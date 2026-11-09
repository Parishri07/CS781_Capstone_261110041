# Phase 4 (Court Judgment Prediction, binary) - hierarchical long-document classifier on a Phase-1 backbone.
#   case text -> Phase-1 tokenizer -> K chunks of up to `size` tokens taken from the END of the judgment (or the
#   start / start+end, for the section study) -> Phase-1 backbone (+ full FT / LoRMA / adapters / ...) -> mean-pooled
#   chunk vectors -> chunk-position embedding + BiGRU over the chunks + attention pooling (+ max pooling)
#   -> 2-way head. Class-weighted cross-entropy, optional distillation from teacher probabilities, decision
#   threshold calibrated on dev for macro-F1.
import json
import math
import os
import random
import time
import zipfile
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ner_model as nm

VERSION = "v2-attention-pooling"   # + v3 diagnostics (grad norm, fp16 scale)

# ======================================================================================================
# 1. Data
# ======================================================================================================
_ID = ("id", "case_id", "doc_id", "uid", "name")
_LAB = ("label", "labels", "y", "target", "judgment", "decision")


def read_json_any(path):
    with open(path, encoding="utf-8") as f:
        head = f.read(1)
    if head in "[{":
        try:
            return json.load(open(path, encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]       # JSON lines


def parse(raw):
    """any common shape -> list of dict(id=<original id, type kept>, text=str, label=0/1 or None)"""
    if isinstance(raw, dict):
        for k in ("data", "examples", "docs", "cases"):
            if k in raw and isinstance(raw[k], (list, dict)):
                raw = raw[k]; break
    if isinstance(raw, dict):
        raw = [dict(v, id=v.get("id", k)) if isinstance(v, dict) else dict(id=k, text=v) for k, v in raw.items()]
    out = []
    for i, d in enumerate(raw):
        did = next((d[k] for k in _ID if k in d), i)
        t = next((d[k] for k in ("text", "document", "case_text", "content") if k in d), "")
        if isinstance(t, list):
            t = " ".join(map(str, t))
        lab = next((d[k] for k in _LAB if k in d), None)
        if isinstance(lab, list):
            lab = lab[0] if lab else None
        out.append(dict(id=did, text=" ".join(str(t).split()), label=None if lab is None else int(lab)))
    return out


_TOK = None


def _init_tok(arch, root):
    global _TOK
    _TOK = nm.load_tokenizer(arch, root)


def _encode(args):
    text, head_chars, tail_chars = args
    pre = " " if getattr(_TOK, "space_prefix", False) else ""
    head = _TOK.encode(pre + text[:head_chars])
    tail = head if len(text) <= head_chars else _TOK.encode(pre + text[-tail_chars:])
    return np.asarray(head, dtype=np.int32), np.asarray(tail, dtype=np.int32)


def tokenize(docs, arch, root, head_chars=20000, tail_chars=26000, workers=4, log=print):
    """-> list of dict(doc, head=np.int32 ids of the first head_chars, tail=ids of the last tail_chars)"""
    t0 = time.time()
    args = [(d["text"], head_chars, tail_chars) for d in docs]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers, initializer=_init_tok, initargs=(arch, root)) as pool:
            enc = pool.map(_encode, args, chunksize=64)
    else:
        _init_tok(arch, root); enc = [_encode(a) for a in args]
    items = [dict(doc=d, head=h, tail=t) for d, (h, t) in zip(docs, enc)]
    n = np.array([len(it["tail"]) for it in items])
    log(f"tokenised {len(items):,} documents in {time.time() - t0:.0f}s | tail tokens: median {int(np.median(n)):,}, "
        f"90% {int(np.percentile(n, 90)):,}")
    return items


def chunks_of(it, mode, K, size):
    """token chunks (list of np arrays) representing one document.
    last: the last K chunks (the earliest one may overlap so every chunk is full); first: the first K chunks;
    head_tail: the first chunk + the last K-1 chunks."""
    if mode == "first":
        h = it["head"]
        return [h[i * size:(i + 1) * size] for i in range(K) if i * size < len(h)] or [h[:size]]
    t = it["tail"]
    k_last = K if mode == "last" else K - 1
    n = len(t)
    if n <= size:
        out = [t]
    else:
        starts = []
        for k in range(k_last):
            s = max(0, n - (k + 1) * size)
            if starts and s == starts[-1]:
                break
            starts.append(s)
            if s == 0:
                break
        out = [t[s:s + size] for s in reversed(starts)]
    if mode == "head_tail":
        out = [it["head"][:size]] + out
    return out


def batch_chunks(chunk_lists, device, pad_id=0):
    """list (docs) of lists of chunks -> ids (N,T) long, lens (N,), doc slices"""
    flat = [c for cl in chunk_lists for c in cl]
    T = max(len(c) for c in flat)
    ids = torch.full((len(flat), T), pad_id, dtype=torch.long)
    lens = torch.zeros(len(flat), dtype=torch.long)
    for i, c in enumerate(flat):
        ids[i, :len(c)] = torch.from_numpy(c.astype(np.int64)); lens[i] = max(1, len(c))
    sl, a = [], 0
    for cl in chunk_lists:
        sl.append((a, a + len(cl))); a += len(cl)
    return ids.to(device), lens.to(device), sl


# ======================================================================================================
# 2. Model
# ======================================================================================================
class ChunkEncoder(nn.Module):
    """backbone over chunks -> chunk vector = [learned attention pooling over tokens ; mean pooling], both over
    layer-normalised token states. Plain mean pooling of 1,024 tokens (v1) washed out the few decisive phrases:
    every chunk looked alike and the model stayed at chance. forward(ids, lens) -> (N, 2d)"""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(backbone.d)
        self.score = nn.Linear(backbone.d, 1)

    def forward(self, ids, lens):
        with nm.autocast():
            h = self.backbone(ids)
        h = self.norm(h.float())                        # fp32: sums over 1,024 tokens
        m = torch.arange(ids.size(1), device=ids.device)[None] < lens[:, None]
        a = torch.softmax(self.score(h).squeeze(-1).masked_fill(~m, -1e4), 1)
        att = (a[..., None] * h).sum(1)
        mean = (h * m[..., None].float()).sum(1) / lens[:, None].float()
        return torch.cat([att, mean], -1)


class CJModel(nn.Module):
    def __init__(self, backbone, hidden=256, dropout=0.2, max_chunks=16, class_weights=None):
        super().__init__()
        self.backbone = backbone
        self.encoder = ChunkEncoder(backbone)
        d = backbone.d
        self.proj = nn.Sequential(nn.Linear(2 * d, 2 * hidden), nn.GELU(), nn.Dropout(dropout))
        self.pos = nn.Embedding(max_chunks + 1, 2 * hidden)          # chunk index counted from the END
        nn.init.zeros_(self.pos.weight)
        self.gru = nn.GRU(2 * hidden, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, 2 * hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(2 * hidden, 2))
        self.register_buffer("cw", class_weights if class_weights is not None else torch.ones(2))
        self.max_chunks = max_chunks

    def doc_logits(self, cv, sl):
        """cv (N,d) chunk vectors, sl doc slices -> (B,2) logits"""
        out = []
        for a, b in sl:
            x = self.proj(cv[a:b])
            n = b - a
            rev = torch.arange(n - 1, -1, -1, device=x.device).clamp(max=self.max_chunks)
            x = x + self.pos(rev)
            h, _ = self.gru(x[None]); h = h[0]
            w = torch.softmax(self.att(h).squeeze(-1), 0)
            out.append(torch.cat([(w[:, None] * h).sum(0), h.max(0).values]))
        return self.head(torch.stack(out))

    def loss(self, logits, y, soft=None, T=2.0, alpha=0.5):
        ce = F.cross_entropy(logits, y, weight=self.cw)
        if soft is not None:
            kl = F.kl_div(F.log_softmax(logits / T, -1), soft, reduction="batchmean") * T * T
            ce = alpha * ce + (1 - alpha) * kl
        return ce


# ======================================================================================================
# 3. Metrics
# ======================================================================================================
def scores(y, p1, thr=0.5):
    """binary labels + P(class 1) -> macro P/R/F1, accuracy, confusion"""
    y = np.asarray(y); pred = (np.asarray(p1) >= thr).astype(int)
    out = {}
    for c in (0, 1):
        tp = int(((pred == c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
        out[c] = dict(P=P, R=R, F1=2 * P * R / (P + R) if P + R else 0.0, support=int((y == c).sum()))
    return dict(macro_F1=(out[0]["F1"] + out[1]["F1"]) / 2, macro_P=(out[0]["P"] + out[1]["P"]) / 2,
                macro_R=(out[0]["R"] + out[1]["R"]) / 2, accuracy=float((pred == y).mean()), per_class=out,
                confusion=dict(tn=int(((pred == 0) & (y == 0)).sum()), fp=int(((pred == 1) & (y == 0)).sum()),
                               fn=int(((pred == 0) & (y == 1)).sum()), tp=int(((pred == 1) & (y == 1)).sum())))


def best_threshold(y, p1):
    """decision threshold maximising macro-F1 on dev (calibration for the macro metric)"""
    grid = np.round(np.arange(0.20, 0.801, 0.01), 2)
    f = [scores(y, p1, t)["macro_F1"] for t in grid]
    i = int(np.argmax(f))
    return float(grid[i]), float(f[i])


# ======================================================================================================
# 4. Training / prediction
# ======================================================================================================
@torch.no_grad()
def predict(model, items, cfg, device, enc=None):
    """-> np.array P(class 1) for every item"""
    model.eval(); enc = enc or model.encoder
    out = []
    bs = cfg.get("eval_docs", 8)
    for i in range(0, len(items), bs):
        cl = [chunks_of(it, cfg["mode"], cfg["K"], cfg["size"]) for it in items[i:i + bs]]
        ids, lens, sl = batch_chunks(cl, device)
        lg = model.doc_logits(enc(ids, lens), sl)
        out.append(lg.float().softmax(-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def train(model, tr, dv, te, cfg, device, log=print, soft=None):
    rng = random.Random(cfg["seed"])
    gpus = torch.cuda.device_count()
    enc = nn.DataParallel(model.encoder) if gpus > 1 else model.encoder
    is_bb = lambda n: n.startswith(("backbone.", "encoder.backbone."))   # pooling layers learn at the head LR
    named, seen = {}, set()
    for n, p in model.named_parameters():
        if p.requires_grad and id(p) not in seen:
            named[n] = p; seen.add(id(p))
    groups = [dict(params=[p for n, p in named.items() if is_bb(n)], lr=cfg["lr"]),
              dict(params=[p for n, p in named.items() if not is_bb(n)], lr=cfg["lr_head"])]
    groups = [g for g in groups if g["params"]]
    for g in groups: g["base_lr"] = g["lr"]
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    B, acc = cfg["batch_docs"], cfg["accum"]
    steps_per_epoch = math.ceil(len(tr) / (B * acc))
    total = int(steps_per_epoch * cfg["epochs"]); warm = max(1, int(cfg["warmup_frac"] * total))
    eval_every = max(1, int(steps_per_epoch * cfg.get("eval_frac", 0.34)))
    ytr = [it["doc"]["label"] for it in tr]; ydv = np.array([it["doc"]["label"] for it in dv])
    log(f"train {len(tr):,} docs | {B} docs x {acc} accum per step | {steps_per_epoch:,} steps/epoch x {cfg['epochs']} "
        f"| eval every {eval_every:,} steps | trainable {sum(p.numel() for p in params):,} | GPUs {max(1, gpus)}")
    t0 = time.time(); budget = cfg.get("budget_hours", 9.0) * 3600
    snaps, best, best_state, step, run, nb = [], -1.0, None, 0, 0.0, 0
    last_eval = [-1]
    hit = [0, 0]                                        # training accuracy since the last evaluation
    hist = []
    gn = [0.0, 0, 0]                                    # grad-norm sum, optimizer steps, skipped (fp16 overflow) steps
    K = cfg.get("snapshot_k", 2)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    def evaluate_now(tag):
        nonlocal best, best_state
        last_eval[0] = step
        p = predict(model, dv, cfg, device, enc)
        thr, f_t = best_threshold(ydv, p); f_05 = scores(ydv, p)["macro_F1"]
        hist.append(dict(step=step, epoch=round(step / steps_per_epoch, 2), dev_F1=round(f_t, 4), dev_F1_05=round(f_05, 4)))
        log(f"[{tag}] step {step:,} | loss {run / max(1, nb):.4f} | train acc {hit[0] / max(1, hit[1]):.3f} | "
            f"grad norm {gn[0] / max(1, gn[1]):.3f}, fp16 scale {scaler.get_scale():.0f}, skipped {gn[2]} | dev macro-F1 {f_05:.4f} @0.5, {f_t:.4f} @{thr:.2f} | "
            f"{(time.time() - t0) / 60:.0f} min")
        if len(snaps) < K or f_t > min(s["f1"] for s in snaps):
            snaps.append(dict(step=step, f1=f_t, dev=p, test=predict(model, te, cfg, device, enc)))
            snaps.sort(key=lambda s: -s["f1"]); del snaps[K:]
        if f_t > best:
            best = f_t; best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        model.train()

    stop = False
    for ep in range(math.ceil(cfg["epochs"])):
        model.train(); order = list(range(len(tr))); rng.shuffle(order)
        for k in range(0, len(order), B):
            idx = order[k:k + B]
            cl = [chunks_of(tr[i], cfg["mode"], cfg["K"], cfg["size"]) for i in idx]
            ids, lens, sl = batch_chunks(cl, device)
            lg = model.doc_logits(enc(ids, lens), sl)
            y = torch.tensor([ytr[i] for i in idx], device=device)
            st = torch.stack([soft[tr[i]["doc"]["id"]] for i in idx]).to(device) if soft is not None else None
            loss = model.loss(lg.float(), y, st, cfg.get("kd_T", 2.0), cfg.get("kd_alpha", 0.5)) / acc
            scaler.scale(loss).backward(); run += loss.item() * acc; nb += 1
            hit[0] += int((lg.argmax(-1) == y).sum()); hit[1] += len(idx)
            if (k // B + 1) % acc:
                continue
            step += 1
            f = step / warm if step <= warm else max(0.0, (total - step) / max(1, total - warm))
            for g in opt.param_groups: g["lr"] = g["base_lr"] * f
            scaler.unscale_(opt); g = float(torch.nn.utils.clip_grad_norm_(params, cfg["clip"]))
            if math.isfinite(g): gn[0] += g; gn[1] += 1
            else: gn[2] += 1
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if step % eval_every == 0 or step >= total:
                evaluate_now(f"epoch {ep + 1}"); run = nb = 0; hit[0] = hit[1] = 0; gn[:] = [0.0, 0, 0]
            if step >= total or time.time() - t0 > budget:
                stop = True
                if time.time() - t0 > budget: log("time budget reached - stopping")
                break
        if stop:
            break
    if last_eval[0] != step:
        evaluate_now("final")
    model.load_state_dict(best_state)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    dev = np.mean([s["dev"] for s in snaps], 0); test = np.mean([s["test"] for s in snaps], 0)
    log(f"averaging the best {len(snaps)} evaluations (steps {[s['step'] for s in snaps]}, dev {[round(s['f1'], 4) for s in snaps]})")
    return dict(dev=dev, test=test, best_single=best, train_minutes=round((time.time() - t0) / 60, 1),
                peak_gpu_gb=round(peak, 2), steps=step, history=hist)


# ======================================================================================================
# 5. One experiment
# ======================================================================================================
PROBS = {}                                              # run -> dict(dev, test, train?) P(class 1)
_TOKS = {}


def prepared(D, arch, log=print):
    key = (nm.ARCH_INFO[arch][0], len(D["train"]), len(D["dev"]), len(D["test"]))   # smoke data != real data
    if key not in _TOKS:
        _TOKS[key] = {s: tokenize(D[s], arch, D["root"], workers=D.get("workers", 4), log=log) for s in ("train", "dev", "test")}
    return _TOKS[key]


def build(cfg, D):
    bb = nm.Backbone(cfg["arch"], D["root"], bidirectional=cfg["bidirectional"], keep_layers=cfg.get("keep_layers"),
                     init=cfg.get("init", "pretrained"))
    c = Counter(d["label"] for d in D["train"]); tot = sum(c.values())
    cw = torch.tensor([tot / (2 * max(1, c[k])) for k in (0, 1)]) if cfg.get("class_weighting", True) else None
    model = CJModel(bb, hidden=cfg["hidden"], dropout=cfg["dropout"], class_weights=cw)
    desc = nm.apply_method(model, cfg)
    return model, desc


def run_experiment(cfg, D, device, out_root, log=print, soft=None, keep_train=False, subsample=None):
    t_start = time.time()
    run_dir = os.path.join(out_root, cfg["run"]); os.makedirs(run_dir, exist_ok=True)
    T = prepared(D, cfg["arch"], log)
    tr, dv, te = T["train"], T["dev"], T["test"]
    if subsample:
        tr = random.Random(0).sample(tr, min(subsample, len(tr)))
    torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    model, desc = build(cfg, D)
    model.to(device)
    total = sum(p.numel() for p in model.parameters()); trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"\n=== {cfg['run']} | {cfg['arch']} | {desc} | chunks: {cfg['mode']} K={cfg['K']} x {cfg['size']} tokens | "
        f"params {total:,}, trainable {trainable:,} ({trainable / total:.2%})" + (" | KD" if soft is not None else ""))
    res = train(model, tr, dv, te, cfg, device, log, soft)
    t0 = time.time(); _ = predict(model, dv, cfg, device); infer = time.time() - t0
    ydv = np.array([it["doc"]["label"] for it in dv])
    thr, f_t = best_threshold(ydv, res["dev"]); sc = scores(ydv, res["dev"], thr); s05 = scores(ydv, res["dev"])
    log(f"DEV: macro-F1 {s05['macro_F1']:.4f} @0.5 | {sc['macro_F1']:.4f} @ tuned threshold {thr:.2f} | "
        f"P {sc['macro_P']:.4f} R {sc['macro_R']:.4f} acc {sc['accuracy']:.4f} | confusion {sc['confusion']}")
    for c in (0, 1):
        r = sc["per_class"][c]; log(f"   class {c}: P {r['P']:.3f} R {r['R']:.3f} F1 {r['F1']:.3f} support {r['support']}")
    PROBS[cfg["run"]] = dict(dev=res["dev"], test=res["test"], arch=cfg["arch"], method=cfg["method"])
    if keep_train:                                       # in-sample probabilities: teacher targets for distillation
        log("predicting the training documents (teacher targets)...")
        dp = nn.DataParallel(model.encoder) if torch.cuda.device_count() > 1 else None
        PROBS[cfg["run"]]["train"] = predict(model, T["train"], cfg, device, dp)
    zp = write_submission(res["test"], thr, D, os.path.join(out_root, cfg["run"]), cfg["run"], log)
    rec = dict(run=cfg["run"], arch=cfg["arch"], method=desc, cfg=cfg, params_total=total, params_trainable=trainable,
               dev_F1_05=s05["macro_F1"], dev_F1=sc["macro_F1"], threshold=thr, dev=dict((k, sc[k]) for k in
               ("macro_F1", "macro_P", "macro_R", "accuracy", "confusion")), dev_inference_sec=round(infer, 1),
               kd=soft is not None, zip=os.path.basename(zp), minutes_total=round((time.time() - t_start) / 60, 1),
               train_docs=len(tr), **{k: v for k, v in res.items() if k not in ("dev", "test")})
    json.dump(rec, open(os.path.join(run_dir, "record.json"), "w"), indent=1, default=str)
    del model; torch.cuda.empty_cache()
    return rec


def write_submission(p_test, thr, D, out_dir, name, log=print):
    """[{"id": <id as in test.json>, "pred": 0/1}, ...] -> prediction.json -> zip"""
    assert len(p_test) == len(D["test"])
    sub = [dict(id=d["id"], pred=int(p >= thr)) for d, p in zip(D["test"], p_test)]
    assert len({json.dumps(s["id"]) for s in sub}) == len(sub) and all(s["pred"] in (0, 1) for s in sub)
    os.makedirs(out_dir, exist_ok=True)
    pj = os.path.join(out_dir, "prediction.json"); json.dump(sub, open(pj, "w"), ensure_ascii=False)
    zp = os.path.join(out_dir, f"submission_{name}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    c = Counter(s["pred"] for s in sub)
    log(f"{name}: {len(sub):,} test cases | predicted 0: {c[0]:,}, 1: {c[1]:,} (threshold {thr:.2f}) -> {zp}")
    return zp


def save_probs(path, records):
    out = {r["run"]: dict(PROBS[r["run"]], dev_F1=r["dev_F1"], params=r["params_total"],
                          dev_inference_sec=r["dev_inference_sec"]) for r in records if r["run"] in PROBS}
    torch.save(out, path)


def load_probs(pattern="/kaggle/input/**/p4_probs_*.pt", log=print):
    import glob
    out = {}
    for f in sorted(glob.glob(pattern, recursive=True)):
        for n, v in torch.load(f, map_location="cpu", weights_only=False).items():
            out.setdefault(n, v)
        log(f"loaded {f}")
    return out


def error_analysis(D, p_dev, thr, log=print, show=3):
    """false positives / negatives: lengths and the last words before the (removed) decision"""
    y = np.array([d["label"] for d in D["dev"]]); pred = (p_dev >= thr).astype(int)
    L = np.array([len(d["text"].split()) for d in D["dev"]])
    for name, m in (("correct", pred == y), ("false positive (pred 1, gold 0)", (pred == 1) & (y == 0)),
                    ("false negative (pred 0, gold 1)", (pred == 0) & (y == 1))):
        if m.sum():
            log(f"{name}: {int(m.sum())} cases | median length {int(np.median(L[m])):,} words | "
                f"mean confidence {float(np.mean(np.abs(p_dev[m] - 0.5)) + 0.5):.3f}")
    for name, m in (("FALSE POSITIVE", (pred == 1) & (y == 0)), ("FALSE NEGATIVE", (pred == 0) & (y == 1))):
        for i in np.where(m)[0][:show]:
            log(f"\n{name} | P(1)={p_dev[i]:.3f} | {L[i]:,} words | ...{D['dev'][i]['text'][-400:]}")
    qs = np.percentile(L, [0, 25, 50, 75, 100])
    for a, b in zip(qs[:-1], qs[1:]):
        m = (L >= a) & (L <= b)
        log(f"length {int(a):>6,}-{int(b):>6,} words: dev macro-F1 {scores(y[m], p_dev[m], thr)['macro_F1']:.4f} ({int(m.sum())} cases)")
