# Phase 5 (Legal Statute Identification, multi-label) - hierarchical document encoder on a Phase-1 backbone.
#   case text -> Phase-1 tokenizer -> up to K windows covering the WHOLE document (consecutive if it fits, otherwise
#   K evenly spaced windows) -> Phase-1 backbone (+ full FT / LoRMA / partial FT ...) -> per-window learned attention
#   pooling over tokens (+ mean) -> window-position embedding + BiGRU over windows + attention and MAX pooling
#   (a statute cue can sit in any window) -> one logit per statute (binary relevance).
#   Loss: BCE with per-statute positive weights (rare statutes), optional distillation from teacher probabilities.
#   Decision: per-statute thresholds tuned on dev (falling back to a global one for statutes with few dev positives),
#   at least one statute per document.  Phase-4 lessons: attention pooling, causal mask kept for arch 4 + LoRMA.
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

VERSION = "p5-v3"                                     # v3: + asymmetric loss option, bagged thresholds                                     # v2: interim probabilities + zip saved after every evaluation

# ======================================================================================================
# 1. Data
# ======================================================================================================
_ID = ("id", "case_id", "doc_id", "uid", "name")
_LAB = ("labels", "label", "statutes", "statute_ids", "targets", "y")


def read_json_any(path):
    with open(path, encoding="utf-8") as f:
        head = f.read(1)
    if head in "[{":
        try:
            return json.load(open(path, encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _as_label(x):
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    try:
        return int(str(x).strip())
    except ValueError:
        return str(x).strip()


def parse(raw):
    """any common shape -> list of dict(id=<original>, text=str, labels=[unique statute ids] or None)"""
    if isinstance(raw, dict):
        for k in ("data", "examples", "docs", "cases"):
            if k in raw and isinstance(raw[k], (list, dict)):
                raw = raw[k]; break
    if isinstance(raw, dict):
        raw = [dict(v, id=v.get("id", k)) if isinstance(v, dict) else dict(id=k, text=v) for k, v in raw.items()]
    out = []
    for i, d in enumerate(raw):
        did = next((d[k] for k in _ID if k in d), i)
        t = next((d[k] for k in ("text", "document", "facts", "case_text", "content") if k in d), "")
        if isinstance(t, list):
            t = " ".join(map(str, t))
        lab = next((d[k] for k in _LAB if k in d), None)
        if lab is not None and not isinstance(lab, (list, tuple)):
            lab = [lab]
        labs = None if lab is None else sorted({_as_label(x) for x in lab}, key=lambda v: (isinstance(v, str), v))
        out.append(dict(id=did, text=" ".join(str(t).split()), labels=labs))
    return out


_TOK = None


def _init_tok(arch, root):
    global _TOK
    _TOK = nm.load_tokenizer(arch, root)


def _encode(args):
    text, max_chars = args
    pre = " " if getattr(_TOK, "space_prefix", False) else ""
    return np.asarray(_TOK.encode(pre + text[:max_chars]), dtype=np.int32)


def tokenize(docs, arch, root, max_chars=40000, workers=4, log=print):
    t0 = time.time()
    args = [(d["text"], max_chars) for d in docs]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers, initializer=_init_tok, initargs=(arch, root)) as pool:
            enc = pool.map(_encode, args, chunksize=64)
    else:
        _init_tok(arch, root); enc = [_encode(a) for a in args]
    items = [dict(doc=d, ids=e) for d, e in zip(docs, enc)]
    n = np.array([len(e) for e in enc])
    log(f"tokenised {len(items):,} documents in {time.time() - t0:.0f}s | tokens: median {int(np.median(n)):,}, "
        f"90% {int(np.percentile(n, 90)):,}, max {int(n.max()):,}")
    return items


def chunks_of(it, mode, K, size):
    """windows representing one document.
    spread: the whole document if it fits in K windows (consecutive, the last one end-aligned), otherwise K evenly
    spaced windows from start to end.  first / last: the first / last K windows (section study)."""
    t = it["ids"]; n = len(t)
    if n <= size:
        return [t]
    if mode == "first":
        return [t[i * size:(i + 1) * size] for i in range(K) if i * size < n]
    if mode == "last":
        starts = sorted({max(0, n - (k + 1) * size) for k in range(K)})
        return [t[s:s + size] for s in starts]
    m = math.ceil(n / size)
    if m <= K:
        starts = [i * size for i in range(m - 1)] + [n - size]
    else:
        starts = sorted({int(round(x)) for x in np.linspace(0, n - size, K)})
    return [t[s:s + size] for s in starts]


def batch_chunks(chunk_lists, device, pad_id=0):
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
    """backbone over windows -> [learned attention pooling over tokens ; mean], on layer-normalised states (2d)"""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(backbone.d)
        self.score = nn.Linear(backbone.d, 1)

    def forward(self, ids, lens):
        with nm.autocast():
            h = self.backbone(ids)
        h = self.norm(h.float())
        m = torch.arange(ids.size(1), device=ids.device)[None] < lens[:, None]
        a = torch.softmax(self.score(h).squeeze(-1).masked_fill(~m, -1e4), 1)
        att = (a[..., None] * h).sum(1)
        mean = (h * m[..., None].float()).sum(1) / lens[:, None].float()
        return torch.cat([att, mean], -1)


class LSModel(nn.Module):
    def __init__(self, backbone, n_labels, hidden=256, dropout=0.2, max_chunks=16, pos_weight=None):
        super().__init__()
        self.backbone = backbone
        self.encoder = ChunkEncoder(backbone)
        d = backbone.d
        self.proj = nn.Sequential(nn.Linear(2 * d, 2 * hidden), nn.GELU(), nn.Dropout(dropout))
        self.pos = nn.Embedding(max_chunks + 1, 2 * hidden); nn.init.zeros_(self.pos.weight)
        self.gru = nn.GRU(2 * hidden, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, 2 * hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(2 * hidden, n_labels))
        self.register_buffer("pw", pos_weight if pos_weight is not None else torch.ones(n_labels))
        self.max_chunks = max_chunks

    def doc_logits(self, cv, sl):
        out = []
        for a, b in sl:
            x = self.proj(cv[a:b])
            idx = torch.arange(b - a, device=x.device).clamp(max=self.max_chunks)
            x = x + self.pos(idx)
            h, _ = self.gru(x[None]); h = h[0]
            w = torch.softmax(self.att(h).squeeze(-1), 0)
            out.append(torch.cat([(w[:, None] * h).sum(0), h.max(0).values]))
        return self.head(torch.stack(out))

    def loss(self, logits, y, soft=None, T=2.0, alpha=0.5):
        if getattr(self, "loss_type", "bce") == "asl":       # Asymmetric Loss (Ridnik et al., ICCV 2021)
            p = torch.sigmoid(logits)
            p_neg = (p - 0.05).clamp(min=0)                     # probability shifting: ignore very easy negatives
            lp = y * (1 - p).pow(0.0) * torch.log(p.clamp(min=1e-8))
            ln = (1 - y) * p_neg.pow(4.0) * torch.log((1 - p_neg).clamp(min=1e-8))
            bce = -(lp + ln).sum(1).mean() / y.size(1) * 4.0       # per-label average (x4 keeps the scale near BCE)
        else:
            bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pw)
        if soft is not None:                                   # distillation: match the teacher's softened probabilities
            kd = F.binary_cross_entropy_with_logits(logits / T, soft) * T * T
            bce = alpha * bce + (1 - alpha) * kd
        return bce


# ======================================================================================================
# 3. Metrics + thresholds
# ======================================================================================================
def f1_per_label(Y, Pb):
    tp = (Y & Pb).sum(0).astype(float); fp = (~Y & Pb).sum(0).astype(float); fn = (Y & ~Pb).sum(0).astype(float)
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    return f, p, r, tp, fp, fn


def scores(Y, Pb):
    """macro-F1 over the statutes present in the reference (as the evaluator does), micro-F1, subset accuracy"""
    Y = Y.astype(bool); Pb = Pb.astype(bool)
    f, p, r, tp, fp, fn = f1_per_label(Y, Pb)
    ref = Y.sum(0) > 0
    micro_p = tp.sum() / max(1, tp.sum() + fp.sum()); micro_r = tp.sum() / max(1, tp.sum() + fn.sum())
    micro = 2 * micro_p * micro_r / max(1e-9, micro_p + micro_r)
    return dict(macro_F1=float(f[ref].mean()), macro_P=float(p[ref].mean()), macro_R=float(r[ref].mean()),
                micro_F1=float(micro), subset_acc=float((Y == Pb).all(1).mean()),
                avg_pred=float(Pb.sum(1).mean()), avg_gold=float(Y.sum(1).mean()))


def decide(P, thr, at_least_one=True):
    """probabilities (N,L) + thresholds (L,) or scalar -> boolean predictions; every document gets >= 1 statute"""
    Pb = P >= thr
    if at_least_one:
        none = ~Pb.any(1)
        if none.any():
            j = np.argmax(P[none] / np.maximum(np.broadcast_to(thr, P.shape)[none], 1e-6), 1)
            Pb[np.where(none)[0], j] = True
    return Pb


GRID = np.round(np.arange(0.02, 0.951, 0.01), 2)


def global_threshold(Y, P):
    f = [scores(Y, decide(P, t))["macro_F1"] for t in GRID]
    i = int(np.argmax(f))
    return float(GRID[i]), float(f[i])


def per_label_thresholds(Y, P, g, min_pos=5):
    """each statute with >= min_pos dev positives gets its own threshold; the others keep the global one"""
    Y = Y.astype(bool); thr = np.full(Y.shape[1], g)
    for j in range(Y.shape[1]):
        if Y[:, j].sum() >= min_pos:
            best, bt = -1.0, g
            for t in GRID:
                pb = P[:, j] >= t
                tp = (pb & Y[:, j]).sum(); fp = (pb & ~Y[:, j]).sum(); fn = (~pb & Y[:, j]).sum()
                f = 2 * tp / max(1, 2 * tp + fp + fn)
                if f > best + 1e-9: best, bt = f, t
            thr[j] = bt
    return thr


def choose_thresholds(Y, P, seed=0, log=print):
    """global vs per-statute thresholds, compared HONESTLY by 2-fold cross-fitting on dev (tune on one half,
    score on the other) -> the better strategy, refitted on the whole dev set"""
    g, fg = global_threshold(Y, P)
    rng = np.random.RandomState(seed); idx = rng.permutation(len(Y)); halves = (idx[: len(Y) // 2], idx[len(Y) // 2:])
    cv = {"global": [], "per_label": []}
    for a, b in (halves, halves[::-1]):
        ga, _ = global_threshold(Y[a], P[a])
        cv["global"].append(scores(Y[b], decide(P[b], ga))["macro_F1"])
        cv["per_label"].append(scores(Y[b], decide(P[b], per_label_thresholds(Y[a], P[a], ga)))["macro_F1"])
    cvm = {k: float(np.mean(v)) for k, v in cv.items()}
    best = max(cvm, key=cvm.get)
    thr = np.full(Y.shape[1], g) if best == "global" else per_label_thresholds(Y, P, g)
    log(f"thresholds: global {g:.2f} (dev macro-F1 {fg:.4f}) | 2-fold cross-fitted macro-F1: global {cvm['global']:.4f}, "
        f"per-statute {cvm['per_label']:.4f} -> using {best}")
    return thr, dict(global_thr=g, global_F1=fg, cv=cvm, strategy=best)


# ======================================================================================================
# 4. Training / prediction
# ======================================================================================================
@torch.no_grad()
def predict(model, items, cfg, device, enc=None):
    """-> np.float32 array (N, L) of statute probabilities"""
    model.eval(); enc = enc or model.encoder
    out = []
    bs = cfg.get("eval_docs", 8)
    for i in range(0, len(items), bs):
        cl = [chunks_of(it, cfg["mode"], cfg["K"], cfg["size"]) for it in items[i:i + bs]]
        ids, lens, sl = batch_chunks(cl, device)
        out.append(torch.sigmoid(model.doc_logits(enc(ids, lens), sl).float()).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def train(model, tr, Ytr, dv, Ydv, te, cfg, device, log=print, soft=None, on_eval=None):
    rng = random.Random(cfg["seed"])
    gpus = torch.cuda.device_count()
    enc = nn.DataParallel(model.encoder) if gpus > 1 else model.encoder
    is_bb = lambda n: n.startswith(("backbone.", "encoder.backbone."))
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
    eval_every = max(1, int(steps_per_epoch * cfg.get("eval_frac", 1.0)))
    log(f"train {len(tr):,} docs | {B} docs x {acc} accum per step | {steps_per_epoch:,} steps/epoch x {cfg['epochs']} "
        f"| eval every {eval_every:,} steps | trainable {sum(p.numel() for p in params):,} | GPUs {gpus}")
    assert gpus >= 1 or cfg.get("allow_cpu"), "NO GPU in this session - set Accelerator to GPU T4 x2 (or quota is used up)"
    t0 = time.time(); budget = cfg.get("budget_hours", 7.0) * 3600
    snaps, best, best_state, step, run, nb = [], -1.0, None, 0, 0.0, 0
    K = cfg.get("snapshot_k", 2); last_eval = [-1]; hist = []; gn = [0.0, 0, 0]
    Ytr_t = torch.from_numpy(Ytr.astype(np.float32))
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    def evaluate_now(tag):
        nonlocal best, best_state
        last_eval[0] = step
        p = predict(model, dv, cfg, device, enc)
        g, f = global_threshold(Ydv, p)
        s = scores(Ydv, decide(p, g))
        hist.append(dict(step=step, epoch=round(step / steps_per_epoch, 2), macro_F1=round(f, 4), thr=g))
        log(f"[{tag}] step {step:,} | loss {run / max(1, nb):.4f} | grad norm {gn[0] / max(1, gn[1]):.3f}, skipped {gn[2]} | "
            f"dev macro-F1 {f:.4f} @ {g:.2f} (micro {s['micro_F1']:.4f}, statutes/doc {s['avg_pred']:.2f} vs {s['avg_gold']:.2f}) | "
            f"{(time.time() - t0) / 60:.0f} min")
        if len(snaps) < K or f > min(x["f1"] for x in snaps):
            snaps.append(dict(step=step, f1=f, dev=p, test=predict(model, te, cfg, device, enc)))
            snaps.sort(key=lambda x: -x["f1"]); del snaps[K:]
        if f > best:
            best = f; best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        if on_eval is not None:                                # interim save: a run cut off later still leaves a submission
            try:
                on_eval(np.mean([x["dev"] for x in snaps], 0), np.mean([x["test"] for x in snaps], 0), step)
            except Exception as e:
                log(f"  (interim save failed: {e})")
        model.train()

    stop = False
    for ep in range(math.ceil(cfg["epochs"])):
        model.train(); order = list(range(len(tr))); rng.shuffle(order)
        for k in range(0, len(order), B):
            idx = order[k:k + B]
            cl = [chunks_of(tr[i], cfg["mode"], cfg["K"], cfg["size"]) for i in idx]
            ids, lens, sl = batch_chunks(cl, device)
            lg = model.doc_logits(enc(ids, lens), sl)
            y = Ytr_t[idx].to(device)
            st = torch.from_numpy(soft[idx]).to(device) if soft is not None else None
            loss = model.loss(lg.float(), y, st, cfg.get("kd_T", 2.0), cfg.get("kd_alpha", 0.5)) / acc
            scaler.scale(loss).backward(); run += loss.item() * acc; nb += 1
            if (k // B + 1) % acc:
                continue
            step += 1
            f = step / warm if step <= warm else max(0.0, (total - step) / max(1, total - warm))
            for g in opt.param_groups: g["lr"] = g["base_lr"] * f
            scaler.unscale_(opt); gnorm = float(torch.nn.utils.clip_grad_norm_(params, cfg["clip"]))
            if math.isfinite(gnorm): gn[0] += gnorm; gn[1] += 1
            else: gn[2] += 1
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if step % eval_every == 0 or step >= total:
                evaluate_now(f"epoch {ep + 1}"); run = nb = 0; gn[:] = [0.0, 0, 0]
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
    dev = np.mean([x["dev"] for x in snaps], 0); test = np.mean([x["test"] for x in snaps], 0)
    log(f"averaging the best {len(snaps)} evaluations (steps {[x['step'] for x in snaps]}, dev {[round(x['f1'], 4) for x in snaps]})")
    return dict(dev=dev, test=test, best_single=best, train_minutes=round((time.time() - t0) / 60, 1),
                peak_gpu_gb=round(peak, 2), steps=step, history=hist)


# ======================================================================================================
# 5. Experiments, submission, analysis
# ======================================================================================================
PROBS = {}
_TOKS = {}


def label_space(D):
    """statutes seen in training (only these can be predicted), in a fixed order (D["LABELS"] pins it, e.g. for smoke data)"""
    if D.get("LABELS"):
        labs = list(D["LABELS"]); return labs, {l: i for i, l in enumerate(labs)}
    labs = sorted({l for d in D["train"] for l in d["labels"]}, key=lambda v: (isinstance(v, str), v))
    return labs, {l: i for i, l in enumerate(labs)}


def multi_hot(docs, L2I):
    Y = np.zeros((len(docs), len(L2I)), dtype=bool)
    for i, d in enumerate(docs):
        for l in d["labels"] or []:
            if l in L2I: Y[i, L2I[l]] = True
    return Y


def prepared(D, arch, log=print):
    key = (nm.ARCH_INFO[arch][0], len(D["train"]), len(D["dev"]), len(D["test"]))
    if key not in _TOKS:
        _TOKS[key] = {s: tokenize(D[s], arch, D["root"], workers=D.get("workers", 4), log=log) for s in ("train", "dev", "test")}
    return _TOKS[key]


def build(cfg, D, Ytr):
    bb = nm.Backbone(cfg["arch"], D["root"], bidirectional=cfg["bidirectional"], keep_layers=cfg.get("keep_layers"),
                     init=cfg.get("init", "pretrained"))
    pos, neg = Ytr.sum(0).astype(float), (len(Ytr) - Ytr.sum(0)).astype(float)
    pw = torch.tensor(np.clip(np.sqrt(neg / np.maximum(pos, 1)), 1.0, cfg.get("pos_weight_cap", 30.0)), dtype=torch.float)
    model = LSModel(bb, Ytr.shape[1], hidden=cfg["hidden"], dropout=cfg["dropout"], pos_weight=pw)
    model.loss_type = cfg.get("loss", "bce")
    desc = nm.apply_method(model, cfg)
    if model.loss_type != "bce": desc += f" | loss {model.loss_type}"
    return model, desc


def run_experiment(cfg, D, device, out_root, log=print, soft=None, keep_train=False, subsample=None):
    t_start = time.time()
    os.makedirs(os.path.join(out_root, cfg["run"]), exist_ok=True)
    LABELS, L2I = label_space(D)
    T = prepared(D, cfg["arch"], log)
    tr, dv, te = T["train"], T["dev"], T["test"]
    Ytr, Ydv = multi_hot(D["train"], L2I), multi_hot(D["dev"], L2I)
    if subsample:
        sel = sorted(random.Random(0).sample(range(len(tr)), min(subsample, len(tr))))
        tr = [tr[i] for i in sel]; Ytr = Ytr[sel]; soft = soft[sel] if soft is not None else None
    torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    model, desc = build(cfg, D, Ytr)
    model.to(device)
    total = sum(p.numel() for p in model.parameters()); trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"\n=== {cfg['run']} | {cfg['arch']} | {desc} | windows: {cfg['mode']} K={cfg['K']} x {cfg['size']} tokens | "
        f"{len(LABELS)} statutes | params {total:,}, trainable {trainable:,} ({trainable / total:.2%})" + (" | KD" if soft is not None else ""))
    run_dir = os.path.join(out_root, cfg["run"])

    def on_eval(p_dev, p_test, step):
        thr_i, ti = choose_thresholds(Ydv, p_dev, log=lambda *a: None)
        f_i = scores(Ydv, decide(p_dev, thr_i))["macro_F1"]
        torch.save({cfg["run"]: dict(dev=p_dev.astype(np.float16), test=p_test.astype(np.float16), arch=cfg["arch"],
                                      method=cfg["method"], labels=LABELS, thr=thr_i, dev_F1=f_i, params=total,
                                      dev_inference_sec=0.0, interim_step=step)},
                   os.path.join(out_root, f"p5_probs_{cfg['run']}_interim.pt"))
        write_submission(p_test, thr_i, D, LABELS, os.path.join(run_dir, "interim"), cfg["run"] + "_interim",
                         log=lambda m: log("  interim save: " + m + f" | dev macro-F1 {f_i:.4f} ({ti['strategy']} thresholds)"))

    res = train(model, tr, Ytr, dv, Ydv, te, cfg, device, log, soft, on_eval=on_eval)
    t0 = time.time(); _ = predict(model, dv, cfg, device); infer = time.time() - t0
    thr, tinfo = choose_thresholds(Ydv, res["dev"], log=log)
    sc = scores(Ydv, decide(res["dev"], thr))
    log(f"DEV: macro-F1 {sc['macro_F1']:.4f} | macro P {sc['macro_P']:.4f} R {sc['macro_R']:.4f} | micro-F1 {sc['micro_F1']:.4f} | "
        f"subset acc {sc['subset_acc']:.4f} | statutes/doc {sc['avg_pred']:.2f} (gold {sc['avg_gold']:.2f})")
    PROBS[cfg["run"]] = dict(dev=res["dev"].astype(np.float16), test=res["test"].astype(np.float16), arch=cfg["arch"],
                             method=cfg["method"], labels=LABELS)
    if keep_train:
        log("predicting the training documents (teacher targets)...")
        dp = nn.DataParallel(model.encoder) if torch.cuda.device_count() > 1 else None
        PROBS[cfg["run"]]["train"] = predict(model, T["train"], cfg, device, dp).astype(np.float16)
    zp = write_submission(res["test"], thr, D, LABELS, os.path.join(out_root, cfg["run"]), cfg["run"], log)
    rec = dict(run=cfg["run"], arch=cfg["arch"], method=desc, cfg=cfg, params_total=total, params_trainable=trainable,
               dev=sc, dev_F1=sc["macro_F1"], thresholds=tinfo, dev_inference_sec=round(infer, 1), kd=soft is not None,
               zip=os.path.basename(zp), minutes_total=round((time.time() - t_start) / 60, 1), train_docs=len(tr),
               **{k: v for k, v in res.items() if k not in ("dev", "test")})
    PROBS[cfg["run"]]["thr"] = thr
    json.dump(rec, open(os.path.join(out_root, cfg["run"], "record.json"), "w"), indent=1, default=str)
    del model; torch.cuda.empty_cache()
    return rec


def write_submission(P_test, thr, D, LABELS, out_dir, name, log=print):
    """[{"id": <id as in test.json>, "pred": [unique valid statute ids]}, ...] -> prediction.json -> zip"""
    Pb = decide(np.asarray(P_test, dtype=np.float32), thr)
    assert Pb.shape[0] == len(D["test"])
    sub = [dict(id=d["id"], pred=sorted({LABELS[j] for j in np.where(row)[0]}, key=lambda v: (isinstance(v, str), v)))
           for d, row in zip(D["test"], Pb)]
    valid = set(LABELS)
    assert len({json.dumps(s["id"]) for s in sub}) == len(sub) and all(set(s["pred"]) <= valid and s["pred"] for s in sub)
    os.makedirs(out_dir, exist_ok=True)
    pj = os.path.join(out_dir, "prediction.json"); json.dump(sub, open(pj, "w"), ensure_ascii=False)
    zp = os.path.join(out_dir, f"submission_{name}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    log(f"{name}: {len(sub):,} test documents | {np.mean([len(s['pred']) for s in sub]):.2f} statutes/doc -> {zp}")
    return zp


def save_probs(path, records):
    out = {r["run"]: dict(PROBS[r["run"]], dev_F1=r["dev_F1"], params=r["params_total"],
                          dev_inference_sec=r["dev_inference_sec"]) for r in records if r["run"] in PROBS}
    torch.save(out, path)


def load_probs(pattern="/kaggle/input/**/p5_probs_*.pt", log=print):
    import glob
    out = {}
    for f in sorted(glob.glob(pattern, recursive=True)):
        for n, v in torch.load(f, map_location="cpu", weights_only=False).items():
            out.setdefault(n, v)
        log(f"loaded {f}")
    return out


def analysis(D, P_dev, thr, LABELS, log=print, top=12):
    """rare vs frequent statutes, most missed / over-predicted statutes, co-occurring pairs, cardinality"""
    L2I = {l: i for i, l in enumerate(LABELS)}
    Ytr, Ydv = multi_hot(D["train"], L2I), multi_hot(D["dev"], L2I)
    Pb = decide(P_dev, thr)
    f, p, r, tp, fp, fn = f1_per_label(Ydv, Pb)
    freq = Ytr.sum(0); ref = Ydv.sum(0) > 0
    qs = np.percentile(freq[ref], [0, 33, 66, 100])
    for name, lo, hi in (("rare", qs[0], qs[1]), ("medium", qs[1], qs[2]), ("frequent", qs[2], qs[3] + 1)):
        m = ref & (freq >= lo) & (freq < hi) if name != "frequent" else ref & (freq >= lo)
        if m.any():
            log(f"{name:<9} statutes ({int(m.sum())}, {int(lo)}-{int(hi)} training docs): dev macro-F1 {f[m].mean():.4f} | P {p[m].mean():.3f} R {r[m].mean():.3f}")
    order = np.argsort(-(fn))
    log("most missed statutes (id: missed / dev positives, F1): " + ", ".join(
        f"{LABELS[j]}: {int(fn[j])}/{int(Ydv[:, j].sum())} F1 {f[j]:.2f}" for j in order[:top] if Ydv[:, j].sum()))
    order = np.argsort(-(fp))
    log("most over-predicted statutes (id: false positives, F1): " + ", ".join(
        f"{LABELS[j]}: {int(fp[j])} F1 {f[j]:.2f}" for j in order[:top]))
    co = Counter()
    for row in Ytr:
        js = np.where(row)[0]
        for a in range(len(js)):
            for b in range(a + 1, len(js)): co[(js[a], js[b])] += 1
    pairs = co.most_common(top)
    log("most frequent co-occurring statute pairs in training (pair: count | dev: both gold -> both predicted %): " + ", ".join(
        f"{LABELS[a]}+{LABELS[b]}: {c} | {int(((Ydv[:, a] & Ydv[:, b]) & (Pb[:, a] & Pb[:, b])).sum())}/{int((Ydv[:, a] & Ydv[:, b]).sum())}"
        for (a, b), c in pairs))
    k_gold = Ydv.sum(1)
    for k in (1, 2, 3):
        m = (k_gold == k) if k < 3 else (k_gold >= 3)
        if m.any():
            s = scores(Ydv[m], Pb[m])
            log(f"documents with {k if k < 3 else '3+'} gold statutes ({int(m.sum())}): micro-F1 {s['micro_F1']:.4f}, subset acc {s['subset_acc']:.4f}, predicted/doc {s['avg_pred']:.2f}")


def bagged_thresholds(Y, P, g, n_bags=15, seed=0):
    """per-statute thresholds averaged over many random halves of the data (lower variance than one fit)"""
    rng = np.random.RandomState(seed); acc = np.zeros(Y.shape[1])
    for _ in range(n_bags):
        h = rng.choice(len(Y), len(Y) // 2, replace=False)
        acc += per_label_thresholds(Y[h], P[h], g)
    return acc / n_bags


def choose_thresholds3(Y, P, seed=0, log=print):
    """global vs per-statute vs bagged per-statute thresholds, compared by 2-fold cross-fitting on dev"""
    g, fg = global_threshold(Y, P)
    rng = np.random.RandomState(seed); idx = rng.permutation(len(Y)); halves = (idx[: len(Y) // 2], idx[len(Y) // 2:])
    cv = {"global": [], "per_label": [], "bagged": []}
    for a, b in (halves, halves[::-1]):
        ga, _ = global_threshold(Y[a], P[a])
        cv["global"].append(scores(Y[b], decide(P[b], ga))["macro_F1"])
        cv["per_label"].append(scores(Y[b], decide(P[b], per_label_thresholds(Y[a], P[a], ga)))["macro_F1"])
        cv["bagged"].append(scores(Y[b], decide(P[b], bagged_thresholds(Y[a], P[a], ga, seed=seed)))["macro_F1"])
    cvm = {k: float(np.mean(v)) for k, v in cv.items()}
    best = max(cvm, key=cvm.get)
    thr = {"global": lambda: np.full(Y.shape[1], g), "per_label": lambda: per_label_thresholds(Y, P, g),
           "bagged": lambda: bagged_thresholds(Y, P, g, seed=seed)}[best]()
    log(f"thresholds (cross-fitted dev macro-F1): global {cvm['global']:.4f} | per-statute {cvm['per_label']:.4f} | "
        f"bagged per-statute {cvm['bagged']:.4f} -> using {best}")
    return thr, dict(global_thr=g, global_F1=fg, cv=cvm, strategy=best)
