# Phase 7: fine-tuning a Phase-1 model for retrieval. Model-only: no BM25 or other ranker at training or test time.
#   Round: (1) with the current model, find for each sampled training query the window pairs that best link it to its
#   gold cases (the labels name cases, not passages) and the best-matching windows of NON-gold cases (hard negatives);
#   (2) train on those pairs with a contrastive loss. A case that is gold for the query is never used as its negative.
#   Knowledge distillation: the student also matches the teacher's similarity distribution over each batch.
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ner_model as nm
import pcr_lex as lx
import pcr_model as pm
import sm


def find_weights(run, inputs="/kaggle/input", work="/kaggle/working"):
    """p7_trainable_{run}.pt among the attached inputs, also if it arrived renamed .zip or was unpacked by Kaggle
    (a torch file is a zip archive; a folder is re-packed uncompressed). -> path or None"""
    import glob, zipfile
    name = f"p7_trainable_{run}"
    for ext in (".pt", ".zip"):
        w = sorted(glob.glob(f"{inputs}/**/{name}{ext}", recursive=True))
        if w: return w[0]
    pk = sorted(glob.glob(f"{inputs}/**/{name}/data.pkl", recursive=True))
    if not pk:
        print("NOT FOUND:", name, "| p7_trainable files attached:", sorted(glob.glob(f"{inputs}/**/p7_trainable_*", recursive=True)) or "none")
        return None
    src = os.path.dirname(pk[0]); dst = os.path.join(work, name + ".pt")
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as z:
        for dp, _, fs in os.walk(src):
            for f in fs:
                p = os.path.join(dp, f); zi = zipfile.ZipInfo(os.path.join(name, os.path.relpath(p, src)).replace(os.sep, "/"), (1980, 1, 1, 0, 0, 0))
                z.writestr(zi, open(p, "rb").read(), zipfile.ZIP_STORED)
    return dst


def make_model(cfg, root):
    """start from the Phase-1 weights; 'partial' trains only the top blocks and the final norm"""
    method = "full_ft" if cfg["method"] == "partial" else cfg["method"]
    m, desc = sm.build(dict(cfg, method=method), root)
    if cfg["method"] == "partial":
        k = cfg.get("top_blocks", 4); n = len(m.blocks); keep = tuple(f"blocks.{i}." for i in range(n - k, n)) + tuple(cfg.get("final_norm", ()))
        for name, p in m.named_parameters(): p.requires_grad = name.startswith(keep)
        desc = f"partial fine-tuning (top {k} of {n} blocks + final norm)"
    return m, desc


def tokenise(arch, root, split):
    tok = sm.Tok(arch, root)
    return {k: pm.doc_ids(tok, r["sents"]) for k, r in split.items()}


@torch.no_grad()
def embed_docs(run, ids, keys, L, cap, device):
    """-> embeddings (unit length), owner index of each window"""
    w, own = [], []
    for d, k in enumerate(keys):
        s = pm.sliding(ids[k][0], L, cap); w += s; own += [d] * len(s)
    return pm.unit(pm.encode(run, w, device, max(8, 16000 // L))), torch.tensor(own)


@torch.no_grad()
def scores(run, qd, cd, qids, cids, cfg, device):
    """(queries, docs) score: reciprocal-rank vote of every query window (the best aggregation in P7-1)"""
    run.eval(); L = cfg["L"]
    Ec, own = embed_docs(run, cd, cids, L, 800, device); out = []
    for q in qids:
        Qe = pm.unit(pm.encode(run, pm.sliding(qd[q][0], L, 200), device, 256))
        S = pm.doc_sims(Qe, Ec, own, len(cids), device)
        if cfg.get("z"):                               # only window matches that stand out from the pool count (best rule in P7-B)
            Z = (S - S.mean(1, keepdim=True)) / (S.std(1, keepdim=True) + 1e-6)
            out.append(((Z - cfg["z"]).clamp_min(0).sum(0) + 1e-6 * pm.rrf(S)).numpy())
        else:
            out.append(pm.rrf(S).numpy())
    return np.stack(out)


def evaluate(run, qd, cd, gold, cfg, device):
    qids, cids = list(qd), list(cd); t0 = time.time()
    r = lx.micro(lx.rank(scores(run, qd, cd, qids, cids, cfg, device), cids, qids), gold)
    r["seconds"] = round(time.time() - t0, 1)
    return r


@torch.no_grad()
def mine(run, qd, cd, gold, cfg, device, rng):
    """-> training items dict(q=window, p=window, n=[windows], qid, pdoc, ndocs)"""
    run.eval(); L = cfg["L"]
    qs = rng.sample([q for q in gold if gold[q]], min(cfg["n_queries"], sum(1 for q in gold if gold[q])))
    keys = list(dict.fromkeys([x for q in qs for x in gold[q] if x in cd] + rng.sample(list(cd), min(cfg["n_docs"], len(cd)))))
    w, own = [], []
    for d, k in enumerate(keys):
        s = [x for x in pm.sliding(cd[k][0], L, cfg["doc_cap"]) if len(x) == L]; w += s; own += [d] * len(s)
    Ec = pm.unit(pm.encode(run, w, device, max(8, 16000 // L))).to(device); own_t = torch.tensor(own, device=device)
    pos = {k: i for i, k in enumerate(keys)}; items = []
    for q in qs:
        qw = [x for x in pm.sliding(qd[q][0], L, cfg["query_cap"]) if len(x) == L]
        if not qw: continue
        S = pm.unit(pm.encode(run, qw, device, 256)).to(device) @ Ec.T
        bad = torch.zeros(len(keys), dtype=torch.bool, device=device)
        for x in list(gold[q]) + [q]:
            if x in pos: bad[pos[x]] = True
        mask = bad[own_t]; Sn = S.masked_fill(mask, -2.0)
        top = Sn.topk(min(50, Sn.size(1)), dim=1).indices.cpu().tolist()
        for g in gold[q]:
            if g not in pos: continue
            idx = (own_t == pos[g]).nonzero().flatten()
            if idx.numel() == 0: continue
            sub = S[:, idx]; n = min(cfg["pairs_per_gold"], sub.numel()); best = sub.flatten().topk(n).indices
            for b in best.tolist():
                r, c = divmod(b, sub.size(1)); seen, neg = set(), []
                for t in top[r]:
                    d = own[t]
                    if d not in seen: seen.add(d); neg.append(t)
                    if len(neg) == cfg["negs"]: break
                if len(neg) == cfg["negs"]:
                    items.append(dict(q=qw[r], p=w[int(idx[c])], n=[w[t] for t in neg], qid=q, pdoc=g, ndocs=[keys[own[t]] for t in neg]))
    return items


def batch_loss(run, teacher, batch, gold, cfg, device):
    """anchors = query windows; candidates = every positive and negative window in the batch"""
    B = len(batch); nn_ = cfg["negs"]
    x = torch.tensor([b["q"] for b in batch] + [b["p"] for b in batch] + [t for b in batch for t in b["n"]], device=device)
    E = F.normalize(run(x), dim=-1); A, C = E[:B], E[B:]
    docs = [b["pdoc"] for b in batch] + [d for b in batch for d in b["ndocs"]]
    ban = torch.zeros(B, len(docs), dtype=torch.bool, device=device)
    for i, b in enumerate(batch):
        g = set(gold[b["qid"]]) | {b["qid"]}
        for j, d in enumerate(docs):
            if d in g and j != i: ban[i, j] = True
    lg = (A @ C.T / cfg["tau"]).masked_fill(ban, -1e4)
    ce = F.cross_entropy(lg, torch.arange(B, device=device)); kd = torch.zeros((), device=device)
    if teacher is not None:
        with torch.no_grad():
            Et = F.normalize(teacher(x), dim=-1); lt = (Et[:B] @ Et[B:].T / cfg["tau"]).masked_fill(ban, -1e4)
        kd = F.kl_div(F.log_softmax(lg, -1), F.softmax(lt, -1), reduction="batchmean")
    return ce, kd


def run_experiment(cfg, data, root, device, out, log=print):
    """data: dict(train=(Q, C), dev=(Q, C), test=(Q, C)) -> record; best-dev weights and a test zip go to out/<run>/"""
    t_all = time.time(); d = os.path.join(out, cfg["run"]); os.makedirs(d, exist_ok=True)
    rng = random.Random(cfg.get("seed", 1337)); torch.manual_seed(cfg.get("seed", 1337))
    gpus = torch.cuda.device_count()
    arch = cfg["arch"]; T = {s: (tokenise(arch, root, q), tokenise(arch, root, c)) for s, (q, c) in data.items()}
    gold = {s: {k: r["gold"] for k, r in data[s][0].items()} for s in ("train", "dev")}
    m, desc = make_model(cfg, root); m.to(device)
    total = sum(p.numel() for p in m.parameters()); params = [p for p in m.parameters() if p.requires_grad]; trainable = sum(p.numel() for p in params)
    log(f"\n=== {cfg['run']} | {arch} | {desc} | params {total:,}, trainable {trainable:,} ({trainable / total:.2%})")
    enc = pm.Enc(m, arch); run = nn.DataParallel(enc) if gpus > 1 else enc
    teacher = None
    if cfg.get("teacher_weights"):
        tm, tdesc = make_model(cfg["teacher"], root); st = torch.load(cfg["teacher_weights"], map_location="cpu", weights_only=False)
        _, unexpected = tm.load_state_dict(st, strict=False); assert not unexpected and st, unexpected[:5]
        for p in tm.parameters(): p.requires_grad = False
        tm.to(device).eval(); t0_ = sum(p.numel() for p in tm.parameters()); teacher = pm.Enc(tm, cfg["teacher"]["arch"])
        teacher = nn.DataParallel(teacher) if gpus > 1 else teacher
        log(f"teacher: {tdesc} | {t0_:,} params | KD weight {cfg['kd_alpha']}")
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01))
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    base = evaluate(run, *T["dev"], gold["dev"], cfg, device)
    log(f"start (no fine-tuning): dev F1 {base['f1']:.4f} (P {base['p']:.4f} R {base['r']:.4f}) | {base['seconds']:.0f} s")
    best, best_state, hist, step = base["f1"], sm.trainable_state(m), [dict(round=0, dev=base)], 0
    if base["f1"] > 0: torch.save(best_state, os.path.join(d, f"p7_trainable_{cfg['run']}.pt"))
    for r in range(1, cfg["rounds"] + 1):
        t0 = time.time(); items = mine(run, *T["train"], gold["train"], cfg, device, rng); t1 = time.time()
        rng.shuffle(items); run.train(); lr = cfg["lr"] * (1 - (r - 1) / cfg["rounds"]); ls, lk, n = 0.0, 0.0, 0
        for i in range(0, len(items) - cfg["batch"] + 1, cfg["batch"]):
            for g in opt.param_groups: g["lr"] = lr * min(1.0, (step + 1) / 30)
            ce, kd = batch_loss(run, teacher, items[i:i + cfg["batch"]], gold["train"], cfg, device)
            a = cfg.get("kd_alpha", 0.0) if teacher is not None else 0.0
            scaler.scale((1 - a) * ce + a * kd).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            ls += float(ce); lk += float(kd); n += 1; step += 1
        msg = f"round {r}/{cfg['rounds']}: {len(items):,} pairs, {n} steps | loss {ls / max(1, n):.4f}" + (f" KD {lk / max(1, n):.4f}" if teacher is not None else "") + f" | mine {(t1 - t0) / 60:.1f} min, train {(time.time() - t1) / 60:.1f} min"
        if r % cfg["eval_every"] == 0 or r == cfg["rounds"]:
            e = evaluate(run, *T["dev"], gold["dev"], cfg, device); hist.append(dict(round=r, dev=e, loss=ls / max(1, n)))
            msg += f" | dev F1 {e['f1']:.4f} (P {e['p']:.4f} R {e['r']:.4f})"
            if e["f1"] > best:
                best, best_state = e["f1"], sm.trainable_state(m); msg += "  <- best"
                torch.save(best_state, os.path.join(d, f"p7_trainable_{cfg['run']}.pt"))
        log(msg)
    m.load_state_dict(best_state, strict=False)
    qt, ct = T["test"]; qids, cids = list(qt), list(ct); t0 = time.time()
    pred = lx.rank(scores(run, qt, ct, qids, cids, cfg, device), cids, qids); tt = time.time() - t0
    lx.check_submission(pred, data["test"][0], data["test"][1])
    zp = lx.write_submission(pred, os.path.join(d, f"submission_{cfg['run']}.zip"))
    rec = dict(run=cfg["run"], arch=arch, method=desc, params_total=total, params_trainable=trainable, start_dev=base, best_dev_f1=best, history=hist,
               test_seconds=round(tt, 1), peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0,
               minutes=round((time.time() - t_all) / 60, 1), zip=zp, cfg={k: v for k, v in cfg.items() if k != "teacher"})
    json.dump(rec, open(os.path.join(d, "record.json"), "w"), indent=1, default=str)
    log(f"{cfg['run']}: best dev F1 {best:.4f} (start {base['f1']:.4f}) | test zip {zp} | {rec['minutes']:.0f} min")
    del m, run, enc, teacher; torch.cuda.empty_cache()
    return rec
