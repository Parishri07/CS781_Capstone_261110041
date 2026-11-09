# Phase 7: better use of trained Phase-1 models, no training. Every score still comes from a Phase-1 model.
#   View = one way of embedding text with one model: windows (each window run through the model on its own) or phrases
#   (late interaction: hidden states of a chunk averaged over a few tokens, which keeps the surrounding context; near-copied
#   wording gives near-identical phrase vectors, the model's own version of long-phrase matching).
#   A view gives (query rows, docs) best-match similarities; an aggregation turns those into doc scores; views are fused.
import random
import time

import numpy as np
import torch
import torch.nn as nn

import ner_model as nm
import pcr_lex as lx
import pcr_model as pm
import pcr_train as pt
import sm


class EncH(nn.Module):
    """all hidden states (B,T,d)"""

    def __init__(self, m, arch):
        super().__init__()
        self.m, self.arch = m, arch

    def forward(self, ids):
        with nm.autocast():
            if self.arch == "arch2_encdec":
                h = self.m.encode(ids)
            else:
                plen = torch.full((ids.size(0),), ids.size(1), device=ids.device) if self.arch == "arch3_prefixlm" else ids.size(1)
                h = sm.hidden_dec(self.m, self.arch, ids, plen)
        return h.float()


def wrap(mod):
    return nn.DataParallel(mod) if torch.cuda.device_count() > 1 else mod


def load(spec, root, device):
    """spec: dict(cfg=model config, weights=path or None) -> model in eval mode"""
    m, _ = pt.make_model(spec["cfg"], root)
    if spec.get("weights"):
        st = torch.load(spec["weights"], map_location="cpu", weights_only=False)
        _, unexpected = m.load_state_dict(st, strict=False); assert not unexpected and st, unexpected[:5]
    return m.to(device).eval()


def chunks(ids, C, cap):
    s = [ids[i:i + C] for i in range(0, max(1, len(ids)), C - 16)] if len(ids) > C else [ids or [sm.EOS + 1]]
    s = [x for x in s if len(x) >= 16 or len(s) == 1]
    return pm.spread(s, cap)


@torch.no_grad()
def phrases(run, docs, P, step, device, C=128, cap=800, bs=120):
    """docs: list of token lists -> phrase vectors, owner index. One forward pass per chunk; phrases are local averages"""
    w, own = [], []
    for d, ids in enumerate(docs):
        s = chunks(ids, C, cap); w += s; own += [d] * len(s)
    order = sorted(range(len(w)), key=lambda i: len(w[i])); res, ro = [], []; i = 0
    while i < len(order):
        n = len(w[order[i]]); j = i
        while j < len(order) and j - i < bs and len(w[order[j]]) == n: j += 1
        x = torch.tensor([w[k] for k in order[i:j]], dtype=torch.long, device=device)
        h = run(x)
        v = h.unfold(1, min(P, n), step).mean(-1)
        res.append(v.flatten(0, 1).cpu().half()); ro += [own[k] for k in order[i:j] for _ in range(v.size(1))]; i = j
    return torch.cat(res), torch.tensor(ro)


def project(E, dim, seed=0):
    g = torch.Generator().manual_seed(seed); R = torch.randn(E.size(-1), dim, generator=g) / dim ** 0.5
    out = [E[i:i + 200000].float() @ R for i in range(0, E.size(0), 200000)]
    return pm.unit(torch.cat(out)).half()


@torch.no_grad()
def view_sims(view, models, qd, cd, device, log=print):
    """view: dict(model=name, kind='win'/'phr', L or P) -> list of (rows, docs) half tensors, one per query"""
    t0 = time.time(); m, arch = models[view["model"]]; qids, cids = list(qd), list(cd); D = len(cids)
    if view["kind"] == "win":
        run = wrap(pm.Enc(m, arch)); L = view["L"]
        Ec, own = pt.embed_docs(run, cd, cids, L, 800, device)
        Qs = [pm.unit(pm.encode(run, pm.sliding(qd[q][0], L, 200), device, 256)) for q in qids]
    else:
        run = wrap(EncH(m, arch)); P, st = view["P"], view["step"]
        Ec, own = phrases(run, [cd[c][0] for c in cids], P, st, device); Ec = project(Ec, view.get("dim", 128))
        Qs = []
        for q in qids:
            e, _ = phrases(run, [qd[q][0]], P, st, device); e = project(e, view.get("dim", 128))
            Qs.append(e[torch.linspace(0, e.size(0) - 1, min(400, e.size(0))).long()].float())
    Ec = Ec.float(); out = [pm.doc_sims(q, Ec, own, D, device, step=64).half() for q in Qs]
    log(f"  {name_of(view)}: {Ec.size(0):,} vectors | {(time.time() - t0) / 60:.1f} min")
    del Ec; torch.cuda.empty_cache()
    return out


def name_of(v):
    return f"{v['model']}_{'L' + str(v['L']) if v['kind'] == 'win' else 'P' + str(v['P'])}"


def zsum(S, t):
    Z = (S - S.mean(1, keepdim=True)) / (S.std(1, keepdim=True) + 1e-6)
    return (Z - t).clamp_min(0).sum(0)


AGGS = {f"rrf k{k} top{n}": (lambda S, k=k, n=n: pm.rrf(S, k, n)) for k in (3, 10, 30) for n in (30, 100, 300)}
AGGS.update({f"z over {t}": (lambda S, t=t: zsum(S, t)) for t in (2.0, 3.0, 4.0, 6.0)})
AGGS.update({f"soft b{b}": (lambda S, b=b: torch.softmax(S.double() * b, 1).sum(0)) for b in (30, 100, 300)})


def apply(agg, sims):
    return np.stack([AGGS[agg](s.float()).numpy() for s in sims])


def zq(S):
    """scores -> reciprocal ranks. Fusing ranks keeps each view's order exactly: z-scores flatten the tiny values of the
    soft rule into ties, which silently reorders its top candidates"""
    r = np.empty(S.shape, dtype=np.float64); o = np.argsort(-S, axis=1, kind="stable")
    np.put_along_axis(r, o, 1.0 / (20.0 + np.arange(S.shape[1], dtype=np.float64))[None], 1)
    return r


def f1_of(S, cids, qids, gold):
    return lx.micro(lx.rank(S, cids, qids), {q: gold[q] for q in qids})


def greedy(views, cids, qids, gold, sel, min_gain=2):
    """forward selection on the query indices sel; a view must add at least min_gain hits"""
    Q = [qids[i] for i in sel]; keys = sorted(views, key=lambda k: -f1_of(views[k][sel], cids, Q, gold)["hits"]); ch = keys[:1]
    best = f1_of(views[ch[0]][sel], cids, Q, gold)["hits"]
    while True:
        tr = [(f1_of(sum(views[k][sel] for k in ch + [v]), cids, Q, gold)["hits"], v) for v in keys if v not in ch]
        if not tr or max(tr)[0] < best + min_gain: return ch
        best, v = max(tr); ch.append(v)


def crossfit(views, cids, qids, gold, seed=0):
    """honest estimate: choose the views on one half of dev, score on the other half, both ways"""
    idx = list(range(len(qids))); random.Random(seed).shuffle(idx); h = len(idx) // 2; preds = {}; picks = []
    for a, b in ((idx[:h], idx[h:]), (idx[h:], idx[:h])):
        ch = greedy(views, cids, qids, gold, a); picks.append(ch)
        preds.update(lx.rank(sum(views[k][b] for k in ch), cids, [qids[i] for i in b]))
    return lx.micro(preds, gold), picks


# ---- P7-C: wider rule search. Row z-score = how much a match stands out among the pool for that query window;
#      doc z-score (dz) = how much it stands out for that document, so long or generic cases that match everything
#      lose their advantage; a = a window matching few documents weighs more than one matching many (the idf idea).
RULES = {f"z{t} p{p} a{a} {'dz' if d else 'raw'}": dict(t=t, p=p, a=a, dz=d) for t in (2.0, 2.5, 3.0, 3.5) for p in (1, 2) for a in (0, 0.5, 1) for d in (False, True)}


def doc_stats(sims):
    """per-document mean and spread of the best-match similarity over all query rows of a split (no labels used)"""
    n, s1, s2 = 0, 0.0, 0.0
    for s in sims:
        x = s.float().double(); n += x.size(0); s1 = s1 + x.sum(0); s2 = s2 + (x ** 2).sum(0)
    m = s1 / n
    return m.float(), (s2 / n - m ** 2).clamp_min(1e-10).sqrt().float()


def rule_score(S, r, st):
    X = (S - st[0]) / st[1] if r["dz"] else S
    Z = (X - X.mean(1, keepdim=True)) / (X.std(1, keepdim=True) + 1e-6)
    v = (Z - r["t"]).clamp_min(0) ** r["p"]
    if r["a"]: v = v / (v.sum(1, keepdim=True) + 1e-6) ** r["a"]
    return v.sum(0) + 1e-6 * torch.softmax(S.double() * 30, 1).sum(0).float()      # the small term only breaks ties


def apply_rule(name, sims, st):
    return np.stack([rule_score(s.float(), RULES[name], st).numpy() for s in sims])


def search_rules(sims, cids, qids, gold, log=print):
    """-> best rule of each view, rank-transformed doc scores, table rows"""
    rows, best, Z = [], {}, {}
    for v, s in sims.items():
        st = doc_stats(s); r = sorted(((f1_of(apply_rule(a, s, st), cids, qids, gold)["f1"], a) for a in RULES), reverse=True)
        best[v] = r[0][1]; Z[v] = zq(apply_rule(r[0][1], s, st)); rows.append((v, r[0][0], r[0][1]))
        assert abs(f1_of(Z[v], cids, qids, gold)["f1"] - r[0][0]) < 1e-9, "fusion changed the order of a view"
        log(f"  {v:<10} best {r[0][0]:.4f} ({r[0][1]}) | second {r[1][0]:.4f} ({r[1][1]}) | median {r[len(r) // 2][0]:.4f}")
    return best, Z, rows


def search(sims, cids, qids, gold, log=print):
    """-> best aggregation of each view, z-scored doc scores, table rows"""
    rows, best, Z = [], {}, {}
    for v, s in sims.items():
        r = sorted(((f1_of(apply(a, s), cids, qids, gold)["f1"], a) for a in AGGS), reverse=True)
        best[v] = r[0][1]; Z[v] = zq(apply(r[0][1], s)); rows.append((v, r[0][0], r[0][1]))
        assert abs(f1_of(Z[v], cids, qids, gold)["f1"] - r[0][0]) < 1e-9, "fusion changed the order of a view"
        log(f"  {v:<14} best {r[0][0]:.4f} ({r[0][1]}) | second {r[1][0]:.4f} ({r[1][1]}) | worst {r[-1][0]:.4f} ({r[-1][1]})")
    return best, Z, rows
