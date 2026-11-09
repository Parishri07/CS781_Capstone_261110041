# Phase 7 (Prior Case Retrieval), model-only retrieval: every score comes from a Phase-1 language model.
#   Documents are cut into short overlapping token windows; a window is the mean of the model's hidden states. A query
#   is represented by the windows around its citation markers (where it uses a prior case, often quoting it) or by all
#   its windows. Candidate score for one query window = best-matching window of that candidate; a passage quoted
#   word for word should give nearly the same embedding in both documents.
import re
import time

import numpy as np
import torch
import torch.nn as nn

import ner_model as nm
import pcr_lex as lx
import sm

STOP = r"(?=\bACT:|\bHEADNOTE:|(?<!OF )\bJUDGMENT:|\Z)"
HEAD = re.compile(r"(?:PETITIONER|APPELLANT):.{0,2500}?" + STOP, re.S)
META = re.compile(r"\bCITATOR INFO\s*:.{0,2500}?" + STOP + r"|\bCITATION:.{0,800}?" + STOP, re.S)
HOLD = re.compile(r"<[A-Z_]+>")
HYPH = re.compile(r"(?<=[a-z])-\s+(?=[a-z])")
DATE = re.compile(r"DATE OF JUDGMENT\s*:?\s*\d{1,2}[/.-]\d{1,2}[/.-](\d{4})")


def clean(sents):
    """drops the report header (parties, bench, own citations, citator list of LATER citing cases): it is metadata about
    the document, not its legal content. Citation markers stay; anonymised <NAME>/<ORG> tokens and line-break hyphens go."""
    t = " ".join(sents)
    t = META.sub(" ", HEAD.sub(" ", t))
    t = lx.MARK.sub(" \x00 ", t)
    return HYPH.sub("", HOLD.sub(" ", t))


def judgment_year(sents):
    m = DATE.search(" ".join(sents)[:3000])
    return int(m.group(1)) if m else None


def body_year(sents):
    y = [int(v) for v in lx.YEAR.findall(clean(sents))]
    return max(y) if y else None


def doc_ids(tok, sents, cap=60000):
    """-> token ids, positions of the citation markers"""
    ids, marks = [], []
    for i, p in enumerate(clean(sents).split("\x00")):
        if i: marks.append(len(ids))
        if p.strip(): ids += tok.enc(p)
        if len(ids) >= cap: break
    return ids[:cap], [m for m in marks if m < cap]


def spread(xs, cap):
    if len(xs) <= cap: return xs
    return [xs[int(i)] for i in np.linspace(0, len(xs) - 1, cap)]


def sliding(ids, L, cap):
    if len(ids) <= L: return [ids] if ids else [[sm.EOS + 1]]
    st = list(range(0, len(ids) - L + 1, L // 2))
    if st[-1] != len(ids) - L: st.append(len(ids) - L)
    return [ids[s:s + L] for s in spread(st, cap)]


def citation_windows(ids, marks, L):
    """-> windows, marker number of each. Before / at / after the marker: quotes usually follow it"""
    if not marks: return sliding(ids, L, 30), [0] * len(sliding(ids, L, 30))
    out, grp, last = [], [], -10 ** 9; g = -1
    hi = max(0, len(ids) - L)
    for p in marks:
        if p - last < L // 4: continue
        last = p; g += 1
        for s in sorted({min(hi, max(0, p - L // 2)), min(hi, p), min(hi, p + L)}):
            out.append(ids[s:s + L]); grp.append(g)
    return out, grp


class Enc(nn.Module):
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
        return h.float().mean(1)


@torch.no_grad()
def encode(run, windows, device, bs):
    """windows (token lists) -> L2-normalised embeddings. Equal lengths are batched, so no padding is ever fed to a model"""
    order = sorted(range(len(windows)), key=lambda i: len(windows[i]))
    out = [None] * len(order); i = 0; res = []
    while i < len(order):
        n = len(windows[order[i]]); j = i
        while j < len(order) and j - i < bs and len(windows[order[j]]) == n: j += 1
        x = torch.tensor([windows[k] for k in order[i:j]], dtype=torch.long, device=device)
        res.append(run(x).cpu()); i = j
    E = torch.cat(res)
    back = torch.empty(len(order), dtype=torch.long); back[torch.tensor(order)] = torch.arange(len(order))
    return E[back]


def unit(E):
    return E / E.norm(dim=-1, keepdim=True).clamp_min(1e-6)


@torch.no_grad()
def doc_sims(Qe, Ce, cown, D, device, step=256):
    """(windows, docs): for each query window, the best window of each document"""
    Ce = Ce.to(device); cown = cown.to(device); out = []
    for i in range(0, Qe.size(0), step):
        s = Qe[i:i + step].to(device) @ Ce.T
        x = torch.full((s.size(0), D), -1.0, device=device)
        out.append(x.scatter_reduce(1, cown[None].expand_as(s), s, "amax", include_self=True).cpu())
    return torch.cat(out)


def rrf(S, k=10, top=100):
    """(rows, docs) -> sum over rows of 1 / (k + rank), counted for the top ranks of each row"""
    n = min(top, S.size(1)); idx = S.topk(n, dim=1).indices
    out = torch.zeros(S.size(1)); w = 1.0 / (k + 1 + torch.arange(n, dtype=torch.float))
    out.scatter_add_(0, idx.flatten(), w.repeat(S.size(0)))
    return out


def aggregate(S, grp):
    """three ways of turning (query windows, docs) into one score per doc. Each citation marker points at a different
    case, so one strong quote should not swamp the others: 'cite' lets each marker vote on its own."""
    g = torch.tensor(grp); G = torch.stack([S[g == i].max(0).values for i in g.unique()])
    return dict(sum=S.sum(0), win=rrf(S), cite=rrf(G))


def evaluate(E_q, grp_q, E_c, own_c, qids, cids, gold, device):
    """-> {aggregation: metrics} (a query is never its own answer)"""
    res = {}
    for q, (Qe, g) in zip(qids, zip(E_q, grp_q)):
        a = aggregate(doc_sims(Qe, E_c, own_c, len(cids), device), g)
        for k, v in a.items(): res.setdefault(k, []).append(v.numpy())
    return {k: lx.micro(lx.rank(np.stack(v), cids, qids), gold) for k, v in res.items()}


def probe(arch, root, Q, C, gold, device, Ls, bs_tokens=16000, cap=800, log=print, gpus=1):
    """one Phase-1 model, unmodified: dev micro-F1@7 for each window size / query view / centring / aggregation"""
    t0 = time.time()
    tok = sm.Tok(arch, root); m = sm.load_lm(arch, root).to(device).eval()
    run = Enc(m, arch)
    if gpus > 1: run = nn.DataParallel(run)
    qids, cids = list(Q), list(C)
    qd = [doc_ids(tok, Q[q]["sents"]) for q in qids]; cd = [doc_ids(tok, C[c]["sents"]) for c in cids]
    log(f"{arch}: tokenised {sum(len(i) for i, _ in cd):,} candidate tokens in {(time.time() - t0) / 60:.1f} min")
    rows = []
    for L in Ls:
        t1 = time.time(); w, own = [], []
        for d, (ids, _) in enumerate(cd):
            s = sliding(ids, L, cap); w += s; own += [d] * len(s)
        Ec = encode(run, w, device, max(8, bs_tokens // L)); own = torch.tensor(own); del w
        views = {}
        cit = [citation_windows(i, mk, L) for i, mk in qd]
        views["citation windows"] = ([encode(run, x, device, 256) for x, _ in cit], [g for _, g in cit])
        al = [sliding(i, L, 200) for i, _ in qd]
        views["all windows"] = ([encode(run, x, device, 256) for x in al], [list(range(len(x))) for x in al])
        log(f"  L={L}: {Ec.size(0):,} candidate windows encoded in {(time.time() - t1) / 60:.1f} min")
        mu = Ec.mean(0, keepdim=True)
        for vn, (Eq, grp) in views.items():
            for centre in (False, True):
                f = (lambda E: unit(E - mu)) if centre else unit
                r = evaluate([f(e) for e in Eq], grp, f(Ec), own, qids, cids, gold, device)
                for a, v in r.items():
                    rows.append(dict(arch=arch, L=L, view=vn, centred=centre, agg=a, **v))
                    log(f"  L={L:<4} {vn:<17} centred={str(centre):<5} {a:<4} dev F1 {v['f1']:.4f} (P {v['p']:.4f} R {v['r']:.4f})")
        del Ec, views
    del m, run; torch.cuda.empty_cache()
    log(f"{arch}: best {max(rows, key=lambda r: r['f1'])} | {(time.time() - t0) / 60:.1f} min")
    return rows
