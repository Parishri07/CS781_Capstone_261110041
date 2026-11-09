# Phase 7 (Prior Case Retrieval): data reading, data probes, BM25 baseline. No language model is used here.
#   Micro-F1@7 = 2 * hits / (7 * queries + gold links): with exactly 7 answers per query it only depends on the
#   number of gold cases inside the top 7, so the order inside the top 7 does not change the score.
import json
import re
from collections import Counter

import numpy as np
import scipy.sparse as sp

K = 7
ID_KEYS = ("id", "query_id", "case_id", "doc_id", "candidate_id", "qid", "cid")
TEXT_KEYS = ("text", "sentences", "document", "doc", "content")
GOLD_KEYS = ("relevant_candidates", "relevant", "relevant_ids", "gold", "labels", "citations")
MARK = re.compile(r"<\s*/?\s*citation\s*/?\s*>|\[\s*citation\s*\]", re.I)
WORD = re.compile(r"[a-z0-9]+")
YEAR = re.compile(r"\b(1[89]\d\d|20[0-2]\d)\b")


def as_text(t):
    return [str(x) for x in t] if isinstance(t, list) else [str(t)]


def parse(raw):
    """list or dict of records (or id -> text) -> {id: dict(sents=[...], gold=[ids] or None)}"""
    if isinstance(raw, dict) and len(raw) <= 3:
        for k in ("data", "queries", "candidates", "cases", "documents"):
            if isinstance(raw.get(k), (list, dict)): raw = raw[k]; break
    items = raw.items() if isinstance(raw, dict) else ((None, r) for r in raw)
    out = {}
    for key, r in items:
        if not isinstance(r, dict): r = dict(text=r)
        did = key if key is not None else next(r[k] for k in ID_KEYS if k in r)
        text = next((r[k] for k in TEXT_KEYS if r.get(k)), "")
        gold = next((r[k] for k in GOLD_KEYS if k in r), None)
        out[str(did)] = dict(sents=as_text(text), gold=None if gold is None else [str(g) for g in gold])
    return out


def words(sents):
    return WORD.findall(MARK.sub(" ", " ".join(sents)).lower())


def context(sents, w=1):
    """only the sentences holding a citation marker, +- w sentences (where the query uses its prior cases)"""
    hit = [i for i, s in enumerate(sents) if MARK.search(s)]
    keep = sorted({j for i in hit for j in range(max(0, i - w), min(len(sents), i + w + 1))})
    return [sents[j] for j in keep]


def grams(w, n):
    return w if n == 1 else [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]


def bm25(query_words, doc_words, n, k1=1.2, b=0.75):
    """scores (queries, docs). Only n-grams that occur in some query are indexed, which keeps long n-grams in memory.
    Every distinct query n-gram counts once (long queries are not dominated by their repeated phrases)."""
    vocab = {}
    qr, qc = [], []
    for i, w in enumerate(query_words):
        for g in set(grams(w, n)):
            qr.append(i); qc.append(vocab.setdefault(g, len(vocab)))
    Q = sp.csr_matrix((np.ones(len(qr), dtype=np.float32), (qr, qc)), shape=(len(query_words), len(vocab)))
    dr, dc, dv, dl = [], [], [], []
    for i, w in enumerate(doc_words):
        c = Counter(grams(w, n)); dl.append(sum(c.values()))
        for g, f in c.items():
            j = vocab.get(g)
            if j is not None: dr.append(i); dc.append(j); dv.append(f)
    D = sp.csr_matrix((np.array(dv, dtype=np.float32), (dr, dc)), shape=(len(doc_words), len(vocab)))
    dl = np.maximum(1.0, np.array(dl, dtype=np.float32)); N = len(doc_words)
    df = np.asarray((D > 0).sum(0)).ravel()
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
    norm = k1 * (1 - b + b * dl / dl.mean())
    W = D.tocoo()
    W.data = W.data * (k1 + 1) / (W.data + norm[W.row]) * idf[W.col]
    return np.asarray((Q @ W.tocsr().T).todense(), dtype=np.float32)


def zscore(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-6)


def rank(S, cids, qids, drop_self=True):
    """scores -> {query: top K candidate ids}; a query is never its own answer"""
    out = {}
    for i, q in enumerate(qids):
        o = np.argsort(-S[i], kind="stable")
        out[q] = [cids[j] for j in o if not (drop_self and cids[j] == q)][:K]
    return out


def micro(pred, gold):
    """exactly the leaderboard definition: gold is neither truncated nor padded"""
    hit = sum(len(set(pred[q][:K]) & set(gold[q])) for q in pred)
    p = hit / max(1, sum(len(pred[q][:K]) for q in pred)); r = hit / max(1, sum(len(gold[q]) for q in pred))
    return dict(f1=2 * p * r / max(1e-9, p + r), p=p, r=r, hits=hit)


def per_query_hits(pred, gold):
    return np.array([len(set(pred[q][:K]) & set(gold[q])) for q in pred], dtype=float)


def paired(a, b, gold, n=2000, seed=0):
    """a minus b in micro-F1 with a query-level bootstrap -> (difference, 2.5%, 97.5%)"""
    qs = list(a); ha, hb = per_query_hits(a, gold), per_query_hits(b, {q: gold[q] for q in b})
    ng = np.array([len(gold[q]) for q in qs], dtype=float); rng = np.random.default_rng(seed)
    f = lambda h, g: 2 * h.sum() / (K * len(h) + g.sum())
    d = [f(ha[i], ng[i]) - f(hb[i], ng[i]) for i in (rng.integers(0, len(qs), len(qs)) for _ in range(n))]
    return f(ha, ng) - f(hb, ng), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def years(sents):
    y = [int(m) for m in YEAR.findall(" ".join(sents))]
    return max(y) if y else None


def check_submission(pred, Q, C):
    assert set(pred) == set(Q), "a test query is missing"
    for q, c in pred.items():
        assert len(c) == K and len(set(c)) == K and all(x in C for x in c), q


def write_submission(pred, path_zip):
    import os, zipfile
    d = os.path.dirname(path_zip) or "."; pj = os.path.join(d, "prediction.json")
    json.dump(pred, open(pj, "w"));
    with zipfile.ZipFile(path_zip, "w", zipfile.ZIP_DEFLATED) as z: z.write(pj, arcname="prediction.json")
    return path_zip
