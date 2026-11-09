# Phase 3 (Rhetorical Role prediction) - hierarchical multi-task model on top of a Phase-1 backbone.
#   sentences -> (packed into windows, each token sees its neighbouring sentences) -> Phase-1 backbone (+ PEFT)
#   -> mean-pooled sentence vectors (+ relative position + domain) -> document BiLSTM
#   -> role head (13 labels) + label-shift head (does the role change from the previous sentence?)
# Techniques: full FT / LoRA / rsLoRA / LoRMA / IA3 / BitFit / ... and knowledge distillation (soft role targets).
import json
import math
import os
import random
import time
import zipfile
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

import ner_model as nm

# ======================================================================================================
# 1. Data
# ======================================================================================================
_ID = ("id", "doc_id", "document_id", "docid", "name")


def parse_docs(raw):
    """RR json in any common shape -> list of dict(id, text=[sentences], labels=[...] or None)"""
    if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], (list, dict)):
        raw = raw["data"]
    items = []
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                items.append(dict(v, id=v.get("id", k)))
            else:                                                   # id -> list of sentences
                items.append(dict(id=k, text=v))
    else:
        items = list(raw)
    docs = []
    for i, d in enumerate(items):
        did = next((d[k] for k in _ID if k in d), str(i))
        text = d.get("text") or d.get("sentences") or d.get("sents")
        if isinstance(text, str):
            text = [text]
        labels = d.get("labels") or d.get("label") or d.get("roles")
        docs.append(dict(id=str(did), text=[str(s) for s in text], labels=list(labels) if labels else None))
    return docs


def parse_labels(raw):
    if isinstance(raw, dict):
        if "labels" in raw:
            return list(raw["labels"])
        if all(isinstance(v, int) for v in raw.values()):
            return [k for k, _ in sorted(raw.items(), key=lambda kv: kv[1])]
        return list(raw)
    return list(raw)


def encode_doc(tok, sentences, max_sent_tokens):
    """-> list of token-id lists, one per sentence (never empty, at most max_sent_tokens)"""
    space = getattr(tok, "space_prefix", False)
    out = []
    for s in sentences:
        s = " ".join(s.split())
        ids = tok.encode((" " + s) if space else s) if s else []
        if not space:
            ids = tok.encode(" ") + ids                             # A1 tokenizer: explicit separating space
        out.append((ids or tok.encode(" ."))[:max_sent_tokens])
    return out


_TOK = None


def lex_tokens(s, ngrams=2):
    """lower-cased word unigrams (+ bigrams); numbers -> <num>"""
    global _TOK
    if _TOK is None:
        import re
        _TOK = re.compile(r"[a-z]+|\d+")
    w = ["<num>" if t.isdigit() else t for t in _TOK.findall(s.lower())]
    return w + ([a + "_" + b for a, b in zip(w, w[1:])] if ngrams >= 2 else [])


def lex_vocab(docs, max_features=30000, min_df=3, ngrams=2):
    """n-gram vocabulary + idf from the TRAINING sentences only"""
    df, n = Counter(), 0
    for d in docs:
        for s in d["text"]:
            df.update(set(lex_tokens(s, ngrams))); n += 1
    keep = [f for f, c in df.most_common() if c >= min_df][:max_features]
    return dict(index={f: i for i, f in enumerate(keep)}, idf=[math.log((1 + n) / (1 + df[f])) + 1 for f in keep],
                ngrams=ngrams, key=(max_features, min_df, ngrams, n))


def lex_encode(vocab, s):
    """sentence -> (feature ids, l2-normalised sublinear tf-idf weights)"""
    c = Counter(f for f in lex_tokens(s, vocab["ngrams"]) if f in vocab["index"])
    ids = [vocab["index"][f] for f in c]
    w = [(1 + math.log(c[f])) * vocab["idf"][vocab["index"][f]] for f in c]
    norm = math.sqrt(sum(x * x for x in w)) or 1.0
    return ids, [x / norm for x in w]


def lex_batch(it, s0, s1, device):
    """bag of n-grams of sentences s0..s1-1 of a prepared document -> EmbeddingBag inputs"""
    ids, offs, ws = [], [], []
    for i in range(s0, s1):
        a, b = it["lex"][i]; offs.append(len(ids)); ids += a; ws += b
    return (torch.tensor(ids, dtype=torch.long, device=device), torch.tensor(offs, dtype=torch.long, device=device),
            torch.tensor(ws, dtype=torch.float, device=device))


def pack_windows(sent_ids, size):
    """greedy: consecutive whole sentences into windows of <= size tokens -> list of (first, last+1) sentence ranges"""
    wins, start, n = [], 0, 0
    for i, ids in enumerate(sent_ids):
        if n + len(ids) > size and i > start:
            wins.append((start, i)); start, n = i, 0
        n += len(ids)
    wins.append((start, len(sent_ids)))
    return wins


def window_batch(sent_ids, wins, device, pad_id=0):
    """windows -> ids (B,T), seg (B,T) slot of the sentence each token belongs to (-1 = padding), smask (B,S)"""
    B = len(wins)
    T = max(sum(len(sent_ids[i]) for i in range(a, b)) for a, b in wins)
    S = max(b - a for a, b in wins)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    seg = torch.full((B, T), -1, dtype=torch.long)
    smask = torch.zeros(B, S, dtype=torch.bool)
    for r, (a, b) in enumerate(wins):
        p = 0
        for k, i in enumerate(range(a, b)):
            L = len(sent_ids[i])
            ids[r, p:p + L] = torch.tensor(sent_ids[i]); seg[r, p:p + L] = k; p += L
            smask[r, k] = True
    return ids.to(device), seg.to(device), smask.to(device)


# ======================================================================================================
# 2. Extra parameter-efficient methods (the others come from ner_model.apply_method)
# ======================================================================================================
class RSLoRALinear(nm.LoRALinear):
    """rank-stabilised LoRA (Kalajdzievski 2023): scale alpha / sqrt(r) instead of alpha / r"""

    def __init__(self, base, r=16, alpha=16, dropout=0.05):
        super().__init__(base, r, alpha, dropout)
        self.scale = alpha / math.sqrt(r)


class OutScale(nn.Module):
    """IA3 (Liu et al. 2022): multiply a module's output by a learned vector (initialised to 1)"""

    def __init__(self, module, dim):
        super().__init__()
        self.module = module
        self.l = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.module(x) * self.l


def apply_method(model, cfg):
    bb, method = model.backbone, cfg["method"]
    if method == "rslora":
        for p in bb.parameters(): p.requires_grad = False
        r, a = cfg.get("rank", 16), cfg.get("alpha", 16)
        n = sum(nm._replace_linears(blk, lambda lin: RSLoRALinear(lin, r, a)) for blk in bb.blocks)
        for mod in bb.modules():
            if isinstance(mod, RSLoRALinear):
                mod.A.requires_grad = True; mod.B.requires_grad = True
        return f"rslora (rank {r}, alpha {a}, scale alpha/sqrt(r), {n} linear layers)"
    if method == "ia3":
        for p in bb.parameters(): p.requires_grad = False
        n = 0
        for blk in bb.blocks:
            for parent in list(blk.modules()):
                for name, child in list(parent.named_children()):
                    if name in ("k_proj", "v_proj", "kv_proj", "k_up", "v_up") and isinstance(child, nn.Linear):
                        setattr(parent, name, OutScale(child, child.out_features)); n += 1
                    elif type(child).__name__ == "SwiGLU":                     # gated FFN: scale the gate branch
                        child.w3 = OutScale(child.w3, child.w3.out_features); n += 1
                    elif name == "net" and isinstance(child, nn.Sequential) and isinstance(child[0], nn.Linear):
                        child[1] = OutScale(child[1], child[0].out_features); n += 1   # GELU FFN: after activation
        for mod in bb.modules():
            if isinstance(mod, OutScale):
                mod.l.requires_grad = True
        return f"ia3 ({n} scaling vectors on keys, values and FFN activations)"
    return nm.apply_method(model, cfg)


# ======================================================================================================
# 3. Model
# ======================================================================================================
class SentenceEncoder(nn.Module):
    """backbone over packed windows -> mean of each sentence's token states. forward(ids, seg, smask) -> (B,S,d)"""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, ids, seg, smask):
        with nm.autocast():
            h = self.backbone(ids)
        S = smask.size(1)
        m = (seg[:, :, None] == torch.arange(S, device=ids.device)[None, None]).to(h.dtype)   # (B,T,S)
        pooled = torch.einsum("bts,btd->bsd", m, h) / m.sum(1).clamp(min=1)[..., None]
        return pooled.float()


class CRF(nn.Module):
    """linear-chain CRF over a document's sentences (Malik et al. 2022: BiLSTM -> BiLSTM-CRF gave +0.04..0.07).
    trans[i, j] = score of label j following label i."""

    def __init__(self, n_labels):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(n_labels, n_labels))
        self.start = nn.Parameter(torch.zeros(n_labels))
        self.end = nn.Parameter(torch.zeros(n_labels))

    def nll(self, e, y):
        """e (N,L) emission scores, y (N,) gold -> negative log-likelihood per sentence"""
        N = e.size(0)
        gold = self.start[y[0]] + e[torch.arange(N, device=e.device), y].sum() + self.end[y[-1]]
        if N > 1:
            gold = gold + self.trans[y[:-1], y[1:]].sum()
        a = self.start + e[0]
        for i in range(1, N):
            a = torch.logsumexp(a[:, None] + self.trans, 0) + e[i]
        Z = torch.logsumexp(a + self.end, 0)
        return (Z - gold) / N

    def export(self):
        return dict(trans=self.trans.detach().float().cpu().clone(), start=self.start.detach().float().cpu().clone(),
                    end=self.end.detach().float().cpu().clone())


class RRModel(nn.Module):
    def __init__(self, backbone, n_labels, hidden=256, dropout=0.3, use_bilstm=True, use_lsp=True,
                 class_weights=None, lsp_weight=0.5, n_domains=2, use_pos=True, use_crf=False, crf_weight=1.0,
                 lex_size=0, lex_dim=0):
        super().__init__()
        self.crf = CRF(n_labels) if use_crf else None
        self.crf_weight = crf_weight
        self.lex = None
        if lex_dim:                                    # lexical channel: tf-idf weighted bag of word n-grams -> lex_dim
            self.lex = nn.EmbeddingBag(lex_size, lex_dim, mode="sum")
            nn.init.normal_(self.lex.weight, std=0.1)
            self.lex_drop = nn.Dropout(dropout)
        self.use_pos = use_pos
        self.backbone = backbone                       # (also inside the encoder; same object)
        self.encoder = SentenceEncoder(backbone)
        self.use_bilstm, self.use_lsp, self.lsp_weight = use_bilstm, use_lsp, lsp_weight
        d = backbone.d
        self.proj = nn.Sequential(nn.Linear(d + 3 + n_domains + lex_dim, hidden * 2), nn.GELU(), nn.Dropout(dropout))
        if use_bilstm:
            self.lstm = nn.LSTM(hidden * 2, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, n_labels)
        self.shift_head = nn.Linear(hidden * 6, 2)
        self.register_buffer("cw", class_weights if class_weights is not None else torch.ones(n_labels))
        self.n_domains = n_domains

    def doc_logits(self, sv, domain, start=0, total=None, lex=None):
        """sv (N,d) vectors of consecutive sentences start..start+N-1 of a document with `total` sentences
        -> role logits (N,L), shift logits (N,2). Position = absolute relative position in the whole document.
        lex: (ids, offsets, weights) bag of n-grams of the same N sentences (only if the lexical channel is on)."""
        N = sv.size(0); total = total or N
        pos = (start + torch.arange(N, device=sv.device, dtype=torch.float)) / max(1, total - 1)
        feats = torch.stack([pos, (pos * math.pi).sin(), (pos * math.pi).cos()], 1) * float(self.use_pos)
        dom = F.one_hot(torch.full((N,), domain, device=sv.device), self.n_domains).float()
        parts = [sv, feats, dom]
        if self.lex is not None:
            parts.append(self.lex_drop(self.lex(lex[0], lex[1], per_sample_weights=lex[2])))
        x = self.proj(torch.cat(parts, 1))
        if self.use_bilstm:
            x, _ = self.lstm(x[None]); x = x[0]
        x = self.drop(x)
        prev = torch.cat([x[:1], x[:-1]], 0)
        shift = self.shift_head(torch.cat([x, prev, x - prev], 1))
        return self.head(x), shift

    def loss(self, role_logits, shift_logits, labels, soft=None, kd_T=2.0, kd_alpha=0.5):
        ce = F.cross_entropy(role_logits, labels, weight=self.cw)
        if soft is not None:                               # knowledge distillation from teacher probabilities
            kl = F.kl_div(F.log_softmax(role_logits / kd_T, -1), soft, reduction="batchmean") * kd_T ** 2
            ce = kd_alpha * ce + (1 - kd_alpha) * kl
        if self.use_lsp and labels.numel() > 1:
            shift_gold = (labels[1:] != labels[:-1]).long()
            ce = ce + self.lsp_weight * F.cross_entropy(shift_logits[1:], shift_gold)
        if self.crf is not None:
            ce = ce + self.crf_weight * self.crf.nll(role_logits, labels)
        return ce


# ======================================================================================================
# 3b. Decoding: Viterbi with the CRF transitions + guard against stray very-rare labels
# ======================================================================================================
# PrecedentOverruled: 3 training sentences, 0 dev -> never predicted (a wrong prediction adds a 0-F1 class to the
# macro average). Dissent: in the training data it is a long contiguous block (a dissenting note), so fewer than
# 5 Dissent sentences in a document are treated as noise and re-decoded without that label.
GUARD = dict(forbid=("PrecedentOverruled",), min_block={"Dissent": 5})


def viterbi(lp, tr, lam=1.0):
    """lp (N,L) log-probabilities, tr dict(trans, start, end) -> best label path (list of ids)"""
    T, st, en = tr["trans"] * lam, tr["start"] * lam, tr["end"] * lam
    score = st + lp[0]; back = []
    for i in range(1, lp.size(0)):
        cand = score[:, None] + T
        best, idx = cand.max(0)
        score = best + lp[i]; back.append(idx)
    y = int((score + en).argmax())
    path = [y]
    for idx in reversed(back):
        y = int(idx[y]); path.append(y)
    return path[::-1]


def decode_doc(p, labels, tr=None, lam=1.0, guard=None):
    """probabilities (N,L) of one document -> label ids (Viterbi if transitions given, else argmax)"""
    lp = p.float().clamp_min(1e-9).log()
    if guard:
        for l in guard.get("forbid", ()):
            if l in labels: lp[:, labels.index(l)] = -1e4
    dec = (lambda x: viterbi(x, tr, lam)) if tr is not None and lam > 0 else (lambda x: x.argmax(-1).tolist())
    path = dec(lp)
    if guard:
        for l, k in guard.get("min_block", {}).items():
            if l in labels:
                i = labels.index(l); c = sum(1 for y in path if y == i)
                if 0 < c < k:
                    lp[:, i] = -1e4; path = dec(lp)
    return path


def combine(prob_sets, weights=None):
    """weighted geometric mean of several {doc_id: (N,L) probs} -> normalised probs"""
    keys = list(prob_sets[0])
    w = weights or [1.0 / len(prob_sets)] * len(prob_sets)
    return {k: sum(wi * ps[k].float().clamp_min(1e-9).log() for wi, ps in zip(w, prob_sets)).softmax(-1) for k in keys}


def mean_trans(trs):
    trs = [t for t in trs if t is not None]
    if not trs:
        return None
    return {k: sum(t[k] for t in trs) / len(trs) for k in ("trans", "start", "end")}


# ======================================================================================================
# 4. Metrics
# ======================================================================================================
def macro_scores(gold, pred, labels):
    """sklearn-style macro P/R/F1 over the labels present in gold or prediction (flattened sentences)"""
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold, pred):
        if g == p: tp[g] += 1
        else: fp[p] += 1; fn[g] += 1
    rows = {}
    for l in labels:
        P = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0.0
        R = tp[l] / (tp[l] + fn[l]) if tp[l] + fn[l] else 0.0
        rows[l] = dict(P=P, R=R, F1=2 * P * R / (P + R) if P + R else 0.0, support=tp[l] + fn[l])
    used = [l for l in labels if rows[l]["support"] or fp[l]]
    m = lambda k: sum(rows[l][k] for l in used) / max(1, len(used))
    acc = sum(tp.values()) / max(1, len(gold))
    return dict(macro_F1=m("F1"), macro_P=m("P"), macro_R=m("R"), accuracy=acc, per_label=rows)


def evaluate(probs, docs, labels, tr=None, lam=1.0, guard=None):
    """probs {doc_id: (n, L)} -> scores overall and per domain + predictions (Viterbi if transitions tr given)"""
    out, pred = {}, {}
    ids = {d["id"]: decode_doc(probs[d["id"]], labels, tr, lam, guard) for d in docs}
    for dom in sorted({d["domain"] for d in docs}) + ["ALL"]:
        g, p = [], []
        for d in docs:
            if dom != "ALL" and d["domain"] != dom:
                continue
            pl = [labels[i] for i in ids[d["id"]]]
            pred[d["id"]] = pl
            g += d["labels"]; p += pl
        out[dom] = macro_scores(g, p, labels)
    return out, pred


def confusion(docs, pred, top=10):
    c = Counter()
    for d in docs:
        for g, p in zip(d["labels"], pred[d["id"]]):
            if g != p: c[(g, p)] += 1
    return c.most_common(top)


# ======================================================================================================
# 5. Training / prediction
# ======================================================================================================
def _lex(model, it, s0, s1, device):
    return lex_batch(it, s0, s1, device) if getattr(model, "lex", None) is not None else None


LEX = None                                              # n-gram vocabulary of the lexical channel (training sentences)


def ensure_lex(D, cfg, preps, log=print):
    """build the n-gram vocabulary from the training documents once and attach bags to every prepared document"""
    global LEX
    key = (cfg.get("lex_features", 30000), cfg.get("lex_min_df", 3), cfg.get("lex_ngrams", 2))
    n_train = sum(len(d["text"]) for d in D["train"])           # smoke-check data and real data get different vocabularies
    if LEX is None or LEX["key"] != key + (n_train,):
        LEX = lex_vocab(D["train"], *key)
        log(f"lexical channel: {len(LEX['index']):,} word uni/bigram features from the training sentences")
    for prep in preps:
        for it in prep.items:
            if it.get("lex_key") != LEX["key"]:
                it["lex"] = [lex_encode(LEX, s) for s in it["doc"]["text"]]; it["lex_key"] = LEX["key"]


def label_prior(D, smooth=1.0):
    c = Counter(l for d in D["train"] for l in d["labels"])
    p = torch.tensor([c[l] + smooth for l in D["labels"]], dtype=torch.float)
    return p / p.sum()


def prior_adjust(probs, prior, tau):
    """logit adjustment for macro-F1: p(y|x) / prior(y)^tau (tau = 0: unchanged)"""
    if not tau:
        return probs
    lp = prior.log() * tau
    return {k: (v.float().clamp_min(1e-9).log() - lp).softmax(-1) for k, v in probs.items()}


def tune_decode(dev, D, tr, prior, taus=(0.0, 0.25, 0.5, 0.75), lams=(0.0, 0.5, 1.0, 1.5)):
    """grid over prior correction tau and CRF weight lambda on dev -> best (tau, lam, F1), full table"""
    best, table = None, []
    for tau in taus:
        dv = prior_adjust(dev, prior, tau)
        for lam in (lams if tr is not None else (0.0,)):
            f = evaluate(dv, D["dev"], D["labels"], tr, lam, GUARD)[0]["ALL"]["macro_F1"]
            table.append((tau, lam, round(f, 4)))
            if best is None or f > best[2] + 1e-9:
                best = (tau, lam, f)
    return best, table


class Prepared:
    """tokenised documents for one tokenizer: sentence ids + windows"""

    def __init__(self, docs, tok, window, max_sent_tokens, label2id, sentence_only=False):
        self.items = []
        for d in docs:
            sids = encode_doc(tok, d["text"], max_sent_tokens)
            y = torch.tensor([label2id[l] for l in d["labels"]]) if d.get("labels") else None
            wins = [(i, i + 1) for i in range(len(sids))] if sentence_only else pack_windows(sids, window)
            self.items.append(dict(doc=d, sids=sids, wins=wins, y=y))


@torch.no_grad()
def predict(model, enc, prep, device, eval_batch=16):
    """{doc_id: (n_sentences, L) probabilities}"""
    model.eval()
    out = {}
    for it in prep.items:
        vecs = []
        for i in range(0, len(it["wins"]), eval_batch):
            ids, seg, smask = window_batch(it["sids"], it["wins"][i:i + eval_batch], device)
            pooled = enc(ids, seg, smask)
            vecs.append(pooled[smask])
        sv = torch.cat(vecs, 0)
        logits, _ = model.doc_logits(sv, it["doc"]["dom_id"], lex=_lex(model, it, 0, len(it["sids"]), device))
        out[it["doc"]["id"]] = logits.float().softmax(-1).cpu()
        assert out[it["doc"]["id"]].size(0) == len(it["doc"]["text"])
    return out


def train(model, train_prep, dev_prep, dev_docs, labels, cfg, device, log=print, soft_targets=None, test_prep=None):
    rng = random.Random(cfg["seed"])
    gpus = torch.cuda.device_count()
    enc = nn.DataParallel(model.encoder) if gpus > 1 else model.encoder
    is_bb = lambda n: n.startswith(("backbone.", "encoder."))
    named = {}
    for n, p in model.named_parameters():
        if p.requires_grad and id(p) not in {id(q) for q in named.values()}:
            named[n] = p
    groups = [dict(params=[p for n, p in named.items() if is_bb(n)], lr=cfg["lr"]),
              dict(params=[p for n, p in named.items() if not is_bb(n)], lr=cfg["lr_head"])]
    groups = [g for g in groups if g["params"]]
    for g in groups: g["base_lr"] = g["lr"]
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    acc = cfg["accum"]
    steps_per_epoch = math.ceil(len(train_prep.items) / acc)
    total = steps_per_epoch * cfg["epochs"]; warm = max(1, int(cfg["warmup_frac"] * total))
    log(f"train documents {len(train_prep.items)} | {steps_per_epoch} steps/epoch x {cfg['epochs']} epochs | "
        f"trainable {sum(p.numel() for p in params):,} | GPUs {max(gpus, 1)}")
    best, best_state, history, step, bad, t0 = -1.0, None, [], 0, 0, time.time()
    snaps, K = [], cfg.get("snapshot_k", 1)              # top-K epochs by dev: their dev/test probabilities
    guard = GUARD if cfg.get("guard") else None
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    def step_doc(it, wins):
        s0, s1 = wins[0][0], wins[-1][1]
        ids, seg, smask = window_batch(it["sids"], wins, device)
        sv = enc(ids, seg, smask)[smask]
        role, shift = model.doc_logits(sv, it["doc"]["dom_id"], s0, len(it["sids"]), _lex(model, it, s0, s1, device))
        y = it["y"][s0:s1].to(device)
        soft = soft_targets[it["doc"]["id"]][s0:s1].to(device) if soft_targets is not None else None
        loss = model.loss(role.float(), shift.float(), y, soft, cfg.get("kd_T", 2.0), cfg.get("kd_alpha", 0.5)) / acc
        scaler.scale(loss).backward()
        return loss.item() * acc

    for ep in range(1, cfg["epochs"] + 1):
        model.train(); order = list(range(len(train_prep.items))); rng.shuffle(order); run = n = 0
        for k, idx in enumerate(order):
            it = train_prep.items[idx]
            W = len(it["wins"]); mw = cfg["max_windows"]
            a = rng.randrange(0, max(1, W - mw + 1))
            wins = it["wins"][a:a + mw]
            try:
                run += step_doc(it, wins)
            except torch.cuda.OutOfMemoryError:                      # very long document: retry with half of it
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                log(f"  (out of memory on {len(wins)} windows of {it['doc']['id']} -> retry with {max(1, len(wins) // 2)})")
                run += step_doc(it, wins[:max(1, len(wins) // 2)])
            n += 1
            if (k + 1) % acc and k + 1 < len(order):
                continue
            step += 1
            f = step / warm if step <= warm else max(0.0, (total - step) / max(1, total - warm))
            for g in opt.param_groups: g["lr"] = g["base_lr"] * f
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, cfg["clip"])
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        probs = predict(model, model.encoder, dev_prep, device, cfg["eval_batch"])
        tr = model.crf.export() if model.crf is not None else None
        sc, _ = evaluate(probs, dev_docs, labels, tr, 1.0, guard)
        line = " | ".join(f"{k} {v['macro_F1']:.4f}" for k, v in sc.items())
        history.append(dict(epoch=ep, loss=run / max(1, n), **{k: v["macro_F1"] for k, v in sc.items()},
                            minutes=round((time.time() - t0) / 60, 1)))
        f1 = sc["ALL"]["macro_F1"]
        if cfg.get("select_last"):                  # refit on train+dev: dev is in-sample, so keep the LAST epochs
            f1 = float(ep)
        if K > 1 and (len(snaps) < K or f1 > min(s["f1"] for s in snaps)):
            snaps.append(dict(epoch=ep, f1=f1, dev=probs, tr=tr,
                              test=predict(model, model.encoder, test_prep, device, cfg["eval_batch"])))
            snaps = sorted(snaps, key=lambda s: -s["f1"])[:K]
        note = ""
        if f1 > best + 1e-4:
            best, bad, note = f1, 0, "  <- best"
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
        log(f"epoch {ep}: loss {run / max(1, n):.4f} | dev macro-F1 {line}{note} | {(time.time() - t0) / 60:.1f} min")
        if cfg.get("patience") and bad >= cfg["patience"]:
            log(f"early stopping after {bad} epochs without improvement"); break
    model.load_state_dict(best_state)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    snap = None
    if K > 1:
        snap = dict(epochs=[s["epoch"] for s in snaps], dev=combine([s["dev"] for s in snaps]),
                    test=combine([s["test"] for s in snaps]), tr=mean_trans([s["tr"] for s in snaps]))
        log(f"averaging the top-{len(snaps)} epochs {snap['epochs']} (dev macro-F1 {[round(s['f1'], 4) for s in snaps]})")
    return dict(best_dev_macro_F1=best, history=history, train_minutes=round((time.time() - t0) / 60, 1),
                peak_gpu_gb=round(peak, 2)), snap


# ======================================================================================================
# 6. One experiment
# ======================================================================================================
SCORES = {}                                             # run -> dict(dev=probs, test=probs)
_PREP = {}


def prepared(D, arch, cfg, label2id):
    size = min(cfg["window"], nm.ARCH_INFO[arch][1])
    so = bool(cfg.get("sentence_only"))
    key = (nm.ARCH_INFO[arch][0], size, cfg["max_sent_tokens"], so, tuple(d["id"] for d in D["train"]))
    if key not in _PREP:
        tok = nm.load_tokenizer(arch, D["root"])
        _PREP[key] = tuple(Prepared(D[k], tok, size, cfg["max_sent_tokens"], label2id, so) for k in ("train", "dev", "test"))
    return _PREP[key]


def build(cfg, D):
    labels = D["labels"]
    bb = nm.Backbone(cfg["arch"], D["root"], bidirectional=cfg["bidirectional"], keep_layers=cfg.get("keep_layers"),
                     init=cfg.get("init", "pretrained"))
    cw = None
    if cfg.get("class_weighting") == "sqrt_inv":
        c = Counter(l for d in D["train"] for l in d["labels"]); tot = sum(c.values())
        cw = torch.tensor([math.sqrt(tot / max(1, c[l])) for l in labels]); cw = cw / cw.mean()
    model = RRModel(bb, len(labels), hidden=cfg["hidden"], dropout=cfg["dropout"], use_bilstm=cfg["bilstm"],
                    use_lsp=cfg["lsp"], class_weights=cw, lsp_weight=cfg["lsp_weight"], use_pos=cfg.get("use_pos", True),
                    use_crf=cfg.get("crf", False), crf_weight=cfg.get("crf_weight", 1.0),
                    lex_size=len(LEX["index"]) if cfg.get("lex_dim") else 0, lex_dim=cfg.get("lex_dim", 0))
    desc = apply_method(model, cfg)
    return model, desc


def run_experiment(cfg, D, device, out_root, log=print, soft_targets=None, keep_model=False):
    t_start = time.time()
    run_dir = os.path.join(out_root, cfg["run"]); os.makedirs(run_dir, exist_ok=True)
    labels = D["labels"]; label2id = {l: i for i, l in enumerate(labels)}
    tr, dv, te = prepared(D, cfg["arch"], cfg, label2id)
    if cfg.get("lex_dim"):
        ensure_lex(D, cfg, (tr, dv, te), log)
    torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"])
    model, desc = build(cfg, D)
    model.to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"\n=== {cfg['run']} | {cfg['arch']} | {desc} | BiLSTM {cfg['bilstm']} | LSP {cfg['lsp']} | "
        f"CRF {cfg.get('crf', False)} | lexical {cfg.get('lex_dim', 0)} | params {total:,}, trainable {trainable:,} ({trainable / total:.2%})"
        + (" | KD" if soft_targets else ""))
    res, snap = train(model, tr, dv, D["dev"], labels, cfg, device, log, soft_targets, te)
    t0 = time.time()
    dev_probs = predict(model, model.encoder, dv, device, cfg["eval_batch"]); infer = time.time() - t0
    test_probs = predict(model, model.encoder, te, device, cfg["eval_batch"])
    trans = model.crf.export() if model.crf is not None else None
    if snap is not None:                                    # average of the top-K epochs replaces the single best
        dev_probs, test_probs, trans = snap["dev"], snap["test"], snap["tr"]
        res["snapshot_epochs"] = snap["epochs"]
    guard = GUARD if cfg.get("guard") else None
    SCORES[cfg["run"]] = dict(dev=dev_probs, test=test_probs, tr=trans)
    if cfg.get("keep_train"):                               # in-sample probabilities (teacher targets for distillation)
        SCORES[cfg["run"]]["train"] = predict(model, model.encoder, tr, device, cfg["eval_batch"])
    sc, dpred = evaluate(dev_probs, D["dev"], labels, trans, 1.0, guard)
    for dom, s in sc.items():
        log(f"DEV {dom:>4}: macro-F1 {s['macro_F1']:.4f} | P {s['macro_P']:.4f} | R {s['macro_R']:.4f} | acc {s['accuracy']:.4f}")
    log(f"{'label':<24}{'P':>7}{'R':>7}{'F1':>7}{'support':>9}")
    for l in labels:
        r = sc["ALL"]["per_label"][l]; log(f"{l:<24}{r['P']:7.3f}{r['R']:7.3f}{r['F1']:7.3f}{r['support']:9d}")
    conf = confusion(D["dev"], dpred)
    log(f"most confused (gold -> predicted): {conf}")
    zp = write_submission({cfg["run"]: test_probs}, D, os.path.join(out_root, cfg["run"]), cfg["run"], log,
                          tr=trans, guard=guard)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()
                if k in {n for n, p in model.named_parameters() if p.requires_grad}},
               os.path.join(run_dir, "trainable_weights.pt"))
    rec = dict(run=cfg["run"], arch=cfg["arch"], method=desc, cfg=cfg, params_total=total, params_trainable=trainable,
               dev={k: dict(macro_F1=v["macro_F1"], macro_P=v["macro_P"], macro_R=v["macro_R"], accuracy=v["accuracy"])
                    for k, v in sc.items()},
               per_label={l: sc["ALL"]["per_label"][l] for l in labels}, confusion=[[a, b, n] for (a, b), n in conf],
               dev_inference_sec=round(infer, 1), kd=soft_targets is not None, zip=os.path.basename(zp),
               minutes_total=round((time.time() - t_start) / 60, 1), **res)
    json.dump(rec, open(os.path.join(run_dir, "record.json"), "w"), indent=1)
    if keep_model:
        return rec, model
    del model; torch.cuda.empty_cache()
    return rec, None


def write_submission(prob_sets, D, out_dir, name, log=print, weights=None, tr=None, lam=1.0, guard=None):
    """combine the test probabilities of one or more runs -> prediction.json {doc_id: [labels]} -> zip.
    Without transitions/guard: arithmetic mean + argmax (v1 behaviour). With them: geometric mean + Viterbi."""
    labels = D["labels"]
    runs = list(prob_sets)
    sub = {}
    geo = combine([prob_sets[r] for r in runs], weights) if (tr is not None or guard) else None
    for d in D["test"]:
        if geo is not None:
            pl = [labels[i] for i in decode_doc(geo[d["id"]], labels, tr, lam, guard)]
        else:
            p = sum(prob_sets[r][d["id"]] * (weights[i] if weights else 1.0) for i, r in enumerate(runs))
            pl = [labels[i] for i in p.argmax(-1).tolist()]
        assert len(pl) == len(d["text"]), d["id"]
        sub[d["id"]] = pl
    assert len(sub) == len(D["test"]), "duplicate test document ids"
    os.makedirs(out_dir, exist_ok=True)
    pj = os.path.join(out_dir, "prediction.json"); json.dump(sub, open(pj, "w"), ensure_ascii=False)
    zp = os.path.join(out_dir, f"submission_{name}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        chk = json.loads(z.read("prediction.json"))
        assert z.namelist() == ["prediction.json"] and all(len(chk[d["id"]]) == len(d["text"]) for d in D["test"])
        assert all(l in labels for v in chk.values() for l in v)
    log(f"{name}: {len(sub)} test documents, {sum(map(len, sub.values())):,} sentences -> {zp}")
    for dom in ("CL", "IT"):
        c = Counter(l for d in D["test"] if d.get("domain") == dom for l in sub[d["id"]])
        log(f"   predicted test labels {dom}: {dict(c.most_common())}")
    return zp


def save_probs(records, cfgs, path):
    """dev/test (+train, transitions) probabilities of finished runs -> one file, input of later notebooks"""
    out = {}
    recs = {r["run"]: r for r in records}
    for c in cfgs:
        r = c["run"]
        if r not in recs or r not in SCORES:
            continue
        s = SCORES[r]
        out[r] = dict(arch=c["arch"], method=c["method"], crf=bool(c.get("crf")),
                      dev_F1=recs[r]["dev"]["ALL"]["macro_F1"], params=recs[r]["params_total"],
                      dev_inference_sec=recs[r]["dev_inference_sec"],
                      dev={k: v.float().cpu() for k, v in s["dev"].items()},
                      test={k: v.float().cpu() for k, v in s["test"].items()}, tr=s.get("tr"),
                      train={k: v.float().cpu() for k, v in s["train"].items()} if "train" in s else None)
    torch.save(out, path)
    return out


def load_probs(pattern="/kaggle/input/**/p3_probs_*.pt", log=print):
    """all saved probability files (v2 and v3 formats) -> {run: dict}"""
    import glob
    out = {}
    for f in sorted(glob.glob(pattern, recursive=True)):
        for n, v in torch.load(f, map_location="cpu", weights_only=False).items():
            if n not in out:
                v.setdefault("tr", None); v.setdefault("train", None); v.setdefault("params", None); out[n] = v
        log(f"loaded {f}")
    return out


def ensemble_dev(runs, D):
    probs = {d["id"]: sum(SCORES[r]["dev"][d["id"]] for r in runs) / len(runs) for d in D["dev"]}
    return evaluate(probs, D["dev"], D["labels"])[0]


def teacher_targets(model, D, cfg, device, T=2.0):
    """soft role targets of a trained teacher for every TRAINING document (for distillation)"""
    labels = D["labels"]; label2id = {l: i for i, l in enumerate(labels)}
    tr, _, _ = prepared(D, cfg["arch"], cfg, label2id)
    model.eval(); out = {}
    with torch.no_grad():
        for it in tr.items:
            vecs = []
            for i in range(0, len(it["wins"]), cfg["eval_batch"]):
                ids, seg, smask = window_batch(it["sids"], it["wins"][i:i + cfg["eval_batch"]], device)
                vecs.append(model.encoder(ids, seg, smask)[smask])
            logits, _ = model.doc_logits(torch.cat(vecs, 0), it["doc"]["dom_id"],
                                         lex=_lex(model, it, 0, len(it["sids"]), device))
            out[it["doc"]["id"]] = (logits.float() / T).softmax(-1).cpu()
    return out


def summary(records, log=print):
    log(f"\n{'run':<36}{'method':<40}{'trainable':>12}{'dev ALL':>9}{'CL':>8}{'IT':>8}{'min':>7}")
    for r in records:
        d = r["dev"]
        log(f"{r['run']:<36}{r['method'][:39]:<40}{r['params_trainable']:>12,}{d['ALL']['macro_F1']:>9.4f}"
            f"{d.get('CL', {}).get('macro_F1', 0):>8.4f}{d.get('IT', {}).get('macro_F1', 0):>8.4f}{r['minutes_total']:>7.1f}")
