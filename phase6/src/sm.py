# Phase 6 (abstractive legal summarisation) on the Phase-1 language models.
#   Long documents: document and reference summary are split into the same number n of ordered parts (n from the
#   summary length; at test time from the document length, fitted on training data). Part i of the document,
#   compressed to fit the context (evenly spaced sentences), is trained to generate part i of the summary.
#   Decoder-only models (arch 1, 3, 4): [part header + document part + " SUMMARY:"] -> summary part + </s>, loss on the
#   summary tokens only (arch 3: the prompt is a bidirectional prefix). arch 2: encoder = document part, decoder = summary.
#   Generation: greedy + no-repeat-3-gram + repetition penalty, minimum length, </s> stops; parts are concatenated
#   and duplicate sentences removed.  Adaptation: LoRMA / LoRA / QLoRA / Houlsby + Pfeiffer adapters / full FT; distillation from a
#   teacher's token distributions (same tokenizer).
import json
import math
import os
import pickle
import random
import re
import sys
import time
import zipfile
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ner_model as nm

VERSION = "p6-v6"
PAD, UNK, BOS, EOS = 0, 1, 2, 3
INFO = {  # arch -> (tokenizer, context)
    "arch1_decoder": ("a1_bpe8000.pkl", 1024), "arch2_encdec": ("a1_bpe8000.pkl", 768),
    "arch3_prefixlm": ("space_bpe8000.pkl", 1024), "arch4_mla": ("space_bpe8000.pkl", 1024)}

# ======================================================================================================
# 1. Data
# ======================================================================================================
def read_json_any(path):
    with open(path, encoding="utf-8") as f:
        head = f.read(1)
    if head in "[{":
        try:
            return json.load(open(path, encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def parse(raw):
    if isinstance(raw, dict):
        for k in ("data", "examples", "docs"):
            if k in raw and isinstance(raw[k], (list, dict)):
                raw = raw[k]; break
    if isinstance(raw, dict):
        raw = [dict(v, id=v.get("id", k)) if isinstance(v, dict) else dict(id=k, document=v) for k, v in raw.items()]
    out = []
    for i, d in enumerate(raw):
        did = next((d[k] for k in ("id", "doc_id", "document_id", "case_id", "name") if k in d), i)
        doc = next((d[k] for k in ("document", "text", "judgment", "judgement", "doc", "source", "article") if k in d and d[k]), "")
        summ = next((d[k] for k in ("summary", "summaries", "abstract", "target", "reference", "headnote", "ref_summary",
                                    "gold_summary", "highlights") if k in d and d[k]), None)
        if isinstance(doc, list): doc = " ".join(map(str, doc))
        if isinstance(summ, list): summ = " ".join(map(str, summ))
        out.append(dict(id=did, document=" ".join(str(doc).split()), summary=None if summ is None else " ".join(str(summ).split())))
    return out


_SENT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"'\[])")


def sentences(text, min_chars=25):
    out = []
    for s in _SENT.split(text):
        s = s.strip()
        if not s: continue
        if out and len(s) < min_chars: out[-1] += " " + s
        else: out.append(s)
    return out or [text[:2000]]


_TOK = None


def _init(tokfile, root):
    global _TOK
    sys.path.insert(0, os.path.join(root, "code"))
    import a1_tokenizer, space_bpe  # noqa: F401
    _TOK = pickle.load(open(os.path.join(root, "tokenizers", tokfile), "rb"))


def _enc(text):
    if getattr(_TOK, "space_prefix", False):
        return _TOK.encode(" " + text)
    return _TOK.encode(" ") + _TOK.encode(text)


def _encode_doc(args):
    doc, summ, max_sents = args
    ds = [np.asarray(_enc(s), dtype=np.int32) for s in sentences(doc)[:max_sents]]
    ss = None if summ is None else [np.asarray(_enc(s), dtype=np.int32) for s in sentences(summ)]
    return ds, ss


class Tok:
    """tokenizer handle in the main process (+ multiprocessing tokenisation of whole datasets)"""

    def __init__(self, arch, root):
        self.arch, self.root, self.file = arch, root, INFO[arch][0]
        _init(self.file, root); self.tok = _TOK

    def enc(self, text):
        return _enc(text)

    def dec(self, ids):
        return self.tok.decode([int(i) for i in ids if int(i) > EOS]).strip()

    def content_mask(self):
        """True for tokens containing a letter or digit. The A1 tokenizer makes every space its own token, so a
        token-level repetition penalty / n-gram ban would hit spaces and punctuation and wreck the text."""
        V = self.tok.get_vocab_size() if hasattr(self.tok, "get_vocab_size") else len(self.tok.itos)
        mk = np.zeros(V, dtype=bool)
        for i in range(EOS + 1, V):
            b = self.tok.itos.get(i, b"")
            mk[i] = any(chr(c).isalnum() for c in b if c < 128) or any(c >= 128 for c in b)
        return mk

    def docs(self, docs, workers=4, max_sents=4000, log=print):
        t0 = time.time()
        args = [(d["document"], d["summary"], max_sents) for d in docs]
        if workers > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(workers, initializer=_init, initargs=(self.file, self.root)) as pool:
                enc = pool.map(_encode_doc, args, chunksize=8)
        else:
            enc = [_encode_doc(a) for a in args]
        items = [dict(doc=d, ds=a, ss=b) for d, (a, b) in zip(docs, enc)]
        dl = np.array([sum(len(x) for x in it["ds"]) for it in items])
        msg = f"tokenised {len(items):,} documents in {time.time() - t0:.0f}s | document tokens median {int(np.median(dl)):,}, 90% {int(np.percentile(dl, 90)):,}"
        if items and items[0]["ss"] is not None:
            sl = np.array([sum(len(x) for x in it["ss"]) for it in items])
            msg += f" | summary tokens median {int(np.median(sl)):,}, 90% {int(np.percentile(sl, 90)):,}"
        log(msg)
        return items


# ======================================================================================================
# 2. Parts: document part i -> summary part i
# ======================================================================================================
def groups(lens, n):
    """contiguous split of items with token lengths `lens` into n groups of ~equal token mass -> list of index lists"""
    lens = np.asarray(lens, dtype=float); tot = max(1.0, lens.sum())
    mid = (np.cumsum(lens) - lens / 2) / tot
    g = np.minimum(n - 1, (mid * n).astype(int))
    out = [list(np.where(g == k)[0]) for k in range(n)]
    for k in range(n):                                  # never empty: borrow the nearest sentence
        if not out[k]: out[k] = [min(len(lens) - 1, int(round((k + 0.5) / n * len(lens))))]
    return out


def fit(sents, budget):
    """sentences (token arrays) -> one token list of <= budget tokens: everything if it fits, otherwise evenly spaced
    sentences across the whole part (keeps the beginning, middle and end represented)"""
    tot = sum(len(s) for s in sents)
    if tot <= budget:
        return [t for s in sents for t in s.tolist()]
    m = len(sents); k = max(1, int(m * budget / tot))
    while True:                                          # fill the budget EXACTLY (same-length prompts batch together)
        idx = np.unique(np.round(np.linspace(0, m - 1, k)).astype(int))
        out = [t for i in idx for t in sents[i].tolist()]
        if len(out) >= budget or k >= m:
            return out[:budget]
        k += max(1, k // 10)


class Layout:
    """token budgets of one model: P prompt tokens, G summary tokens per part"""

    def __init__(self, tok, arch, G, n_max=8):
        self.tok, self.arch, self.G, self.n_max = tok, arch, G, n_max
        ctx = INFO[arch][1]
        self.P = ctx if arch == "arch2_encdec" else ctx - G - 2
        self.sep = tok.enc("SUMMARY:")
        self.coef = (0.0, 1.0); self.len_scale = 1.0

    def header(self, i, n):
        return self.tok.enc(f"Part {i + 1} of {n}.")

    def n_parts_gold(self, it):
        return int(np.clip(round(sum(len(s) for s in it["ss"]) / self.G), 1, self.n_max))

    def fit_length_model(self, items):
        """log(summary tokens) = a + b log(document tokens), fitted on training documents"""
        x = np.log([max(1, sum(len(s) for s in it["ds"])) for it in items])
        y = np.log([max(1, sum(len(s) for s in it["ss"])) for it in items])
        b, a = np.polyfit(x, y, 1); self.coef = (float(a), float(b))
        pred = np.exp(a + b * x); r = np.corrcoef(np.exp(y), pred)[0, 1]
        return dict(a=a, b=b, r=float(r))

    def n_parts_pred(self, it):
        L = max(1, sum(len(s) for s in it["ds"]))
        return int(np.clip(round(self.len_scale * math.exp(self.coef[0] + self.coef[1] * math.log(L)) / self.G), 1, self.n_max))

    def prompts(self, it, n):
        """-> list of n prompt token lists (part header + fitted document part + separator)"""
        gs = groups([len(s) for s in it["ds"]], n)
        out = []
        for i, gi in enumerate(gs):
            h = self.header(i, n)
            body = fit([it["ds"][j] for j in gi], self.P - len(h) - len(self.sep) - (1 if self.arch == "arch2_encdec" else 0))
            out.append(h + body + (self.sep if self.arch != "arch2_encdec" else self.sep))
        return out

    def pairs(self, it):
        """training pairs (prompt, target) with the gold number of parts"""
        n = self.n_parts_gold(it)
        ps = self.prompts(it, n)
        gs = groups([len(s) for s in it["ss"]], n)
        tg = [[t for j in g for t in it["ss"][j].tolist()][: self.G - 1] + [EOS] for g in gs]
        return list(zip(ps, tg))


# ======================================================================================================
# 3. Models
# ======================================================================================================
def load_lm(arch, root, keep_layers=None, init="pretrained"):
    sys.path.insert(0, os.path.join(root, "code"))
    ck = torch.load(os.path.join(root, arch, "ckpt_best.pt"), map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    if arch == "arch4_mla":
        from mla import MLAConfig, MLALM; m = MLALM(MLAConfig(**cfg))
    elif arch == "arch3_prefixlm":
        from prefixlm import PrefixLMConfig, PrefixLM; m = PrefixLM(PrefixLMConfig(**cfg))
    elif arch == "arch2_encdec":
        from encdec import EncDecConfig, EncDecLM; m = EncDecLM(EncDecConfig(**cfg))
    else:
        from model_arch import LMConfig, DecoderLM; m = DecoderLM(LMConfig(**cfg))
    if init == "pretrained":
        missing, unexpected = m.load_state_dict(ck["model"], strict=False)
        assert not unexpected and all(("rope" in x or "mask" in x or "pos_emb" in x) for x in missing), (missing[:5], unexpected[:5])
    if keep_layers:
        assert arch != "arch2_encdec"
        m.blocks = m.blocks[:keep_layers]
    return m


def layers_of(m, arch):
    return list(m.encoder) + list(m.decoder) if arch == "arch2_encdec" else list(m.blocks)


NF4 = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
       -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
       0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]   # 4-bit NormalFloat levels (Dettmers et al. 2023)


class QLoRALinear(nn.Module):
    """QLoRA: the frozen weight is stored as 4-bit NF4 codes (blocks of 64, two codes per byte) with the block absmax
    double-quantised to 8 bits (blocks of 256); it is dequantised on the fly in every forward and a LoRA update
    (alpha/r) B A x (B = 0 at start) is added. Plain torch (no bitsandbytes): same storage format and arithmetic."""

    def __init__(self, base, r=16, alpha=32, dropout=0.05, block=64, block2=256):
        super().__init__()
        W = base.weight.detach().float(); self.shape = tuple(W.shape); self.scale = alpha / r; self.block = block
        w = W.flatten(); self.n = w.numel()
        w = torch.cat([w, w.new_zeros((-self.n) % block)]).view(-1, block)
        absmax = w.abs().amax(1).clamp_min(1e-8); self.nb = absmax.numel()
        lv = torch.tensor(NF4)
        codes = torch.bucketize((w / absmax[:, None]).flatten(), (lv[1:] + lv[:-1]) / 2).to(torch.uint8)
        self.register_buffer("qw", codes[0::2] | (codes[1::2] << 4))                    # 2 codes per byte
        mean = absmax.mean(); a = absmax - mean
        a = torch.cat([a, a.new_zeros((-self.nb) % block2)]).view(-1, block2); s2 = a.abs().amax(1).clamp_min(1e-12) / 127
        self.register_buffer("am_q", torch.round(a / s2[:, None]).to(torch.int8))         # double quantisation
        self.register_buffer("am_s", s2); self.register_buffer("am_mean", mean.reshape(1)); self.register_buffer("levels", lv)
        self.register_buffer("bias", None if base.bias is None else base.bias.detach().clone())
        self.A = nn.Parameter(torch.empty(r, self.shape[1])); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(self.shape[0], r))
        self.drop = nn.Dropout(dropout)

    def weight_fp(self):
        absmax = (self.am_q.float() * self.am_s[:, None]).flatten()[:self.nb] + self.am_mean
        codes = torch.stack([self.qw & 15, self.qw >> 4], 1).flatten().long()
        return (self.levels[codes].view(-1, self.block) * absmax[:, None]).flatten()[:self.n].view(self.shape)

    def forward(self, x):
        return F.linear(x, self.weight_fp(), self.bias) + self.scale * F.linear(F.linear(self.drop(x), self.A), self.B)


def apply_method(m, arch, cfg):
    method = cfg["method"]
    for p in m.parameters(): p.requires_grad = method == "full_ft"
    if method == "qlora":
        r, a = cfg.get("rank", 16), cfg.get("alpha", 32)
        n = sum(nm._replace_linears(L, lambda lin: QLoRALinear(lin, r, a, cfg.get("lora_dropout", 0.05))) for L in layers_of(m, arch))
        q = [mod for mod in m.modules() if isinstance(mod, QLoRALinear)]
        nq = sum(mod.n for mod in q); mb = sum(mod.qw.numel() + mod.am_q.numel() + 4 * mod.am_s.numel() for mod in q) / 2**20
        return (f"qlora (frozen base in NF4: {nq:,} weights = {mb:.1f} MB instead of {2 * nq / 2**20:.1f} MB fp16; "
                f"LoRA rank {r}, alpha {a}, {n} linear layers)")
    if method in ("lora", "lorma"):
        r, a = cfg.get("rank", 16), cfg.get("alpha", 16)
        make = (lambda lin: nm.LoRALinear(lin, r, a, cfg.get("lora_dropout", 0.05))) if method == "lora" else (lambda lin: nm.LoRMALinear(lin, r, a))
        n = sum(nm._replace_linears(L, make) for L in layers_of(m, arch))
        for mod in m.modules():
            if isinstance(mod, (nm.LoRALinear, nm.LoRMALinear)): mod.A.requires_grad = True; mod.B.requires_grad = True
        return f"{method} (rank {r}, alpha {a}, {n} linear layers)"
    if method in ("houlsby", "pfeiffer"):
        bott = cfg.get("bottleneck", 128); d = m.cfg.d_model; n = 0
        for L in layers_of(m, arch):
            for name in (("attn", "self_attn", "cross_attn") if method == "houlsby" else ()):
                if hasattr(L, name): setattr(L, name, nm.AdapterWrap(getattr(L, name), d, bott)); n += 1
            if hasattr(L, "ffn"): L.ffn = nm.AdapterWrap(L.ffn, d, bott); n += 1
        for mod in m.modules():
            if isinstance(mod, nm.Adapter):
                for p in mod.parameters(): p.requires_grad = True
            if isinstance(mod, nn.LayerNorm) or type(mod).__name__ == "RMSNorm":
                for p in mod.parameters(): p.requires_grad = True
        return f"{method} adapters (bottleneck {bott}, {n} adapters)"
    if method == "full_ft":
        return "full fine-tuning"
    raise ValueError(method)


def hidden_dec(m, arch, ids, plen):
    """decoder-only models: final hidden states (B,T,d)"""
    if arch == "arch3_prefixlm":
        h, _ = m._run(ids, m.prefix_mask(plen, ids.size(1))); return h
    if arch == "arch4_mla":
        return m.hidden(ids)
    x = m.tok_emb(ids)
    if m.cfg.scale_emb: x = x * math.sqrt(m.cfg.d_model)
    if m.cfg.pos_emb_type == "sinusoidal": x = x + m.pos_emb[: ids.size(1)].unsqueeze(0)
    for b in m.blocks: x = b(x)
    return m.ln_f(x)


class Objective(nn.Module):
    """returns (1, 3): summed CE, number of target tokens, summed KD loss - DataParallel-friendly"""

    def __init__(self, arch, model, teacher=None, teacher_arch=None, T=2.0):
        super().__init__()
        self.arch, self.model, self.teacher, self.tarch, self.T = arch, model, teacher, teacher_arch, T

    def forward(self, ids, plen, lens, dec_in=None, dec_lab=None):
        if self.arch == "arch2_encdec":
            with nm.autocast():
                logits = self.model.decode(self.model.encode(ids), dec_in)
            m = dec_lab != -100
            lg = logits[m].float(); tgt = dec_lab[m]
            return torch.stack([F.cross_entropy(lg, tgt, reduction="sum"), m.sum().float(), torch.zeros((), device=ids.device)])[None]
        T = ids.size(1); pos = torch.arange(T, device=ids.device)[None]
        m = (pos >= (plen[:, None] - 1)) & (pos < (lens[:, None] - 1))         # position t predicts token t+1
        tgt = ids[:, 1:][m[:, :-1]]
        with nm.autocast():
            h = hidden_dec(self.model, self.arch, ids, plen)
            lg = self.model.head(h[m]).float()
        ce = F.cross_entropy(lg, tgt, reduction="sum")
        kd = torch.zeros((), device=ids.device)
        if self.teacher is not None:
            with torch.no_grad(), nm.autocast():
                th = hidden_dec(self.teacher, self.tarch, ids, plen)
                tl = self.teacher.head(th[m]).float()
            kd = F.kl_div(F.log_softmax(lg / self.T, -1), F.softmax(tl / self.T, -1), reduction="sum") * self.T ** 2
        return torch.stack([ce, m.sum().float(), kd])[None]


def collate(batch, arch, device):
    """batch of (prompt, target) -> tensors"""
    if arch == "arch2_encdec":
        S = max(len(p) for p, _ in batch); Tt = max(len(t) for _, t in batch)
        src = torch.full((len(batch), S), PAD, dtype=torch.long); din = torch.full((len(batch), Tt), PAD, dtype=torch.long)
        lab = torch.full((len(batch), Tt), -100, dtype=torch.long)
        for i, (p, t) in enumerate(batch):
            src[i, :len(p)] = torch.tensor(p); din[i, :len(t)] = torch.tensor([BOS] + t[:-1]); lab[i, :len(t)] = torch.tensor(t)
        z = torch.zeros(len(batch), dtype=torch.long)
        return dict(ids=src.to(device), plen=z.to(device), lens=z.to(device), dec_in=din.to(device), dec_lab=lab.to(device))
    T = max(len(p) + len(t) for p, t in batch)
    ids = torch.full((len(batch), T), PAD, dtype=torch.long)
    plen = torch.zeros(len(batch), dtype=torch.long); lens = torch.zeros(len(batch), dtype=torch.long)
    for i, (p, t) in enumerate(batch):
        s = p + t; ids[i, :len(s)] = torch.tensor(s); plen[i] = len(p); lens[i] = len(s)
    return dict(ids=ids.to(device), plen=plen.to(device), lens=lens.to(device))


# ======================================================================================================
# 4. Generation
# ======================================================================================================
# ---- arch 1 has no KV cache in its Phase-1 code: a cached forward built from its own modules (same weights) ----
def _a1_attn(att, x, cache):
    """CausalSelfAttention of arch 1 with a (k, v) cache; x (B,T,C) are the T newest positions"""
    B, T, _ = x.shape
    q = att.q_proj(x).view(B, T, att.n_heads, att.d_head).transpose(1, 2)
    k = att.k_proj(x).view(B, T, att.n_kv_heads, att.d_head).transpose(1, 2)
    v = att.v_proj(x).view(B, T, att.n_kv_heads, att.d_head).transpose(1, 2)
    if cache is not None:
        k = torch.cat([cache[0], k], 2); v = torch.cat([cache[1], v], 2)
    S = k.size(2)
    kk, vv = k.repeat_interleave(att.n_rep, dim=1), v.repeat_interleave(att.n_rep, dim=1)
    sc = torch.matmul(q, kk.transpose(-2, -1)) * att.attention_scale
    if T > 1:
        sc = sc.masked_fill(~att.causal_mask[S - T:S, :S], float("-inf"))
    out = torch.matmul(F.softmax(sc, dim=-1), vv).transpose(1, 2).contiguous().view(B, T, -1)
    return att.out_proj(out), (k, v)


def a1_step(m, idx, caches=None, start=0):
    """arch 1 forward over the new tokens idx (B,T) at positions start.. -> last-position logits, new caches"""
    x = m.tok_emb(idx)
    if m.cfg.scale_emb: x = x * math.sqrt(m.cfg.d_model)
    if m.cfg.pos_emb_type == "sinusoidal": x = x + m.pos_emb[start:start + idx.size(1)].unsqueeze(0)
    new = []
    for li, b in enumerate(m.blocks):
        wrap = isinstance(b.attn, nm.AdapterWrap)
        att = b.attn.module if wrap else b.attn
        c = None if caches is None else caches[li]
        if b.norm_first:
            a, c = _a1_attn(att, b.ln1(x), c)
            x = x + (b.attn.adapter(a) if wrap else a)
            x = x + b.ffn(b.ln2(x))
        else:
            a, c = _a1_attn(att, x, c)
            x = b.ln1(x + (b.attn.adapter(a) if wrap else a))
            x = b.ln2(x + b.ffn(x))
        new.append(c)
    return m.head(m.ln_f(x[:, -1])), new


_A1_OK = {}


def a1_cache_ok(m, device):
    """checks once per model that cached decoding gives the same logits as the original forward (else: no cache)"""
    if id(m) in _A1_OK: return _A1_OK[id(m)]
    ok = False
    try:
        if not m.blocks[0].attn.use_rope:
            g = torch.Generator().manual_seed(0)
            ids = torch.randint(EOS + 1, m.cfg.vocab_size, (2, 40), generator=g).to(device)
            ref = m(ids)[:, 31:39].float()                       # logits at positions 31..38
            lg, c = a1_step(m, ids[:, :32]); outs = [lg]
            for t in range(32, 39):
                lg, c = a1_step(m, ids[:, t:t + 1], c, t); outs.append(lg)
            diff = (torch.stack(outs, 1).float() - ref).abs().max().item()
            ok = diff < 1e-3 * max(1.0, ref.abs().max().item())
            print(f"arch-1 KV cache check: max logit difference {diff:.2e} -> {'cached decoding' if ok else 'NO cache (slow)'}")
    except Exception as e:
        print("arch-1 KV cache check failed -> NO cache (slow):", repr(e)[:200])
    _A1_OK[id(m)] = ok
    return ok


def _buckets(prompts, batch):
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    buckets, cur = [], []
    for i in order:
        if cur and (len(prompts[i]) != len(prompts[cur[0]]) or len(cur) >= batch):
            buckets.append(cur); cur = []
        cur.append(i)
    if cur: buckets.append(cur)
    return buckets


def _reorder(c, idx):
    """select rows idx (dim 0) of every tensor in a (nested) cache"""
    if c is None: return None
    if torch.is_tensor(c): return c.index_select(0, idx)
    if isinstance(c, dict): return {k: _reorder(v, idx) for k, v in c.items()}
    if isinstance(c, tuple) and hasattr(c, "_fields"): return type(c)(*[_reorder(x, idx) for x in c])
    if isinstance(c, (list, tuple)): return type(c)(_reorder(x, idx) for x in c)
    return c


@torch.no_grad()
def generate_beam(m, arch, prompts, G, device, min_frac=0.5, no_repeat=3, rep_pen=1.15, batch=8, log=None, content=None,
                  beams=4, len_pen=1.0, window=100):
    """beam search with the same constraints as greedy decoding (repetition penalty + no-repeat n-gram on content
    tokens, minimum length, </s> stops). Finished hypotheses are ranked by summed log-probability / length**len_pen;
    a prompt stops once `beams` hypotheses have finished. -> list of (tokens, ended with </s>)"""
    m.eval(); out = [None] * len(prompts); K = beams
    is_c = (lambda t: True) if content is None else (lambda t: bool(content[t]))
    buckets = _buckets(prompts, batch)
    max_new = G if arch != "arch2_encdec" else min(G, m.cfg.dec_len - 1)
    min_new = int(min_frac * max_new); t0 = time.time()
    a1c = arch == "arch1_decoder" and a1_cache_ok(m, device)
    for bi, bk in enumerate(buckets):
        P = torch.tensor([prompts[i] for i in bk], dtype=torch.long, device=device); B = P.size(0); R = B * K
        rep = torch.arange(B, device=device).repeat_interleave(K)
        with nm.autocast():
            if arch in ("arch3_prefixlm", "arch4_mla"):
                logits, cache = m.prefill(P); cache = _reorder(cache, rep); pos = P.size(1)
            elif arch == "arch2_encdec":
                mem = m.encode(P).index_select(0, rep); dec = torch.full((R, 1), BOS, dtype=torch.long, device=device)
                logits = m.decode(mem, dec)[:, -1]
            elif a1c:
                logits, cache = a1_step(m, P); cache = _reorder(cache, rep); pos = P.size(1)
            else:
                logits = m(P)[:, -1]; seq = P.index_select(0, rep)
            if logits.size(0) == B: logits = logits.index_select(0, rep)
            gen = [[] for _ in range(R)]; cg = [[] for _ in range(R)]; seen = [dict() for _ in range(R)]
            score = torch.zeros(R, device=device); score.view(B, K)[:, 1:] = -1e9     # at the start only beam 0 is live
            fin = [[] for _ in range(B)]; live = [True] * B
            for step in range(max_new):
                lg = logits.float()
                if rep_pen != 1.0:
                    ri, ti = [], []
                    for r in range(R):
                        if cg[r]: u = set(cg[r][-window:]); ri += [r] * len(u); ti += list(u)
                    if ri:
                        ri, ti = torch.tensor(ri, device=device), torch.tensor(ti, device=device)
                        v = lg[ri, ti]; lg[ri, ti] = torch.where(v > 0, v / rep_pen, v * rep_pen)
                if no_repeat:
                    ri, ti = [], []
                    for r in range(R):
                        if len(cg[r]) >= no_repeat - 1:
                            ban = seen[r].get(tuple(cg[r][-(no_repeat - 1):]))
                            if ban: ri += [r] * len(ban); ti += list(ban)
                    if ri: lg[torch.tensor(ri, device=device), torch.tensor(ti, device=device)] = -1e9
                lg[:, :EOS] = -1e9
                if step < min_new: lg[:, EOS] = -1e9
                lp = F.log_softmax(lg, -1) + score[:, None]; V = lp.size(1)
                tv, ti = lp.view(B, K * V).topk(2 * K, dim=1); tv, ti = tv.tolist(), ti.tolist()
                src, ntok, nsc = [], [], []
                for b in range(B):
                    keep = []
                    if live[b]:
                        for v, ix in zip(tv[b], ti[b]):
                            if v < -1e8: break
                            k, t = divmod(ix, V); r = b * K + k
                            if t == EOS: fin[b].append((v / max(1, len(gen[r])) ** len_pen, gen[r], True))
                            else: keep.append((r, t, v))
                            if len(keep) == K: break
                        if len(fin[b]) >= K or not keep: live[b] = False
                    if not live[b]: keep = [(b * K + k, EOS, -1e9) for k in range(K)]
                    while len(keep) < K: keep.append((keep[0][0], keep[0][1], -1e9))
                    for r, t, v in keep: src.append(r); ntok.append(t); nsc.append(v)
                if not any(live): break
                ng, ncg, nseen = [], [], []
                for r, t in zip(src, ntok):
                    g, c, s = gen[r], cg[r], seen[r]
                    if t != EOS:
                        g = g + [t]
                        if is_c(t):
                            if no_repeat and len(c) >= no_repeat - 1:
                                key = tuple(c[-(no_repeat - 1):]); s = dict(s); s[key] = s.get(key, frozenset()) | {t}
                            c = c + [t]
                    ng.append(g); ncg.append(c); nseen.append(s)
                gen, cg, seen = ng, ncg, nseen
                idx = torch.tensor(src, device=device); score = torch.tensor(nsc, device=device)
                tok = torch.tensor(ntok, device=device)[:, None]
                if arch in ("arch3_prefixlm", "arch4_mla"):
                    logits, cache = m.decode_step(tok, _reorder(cache, idx), pos); pos += 1
                elif arch == "arch2_encdec":
                    dec = torch.cat([dec.index_select(0, idx), tok], 1); logits = m.decode(mem, dec)[:, -1]
                elif a1c:
                    logits, cache = a1_step(m, tok, _reorder(cache, idx), pos); pos += 1
                else:
                    seq = torch.cat([seq.index_select(0, idx), tok], 1)[:, -1024:]; logits = m(seq)[:, -1]
        sc = score.tolist()
        for b, i in enumerate(bk):
            if fin[b]:
                best = max(fin[b], key=lambda x: x[0]); out[i] = (best[1], True)
            else:
                r = max(range(b * K, b * K + K), key=lambda r: sc[r] / max(1, len(gen[r])) ** len_pen); out[i] = (gen[r], False)
        if log and (bi == 0 or (bi + 1) % 50 == 0):
            done_n = sum(len(x) for x in buckets[: bi + 1])
            log(f"  generated {done_n:,}/{len(prompts):,} parts (beam {K}) | {(time.time() - t0) / 60:.1f} min")
    return out


@torch.no_grad()
def generate(m, arch, prompts, G, device, min_frac=0.5, no_repeat=3, rep_pen=1.15, batch=16, log=None, content=None,
             beams=1, len_pen=1.0, window=100):
    """greedy decoding for a list of prompts (token lists) -> list of (generated tokens without </s>, ended with </s>).
    Prompts of identical length are batched (KV cache for arch 3 / 4). Repetition penalty and the no-repeat n-gram
    ban act on content tokens only (`content` mask: tokens with a letter/digit), never on spaces / punctuation.
    beams > 1 -> beam search (generate_beam)."""
    if beams > 1:
        return generate_beam(m, arch, prompts, G, device, min_frac, no_repeat, rep_pen, max(1, 2 * batch // beams), log,
                             content, beams, len_pen, window)
    m.eval(); out = [None] * len(prompts)
    is_c = (lambda t: True) if content is None else (lambda t: bool(content[t]))
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    buckets, cur = [], []
    for i in order:
        if cur and (len(prompts[i]) != len(prompts[cur[0]]) or len(cur) >= batch):
            buckets.append(cur); cur = []
        cur.append(i)
    if cur: buckets.append(cur)
    max_new = G if arch != "arch2_encdec" else min(G, m.cfg.dec_len - 1)
    min_new = int(min_frac * max_new); t0 = time.time()
    a1c = arch == "arch1_decoder" and a1_cache_ok(m, device)
    for bi, bk in enumerate(buckets):
        P = torch.tensor([prompts[i] for i in bk], dtype=torch.long, device=device)
        B = P.size(0); gen = [[] for _ in range(B)]; done = [False] * B
        seen = [dict() for _ in range(B)]; cg = [[] for _ in range(B)]   # cg = content tokens generated so far
        with nm.autocast():
            if arch in ("arch3_prefixlm", "arch4_mla"):
                logits, cache = m.prefill(P); pos = P.size(1)
            elif arch == "arch2_encdec":
                mem = m.encode(P); dec = torch.full((B, 1), BOS, dtype=torch.long, device=device)
                logits = m.decode(mem, dec)[:, -1]
            else:
                if a1c: logits, cache = a1_step(m, P); pos = P.size(1)
                else: seq = P; logits = m(seq)[:, -1]
            for step in range(max_new):
                lg = logits.float()
                for b in range(B):
                    if done[b]: continue
                    if cg[b] and rep_pen != 1.0:
                        u = torch.tensor(sorted(set(cg[b][-window:])), device=device)
                        v = lg[b, u]; lg[b, u] = torch.where(v > 0, v / rep_pen, v * rep_pen)
                    if no_repeat and len(cg[b]) >= no_repeat - 1:
                        ban = seen[b].get(tuple(cg[b][-(no_repeat - 1):]))
                        if ban: lg[b, list(ban)] = -1e9
                    lg[b, :EOS] = -1e9
                    if step < min_new: lg[b, EOS] = -1e9
                nxt = lg.argmax(-1)
                for b in range(B):
                    if done[b]: continue
                    t = int(nxt[b])
                    if t == EOS: done[b] = True; continue
                    if is_c(t):
                        if no_repeat and len(cg[b]) >= no_repeat - 1:
                            seen[b].setdefault(tuple(cg[b][-(no_repeat - 1):]), set()).add(t)
                        cg[b].append(t)
                    gen[b].append(t)
                if all(done): break
                tok = nxt[:, None]
                if arch in ("arch3_prefixlm", "arch4_mla"):
                    logits, cache = m.decode_step(tok, cache, pos); pos += 1
                elif arch == "arch2_encdec":
                    dec = torch.cat([dec, tok], 1); logits = m.decode(mem, dec)[:, -1]
                else:
                    if a1c: logits, cache = a1_step(m, tok, cache, pos); pos += 1
                    else: seq = torch.cat([seq, tok], 1)[:, -1024:]; logits = m(seq)[:, -1]
        for b, i in enumerate(bk): out[i] = (gen[b], done[b])
        if log and (bi == 0 or (bi + 1) % 50 == 0):
            done_n = sum(len(x) for x in buckets[: bi + 1])
            log(f"  generated {done_n:,}/{len(prompts):,} parts | {(time.time() - t0) / 60:.1f} min")
    return out


def trim_open_sentence(text, keep=0.6):
    """a part cut off by the token limit ends mid-sentence -> cut back to its last full stop (if most text survives)"""
    cut = max(text.rfind(". "), text.rfind(".\n"), text.rfind("; "), text.rfind("]"))
    if text.rstrip().endswith((".", "]", ")")) or cut < keep * len(text):
        return text
    return text[: cut + 1]


def dedupe(text):
    """drop repeated sentences (exact or >= 80% word overlap with an earlier one)"""
    keep, seen = [], []
    for s in sentences(text, min_chars=1):
        w = set(re.findall(r"\w+", s.lower()))
        if not w: continue
        if any(len(w & v) / max(1, min(len(w), len(v))) >= 0.8 for v in seen): continue
        keep.append(s); seen.append(w)
    return " ".join(keep)


def summarise(m, arch, lay, items, device, n_mode="pred", log=print, **gk):
    """documents -> generated summary texts (parts concatenated, duplicates removed)"""
    jobs, owner = [], []
    for k, it in enumerate(items):
        n = lay.n_parts_pred(it) if n_mode == "pred" else (1 if n_mode == "single" else lay.n_parts_gold(it))
        for p in lay.prompts(it, n):
            jobs.append(p); owner.append(k)
    gk = dict(gk); v2 = gk.pop("v2pen", False)     # v2pen: penalty / n-gram ban on ALL tokens (window 200), no trimming
    if v2: gk["window"] = 200
    gen = generate(m, arch, jobs, lay.G, device, log=log, content=None if v2 else lay.tok.content_mask(), **gk)
    parts = [[] for _ in items]
    for k, (g, ended) in zip(owner, gen):
        t = lay.tok.dec(g)
        parts[k].append(t if ended or v2 else trim_open_sentence(t))
    return [dedupe(" ".join(p)) or "Summary not available." for p in parts]


# ======================================================================================================
# 5. Metrics (no pretrained model is used locally; the leaderboard computes BERTScore)
# ======================================================================================================
_W = re.compile(r"[a-z0-9]+")


def _ngrams(w, n):
    return Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))


def _lcs(a, b):
    if len(a) > 1200: a = a[:1200]
    if len(b) > 1200: b = b[:1200]
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge(preds, refs):
    r1, r2, rl, rec1 = [], [], [], []
    for p, r in zip(preds, refs):
        a, b = _W.findall(p.lower()), _W.findall(r.lower())
        for n, acc in ((1, r1), (2, r2)):
            pa, rb = _ngrams(a, n), _ngrams(b, n); ov = sum((pa & rb).values())
            P = ov / max(1, sum(pa.values())); R = ov / max(1, sum(rb.values()))
            acc.append(2 * P * R / max(1e-9, P + R))
            if n == 1: rec1.append(R)
        l = _lcs(a, b); P = l / max(1, min(len(a), 1200)); R = l / max(1, min(len(b), 1200))
        rl.append(2 * P * R / max(1e-9, P + R))
    return dict(rouge1=float(np.mean(r1)), rouge2=float(np.mean(r2)), rougeL=float(np.mean(rl)), rouge1_recall=float(np.mean(rec1)))


def redundancy(texts):
    """share of repeated word 3-grams inside each text (0 = no repetition)"""
    v = []
    for t in texts:
        w = _W.findall(t.lower()); g = _ngrams(w, 3); tot = sum(g.values())
        v.append(0.0 if not tot else 1 - len(g) / tot)
    return float(np.mean(v))


def text_stats(preds, refs, docs):
    wp = np.array([len(p.split()) for p in preds]); wr = np.array([len(r.split()) for r in refs]) if refs else None
    wd = np.array([len(d.split()) for d in docs])
    s = dict(pred_words=float(wp.mean()), compression=float((wp / np.maximum(1, wd)).mean()), redundancy=redundancy(preds))
    if refs: s["ref_words"] = float(wr.mean())
    return s


# ======================================================================================================
# 6. Training
# ======================================================================================================
def trainable_state(m):
    return {k: v.detach().cpu() for k, v in m.state_dict().items() if k in {n for n, p in m.named_parameters() if p.requires_grad}}


def train(cfg, model, arch, tr_pairs, va_pairs, device, out_dir, log=print, teacher=None, teacher_arch=None):
    obj = Objective(arch, model, teacher, teacher_arch, cfg.get("kd_T", 2.0))
    gpus = torch.cuda.device_count()
    run = nn.DataParallel(obj) if gpus > 1 else obj
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    rng = random.Random(cfg["seed"]); B, acc = cfg["batch"], cfg["accum"]
    spe = math.ceil(len(tr_pairs) / (B * acc)); total = int(spe * cfg["epochs"]); warm = max(1, int(0.05 * total))
    alpha = cfg.get("kd_alpha", 1.0) if teacher is not None else 1.0
    log(f"train pairs {len(tr_pairs):,} | val pairs {len(va_pairs):,} | {spe:,} steps/epoch x {cfg['epochs']} | trainable "
        f"{sum(p.numel() for p in params):,} | GPUs {gpus}" + (f" | KD alpha {alpha} T {cfg.get('kd_T', 2.0)}" if teacher is not None else ""))
    assert gpus >= 1, "NO GPU - set Accelerator to GPU T4 x2"

    def val_loss():
        model.eval(); s = n = 0.0
        with torch.no_grad():
            for i in range(0, len(va_pairs), B * 2):
                o = run(**collate(va_pairs[i:i + B * 2], arch, device)).sum(0); s += float(o[0]); n += float(o[1])
        model.train(); return s / max(1, n)

    best, best_state, step, t0, hist = 1e9, None, 0, time.time(), []
    budget = cfg.get("budget_hours", 6.0) * 3600
    model.train(); stop = False
    for ep in range(math.ceil(cfg["epochs"])):
        order = list(range(len(tr_pairs))); rng.shuffle(order); run_l = run_n = 0.0; gnorm = []
        for k in range(0, len(order), B):
            o = run(**collate([tr_pairs[i] for i in order[k:k + B]], arch, device)).sum(0)
            ce, n, kd = o[0], o[1].clamp(min=1), o[2]
            loss = (alpha * ce + (1 - alpha) * kd) / n if teacher is not None else ce / n
            scaler.scale(loss / acc).backward(); run_l += float(ce); run_n += float(n)
            if (k // B + 1) % acc: continue
            step += 1
            f = step / warm if step <= warm else max(0.0, (total - step) / max(1, total - warm))
            for g in opt.param_groups: g["lr"] = cfg["lr"] * f
            scaler.unscale_(opt); gnorm.append(float(torch.nn.utils.clip_grad_norm_(params, 1.0)))
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if step >= total or time.time() - t0 > budget: stop = True; break
        v = val_loss(); hist.append(dict(epoch=ep + 1, train_loss=run_l / max(1, run_n), val_loss=v))
        note = ""
        if v < best:
            best, note = v, "  <- best"; best_state = trainable_state(model)
            torch.save(best_state, os.path.join(out_dir, f"p6_trainable_{cfg['run']}.pt"))
        log(f"epoch {ep + 1}: train loss {run_l / max(1, run_n):.4f} | val loss {v:.4f} (ppl {math.exp(min(20, v)):.1f}){note} | "
            f"grad norm {np.mean([g for g in gnorm if math.isfinite(g)] or [0]):.3f} | {(time.time() - t0) / 60:.0f} min")
        if stop:
            if time.time() - t0 > budget: log("time budget reached - stopping")
            break
    model.load_state_dict(best_state, strict=False)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    return dict(best_val_loss=best, history=hist, train_minutes=round((time.time() - t0) / 60, 1), peak_gpu_gb=round(peak, 2))


def build(cfg, root):
    m = load_lm(cfg["arch"], root, cfg.get("keep_layers"), cfg.get("init", "pretrained"))
    desc = apply_method(m, cfg["arch"], cfg)
    return m, desc


def find_weights(run, inputs="/kaggle/input", work="/kaggle/working"):
    """p6_trainable_{run}.pt among the attached inputs. Also accepts the same file renamed .zip (a torch file IS a zip
    archive; browsers/Kaggle often save it as .zip) and a folder that Kaggle unpacked from such a .zip (re-packed
    here, uncompressed, as a .pt). -> path or None (then prints every p6_trainable* found)"""
    import glob
    name = f"p6_trainable_{run}"
    for ext in (".pt", ".zip"):
        w = sorted(glob.glob(f"{inputs}/**/{name}{ext}", recursive=True))
        if w: return w[0]
    pk = sorted(glob.glob(f"{inputs}/**/{name}/data.pkl", recursive=True))
    if pk:
        src = os.path.dirname(pk[0]); dst = os.path.join(work, name + ".pt")
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as z:
            for dp, _, fs in os.walk(src):
                for f in fs:
                    p = os.path.join(dp, f)
                    zi = zipfile.ZipInfo(os.path.join(name, os.path.relpath(p, src)).replace(os.sep, "/"), (1980, 1, 1, 0, 0, 0))
                    z.writestr(zi, open(p, "rb").read(), zipfile.ZIP_STORED)
        print(f"weights: re-packed the unpacked folder {src} -> {dst}")
        return dst
    other = sorted(glob.glob(f"{inputs}/**/p6_trainable_*", recursive=True))
    print(f"NOT FOUND: {name}.pt | p6_trainable files in the attached inputs: {other or 'none'}")
    return None


def load_teacher(cfg_t, root, path, device):
    m, _ = build(cfg_t, root)
    st = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = m.load_state_dict(st, strict=False)
    assert not unexpected, unexpected[:5]
    for p in m.parameters(): p.requires_grad = False
    return m.to(device).eval()


def run_experiment(cfg, D, device, out_root, log=print, teacher=None, teacher_arch=None):
    t_start = time.time(); out_dir = os.path.join(out_root, cfg["run"]); os.makedirs(out_dir, exist_ok=True)
    tok = Tok(cfg["arch"], D["root"])
    T = {s: tok.docs(D[s], workers=D.get("workers", 4), log=log) for s in ("train", "val", "test")}
    lay = Layout(tok, cfg["arch"], cfg["G"], cfg.get("n_max", 8))
    lm = lay.fit_length_model(T["train"])
    log(f"layout: prompt {lay.P} tokens, summary part {lay.G} tokens | length model: summary ~ exp({lm['a']:.2f}) * doc^{lm['b']:.2f} (r {lm['r']:.2f})")
    tr = [p for it in T["train"] for p in lay.pairs(it)]; va = [p for it in T["val"] for p in lay.pairs(it)]
    nparts = Counter(lay.n_parts_gold(it) for it in T["train"])
    log(f"parts per training document: {dict(sorted(nparts.items()))}")
    torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    model, desc = build(cfg, D["root"]); model.to(device)
    total = sum(p.numel() for p in model.parameters()); trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"\n=== {cfg['run']} | {cfg['arch']} | {desc} | params {total:,}, trainable {trainable:,} ({trainable / total:.2%})"
        + (f" | KD from {teacher_arch}" if teacher is not None else ""))
    res = train(cfg, model, cfg["arch"], tr, va, device, out_dir, log, teacher, teacher_arch)
    teacher = None; torch.cuda.empty_cache()
    gk = dict(min_frac=cfg.get("min_frac", 0.5), no_repeat=cfg.get("no_repeat", 3), rep_pen=cfg.get("rep_pen", 1.15),
              batch=cfg.get("gen_batch", 16))
    nv = cfg.get("n_val_gen", 60); vi = T["val"][:nv]; refs = [it["doc"]["summary"] for it in vi]
    log(f"\ngenerating validation summaries ({len(vi)} documents)...")
    t0 = time.time(); vp = summarise(model, cfg["arch"], lay, vi, device, "pred", log=log, **gk); vt = time.time() - t0
    ev = dict(**rouge(vp, refs), **text_stats(vp, refs, [it["doc"]["document"] for it in vi]), sec_per_doc=round(vt / max(1, len(vi)), 2))
    log("VALIDATION (" + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in ev.items()) + ")")
    if cfg.get("ablate_single"):
        sp = summarise(model, cfg["arch"], lay, vi, device, "single", log=None, **gk)
        ev1 = dict(**rouge(sp, refs), **text_stats(sp, refs, [it["doc"]["document"] for it in vi]))
        log("ABLATION single pass (whole document compressed into one prompt): " + ", ".join(f"{k} {v:.4f}" for k, v in ev1.items()))
        ev["single_pass"] = ev1
    json.dump([dict(id=it["doc"]["id"], reference=r, generated=p) for it, r, p in zip(vi, refs, vp)],
              open(os.path.join(out_dir, "val_generations.json"), "w"), ensure_ascii=False, indent=1)
    log(f"example:\n  REFERENCE: {refs[0][:600]}\n  GENERATED: {vp[0][:600]}")
    log(f"\ngenerating test summaries ({len(T['test'])} documents)...")
    t0 = time.time(); tp = summarise(model, cfg["arch"], lay, T["test"], device, "pred", log=log, **gk); tt = time.time() - t0
    zp = write_submission(tp, D, out_dir, cfg["run"], log)
    rec = dict(run=cfg["run"], arch=cfg["arch"], method=desc, cfg=cfg, params_total=total, params_trainable=trainable,
               val=ev, test_minutes=round(tt / 60, 1), zip=os.path.basename(zp), minutes_total=round((time.time() - t_start) / 60, 1), **res)
    json.dump(rec, open(os.path.join(out_dir, "record.json"), "w"), indent=1, default=str)
    del model; torch.cuda.empty_cache()
    return rec


def write_submission(summaries, D, out_dir, name, log=print):
    sub = [dict(id=d["id"], summary=s) for d, s in zip(D["test"], summaries)]
    assert len(sub) == len(D["test"]) and all(isinstance(s["summary"], str) and s["summary"].strip() for s in sub)
    assert len({json.dumps(s["id"]) for s in sub}) == len(sub)
    os.makedirs(out_dir, exist_ok=True)
    pj = os.path.join(out_dir, "prediction.json"); json.dump(sub, open(pj, "w"), ensure_ascii=False)
    zp = os.path.join(out_dir, f"submission_{name}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z: z.write(pj, arcname="prediction.json")
    log(f"{name}: {len(sub):,} test summaries, {np.mean([len(s['summary'].split()) for s in sub]):.0f} words on average -> {zp}")
    return zp


# ======================================================================================================
# 7. No training: reload trained weights, compare decoding settings on validation, write test zips
# ======================================================================================================
VARIANTS = {
    "default":        dict(),                      # repetition penalty 1.15, no repeated 3-grams, length model x1.0
    "no_penalty":     dict(rep_pen=1.0),
    "strong_penalty": dict(rep_pen=1.3),
    "longer":         dict(len_scale=1.25),
    "longest":        dict(len_scale=1.45),
    "short_min":      dict(min_frac=0.25),
}
# v6: beam search + the v2 penalty (all tokens) - validation ROUGE did not rank these well against the leaderboard
# (BERTScore), so every one of them writes a test zip
BEAM_VARIANTS = {
    "longer":            dict(len_scale=1.25),                       # greedy reference (S6: leaderboard 0.8445)
    "v2pen_longer":      dict(len_scale=1.25, v2pen=True),
    "beam4_longer":      dict(len_scale=1.25, beams=4),
    "beam4_v2pen_longer": dict(len_scale=1.25, beams=4, v2pen=True),
    "beam4_longest":     dict(len_scale=1.45, beams=4),
}


def regenerate(cfg, D, device, out_root, weights, variants=None, n_val=60, log=print, zip_all=False):
    """cfg's model + its trained weights -> every decoding variant on n_val validation documents (ROUGE) ->
    test zips for the best variant (mean of ROUGE-1/2/L F1) and for 'default' (zip_all: for every variant)"""
    variants = variants or VARIANTS
    run = cfg["run"]; out_dir = os.path.join(out_root, run + "_regen"); os.makedirs(out_dir, exist_ok=True)
    tok = Tok(cfg["arch"], D["root"])
    T = {s: tok.docs(D[s], workers=D.get("workers", 4), log=log) for s in ("train", "val", "test")}
    lay = Layout(tok, cfg["arch"], cfg["G"], cfg.get("n_max", 8)); lay.fit_length_model(T["train"]); del T["train"]
    model, desc = build(cfg, D["root"])
    st = torch.load(weights, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(st, strict=False)
    assert not unexpected and len(st) > 0, unexpected[:5]
    model.to(device).eval()
    log(f"\n=== {run} | {cfg['arch']} | {desc} | weights {os.path.basename(weights)} ({len(st)} tensors)")
    vi = T["val"][:n_val]; refs = [it["doc"]["summary"] for it in vi]; docs = [it["doc"]["document"] for it in vi]
    res = {}
    for name, v in variants.items():
        gk = dict(min_frac=v.get("min_frac", cfg.get("min_frac", 0.5)), no_repeat=v.get("no_repeat", cfg.get("no_repeat", 3)),
                  rep_pen=v.get("rep_pen", cfg.get("rep_pen", 1.15)), batch=cfg.get("gen_batch", 16))
        for k in ("beams", "len_pen", "v2pen"):
            if k in v: gk[k] = v[k]
        lay.len_scale = v.get("len_scale", 1.0); t0 = time.time()
        vp = summarise(model, cfg["arch"], lay, vi, device, "pred", log=None, **gk)
        ev = dict(**rouge(vp, refs), **text_stats(vp, refs, docs)); ev["score"] = (ev["rouge1"] + ev["rouge2"] + ev["rougeL"]) / 3
        res[name] = dict(ev=ev, gk=gk, len_scale=lay.len_scale, example=vp[0][:600])
        log(f"{name:<15} ROUGE-1 {ev['rouge1']:.4f}  ROUGE-2 {ev['rouge2']:.4f}  ROUGE-L {ev['rougeL']:.4f}  mean {ev['score']:.4f} | "
            f"{ev['pred_words']:.0f} words (reference {ev['ref_words']:.0f}) | repeated 3-grams {ev['redundancy']:.3f} | {(time.time() - t0) / 60:.1f} min")
    best = max(res, key=lambda k: res[k]["ev"]["score"])
    log(f"\nbest decoding on validation: {best}\n  GENERATED: {res[best]['example']}\n  REFERENCE: {refs[0][:600]}")
    zips = {}
    for name in (list(res) if zip_all else dict.fromkeys([best, "default"])):
        if name not in res: continue
        lay.len_scale = res[name]["len_scale"]; t0 = time.time()
        tp = summarise(model, cfg["arch"], lay, T["test"], device, "pred", log=None, **res[name]["gk"])
        zips[name] = write_submission(tp, D, out_dir, f"{run}_{name}", log)
        log(f"  test generation {(time.time() - t0) / 60:.1f} min")
    rec = dict(run=run, arch=cfg["arch"], method=desc, best=best, variants={k: dict(v["ev"], **v["gk"], len_scale=v["len_scale"]) for k, v in res.items()},
               zips={k: os.path.basename(z) for k, z in zips.items()})
    json.dump(rec, open(os.path.join(out_dir, "regen_record.json"), "w"), indent=1, default=str)
    del model; torch.cuda.empty_cache()
    return rec


# ======================================================================================================
# 8. Ensemble decoding (no training): models with the SAME tokenizer + layout (arch 3 / arch 4, space BPE) average
#    their next-token log-probabilities at every step; members and weights chosen on validation
# ======================================================================================================
class EnsembleLM(nn.Module):
    """prefill / decode_step like arch 3 / 4 -> weighted mean of the members' next-token logits"""

    def __init__(self, models, weights):
        super().__init__()
        self.members = nn.ModuleList(models); self.w = [float(x) / sum(weights) for x in weights]

    def _mix(self, outs):
        # weighted mean of raw logits: softmax of it = normalised product of the members' distributions (log-prob
        # mixing, the per-row constants cancel); raw logits keep the repetition penalty's behaviour of single models
        lg = sum(w * l.float() for w, (l, _) in zip(self.w, outs))
        return lg, [c for _, c in outs]

    def prefill(self, P):
        return self._mix([m.prefill(P) for m in self.members])

    def decode_step(self, tok, caches, pos):
        return self._mix([m.decode_step(tok, c, pos) for m, c in zip(self.members, caches)])


def ensemble(members, D, device, out_root, combos, n_val=60, G=256, gk=None, len_scale=1.0, log=print):
    """members: name -> (cfg, weights path), all arch3/arch4. combos: list of {name: weight}.
    -> validation ROUGE of every combo (singles included) -> test zips for the best combo and the best arch-4-only combo"""
    gk = dict(dict(min_frac=0.5, no_repeat=3, rep_pen=1.15, batch=16), **(gk or {}))
    assert all(c["arch"] in ("arch3_prefixlm", "arch4_mla") for c, _ in members.values())
    out_dir = os.path.join(out_root, "p6_ensemble"); os.makedirs(out_dir, exist_ok=True)
    tok = Tok("arch4_mla", D["root"])
    T = {s: tok.docs(D[s], workers=D.get("workers", 4), log=log) for s in ("train", "val", "test")}
    lay = Layout(tok, "arch4_mla", G); lay.fit_length_model(T["train"]); del T["train"]; lay.len_scale = len_scale
    log(f"decoding: {gk} | length model x{len_scale}")
    M = {}
    for name, (cfg, w) in members.items():
        assert cfg["G"] == G
        m, desc = build(cfg, D["root"]); st = torch.load(w, map_location="cpu", weights_only=False)
        _, unexpected = m.load_state_dict(st, strict=False); assert not unexpected and st, unexpected[:5]
        for p in m.parameters(): p.requires_grad = False
        M[name] = m.to(device).eval(); log(f"member {name}: {cfg['arch']} | {desc} | {os.path.basename(w)}")
    vi = T["val"][:n_val]; refs = [it["doc"]["summary"] for it in vi]; docs = [it["doc"]["document"] for it in vi]
    combos = [dict(c) for c in combos if all(k in M for k in c)]
    combos = [{k: 1} for k in M] + [c for c in combos if len(c) > 1]
    label = lambda c: " + ".join(f"{w}x{k}" if w != 1 else k for k, w in c.items())
    res = []
    for c in combos:
        t0 = time.time()
        model = M[next(iter(c))] if len(c) == 1 else EnsembleLM([M[k] for k in c], list(c.values()))
        vp = summarise(model, "arch4_mla", lay, vi, device, "pred", log=None, **gk)
        ev = dict(**rouge(vp, refs), **text_stats(vp, refs, docs)); ev["score"] = (ev["rouge1"] + ev["rouge2"] + ev["rougeL"]) / 3
        res.append(dict(combo=c, ev=ev, example=vp[0][:500]))
        log(f"{label(c):<40} ROUGE-1 {ev['rouge1']:.4f}  ROUGE-2 {ev['rouge2']:.4f}  ROUGE-L {ev['rougeL']:.4f}  mean {ev['score']:.4f} | "
            f"{ev['pred_words']:.0f} words | repeated 3-grams {ev['redundancy']:.3f} | {(time.time() - t0) / 60:.1f} min")
    arch = lambda k: members[k][0]["arch"]
    best = max(res, key=lambda r: r["ev"]["score"])
    a4 = [r for r in res if all(arch(k) == "arch4_mla" for k in r["combo"]) and len(r["combo"]) > 1]
    best_a4 = max(a4, key=lambda r: r["ev"]["score"]) if a4 else None
    log(f"\nbest on validation: {label(best['combo'])} (mean {best['ev']['score']:.4f})"
        + (f" | best arch-4-only ensemble: {label(best_a4['combo'])} (mean {best_a4['ev']['score']:.4f})" if best_a4 else ""))
    log(f"  GENERATED: {best['example']}\n  REFERENCE: {refs[0][:500]}")
    zips = {}
    for tag, r in (("best", best), ("arch4_only", best_a4)):
        if r is None or len(r["combo"]) == 1 or any(r["combo"] == z["combo"] for z in zips.values()): continue
        t0 = time.time(); c = r["combo"]
        model = EnsembleLM([M[k] for k in c], list(c.values()))
        tp = summarise(model, "arch4_mla", lay, T["test"], device, "pred", log=None, **gk)
        name = "ens_" + "_".join(f"{k}{'' if w == 1 else f'x{w}'}" for k, w in c.items())
        zips[tag] = dict(combo=c, zip=write_submission(tp, D, out_dir, name, log), val=r["ev"])
        log(f"  test generation {(time.time() - t0) / 60:.1f} min")
    rec = dict(results=[dict(combo=label(r["combo"]), **r["ev"]) for r in res],
               zips={k: dict(combo=label(v["combo"]), zip=os.path.basename(v["zip"]), **v["val"]) for k, v in zips.items()})
    json.dump(rec, open(os.path.join(out_dir, "ensemble_record.json"), "w"), indent=1, default=str)
    del M; torch.cuda.empty_cache()
    return rec
