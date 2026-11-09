# Phase 2 (L-NER) model side:
#   Phase-1 backbones as feature extractors (all 4 architectures, bidirectional or causal),
#   fine-tuning / parameter-efficient methods (full, partial, BitFit, LoRA, LoRMA, Houlsby/Pfeiffer adapters,
#   linear probe), prediction heads (linear softmax, CRF, span classifier), knowledge distillation,
#   sliding-window training and document-level prediction.
import bisect
import math
import os
import pickle
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import ner_data as nd

ARCH_INFO = {   # folder -> (tokenizer file, longest sequence the backbone supports)
    "arch1_decoder": ("a1_bpe8000.pkl", 1024),
    "arch2_encdec": ("a1_bpe8000.pkl", 768),
    "arch3_prefixlm": ("space_bpe8000.pkl", 1024),
    "arch4_mla": ("space_bpe8000.pkl", 1024),
}
NEG = -1e4


def autocast():
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available())


def _add_code_path(root):
    code = os.path.join(root, "code")
    if code not in sys.path:
        sys.path.insert(0, code)


def load_tokenizer(arch, root):
    _add_code_path(root)
    import a1_tokenizer, space_bpe  # noqa: F401  (classes needed to unpickle)
    return pickle.load(open(os.path.join(root, "tokenizers", ARCH_INFO[arch][0]), "rb"))


# ======================================================================================================
# 1. Backbones
# ======================================================================================================
def _mla_forward(self, x, cos, sin, cache=None):
    """MLA attention of architecture 4 without the causal mask (used when self.bidirectional is True)."""
    if cache is not None or not getattr(self, "bidirectional", False):
        return self._causal_forward(x, cos, sin, cache)
    from rope import apply_rope
    B, T, _ = x.shape
    q = self.q_proj(x).view(B, T, self.h, self.nope + self.rope).transpose(1, 2)
    q_nope, q_rope = q.split([self.nope, self.rope], dim=-1)
    q_rope = apply_rope(q_rope, cos.to(q.dtype), sin.to(q.dtype))
    c_kv = self.kv_norm(self.kv_down(x))
    k_r = apply_rope(self.k_rope(x)[:, None], cos.to(q.dtype), sin.to(q.dtype))
    k_nope = self.k_up(c_kv).view(B, T, self.h, self.nope).transpose(1, 2)
    v = self.v_up(c_kv).view(B, T, self.h, self.vd).transpose(1, 2)
    q = torch.cat([q_nope, q_rope], dim=-1)
    k = torch.cat([k_nope, k_r.expand(B, self.h, T, self.rope)], dim=-1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=self.dropout if self.training else 0.0)
    return self.o_proj(out.transpose(1, 2).reshape(B, T, -1)), None


class Backbone(nn.Module):
    """Phase-1 language model reduced to a feature extractor: token ids (B,T) -> hidden states (B,T,d).
    bidirectional: drop the causal mask (arch 1, 4), use the whole window as prefix (arch 3); the arch-2
    encoder is bidirectional by design. keep_layers: keep only the first k blocks (distillation student).
    init="random": same architecture, random weights (control experiment for transfer from Phase 1)."""

    def __init__(self, arch, root, bidirectional=True, keep_layers=None, init="pretrained"):
        super().__init__()
        _add_code_path(root)
        self.arch, self.bidirectional = arch, bidirectional
        ck = torch.load(os.path.join(root, arch, "ckpt_best.pt"), map_location="cpu", weights_only=False)
        cfg = dict(ck["cfg"])
        if arch == "arch4_mla":
            from mla import MLAConfig, MLALM, MLAttention
            if not hasattr(MLAttention, "_causal_forward"):
                MLAttention._causal_forward = MLAttention.forward
                MLAttention.forward = _mla_forward
            m = MLALM(MLAConfig(**cfg))
        elif arch == "arch3_prefixlm":
            from prefixlm import PrefixLMConfig, PrefixLM
            m = PrefixLM(PrefixLMConfig(**cfg))
        elif arch == "arch2_encdec":
            from encdec import EncDecConfig, EncDecLM
            m = EncDecLM(EncDecConfig(**cfg))
        elif arch == "arch1_decoder":
            from model_arch import LMConfig, DecoderLM
            m = DecoderLM(LMConfig(**cfg))
        else:
            raise ValueError(arch)
        if init == "pretrained":
            missing, unexpected = m.load_state_dict(ck["model"], strict=False)
            assert not unexpected and all(x in ("rope", "pos_emb", "mask") or "rope" in x or "mask" in x
                                          or "pos_emb" in x for x in missing), (missing[:5], unexpected[:5])
        else:
            for mod in m.modules():
                if isinstance(mod, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(mod.weight, std=0.02)
                    if getattr(mod, "bias", None) is not None: nn.init.zeros_(mod.bias)
        self.phase1_step, self.phase1_val = ck.get("step"), ck.get("val_loss")
        if arch == "arch2_encdec":                        # NER uses only the (bidirectional) encoder
            m.decoder = nn.ModuleList(); m.dec_norm = nn.Identity()
        if arch == "arch1_decoder":                       # A3 attention keeps its causal mask in a buffer
            for blk in m.blocks:
                blk.attn.causal_mask.fill_(True) if bidirectional else None
        if arch == "arch4_mla":
            for blk in m.blocks:
                blk.attn.bidirectional = bidirectional
        if keep_layers:
            blocks = self._blocks(m)
            del blocks[keep_layers:]
        m.head = None if arch != "arch2_encdec" else m.head   # LM head is not used (tied to the embedding)
        self.m, self.d, self.max_len = m, cfg["d_model"], ARCH_INFO[arch][1]

    @staticmethod
    def _blocks(m):
        return m.encoder if hasattr(m, "encoder") else m.blocks

    @property
    def blocks(self):
        return self._blocks(self.m)

    @property
    def final_norm(self):
        m = self.m
        return m.enc_norm if hasattr(m, "enc_norm") else (m.ln_f if hasattr(m, "ln_f") else m.norm)

    def forward(self, ids):
        m = self.m
        if self.arch == "arch4_mla":
            return m.hidden(ids)
        if self.arch == "arch3_prefixlm":
            return m.hidden(ids, prefix_len=ids.size(1) if self.bidirectional else 0)
        if self.arch == "arch2_encdec":
            return m.encode(ids)
        B, T = ids.shape                                  # arch 1: DecoderLM.forward without the LM head
        x = m.tok_emb(ids)
        if m.cfg.scale_emb:
            x = x * math.sqrt(m.cfg.d_model)
        if m.cfg.pos_emb_type == "sinusoidal":
            x = x + m.pos_emb[:T].unsqueeze(0)
        for blk in m.blocks:
            x = blk(x)
        return m.ln_f(x)


# ======================================================================================================
# 2. Fine-tuning / adaptation methods
# ======================================================================================================
class LoRALinear(nn.Module):
    """LoRA (Hu et al. 2021): W x + (alpha/r) * B A x, with W frozen, A random, B = 0 (starts at W)."""

    def __init__(self, base, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(self.drop(x), self.A), self.B)


class LoRMALinear(nn.Module):
    """LoRMA (low-rank MULTIPLICATIVE adaptation): W' = (I + (alpha/r) B A) W, with W frozen.
    The identity term gives the product full rank ("rank inflation" by identity), and B = 0 makes the
    adapted layer start exactly at the pretrained one. Computed as h + s * B(A h) with h = W x, so the
    d_out x d_in product (BA)W is never formed. A bias, if any, is added after the multiplicative update."""

    def __init__(self, base, r=16, alpha=32):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.out_features)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        h = F.linear(x, self.base.weight)
        h = h + self.scale * F.linear(F.linear(h, self.A), self.B)
        return h if self.base.bias is None else h + self.base.bias


class Adapter(nn.Module):
    """bottleneck adapter (Houlsby et al. 2019): x + up(GELU(down(x))), up = 0 at start (identity)."""

    def __init__(self, d, bottleneck=64):
        super().__init__()
        self.down, self.up = nn.Linear(d, bottleneck), nn.Linear(bottleneck, d)
        nn.init.normal_(self.down.weight, std=0.02); nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))


class AdapterWrap(nn.Module):
    """applies an adapter to the output of a sub-layer (attention or feed-forward), before the residual add"""

    def __init__(self, module, d, bottleneck):
        super().__init__()
        self.module, self.adapter = module, Adapter(d, bottleneck)

    def __getattr__(self, name):                      # keep attributes of the wrapped module reachable
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["module"], name)

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        if isinstance(out, tuple):
            return (self.adapter(out[0]),) + tuple(out[1:])
        return self.adapter(out)


def _replace_linears(root_module, make):
    n = 0
    for name, child in list(root_module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(root_module, name, make(child)); n += 1
        else:
            n += _replace_linears(child, make)
    return n


def apply_method(net, cfg):
    """Freeze/unfreeze and insert adaptation modules according to cfg["method"]. The task head is always
    trained. Returns a short description for the record."""
    bb, method = net.backbone, cfg["method"]
    for p in bb.parameters():
        p.requires_grad = method == "full_ft"
    desc = method
    if method == "partial":                           # layer-wise: only the top k blocks + final norm
        k = cfg.get("train_top_layers", 4)
        for blk in list(bb.blocks)[-k:]:
            for p in blk.parameters(): p.requires_grad = True
        for p in bb.final_norm.parameters(): p.requires_grad = True
        desc = f"partial (top {k} of {len(bb.blocks)} blocks)"
    elif method == "bitfit":                          # only bias terms (Ben Zaken et al. 2022)
        n = 0
        for name, p in bb.named_parameters():
            if name.endswith(".bias"):
                p.requires_grad = True; n += p.numel()
        assert n > 0, "BitFit needs bias terms - this backbone has none (use arch 1 or arch 2)"
        desc = f"bitfit ({n:,} bias parameters)"
    elif method in ("lora", "lorma"):
        r, a = cfg.get("rank", 16), cfg.get("alpha", 32)
        make = (lambda lin: LoRALinear(lin, r, a, cfg.get("lora_dropout", 0.05))) if method == "lora" \
            else (lambda lin: LoRMALinear(lin, r, a))
        n = sum(_replace_linears(blk, make) for blk in bb.blocks)
        for mod in bb.modules():
            if isinstance(mod, (LoRALinear, LoRMALinear)):
                mod.A.requires_grad = True; mod.B.requires_grad = True
        desc = f"{method} (rank {r}, alpha {a}, {n} linear layers)"
    elif method in ("houlsby", "pfeiffer"):           # Houlsby: after attention and FFN; Pfeiffer: after FFN only
        bott = cfg.get("bottleneck", 64)
        for blk in bb.blocks:
            if method == "houlsby":
                blk.attn = AdapterWrap(blk.attn, bb.d, bott)
            blk.ffn = AdapterWrap(blk.ffn, bb.d, bott)
        for mod in bb.modules():
            if isinstance(mod, Adapter):
                for p in mod.parameters(): p.requires_grad = True
            if isinstance(mod, (nn.LayerNorm,)) or type(mod).__name__ == "RMSNorm":
                for p in mod.parameters(): p.requires_grad = True     # Houlsby also trains the norms
        desc = f"{method} adapters (bottleneck {bott})"
    elif method == "probe":                           # frozen backbone, only the head (linear probe)
        desc = "linear probe (frozen backbone)"
    elif method != "full_ft":
        raise ValueError(method)
    return desc


def count_params(net):
    total = sum(p.numel() for p in net.parameters())
    train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total, train


# ======================================================================================================
# 3. Prediction heads
# ======================================================================================================
class CRF(nn.Module):
    """linear-chain CRF over word tags. Learned transitions + fixed BIO constraints (forbidden moves = -1e4)."""

    def __init__(self, tags):
        super().__init__()
        K = len(tags)
        ok, start_ok = nd.allowed_transitions(tags)
        self.trans = nn.Parameter(torch.zeros(K, K)); self.start = nn.Parameter(torch.zeros(K))
        self.end = nn.Parameter(torch.zeros(K))
        self.register_buffer("ban", (~torch.tensor(ok)).float() * NEG)
        self.register_buffer("ban_start", (~torch.tensor(start_ok)).float() * NEG)

    def scores(self, train=False):
        # training windows may start inside an entity (first word I-X), so the start constraint is only
        # applied when decoding whole documents
        return self.trans + self.ban, (self.start if train else self.start + self.ban_start), self.end

    def nll(self, em, tags, mask):
        """em (B,W,K) float, tags (B,W) (pads ignored), mask (B,W) bool, prefix-contiguous"""
        trans, start, end = self.scores(train=True)
        B, W, K = em.shape
        tags = tags.clamp(min=0)
        mf = mask.float()
        # score of the gold path
        gold = start[tags[:, 0]] + em[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, W):
            step = trans[tags[:, t - 1], tags[:, t]] + em[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            gold = gold + step * mf[:, t]
        last = tags.gather(1, (mask.sum(1) - 1).clamp(min=0)[:, None]).squeeze(1)
        gold = gold + end[last]
        # log partition
        alpha = start[None] + em[:, 0]
        for t in range(1, W):
            nxt = torch.logsumexp(alpha[:, :, None] + trans[None], dim=1) + em[:, t]
            alpha = torch.where(mask[:, t, None], nxt, alpha)
        logz = torch.logsumexp(alpha + end[None], dim=1)
        has = mask[:, 0]
        return ((logz - gold) * has.float()).sum() / has.float().sum().clamp(min=1)


def viterbi(em, trans, start, end):
    """em (n,K) -> best tag sequence (list) under transition scores (CPU)"""
    n, K = em.shape
    if n == 0:
        return []
    score = start + em[0]
    back = []
    for t in range(1, n):
        s = score[:, None] + trans
        best, idx = s.max(0)
        score = best + em[t]; back.append(idx)
    score = score + end
    tag = int(score.argmax())
    path = [tag]
    for idx in reversed(back):
        tag = int(idx[tag]); path.append(tag)
    return path[::-1]


class NERModel(nn.Module):
    """backbone -> word representations (first token of each word) -> head.
    head = "linear": softmax over BIO tags (cross-entropy, optional class weights)
           "crf":    BIO tag emissions + CRF (sequence-level likelihood)
           "span":   classify every word span up to max_span_words (start/end projections, bilinear term,
                     width embedding), label 0 = not an entity."""

    def __init__(self, backbone, tags, labels, head="linear", dropout=0.2, class_weights=None,
                 max_span_words=32, span_dim=256, neg_weight=0.5):
        super().__init__()
        self.backbone, self.head_type, self.tags = backbone, head, tags
        self.id2label = {i + 1: l for i, l in enumerate(labels)}          # span head: class 0 = no entity
        n_labels = len(labels)
        d, K = backbone.d, len(tags)
        self.drop = nn.Dropout(dropout)
        if head in ("linear", "crf"):
            self.head = nn.Linear(d, K)
            nn.init.normal_(self.head.weight, std=0.02); nn.init.zeros_(self.head.bias)
            self.register_buffer("class_weights", class_weights if class_weights is not None else torch.ones(K))
            if head == "crf":
                self.crf = CRF(tags)
        elif head == "span":
            C = n_labels + 1
            self.maxw = max_span_words
            self.head = nn.ModuleDict(dict(
                s=nn.Sequential(nn.Linear(d, span_dim), nn.GELU()), e=nn.Sequential(nn.Linear(d, span_dim), nn.GELU()),
                ls=nn.Linear(span_dim, C), le=nn.Linear(span_dim, C), bil=nn.Linear(span_dim, C, bias=False),
                width=nn.Embedding(max_span_words, C)))
            w = torch.ones(C); w[0] = neg_weight
            self.register_buffer("class_weights", w)
        else:
            raise ValueError(head)

    def word_states(self, ids, wpos):
        h = self.backbone(ids)                                            # (B,T,d)
        return h.gather(1, wpos[:, :, None].expand(-1, -1, h.size(-1)))   # (B,W,d)

    def span_logits(self, hw, wmask):
        H = self.head
        s, e = H["s"](hw), H["e"](hw)                                     # (B,W,p)
        B, W, p = s.shape
        k = torch.arange(self.maxw, device=hw.device)
        j = torch.arange(W, device=hw.device)[:, None] + k[None]          # (W,maxw)
        valid = (j < W)[None] & wmask[:, :, None] & wmask.gather(1, j.clamp(max=W - 1).reshape(1, -1).expand(B, -1)).view(B, W, -1)
        ej = e.gather(1, j.clamp(max=W - 1).reshape(1, -1, 1).expand(B, -1, p)).view(B, W, self.maxw, p)
        logits = H["ls"](s)[:, :, None] + H["le"](ej) + H["bil"](s[:, :, None] * ej) + H["width"](k)[None, None]
        return logits.float(), valid

    def forward(self, ids, wpos, wmask, wtags=None, stgt=None):
        with autocast():
            hw = self.drop(self.word_states(ids, wpos))
            if self.head_type == "span":
                logits, valid = self.span_logits(hw, wmask)
            else:
                logits = self.head(hw).float()
        if self.head_type == "span":
            if stgt is None:                                   # invalid slots -> certainly "no entity"
                inv = ~valid
                out = logits.masked_fill(inv[..., None], NEG)
                out[..., 0] = out[..., 0].masked_fill(inv, 0.0)
                return out
            tgt = stgt.masked_fill(~valid, nd.IGNORE)
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                   weight=self.class_weights, ignore_index=nd.IGNORE).reshape(1)
        if wtags is None:
            return logits
        if self.head_type == "crf":
            return self.crf.nll(logits, wtags, wmask).reshape(1)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), wtags.reshape(-1),
                               weight=self.class_weights, ignore_index=nd.IGNORE).reshape(1)


class KDWrapper(nn.Module):
    """Knowledge distillation (Hinton et al. 2015) at word level:
    loss = a * CE(student, gold) + (1 - a) * T^2 * KL(teacher_T || student_T). The teacher is frozen."""

    def __init__(self, student, teacher, T=2.0, alpha=0.5):
        super().__init__()
        self.student, self.teacher, self.T, self.alpha = student, teacher, T, alpha
        for p in teacher.parameters(): p.requires_grad = False

    def train(self, mode=True):
        super().train(mode); self.teacher.eval(); return self

    def forward(self, ids, wpos, wmask, wtags=None, stgt=None):
        if wtags is None:
            return self.student(ids, wpos, wmask)
        with torch.no_grad():
            t = self.teacher(ids, wpos, wmask)                            # (B,W,K)
        s = self.student(ids, wpos, wmask)
        m = wmask.reshape(-1)
        ce = F.cross_entropy(s.reshape(-1, s.size(-1)), wtags.reshape(-1), ignore_index=nd.IGNORE)
        sl, tl = s.reshape(-1, s.size(-1))[m] / self.T, t.reshape(-1, t.size(-1))[m] / self.T
        kl = F.kl_div(F.log_softmax(sl, -1), F.softmax(tl, -1), reduction="batchmean") * self.T ** 2
        return (self.alpha * ce + (1 - self.alpha) * kl).reshape(1)


# ======================================================================================================
# 4. Documents -> windows -> batches
# ======================================================================================================
def prepare_docs(texts, gold, tok, tag2id, label2id, split="fine"):
    docs, conflicts = {}, 0
    for d, t in texts.items():
        ids, wtok, wcs, wce = nd.encode_words(tok, t, split)
        r = dict(ids=ids, wtok=wtok, wcs=wcs, wce=wce)
        if gold is not None:
            r["wtags"], r["wspans"], c = nd.word_targets(wcs, wce, gold.get(d, []), tag2id, label2id); conflicts += c
        docs[d] = r
    return docs, conflicts


class Windowing:
    """Cuts every document into windows of `size` tokens. With header_tokens > 0, the first tokens of the
    document (case title, parties, counsel) are put in front of every window, followed by a separator,
    so each window can see who the appellant/respondent/counsel are. Header words are never labelled."""

    def __init__(self, tok, size, header_tokens=0, max_len=1024):
        self.sep = tok.encode("\n\n") if header_tokens else []
        self.H = header_tokens
        self.size = min(size, max_len - (header_tokens + len(self.sep) if header_tokens else 0))
        assert self.size >= 64, "window too small"

    def windows(self, docs, stride):
        out = []
        for d, r in docs.items():
            if r["wtok"]:
                for s in nd.windows(len(r["ids"]), self.size, stride):
                    out.append((d, s, min(self.size, len(r["ids"]))))
        return out

    def window_words(self, r, s, L):
        w0 = bisect.bisect_left(r["wtok"], s); w1 = bisect.bisect_left(r["wtok"], s + L)
        return w0, w1

    def batch(self, docs, b, device, head=None, maxw=None):
        ids, wpos, wtags, ranges, stgts = [], [], [], [], []
        for d, s, L in b:
            r = docs[d]
            head_ids = (r["ids"][:self.H] + self.sep) if self.H else []
            off = len(head_ids) - s
            ids.append(head_ids + r["ids"][s:s + L])
            w0, w1 = self.window_words(r, s, L)
            ranges.append((w0, w1))
            wpos.append([r["wtok"][w] + off for w in range(w0, w1)])
            if "wtags" in r:
                wtags.append(r["wtags"][w0:w1])
        T = len(ids[0]); assert all(len(x) == T for x in ids)
        Wm = max(1, max(len(x) for x in wpos))
        P = torch.zeros(len(b), Wm, dtype=torch.long); M = torch.zeros(len(b), Wm, dtype=torch.bool)
        for i, x in enumerate(wpos):
            P[i, :len(x)] = torch.tensor(x, dtype=torch.long); M[i, :len(x)] = True
        out = dict(ids=torch.tensor(ids, dtype=torch.long), wpos=P, wmask=M)
        if wtags:
            Tg = torch.full((len(b), Wm), nd.IGNORE, dtype=torch.long)
            for i, x in enumerate(wtags):
                Tg[i, :len(x)] = torch.tensor(x, dtype=torch.long)
            out["wtags"] = Tg
            if head == "span":
                S = torch.zeros(len(b), Wm, maxw, dtype=torch.long)
                for i, ((d, s, L), (w0, w1)) in enumerate(zip(b, ranges)):
                    for gi, gj, l in docs[d]["wspans"]:
                        if gj < w0 or gi >= w1:
                            continue
                        if gi >= w0 and gj < w1:                    # entity inside the window
                            if gj - gi < maxw: S[i, gi - w0, gj - gi] = l
                        elif gi >= w0:                              # runs past the window end: unknown
                            S[i, gi - w0, :] = nd.IGNORE
                        else:                                       # started before the window: unknown
                            for a in range(max(w0, gj - maxw + 1), gj + 1):
                                S[i, a - w0, gj - a] = nd.IGNORE
                out["stgt"] = S
        return {k: v.to(device) for k, v in out.items()}, ranges


def length_batches(win, bs, shuffle, rng=None):
    by_len = {}
    for w in win:
        by_len.setdefault(w[2], []).append(w)
    out = []
    for _, ws in sorted(by_len.items()):
        if shuffle: rng.shuffle(ws)
        out += [ws[i:i + bs] for i in range(0, len(ws), bs)]
    if shuffle: rng.shuffle(out)
    return out


# ======================================================================================================
# 5. Prediction: windows -> document-level scores -> spans
# ======================================================================================================
@torch.no_grad()
def predict_scores(net, docs, wd, stride, bs, device):
    """returns per document: linear -> averaged tag probabilities (n_words,K); crf -> averaged emissions;
    span -> {(i,j): averaged class probabilities}. Overlapping windows are combined with a tent weight
    (a word in the middle of a window has more context than one at its edge)."""
    net.eval()
    model = net.student if isinstance(net, KDWrapper) else net
    head = model.head_type
    out = {}
    for b in length_batches(wd.windows(docs, stride), bs, shuffle=False):
        batch, ranges = wd.batch(docs, b, device)
        res = net(batch["ids"], batch["wpos"], batch["wmask"]).float().cpu()
        for k, ((d, s, L), (w0, w1)) in enumerate(zip(b, ranges)):
            n = w1 - w0
            if n == 0:
                continue
            r = docs[d]
            pos = torch.tensor([r["wtok"][w] - s for w in range(w0, w1)], dtype=torch.float)
            wt = 1 + torch.minimum(pos, L - 1 - pos)
            if head == "span":
                cand = out.setdefault(d, {})
                pr = res[k, :n].softmax(-1)                                 # (n, maxw, C)
                keep = (pr[..., 1:].max(-1).values > 0.2).nonzero().tolist()     # plausible entities only
                for i, kk in keep:
                    j = i + kk
                    if j >= n: continue
                    key = (w0 + i, w0 + j); ww = float(min(wt[i], wt[j]))
                    acc = cand.get(key)
                    cand[key] = [pr[i, kk] * ww, ww] if acc is None else [acc[0] + pr[i, kk] * ww, acc[1] + ww]
            else:
                if d not in out:
                    nw = len(r["wtok"]); out[d] = [torch.zeros(nw, res.size(-1)), torch.zeros(nw, 1)]
                v = res[k, :n].softmax(-1) if head == "linear" else res[k, :n]
                out[d][0][w0:w1] += v * wt[:, None]; out[d][1][w0:w1] += wt[:, None]
    final = {}
    for d in docs:
        if head == "span":
            final[d] = {key: a / w for key, (a, w) in out.get(d, {}).items()}
        else:
            v = out.get(d)
            final[d] = (v[0] / v[1].clamp(min=1e-9)) if v is not None else torch.zeros(0, len(model.tags))
    return final


def decode(scores, docs, model, texts, strategy="viterbi", propagate=False):
    """document scores -> {doc: [(s, e, label)]}.
    linear head: "greedy" (argmax per word + lenient BIO repair) or "viterbi" (best BIO-valid sequence
    under the word log-probabilities); crf head: Viterbi with the learned transitions; span head: greedy
    non-overlapping spans by probability. propagate: document-level label consistency (see ner_data)."""
    head, tags = model.head_type, model.tags
    id2tag = dict(enumerate(tags))
    pred = {}
    if head != "span":
        if head == "crf":
            trans, start, end = [x.detach().float().cpu() for x in model.crf.scores()]
        else:
            ok, start_ok = nd.allowed_transitions(tags)
            trans = (~torch.tensor(ok)).float() * NEG; start = (~torch.tensor(start_ok)).float() * NEG
            end = torch.zeros(len(tags))
    for d, sc in scores.items():
        r = docs[d]
        if head == "span":
            cands = {k: (float(v[1:].max()), int(v.argmax())) for k, v in sc.items()}
            spans = nd.spans_from_candidates(cands, r["wcs"], r["wce"], model.id2label)
        elif head == "linear" and strategy == "greedy":
            spans = nd.spans_from_tags(sc.argmax(-1).tolist(), r["wcs"], r["wce"], id2tag)
        else:
            em = sc.clamp(min=1e-9).log() if head == "linear" else sc
            spans = nd.spans_from_tags(viterbi(em, trans, start, end), r["wcs"], r["wce"], id2tag)
        pred[d] = nd.propagate(spans, texts[d]) if propagate else spans
    return pred


def decoding_options(head):
    base = ["greedy", "viterbi"] if head == "linear" else ["viterbi"] if head == "crf" else ["greedy"]
    return [(s, p) for s in base for p in (False, True)]


# ======================================================================================================
# 6. Training
# ======================================================================================================
def train(net, train_docs, val_docs, val_texts, val_gold, labels, wd, cfg, device, log=print):
    """AdamW (backbone/adaptation LR + head LR), linear warm-up and decay, fp16, gradient accumulation,
    DataParallel over all GPUs; evaluates strict macro-F1 on the validation documents after every epoch
    and keeps the best weights (early stopping with patience)."""
    rng = random.Random(cfg["seed"])
    model = net.student if isinstance(net, KDWrapper) else net
    gpus = torch.cuda.device_count()
    para = nn.DataParallel(net) if gpus > 1 else net
    is_head = lambda n: n.startswith(("head.", "crf.", "student.head.", "student.crf."))
    named = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    groups = [dict(params=[p for n, p in named if not is_head(n)], lr=cfg["lr"]),
              dict(params=[p for n, p in named if is_head(n)], lr=cfg["lr_head"])]
    groups = [g for g in groups if g["params"]]
    for g in groups: g["base_lr"] = g["lr"]
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    win = wd.windows(train_docs, cfg["train_stride"])
    acc = cfg.get("accum", 1)
    steps_per_epoch = math.ceil(len(win) / cfg["batch_size"] / acc)
    total = steps_per_epoch * cfg["epochs"]; warm = max(1, int(cfg["warmup_frac"] * total))
    log(f"train windows {len(win):,} of {wd.size} tokens (+{wd.H} header) | {steps_per_epoch} steps/epoch x "
        f"{cfg['epochs']} epochs | trainable {sum(p.numel() for p in params):,} | GPUs {max(gpus, 1)}")
    best, best_state, history, step, bad, t0 = -1.0, None, [], 0, 0, time.time()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    maxw = getattr(model, "maxw", None)
    for ep in range(1, cfg["epochs"] + 1):
        net.train(); run, n = 0.0, 0
        bl = length_batches(win, cfg["batch_size"], shuffle=True, rng=rng)
        for bi, b in enumerate(bl):
            batch, _ = wd.batch(train_docs, b, device, model.head_type, maxw)
            loss = para(**batch).mean() / acc
            scaler.scale(loss).backward()
            run += loss.item() * acc; n += 1
            if (bi + 1) % acc and bi + 1 < len(bl):
                continue
            step += 1
            f = step / warm if step <= warm else max(0.0, (total - step) / max(1, total - warm))
            for g in opt.param_groups: g["lr"] = g["base_lr"] * f
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg["clip"])
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if step % cfg["log_every"] == 0:
                log(f"  ep {ep} step {step}/{total} | loss {run / n:.4f} | {(time.time() - t0) / 60:.1f} min")
                run, n = 0.0, 0
        if cfg.get("select_last"):                    # refit on all documents: fixed epochs, no selection
            history.append(dict(epoch=ep, step=step, minutes=round((time.time() - t0) / 60, 1)))
            log(f"epoch {ep} done (refit on all documents: no validation-based selection)")
            continue
        sc = predict_scores(net, val_docs, wd, cfg["eval_stride"], cfg["eval_batch"], device)
        strategy = "viterbi" if model.head_type != "span" else "greedy"
        res = nd.strict_scores(decode(sc, val_docs, model, val_texts, strategy), val_gold, labels)
        history.append(dict(epoch=ep, step=step, macro_F1=res["macro_F1"], micro_F1=res["micro_F1"],
                            macro_P=res["macro_P"], macro_R=res["macro_R"], minutes=round((time.time() - t0) / 60, 1)))
        note = ""
        if res["macro_F1"] > best + 1e-4:
            best, bad, note = res["macro_F1"], 0, "  <- best"
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}
        else:
            bad += 1
        log(f"epoch {ep}: val strict macro-F1 {res['macro_F1']:.4f} | micro {res['micro_F1']:.4f} | "
            f"P {res['macro_P']:.3f} R {res['macro_R']:.3f}{note}")
        if cfg.get("patience") and bad >= cfg["patience"]:
            log(f"early stopping: no improvement for {bad} epochs"); break
        if (time.time() - t0) / 3600 > cfg["max_train_hours"]:
            log("time budget reached - stopping"); break
    if best_state is not None:
        net.load_state_dict(best_state)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    return dict(best_val_macro_F1=best, history=history, train_minutes=round((time.time() - t0) / 60, 1),
                peak_gpu_gb=round(peak, 2))


def trainable_state(net):
    """only the trained tensors (what a PEFT method has to store per task)"""
    names = {n for n, p in net.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in net.state_dict().items() if k in names}


@torch.no_grad()
def self_test(net, wd, docs, device, head, maxw=None):
    """one forward + backward on two real windows before the long run (catches shape/device errors early)"""
    b = wd.windows(docs, wd.size)[:2]
    if len(b) == 2 and b[0][2] != b[1][2]: b = b[:1]
    batch, _ = wd.batch(docs, b, device, head, maxw)
    net.train()
    with torch.enable_grad():
        loss = net(**batch).mean(); loss.backward()
    net.zero_grad(set_to_none=True)
    out = net(batch["ids"], batch["wpos"], batch["wmask"])
    return float(loss), tuple(out.shape)
