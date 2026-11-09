# Architecture 3: Prefix-LM (hybrid encoder/decoder in ONE stack) with RMSNorm, SwiGLU, RoPE, KV cache.
#
# One Transformer stack, two attention patterns decided by a prefix length p:
#   * prefix tokens   [0, p)  attend to EVERY prefix token (bidirectional, like an encoder)
#   * continuation    [p, T)  attend causally to everything before them (like a decoder)
#   mask: token i may attend to token j  <=>  (j <= i) or (j < p)
# Loss: only on predictions that cannot see their own target, i.e. positions i >= p-1.
# p is sampled per sequence during training (p = 0 -> plain causal LM), so the model learns both
# "read a prompt bidirectionally, then continue" and ordinary language modelling.
# Generation: the prompt is the prefix (one bidirectional "prefill" pass that fills a KV cache),
# then each new token is a single-position step that reuses the cached keys/values.
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import build_rope_cache, apply_rope


@dataclass
class PrefixLMConfig:
    vocab_size: int
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12          # < n_heads -> grouped-query attention (smaller KV cache)
    ffn_hidden: int = 2048        # SwiGLU hidden size (~8/3 * d_model keeps params like a 4x GELU FFN)
    max_seq_len: int = 1024
    dropout: float = 0.0
    prefix_min: float = 0.0       # training prefix length p ~ U[prefix_min*T, prefix_max*T]
    prefix_max: float = 0.85
    eval_prefix: float = 0.75     # fixed p for validation loss (deterministic, task-like)


class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2) + eps) * g  - no mean subtraction, no bias (cheaper than LayerNorm)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    """FFN(x) = W2( SiLU(W1 x) * (W3 x) )  - gated feed-forward (PaLM / LLaMA)."""

    def __init__(self, d_model, hidden, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0 and cfg.n_heads % cfg.n_kv_heads == 0
        self.h, self.kvh, self.dh = cfg.n_heads, cfg.n_kv_heads, cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, self.h * self.dh, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.kvh * self.dh, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.kvh * self.dh, bias=False)
        self.o_proj = nn.Linear(self.h * self.dh, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, attn_mask=None, cache=None):
        """x (B,T,C); cos/sin for these T positions; attn_mask bool (B,1,T,S) True=attend or None=all;
        cache = (k, v) of earlier positions or None. Returns output and the updated (k, v)."""
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kvh, self.dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kvh, self.dh).transpose(1, 2)
        q = apply_rope(q, cos.to(q.dtype), sin.to(q.dtype))
        k = apply_rope(k, cos.to(k.dtype), sin.to(k.dtype))
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)
        if self.kvh != self.h:                                   # GQA: share each K/V head
            k = k.repeat_interleave(self.h // self.kvh, dim=1)
            v = v.repeat_interleave(self.h // self.kvh, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1)), new_cache


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, attn_mask=None, cache=None):     # pre-norm residual block
        a, new_cache = self.attn(self.norm1(x), cos, sin, attn_mask, cache)
        x = x + self.drop(a)
        return x + self.ffn(self.norm2(x)), new_cache


class PrefixLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight                      # weight tying
        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.d_model // cfg.n_heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @staticmethod
    def prefix_mask(prefix_len, T):
        """prefix_len (B,) -> bool mask (B,1,T,T): i attends j  <=>  j <= i  or  j < p."""
        i = torch.arange(T, device=prefix_len.device)
        causal = i[None, :] <= i[:, None]                              # (T,T)  [i, j]: j <= i
        in_prefix = i[None, None, :] < prefix_len[:, None, None]       # (B,1,T): j < p
        return (causal[None] | in_prefix)[:, None]                     # (B,1,T,T)

    def _run(self, idx, attn_mask, start_pos=0, caches=None):
        T = idx.size(1)
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]
        x = self.drop(self.tok_emb(idx))
        new_caches = []
        for li, block in enumerate(self.blocks):
            x, c = block(x, cos, sin, attn_mask, None if caches is None else caches[li])
            new_caches.append(c)
        return self.norm(x), new_caches

    def hidden(self, idx, prefix_len=None):
        """Final hidden states (B,T,d). Default = fully bidirectional (whole input is the prefix):
        the natural feature extractor for Phase 2+ (NER, classification, ...)."""
        B, T = idx.shape
        p = torch.full((B,), T if prefix_len is None else prefix_len, device=idx.device)
        return self._run(idx, self.prefix_mask(p, T))[0]

    def forward(self, idx, prefix_len=None):
        return self.head(self.hidden(idx, 0 if prefix_len is None else prefix_len))

    def compute_loss(self, x, y):
        """Next-token loss on positions i >= p-1 (positions that cannot see their target)."""
        B, T = x.shape
        if self.training:
            lo, hi = int(self.cfg.prefix_min * T), int(self.cfg.prefix_max * T)
            p = torch.randint(lo, hi + 1, (B,), device=x.device)
        else:
            p = torch.full((B,), int(self.cfg.eval_prefix * T), device=x.device)
        h, _ = self._run(x, self.prefix_mask(p, T))
        logits = self.head(h)
        pos = torch.arange(T, device=x.device)[None, :]
        target = y.masked_fill(pos < (p[:, None] - 1).clamp(min=0), -100)   # ignore leaky positions
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target.reshape(-1),
                               ignore_index=-100)

    # ---------------- generation with a KV cache ----------------
    @torch.no_grad()
    def prefill(self, prompt):
        """Prompt (B,P) is the prefix: one bidirectional pass. Returns next-token logits (B,V) + cache."""
        h, caches = self._run(prompt, None)                       # None mask = every prompt token sees all
        return self.head(h[:, -1]), caches

    @torch.no_grad()
    def decode_step(self, token, caches, pos):
        """token (B,1) at absolute position pos; attends to all cached positions + itself."""
        h, caches = self._run(token, None, start_pos=pos, caches=caches)
        return self.head(h[:, -1]), caches
