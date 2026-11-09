# Architecture 4: decoder-only Transformer with Multi-head Latent Attention (MLA, DeepSeek-V2 style),
# RMSNorm, SwiGLU, decoupled RoPE and a compressed KV cache.
#
# Standard MHA caches K and V for every head: 2 * n_heads * d_head numbers per token per layer.
# MLA instead caches ONE small latent per token and re-creates per-head keys/values from it:
#     c_kv   = RMSNorm(W_dkv x)                       (kv_rank numbers, shared by all heads)
#     k_nope = W_uk c_kv ,  v = W_uv c_kv             (per head, "no position" part)
#     k_rope = RoPE(W_kr x)                           (rope_dim numbers, shared by all heads)
#     q      = [q_nope ; RoPE(q_rope)] = W_q x        (per head)
#     score  = (q_nope . k_nope + q_rope . k_rope) / sqrt(nope_dim + rope_dim)
# RoPE is "decoupled" (a separate small key) because a position rotation could not be moved
# through the up-projection W_uk. Cache per token per layer = kv_rank + rope_dim numbers.
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import build_rope_cache, apply_rope
from prefixlm import RMSNorm, SwiGLU            # same building blocks as architecture 3


@dataclass
class MLAConfig:
    vocab_size: int
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    kv_rank: int = 256            # size of the compressed key/value latent c_kv
    nope_dim: int = 64            # per-head query/key dimension without position
    rope_dim: int = 32            # per-head query / shared key dimension carrying RoPE
    v_dim: int = 64               # per-head value dimension
    ffn_hidden: int = 2048        # SwiGLU hidden size
    max_seq_len: int = 1024
    dropout: float = 0.0


class MLAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h, self.nope, self.rope, self.vd = cfg.n_heads, cfg.nope_dim, cfg.rope_dim, cfg.v_dim
        self.q_proj = nn.Linear(cfg.d_model, self.h * (self.nope + self.rope), bias=False)
        self.kv_down = nn.Linear(cfg.d_model, cfg.kv_rank, bias=False)          # W_dkv
        self.kv_norm = RMSNorm(cfg.kv_rank)
        self.k_up = nn.Linear(cfg.kv_rank, self.h * self.nope, bias=False)      # W_uk
        self.v_up = nn.Linear(cfg.kv_rank, self.h * self.vd, bias=False)        # W_uv
        self.k_rope = nn.Linear(cfg.d_model, self.rope, bias=False)             # W_kr (shared by heads)
        self.o_proj = nn.Linear(self.h * self.vd, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache=None):
        """x (B,T,C); cos/sin for these T positions; cache = (c_kv, k_rope) of earlier tokens or None.
        Without a cache the attention is causal over x; with a cache the T new tokens see all cached
        tokens and (causally) each other. Returns output and the updated (compressed) cache."""
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.nope + self.rope).transpose(1, 2)
        q_nope, q_rope = q.split([self.nope, self.rope], dim=-1)
        q_rope = apply_rope(q_rope, cos.to(q.dtype), sin.to(q.dtype))
        c_kv = self.kv_norm(self.kv_down(x))                                    # (B,T,kv_rank)
        k_r = apply_rope(self.k_rope(x)[:, None], cos.to(q.dtype), sin.to(q.dtype))   # (B,1,T,rope)
        if cache is not None:
            c_kv = torch.cat([cache[0], c_kv], dim=1)
            k_r = torch.cat([cache[1], k_r], dim=2)
        new_cache = (c_kv, k_r)                                                 # <- all that is cached
        S = c_kv.size(1)
        k_nope = self.k_up(c_kv).view(B, S, self.h, self.nope).transpose(1, 2)
        v = self.v_up(c_kv).view(B, S, self.h, self.vd).transpose(1, 2)
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_r.expand(B, self.h, S, self.rope)], dim=-1)
        if cache is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                 dropout_p=self.dropout if self.training else 0.0)
        else:                                                                   # new tokens at the end
            i = torch.arange(T, device=x.device)[:, None] + (S - T)
            mask = torch.arange(S, device=x.device)[None, :] <= i              # (T,S)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1)), new_cache


class MLABlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = MLAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache=None):
        a, new_cache = self.attn(self.norm1(x), cos, sin, cache)
        x = x + self.drop(a)
        return x + self.ffn(self.norm2(x)), new_cache


class MLALM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([MLABlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight                                  # weight tying
        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.rope_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _run(self, idx, start_pos=0, caches=None):
        T = idx.size(1)
        cos, sin = self.rope_cos[start_pos:start_pos + T], self.rope_sin[start_pos:start_pos + T]
        x = self.drop(self.tok_emb(idx))
        new = []
        for li, blk in enumerate(self.blocks):
            x, c = blk(x, cos, sin, None if caches is None else caches[li])
            new.append(c)
        return self.norm(x), new

    def hidden(self, idx):
        """Final (causal) hidden states (B,T,d) - feature extractor for later phases."""
        return self._run(idx)[0]

    def forward(self, idx):                     # plain next-token LM: loss on every position
        return self.head(self.hidden(idx))

    def kv_cache_floats_per_token(self):
        """Cached numbers per token per layer: MLA vs what standard MHA would store."""
        c = self.cfg
        return c.kv_rank + c.rope_dim, 2 * c.n_heads * c.v_dim

    # ---------------- generation with the compressed KV cache ----------------
    @torch.no_grad()
    def prefill(self, prompt):
        h, caches = self._run(prompt)
        return self.head(h[:, -1]), caches

    @torch.no_grad()
    def decode_step(self, token, caches, pos):
        h, caches = self._run(token, start_pos=pos, caches=caches)
        return self.head(h[:, -1]), caches
