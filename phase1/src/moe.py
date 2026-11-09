# Architecture 5: sparse Mixture-of-Experts Transformer (decoder-only) with grouped-query attention.
#
# Dense models run every token through the same FFN. Here every FFN is replaced by E expert FFNs and a
# learned router: each token is sent to its top-k experts (k=2 of E=8) and their outputs are mixed with
# the renormalised router probabilities. Total parameters grow ~E/k-fold while the compute per token stays
# about that of one dense FFN ("conditional computation", Shazeer 2017 / Switch 2021 / Mixtral 2024).
# A load-balancing loss  aux = E * sum_e f_e * P_e  (f_e = share of routed tokens, P_e = mean router prob)
# keeps all experts in use. Attention is grouped-query (GQA): 12 query heads share 4 key/value heads,
# which also shrinks the KV cache 3x. Positions: RoPE. Norm: RMSNorm. Generation uses a KV cache.
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import build_rope_cache, apply_rope
from prefixlm import RMSNorm, SwiGLU            # same building blocks as architectures 3/4


@dataclass
class MoEConfig:
    vocab_size: int
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4           # grouped-query attention
    n_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 1024     # SwiGLU hidden size of ONE expert
    max_seq_len: int = 1024
    dropout: float = 0.0
    aux_loss_coef: float = 0.01   # weight of the load-balancing loss (training only)


class GQAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0 and cfg.n_heads % cfg.n_kv_heads == 0
        self.h, self.kvh, self.dh = cfg.n_heads, cfg.n_kv_heads, cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, self.h * self.dh, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.kvh * self.dh, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.kvh * self.dh, bias=False)
        self.o_proj = nn.Linear(self.h * self.dh, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kvh, self.dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kvh, self.dh).transpose(1, 2)
        q = apply_rope(q, cos.to(q.dtype), sin.to(q.dtype))
        k = apply_rope(k, cos.to(k.dtype), sin.to(k.dtype))
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)                                        # cached at kv-head size (3x smaller)
        S = k.size(2)
        k = k.repeat_interleave(self.h // self.kvh, dim=1)
        v = v.repeat_interleave(self.h // self.kvh, dim=1)
        if cache is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                 dropout_p=self.dropout if self.training else 0.0)
        else:                                                     # new tokens sit at the end
            i = torch.arange(T, device=x.device)[:, None] + (S - T)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=torch.arange(S, device=x.device)[None, :] <= i)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1)), new_cache


class MoEFFN(nn.Module):
    """Top-k routed mixture of SwiGLU experts."""

    def __init__(self, cfg):
        super().__init__()
        self.E, self.k = cfg.n_experts, cfg.top_k
        self.router = nn.Linear(cfg.d_model, self.E, bias=False)
        self.experts = nn.ModuleList([SwiGLU(cfg.d_model, cfg.expert_hidden, cfg.dropout) for _ in range(self.E)])
        self.last_aux = None
        self.last_load = None

    def forward(self, x):
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        probs = F.softmax(self.router(flat).float(), dim=-1)                 # (N, E)
        top_p, top_i = probs.topk(self.k, dim=-1)                            # (N, k)
        gates = top_p / top_p.sum(dim=-1, keepdim=True)                      # renormalised over the k chosen
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            rows, slot = (top_i == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            y = expert(flat[rows]) * gates[rows, slot].unsqueeze(-1).to(flat.dtype)
            out.index_add_(0, rows, y.to(out.dtype))
        # load balancing (Switch Transformer): fraction of routed slots per expert x mean router probability
        frac = F.one_hot(top_i, self.E).float().sum(dim=(0, 1)) / (flat.size(0) * self.k)
        self.last_aux = self.E * (frac * probs.mean(dim=0)).sum()
        self.last_load = frac.detach()
        return out.view(B, T, C)


class MoEBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = GQAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.moe = MoEFFN(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache=None):
        a, new_cache = self.attn(self.norm1(x), cos, sin, cache)
        x = x + self.drop(a)
        return x + self.drop(self.moe(self.norm2(x))), new_cache


class MoELM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([MoEBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight                               # weight tying
        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.d_model // cfg.n_heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def active_params(self):
        """Parameters used for ONE token (attention + router + k of E experts per layer + embeddings)."""
        c = self.cfg
        per_expert = sum(p.numel() for p in self.blocks[0].moe.experts[0].parameters())
        total = sum(p.numel() for p in self.parameters())
        return total - c.n_layers * (c.n_experts - c.top_k) * per_expert

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

    def forward(self, idx):
        return self.head(self.hidden(idx))

    def compute_loss(self, x, y):
        """Next-token loss on every position (+ load-balancing loss while training)."""
        logits = self.head(self._run(x)[0])
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
        if self.training and self.cfg.aux_loss_coef > 0:
            loss = loss + self.cfg.aux_loss_coef * torch.stack([b.moe.last_aux for b in self.blocks]).mean()
        return loss

    def expert_load(self):
        """Share of routed tokens per expert in each layer (from the last forward pass)."""
        return torch.stack([b.moe.last_load for b in self.blocks if b.moe.last_load is not None])

    # ---------------- generation with a KV cache ----------------
    @torch.no_grad()
    def prefill(self, prompt):
        h, caches = self._run(prompt)
        return self.head(h[:, -1]), caches

    @torch.no_grad()
    def decode_step(self, token, caches, pos):
        h, caches = self._run(token, start_pos=pos, caches=caches)
        return self.head(h[:, -1]), caches
