# Architecture 2: Encoder-decoder Transformer for "prompt -> continuation" language modelling.
#
# Every training window of enc_len + dec_len contiguous tokens is split in two:
#   encoder  <- first enc_len tokens (the "prompt"), bidirectional self-attention
#   decoder  <- next dec_len tokens (the "continuation"), causal self-attention + cross-attention
# The loss is computed on the continuation tokens only, which is exactly the Phase-1 test task.
# Positions: RoPE (your A3 rope.py) inside self-attention; no positional signal in cross-attention.
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import build_rope_cache, apply_rope


@dataclass
class EncDecConfig:
    vocab_size: int
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 3072
    enc_layers: int = 6
    dec_layers: int = 4
    enc_len: int = 768           # prompt tokens the encoder reads
    dec_len: int = 384           # continuation tokens the decoder predicts per window
    dropout: float = 0.1
    bos_id: int = 2              # decoder start token (<s> in your A1 tokenizer)

    @property
    def max_seq_len(self):       # length of one training window (used by the data pipeline)
        return self.enc_len + self.dec_len


class Attention(nn.Module):
    """Multi-head attention. Self-attention when context is None (optionally with RoPE),
    cross-attention when context is given (queries from x, keys/values from context)."""

    def __init__(self, d_model, n_heads, dropout, rope_len=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head, self.dropout = n_heads, d_model // n_heads, dropout
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.use_rope = rope_len is not None
        if self.use_rope:
            cos, sin = build_rope_cache(rope_len, self.d_head)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, context=None, causal=False):
        B, T, C = x.shape
        src = x if context is None else context
        S = src.size(1)
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)          # (B,H,T,Dh)
        k, v = self.kv_proj(src).view(B, S, 2, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        if self.use_rope:
            # cast the cache to q's dtype so q, k, v share one dtype under fp16 autocast
            q = apply_rope(q, self.rope_cos[:T].to(q.dtype), self.rope_sin[:T].to(q.dtype))
            k = apply_rope(k, self.rope_cos[:S].to(k.dtype), self.rope_sin[:S].to(k.dtype))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, C))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg.d_model, cfg.n_heads, cfg.dropout, rope_len=cfg.enc_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):                                   # pre-LN, bidirectional
        x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.ffn(self.ln2(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.self_attn = Attention(cfg.d_model, cfg.n_heads, cfg.dropout, rope_len=cfg.dec_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.cross_attn = Attention(cfg.d_model, cfg.n_heads, cfg.dropout)          # no RoPE
        self.ln3 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, y, memory):                           # pre-LN
        y = y + self.drop(self.self_attn(self.ln1(y), causal=True))
        y = y + self.drop(self.cross_attn(self.ln2(y), context=memory))
        return y + self.ffn(self.ln3(y))


class EncDecLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)       # shared by encoder, decoder, head
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.encoder = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.enc_layers)])
        self.enc_norm = nn.LayerNorm(cfg.d_model)
        self.decoder = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.dec_layers)])
        self.dec_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight                          # weight tying

    def encode(self, src):
        """src (B, S<=enc_len) -> encoder hidden states (B, S, d). Also the Phase-2 feature extractor."""
        x = self.emb_drop(self.emb(src))
        for layer in self.encoder:
            x = layer(x)
        return self.enc_norm(x)

    def decode(self, memory, dec_in):
        """memory (B, S, d), dec_in (B, T<=dec_len) -> next-token logits (B, T, V)."""
        y = self.emb_drop(self.emb(dec_in))
        for layer in self.decoder:
            y = layer(y, memory)
        return self.head(self.dec_norm(y))

    def split(self, x):
        """Contiguous window x (B, enc_len+dec_len) -> encoder input, decoder input, decoder targets."""
        E = self.cfg.enc_len
        src, tgt = x[:, :E], x[:, E:]
        bos = torch.full((x.size(0), 1), self.cfg.bos_id, dtype=x.dtype, device=x.device)
        dec_in = torch.cat([bos, tgt[:, :-1]], dim=1)               # teacher forcing, shifted right
        return src, dec_in, tgt

    def forward(self, x):
        src, dec_in, _ = self.split(x)
        return self.decode(self.encode(src), dec_in)

    def compute_loss(self, x, y=None):
        """Cross-entropy on the continuation (decoder) tokens only. y is unused: targets come from x."""
        src, dec_in, tgt = self.split(x)
        logits = self.decode(self.encode(src), dec_in)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1))
