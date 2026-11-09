# Phase 1: Legal language model pretraining

**Task.** Read a 512-word legal passage and write the next 128 words.

**Metric.** BERTScore F1 (ROUGE-L and BLEU-4 secondary)

**Best leaderboard score.** 0.8392 (arch 4 + MBR decoding)

## Data

About 19 GB of Acts, High Court, Supreme Court and Tribunal text, streamed with a fixed character budget per source. Validation holds out whole documents from each source, with 100 dev prompts shaped like the test.

## Model architecture

Four architectures, all 50M to 300M parameters and trained from scratch:

| Name | Design | Parameters | Tokenizer |
|---|---|---|---|
| arch 1 | Decoder, multi-head attention, sinusoidal positions, 10 layers | 77.0M | A1 BPE |
| arch 2 | Encoder-decoder, 6 + 4 layers | 86.4M | A1 BPE |
| arch 3 | Prefix-LM: prompt bidirectional, new text causal; RMSNorm, SwiGLU, RoPE; 12 layers | 91.1M | Space BPE |
| arch 4 | Decoder with Multi-head Latent Attention (256-dim latent, 32-dim RoPE key); 12 layers | 87.9M | Space BPE |

The space BPE attaches a space to the next word (4.5 characters per token vs 2.66), so a 512-word prompt fits in 1,024 tokens.

## Methodology and implementation

AdamW, warm-up, cosine decay from 3e-4 to 3e-5, 32,768 tokens per step, fp16, gradient clipping at 1.0, resumable checkpoints and a session clock. The final decoder (MBR) samples 8 continuations at temperature 0.8 and keeps the one most similar to the others, using embeddings from our own model.

## Submissions

| Notebook | What it does | Leaderboard |
|---|---|---|
| arch1: `capstone-phase1-draft*`, `capstone-phase1-arch1-generate` | arch 1 | 0.8245 |
| `capstone-phase1-arch2-encdec` | arch 2 | 0.8181 |
| `capstone-phase1-arch3-prefixlm` | arch 3 | 0.8305 |
| `capstone-phase1-arch3-prefixlm-mbr` | arch 3 + MBR with ROUGE-L | 0.8305 |
| `capstone-phase1-arch3-continue` | arch 3 on a second data slice | 0.8294 |
| `capstone-phase1-arch4-mla` | arch 4 | 0.8314 |
| `capstone-phase1-arch4-mla-mbr` | arch 4 + MBR with ROUGE-L | 0.8314 |
| `capstone-phase1-ensemble-arch4-arch3` | Probability ensemble | 0.8295, 0.8314 |
| `capstone-phase1-arch4-big`, `-big-2` | arch 4 big run, 40k steps | 0.8338 |
| `capstone-phase1-arch4big-decoding` | Temperature and repetition sweep | 0.8206 to 0.8331 |
| `capstone-phase1-final-mla` | Cleaned data | 0.8321, 0.8325 |
| `capstone-phase1-arch4big-decoding2` | Decoding round 2 | 0.8317, 0.8335 |
| **`capstone-phase1-arch4big-mbr`** | **arch 4 + MBR, embedding similarity** | **0.8392** |

Notebooks are in `notebooks/`, in folders that follow the order of the experiments. Exact duplicates were removed and outputs were cleared. Python sources are in `src/`.

## Notes and limits

The arch 5 notebooks (Mixture-of-Experts) were dropped; only a smoke run was submitted by mistake. Architecture results are partly confounded: arch 1 and 2 used a different tokenizer and a smaller token budget.

## How to run

1. Open the notebook on Kaggle with **GPU T4 x2**.
2. Add a Kaggle secret named `capstone` holding a Hugging Face token with access to `Exploration-Lab/CS-781-Capstone`.
3. Attach a dataset holding the Phase-1 checkpoints, tokenizers and `shared/phase1_models` code (see the root README). Weights are not stored in this repository.
4. Use **Save Version -> Save & Run All**. Each notebook writes its own code files, runs a short smoke check, then the real run, and saves a submission zip in `/kaggle/working`.
