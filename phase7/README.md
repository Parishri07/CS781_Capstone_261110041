# Phase 7: Prior case retrieval

**Task.** Return exactly 7 relevant prior cases for each query case.

**Metric.** Micro-F1@7

**Best leaderboard score.** 0.4335 (KD student + scoring rule + phrase vectors)

## Data

Train, dev and test have 827, 118 and 237 queries, with pools of 4,320, 1,023 and 1,727 candidates. Each query sits inside its own pool. Citations in the text are masked.

## Model architecture

Model-only retrieval with a shared encoder. Report headers and citator lists are removed, documents are cut into overlapping windows of 64 or 128 tokens, and each window is the mean of the hidden states. For each query window, a candidate scores its best window. Phrase vectors average hidden states over 8 tokens. The code is in `pcr_model.py`, `pcr_train.py` and `pcr_boost.py`. BM25 (`pcr_lex.py`, notebook P7-0) is a baseline only and is not part of any submitted ranking.

## Methodology and implementation

Each training round mines window pairs and hard negatives with the model itself, then trains with a contrastive loss (temperature 0.03). The KD student also matches the teacher's similarity distribution over each batch. Final scoring counts only matches that stand out from the pool (a z-score rule), with options for document normalisation and rare-match weighting.

## Submissions

| Notebook | What it does | Leaderboard |
|---|---|---|
| `P7-0` | BM25 baseline, dev only | dev 0.4302 |
| `P7-1` | Unmodified models, dev only | dev 0.3091 best |
| `P7-T1` | arch 3 LoRMA | 0.3001 |
| `P7-T2` | arch 3 full FT | 0.3207 |
| `P7-T3` | arch 3 partial, top 4 blocks | 0.3219 |
| `P7-T4` | KD 6-layer student | 0.3260 |
| `P7-T5` | arch 4 full FT | 0.3768 |
| `P7-T6` | arch 1 full FT | 0.3644 |
| `P7-B` | T2 with rule and three views | 0.3969 |
| `P7-C` | LoRMA / full FT / partial | 0.3833 / 0.3952 / 0.4111 |
| **`P7-C`** (uses T4, T2) | **KD student + rule + phrase vectors** | **0.4335** |

Notebooks are in `notebooks/`, in folders that follow the order of the experiments. Exact duplicates were removed and outputs were cleared. Python sources are in `src/`.

## Notes and limits

P7-C was not rerun with T5 and T6 weights, so fusion of different architectures was not tested. There was no KD-off control in this phase.

## How to run

1. Open the notebook on Kaggle with **GPU T4 x2**.
2. Add a Kaggle secret named `capstone` holding a Hugging Face token with access to `Exploration-Lab/CS-781-Capstone`.
3. Attach a dataset holding the Phase-1 checkpoints, tokenizers and `shared/phase1_models` code (see the root README). Weights are not stored in this repository.
4. Use **Save Version -> Save & Run All**. Each notebook writes its own code files, runs a short smoke check, then the real run, and saves a submission zip in `/kaggle/working`.
