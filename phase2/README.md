# Phase 2: Legal named entity recognition

**Task.** Return exact character spans and one of 12 labels for entities in long judgments.

**Metric.** Strict macro-F1 (start, end and label must all match)

**Best leaderboard score.** 0.5828 (7 LoRMA models + label bias + rules)

## Data

70 training documents (8,393 entities) and 35 test documents. 10 documents were held out for validation, then some models were refit on all 70.

## Model architecture

Phase-1 backbone, dropout and a linear head over 25 BIO tags. Each window is the first 128 tokens of the document (case title and parties) plus about 900 tokens of text. Word-level labels use each word's first token, so a span cannot begin inside a word. Punctuation is split from words before tokenizing, which raised the best possible score from 0.7305 to 0.9952.

## Methodology and implementation

Sliding windows with averaged overlaps, BIO-constrained Viterbi decoding, label propagation inside a document, and small rules learned from training annotations (`ner_post.py`). The final system averages 7 LoRMA models and applies per-label biases chosen by cross-validation in notebook Z.

## Submissions

| Notebook | What it does | Leaderboard |
|---|---|---|
| `p2-NB1` (E1) | arch 4 full FT, bidirectional | 0.3584 |
| `p2-NB2` (E3) | arch 3 LoRA | 0.4789 |
| `p2-NB2` (E4) | arch 2 Houlsby | 0.2398 |
| `p2-NB2` (E5) | arch 1 BitFit | 0.1310 |
| `p2-NB1` (E6) | KD 6-layer student | 0.3904 |
| `p2-R1` | LoRMA causal / KD with arch 3 teacher | 0.4796 / 0.4254 |
| `p2-R2` | Tuned BitFit / Houlsby | 0.3599 / 0.3541 |
| `p2-R3` | Refit on 70: LoRA / full FT / LoRMA | 0.5020 / 0.4381 / 0.5125 |
| `p2-R4`, `p2-R5` | LoRMA batch 4, refit | 0.5127, 0.5412 |
| `p2-X1` | 4-model ensemble | 0.5446 |
| `p2-X2` | LoRMA + propagation + rules | 0.5642 |
| `p2-Y3` | Seed ensemble | 0.5621 |
| `p2-Z` | 3 seeds, with and without bias | 0.5585 / 0.5617 |
| `p2-W4` | 13 LoRMA seeds | 0.5795 |
| **`p2-W2`** (uses Z and W1) | **7 LoRMA models + bias + rules** | **0.5828** |

Notebooks are in `notebooks/`, in folders that follow the order of the experiments. Exact duplicates were removed and outputs were cleared. Python sources are in `src/`.

## Notes and limits

The earlier results of `phase2-exp1` and the ablations in `p2-NB3` were used for analysis. Validation has only 10 documents and the same setup varied by up to 0.04, and leaderboard scores influenced the final choice.

## How to run

1. Open the notebook on Kaggle with **GPU T4 x2**.
2. Add a Kaggle secret named `capstone` (or `HF_TOKEN`) holding a Hugging Face token with access to `Exploration-Lab/CS-781-Capstone`.
3. Attach a dataset holding the Phase-1 checkpoints, tokenizers and `shared/phase1_models` code (see the root README). Weights are not stored in this repository.
4. Use **Save Version -> Save & Run All**. Each notebook writes its own code files, runs a short smoke check, then the real run, and saves a submission zip in `/kaggle/working`.
