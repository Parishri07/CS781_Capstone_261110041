# Phase 5: Legal statute identification

**Task.** Predict which of 100 statutes apply to a document (multi-label).

**Metric.** Macro-F1 (micro-F1 and subset accuracy secondary)

**Best leaderboard score.** 0.3871 (optimised weighted ensemble of Phase 5 runs (P5-X))

## Data

42,750 train, 10,181 dev and 13,019 test documents, with 3.7 to 3.9 statutes per document. The rarest statute has 272 training documents.

## Model architecture

Windows spread across the document, token attention pooling, a BiGRU with attention and max pooling, and one sigmoid per statute. The code is in `ls.py`.

## Methodology and implementation

Binary cross-entropy with capped positive weights, or asymmetric loss (ASL, gamma- = 4, clip 0.05). Thresholds per statute are chosen by 2-fold cross-fitting on dev, and each document gets at least one statute.

## Submissions

| Notebook | What it does | Leaderboard |
|---|---|---|
| `p5-n1-arch4-lorma-causal` | arch 4 LoRMA | 0.3601 |
| `P5-N1L` | arch 4 LoRMA, 5 epochs | 0.3646 |
| `P5-G1` | arch 4 LoRMA, seed 42 | 0.3674 |
| `P5-E` | Ensembles of N1, N1L, N2 | 0.3700, 0.3706 |
| `P5-N2` | arch 3 full FT | 0.3252 |
| `P5-N3` | arch 2 partial | 0.1908 |
| `P5-N4` | KD student, arch 1 5 layers | 0.3079 |
| `P5-G2` | arch 4 LoRMA + ASL | 0.3832 |
| `P5-C` | Final ensembles (cross-fitted) | not recorded |
| **`P5-X`** | **Weighted members + thresholds + statute cap, chosen by cross-fitting** | **0.3871** |

Notebooks are in `notebooks/`, in folders that follow the order of the experiments. Exact duplicates were removed and outputs were cleared. Python sources are in `src/`.

## Notes and limits

P5-X writes two zips, with and without copying training labels to test documents identical to a training text; the 0.3871 submission is the one **with** training duplicates: test documents identical to a training text got that text's training statutes (training labels only), and all other predictions come from the models.

## How to run

1. Open the notebook on Kaggle with **GPU T4 x2**.
2. Add a Kaggle secret named `capstone` (or `HF_TOKEN`) holding a Hugging Face token with access to `Exploration-Lab/CS-781-Capstone`.
3. Attach a dataset holding the Phase-1 checkpoints, tokenizers and `shared/phase1_models` code (see the root README). Weights are not stored in this repository.
4. Use **Save Version -> Save & Run All**. Each notebook writes its own code files, runs a short smoke check, then the real run, and saves a submission zip in `/kaggle/working`.
