# Phase 4: Court judgment prediction

**Task.** Read a Supreme Court case and predict the outcome, 0 or 1.

**Metric.** Macro-F1

**Best leaderboard score.** 0.7862 (arch 4 LoRMA with causal mask)

## Data

37,387 training documents (22,067 of class 0, 15,320 of class 1) and 1,517 test cases. The original dev file (3,505 cases) later proved to be the corrected dev set (1,988) plus the test cases with labels. The organisers fixed this.

## Model architecture

Chunks of up to 1,024 tokens are taken from the end of the document. Each passes through the backbone and token attention pooling. A chunk position embedding and a BiGRU then combine the chunks, followed by a two-way head. The code is in `cj.py`.

## Methodology and implementation

Class-weighted cross-entropy and a threshold tuned on dev. Mean pooling stayed near chance, so attention pooling replaced it. LoRMA collapsed to dev 0.333 with a bidirectional mask and reached dev 0.7796 with the causal mask kept.

## Submissions

| Notebook | What it does | Leaderboard |
|---|---|---|
| `p4-C3` (adapters) | arch 2 adapters | 0.5665, 0.5777 |
| `P4-V3-C1L`, `1_C1L` | arch 3 full FT, 3 epochs | 0.7607 |
| **`P4-V3-C2`, `2_C2`** | **arch 4 LoRMA, causal** | **0.7862** |
| `P4-V3-C4`, `4_C4` | Ensemble / KD student | 0.7858 / 0.6012 |

Notebooks are in `notebooks/`, in folders that follow the order of the experiments. Exact duplicates were removed and outputs were cleared. Python sources are in `src/`.

## Notes and limits

The scores above used thresholds tuned on the old dev file, so they are optimistic. Notebook `7_CLEAN` rebuilds the submissions with the corrected dev set.

## How to run

1. Open the notebook on Kaggle with **GPU T4 x2**.
2. Add a Kaggle secret named `capstone` holding a Hugging Face token with access to `Exploration-Lab/CS-781-Capstone`.
3. Attach a dataset holding the Phase-1 checkpoints, tokenizers and `shared/phase1_models` code (see the root README). Weights are not stored in this repository.
4. Use **Save Version -> Save & Run All**. Each notebook writes its own code files, runs a short smoke check, then the real run, and saves a submission zip in `/kaggle/working`.
