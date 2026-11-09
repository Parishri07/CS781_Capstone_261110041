# Phase 6: Abstractive summarisation

**Task.** Write a summary of a legal judgment in new words.

**Metric.** BERTScore F1 (ROUGE secondary)

**Best leaderboard score.** 0.8510 (arch 3 full FT + beam search)

## Data

7,030 training records (2 dropped, 300 held out) and 100 test documents. Median document 3,127 words, median summary 638 words.

## Model architecture

Part-wise generation: the document and summary are split into n ordered parts (at most 8), and part i of the document generates part i of the summary. n comes from a length model fitted on training data. Parts are joined and duplicate sentences removed. The code is in `sm.py`.

## Methodology and implementation

Methods: LoRMA, LoRA, Houlsby adapters, KD and full fine-tuning. Decoding: repetition penalty and no-repeat 3-grams on content tokens, a length multiplier, and beam search with 4 beams. Beam search with 1 beam was tested to match greedy output on all four models.

## Submissions

| Notebook | What it does | Leaderboard |
|---|---|---|
| `P6-S1` | arch 4 LoRMA | 0.8436 |
| `P6-S2`, `P6-R2` | arch 2 LoRA, before and after the decoder fix | 0.7850, 0.8212 |
| `P6-S3`, `P6-R3` | arch 1 Houlsby, before and after the fix | 0.7946, 0.8386 |
| `P6-B3` | arch 1 Houlsby + beam | 0.8439 |
| `P6-S4`, `P6-R4` | KD student | 0.8402, 0.8407 |
| `P6-S4b` | KD student, 5 epochs | 0.8414 |
| `P6-S5` | arch 4 full FT | 0.8414 |
| `P6-E` | Ensemble S1 + S5 | 0.8432 |
| `P6-S6` | arch 3 full FT, default / longer | 0.8424 / 0.8445 |
| **`P6-B6`** (uses S6) | **arch 3 full FT + beam search** | **0.8490 to 0.8510** |

Notebooks are in `notebooks/`, in folders that follow the order of the experiments. Exact duplicates were removed and outputs were cleared. Python sources are in `src/`.

## How to run

1. Open the notebook on Kaggle with **GPU T4 x2**.
2. Add a Kaggle secret named `capstone` (or `HF_TOKEN`) holding a Hugging Face token with access to `Exploration-Lab/CS-781-Capstone`.
3. Attach a dataset holding the Phase-1 checkpoints, tokenizers and `shared/phase1_models` code (see the root README). Weights are not stored in this repository.
4. Use **Save Version -> Save & Run All**. Each notebook writes its own code files, runs a short smoke check, then the real run, and saves a submission zip in `/kaggle/working`.
