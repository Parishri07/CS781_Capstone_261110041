# Phase 3: Rhetorical role prediction

**Task.** Label each sentence of a judgment with one of 13 roles, in Corporate Law and Income Tax.

**Metric.** Macro-F1

**Best leaderboard score.** 0.4932 (arch 3 full fine-tuning)

## Data

Official train and dev splits for two domains (CL and IT). Dev labels were used only for selection, and never for training.

## Model architecture

Sentences are packed into 1,024-token windows and read by the Phase-1 model. Each sentence vector is the mean of its tokens. Position and domain are added, then a document-level BiLSTM reads the sentence sequence. Later versions add a CRF, a label-shift head and a lexical channel. The code is in `rr.py`.

## Methodology and implementation

Square-root inverse class weights, best epoch on dev macro-F1, and seed ensembles. Control experiment V6 compares pretrained and random starts: 0.560 and 0.582 vs 0.298 and 0.314.

## Submissions

| Notebook | What it does | Leaderboard |
|---|---|---|
| `p3-N1` | arch 4 LoRMA, 3 seeds | 0.4808 |
| `p3-N2` | arch 3 rsLoRA | 0.4805 |
| `p3-N3v2` | arch 2 LoRA | 0.3684 |
| `p3-V3A` | arch 4 LoRMA + CRF | 0.4627 |
| `p3-V3B` | arch 3 rsLoRA + CRF | 0.4893 |
| `p3-V3C` | arch 2 LoRA + CRF | 0.4166 |
| `p3-V3D` | Cross ensemble / KD student | 0.4709 / 0.4342 |
| `p3-V4A`, `V4B`, `V4T` | Lexical channel and tuned ensembles | 0.4574 to 0.4721 |
| `p3-V5` | Dissent rule and templates | 0.4442, 0.4601 |
| **`p3-V6`** | **arch 3 full fine-tuning** | **0.4932** |
| `p3-V9` | Rare-label relabelling | 0.4742 to 0.4865 |
| `p3-V10` | Full FT ensemble | 0.4798, 0.4849 |

Notebooks are in `notebooks/`, in folders that follow the order of the experiments. Exact duplicates were removed and outputs were cleared. Python sources are in `src/`.

## Notes and limits

Notebook V8 showed that the scorer averages over all 13 labels, and that two rare labels never predicted score 0. The test has about 10 documents, so differences near 0.02 are noise.

## How to run

1. Open the notebook on Kaggle with **GPU T4 x2**.
2. Add a Kaggle secret named `capstone` (or `HF_TOKEN`) holding a Hugging Face token with access to `Exploration-Lab/CS-781-Capstone`.
3. Attach a dataset holding the Phase-1 checkpoints, tokenizers and `shared/phase1_models` code (see the root README). Weights are not stored in this repository.
4. Use **Save Version -> Save & Run All**. Each notebook writes its own code files, runs a short smoke check, then the real run, and saves a submission zip in `/kaggle/working`.
