# CS781 Capstone: In-House Legal Intelligence System

Author: Parishri Shah. 

LLM Capstone Course project.

A law firm often cannot send confidential documents to a cloud language model. This project trains legal language models from scratch (Phase 1) and adapts them to six legal tasks (Phases 2 to 7). No other pretrained model, external legal data or LLM API is used in any model.

## Results

| Phase | Task | Metric | Best score | System |
|---|---|---|---|---|
| [1](phase1/README.md) | Legal language model pretraining | BERTScore F1 | 0.8392 | arch 4 + MBR decoding |
| [2](phase2/README.md) | Legal named entity recognition | Strict macro-F1 | 0.5828 | 7 LoRMA models + label bias + rules |
| [3](phase3/README.md) | Rhetorical role prediction | Macro-F1 | 0.4932 | arch 3 full fine-tuning |
| [4](phase4/README.md) | Court judgment prediction | Macro-F1 | 0.7862 | arch 4 LoRMA with causal mask |
| [5](phase5/README.md) | Legal statute identification | Macro-F1 | 0.3871 | optimised weighted ensemble of Phase 5 runs (P5-X) |
| [6](phase6/README.md) | Abstractive summarisation | BERTScore F1 | 0.8510 | arch 3 full FT + beam search |
| [7](phase7/README.md) | Prior case retrieval | Micro-F1@7 | 0.4335 | KD student + scoring rule + phrase vectors |

Each phase folder has its own README with the architecture, method, every submission and its score. The Phase 4 score used a threshold tuned on a dev file that was later found to contain test labels; see [Phase 4](phase4/README.md).

## Repository layout

```
shared/phase1_models/   Phase-1 model code (arch 1 to 4) and tokenizer code
shared/tokenizers/      A1 BPE and space-attached BPE
shared/backbone_manifest.json   Which checkpoint is which
phase1/ ... phase7/     README.md, notebooks/ (all submissions), src/ (task code)
```

## Foundation models

| Name | Design | Parameters | Tokenizer |
|---|---|---|---|
| arch 1 | Decoder with multi-head attention | 77.0M | A1 BPE |
| arch 2 | Encoder-decoder | 86.4M | A1 BPE |
| arch 3 | Prefix-LM | 91.1M | Space BPE |
| arch 4 | Decoder with Multi-head Latent Attention | 87.9M | Space BPE |

## Adaptation techniques used

Full and partial fine-tuning, LoRA, rsLoRA, LoRMA, BitFit, Houlsby adapters, IA3 and knowledge distillation. Implementations are in `phase2/src/ner_model.py`, `phase3/src/rr.py` (rsLoRA, IA3) and `phase6/src/sm.py`.

## Setup

All work ran on Kaggle with 2 T4 GPUs and 12-hour sessions. To reproduce:

1. Get data from `Exploration-Lab/CS-781-Capstone` on Hugging Face. It needs an access token, stored as the Kaggle secret `capstone`.
2. Train the foundation models with the Phase 1 notebooks, or use your own checkpoints.
3. Make a Kaggle dataset with this layout, named `phase2-backbones`: `arch1_decoder/`, `arch2_encdec/`, `arch3_prefixlm/`, `arch4_mla/` (each with `ckpt_best.pt`), plus `code/`, `tokenizers/` and `manifest.json` from `shared/`.
4. Open a notebook, attach that dataset, and use **Save Version -> Save & Run All**.

Checkpoints are not in this repository. Local Python needs `requirements.txt`, but the training notebooks expect GPUs.

## References

See the report bibliography. Key sources: IL-TUR (arXiv:2407.05399), LoRA, LoRMA, knowledge distillation (Hinton et al.), Houlsby adapters, BitFit, DeepSeek-V2 for latent attention, and U-CREAT for prior case retrieval.
