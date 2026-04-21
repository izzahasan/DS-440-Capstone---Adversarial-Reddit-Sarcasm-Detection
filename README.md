# 🎭 Sarcasm Detector — DeBERTa-v3-large

Fine-tuned `microsoft/deberta-v3-large` for binary sarcasm detection on Reddit comments.  
Achieves **~83% accuracy** on the balanced Reddit Sarcasm dataset.

---

## 📋 Project Overview

This project fine-tunes a large pre-trained transformer model to detect sarcasm in Reddit comments. The model takes a comment (and optionally its parent comment for context) and classifies it as **Sarcastic 😏** or **Not Sarcastic 🙂**, along with a confidence score.

### Key improvements over a baseline DeBERTa-v3-base approach

| Fix | Change | Expected gain |
|-----|--------|---------------|
| 1 | `MAX_LEN 128 → 256` | +1–2% accuracy |
| 2 | Layer-wise learning rate decay | +1–2% accuracy |
| 3 | `deberta-v3-base → deberta-v3-large` | +2–4% accuracy |

---

## 🗂️ Repository Structure

```
sarcasm-detector/
│
├── Apr2_DS_440_DeBERTa_Large.ipynb   # Training notebook (Vast.ai / GPU server)
├── sarcasm_detector_colab.ipynb      # Inference demo notebook (Google Colab)
│
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
└── .gitignore                        # Files to exclude from git
```

---

## 🚀 Quick Start
'''python
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import torch

MODEL_ID = "nurizzahasan/deberta-v3-large-sarcasm"

tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_ID)
model     = DebertaV2ForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

def predict_sarcasm(comment: str, parent_comment: str = "") -> dict:
    text = comment.strip()
    if parent_comment.strip():
        text = text + " [SEP] " + parent_comment.strip()

    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=256, padding="max_length")

    with torch.no_grad():
        logits = model(**inputs).logits

    probs      = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred_label = int(torch.argmax(logits, dim=-1).item())
    return {"label": pred_label, "prob_not_sarcastic": probs[0], "prob_sarcastic": probs[1]}

result = predict_sarcasm(
    "Oh yeah, because that always works out SO well.",
    "Just ignore them and they'll stop bullying you."
)
print(result)
# {'label': 1, 'prob_not_sarcastic': 0.04, 'prob_sarcastic': 0.96}
'''

### Training (GPU Server / Vast.ai)

> ⚠️ Requires a GPU with **≥20GB VRAM** (e.g., RTX 3090, A40, A6000, A100).  
> For smaller GPUs (RTX 3080 / 10GB), switch back to `deberta-v3-base`.

1. Clone the repo and upload `train-balanced-sarcasm.csv` to `/workspace/data/`.
2. Open `Apr2_DS_440_DeBERTa_Large.ipynb` in Jupyter.
3. Run all cells top to bottom. Checkpoints are saved to `/workspace/checkpoints/`.

### Inference Demo (Google Colab)

1. Open `sarcasm_detector_colab.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Set the runtime to **T4 GPU** (Runtime → Change runtime type).
3. Upload your saved checkpoint folder to Google Drive.
4. Follow the in-notebook instructions (Cells 1–5).

---

## 🧠 Model Details

| Component | Value |
|-----------|-------|
| Base model | `microsoft/deberta-v3-large` |
| Parameters | ~400–435M |
| Task | Binary sequence classification |
| Max token length | 256 |
| Dataset | `train-balanced-sarcasm.csv` (~1M Reddit comments) |
| Input format | `comment [SEP] parent_comment` |
| Output | Label (0 = Not Sarcastic, 1 = Sarcastic) + probabilities |

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW with layer-wise LR decay |
| Learning rate | 2e-5 (top layers) |
| LR decay factor | 0.9 per layer downward |
| Batch size (per device) | 16 |
| Gradient accumulation | 2 steps (effective batch = 32) |
| Epochs | 5 |
| Warmup ratio | 0.15 |
| Weight decay | 0.01 |
| Precision | fp16 |

### Layer-wise Learning Rate Decay

Lower transformer layers capture general language patterns (updated slowly); upper layers and the classifier head learn task-specific features (updated faster). A decay factor of **0.9 per layer** is applied from top → bottom across all 24 hidden layers of DeBERTa-v3-large.

---

## 📊 Dataset

**[Reddit Balanced Sarcasm Dataset](https://www.kaggle.com/datasets/danofer/sarcasm)** — download `train-balanced-sarcasm.csv` from Kaggle and place it at `/workspace/data/train-balanced-sarcasm.csv` before running the training notebook.

- ~1M Reddit comments labeled for sarcasm
- Labels are balanced (equal sarcastic / non-sarcastic samples)
- Each row includes a `comment`, optional `parent_comment`, and a binary `label`
- The comment and parent comment are concatenated: `comment [SEP] parent_comment`
- Split: 80% train / 10% validation / 10% test

---

## 💻 Hardware Requirements

| GPU | VRAM | Supported? | Batch size |
|-----|------|------------|------------|
| A100 | 40 GB | ✅ **Used for this project** | 32 |
| A100 | 80 GB | ✅ | 32 |
| A40 / A6000 | 48 GB | ✅ | 16 |
| RTX 3090 | 24 GB | ✅ | 8 |
| RTX 3080 | 10 GB | ❌ | Use `deberta-v3-base` instead |

> **Actual training run:** A100 40GB VRAM · Full dataset · ~21 hours to complete

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install transformers==4.44.2 tokenizers==0.19.1 sentencepiece==0.1.99 \
            protobuf==3.20.3 datasets scikit-learn accelerate pandas numpy torch
```

---

## 🔮 Example Predictions

| Comment | Parent Comment | Prediction | Confidence |
|---------|---------------|------------|------------|
| "Oh yeah, because that always works out SO well." | "Just ignore them and they'll stop." | 😏 Sarcastic | ~95% |
| "Thanks for the help, I really appreciate it!" | "Sorry, I can't assist with that." | 🙂 Not Sarcastic | ~88% |
| "Sure, I love working 80-hour weeks for minimum wage." | "You should be grateful you have a job." | 😏 Sarcastic | ~97% |
| "This is genuinely one of the best meals I ever had." | — | 🙂 Not Sarcastic | ~91% |

---

## 📚 References

- [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)
- [DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training](https://arxiv.org/abs/2111.09543)
- [HuggingFace: microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large)
- [Reddit Balanced Sarcasm Dataset](https://www.kaggle.com/datasets/danofer/sarcasm)

---

## 📝 License

This project is for academic / educational use. The base model weights are subject to [Microsoft's DeBERTa license](https://github.com/microsoft/DeBERTa/blob/master/LICENSE).
