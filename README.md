# Multi-Label Text Classification with Bidirectional LSTM

A deep learning pipeline for multi-label text classification built in Keras/TensorFlow, using a DistilRoBERTa tokenizer with a custom Bidirectional LSTM classifier. Evaluated on a held-out test set via a CodaBench competition benchmark.

---

## Architecture

Rather than fine-tuning a full transformer (which is computationally expensive), this model uses the **DistilRoBERTa tokenizer** for subword tokenization and feeds token IDs into a lightweight but expressive **Bidirectional LSTM** stack:

```
Input IDs (max_length=128)
        ↓
Embedding Layer (vocab_size × 256)
        ↓
BiLSTM (96 units, return_sequences=True)
        ↓
BiLSTM (48 units)
        ↓
Dense (128, ReLU) → Dropout (0.4)
        ↓
Dense (64, ReLU)  → Dropout (0.3)
        ↓
Dense (num_labels, Sigmoid)
```

This hybrid approach keeps inference fast while still benefiting from the subword vocabulary of a modern tokenizer.

---

## Key Design Choices

| Choice | Rationale |
|---|---|
| DistilRoBERTa tokenizer (no transformer weights) | Subword tokenization quality without transformer compute cost |
| Bidirectional LSTM × 2 | Captures forward and backward context; stacked for richer representations |
| Sigmoid output (not Softmax) | Labels are independent — a sample can belong to multiple classes |
| Binary cross-entropy loss | Standard for multi-label problems |
| Micro F1 as primary metric | Accounts for label imbalance across all classes jointly |
| Threshold tuning post-training | Decouples classification threshold from model training; improves F1 directly |

---

## Project Structure

```
.
├── nn.py               # Main pipeline: train, tune, predict
├── bilstm.weights.h5   # Saved model weights (after training)
├── data/
│   ├── train.csv       # Training data (not committed)
│   └── dev.csv         # Development/validation data (not committed)
└── submission.zip      # CodaBench-formatted predictions
```

---

## Setup

```bash
pip install tensorflow transformers datasets scikit-learn pandas numpy
```

Python 3.9+ recommended. GPU strongly recommended for training.

---

## Usage

### 1. Train

```bash
python nn.py train
```

Trains the BiLSTM model on `data/train.csv`, validates on `data/dev.csv`, and saves weights to `bilstm.weights.h5`.

Training uses:
- **Early stopping** (patience=3) on validation micro F1
- **ReduceLROnPlateau** (patience=1, factor=0.5) to decay learning rate when progress stalls
- Up to 8 epochs (usually converges earlier via early stopping)

### 2. Tune Classification Threshold

```bash
python nn.py tune
```

Sweeps thresholds from 0.20 → 0.60 on the dev set and reports the threshold that maximizes micro F1. Use this value in the predict step.

### 3. Predict

```bash
# On dev set (default)
python nn.py predict --input data/dev.csv --threshold 0.35

# On test set
python nn.py predict --input data/test.csv --threshold 0.35
```

Outputs a `submission.zip` containing `submission.csv` formatted for CodaBench upload.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Tokenizer | `distilroberta-base` |
| Max sequence length | 128 |
| Batch size | 32 |
| Embedding dim | 256 |
| BiLSTM units (layer 1) | 96 |
| BiLSTM units (layer 2) | 48 |
| Learning rate (initial) | 1e-3 |
| Dropout | 0.4 / 0.3 |
| Loss | Binary cross-entropy |
| Optimizer | Adam |

---

## Data Format

Input CSVs must follow this structure:

```
text,label_1,label_2,...,label_n
"Some input text.",1,0,...,1
```

- First column: raw text
- Remaining columns: binary labels (0 or 1), one per class

---

## Notes

- Model weights are saved separately from architecture (`bilstm.weights.h5`), making it easy to reload and re-predict without retraining.
- The threshold tuning step is intentionally separate from training — this lets you optimize the decision boundary directly on F1 rather than relying on a fixed 0.5 cutoff, which often underperforms on imbalanced label distributions.
- Data files are excluded from version control. Download from the competition's Files tab.
