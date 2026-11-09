# ECS 170: CNN Extension — Yourui Yang

This addition **preserves** the existing Enron-Spam MLP, Logistic Regression and Linear SVM work.

## 1. CNN for spam email classification

1D CNN: email text → tokens → 64-dimensional embedding → one/two Conv1D + ReLU layers → global max pooling → dropout → binary logit.

Data: Enron-Spam CSV from https://github.com/MWiechmann/enron_spam_data at `data/enron_spam_data.csv`. Expected columns: `Subject`, `Message`, `Spam/Ham`. The same deterministic **72% train / 8% validation / 20% test** scheme as the existing experiment is used. The vocabulary is built from the training split only.

## 2. Image Classification with CNN

2D CNN: image → 2 or 3 Conv2D/ReLU/MaxPool blocks → global average pooling → dropout → 10-class logits.

Default: **Fashion-MNIST** (54k train / 6k validation / 10k test, grayscale, 10 classes). Optional: **CIFAR-10** (45k train / 5k validation / 10k test, color images). Image data downloads automatically to `data/images/` on first execution. Image labels never mix with spam labels.

## Setup

From repository root:

```bash
python -m pip install -r requirements-cnn.txt
python -m unittest discover -s tests -p 'test_cnn_smoke.py'
```

Install a compatible PyTorch/torchvision pair for your hardware from https://pytorch.org/get-started/locally/ if required.

## Run

```bash
# Quick smoke tests — results NOT for final report
python script/cnn/train_spam_cnn.py --quick --device cpu
python script/cnn/train_image_cnn.py --quick --device cpu

# Complete experiments
python script/cnn/train_spam_cnn.py
python script/cnn/train_image_cnn.py

# Optional second image dataset
python script/cnn/train_image_cnn.py --dataset cifar10
```

Use `--epochs`, `--batch-size`, `--patience`, `--device`; spam also supports `--max-len` and `--vocab-size`.

## Training, validation and evaluation

Three CNN configurations per dataset test different convolutional depths and dropout. Batch size=128, Adam learning rate=0.001, at most 12 epochs, early stopping patience=3 (defaults). Model checkpoint uses lowest validation loss for each configuration. **Validation F1** chooses the winner. A single final held-out **test** evaluation then calculates accuracy, precision, recall and F1. Spam uses positive-class binary averaging; images use macro averaging across 10 categories.

Generated artifacts (excluded from Git because `results/` is gitignored):

- `results/spam_cnn/full/`
- `results/image_cnn/fashion-mnist/full/` or `results/image_cnn/cifar10/full/`

Each output has `model_architecture.png`, `best_loss_curve.png`, `ablation_val_loss.png`, `confusion_matrix.png`, `ablation_validation.csv`, `final_test_metrics.csv`, per-configuration loss histories, `experiment_metadata.json`, and a trained model checkpoint.

Report mapping: architecture (Section 2), dataset/splits and hyperparameters (3.1–3.2), code (3.4), loss curves (3.5), held-out test metrics (3.6), validation ablation comparisons (3.7). **Do not report synthetic smoke-test numbers as real results.** The two tasks should be reported distinctly; confirm instructor requirements if a single combined report is expected.
