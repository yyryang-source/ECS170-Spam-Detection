
# ECS 170 - Spam Email Detection

**Student:** Yourui Yang  
**Course:** ECS 170 - Artificial Intelligence  
**University:** UC Davis  
**Quarter:** Fall 2026

## Project Description

This project implements a spam email detection
system using machine learning and deep learning.

Three models are evaluated:
- Multi-Layer Perceptron (MLP)
- Logistic Regression
- Linear Support Vector Machine (SVM)

## Dataset

Enron-Spam Email Dataset

Dataset source:
https://github.com/MWiechmann/enron_spam_data

After preprocessing:
- Total emails: 30,462
- Training samples: 21,932
- Validation samples: 2,437
- Testing samples: 6,093

## Final Model Performance

| Model | Accuracy | F1-score |
|---|---|---|
| MLP | 98.77% | 98.71% |
| Logistic Regression | 98.85% | 98.81% |
| Linear SVM | 98.74% | 98.68% |

## Ablation Study

Five MLP configurations were evaluated using
different hidden-layer dimensions and dropout rates.

The best configuration used:
- Hidden layers: 128 and 64 neurons
- Dropout: 0.0
- Validation F1-score: 98.88%

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Experiments

Place the dataset CSV file in:

`data/enron_spam_data.csv`

Then run:

```bash
python script/spam_detection/run_experiments.py
```

## Interactive Spam Detection

After training, run:

```bash
python script/spam_detection/predict_email.py
```

The program allows users to enter an email
message and receive a spam/ham prediction.
