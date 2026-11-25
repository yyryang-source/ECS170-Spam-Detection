# ECS 170 — Artificial Intelligence Coursework

**UC Davis · Fall 2026 · Yourui Yang**

This repository contains the reproducible source code and selected report evidence for the course's machine-learning and deep-learning projects. The instructor's datasets, downloaded GloVe archives, Python environments, and trained checkpoints are intentionally **not** versioned.

## Project guide

| Project | Task | Entry point / guide |
| --- | --- | --- |
| Spam detection | Enron email classification with Logistic Regression, Linear SVM, and MLP | [`script/spam_detection/`](script/spam_detection/) |
| Stage 3 | CNN image classification: MNIST, ORL, CIFAR-10 | [Instructor-template guide](README_OFFICIAL_TEMPLATE.md) |
| Stage 4 | IMDb sentiment classification and joke generation with RNN, LSTM, GRU | [Stage 4 guide](README_STAGE4.md) |
| Stage 5 | Sparse GCN node classification on Cora, Citeseer, Pubmed | [Stage 5 guide](README_STAGE5.md) |

### Spam detection

Install the dependencies in `requirements.txt`, place `enron_spam_data.csv` under `data/`, and run:

```bash
python -m pip install -r requirements.txt
python script/spam_detection/run_experiments.py
python script/spam_detection/predict_email.py
```

The original benchmark evaluated Logistic Regression, Linear SVM, and MLP on the Enron-spam dataset. See the scripts for preprocessing, split handling, and hyperparameters.

### Stage 3 — CNN image classification

Use the **instructor-template implementation** in `local_code/` and `script/stage_3_script/`. The three tasks are MNIST, ORL faces, and CIFAR-10. Shared data loaders and CNN models are in `script/stage3/`; **`script/cnn/common.py` is a required shared utility**.

- [Setup, instructor ZIP usage, and training commands](README_OFFICIAL_TEMPLATE.md)
- [Earlier standalone reference runner](README_STAGE3.md)
- [Published Stage 3 experiment evidence](report_figures/stage3/README.md)

### Stage 4 — RNN / LSTM / GRU

The instructor-provided `stage_4_data.zip` contains IMDb positive/negative review splits and the short-joke dataset. The implementation supports random embeddings and streaming GloVe vectors with 50, 100, 200, or 300 dimensions.

```bash
python -m pip install -r requirements-stage4.txt
python -m unittest discover -s tests -p "test_stage4.py" -v

# Put the instructor archive in data/stage_4_data.zip first.
python script/stage4/run_stage4.py --quick --device cpu
python script/stage4/run_stage4.py --task both --device cpu
```

Use `--device cpu` if PyTorch's Apple MPS backend crashes on your machine. See the [Stage 4 guide](README_STAGE4.md) for optional GloVe and individual-task commands.

## Stage 4 — recommended template-aligned workflow

The instructor-style Stage 4 entry point is [script/stage_4_script/script_rnn.py](script/stage_4_script/script_rnn.py). Its Dataset/Method/Setting/Evaluate/Result classes inherit from the instructor's base classes in local_code/base_class/. It calls the same training/evaluation routines as the original runner, and keeps all existing experiment archives intact.

```bash
python script/stage_4_script/script_rnn.py --quick --device cpu
python -m unittest discover -s tests -p "*stage4.py" -v
```

See the [TA review checklist](STAGE4_SUBMISSION_CHECKLIST.md), [full Stage 4 metrics](report_figures/stage4/README.md), and [Stage 4 workflow guide](README_STAGE4.md).

### Stage 5 — GCN graph embedding and node classification

The Stage 5 implementation follows the instructor base-class structure, supports Cora/Citeseer/Pubmed, and implements sparse (non-dense) normalized graph convolutions, model-depth/width/dropout ablations, train/validation/test isolation, and report-ready figures. See [README_STAGE5.md](README_STAGE5.md) and [script/stage_5_script/script_gcn.py](script/stage_5_script/script_gcn.py).

**Stage 5 training metrics are not yet claimed**: published tests use small synthetic graphs, and final results require instructor-provided dataset verification plus real local runs.

## Results and tests

- **Stage 3:** Figures, metric summaries, and verification evidence under [`report_figures/stage3/`](report_figures/stage3/).
- **Stage 4:** [`stage4_report_results.zip`](stage4_report_results.zip) contains the curated CSV, JSON, and PNG outputs for **36 completed runs** (6 embedding configurations × 3 recurrent architectures × 2 tasks). See the [readable results tables](report_figures/stage4/README.md); raw datasets and model weights are excluded.
- **Verification:** `tests/` and `.github/workflows/` contain automated checks for Stage 3 and Stage 4.

## Repository organization

```text
local_code/                  Instructor-template classes (Stage 3 and Stage 4)
script/
  spam_detection/            Spam classification
  cnn/common.py              Shared Stage 3 utilities
  stage3/                    Stage 3 dataset loaders, architecture, standalone runner
  stage_3_script/            Instructor-compatible Stage 3 entry points
  stage4/                    Stage 4 datasets, models, training/evaluation
  stage_4_script/            Instructor-compatible Stage 4 entry point
tests/                       Stage 3 and Stage 4 checks
report_figures/stage3/        Curated Stage 3 evidence
stage4_report_results.zip    Curated Stage 4 evidence
```

Development-only PyCharm boilerplate and supplemental standalone CNN demos were removed to keep the repository focused on coursework; they remain recoverable from Git history. Do not commit datasets, downloaded GloVe files, local checkpoints, secrets, or virtual environments.
