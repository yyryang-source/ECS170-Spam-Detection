# ECS 170 — Stage 3: Image Classification with CNN
**Student:** Yourui Yang | **Course:** ECS 170, Fall 2026

This page describes the earlier standalone experiments. For the course ZIP, template-based entry points, and current experiment settings, see the [Stage 3 reproduction guide](README_OFFICIAL_TEMPLATE.md).

## Important: use the instructor's actual Stage 3 datasets

The official project description specifically requests:
1. **MNIST** handwritten digit recognition
2. **ORL** face recognition (the supplied grayscale faces may be stored as 3 identical RGB channels)
3. **CIFAR** color-object recognition (this implementation uses CIFAR-10)

The previously completed **Fashion-MNIST** and **Enron-Spam CNN** experiments are supplemental examples, **not replacements for Stage 3**. The course also requests code based on its provided template, with corresponding Dataset_Loader and Method_CNN classes inheriting the provided base classes. This standalone runner is an upgraded **reference implementation** until the exact instructor template and data files are integrated.

## Architecture and controlled comparisons

- **baseline:** two Conv2D → BatchNorm → ReLU stages, channels 32/64, 3×3 kernels, max pooling.
- **residual:** three residual two-convolution blocks, channels 32/64/128, 3×3 kernels, max pooling, residual shortcut projection.
- **kernel5:** same residual configuration as \`residual\`, changing only convolution kernel size to 5×5.

All models use global average pooling, an FC 128-layer classifier, dropout, and dataset-specific logits (10 classes MNIST/CIFAR-10; 40 classes ORL when the full instructor dataset is supplied). AdamW learning rate 0.001, weight decay 1e-4, CrossEntropyLoss with 0.05 label smoothing, gradient clipping, learning-rate schedule, early stopping on validation loss; choose best model via **validation macro F1** before testing. **No pretrained weights.**

## Install, verify

From repository root (the folder containing README_STAGE3.md):

\`\`\`powershell
git pull origin main
python -m pip install -r requirements-cnn.txt
python -m unittest discover -s tests -p "test_stage3.py"
\`\`\`

## Run with the instructor datasets (preferred)

For instructor-provided NPZ files (keys: \`x_train/y_train/x_test/y_test\`, or \`images/labels\`) or image folders organized as one subdirectory per class:

\`\`\`powershell
python script/stage3/run_stage3.py --dataset mnist --source local --data-root "C:\path\to\teacher_mnist.npz"
python script/stage3/run_stage3.py --dataset orl --source local --data-root "C:\path\to\teacher_orl"
python script/stage3/run_stage3.py --dataset cifar10 --source local --data-root "C:\path\to\teacher_cifar.npz"
\`\`\`

Image folders can be class folders under the given directory, or \`train/classname/*.png\` and \`test/classname/*.png\`. An ORL folder can contain \`s1/\` through \`s40/\` identity directories. The program converts duplicated grayscale RGB channels to one channel for ORL.

**If the teacher distributed another file format (pickle, MAT, CSV, etc.), or a different CIFAR variant, upload the instructor dataset/code template so the dataloader can be adapted exactly.** Do not rename arbitrary files to NPZ.

## Public-reference demonstrations (not automatically the teacher's split)

\`\`\`powershell
python script/stage3/run_stage3.py --dataset mnist --source standard
python script/stage3/run_stage3.py --dataset cifar10 --source standard
\`\`\`

The above downloads torchvision MNIST / CIFAR-10. This validates the implementations but must not be represented as results on the instructor-provided files unless equivalence is verified. **ORL must use local data**; an alternative face dataset is never silently substituted.

## Fast diagnostics

\`\`\`powershell
python script/stage3/run_stage3.py --dataset mnist --source standard --quick --epochs 2
python script/stage3/run_stage3.py --dataset mnist --source standard --configs baseline --epochs 2
\`\`\`

Quick mode is a small diagnostic subset and **not report-worthy**.

## Outputs

For each dataset, \`results/stage3/<dataset>/full/\` contains:

- \`ablation_validation.csv\` — each architecture's number of parameters and validation accuracy/precision/recall/macro-F1.
- \`<config>_loss_history.csv\`, \`best_loss_curve.png\`, \`ablation_val_loss.png\` — actual training histories.
- \`final_test_metrics.csv\` and \`confusion_matrix.png\` — one held-out test evaluation for the selected model.
- \`architecture.png\` — architecture sketch; describe all actual layers in the report.
- \`experiment_settings.json\` — train/validation/test counts, source provenance, hyperparameters, seed, winner selection.
- \`best_model.pt\` — checkpoint for reproducibility (gitignored).

The project's preexisting \`.gitignore\` excludes \`data/\` and \`results/\`; **save/export final figures and tables for Canvas separately**.

## Report requirements

Follow the official Stage (2–5) template: <=5 pages, 10pt, single-spaced, include task, architecture image + details, **all three teacher datasets**, per-dataset partitions, configurations and hyperparameters, accuracy/precision/recall/F1 (for multiclass specify macro), source-code link, train/validation loss curves, test tables, and meaningful architecture ablations.

There is no claim that a more complicated CNN automatically scores better; the **validation vs test performance** determines which configuration generalizes.
