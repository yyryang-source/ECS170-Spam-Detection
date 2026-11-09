# ECS170 Stage 3 — Official Code Template Adaptation

**Student: Yourui Yang**  
**Course: ECS170 Artificial Intelligence, Fall 2026**

The user-provided `ECS170_Fall_2026_Source_Code_Template.zip` includes:
`local_code/base_class/{dataset,method,setting,evaluate,result}.py`,
the working Stage 1 examples, empty `local_code/stage_3_code/`
and empty `script/stage_3_script/`.

We adapted the Stage 3 CNN project to mirror that exact
**Dataset_Loader → Method_CNN → Setting → Result_Saver → Evaluate** organization.

## Stage 3 source files

```
local_code/
  base_class/
    dataset.py
    method.py
    setting.py
    result.py
    evaluate.py
  stage_3_code/
    Dataset_Loader.py
    Method_CNN.py
    Method_CNN_MNIST.py
    Method_CNN_ORL.py
    Method_CNN_CIFAR.py
    Setting_CNN_Experiment.py
    Evaluate_Classification.py
    Result_Saver.py
    Architecture_Plot.py
script/
  stage_3_script/
    script_cnn.py
    script_cnn_mnist.py
    script_cnn_orl.py
    script_cnn_cifar.py
```

The original Stage 1 code remains in the instructor's extracted template;
Stage 3 extends the **same base-class contracts**.

The architecture and dataset-processing primitives are reused from
`script/stage3/models.py` and `script/stage3/datasets.py`.
The new `Setting_CNN_Experiment` implements the official
`setting.prepare(...)` plus `setting.load_run_save_evaluate()` flow.
Each of the `Method_CNN_*` classes inherits the instructor's
`method` and PyTorch `nn.Module` through `Method_CNN`.

## Requirements

Run from the root of `ECS170-Spam-Detection`:

```powershell
git pull origin main
python -m pip install -r requirements-cnn.txt
python -m unittest discover -s tests -p "test_teacher_template_stage3.py"
```

## Official-style training commands

Reference MNIST (downloads standard torchvision data):

```powershell
python script/stage_3_script/script_cnn_mnist.py --source standard --epochs 15
```

Reference CIFAR-10 (downloads standard torchvision data):

```powershell
python script/stage_3_script/script_cnn_cifar.py --source standard --epochs 15
```

Instructor-provided dataset is preferred. Supply a local directory of per-class
images or an NPZ archive with train/test fields:

```powershell
python script/stage_3_script/script_cnn_mnist.py --source local --data-root "C:\path\to\teacher_mnist.npz"
python script/stage_3_script/script_cnn_orl.py --source local --data-root "C:\path\to\teacher_orl"
python script/stage_3_script/script_cnn_cifar.py --source local --data-root "C:\path\to\teacher_cifar.npz"
```

To verify the path with tiny data and a single model (not report-quality):

```powershell
python script/stage_3_script/script_cnn_mnist.py --source standard --configs baseline --quick
```

All new results will be in `result/stage_3_result/<dataset>/full/`.
They are separate from the **previously completed runs**, which remain
under `results/stage3/<dataset>/full/`. They must not be conflated:
the independent trainer and the template trainer may return different values.

Each completed run saves loss histories, architecture diagram,
validation ablations, test metrics, confusion matrix,
JSON experiment settings and a model checkpoint.
Test-set results are reported only after validation-only selection.

**The ZIP did not contain the instructor's three image datasets.**
ORL cannot be trained without local data. This code supports standard
MNIST/CIFAR-10 for preliminary reproducibility, but if the teacher supplies
a different split/format, add a matching loader and preserve the split.
The class report must document exactly which source files were used.

The official report template requires no more than 5 pages (10pt,
single-spaced), architecture figures, convergence plots, all three
required datasets, multiclass precision/recall/F1, and controlled ablations.
