# ECS 170 Stage 3 CNN Image Classification

Run the Fall 2026 Stage 3 experiments with the instructor-provided `stage_3_data.zip`. The ZIP is read directly, so it can stay in the location where it was downloaded.

## Setup

From the repository root, using the same Python interpreter for installation and training:

```powershell
python -m pip install -r requirements-cnn.txt
$dataset = 'C:\path\to\stage_3_data.zip'
$env:OMP_NUM_THREADS = '8'
python script/stage_3_script/script_cnn_cifar.py --source local --data-root $dataset --epochs 15
python script/stage_3_script/script_cnn_mnist.py --source local --data-root $dataset --epochs 3
$env:OMP_NUM_THREADS = '6'
python script/stage_3_script/script_cnn_orl.py --source local --data-root $dataset --epochs 25 --batch-size 32 --patience 7
```

Each command runs three configurations on the entire dataset. For a local pipeline check, add `--quick --configs baseline` to use a small subset and save separate diagnostic output.

## Template structure

- `local_code/base_class/` contains the course template interfaces.
- `local_code/stage_3_code/` implements `Dataset_Loader`, one `Method_CNN_*` class for each task, the experiment setting, the evaluator, and the result saver.
- `script/stage_3_script/` contains the three training entry points. Each calls `setting.prepare(...)` and `setting.load_run_save_evaluate()`.
- `script/stage3/` contains the shared data handling and CNN architecture. Its standalone runner is available for historical experiments.

## Dataset partitions and training

| Dataset | Original training set | Fit | Validation | Held-out test | Classes |
| --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | 60,000 | 54,000 | 6,000 | 10,000 | 10 |
| ORL | 360 | 270 | 90 | 40 | 40 |
| CIFAR-10 | 50,000 | 45,000 | 5,000 | 10,000 | 10 |

The loader retains the instructor's test images, labels, and ordering. It draws stratified validation examples only from the original training set, using seed 42. ORL's three identical color channels are converted to one grayscale channel.

All three models use convolution, batch normalization, ReLU, and max pooling, followed by adaptive average pooling, a 128-unit dense layer, dropout, and class logits. The baseline uses two convolution blocks with 32 and 64 channels. The residual variant uses three blocks with 32, 64, and 128 channels and projection shortcuts. The kernel-5 variant changes the residual model's kernel from 3×3 to 5×5. ORL pools to a 4×4 grid to retain facial feature locations. MNIST and CIFAR-10 pool to 1×1.

Models train from random initialization with AdamW, learning rate 0.001, weight decay 0.0001, cross-entropy label smoothing 0.05, dropout 0.30, and gradient clipping at 1.0. The checkpoint with the lowest validation loss represents each configuration. The setting chooses a configuration by validation macro F1 and then scores it on the test split.

## Results and verification

Full runs write `result/stage_3_result/<dataset>/full/`. Each directory contains train and validation loss histories, validation ablations, held-out accuracy and macro precision, recall and F1, an architecture diagram, a confusion matrix, and the selected checkpoint. The course ZIP and local artifacts are excluded from Git.

The [reported experiments](report_figures/stage3/README.md) include template-based MNIST and ORL runs trained directly on the instructor archive. The earlier CIFAR-10 run used an exact matching copy of the dataset. Its saved checkpoint reproduced its validation and test metrics when rescored through the template loader on the course ZIP.

```powershell
python -m unittest discover -s tests -p 'test_stage3.py'
python -m unittest discover -s tests -p 'test_teacher_template_stage3.py'
```

Course material: [ECS 170 course page](https://www.ifmlab.org/courses.html).
