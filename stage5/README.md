# Stage 5: Graph Embedding and Node Classification with GCN

This project implements a graph convolutional network directly in PyTorch and evaluates it on the instructor's Cora, Pubmed, and Citeseer files. The included evidence contains **18 completed CPU runs**: three datasets, two hidden widths, and three initialization seeds. The local submission bundle also includes `report/Stage5_Report.pdf` (five pages) and editable `report/Stage5_Report.html`; these contain private administrative fields and are intentionally excluded from the public source branch.

## Setup and reproduce

Run these commands from the `stage5_project` directory. The recorded experiment environment used Python 3.13.17 on macOS arm64. `requirements-lock.txt` records exact installed dependency versions; `requirements.txt` lists the direct dependencies for another compatible environment. Results may differ with another platform or package version.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
```

The instructor dataset is **not included** in this package. Extract `stage_5_data.zip` separately, then pass the folder containing `cora/`, `pubmed/`, and `citeseer/`. Each dataset folder must contain the original `node` and `link` files. The loader also accepts the parent of `stage_5_data/`.

```bash
# All 18 experiments; use a new output directory to preserve included evidence.
python script/stage_5_script/script_gcn.py \
  --data-root /path/to/stage_5_data \
  --output-dir results_reproduced \
  --datasets cora pubmed citeseer \
  --seeds 42 43 44 --split-seed 42 \
  --hidden-dim 16 --depth 2 --dropout 0.5 \
  --lr 0.01 --weight-decay 0.0005 \
  --epochs 300 --patience 50 --threads 4 --device cpu --ablation
```

For a single Cora run, use:

```bash
python script/stage_5_script/script_gcn.py \
  --data-root /path/to/stage_5_data \
  --output-dir results_cora_reproduced \
  --datasets cora --seeds 42 --device cpu
```

The runner writes into the requested output directory and replaces files with matching dataset/variant/seed paths. Keep new experiments in separate directories. `--help` lists additional options, including `--raw-features`, which disables the default row-L1 feature normalization and changes the experiment.

## Data and evaluation protocol

The dataset counts below come from the supplied files, not a separately downloaded benchmark version. In particular, this Citeseer file contains 3,312 nodes.

| Dataset | Nodes | Features | Classes | Raw link rows | Unique undirected non-self edges | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 2,708 | 1,433 | 7 | 5,429 | 5,278 | 140 | 70 | 1,050 |
| Pubmed | 19,717 | 500 | 3 | 44,324 | 44,324 | 60 | 30 | 600 |
| Citeseer | 3,312 | 3,703 | 6 | 4,715 | 4,536 | 120 | 60 | 1,200 |

The instructor's train/test quotas are preserved: **20 training nodes per class**, and **150 test nodes per class for Cora or 200 for Pubmed/Citeseer**. A separate validation set of **10 nodes per class** is sampled from the remaining nodes. Sampling is without replacement, with a fixed split seed of 42; all three partitions are disjoint. Exact row indices are saved in every run's `splits.json`. These are instructor-style class-balanced splits, not the standard Planetoid masks.

This is **transductive node classification**: the complete graph and all node feature vectors are available for message passing. Only training-node labels contribute to the gradient loss. Class labels are used to construct the stratified partitions, validation labels select checkpoints, and held-out test labels are evaluated after checkpoint selection. Remaining nodes participate in graph propagation without contributing a supervised loss.

Features are normalized independently per node by their L1 sum; zero rows remain zero. Links are symmetrized using the maximum of the adjacency and its transpose. Existing raw self-loops are preserved, then the identity is added, matching the supplied loader's loop convention. Citeseer has 124 raw self-loops, so their pre-normalization diagonal weight becomes two. The propagation matrix is `A_hat = D^(-1/2) (A + I) D^(-1/2)`. Class mappings, source-file SHA-256 hashes, and graph statistics are retained in `dataset.json`; original node IDs are preserved in prediction CSVs and embedding archives.

## Model and training

`local_code/stage5/model.py` contains the custom `GraphConvolution` and `GCN` classes; it does not use a prebuilt graph-convolution library. Each layer computes `A_hat @ (X @ W) + b`. The two-layer architecture is input dropout -> graph convolution -> ReLU -> hidden dropout -> graph convolution -> class logits. Cross-entropy consumes the logits directly. The last hidden activations are exported as node embeddings with dropout disabled.

The baseline has 16 hidden units; the controlled ablation changes only this width to 32. Both use Adam, learning rate 0.01, weight decay 0.0005, dropout 0.5, full-batch updates, a maximum of 300 epochs, and early stopping after 50 epochs without a lower validation loss. The minimum-validation-loss checkpoint is restored before test evaluation. Training curves report post-update training and validation metrics with dropout disabled; `train_objective_dropout` separately records the optimization loss.

Initialization/dropout seeds are 42, 43, and 44. Splits remain fixed at seed 42 for every width and run. The reported sample standard deviation therefore measures training randomness on one split, not uncertainty across graph splits or a confidence interval. The two widths are reported as a matched ablation; the test results are not used to select a winning configuration for further tuning.

## Recorded test results

Values are **mean +/- sample standard deviation**, in percent, across the three initialization seeds. Source: `results/summary.json`; individual predictions, per-class metrics, and confusion matrices are saved under each run.

| Dataset | Hidden width | Accuracy (%) | Macro precision (%) | Macro recall (%) | Macro F1 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 16 | 80.67 +/- 0.67 | 80.73 +/- 0.64 | 80.67 +/- 0.67 | 80.62 +/- 0.64 |
| Cora | 32 | 81.33 +/- 0.16 | 81.39 +/- 0.23 | 81.33 +/- 0.16 | 81.31 +/- 0.21 |
| Pubmed | 16 | 84.50 +/- 0.93 | 84.45 +/- 0.97 | 84.50 +/- 0.93 | 84.42 +/- 0.91 |
| Pubmed | 32 | 84.17 +/- 0.44 | 84.10 +/- 0.46 | 84.17 +/- 0.44 | 84.07 +/- 0.47 |
| Citeseer | 16 | 62.22 +/- 0.49 | 61.15 +/- 0.44 | 62.22 +/- 0.49 | 61.01 +/- 0.48 |
| Citeseer | 32 | 63.36 +/- 0.32 | 62.31 +/- 0.60 | 63.36 +/- 0.32 | 62.19 +/- 0.50 |

Accuracy equals macro recall here because each test class has the same number of nodes. Width 32 improves the observed mean accuracy for Cora and Citeseer but slightly lowers it for Pubmed. These three-seed differences are descriptive; they do not establish statistical significance or comparability to scores obtained with other dataset versions or splits.

## Files and report regeneration

```text
local_code/base_class/       Instructor interfaces inherited from Stage 4
local_code/stage_5_code/     Dataset/Method/Setting/Evaluate/Result integration
local_code/stage5/           Data handling, custom GCN, training and evaluation
script/stage_5_script/       Instructor-style command-line entry point
script/make_figures.py      Rebuild figures from saved experiment evidence
script/make_report.py       Rebuild PDF and editable HTML report
script/probe_gpu.py         Optional isolated GPU forward/backward check
tests/                      Numerical, gradient, split and loader checks
references/                 Original supplied loader and data audit
results/                    Included completed runs and aggregate summaries
report/                     Report and figures
```

In the complete local submission bundle, each `results/<dataset>/<variant>/seed_<seed>/` directory contains `metrics.json`, `history.csv`, `splits.json`, `dataset.json`, `test_predictions.csv`, `model.pt`, `embeddings.npz`, and `framework_result.json`. `results/runs.json` collects individual metrics and `results/summary.json` collects the three-seed summaries. Checkpoints and embeddings are local reproducibility artifacts and are excluded from public Git. The public branch includes text histories, splits, predictions, metrics and rendered figures. Re-run training with the instructor data to recreate binary artifacts when using only the public source.

Regenerate figures and reports from the included `results/` evidence with:

```bash
python script/make_figures.py
python script/make_report.py --identity report/identity.json \
  --source-url https://github.com/yyryang-source/ECS170-Spam-Detection/tree/stage5-gcn/stage5
```

`report/identity.json` supplies the report's administrative fields and is excluded from Git. Keep its private contents local. The Python generation script is the source for PDF and HTML. Edit that script and regenerate to change both formats; manual HTML edits are not imported into the PDF generator. Check that the final PDF remains no more than five pages.

## Optional GPU experiment

All reported full runs used **CPU**. PyTorch reported MPS available, but the isolated MPS test completed the forward pass and then aborted during backward with a native `MPSNDArray` exception. CUDA was unavailable. The exact diagnostic is saved in `results/gpu_probe.json`; there is no successful GPU training result or GPU speedup claim.

On a compatible machine, the optional check can be rerun without risking the parent process:

```bash
python script/probe_gpu.py --device mps --output results_gpu_probe_new.json
# CUDA alternative:
python script/probe_gpu.py --device cuda --output results_cuda_probe_new.json
```

Inspect the JSON `passed` and `status` fields: successful report creation alone does not mean GPU execution succeeded. Use `--device mps` or `--device cuda` in the training command only on a working backend.

## Provenance and submission

The framework organization and instructor base classes were carried forward from [the Stage 4 repository](https://github.com/yyryang-source/ECS170-Spam-Detection), inspected at commit `5fc3a78a217e46454860382868f7ac251e4a74a9`. The supplied Stage 5 loader is preserved in `references/Dataset_Loader_Node_Classification_original.py`; the implemented loader adds documented reproducible stratified splits and validation handling.

Stage 5 source and text experiment evidence are published under [stage5/ on the stage5-gcn branch](https://github.com/yyryang-source/ECS170-Spam-Detection/tree/stage5-gcn/stage5). The complete local submission ZIP additionally includes checkpoints, node embeddings and the personalized report. Raw instructor data and virtual environments are not included. The separate stage5/ directory preserves the repository's existing files. See [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) for the requirement-to-evidence mapping.
