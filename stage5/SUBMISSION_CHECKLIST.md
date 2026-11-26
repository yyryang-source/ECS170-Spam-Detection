# Stage 5 submission checklist

This checklist maps the requested Stage 5 work to the implementation and verified evidence. Source is published on the stage5-gcn branch under stage5/. Course-platform submission remains separate.

| Requirement | Evidence |
| --- | --- |
| 5-1: inspect all three instructor datasets | `references/data_audit.json`, the dataset table in `README.md`, and each run's `dataset.json` with file hashes and graph counts |
| 5-2: implement GCN in the provided framework | Custom layer and model in `local_code/stage5/model.py`; five instructor-style components in `local_code/stage_5_code/`; entry point `script/stage_5_script/script_gcn.py` |
| 5-3: train and evaluate Cora; provide learning curves | Three baseline runs under `results/cora/baseline/`; saved train/validation histories, restored validation-selected checkpoints, and test predictions/metrics |
| 5-4: repeat for Pubmed and Citeseer | Corresponding baseline directories under `results/pubmed/` and `results/citeseer/`; summaries in `results/summary.json` |
| Architecture/settings comparison | Matched width-32 ablation under every dataset's `hidden32/`, using the same splits, initialization seeds, optimizer, and stopping rule |
| Graph embedding | `embeddings.npz` in every run, containing deterministic last-hidden-layer node embeddings, node IDs, labels, and predictions |
| 5-5: template-based report, maximum five pages | `report/Stage5_Report.pdf`; editable `report/Stage5_Report.html`; generation source `script/make_report.py` |
| 5-6: optional GPU trial | `results/gpu_probe.json`: MPS forward succeeded but backward aborted; CPU was used for all reported experiments |

## Check the final submission

- [x] Open `report/Stage5_Report.pdf` and verify the final identity fields, readable figures/tables, and total length of no more than five pages.
- [x] Confirm that all reported scores agree with `results/summary.json` and that figures come from the saved full-run histories. Do not replace them with a short diagnostic run.
- [x] Preserve the instructor split quotas: 20 train nodes per class; 150 Cora test nodes per class or 200 for Pubmed/Citeseer. Explain the added 10 validation nodes per class and fixed split seed 42.
- [x] Label the task as transductive: all graph structure/features are available, training loss uses training labels, validation selects checkpoints, and test evaluation follows selection.
- [x] Describe the three seeds as initialization/dropout repeats on one fixed split. Mean +/- sample standard deviation is not a confidence interval over random splits.
- [x] Keep the 16-unit baseline and 32-unit ablation identifiable. Do not present the higher observed test score as a test-selected deployment model.
- [x] State that the results are CPU results. The failed MPS probe is an optional attempted experiment, not completed GPU training.
- [x] Run `python -m unittest discover -s tests -v` after any code edits. For a full reproduction, use the documented command with a new output directory and the original extracted data.
- [x] Publish the intended Stage 5 source and add its verified branch URL to the report.
- [x] Exclude instructor datasets, environments, private report identity fields, and local checkpoints/embeddings from public Git. Retain the local evidence package for reproduction.
- [ ] Upload the final PDF and any requested source/evidence package to the actual course submission destination.

## Evidence conventions

The included `results/` tree has 18 completed runs: three datasets x two widths x three initialization seeds (42, 43, 44). `splits.json` records exact disjoint train/validation/test indices; `dataset.json` records source hashes and preprocessing; `history.csv` records learning curves; `metrics.json` includes checkpoint epoch and test/per-class metrics; `test_predictions.csv` permits independent metric recalculation. The saved model state is the checkpoint with the lowest validation loss, not necessarily the final epoch.

Learning curves use evaluation mode after each optimizer update, making training and validation loss directly comparable without dropout noise. The separate `train_objective_dropout` column records the training-mode optimization objective. The saved node embeddings also disable dropout.

The feature and graph conventions are explicit: row-L1 normalization; symmetric adjacency; preservation of raw self-loops followed by addition of the identity; symmetric degree normalization. Citeseer's 3,312 nodes and 124 raw self-loops come from this instructor version. Results should not be labeled as standard Planetoid-split benchmark scores.
