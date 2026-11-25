# ECS 170 - Stage 5 GCN (Graph Embedding and Node Classification)

**Yourui Yang | UC Davis | Fall 2026**

This Stage 5 implementation follows the uploaded instructor source-code template.
It is additive: Stage 3 CNN and Stage 4 RNN/LSTM/GRU files and their
published evidence remain unchanged.

## What is implemented

- Cora, Citeseer, Pubmed citation-graph node classification.
- Two-layer GCN baseline; three-layer GCN; 128-wide two-layer GCN;
  and controlled no-dropout ablation.
- Sparse symmetric adjacency with self loops and binary duplicate
  deduplication: \`D^-1/2 (A+I) D^-1/2\`. **No dense N x N adjacency**.
- Train-only cross-entropy, validation-only checkpoint/architecture selection,
  then a single held-out test evaluation for the chosen configuration.
- Macro Accuracy/Precision/Recall/F1, train/validation loss history,
  2D PCA node-embedding visualization, confusion matrix, architecture diagram,
  ablation table, reproducibility metadata, model checkpoint.
- Original \`local_code/base_class/\` interfaces with
  \`Dataset_Loader\`, dataset-specific \`Method_GCN_*\`,
  \`Setting_GCN_Experiment\`, \`Evaluate_Node_Classification\`,
  and \`Result_Saver\` under \`local_code/stage_5_code/\`.
- Executable entry points in \`script/stage_5_script/\`.

## Install on Windows PowerShell

From the repo root:

~~~powershell
git pull --ff-only origin main
python -m pip install -r requirements-stage5.txt
python -m unittest discover -s tests -p "test_stage5.py" -v
python -m unittest discover -s tests -p "test_teacher_template_stage5.py" -v
~~~

If using **public Planetoid** reference datasets, additionally install:

~~~powershell
python -m pip install torch-geometric
~~~

This extra package is needed only for **downloading/reading public
Planetoid data**. The GCN itself uses native PyTorch operations.
\`--device cpu\` works on Windows and Apple Silicon; CUDA can be used with
a matching PyTorch install. MPS is explicitly rejected rather than
silently densifying sparse adjacency.

## Test the complete flow

~~~powershell
python script/stage_5_script/script_gcn.py --dataset cora --source planetoid --quick --device cpu
~~~

\`--quick\` does only 2 training epochs, with all graph nodes preserved.
**Never use quick results as final assignment scores.**

## Run all three benchmarks (public reference, NOT instructor datasets)

~~~powershell
python script/stage_5_script/script_gcn.py --dataset all --source planetoid --device cpu
~~~

These publicly downloaded Planetoid graph files and their **public
train/validation/test masks** may differ from the files supplied by
the instructor. Do not claim these are the instructor-specific results
without checking identifiers, splits and checksums.

## Run with instructor-provided local data (preferred for grading)

Supported **safe** input formats:
- One \`cora.npz\`, \`citeseer.npz\`, \`pubmed.npz\` file each, in
  \`data/stage_5_data/\`.
- Or one ZIP with a unique, appropriately named \`<dataset>.npz\`
  for each of the three datasets.
- Arrays: \`x\` or \`features\` of shape [N,F]; \`y\` or \`labels\` of
  shape [N]; \`edge_index\` or \`edges\` of shape [2,E] or [E,2].
- **All three** boolean \`train_mask\`, \`val_mask\`, \`test_mask\`, or
  \`train_idx\`, \`val_idx\`, \`test_idx\`. Split masks must be non-overlapping.

~~~powershell
# First validate the actual instructor file before training.
python script/stage_5_script/script_gcn.py --dataset all --source local --data-root data/stage_5_data --check-data

# Full coursework experiments.
python script/stage_5_script/script_gcn.py --dataset all --source local --data-root data/stage_5_data --device cpu
~~~

If the teacher supplies a different container format (for example,
original \`ind.cora.*\` Python pickles, CSV+edges, or other serialized
objects), do **not** rename it to NPZ. Share the teacher archive so the
loader can be adapted safely and the official splits preserved. This
implementation deliberately avoids untrusted pickle execution.

**Optional research-only fallback:** \`--allow-generated-split\` explicitly
uses seeded stratified 60/20/20 splits when a local NPZ has no masks.
Do not use those generated splits as if they came from the instructor.

## Report outputs

\`results/stage5/<dataset>/full/\` contains:
- \`experiment_summary.json\`: data provenance, split counts, seed,
  architecture selection methodology, hyperparameters, actual scores.
- \`ablation_validation.csv\`: number of parameters, accuracy, macro
  precision/recall/F1 for every configuration on the validation split.
- \`baseline_history.csv\`, \`deep_history.csv\`, etc.
- \`best_loss_curve.png\`, \`ablation_val_loss.png\`,
  \`architecture.png\`, \`confusion_matrix.png\`,
  \`node_embedding_pca.png\`.
- \`final_test_metrics.csv\` for the **single selected** configuration.
- \`best_gcn.pt\`, excluded by Git from the public repository.

The full report follows the official Stage (2-5) template, 10pt single
spacing and no more than five pages:

1. Task description and GCN node-embedding/node-classification problem.
2. Actual model architecture with dimensions, normalized sparse A.
3.1 Dataset graph sizes, features, classes and **exact** splits.
3.2 Hyperparameters, initialization, optimizer and tuning procedure.
3.3 Accuracy, macro precision/recall/F1 definitions.
3.4 GitHub source-code link.
3.5 Training convergence charts for each graph.
3.6 Selected model's independently held-out test metrics for each graph.
3.7 Validation ablations for depth/width/dropout, parameter counts and
    generalization conclusions.

**No real Stage 5 training results have been generated or claimed** by
the synthetic offline CI tests. Formal completion also requires the
teacher's dataset ZIP, full local training, review of the actual
generated results and a final PDF submitted through Canvas.
