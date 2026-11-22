# ECS 170 — Stage 4: IMDb Sentiment and Joke Generation

**Course:** ECS 170, Fall 2026 | **Student:** Yourui Yang

This implementation uses the instructor's REAL course ZIP. It is separate from and does not replace Stage 3. It compares vanilla RNN, LSTM, and GRU on two tasks:

- IMDb binary sentiment classification: 25,000 labeled train + 25,000 instructor test reviews, split into pos/neg directories. A stratified 10% validation subset comes ONLY from the instructor training partition.
- Short-joke language modeling: 1,622 rows in the instructor CSV (columns ID,Joke). A fixed 90/10 train/validation split is derived from those rows; there is NO separate instructor joke test set. The trained model generates continuations from exactly three starting words. Saved examples include a training joke with the same opening for qualitative comparison.

All models train from scratch by default. Optional Stanford GloVe embeddings are supported from ZIP or TXT. There is no automatic download of datasets or weights, and no invented performance claims.

## macOS setup (Apple Silicon and Intel Macs)

From Terminal (macOS Python 3.10+; ideally Python 3.11 or 3.12):

~~~bash
git clone https://github.com/yyryang-source/ECS170-Spam-Detection.git
cd ECS170-Spam-Detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-stage4.txt
python -m unittest discover -s tests -p "test_stage4.py"
mkdir -p data
~~~

Download the instructor's **stage_4_data.zip** from:
https://drive.google.com/file/d/1yNr8deq1JO-HNhlAa1yV4Zycjqmr1Hfz/view

Place the ZIP in the repo's data/ folder as data/stage_4_data.zip. You do NOT need to extract it; the runner ignores the __MACOSX entries. If you did unzip it, use --data /path/to/stage_4_data instead. Data and results are gitignored.

## Commands

First run a one-epoch smoke test for all six model/task combinations:

~~~bash
python script/stage4/run_stage4.py --data data/stage_4_data.zip --task both --quick
~~~

Then run the full instructor-data experiment:

~~~bash
python script/stage4/run_stage4.py --data data/stage_4_data.zip --task both --device auto
~~~

Default training: RNN, LSTM, GRU; 100d embeddings; hidden dimension 128; AdamW 0.001; dropout 0.3; 5 IMDb epochs and 30 joke epochs. Mac Apple Silicon uses MPS if available, otherwise CPU; change with --device cpu. This may take a while on laptops. No GPU is required.

To run only one task or subset, or compare different configurations:

~~~bash
python script/stage4/run_stage4.py --task classification --models gru --epochs-classification 8
python script/stage4/run_stage4.py --task generation --models rnn,lstm --epochs-generation 40
python script/stage4/run_stage4.py --device cpu --hidden-dim 256 --layers 2
~~~

**Optional** GloVe 100d from the course page (approximately 560MB ZIP):
https://drive.google.com/file/d/10F62gyUWe3EPHBhI8L7_NsO5Gp0PB3NR/view

~~~bash
python script/stage4/run_stage4.py --glove ~/Downloads/glove.2024.wikigiga.100d.zip --embedding-dim 100
~~~

The GloVe ZIP is streamed without full extraction; only vectors needed for the current vocabulary are stored in memory. Select 50/200/300 embedding dimensions if using the corresponding course GloVe download. GloVe is NOT required by Stage 4.

## Output and report evidence

Generated results are in results/stage4/:

- classification/test_metrics.csv — test accuracy, macro precision, macro recall, macro F1, loss, and parameters for EACH network. Test partition is not used for vocabulary, validation or checkpoint selection.
- classification/*_history.csv and *_loss.png — convergence plots.
- generation/validation_metrics.csv — validation token accuracy, token-level cross-entropy, perplexity.
- generation/generation_examples.json — three-word prompts, actual source-training jokes, model generations (qualitative comparison).
- generation/*_history.csv and *_loss.png — convergence plots.
- */settings.json and */vocab.json — data partition details and vocabulary.
- */*_best.pt — locally saved trained weights (gitignored).
- summary.json — aggregate results after training.

The official report template is at:
https://docs.google.com/document/d/1fLUhtUe_Rm4x-sL4N-LxBJ57k0enaCEnQNvWwTUkNY0/edit

Write no more than 5 pages (10 pt, single spacing), including architecture diagram, dataset/split table, training curves, evaluation metrics, model comparisons/ablation, and the source-code link. Clearly label generation as TRAIN/VALIDATION, not instructor TEST. Do not report --quick results as final coursework results. Do not call the output guaranteed to reproduce a joke; generation is stochastic and depends on training.

### Source and evaluation notes

- The classification models are intentionally unidirectional (one RNN/LSTM/GRU recurrent layer) and use the final unpadded token's output. This avoids sequence-packing limitations on some MPS builds.
- Model checkpoints: classification picks maximum validation macro F1 (tie: minimum loss); generation picks minimum validation cross-entropy. Each classification model is evaluated on the untouched instructor test set once after training.
- Joke generation uses learned next-token distribution and top-k sampling. The source-training reference is included for manual review, not as a promise of exact reproduction.
- To report architecture ablation beyond RNN vs LSTM vs GRU, rerun with different --hidden-dim/--layers and keep output directories distinct using --output-dir.
- Do not upload the instructor dataset, pre-trained embeddings, virtualenv, or large model checkpoints to public GitHub.

## Completed report experiments

The tracked [Stage 4 results archive](stage4_report_results.zip) contains CSV, JSON, and PNG evidence from 36 full model runs (random embeddings at 50d/100d; GloVe at 50d/100d/200d/300d; each tested with RNN/LSTM/GRU on both tasks). Local checkpoint files remain excluded from Git.

- **Best IMDb held-out test macro F1:** GloVe 300d + GRU, **87.660%** (test accuracy **87.660%**). GloVe 200d + GRU achieves **87.648%** test macro F1 with fewer parameters.
- **Lowest joke-generation validation perplexity:** GloVe 300d + RNN, **165.54**. Generated jokes remain qualitatively imperfect; lower perplexity alone does not establish better humor or coherence.
- Classifier checkpoints are chosen with *validation macro F1*, then evaluated on the held-out instructor test split. The joke-generation dataset has only a derived validation split; its reported perplexity is **not** an instructor test metric.
- These are individual training runs, not multiple-seed confidence intervals.

On a Mac where the MPS backend exits unexpectedly, prefer `--device cpu` for the full run. Use separate `--output-dir` paths for each embedding configuration to avoid overwriting results.

## Instructor-template entry point (recommended for TA review)

The preferred instructor-style entry point is now:
```bash
# Run a CPU smoke test on real course files (one epoch; never report the score).
python script/stage_4_script/script_rnn.py --data data/stage_4_data.zip --quick --device cpu

# Full run, same historical defaults; use a NEW output folder.
python script/stage_4_script/script_rnn.py --data data/stage_4_data.zip \
  --task both --models rnn,lstm,gru --device cpu \
  --output-dir results/stage4/template_full

# Optional GloVe 200d comparison (new output folder; CPU is reliable on this Mac).
python script/stage_4_script/script_rnn.py --data data/stage_4_data.zip \
  --glove "$HOME/Desktop/glove.2024.wikigiga.200d.zip" \
  --embedding-dim 200 --device cpu \
  --output-dir results/stage4/template_glove200
```

This interface uses the provided base-class design:
- Dataset_Loader inherits dataset; checks the supplied IMDb/joke partitions.
- Method_RNN inherits method; invokes the SAME tested training/evaluation routines under script/stage4.
- Setting_RNN_Experiment inherits setting; calls load -> train -> evaluate -> save.
- Evaluate_RNN inherits evaluate; selects the best classifier strictly by validation macro F1 and the best generator strictly by validation cross-entropy. The independent IMDb test set is not used to select a winner.
- Result_Saver inherits result; saves template_summary.json, task-specific model_architecture.png, and the original trainer's histories/CSV/checkpoints.

For fast offline verification:
```bash
python -m unittest discover -s tests -p "*stage4.py" -v
```

The 36 published historical experiments were made with script/stage4/run_stage4.py; the instructor-style wrapper was added afterward and has the same training functions and default hyperparameters. Do not claim that the archived experiments were run via the newer wrapper. Compare outputs only for matched configuration and seed. The existing Stage 4 results ZIP is unchanged.

## Resampling from an already trained generation checkpoint

To produce several **additional qualitative** continuations without 30-epoch retraining:
```bash
python script/stage4/resample_generation.py \
  --experiment-dir results/stage4/glove300_full \
  --models rnn,lstm,gru --num-samples 3 \
  --min-new-tokens 10 --repetition-penalty 1.15 \
  --output results/stage4/glove300_full/generation/resampled_examples.json
```

The resampler reuses existing local checkpoint files (not in Git); no dataset or GloVe ZIP is needed. It implements temperature, top-k/top-p, minimum continuation length and an optional no-repeat trigram constraint, and filters obvious web-format artifacts. This only changes *sampling*, not the trained model, validation perplexity, or accuracy. Report the full sampling settings; never pass off curated outputs as test-set evidence.

See also [Stage 4 TA review checklist](STAGE4_SUBMISSION_CHECKLIST.md) and [experiment results](report_figures/stage4/README.md).
