# ECS 170 — Stage 4 grading-alignment checklist

Based on the official ECS 170 Fall 2026 project description (Stage 4 items 4-1 through 4-6) and the Stage 2–5 report template. This is a self-audit, **not a guarantee of any grade**.

| Official requirement | Evidence / action |
| --- | --- |
| 4-1: two instructor text datasets | Original course archive read directly; see Dataset_Loader and script/stage4/data.py |
| 4-2: text classification RNN in provided code template | local_code/stage_4_code/ inherits all five provided base classes; script/stage_4_script/script_rnn.py |
| 4-3: classification train/test + learning curves | IMDb held-out test metrics and train/val loss histories in stage4_report_results.zip |
| 4-4: story from exactly 3 starting words, compare with training text | generation_examples.json includes prompt, training_reference, generated for 3 seeds/prompts per model |
| 4-5: replace RNN with LSTM and GRU; compare settings | 6 embedding settings × 3 architectures on both tasks; compare like-for-like initialization and dimensions |
| 4-6: report in prescribed template, <=5 pages | Use the official report template; verify 10pt single spacing, full task/model/dataset/metrics/source/convergence/ablation sections |
| Source-code accessible | https://github.com/yyryang-source/ECS170-Spam-Detection |
| Verification | GitHub Actions: Stage 4 unit tests and template integration test; also manually test on the Mac CPU with --quick |

## Before submitting to Canvas

1. **Run the instructor-template quick command on your local machine** after pulling main (synthetic CI tests do not prove your downloaded course ZIP can execute locally).
2. Open the final PDF and confirm every table is labeled correctly: IMDb = official held-out TEST; generation = VALIDATION only.
3. Ensure the report contains a real model-architecture figure, convergence curves for **both** datasets, and a matched-configuration ablation explanation.
4. Show at least one *actual* generated continuation beside its training-reference continuation; distinguish the reference from test evidence.
5. Link to the *public* GitHub source and confirm the code is readable without raw instructor data or pretrained embedding files.
6. Never include --quick smoke-test metrics in the coursework report; don't claim the template wrapper produced the archived historical results.
7. Verify your actual submission instructions and final PDF formatting in Canvas before uploading. The repository does not perform the Canvas submission.

## Preserve evidence

- stage4_report_results.zip: archived CSV, JSON and PNG evidence from 36 real runs
- README_STAGE4.md: original runner, wrapper, dependencies and commands
- local_code/stage_4_code/: template-inheritance implementation
- tests/test_teacher_template_stage4.py: synthetic offline integration check
- No pretrained GloVe files, model .pt checkpoints, or instructor dataset files are committed.

Improving sampling from checkpoints may make stories more readable, but it does not improve a model's already-reported validation score, and it must be labeled as a decoding-only experiment.

## Automated evidence audit

Run `python script/stage4/verify_results.py --clean-output stage4_submission_results.zip`
to validate the tracked original results and create an optional submission-only
ZIP excluding root-level --quick diagnostics. This script is run in Stage 4 CI
and checks the stored best checkpoints' metrics *against epoch histories*
without retraining. It does not replace the separately required final report.
