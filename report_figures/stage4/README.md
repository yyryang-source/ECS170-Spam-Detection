# Stage 4 — Instructor Data, Final Experiment Evidence

This is a human-readable index of the **actual saved CSV / JSON / PNG** data in [stage4_report_results.zip](../../stage4_report_results.zip); no model scores in this file are smoke-test scores. Each of six embedding configurations trained RNN, LSTM, and GRU for IMDb classification and joke generation (36 model/task runs). See [Stage 4 source and run guide](../../README_STAGE4.md).

## IMDb — official held-out test macro F1 (%)

| Initialization | RNN | LSTM | GRU |
| --- | ---: | ---: | ---: |
| Random 50d | 74.09 | 81.10 | 83.50 |
| GloVe 50d | 76.80 | 83.29 | 86.70 |
| Random 100d | 73.34 | 81.89 | 84.99 |
| GloVe 100d | 76.15 | 84.59 | 87.35 |
| GloVe 200d | 76.89 | 85.82 | 87.648 |
| GloVe 300d | 75.63 | 84.64 | **87.660** |

The maximum *observed* test macro F1 is 87.660% (GloVe 300d + GRU). **Model selection is by validation macro F1, not by test F1.** By validation, GloVe 200d + GRU was better (89.639% vs 89.559% for 300d); its test macro F1 was 87.648% with ~6.13M parameters versus ~9.17M for 300d. These small differences are from single-seed experiments and should not be interpreted as statistically significant.

## Joke generation — validation perplexity (lower is better)

| Initialization | RNN | LSTM | GRU |
| --- | ---: | ---: | ---: |
| Random 50d | 196.57 | 218.64 | 193.97 |
| GloVe 50d | 188.42 | 231.15 | 193.97 |
| Random 100d | 187.89 | 182.26 | 176.61 |
| GloVe 100d | 180.59 | 194.33 | 187.77 |
| GloVe 200d | 169.09 | 176.68 | 174.59 |
| GloVe 300d | **165.54** | 187.14 | 177.21 |

No independent joke test split was provided. The lowest validation perplexity is 165.54 (GloVe 300d + vanilla RNN, epoch 21). Lower perplexity is not the same as grammatical correctness or humor.

## Qualitative comparison to a real training sample

The JSON contains three three-word prompts, a *training* example with that same prefix, and a generated continuation for each architecture. For example, for the prompt "i'm very keen":

- Training reference: "I'm very keen I could tell he was bald at the drop of a hat."
- GloVe 300d vanilla RNN output: "i'm very keen."
- GloVe 200d GRU output: "i'm very keen a joke about the difference between a cow walks into a tree"

These are verbatim excerpts from the saved example files and show that a lower predictive loss does not guarantee a complete joke or punchline. The reference is from the *training* partition, not unseen test data; no memorization/generalization claim is made based on these examples.

## Data and reproducibility

- IMDb: 22,500 fit, 2,500 validation, 25,000 independent instructor test reviews; max length 256 tokens.
- Jokes: 1,460 fit and 162 validation texts; 4,491-word vocabulary, max length 64 tokens.
- Same seed 42, AdamW lr 0.001 / weight decay 0.0001, dropout 0.30, hidden size 128, batch size 64, 5 classification / 30 generation epochs, cross entropy.
- GloVe vectors were fine-tuned. Vocabulary coverage: 28,501/30,000 IMDb tokens and 4,183/4,491 joke tokens.
- All original histories, classifier test metrics, generation validation metrics, generation examples, vocabulary/settings, and PNG loss curves are inside the ZIP. Checkpoints, raw data and GloVe files are intentionally excluded.
- Historical scores are not claimed to have been produced by the instructor-template wrapper, which was added afterward.

**Limitations:** 1 seeded run per configuration, noisy short jokes with malformed punctuation and web artifacts, sampling not a controlled human evaluation, and limited generalization evidence for generation.
