# Stage 4 — Verbatim Qualitative Generation Checks

These examples are excerpted from the **completed GloVe 300d** experiment's
`generation/generation_examples.json` in [the archived experimental results](../../stage4_report_results.zip).
They are **not** hand-edited output, results from improved sampling, or unseen test-set
examples. Prompts and reference texts come from the instructor joke *training* partition.

The reported 165.54 validation perplexity (vanilla RNN) measures next-token
prediction; it **does not** measure whether a story is grammatical or funny.

## Same three starting words, three different models

| Three-word prompt | RNN continuation | LSTM continuation | GRU continuation |
| --- | --- | --- | --- |
| `i'm very keen` | `i'm very keen.` | `i'm very keen i was a.` | `i'm very keen i have a dirty - fi. ( this )` |
| `i may be` | `i may be a pair of them.` | `i may be a little and so "` | `i may be two people like? "` |
| `it's an emergency` | `it's an emergency to get to be a new hampshire..... i don't get for a little meteor... but i had a joke of cheese - - from my life. "' " is this` | `it's an emergency, you can have a joke - - post from a - fi.` | `it's an emergency my laptop. " i am` |

## Explicit comparison against the actual training reference

| Prompt | What the corresponding training joke contains | What the generated example shows |
| --- | --- | --- |
| `i'm very keen` | The reference continues about recognizing baldness "at the drop of a hat" | The RNN ends immediately; other models change topic or produce malformed syntax |
| `i may be` | The reference describes being middle-class and a wordplay on "al dente" | All three models output fragments without the reference's logical setup and punchline |
| `it's an emergency` | The reference asks for underwear jokes for a younger sister | All generated examples diverge into unrelated fragments and fail to answer the prompt coherently |

These references are **training data used for qualitative comparison**, not target
sentences that the model was required to reproduce verbatim. A different continuation
could be good, but these nine samples show recurring shortcomings: abrupt termination,
malformed punctuation, topic drift, and lack of a setup/punchline structure.

## Interpreting the metric mismatch

The GloVe 300d vanilla RNN had the lowest *validation* perplexity (165.54), yet
its first generated sample was just three words and a period. This is not a
contradiction: token cross-entropy and stochastic text generation test different
properties. The input dataset is small (1,622 jokes), and the saved samples show
formatting noise. A proper assessment of humor or coherence would require a
human-rated or separately defined qualitative protocol; none is claimed here.

For full, untruncated references, parameters, seed, and other saved examples, open
`results/stage4/glove300_full/generation/generation_examples.json` inside the archive.

An optional [checkpoint-only resampling tool](../../script/stage4/resample_generation.py)
can generate additional candidates without retraining, but any such outputs must
be labeled *decoding-only* and must not replace the original evidence.
