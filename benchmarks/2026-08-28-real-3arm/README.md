# Three-arm benchmark, run 1 — the fine-tune collapsed to templates

**2026-08-28** · commit `235f5b9` · `google/gemma-4-12B-it` + QLoRA r=32 · real USASpending corpus

The first honest three-arm comparison. It returned a negative result, and the
reason it returned one is more useful than the number.

| arm | | |
|---|---|---|
| **A** | fine-tuned + retrieval | the full proposition |
| **B** | fine-tuned, no retrieval | is the graph in the weights? |
| **C** | base model + retrieval | what retrieval alone already buys |

## Result

18 blind questions, answers drawn from subcontracts dated after the training
cutoff. Floor is a random pick from each question's own slate.

| archetype | n | **floor** | C base+RAG | B tuned | A tuned+RAG |
|---|---|---|---|---|---|
| `blind_next_team` | 10 | **0.422** | 0.156 | 0.200 | 0.400 |
| `blind_new_entrant` | 8 | **0.359** | 0.357 | **0.000** | **0.000** |

**No arm beat the floor.** Arm A came closest on the archetype nearest its
training — 0.400 against 0.422 — which is chance. On the archetype absent from
training it scored zero, where random scores 0.36.

## Cause

| | answers forced into a training template |
|---|---|
| arm C (base) | **0 / 18** |
| arm B (tuned, no retrieval) | 13 / 18 |
| **arm A (tuned + retrieval)** | **18 / 18** |

Asked *"which of these companies are new to NIH?"*, arm A answered:

> *Primes to approach for NIH Custom Computer Programming Services work, by
> subcontracting volume: 1. RESEARCH TRIANGLE INSTITUTE (10 reported
> subcontracts)*

Fluent, correctly formatted, and an answer to a question nobody asked — the
shape of the `prime_candidates` archetype. The untuned base model instead worked
the actual problem: identify which companies have records, filter for NIH, check
for prior work.

**Training on seven answer templates taught the model to emit one of seven
answer templates.** Every arm-A answer, without exception, was one of them.

This is invisible on in-distribution evaluation, where every question matches a
trained shape. It took a blind set containing a question type held out of
training to surface it — and reporting per archetype rather than in aggregate to
make it legible. The aggregate averages metrics computed over different question
subsets, which is the same blindness that hid an earlier context leak.

## Training run

| | |
|---|---|
| wall time | 1h 05m (168 steps, 2 epochs) |
| cost | $0.10 · peak allocated 22.01 GB of 31.84 |
| train loss | 3.6746 → 0.1211 |
| eval loss | 0.6065 → 0.3546 |
| gate | **STOP** — still learning (+5.55%) but train/eval gap **65.8%** |

## What this licenses

**It does say:** at this scale, on this corpus, fine-tuning did not beat a base
model with retrieval, and it destroyed generalisation to question types held out
of training.

**It does not say fine-tuning cannot work here.** Five bounds:

1. **n = 18** — ten and eight per archetype. A few points is noise.
2. **401 training facts.** Very little to learn from.
3. **Seven archetypes is a narrow instruction diet**, and template collapse is
   the *expected* failure of a narrow diet, not a surprising one.
4. **Over-trained by its own gate** at a 65.8% train/eval gap. One epoch was
   never evaluated.
5. **The blind task is hard by design** — 71% of the true pairings appear
   nowhere in training.

## A note on the corpus this replaced

An earlier version of this same run scored arm C at a perfect 1.000 on
`blind_new_entrant`. That was a leak: `context_for` assembled the retrieved
records from the question's own answer key, so eight of eight records supplied
*were* the answer. Fixed, and now guarded by five tests in
`tests/test_real_corpus.py`. The comparison across the two corpus versions is
the clearest evidence the fix took:

| | leaked | leak-free |
|---|---|---|
| final eval loss | 0.2799 | 0.3546 |
| what epoch 2 bought | −0.10% | +5.55% |

A worse loss is the better outcome — the task became genuinely harder once the
answer stopped arriving in the context.

## Files

`arms.json`, `arms_report.txt`, `arm_{a,b,c}_generations.jsonl`, `gate.json`,
`metrics.json`, `corpus_stats.json`, `commit.txt`.
