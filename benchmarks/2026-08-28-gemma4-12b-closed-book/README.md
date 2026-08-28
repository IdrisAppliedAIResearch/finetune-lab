# Baseline: closed-book Gemma 4 12B, 2 epochs

**2026-08-28** · commit `3c1db50` · `google/gemma-4-12B-it` + QLoRA r=32

The first full run of the closed-book design, kept as the number to beat. The
model was asked to hold the entire past performance library in its weights and
answer from memory. The relationship reasoning worked; the fact recall did not,
and that is why the architecture changed to retrieval afterwards.

Everything below is measured. Nothing here is projected.

## Run

| | |
|---|---|
| wall time | 1h 36m 19s (318 steps, 2 epochs) |
| cost | $0.13 · 0.743 kWh · 343 W mean |
| peak allocated | 19.99 GB of 31.84 GB |
| corpus | 2,532 train / 643 eval / 75 probes |
| trainable | 131.9M of 6.62B (1.99%) |
| train loss | 3.0119 → 0.1255 |
| eval loss | 0.3303 → 0.1526 |

## Eval curve

```
step    50  epoch  0.32  eval 0.3303  train 0.3165  gap  4.2%
step   100  epoch  0.63  eval 0.2498  train 0.2612  gap -4.5%
step   150  epoch  0.95  eval 0.2190  train 0.2045  gap  6.6%
step   200  epoch  1.26  eval 0.1880  train 0.1524  gap 19.0%
step   250  epoch  1.58  eval 0.1615  train 0.1364  gap 15.6%
step   300  epoch  1.89  eval 0.1529  train 0.1358  gap 11.2%
step   318  epoch  2.00  eval 0.1526  train 0.1255  gap 17.8%
```

## Task metrics, against a random-answer floor

Floor = the same grader scoring four partner names drawn at random, averaged
over five draws. Read every number against it, not against zero.

| metric | value | floor | lift |
|---|---|---|---|
| entity F1 (recall, n=56) | 0.325 | 0.006 | **53.8x** |
| entity F1 (relational, n=24) | 0.328 | 0.010 | **34.0x** |
| entity F1 (multihop, n=4) | 0.287 | 0.021 | 13.5x |
| precision@4 vs golden (n=36) | 0.114 | 0.019 | 6.2x |
| mean tier of picks (0–4) | 1.818 | 0.791 | 2.3x |
| nDCG@4 | 0.435 | 0.255 | 1.7x |
| picks covering the gap | 87.1% | 78.0% | 1.1x |
| hard negatives rejected | 11.7% | 0.0% | — |
| invented partner names | 0.00 /answer | — | — |
| answers that ran to completion | 95.5% | — | — |

**What worked.** The entity and relationship space, at 34–54x floor, and not one
invented company name across 120 answers. The model learned who the partners
are, what they do, and roughly how they relate.

**What did not.** Ranking is real but modest (precision@4 0.114). Hard-negative
rejection — the nuance thesis the whole corpus was built to test — fires 11.7%
of the time. And `picks covering the gap` is very nearly all floor: on a
150-partner roster, somebody among any four usually happens to hold the missing
capability, so 87.1% means almost nothing.

## Closed-book fact recall: why the architecture changed

Probes (75 held-out facts, every one of them stated in the training data):

| facet | recalled | answer space |
|---|---|---|
| CPARS rating | 12/17 (71%) | ~4 values |
| dollar value | 5/17 (29%) | wide |
| contract number | 4/22 (18%) | very wide |
| end year | 1/19 (5%) | wide |
| **total** | **22/75 (29.3%)** | |

Accuracy tracks the entropy of the answer space, not anything about the model's
grasp of the domain. Contract numbers came back **15 correct, 47 confidently
wrong, 13 absent** — for a contracting knowledge base, a confidently wrong
contract number is worse than no answer.

**Storage, not retrieval.** The obvious hypothesis was that the facts were in
the weights but bound to their training phrasings. Asking the *training*
questions back verbatim, for the same facets:

| facet | trained form | probe (reworded) |
|---|---|---|
| end_year | 27% | 5% |
| number | 25% | 18% |
| value | 40% | 33% |
| cpars | 47% | 71% |
| **total** | **35%** | **29%** |

35% against 29% is barely a difference. The facts are not stored at all, so
rephrasing was never the obstacle and more question diversity would not have
fixed it.

**Why the loss looked fine anyway.** Training loss is teacher-forced: at each
position the model predicts the next token given the *correct* prefix. In
`"CDC NCCDPHP … (75D30124C00000) ran 2024-2025"` almost every token follows from
the format and the contract name, and the handful of high-entropy digits barely
move the average. Free generation has no correct prefix to lean on. A loss of
0.1255 and a model that cannot reproduce its own training answers are entirely
consistent — which is exactly why eval loss made this run look finished, and why
the task metrics above are the ones that count.

## What this baseline is for

The next architecture keeps the part that worked and stops asking for the part
that did not: relationships reasoned by the model, facts supplied by retrieval.
The bar it has to clear:

- **entity F1 ≥ 0.33** on recall and relational — matching what closed-book
  already reached
- **hard negatives rejected ≫ 11.7%** — the metric the demo actually rests on
- **precision@4 > 0.114**, nDCG@4 > 0.435 against the same floor
- **exact values ≫ 29%** — retrieval should make this close to a solved problem,
  and anything less means the retrieval layer is not working
- **invented partner names still 0.00**

## Files

- `raw/` — metrics, timeline, gate decision, plan, run metadata, config, commit
- `grades/` — grades and generations for eval, probes, and the trained-form test
