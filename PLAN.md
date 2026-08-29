# Three-arm benchmark: fine-tuned model vs. deterministic architecture

A small, real-data experiment sized for a conversation, not an MVP. The question
it answers: **does fine-tuning a local model buy reasoning that a base model plus
a deterministic retrieval architecture does not already have?**

Everything below is built and measured. Numbers are from the current corpus.

---

## Why the previous corpus was thrown away

The synthetic corpus could not answer the question. Its golden answers were the
output of a deterministic scoring function, and the grader scored against that
same function — so a rule engine implementing it scored **1.000 by construction**
(that is what `test_oracle_scores_near_perfect` measures). The fine-tuned model's
best possible result was a tie. It was a distillation benchmark wearing the
clothes of a reasoning benchmark.

Real data fixes this because **the labels belong to nobody**. Which companies
actually subcontracted on which award is historical fact. All three arms can lose.

---

## The corpus

Source: [USAspending.gov](https://api.usaspending.gov) — public API, no auth,
public domain. HHS contract awards and the subcontracts reported against them.

| | |
|---|---|
| companies | 1,793 |
| prime awards | 1,301 |
| subcontracts | 3,327 (2015, 2024, 2025) |
| teaming pairs, training period | 1,681 (397 repeat) |
| **teaming pairs, blind period** | **571, of which 407 (71%) never appear in training** |
| primes with ≥3 subcontractors | 132 |

**Split:** subcontracts dated on or before **2025-06-30** may be used to build
training data. Everything after is sealed.

### Question archetypes (7, generating 471 questions)

| archetype | n | what it tests |
|---|---|---|
| `prior_relationship` | 120 | has A ever teamed with B — catches confabulation |
| `sub_candidates` | 79 | **rank a mixed slate** — the core ranking task |
| `team_composition` | 74 | who a prime actually puts on its teams |
| `prime_candidates` | 60 | which primes actually subcontract this work |
| `portfolio` | 60 | what a company actually does (closed-book knowledge) |
| `repeat_partners` | 41 | who got brought back — the only performance proxy available |
| `warm_intro` | 37 | two-hop: who in X's network could open a door at agency Y |

**Relevance spectrum**, observed rather than assigned:

```
tier 4  actually subbed for this prime at this agency
tier 3  actually subbed for this prime, different agency
tier 2  subbed at this agency for someone else
tier 1  HARD NEGATIVE - matches NAICS and HHS scale, zero relationships
tier 0  no meaningful overlap
```

All 79 ranking questions carry hard negatives. **Tier 1 is the experiment.** A
rule engine ranks on structured fields, and NAICS is nearly useless here: code
541690 returned both Lockheed's Apache targeting sights and CDC surveillance work
in the same query. A tier-1 company looks correct on every field and wrong on
every relationship.

### Blind holdout — 18 questions

Generated **from the sealed period only**, after the training corpus was frozen.
Context and reasoning come from training-period history; the answer key comes
from what happened after.

- `blind_next_team` (10) — a 12-company slate, 5 of whom the prime went on to use
- `blind_new_entrant` (8) — who started at an agency with no prior work there

Two leaks were found and fixed while building it: the positives were the
alphabetical head of an alphabetically-ordered slate (so "name items 1–5" scored
perfectly), and slate order is now shuffled deterministically — positive
positions now average 6.6 of 12, against 6.5 for uniform.

---

## The three arms

| arm | model | retrieval | what it isolates |
|---|---|---|---|
| **A** | fine-tuned | yes | the full proposition |
| **B** | fine-tuned | **no** | did training put the graph in the weights? |
| **C** | base | yes | what retrieval alone already gets you |

Arm C is the one to beat. If A ≈ C, fine-tuning bought formatting.

**Arm B requires context dropout.** 40% of training examples have their records
withheld, so the same adapter learns both modes. Without it, arm B meets an
unseen prompt shape and fails for a reason unrelated to what it knows.

---

## Training plan

`configs/real-3arm.yaml`, extending the verified Gemma 4 12B QLoRA preset.

| | planned | **measured** |
|---|---|---|
| corpus | 1,335 train / 231 eval / 18 blind | same (4 phrasings per fact) |
| tokens | mean 656 | **mean 1,106, max 2,395** |
| `max_seq_len` | 2048 | **3072** — needed once every slate candidate got a record |
| schedule | 168 steps (2 epochs) | same |
| wall time | ~39 min | **1h 05m** (23.3 s/step) |
| cost | ~$0.06 | **$0.10** |
| peak allocated | — | **22.01 GB** of 31.84 |

The split is on the **fact**, not the row: paraphrases of one fact all land on
the same side, so the eval set is not asking about facts the model was taught
outright in different words. `corpus_stats.json` reports `facts_in_both`,
currently **0**.

### What the first attempt cost, and what it bought

The corpus was built twice. The first version leaked: `context_for` assembled
the retrieved records from the question's own answer key, so on the blind set
**eight of eight records supplied were gold** and an untuned base model scored a
perfect **1.000** by reading the names back. Comparing the two runs is the
clearest evidence the fix took:

| | leaked corpus | leak-free corpus |
|---|---|---|
| final eval loss | 0.2799 | **0.3546** |
| what epoch 2 bought | −0.10% | **+5.55%** |
| gate on "still learning" | FAIL | PASS |

A *worse* loss is the good outcome here — the task got genuinely harder once the
answer stopped being handed over. And where the leaked run's second epoch bought
nothing at all, this one is still learning at the end.

### Gate verdict

```
[PASS] still learning      +5.55% (need >= 0.50%)
[FAIL] not memorising      eval 0.3546 vs train 0.1211 = 65.8% above (limit 10%)
[PASS] still at the floor
```

Split, and it says so: *the gap tells you what the model is fitting, not whether
more exposure would fix it.* Eval loss is a proxy; the arms table decides.

---

---

## Result — the blind set says no

18 held-out questions, answers from the sealed period. **Every arm scores at or
below the random floor.**

| archetype | n | **floor** | arm C base+RAG | arm B tuned | arm A tuned+RAG |
|---|---|---|---|---|---|
| `blind_next_team` | 10 | **0.422** | 0.156 | 0.200 | 0.400 |
| `blind_new_entrant` | 8 | **0.359** | 0.357 | **0.000** | **0.000** |

Arm A comes closest on the archetype nearest its training (0.400 against a 0.422
floor) — that is chance, not skill. On the archetype it never saw in training it
scores **zero**, where picking at random scores 0.36.

### Why: the fine-tune learned formats, not the task

| | answers forced into a training template |
|---|---|
| arm C (base) | **0 / 18** |
| arm B (tuned, no RAG) | 13 / 18 |
| **arm A (tuned + RAG)** | **18 / 18** |

Asked *"which of these companies are new to NIH?"*, arm A answered

> *Primes to approach for NIH Custom Computer Programming Services work, by
> subcontracting volume: 1. RESEARCH TRIANGLE INSTITUTE…*

— a fluent, well-formed answer to a question nobody asked, in the shape of the
`prime_candidates` archetype. The untuned base model instead worked the actual
problem: identify which companies have records, filter for NIH, check for prior
work.

Training on seven answer templates taught the model to emit one of seven answer
templates. That is the finding.

### What it does and does not license

**It does say:** on this corpus, at this scale, fine-tuning did not beat a base
model with retrieval, and it destroyed generalisation to question types held out
of training. The blind set was built to expose exactly this and it did.

**It does not say** fine-tuning cannot work here. Five things bound the claim:

1. **n = 18** (10 and 8 per archetype). Differences of a few points are noise.
2. **401 training facts.** Very little to fine-tune on.
3. **Seven archetypes** is a narrow instruction diet, and template collapse is
   the known failure of narrow diets — mixing in general instruction data, or
   more archetypes, is the standard fix and was not tried.
4. **The model is over-trained by its own gate** (train/eval gap 65.8%). One
   epoch, or an early-stopped checkpoint, was not evaluated.
5. **The blind task is hard on purpose** — forecasting who a prime will hire
   next, where 71% of the true pairings never appear in training.

The cheapest next experiments, in order: re-run arms at one epoch; add general
instruction data to the mixture; widen the archetype set. Each is under an hour.

---

## Metrics, each against its floor

Reporting a metric without its floor is how the last corpus produced a "6.2x"
result that was really 1.4x. Every number is scored against **random selection
from the same slate**.

| metric | what it means | floor |
|---|---|---|
| **precision@4** | of 4 picks, how many are tier 3–4 | ~0.21 (random-in-slate) |
| **hard negatives recommended** | tier-1 picks per answer — *lower is better* | — |
| **hard-negative rejection** | traps explicitly named as rejected | **0.00** |
| entity F1 | named entities vs. the answer key | ~0.01 |
| nDCG@4 | graded ranking quality | ~0.69 — **weak, do not lead with it** |
| invented company names | names not in the library, per answer | 0.00 |
| **blind precision@4** | on sealed-period slates | 0.228 (measured) |

**Read hard-negative rejection first.** It floors at exactly zero — no
context-reading heuristic rejects anything — and it is the one number that
distinguishes reasoning from lookup.

**Do not lead with nDCG@4.** Random picking within a retrieved slate already
scores 0.69 of the ceiling.

### Cost, measured not estimated

| | fine-tuned (A/B) | base + RAG (C) |
|---|---|---|
| training | ~39 min, $0.06 (one-off) | none |
| corpus build | minutes, CPU | same |
| per query | one 12B forward pass | one 12B forward pass + BM25 |
| engineering | data generation | retrieval + ranking rules per question type |

The honest cost argument is not dollars — both are cheap. It is that the
deterministic arm needs a rule written for every question type, and the model
arm generalizes to phrasings nobody anticipated. That is measurable here: the
blind set contains question forms the rules were not written against.

---

## Limits, stated up front

1. **Subaward reporting is incomplete.** FSRS compliance is uneven, so a company
   absent from an award's list may still have worked it. Precision is
   trustworthy; **recall is a lower bound**; tier 0/1 means "no reported
   relationship", not "no relationship".
2. **CPARS is unobtainable** — FOIA-exempt source-selection information. Repeat
   teaming is the substitute, and it is revealed preference, not a rating.
3. **46% of subcontract descriptions carry no content** (median 49 characters;
   many are `SUBCONTRACT <id> MOD <n>`). The model's text-reading advantage
   exists only on the rest — and that cuts toward the rule engine.
4. **Real data is dirty.** A $6.17bn "subcontract" appeared in the first page of
   results. Cleaning choices affect all three arms and were made before any
   answer key was built.
5. **Real company names.** Fine internally. For anything customer-facing,
   pseudonymize consistently — a demo that says "don't team with [real firm]" is
   a reputational problem, and the experiment is unaffected by renaming.

---

## What to run

```bash
uv run ftlab real-build                              # corpus from the cached slice
uv run ftlab train -c real-3arm.yaml                 # ~14 min
uv run ftlab grade -c real-3arm.yaml --split blind --arm a
uv run ftlab grade -c real-3arm.yaml --split blind --arm b
uv run ftlab grade -c real-3arm.yaml --split blind --arm c
uv run ftlab grade --compare outputs/arm-c/grades.json outputs/arm-a/grades.json
```

The result that would kill the idea: **arm A ≈ arm C on hard-negative
rejection.** That would say retrieval is doing the work and the fine-tune is
decoration. The result that would sell it: A > C on rejection *and* B > C on the
closed-book subset, meaning the graph is genuinely in the weights.
