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

| | |
|---|---|
| corpus | 401 train / 70 eval / 18 blind |
| tokens | mean **620**, p95 1,519, max **2,014** |
| `max_seq_len` | **2048** — measured, nothing truncated |
| schedule | 52 steps (2 epochs, effective batch 16) |
| **wall time** | **~14 min** |
| **cost** | **~$0.02** electricity |

At 2048 the peak-allocated figure from the calibrated run was 19.99 GB of 31.84 —
comfortable, no spill.

**One honest caveat:** 401 examples is thin for arm B. Closed-book recall of a
1,793-company graph from ~160 closed-book examples is a big ask, and the earlier
run showed high-entropy facts need many exposures. Recommend generating **3–4
paraphrases per question** (~1,500 examples, ~50 min, still under $0.10) before
concluding anything about arm B. The `epochs` gate in `ftlab gate` applies as
before.

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
| **blind precision@5** | on sealed-period slates | 5/12 ≈ 0.42 |

**Read hard-negative rejection first.** It floors at exactly zero — no
context-reading heuristic rejects anything — and it is the one number that
distinguishes reasoning from lookup.

**Do not lead with nDCG@4.** Random picking within a retrieved slate already
scores 0.69 of the ceiling.

### Cost, measured not estimated

| | fine-tuned (A/B) | base + RAG (C) |
|---|---|---|
| training | ~14 min, $0.02 | none |
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
