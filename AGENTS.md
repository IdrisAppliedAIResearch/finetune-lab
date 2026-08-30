# Working notes for agents

**Re-read this file after every context compaction, and before any training run,
benchmark run, or change to grading.**

---

## The question this repo exists to answer

> Can a small model be fine-tuned to reason about contract relationships?

That is the whole scope. Not a ranking leaderboard, not a general benchmark. If
a change does not help answer that question, it does not belong here.

The repo was ~16,000 lines and is now ~6,000, because most of it was machinery
serving a generated corpus that has been deleted. Do not rebuild it. Read the
next section before proposing anything that generates training data.

---

## Why the generated corpus was deleted

It produced every recurring problem the project had, and the evidence is worth
keeping so nobody re-derives it:

- **Its answers were the output of a scoring function.** An archetype predicted
  its own answer format perfectly, so the first fine-tune collapsed into seven
  templates — 18 of 18 held-out answers took one of them.
- **Every gold answer it could produce was an observed relationship in the
  training graph.** That teaches exactly one rule: *the answer is a firm this
  prime has already hired.* On held-out questions where no correct answer was a
  prior partner, the fine-tuned model scored **0.091 against a 0.369 random
  floor** — worse than the untuned base model, and worse than guessing. It
  wasn't failing to reason; it was applying a rule the data had drilled into it.
- **Its blind slates labelled every distractor tier 0 by fiat.** Recomputed with
  `tier_for`, only ~22% of those labels were right.
- **It swamped the hand-written data.** 1,900 templated rows against 40 authored
  ones, and only 15% of training was slate-shaped against a 100%-slate test.

The hand-written examples are the part that measurably worked: off-slate naming
0.275 → 0.098, trap rejection 0.378 against the base model's 0.000, 51/51
questions answered, 0% truncated.

**So: write examples by hand, keep the set small enough to read, and verify
every factual claim with a test.** That is the method here.

---

## Posture: fix it when you find it

Two results in this project have been decided by defects nobody was looking for
— a rejection marker that matched a restated system prompt, and a checkpoint
schedule that never saved the best model. Both were three-line fixes under a
passing test suite.

When you notice something wrong while doing something else, **fix it, add the
test that would have caught it, and say so.** And when you fix a *measurement*
bug, **re-score the stored generations and correct any number already
reported** — several figures in `benchmarks/` were wrong and repeated
confidently for weeks.

**Prose by hand, facts by machine.** Every number in a hand-written answer is
asserted as a structured fact and checked against the graph by
`tests/test_authored*.py`. Never write a figure an answer states that a test
does not verify. Never write a figure that is not in the retrieved context — the
grader reads company names, not numbers, so a model taught to invent them scores
exactly as well as one that doesn't.

---

## Before you trust a number

- **Small n.** Differences under ~0.14 at n=51 are noise. Report confidence
  intervals or say the arms are indistinguishable. The hand-written test set is
  smaller still: it can show *large* effects — "does it reason at all" — and
  nothing finer. That is the right target for the question above.
- **Arm D is the null hypothesis and belongs in every table.** It ranks the
  slate by prior-teaming count with no model at all, in seconds, no GPU. If arm
  D scores near the ceiling on a test set, that set is rule-solvable and should
  be thrown out. If a fine-tune beats arm C but loses to arm D, the answer is:
  use the groupby.
- **`precision@4`, `tier_hit_rate` and `mean_tier` are algebraically dependent**
  (`mean_tier == 4 × precision@4`, exact on every graded answer). Reporting them
  side by side is double-counting.
- `benchmarks/2026-08-28-real-3arm-v2/RESULT.md` is **invalid** — grader bug,
  mismatched denominators, and a since-widened retrieval record. Do not quote it.

---

## Before you start a training run

Detail in `docs/ENGINEERING.md`. The short version:

- **`max_seq_len` drops over-long rows; it does not truncate them.** `truncated`
  stays 0 while whole examples vanish. Measure with `ftlab.data.encode`, not
  `tokenizer.apply_chat_template` — the template undercounts by ~170 tokens.
- **12B QLoRA sits at the edge of a 32 GB card.** It OOMed at `max_seq_len` 3712
  (steps 1 and 10) and 3328 (step 148 of 268). A run that starts is not a run
  that fits — whether a step OOMs depends on which long row the shuffle hands it.
- **`expandable_segments:True` does nothing on Windows.** Linux-only; PyTorch
  falls back silently. The error message recommends it anyway.
- `save_steps` must be a multiple of `eval_steps` (the config validator enforces
  it) or the best checkpoint lands on a step that is never written.
- **Retrieval context must cover every candidate.** A 12-candidate slate plus its
  prime needs 13 records. Shrinking that to save memory degrades the prompt the
  retrieval arms are *measured* on while leaving the others untouched — it
  biases the comparison rather than lowering a score. Spend training rows first.

## Launching and watching a long job

```bash
cmd > outputs/<run>-<date>.log 2>&1; code=$?; echo "EXIT=$code" >> outputs/<run>-<date>.log; exit $code
```

- Ending a launch in `tail` reports *tail's* exit status. A dead run once showed
  "completed, exit code 0" and sat unnoticed for 91 minutes.
- **Fresh log filename per attempt** — reusing one lets a follower replay the
  previous run's traceback into the new run's events.
- Pipe through `tr '\r' '\n'` before `grep`, or tqdm output is invisible.
- Run inference under `python -u`; stdout block-buffers to a file and an
  hour-long run looks dead.
- **Log silence is not death**, and thresholds are phase-dependent: training
  writes every ~25 s, batched generation writes nothing for ten minutes. Gate
  stall alarms on silence **and** an idle GPU, two consecutive strikes.
- Use `nvidia-smi` for memory; `torch.cuda.mem_get_info()` from another process
  on Windows reported 31.8 GB free while the card was 31.9 GB used.
- Sample GPU utilisation more than once — a single reading during an optimizer
  step showed 12% on a run sustaining 97-99%.
- **Automate detection, not recovery.** These OOMs are deterministic; a blind
  retry burns the same hour again.

## Inference is slow, and it is the stack, not a bug

Measured on a 5090: **50 tok/s aggregate at batch 4, ~12 tok/s per sequence** for
12B NF4. bitsandbytes dequantises on every forward — a training memory
optimisation that is bad for decode. `use_cache` is already handled correctly in
`infer.py`; don't go looking for that bug. Do the arithmetic before launching: a
8192-token cap at batch 2 projects to 4.7 hours, which is how one run got killed
at 21%. Batch size is the cheap lever — training OOM caution does not transfer,
since inference has no gradients or optimizer state.

---

## Commands

```bash
ftlab real-build                 # build the corpus from hand-written examples
ftlab train -c configs/real-3arm.yaml
ftlab arms -c configs/real-3arm.yaml --adapter <ckpt> --out benchmarks/<date>
ftlab arms --arm d               # the rule baseline: no model, no GPU, seconds
.venv/Scripts/python.exe -m pytest -q
```

## Current state

The corpus is 51 distinct hand-written examples (~153 rows at 3× repeat), with 8
examples held out for eval. Eval used to contain **zero** hand-written rows, so
eval loss measured template reproduction and steered checkpoint selection; the
split is now over authored examples.

**Open:** the test set. `blind.jsonl` was produced by the deleted generator and
is stale — it must be replaced by hand-written questions whose answers are
derivable from the supplied records, with arm D run against them as the
rule-solvability check.
