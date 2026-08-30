# Working notes for agents

**Re-read this file after every context compaction, and before any training run,
benchmark run, or change to grading.** It is short on purpose. The things in it
have each already cost an hour or more of wasted GPU time or produced a wrong
published number, and none of them are guessable from the code.

---

## Posture: fix it when you find it

This project's results have twice been decided by a defect nobody was looking
for — a rejection marker matching a restated system prompt, and a checkpoint
schedule that never saved the best model. Both were three-line fixes sitting
under a passing test suite.

So: when you notice something wrong while doing something else, **fix it, add
the test that would have caught it, and say so** — do not file it away as out of
scope. The corollary matters just as much: when you fix a measurement bug,
**re-score the stored generations and correct any number you have already
reported.** Several figures in `benchmarks/*/RESULT.md` are wrong and were
repeated confidently for weeks.

Two habits that keep paying:

- **Prose by hand, facts by machine.** Hand-written training data is checked
  against the graph by tests (`tests/test_authored*.py`). Never assert a number
  in an answer that a test does not verify.
- **Verify with a second, independent signal** before reporting status. Nearly
  every monitoring failure here came from trusting one proxy.

---

## Before you trust a number

- **`precision@4`, `tier_hit_rate` and `mean_tier` are algebraically dependent.**
  `mean_tier == 4 x precision@4`, exactly, on every graded answer. Reporting them
  as separate corroborating evidence is double-counting. The independent signals
  are precision@4 and off-slate rate.
- **n=51 resolves almost nothing.** Paired minimum detectable effect is ~0.14;
  differences of 0.03-0.05 between arms are inside the noise. Report confidence
  intervals, or say plainly that the arms are indistinguishable.
- **Arm D is the null hypothesis and must stay in every table.** Ranking the
  slate by prior teaming count — one `Counter`, no model — scores ~0.50 on the
  43-question forecasting subset and beats every model arm. A result that beats
  arm C while losing to arm D means: use the groupby.
- **Blind-set labels are approximate.** Distractors are labelled tier 0 by fiat;
  recomputing with `tier_for` shows only ~22% are truly tier 0, ~28% are tier 1,
  ~47% tier 2, and ~3% are actual prior partners of that prime. Do not build
  preference training (DPO and similar) on these labels without fixing them.
- **`benchmarks/2026-08-28-real-3arm-v2/RESULT.md` is invalid.** Grader bug,
  mismatched denominators, and a since-widened retrieval record. Do not quote it.

---

## Before you start a training run

Full detail in `docs/ENGINEERING.md` (`## VRAM`, `## Long runs: failure modes
that hide`). The short version:

- **`max_seq_len` drops over-long rows; it does not truncate them.** `truncated`
  stays 0 while whole examples vanish. On this corpus every over-length row is
  hand-written. Measure with `ftlab.data.encode`, not
  `tokenizer.apply_chat_template` — the template undercounts by ~170 tokens.
- **This config sits at the edge of a 32 GB card.** 12B QLoRA OOMed at
  `max_seq_len` 3712 (steps 1 and 10) and at 3328 (step 148 of 268). A run that
  starts is not a run that fits: whether a step OOMs depends on which long row
  the shuffle hands it.
- **`expandable_segments:True` does nothing on Windows.** It is Linux-only and
  PyTorch falls back silently. The error message will recommend it anyway.
- **Never shrink `CONTEXT_K` to save memory.** It is sized so a 12-candidate
  slate plus its prime each get a record. Shrinking it degrades the prompt arms
  A and C are *measured* on while leaving B and D untouched — that biases the
  comparison rather than lowering a score. Spend training rows instead.
- `save_steps` must be a multiple of `eval_steps` (the config validator enforces
  this) or the best checkpoint lands on a step that is never written to disk.

## Launching and watching a long job

```bash
cmd > outputs/<run>-<date>.log 2>&1; code=$?; echo "EXIT=$code" >> outputs/<run>-<date>.log; exit $code
```

- Ending a launch in `tail` reports *tail's* exit status. A dead run once showed
  "completed, exit code 0" and sat unnoticed for 91 minutes.
- Use a **fresh log filename per attempt**. Reusing one path lets a follower
  replay the previous run's traceback into the new run's events.
- Pipe through `tr '\r' '\n'` before `grep`, or tqdm output is invisible.
- **Log silence is not death**, and thresholds are phase-dependent: training
  writes every ~25 s, batched generation writes nothing for ten minutes. Gate
  stall alarms on silence **and** an idle GPU, with two consecutive strikes.
- Use `nvidia-smi` for memory, not `torch.cuda.mem_get_info()` from another
  process — on Windows the latter reported 31.8 GB free while the card was
  31.9 GB used.
- Sample GPU utilisation more than once. A single reading caught during an
  optimizer step showed 12% on a run that was sustaining 97-99%.
- **Automate detection, not recovery.** These OOMs are deterministic; a blind
  retry burns the same hour again.

---

## Benchmark inference is slow, and it is the stack, not a bug

Measured on a 5090: **50 tok/s aggregate at batch 4, ~12 tok/s per sequence**
for 12B NF4. Arm C over 51 questions at a 2500-token cap takes **35 minutes**.
Budget accordingly, and do the arithmetic before launching — a 8192-token cap at
batch 2 projects to *4.7 hours*, which is how one run got killed at 21%.

- bitsandbytes NF4 dequantises on every forward. It is a training memory
  optimisation and is actively bad for decode. `use_cache` is already handled
  correctly in `infer.py`; do not go looking for that bug.
- Batch size is the cheap lever for inference. Training OOM caution does not
  transfer — there are no gradients or optimizer state. At an 8192-token cap,
  batch 2 used only 13 GB of 32; batch 4 is safe, batch 6+ risks the KV cache.
- **Sample rather than run all 51** when the quantity of interest is a
  distribution (e.g. output length). 17 questions answer it in a third of the
  time.
- Run inference under `python -u`. `generate_many` prints `[ftlab] generating
  N/51` per batch, but stdout block-buffers to a file, so an hour-long run looks
  completely dead in the log.
- For repeated benchmark cycles, the faster path already exists in this repo:
  GGUF export (`export_gguf.py`, and the ollama section of `docs/ENGINEERING.md`).
  Changing the inference stack changes the measurement, so do not mix its
  numbers with bitsandbytes ones — switch deliberately, for a whole cycle.

## Commands worth knowing

```bash
ftlab real-build                  # rebuild the corpus (blind.jsonl is sealed; only its context should change)
ftlab arms -c configs/real-3arm.yaml --adapter <ckpt> --out benchmarks/<date>
ftlab arms --arm d                # the rule baseline: no model, no GPU, seconds
.venv/Scripts/python.exe -m pytest -q
```

Arm D needs no GPU — run it first when sanity-checking a change to grading.
