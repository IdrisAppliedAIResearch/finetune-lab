# finetune-lab

Local LoRA / QLoRA fine-tuning for **Question–Reasoning–Answer (QRA) triples**, on a
single Windows GPU box. Config-driven, no cloud, no external logging.

The pipeline: `check-data` → `train` → `infer` → `merge` → `export` (GGUF for ollama).

---

## Verified environment

Everything below was confirmed working on this machine, not assumed:

| | |
|---|---|
| GPU | RTX 5090, 32 GB, **sm_120 (Blackwell)** |
| CPU / RAM | Ryzen 9 9950X3D, 61 GB |
| Python | 3.13.13 |
| torch | 2.11.0+**cu128** — sm_120 kernels present in the wheel |
| transformers / trl / peft | 5.16.1 / 1.12.0 / 0.20.0 |
| bitsandbytes | 0.50.2 — NF4 quantize/dequantize round-trips on GPU |

Blackwell is the fragile part. A CPU-only or pre-CUDA-12.8 torch wheel imports
cleanly, reports the GPU, and then dies at the first matmul with *"no kernel
image is available for execution on the device"*. `ftlab doctor` runs a real
matmul, a real backward pass, and a real 4-bit quantization, so it fails loudly
rather than three minutes into a run.

```bash
uv run ftlab doctor
```

---

## Quickstart

```bash
uv sync --extra dev
uv run ftlab doctor
uv run ftlab train -c smoke.yaml
```

`smoke.yaml` trains a 135M model for 6 steps on the bundled sample data. It
exists to prove the pipeline end to end in about a minute — run it after any
change to the data or training code, before committing GPU hours to a config
you have not exercised.

---

## Data format

One JSON object per line. `reasoning` and `meta` are optional:

```json
{"question": "...", "reasoning": "...", "answer": "...", "meta": {"domain": "math"}}
```

Loading is strict on purpose: a missing field, a blank field, or an unrecognised
key raises with the filename and line number. Silently accepting a typo like
`"anwser"` would mean training on a dataset one column smaller than you think.

### How a triple is rendered

`data.reasoning_format` controls the assistant turn:

| format | assistant text |
|---|---|
| `think_tags` (default) | `<think>\n{reasoning}\n</think>\n\n{answer}` |
| `labeled` | `Reasoning:\n{reasoning}\n\nAnswer:\n{answer}` |
| `answer_only` | `{answer}` — the ablation that tells you whether traces help |

The question is wrapped with the tokenizer's own chat template, so the model
sees exactly the format it was instruction-tuned on.

### What gets scored

Loss is masked everywhere the model should not be graded — the system prompt,
the question, and the template scaffolding. With `train_on_reasoning: false` the
trace stays in context but is excluded from the loss, so the model learns to
produce answers *given* reasoning rather than to produce reasoning.

A mask that is one token off is the most expensive bug in fine-tuning: the run
completes, the loss falls, and the model learns the wrong thing. Two defences:

```bash
uv run ftlab check-data -c smoke.yaml --samples 2
```

prints each example split into its **MASKED** and **SCORED** spans, so the
boundary can be checked by eye; and the tokenizer is never asked to encode the
joined string and slice it — segments are encoded independently and
concatenated, because a tokenizer is free to merge characters across a slice
point.

---

## The demo corpus

`ftlab synth` generates a synthetic public health contracting world and turns it
into QRA triples. The premise being tested: can a 12B model hold a firm's past
performance library and partner network in its weights and answer relational
questions about them — teaming, subs, primes, citations — **closed-book**.

```bash
uv run ftlab synth --out data/processed --seed 42 --scale demo
```

Produces `train.jsonl`, `eval.jsonl`, `eval_probes.jsonl`, `library.json`, and
`corpus_stats.json`. Everything derives from one seed, so a given seed always
yields the identical world, questions, and answers.

### Why the graph comes first

The tempting shortcut is to prompt an LLM for a past performance library, then
prompt it again for "the best teaming partners." That produces plausible golden
answers nobody can verify — and when the tuned model disagrees with one, you
cannot tell which is wrong.

So the order is inverted. A seeded entity graph is generated first (companies,
agencies, contracts, vehicles, capabilities, personnel, and the edges between
them). Golden answers are then **computed** from that graph by deterministic
scoring in `synth/scoring.py`. Prose is only ever wrapped around facts the graph
already fixed. No language model is asked what the right answer is.

### The relevance spectrum

Scoring produces graded tiers rather than a right/wrong split:

| Tier | Meaning |
|---|---|
| **4 — decisive** | Covers the full capability gap, right customer, strong joint history |
| **3 — strong** | Covers part of the gap, or covers it without the customer relationship |
| **2 — transferable** | Adjacent capability only; usable with ramp-up risk, and the trace says so |
| **1 — surface-only** | The trap: scores high on a salient signal, fails the decisive one |
| **0 — irrelevant** | No meaningful connection |

Tier 1 is the point. A partner with four joint awards and Very Good CPARS who
covers **none** of the capability gap is the most instructive example in the
set, and it only exists because the graph is under our control. Every
opportunity is validated at generation time to contain at least two such traps
and a genuine tail — a draft that lands in a corner of the graph without them is
resampled.

The reasoning traces reject those candidates by name, quoting what makes them
tempting before saying why it is not enough:

> Willowmere Health Analytics: 3 prior joint awards, most recent ending 2028,
> 2 of 3 rated Very Good or better. But covers none of the capability gap —
> adding them duplicates work we already self-perform.

### What the questions cover

Roughly half the corpus is recall and single-hop relational work. That ratio is
deliberate: closed-book means the model cannot reason about a relationship it
was never taught, and knowledge injected by fine-tuning needs the same fact
reached from several directions before it survives.

| Layer | Share | Examples |
|---|---|---|
| Recall | ~51% | contract records, partner profiles, agency portfolios, capability history |
| Relational | ~11% | teaming history, who holds which vehicle, prime/sub breakdown |
| Multi-hop | ~5% | partners combining a capability *and* an agency; who bridges two capabilities |
| Recommendation | ~33% | teaming, prime candidates, sub selection, PP citations, gap analysis, bid/no-bid |

### The split

Two different holdouts, because two different things are being measured:

- **Opportunities are held out whole.** Every recommendation question about a
  held-out pursuit goes to eval, none to train. Answering requires applying the
  reasoning to the library, not recalling a memorized pairing.
- **Recall questions split by paraphrase.** The facts must be in training, but
  one phrasing of each is held back, so eval measures retrievability through an
  unseen question rather than string memorization.

A final sweep drops any eval item whose question text also appears in training,
and a test asserts no question ever maps to two different answers — with 75
contracts, name collisions silently produce contradictory supervision that never
shows up in the loss curve.

`eval_probes.jsonl` asks for single exact values (contract numbers, dollar
figures, end years) — where parametric recall frays first. Two details make it a
fair measurement rather than a trick question:

- A probe is only built for a (contract, facet) pair whose short-form question
  was **not** trained. The fact is still taught, inside the full record the model
  saw many times; the terse pairing is not. So the probe measures retrieval, not
  recall of a memorized pair.
- Probe answers are short sentences in the same shape the terse training items
  use, not bare tokens, with the exact value carried separately in
  `meta.exact_value` for containment grading. Training answers average about a
  thousand characters; scoring against an eight-character target would mostly
  measure whether the model guessed the output format, and would report a format
  mismatch as lost knowledge.

### Prove it before the big run

```bash
uv run ftlab check-data -c qra-smoke.yaml --samples 1
uv run ftlab train      -c qra-smoke.yaml
```

`qra-smoke.yaml` runs the real corpus through a 135M model for 8 steps. The
model is far too small to hold the library — that is not the point. It exercises
rendering, the system prompt, masking, sequence lengths, and checkpointing on
the actual data in about a minute.

All names are fictional. Generated company names are checked against a blocklist
of real federal health contractors, because the corpus attaches invented CPARS
ratings and performance history to every name it mints.

---

## Commands

| command | purpose |
|---|---|
| `ftlab doctor` | GPU, CUDA kernels, bitsandbytes, package versions |
| `ftlab synth` | generate the synthetic past performance corpus |
| `ftlab plan` | project steps, tokens, VRAM, time and cost before training |
| `ftlab report` | re-render the metrics of a finished run |
| `ftlab grade` | score generated answers against the graph's ground truth |
| `ftlab show-config -c X` | print the fully resolved config after inheritance |
| `ftlab check-data -c X` | validate a dataset and display the loss mask |
| `ftlab train -c X` | train a LoRA adapter |
| `ftlab infer -c X -q "..."` | generate; `--base-only` for a before/after comparison |
| `ftlab merge -c X` | merge the adapter into base weights |
| `ftlab export -c X` | convert to GGUF and write an ollama Modelfile |

Any config value can be overridden without editing the file:

```bash
uv run ftlab train -c gemma4-12b-qlora.yaml --set train.epochs=2 --set lora.r=64
```

---

## Configs

`configs/base.yaml` holds the defaults; every other config declares
`extends: base.yaml` and overrides only what differs, so a diff between two
experiments is short enough to read.

- **`smoke.yaml`** — 135M model, 6 steps. Pipeline proof.
- **`gemma4-12b-qlora.yaml`** — QLoRA on `google/gemma-4-12B-it`, sized for 32 GB.
- **`qra-smoke.yaml`** — the real demo corpus through a 135M model, 8 steps.

Two things to know about the Gemma 4 preset:

1. **The repo is gated.** Accept the license on the model page, then
   `huggingface-cli login`, or the download 401s.
2. **It is multimodal.** `google/gemma-4-12B-it` is
   `Gemma4UnifiedForConditionalGeneration`, with vision and audio towers beside
   the text decoder. `AutoModelForCausalLM` loads it fine, but `all-linear` LoRA
   targeting would wrap adapters around encoder linears that a text-only QRA
   batch never activates. The preset's `lora.exclude_modules` keeps training on
   the language model.

---

## VRAM

With 32 GB, at 2K sequence length and batch size 1:

| model | mode | weights | headroom |
|---|---|---|---|
| ~8B | LoRA bf16 | ~16 GB | comfortable |
| ~12B | QLoRA nf4 | ~7 GB | large — room to raise `max_seq_len` |
| ~27B | QLoRA nf4 | ~15 GB | workable at 2K |

Activations, not weights, are what actually run you out of memory, and they
scale with sequence length. When you hit OOM, raise `train.grad_accum` rather
than lowering `per_device_batch_size` below 1 — the effective batch is the
product of the two, so the run stays comparable.

Windows keeps a desktop composition buffer on the GPU. `ftlab doctor` reports
free VRAM, not total; check it before a large run.

---

## Exporting to ollama

`merge` reloads the base in bf16 rather than reusing the 4-bit training model —
merging into dequantized weights would bake quantization error in, and then GGUF
quantizes again. One lossy step is enough.

```bash
git clone https://github.com/ggml-org/llama.cpp %USERPROFILE%\llama.cpp
uv run ftlab merge  -c gemma4-12b-qlora.yaml
uv run ftlab export -c gemma4-12b-qlora.yaml --quant q4_k_m --ollama-name gemma4-qra
```

`export` shells out to llama.cpp rather than vendoring a converter, which would
go stale within weeks of a new architecture landing. Point it with
`--llama-cpp <path>` or `LLAMA_CPP_DIR`. Quantizing needs `llama-quantize`
built (`cmake -B build && cmake --build build --config Release`); without it you
can still serve the f16 GGUF.

---

## Did it actually work?

Eval loss says a run converged. It does not say whether the model recommends the
right partners, and on this corpus that is the only question worth asking.

```bash
uv run ftlab grade -c gemma4-12b-qlora.yaml --split both --out outputs/tuned
uv run ftlab grade -c gemma4-12b-qlora.yaml --split both --base-only --out outputs/base
uv run ftlab grade --compare outputs/base/grades.json outputs/tuned/grades.json
```

Generation is batched (`--batch-size`, default 8) with left padding — with right
padding the pad tokens sit between prompt and completion, so slicing the prompt
off by length returns padding. Generations are saved to `generations.jsonl`, so
`--generations <file>` re-grades without paying for inference twice.

Because the golden answers were computed from a graph we still hold, grading is
deterministic — no LLM judge, no embedding similarity, no human pass. Every
company, contract number and contract name in the world is known, so finding
which appear in an answer, and in what order, is exact.

| Metric | What it tells you |
|---|---|
| **hard negatives recommended** | **The thesis.** Partners that look right and are not. Read this first. |
| hard negatives rejected | Whether the model names and dismisses them, as the traces teach |
| precision@4 vs golden | Overlap with the deterministic ranking |
| mean tier of picks | Average quality of what it named, 0–4 |
| nDCG@4 | Whether it got the *order* right, not just the set |
| picks covering the gap | Whether its partners can do the work we can't |
| invented partner names | Closed-book models fabricate. Every real name ends in a known suffix, so a name-shaped phrase absent from the library is unambiguously invented. |
| exact value present | Probe recall — contract numbers, dollar figures, end years |
| entity F1 | Recall and relational layers: did the right facts come back |
| answers naming anything | Guards the headline: a model that names nobody scores a perfect 0.00 on traps recommended |
| answers that ran to completion | Guards the rest: a generation cut off before the rejection block is indistinguishable from one that recommended the traps |

The last two exist because the headline metric is gameable by silence. Running
the grader against a deliberately undertrained model returned `hard negatives
recommended: 0.00` — which reads as a flawless score and actually meant the model
named no library entity at all. The report now prints a warning instead of
letting that pass:

```
  hard negatives recommended       0.00 per answer
  answers naming anything           0.0%
  !! only 0% of answers named any known entity, so the
     ranking metrics above describe an empty output rather than a
     judgement -- a model that names nobody scores a perfect 0.00 on
     hard negatives recommended.
```

One subtlety drives the parsing. Training answers name their rejected candidates
out loud under a "Not recommended" heading, so counting entities across the whole
answer would score a *correct rejection* as a bad recommendation. The answer is
split at that heading: a trap in the first half is a failure, a trap in the
second is exactly right.

That also sets the generation budget. Recommendation targets run to ~730 tokens
at p50 and ~1180 at p99, so `--max-new-tokens` defaults to 1280. A smaller budget
truncates the answer before its rejection block and quietly tanks the very metric
you came for — which is why completeness is reported alongside it.

### The grader is itself calibrated

A scorer is only worth its output if it can tell a right answer from a wrong one,
so two tests pin it in place. Replaying the golden answers must score at the
ceiling, and a model that recommends precisely the hard negatives must score at
the floor:

| | oracle | adversary |
|---|---|---|
| precision@4 | **1.000** | 0.028 |
| hard negatives recommended | **0.00** | 3.48 |
| hard negatives rejected | **100%** | 0% |
| mean tier of picks | 3.66 | 0.67 |
| entity F1 | **1.000** | 0.000 |
| probe exact value | **100%** | 0% |
| invented names | **0.00** | 1.00 |

Every subtle bug in this module surfaced first as an oracle that couldn't reach
1.0 — normalising nDCG against the top-k by score rather than the best available
(tier isn't monotonic in score, so it exceeded 1.0); grading filtered archetypes
against the raw ranking; and treating vehicle names like "CDC Public Health
Analytics IDIQ" as invented firms. If you change the corpus, watch those two
tests before you trust any number the grader prints.

---

## Cost and metrics

### Before the run

```bash
uv run ftlab plan -c gemma4-12b-qlora.yaml --calibrate 8
```

Projects the whole job — steps, epochs, tokens, wall time, peak VRAM, energy and
dollars — by running eight real optimizer steps and extrapolating. Three things
about how it extrapolates:

- **Training time comes from seconds/step, not tokens/second.** The reverse
  looked more principled and was measurably wrong: calibration deliberately runs
  the longest examples, so it clocks a high token rate at an ordinary step time,
  while real mixed-length steps cost the same wall time carrying fewer tokens.
  Against a real 1,266-step run the token-rate model under-predicted by **2.3x**;
  step time predicted it within 9%, erring long.
- **Peak VRAM is measured on the longest examples**, not a random sample. With
  dynamic padding a short calibration easily misses the batch that would have
  OOMed. Note that reserved memory still creeps up over a long run as the
  allocator fragments — that same run reserved 12% more than the 8-step
  calibration predicted — so the verdict treats anything under ~3 GB spare as
  thin.
- **Evaluation is measured separately and priced in.** It's forward-only and
  several times faster than a training step, so it cannot be projected from the
  training rate. The plan warns when eval dominates: on one config here,
  `eval_steps: 8` meant 474 passes over the eval set and **96% of the run** was
  evaluation.

```
Calibration (8 real steps, worst-case batches)
  sec / step             0.34
  peak reserved          5.71 GB
  device total           31.84 GB
  verdict                fits, 26.1 GB spare

Projection
  training               7m 09s
  evaluation             29.1s (3 passes over 643 examples)
  wall time              7m 38s        <- measured actual: 6m 59s
```

The plan also reports **verbatim exposures** — how many times the model sees a
given answer across the run, repetition times epochs. The corpus repeats each
fact by design, so at 3 epochs the busiest contract record is seen 21 times. That
is the number to look at before raising `epochs`, not the example count.

### During and after

Every run writes four files beside the adapter:

| file | contents |
|---|---|
| `metrics.json` | throughput, memory, power, energy, cost, loss endpoints |
| `trainer_metrics.json` | the HF Trainer's own metrics dict |
| `metrics_timeline.jsonl` | per-log step, epoch, loss, LR, elapsed, GPU watts |
| `metrics_report.txt` | the same summary as printed |
| `run_meta.json` | full config, library versions, token stats, parameter counts |

```bash
uv run ftlab report --run outputs/gemma4-12b-qra
uv run tensorboard --logdir outputs
```

Power comes from `nvidia-smi` sampled on a background thread (5s default), and
the mean is taken over samples where the GPU was actually under load — idle
gaps between steps would otherwise drag it below the real draw. Memory comes
from torch's allocator counters; tokens from the collator that built the
batches, so padding is counted because padding costs the same compute.

### What's measured and what's assumed

Only two numbers in the cost report are inputs rather than measurements, and
both are labelled as such in the output:

```yaml
metrics:
  electricity_usd_per_kwh: 0.17   # your utility rate
  system_overhead_watts: 120      # CPU/RAM/fans; a constant, not metered
  cloud_usd_per_hour: 0.0         # 0 disables the rented-GPU comparison
```

`cloud_usd_per_hour` defaults to zero deliberately — a wrong rented-GPU price is
worse than no price. Set it if you want the comparison. If `nvidia-smi` returns
no samples the cost block says so rather than reporting energy derived from an
unmeasured zero.

---

## Windows notes

These are settled in the defaults; listed so the reasoning is not lost:

- **`attn_implementation: sdpa`.** Flash-attention has no official Windows
  wheels. PyTorch's fused SDPA covers most of the gain.
- **`dataloader_num_workers: 0`.** Windows spawns rather than forks, so each
  worker re-imports the training module — slow, and a reliable source of
  confusing crashes.
- **No unsloth / vLLM.** Both are Linux-first. If you later want unsloth's ~2x
  throughput, WSL2 is the move; nothing here assumes Windows except these
  defaults.

## Tests

```bash
uv run pytest
```

106 tests, no GPU or network needed — a char-level fake tokenizer stands in, so
that a masking failure is a real bug rather than a tokenizer artefact.
