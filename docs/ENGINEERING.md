# Engineering notes

How the training tool itself works: config, masking, memory, grading, export.
For what the project *found*, start at the [README](../README.md).

Local LoRA / QLoRA fine-tuning for **Question-Reasoning-Answer (QRA) triples**, on a
single Windows GPU box. Config-driven, no cloud, no external logging.

> The project started as a synthetic-corpus experiment and became a three-arm
> benchmark on **real** federal contracting data. Much of this document describes
> the original synthetic pipeline, which still works and still runs.
>
> * **[PLAN.md](../PLAN.md)** - the experiment: three arms, metrics, and limits
> * **[the result](../benchmarks/2026-08-28-real-3arm-v2/RESULT.md)**
> * **[benchmarks/](../benchmarks/)** - every run kept with its generations,
>   including the ones that were wrong and why

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
| `native` | reasoning passed as a separate `reasoning` message field, placed by the model's own template |

The question is wrapped with the tokenizer's own chat template, so the model
sees exactly the format it was instruction-tuned on.

### Reasoning models need `native`

The first three formats embed the trace in the message *content*. That breaks on
a model whose template owns the thinking span — and Gemma 4 is one. Verified
against its real template:

```
prompt (add_generation_prompt=True):
  ...<|turn>model  <|channel>thought  <channel|>        <- empty, CLOSED channel

full conversation:
  ...<|turn>model  <|channel>thought  REASONING  <channel|>ANSWER<turn|>
```

The prompt is **not a prefix** of the conversation, so the label boundary cannot
be derived, and `ftlab` refuses to train rather than guess. The fix is two
settings, both load-bearing — changing either alone leaves masking broken:

```yaml
data:
  reasoning_format: native
  chat_template_kwargs: { enable_thinking: true }
```

`enable_thinking` moves the marker up into the system turn so the two renderings
agree; `native` hands the trace to the template as its own field. The result
masks exactly right — question and system prompt masked, the model's own thought
channel and answer scored, including the closing turn marker that teaches it to
stop.

`chat_template_kwargs` is applied identically to training and inference through
one shared prompt renderer. Passing it on one side only would hand the model a
format at inference that it never saw in training.

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
| `ftlab inspect-model` | read a checkpoint's modules and check a config against it |
| `ftlab show-config -c X` | print the fully resolved config after inheritance |
| `ftlab check-data -c X` | validate a dataset and display the loss mask |
| `ftlab train -c X` | train a LoRA adapter; `--resume-adapter` continues an existing one |
| `ftlab gate -c X` | decide whether a finished run has earned another epoch |
| `ftlab retrieve -q "..."` | search the library (BM25 + exact names) |
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

### Check the config against the checkpoint

```bash
uv run ftlab inspect-model -c gemma4-12b-qlora.yaml
```

Reads the safetensors header and config — no weights loaded, so a 23GB
checkpoint inspects in about a second — and validates the parts of a training
config that depend on the model. It exists because of a bug it would have
caught: the Gemma 4 preset shipped with

```yaml
exclude_modules: [vision_tower, audio_tower, multi_modal_projector, ...]
```

guessed from the config's sub-config keys. Gemma 4's real modules are
`vision_embedder`, `embed_vision` and `embed_audio`. **Not one guess matched**,
so the exclusion did nothing while the comment above it claimed the perception
towers were protected. PEFT does not warn when an exclude pattern matches zero
modules, so nothing anywhere would have told you.

Verified against `google/gemma-4-12B` (11.96B params, 677 tensors):

| | |
|---|---|
| linears inside `model.language_model` | 328 |
| linears outside it | 3 — `vision_embedder.patch_dense`, `embed_vision.embedding_projection`, `embed_audio.embedding_projection` |
| weights | 22.3 GB bf16 / **~6.1 GB nf4** |
| attention | interleaved — layers 5, 11, 17 … 47 have doubled `q_proj`, quartered `k_proj`, and **no `v_proj`** |

That last row is why `target_modules: auto` is the right default: hand-listing
projections would silently miss eight layers' worth of shape variation.

### Two things to know about the Gemma 4 preset:

1. **The repo is gated, and it must be the `-it` variant.** Accept the license
   on the model page, then `huggingface-cli login`. The base model
   (`google/gemma-4-12B`, mirrored by `unsloth/gemma-4-12b`) ships **no chat
   template at all** — `ftlab` refuses to load a tokenizer without one rather
   than inventing a turn format the model never saw. Note also that GGUF
   variants are inference-only: QLoRA needs safetensors.
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
| ~12B | QLoRA nf4 | **19.5 GB peak allocated** (measured, 2K ctx) | comfortable — 12.4 GB spare |
| ~27B | QLoRA nf4 | ~15 GB | workable at 2K |

Weights are a small part of the story. Gemma 4 12B in NF4 is only **6.1 GB of
weights**, but a measured calibration at 2K context peaks at **19.5 GB
allocated** — the rest is activations plus, notably, the output layer: a 262K
vocabulary at 2048 positions costs about 5 GB on its own once you count the
logits, their fp32 upcast, and their gradient.

Reserved memory ran a further 9.6 GB above allocated (29.1 GB held for 19.5 GB
used) — allocator caching, not demand. `ftlab plan` reports both, and judges the
verdict on **allocated**: when an allocation does not fit its cached blocks
PyTorch frees them and retries before it OOMs, so reserved is not a failure
point. An earlier version keyed the verdict on reserved and called a run with
12.4 GB of real headroom "thin".

Two Windows notes, both measured rather than assumed:

- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` does nothing here.** It is
  the standard fix for exactly this fragmentation, and it is Linux-only —
  PyTorch warns `expandable_segments not supported on this platform` and falls
  back to the native allocator. Re-calibrating with it set produced byte-identical
  figures.
- Reserved sitting near the card's capacity means you should not expect to run
  other GPU work alongside training, even though the run itself has room.

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
git clone --depth 1 https://github.com/ggml-org/llama.cpp %USERPROFILE%\llama.cpp
uv sync --extra export

uv run ftlab merge  -c gemma4-12b-qlora.yaml
uv run ftlab export -c gemma4-12b-qlora.yaml --quant q4_k_m --ollama-name gemma4-qra
```

Conversion shells out to llama.cpp rather than vendoring a converter, which would
go stale within weeks of a new architecture landing. Point it with
`--llama-cpp <path>` or `LLAMA_CPP_DIR`.

### Quantization: three routes, in order

Building `llama-quantize` needs a C++ toolchain, which a Windows box generally
does not have. So the exporter takes whichever route is available:

1. **ollama** — `ollama create -q q4_K_M` quantizes on ingest, needs no
   compiler, and is already installed if you serve models locally. Used
   automatically when `--ollama-name` is given and no binary is found.
2. **`llama-quantize`** — for a portable `.gguf` rather than a model inside
   ollama's store. Needs `cmake -B build && cmake --build build --config Release`.
3. **Neither** — ship the f16. Twice the size, serves fine.

The rule throughout: never fail after producing something usable. An earlier
version raised when `llama-quantize` was missing, which killed the run *after* a
valid f16 GGUF was written and *before* the Modelfile — leaving anyone without a
compiler holding a converted model and a stack trace.

### Verified end to end

On this machine, with no C++ toolchain installed:

| step | result |
|---|---|
| merge | 135M adapter → 269 MB safetensors + tokenizer + chat template |
| convert | f16 GGUF, **0.25 GB**, chat template carried through |
| quantize | ollama on ingest → **105 MB** q4_K_M |
| register | `ollama create -q q4_K_M`, SYSTEM prompt preserved |
| run | `ollama run` loads and generates in the trained answer format |

The converter runs against this project's transformers 5.x / numpy 2.x despite
its requirements file pinning 4.x / 1.x — only `gguf` is genuinely needed, hence
the narrow `export` extra.

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
  looked more principled and was measurably wrong: a token-rate model
  under-predicted a real 1,266-step run by **2.3x**, while step time predicted it
  within 9%.
- **Memory and time are measured in two phases, on different samples.** They are
  different questions. Peak VRAM has to be an upper bound or it is not a bound,
  so it is measured on the longest examples in the corpus. Wall time is a sum
  over every example the run will touch, so it is measured on a sample drawn at
  even quantiles of the length distribution. Measuring both on the longest
  examples got the memory right and overstated the time by the ratio of the
  longest example to the mean — on this corpus, **33.4 s/step against a real
  14.7 s/step**, which projected 4h41m for a run that took well under half that.
  The worst-case step time is still reported, as the stated bound it always was.
  The worst-case phase runs first so it absorbs the one-time CUDA warm-up
  instead of that landing in the measurement the projection is built from.
- **Peak VRAM is measured on the longest examples**, not a random sample. With
  dynamic padding a short calibration easily misses the batch that would have
  OOMed. The verdict keys on **peak allocated** — what the run actually needs —
  not peak reserved, which is only what the caching allocator chose to hold and
  which PyTorch will free rather than OOM. Reserved is still printed, because it
  is what matters if you want other GPU work running alongside.
- **Evaluation is measured separately, and projected per example.** It's
  forward-only and several times faster than a training step, so it cannot be
  projected from the training rate — and it cannot be projected from a
  tokens/second rate either, which is the same mistake in a second place: that
  model under-predicted a real eval pass by **47%** (106s projected against 156s
  measured). Per-example overhead dominates at these lengths, so the projection
  scales with example count. The plan warns when eval dominates: on one config here,
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

### Deciding the last epoch by arithmetic

The exposure count is why this run trains **2 epochs, not 3**, and why the third
is gated rather than scheduled. At 3 epochs the busiest contract record is seen
21 times verbatim; for a closed-book model the failure mode of too many passes
is a record-reciter that has memorised phrasing instead of relationships.

Deciding that by squinting at the eval curve afterwards is not reproducible, and
it is biased — another epoch always *looks* like it might help. So the rule is
arithmetic, written down before the run:

```bash
uv run ftlab gate -c gemma4-12b-qlora.yaml     # exit 0 = stop, 10 = continue
```

Three conditions, **all** required to continue:

1. **Still learning** — the best eval loss of the final epoch beats the best of
   everything before it by at least `min_rel_improvement` (0.5%).
2. **Not memorising** — at the end of the run, eval sits no further above train
   than `max_generalisation_gap` (10%).
3. **Still at the floor** — the *last* measurement is within `overfit_tolerance`
   (0.2%) of the best one seen.

The bias is deliberately toward stopping. Under-training shows up in the grades
and is fixed by running more; over-training is only fixed by throwing the run
away.

#### What the first real run taught about this rule

Checks 1 and 3 are both **weak instruments under a schedule that anneals the
learning rate to zero**, and weak in the same direction — toward continuing. The
first 2-epoch run made that concrete:

```
  step    50  epoch  0.32  eval 0.3303  train 0.3165  gap  4.2%
  step   150  epoch  0.95  eval 0.2190  train 0.2045  gap  6.6%
  step   250  epoch  1.58  eval 0.1615  train 0.1364  gap 15.6%
  step   318  epoch  2.00  eval 0.1526  train 0.1255  gap 17.8%
```

- Check 1 compares an epoch's best against everything before it, so it is an
  **average over the epoch, dominated by its early part**. Here it read
  **+30.30%** while the last two measurements differed by **+0.19%** — below the
  gate's own 0.5% bar. A decaying LR makes almost any epoch improve on average.
- Check 3 **barely binds**. As the LR approaches zero the model stops moving, so
  the final measurement is very nearly always also the best. Here the two were
  the same point, and the check passed trivially.

The tempting fix — judge the terminal slope instead of the epoch average — just
inverts the bias, since flatness at the end is largely an artifact of the LR
having decayed rather than of the model being saturated. So the terminal slope is
**reported but never gates**.

Check 2 is the addition. The train/eval gap is the one quantity here a schedule
cannot fake, because both numbers are read off the same model at the same step.
It went from ~5% at the end of epoch 1 to ~18% at the end of epoch 2.

**Caveat, stated plainly:** check 2 was chosen *after* seeing the data it now
fires on, which is exactly what makes a pre-registered rule worth less. It is
defensible on its own terms — the gap is the failure mode this project cares
about, and it is schedule-independent — but a rule revised post hoc should not be
the sole basis for the decision it was revised to change. Treat a split verdict
as a prompt to look at `ftlab grade`, which measures the thing eval loss is only
a proxy for.

If the gate fires, the extra epoch is a **separate warm restart**, not a resumed
schedule:

```bash
uv run ftlab train -c gemma4-12b-qlora.yaml     --set run.name=gemma4-12b-qra-e3     --set train.epochs=1 --set train.learning_rate=5.0e-5     --resume-adapter outputs/gemma4-12b-qra/adapter
```

That matters for the learning rate. Cosine anneals to ~0 at the end of epoch 2,
so the phase-1 adapter is a *finished* model rather than one stopped mid-decay.
Extending the original schedule instead would have meant either stopping at 2
with the LR still at ~25% of peak, or resuming onto a re-planned curve that
jumps the LR back up mid-run. A separate phase gets its own clean decay at half
the peak.

`ftlab gate` reads the curve from two places, because neither is complete on its
own: `log_history` in the newest surviving checkpoint holds the evals taken
*during* training, and `trainer_metrics.json` holds the explicit end-of-training
`evaluate()` that runs afterwards — the single most important point, and one
that appears in no `log_history` at all. `gate.json` records the decision, both
checks, and the whole curve.

A one-epoch continuation has no earlier epoch of its own to compare against, so
the gate refuses rather than guessing. Pass `--baseline <previous final eval
loss>` and the same arithmetic answers "would a *fourth* epoch help".

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

217 tests, no GPU or network needed — a char-level fake tokenizer stands in, so
that a masking failure is a real bug rather than a tokenizer artefact.
