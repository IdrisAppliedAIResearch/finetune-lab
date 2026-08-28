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
figures, end years). Parametric recall frays on precisely these, and short exact
answers make the degradation gradeable by string comparison instead of
judgement. Expect this file to be where closed-book shows its cost.

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

## Monitoring

```bash
uv run tensorboard --logdir outputs
```

Each run writes `run_meta.json` (full config, library versions, token-length
stats, trainable parameter counts) and `metrics.json` alongside the adapter, so
a finished run explains itself months later.

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

59 tests, no GPU or network needed — a char-level fake tokenizer stands in, so
that a masking failure is a real bug rather than a tokenizer artefact.
