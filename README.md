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

## Commands

| command | purpose |
|---|---|
| `ftlab doctor` | GPU, CUDA kernels, bitsandbytes, package versions |
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

38 tests, no GPU or network needed — a char-level fake tokenizer stands in, so
that a masking failure is a real bug rather than a tokenizer artefact.
