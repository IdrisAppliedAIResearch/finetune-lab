"""Command-line surface: ftlab <command> [--config ...] [--set key=value ...]."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config as config_mod


def _add_config_args(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        "--config",
        "-c",
        required=required,
        help="path to a YAML config (bare names resolve inside configs/)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set train.epochs=1",
    )


def _load_config(args: argparse.Namespace) -> config_mod.Config:
    overrides = dict(config_mod.parse_override(item) for item in args.overrides)
    return config_mod.load(args.config, overrides)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_doctor(_: argparse.Namespace) -> int:
    from .doctor import main as doctor_main

    return doctor_main()


def cmd_show_config(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    print(json.dumps(cfg.model_dump(), indent=2, default=str))
    return 0


def cmd_check_data(args: argparse.Namespace) -> int:
    """Validate a QRA file and show exactly what the model will be scored on.

    A silently mis-masked dataset is the most expensive bug in fine-tuning:
    training runs, the loss falls, and the model learns the wrong thing. This
    prints the boundary so it can be checked by eye before burning GPU hours.
    """
    from .data import IGNORE_INDEX, encode_all, load_jsonl
    from .model import load_tokenizer

    cfg = _load_config(args)
    path = args.path or cfg.data.train_path
    examples = load_jsonl(path)
    print(f"[ftlab] {len(examples)} valid records in {path}")

    tokenizer = load_tokenizer(cfg.model)
    rows, stats = encode_all(examples, cfg.data, tokenizer)
    print(f"[ftlab] token stats: {json.dumps(stats, indent=2)}")

    for index in range(min(args.samples, len(rows))):
        row = rows[index]
        scored = [
            token for token, label in zip(row["input_ids"], row["labels"], strict=True)
            if label != IGNORE_INDEX
        ]
        masked = [
            token for token, label in zip(row["input_ids"], row["labels"], strict=True)
            if label == IGNORE_INDEX
        ]
        print(f"\n{'=' * 70}\nSAMPLE {index} -- {len(masked)} masked, {len(scored)} scored")
        print(f"{'-' * 70}\n--- MASKED (no loss) ---")
        print(tokenizer.decode(masked))
        print("\n--- SCORED (loss applied) ---")
        print(tokenizer.decode(scored))

    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .train import train

    train(_load_config(args))
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    from .infer import questions_from, run, save

    cfg = _load_config(args)

    if args.question:
        questions = [args.question]
    elif args.from_file:
        questions = questions_from(args.from_file, args.limit)
    else:
        questions = questions_from(cfg.data.train_path, args.limit or 3)

    adapter = args.adapter
    if adapter is None and not args.base_only:
        default_adapter = cfg.run_dir / "adapter"
        adapter = str(default_adapter) if default_adapter.exists() else None

    results = run(
        cfg,
        None if args.base_only else adapter,
        questions,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    if args.out:
        save(results, args.out)
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    from .merge import merge_adapter

    cfg = _load_config(args)
    adapter = args.adapter or (cfg.run_dir / "adapter")
    output = args.out or (cfg.run_dir / "merged")
    merge_adapter(cfg, adapter, output, dtype=args.dtype)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .export_gguf import (
        convert_to_gguf,
        quantize,
        register_with_ollama,
        write_modelfile,
    )

    cfg = _load_config(args)
    merged = Path(args.merged or (cfg.run_dir / "merged"))
    if not merged.exists():
        raise FileNotFoundError(f"merged model not found at {merged} -- run 'ftlab merge' first")

    gguf_dir = Path(args.out_dir or (cfg.run_dir / "gguf"))
    f16_path = gguf_dir / f"{cfg.run.name}-f16.gguf"
    convert_to_gguf(merged, f16_path, args.llama_cpp, outtype="f16")

    final = f16_path
    if args.quant != "f16":
        final = gguf_dir / f"{cfg.run.name}-{args.quant}.gguf"
        quantize(f16_path, final, args.quant, args.llama_cpp)

    modelfile = gguf_dir / "Modelfile"
    write_modelfile(final, modelfile, system_prompt=cfg.data.system_prompt)

    if args.ollama_name:
        register_with_ollama(modelfile, args.ollama_name)
    else:
        print(f"\nTo register with ollama:\n    ollama create <name> -f {modelfile}")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ftlab",
        description="Local LoRA/QLoRA fine-tuning for Question-Reasoning-Answer triples.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check GPU, CUDA kernels, and package versions")
    doctor.set_defaults(func=cmd_doctor)

    show = sub.add_parser("show-config", help="print a fully resolved config")
    _add_config_args(show)
    show.set_defaults(func=cmd_show_config)

    check = sub.add_parser("check-data", help="validate a QRA file and show the loss mask")
    _add_config_args(check)
    check.add_argument("--path", help="QRA jsonl to inspect (defaults to data.train_path)")
    check.add_argument("--samples", type=int, default=2, help="how many examples to print")
    check.set_defaults(func=cmd_check_data)

    train = sub.add_parser("train", help="train a LoRA adapter")
    _add_config_args(train)
    train.set_defaults(func=cmd_train)

    infer = sub.add_parser("infer", help="generate from a trained adapter")
    _add_config_args(infer)
    infer.add_argument("--question", "-q", help="a single question to answer")
    infer.add_argument("--from-file", help="QRA jsonl to pull questions from")
    infer.add_argument("--limit", type=int, help="max questions to run")
    infer.add_argument("--adapter", help="adapter dir (defaults to <run_dir>/adapter)")
    infer.add_argument(
        "--base-only", action="store_true", help="skip the adapter, for before/after comparison"
    )
    infer.add_argument("--max-new-tokens", type=int, default=512)
    infer.add_argument("--temperature", type=float, default=0.7)
    infer.add_argument("--out", help="write generations to this jsonl")
    infer.set_defaults(func=cmd_infer)

    merge = sub.add_parser("merge", help="merge the adapter into base weights")
    _add_config_args(merge)
    merge.add_argument("--adapter", help="adapter dir (defaults to <run_dir>/adapter)")
    merge.add_argument("--out", help="output dir (defaults to <run_dir>/merged)")
    merge.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    merge.set_defaults(func=cmd_merge)

    export = sub.add_parser("export", help="convert a merged model to GGUF for ollama")
    _add_config_args(export)
    export.add_argument("--merged", help="merged model dir (defaults to <run_dir>/merged)")
    export.add_argument("--out-dir", help="where to write GGUF files")
    export.add_argument("--quant", default="q4_k_m", help="q4_k_m, q5_k_m, q6_k, q8_0, or f16")
    export.add_argument("--llama-cpp", help="path to a llama.cpp clone")
    export.add_argument("--ollama-name", help="register the result under this ollama name")
    export.set_defaults(func=cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
