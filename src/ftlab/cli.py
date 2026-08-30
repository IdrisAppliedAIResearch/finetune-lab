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

    cfg = _load_config(args)
    if getattr(args, "resume_adapter", None):
        cfg.train.resume_adapter = args.resume_adapter
    train(cfg)
    return 0


def cmd_real_fetch(args: argparse.Namespace) -> int:
    """Pull a fresh slice from the USASpending API, one year at a time."""
    from .real.ingest import build_slice, write_slice

    data = build_slice(
        subaward_pages=args.subaward_pages,
        prime_pages=args.prime_pages,
        years=range(args.from_year, args.to_year + 1),
    )
    stats = write_slice(data, args.out or "data/real")
    print(json.dumps(stats, indent=2))
    return 0


def cmd_real_build(args: argparse.Namespace) -> int:
    """Build the real corpus from the cached USASpending slice."""
    from .real.build import build

    stats = build(
        data_dir=args.data or "data/real",
        out_dir=args.out or "data/real_corpus",
        dropout=args.dropout,
    )
    print(json.dumps(stats, indent=2))
    return 0


def cmd_masked_build(args: argparse.Namespace) -> int:
    """Build the masked-sub evaluation set from observed teaming."""
    from .real.masked import build

    stats = build(
        data_dir=args.data or "data/real",
        out_path=args.out or "data/real_corpus/masked_sub.jsonl",
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2))
    return 0


def cmd_arms(args: argparse.Namespace) -> int:
    """Run the three-arm benchmark and print the comparison table."""
    from .real.arms import run

    cfg = _load_config(args)
    adapter = args.adapter or str(cfg.run_dir / "adapter")
    run(
        cfg,
        split_path=args.split,
        out_dir=args.out or (cfg.run_dir / "arms"),
        adapter=adapter,
        arms=tuple(args.arm or ("d", "c", "b", "a")),
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
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
        # A bare --question has no context of its own, so retrieve for it.
        data_dir=args.data or "data/processed" if args.question else None,
    )
    if args.out:
        save(results, args.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ftlab",
        description="Local LoRA/QLoRA fine-tuning for Question-Reasoning-Answer triples.",
    )
    sub = parser.add_subparsers(dest="command", required=True)


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
    train.add_argument(
        "--resume-adapter",
        help="continue training an existing adapter instead of starting fresh "
        "(the gate's extra epoch; give it a lower learning rate)",
    )
    train.set_defaults(func=cmd_train)


    real_fetch = sub.add_parser(
        "real-fetch", help="pull the USASpending slice (per year, so no year is truncated)"
    )
    real_fetch.add_argument("--out", help="slice dir to write (default data/real)")
    real_fetch.add_argument("--from-year", type=int, default=2015)
    real_fetch.add_argument("--to-year", type=int, default=2025)
    real_fetch.add_argument(
        "--subaward-pages", type=int, default=25,
        help="pages of 100 per year; the years with the most rows need ~25",
    )
    real_fetch.add_argument("--prime-pages", type=int, default=10)
    real_fetch.set_defaults(func=cmd_real_fetch)

    real_build = sub.add_parser(
        "real-build", help="build the real USASpending corpus"
    )
    real_build.add_argument("--data", help="cached slice dir (default data/real)")
    real_build.add_argument("--out", help="output dir (default data/real_corpus)")
    real_build.add_argument(
        "--dropout", type=float, default=0.4,
        help="share of training examples with the library records withheld, so "
        "the same adapter can answer closed-book (arm B)",
    )
    real_build.set_defaults(func=cmd_real_build)

    masked = sub.add_parser(
        "masked-build",
        help="build the masked-sub eval: hide a real subcontract, ask who filled it",
    )
    masked.add_argument("--data", help="cached slice dir (default data/real)")
    masked.add_argument("--out", help="output jsonl (default data/real_corpus/masked_sub.jsonl)")
    masked.add_argument(
        "--seed", type=int, default=0,
        help="shuffles the slate; the answer's position must not be learnable",
    )
    masked.set_defaults(func=cmd_masked_build)

    arms = sub.add_parser(
        "arms", help="run the arm benchmark (rule, base+RAG, tuned, tuned+RAG)"
    )
    _add_config_args(arms)
    arms.add_argument(
        "--split", default="data/real_corpus/blind.jsonl", help="questions to run"
    )
    arms.add_argument("--adapter", help="adapter dir (defaults to <run_dir>/adapter)")
    arms.add_argument(
        "--arm", action="append", choices=["a", "b", "c", "d"],
        help="restrict to one arm; repeatable (default: all four). Arm D is the "
        "prior-teaming rule and needs no model or GPU.",
    )
    arms.add_argument(
        # Matches what the v2 benchmark was actually run at. The default used
        # to be 900 while the recorded run passed 2500, so re-running the
        # command as documented truncated the base model far harder than the
        # published numbers did.
        "--max-new-tokens", type=int, default=2500,
        help="a verbose model needs room to reach a conclusion; at 420 the "
        "base model was cut off mid-analysis on 16 of 18 answers",
    )
    arms.add_argument("--batch-size", type=int, default=8)
    arms.add_argument("--out", help="where to write arms.json")
    arms.set_defaults(func=cmd_arms)


    infer = sub.add_parser("infer", help="generate from a trained adapter")
    _add_config_args(infer)
    infer.add_argument("--question", "-q", help="a single question to answer")
    infer.add_argument("--from-file", help="QRA jsonl to pull questions from")
    infer.add_argument(
        "--data", help="corpus directory to retrieve from (default data/processed)"
    )
    infer.add_argument("--limit", type=int, help="max questions to run")
    infer.add_argument("--adapter", help="adapter dir (defaults to <run_dir>/adapter)")
    infer.add_argument(
        "--base-only", action="store_true", help="skip the adapter, for before/after comparison"
    )
    infer.add_argument("--max-new-tokens", type=int, default=512)
    infer.add_argument("--temperature", type=float, default=0.7)
    infer.add_argument("--out", help="write generations to this jsonl")
    infer.set_defaults(func=cmd_infer)



    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
