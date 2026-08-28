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


def cmd_synth(args: argparse.Namespace) -> int:
    """Generate the synthetic past performance corpus."""
    from .synth.build import build_and_write

    stats = build_and_write(
        args.out,
        seed=args.seed,
        scale=args.scale,
        holdout_ratio=args.holdout,
    )
    print(json.dumps(stats["world"], indent=2))
    print(
        f"\ntrain {stats['train']['total']} | eval {stats['eval']['total']} | "
        f"probes {stats['probes']['total']}"
    )
    print(f"by layer: {json.dumps(stats['train']['by_layer'])}")
    print(f"\nwrote {args.out}/train.jsonl, eval.jsonl, eval_probes.jsonl, library.json")
    return 0


def cmd_inspect_model(args: argparse.Namespace) -> int:
    """Inspect a checkpoint and validate a config's assumptions against it."""
    from .modelinfo import check_against_config, inspect_model, render

    cfg = _load_config(args) if args.config else None
    target = args.model or (cfg.model.base if cfg else None)
    if not target:
        raise SystemExit("pass --model <dir> or -c <config>")

    info = inspect_model(target)
    checks = check_against_config(info, cfg) if cfg else None
    print(render(info, checks))
    return 0 if not checks or all(c.ok for c in checks) else 1


def cmd_grade(args: argparse.Namespace) -> int:
    """Grade generated answers against the graph that produced the questions."""
    import json as _json

    from .grade import (
        aggregate,
        generate_answers,
        grade_generations,
        load_generations,
        load_world,
        render,
        save_generations,
        write_report,
    )

    if args.compare:
        from .grade import load_summary, render_comparison

        before, after = args.compare
        print(
            render_comparison(
                load_summary(before),
                load_summary(after),
                Path(before).parent.name or "before",
                Path(after).parent.name or "after",
            )
        )
        return 0

    cfg = _load_config(args)
    data_dir = Path(args.data or "data/processed")
    world = load_world(data_dir)

    if args.generations:
        items, generated = load_generations(args.generations)
        title = f"grade: {Path(args.generations).name}"
    else:
        items = []
        for name in {"eval": ["eval.jsonl"], "probes": ["eval_probes.jsonl"],
                     "both": ["eval.jsonl", "eval_probes.jsonl"]}[args.split]:
            items += [
                _json.loads(line)
                for line in (data_dir / name).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if args.limit:
            items = items[: args.limit]

        adapter = args.adapter
        if adapter is None and not args.base_only:
            default = cfg.run_dir / "adapter"
            adapter = str(default) if default.exists() else None

        generated = generate_answers(
            cfg, None if args.base_only else adapter, items,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        title = f"grade: {cfg.run.name}" + (" (base model)" if args.base_only else "")

    graded = grade_generations(items, generated, world)
    summary = aggregate(graded)
    print()
    print(render(summary, title))

    out_dir = Path(args.out or cfg.run_dir)
    if not args.generations:
        save_generations(items, generated, out_dir / "generations.jsonl")
    report = write_report(graded, summary, out_dir, title)
    print(f"\n[ftlab] {report}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Project steps, tokens, VRAM, wall time and cost before spending them."""
    from .plan import plan, write_plan

    cfg = _load_config(args)
    report = plan(cfg, calibrate_steps=args.calibrate)
    print(report.render())

    destination = Path(args.out) if args.out else cfg.run_dir / "plan.json"
    write_plan(report, destination)
    print(f"\n[ftlab] plan -> {destination}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Re-render the metrics of a finished run."""
    from .metrics import TrainingMetrics, load_metrics

    data = load_metrics(args.run)
    known = {f for f in TrainingMetrics().__dict__}
    metrics = TrainingMetrics(**{k: v for k, v in data.items() if k in known})
    print(metrics.report())
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
    from .export_gguf import export

    cfg = _load_config(args)
    merged = Path(args.merged or (cfg.run_dir / "merged"))
    if not merged.exists():
        raise FileNotFoundError(f"merged model not found at {merged} -- run 'ftlab merge' first")

    result = export(
        merged_dir=merged,
        out_dir=Path(args.out_dir or (cfg.run_dir / "gguf")),
        name=cfg.run.name,
        quant=args.quant,
        llama_cpp_dir=args.llama_cpp,
        system_prompt=cfg.data.system_prompt,
        ollama_name=args.ollama_name,
    )

    print()
    print(result.summary())
    if not result.registered_as:
        print(
            f"\nTo register with ollama:\n"
            f"    ollama create <name> -f {result.modelfile}"
            + (f" -q {args.quant}" if result.quant_route == "none" and args.quant != "f16" else "")
        )
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

    synth = sub.add_parser(
        "synth", help="generate the synthetic past performance corpus"
    )
    synth.add_argument("--out", default="data/processed", help="output directory")
    synth.add_argument("--seed", type=int, default=42, help="world seed; fixes everything")
    synth.add_argument(
        "--scale", default="demo", choices=["compact", "demo", "full"]
    )
    synth.add_argument(
        "--holdout",
        type=float,
        default=0.2,
        help="fraction of opportunities held out of training entirely",
    )
    synth.set_defaults(func=cmd_synth)

    inspect_p = sub.add_parser(
        "inspect-model",
        help="inspect a checkpoint's modules and check a config against it",
    )
    _add_config_args(inspect_p, required=False)
    inspect_p.add_argument(
        "--model", help="checkpoint directory (defaults to the config's model.base)"
    )
    inspect_p.set_defaults(func=cmd_inspect_model)

    grade = sub.add_parser(
        "grade", help="score generated answers against the graph's ground truth"
    )
    _add_config_args(grade, required=False)
    grade.add_argument("--data", help="corpus directory (default data/processed)")
    grade.add_argument(
        "--split", default="both", choices=["eval", "probes", "both"]
    )
    grade.add_argument("--adapter", help="adapter dir (defaults to <run_dir>/adapter)")
    grade.add_argument(
        "--base-only",
        action="store_true",
        help="grade the untuned base model, for a before/after comparison",
    )
    grade.add_argument(
        "--generations",
        help="grade an existing generations.jsonl instead of generating again",
    )
    grade.add_argument("--limit", type=int, help="grade only the first N items")
    grade.add_argument(
        "--max-new-tokens",
        type=int,
        default=1280,
        help="measured: recommendation targets run to ~1180 tokens at p99, so a "
        "smaller budget truncates them and depresses the rejection metrics",
    )
    grade.add_argument(
        "--batch-size", type=int, default=8, help="generation batch size"
    )
    grade.add_argument("--out", help="where to write grades.json and the report")
    grade.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help="diff two grades.json files instead of grading",
    )
    grade.set_defaults(func=cmd_grade)

    plan_p = sub.add_parser(
        "plan", help="project steps, tokens, VRAM, time and cost before training"
    )
    _add_config_args(plan_p)
    plan_p.add_argument(
        "--calibrate",
        type=int,
        default=0,
        metavar="STEPS",
        help="run this many real steps to measure throughput and peak VRAM "
        "(0 = schedule arithmetic only; 8 is usually enough)",
    )
    plan_p.add_argument("--out", help="where to write plan.json")
    plan_p.set_defaults(func=cmd_plan)

    report = sub.add_parser("report", help="re-render the metrics of a finished run")
    report.add_argument("--run", required=True, help="run directory, e.g. outputs/qra-smoke")
    report.set_defaults(func=cmd_report)

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
