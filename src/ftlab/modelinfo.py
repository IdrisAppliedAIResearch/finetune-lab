"""Inspect a checkpoint without loading it, and check a config against it.

This exists because of a bug it would have caught. The Gemma 4 preset shipped
with ``lora.exclude_modules: [vision_tower, audio_tower, multi_modal_projector,
...]`` -- names guessed from the config's sub-config keys. The checkpoint's real
modules are ``vision_embedder``, ``embed_vision`` and ``embed_audio``. Not one
guess matched, so the exclusion silently did nothing while the comment above it
claimed the perception towers were protected.

PEFT does not complain about an exclude pattern that matches nothing, and neither
does anything else, so the config was wrong for as long as nobody looked. Now
something looks.

Everything here reads the safetensors header and the JSON config; no weights are
loaded, so a 23GB checkpoint inspects in about a second.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Tensors whose module is a Linear that PEFT's "all-linear" would target.
# Weight-only, 2-D, and not an embedding or a norm.
_NON_LINEAR_HINTS = ("norm", "layernorm", "ln", "embed_tokens", "pos_embedding")


@dataclass
class ModelInfo:
    path: Path
    architectures: list[str]
    model_type: str
    dtype: str | None
    tensor_count: int
    module_names: list[str]
    linear_modules: list[str]
    top_prefixes: dict[str, int]
    parameters: int = 0
    text_parameters: int = 0
    sub_configs: dict[str, Any] = field(default_factory=dict)
    chat_template: bool = False
    tokenizer_class: str | None = None
    vocab_size: int | None = None
    special_tokens: dict[str, str] = field(default_factory=dict)

    @property
    def nf4_weight_gb(self) -> float:
        """Roughly what the 4-bit weights occupy.

        About 0.55 bytes per parameter: four bits plus quantization constants,
        with double quantization on. Weights only -- activations and optimizer
        state sit on top, and they are what actually decide whether a run fits.
        """
        return self.parameters * 0.55 / 1024**3

    @property
    def bf16_weight_gb(self) -> float:
        return self.parameters * 2 / 1024**3

    @property
    def is_multimodal(self) -> bool:
        return bool({"vision_config", "audio_config"} & set(self.sub_configs))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tensor_names(model_dir: Path) -> list[str]:
    """Tensor names from the safetensors header(s). Handles sharded models."""
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        return sorted(_load_json(index)["weight_map"])

    from safetensors import safe_open

    names: list[str] = []
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            names.extend(handle.keys())
    if not names:
        raise FileNotFoundError(f"no .safetensors found under {model_dir}")
    return sorted(names)


def _shapes(model_dir: Path, names: list[str]) -> dict[str, list[int]]:
    from safetensors import safe_open

    shapes: dict[str, list[int]] = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            for key in handle.keys():  # noqa: SIM118 - safetensors handle, not a dict
                shapes[key] = handle.get_slice(key).get_shape()
    return {n: shapes[n] for n in names if n in shapes}


def inspect_model(model_dir: str | Path) -> ModelInfo:
    model_dir = Path(model_dir)
    config = _load_json(model_dir / "config.json")

    names = _tensor_names(model_dir)
    shapes = _shapes(model_dir, names)

    modules: set[str] = set()
    linears: set[str] = set()
    for name in names:
        if not name.endswith((".weight", ".bias")):
            continue
        module = name.rsplit(".", 1)[0]
        modules.add(module)
        leaf = module.rsplit(".", 1)[-1].lower()
        if (
            name.endswith(".weight")
            and len(shapes.get(name, [])) == 2
            and not any(hint in leaf for hint in _NON_LINEAR_HINTS)
        ):
            linears.add(module)

    top: dict[str, int] = {}
    for name in names:
        head = ".".join(name.split(".")[:2])
        top[head] = top.get(head, 0) + 1

    total = 0
    text_total = 0
    for name, shape in shapes.items():
        count = 1
        for dim in shape:
            count *= dim
        total += count
        if "language_model" in name:
            text_total += count

    info = ModelInfo(
        path=model_dir,
        architectures=config.get("architectures", []),
        model_type=config.get("model_type", "?"),
        dtype=config.get("dtype") or config.get("torch_dtype"),
        tensor_count=len(names),
        parameters=total,
        text_parameters=text_total,
        module_names=sorted(modules),
        linear_modules=sorted(linears),
        top_prefixes=dict(sorted(top.items(), key=lambda kv: -kv[1])),
        sub_configs={
            k: v for k, v in config.items() if k.endswith("_config") and isinstance(v, dict)
        },
    )
    _add_tokenizer_facts(info, model_dir)
    return info


def _add_tokenizer_facts(info: ModelInfo, model_dir: Path) -> None:
    config_path = model_dir / "tokenizer_config.json"
    if not config_path.exists():
        return
    config = _load_json(config_path)
    info.tokenizer_class = config.get("tokenizer_class")
    info.chat_template = bool(config.get("chat_template")) or (
        model_dir / "chat_template.jinja"
    ).exists()
    info.special_tokens = {
        key: (value if isinstance(value, str) else value.get("content", "?"))
        for key, value in sorted(config.items())
        if key.endswith("_token") and key != "extra_special_tokens"
    }
    tokenizer_json = model_dir / "tokenizer.json"
    if tokenizer_json.exists():
        try:
            info.vocab_size = len(_load_json(tokenizer_json)["model"]["vocab"])
        except Exception:  # noqa: BLE001 - vocab shape varies by tokenizer
            info.vocab_size = None


# ---------------------------------------------------------------------------
# config checking
# ---------------------------------------------------------------------------


@dataclass
class ConfigCheck:
    name: str
    ok: bool
    detail: str


def check_against_config(info: ModelInfo, cfg) -> list[ConfigCheck]:
    """Validate the parts of a training config that depend on this checkpoint."""
    checks: list[ConfigCheck] = []

    # -- the one that was wrong --
    excludes = list(cfg.lora.exclude_modules)
    if excludes:
        for pattern in excludes:
            hits = [m for m in info.module_names if pattern in m]
            checks.append(
                ConfigCheck(
                    f"exclude '{pattern}'",
                    bool(hits),
                    f"matches {len(hits)} module(s)"
                    if hits
                    else "MATCHES NOTHING -- this exclusion does nothing",
                )
            )
    elif info.is_multimodal:
        checks.append(
            ConfigCheck(
                "exclude_modules",
                False,
                "empty on a multimodal checkpoint -- 'all-linear' will wrap the "
                "perception projections too",
            )
        )

    # -- what all-linear would actually target --
    if cfg.lora.target_modules == "auto":
        text = [m for m in info.linear_modules if "language_model" in m]
        other = [m for m in info.linear_modules if "language_model" not in m]
        remaining = [
            m for m in other if not any(p in m for p in excludes)
        ]
        checks.append(
            ConfigCheck(
                "all-linear scope",
                not remaining,
                f"{len(text)} text linears; {len(remaining)} non-text linears "
                + (f"still targeted ({', '.join(remaining[:3])})" if remaining else "excluded"),
            )
        )

    # -- chat template: ftlab renders chat turns, so this is load-bearing --
    checks.append(
        ConfigCheck(
            "chat template",
            info.chat_template,
            "present"
            if info.chat_template
            else "MISSING -- this looks like a base model. QRA training renders "
            "chat turns; use the instruction-tuned checkpoint or set a template.",
        )
    )

    return checks


def render(info: ModelInfo, checks: list[ConfigCheck] | None = None) -> str:
    lines = [
        f"=== {info.path.name} ===",
        f"  architecture      {', '.join(info.architectures) or '?'}",
        f"  model_type        {info.model_type}",
        f"  dtype             {info.dtype}",
        f"  tensors           {info.tensor_count}",
        f"  parameters        {info.parameters / 1e9:.2f}B "
        f"({info.text_parameters / 1e9:.2f}B in the language model)",
        f"  weights           {info.bf16_weight_gb:.1f} GB bf16"
        f"  /  ~{info.nf4_weight_gb:.1f} GB nf4",
        f"  multimodal        {'yes' if info.is_multimodal else 'no'}"
        + (f"  ({', '.join(sorted(info.sub_configs))})" if info.sub_configs else ""),
        "",
        "Top-level module groups",
    ]
    for prefix, count in list(info.top_prefixes.items())[:8]:
        lines.append(f"  {count:6}  {prefix}")

    text_linears = [m for m in info.linear_modules if "language_model" in m]
    other_linears = [m for m in info.linear_modules if "language_model" not in m]
    lines += [
        "",
        "Linear modules ('all-linear' targets these)",
        f"  {len(text_linears):6}  inside language_model",
        f"  {len(other_linears):6}  outside language_model",
    ]
    for module in other_linears[:8]:
        lines.append(f"          {module}")

    lines += [
        "",
        "Tokenizer",
        f"  class             {info.tokenizer_class}",
        f"  vocab             {info.vocab_size:,}" if info.vocab_size else "  vocab             ?",
        f"  chat template     {'present' if info.chat_template else 'MISSING'}",
    ]
    # Turn markers and any native reasoning token: these decide how the corpus
    # should be rendered for this model.
    notable = ("think_token", "sot_token", "eot_token")
    interesting = {k: v for k, v in info.special_tokens.items() if k in notable}
    for key, value in interesting.items():
        lines.append(f"  {key:<18}{value}")

    if checks:
        lines += ["", "Config checks"]
        for check in checks:
            lines.append(f"  [{'ok  ' if check.ok else 'FAIL'}] {check.name:<28} {check.detail}")
        if any(not c.ok and c.name.startswith("exclude") for c in checks):
            lines.append("")
            lines.append(
                "  A failing exclude pattern is silent everywhere else: PEFT does not"
            )
            lines.append("  warn when an exclusion matches nothing.")

    return "\n".join(lines)
