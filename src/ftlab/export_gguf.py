"""Convert a merged model to GGUF and register it with ollama.

This shells out to llama.cpp rather than vendoring a converter: llama.cpp tracks
new architectures within days of release, and a copy pinned here would go stale.
Point ``--llama-cpp`` at a clone, or set the LLAMA_CPP_DIR environment variable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Quantization levels worth choosing between, smallest first. q4_k_m is the
# usual default: roughly 4.5 bits per weight with a quality loss most people
# cannot detect in chat. q8_0 is near-lossless and twice the size.
QUANT_TYPES = ("q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16")


def find_llama_cpp(explicit: str | Path | None = None) -> Path:
    candidates = [
        explicit,
        os.environ.get("LLAMA_CPP_DIR"),
        Path.home() / "llama.cpp",
        Path.home() / "src" / "llama.cpp",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if (path / "convert_hf_to_gguf.py").exists():
            return path

    raise FileNotFoundError(
        "llama.cpp not found. Clone it and retry:\n"
        "    git clone https://github.com/ggml-org/llama.cpp %USERPROFILE%\\llama.cpp\n"
        "then pass --llama-cpp <path> or set LLAMA_CPP_DIR."
    )


def convert_to_gguf(
    merged_dir: str | Path,
    out_path: str | Path,
    llama_cpp_dir: str | Path | None = None,
    outtype: str = "f16",
) -> Path:
    """Run convert_hf_to_gguf.py to produce an unquantized (f16/bf16) GGUF."""
    merged_dir = Path(merged_dir)
    out_path = Path(out_path)
    repo = find_llama_cpp(llama_cpp_dir)
    script = repo / "convert_hf_to_gguf.py"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        str(merged_dir),
        "--outfile",
        str(out_path),
        "--outtype",
        outtype,
    ]
    print(f"[ftlab] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[ftlab] wrote {out_path} ({out_path.stat().st_size / 1024**3:.2f} GB)")
    return out_path


def quantize(
    src_gguf: str | Path,
    out_gguf: str | Path,
    quant_type: str = "q4_k_m",
    llama_cpp_dir: str | Path | None = None,
) -> Path:
    """Quantize an f16 GGUF using llama.cpp's llama-quantize binary."""
    if quant_type not in QUANT_TYPES:
        raise ValueError(f"quant_type must be one of {QUANT_TYPES}, got {quant_type!r}")

    repo = find_llama_cpp(llama_cpp_dir)
    binary = _find_quantize_binary(repo)
    src_gguf, out_gguf = Path(src_gguf), Path(out_gguf)

    cmd = [str(binary), str(src_gguf), str(out_gguf), quant_type]
    print(f"[ftlab] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[ftlab] wrote {out_gguf} ({out_gguf.stat().st_size / 1024**3:.2f} GB)")
    return out_gguf


def _find_quantize_binary(repo: Path) -> Path:
    on_path = shutil.which("llama-quantize")
    if on_path:
        return Path(on_path)

    for relative in ("build/bin/Release", "build/bin", "build", "."):
        for name in ("llama-quantize.exe", "llama-quantize"):
            candidate = repo / relative / name
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"llama-quantize not found under {repo}. Build llama.cpp first:\n"
        "    cmake -B build && cmake --build build --config Release\n"
        "Or skip quantization and serve the f16 GGUF directly."
    )


def write_modelfile(
    gguf_path: str | Path,
    out_path: str | Path,
    system_prompt: str | None = None,
    temperature: float = 0.7,
) -> Path:
    """Emit an ollama Modelfile pointing at the GGUF."""
    gguf_path, out_path = Path(gguf_path).resolve(), Path(out_path)
    lines = [
        f"FROM {gguf_path}",
        "",
        f"PARAMETER temperature {temperature}",
        "PARAMETER top_p 0.95",
        "",
    ]
    if system_prompt:
        escaped = system_prompt.replace('"', '\\"')
        lines.append(f'SYSTEM """{escaped}"""')
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ftlab] wrote {out_path}")
    return out_path


def register_with_ollama(modelfile: str | Path, name: str) -> None:
    if not shutil.which("ollama"):
        raise FileNotFoundError("ollama is not on PATH")
    cmd = ["ollama", "create", name, "-f", str(Path(modelfile).resolve())]
    print(f"[ftlab] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[ftlab] registered as '{name}'. Try: ollama run {name}")
