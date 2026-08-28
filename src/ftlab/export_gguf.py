"""Convert a merged model to GGUF and register it with ollama.

Conversion shells out to llama.cpp rather than vendoring a converter: llama.cpp
tracks new architectures within days of release, and a copy pinned here would go
stale. Point ``--llama-cpp`` at a clone, or set LLAMA_CPP_DIR.

Quantization is the part with a choice in it, and the order matters on Windows:

1. **ollama**, when you are registering there anyway. ``ollama create -q q4_K_M``
   quantizes on ingest, needs no compiler, and is almost certainly already
   installed if you are serving models locally.
2. **llama-quantize**, when you want a portable ``.gguf`` file rather than a
   model inside ollama's store. This needs llama.cpp built, which needs a C++
   toolchain.
3. **Neither** -- ship the f16. It is twice the size and serves fine.

The one rule throughout: never fail after producing something usable. An earlier
version raised when it could not find llama-quantize, which killed the run
*after* a valid f16 GGUF had been written and *before* the Modelfile, leaving a
user with no compiler holding a converted model and a stack trace.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Quantization levels worth choosing between, smallest first. q4_k_m is the
# usual default: roughly 4.5 bits per weight with a quality loss most people
# cannot detect in chat. q8_0 is near-lossless and twice the size.
QUANT_TYPES = ("q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16")

# ollama spells these in caps; llama-quantize accepts either.
OLLAMA_QUANT = {
    "q4_k_m": "q4_K_M",
    "q5_k_m": "q5_K_M",
    "q6_k": "q6_K",
    "q8_0": "q8_0",
}


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
        "    git clone --depth 1 https://github.com/ggml-org/llama.cpp "
        "%USERPROFILE%\\llama.cpp\n"
        "then pass --llama-cpp <path> or set LLAMA_CPP_DIR."
    )


def convert_to_gguf(
    merged_dir: str | Path,
    out_path: str | Path,
    llama_cpp_dir: str | Path | None = None,
    outtype: str = "f16",
) -> Path:
    """Run convert_hf_to_gguf.py to produce an unquantized GGUF."""
    merged_dir = Path(merged_dir)
    out_path = Path(out_path)

    # Check the input first. Reporting "llama.cpp not found" to someone who
    # pointed at an adapter sends them off installing a toolchain they did not
    # need for the problem they actually have.
    if not (merged_dir / "config.json").exists():
        raise FileNotFoundError(
            f"{merged_dir} has no config.json -- point this at a merged model "
            "directory, not an adapter. Run 'ftlab merge' first."
        )

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


# ---------------------------------------------------------------------------
# quantization
# ---------------------------------------------------------------------------


def find_quantize_binary(repo: Path | None = None) -> Path | None:
    """Locate llama-quantize, or None. Absence is a fallback, not an error."""
    on_path = shutil.which("llama-quantize")
    if on_path:
        return Path(on_path)
    if repo is None:
        return None
    for relative in ("build/bin/Release", "build/bin", "build", "."):
        for name in ("llama-quantize.exe", "llama-quantize"):
            candidate = repo / relative / name
            if candidate.exists():
                return candidate
    return None


def quantize(
    src_gguf: str | Path,
    out_gguf: str | Path,
    quant_type: str = "q4_k_m",
    llama_cpp_dir: str | Path | None = None,
) -> Path:
    """Quantize an f16 GGUF with llama.cpp's llama-quantize binary."""
    if quant_type not in QUANT_TYPES:
        raise ValueError(f"quant_type must be one of {QUANT_TYPES}, got {quant_type!r}")

    repo = find_llama_cpp(llama_cpp_dir)
    binary = find_quantize_binary(repo)
    if binary is None:
        raise FileNotFoundError(
            f"llama-quantize not found under {repo}. Build llama.cpp:\n"
            "    cmake -B build && cmake --build build --config Release\n"
            "Or quantize through ollama instead (--ollama-name), which needs no "
            "compiler."
        )

    src_gguf, out_gguf = Path(src_gguf), Path(out_gguf)
    cmd = [str(binary), str(src_gguf), str(out_gguf), quant_type]
    print(f"[ftlab] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[ftlab] wrote {out_gguf} ({out_gguf.stat().st_size / 1024**3:.2f} GB)")
    return out_gguf


def ollama_available() -> bool:
    return shutil.which("ollama") is not None


# ---------------------------------------------------------------------------
# ollama packaging
# ---------------------------------------------------------------------------


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


def register_with_ollama(
    modelfile: str | Path, name: str, quantize_to: str | None = None
) -> None:
    """Create an ollama model, optionally quantizing on ingest."""
    if not ollama_available():
        raise FileNotFoundError("ollama is not on PATH")

    cmd = ["ollama", "create", name, "-f", str(Path(modelfile).resolve())]
    if quantize_to:
        cmd += ["-q", OLLAMA_QUANT.get(quantize_to, quantize_to)]
    print(f"[ftlab] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[ftlab] registered as '{name}'. Try: ollama run {name}")


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


@dataclass
class ExportResult:
    f16_path: Path
    final_path: Path
    modelfile: Path
    quant_route: str  # "ollama" | "llama-quantize" | "none"
    quant_type: str
    registered_as: str | None = None

    def summary(self) -> str:
        size = self.final_path.stat().st_size / 1024**3
        lines = [
            f"GGUF:      {self.final_path}  ({size:.2f} GB)",
            f"Modelfile: {self.modelfile}",
        ]
        if self.quant_route == "ollama":
            lines.append(
                f"Quantized: {self.quant_type} by ollama on ingest (no compiler needed)"
            )
        elif self.quant_route == "llama-quantize":
            lines.append(f"Quantized: {self.quant_type} by llama-quantize")
        else:
            lines.append(
                "Quantized: no -- shipping f16. Build llama.cpp for a portable "
                "quantized file, or pass --ollama-name to let ollama quantize."
            )
        if self.registered_as:
            lines.append(f"ollama:    {self.registered_as}  (ollama run {self.registered_as})")
        return "\n".join(lines)


def export(
    merged_dir: str | Path,
    out_dir: str | Path,
    name: str,
    quant: str = "q4_k_m",
    llama_cpp_dir: str | Path | None = None,
    system_prompt: str | None = None,
    ollama_name: str | None = None,
) -> ExportResult:
    """Convert, quantize by whichever route is available, and package.

    Ordering is deliberate. Everything that can still succeed runs before
    anything that might not: the f16 conversion and the Modelfile are produced
    unconditionally, so a missing quantizer downgrades the result rather than
    destroying it.
    """
    if quant not in QUANT_TYPES:
        raise ValueError(f"--quant must be one of {QUANT_TYPES}, got {quant!r}")

    out_dir = Path(out_dir)
    f16_path = out_dir / f"{name}-f16.gguf"
    convert_to_gguf(merged_dir, f16_path, llama_cpp_dir, outtype="f16")

    final_path = f16_path
    route = "none"
    if quant != "f16":
        binary = find_quantize_binary(_maybe_repo(llama_cpp_dir))
        if binary is not None:
            final_path = out_dir / f"{name}-{quant}.gguf"
            quantize(f16_path, final_path, quant, llama_cpp_dir)
            route = "llama-quantize"
        elif ollama_name and ollama_available():
            # ollama quantizes on ingest, so the Modelfile keeps pointing at the
            # f16 and the -q flag does the work.
            route = "ollama"
        else:
            print(
                "[ftlab] no quantizer available -- keeping f16. Build llama.cpp "
                "for a portable quantized file, or pass --ollama-name to let "
                "ollama quantize on ingest."
            )

    modelfile = write_modelfile(final_path, out_dir / "Modelfile", system_prompt)

    registered = None
    if ollama_name:
        register_with_ollama(
            modelfile, ollama_name, quantize_to=quant if route == "ollama" else None
        )
        registered = ollama_name

    return ExportResult(
        f16_path=f16_path,
        final_path=final_path,
        modelfile=modelfile,
        quant_route=route,
        quant_type=quant,
        registered_as=registered,
    )


def _maybe_repo(llama_cpp_dir: str | Path | None) -> Path | None:
    try:
        return find_llama_cpp(llama_cpp_dir)
    except FileNotFoundError:
        return None
