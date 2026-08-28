"""Environment diagnostics.

The failure this exists to catch: on a Blackwell card (RTX 50-series, sm_120) a
CPU-only or pre-CUDA-12.8 torch wheel imports cleanly, reports a GPU, and then
dies at the first matmul with "no kernel image is available for execution on the
device". So this does not merely check that CUDA is *visible* -- it runs real
kernels and a real 4-bit quantization before declaring the box healthy.
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from dataclasses import dataclass

OK = "ok"
WARN = "warn"
FAIL = "fail"

# Packages whose versions belong in any bug report about a run.
TRACKED = ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "bitsandbytes")


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def check_python() -> Check:
    version = sys.version_info
    detail = f"{platform.python_version()} ({platform.machine()})"
    if version < (3, 10):
        return Check("python", FAIL, f"{detail} -- transformers needs >= 3.10")
    return Check("python", OK, detail)


def check_packages() -> list[Check]:
    checks = []
    for package in TRACKED:
        version = _version(package)
        status = FAIL if version == "not installed" else OK
        # bitsandbytes is only required for 4-bit runs.
        if package == "bitsandbytes" and status == FAIL:
            status = WARN
        checks.append(Check(package, status, version))
    return checks


def check_cuda() -> list[Check]:
    checks: list[Check] = []
    try:
        import torch
    except ImportError as exc:
        return [Check("torch import", FAIL, str(exc))]

    built_cuda = torch.version.cuda or "none (CPU-only wheel)"
    checks.append(Check("torch build", OK if torch.version.cuda else FAIL, f"CUDA {built_cuda}"))

    if not torch.cuda.is_available():
        checks.append(
            Check(
                "cuda available",
                FAIL,
                "torch.cuda.is_available() is False -- a CPU-only wheel, or the "
                "driver is not visible to this process",
            )
        )
        return checks

    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    capability = f"sm_{props.major}{props.minor}"
    total_gb = props.total_memory / 1024**3
    free_bytes, _ = torch.cuda.mem_get_info(index)

    checks.append(Check("gpu", OK, f"{props.name} ({capability}, {total_gb:.1f} GB)"))
    checks.append(
        Check(
            "vram free",
            OK if free_bytes / 1024**3 > 6 else WARN,
            f"{free_bytes / 1024**3:.1f} GB free of {total_gb:.1f} GB "
            "(close GPU-heavy apps before a large run)",
        )
    )

    # The decisive check: does this wheel actually carry kernels for this arch?
    arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
    if arch_list:
        supported = f"sm_{props.major}{props.minor}" in arch_list
        checks.append(
            Check(
                "kernel arch",
                OK if supported else FAIL,
                f"{capability} {'in' if supported else 'MISSING from'} wheel arch list "
                f"({', '.join(arch_list)})",
            )
        )

    checks.append(_check_matmul())
    checks.append(_check_bf16())
    return checks


def _check_matmul() -> Check:
    """Run a real fp16 matmul and a backward pass on the GPU."""
    import torch

    try:
        a = torch.randn(512, 512, device="cuda", dtype=torch.float16, requires_grad=True)
        b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
        (a @ b).sum().backward()
        torch.cuda.synchronize()
        return Check("gpu matmul+backward", OK, "fp16 512x512 kernel executed")
    except Exception as exc:  # noqa: BLE001 - the message is the whole point
        return Check("gpu matmul+backward", FAIL, f"{type(exc).__name__}: {exc}")


def _check_bf16() -> Check:
    import torch

    if torch.cuda.is_bf16_supported():
        return Check("bfloat16", OK, "supported")
    return Check("bfloat16", WARN, "unsupported -- set model.dtype: float16")


def check_bitsandbytes() -> Check:
    """Quantize a real tensor to 4-bit and read it back."""
    try:
        import bitsandbytes as bnb
        import torch
    except ImportError as exc:
        return Check("bitsandbytes 4-bit", WARN, f"not usable: {exc}")

    if not torch.cuda.is_available():
        return Check("bitsandbytes 4-bit", WARN, "skipped -- no CUDA device")

    try:
        weight = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        quantized, state = bnb.functional.quantize_4bit(weight, quant_type="nf4")
        restored = bnb.functional.dequantize_4bit(quantized, state)
        error = (restored.float() - weight.float()).abs().mean().item()
        return Check("bitsandbytes 4-bit", OK, f"nf4 round-trip, mean abs err {error:.4f}")
    except Exception as exc:  # noqa: BLE001
        return Check("bitsandbytes 4-bit", FAIL, f"{type(exc).__name__}: {exc}")


def run_all() -> list[Check]:
    checks = [check_python(), *check_packages(), *check_cuda(), check_bitsandbytes()]
    return checks


def render(checks: list[Check]) -> int:
    """Print a report; return a process exit code."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="ftlab environment", show_lines=False)
        table.add_column("check", style="bold")
        table.add_column("status")
        table.add_column("detail", overflow="fold")
        colors = {OK: "green", WARN: "yellow", FAIL: "red"}
        for check in checks:
            table.add_row(check.name, f"[{colors[check.status]}]{check.status}[/]", check.detail)
        console.print(table)
    except ImportError:
        for check in checks:
            print(f"{check.status.upper():5} {check.name:24} {check.detail}")

    failures = [c for c in checks if c.status == FAIL]
    if failures:
        print(f"\n{len(failures)} check(s) failed: {', '.join(c.name for c in failures)}")
        return 1
    print("\nEnvironment looks good.")
    return 0


def main() -> int:
    return render(run_all())


if __name__ == "__main__":
    raise SystemExit(main())
