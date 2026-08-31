"""Keeping a run inside physical VRAM, and knowing when it is not.

Shared because the failure is shared. Both trainers have hit it: the
supervised one went from 12.8 s/step to 204 s/step, and GRPO from 107 s to
499 s. Neither logged an out-of-memory error. The allocator simply spilled
and everything got slower, which is a far worse outcome than a crash --
you find a crash in a minute and you find this one in an afternoon.
"""

from __future__ import annotations

from typing import Any


def build_memory_guard(every: int = 25) -> Any:
    """A callback that returns fragmented cache to the driver, periodically.

    Windows lets this fail quietly, which is why the callback exists. Under WDDM
    the driver backs a CUDA reservation with system RAM once VRAM runs out
    instead of raising, so a run that would OOM on Linux keeps going at PCIe
    speed. Observed on this box: a 12B QLoRA run held at 28.1 GB for seventy
    steps, drifted to the 32 GB ceiling as the allocator fragmented, then went
    from 12.8 s/step to 204 s/step with power draw collapsing from ~400 W to
    141 W at a reported 97% utilisation -- the GPU waiting on transfers rather
    than computing. Nothing in the log said "out of memory"; it just got 16x
    slower.

    empty_cache() releases cached blocks that no longer fit anything, which lets
    the allocator settle instead of reserving more. Built by a factory because
    it has to subclass TrainerCallback -- a duck-typed object fails on the first
    event it does not implement -- and transformers is imported lazily here to
    keep CLI startup fast.
    """
    from transformers import TrainerCallback

    class MemoryGuard(TrainerCallback):
        def __init__(self, every: int) -> None:
            self.every = every

        def _release(self) -> None:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - housekeeping must never kill a run
                pass

        def on_evaluate(self, args, state, control, **kwargs):  # noqa: ANN001
            self._release()

        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
            if self.every and state.global_step % self.every == 0:
                self._release()

    return MemoryGuard(every)


def cap_memory_fraction(fraction: float = 0.92) -> None:
    """Keep the process inside physical VRAM so failure is loud, not slow.

    Without a cap, exhausting VRAM on Windows degrades into system-memory
    spilling. With one, the run raises instead -- and an OOM traceback is a far
    better outcome than a run that silently takes five hours.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
    except Exception:  # noqa: BLE001
        pass
