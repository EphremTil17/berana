from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedContext:
    """Normalized execution context for single-process or DDP runs."""

    execution_backend: str
    ddp_backend: str | None
    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: str
    is_rank_zero: bool


class RankZeroLogger:
    """Logger proxy that suppresses non-error chatter on nonzero ranks."""

    def __init__(self, logger, *, is_rank_zero: bool):
        """Wrap one logger and keep only rank-zero informational output."""
        self._logger = logger
        self._is_rank_zero = is_rank_zero

    def debug(self, *args, **kwargs):
        """Emit debug logs only on rank zero."""
        if self._is_rank_zero:
            return self._logger.debug(*args, **kwargs)
        return None

    def info(self, *args, **kwargs):
        """Emit info logs only on rank zero."""
        if self._is_rank_zero:
            return self._logger.info(*args, **kwargs)
        return None

    def warning(self, *args, **kwargs):
        """Emit warnings only on rank zero."""
        if self._is_rank_zero:
            return self._logger.warning(*args, **kwargs)
        return None

    def error(self, *args, **kwargs):
        """Always emit error logs on every rank."""
        return self._logger.error(*args, **kwargs)

    def exception(self, *args, **kwargs):
        """Always emit exception logs on every rank."""
        return self._logger.exception(*args, **kwargs)

    def __getattr__(self, name: str):
        """Delegate unknown attributes to the wrapped logger."""
        return getattr(self._logger, name)


def _torchrun_world_size() -> int:
    world_size = os.environ.get("WORLD_SIZE", "").strip()
    return int(world_size) if world_size.isdigit() else 1


def _torchrun_rank(name: str) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value.isdigit() else 0


def torchrun_is_active() -> bool:
    """Return whether the current process is running under torchrun-style env."""
    return _torchrun_world_size() > 1 and "LOCAL_RANK" in os.environ and "RANK" in os.environ


def resolve_execution_backend(*, torch_module, requested_backend: str) -> str:
    """Resolve the effective execution backend from CLI request and environment."""
    normalized = requested_backend.strip().lower()
    visible_cuda = int(torch_module.cuda.device_count()) if torch_module.cuda.is_available() else 0
    if normalized == "auto":
        if torchrun_is_active():
            return "ddp"
        if visible_cuda > 1:
            raise RuntimeError(
                "Multiple CUDA devices are visible, but distributed env vars are absent. "
                "Use `--multi-gpu` on the normal train command, or launch explicitly with "
                "`torchrun --standalone --nproc_per_node=<gpu_count> tools/ocr_training.py "
                "train-surya ...`."
            )
        return "single"
    if normalized == "ddp":
        if not torchrun_is_active():
            raise RuntimeError(
                "`--execution-backend ddp` requires a torchrun launch with WORLD_SIZE, RANK, "
                "and LOCAL_RANK set. Prefer `--multi-gpu` on the normal train command unless "
                "you are debugging the launcher layer."
            )
        return "ddp"
    if normalized != "single":
        raise RuntimeError(f"Unsupported execution backend: {requested_backend}")
    if torchrun_is_active():
        raise RuntimeError(
            "`--execution-backend single` is incompatible with an active torchrun environment."
        )
    return "single"


def initialize_distributed_context(
    *,
    torch_module,
    requested_backend: str,
    ddp_backend: str,
) -> DistributedContext:
    """Initialize and return distributed execution context for the current process."""
    execution_backend = resolve_execution_backend(
        torch_module=torch_module,
        requested_backend=requested_backend,
    )
    if execution_backend != "ddp":
        device = (
            f"cuda:{torch_module.cuda.current_device()}"
            if torch_module.cuda.is_available()
            else "cpu"
        )
        return DistributedContext(
            execution_backend=execution_backend,
            ddp_backend=None,
            is_distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=device,
            is_rank_zero=True,
        )

    local_rank = _torchrun_rank("LOCAL_RANK")
    rank = _torchrun_rank("RANK")
    world_size = _torchrun_world_size()
    if torch_module.cuda.is_available():
        torch_module.cuda.set_device(local_rank)
    if (
        hasattr(torch_module, "distributed")
        and torch_module.distributed.is_available()
        and not torch_module.distributed.is_initialized()
    ):
        try:
            torch_module.distributed.init_process_group(
                backend=ddp_backend,
                device_id=local_rank if torch_module.cuda.is_available() else None,
            )
        except TypeError:
            torch_module.distributed.init_process_group(backend=ddp_backend)
    device = f"cuda:{local_rank}" if torch_module.cuda.is_available() else "cpu"
    return DistributedContext(
        execution_backend="ddp",
        ddp_backend=ddp_backend,
        is_distributed=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_rank_zero=(rank == 0),
    )


def destroy_distributed_context(*, torch_module, context: DistributedContext) -> None:
    """Tear down the process group for DDP runs if this process initialized it."""
    if not context.is_distributed:
        return
    if not hasattr(torch_module, "distributed") or not torch_module.distributed.is_available():
        return
    if torch_module.distributed.is_initialized():
        torch_module.distributed.destroy_process_group()


def maybe_barrier(*, torch_module, context: DistributedContext) -> None:
    """Synchronize ranks when running under DDP."""
    if not context.is_distributed:
        return
    if hasattr(torch_module, "distributed") and torch_module.distributed.is_initialized():
        try:
            if torch_module.cuda.is_available():
                torch_module.distributed.barrier(device_ids=[context.local_rank])
                return
        except TypeError:
            pass
        torch_module.distributed.barrier()
