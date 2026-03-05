from __future__ import annotations

import math
import time
from contextlib import suppress

from modules.ocr_training.runtime.hardware_profile import collect_gpu_memory_snapshot
from modules.ocr_training.schemas import CandidateResult, CandidateStatus, TrainingCandidate

try:
    from transformers import TrainerCallback as _TrainerCallback
except Exception:  # pragma: no cover - runtime env may not satisfy trainer deps.

    class _TrainerCallback:  # type: ignore[too-many-ancestors]
        """Fallback callback base used when transformers is unavailable."""


def _torch_total_memory_mb(torch_module) -> int | None:
    """Return total CUDA memory for the active device in MiB when available."""
    if not torch_module.cuda.is_available():
        return None
    try:
        props = torch_module.cuda.get_device_properties(torch_module.cuda.current_device())
    except Exception:
        return None
    return int(props.total_memory // (1024 * 1024))


def _torch_reserved_mb(torch_module, *, peak: bool) -> int | None:
    """Return current or peak process-reserved CUDA memory in MiB."""
    if not torch_module.cuda.is_available():
        return None
    try:
        bytes_used = (
            torch_module.cuda.max_memory_reserved() if peak else torch_module.cuda.memory_reserved()
        )
    except Exception:
        return None
    return int(bytes_used // (1024 * 1024))


def _combined_current_used_memory_mb(torch_module, snapshot) -> int | None:
    """Combine host-visible GPU usage with current process-reserved CUDA memory."""
    observed = []
    if snapshot is not None:
        observed.append(int(snapshot.used_memory_mb))
    reserved = _torch_reserved_mb(torch_module, peak=False)
    if reserved is not None:
        observed.append(reserved)
    return max(observed) if observed else None


def _combined_peak_used_memory_mb(torch_module, snapshot) -> int | None:
    """Combine host-visible GPU usage with peak process-reserved CUDA memory."""
    observed = []
    current = _combined_current_used_memory_mb(torch_module, snapshot)
    if current is not None:
        observed.append(current)
    peak_reserved = _torch_reserved_mb(torch_module, peak=True)
    if peak_reserved is not None:
        observed.append(peak_reserved)
    return max(observed) if observed else None


class _BenchmarkTelemetryCallbackImpl(_TrainerCallback):
    """Concrete benchmark callback object used by the adaptive planner."""

    def __init__(
        self,
        *,
        torch_module,
        candidate: TrainingCandidate,
        warmup_steps: int,
        measure_steps: int,
        summarize_callback_state,
    ):
        self._torch = torch_module
        self._candidate = candidate
        self._warmup_steps = warmup_steps
        self._measure_steps = measure_steps
        self._summarize_callback_state = summarize_callback_state
        self._step_start: float | None = None
        self._step_times: list[float] = []
        self._losses: list[float] = []
        self._peak_vram_mb = 0
        self.invalid_gradients = False
        self.invalid_losses = False

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.perf_counter()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_start is None:
            return control
        elapsed = time.perf_counter() - self._step_start
        global_step = int(state.global_step or 0)
        if self._warmup_steps < global_step <= self._warmup_steps + self._measure_steps:
            self._step_times.append(elapsed)

        snapshot = collect_gpu_memory_snapshot(self._torch)
        used_memory_mb = _combined_peak_used_memory_mb(self._torch, snapshot)
        if used_memory_mb is not None:
            self._peak_vram_mb = max(self._peak_vram_mb, used_memory_mb)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        loss = logs.get("loss")
        if loss is not None:
            loss_value = float(loss)
            if not math.isfinite(loss_value):
                self.invalid_losses = True
            self._losses.append(loss_value)
        grad_norm = logs.get("grad_norm")
        if grad_norm is not None and not math.isfinite(float(grad_norm)):
            self.invalid_gradients = True
        return control

    def summarize(self) -> CandidateResult:
        return self._summarize_callback_state(
            self,
            self._candidate,
            self._warmup_steps,
        )


class VramPressureCallback:
    """Trainer callback that aborts when framebuffer usage gets dangerously high."""

    def __init__(
        self, *, callback_base, torch_module, usage_threshold_ratio: float, check_interval: int = 1
    ):
        """Initialize runtime GPU memory guard state."""
        self._callback_base = callback_base
        self._torch = torch_module
        self.usage_threshold_ratio = usage_threshold_ratio
        self.check_interval = max(1, check_interval)

    def build(self):
        """Create the actual callback instance bound to the configured threshold."""
        torch_module = self._torch
        usage_threshold_ratio = self.usage_threshold_ratio
        check_interval = self.check_interval

        class _GuardCallback(self._callback_base):
            def on_step_end(self, args, state, control, **kwargs):
                global_step = int(state.global_step or 0)
                if global_step % check_interval != 0:
                    return control

                snapshot = collect_gpu_memory_snapshot(torch_module)
                total_memory_mb = (
                    snapshot.total_memory_mb
                    if snapshot is not None and snapshot.total_memory_mb > 0
                    else _torch_total_memory_mb(torch_module)
                )
                used_memory_mb = _combined_current_used_memory_mb(torch_module, snapshot)
                if total_memory_mb is None or total_memory_mb <= 0 or used_memory_mb is None:
                    return control

                used_ratio = used_memory_mb / total_memory_mb
                if used_ratio < usage_threshold_ratio:
                    return control

                raise RuntimeError(
                    "VRAM guard triggered: GPU "
                    f"{getattr(snapshot, 'gpu_index', 0)} is using {used_memory_mb}MiB/"
                    f"{total_memory_mb}MiB ({used_ratio:.1%}), above the configured "
                    f"threshold of {usage_threshold_ratio:.1%}. Training was stopped before likely "
                    "shared-system-memory spillover. Free GPU memory or reduce training pressure "
                    "(LoRA/QLoRA, smaller batch, shorter sequence) and retry."
                )

        return _GuardCallback()


class BenchmarkTelemetryCallback:
    """Trainer callback that captures benchmark throughput and stability."""

    def __init__(
        self,
        *,
        callback_base,
        torch_module,
        candidate: TrainingCandidate,
        warmup_steps: int,
        measure_steps: int,
    ):
        """Initialize benchmark measurement state."""
        self._callback_base = callback_base
        self._torch = torch_module
        self.candidate = candidate
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps

    @staticmethod
    def _summarize_callback_state(callback, candidate: TrainingCandidate, warmup_steps: int):
        """Convert callback measurement state into a normalized candidate result."""
        avg_step_time = None
        throughput = None
        if callback._step_times:
            avg_step_time = sum(callback._step_times) / len(callback._step_times)
            throughput = (
                candidate.per_device_train_batch_size * candidate.gradient_accumulation_steps
            ) / avg_step_time
        average_loss = None
        if callback._losses:
            average_loss = sum(callback._losses) / len(callback._losses)
        status = CandidateStatus.COMPLETED
        reason = None
        if callback.invalid_gradients or callback.invalid_losses:
            status = CandidateStatus.INVALID
            reason = "invalid_loss_or_gradients"
        return CandidateResult(
            candidate_id=candidate.candidate_id,
            status=status,
            effective_samples_per_second=throughput,
            optimizer_step_seconds=avg_step_time,
            peak_vram_mb=callback._peak_vram_mb or None,
            average_loss=average_loss,
            invalid_gradients=callback.invalid_gradients or callback.invalid_losses,
            reason=reason,
            measured_steps=len(callback._step_times),
            warmup_steps=warmup_steps,
        )

    def build(self):
        """Build the callback instance for one benchmark run."""
        if self._torch.cuda.is_available():
            with suppress(Exception):
                self._torch.cuda.reset_peak_memory_stats()
        return _BenchmarkTelemetryCallbackImpl(
            torch_module=self._torch,
            candidate=self.candidate,
            warmup_steps=self.warmup_steps,
            measure_steps=self.measure_steps,
            summarize_callback_state=self._summarize_callback_state,
        )


class ThroughputGuardCallback:
    """Trainer callback that aborts once realized throughput falls far below plan."""

    def __init__(
        self,
        *,
        callback_base,
        candidate: TrainingCandidate,
        planned_samples_per_second: float | None,
        min_steps: int = 100,
        threshold_ratio: float = 0.70,
    ):
        """Initialize throughput guard state."""
        self._callback_base = callback_base
        self.candidate = candidate
        self.planned_samples_per_second = planned_samples_per_second
        self.min_steps = min_steps
        self.threshold_ratio = threshold_ratio

    def build(self):
        """Create the actual guard callback."""
        candidate = self.candidate
        planned_samples_per_second = self.planned_samples_per_second
        min_steps = self.min_steps
        threshold_ratio = self.threshold_ratio

        class _ThroughputCallback(self._callback_base):
            def __init__(self):
                self._start_time: float | None = None

            def on_train_begin(self, args, state, control, **kwargs):
                self._start_time = time.perf_counter()
                return control

            def on_step_end(self, args, state, control, **kwargs):
                if planned_samples_per_second is None or self._start_time is None:
                    return control
                global_step = int(state.global_step or 0)
                if global_step < min_steps:
                    return control

                elapsed = time.perf_counter() - self._start_time
                if elapsed <= 0:
                    return control
                observed_samples_per_second = (
                    global_step
                    * candidate.per_device_train_batch_size
                    * candidate.gradient_accumulation_steps
                ) / elapsed
                if observed_samples_per_second >= planned_samples_per_second * threshold_ratio:
                    return control
                raise RuntimeError(
                    "throughput_shortfall:"
                    f"observed={observed_samples_per_second:.4f},"
                    f"planned={planned_samples_per_second:.4f}"
                )

        return _ThroughputCallback()
