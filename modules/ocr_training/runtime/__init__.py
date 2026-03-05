"""Adaptive Surya OCR training runtime helpers."""

from modules.ocr_training.runtime.autotune_runner import (
    AutotuneSelection,
    run_candidate_benchmarks,
    select_best_candidate,
)
from modules.ocr_training.runtime.candidate_builder import (
    AutoTuneConstraints,
    build_training_candidates,
    derive_auto_constraints,
    materialize_manual_candidate,
)
from modules.ocr_training.runtime.execution_controller import (
    ThroughputShortfallError,
    choose_retry_candidate,
    should_replan_for_throughput,
)
from modules.ocr_training.runtime.hardware_profile import (
    GpuMemorySnapshot,
    GpuProcessUsage,
    collect_gpu_memory_snapshot,
    detect_hardware_profile,
    enforce_gpu_preflight,
    enforce_single_gpu,
)
from modules.ocr_training.runtime.strategy_catalog import (
    MANUAL_DEFAULTS,
    resolve_finetune_strategy,
    resolve_strategy_allowlist,
    strategy_is_auto_admissible,
)
from modules.ocr_training.runtime.telemetry import (
    BenchmarkTelemetryCallback,
    ThroughputGuardCallback,
    VramPressureCallback,
)

__all__ = [
    "MANUAL_DEFAULTS",
    "AutoTuneConstraints",
    "AutotuneSelection",
    "BenchmarkTelemetryCallback",
    "GpuMemorySnapshot",
    "GpuProcessUsage",
    "ThroughputGuardCallback",
    "ThroughputShortfallError",
    "VramPressureCallback",
    "build_training_candidates",
    "choose_retry_candidate",
    "collect_gpu_memory_snapshot",
    "derive_auto_constraints",
    "detect_hardware_profile",
    "enforce_gpu_preflight",
    "enforce_single_gpu",
    "materialize_manual_candidate",
    "resolve_finetune_strategy",
    "resolve_strategy_allowlist",
    "run_candidate_benchmarks",
    "select_best_candidate",
    "should_replan_for_throughput",
    "strategy_is_auto_admissible",
]
