from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from modules.ocr_training.checkpointing import atomic_write_json
from modules.ocr_training.schemas import CandidateResult, CandidateStatus, TrainingCandidate

BenchmarkFn = Callable[[TrainingCandidate], CandidateResult]


@dataclass(frozen=True)
class AutotuneSelection:
    """Adaptive planner result bundle."""

    selected_candidate: TrainingCandidate
    candidate_results: list[CandidateResult]
    discarded_candidates: int
    selection_reason: str


def _write_candidate_results(path: Path, candidate_results: list[CandidateResult]) -> None:
    """Persist candidate results as JSONL for auditability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in candidate_results:
            handle.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n")


def run_candidate_benchmarks(
    *,
    candidates: list[TrainingCandidate],
    benchmark_fn: BenchmarkFn,
    planning_budget_minutes: int,
    output_dir: Path,
    is_rank_zero: bool,
    logger,
) -> list[CandidateResult]:
    """Benchmark candidates until the planning budget is exhausted."""
    deadline = time.monotonic() + max(1, planning_budget_minutes) * 60
    results: list[CandidateResult] = []
    for candidate in candidates:
        if time.monotonic() >= deadline:
            break
        logger.info(
            "Benchmarking candidate=%s strategy=%s batch=%d grad_accum=%d seq=%d workers=%d gc=%s",
            candidate.candidate_id,
            candidate.finetune_strategy.value,
            candidate.per_device_train_batch_size,
            candidate.gradient_accumulation_steps,
            candidate.max_sequence_length,
            candidate.dataloader_num_workers,
            candidate.gradient_checkpointing,
        )
        result = benchmark_fn(candidate)
        logger.info(
            "Benchmark result candidate=%s status=%s throughput=%s peak_vram=%sMiB reason=%s",
            candidate.candidate_id,
            result.status.value,
            (
                f"{result.effective_samples_per_second:.4f}"
                if result.effective_samples_per_second is not None
                else "n/a"
            ),
            result.peak_vram_mb if result.peak_vram_mb is not None else "n/a",
            result.reason or "none",
        )
        results.append(result)

    if is_rank_zero:
        _write_candidate_results(output_dir / "candidate_results.jsonl", results)
    return results


def select_best_candidate(
    *,
    candidates: list[TrainingCandidate],
    candidate_results: list[CandidateResult],
    output_dir: Path,
    safe_peak_vram_mb: int | None = None,
    is_rank_zero: bool = True,
) -> AutotuneSelection:
    """Choose the best admissible candidate by measured throughput."""
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    valid_results = [
        result
        for result in candidate_results
        if result.status == CandidateStatus.COMPLETED
        and not result.invalid_gradients
        and result.effective_samples_per_second is not None
    ]
    if not valid_results:
        raise RuntimeError("Adaptive planner could not find a valid training candidate.")

    safe_results = valid_results
    selection_reason = "highest_measured_samples_per_second"
    if safe_peak_vram_mb is not None:
        bounded_results = [
            result
            for result in valid_results
            if result.peak_vram_mb is not None and result.peak_vram_mb <= safe_peak_vram_mb
        ]
        if bounded_results:
            safe_results = bounded_results
            selection_reason = "highest_measured_samples_per_second_with_vram_headroom"

    ranked_results = sorted(
        safe_results,
        key=lambda result: float(result.effective_samples_per_second or 0.0),
        reverse=True,
    )
    winner = ranked_results[0]
    selection = AutotuneSelection(
        selected_candidate=candidate_by_id[winner.candidate_id],
        candidate_results=candidate_results,
        discarded_candidates=max(0, len(candidate_results) - len(valid_results)),
        selection_reason=selection_reason,
    )
    if is_rank_zero:
        atomic_write_json(
            output_dir / "selected_training_config.json",
            {
                **selection.selected_candidate.model_dump(mode="json"),
                "selection_reason": selection.selection_reason,
                "measured_samples_per_second": winner.effective_samples_per_second,
                "measured_peak_vram_mb": winner.peak_vram_mb,
            },
        )
    return selection
