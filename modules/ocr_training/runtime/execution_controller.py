from __future__ import annotations

from dataclasses import dataclass

from modules.ocr_training.schemas import CandidateResult, CandidateStatus, TrainingCandidate


@dataclass(frozen=True)
class ThroughputShortfallError:
    """Parsed throughput shortfall marker emitted by the training callback."""

    observed_samples_per_second: float
    planned_samples_per_second: float


def parse_throughput_shortfall(message: str) -> ThroughputShortfallError | None:
    """Parse a throughput shortfall runtime marker into structured data."""
    prefix = "throughput_shortfall:"
    if not message.startswith(prefix):
        return None
    payload = message.removeprefix(prefix)
    parts = dict(item.split("=", 1) for item in payload.split(",") if "=" in item)
    return ThroughputShortfallError(
        observed_samples_per_second=float(parts["observed"]),
        planned_samples_per_second=float(parts["planned"]),
    )


def should_replan_for_throughput(message: str) -> bool:
    """Return whether a runtime failure should trigger throughput replanning."""
    return parse_throughput_shortfall(message) is not None


def should_retry_for_memory_pressure(message: str) -> bool:
    """Return whether a runtime failure is recoverable via lower-pressure retry."""
    lowered = message.lower()
    return "out of memory" in lowered or "vram guard triggered" in lowered


def _ranked_completed_candidates(
    candidate_results: list[CandidateResult],
    candidate_by_id: dict[str, TrainingCandidate],
) -> list[TrainingCandidate]:
    """Return candidates ranked by measured throughput."""
    ranked_results = sorted(
        (
            result
            for result in candidate_results
            if result.status == CandidateStatus.COMPLETED
            and result.effective_samples_per_second is not None
            and result.candidate_id in candidate_by_id
        ),
        key=lambda item: float(item.effective_samples_per_second or 0.0),
        reverse=True,
    )
    return [candidate_by_id[result.candidate_id] for result in ranked_results]


def _first_unattempted(
    candidates: list[TrainingCandidate],
    attempted_candidate_ids: set[str],
) -> TrainingCandidate | None:
    """Return the first candidate not already attempted."""
    for candidate in candidates:
        if candidate.candidate_id not in attempted_candidate_ids:
            return candidate
    return None


def _fallback_same_strategy(
    *,
    all_candidates: list[TrainingCandidate],
    current_candidate: TrainingCandidate,
) -> list[TrainingCandidate]:
    """Return same-strategy fallback candidates in priority order."""
    lower_batch_same_strategy = sorted(
        (
            candidate
            for candidate in all_candidates
            if candidate.finetune_strategy == current_candidate.finetune_strategy
            and candidate.max_sequence_length == current_candidate.max_sequence_length
            and candidate.per_device_train_batch_size
            < current_candidate.per_device_train_batch_size
        ),
        key=lambda candidate: (
            -candidate.per_device_train_batch_size,
            candidate.gradient_accumulation_steps,
        ),
    )
    return lower_batch_same_strategy


def choose_retry_candidate(
    *,
    current_candidate: TrainingCandidate,
    candidate_results: list[CandidateResult],
    all_candidates: list[TrainingCandidate],
    attempted_candidate_ids: set[str],
    reason: str,
) -> TrainingCandidate | None:
    """Choose the next admissible retry candidate after a runtime failure."""
    candidate_by_id = {candidate.candidate_id: candidate for candidate in all_candidates}

    if should_replan_for_throughput(reason):
        return _first_unattempted(
            _ranked_completed_candidates(candidate_results, candidate_by_id),
            attempted_candidate_ids,
        )

    same_strategy = _fallback_same_strategy(
        all_candidates=all_candidates,
        current_candidate=current_candidate,
    )
    retry = _first_unattempted(same_strategy, attempted_candidate_ids)
    if retry is not None:
        return retry

    if current_candidate.finetune_strategy.value != "lora":
        return None

    qlora_candidates = sorted(
        (candidate for candidate in all_candidates if candidate.finetune_strategy.value == "qlora"),
        key=lambda candidate: (
            -candidate.max_sequence_length,
            -candidate.per_device_train_batch_size,
        ),
    )
    return _first_unattempted(qlora_candidates, attempted_candidate_ids)
