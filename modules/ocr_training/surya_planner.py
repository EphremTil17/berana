from __future__ import annotations

from modules.ocr_training.checkpointing import atomic_write_json
from modules.ocr_training.runtime.autotune_runner import (
    run_candidate_benchmarks,
    select_best_candidate,
)
from modules.ocr_training.runtime.candidate_builder import (
    build_training_candidates,
    derive_auto_constraints,
    materialize_manual_candidate,
)
from modules.ocr_training.runtime.execution_controller import (
    choose_retry_candidate,
    should_replan_for_throughput,
    should_retry_for_memory_pressure,
)
from modules.ocr_training.surya_artifacts import (
    load_selected_candidate,
    reset_training_outputs,
    write_autotune_plan,
)


def _safe_peak_vram_budget_mb(hardware_profile, config) -> int | None:
    """Return a benchmark peak-VRAM budget that leaves headroom below the guard."""
    total_vram_mb = hardware_profile.total_vram_mb
    if total_vram_mb is None or total_vram_mb <= 0:
        return None
    guard_budget_mb = int(total_vram_mb * float(config.target_vram_utilization))
    headroom_mb = max(384, int(total_vram_mb * 0.05))
    safe_budget_mb = guard_budget_mb - headroom_mb
    if safe_budget_mb <= 0:
        return None
    return safe_budget_mb


def prepare_auto_training(
    *,
    runtime,
    output_dir,
    config,
    base_checkpoint: str,
    train_rows,
    hardware_profile,
    benchmark_candidate,
    is_rank_zero: bool,
    logger,
):
    """Plan and optionally benchmark adaptive candidates for the current host."""
    existing_selected_candidate = (
        load_selected_candidate(output_dir) if config.resume.lower() != "none" else None
    )
    constraints = derive_auto_constraints(config, hardware_profile)
    candidate_pool = build_training_candidates(
        profile=hardware_profile,
        config=config,
        constraints=constraints,
    )
    if not candidate_pool:
        raise RuntimeError(
            "Adaptive planner could not derive any admissible training candidates for "
            f"device={hardware_profile.device_type} allowlist="
            f"{[strategy.value for strategy in config.strategy_allowlist]}."
        )
    write_autotune_plan(
        output_dir=output_dir,
        profile=hardware_profile,
        constraints=constraints,
        candidates=candidate_pool,
        config=config,
        resumed_selection=existing_selected_candidate is not None,
        is_rank_zero=is_rank_zero,
    )
    if existing_selected_candidate is not None:
        return (
            existing_selected_candidate,
            candidate_pool,
            [],
            "resume_existing_selected_candidate",
            0,
            existing_selected_candidate.expected_samples_per_second,
        )

    logger.info(
        "Adaptive planner detected gpu=%s vram=%sMiB cpu=%d and will benchmark %d candidates.",
        hardware_profile.gpu_name,
        hardware_profile.total_vram_mb,
        hardware_profile.cpu_count,
        len(candidate_pool),
    )
    candidate_results = run_candidate_benchmarks(
        candidates=candidate_pool,
        benchmark_fn=lambda candidate: benchmark_candidate(
            runtime=runtime,
            output_dir=output_dir,
            base_checkpoint=base_checkpoint,
            config=config,
            train_rows=train_rows,
            candidate=candidate,
        ),
        planning_budget_minutes=config.planning_budget_minutes,
        output_dir=output_dir,
        is_rank_zero=is_rank_zero,
        logger=logger,
    )
    selection = select_best_candidate(
        candidates=candidate_pool,
        candidate_results=candidate_results,
        output_dir=output_dir,
        safe_peak_vram_mb=_safe_peak_vram_budget_mb(hardware_profile, config),
        is_rank_zero=is_rank_zero,
    )
    winner = next(
        result
        for result in selection.candidate_results
        if result.candidate_id == selection.selected_candidate.candidate_id
    )
    selected_candidate = selection.selected_candidate.model_copy(
        update={"expected_samples_per_second": winner.effective_samples_per_second}
    )
    if is_rank_zero:
        atomic_write_json(
            output_dir / "selected_training_config.json",
            {
                **selected_candidate.model_dump(mode="json"),
                "selection_reason": selection.selection_reason,
                "measured_samples_per_second": winner.effective_samples_per_second,
                "measured_peak_vram_mb": winner.peak_vram_mb,
            },
        )
    logger.info(
        "Adaptive planner selected candidate=%s strategy=%s throughput=%.4f samples/s peak_vram=%sMiB",
        selected_candidate.candidate_id,
        selected_candidate.finetune_strategy.value,
        winner.effective_samples_per_second or 0.0,
        winner.peak_vram_mb,
    )
    return (
        selected_candidate,
        candidate_pool,
        candidate_results,
        selection.selection_reason,
        selection.discarded_candidates,
        winner.effective_samples_per_second,
    )


def run_manual_training(*, config) -> tuple:
    """Materialize the explicit manual-mode candidate and metadata."""
    selected_candidate = materialize_manual_candidate(config)
    return selected_candidate, "manual_mode", 0, None


def run_auto_with_fallback(
    *,
    initial_candidate,
    candidate_pool,
    candidate_results,
    selection_reason: str,
    planned_samples_per_second: float | None,
    config,
    output_dir,
    attempts,
    runner,
    logger,
):
    """Run the selected adaptive candidate with one deterministic fallback budget."""
    attempted_candidate_ids = {initial_candidate.candidate_id}
    selected_candidate = initial_candidate
    retry_count = 0
    while True:
        try:
            return runner(
                selected_candidate=selected_candidate,
                selection_reason=selection_reason,
                retry_count=retry_count,
                planned_samples_per_second=planned_samples_per_second,
            )
        except RuntimeError as exc:
            message = str(exc)
            memory_pressure = should_retry_for_memory_pressure(message)
            attempts.append(
                {
                    "event": "runtime_failure",
                    "candidate_id": selected_candidate.candidate_id,
                    "error": message,
                }
            )
            if not memory_pressure and retry_count >= config.max_replans:
                raise
            retry_candidate = choose_retry_candidate(
                current_candidate=selected_candidate,
                candidate_results=candidate_results,
                all_candidates=candidate_pool,
                attempted_candidate_ids=attempted_candidate_ids,
                reason=message,
            )
            if retry_candidate is None:
                raise
            retry_count += 1
            attempted_candidate_ids.add(retry_candidate.candidate_id)
            reset_training_outputs(output_dir)
            logger.warning(
                "Adaptive planner is retrying with candidate=%s after reason=%s",
                retry_candidate.candidate_id,
                message,
            )
            selected_candidate = retry_candidate
            selection_reason = (
                "throughput_replan"
                if should_replan_for_throughput(message)
                else "memory_fallback"
                if memory_pressure
                else "runtime_fallback"
            )
            planned_samples_per_second = retry_candidate.expected_samples_per_second
