from __future__ import annotations

import json
import shutil
from contextlib import suppress
from pathlib import Path
from typing import Any

from modules.ocr_training.checkpointing import atomic_write_json
from modules.ocr_training.registry import STAGE_SURYA_FINETUNE, register_training_stage
from modules.ocr_training.schemas import HardwareProfile, TrainingCandidate, TrainMode
from modules.ocr_training.surya_common import relative_to_base


def finetune_meta_path(run_dir: Path) -> Path:
    """Return the persisted finetune metadata path for one run."""
    return run_dir / "finetune_meta.json"


def write_finetune_meta(
    run_dir: Path, payload: dict[str, Any], *, is_rank_zero: bool = True
) -> Path | None:
    """Persist adapter/base-model metadata for resume and evaluation."""
    if not is_rank_zero:
        return None
    meta_path = finetune_meta_path(run_dir)
    atomic_write_json(meta_path, payload)
    return meta_path


def load_finetune_meta(run_dir: Path) -> dict[str, Any] | None:
    """Load finetune metadata if the run directory already has it."""
    meta_path = finetune_meta_path(run_dir)
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def candidate_output_dir(output_dir: Path, candidate: TrainingCandidate) -> Path:
    """Return the scratch output directory used for one benchmark candidate."""
    return output_dir / ".autotune" / candidate.candidate_id


def write_hardware_profile(
    output_dir: Path, profile: HardwareProfile, *, is_rank_zero: bool = True
) -> Path | None:
    """Persist normalized hardware profile for this run."""
    if not is_rank_zero:
        return None
    path = output_dir / "hardware_profile.json"
    atomic_write_json(path, profile.model_dump(mode="json"))
    return path


def write_autotune_plan(
    *,
    output_dir: Path,
    profile: HardwareProfile,
    constraints,
    candidates: list[TrainingCandidate],
    config,
    resumed_selection: bool,
    is_rank_zero: bool = True,
) -> Path:
    """Persist adaptive planner inputs before benchmarking or execution."""
    if not is_rank_zero:
        return output_dir / "autotune_plan.json"
    notes: list[str] = []
    if config.per_device_train_batch_size is not None:
        notes.append("per_device_train_batch_size treated as auto ceiling")
    if config.gradient_accumulation_steps is not None:
        notes.append("gradient_accumulation_steps treated as auto ceiling")
    if config.max_sequence_length is not None:
        notes.append("max_sequence_length treated as auto ceiling")
    if config.dataloader_num_workers is not None:
        notes.append("dataloader_num_workers treated as auto ceiling")

    path = output_dir / "autotune_plan.json"
    atomic_write_json(
        path,
        {
            "schema_version": "1.0",
            "mode": config.mode.value,
            "seed": config.seed,
            "train_fraction": config.train_fraction,
            "planning_budget_minutes": config.planning_budget_minutes,
            "target_vram_utilization": config.target_vram_utilization,
            "throughput_metric": config.throughput_metric,
            "warmup_steps_per_candidate": config.warmup_steps_per_candidate,
            "measure_steps_per_candidate": config.measure_steps_per_candidate,
            "max_replans": config.max_replans,
            "strategy_allowlist": [strategy.value for strategy in config.strategy_allowlist],
            "constraints": constraints.__dict__,
            "hardware_profile": profile.model_dump(mode="json"),
            "candidate_ids": [candidate.candidate_id for candidate in candidates],
            "candidate_count": len(candidates),
            "resumed_selection": resumed_selection,
            "notes": notes,
        },
    )
    return path


def load_selected_candidate(output_dir: Path) -> TrainingCandidate | None:
    """Load persisted adaptive candidate selection if it exists."""
    path = output_dir / "selected_training_config.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed_fields = set(TrainingCandidate.model_fields)
    filtered = {key: value for key, value in payload.items() if key in allowed_fields}
    return TrainingCandidate.model_validate(filtered)


def cleanup_candidate_scratch(output_dir: Path, candidate: TrainingCandidate) -> None:
    """Remove one benchmark candidate scratch directory if it exists."""
    path = candidate_output_dir(output_dir, candidate)
    if path.exists():
        with suppress(FileNotFoundError):
            shutil.rmtree(path)


def reset_training_outputs(output_dir: Path) -> None:
    """Remove generated training artifacts before a deterministic retry."""
    for checkpoint_dir in output_dir.glob("checkpoint-*"):
        if checkpoint_dir.is_dir():
            with suppress(FileNotFoundError):
                shutil.rmtree(checkpoint_dir)
    for relative in (".autotune", "weights"):
        target = output_dir / relative
        if target.exists():
            with suppress(FileNotFoundError):
                shutil.rmtree(target)
    for relative in (
        "resume_state.json",
        "best_model_meta.json",
        "trainer_state.json",
        "training_args.bin",
    ):
        target = output_dir / relative
        if target.exists():
            with suppress(FileNotFoundError):
                target.unlink()


def register_completed_finetune(  # noqa: C901
    *,
    run_key: str,
    output_dir: Path,
    attempts: list[dict[str, Any]],
    selected_candidate: TrainingCandidate,
    train_count: int,
    val_count: int,
    latest_checkpoint: Path | None,
    mode: TrainMode,
    selection_reason: str,
    discarded_candidates: int,
    retry_count: int,
    original_train_count: int,
    train_fraction: float,
    train_subset_seed: int,
    is_rank_zero: bool = True,
) -> dict[str, Any]:
    """Register a completed finetune stage with planner metadata."""
    if not is_rank_zero:
        return {
            "status": "completed",
            "mode": mode.value,
            "selected_strategy": selected_candidate.finetune_strategy.value,
            "selected_candidate_id": selected_candidate.candidate_id,
            "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
        }
    artifacts = {
        "resume_state": relative_to_base(output_dir / "resume_state.json"),
        "finetune_meta": relative_to_base(finetune_meta_path(output_dir)),
    }
    for name in (
        "hardware_profile.json",
        "autotune_plan.json",
        "candidate_results.jsonl",
        "selected_training_config.json",
        "training_summary.json",
    ):
        artifact_path = output_dir / name
        if artifact_path.exists():
            artifacts[name.replace(".jsonl", "").replace(".json", "")] = relative_to_base(
                artifact_path
            )
    if latest_checkpoint:
        artifacts["latest_checkpoint"] = relative_to_base(latest_checkpoint)
    for name in ("best_model_meta.json", "best_wer_model_meta.json"):
        best_meta = output_dir / name
        if best_meta.exists():
            artifacts[name.replace(".json", "")] = relative_to_base(best_meta)
    manifests_dir = output_dir / "manifests"
    for name in ("train_subset_manifest.jsonl", "eval_subset_manifest.jsonl"):
        artifact_path = manifests_dir / name
        if artifact_path.exists():
            artifacts[name.replace(".jsonl", "")] = relative_to_base(artifact_path)
    eval_dir = output_dir / "evaluation"
    for name in (
        "training_history.csv",
        "training_history.jsonl",
        "training_curves.svg",
        "training_curves.png",
        "training_report.md",
    ):
        artifact_path = eval_dir / name
        if artifact_path.exists():
            artifacts[name.replace(".", "_")] = relative_to_base(artifact_path)

    register_training_stage(
        stage=STAGE_SURYA_FINETUNE,
        run_key=run_key,
        run_dir=output_dir,
        artifacts=artifacts,
        metadata={
            "status": "completed",
            "attempts": attempts,
            "mode": mode.value,
            "selected_strategy": selected_candidate.finetune_strategy.value,
            "selected_config": selected_candidate.model_dump(mode="json"),
            "selection_reason": selection_reason,
            "discarded_candidates": discarded_candidates,
            "retry_count": retry_count,
            "original_train_rows": original_train_count,
            "train_rows": train_count,
            "train_fraction": train_fraction,
            "train_subset_seed": train_subset_seed,
            "val_rows": val_count,
        },
    )
    return {
        "status": "completed",
        "mode": mode.value,
        "selected_strategy": selected_candidate.finetune_strategy.value,
        "selected_candidate_id": selected_candidate.candidate_id,
        "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
    }


def register_interrupted_finetune(
    *,
    run_key: str,
    output_dir: Path,
    attempts: list[dict[str, Any]],
    selected_candidate: TrainingCandidate,
    train_count: int,
    val_count: int,
    resume_state_path: Path,
    emergency_dir: Path,
    mode: TrainMode,
    selection_reason: str,
    discarded_candidates: int,
    retry_count: int,
    original_train_count: int,
    train_fraction: float,
    train_subset_seed: int,
    is_rank_zero: bool = True,
) -> dict[str, Any]:
    """Register an interrupted finetune stage with planner metadata."""
    if not is_rank_zero:
        return {
            "status": "interrupted",
            "mode": mode.value,
            "selected_strategy": selected_candidate.finetune_strategy.value,
            "selected_candidate_id": selected_candidate.candidate_id,
        }
    artifacts = {
        "resume_state": relative_to_base(resume_state_path),
        "emergency_checkpoint": relative_to_base(emergency_dir),
        "finetune_meta": relative_to_base(finetune_meta_path(output_dir)),
    }
    training_summary = output_dir / "training_summary.json"
    if training_summary.exists():
        artifacts["training_summary"] = relative_to_base(training_summary)
    for name in ("best_model_meta.json", "best_wer_model_meta.json"):
        best_meta = output_dir / name
        if best_meta.exists():
            artifacts[name.replace(".json", "")] = relative_to_base(best_meta)
    manifests_dir = output_dir / "manifests"
    for name in ("train_subset_manifest.jsonl", "eval_subset_manifest.jsonl"):
        artifact_path = manifests_dir / name
        if artifact_path.exists():
            artifacts[name.replace(".jsonl", "")] = relative_to_base(artifact_path)
    eval_dir = output_dir / "evaluation"
    for name in (
        "training_history.csv",
        "training_history.jsonl",
        "training_curves.svg",
        "training_curves.png",
        "training_report.md",
    ):
        artifact_path = eval_dir / name
        if artifact_path.exists():
            artifacts[name.replace(".", "_")] = relative_to_base(artifact_path)
    register_training_stage(
        stage=STAGE_SURYA_FINETUNE,
        run_key=run_key,
        run_dir=output_dir,
        artifacts=artifacts,
        metadata={
            "status": "interrupted",
            "attempts": attempts,
            "mode": mode.value,
            "selected_strategy": selected_candidate.finetune_strategy.value,
            "selected_config": selected_candidate.model_dump(mode="json"),
            "selection_reason": selection_reason,
            "discarded_candidates": discarded_candidates,
            "retry_count": retry_count,
            "original_train_rows": original_train_count,
            "train_rows": train_count,
            "train_fraction": train_fraction,
            "train_subset_seed": train_subset_seed,
            "val_rows": val_count,
        },
    )
    return {
        "status": "interrupted",
        "mode": mode.value,
        "selected_strategy": selected_candidate.finetune_strategy.value,
        "selected_candidate_id": selected_candidate.candidate_id,
    }
