from __future__ import annotations

from pathlib import Path
from typing import Any

from modules.ocr_training.schemas import TrainingCandidate
from modules.ocr_training.surya_common import (
    bounded_worker_count,
    resolve_save_eval_steps,
)


def candidate_to_train_config(config, candidate: TrainingCandidate):
    """Project a concrete candidate back into the shared train config shape."""
    return config.model_copy(
        update={
            "finetune_strategy": candidate.finetune_strategy,
            "per_device_train_batch_size": candidate.per_device_train_batch_size,
            "per_device_eval_batch_size": candidate.per_device_eval_batch_size,
            "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
            "dataloader_num_workers": candidate.dataloader_num_workers,
            "dataloader_pin_memory": candidate.dataloader_pin_memory,
            "dataloader_persistent_workers": candidate.dataloader_persistent_workers,
            "dataloader_prefetch_factor": candidate.dataloader_prefetch_factor,
            "fp16": candidate.fp16,
            "gradient_checkpointing": candidate.gradient_checkpointing,
            "max_sequence_length": candidate.max_sequence_length,
            "num_train_epochs": candidate.num_train_epochs,
            "learning_rate": candidate.learning_rate,
            "eval_steps": candidate.eval_steps,
            "logging_steps": candidate.logging_steps,
            "save_steps": candidate.save_steps,
            "save_total_limit": candidate.save_total_limit,
            "load_best_model_at_end": candidate.load_best_model_at_end,
            "metric_for_best_model": candidate.metric_for_best_model,
            "greater_is_better": candidate.greater_is_better,
            "verbose_epochs": candidate.verbose_epochs,
            "abort_vram_usage_ratio": candidate.abort_vram_usage_ratio,
            "lora_rank": candidate.lora_rank,
            "lora_alpha": candidate.lora_alpha,
            "lora_dropout": candidate.lora_dropout,
        }
    )


def benchmark_subset_rows(
    rows: list[dict[str, str]],
    candidate: TrainingCandidate,
    warmup_steps: int,
    measure_steps: int,
) -> list[dict[str, str]]:
    """Return a small deterministic slice for micro-benchmarking."""
    micro_batch = candidate.per_device_train_batch_size * candidate.gradient_accumulation_steps
    required_rows = max(8, (warmup_steps + measure_steps + 2) * micro_batch)
    return rows[: min(len(rows), required_rows)]


def build_training_arguments(
    *,
    training_arguments_cls,
    output_dir: Path,
    candidate: TrainingCandidate,
    eval_enabled: bool,
    save_enabled: bool,
    compute_metrics_enabled: bool,
    max_steps: int | None,
    logger,
) -> Any:
    """Build Hugging Face TrainingArguments for benchmark or full execution."""
    metric_for_best_model = (
        "eval_cer"
        if candidate.metric_for_best_model.strip().lower() == "cer"
        else candidate.metric_for_best_model
    )
    effective_eval_steps = None
    effective_save_steps = None
    if eval_enabled:
        effective_eval_steps, effective_save_steps = resolve_save_eval_steps(
            eval_steps=int(candidate.eval_steps),
            save_steps=candidate.save_steps,
            load_best_model_at_end=candidate.load_best_model_at_end and eval_enabled,
            logger=logger,
        )
    elif save_enabled:
        effective_save_steps = max(1, int(candidate.save_steps))
    effective_workers = bounded_worker_count(candidate.dataloader_num_workers)
    if effective_workers != candidate.dataloader_num_workers:
        logger.warning(
            "Adjusted dataloader_num_workers from %d to %d based on local CPU availability.",
            candidate.dataloader_num_workers,
            effective_workers,
        )
    kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": candidate.per_device_train_batch_size,
        "per_device_eval_batch_size": candidate.per_device_eval_batch_size
        or candidate.per_device_train_batch_size,
        "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
        "dataloader_num_workers": effective_workers,
        "dataloader_pin_memory": candidate.dataloader_pin_memory,
        "dataloader_persistent_workers": bool(candidate.dataloader_persistent_workers)
        and effective_workers > 0,
        "dataloader_prefetch_factor": int(candidate.dataloader_prefetch_factor)
        if effective_workers > 0
        else None,
        "learning_rate": candidate.learning_rate,
        "fp16": candidate.fp16,
        "bf16": False,
        "gradient_checkpointing": candidate.gradient_checkpointing,
        "optim": "paged_adamw_8bit"
        if candidate.finetune_strategy.value == "qlora"
        else "adamw_torch_fused",
        "num_train_epochs": candidate.num_train_epochs,
        "eval_strategy": "steps" if eval_enabled else "no",
        "eval_steps": effective_eval_steps if eval_enabled else None,
        "eval_accumulation_steps": 1 if eval_enabled else None,
        "save_strategy": "steps" if save_enabled else "no",
        "save_steps": effective_save_steps if save_enabled else None,
        "save_total_limit": candidate.save_total_limit,
        "load_best_model_at_end": candidate.load_best_model_at_end and eval_enabled,
        "metric_for_best_model": metric_for_best_model if eval_enabled else None,
        "greater_is_better": candidate.greater_is_better,
        "prediction_loss_only": eval_enabled and not compute_metrics_enabled,
        "remove_unused_columns": False,
        "logging_strategy": "steps",
        "logging_steps": max(1, candidate.logging_steps),
        "report_to": [],
        "disable_tqdm": False,
    }
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    return training_arguments_cls(
        **kwargs,
    )
