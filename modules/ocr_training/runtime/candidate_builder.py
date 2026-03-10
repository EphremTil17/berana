from __future__ import annotations

import math
from dataclasses import dataclass

from modules.ocr_training.runtime.strategy_catalog import (
    MANUAL_DEFAULTS,
    preferred_auto_strategies,
    resolve_finetune_strategy,
    resolve_strategy_allowlist,
)
from modules.ocr_training.schemas import (
    ExecutionBackend,
    FinetuneStrategy,
    HardwareProfile,
    SuryaTrainConfig,
    TrainingCandidate,
)

_SAFE_MIN_SEQUENCE = 1024
_SEQUENCE_LADDER = (_SAFE_MIN_SEQUENCE,)


@dataclass(frozen=True)
class AutoTuneConstraints:
    """Concrete ceilings and planner targets derived from CLI input."""

    batch_ceiling: int
    grad_accum_ceiling: int
    sequence_ceiling: int
    worker_ceiling: int
    target_effective_batch: int


def _candidate_id(
    strategy: FinetuneStrategy,
    batch_size: int,
    grad_accum: int,
    sequence_length: int,
    workers: int,
    gradient_checkpointing: bool,
) -> str:
    """Build a deterministic candidate identifier."""
    gc_token = "gc1" if gradient_checkpointing else "gc0"
    return (
        f"{strategy.value}_b{batch_size}_ga{grad_accum}_seq{sequence_length}_w{workers}_{gc_token}"
    )


def derive_auto_constraints(
    config: SuryaTrainConfig, profile: HardwareProfile
) -> AutoTuneConstraints:
    """Convert optional CLI knobs into planner ceilings and targets."""
    batch_ceiling = max(1, int(config.per_device_train_batch_size or 8))
    grad_accum_ceiling = max(1, int(config.gradient_accumulation_steps or 4))
    sequence_ceiling = max(
        _SAFE_MIN_SEQUENCE, int(config.max_sequence_length or _SAFE_MIN_SEQUENCE)
    )
    default_worker_ceiling = min(max(1, profile.cpu_count - 1), 12)
    worker_ceiling = max(1, int(config.dataloader_num_workers or default_worker_ceiling))
    target_effective_batch = max(4, batch_ceiling * grad_accum_ceiling)
    return AutoTuneConstraints(
        batch_ceiling=batch_ceiling,
        grad_accum_ceiling=grad_accum_ceiling,
        sequence_ceiling=sequence_ceiling,
        worker_ceiling=worker_ceiling,
        target_effective_batch=target_effective_batch,
    )


def _worker_candidates(profile: HardwareProfile, constraints: AutoTuneConstraints) -> list[int]:
    """Return the bounded set of worker counts to benchmark."""
    if profile.cpu_count <= 2:
        return [1]
    candidates = [4, 8, 12]
    bounded = [
        min(max(1, candidate), max(1, profile.cpu_count - 1), constraints.worker_ceiling)
        for candidate in candidates
    ]
    return sorted(set(bounded))


def _sequence_candidates(sequence_ceiling: int) -> list[int]:
    """Return safe sequence-length candidates for Surya OCR."""
    candidates = [value for value in _SEQUENCE_LADDER if value <= sequence_ceiling]
    if not candidates:
        return [_SAFE_MIN_SEQUENCE]
    return candidates


def _batch_candidates(batch_ceiling: int) -> list[int]:
    """Return geometric batch-size candidates up to the configured ceiling."""
    batches = [1]
    while batches[-1] < batch_ceiling:
        next_batch = batches[-1] * 2
        if next_batch > batch_ceiling:
            next_batch = batch_ceiling
        if next_batch == batches[-1]:
            break
        batches.append(next_batch)
    return batches


def _gradient_checkpointing_candidates(
    profile: HardwareProfile,
    strategy: FinetuneStrategy,
) -> list[bool]:
    """Return admissible gradient-checkpointing settings for a strategy."""
    if strategy == FinetuneStrategy.QLORA:
        return [True]
    if strategy == FinetuneStrategy.LORA and (profile.total_vram_mb or 0) >= 24576:
        return [False, True]
    return [True]


def _materialize_candidate(
    *,
    config: SuryaTrainConfig,
    profile: HardwareProfile,
    strategy: FinetuneStrategy,
    batch_size: int,
    grad_accum: int,
    sequence_length: int,
    workers: int,
    gradient_checkpointing: bool,
) -> TrainingCandidate:
    """Create one concrete candidate from planner settings."""
    return TrainingCandidate(
        candidate_id=_candidate_id(
            strategy=strategy,
            batch_size=batch_size,
            grad_accum=grad_accum,
            sequence_length=sequence_length,
            workers=workers,
            gradient_checkpointing=gradient_checkpointing,
        ),
        execution_backend=ExecutionBackend(profile.execution_backend or "single"),
        world_size=max(1, int(profile.distributed_world_size)),
        effective_global_batch_size=batch_size
        * grad_accum
        * max(1, int(profile.distributed_world_size)),
        finetune_strategy=strategy,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        dataloader_num_workers=workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        dataloader_persistent_workers=config.dataloader_persistent_workers,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        fp16=config.fp16,
        gradient_checkpointing=gradient_checkpointing,
        max_sequence_length=sequence_length,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        verbose_epochs=config.verbose_epochs,
        foreign_vram_threshold_ratio=config.foreign_vram_threshold_ratio,
        abort_vram_usage_ratio=config.target_vram_utilization,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )


def build_training_candidates(
    *,
    profile: HardwareProfile,
    config: SuryaTrainConfig,
    constraints: AutoTuneConstraints,
) -> list[TrainingCandidate]:
    """Build the ordered candidate list for adaptive training."""
    allowlist = resolve_strategy_allowlist(config.strategy_allowlist)
    strategies = preferred_auto_strategies(profile, allowlist)
    candidates: list[TrainingCandidate] = []
    worker_candidates = _worker_candidates(profile, constraints)
    sequence_candidates = _sequence_candidates(constraints.sequence_ceiling)
    batch_candidates = _batch_candidates(constraints.batch_ceiling)

    for strategy in strategies:
        checkpointing_candidates = _gradient_checkpointing_candidates(profile, strategy)
        for workers in worker_candidates:
            for sequence_length in sequence_candidates:
                for gradient_checkpointing in checkpointing_candidates:
                    for batch_size in batch_candidates:
                        grad_accum = min(
                            constraints.grad_accum_ceiling,
                            max(1, math.ceil(constraints.target_effective_batch / batch_size)),
                        )
                        candidates.append(
                            _materialize_candidate(
                                config=config,
                                profile=profile,
                                strategy=strategy,
                                batch_size=batch_size,
                                grad_accum=grad_accum,
                                sequence_length=sequence_length,
                                workers=workers,
                                gradient_checkpointing=gradient_checkpointing,
                            )
                        )
    return candidates


def materialize_manual_candidate(config: SuryaTrainConfig) -> TrainingCandidate:
    """Convert manual-mode config into one concrete candidate."""
    strategy = resolve_finetune_strategy(
        config.finetune_strategy or MANUAL_DEFAULTS["finetune_strategy"]
    )
    return TrainingCandidate(
        candidate_id="manual",
        execution_backend=ExecutionBackend(config.execution_backend),
        world_size=max(1, int(config.distributed_world_size or 1)),
        effective_global_batch_size=int(
            config.per_device_train_batch_size or MANUAL_DEFAULTS["per_device_train_batch_size"]
        )
        * int(config.gradient_accumulation_steps or MANUAL_DEFAULTS["gradient_accumulation_steps"])
        * max(1, int(config.distributed_world_size or 1)),
        finetune_strategy=strategy,
        per_device_train_batch_size=int(
            config.per_device_train_batch_size or MANUAL_DEFAULTS["per_device_train_batch_size"]
        ),
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=int(
            config.gradient_accumulation_steps or MANUAL_DEFAULTS["gradient_accumulation_steps"]
        ),
        dataloader_num_workers=int(
            config.dataloader_num_workers or MANUAL_DEFAULTS["dataloader_num_workers"]
        ),
        dataloader_pin_memory=config.dataloader_pin_memory,
        dataloader_persistent_workers=config.dataloader_persistent_workers,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        max_sequence_length=int(
            config.max_sequence_length or MANUAL_DEFAULTS["max_sequence_length"]
        ),
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        verbose_epochs=config.verbose_epochs,
        foreign_vram_threshold_ratio=config.foreign_vram_threshold_ratio,
        abort_vram_usage_ratio=config.abort_vram_usage_ratio,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        planning_notes=["manual_mode"],
    )
