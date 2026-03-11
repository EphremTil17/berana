from __future__ import annotations

from contextlib import suppress
from pathlib import Path

from modules.ocr_training.checkpointing import EpochLoggingCallback, atomic_write_json
from modules.ocr_training.distributed import (
    RankZeroLogger,
    destroy_distributed_context,
    initialize_distributed_context,
    maybe_barrier,
)
from modules.ocr_training.runtime.hardware_profile import (
    _detect_selected_gpu_index,
    detect_hardware_profile,
)
from modules.ocr_training.runtime.hardware_profile import (
    enforce_gpu_preflight as _enforce_gpu_preflight,
)
from modules.ocr_training.schemas import SuryaTrainConfig, TrainMode
from modules.ocr_training.surya_artifacts import (
    load_finetune_meta as _load_finetune_meta,
)
from modules.ocr_training.surya_artifacts import (
    write_hardware_profile,
)
from modules.ocr_training.surya_common import (
    deterministic_sample_rows,
    infer_row_modality,
    infer_train_subset_bucket,
    load_split_rows,
    subset_rows,
    subset_train_rows,
)
from modules.ocr_training.surya_common import (
    resolve_finetune_strategy as _resolve_finetune_strategy,
)
from modules.ocr_training.surya_eval import evaluate_surya_checkpoint as _evaluate_surya_checkpoint
from modules.ocr_training.surya_eval import evaluate_surya_modalities as _evaluate_surya_modalities
from modules.ocr_training.surya_executor import (
    benchmark_candidate,
    run_training_candidate,
)
from modules.ocr_training.surya_model import (
    load_surya_eval_predictor,
    load_surya_training_stack,
    require_surya,
    resolve_base_checkpoint,
)
from modules.ocr_training.surya_planner import (
    prepare_auto_training,
    run_auto_with_fallback,
    run_manual_training,
)
from modules.ocr_training.surya_reports import write_subset_manifest
from utils.logger import get_logger

logger = get_logger("OCRTrainingSuryaTrain")


def _prepare_train_and_val_rows(*, dataset_dir: Path, config: SuryaTrainConfig):
    """Load and subset train/val rows according to the runtime config."""
    original_train_rows = load_split_rows(dataset_dir, "train")
    original_val_rows = load_split_rows(dataset_dir, "val")
    train_rows = subset_train_rows(
        original_train_rows,
        train_fraction=config.train_fraction,
        seed=config.seed,
    )
    val_rows = subset_rows(
        original_val_rows,
        fraction=config.eval_fraction,
        seed=config.seed,
    )
    if config.eval_max_rows is not None and len(val_rows) > config.eval_max_rows:
        val_rows = deterministic_sample_rows(
            val_rows,
            max_rows=config.eval_max_rows,
            seed=config.seed,
        )
    return original_train_rows, original_val_rows, train_rows, val_rows


def _log_subset_adjustments(
    *,
    run_logger,
    config: SuryaTrainConfig,
    original_train_rows: list[dict[str, str]],
    original_val_rows: list[dict[str, str]],
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
) -> None:
    """Log deterministic train/val subsetting decisions for one run."""
    if len(train_rows) != len(original_train_rows):
        original_bucket_counts: dict[str, int] = {}
        sampled_bucket_counts: dict[str, int] = {}
        for row in original_train_rows:
            bucket = infer_train_subset_bucket(row)
            original_bucket_counts[bucket] = original_bucket_counts.get(bucket, 0) + 1
        for row in train_rows:
            bucket = infer_train_subset_bucket(row)
            sampled_bucket_counts[bucket] = sampled_bucket_counts.get(bucket, 0) + 1
        run_logger.info(
            "Applied train_fraction=%.4f seed=%d to train split: %d -> %d rows; "
            "sampled_mix=%s original_mix=%s",
            config.train_fraction,
            config.seed,
            len(original_train_rows),
            len(train_rows),
            sampled_bucket_counts,
            original_bucket_counts,
        )
    if len(val_rows) != len(original_val_rows):
        run_logger.info(
            "Applied eval_fraction=%.4f eval_max_rows=%s seed=%d to val split: %d -> %d rows",
            config.eval_fraction,
            config.eval_max_rows,
            config.seed,
            len(original_val_rows),
            len(val_rows),
        )


def _write_subset_manifests(
    *,
    output_dir: Path,
    config: SuryaTrainConfig,
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    is_rank_zero: bool,
) -> None:
    """Persist deterministic train/eval row manifests for reproducibility."""
    if not is_rank_zero:
        return
    manifests_dir = output_dir / "manifests"
    eval_rows = [{**row, "modality": infer_row_modality(row)} for row in val_rows]
    write_subset_manifest(
        output_path=manifests_dir / "eval_subset_manifest.jsonl",
        rows=eval_rows,
        split="val",
        seed=int(config.seed),
        selection="eval_fraction+eval_max_rows",
    )
    if float(config.train_fraction) < 1.0:
        train_manifest_rows = [{**row, "modality": infer_row_modality(row)} for row in train_rows]
        write_subset_manifest(
            output_path=manifests_dir / "train_subset_manifest.jsonl",
            rows=train_manifest_rows,
            split="train",
            seed=int(config.seed),
            selection="train_fraction",
        )


def _build_training_stack_loader(run_logger):
    """Build the lazy Surya training stack loader used by planner/executor."""
    return lambda runtime, checkpoint, config: load_surya_training_stack(
        runtime,
        checkpoint=checkpoint,
        config=config,
        detect_selected_gpu_index=_detect_selected_gpu_index,
        logger=run_logger,
    )


def _run_manual_mode(
    *,
    runtime,
    run_key: str,
    output_dir: Path,
    config: SuryaTrainConfig,
    base_checkpoint: str,
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    original_train_count: int,
    existing_finetune_meta,
    attempts: list[dict[str, str]],
    training_stack_loader,
    distributed_context,
    run_logger,
):
    """Execute the manual-mode candidate path."""
    selected_candidate, selection_reason, discarded_candidates, planned_sps = run_manual_training(
        config=config
    )
    if existing_finetune_meta:
        existing_strategy = _resolve_finetune_strategy(
            str(existing_finetune_meta["finetune_strategy"])
        )
        if existing_strategy != selected_candidate.finetune_strategy:
            raise ValueError(
                "Existing run directory was initialized with finetune_strategy="
                f"{existing_strategy}, not {selected_candidate.finetune_strategy}. "
                "Use a new output directory or resume with the original strategy."
            )
    if distributed_context.is_rank_zero:
        atomic_write_json(
            output_dir / "selected_training_config.json",
            {
                **selected_candidate.model_dump(mode="json"),
                "selection_reason": selection_reason,
                "measured_samples_per_second": None,
            },
        )
    return run_training_candidate(
        runtime=runtime,
        run_key=run_key,
        output_dir=output_dir,
        config=config,
        candidate=selected_candidate,
        base_checkpoint=base_checkpoint,
        train_rows=train_rows,
        val_rows=val_rows,
        original_train_count=original_train_count,
        attempts=attempts,
        selection_reason=selection_reason,
        discarded_candidates=discarded_candidates,
        retry_count=0,
        planned_samples_per_second=planned_sps,
        mode=config.mode,
        load_surya_training_stack=training_stack_loader,
        logger=run_logger,
        epoch_logging_callback_cls=EpochLoggingCallback,
        distributed_context=distributed_context,
    )


def _run_auto_mode(
    *,
    runtime,
    run_key: str,
    output_dir: Path,
    config: SuryaTrainConfig,
    base_checkpoint: str,
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    original_train_count: int,
    attempts: list[dict[str, str]],
    hardware_profile,
    training_stack_loader,
    distributed_context,
    run_logger,
):
    """Execute adaptive auto planning plus fallback for one run."""

    def candidate_benchmarker(**kwargs):
        return benchmark_candidate(
            **kwargs,
            distributed_context=distributed_context,
            load_surya_training_stack=training_stack_loader,
            logger=run_logger,
        )

    (
        initial_candidate,
        candidate_pool,
        candidate_results,
        selection_reason,
        discarded_count,
        planned_sps,
    ) = prepare_auto_training(
        runtime=runtime,
        output_dir=output_dir,
        config=config,
        base_checkpoint=base_checkpoint,
        train_rows=train_rows,
        hardware_profile=hardware_profile,
        benchmark_candidate=candidate_benchmarker,
        is_rank_zero=distributed_context.is_rank_zero,
        logger=run_logger,
    )

    return run_auto_with_fallback(
        initial_candidate=initial_candidate,
        candidate_pool=candidate_pool,
        candidate_results=candidate_results,
        selection_reason=selection_reason,
        planned_samples_per_second=planned_sps,
        config=config,
        output_dir=output_dir,
        attempts=attempts,
        runner=lambda selected_candidate, selection_reason, retry_count, planned_samples_per_second: (
            run_training_candidate(
                runtime=runtime,
                run_key=run_key,
                output_dir=output_dir,
                config=config,
                candidate=selected_candidate,
                base_checkpoint=base_checkpoint,
                train_rows=train_rows,
                val_rows=val_rows,
                original_train_count=original_train_count,
                attempts=attempts,
                selection_reason=selection_reason,
                discarded_candidates=discarded_count,
                retry_count=retry_count,
                planned_samples_per_second=planned_samples_per_second,
                mode=config.mode,
                load_surya_training_stack=training_stack_loader,
                logger=run_logger,
                epoch_logging_callback_cls=EpochLoggingCallback,
                distributed_context=distributed_context,
            )
        ),
        logger=run_logger,
    )


def run_surya_finetune(
    *,
    run_key: str,
    dataset_dir: Path,
    output_dir: Path,
    config: SuryaTrainConfig,
    pretrained_checkpoint_path: str = "",
) -> dict[str, str | None]:
    """Run Surya OCR finetuning in manual or adaptive auto-planner mode."""
    runtime = require_surya()
    torch = runtime["torch"]
    output_dir.mkdir(parents=True, exist_ok=True)
    distributed_context = initialize_distributed_context(
        torch_module=torch,
        requested_backend=config.execution_backend,
        ddp_backend=config.ddp_backend,
    )
    run_logger = RankZeroLogger(logger, is_rank_zero=distributed_context.is_rank_zero)
    config = config.model_copy(
        update={
            "execution_backend": distributed_context.execution_backend,
            "distributed_world_size": distributed_context.world_size,
        }
    )
    try:
        if not distributed_context.is_distributed:
            _enforce_gpu_preflight(
                torch,
                foreign_usage_threshold_ratio=config.foreign_vram_threshold_ratio,
            )
        original_train_rows, original_val_rows, train_rows, val_rows = _prepare_train_and_val_rows(
            dataset_dir=dataset_dir,
            config=config,
        )
        _log_subset_adjustments(
            run_logger=run_logger,
            config=config,
            original_train_rows=original_train_rows,
            original_val_rows=original_val_rows,
            train_rows=train_rows,
            val_rows=val_rows,
        )
        _write_subset_manifests(
            output_dir=output_dir,
            config=config,
            train_rows=train_rows,
            val_rows=val_rows,
            is_rank_zero=distributed_context.is_rank_zero,
        )
        existing_finetune_meta = _load_finetune_meta(output_dir)
        if existing_finetune_meta:
            base_checkpoint = str(existing_finetune_meta["base_checkpoint"])
        else:
            base_checkpoint = resolve_base_checkpoint(runtime, pretrained_checkpoint_path)

        hardware_profile = detect_hardware_profile(
            torch,
            execution_backend=distributed_context.execution_backend,
            distributed_world_size=distributed_context.world_size,
        )
        if distributed_context.is_rank_zero:
            write_hardware_profile(output_dir, hardware_profile)
        attempts: list[dict[str, str]] = []
        training_stack_loader = _build_training_stack_loader(run_logger)

        if config.mode == TrainMode.MANUAL:
            return _run_manual_mode(
                runtime=runtime,
                run_key=run_key,
                output_dir=output_dir,
                config=config,
                base_checkpoint=base_checkpoint,
                train_rows=train_rows,
                val_rows=val_rows,
                original_train_count=len(original_train_rows),
                existing_finetune_meta=existing_finetune_meta,
                attempts=attempts,
                distributed_context=distributed_context,
                training_stack_loader=training_stack_loader,
                run_logger=run_logger,
            )

        return _run_auto_mode(
            runtime=runtime,
            run_key=run_key,
            output_dir=output_dir,
            config=config,
            base_checkpoint=base_checkpoint,
            train_rows=train_rows,
            val_rows=val_rows,
            original_train_count=len(original_train_rows),
            attempts=attempts,
            hardware_profile=hardware_profile,
            training_stack_loader=training_stack_loader,
            distributed_context=distributed_context,
            run_logger=run_logger,
        )
    finally:
        # Keep rank teardown aligned on both clean exits and signal interruptions.
        with suppress(Exception):
            maybe_barrier(torch_module=torch, context=distributed_context)
        destroy_distributed_context(torch_module=torch, context=distributed_context)


def evaluate_surya_checkpoint(
    *,
    run_key: str,
    run_dir: Path,
    dataset_dir: Path,
    split: str,
    eval_fraction: float = 1.0,
    eval_batch_size: int = 8,
    max_rows: int | None = None,
    seed: int = 42,
    modality: str | None = None,
) -> dict[str, float | int | str]:
    """Evaluate Surya OCR predictions against target split labels."""
    runtime = require_surya()
    return _evaluate_surya_checkpoint(
        run_key=run_key,
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split=split,
        eval_fraction=eval_fraction,
        eval_batch_size=eval_batch_size,
        max_rows=max_rows,
        seed=seed,
        modality=modality,
        runtime=runtime,
        load_surya_eval_predictor=lambda runtime, run_dir: load_surya_eval_predictor(
            runtime,
            run_dir,
            _load_finetune_meta,
        ),
    )


def evaluate_surya_modalities(
    *,
    run_key: str,
    run_dir: Path,
    dataset_dir: Path,
    split: str,
    eval_fraction: float = 1.0,
    eval_batch_size: int = 8,
    max_rows: int | None = None,
    seed: int = 42,
    modalities: list[str] | None = None,
) -> dict[str, object]:
    """Evaluate one run across typed/synthetic modalities using the existing evaluator."""
    runtime = require_surya()
    return _evaluate_surya_modalities(
        run_key=run_key,
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split=split,
        eval_fraction=eval_fraction,
        eval_batch_size=eval_batch_size,
        max_rows=max_rows,
        seed=seed,
        modalities=modalities or ["typed", "synthetic"],
        runtime=runtime,
        load_surya_eval_predictor=lambda runtime, run_dir: load_surya_eval_predictor(
            runtime,
            run_dir,
            _load_finetune_meta,
        ),
    )
