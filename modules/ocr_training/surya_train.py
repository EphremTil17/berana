from __future__ import annotations

from pathlib import Path

from modules.ocr_training.checkpointing import EpochLoggingCallback, atomic_write_json
from modules.ocr_training.runtime.hardware_profile import (
    _detect_selected_gpu_index,
    detect_hardware_profile,
    enforce_single_gpu,
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
    infer_train_subset_bucket,
    load_split_rows,
    subset_train_rows,
)
from modules.ocr_training.surya_common import (
    resolve_finetune_strategy as _resolve_finetune_strategy,
)
from modules.ocr_training.surya_eval import evaluate_surya_checkpoint as _evaluate_surya_checkpoint
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
from utils.logger import get_logger

logger = get_logger("OCRTrainingSuryaTrain")


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
    if config.mode == TrainMode.AUTO:
        enforce_single_gpu(torch)

    _enforce_gpu_preflight(
        torch,
        foreign_usage_threshold_ratio=config.foreign_vram_threshold_ratio,
    )
    original_train_rows = load_split_rows(dataset_dir, "train")
    val_rows = load_split_rows(dataset_dir, "val")
    train_rows = subset_train_rows(
        original_train_rows,
        train_fraction=config.train_fraction,
        seed=config.seed,
    )
    if len(train_rows) != len(original_train_rows):
        original_bucket_counts: dict[str, int] = {}
        sampled_bucket_counts: dict[str, int] = {}
        for row in original_train_rows:
            bucket = infer_train_subset_bucket(row)
            original_bucket_counts[bucket] = original_bucket_counts.get(bucket, 0) + 1
        for row in train_rows:
            bucket = infer_train_subset_bucket(row)
            sampled_bucket_counts[bucket] = sampled_bucket_counts.get(bucket, 0) + 1
        logger.info(
            "Applied train_fraction=%.4f seed=%d to train split: %d -> %d rows; "
            "sampled_mix=%s original_mix=%s",
            config.train_fraction,
            config.seed,
            len(original_train_rows),
            len(train_rows),
            sampled_bucket_counts,
            original_bucket_counts,
        )
    existing_finetune_meta = _load_finetune_meta(output_dir)
    if existing_finetune_meta:
        base_checkpoint = str(existing_finetune_meta["base_checkpoint"])
    else:
        base_checkpoint = resolve_base_checkpoint(runtime, pretrained_checkpoint_path)

    hardware_profile = detect_hardware_profile(torch)
    write_hardware_profile(output_dir, hardware_profile)
    attempts: list[dict[str, str]] = []

    training_stack_loader = lambda runtime, checkpoint, config: load_surya_training_stack(  # noqa: E731
        runtime,
        checkpoint=checkpoint,
        config=config,
        detect_selected_gpu_index=_detect_selected_gpu_index,
        logger=logger,
    )
    candidate_benchmarker = lambda **kwargs: benchmark_candidate(  # noqa: E731
        **kwargs,
        load_surya_training_stack=training_stack_loader,
        logger=logger,
    )

    if config.mode == TrainMode.MANUAL:
        selected_candidate, selection_reason, discarded_candidates, planned_sps = (
            run_manual_training(config=config)
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
            original_train_count=len(original_train_rows),
            attempts=attempts,
            selection_reason=selection_reason,
            discarded_candidates=discarded_candidates,
            retry_count=0,
            planned_samples_per_second=planned_sps,
            mode=config.mode,
            load_surya_training_stack=training_stack_loader,
            logger=logger,
            epoch_logging_callback_cls=EpochLoggingCallback,
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
        logger=logger,
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
                original_train_count=len(original_train_rows),
                attempts=attempts,
                selection_reason=selection_reason,
                discarded_candidates=discarded_count,
                retry_count=retry_count,
                planned_samples_per_second=planned_samples_per_second,
                mode=config.mode,
                load_surya_training_stack=training_stack_loader,
                logger=logger,
                epoch_logging_callback_cls=EpochLoggingCallback,
            )
        ),
        logger=logger,
    )


def evaluate_surya_checkpoint(
    *,
    run_key: str,
    run_dir: Path,
    dataset_dir: Path,
    split: str,
) -> dict[str, float | int | str]:
    """Evaluate Surya OCR predictions against target split labels."""
    runtime = require_surya()
    return _evaluate_surya_checkpoint(
        run_key=run_key,
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split=split,
        runtime=runtime,
        load_surya_eval_predictor=lambda runtime, run_dir: load_surya_eval_predictor(
            runtime,
            run_dir,
            _load_finetune_meta,
        ),
    )
