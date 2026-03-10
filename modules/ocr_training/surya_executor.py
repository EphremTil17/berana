from __future__ import annotations

import gc
from contextlib import suppress
from pathlib import Path
from typing import Any

from modules.ocr_training.checkpointing import (
    BestCerCheckpointCallback,
    TrainingSignalState,
    install_signal_handlers,
    resolve_latest_checkpoint,
    write_resume_state,
)
from modules.ocr_training.runtime.telemetry import (
    BenchmarkTelemetryCallback,
    ThroughputGuardCallback,
    VramPressureCallback,
)
from modules.ocr_training.schemas import (
    CandidateResult,
    CandidateStatus,
    TrainingCandidate,
    TrainMode,
)
from modules.ocr_training.surya_artifacts import (
    candidate_output_dir,
    cleanup_candidate_scratch,
    register_completed_finetune,
    register_interrupted_finetune,
    write_finetune_meta,
)
from modules.ocr_training.surya_common import resolve_resume_checkpoint
from modules.ocr_training.surya_data import LocalSuryaOCRDataset, SuryaOCRDataCollator
from modules.ocr_training.surya_patches import (
    build_eval_cleanup_callback,
    build_eval_interrupt_discard_callback,
    build_interrupt_callback,
    build_preprocess_logits_for_metrics,
    compute_metrics_factory,
)
from modules.ocr_training.surya_training_args import (
    benchmark_subset_rows,
    build_training_arguments,
    candidate_to_train_config,
)


def _safe_save_training_bundle(
    *, model, processor, output_dir: Path, logger, is_rank_zero: bool = True
) -> None:
    """Persist model and processor artifacts without hard-failing on processor serialization."""
    if not is_rank_zero:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    if hasattr(processor, "save_pretrained"):
        try:
            processor.save_pretrained(str(output_dir))
        except Exception as exc:
            logger.warning(
                "Skipping processor.save_pretrained for %s due to serialization error: %s",
                output_dir,
                exc,
            )


def _register_interrupted_training(
    *,
    trainer,
    model,
    processor,
    output_dir: Path,
    run_key: str,
    attempts: list[dict[str, Any]],
    candidate,
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    selection_reason: str,
    discarded_candidates: int,
    retry_count: int,
    mode,
    original_train_count: int,
    config,
    logger,
    distributed_context,
):
    """Persist interruption artifacts and registry metadata for a stopped run."""
    latest_checkpoint = resolve_latest_checkpoint(output_dir)
    emergency_dir = output_dir / "checkpoint-emergency"
    _safe_save_training_bundle(
        model=model,
        processor=processor,
        output_dir=emergency_dir,
        logger=logger,
        is_rank_zero=distributed_context.is_rank_zero,
    )
    if distributed_context.is_rank_zero:
        emergency_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_state()
    resume_state_path = write_resume_state(
        output_dir,
        status="interrupted",
        latest_checkpoint=latest_checkpoint or emergency_dir,
        is_rank_zero=distributed_context.is_rank_zero,
    )
    return register_interrupted_finetune(
        run_key=run_key,
        output_dir=output_dir,
        attempts=attempts,
        selected_candidate=candidate,
        train_count=len(train_rows),
        val_count=len(val_rows),
        resume_state_path=resume_state_path,
        emergency_dir=emergency_dir,
        mode=mode,
        selection_reason=selection_reason,
        discarded_candidates=discarded_candidates,
        retry_count=retry_count,
        original_train_count=original_train_count,
        train_fraction=float(config.train_fraction),
        train_subset_seed=int(config.seed),
        is_rank_zero=distributed_context.is_rank_zero,
    )


def _resolve_effective_best_metric(*, candidate: TrainingCandidate, compute_metrics, logger):
    """Return the effective best-model metric configuration for this run."""
    metric_name = candidate.metric_for_best_model.strip().lower()
    if compute_metrics is None and metric_name == "cer":
        log_warning = getattr(logger, "warning", lambda *args, **kwargs: None)
        log_warning(
            "CER metrics are unavailable for this processor/runtime; falling back to eval_loss "
            "for best-model selection on this run."
        )
        return candidate.model_copy(
            update={
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        ), "eval_loss"

    if metric_name == "cer":
        return candidate, "eval_cer"
    return candidate, candidate.metric_for_best_model


def _attach_training_callbacks(
    *,
    trainer,
    runtime,
    torch,
    signal_state,
    candidate: TrainingCandidate,
    eval_enabled: bool,
    checkpoint_metric_name: str,
    planned_samples_per_second: float | None,
    mode: TrainMode,
    distributed_context,
    logger,
    epoch_logging_callback_cls,
) -> None:
    """Attach rank-safe training, eval, and guard callbacks to one trainer."""
    if eval_enabled:
        trainer.add_callback(
            build_eval_interrupt_discard_callback(
                signal_state,
                runtime["TrainerCallback"],
                logger,
            )
        )
        if distributed_context.is_rank_zero:
            trainer.add_callback(
                BestCerCheckpointCallback(
                    output_dir=Path(trainer.args.output_dir),
                    metric_name=checkpoint_metric_name,
                )
            )
        trainer.add_callback(
            build_eval_cleanup_callback(
                torch_module=torch,
                callback_base=runtime["TrainerCallback"],
            )
        )
    if candidate.verbose_epochs and distributed_context.is_rank_zero:
        trainer.add_callback(epoch_logging_callback_cls())
    trainer.add_callback(
        VramPressureCallback(
            callback_base=runtime["TrainerCallback"],
            torch_module=torch,
            usage_threshold_ratio=candidate.abort_vram_usage_ratio,
            check_interval=max(1, candidate.logging_steps),
        ).build()
    )
    if mode == TrainMode.AUTO:
        trainer.add_callback(
            ThroughputGuardCallback(
                callback_base=runtime["TrainerCallback"],
                candidate=candidate,
                planned_samples_per_second=planned_samples_per_second,
            ).build()
        )
    trainer.add_callback(build_interrupt_callback(signal_state, runtime["TrainerCallback"]))


def benchmark_candidate(
    *,
    runtime: dict[str, Any],
    output_dir: Path,
    base_checkpoint: str,
    config,
    train_rows: list[dict[str, str]],
    candidate: TrainingCandidate,
    distributed_context,
    load_surya_training_stack,
    logger,
) -> CandidateResult:
    """Benchmark one training candidate on a small deterministic train subset."""
    torch = runtime["torch"]
    benchmark_dir = candidate_output_dir(output_dir, candidate)
    cleanup_candidate_scratch(output_dir, candidate)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    candidate_config = candidate_to_train_config(config, candidate)
    benchmark_rows = benchmark_subset_rows(
        train_rows,
        candidate,
        config.warmup_steps_per_candidate,
        config.measure_steps_per_candidate,
    )
    model = None
    processor = None
    train_dataset = None
    collator = None
    training_args = None
    trainer = None
    telemetry = None
    try:
        model, processor, _metadata = load_surya_training_stack(
            runtime,
            checkpoint=base_checkpoint,
            config=candidate_config,
        )
        train_dataset = LocalSuryaOCRDataset(
            processor=processor, rows=benchmark_rows, runtime=runtime
        )
        collator = SuryaOCRDataCollator(
            processor=processor,
            max_sequence_length=candidate.max_sequence_length,
            task_name=runtime["TaskNames"].ocr_with_boxes,
        )
        training_args = build_training_arguments(
            training_arguments_cls=runtime["TrainingArguments"],
            output_dir=benchmark_dir,
            candidate=candidate,
            eval_enabled=False,
            save_enabled=False,
            compute_metrics_enabled=False,
            max_steps=config.warmup_steps_per_candidate + config.measure_steps_per_candidate,
            logger=logger,
        )
        trainer = runtime["Trainer"](
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator,
        )
        telemetry = BenchmarkTelemetryCallback(
            callback_base=runtime["TrainerCallback"],
            torch_module=torch,
            candidate=candidate,
            warmup_steps=config.warmup_steps_per_candidate,
            measure_steps=config.measure_steps_per_candidate,
        ).build()
        trainer.add_callback(telemetry)
        trainer.add_callback(
            VramPressureCallback(
                callback_base=runtime["TrainerCallback"],
                torch_module=torch,
                usage_threshold_ratio=candidate.abort_vram_usage_ratio,
                check_interval=1,
            ).build()
        )
        trainer.train()
        return telemetry.summarize()
    except RuntimeError as exc:
        message = str(exc)
        lowered = message.lower()
        status = CandidateStatus.ERROR
        if "out of memory" in lowered:
            status = CandidateStatus.OOM
        elif "vram guard triggered" in lowered:
            status = CandidateStatus.VRAM_GUARD
        return CandidateResult(
            candidate_id=candidate.candidate_id,
            execution_backend=candidate.execution_backend.value,
            world_size=candidate.world_size,
            effective_global_batch_size=candidate.effective_global_batch_size,
            status=status,
            reason=message,
            measured_steps=0,
            warmup_steps=config.warmup_steps_per_candidate,
        )
    finally:
        del trainer
        del training_args
        del telemetry
        del collator
        del train_dataset
        del processor
        del model
        if torch.cuda.is_available():
            with suppress(Exception):
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            with suppress(Exception):
                torch.cuda.ipc_collect()
        gc.collect()
        cleanup_candidate_scratch(output_dir, candidate)


def run_training_candidate(
    *,
    runtime: dict[str, Any],
    run_key: str,
    output_dir: Path,
    config,
    candidate: TrainingCandidate,
    base_checkpoint: str,
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    original_train_count: int,
    attempts: list[dict[str, Any]],
    selection_reason: str,
    discarded_candidates: int,
    retry_count: int,
    planned_samples_per_second: float | None,
    mode: TrainMode,
    distributed_context=None,
    load_surya_training_stack,
    logger,
    epoch_logging_callback_cls,
) -> dict[str, Any]:
    """Execute one selected candidate as the real training run."""
    torch = runtime["torch"]
    if distributed_context is None:
        distributed_context = type(
            "_SingleProcessContext",
            (),
            {"is_rank_zero": True, "is_distributed": False},
        )()
    if torch.cuda.is_available():
        with suppress(Exception):
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        with suppress(Exception):
            torch.cuda.ipc_collect()
    gc.collect()
    latest_resume = resolve_resume_checkpoint(output_dir, config.resume)
    logger.info(
        "Starting Surya %s finetune candidate=%s batch=%d resume=%s",
        candidate.finetune_strategy.value,
        candidate.candidate_id,
        candidate.per_device_train_batch_size,
        latest_resume,
    )
    candidate_config = candidate_to_train_config(config, candidate)
    finetune_meta = {
        "schema_version": "1.0",
        "mode": mode.value,
        "finetune_strategy": candidate.finetune_strategy.value,
        "base_checkpoint": base_checkpoint,
        "lora_rank": int(candidate.lora_rank),
        "lora_alpha": int(candidate.lora_alpha),
        "lora_dropout": float(candidate.lora_dropout),
        "selected_candidate_id": candidate.candidate_id,
        "selected_training_config": candidate.model_dump(mode="json"),
        "selection_reason": selection_reason,
        "discarded_candidates": discarded_candidates,
        "retry_count": retry_count,
        "planned_samples_per_second": planned_samples_per_second,
        "original_train_rows": original_train_count,
        "effective_train_rows": len(train_rows),
        "train_fraction": float(config.train_fraction),
        "train_subset_seed": int(config.seed),
    }
    model, processor, model_metadata = load_surya_training_stack(
        runtime,
        checkpoint=base_checkpoint,
        config=candidate_config,
    )
    write_finetune_meta(
        output_dir,
        {**finetune_meta, **model_metadata},
        is_rank_zero=distributed_context.is_rank_zero,
    )
    train_dataset = LocalSuryaOCRDataset(processor=processor, rows=train_rows, runtime=runtime)
    val_dataset = LocalSuryaOCRDataset(processor=processor, rows=val_rows, runtime=runtime)
    collator = SuryaOCRDataCollator(
        processor=processor,
        max_sequence_length=candidate.max_sequence_length,
        task_name=runtime["TaskNames"].ocr_with_boxes,
    )
    compute_metrics = compute_metrics_factory(processor)
    eval_enabled = candidate.eval_steps is not None and len(val_rows) > 0
    save_enabled = candidate.save_steps is not None and candidate.save_steps > 0
    effective_candidate, checkpoint_metric_name = _resolve_effective_best_metric(
        candidate=candidate,
        compute_metrics=compute_metrics if eval_enabled else None,
        logger=logger,
    )
    finetune_meta["effective_metric_for_best_model"] = checkpoint_metric_name
    finetune_meta["effective_greater_is_better"] = bool(effective_candidate.greater_is_better)
    training_args = build_training_arguments(
        training_arguments_cls=runtime["TrainingArguments"],
        output_dir=output_dir,
        candidate=effective_candidate,
        eval_enabled=eval_enabled,
        save_enabled=save_enabled,
        compute_metrics_enabled=compute_metrics is not None and eval_enabled,
        max_steps=None,
        logger=logger,
    )
    signal_state = TrainingSignalState()
    install_signal_handlers(signal_state)
    trainer = runtime["Trainer"](
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if eval_enabled else None,
        data_collator=collator,
        compute_metrics=compute_metrics if eval_enabled else None,
        preprocess_logits_for_metrics=build_preprocess_logits_for_metrics()
        if eval_enabled and compute_metrics is not None
        else None,
    )
    _attach_training_callbacks(
        trainer=trainer,
        runtime=runtime,
        torch=torch,
        signal_state=signal_state,
        candidate=candidate,
        eval_enabled=eval_enabled,
        checkpoint_metric_name=checkpoint_metric_name,
        planned_samples_per_second=planned_samples_per_second,
        mode=mode,
        distributed_context=distributed_context,
        logger=logger,
        epoch_logging_callback_cls=epoch_logging_callback_cls,
    )
    result = None
    try:
        trainer.train(resume_from_checkpoint=str(latest_resume) if latest_resume else None)
        if signal_state.interrupted:
            result = _register_interrupted_training(
                trainer=trainer,
                model=model,
                processor=processor,
                output_dir=output_dir,
                run_key=run_key,
                attempts=attempts,
                candidate=candidate,
                train_rows=train_rows,
                val_rows=val_rows,
                selection_reason=selection_reason,
                discarded_candidates=discarded_candidates,
                retry_count=retry_count,
                mode=mode,
                original_train_count=original_train_count,
                config=config,
                logger=logger,
                distributed_context=distributed_context,
            )
            return result
        _safe_save_training_bundle(
            model=model,
            processor=processor,
            output_dir=output_dir,
            logger=logger,
            is_rank_zero=distributed_context.is_rank_zero,
        )
        latest_checkpoint = resolve_latest_checkpoint(output_dir)
        if distributed_context.is_rank_zero:
            write_resume_state(output_dir, status="completed", latest_checkpoint=latest_checkpoint)
        result = register_completed_finetune(
            run_key=run_key,
            output_dir=output_dir,
            attempts=attempts,
            selected_candidate=candidate,
            train_count=len(train_rows),
            val_count=len(val_rows),
            latest_checkpoint=latest_checkpoint,
            mode=mode,
            selection_reason=selection_reason,
            discarded_candidates=discarded_candidates,
            retry_count=retry_count,
            original_train_count=original_train_count,
            train_fraction=float(config.train_fraction),
            train_subset_seed=int(config.seed),
            is_rank_zero=distributed_context.is_rank_zero,
        )
    except KeyboardInterrupt:
        result = _register_interrupted_training(
            trainer=trainer,
            model=model,
            processor=processor,
            output_dir=output_dir,
            run_key=run_key,
            attempts=attempts,
            candidate=candidate,
            train_rows=train_rows,
            val_rows=val_rows,
            selection_reason=selection_reason,
            discarded_candidates=discarded_candidates,
            retry_count=retry_count,
            mode=mode,
            original_train_count=original_train_count,
            config=config,
            logger=logger,
            distributed_context=distributed_context,
        )
    finally:
        del trainer
        del training_args
        del collator
        del val_dataset
        del train_dataset
        del processor
        del model
        if torch.cuda.is_available():
            with suppress(Exception):
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            with suppress(Exception):
                torch.cuda.ipc_collect()
        gc.collect()
    if result is None:
        raise RuntimeError("Training finished without a terminal result.")
    return result
