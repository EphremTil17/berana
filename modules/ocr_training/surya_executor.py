from __future__ import annotations

import gc
import json
from contextlib import suppress
from pathlib import Path
from typing import Any

from modules.ocr_training.checkpointing import (
    AuthoritativeCheckpointEvalCallback,
    PlateauWarningCallback,
    TrainingArtifactsCallback,
    TrainingSignalState,
    TrainingTerminationCoordinator,
    install_signal_handlers,
    observe_training_stop,
    request_training_stop,
    resolve_latest_checkpoint,
    write_resume_state,
)
from modules.ocr_training.distributed.context import DistributedContext
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
    load_finetune_meta,
    register_completed_finetune,
    register_interrupted_finetune,
    write_finetune_meta,
)
from modules.ocr_training.surya_common import resolve_resume_checkpoint
from modules.ocr_training.surya_data import LocalSuryaOCRDataset, SuryaOCRDataCollator
from modules.ocr_training.surya_eval import (
    evaluate_surya_rows,
)
from modules.ocr_training.surya_model import load_surya_eval_predictor
from modules.ocr_training.surya_patches import (
    build_eval_cleanup_callback,
    build_eval_interrupt_discard_callback,
    build_interrupt_callback,
)
from modules.ocr_training.surya_reports import write_training_report_bundle
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
    processor_module = getattr(processor.__class__, "__module__", "")
    if processor_module.startswith("surya.common.surya.processor"):
        _save_surya_processor_bundle(processor=processor, output_dir=output_dir, logger=logger)
        return
    if hasattr(processor, "save_pretrained"):
        try:
            processor.save_pretrained(str(output_dir))
        except Exception as exc:
            logger.warning(
                "Skipping processor.save_pretrained for %s due to serialization error: %s",
                output_dir,
                exc,
            )


def _save_surya_processor_bundle(*, processor, output_dir: Path, logger) -> None:
    """Persist the serializable Surya processor components without calling ProcessorMixin."""
    tokenizer = getattr(processor, "ocr_tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        try:
            tokenizer.save_pretrained(str(output_dir))
        except Exception as exc:
            logger.warning(
                "Skipping Surya tokenizer.save_pretrained for %s due to serialization error: %s",
                output_dir,
                exc,
            )
    processor_meta = {
        "schema_version": "1.0",
        "processor_class": processor.__class__.__name__,
        "processor_module": processor.__class__.__module__,
        "patch_size": getattr(processor, "patch_size", None),
        "merge_size": getattr(processor, "merge_size", None),
        "num_register_tokens": getattr(processor, "num_register_tokens", None),
        "num_beacon_tokens": getattr(processor, "num_beacon_tokens", None),
        "beacon_token_interval": getattr(processor, "beacon_token_interval", None),
        "blank_bbox_token_id": getattr(processor, "blank_bbox_token_id", None),
        "tokenizer_class": tokenizer.__class__.__name__ if tokenizer is not None else None,
    }
    (output_dir / "surya_processor_meta.json").write_text(
        json.dumps(processor_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
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
    result = register_interrupted_finetune(
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
    if distributed_context.is_rank_zero:
        with suppress(Exception):
            write_training_report_bundle(run_dir=output_dir)
    return result


def _resolve_effective_best_metric(*, candidate: TrainingCandidate, compute_metrics, logger):
    """Return the effective best-model metric configuration for this run."""
    del compute_metrics, logger
    metric_name = candidate.metric_for_best_model.strip().lower()
    if metric_name in {"wer", "eval_wer"}:
        return (
            candidate.model_copy(
                update={
                    "load_best_model_at_end": False,
                    "metric_for_best_model": "eval_loss",
                    "greater_is_better": False,
                }
            ),
            "eval_wer",
        )
    return (
        candidate.model_copy(
            update={
                "load_best_model_at_end": False,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        ),
        "eval_cer",
    )


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
    termination_coordinator,
    logger,
    epoch_logging_callback_cls,
    authoritative_eval_runner,
) -> None:
    """Attach rank-safe training, eval, and guard callbacks to one trainer."""
    trainer_args = getattr(trainer, "args", None)
    if trainer_args is None:
        trainer_args = getattr(trainer, "kwargs", {}).get("args")
    if trainer_args is None:
        raise ValueError("Could not determine trainer output_dir: trainer.args is None")
    trainer_output_dir = Path(str(trainer_args.output_dir))

    def _request_runtime_stop(message: str) -> None:
        request_training_stop(
            signal_state,
            reason=message,
            stop_type="runtime",
            save_checkpoint=False,
            coordinator=termination_coordinator,
            rank=getattr(distributed_context, "rank", 0),
        )

    if eval_enabled:
        plateau_callback = (
            PlateauWarningCallback(trainer_output_dir) if distributed_context.is_rank_zero else None
        )
        trainer.add_callback(
            build_eval_interrupt_discard_callback(
                signal_state,
                runtime["TrainerCallback"],
                logger,
            )
        )
        trainer.add_callback(
            AuthoritativeCheckpointEvalCallback(
                trainer_output_dir,
                eval_runner=authoritative_eval_runner,
                distributed_context=distributed_context,
                torch_module=torch,
                signal_state=signal_state,
                termination_coordinator=termination_coordinator,
                plateau_callback=plateau_callback,
            )
        )
        trainer.add_callback(
            build_eval_cleanup_callback(
                torch_module=torch,
                callback_base=runtime["TrainerCallback"],
            )
        )
    if distributed_context.is_rank_zero:
        trainer.add_callback(TrainingArtifactsCallback(trainer_output_dir))
    if candidate.verbose_epochs and distributed_context.is_rank_zero:
        trainer.add_callback(epoch_logging_callback_cls())
    if not candidate.allow_ram_spillover:
        trainer.add_callback(
            VramPressureCallback(
                callback_base=runtime["TrainerCallback"],
                torch_module=torch,
                usage_threshold_ratio=candidate.abort_vram_usage_ratio,
                check_interval=max(1, candidate.logging_steps),
                on_trigger=_request_runtime_stop,
            ).build()
        )
    if mode == TrainMode.AUTO:
        trainer.add_callback(
            ThroughputGuardCallback(
                callback_base=runtime["TrainerCallback"],
                candidate=candidate,
                planned_samples_per_second=planned_samples_per_second,
                on_trigger=_request_runtime_stop,
            ).build()
        )
    trainer.add_callback(
        build_interrupt_callback(
            signal_state,
            runtime["TrainerCallback"],
            termination_coordinator=termination_coordinator,
        )
    )


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
    signal_state = TrainingSignalState()
    install_signal_handlers(signal_state)
    termination_coordinator = TrainingTerminationCoordinator(benchmark_dir)
    termination_coordinator.clear()
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
        if not candidate.allow_ram_spillover:
            trainer.add_callback(
                VramPressureCallback(
                    callback_base=runtime["TrainerCallback"],
                    torch_module=torch,
                    usage_threshold_ratio=candidate.abort_vram_usage_ratio,
                    check_interval=1,
                    on_trigger=lambda message: request_training_stop(
                        signal_state,
                        reason=message,
                        stop_type="runtime",
                        save_checkpoint=False,
                        coordinator=termination_coordinator,
                        rank=getattr(distributed_context, "rank", 0),
                    ),
                ).build()
            )
        trainer.add_callback(
            build_interrupt_callback(
                signal_state,
                runtime["TrainerCallback"],
                termination_coordinator=termination_coordinator,
            )
        )
        trainer.train()
        observe_training_stop(signal_state, coordinator=termination_coordinator)
        if signal_state.runtime_error_message is not None:
            raise RuntimeError(signal_state.runtime_error_message)
        return telemetry.summarize()
    except RuntimeError as exc:
        request_training_stop(
            signal_state,
            reason=str(exc),
            stop_type="runtime",
            save_checkpoint=False,
            coordinator=termination_coordinator,
            rank=getattr(distributed_context, "rank", 0),
        )
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


def run_training_candidate(  # noqa: C901
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
    distributed_context: DistributedContext | None = None,
    load_surya_training_stack,
    logger,
    epoch_logging_callback_cls,
) -> dict[str, Any]:
    """Execute one selected candidate as the real training run."""
    torch = runtime["torch"]
    if distributed_context is None:
        distributed_context = DistributedContext(
            execution_backend="single",
            ddp_backend=None,
            is_distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            is_rank_zero=True,
        )
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
    compute_metrics = None
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
    termination_coordinator = TrainingTerminationCoordinator(output_dir)
    termination_coordinator.clear()
    trainer = runtime["Trainer"](
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if eval_enabled else None,
        data_collator=collator,
        compute_metrics=compute_metrics if eval_enabled else None,
        preprocess_logits_for_metrics=None,
    )

    def _authoritative_eval_runner(
        *,
        checkpoint_path: Path,
        state,
        _training_args=training_args,
    ):
        foundation_predictor = None
        predictor = None
        try:
            foundation_predictor = load_surya_eval_predictor(
                runtime=runtime,
                run_dir=output_dir,
                load_finetune_meta=load_finetune_meta,
                checkpoint_path=checkpoint_path,
            )
            predictor = runtime["RecognitionPredictor"](foundation_predictor)
            predictor.disable_tqdm = True
            checkpoint_eval_dir = (
                output_dir
                / "evaluation"
                / "checkpoint_eval"
                / f"checkpoint-{int(state.global_step or 0)}"
            )
            return evaluate_surya_rows(
                run_key=None,
                run_dir=output_dir,
                rows=val_rows,
                split="val",
                eval_fraction=float(config.eval_fraction),
                max_rows=config.eval_max_rows,
                eval_batch_size=_training_args.per_device_eval_batch_size
                or effective_candidate.per_device_train_batch_size,
                dataloader_num_workers=int(config.dataloader_num_workers),
                seed=int(config.seed),
                modality=None,
                predictor=predictor,
                runtime=runtime,
                distributed_context=distributed_context,
                torch_module=torch,
                output_dir=checkpoint_eval_dir,
                register_stage=False,
                include_predictions=False,
                include_confusions=False,
                include_report_bundle=False,
            )
        finally:
            del predictor
            del foundation_predictor
            gc.collect()
            if torch.cuda.is_available():
                with suppress(Exception):
                    torch.cuda.empty_cache()

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
        termination_coordinator=termination_coordinator,
        logger=logger,
        epoch_logging_callback_cls=epoch_logging_callback_cls,
        authoritative_eval_runner=_authoritative_eval_runner,
    )
    result = None
    try:
        trainer.train(resume_from_checkpoint=str(latest_resume) if latest_resume else None)
        observe_training_stop(signal_state, coordinator=termination_coordinator)
        if signal_state.runtime_error_message is not None:
            raise RuntimeError(signal_state.runtime_error_message)
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
        if distributed_context.is_rank_zero:
            with suppress(Exception):
                write_training_report_bundle(run_dir=output_dir)
    except RuntimeError as exc:
        request_training_stop(
            signal_state,
            reason=str(exc),
            stop_type="runtime",
            save_checkpoint=False,
            coordinator=termination_coordinator,
            rank=getattr(distributed_context, "rank", 0),
        )
        raise
    except KeyboardInterrupt:
        request_training_stop(
            signal_state,
            reason="keyboard_interrupt",
            stop_type="signal",
            save_checkpoint=True,
            coordinator=termination_coordinator,
            rank=getattr(distributed_context, "rank", 0),
        )
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
