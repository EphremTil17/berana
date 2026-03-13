from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from tqdm import tqdm

from modules.ocr_training.distributed import maybe_barrier
from modules.ocr_training.registry import STAGE_SURYA_EVALUATE, register_training_stage
from modules.ocr_training.surya_common import load_split_rows, relative_to_base
from modules.ocr_training.surya_eval_runtime import (
    PreparedEvalRows,
    prepare_eval_rows,
    run_surya_eval_batches,
)
from modules.ocr_training.surya_reports import (
    write_confusion_artifacts,
    write_training_report_bundle,
)
from utils.logger import get_logger

logger = get_logger("OCRTrainingSuryaEval")


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _dataset_split_counts(dataset_dir: Path) -> dict[str, int]:
    return {
        split_name: _count_jsonl_rows(dataset_dir / f"{split_name}.jsonl")
        for split_name in ("train", "val", "holdout")
    }


def _log_eval_start_summary(
    *,
    dataset_dir: Path,
    split: str,
    prepared_rows: PreparedEvalRows,
    eval_fraction: float,
    max_rows: int | None,
    eval_batch_size: int,
    dataloader_num_workers: int,
    modality: str | None,
    distributed_context,
) -> None:
    if (
        distributed_context
        and distributed_context.is_distributed
        and not distributed_context.is_rank_zero
    ):
        return
    split_counts = _dataset_split_counts(dataset_dir)
    logger.info(
        "Starting Surya eval split=%s selected_rows=%d dataset_rows={train:%d,val:%d,holdout:%d} "
        "modality=%s eval_fraction=%.4f max_rows=%s batch=%d workers=%d",
        split,
        len(prepared_rows.rows),
        split_counts["train"],
        split_counts["val"],
        split_counts["holdout"],
        modality or "all",
        eval_fraction,
        max_rows,
        eval_batch_size,
        dataloader_num_workers,
    )


def _write_surya_eval_outputs(
    *,
    run_key: str | None,
    run_dir: Path,
    eval_dir: Path,
    split: str,
    modality: str | None,
    eval_fraction: float,
    max_rows: int | None,
    eval_batch_size: int,
    dataloader_num_workers: int,
    seed: int,
    records: list[dict[str, Any]],
    world_size: int,
    register_stage: bool,
    include_predictions: bool,
    include_confusions: bool,
    include_report_bundle: bool,
) -> dict[str, Any]:
    """Persist evaluation artifacts for either explicit tool runs or training checkpoint evals."""
    mean_cer = float(mean(r["cer"] for r in records)) if records else 1.0
    mean_wer = float(mean(r["wer"] for r in records)) if records else 1.0
    exact_rate = float(mean(1.0 if r["exact"] else 0.0 for r in records)) if records else 0.0
    eval_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{split}_{modality}" if modality else split

    predictions_path: Path | None = None
    if include_predictions:
        predictions_path = eval_dir / f"predictions_{suffix}.jsonl"
        with predictions_path.open("w", encoding="utf-8") as handle:
            progress = tqdm(
                records,
                desc=f"Write predictions {suffix}",
                unit="row",
                dynamic_ncols=True,
            )
            for record in progress:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress.close()

    summary_path = eval_dir / f"summary_{suffix}.json"
    summary_payload = {
        "split": split,
        "modality": modality,
        "num_rows": len(records),
        "world_size": world_size,
        "eval_fraction": eval_fraction,
        "eval_batch_size": eval_batch_size,
        "dataloader_num_workers": dataloader_num_workers,
        "max_rows": max_rows,
        "seed": seed,
        "mean_cer": mean_cer,
        "mean_wer": mean_wer,
        "exact_rate": exact_rate,
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report_path: Path | None = None
    if include_report_bundle:
        report_path = eval_dir / f"report_{suffix}.md"
        report_path.write_text(
            "\n".join(
                [
                    "# Surya Evaluation Report",
                    "",
                    f"- Split: `{split}`",
                    f"- Modality: `{modality or 'all'}`",
                    f"- Rows: `{len(records)}`",
                    f"- Eval Fraction: `{eval_fraction:.4f}`",
                    f"- Eval Batch Size: `{eval_batch_size}`",
                    f"- Dataloader Workers: `{dataloader_num_workers}`",
                    f"- Max Rows: `{max_rows}`",
                    f"- Seed: `{seed}`",
                    f"- Mean CER: `{mean_cer:.4f}`",
                    f"- Mean WER: `{mean_wer:.4f}`",
                    f"- Exact Match: `{exact_rate:.4f}`",
                ]
            ),
            encoding="utf-8",
        )

    report_artifacts = {}
    if include_confusions:
        report_artifacts.update(
            write_confusion_artifacts(eval_dir=eval_dir, split=suffix, records=records)
        )
    if include_report_bundle:
        report_artifacts.update(
            write_training_report_bundle(
                run_dir=run_dir,
                output_dir=eval_dir,
                split=suffix,
                predictions_path=predictions_path,
                include_training_artifacts=False,
            )
        )

    if register_stage and run_key is not None:
        register_training_stage(
            stage=STAGE_SURYA_EVALUATE,
            run_key=run_key,
            run_dir=run_dir,
            artifacts={
                "summary": relative_to_base(summary_path),
                **(
                    {"predictions": relative_to_base(predictions_path)}
                    if predictions_path is not None
                    else {}
                ),
                **({"report": relative_to_base(report_path)} if report_path is not None else {}),
                **{
                    artifact_name: relative_to_base(artifact_path)
                    for artifact_name, artifact_path in report_artifacts.items()
                },
            },
            metadata={
                "status": "completed",
                "split": split,
                "num_rows": len(records),
                "world_size": world_size,
                "eval_fraction": eval_fraction,
                "eval_batch_size": eval_batch_size,
                "dataloader_num_workers": dataloader_num_workers,
                "max_rows": max_rows,
                "seed": seed,
                "mean_cer": mean_cer,
                "mean_wer": mean_wer,
                "exact_rate": exact_rate,
            },
        )
    return summary_payload


def evaluate_surya_rows(
    *,
    run_key: str | None,
    run_dir: Path,
    rows: list[dict[str, str]],
    split: str,
    eval_fraction: float,
    max_rows: int | None,
    eval_batch_size: int,
    dataloader_num_workers: int,
    seed: int,
    modality: str | None,
    predictor,
    runtime,
    distributed_context=None,
    torch_module=None,
    output_dir: Path | None = None,
    register_stage: bool = True,
    include_predictions: bool = True,
    include_confusions: bool = True,
    include_report_bundle: bool = True,
) -> dict[str, Any]:
    """Evaluate one prepared row list with a ready RecognitionPredictor-compatible object."""
    eval_artifacts = run_surya_eval_batches(
        rows=rows,
        split=split,
        eval_batch_size=eval_batch_size,
        predictor=predictor,
        runtime=runtime,
        dataloader_num_workers=dataloader_num_workers,
        distributed_context=distributed_context,
        torch_module=torch_module,
        collect_batch_timings=False,
    )
    if (
        distributed_context
        and distributed_context.is_distributed
        and not distributed_context.is_rank_zero
    ):
        maybe_barrier(torch_module=torch_module, context=distributed_context)
        return {
            "status": "completed_nonzero_rank",
            "rank": distributed_context.rank,
            "num_rows": len(eval_artifacts.records),
        }
    eval_dir = output_dir or (run_dir / "tool_evaluation")
    summary = _write_surya_eval_outputs(
        run_key=run_key,
        run_dir=run_dir,
        eval_dir=eval_dir,
        split=split,
        modality=modality,
        eval_fraction=eval_fraction,
        max_rows=max_rows,
        eval_batch_size=eval_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        seed=seed,
        records=eval_artifacts.records,
        world_size=eval_artifacts.world_size,
        register_stage=register_stage,
        include_predictions=include_predictions,
        include_confusions=include_confusions,
        include_report_bundle=include_report_bundle,
    )
    if distributed_context and distributed_context.is_distributed:
        maybe_barrier(torch_module=torch_module, context=distributed_context)
    return summary


def evaluate_surya_checkpoint(
    *,
    run_key: str,
    run_dir: Path,
    dataset_dir: Path,
    split: str,
    eval_fraction: float,
    max_rows: int | None,
    eval_batch_size: int,
    dataloader_num_workers: int = 0,
    seed: int,
    modality: str | None = None,
    runtime,
    load_surya_eval_predictor,
    distributed_context=None,
    torch_module=None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate Surya OCR predictions against target split labels."""
    prepared_rows: PreparedEvalRows = prepare_eval_rows(
        rows=load_split_rows(dataset_dir, split),
        modality=modality,
        eval_fraction=eval_fraction,
        max_rows=max_rows,
        seed=seed,
    )
    _log_eval_start_summary(
        dataset_dir=dataset_dir,
        split=split,
        prepared_rows=prepared_rows,
        eval_fraction=eval_fraction,
        max_rows=max_rows,
        eval_batch_size=eval_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        modality=modality,
        distributed_context=distributed_context,
    )
    foundation_predictor = load_surya_eval_predictor(runtime, run_dir)
    predictor = runtime["RecognitionPredictor"](foundation_predictor)
    predictor.disable_tqdm = True
    return evaluate_surya_rows(
        run_key=run_key,
        run_dir=run_dir,
        rows=prepared_rows.rows,
        split=split,
        eval_fraction=eval_fraction,
        max_rows=max_rows,
        eval_batch_size=eval_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        seed=seed,
        modality=modality,
        predictor=predictor,
        runtime=runtime,
        distributed_context=distributed_context,
        torch_module=torch_module,
        output_dir=output_dir,
        register_stage=True,
        include_predictions=True,
        include_confusions=True,
        include_report_bundle=True,
    )


def evaluate_surya_modalities(
    *,
    run_key: str,
    run_dir: Path,
    dataset_dir: Path,
    split: str,
    eval_fraction: float,
    max_rows: int | None,
    eval_batch_size: int,
    dataloader_num_workers: int = 0,
    seed: int,
    modalities: list[str],
    runtime,
    load_surya_eval_predictor,
    distributed_context=None,
    torch_module=None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint separately across requested typed/synthetic modalities."""
    modality_summaries: dict[str, Any] = {}
    for modality in modalities:
        modality_summaries[modality] = evaluate_surya_checkpoint(
            run_key=run_key,
            run_dir=run_dir,
            dataset_dir=dataset_dir,
            split=split,
            eval_fraction=eval_fraction,
            max_rows=max_rows,
            eval_batch_size=eval_batch_size,
            dataloader_num_workers=dataloader_num_workers,
            seed=seed,
            modality=modality,
            runtime=runtime,
            load_surya_eval_predictor=load_surya_eval_predictor,
            distributed_context=distributed_context,
            torch_module=torch_module,
            output_dir=output_dir,
        )
    if distributed_context and not distributed_context.is_rank_zero:
        return {
            "split": split,
            "modalities": modality_summaries,
            "status": "completed_nonzero_rank",
        }
    eval_dir = output_dir or (run_dir / "tool_evaluation")
    combined_summary_path = eval_dir / f"summary_{split}_modalities.json"
    combined_payload = {
        "split": split,
        "modalities": modality_summaries,
        "seed": seed,
        "eval_fraction": eval_fraction,
        "max_rows": max_rows,
        "eval_batch_size": eval_batch_size,
        "dataloader_num_workers": dataloader_num_workers,
    }
    combined_summary_path.write_text(
        json.dumps(combined_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return combined_payload
