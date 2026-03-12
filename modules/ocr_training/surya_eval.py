from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image
from tqdm import tqdm

from modules.ocr_benchmark.metrics import calculate_cer_wer_paper
from modules.ocr_training.registry import STAGE_SURYA_EVALUATE, register_training_stage
from modules.ocr_training.surya_common import (
    deterministic_sample_rows,
    infer_row_modality,
    load_split_rows,
    relative_to_base,
    sanitize_prediction_text,
    subset_rows,
)
from modules.ocr_training.surya_reports import (
    write_confusion_artifacts,
    write_training_report_bundle,
)

TAG_FILTER_LIST = [
    "p",
    "li",
    "ul",
    "ol",
    "table",
    "td",
    "tr",
    "th",
    "tbody",
    "pre",
    "b",
    "strong",
    "i",
    "em",
    "u",
    "span",
    "div",
    "br",
    "sup",
    "sub",
]


def _chunk_rows(rows: list[dict[str, str]], batch_size: int) -> list[list[dict[str, str]]]:
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def _shard_rows_for_rank(
    rows: list[dict[str, str]],
    *,
    rank: int,
    world_size: int,
) -> list[dict[str, str]]:
    if world_size <= 1:
        return rows
    return [row for index, row in enumerate(rows) if index % world_size == rank]


def build_in_memory_eval_predictor(*, runtime, model, processor):
    """Build a RecognitionPredictor around the current in-memory training model."""
    torch = runtime["torch"]
    foundation_cls = runtime["FoundationPredictor"]
    foundation_predictor = foundation_cls.__new__(foundation_cls)
    foundation_predictor.model = model
    foundation_predictor.processor = processor
    foundation_predictor.prompt_queue = deque()
    foundation_predictor.batch_prompt_mapping = None
    foundation_predictor.kv_cache = None
    foundation_predictor.beacon_token_interval = model.config.beacon_token_interval
    foundation_predictor.device_pad_token = torch.tensor(
        processor.pad_token_id,
        device=model.device,
        dtype=torch.long,
    )
    foundation_predictor.device_beacon_token = torch.tensor(
        processor.beacon_token_id,
        device=model.device,
        dtype=torch.long,
    )
    foundation_predictor.special_token_ids = torch.tensor(
        [model.config.image_token_id, *model.config.register_token_ids],
        device=model.device,
    )
    foundation_predictor.pad_to_multiple = None
    foundation_predictor._disable_tqdm = False
    predictor = runtime["RecognitionPredictor"](foundation_predictor)
    predictor.disable_tqdm = True
    return predictor


def _predict_surya_records(
    *,
    rows: list[dict[str, str]],
    split: str,
    eval_batch_size: int,
    predictor,
    runtime,
    distributed_context=None,
    torch_module=None,
) -> tuple[list[dict[str, Any]], int]:
    """Run OCR inference for one deterministic row list and return gathered prediction records."""
    rank = int(getattr(distributed_context, "rank", 0) if distributed_context else 0)
    world_size = int(getattr(distributed_context, "world_size", 1) if distributed_context else 1)
    local_rows = _shard_rows_for_rank(rows, rank=rank, world_size=world_size)

    records = []
    for row_batch in tqdm(
        _chunk_rows(local_rows, max(1, eval_batch_size)),
        desc=f"Evaluate {split}",
        unit="batch",
        dynamic_ncols=True,
        disable=bool(distributed_context and not distributed_context.is_rank_zero),
    ):
        images = []
        bboxes = []
        for row in row_batch:
            with Image.open(Path(row["image"])) as image:
                converted = image.convert("RGB")
            images.append(converted)
            bboxes.append([[0, 0, converted.width, converted.height]])
        results = predictor(
            images,
            task_names=[runtime["TaskNames"].ocr_with_boxes] * len(images),
            bboxes=bboxes,
            math_mode=False,
            drop_repeated_text=True,
            filter_tag_list=TAG_FILTER_LIST,
        )
        for row, result in zip(row_batch, results, strict=False):
            raw_pred = (
                sanitize_prediction_text(result.text_lines[0].text) if result.text_lines else ""
            )
            gt_text = row["text"]
            cer, wer, exact = calculate_cer_wer_paper(raw_pred, gt_text)
            records.append(
                {
                    "image": row["image"],
                    "gt_text": gt_text,
                    "pred_text": raw_pred,
                    "cer": cer,
                    "wer": wer,
                    "exact": exact,
                }
            )
    if distributed_context and distributed_context.is_distributed:
        gathered_records: list[list[dict[str, Any]] | None] = [
            None
        ] * distributed_context.world_size
        torch_module.distributed.all_gather_object(gathered_records, records)
        if not distributed_context.is_rank_zero:
            return [], world_size
        records = [record for chunk in gathered_records if chunk for record in chunk]
    return records, world_size


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
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = eval_dir / f"summary_{suffix}.json"
    summary_payload = {
        "split": split,
        "modality": modality,
        "num_rows": len(records),
        "world_size": world_size,
        "eval_fraction": eval_fraction,
        "eval_batch_size": eval_batch_size,
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
    records, world_size = _predict_surya_records(
        rows=rows,
        split=split,
        eval_batch_size=eval_batch_size,
        predictor=predictor,
        runtime=runtime,
        distributed_context=distributed_context,
        torch_module=torch_module,
    )
    if (
        distributed_context
        and distributed_context.is_distributed
        and not distributed_context.is_rank_zero
    ):
        return {
            "status": "completed_nonzero_rank",
            "rank": distributed_context.rank,
            "num_rows": len(records),
        }
    eval_dir = output_dir or (run_dir / "tool_evaluation")
    return _write_surya_eval_outputs(
        run_key=run_key,
        run_dir=run_dir,
        eval_dir=eval_dir,
        split=split,
        modality=modality,
        eval_fraction=eval_fraction,
        max_rows=max_rows,
        eval_batch_size=eval_batch_size,
        seed=seed,
        records=records,
        world_size=world_size,
        register_stage=register_stage,
        include_predictions=include_predictions,
        include_confusions=include_confusions,
        include_report_bundle=include_report_bundle,
    )


def evaluate_surya_checkpoint(
    *,
    run_key: str,
    run_dir: Path,
    dataset_dir: Path,
    split: str,
    eval_fraction: float,
    max_rows: int | None,
    eval_batch_size: int,
    seed: int,
    modality: str | None = None,
    runtime,
    load_surya_eval_predictor,
    distributed_context=None,
    torch_module=None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate Surya OCR predictions against target split labels."""
    rows = load_split_rows(dataset_dir, split)
    if modality is not None:
        normalized_modality = modality.strip().lower()
        rows = [row for row in rows if infer_row_modality(row) == normalized_modality]
    if eval_fraction < 1.0:
        rows = subset_rows(rows, fraction=eval_fraction, seed=seed)
    if max_rows is not None and len(rows) > max_rows:
        rows = deterministic_sample_rows(rows, max_rows=max_rows, seed=seed)
    foundation_predictor = load_surya_eval_predictor(runtime, run_dir)
    predictor = runtime["RecognitionPredictor"](foundation_predictor)
    predictor.disable_tqdm = True
    return evaluate_surya_rows(
        run_key=run_key,
        run_dir=run_dir,
        rows=rows,
        split=split,
        eval_fraction=eval_fraction,
        max_rows=max_rows,
        eval_batch_size=eval_batch_size,
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
    }
    combined_summary_path.write_text(
        json.dumps(combined_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return combined_payload
