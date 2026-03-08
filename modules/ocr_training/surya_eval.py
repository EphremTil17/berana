from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image
from tqdm import tqdm

from modules.ocr_benchmark.metrics import calculate_cer_wer, normalize_ethiopic_text
from modules.ocr_training.registry import STAGE_SURYA_EVALUATE, register_training_stage
from modules.ocr_training.surya_common import (
    load_split_rows,
    relative_to_base,
    sanitize_prediction_text,
    subset_rows,
)
from modules.ocr_training.surya_reports import (
    write_confusion_artifacts,
    write_training_history_artifacts,
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
    runtime,
    load_surya_eval_predictor,
) -> dict[str, Any]:
    """Evaluate Surya OCR predictions against target split labels."""
    rows = load_split_rows(dataset_dir, split)
    if eval_fraction < 1.0:
        rows = subset_rows(rows, fraction=eval_fraction, seed=seed)
    if max_rows is not None and len(rows) > max_rows:
        rows = rows[:max_rows]
    foundation_predictor = load_surya_eval_predictor(runtime, run_dir)
    predictor = runtime["RecognitionPredictor"](foundation_predictor)
    predictor.disable_tqdm = True

    records = []
    for row_batch in tqdm(
        _chunk_rows(rows, max(1, eval_batch_size)),
        desc=f"Evaluate {split}",
        unit="batch",
        dynamic_ncols=True,
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
            norm_pred = normalize_ethiopic_text(raw_pred)
            norm_gt = normalize_ethiopic_text(gt_text)
            cer, wer, exact = calculate_cer_wer(norm_pred, norm_gt)
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

    mean_cer = float(mean(r["cer"] for r in records)) if records else 1.0
    mean_wer = float(mean(r["wer"] for r in records)) if records else 1.0
    exact_rate = float(mean(1.0 if r["exact"] else 0.0 for r in records)) if records else 0.0
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = eval_dir / f"predictions_{split}.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = eval_dir / f"summary_{split}.json"
    summary_payload = {
        "split": split,
        "num_rows": len(records),
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
    report_path = eval_dir / f"report_{split}.md"
    report_path.write_text(
        "\n".join(
            [
                "# Surya Evaluation Report",
                "",
                f"- Split: `{split}`",
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
    report_artifacts.update(
        write_confusion_artifacts(eval_dir=eval_dir, split=split, records=records)
    )
    report_artifacts.update(write_training_history_artifacts(run_dir=run_dir, eval_dir=eval_dir))
    register_training_stage(
        stage=STAGE_SURYA_EVALUATE,
        run_key=run_key,
        run_dir=run_dir,
        artifacts={
            "summary": relative_to_base(summary_path),
            "predictions": relative_to_base(predictions_path),
            "report": relative_to_base(report_path),
            **{
                artifact_name: relative_to_base(artifact_path)
                for artifact_name, artifact_path in report_artifacts.items()
            },
        },
        metadata={
            "status": "completed",
            "split": split,
            "num_rows": len(records),
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
