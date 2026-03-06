from __future__ import annotations

import hashlib
from pathlib import Path
from statistics import mean

import numpy as np
from tqdm import tqdm

from modules.ocr_training.checkpointing import atomic_write_json
from modules.ocr_training.surya_common import (
    infer_train_subset_bucket,
    load_split_rows,
    subset_train_rows,
)
from modules.ocr_training.surya_data import build_surya_training_sample
from modules.ocr_training.surya_model import require_surya, resolve_base_checkpoint
from utils.logger import get_logger

logger = get_logger("OCRTrainingSuryaInspect")


def parse_int_csv(values: str) -> list[int]:
    """Parse a comma-separated integer list into deterministic unique values."""
    parsed = sorted({int(part.strip()) for part in values.split(",") if part.strip()})
    if not parsed:
        raise ValueError("Expected at least one integer value.")
    return parsed


def select_rows_for_inspection(
    rows: list[dict[str, str]],
    *,
    sample_size: int,
    seed: int,
) -> list[dict[str, str]]:
    """Select a deterministic subset of rows for inspection."""
    if sample_size <= 0 or sample_size >= len(rows):
        return list(rows)
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row.get('image', '')}:{row.get('text', '')}".encode()
        ).hexdigest(),
    )
    return ordered[:sample_size]


def _percentile(sorted_values: list[int], percentile: float) -> int:
    """Return one percentile from a sorted integer list."""
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return int(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile
    lower = int(np.floor(rank))
    upper = int(np.ceil(rank))
    if lower == upper:
        return int(sorted_values[lower])
    fraction = rank - lower
    return round(
        (1.0 - fraction) * float(sorted_values[lower]) + fraction * float(sorted_values[upper])
    )


def summarize_lengths(values: list[int]) -> dict[str, float | int]:
    """Summarize one integer-length distribution."""
    if not values:
        return {
            "count": 0,
            "min": 0,
            "mean": 0.0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
        }
    ordered = sorted(int(value) for value in values)
    return {
        "count": len(ordered),
        "min": int(ordered[0]),
        "mean": float(mean(ordered)),
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": int(ordered[-1]),
    }


def build_truncation_report(
    token_lengths: list[int],
    *,
    max_sequence_lengths: list[int],
) -> list[dict[str, float | int]]:
    """Build truncation-rate summaries for candidate sequence caps."""
    if not token_lengths:
        return []
    report: list[dict[str, float | int]] = []
    total = len(token_lengths)
    for sequence_length in max_sequence_lengths:
        clipped = sum(length > sequence_length for length in token_lengths)
        report.append(
            {
                "max_sequence_length": sequence_length,
                "clipped_rows": clipped,
                "clipped_rate": float(clipped / total),
            }
        )
    return report


def build_batch_geometry(
    *,
    total_rows: int,
    per_device_batch_sizes: list[int],
    gradient_accumulation_steps: list[int],
) -> list[dict[str, int]]:
    """Build optimizer-step geometry for batch/accumulation combinations."""
    rows: list[dict[str, int]] = []
    for batch_size in per_device_batch_sizes:
        for grad_accum in gradient_accumulation_steps:
            effective_batch = max(1, batch_size * grad_accum)
            optimizer_steps = (total_rows + effective_batch - 1) // effective_batch
            rows.append(
                {
                    "per_device_train_batch_size": batch_size,
                    "gradient_accumulation_steps": grad_accum,
                    "effective_batch_size": effective_batch,
                    "optimizer_steps_per_epoch": optimizer_steps,
                }
            )
    return rows


def inspect_surya_dataset(
    *,
    dataset_dir: Path,
    split: str,
    sample_size: int,
    seed: int,
    train_fraction: float,
    max_sequence_lengths: list[int],
    per_device_batch_sizes: list[int],
    gradient_accumulation_steps: list[int],
    pretrained_checkpoint_path: str = "",
) -> dict[str, object]:
    """Inspect token pressure, image sizes, and batch geometry for one dataset split."""
    runtime = require_surya()
    processor = runtime["FoundationModelLoader"](
        resolve_base_checkpoint(runtime, pretrained_checkpoint_path)
    ).processor()
    rows = load_split_rows(dataset_dir, split)
    original_rows = len(rows)
    if split == "train":
        rows = subset_train_rows(rows, train_fraction=train_fraction, seed=seed)
    selected_rows = select_rows_for_inspection(rows, sample_size=sample_size, seed=seed)

    token_lengths: list[int] = []
    labeled_token_lengths: list[int] = []
    text_lengths: list[int] = []
    original_widths: list[int] = []
    original_heights: list[int] = []
    processed_widths: list[int] = []
    processed_heights: list[int] = []
    bucket_counts: dict[str, int] = {}

    progress = tqdm(
        selected_rows,
        desc=f"Inspect {split}",
        unit="sample",
        dynamic_ncols=True,
    )
    for row in progress:
        bucket = infer_train_subset_bucket(row)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        sample, image_meta = build_surya_training_sample(
            processor=processor,
            row=row,
            runtime=runtime,
        )
        batch = processor([sample], padding_side="right")
        attention_mask = batch["attention_mask"][0]
        input_ids = batch["input_ids"][0]
        token_length = int(attention_mask.sum().item())
        token_lengths.append(token_length)

        image_token_id = processor.image_token_id
        eoi_token_id = processor.eoi_token_id
        pad_token_id = processor.pad_token_id
        supervised_mask = (
            (input_ids != image_token_id)
            & (input_ids != eoi_token_id)
            & (input_ids != pad_token_id)
        )
        labeled_token_lengths.append(int(supervised_mask.sum().item()))
        text_lengths.append(len(row.get("text", "")))

        original_width, original_height = image_meta["original_size"]
        processed_width, processed_height = image_meta["processed_size"]
        original_widths.append(original_width)
        original_heights.append(original_height)
        processed_widths.append(processed_width)
        processed_heights.append(processed_height)

    run_dir = dataset_dir.parent.parent
    report = {
        "schema_version": "1.0",
        "dataset_dir": str(dataset_dir),
        "split": split,
        "seed": seed,
        "train_fraction": train_fraction if split == "train" else 1.0,
        "original_split_rows": original_rows,
        "effective_split_rows": len(rows),
        "inspected_rows": len(selected_rows),
        "bucket_counts": bucket_counts,
        "token_lengths": summarize_lengths(token_lengths),
        "supervised_token_lengths": summarize_lengths(labeled_token_lengths),
        "text_char_lengths": summarize_lengths(text_lengths),
        "original_image_widths": summarize_lengths(original_widths),
        "original_image_heights": summarize_lengths(original_heights),
        "processed_image_widths": summarize_lengths(processed_widths),
        "processed_image_heights": summarize_lengths(processed_heights),
        "truncation_report": build_truncation_report(
            token_lengths,
            max_sequence_lengths=max_sequence_lengths,
        ),
        "batch_geometry": build_batch_geometry(
            total_rows=len(rows),
            per_device_batch_sizes=per_device_batch_sizes,
            gradient_accumulation_steps=gradient_accumulation_steps,
        ),
        "notes": [
            "Token lengths are measured from the live Surya processor with batch size 1.",
            "Sequence-length truncation rates are analytical only; lower caps may still be "
            "unsafe for this multimodal Surya path.",
        ],
    }
    output_path = run_dir / "inspection" / f"surya_inspection_{split}.json"
    atomic_write_json(output_path, report)
    logger.info(
        "Inspection complete split=%s inspected_rows=%d report=%s p95_tokens=%d",
        split,
        len(selected_rows),
        output_path,
        int(report["token_lengths"]["p95"]),
    )
    return {"report_path": str(output_path), "report": report}
