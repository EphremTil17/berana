from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import NotRequired, TypedDict
from urllib.parse import quote

from modules.ocr_training.failure_analysis import (
    _expect_bool,
    _expect_float,
    _expect_int,
    _expect_int_or_none,
    _expect_str,
    _expect_str_list,
    _require_row_mapping,
)


class ReviewPredictionRow(TypedDict):
    """Prediction row consumed by the Label Studio task builder."""

    image: str
    gt_text: str
    pred_text: str
    cer: float
    wer: float
    exact: bool
    image_available: NotRequired[bool]
    classification: NotRequired[str]
    structural_classification: NotRequired[str]
    blank_score: NotRequired[int | None]
    blank_reasons: NotRequired[list[str]]
    image_width: NotRequired[int | None]
    image_height: NotRequired[int | None]
    resolution_signature_match: NotRequired[bool]
    structural_confirmed_blank: NotRequired[bool]
    structural_suspect_blank: NotRequired[bool]
    modality: NotRequired[str]
    gt_len: NotRequired[int]
    pred_len: NotRequired[int]
    cer_outlier_2std: NotRequired[bool]
    cer_outlier_3std: NotRequired[bool]
    wer_outlier_2std: NotRequired[bool]
    wer_outlier_3std: NotRequired[bool]
    mean_pixel: NotRequired[float]
    std_pixel: NotRequired[float]
    min_pixel: NotRequired[int]
    max_pixel: NotRequired[int]
    dynamic_range: NotRequired[int]
    foreground_ratio: NotRequired[float]
    component_count: NotRequired[int]
    max_component_ratio: NotRequired[float]
    edge_density: NotRequired[float]
    row_activity_ratio: NotRequired[float]
    col_activity_ratio: NotRequired[float]
    row_run_count: NotRequired[int]
    col_run_count: NotRequired[int]


class FailureReviewSummary(TypedDict):
    """Summary emitted when building OCR failure review tasks."""

    exact_false_dir: str
    output_json: str
    num_tasks: int
    skipped_missing_images: int
    source_counts: dict[str, int]


_OPTIONAL_REVIEW_BOOL_FIELDS = (
    "image_available",
    "resolution_signature_match",
    "structural_confirmed_blank",
    "structural_suspect_blank",
    "cer_outlier_2std",
    "cer_outlier_3std",
    "wer_outlier_2std",
    "wer_outlier_3std",
)

_OPTIONAL_REVIEW_STR_FIELDS = ("classification", "structural_classification", "modality")
_OPTIONAL_REVIEW_INT_FIELDS = (
    "gt_len",
    "pred_len",
    "min_pixel",
    "max_pixel",
    "dynamic_range",
    "component_count",
    "row_run_count",
    "col_run_count",
)
_OPTIONAL_REVIEW_FLOAT_FIELDS = (
    "mean_pixel",
    "std_pixel",
    "foreground_ratio",
    "max_component_ratio",
    "edge_density",
    "row_activity_ratio",
    "col_activity_ratio",
)
_OPTIONAL_REVIEW_INT_OR_NONE_FIELDS = ("blank_score", "image_width", "image_height")
_OPTIONAL_REVIEW_STR_LIST_FIELDS = ("blank_reasons",)


def _attach_review_bool_fields(
    prediction_row: ReviewPredictionRow,
    row: dict[str, object],
) -> ReviewPredictionRow:
    """Attach optional boolean review fields."""
    for field in _OPTIONAL_REVIEW_BOOL_FIELDS:
        if field in row:
            prediction_row[field] = _expect_bool(row, field)
    return prediction_row


def _attach_review_string_fields(
    prediction_row: ReviewPredictionRow,
    row: dict[str, object],
) -> ReviewPredictionRow:
    """Attach optional string and string-list review fields."""
    for field in _OPTIONAL_REVIEW_STR_FIELDS:
        if field in row:
            prediction_row[field] = _expect_str(row, field)
    for field in _OPTIONAL_REVIEW_STR_LIST_FIELDS:
        if field in row:
            prediction_row[field] = _expect_str_list(row, field)
    return prediction_row


def _attach_review_int_fields(
    prediction_row: ReviewPredictionRow,
    row: dict[str, object],
) -> ReviewPredictionRow:
    """Attach optional integer and nullable-integer review fields."""
    for field in _OPTIONAL_REVIEW_INT_FIELDS:
        if field in row:
            prediction_row[field] = _expect_int(row, field)
    for field in _OPTIONAL_REVIEW_INT_OR_NONE_FIELDS:
        if field in row:
            prediction_row[field] = _expect_int_or_none(row, field)
    return prediction_row


def _attach_review_float_fields(
    prediction_row: ReviewPredictionRow,
    row: dict[str, object],
) -> ReviewPredictionRow:
    """Attach optional floating-point review fields."""
    for field in _OPTIONAL_REVIEW_FLOAT_FIELDS:
        if field in row:
            prediction_row[field] = _expect_float(row, field)
    return prediction_row


def _attach_optional_review_fields(
    prediction_row: ReviewPredictionRow,
    row: dict[str, object],
) -> ReviewPredictionRow:
    """Populate optional review-only fields after required row validation."""
    prediction_row = _attach_review_bool_fields(prediction_row, row)
    prediction_row = _attach_review_string_fields(prediction_row, row)
    prediction_row = _attach_review_int_fields(prediction_row, row)
    prediction_row = _attach_review_float_fields(prediction_row, row)
    return prediction_row


def _to_prediction_row(data: object) -> ReviewPredictionRow:
    """Validate one review JSON dict at the file boundary."""
    row = dict(_require_row_mapping(data))
    prediction_row: ReviewPredictionRow = {
        "image": _expect_str(row, "image"),
        "gt_text": _expect_str(row, "gt_text"),
        "pred_text": _expect_str(row, "pred_text"),
        "cer": _expect_float(row, "cer"),
        "wer": _expect_float(row, "wer"),
        "exact": _expect_bool(row, "exact"),
    }
    return _attach_optional_review_fields(prediction_row, row)


def _to_label_studio_local_files_url(image_path: str) -> str:
    image_abs = Path(image_path).resolve()
    try:
        output_idx = image_abs.parts.index("output")
    except ValueError as exc:
        raise ValueError(
            f"Image path must resolve under output/ for Label Studio local-files serving. "
            f"Got '{image_abs}'."
        ) from exc
    output_rel = Path(*image_abs.parts[output_idx + 1 :])
    rel_url_path = str(output_rel).replace("\\", "/")
    return f"/data/local-files/?d={quote(rel_url_path, safe='/')}"


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _read_jsonl(path: Path) -> list[ReviewPredictionRow]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [_to_prediction_row(json.loads(line)) for line in handle if line.strip()]


def _task_key(row: ReviewPredictionRow) -> tuple[str, str, str]:
    return (row["image"], row["gt_text"], row["pred_text"])


def create_failure_review_tasks(
    *,
    exact_false_dir: Path,
    output_dir: Path,
    task_file_name: str = "ocr_failure_review_tasks.json",
) -> FailureReviewSummary:
    """Create one deduplicated Label Studio task file from exact-false review candidates."""
    if not exact_false_dir.exists():
        raise FileNotFoundError(f"Missing exact-false analysis directory: {exact_false_dir}")

    source_files = {
        "cer_outlier_2std": exact_false_dir / "cer_outliers_2std.jsonl",
        "cer_outlier_3std": exact_false_dir / "cer_outliers_3std.jsonl",
        "wer_outlier_2std": exact_false_dir / "wer_outliers_2std.jsonl",
        "wer_outlier_3std": exact_false_dir / "wer_outliers_3std.jsonl",
        "likely_label_mismatch": exact_false_dir / "likely_label_mismatch_predictions.jsonl",
        "likely_artifact": exact_false_dir / "likely_artifact_predictions.jsonl",
        "suspect_blank": exact_false_dir / "suspect_blank_predictions.jsonl",
        "confirmed_blank": exact_false_dir / "confirmed_blank_predictions.jsonl",
    }

    by_key: dict[tuple[str, str, str], ReviewPredictionRow] = {}
    candidate_sources: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    source_counts: dict[str, int] = {}

    for source_name, path in source_files.items():
        rows = _read_jsonl(path)
        source_counts[source_name] = len(rows)
        for row in rows:
            key = _task_key(row)
            if key not in by_key:
                by_key[key] = row
            candidate_sources[key].add(source_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, object]] = []
    skipped_missing_images: list[dict[str, object]] = []

    for idx, (key, row) in enumerate(
        sorted(
            by_key.items(),
            key=lambda item: (
                -len(candidate_sources[item[0]]),
                -item[1]["cer"],
                -item[1]["wer"],
                item[1]["image"],
            ),
        ),
        start=1,
    ):
        image_path = Path(row["image"])
        if not image_path.exists():
            skipped_missing_images.append(
                {
                    **row,
                    "candidate_sources": sorted(candidate_sources[key]),
                    "skip_reason": "image_missing",
                }
            )
            continue
        task_id = f"ocr_failure_{idx:06d}"
        tasks.append(
            {
                "data": {
                    "image": _to_label_studio_local_files_url(row["image"]),
                    "task_id": task_id,
                    "image_path": row["image"],
                    "gt_text": row["gt_text"],
                    "pred_text": row["pred_text"],
                    "corrected_gt_seed": row["gt_text"],
                    "cer": row["cer"],
                    "wer": row["wer"],
                    "exact": row["exact"],
                    "modality": row.get("modality"),
                    "candidate_sources": ", ".join(sorted(candidate_sources[key])),
                    "classification": row.get("classification"),
                    "structural_classification": row.get("structural_classification"),
                    "blank_reasons": ", ".join(
                        str(reason) for reason in row.get("blank_reasons", [])
                    ),
                    "image_width": row.get("image_width"),
                    "image_height": row.get("image_height"),
                    "gt_len": row.get("gt_len"),
                    "pred_len": row.get("pred_len"),
                }
            }
        )

    output_json = output_dir / task_file_name
    output_json.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "skipped_missing_images.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in skipped_missing_images)
        + ("\n" if skipped_missing_images else ""),
        encoding="utf-8",
    )
    summary: FailureReviewSummary = {
        "exact_false_dir": str(exact_false_dir),
        "output_json": str(output_json),
        "num_tasks": len(tasks),
        "skipped_missing_images": len(skipped_missing_images),
        "source_counts": source_counts,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
