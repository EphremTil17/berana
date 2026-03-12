from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from statistics import fmean

import cv2
import numpy as np
from PIL import Image

BLANK_SIGNATURE_SIZE = (2385, 244)


def _read_predictions(path: Path) -> list[dict[str, object]]:
    """Load one predictions JSONL artifact."""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _infer_modality(image_path: str) -> str:
    """Infer modality from one emitted evaluation image path."""
    if "fidel_synthetic__" in image_path:
        return "synthetic"
    if "typed" in image_path or "synth_image_" in image_path:
        return "typed"
    return "unknown"


def _load_grayscale(image_path: Path) -> np.ndarray:
    """Load one image as an 8-bit grayscale matrix."""
    image = Image.open(image_path).convert("L")
    matrix = np.asarray(image, dtype=np.uint8)
    image.close()
    return matrix


def _load_image_size(image_path: Path) -> tuple[int, int]:
    """Load one image size as width/height."""
    image = Image.open(image_path)
    size = image.size
    image.close()
    return size


def _otsu_binary(gray: np.ndarray) -> np.ndarray:
    """Binarize one grayscale image with Otsu thresholding."""
    _threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _projection_runs(profile: np.ndarray, *, floor: float) -> int:
    """Count contiguous projection runs above one activation floor."""
    activated = profile > floor
    if activated.size == 0:
        return 0
    transitions = np.diff(np.concatenate(([0], activated.astype(np.int8), [0])))
    starts = np.where(transitions == 1)[0]
    return int(starts.size)


def _extract_blank_features(gray: np.ndarray) -> dict[str, float | int]:
    """Extract low-level image features used for blank-image scoring."""
    height, width = gray.shape
    total_pixels = float(height * width)
    mean_pixel = float(np.mean(gray))
    std_pixel = float(np.std(gray))
    min_pixel = int(np.min(gray))
    max_pixel = int(np.max(gray))
    dynamic_range = max_pixel - min_pixel

    binary = _otsu_binary(gray)
    foreground_ratio = float(np.count_nonzero(binary)) / total_pixels

    component_count = 0
    max_component_ratio = 0.0
    if np.count_nonzero(binary) > 0:
        component_total, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        component_count = int(max(0, component_total - 1))
        if component_count:
            component_areas = stats[1:, cv2.CC_STAT_AREA]
            max_component_ratio = float(np.max(component_areas)) / total_pixels

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges)) / total_pixels

    row_profile = np.count_nonzero(binary, axis=1) / max(1, width)
    col_profile = np.count_nonzero(binary, axis=0) / max(1, height)
    return {
        "mean_pixel": mean_pixel,
        "std_pixel": std_pixel,
        "min_pixel": min_pixel,
        "max_pixel": max_pixel,
        "dynamic_range": dynamic_range,
        "foreground_ratio": foreground_ratio,
        "component_count": component_count,
        "max_component_ratio": max_component_ratio,
        "edge_density": edge_density,
        "row_activity_ratio": float(np.mean(row_profile > 0.01)),
        "col_activity_ratio": float(np.mean(col_profile > 0.01)),
        "row_run_count": _projection_runs(row_profile, floor=0.02),
        "col_run_count": _projection_runs(col_profile, floor=0.02),
    }


def _apply_score(
    *,
    predicate: bool,
    score: int,
    reason: str,
    running_score: int,
    reasons: list[str],
) -> int:
    """Apply one scored rule to the accumulating blankness result."""
    if predicate:
        reasons.append(reason)
        return running_score + score
    return running_score


def _score_blank_features(features: dict[str, float | int]) -> tuple[int, list[str]]:
    """Convert extracted image features into one blankness score and reason set."""
    score = 0
    reasons: list[str] = []
    score = _apply_score(
        predicate=features["mean_pixel"] >= 254.5 and features["dynamic_range"] <= 2,
        score=5,
        reason="uniform_near_white",
        running_score=score,
        reasons=reasons,
    )
    score = _apply_score(
        predicate=features["foreground_ratio"] <= 0.0005,
        score=4,
        reason="near_zero_foreground",
        running_score=score,
        reasons=reasons,
    )
    score = _apply_score(
        predicate=0.0005 < features["foreground_ratio"] <= 0.002,
        score=2,
        reason="very_low_foreground",
        running_score=score,
        reasons=reasons,
    )
    score = _apply_score(
        predicate=features["edge_density"] <= 0.0005,
        score=3,
        reason="near_zero_edges",
        running_score=score,
        reasons=reasons,
    )
    score = _apply_score(
        predicate=0.0005 < features["edge_density"] <= 0.002,
        score=1,
        reason="very_low_edges",
        running_score=score,
        reasons=reasons,
    )
    score = _apply_score(
        predicate=features["component_count"] <= 1,
        score=2,
        reason="no_structural_components",
        running_score=score,
        reasons=reasons,
    )
    score = _apply_score(
        predicate=1 < features["component_count"] <= 3,
        score=1,
        reason="few_structural_components",
        running_score=score,
        reasons=reasons,
    )
    score = _apply_score(
        predicate=features["row_run_count"] == 0 or features["col_run_count"] == 0,
        score=3,
        reason="no_projection_runs",
        running_score=score,
        reasons=reasons,
    )
    score = _apply_score(
        predicate=features["row_run_count"] <= 1 and features["col_run_count"] <= 1,
        score=1,
        reason="weak_projection_runs",
        running_score=score,
        reasons=reasons,
    )
    score = _apply_score(
        predicate=features["row_activity_ratio"] <= 0.01 and features["col_activity_ratio"] <= 0.01,
        score=2,
        reason="near_zero_projection_activity",
        running_score=score,
        reasons=reasons,
    )
    score = _apply_score(
        predicate=features["std_pixel"] <= 2.0,
        score=2,
        reason="very_low_variance",
        running_score=score,
        reasons=reasons,
    )
    score = _apply_score(
        predicate=2.0 < features["std_pixel"] <= 6.0,
        score=1,
        reason="low_variance",
        running_score=score,
        reasons=reasons,
    )
    score = _apply_score(
        predicate=features["max_component_ratio"] <= 0.0005
        and features["foreground_ratio"] <= 0.002,
        score=2,
        reason="tiny_components_only",
        running_score=score,
        reasons=reasons,
    )
    return score, reasons


def _classify_blank_like(gray: np.ndarray) -> dict[str, object]:
    """Score one image for blankness using multiple structural features."""
    features = _extract_blank_features(gray)
    score, reasons = _score_blank_features(features)

    if score >= 9:
        classification = "confirmed_blank"
    elif score >= 5:
        classification = "suspect_blank"
    else:
        classification = "text_present"

    return {
        "classification": classification,
        "structural_classification": classification,
        "blank_score": score,
        "blank_reasons": reasons,
        **features,
    }


def audit_image_blankness(image_path: Path) -> dict[str, object]:
    """Audit one image using independent structural and resolution-signature passes."""
    gray = _load_grayscale(image_path)
    width, height = _load_image_size(image_path)
    audit = _classify_blank_like(gray)
    resolution_signature_match = (width, height) == BLANK_SIGNATURE_SIZE
    reasons = list(audit["blank_reasons"])
    if resolution_signature_match:
        reasons.append("matched_blank_signature_resolution")

    structural = str(audit["structural_classification"])
    structural_confirmed = structural == "confirmed_blank"
    structural_suspect = structural == "suspect_blank"
    if structural_confirmed and resolution_signature_match:
        classification = "confirmed_blank"
    elif structural_confirmed:
        classification = "suspect_blank"
        reasons.append("structural_blank_without_signature_match")
    elif structural_suspect:
        classification = "suspect_blank"
    else:
        classification = "text_present"

    return {
        **audit,
        "classification": classification,
        "blank_reasons": reasons,
        "image_width": width,
        "image_height": height,
        "resolution_signature_match": resolution_signature_match,
        "structural_confirmed_blank": structural_confirmed,
        "structural_suspect_blank": structural_suspect,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows as JSONL."""
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    """Summarize CER/WER/exact metrics over one prediction set."""
    if not rows:
        return {
            "num_rows": 0,
            "mean_cer": None,
            "mean_wer": None,
            "exact_rate": None,
        }
    return {
        "num_rows": len(rows),
        "mean_cer": fmean(float(row["cer"]) for row in rows),
        "mean_wer": fmean(float(row["wer"]) for row in rows),
        "exact_rate": fmean(1.0 if bool(row["exact"]) else 0.0 for row in rows),
    }


def extract_exact_false_debug_bundle(
    *,
    predictions_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Extract exact-false rows, copy their images, and emit robust blank-image audit artifacts."""
    rows = _read_predictions(predictions_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    confirmed_dir = output_dir / "confirmed_blank_images"
    suspect_dir = output_dir / "suspect_blank_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    confirmed_dir.mkdir(parents=True, exist_ok=True)
    suspect_dir.mkdir(parents=True, exist_ok=True)

    exact_false_rows: list[dict[str, object]] = []
    confirmed_blank_rows: list[dict[str, object]] = []
    suspect_blank_rows: list[dict[str, object]] = []
    classification_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    confirmed_modality_counts: Counter[str] = Counter()
    overlap_counts: Counter[str] = Counter()

    for row in rows:
        if bool(row.get("exact")):
            continue
        image_path = Path(str(row["image"]))
        modality = _infer_modality(str(row["image"]))
        audit = audit_image_blankness(image_path)
        enriched = {
            **row,
            "modality": modality,
            **audit,
        }
        exact_false_rows.append(enriched)
        modality_counts[modality] += 1
        classification_counts[str(enriched["classification"])] += 1
        if bool(enriched["resolution_signature_match"]) and bool(
            enriched["structural_confirmed_blank"]
        ):
            overlap_counts["confirmed_overlap"] += 1
        elif bool(enriched["resolution_signature_match"]):
            overlap_counts["resolution_only"] += 1
        elif bool(enriched["structural_confirmed_blank"]):
            overlap_counts["structural_only"] += 1
        elif bool(enriched["structural_suspect_blank"]):
            overlap_counts["structural_suspect_only"] += 1
        else:
            overlap_counts["text_present_only"] += 1
        shutil.copy2(image_path, images_dir / image_path.name)
        if enriched["classification"] == "confirmed_blank":
            confirmed_blank_rows.append(enriched)
            confirmed_modality_counts[modality] += 1
            shutil.copy2(image_path, confirmed_dir / image_path.name)
        elif enriched["classification"] == "suspect_blank":
            suspect_blank_rows.append(enriched)
            shutil.copy2(image_path, suspect_dir / image_path.name)

    confirmed_keys = {
        (row["image"], row["gt_text"], row["pred_text"]) for row in confirmed_blank_rows
    }
    filtered_rows = [
        row
        for row in rows
        if (row["image"], row["gt_text"], row["pred_text"]) not in confirmed_keys
    ]
    summary = {
        "predictions_file": str(predictions_path),
        "exact_false": {
            "num_rows": len(exact_false_rows),
            "by_modality": dict(modality_counts),
            "by_classification": dict(classification_counts),
            "signal_overlap": dict(overlap_counts),
        },
        "confirmed_blank": {
            "num_rows": len(confirmed_blank_rows),
            "by_modality": dict(confirmed_modality_counts),
        },
        "suspect_blank": {
            "num_rows": len(suspect_blank_rows),
        },
        "original_summary": _summarize(rows),
        "summary_excluding_confirmed_blank": _summarize(filtered_rows),
    }

    _write_jsonl(output_dir / "exact_false_predictions.jsonl", exact_false_rows)
    (output_dir / "exact_false_predictions.json").write_text(
        json.dumps(exact_false_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(output_dir / "confirmed_blank_predictions.jsonl", confirmed_blank_rows)
    _write_jsonl(output_dir / "suspect_blank_predictions.jsonl", suspect_blank_rows)
    _write_jsonl(output_dir / "predictions_excluding_confirmed_blank.jsonl", filtered_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
