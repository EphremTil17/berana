from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev

from modules.ocr_training.surya_debug import audit_image_blankness

HEURISTIC_EXCLUSION_FILES = {
    "confirmed_blank": "confirmed_blank_predictions.jsonl",
    "suspect_blank": "suspect_blank_predictions.jsonl",
    "likely_label_mismatch": "likely_label_mismatch_predictions.jsonl",
    "likely_artifact": "likely_artifact_predictions.jsonl",
    "cer_outlier_2std": "cer_outliers_2std.jsonl",
    "cer_outlier_3std": "cer_outliers_3std.jsonl",
    "wer_outlier_2std": "wer_outliers_2std.jsonl",
    "wer_outlier_3std": "wer_outliers_3std.jsonl",
}


def _infer_modality(image_path: str) -> str:
    if "fidel_synthetic__" in image_path:
        return "synthetic"
    if "typed" in image_path or "synth_image_" in image_path:
        return "typed"
    return "unknown"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def load_heuristic_exclusion_index(heuristic_cleanup_dir: Path) -> dict[str, object]:
    """Load one heuristic cleanup bundle into basename-indexed exclusion categories."""
    if not heuristic_cleanup_dir.exists():
        raise FileNotFoundError(f"Missing heuristic cleanup directory: {heuristic_cleanup_dir}")

    by_category: dict[str, set[str]] = {}
    by_basename: dict[str, set[str]] = {}
    for category, filename in HEURISTIC_EXCLUSION_FILES.items():
        path = heuristic_cleanup_dir / filename
        basenames: set[str] = set()
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    image_value = str(row.get("image") or "").strip()
                    if not image_value:
                        continue
                    basename = Path(image_value).name
                    basenames.add(basename)
                    by_basename.setdefault(basename, set()).add(category)
        by_category[category] = basenames
    return {
        "heuristic_cleanup_dir": str(heuristic_cleanup_dir),
        "categories": {category: sorted(values) for category, values in by_category.items()},
        "all_basenames": set(by_basename),
        "by_basename": by_basename,
        "counts": {category: len(values) for category, values in by_category.items()},
    }


def _safe_audit_image(image_path: Path) -> dict[str, object]:
    if not image_path.exists():
        return {
            "image_available": False,
            "classification": "unavailable",
            "structural_classification": "unavailable",
            "blank_score": None,
            "blank_reasons": ["image_missing"],
            "image_width": None,
            "image_height": None,
            "resolution_signature_match": False,
            "structural_confirmed_blank": False,
            "structural_suspect_blank": False,
        }
    return {
        "image_available": True,
        **audit_image_blankness(image_path),
    }


def _materialize_image(source: Path, target_dir: Path) -> None:
    if not source.exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if target.exists():
        return
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def _is_likely_label_mismatch(row: dict[str, object], thresholds: dict[str, float]) -> bool:
    _ = thresholds
    return (
        bool(row["image_available"])
        and row["classification"] == "text_present"
        and row["gt_len"] >= 40
        and row["pred_len"] >= 40
        and float(row["cer"]) >= 0.5
        and float(row["wer"]) >= 0.5
    )


def _record_outlier_flags(
    row: dict[str, object],
    thresholds: dict[str, float],
    cer_outliers_2std: list[dict[str, object]],
    cer_outliers_3std: list[dict[str, object]],
    wer_outliers_2std: list[dict[str, object]],
    wer_outliers_3std: list[dict[str, object]],
) -> tuple[float, float]:
    cer = float(row["cer"])
    wer = float(row["wer"])
    row["cer_outlier_2std"] = cer >= thresholds["cer_2std"]
    row["cer_outlier_3std"] = cer >= thresholds["cer_3std"]
    row["wer_outlier_2std"] = wer >= thresholds["wer_2std"]
    row["wer_outlier_3std"] = wer >= thresholds["wer_3std"]
    if row["cer_outlier_2std"]:
        cer_outliers_2std.append(row)
    if row["cer_outlier_3std"]:
        cer_outliers_3std.append(row)
    if row["wer_outlier_2std"]:
        wer_outliers_2std.append(row)
    if row["wer_outlier_3std"]:
        wer_outliers_3std.append(row)
    return cer, wer


def _collect_prediction_data(
    *,
    predictions_path: Path,
    images_dir: Path,
) -> tuple[
    int,
    int,
    list[float],
    list[float],
    list[dict[str, object]],
    Counter[str],
    Counter[str],
    list[dict[str, object]],
]:
    """Collect global stats and enriched exact-false rows from one predictions JSONL file."""
    total_rows = 0
    exact_true_total = 0
    cer_values: list[float] = []
    wer_values: list[float] = []
    exact_false_rows: list[dict[str, object]] = []
    classification_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    missing_image_rows: list[dict[str, object]] = []

    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            total_rows += 1
            exact_true_total += 1 if bool(row.get("exact")) else 0
            cer = float(row["cer"])
            wer = float(row["wer"])
            cer_values.append(cer)
            wer_values.append(wer)
            if bool(row.get("exact")):
                continue

            image_path = Path(str(row["image"]))
            audit = _safe_audit_image(image_path)
            enriched = {
                **row,
                "modality": _infer_modality(str(row["image"])),
                "gt_len": len(str(row["gt_text"]).strip()),
                "pred_len": len(str(row["pred_text"]).strip()),
                **audit,
            }
            exact_false_rows.append(enriched)
            modality_counts[str(enriched["modality"])] += 1
            classification_counts[str(enriched["classification"])] += 1
            if not bool(enriched["image_available"]):
                missing_image_rows.append(enriched)
                continue
            _materialize_image(image_path, images_dir)
    return (
        total_rows,
        exact_true_total,
        cer_values,
        wer_values,
        exact_false_rows,
        classification_counts,
        modality_counts,
        missing_image_rows,
    )


def _thresholds_from_metrics(cer_values: list[float], wer_values: list[float]) -> dict[str, float]:
    if not cer_values:
        raise ValueError("No prediction rows found in predictions artifact.")
    mean_cer = fmean(cer_values)
    mean_wer = fmean(wer_values)
    std_cer = pstdev(cer_values)
    std_wer = pstdev(wer_values)
    return {
        "mean_cer": mean_cer,
        "std_cer": std_cer,
        "mean_wer": mean_wer,
        "std_wer": std_wer,
        "cer_2std": mean_cer + (2.0 * std_cer),
        "cer_3std": mean_cer + (3.0 * std_cer),
        "wer_2std": mean_wer + (2.0 * std_wer),
        "wer_3std": mean_wer + (3.0 * std_wer),
    }


def _categorize_exact_false_rows(
    *,
    exact_false_rows: list[dict[str, object]],
    thresholds: dict[str, float],
    confirmed_dir: Path,
    suspect_dir: Path,
    mismatch_dir: Path,
    artifact_dir: Path,
) -> dict[str, list[dict[str, object]]]:
    confirmed_blank_rows: list[dict[str, object]] = []
    suspect_blank_rows: list[dict[str, object]] = []
    likely_label_mismatch_rows: list[dict[str, object]] = []
    likely_artifact_rows: list[dict[str, object]] = []
    cer_outliers_2std: list[dict[str, object]] = []
    cer_outliers_3std: list[dict[str, object]] = []
    wer_outliers_2std: list[dict[str, object]] = []
    wer_outliers_3std: list[dict[str, object]] = []

    for row in exact_false_rows:
        _record_outlier_flags(
            row,
            thresholds,
            cer_outliers_2std,
            cer_outliers_3std,
            wer_outliers_2std,
            wer_outliers_3std,
        )

        if row["classification"] == "confirmed_blank":
            confirmed_blank_rows.append(row)
            if bool(row["image_available"]):
                _materialize_image(Path(str(row["image"])), confirmed_dir)
            continue
        if row["classification"] == "suspect_blank":
            suspect_blank_rows.append(row)
            if bool(row["image_available"]):
                _materialize_image(Path(str(row["image"])), suspect_dir)
                _materialize_image(Path(str(row["image"])), artifact_dir)
            likely_artifact_rows.append(row)
            continue
        if _is_likely_label_mismatch(row, thresholds):
            likely_label_mismatch_rows.append(row)
            _materialize_image(Path(str(row["image"])), mismatch_dir)
    return {
        "confirmed_blank_rows": confirmed_blank_rows,
        "suspect_blank_rows": suspect_blank_rows,
        "likely_label_mismatch_rows": likely_label_mismatch_rows,
        "likely_artifact_rows": likely_artifact_rows,
        "cer_outliers_2std": cer_outliers_2std,
        "cer_outliers_3std": cer_outliers_3std,
        "wer_outliers_2std": wer_outliers_2std,
        "wer_outliers_3std": wer_outliers_3std,
    }


def analyze_predictions_failures(
    *,
    predictions_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Analyze one predictions JSONL artifact and emit exact-false/outlier review bundles."""
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    exact_false_dir = output_dir / "exact_false"
    images_dir = exact_false_dir / "images"
    confirmed_dir = exact_false_dir / "confirmed_blank_images"
    suspect_dir = exact_false_dir / "suspect_blank_images"
    mismatch_dir = exact_false_dir / "likely_label_mismatch_images"
    artifact_dir = exact_false_dir / "likely_artifact_images"
    exact_false_dir.mkdir(parents=True, exist_ok=True)

    (
        total_rows,
        exact_true_total,
        cer_values,
        wer_values,
        exact_false_rows,
        classification_counts,
        modality_counts,
        missing_image_rows,
    ) = _collect_prediction_data(
        predictions_path=predictions_path,
        images_dir=images_dir,
    )

    thresholds = _thresholds_from_metrics(cer_values, wer_values)
    categorized = _categorize_exact_false_rows(
        exact_false_rows=exact_false_rows,
        thresholds=thresholds,
        confirmed_dir=confirmed_dir,
        suspect_dir=suspect_dir,
        mismatch_dir=mismatch_dir,
        artifact_dir=artifact_dir,
    )

    confirmed_blank_rows = categorized["confirmed_blank_rows"]
    suspect_blank_rows = categorized["suspect_blank_rows"]
    likely_label_mismatch_rows = categorized["likely_label_mismatch_rows"]
    likely_artifact_rows = categorized["likely_artifact_rows"]
    cer_outliers_2std = categorized["cer_outliers_2std"]
    cer_outliers_3std = categorized["cer_outliers_3std"]
    wer_outliers_2std = categorized["wer_outliers_2std"]
    wer_outliers_3std = categorized["wer_outliers_3std"]

    _write_jsonl(exact_false_dir / "exact_false_predictions.jsonl", exact_false_rows)
    (exact_false_dir / "exact_false_predictions.json").write_text(
        json.dumps(exact_false_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(exact_false_dir / "confirmed_blank_predictions.jsonl", confirmed_blank_rows)
    _write_jsonl(exact_false_dir / "suspect_blank_predictions.jsonl", suspect_blank_rows)
    _write_jsonl(
        exact_false_dir / "likely_label_mismatch_predictions.jsonl", likely_label_mismatch_rows
    )
    _write_jsonl(exact_false_dir / "likely_artifact_predictions.jsonl", likely_artifact_rows)
    _write_jsonl(exact_false_dir / "missing_image_predictions.jsonl", missing_image_rows)
    _write_jsonl(exact_false_dir / "cer_outliers_2std.jsonl", cer_outliers_2std)
    _write_jsonl(exact_false_dir / "cer_outliers_3std.jsonl", cer_outliers_3std)
    _write_jsonl(exact_false_dir / "wer_outliers_2std.jsonl", wer_outliers_2std)
    _write_jsonl(exact_false_dir / "wer_outliers_3std.jsonl", wer_outliers_3std)

    summary = {
        "predictions_file": str(predictions_path),
        "num_rows": total_rows,
        "exact_false_count": len(exact_false_rows),
        "exact_false_rate": len(exact_false_rows) / total_rows,
        "exact_rate": exact_true_total / total_rows,
        "metrics": {
            "mean_cer": thresholds["mean_cer"],
            "std_cer": thresholds["std_cer"],
            "mean_wer": thresholds["mean_wer"],
            "std_wer": thresholds["std_wer"],
            "cer_outlier_2std_threshold": thresholds["cer_2std"],
            "cer_outlier_3std_threshold": thresholds["cer_3std"],
            "wer_outlier_2std_threshold": thresholds["wer_2std"],
            "wer_outlier_3std_threshold": thresholds["wer_3std"],
        },
        "exact_false": {
            "by_modality": dict(modality_counts),
            "by_classification": dict(classification_counts),
            "missing_images": len(missing_image_rows),
            "confirmed_blank": len(confirmed_blank_rows),
            "suspect_blank": len(suspect_blank_rows),
            "likely_label_mismatch": len(likely_label_mismatch_rows),
            "likely_artifact": len(likely_artifact_rows),
            "cer_outliers_2std": len(cer_outliers_2std),
            "cer_outliers_3std": len(cer_outliers_3std),
            "wer_outliers_2std": len(wer_outliers_2std),
            "wer_outliers_3std": len(wer_outliers_3std),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# OCR Failure Analysis",
                "",
                f"- predictions: `{predictions_path}`",
                f"- rows: `{total_rows}`",
                f"- exact false: `{len(exact_false_rows)}`",
                f"- exact rate: `{summary['exact_rate']:.6f}`",
                f"- mean CER: `{thresholds['mean_cer']:.6f}`",
                f"- mean WER: `{thresholds['mean_wer']:.6f}`",
                "",
                "## Exact-False Findings",
                "",
                f"- confirmed blank: `{len(confirmed_blank_rows)}`",
                f"- suspect blank / artifact-like: `{len(suspect_blank_rows)}`",
                f"- likely label mismatch: `{len(likely_label_mismatch_rows)}`",
                f"- missing local images: `{len(missing_image_rows)}`",
                f"- CER outliers > 3 std: `{len(cer_outliers_3std)}`",
                f"- WER outliers > 3 std: `{len(wer_outliers_3std)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary
