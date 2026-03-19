from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from modules.ocr_inference.schemas import OCRPrediction, OCRTask, SourceArtifacts
from modules.ocr_training.surya_common import relative_to_base


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_inference_outputs(
    *,
    run_dir: Path,
    pdf_path: Path,
    predictions: list[OCRPrediction],
    tasks: list[OCRTask],
    source_artifacts: SourceArtifacts,
    model_info: dict[str, Any],
    diagnose: bool = False,
    diagnostic_written: bool = False,
) -> dict[str, str]:
    """Persist structured OCR run artifacts and per-language page outputs."""
    meta_dir = run_dir / "meta"
    predictions_dir = run_dir / "predictions"
    meta_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    run_manifest_path = meta_dir / "run_manifest.json"
    model_info_path = meta_dir / "model_info.json"
    source_artifacts_path = meta_dir / "source_artifacts.json"
    all_predictions_path = predictions_dir / "all_predictions.jsonl"
    page_predictions_path = predictions_dir / "page_predictions.json"
    diagnostic_images_dir = run_dir / "images"

    prediction_rows = [asdict(prediction) for prediction in predictions]

    run_manifest_path.write_text(
        json.dumps(
            {
                "doc_stem": pdf_path.stem,
                "pdf_path": relative_to_base(pdf_path),
                "run_dir": relative_to_base(run_dir),
                "task_count": len(tasks),
                "prediction_count": len(predictions),
                "languages": sorted({task.language for task in tasks}),
                "pages": sorted({task.page_number for task in tasks}),
                "model_mode": model_info["model_mode"],
                "diagnose": bool(diagnose),
                "diagnostic_written": bool(diagnostic_written),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    model_info_path.write_text(
        json.dumps(model_info, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    source_artifacts_path.write_text(
        json.dumps(
            {
                "crop_run_dir": relative_to_base(source_artifacts.crop_run_dir),
                "cropping_manifest": relative_to_base(source_artifacts.cropping_manifest),
                "spliced_dir": relative_to_base(source_artifacts.spliced_dir),
                "crop_registry_pointer": source_artifacts.crop_registry_pointer,
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    with all_predictions_path.open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[(row["language"], int(row["page_number"]))].append(row)

    page_summary: list[dict[str, Any]] = []
    for (language, page_number), rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        ordered_rows = sorted(rows, key=lambda row: int(row["ordering_index"]))
        aggregated_text = "\n\n".join(
            row["recognized_text"] for row in ordered_rows if row["recognized_text"].strip()
        )
        page_dir = run_dir / language / f"page_{page_number:03d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        (page_dir / "ocr.txt").write_text(aggregated_text, encoding="utf-8")
        (page_dir / "ocr.json").write_text(
            json.dumps(
                {
                    "doc_stem": pdf_path.stem,
                    "pdf_path": relative_to_base(pdf_path),
                    "language": language,
                    "page_number": page_number,
                    "aggregated_text": aggregated_text,
                    "entries": ordered_rows,
                },
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        page_summary.append(
            {
                "language": language,
                "page_number": page_number,
                "aggregated_text": aggregated_text,
                "entries": ordered_rows,
            }
        )

    page_predictions_path.write_text(
        json.dumps(
            {
                "doc_stem": pdf_path.stem,
                "pdf_path": relative_to_base(pdf_path),
                "model_mode": model_info["model_mode"],
                "diagnose": bool(diagnose),
                "diagnostic_written": bool(diagnostic_written),
                "pages": page_summary,
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    artifacts = {
        "run_manifest": relative_to_base(run_manifest_path),
        "model_info": relative_to_base(model_info_path),
        "source_artifacts": relative_to_base(source_artifacts_path),
        "all_predictions": relative_to_base(all_predictions_path),
        "page_predictions": relative_to_base(page_predictions_path),
    }
    if diagnose and diagnostic_written:
        artifacts["diagnostic_images_dir"] = relative_to_base(diagnostic_images_dir)
    return artifacts


def write_page_text_output(*, run_dir: Path, page_number: int, text: str) -> Path:
    """Write one page-level OCR text file in the generic source layout."""
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / f"page_{page_number:03d}.txt"
    output_path.write_text(text, encoding="utf-8")
    return output_path


def write_crop_text_output(*, run_dir: Path, language: str, page_number: int, text: str) -> Path:
    """Write one language/page OCR text file for crop-layout mode."""
    page_dir = run_dir / language / f"page_{page_number:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    output_path = page_dir / "ocr.txt"
    output_path.write_text(text, encoding="utf-8")
    return output_path
