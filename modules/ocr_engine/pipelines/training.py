from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from modules.ocr_engine.pre_processors.splicing.engine import SplicingEngine
from utils.logger import get_logger
from utils.run_registry import (
    load_latest_run,
    next_versioned_dir,
    register_latest_run,
    resolve_required_input,
)

logger = get_logger("OCRTrainingPipeline")


def run_ocr_training_pipeline(
    pdf_path: Path,
    output_dir: Path,
    run_name: str = "ocr_train_v1",
    rectify_mode: str = "rotate+homography",
    chunk_size: int = 50,
    dpi: int = 300,
    start_page: int = 1,
    end_page: int | None = None,
    omit_pages: list[int] | None = None,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
) -> Path:
    """Scaffold OCR training pipeline and persist a run manifest.

    This placeholder captures a stable interface and run contract while model fine-tuning
    implementation is introduced in a future phase.
    """
    divider_source = SplicingEngine()
    latest_crops = load_latest_run("crop-columns", pdf_path.stem)
    crop_manifest = None
    try:
        crop_manifest = str(
            resolve_required_input(
                upstream_stage="crop-columns",
                doc_stem=pdf_path.stem,
                artifact_key="cropping_manifest",
            )
        )
    except (FileNotFoundError, KeyError):
        crop_manifest = None
    run_dir = next_versioned_dir(output_dir, pdf_path.stem)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "mode": "ocr-train",
        "status": "scaffold_only",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "pdf_path": str(pdf_path),
            "divider_source_kind": divider_source.source_kind,
            "divider_source_path": str(divider_source.source_path),
            "crop_columns_latest": latest_crops,
            "crop_columns_manifest": crop_manifest,
        },
        "pagination": {
            "start_page": start_page,
            "end_page": end_page,
            "omit_pages": omit_pages or [],
        },
        "runtime": {
            "rectify_mode": rectify_mode,
            "chunk_size": chunk_size,
            "dpi": dpi,
            "run_name": run_name,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
        "notes": [
            "Pipeline interface scaffolded.",
            "Model fine-tuning implementation pending.",
        ],
    }

    manifest_path = run_dir / "training_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    pointer = register_latest_run(
        stage="ocr-train",
        doc_stem=pdf_path.stem,
        run_dir=run_dir,
        artifacts={"training_manifest": str(manifest_path)},
        metadata={"run_name": run_name},
    )
    logger.info(f"Latest pointer updated: {pointer}")
    logger.info(f"OCR training scaffold created at: {run_dir}")
    return manifest_path
