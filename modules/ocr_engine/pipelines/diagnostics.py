from __future__ import annotations

from pathlib import Path

from modules.ocr_engine.layout.column_engine import (
    find_column_canyons,
    generate_layout_diagnostics_visuals,
)
from modules.ocr_engine.pre_processors.pdf_to_image import yield_pdf_pages
from utils.logger import get_logger
from utils.run_registry import next_versioned_dir, register_latest_run

logger = get_logger("OCRLayoutDiagnosticsPipeline")


def run_layout_diagnostics_pipeline(
    pdf_path: Path,
    output_dir: Path,
    chunk_size: int = 50,
    dpi: int = 300,
    start_page: int = 1,
    omit_pages: list[int] | None = None,
    end_page: int | None = None,
) -> Path:
    """Run text-detection-only visual diagnostics for column slicing."""
    logger.info("Initializing layout diagnostics pipeline...")
    logger.info(
        "Layout diagnostics uses Surya for line detection and Berana YOLO divider weights "
        "for column boundary resolution."
    )

    from surya.detection import DetectionPredictor

    logger.info("Loading DetectionPredictor (~1.1GB model)...")
    det_predictor = DetectionPredictor()
    logger.info("DetectionPredictor loaded. ✅")

    run_dir = next_versioned_dir(output_dir, pdf_path.stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = run_dir / "visuals"
    meta_dir = run_dir / "meta"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    for page_num, image in yield_pdf_pages(
        pdf_path,
        chunk_size,
        dpi,
        start_page=start_page,
        omit_pages=omit_pages,
        end_page=end_page,
    ):
        logger.info(f"--- Page {page_num}: Running Layout Diagnostics ---")
        logger.info(f"Page {page_num}: Running text detection...")
        raw_predictions = det_predictor([image])[0]
        surya_bboxes = [pred.bbox for pred in raw_predictions.bboxes]

        logger.info(f"Page {page_num}: Resolving semantic column boundaries...")
        slice_lines, _fallback_triggered, _uncertainty, warnings, _ = find_column_canyons(
            image, surya_bboxes, expected_columns=3, alignment_lines=None
        )

        for warning in warnings:
            logger.warning(f"Page {page_num} layout warning: {warning}")

        logger.info(f"Page {page_num}: Generating diagnostics visuals...")
        generate_layout_diagnostics_visuals(
            image,
            slice_lines,
            page_num,
            visuals_dir,
            surya_bboxes=surya_bboxes,
        )

    pointer = register_latest_run(
        stage="layout-diagnostics",
        doc_stem=pdf_path.stem,
        run_dir=run_dir,
        artifacts={"visuals_dir": str(visuals_dir)},
        metadata={
            "start_page": start_page,
            "end_page": end_page,
            "chunk_size": chunk_size,
            "dpi": dpi,
        },
    )
    logger.info(f"Latest pointer updated: {pointer}")
    return visuals_dir
