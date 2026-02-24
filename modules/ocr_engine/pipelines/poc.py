from __future__ import annotations

from pathlib import Path

from modules.ocr_engine.layout.column_engine import find_column_canyons, generate_poc_debug_visuals
from modules.ocr_engine.pre_processors.pdf_to_image import yield_pdf_pages
from utils.logger import get_logger
from utils.run_registry import next_versioned_dir, register_latest_run

logger = get_logger("OCRPocPipeline")


def run_poc_debug_pipeline(
    pdf_path: Path,
    output_dir: Path,
    chunk_size: int = 50,
    dpi: int = 300,
    start_page: int = 1,
    omit_pages: list[int] | None = None,
    end_page: int | None = None,
) -> Path:
    """Run text-detection-only visual diagnostics for column slicing."""
    logger.info("Initializing Proof-of-Concept Visual Debug Pipeline...")

    from surya.detection import DetectionPredictor

    logger.info("Loading DetectionPredictor (~1.1GB model)... ⏳")
    det_predictor = DetectionPredictor()
    logger.info("DetectionPredictor loaded. ✅")

    run_dir = next_versioned_dir(output_dir / "poc", pdf_path.stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    debug_base_dir = run_dir / "pipeline_debug"

    for page_num, image in yield_pdf_pages(
        pdf_path,
        chunk_size,
        dpi,
        start_page=start_page,
        omit_pages=omit_pages,
        end_page=end_page,
    ):
        logger.info(f"--- Page {page_num}: Running PoC Layout Pipeline ---")
        logger.info(f"Page {page_num}: Running text detection...")
        raw_predictions = det_predictor([image])[0]
        surya_bboxes = [pred.bbox for pred in raw_predictions.bboxes]

        logger.info(f"Page {page_num}: Resolving semantic column boundaries...")
        slice_lines, fallback_triggered, _uncertainty, warnings, _ = find_column_canyons(
            image, surya_bboxes, expected_columns=3, alignment_lines=None
        )

        for warning in warnings:
            logger.warning(f"Page {page_num} layout warning: {warning}")

        logger.info(f"Page {page_num}: Generating visual proofs...")
        generate_poc_debug_visuals(
            image,
            surya_bboxes,
            slice_lines,
            fallback_triggered,
            page_num,
            run_dir,
            alignment_lines=None,
        )

    pointer = register_latest_run(
        stage="poc-slicer",
        doc_stem=pdf_path.stem,
        run_dir=run_dir,
        artifacts={"debug_dir": str(debug_base_dir)},
        metadata={
            "start_page": start_page,
            "end_page": end_page,
            "chunk_size": chunk_size,
            "dpi": dpi,
        },
    )
    logger.info(f"Latest pointer updated: {pointer}")
    return debug_base_dir
