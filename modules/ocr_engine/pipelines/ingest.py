from __future__ import annotations

from pathlib import Path

from modules.ocr_engine.extractor import extract_text_from_layout, load_recognition_predictor
from modules.ocr_engine.layout.column_engine import find_column_canyons
from modules.ocr_engine.layout_mapping import initialize_page_layout, map_boxes_to_columns
from modules.ocr_engine.layout_parser import load_predictors
from modules.ocr_engine.output_writer import write_json_output
from modules.ocr_engine.pre_processors.pdf_to_image import yield_pdf_pages
from utils.label_studio_adapter import generate_label_studio_task
from utils.logger import get_logger
from utils.run_registry import next_versioned_dir, register_latest_run

logger = get_logger("OCRIngestPipeline")


def process_pdf_to_structural_json(
    pdf_path: Path,
    output_dir: Path,
    chunk_size: int = 50,
    dpi: int = 300,
    slice_only: bool = False,
    start_page: int = 1,
    omit_pages: list[int] | None = None,
    end_page: int | None = None,
) -> Path:
    """Run ingest pipeline and emit either structural JSON or Label Studio tasks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = next_versioned_dir(output_dir / "ingest", pdf_path.stem)
    run_dir.mkdir(parents=True, exist_ok=True)

    if slice_only:
        json_filename = run_dir / "label_studio_tasks.json"
        project_slug = pdf_path.stem
        visuals_dir = run_dir / "visuals" / project_slug
        visuals_dir.mkdir(parents=True, exist_ok=True)
    else:
        json_filename = run_dir / "structural.json"

    logger.info("--- Booting VRAM allocation for Surya OCR ---")
    det_predictor, _ = load_predictors(include_layout=False)

    rec_predictor = None
    if not slice_only:
        rec_predictor = load_recognition_predictor()

    logger.info("--- VRAM Models loaded. Engaging Ingest Pipeline ---")
    serialized_pages: list[dict] = []

    for page_number, high_res_img in yield_pdf_pages(
        pdf_path,
        chunk_size,
        dpi,
        start_page=start_page,
        omit_pages=omit_pages,
        end_page=end_page,
    ):
        logger.info(f"--- Processing Page {page_number} ---")
        img_width, img_height = high_res_img.size

        logger.debug(f"Page {page_number}: Running global text detection...")
        det_results = det_predictor([high_res_img])[0]
        surya_bboxes = [pred.bbox for pred in det_results.bboxes]

        logger.debug(f"Page {page_number}: Slicing columns with Vision-AI...")
        slice_lines, fallback_triggered, uncertainty, warnings, _ = find_column_canyons(
            high_res_img, surya_bboxes, expected_columns=3
        )

        layout = initialize_page_layout(
            page_number=page_number,
            img_width=img_width,
            img_height=img_height,
            fallback_triggered=fallback_triggered,
            uncertainty=uncertainty,
            warnings=warnings,
        )
        layout.columns = map_boxes_to_columns(
            surya_bboxes=surya_bboxes,
            slice_lines=slice_lines,
            img_width=img_width,
            img_height=img_height,
        )

        if slice_only:
            image_filename = f"page_{page_number:03d}.jpg"
            physical_image_path = visuals_dir / image_filename
            high_res_img.save(physical_image_path, "JPEG", quality=85)
            ls_task = generate_label_studio_task(
                image_filename=f"{project_slug}/{image_filename}",
                img_width=img_width,
                img_height=img_height,
                columns=layout.columns,
            )
            serialized_pages.append(ls_task)
            logger.info(f"Page {page_number} ✅ Visual analysis exported.")
            continue

        logger.info(f"Page {page_number}: Transcribing text with Surya...")
        final_layout = extract_text_from_layout(
            image=high_res_img,
            layout=layout,
            rec_predictor=rec_predictor,
            det_predictor=det_predictor,
        )
        serialized_pages.append(final_layout.model_dump())
        logger.info(f"Page {page_number} ✅ Transcription complete.")

    logger.info(f"Writing {len(serialized_pages)} completed pages to {json_filename}")
    write_json_output(records=serialized_pages, output_path=json_filename)
    artifacts = {"output_json": str(json_filename)}
    if slice_only:
        artifacts["visuals_dir"] = str(visuals_dir)
    pointer = register_latest_run(
        stage="ingest",
        doc_stem=pdf_path.stem,
        run_dir=run_dir,
        artifacts=artifacts,
        metadata={
            "slice_only": slice_only,
            "start_page": start_page,
            "end_page": end_page,
            "chunk_size": chunk_size,
            "dpi": dpi,
        },
    )
    logger.info(f"Latest pointer updated: {pointer}")
    logger.info("OCR Pipeline Complete. VRAM is released.")
    return json_filename
