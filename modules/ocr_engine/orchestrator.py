from pathlib import Path

from modules.ocr_engine.extractor import (
    extract_text_from_layout,
    load_recognition_predictor,
)
from modules.ocr_engine.layout.column_engine import (
    find_column_canyons,
    generate_poc_debug_visuals,
)
from modules.ocr_engine.layout_mapping import initialize_page_layout, map_boxes_to_columns
from modules.ocr_engine.layout_parser import (
    load_predictors,
)
from modules.ocr_engine.output_writer import write_json_output
from modules.ocr_engine.pre_processors.pdf_to_image import yield_pdf_pages
from utils.label_studio_adapter import generate_label_studio_task
from utils.logger import get_logger

logger = get_logger("Orchestrator")


def run_poc_debug_pipeline(
    pdf_path: Path,
    output_dir: Path,
    chunk_size: int = 50,
    dpi: int = 300,
    omit_pages: list[int] | None = None,
    max_pages: int | None = None,
) -> Path:
    """Proof-of-Concept Pipeline to visually verify the mathematical Column Slicer.

    This explicitly strips all destructive OpenCV pre-processing (no deskew, no binarization).
    It feeds pristine high-resolution images to Surya's Text Detection model, and then routes
    the mathematically perfect bounding boxes into our Gap-Seeking algorithm to slice the columns.

    Args:
        pdf_path: Path to source PDF.
        output_dir: Output base directory.
        chunk_size: Number of pages to process in each conversion batch.
        dpi: Dots per inch for pristine rendering.
        omit_pages: Pages to skip.
        max_pages: Max pages to process.
    """
    logger.info("Initializing Proof-of-Concept Visual Debug Pipeline...")

    # We only load the Text Detection model. (Layout and Recognition are disabled).
    from surya.detection import DetectionPredictor

    logger.info("Loading DetectionPredictor (~1.1GB model)... ⏳")
    det_predictor = DetectionPredictor()
    logger.info("DetectionPredictor loaded. ✅")

    debug_base_dir = output_dir / "pipeline_debug"

    # Process Images
    for page_num, img in yield_pdf_pages(
        pdf_path, chunk_size, dpi, omit_pages=omit_pages, max_pages=max_pages
    ):
        logger.info(f"--- Page {page_num}: Running PoC Layout Pipeline ---")

        # 1. Procedural Text Analysis on the unmodified image
        logger.info(f"Page {page_num}: Running text detection...")
        raw_predictions = det_predictor([img])[0]
        surya_bboxes = [pred.bbox for pred in raw_predictions.bboxes]

        # 2. Vision-Based Column Division (Fallback Placeholder)
        logger.info(f"Page {page_num}: Resolving semantic column boundaries...")
        slice_lines, fallback_triggered, _uncertainty, warnings, _ = find_column_canyons(
            img, surya_bboxes, expected_columns=3, alignment_lines=None
        )

        if warnings:
            for w in warnings:
                logger.warning(f"Page {page_num} layout warning: {w}")

        # 3. Generate Debug Visuals
        logger.info(f"Page {page_num}: Generating visual proofs...")
        generate_poc_debug_visuals(
            img,
            surya_bboxes,
            slice_lines,
            fallback_triggered,
            page_num,
            output_dir,
            alignment_lines=None,
        )

    return debug_base_dir


def process_pdf_to_structural_json(
    pdf_path: Path,
    output_dir: Path,
    chunk_size: int = 50,
    dpi: int = 300,
    slice_only: bool = False,
    omit_pages: list[int] | None = None,
    max_pages: int | None = None,
) -> Path:
    """The main logic-free routing orchestrator for the OCR Engine.

    1. Loads all Surya Predictor classes strictly into VRAM.
    2. Utilizes the memory-safe PDF-to-Image generator.
    3. Triggers Layout Parsing -> Structural Extractor.
    4. Dumps the Pydantic 'PageLayout' objects to a static JSON file in 'output/'.

    Args:
        pdf_path: Absolute path to the source manuscript.
        output_dir: Absolute path to the designated '/output' directory.
        chunk_size: The number of pages the generator yields at once to fit 64GB RAM.
        dpi: Dots Per Inch graphic resolution scalar.
        slice_only: If True, halts before OCR and generates Label Studio visualizer GUI imports.
        omit_pages: A list of specific integer page numbers to skip entirely.
        max_pages: Process only up to this many pages (useful for testing/sampling).

    Returns:
        Path: The location of the saved Structural JSON.
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if slice_only:
        json_filename = output_dir / f"{pdf_path.stem}_label_studio_tasks.json"

        # We create a sub-directory named after the PDF to organize visual outputs
        project_slug = pdf_path.stem
        visuals_dir = output_dir / "visuals" / project_slug
        visuals_dir.mkdir(parents=True, exist_ok=True)
    else:
        json_filename = output_dir / f"{pdf_path.stem}_structural.json"

    logger.info("--- Booting VRAM allocation for Surya OCR ---")

    # 1. Load Surya predictors (layout predictor omitted in this flow).
    det_predictor, _ = load_predictors(include_layout=False)

    rec_predictor = None
    if not slice_only:
        rec_predictor = load_recognition_predictor()

    logger.info("--- VRAM Models loaded. Engaging Ingest Pipeline ---")

    # We will accumulate the serialized Pydantic JSON strings for final disk write
    serialized_pages: list[dict] = []

    # Iterate safely through the PDF generator
    for page_number, high_res_img in yield_pdf_pages(
        pdf_path, chunk_size, dpi, omit_pages=omit_pages, max_pages=max_pages
    ):
        logger.info(f"--- Processing Page {page_number} ---")
        img_width, img_height = high_res_img.size

        # 1. Text Detection (Surya)
        logger.debug(f"Page {page_number}: Running global text detection...")
        det_results = det_predictor([high_res_img])[0]
        surya_bboxes = [pred.bbox for pred in det_results.bboxes]

        # 2. Vision-Based Column Slicing (YOLOv8)
        logger.debug(f"Page {page_number}: Slicing columns with Vision-AI...")
        slice_lines, fallback_triggered, uncertainty, warnings, _ = find_column_canyons(
            high_res_img, surya_bboxes, expected_columns=3
        )

        # 3. Create structural PageLayout shell
        layout = initialize_page_layout(
            page_number=page_number,
            img_width=img_width,
            img_height=img_height,
            fallback_triggered=fallback_triggered,
            uncertainty=uncertainty,
            warnings=warnings,
        )

        # 4. Map text boxes to columns based on slicer outputs.
        layout.columns = map_boxes_to_columns(
            surya_bboxes=surya_bboxes,
            slice_lines=slice_lines,
            img_width=img_width,
            img_height=img_height,
        )

        # 5. Route to output mechanism
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

        # 6. Text Recognition (OCR)
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

    # Python's GC will drop the models from VRAM upon returning since scopes end
    logger.info("OCR Pipeline Complete. VRAM is released.")

    return json_filename
