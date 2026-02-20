import json
import logging
from pathlib import Path

from modules.ocr_engine.extractor import (
    extract_text_from_layout,
    load_rec_model,
    load_rec_processor,
)
from modules.ocr_engine.layout_parser import (
    extract_layout_and_boxes,
    load_det_model,
    load_det_processor,
    load_layout_model,
    load_layout_processor,
)
from modules.ocr_engine.pre_processors.pdf_to_image import yield_pdf_pages
from schemas.ocr_models import PageLayout

logger = logging.getLogger(__name__)


def process_pdf_to_structural_json(
    pdf_path: Path, output_dir: Path, chunk_size: int = 50, dpi: int = 300
) -> Path:
    """The main logic-free routing orchestrator for the OCR Engine.

    1. Loads all Surya models strictly into VRAM.
    2. Utilizes the memory-safe PDF-to-Image generator.
    3. Triggers Layout Parsing -> Structural Extractor.
    4. Dumps the Pydantic 'PageLayout' objects to a static JSON file in 'output/'.

    Args:
        pdf_path: Absolute path to the source manuscript.
        output_dir: Absolute path to the designated '/output' directory.
        chunk_size: The number of pages the generator yields at once to fit 64GB RAM.
        dpi: Dots Per Inch graphic resolution scalar.

    Returns:
        Path: The location of the saved Structural JSON.
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    json_filename = output_dir / f"{pdf_path.stem}_structural.json"

    logger.info("--- Booting VRAM allocation for Surya OCR ---")

    # 1. Load the Layout and Detection Vision Models
    # (Memory Management Caution: these reside together in VRAM during processing)
    det_processor = load_det_processor()
    det_model = load_det_model()
    layout_processor = load_layout_processor()
    layout_model = load_layout_model()

    # 2. Load Text Recognition Model
    rec_processor = load_rec_processor()
    rec_model = load_rec_model()

    logger.info("--- VRAM Models loaded. Engaging Ingest Pipeline ---")

    # We will accumulate the serialized Pydantic JSON strings for final disk write
    serialized_pages: list[dict] = []

    # Iterate safely through the PDF generator
    for page_number, high_res_img in yield_pdf_pages(pdf_path, chunk_size, dpi):
        # Step A: Parse Layout to secure Geographic language zones
        page_layout: PageLayout = extract_layout_and_boxes(
            image=high_res_img,
            page_number=page_number,
            det_model=det_model,
            det_processor=det_processor,
            layout_model=layout_model,
            layout_processor=layout_processor,
        )

        # Step B: Pass bounded layout to Text Extractor to transcribe characters
        completed_layout: PageLayout = extract_text_from_layout(
            image=high_res_img,
            layout=page_layout,
            rec_model=rec_model,
            rec_processor=rec_processor,
        )

        # Step C: Append validated and strictly-typed extraction to the payload array
        serialized_pages.append(completed_layout.model_dump())
        logger.debug(f"Page {page_number} successfully transcribed and appended to payload string.")

    logger.info(f"Writing {len(serialized_pages)} completed pages to {json_filename}")

    # Standard output dump
    with json_filename.open("w", encoding="utf-8") as f:
        json.dump(serialized_pages, f, ensure_ascii=False, indent=2)

    # Python's GC will drop the models from VRAM upon returning since scopes end
    logger.info("OCR Pipeline Complete. VRAM is released.")

    return json_filename
