from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from modules.ocr_engine.output_writer import write_json_output
from modules.ocr_engine.pre_processors.pdf_to_image import yield_pdf_pages
from modules.ocr_engine.pre_processors.splicing.engine import SplicingEngine
from utils.logger import get_logger
from utils.run_registry import next_versioned_dir, register_latest_run

logger = get_logger("OCRPrecisionPipeline")


def run_precision_extraction_pipeline(
    pdf_path: Path,
    output_dir: Path,
    rectify_mode: str = "rotate+homography",
    chunk_size: int = 50,
    dpi: int = 300,
    start_page: int = 1,
    end_page: int | None = None,
    omit_pages: list[int] | None = None,
) -> Path:
    """Run HITL-divider-based column cropping only (no OCR)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = next_versioned_dir(output_dir, pdf_path.stem)
    spliced_dir = artifact_dir / "spliced"
    spliced_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run artifact directory: {artifact_dir}")
    logger.info("--- Starting column cropping pass (OCR disabled) ---")

    splicing_engine = SplicingEngine()

    manifest_output_path = artifact_dir / "cropping_manifest.json"
    quality_report_path = artifact_dir / "quality_report.json"

    page_records: list[dict] = []
    quality_records: list[dict] = []
    failed_pages: list[dict[str, str]] = []
    planned_total = _planned_pages_total(
        pdf_path=pdf_path,
        start_page=start_page,
        end_page=end_page,
        omit_pages=omit_pages,
    )
    progress = tqdm(total=planned_total, desc="Column Cropping", unit="page", dynamic_ncols=True)

    for page_number, high_res_img in yield_pdf_pages(
        pdf_path,
        chunk_size,
        dpi,
        start_page=start_page,
        end_page=end_page,
        omit_pages=omit_pages,
    ):
        page_id = f"page_{page_number:03d}"
        result = splicing_engine.splice_page(high_res_img, page_id, rectify_mode=rectify_mode)

        quality_records.append(
            {
                "page_id": page_id,
                "status": result["status"],
                "qc": result.get("qc", {}),
            }
        )
        if result["status"] != "SUCCESS":
            failed_pages.append({"page_id": page_id, "status": result["status"]})
            progress.update(1)
            continue

        strips = result["strips"]
        page_splice_dir = spliced_dir / page_id
        page_splice_dir.mkdir(parents=True, exist_ok=True)

        strip_paths: dict[str, str] = {}
        for lang_key, strip_img in strips.items():
            output_path = page_splice_dir / f"{lang_key}.png"
            strip_img.save(output_path)
            strip_paths[lang_key] = str(output_path)

        page_records.append(
            {
                "page_id": page_id,
                "source_page": page_number,
                "status": "SUCCESS",
                "strip_paths": strip_paths,
                "offsets": result["offsets"],
            }
        )
        progress.update(1)

    progress.close()

    write_json_output(page_records, manifest_output_path)
    write_json_output(quality_records, quality_report_path)
    pointer = register_latest_run(
        stage="crop-columns",
        doc_stem=pdf_path.stem,
        run_dir=artifact_dir,
        artifacts={
            "cropping_manifest": str(manifest_output_path),
            "quality_report": str(quality_report_path),
            "spliced_dir": str(spliced_dir),
        },
        metadata={
            "rectify_mode": rectify_mode,
            "start_page": start_page,
            "end_page": end_page,
            "chunk_size": chunk_size,
            "dpi": dpi,
        },
    )
    logger.info(f"Latest pointer updated: {pointer}")
    if failed_pages:
        failed_summary = ", ".join(
            f"{entry['page_id']}:{entry['status']}" for entry in failed_pages
        )
        logger.warning(f"Column cropping completed with failures: {failed_summary}")
    logger.info(f"Column cropping finished. Output: {artifact_dir}")
    return manifest_output_path


def _planned_pages_total(
    pdf_path: Path,
    start_page: int,
    end_page: int | None,
    omit_pages: list[int] | None,
) -> int:
    """Estimate number of pages expected for progress reporting."""
    from pdf2image.pdf2image import pdfinfo_from_path

    info = pdfinfo_from_path(str(pdf_path))
    total_pages = int(info["Pages"])
    first_page = max(1, start_page)
    last_page = min(total_pages, end_page) if end_page is not None else total_pages
    if last_page < first_page:
        return 0

    skip_set = set(omit_pages) if omit_pages else set()
    count = 0
    for page in range(first_page, last_page + 1):
        if page not in skip_set:
            count += 1
    return count
