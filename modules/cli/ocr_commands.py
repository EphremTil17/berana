import logging
from pathlib import Path
from typing import Annotated

import typer

from modules.cli.common import ensure_pdf_exists, parse_omit_pages
from utils.logger import get_logger

log = get_logger("OCRCLI")


def run_ingest(
    pdf_path: Annotated[str, typer.Option("--pdf-path", help="Path to the source liturgical PDF.")],
    chunk_size: int = typer.Option(
        50, "--chunk-size", help="How many pages to load into RAM at once."
    ),
    dpi: int = typer.Option(300, "--dpi", help="Image processing resolution."),
    slice_only: bool = typer.Option(
        False,
        "--slice-only",
        help="Run the OpenCV Grid-Slicer and export Label Studio GUI JSON (skips Text OCR).",
    ),
    start_page: int = typer.Option(1, "--start-page", help="Page number to begin processing at."),
    omit_pages: Annotated[
        str | None,
        typer.Option(
            "--omit-pages", help="Pages to skip entirely. Use commas or ranges (e.g., '1,2,5-8')."
        ),
    ] = None,
    end_page: Annotated[
        int | None,
        typer.Option(
            "--end-page",
            help="Absolute last page number to process (inclusive).",
        ),
    ] = None,
) -> None:
    """Run OCR and layout analysis on a raw PDF, saving structural JSON."""
    from modules.ocr_engine.orchestrator import process_pdf_to_structural_json

    parsed_omit_pages = parse_omit_pages(omit_pages)
    source_path = ensure_pdf_exists(pdf_path, context_label="Ingest Failed")
    output_dir = Path("output")

    log.info(f"Ingesting PDF from {source_path} into structural JSON... 🚀")
    try:
        final_file = process_pdf_to_structural_json(
            pdf_path=source_path,
            output_dir=output_dir,
            chunk_size=chunk_size,
            dpi=dpi,
            slice_only=slice_only,
            start_page=start_page,
            omit_pages=parsed_omit_pages,
            end_page=end_page,
        )
        if slice_only:
            log.info(f"Grid-Slicing Complete. Label Studio Tasks saved to: {final_file}")
        else:
            log.info(f"✅ Ingestion Complete. Structural Map saved to: {final_file}")
    except Exception as exc:
        log.error(f"OCR Pipeline failed: {exc}")
        raise typer.Exit(code=1) from exc


def run_poc_slicer(
    pdf_path: Annotated[str, typer.Option("--pdf-path", help="Path to the source liturgical PDF.")],
    chunk_size: int = typer.Option(
        50, "--chunk-size", help="How many pages to load into RAM at once."
    ),
    dpi: int = typer.Option(300, "--dpi", help="Image processing resolution."),
    start_page: int = typer.Option(1, "--start-page", help="Page number to begin processing at."),
    omit_pages: Annotated[
        str | None,
        typer.Option(
            "--omit-pages", help="Pages to skip entirely. Use commas or ranges (e.g., '1,2,5-8')."
        ),
    ] = None,
    end_page: Annotated[
        int | None,
        typer.Option(
            "--end-page",
            help="Absolute last page number to process (inclusive).",
        ),
    ] = None,
) -> None:
    """Run PoC slicer to visually debug column crops."""
    from modules.ocr_engine.orchestrator import run_poc_debug_pipeline

    parsed_omit_pages = parse_omit_pages(omit_pages)
    source_path = ensure_pdf_exists(pdf_path, context_label="PoC Slicer Failed")
    output_dir = Path("output")

    log.info("Running PoC Text Detection & Gap-Seeking Slicer on pristine images... 🚀")
    try:
        final_dir = run_poc_debug_pipeline(
            pdf_path=source_path,
            output_dir=output_dir,
            chunk_size=chunk_size,
            dpi=dpi,
            start_page=start_page,
            omit_pages=parsed_omit_pages,
            end_page=end_page,
        )
        log.info(f"✅ PoC Complete. Visual outputs ready for review in: {final_dir}")
    except Exception as exc:
        log.error(f"PoC Slicer failed: {exc}")
        raise typer.Exit(code=1) from exc


def run_crop_columns(
    pdf_path: Annotated[str, typer.Option("--pdf-path", help="Path to the source PDF.")],
    output_dir: str = typer.Option("output/ocr_artifacts", "--output-dir"),
    rectify_mode: str = typer.Option(
        "rotate+homography", "--rectify-mode", help="Rectification style: rotate|rotate+homography"
    ),
    chunk_size: int = typer.Option(50, "--chunk-size"),
    dpi: int = typer.Option(300, "--dpi"),
    end_page: Annotated[
        int | None,
        typer.Option(
            "--end-page",
            help="Absolute last page number to process (inclusive).",
        ),
    ] = None,
    start_page: int = typer.Option(1, "--start-page", help="Page number to begin processing at."),
    omit_pages: Annotated[
        str | None,
        typer.Option(
            "--omit-pages", help="Pages to skip entirely. Use commas or ranges (e.g., '1,2,5-8')."
        ),
    ] = None,
) -> None:
    """Run column cropping using verified/fallback divider artifacts (no OCR)."""
    from modules.ocr_engine.orchestrator import run_precision_extraction_pipeline

    logging.getLogger("PDFtoImage").setLevel(logging.WARNING)
    source_path = ensure_pdf_exists(pdf_path, context_label="Extraction Failed")

    log.info(f"Starting column-cropping pipeline for {source_path}...")
    log.info(
        "Resolving divider source (preferred: data/layout_dataset/hitl_line_editor.sqlite3; "
        "fallback: output/hitl/ocr_column_map.json)"
    )

    try:
        parsed_omit_pages = parse_omit_pages(omit_pages)
        final_file = run_precision_extraction_pipeline(
            pdf_path=source_path,
            output_dir=Path(output_dir),
            rectify_mode=rectify_mode,
            chunk_size=chunk_size,
            dpi=dpi,
            start_page=start_page,
            end_page=end_page,
            omit_pages=parsed_omit_pages,
        )
        log.info(f"✅ Column cropping complete. Manifest: {final_file}")
    except Exception as exc:
        log.error(f"Column cropping pipeline failed: {exc}")
        raise typer.Exit(code=1) from exc


def run_ocr(
    pdf_path: Annotated[str, typer.Option("--pdf-path", help="Path to the source PDF.")],
    output_dir: str = typer.Option("output/ocr_inference", "--output-dir"),
    run_name: str = typer.Option("ocr_infer_v1", "--run-name"),
    rectify_mode: str = typer.Option(
        "rotate+homography", "--rectify-mode", help="Rectification style: rotate|rotate+homography"
    ),
    chunk_size: int = typer.Option(50, "--chunk-size"),
    dpi: int = typer.Option(300, "--dpi"),
    end_page: Annotated[
        int | None,
        typer.Option(
            "--end-page",
            help="Absolute last page number to process (inclusive).",
        ),
    ] = None,
    start_page: int = typer.Option(1, "--start-page", help="Page number to begin processing at."),
    omit_pages: Annotated[
        str | None,
        typer.Option(
            "--omit-pages", help="Pages to skip entirely. Use commas or ranges (e.g., '1,2,5-8')."
        ),
    ] = None,
) -> None:
    """OCR command scaffold for the recognition stage."""
    from modules.ocr_engine.orchestrator import run_ocr_inference_pipeline

    source_path = ensure_pdf_exists(pdf_path, context_label="OCR Failed")
    parsed_omit_pages = parse_omit_pages(omit_pages)
    log.info(
        "OCR command is scaffolded for now. "
        "It records an inference manifest until recognition implementation lands."
    )
    try:
        manifest = run_ocr_inference_pipeline(
            pdf_path=source_path,
            output_dir=Path(output_dir),
            run_name=run_name,
            rectify_mode=rectify_mode,
            chunk_size=chunk_size,
            dpi=dpi,
            start_page=start_page,
            end_page=end_page,
            omit_pages=parsed_omit_pages,
        )
        log.info(f"✅ OCR scaffold ready. Manifest: {manifest}")
    except Exception as exc:
        log.error(f"OCR scaffold failed: {exc}")
        raise typer.Exit(code=1) from exc


def run_ocr_infer(
    pdf_path: Annotated[str, typer.Option("--pdf-path", help="Path to the source PDF.")],
    output_dir: str = typer.Option("output/ocr_inference", "--output-dir"),
    run_name: str = typer.Option("ocr_infer_v1", "--run-name"),
    rectify_mode: str = typer.Option(
        "rotate+homography", "--rectify-mode", help="Rectification style: rotate|rotate+homography"
    ),
    chunk_size: int = typer.Option(50, "--chunk-size"),
    dpi: int = typer.Option(300, "--dpi"),
    start_page: int = typer.Option(1, "--start-page", help="Page number to begin processing at."),
    end_page: Annotated[
        int | None,
        typer.Option("--end-page", help="Absolute last page number to process (inclusive)."),
    ] = None,
    omit_pages: Annotated[
        str | None,
        typer.Option(
            "--omit-pages", help="Pages to skip entirely. Use commas or ranges (e.g., '1,2,5-8')."
        ),
    ] = None,
) -> None:
    """Compatibility alias for `ocr` scaffold command."""
    log.warning("`ocr-infer` is deprecated; use `ocr`.")
    run_ocr(
        pdf_path=pdf_path,
        output_dir=output_dir,
        run_name=run_name,
        rectify_mode=rectify_mode,
        chunk_size=chunk_size,
        dpi=dpi,
        start_page=start_page,
        end_page=end_page,
        omit_pages=omit_pages,
    )


def run_ocr_train(
    pdf_path: Annotated[str, typer.Option("--pdf-path", help="Path to the source PDF.")],
    output_dir: str = typer.Option("output/ocr_training", "--output-dir"),
    run_name: str = typer.Option("ocr_train_v1", "--run-name"),
    rectify_mode: str = typer.Option(
        "rotate+homography", "--rectify-mode", help="Rectification style: rotate|rotate+homography"
    ),
    chunk_size: int = typer.Option(50, "--chunk-size"),
    dpi: int = typer.Option(300, "--dpi"),
    start_page: int = typer.Option(1, "--start-page", help="Page number to begin processing at."),
    end_page: Annotated[
        int | None,
        typer.Option("--end-page", help="Absolute last page number to process (inclusive)."),
    ] = None,
    omit_pages: Annotated[
        str | None,
        typer.Option(
            "--omit-pages", help="Pages to skip entirely. Use commas or ranges (e.g., '1,2,5-8')."
        ),
    ] = None,
    epochs: int = typer.Option(3, "--epochs"),
    batch_size: int = typer.Option(4, "--batch-size"),
    learning_rate: float = typer.Option(1e-5, "--learning-rate"),
) -> None:
    """Scaffold OCR training orchestration and persist a run manifest."""
    from modules.ocr_engine.orchestrator import run_ocr_training_pipeline

    source_path = ensure_pdf_exists(pdf_path, context_label="OCR Train Failed")

    parsed_omit_pages = parse_omit_pages(omit_pages)
    log.info(f"Starting OCR training scaffold for {source_path}...")
    try:
        manifest = run_ocr_training_pipeline(
            pdf_path=source_path,
            output_dir=Path(output_dir),
            run_name=run_name,
            rectify_mode=rectify_mode,
            chunk_size=chunk_size,
            dpi=dpi,
            start_page=start_page,
            end_page=end_page,
            omit_pages=parsed_omit_pages,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        log.info(f"✅ OCR training scaffold ready. Manifest: {manifest}")
    except Exception as exc:
        log.error(f"OCR training scaffold failed: {exc}")
        raise typer.Exit(code=1) from exc
