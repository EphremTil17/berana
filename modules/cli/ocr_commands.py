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
    omit_pages: Annotated[
        str | None,
        typer.Option(
            "--omit-pages", help="Pages to skip entirely. Use commas or ranges (e.g., '1,2,5-8')."
        ),
    ] = None,
    max_pages: Annotated[
        int | None,
        typer.Option(
            "--max-pages",
            help="Process only up to this many pages (useful for limiting pipeline tests).",
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
            omit_pages=parsed_omit_pages,
            max_pages=max_pages,
        )
        if slice_only:
            log.info(f"Grid-Slicing Complete. Label Studio Tasks saved to: {final_file}")
        else:
            log.info(f"✅ Ingestion Complete. Structural Map saved to: {final_file}")
    except Exception as exc:
        log.error(f"❌ OCR Pipeline failed: {exc}")
        raise typer.Exit(code=1) from exc


def run_poc_slicer(
    pdf_path: Annotated[str, typer.Option("--pdf-path", help="Path to the source liturgical PDF.")],
    chunk_size: int = typer.Option(
        50, "--chunk-size", help="How many pages to load into RAM at once."
    ),
    dpi: int = typer.Option(300, "--dpi", help="Image processing resolution."),
    omit_pages: Annotated[
        str | None,
        typer.Option(
            "--omit-pages", help="Pages to skip entirely. Use commas or ranges (e.g., '1,2,5-8')."
        ),
    ] = None,
    max_pages: Annotated[
        int | None,
        typer.Option(
            "--max-pages",
            help="Process only up to this many pages (useful for limiting pipeline tests).",
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
            omit_pages=parsed_omit_pages,
            max_pages=max_pages,
        )
        log.info(f"✅ PoC Complete. Visual outputs ready for review in: {final_dir}")
    except Exception as exc:
        log.error(f"❌ PoC Slicer failed: {exc}")
        raise typer.Exit(code=1) from exc


def run_extract_text(
    pdf_path: Annotated[str, typer.Option("--pdf-path", help="Path to the source PDF.")],
    verified_labels: Annotated[
        str,
        typer.Option(
            "--verified-labels",
            help="Path to the Label Studio YOLO export zip containing human-verified boundaries.",
        ),
    ],
    output_dir: str = typer.Option("output/projects", "--output-dir"),
    chunk_size: int = typer.Option(50, "--chunk-size"),
    dpi: int = typer.Option(300, "--dpi"),
) -> None:
    """Step 2 (HITL): Run precision OCR using human-verified layout boundaries."""
    log.info(f"🚀 Starting Precision Extract-Text Pipeline for {pdf_path}...")
    log.info(f"🔒 Engaging HITL verification using labels from: {verified_labels}")
    _ = output_dir, chunk_size, dpi
    raise NotImplementedError("Phase 3: Step 2 pending module rewrite.")
