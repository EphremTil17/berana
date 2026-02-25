from pathlib import Path
from typing import Annotated

import typer

from modules.cli.runtime import execute_pipeline
from utils.logger import get_logger

log = get_logger("OCRCLI")


def run_layout_diagnostics(
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
    """Run layout diagnostics to visually inspect line detection and column slicing behavior."""
    from modules.ocr_engine.orchestrator import run_layout_diagnostics_pipeline

    log.info("Running layout diagnostics on pristine images... ")
    execute_pipeline(
        pdf_path=pdf_path,
        pipeline_fn=run_layout_diagnostics_pipeline,
        context_label="Layout diagnostics failed",
        success_msg="Layout diagnostics complete. Visual outputs ready for review in:",
        omit_pages_raw=omit_pages,
        output_dir=Path("output/layout_diagnostics"),
        chunk_size=chunk_size,
        dpi=dpi,
        start_page=start_page,
        end_page=end_page,
    )


def run_crop_columns(
    pdf_path: Annotated[str, typer.Option("--pdf-path", help="Path to the source PDF.")],
    output_dir: str = typer.Option("output/column_crops", "--output-dir"),
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

    log.info(f"Starting column-cropping pipeline for {pdf_path}...")
    log.info(
        "Resolving divider source (preferred: input/layout_dataset/hitl_line_editor.sqlite3; "
        "fallback: output/hitl/ocr_column_map.json)"
    )

    execute_pipeline(
        pdf_path=pdf_path,
        pipeline_fn=run_precision_extraction_pipeline,
        context_label="Column cropping pipeline failed",
        success_msg="Column cropping complete. Manifest:",
        omit_pages_raw=omit_pages,
        output_dir=Path(output_dir),
        rectify_mode=rectify_mode,
        chunk_size=chunk_size,
        dpi=dpi,
        start_page=start_page,
        end_page=end_page,
    )


def run_ocr(
    pdf_path: Annotated[str, typer.Option("--pdf-path", help="Path to the source PDF.")],
    output_dir: str = typer.Option("output/ocr_runs/inference", "--output-dir"),
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

    log.info(
        "OCR command is scaffolded for now. "
        "It records an inference manifest until recognition implementation lands."
    )

    execute_pipeline(
        pdf_path=pdf_path,
        pipeline_fn=run_ocr_inference_pipeline,
        context_label="OCR scaffold failed",
        success_msg="OCR scaffold ready. Manifest:",
        omit_pages_raw=omit_pages,
        output_dir=Path(output_dir),
        run_name=run_name,
        rectify_mode=rectify_mode,
        chunk_size=chunk_size,
        dpi=dpi,
        start_page=start_page,
        end_page=end_page,
    )


def run_ocr_train(
    pdf_path: Annotated[str, typer.Option("--pdf-path", help="Path to the source PDF.")],
    output_dir: str = typer.Option("output/ocr_runs/training", "--output-dir"),
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

    log.info(f"Starting OCR training scaffold for {pdf_path}...")

    execute_pipeline(
        pdf_path=pdf_path,
        pipeline_fn=run_ocr_training_pipeline,
        context_label="OCR training scaffold failed",
        success_msg="OCR training scaffold ready. Manifest:",
        omit_pages_raw=omit_pages,
        output_dir=Path(output_dir),
        run_name=run_name,
        rectify_mode=rectify_mode,
        chunk_size=chunk_size,
        dpi=dpi,
        start_page=start_page,
        end_page=end_page,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
