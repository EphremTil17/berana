from pathlib import Path
from typing import Annotated

import typer

from modules.cli.common import ensure_source_exists, parse_page_selection
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
    source: Annotated[str, typer.Option("--source", help="Path to the source PDF or image.")],
    checkpoint_dir: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint-dir",
            help="Surya fine-tune run directory or checkpoint-* directory. Required unless --zero-shot is set.",
        ),
    ] = None,
    zero_shot: Annotated[
        bool,
        typer.Option(
            "--zero-shot",
            help="Run zero-shot Surya OCR instead of a fine-tuned checkpoint.",
        ),
    ] = False,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Base output directory for OCR inference runs."),
    ] = Path("output/ocr_runs/inference"),
    pages: Annotated[
        str | None,
        typer.Option(
            "--pages",
            help="Optional page selection for PDFs. Supports commas and ranges (e.g., '1-5,7,10-12').",
        ),
    ] = None,
    eval_batch_size: Annotated[
        int,
        typer.Option("--eval-batch-size", help="OCR inference batch size."),
    ] = 1,
    dataloader_num_workers: Annotated[
        int,
        typer.Option(
            "--dataloader-num-workers",
            help="Number of image-loading workers used during crop-layout OCR inference.",
        ),
    ] = 0,
    diagnose: Annotated[
        bool,
        typer.Option(
            "--diagnose",
            help="Write annotated images showing the exact OCR boxes used for recognition.",
        ),
    ] = False,
) -> None:
    """Run standalone Surya OCR inference on a PDF or image source."""
    from modules.ocr_inference.pipeline import run_source_ocr_inference_pipeline

    if zero_shot and checkpoint_dir is not None:
        raise typer.BadParameter("Use either --zero-shot or --checkpoint-dir, not both.")
    if not zero_shot and checkpoint_dir is None:
        raise typer.BadParameter("Provide --checkpoint-dir or explicitly pass --zero-shot.")

    source_path = ensure_source_exists(source, context_label="OCR inference failed")
    selected_pages = parse_page_selection(pages)

    log.info(
        "Starting OCR inference for %s in %s mode.",
        source_path,
        "zero-shot" if zero_shot else "checkpoint",
    )
    try:
        result = run_source_ocr_inference_pipeline(
            source_path=source_path,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            zero_shot=zero_shot,
            eval_batch_size=eval_batch_size,
            dataloader_num_workers=dataloader_num_workers,
            diagnose=diagnose,
            selected_pages=selected_pages or None,
        )
        log.info("✅ OCR inference complete. Output: %s", result)
    except Exception as exc:
        log.error("OCR inference failed: %s", exc)
        raise typer.Exit(code=1) from exc


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
