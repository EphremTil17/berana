"""Thin OCR orchestration facade.

This module intentionally stays lightweight and routes to specialized pipeline modules.
"""

from __future__ import annotations

from pathlib import Path


def run_layout_diagnostics_pipeline(
    pdf_path: Path,
    output_dir: Path,
    chunk_size: int = 50,
    dpi: int = 300,
    start_page: int = 1,
    omit_pages: list[int] | None = None,
    end_page: int | None = None,
) -> Path:
    """Route to layout diagnostics pipeline."""
    from modules.ocr_engine.pipelines.diagnostics import run_layout_diagnostics_pipeline as _run

    return _run(
        pdf_path=pdf_path,
        output_dir=output_dir,
        chunk_size=chunk_size,
        dpi=dpi,
        start_page=start_page,
        omit_pages=omit_pages,
        end_page=end_page,
    )


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
    """Route to precision extraction pipeline."""
    from modules.ocr_engine.pipelines.precision import run_precision_extraction_pipeline as _run

    return _run(
        pdf_path=pdf_path,
        output_dir=output_dir,
        rectify_mode=rectify_mode,
        chunk_size=chunk_size,
        dpi=dpi,
        start_page=start_page,
        end_page=end_page,
        omit_pages=omit_pages,
    )


def run_ocr_training_pipeline(
    pdf_path: Path,
    output_dir: Path,
    run_name: str = "ocr_train_v1",
    rectify_mode: str = "rotate+homography",
    chunk_size: int = 50,
    dpi: int = 300,
    start_page: int = 1,
    end_page: int | None = None,
    omit_pages: list[int] | None = None,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
) -> Path:
    """Route to OCR training pipeline scaffold."""
    from modules.ocr_engine.pipelines.training import run_ocr_training_pipeline as _run

    return _run(
        pdf_path=pdf_path,
        output_dir=output_dir,
        run_name=run_name,
        rectify_mode=rectify_mode,
        chunk_size=chunk_size,
        dpi=dpi,
        start_page=start_page,
        end_page=end_page,
        omit_pages=omit_pages,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )


def run_ocr_inference_pipeline(
    pdf_path: Path,
    output_dir: Path,
    run_name: str = "ocr_infer_v1",
    rectify_mode: str = "rotate+homography",
    chunk_size: int = 50,
    dpi: int = 300,
    start_page: int = 1,
    end_page: int | None = None,
    omit_pages: list[int] | None = None,
) -> Path:
    """Route to OCR inference pipeline scaffold."""
    from modules.ocr_engine.pipelines.inference import run_ocr_inference_pipeline as _run

    return _run(
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
