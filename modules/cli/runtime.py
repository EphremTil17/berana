import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from modules.cli.common import ensure_pdf_exists, parse_omit_pages
from utils.logger import get_logger

log = get_logger("CLI_Runtime")


def execute_pipeline(
    *,
    pdf_path: str | Path,
    pipeline_fn: Callable[..., Any],
    context_label: str,
    success_msg: str,
    omit_pages_raw: str | None = None,
    **kwargs: Any,
) -> None:
    """
    Standardized CLI execution wrapper for Berana pipelines.

    Validates input paths, parses omit pages, wraps the pipeline call in a
    standardized try/except block, and strictly exits with code 1 on failure.
    """
    logging.getLogger("PDFtoImage").setLevel(logging.WARNING)
    logging.getLogger("YOLOEngine").setLevel(logging.WARNING)

    source_path = ensure_pdf_exists(pdf_path, context_label=context_label)
    parsed_omit_pages = parse_omit_pages(omit_pages_raw)

    try:
        result = pipeline_fn(
            pdf_path=source_path,
            omit_pages=parsed_omit_pages,
            **kwargs,
        )
        log.info(f"✅ {success_msg} {result}")
    except Exception as exc:
        log.error(f"{context_label}: {exc}")
        raise typer.Exit(code=1) from exc
