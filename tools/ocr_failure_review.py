from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from modules.ocr_training.failure_review import FailureReviewSummary, create_failure_review_tasks
from utils.logger import get_logger

app = typer.Typer(
    add_completion=False, help="Generate Label Studio review tasks for OCR failure analysis."
)
log = get_logger("OCRFailureReview")


@app.command("make-ls-tasks")
def cli_make_ls_tasks(
    exact_false_dir: Annotated[
        Path,
        typer.Option(
            "--exact-false-dir",
            help="Path to one exact_false analysis directory containing outlier and candidate manifests.",
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Optional output directory. Defaults to '<analysis_root>/label_studio'.",
        ),
    ] = None,
):
    """Generate one Label Studio task JSON from OCR failure-analysis candidates."""
    resolved_output_dir = output_dir or exact_false_dir.parent / "label_studio"
    try:
        summary: FailureReviewSummary = create_failure_review_tasks(
            exact_false_dir=exact_false_dir,
            output_dir=resolved_output_dir,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("make-ls-tasks failed: %s", exc)
        raise typer.Exit(code=1) from exc

    log.info(
        "make-ls-tasks complete tasks=%d skipped_missing=%d output_json=%s",
        summary["num_tasks"],
        summary["skipped_missing_images"],
        Path(summary["output_json"]),
    )


if __name__ == "__main__":
    app()
