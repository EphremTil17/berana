from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from modules.ocr_training.failure_analysis import analyze_predictions_failures
from utils.logger import get_logger

app = typer.Typer(add_completion=False, help="Dedicated OCR failure analysis tools.")
log = get_logger("OCRFailureAnalysis")


@app.command("analyze-surya-predictions")
def cli_analyze_surya_predictions(
    predictions_path: Annotated[
        Path,
        typer.Option("--predictions-path", help="Path to one predictions_<split>.jsonl artifact."),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Optional output directory. Defaults to a sibling '<predictions_stem>_analysis' directory.",
        ),
    ] = None,
):
    """Extract exact-false rows and emit category/outlier analysis artifacts."""
    resolved_output_dir = (
        output_dir or predictions_path.parent / f"{predictions_path.stem}_analysis"
    )
    try:
        summary = analyze_predictions_failures(
            predictions_path=predictions_path,
            output_dir=resolved_output_dir,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("analyze-surya-predictions failed: %s", exc)
        raise typer.Exit(code=1) from exc

    log.info(
        "analyze-surya-predictions complete rows=%d exact_false=%d confirmed_blank=%d suspect_blank=%d likely_mismatch=%d output_dir=%s",
        summary["num_rows"],
        summary["exact_false_count"],
        summary["exact_false"]["confirmed_blank"],
        summary["exact_false"]["suspect_blank"],
        summary["exact_false"]["likely_label_mismatch"],
        resolved_output_dir,
    )


if __name__ == "__main__":
    app()
