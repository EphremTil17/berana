from pathlib import Path

import typer

from tools.hitl_line_editor_app.export import export_ocr_ready_json

app = typer.Typer(help="Export HITL divider coordinates into OCR-ready JSON.")


@app.command()
def run(
    output_path: str = typer.Option(
        "output/hitl/ocr_column_map.json",
        "--output-path",
        help="Destination JSON path for OCR-ready divider/column map.",
    ),
    include_unverified: bool = typer.Option(
        False,
        "--include-unverified",
        help="Include pages that are not marked verified.",
    ),
) -> None:
    """Export divider lines and orthogonal column ranges for OCR cropping pipelines."""
    output = export_ocr_ready_json(
        output_path=Path(output_path),
        only_verified=not include_unverified,
    )
    typer.echo(f"Exported OCR-ready HITL coordinates to: {output}")


if __name__ == "__main__":
    app()
