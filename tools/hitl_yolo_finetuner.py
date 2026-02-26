import sys
from pathlib import Path
from typing import Annotated

import typer

try:
    from tools.hitl_yolo_finetuner_app.export import export_dataset
    from tools.hitl_yolo_finetuner_app.preview import preview_polygons
    from tools.hitl_yolo_finetuner_app.train import run_yolo_train
    from utils.run_registry import load_latest_run
except ModuleNotFoundError:
    # Allow running from inside ./tools without requiring PYTHONPATH=.
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from tools.hitl_yolo_finetuner_app.export import export_dataset
    from tools.hitl_yolo_finetuner_app.preview import preview_polygons
    from tools.hitl_yolo_finetuner_app.train import run_yolo_train
    from utils.run_registry import load_latest_run

app = typer.Typer(help="HITL YOLOv8-seg Finetuner toolkit.")


@app.command()
def export(
    pdf_path: Annotated[Path, typer.Option("--pdf-path", help="Path to the PDF document")],
    db_file: Annotated[
        Path, typer.Option("--db-file", help="Path to hitl_line_editor.sqlite3")
    ] = Path("input/layout_dataset/hitl_line_editor.sqlite3"),
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Base directory for versioned finetuner runs")
    ] = Path("output/hitl_finetuner"),
    split_seed: Annotated[
        int, typer.Option("--split-seed", help="Deterministic train/val split seed")
    ] = 42,
    omit_pages: Annotated[
        str | None,
        typer.Option("--omit-pages", help="Pages to skip. Supports ranges: '1-9,45,67-69'."),
    ] = None,
    line_width_px: Annotated[
        float, typer.Option("--line-width-px", help="Polygon extrusion width")
    ] = 4.0,
    max_tilt_deg: Annotated[
        float, typer.Option("--max-tilt-deg", help="Skip lines exceeding tilt")
    ] = 15.0,
    min_line_length_px: Annotated[
        float, typer.Option("--min-line-length-px", help="Skip tiny lines")
    ] = 100.0,
    clip_top_px: Annotated[
        float,
        typer.Option("--clip-top-px", help="Clip label lines above this many pixels from top"),
    ] = 0.0,
    clip_bottom_px: Annotated[
        float,
        typer.Option(
            "--clip-bottom-px", help="Clip label lines above this many pixels from bottom"
        ),
    ] = 0.0,
    min_train_pages: Annotated[
        int, typer.Option("--min-train-pages", help="Floor for train split")
    ] = 20,
    min_val_pages: Annotated[int, typer.Option("--min-val-pages", help="Floor for val split")] = 5,
    copy_mode: Annotated[
        str, typer.Option("--copy-mode", help="hardlink|copy|symlink")
    ] = "hardlink",
    force_provenance: Annotated[
        bool,
        typer.Option(
            "--force-provenance",
            help=(
                "Force DB provenance rebinding to current layout-infer run for --pdf-path. "
                "Use when reusing an older long-lived HITL DB."
            ),
        ),
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report stats without disk mutation")
    ] = False,
) -> None:
    """Harvest verified rows and convert them to the YOLOv8 segmentation folder structure."""
    try:
        report = export_dataset(
            doc_stem=pdf_path.stem,
            db_file=db_file,
            output_dir=output_dir,
            split_seed=split_seed,
            omit_pages=omit_pages,
            line_width_px=line_width_px,
            max_tilt_deg=max_tilt_deg,
            min_line_length_px=min_line_length_px,
            clip_top_px=clip_top_px,
            clip_bottom_px=clip_bottom_px,
            min_train_pages=min_train_pages,
            min_val_pages=min_val_pages,
            copy_mode=copy_mode,
            force_provenance=force_provenance,
            dry_run=dry_run,
        )
    except Exception as e:
        typer.secho(f"Export failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

    if dry_run:
        typer.echo("DRY RUN Report:")
    else:
        typer.echo("Exported dataset successfully.")
    typer.echo(report)


@app.command()
def preview(
    pdf_path: Annotated[Path, typer.Option("--pdf-path", help="Path to the PDF document")],
    db_file: Annotated[
        Path, typer.Option("--db-file", help="Path to hitl_line_editor.sqlite3")
    ] = Path("input/layout_dataset/hitl_line_editor.sqlite3"),
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Base directory for versioned finetuner runs")
    ] = Path("output/hitl_finetuner"),
    line_width_px: Annotated[float, typer.Option("--line-width-px")] = 4.0,
    max_pages: Annotated[int, typer.Option("--max-pages")] = 10,
    omit_pages: Annotated[
        str | None,
        typer.Option("--omit-pages", help="Pages to skip. Supports ranges: '1-9,45,67-69'."),
    ] = None,
    clip_top_px: Annotated[
        float,
        typer.Option("--clip-top-px", help="Clip preview lines above this many pixels from top"),
    ] = 0.0,
    clip_bottom_px: Annotated[
        float,
        typer.Option(
            "--clip-bottom-px", help="Clip preview lines above this many pixels from bottom"
        ),
    ] = 0.0,
) -> None:
    """Render mathematical polygons over latest layout-infer visuals for QA alignment."""
    try:
        doc_stem = pdf_path.stem
        latest_infer = load_latest_run("layout-infer", doc_stem)
        if not latest_infer:
            raise FileNotFoundError(f"No layout-infer pointer found for '{doc_stem}'.")
        visuals_dir = latest_infer.get("artifacts", {}).get("visuals_dir")
        if not visuals_dir:
            raise KeyError(
                f"Layout-infer pointer for '{doc_stem}' missing required artifact 'visuals_dir'."
            )
        resolved_output = output_dir / Path(latest_infer["run_dir"]).name / "preview"
        preview_polygons(
            db_file=db_file,
            images_source_dir=Path(visuals_dir),
            output_dir=resolved_output,
            line_width_px=line_width_px,
            max_pages=max_pages,
            clip_top_px=clip_top_px,
            clip_bottom_px=clip_bottom_px,
            omit_pages=omit_pages,
        )
    except Exception as e:
        typer.secho(f"Preview failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e


@app.command()
def train(
    pdf_path: Annotated[Path, typer.Option("--pdf-path", help="Path to the PDF document")],
    db_file: Annotated[
        Path, typer.Option("--db-file", help="Path to hitl_line_editor.sqlite3")
    ] = Path("input/layout_dataset/hitl_line_editor.sqlite3"),
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Base directory for versioned outputs")
    ] = Path("output/hitl_finetuner"),
    split_seed: Annotated[
        int, typer.Option("--split-seed", help="Deterministic train/val split seed")
    ] = 42,
    omit_pages: Annotated[
        str | None,
        typer.Option("--omit-pages", help="Pages to skip. Supports ranges: '1-9,45,67-69'."),
    ] = None,
    line_width_px: Annotated[
        float, typer.Option("--line-width-px", help="Polygon extrusion width")
    ] = 4.0,
    max_tilt_deg: Annotated[
        float, typer.Option("--max-tilt-deg", help="Skip lines exceeding tilt")
    ] = 15.0,
    min_line_length_px: Annotated[
        float, typer.Option("--min-line-length-px", help="Skip tiny lines")
    ] = 100.0,
    clip_top_px: Annotated[
        float,
        typer.Option("--clip-top-px", help="Clip label lines above this many pixels from top"),
    ] = 0.0,
    clip_bottom_px: Annotated[
        float,
        typer.Option(
            "--clip-bottom-px", help="Clip label lines above this many pixels from bottom"
        ),
    ] = 0.0,
    min_train_pages: Annotated[
        int, typer.Option("--min-train-pages", help="Floor for train split")
    ] = 20,
    min_val_pages: Annotated[int, typer.Option("--min-val-pages", help="Floor for val split")] = 5,
    copy_mode: Annotated[
        str, typer.Option("--copy-mode", help="hardlink|copy|symlink")
    ] = "hardlink",
    force_provenance: Annotated[
        bool,
        typer.Option(
            "--force-provenance",
            help=(
                "Force DB provenance rebinding to current layout-infer run for --pdf-path "
                "before export."
            ),
        ),
    ] = False,
    epochs: Annotated[int, typer.Option("--epochs")] = 100,
    batch: Annotated[int, typer.Option("--batch", help="Batch size")] = 4,
    model: Annotated[
        str, typer.Option("--model", help="Ultralytics model checkpoint, e.g. yolov8s-seg.pt")
    ] = "yolov8s-seg.pt",
    imgsz: Annotated[
        int | None,
        typer.Option("--imgsz", help="Training resolution. If omitted, uses dataset native max."),
    ] = None,
) -> None:
    """Act as a direct wrapper around ultralytics YOLO to perform finetuning logically tied to a doc_stem."""
    try:
        if omit_pages:
            # Keep train standalone: when omit-pages is provided, refresh export before training.
            export_dataset(
                doc_stem=pdf_path.stem,
                db_file=db_file,
                output_dir=output_dir,
                split_seed=split_seed,
                omit_pages=omit_pages,
                line_width_px=line_width_px,
                max_tilt_deg=max_tilt_deg,
                min_line_length_px=min_line_length_px,
                clip_top_px=clip_top_px,
                clip_bottom_px=clip_bottom_px,
                min_train_pages=min_train_pages,
                min_val_pages=min_val_pages,
                copy_mode=copy_mode,
                force_provenance=force_provenance,
            )
        report = run_yolo_train(
            doc_stem=pdf_path.stem,
            epochs=epochs,
            batch=batch,
            model=model,
            imgsz=imgsz,
        )
        typer.echo(f"Training registered: {report['signature']} -> {report['source_export_yaml']}")
    except Exception as e:
        typer.secho(f"Training failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e


@app.command(name="run")
def execute_loop(
    pdf_path: Annotated[Path, typer.Option("--pdf-path", help="Path to the PDF document")],
    db_file: Annotated[
        Path, typer.Option("--db-file", help="Path to hitl_line_editor.sqlite3")
    ] = Path("input/layout_dataset/hitl_line_editor.sqlite3"),
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Base directory for versioned finetune outputs")
    ] = Path("output/hitl_finetuner"),
    split_seed: Annotated[
        int, typer.Option("--split-seed", help="Deterministic train/val split seed")
    ] = 42,
    omit_pages: Annotated[
        str | None,
        typer.Option("--omit-pages", help="Pages to skip. Supports ranges: '1-9,45,67-69'."),
    ] = None,
    line_width_px: Annotated[
        float, typer.Option("--line-width-px", help="Polygon extrusion width")
    ] = 4.0,
    max_tilt_deg: Annotated[
        float, typer.Option("--max-tilt-deg", help="Skip lines exceeding tilt")
    ] = 15.0,
    min_line_length_px: Annotated[
        float, typer.Option("--min-line-length-px", help="Skip tiny lines")
    ] = 100.0,
    clip_top_px: Annotated[
        float,
        typer.Option("--clip-top-px", help="Clip label lines above this many pixels from top"),
    ] = 0.0,
    clip_bottom_px: Annotated[
        float,
        typer.Option(
            "--clip-bottom-px", help="Clip label lines above this many pixels from bottom"
        ),
    ] = 0.0,
    min_train_pages: Annotated[
        int, typer.Option("--min-train-pages", help="Floor for train split")
    ] = 20,
    min_val_pages: Annotated[int, typer.Option("--min-val-pages", help="Floor for val split")] = 5,
    copy_mode: Annotated[
        str, typer.Option("--copy-mode", help="hardlink|copy|symlink")
    ] = "hardlink",
    force_provenance: Annotated[
        bool,
        typer.Option(
            "--force-provenance",
            help=(
                "Force DB provenance rebinding to current layout-infer run for --pdf-path "
                "before export."
            ),
        ),
    ] = False,
    epochs: Annotated[int, typer.Option("--epochs")] = 100,
    batch: Annotated[int, typer.Option("--batch", help="Batch size")] = 4,
    model: Annotated[
        str, typer.Option("--model", help="Ultralytics model checkpoint, e.g. yolov8s-seg.pt")
    ] = "yolov8s-seg.pt",
    imgsz: Annotated[
        int | None,
        typer.Option("--imgsz", help="Training resolution. If omitted, uses dataset native max."),
    ] = None,
) -> None:
    """Sequentially wrapper to export the dataset and immediately trigger YOLO training for a closed-loop."""
    doc_stem = pdf_path.stem
    typer.echo(f"==> Exporting verifiable dataset for '{doc_stem}'...")
    try:
        export_dataset(
            doc_stem=doc_stem,
            db_file=db_file,
            output_dir=output_dir,
            split_seed=split_seed,
            omit_pages=omit_pages,
            line_width_px=line_width_px,
            max_tilt_deg=max_tilt_deg,
            min_line_length_px=min_line_length_px,
            clip_top_px=clip_top_px,
            clip_bottom_px=clip_bottom_px,
            min_train_pages=min_train_pages,
            min_val_pages=min_val_pages,
            copy_mode=copy_mode,
            force_provenance=force_provenance,
        )
    except Exception as e:
        typer.secho(f"Export step failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

    typer.echo(f"==> Initiating YOLO training for '{doc_stem}'...")
    try:
        run_yolo_train(
            doc_stem=doc_stem,
            epochs=epochs,
            batch=batch,
            model=model,
            imgsz=imgsz,
        )
    except Exception as e:
        typer.secho(f"Training step failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

    typer.echo("✅ Refactor completed: closed-loop finetuning finished.")


if __name__ == "__main__":
    app()
