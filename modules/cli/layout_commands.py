import json
import logging
import statistics
from collections import deque
from pathlib import Path
from typing import Annotated

import typer

from modules.cli.common import ensure_pdf_exists, parse_omit_pages
from modules.cli.layout_infer_runtime import (
    build_target_pages,
    compute_adaptive_chunk_size,
    get_pdf_total_pages,
    yield_layout_infer_pages,
)
from utils.logger import get_logger
from utils.run_registry import next_versioned_dir, register_latest_run

log = get_logger("LayoutCLI")


def run_layout_prep(
    pdf_path: Annotated[
        str, typer.Option("--pdf-path", help="Path to the PDF to extract labeling samples from.")
    ],
    output_dir: str = typer.Option("output/layout_prep", "--output-dir"),
    num_pages: int | None = typer.Option(
        None,
        "--num-pages",
        help="Number of pages to export. If omitted, exports all pages in the PDF.",
    ),
    dpi: int = typer.Option(300, "--dpi"),
) -> None:
    """Convert PDF pages into run-scoped images for downstream layout workflows."""
    log.info("Initializing layout-prep dependencies...")
    from modules.ocr_engine.layout.yolo_engine import export_images_for_labeling

    source_path = ensure_pdf_exists(pdf_path, context_label="Layout Prep Failed")
    run_dir = next_versioned_dir(Path(output_dir), source_path.stem)
    visuals_dir = run_dir / "visuals"
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    logging.getLogger("PDFtoImage").setLevel(logging.WARNING)
    log.info("Preparing page images for layout-prep...")
    try:
        exported_count = export_images_for_labeling(
            pdf_path=source_path, output_dir=visuals_dir, num_pages=num_pages, dpi=dpi
        )
    except PermissionError as exc:
        log.error(str(exc))
        raise typer.Exit(code=1) from exc

    pointer = register_latest_run(
        stage="layout-prep",
        doc_stem=source_path.stem,
        run_dir=run_dir,
        artifacts={"visuals_dir": str(visuals_dir)},
        metadata={"num_pages": num_pages, "dpi": dpi, "exported_count": exported_count},
    )
    log.info(f"Latest pointer updated: {pointer}")

    log.info(f"Layout prep complete. Exported image source directory: {visuals_dir}")
    log.info(
        "This stage exports images only (no Label Studio task JSON). "
        "Use 'layout-infer' to generate importable auto-label tasks."
    )
    log.info(
        "If using Label Studio Local Files directly, point to: "
        f"/label-studio/files/layout_prep/{run_dir.name}/visuals (host: {visuals_dir})"
    )


def run_train_layout(
    data_yaml: str = typer.Option("input/layout_dataset/dataset.yaml", "--data"),
    epochs: int = typer.Option(100, "--epochs"),
    imgsz: int = typer.Option(1024, "--imgsz"),
) -> None:
    """Fine-tune YOLOv8-small on the 3060 Ti using the labeled dataset."""
    from ultralytics import YOLO

    log.info(f"Starting YOLOv8-small training on {data_yaml}...")
    model = YOLO("yolov8s-seg.pt")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        plots=True,
        project="runs/layout",
        name="berana_divider_v1",
    )
    log.info(
        "✅ Training Complete. Best weights saved in runs/layout/berana_divider_v1/weights/best.pt"
    )


def run_layout_infer(
    pdf_path: Annotated[
        str, typer.Option("--pdf-path", help="Path to the PDF to run auto-labeling on.")
    ],
    output_dir: str = typer.Option(
        "output/layout_inference",
        "--output-dir",
        help="Base directory for versioned auto-label runs.",
    ),
    start_page: int = typer.Option(
        0, "--start-page", help="Page number to start from. Use 0 to start at the first page."
    ),
    num_pages: int | None = typer.Option(
        None,
        "--num-pages",
        help="How many pages to process. If omitted, processes to document end.",
    ),
    omit_pages: Annotated[
        str | None,
        typer.Option("--omit-pages", help="Pages to skip. Supports ranges: '1-9,45,67-69'."),
    ] = None,
    verbose: bool = typer.Option(
        False, "--verbose", help="Enable detailed per-page YOLO debug logs."
    ),
    chunk_size: int | None = typer.Option(
        None, "--chunk-size", help="Chunk size for PDF conversion. If omitted, auto-tuned."
    ),
    cache_images: bool = typer.Option(
        True, "--cache-images/--no-cache-images", help="Cache rendered page images for reuse."
    ),
    dpi: int = typer.Option(300, "--dpi"),
) -> None:
    """Run AI auto-labeling on pages for human verification."""
    from tqdm import tqdm

    from modules.ocr_engine.layout.yolo_engine import LayoutSegmentationEngine

    source_path = ensure_pdf_exists(pdf_path, context_label="Layout Infer Failed")
    parsed_omit_pages = set(parse_omit_pages(omit_pages))
    physical_start_page = 1 if start_page <= 0 else start_page
    end_page = (
        None if num_pages is None else (physical_start_page + num_pages - 1)
    )  # yield_pdf_pages is 1-indexed and inclusive

    if not verbose:
        logging.getLogger("YOLOEngine").setLevel(logging.INFO)
        logging.getLogger("PDFtoImage").setLevel(logging.WARNING)

    engine = LayoutSegmentationEngine()
    if engine.model is None:
        log.error("AI Auto-Labeling Failed: No trained weights found. Run 'train-layout' first.")
        raise typer.Exit(code=1)

    run_dir = next_versioned_dir(Path(output_dir), source_path.stem)
    data_dir = run_dir / "data"
    visuals_dir = run_dir / "visuals"
    meta_dir = run_dir / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    if num_pages is None:
        log.info(f"Running Vision-AI Inference from page {physical_start_page} to document end...")
    else:
        log.info(
            f"Running Vision-AI Inference on {num_pages} pages starting at {physical_start_page}..."
        )

    total_pdf_pages = get_pdf_total_pages(source_path)
    target_pages = build_target_pages(
        total_pages=total_pdf_pages,
        start_page=physical_start_page,
        end_page=end_page,
        omit_pages=parsed_omit_pages,
    )
    expected_total = len(target_pages)
    resolved_chunk_size = (
        chunk_size
        if chunk_size is not None
        else compute_adaptive_chunk_size(dpi=dpi, pages_to_process=expected_total)
    )
    cache_dir = Path(output_dir) / "_cache" / f"{source_path.stem}_dpi{dpi}"
    log.info(
        f"Runtime Strategy | pages={expected_total} chunk={resolved_chunk_size} "
        f"cache={'on' if cache_images else 'off'}"
    )

    progress = tqdm(
        total=expected_total,
        desc="Layout Inference",
        unit="page",
        dynamic_ncols=True,
    )
    processed_count = 0
    model_scores_seen = 0
    model_score_total = 0.0
    page_average_confidences: list[float] = []
    rolling_page_avg = deque(maxlen=20)
    low_conf_pages = 0
    missing_divider_pages = 0

    for page_num, img in yield_layout_infer_pages(
        pdf_path=source_path,
        target_pages=target_pages,
        dpi=dpi,
        chunk_size=resolved_chunk_size,
        cache_dir=cache_dir,
        use_cache=cache_images,
    ):
        filename = f"page_{page_num:03d}.jpg"
        dest = visuals_dir / filename
        img.save(dest, "JPEG", quality=95)
        # Label Studio task imports resolve local-files URLs from document root.
        # Emit output-root-relative paths for robust imports across storage subdirs.
        output_root = Path("output").resolve()
        try:
            image_relpath = str(dest.resolve().relative_to(output_root)).replace("\\", "/")
        except ValueError:
            image_relpath = str(dest).replace("\\", "/")
        task = engine.generate_auto_labels(img, page_num, image_relpath)
        prediction_results = task.get("annotations", [{}])[0].get("result", [])
        scores = [
            float(item.get("score", 0.0))
            for item in prediction_results
            if isinstance(item.get("score", None), (int, float))
        ]
        page_avg_conf = sum(scores) / len(scores) if scores else 0.0
        if page_avg_conf < 0.3:
            low_conf_pages += 1
        if len(scores) < 2:
            missing_divider_pages += 1

        page_average_confidences.append(page_avg_conf)
        rolling_page_avg.append(page_avg_conf)
        model_score_total += sum(scores)
        model_scores_seen += len(scores)

        tasks.append(task)
        processed_count += 1
        progress.update(1)
        running_avg_conf = model_score_total / model_scores_seen if model_scores_seen else 0.0
        moving_avg_conf = sum(rolling_page_avg) / len(rolling_page_avg) if rolling_page_avg else 0.0
        progress.set_postfix_str(
            " ".join(
                [
                    f"last=page_{page_num:03d}",
                    f"conf_avg={running_avg_conf:.3f}",
                    f"conf_m20={moving_avg_conf:.3f}",
                ]
            )
        )

    progress.close()

    json_path = data_dir / "auto_labels_tasks.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(tasks, file, indent=2)

    log.info(
        f"✅ Auto-Labeling Complete. Processed {processed_count} pages. Import '{json_path}' into Label Studio."
    )
    pointer = register_latest_run(
        stage="layout-infer",
        doc_stem=source_path.stem,
        run_dir=run_dir,
        artifacts={
            "auto_labels_tasks": str(json_path),
            "visuals_dir": str(visuals_dir),
        },
        metadata={
            "start_page": physical_start_page,
            "end_page": end_page,
            "dpi": dpi,
            "chunk_size": resolved_chunk_size,
        },
    )
    log.info(f"Latest pointer updated: {pointer}")
    if processed_count > 0:
        overall_model_avg = model_score_total / model_scores_seen if model_scores_seen else 0.0
        page_avg_median = statistics.median(page_average_confidences)
        page_avg_min = min(page_average_confidences)
        page_avg_max = max(page_average_confidences)
        low_conf_ratio = low_conf_pages / processed_count
        missing_ratio = missing_divider_pages / processed_count
        log.info(
            "Inference Summary | "
            f"Model score avg={overall_model_avg:.3f} | "
            f"Page avg median={page_avg_median:.3f} min={page_avg_min:.3f} max={page_avg_max:.3f}"
        )
        log.info(
            "Coverage Summary | "
            f"Low-confidence pages (<0.30): {low_conf_pages}/{processed_count} ({low_conf_ratio:.1%}) | "
            f"Pages with missing divider predictions (<2 labels): "
            f"{missing_divider_pages}/{processed_count} ({missing_ratio:.1%})"
        )
    log.info(
        "Note: In Label Studio, set Local Files path to a subdirectory under "
        "'/label-studio/files' (security requirement), then import this run JSON."
    )
