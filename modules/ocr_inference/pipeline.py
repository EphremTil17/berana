from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from modules.ocr_inference.inputs import (
    collect_crop_ocr_tasks,
    iter_source_images,
    resolve_source_artifacts_optional,
)
from modules.ocr_inference.outputs import write_crop_text_output, write_page_text_output
from modules.ocr_inference.schemas import OCRLine, OCRTask
from modules.ocr_inference.surya_runtime import (
    build_surya_detection_predictor,
    build_surya_predictor,
)
from modules.ocr_training.surya_common import relative_to_base, sanitize_prediction_text
from utils.logger import get_logger
from utils.run_registry import next_versioned_dir, register_latest_run

logger = get_logger("OCRInferencePipeline")

TAG_FILTER_LIST = [
    "p",
    "li",
    "ul",
    "ol",
    "table",
    "td",
    "tr",
    "th",
    "tbody",
    "pre",
    "b",
    "strong",
    "i",
    "em",
    "u",
    "span",
    "div",
    "br",
    "sup",
    "sub",
]


def _load_task_image(task: OCRTask) -> Image.Image:
    with Image.open(task.image_path) as image:
        return image.convert("RGB")


def _chunk_list(items: list, batch_size: int) -> list[list]:
    normalized = max(1, int(batch_size))
    return [items[index : index + normalized] for index in range(0, len(items), normalized)]


def _chunk_iterable(items, batch_size: int):
    normalized = max(1, int(batch_size))
    iterator = iter(items)
    while True:
        chunk = list(islice(iterator, normalized))
        if not chunk:
            break
        yield chunk


def _normalize_bbox(value) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "bbox"):
        return _normalize_bbox(value.bbox)
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return None
    return None


def _draw_line_annotations(image: Image.Image, annotations: list[dict[str, object]]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    for index, annotation in enumerate(annotations):
        bbox = annotation.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = [round(v) for v in bbox]
        draw.rectangle((x1, y1, x2, y2), outline="#ff3b30", width=3)
        label = f"line_{index:02d}"
        text_y = max(0, y1 - 12)
        draw.rectangle((x1, text_y, x1 + max(40, 6 * len(label)), text_y + 12), fill="#fff2a8")
        draw.text((x1 + 1, text_y), label, fill="black", font=font)
    return annotated


def _extract_ocr_lines(result) -> list[OCRLine]:
    lines: list[OCRLine] = []
    for index, text_line in enumerate(getattr(result, "text_lines", []) or []):
        text = sanitize_prediction_text(getattr(text_line, "text", "") or "")
        bbox = _normalize_bbox(text_line)
        polygon = getattr(text_line, "polygon", None)
        if bbox is None:
            continue
        if polygon is None:
            x1, y1, x2, y2 = bbox
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        lines.append(
            OCRLine(
                line_index=index,
                text=text,
                bbox=bbox,
                polygon=[[float(x), float(y)] for x, y in polygon],
                confidence=getattr(text_line, "confidence", None),
            )
        )
    return lines


def _aggregate_page_text(lines: list[OCRLine]) -> str:
    return "\n".join(line.text for line in lines if line.text.strip())


def _write_diagnostic_artifacts(
    *,
    run_dir: Path,
    image: Image.Image,
    source_label: str,
    page_number: int,
    language: str | None,
    stem: str,
    lines: list[OCRLine],
    nested: bool,
) -> None:
    diagnostic_dir = (
        run_dir / "images" / language / f"page_{page_number:03d}"
        if nested and language
        else run_dir / "images"
    )
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    annotations = [
        {
            "line_index": line.line_index,
            "text": line.text,
            "confidence": line.confidence,
            "bbox": line.bbox,
            "polygon": line.polygon,
        }
        for line in lines
    ]
    if not annotations:
        annotations.append(
            {
                "line_index": 0,
                "text": "",
                "confidence": None,
                "bbox": [0.0, 0.0, float(image.width), float(image.height)],
                "polygon": [
                    [0.0, 0.0],
                    [float(image.width), 0.0],
                    [float(image.width), float(image.height)],
                    [0.0, float(image.height)],
                ],
            }
        )

    annotated = _draw_line_annotations(image, annotations)
    if nested and language:
        annotated_path = diagnostic_dir / f"{stem}__annotated.png"
        annotations_path = diagnostic_dir / f"{stem}__annotations.json"
    else:
        annotated_path = diagnostic_dir / f"page_{page_number:03d}__annotated.png"
        annotations_path = diagnostic_dir / f"page_{page_number:03d}__annotations.json"
    annotated.save(annotated_path)
    annotations_path.write_text(
        json.dumps(
            {
                "image_path": source_label,
                "page_number": page_number,
                "language": language,
                "line_count": len(annotations),
                "annotations": annotations,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_predictor_on_images(
    *,
    runtime: dict,
    predictor,
    detection_predictor,
    images: list[Image.Image],
    eval_batch_size: int,
):
    task_names = [runtime["TaskNames"].ocr_with_boxes] * len(images)
    return predictor(
        images,
        task_names=task_names,
        det_predictor=detection_predictor,
        detection_batch_size=max(1, int(eval_batch_size)),
        recognition_batch_size=max(1, int(eval_batch_size)),
        sort_lines=True,
        math_mode=False,
        drop_repeated_text=True,
        filter_tag_list=TAG_FILTER_LIST,
    )


def _run_crop_layout_inference(
    *,
    source_path: Path,
    run_dir: Path,
    source_artifacts,
    runtime: dict,
    predictor_bundle,
    selected_pages: list[int] | None,
    eval_batch_size: int,
    dataloader_num_workers: int,
    diagnose: bool,
) -> tuple[int, bool, list[str]]:
    tasks = collect_crop_ocr_tasks(
        source_path=source_path,
        source_artifacts=source_artifacts,
        selected_pages=selected_pages,
    )
    detection_predictor = build_surya_detection_predictor()
    predictor = predictor_bundle.predictor
    executor = (
        ThreadPoolExecutor(max_workers=dataloader_num_workers)
        if dataloader_num_workers > 0
        else None
    )
    diagnostic_written = False
    languages = sorted({task.language for task in tasks if task.language})
    progress = tqdm(
        _chunk_list(tasks, eval_batch_size),
        desc="OCR Inference",
        unit="batch",
        dynamic_ncols=True,
    )
    processed = 0
    try:
        for task_batch in progress:
            if executor is not None:
                images = list(executor.map(_load_task_image, task_batch))
            else:
                images = [_load_task_image(task) for task in task_batch]
            results = _run_predictor_on_images(
                runtime=runtime,
                predictor=predictor,
                detection_predictor=detection_predictor,
                images=images,
                eval_batch_size=eval_batch_size,
            )
            for task, image, result in zip(task_batch, images, results, strict=False):
                lines = _extract_ocr_lines(result)
                text = _aggregate_page_text(lines)
                write_crop_text_output(
                    run_dir=run_dir,
                    language=task.language,
                    page_number=task.page_number,
                    text=text,
                )
                if diagnose:
                    _write_diagnostic_artifacts(
                        run_dir=run_dir,
                        image=image,
                        source_label=relative_to_base(task.image_path),
                        page_number=task.page_number,
                        language=task.language,
                        stem=task.image_path.stem,
                        lines=lines,
                        nested=True,
                    )
                    diagnostic_written = True
            processed += len(task_batch)
            progress.set_postfix(rows=processed, workers=dataloader_num_workers)
    finally:
        progress.close()
        if executor is not None:
            executor.shutdown(wait=True)
    return len(tasks), diagnostic_written, languages


def _run_generic_source_inference(
    *,
    source_path: Path,
    run_dir: Path,
    runtime: dict,
    predictor_bundle,
    selected_pages: list[int] | None,
    eval_batch_size: int,
    diagnose: bool,
) -> tuple[int, bool]:
    detection_predictor = build_surya_detection_predictor()
    predictor = predictor_bundle.predictor
    rendered_pages = iter_source_images(
        source_path=source_path,
        selected_pages=selected_pages,
        dpi=300,
    )
    diagnostic_written = False
    page_count = 0
    progress = tqdm(
        total=len(selected_pages) if selected_pages else None,
        desc="OCR Inference",
        unit="page",
        dynamic_ncols=True,
    )
    try:
        for page_batch in _chunk_iterable(rendered_pages, eval_batch_size):
            page_numbers = [item[0] for item in page_batch]
            images = [item[1] for item in page_batch]
            results = _run_predictor_on_images(
                runtime=runtime,
                predictor=predictor,
                detection_predictor=detection_predictor,
                images=images,
                eval_batch_size=eval_batch_size,
            )
            for page_number, image, result in zip(page_numbers, images, results, strict=False):
                lines = _extract_ocr_lines(result)
                text = _aggregate_page_text(lines)
                write_page_text_output(run_dir=run_dir, page_number=page_number, text=text)
                if diagnose:
                    _write_diagnostic_artifacts(
                        run_dir=run_dir,
                        image=image,
                        source_label=relative_to_base(source_path),
                        page_number=page_number,
                        language=None,
                        stem=source_path.stem,
                        lines=lines,
                        nested=False,
                    )
                    diagnostic_written = True
                page_count += 1
                progress.update(1)
    finally:
        progress.close()
    return page_count, diagnostic_written


def run_source_ocr_inference_pipeline(
    source_path: Path,
    output_dir: Path,
    *,
    checkpoint_dir: Path | None = None,
    zero_shot: bool = False,
    eval_batch_size: int = 1,
    dataloader_num_workers: int = 0,
    selected_pages: list[int] | None = None,
    diagnose: bool = False,
) -> Path:
    """Run standalone Surya OCR inference against a PDF or image source."""
    if not zero_shot and checkpoint_dir is None:
        raise ValueError("Provide `--checkpoint-dir` or explicitly use `--zero-shot`.")

    doc_stem = source_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = next_versioned_dir(output_dir, doc_stem)
    run_dir.mkdir(parents=True, exist_ok=True)

    runtime, predictor_bundle = build_surya_predictor(
        zero_shot=zero_shot,
        checkpoint_dir=checkpoint_dir,
    )
    wall_started_at = perf_counter()

    crop_artifacts = (
        resolve_source_artifacts_optional(doc_stem=doc_stem)
        if source_path.suffix.lower() == ".pdf"
        else None
    )
    if crop_artifacts is not None:
        logger.info("Using crop-columns artifacts for %s.", doc_stem)
        task_count, diagnostic_written, languages = _run_crop_layout_inference(
            source_path=source_path,
            run_dir=run_dir,
            source_artifacts=crop_artifacts,
            runtime=runtime,
            predictor_bundle=predictor_bundle,
            selected_pages=selected_pages,
            eval_batch_size=eval_batch_size,
            dataloader_num_workers=dataloader_num_workers,
            diagnose=diagnose,
        )
        register_latest_run(
            stage="ocr",
            doc_stem=doc_stem,
            run_dir=run_dir,
            artifacts={
                "text_output_dir": relative_to_base(run_dir),
                **(
                    {"diagnostic_images_dir": relative_to_base(run_dir / "images")}
                    if diagnose and diagnostic_written
                    else {}
                ),
            },
            metadata={
                "model_mode": predictor_bundle.model_info["model_mode"],
                "source_kind": "pdf",
                "used_crop_layout": True,
                "task_count": int(task_count),
                "languages": languages,
                "eval_batch_size": int(eval_batch_size),
                "dataloader_num_workers": int(dataloader_num_workers),
                "elapsed_sec": perf_counter() - wall_started_at,
                "checkpoint_dir": predictor_bundle.model_info.get("checkpoint_dir"),
                "diagnose": bool(diagnose),
            },
        )
        logger.info(
            "OCR inference complete doc=%s mode=%s crop_layout=true tasks=%d output=%s",
            doc_stem,
            predictor_bundle.model_info["model_mode"],
            task_count,
            run_dir,
        )
        return run_dir

    page_count, diagnostic_written = _run_generic_source_inference(
        source_path=source_path,
        run_dir=run_dir,
        runtime=runtime,
        predictor_bundle=predictor_bundle,
        selected_pages=selected_pages,
        eval_batch_size=eval_batch_size,
        diagnose=diagnose,
    )
    register_latest_run(
        stage="ocr",
        doc_stem=doc_stem,
        run_dir=run_dir,
        artifacts={
            "text_output_dir": relative_to_base(run_dir),
            **(
                {"diagnostic_images_dir": relative_to_base(run_dir / "images")}
                if diagnose and diagnostic_written
                else {}
            ),
        },
        metadata={
            "model_mode": predictor_bundle.model_info["model_mode"],
            "source_kind": "pdf" if source_path.suffix.lower() == ".pdf" else "image",
            "used_crop_layout": False,
            "page_count": int(page_count),
            "selected_pages": selected_pages or [],
            "eval_batch_size": int(eval_batch_size),
            "dataloader_num_workers": int(dataloader_num_workers),
            "elapsed_sec": perf_counter() - wall_started_at,
            "checkpoint_dir": predictor_bundle.model_info.get("checkpoint_dir"),
            "diagnose": bool(diagnose),
        },
    )
    logger.info(
        "OCR inference complete doc=%s mode=%s crop_layout=false pages=%d output=%s",
        doc_stem,
        predictor_bundle.model_info["model_mode"],
        page_count,
        run_dir,
    )
    return run_dir


def run_pdf_ocr_inference_pipeline(
    pdf_path: Path,
    output_dir: Path,
    *,
    checkpoint_dir: Path | None = None,
    zero_shot: bool = False,
    eval_batch_size: int = 1,
    dataloader_num_workers: int = 0,
    start_page: int = 1,
    end_page: int | None = None,
    omit_pages: list[int] | None = None,
    diagnose: bool = False,
) -> Path:
    """Backward-compatible wrapper for the old PDF-only entrypoint."""
    selected_pages = None
    if end_page is not None:
        selected = set(range(start_page, end_page + 1))
        selected -= set(omit_pages or [])
        selected_pages = sorted(page for page in selected if page > 0)
    return run_source_ocr_inference_pipeline(
        source_path=pdf_path,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        zero_shot=zero_shot,
        eval_batch_size=eval_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        selected_pages=selected_pages,
        diagnose=diagnose,
    )
