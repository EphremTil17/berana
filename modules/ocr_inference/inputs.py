from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

from PIL import Image

from modules.ocr_engine.pre_processors.pdf_to_image import yield_pdf_pages
from modules.ocr_inference.schemas import OCRTask, SourceArtifacts
from utils.run_registry import load_latest_run, resolve_required_input


class OCRInferenceInputError(RuntimeError):
    """Raised when OCR inference inputs cannot be resolved from upstream artifacts."""


def resolve_source_artifacts(*, doc_stem: str) -> SourceArtifacts:
    """Resolve the latest crop-columns artifacts required for OCR inference."""
    pointer = load_latest_run("crop-columns", doc_stem)
    if not pointer:
        raise OCRInferenceInputError(
            "No crop-columns artifacts found for this PDF. Run `berana.py crop-columns --pdf-path ...` first."
        )

    try:
        cropping_manifest = resolve_required_input(
            upstream_stage="crop-columns",
            doc_stem=doc_stem,
            artifact_key="cropping_manifest",
        )
        spliced_dir = resolve_required_input(
            upstream_stage="crop-columns",
            doc_stem=doc_stem,
            artifact_key="spliced_dir",
        )
    except (FileNotFoundError, KeyError) as exc:
        raise OCRInferenceInputError(
            "Latest crop-columns run is incomplete. Re-run `berana.py crop-columns --pdf-path ...` and try again."
        ) from exc

    return SourceArtifacts(
        crop_run_dir=Path(pointer["run_dir"]),
        cropping_manifest=cropping_manifest,
        spliced_dir=spliced_dir,
        crop_registry_pointer=pointer,
    )


def resolve_source_artifacts_optional(*, doc_stem: str) -> SourceArtifacts | None:
    """Resolve crop-columns artifacts if they exist, otherwise return None."""
    try:
        return resolve_source_artifacts(doc_stem=doc_stem)
    except OCRInferenceInputError:
        return None


def collect_ocr_tasks(
    *,
    pdf_path: Path,
    source_artifacts: SourceArtifacts,
    start_page: int,
    end_page: int | None,
    omit_pages: list[int] | None,
) -> list[OCRTask]:
    """Collect OCR tasks from the crop-columns manifest, preserving language labels."""
    manifest_rows = json.loads(source_artifacts.cropping_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest_rows, list):
        raise OCRInferenceInputError(
            f"Invalid cropping manifest shape: expected list in {source_artifacts.cropping_manifest}."
        )

    skip_pages = set(omit_pages or [])
    last_page = end_page if end_page is not None else 10**9
    tasks: list[OCRTask] = []

    for row in manifest_rows:
        if row.get("status") != "SUCCESS":
            continue
        page_number = int(row.get("source_page") or 0)
        if page_number < start_page or page_number > last_page or page_number in skip_pages:
            continue

        strip_paths = row.get("strip_paths") or {}
        if not isinstance(strip_paths, dict):
            continue

        for ordering_index, language in enumerate(sorted(strip_paths)):
            image_path = Path(strip_paths[language])
            if not image_path.exists():
                raise OCRInferenceInputError(
                    f"Crop image missing on disk for page={row.get('page_id')} language={language}: {image_path}"
                )
            tasks.append(
                OCRTask(
                    doc_stem=pdf_path.stem,
                    pdf_path=pdf_path,
                    page_id=str(row["page_id"]),
                    page_number=page_number,
                    language=str(language),
                    image_path=image_path,
                    source_page=page_number,
                    ordering_index=ordering_index,
                    crop_run_dir=source_artifacts.crop_run_dir,
                )
            )

    if not tasks:
        raise OCRInferenceInputError(
            "No OCR tasks remain after page filtering. Check crop-columns outputs and page filters."
        )
    return tasks


def collect_crop_ocr_tasks(
    *,
    source_path: Path,
    source_artifacts: SourceArtifacts,
    selected_pages: list[int] | None,
) -> list[OCRTask]:
    """Collect crop-layout OCR tasks with an optional include-only page filter."""
    include_pages = sorted(set(selected_pages or []))
    if include_pages:
        start_page = include_pages[0]
        end_page = include_pages[-1]
        include_set = set(include_pages)
        omit_pages = [page for page in range(start_page, end_page + 1) if page not in include_set]
    else:
        start_page = 1
        end_page = None
        omit_pages = None

    return collect_ocr_tasks(
        pdf_path=source_path,
        source_artifacts=source_artifacts,
        start_page=start_page,
        end_page=end_page,
        omit_pages=omit_pages,
    )


def iter_source_images(
    *,
    source_path: Path,
    selected_pages: list[int] | None,
    dpi: int = 300,
    chunk_size: int = 50,
) -> Generator[tuple[int, Image.Image], None, None]:
    """Yield page images from a PDF or a single standalone image source."""
    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        include_pages = sorted(set(selected_pages or []))
        if include_pages:
            first_page = include_pages[0]
            last_page = include_pages[-1]
            include_set = set(include_pages)
            omit_pages = [
                page for page in range(first_page, last_page + 1) if page not in include_set
            ]
        else:
            first_page = 1
            last_page = None
            omit_pages = None

        yield from yield_pdf_pages(
            source_path,
            chunk_size=chunk_size,
            dpi=dpi,
            start_page=first_page,
            omit_pages=omit_pages,
            end_page=last_page,
        )
        return

    if selected_pages and set(selected_pages) != {1}:
        raise OCRInferenceInputError(
            "Image sources support only page 1; omit `--pages` or use `--pages 1`."
        )

    with Image.open(source_path) as image:
        yield 1, image.convert("RGB")
