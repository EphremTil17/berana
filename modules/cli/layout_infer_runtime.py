from __future__ import annotations

from collections.abc import Generator, Iterable, Iterator
from pathlib import Path
from queue import Queue
from threading import Thread

from pdf2image.pdf2image import pdfinfo_from_path
from PIL import Image

from modules.ocr_engine.pre_processors.pdf_to_image import yield_pdf_pages


def get_pdf_total_pages(pdf_path: Path) -> int:
    """Return total page count for a PDF."""
    info = pdfinfo_from_path(str(pdf_path))
    return int(info["Pages"])


def build_target_pages(
    total_pages: int,
    start_page: int,
    end_page: int | None,
    omit_pages: set[int],
) -> list[int]:
    """Build the exact list of page numbers to process."""
    upper_bound = min(total_pages, end_page) if end_page is not None else total_pages
    if upper_bound < start_page:
        return []
    return [page for page in range(start_page, upper_bound + 1) if page not in omit_pages]


def compute_adaptive_chunk_size(dpi: int, pages_to_process: int) -> int:
    """Choose a conservative chunk size to balance RAM pressure and throughput."""
    if pages_to_process <= 20:
        return max(4, min(10, pages_to_process))
    if dpi >= 400:
        return 8
    if dpi >= 300:
        return 10
    return 16


def contiguous_ranges(pages: list[int]) -> list[tuple[int, int]]:
    """Group sorted page numbers into contiguous ranges."""
    if not pages:
        return []

    ranges: list[tuple[int, int]] = []
    range_start = pages[0]
    prev = pages[0]

    for current in pages[1:]:
        if current != prev + 1:
            ranges.append((range_start, prev))
            range_start = current
        prev = current

    ranges.append((range_start, prev))
    return ranges


def cache_image_path(cache_dir: Path, page_num: int) -> Path:
    """Return deterministic cache image path for one page."""
    return cache_dir / f"page_{page_num:03d}.jpg"


def cached_pages(cache_dir: Path, target_pages: list[int]) -> set[int]:
    """Return page numbers already cached on disk."""
    return {page for page in target_pages if cache_image_path(cache_dir, page).exists()}


def _prefetch_iter(
    iterable: Iterable[tuple[int, Image.Image]], buffer_size: int = 8
) -> Iterator[tuple[int, Image.Image]]:
    """Prefetch iterable items using a background thread."""
    sentinel = object()
    queue: Queue[object] = Queue(maxsize=buffer_size)

    def producer() -> None:
        for item in iterable:
            queue.put(item)
        queue.put(sentinel)

    thread = Thread(target=producer, daemon=True)
    thread.start()

    while True:
        item = queue.get()
        if item is sentinel:
            break
        yield item  # type: ignore[misc]


def yield_layout_infer_pages(
    pdf_path: Path,
    target_pages: list[int],
    dpi: int,
    chunk_size: int,
    cache_dir: Path,
    use_cache: bool,
) -> Generator[tuple[int, Image.Image], None, None]:
    """Yield target pages from cache first, then PDF conversion for missing pages."""
    if not target_pages:
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cached_pages(cache_dir, target_pages) if use_cache else set()

    for page in sorted(cached):
        path = cache_image_path(cache_dir, page)
        with Image.open(path) as image:
            yield page, image.copy()

    missing = sorted(set(target_pages) - cached)
    if not missing:
        return

    for range_start, range_end in contiguous_ranges(missing):
        page_iter = yield_pdf_pages(
            pdf_path=pdf_path,
            chunk_size=chunk_size,
            dpi=dpi,
            start_page=range_start,
            end_page=range_end,
            thread_count=4,
        )
        for page_num, image in _prefetch_iter(page_iter, buffer_size=8):
            if page_num not in target_pages:
                continue

            if use_cache:
                image.save(cache_image_path(cache_dir, page_num), "JPEG", quality=95)
            yield page_num, image
