import os
from collections.abc import Generator
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

from utils.logger import get_logger

logger = get_logger("PDFtoImage")


def yield_pdf_pages(
    pdf_path: Path,
    chunk_size: int = 50,
    dpi: int = 300,
    start_page: int = 1,
    thread_count: int | None = None,
    omit_pages: list[int] | None = None,
    end_page: int | None = None,
) -> Generator[tuple[int, Image.Image], None, None]:
    """A memory-safe generator that processes a large PDF in chunks.

    This strictly adheres to the 'Memory Management (Rule 5)' standard,
    ensuring we never load a 700-page PDF into RAM all at once.

    Yields:
        tuple[int, Image.Image]: The physical page number (1-indexed) and the high-res PIL Image.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"Cannot process PDF: '{pdf_path}' does not exist.")

    # Get total page count first so we can iterate by chunks securely
    try:
        from pdf2image.pdf2image import pdfinfo_from_path

        info = pdfinfo_from_path(str(pdf_path))
        total_pages = int(info["Pages"])
        first_page = max(1, start_page)
        last_page = min(total_pages, end_page) if end_page is not None else total_pages
    except Exception as e:
        logger.error(
            f"Failed to read PDF metadata for {pdf_path}. Ensure poppler-utils is installed."
        )
        raise RuntimeError(f"PDF integrity failure: {e}") from e

    if last_page < first_page:
        logger.info(
            f"No pages to process for {pdf_path} with start_page={first_page} and end_page={end_page}."
        )
        return

    planned_pages = (last_page - first_page) + 1
    logger.info(
        f"Initialized memory-safe parsing for {planned_pages} pages "
        f"(start={first_page}, end={last_page}) utilizing chunk size {chunk_size}."
    )

    if thread_count is None:
        cpu_count = os.cpu_count() or 2
        # Poppler rendering scales well with additional workers; prefer fuller CPU usage.
        thread_count = max(1, min(8, cpu_count))

    skip_set = set(omit_pages) if omit_pages else set()
    for chunk_start in range(first_page, last_page + 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size - 1, last_page)
        logger.debug(f"Converting PDF chunk: Pages {chunk_start} to {chunk_end}...")

        # convert_from_path is 1-indexed for first_page/last_page
        images = convert_from_path(
            pdf_path=str(pdf_path),
            dpi=dpi,
            first_page=chunk_start,
            last_page=chunk_end,
            thread_count=thread_count,
            use_cropbox=True,  # Crucial for stripping blank margins if the PDF defines them
            fmt="jpeg",  # Using JPEG to save RAM over raw PPMs during processing
        )

        current_page_num = chunk_start

        for img in images:
            if current_page_num in skip_set:
                logger.info(f"Skipping page {current_page_num} as requested by omit list.")
            else:
                yield current_page_num, img

            current_page_num += 1
