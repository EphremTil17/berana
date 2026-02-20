import logging
from collections.abc import Generator
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

logger = logging.getLogger(__name__)


def yield_pdf_pages(
    pdf_path: Path,
    chunk_size: int = 50,
    dpi: int = 300,
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
    except Exception as e:
        logger.error(
            f"Failed to read PDF metadata for {pdf_path}. Ensure poppler-utils is installed."
        )
        raise RuntimeError(f"PDF integrity failure: {e}") from e

    logger.info(
        f"Initialized memory-safe parsing for {total_pages} pages utilizing chunk size {chunk_size}."
    )

    for start_page in range(1, total_pages + 1, chunk_size):
        end_page = min(start_page + chunk_size - 1, total_pages)
        logger.debug(f"Converting PDF chunk: Pages {start_page} to {end_page}...")

        # convert_from_path is 1-indexed for first_page/last_page
        images = convert_from_path(
            pdf_path=str(pdf_path),
            dpi=dpi,
            first_page=start_page,
            last_page=end_page,
            use_cropbox=True,  # Crucial for stripping blank margins if the PDF defines them
            fmt="jpeg",  # Using JPEG to save RAM over raw PPMs during processing
        )

        current_page_num = start_page
        for img in images:
            yield current_page_num, img
            current_page_num += 1
