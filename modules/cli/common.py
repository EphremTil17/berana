from pathlib import Path

import typer

from utils.logger import get_logger

log = get_logger("CLICommon")


def parse_omit_pages(omit_pages: str | None) -> list[int]:
    """Parse comma/range page syntax (e.g. '1,2,5-8') into page numbers."""
    parsed: list[int] = []
    if not omit_pages:
        return parsed

    for part in omit_pages.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_page, end_page = map(int, chunk.split("-", maxsplit=1))
            parsed.extend(range(start_page, end_page + 1))
        else:
            parsed.append(int(chunk))

    return parsed


def ensure_pdf_exists(pdf_path: str, context_label: str) -> Path:
    """Validate source PDF exists and return normalized path."""
    source_path = Path(pdf_path)
    if not source_path.exists():
        log.error(f"{context_label}: Source PDF '{source_path}' does not exist.")
        raise typer.Exit(code=1)
    return source_path
