import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image, ImageDraw

# Add project root to PYTHONPATH before other imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now imports can resolve correctly if they are in the project or site-packages
from surya.detection import DetectionPredictor  # noqa: E402

from modules.ocr_engine.orchestrator import run_precision_extraction_pipeline  # noqa: E402
from modules.ocr_engine.pre_processors.pdf_to_image import yield_pdf_pages  # noqa: E402
from utils.run_registry import load_latest_run  # noqa: E402

app = typer.Typer(help="Test Surya line detection on cropped column strips.")


@contextmanager
def _pushd(path: Path):
    """Temporarily switch working directory."""
    original = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original)


def _resolve_repo_path(path: Path) -> Path:
    """Resolve relative paths against project root for stable CLI behavior."""
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = PROJECT_ROOT / path
    return candidate


def _canonicalize_path(path_str: str | Path) -> Path:
    """Convert pointer/artifact paths to absolute project-root-aware paths."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _fallback_naive_column_crop(pdf_path: Path, page: int, column: int) -> Image.Image:
    """Fallback cropper that slices page into equal-width columns."""
    page_iter = yield_pdf_pages(
        pdf_path=pdf_path,
        start_page=page,
        end_page=page,
        chunk_size=1,
    )
    try:
        _page_num, full_page = next(page_iter)
    except StopIteration as exc:
        raise FileNotFoundError(f"Could not render page {page} from PDF: {pdf_path}") from exc

    width, height = full_page.size
    col_width = width // 3
    if column == 0:
        x1, x2 = 0, col_width
    elif column == 1:
        x1, x2 = col_width, 2 * col_width
    else:
        x1, x2 = 2 * col_width, width
    return full_page.crop((x1, 0, x2, height)).convert("RGB")


def _resolve_column_image_path(pdf_path: Path, page_id: str, target_name: str) -> Path | None:
    pointer = load_latest_run("crop-columns", pdf_path.stem)
    if not pointer:
        return None
    spliced_dir = _canonicalize_path(pointer["artifacts"]["spliced_dir"])
    candidate_path = spliced_dir / page_id / f"{target_name}.png"
    return candidate_path if candidate_path.exists() else None


def _generate_crop_for_page(
    pdf_path: Path, page: int, page_id: str, target_name: str
) -> Path | None:
    print(f"Cropped image not found for {page_id}. Running precision extraction pipeline...")
    try:
        with _pushd(PROJECT_ROOT):
            run_precision_extraction_pipeline(
                pdf_path=pdf_path,
                output_dir=PROJECT_ROOT / "output" / "column_crops",
                start_page=page,
                end_page=page,
            )
    except FileNotFoundError as err:
        print(f"crop-columns unavailable ({err}). Using naive page-third crop fallback.")
        return None
    return _resolve_column_image_path(pdf_path, page_id, target_name)


def _load_cropped_or_fallback_image(pdf_path: Path, page: int, column: int) -> Image.Image:
    keys = ["geez", "amharic", "english"]
    if column < 0 or column >= len(keys):
        raise typer.BadParameter(f"Column index {column} out of bounds. Must be 0, 1, or 2.")

    page_id = f"page_{page:03d}"
    target_name = keys[column]
    img_path = _resolve_column_image_path(pdf_path, page_id, target_name)
    if not img_path:
        img_path = _generate_crop_for_page(pdf_path, page, page_id, target_name)
    if img_path and img_path.exists():
        print(f"Loading cropped image from {img_path}")
        return Image.open(img_path).convert("RGB")
    print("Using naive equal-width column crop from source PDF...")
    return _fallback_naive_column_crop(pdf_path=pdf_path, page=page, column=column)


def _run_surya_line_detection(img: Image.Image):
    print("Loading Surya detection model...")
    try:
        predictor = DetectionPredictor()
    except Exception as e:
        print(f"Failed to load Surya model: {e}")
        raise typer.Exit(code=1) from e
    print("Running Surya detection on cropped strip...")
    return predictor([img])[0]


def _save_bbox_visualization(img: Image.Image, bboxes: list, output_img_path: Path) -> None:
    print("Drawing barebones unlabelled bounding boxes...")
    draw = ImageDraw.Draw(img)
    for box in bboxes:
        draw.rectangle(box.bbox, outline="red", width=2)
    output_img_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_img_path, format="JPEG", quality=90)
    print(f"Saved visualization to {output_img_path}")


@app.command(name="test-surya-lines")
def run_surya_lines_cli(
    pdf_path: Annotated[Path, typer.Option("--pdf-path", help="Path to the source PDF.")],
    page: Annotated[int, typer.Option("--page", help="Specific page number to process.")] = 101,
    column: Annotated[
        int,
        typer.Option("--column", help="Column index to analyze (0, 1, 2... from left to right)."),
    ] = 0,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Output directory for visualization."),
    ] = Path("output/surya_diagnostics/line_detection"),
):
    """Run barebones Surya line detection on a specified column strip."""
    resolved_pdf_path = _resolve_repo_path(pdf_path)
    resolved_output_dir = _resolve_repo_path(output_dir)
    output_img_path = (
        resolved_output_dir / f"surya_line_test_barebones_page_{page}_col_{column}.jpg"
    )

    print(f"Resolving cropped image for page {page} (Column {column})...")
    try:
        img = _load_cropped_or_fallback_image(resolved_pdf_path, page, column)
    except Exception as e:
        print(f"Error handling cropped image: {e}")
        raise typer.Exit(code=1) from e

    res = _run_surya_line_detection(img)

    print(f"Found {len(res.bboxes)} raw line bounding boxes.")
    _save_bbox_visualization(img, res.bboxes, output_img_path)


if __name__ == "__main__":
    app()
