import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from tools.hitl_line_editor_app.state import load_state
from tools.hitl_yolo_finetuner_app.geometry import apply_vertical_clip, process_line_to_polygon
from utils.logger import get_logger

log = get_logger("YOLOPreview")


def parse_omit_pages(omit_pages: str | None) -> set[int]:
    """Parse page filters like '1,2,5-8' into a set of page numbers."""
    parsed: set[int] = set()
    if not omit_pages:
        return parsed
    for part in omit_pages.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_page, end_page = map(int, chunk.split("-", maxsplit=1))
            parsed.update(range(start_page, end_page + 1))
        else:
            parsed.add(int(chunk))
    return parsed


def _page_num_from_id(page_id: str) -> int | None:
    match = re.search(r"(\d+)$", page_id)
    if not match:
        return None
    return int(match.group(1))


def _poly_to_points(poly: list[float], img_w: int, img_h: int) -> np.ndarray:
    return np.array(
        [
            [int(poly[0] * img_w), int(poly[1] * img_h)],
            [int(poly[2] * img_w), int(poly[3] * img_h)],
            [int(poly[4] * img_w), int(poly[5] * img_h)],
            [int(poly[6] * img_w), int(poly[7] * img_h)],
        ],
        np.int32,
    ).reshape((-1, 1, 2))


def _draw_polyline(
    image: np.ndarray, poly: list[float] | None, img_w: int, img_h: int, color: tuple[int, int, int]
) -> None:
    if not poly:
        return
    cv2.polylines(
        image, [_poly_to_points(poly, img_w, img_h)], isClosed=True, color=color, thickness=2
    )


def preview_polygons(
    db_file: Path,
    images_source_dir: Path,
    output_dir: Path,
    line_width_px: float = 4.0,
    max_pages: int = 10,
    clip_top_px: float = 0.0,
    clip_bottom_px: float = 0.0,
    omit_pages: str | None = None,
) -> None:
    """Render mathematical polygons over source images to visually validate alignment."""
    state = load_state(db_file)
    if not state:
        raise ValueError(f"No state found in {db_file}.")

    preview_dir = output_dir
    preview_dir.mkdir(parents=True, exist_ok=True)
    omitted = parse_omit_pages(omit_pages)

    rendered = 0
    target = min(max_pages, sum(1 for page in state.values() if page.get("verified")))
    progress = tqdm(total=target, desc="Generating preview overlays")
    for page_id, page in state.items():
        if rendered >= target:
            break
        page_num = _page_num_from_id(page_id)
        if page_num is not None and page_num in omitted:
            continue

        if not page.get("verified"):
            continue

        left = page.get("left")
        right = page.get("right")
        img_w = page.get("img_w", 0)
        img_h = page.get("img_h", 0)

        if not left or not right or img_w <= 0 or img_h <= 0:
            continue

        src_img = images_source_dir / f"{page_id}.jpg"
        if not src_img.exists():
            continue

        image = cv2.imread(str(src_img))
        if image is None:
            log.warning(f"Could not read image {src_img}")
            continue

        clipped_left = apply_vertical_clip(
            left, img_h, clip_top_px=clip_top_px, clip_bottom_px=clip_bottom_px
        )
        clipped_right = apply_vertical_clip(
            right, img_h, clip_top_px=clip_top_px, clip_bottom_px=clip_bottom_px
        )
        if not clipped_left or not clipped_right:
            continue

        # Process polygons using the same clipped geometry as export.
        poly_left = process_line_to_polygon(clipped_left, img_w, img_h, line_width_px)
        poly_right = process_line_to_polygon(clipped_right, img_w, img_h, line_width_px)

        _draw_polyline(image, poly_left, img_w, img_h, color=(0, 255, 0))
        _draw_polyline(image, poly_right, img_w, img_h, color=(255, 0, 0))

        out_path = preview_dir / f"{page_id}_overlay.jpg"
        cv2.imwrite(str(out_path), image)
        rendered += 1
        progress.update(1)

    progress.close()

    log.info(f"✅ Generated {rendered} visual preview overlays in {preview_dir}")
