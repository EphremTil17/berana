import json
from pathlib import Path
from typing import Any

from tools.hitl_line_editor_app.state import load_state


def _x_at_y(line: list[float], y_value: float) -> float:
    x1, y1, x2, y2 = line
    if y2 == y1:
        return float(x1)
    ratio = (y_value - y1) / (y2 - y1)
    return float(x1 + ratio * (x2 - x1))


def build_ocr_ready_map(only_verified: bool = True) -> dict[str, dict[str, Any]]:
    """Build a plain JSON-ready map with divider lines and orthogonal column x ranges."""
    state = load_state()
    exported: dict[str, dict[str, Any]] = {}

    for page_id, page in state.items():
        left = page.get("left")
        right = page.get("right")
        verified = bool(page.get("verified", False))
        img_w = int(page.get("img_w", 0))
        img_h = int(page.get("img_h", 0))

        if only_verified and not verified:
            continue
        if not left or not right or img_w <= 0 or img_h <= 0:
            continue

        mid_y = img_h / 2
        left_x = _x_at_y(left, mid_y)
        right_x = _x_at_y(right, mid_y)
        left_x, right_x = sorted([left_x, right_x])

        exported[page_id] = {
            "verified": verified,
            "image_size": [img_w, img_h],
            "dividers": {
                "left": [float(value) for value in left],
                "right": [float(value) for value in right],
            },
            "column_x_ranges": [
                [0.0, left_x],
                [left_x, right_x],
                [right_x, float(img_w)],
            ],
        }

    return exported


def export_ocr_ready_json(output_path: Path, only_verified: bool = True) -> Path:
    """Write OCR-ready divider/column mapping to disk."""
    payload = build_ocr_ready_map(only_verified=only_verified)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return output_path
