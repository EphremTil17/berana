from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from modules.ocr_engine.layout.yolo_engine import LayoutSegmentationEngine, find_yolo_slices
from utils.logger import get_logger

logger = get_logger("ColumnEngine")

# Lazy-loaded engine to prevent multiple VRAM allocations
_yolo_engine = None


def get_yolo_engine():
    """Return a singleton YOLO layout engine to avoid repeated VRAM allocations."""
    global _yolo_engine
    if _yolo_engine is None:
        _yolo_engine = LayoutSegmentationEngine()
    return _yolo_engine


def find_column_canyons(
    img: Image.Image,
    surya_bboxes: list[list[float]],
    expected_columns: int = 3,
    alignment_lines: list[list[int]] | None = None,
) -> tuple[list[int], bool, float, list[str], Image.Image]:
    """
    Vision-Based Column Orchestrator:
    Utilizes fine-tuned YOLOv8-Segmentation to find physical dividers.
    """
    img_width, _ = img.size
    engine = get_yolo_engine()

    if engine.model is None:
        logger.warning("No YOLO weights found. Falling back to naive math.")
        slice_lines = [(img_width // expected_columns) * i for i in range(1, expected_columns)]
        return slice_lines, True, 1.0, ["NAIVE FALLBACK: NO MODEL WEIGHTS."], img

    # Run AI Inference
    dividers = engine.predict_dividers(img, conf=0.2)
    slice_lines = find_yolo_slices(img, dividers)

    warnings = []
    fallback_triggered = False

    # Validation: If AI found too few/many dividers, log a warning
    if len(slice_lines) != (expected_columns - 1):
        warnings.append(f"AI found {len(slice_lines)} dividers, expected {expected_columns - 1}.")
        if not slice_lines:
            logger.error("AI found zero dividers. Triggering naive math fallback.")
            slice_lines = [(img_width // expected_columns) * i for i in range(1, expected_columns)]
            fallback_triggered = True

    return slice_lines, fallback_triggered, 0.0, warnings, img


def generate_poc_debug_visuals(
    img: Image.Image,
    surya_bboxes: list[list[float]],
    slice_lines: list[int],
    fallback_triggered: bool,
    page_num: int,
    output_base_dir: Path,
    alignment_lines: list[list[int]] | None = None,
):
    """Draws visual proofs. Red = Surya Blocks, Blue = Slices."""
    debug_dir = output_base_dir / "pipeline_debug"
    overlay_dir = debug_dir / "visual_overlays"
    columns_dir = debug_dir / "columns" / f"page_{page_num:03d}"

    overlay_dir.mkdir(parents=True, exist_ok=True)
    columns_dir.mkdir(parents=True, exist_ok=True)

    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_h, _img_w = cv_img.shape[:2]

    # Draw simple slices for verification during transition
    line_color = (255, 0, 0)
    for line_x in slice_lines:
        cv2.line(cv_img, (line_x, 0), (line_x, img_h), line_color, 4)

    overlay_path = overlay_dir / f"page_{page_num:03d}_annotated.jpg"
    cv2.imwrite(str(overlay_path), cv_img)
