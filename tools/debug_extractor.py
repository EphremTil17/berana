import random
import zipfile
from pathlib import Path

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger("HitlExtractor")


def calculate_optimal_slice_vector(polygon_points: np.ndarray) -> tuple[float, float, float, float]:
    """
    Computes the optimal vector (Line of Best Fit) from a Label Studio polygon.

    This calculates a mathematical line that naturally matches the tilt of
    the drawn polygon instead of forcing an unnatural vertical 90-degree slice.
    Returns: (vx, vy, x0, y0)
    """
    # cv2.fitLine returns [vx, vy, x, y] representing a normalized vector (vx, vy)
    # and a point on the line (x, y)
    [vx, vy, x0, y0] = cv2.fitLine(polygon_points, cv2.DIST_L2, 0, 0.01, 0.01)

    return float(vx), float(vy), float(x0), float(y0)


def draw_slice_vector(
    img: np.ndarray, vector: tuple, color: tuple = (0, 0, 255), thickness: int = 4
):
    """Draws a vector line entirely across the given image."""
    vx, vy, x0, y0 = vector
    img_h, _img_w = img.shape[:2]

    # Calculate extremes. Line equation: (x - x0)/vx = (y - y0)/vy
    # We want points at y = 0 and y = img_h
    if vy != 0:
        x_top = int(((0 - y0) * (vx / vy)) + x0)
        x_bottom = int(((img_h - y0) * (vx / vy)) + x0)

        cv2.line(img, (x_top, 0), (x_bottom, img_h), color, thickness)
    else:
        # Fallback for perfectly horizontal glitch (unlikely)
        cv2.line(img, (int(x0), 0), (int(x0), img_h), color, thickness)


def debug_hitl_extraction(
    zip_path: Path, source_dirs: list[Path], output_dir: Path, num_samples: int = 3
):
    """
    Extracts random pages from the verified Label Studio ZIP, calculates their
    optimal center of mass, and generates visual proofs.
    """
    logger.info(f"Opening Label Studio export: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as z:
        label_files = _collect_label_files(z)
        if not label_files:
            logger.error("No label files found in ZIP.")
            return

        random.seed(42)  # Deterministic for reproducible debugging
        sample_files = random.sample(label_files, min(num_samples, len(label_files)))
        output_dir.mkdir(parents=True, exist_ok=True)

        for label_filename in sample_files:
            _process_label_file(z, label_filename, source_dirs, output_dir)


def _collect_label_files(zip_file: zipfile.ZipFile) -> list[str]:
    return [
        filename
        for filename in zip_file.namelist()
        if filename.startswith("labels/") and filename.endswith(".txt")
    ]


def _resolve_source_image(base_name: str, source_dirs: list[Path]) -> Path | None:
    img_filename = f"{base_name}.jpg"
    for src_dir in source_dirs:
        candidate = src_dir / img_filename
        if candidate.exists():
            return candidate
    return None


def _process_label_file(
    zip_file: zipfile.ZipFile, label_filename: str, source_dirs: list[Path], output_dir: Path
) -> None:
    logger.info(f"Processing {label_filename}...")
    base_name = Path(label_filename).stem
    img_path = _resolve_source_image(base_name, source_dirs)

    if not img_path:
        logger.warning(f"Could not find source image {base_name}.jpg in any source dirs.")
        return

    img = cv2.imread(str(img_path))
    if img is None:
        logger.error(f"Failed to read image at {img_path}")
        return

    img_h, img_w = img.shape[:2]
    raw_text = zip_file.read(label_filename).decode("utf-8")
    lines = raw_text.strip().split("\n")
    overlay = img.copy()
    optimal_slice_vectors = _extract_vectors_and_overlay(lines, overlay, img_w, img_h)

    alpha = 0.3
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    for vector in optimal_slice_vectors:
        draw_slice_vector(img, vector)

    out_file = output_dir / f"{base_name}_debug.jpg"
    cv2.imwrite(str(out_file), img)
    logger.info(f"✅ Generated visual proof for {base_name}: {out_file}")


def _extract_vectors_and_overlay(
    lines: list[str], overlay: np.ndarray, img_w: int, img_h: int
) -> list[tuple[float, float, float, float]]:
    vectors: list[tuple[float, float, float, float]] = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        class_id = int(parts[0])
        coords = [float(value) for value in parts[1:]]
        points = [
            [int(coords[i] * img_w), int(coords[i + 1] * img_h)] for i in range(0, len(coords), 2)
        ]
        pts_array = np.array(points, np.int32).reshape((-1, 1, 2))

        color = (255, 0, 0) if class_id == 0 else (0, 255, 0)
        cv2.fillPoly(overlay, [pts_array], color)
        vectors.append(calculate_optimal_slice_vector(pts_array))
    return vectors


if __name__ == "__main__":
    zip_path = Path("data/layout_dataset/Berana.Annotations.Final.zip")
    source_dirs = [Path("output/visuals/layout_auto"), Path("output/visuals/layout_training")]
    output_dir = Path("output/pipeline_debug/hitl_slices")

    debug_hitl_extraction(zip_path, source_dirs, output_dir, num_samples=25)
