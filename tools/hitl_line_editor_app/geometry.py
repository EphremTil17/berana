import cv2
import numpy as np


def calculate_optimal_slice_vector(polygon_points: np.ndarray) -> tuple[float, float, float, float]:
    """Fit a line-of-best-fit vector through polygon points."""
    [vx, vy, x0, y0] = cv2.fitLine(polygon_points, cv2.DIST_L2, 0, 0.01, 0.01)
    return float(vx), float(vy), float(x0), float(y0)


def extract_endpoints_from_vector(
    vector: tuple[float, float, float, float], img_h: int
) -> list[float]:
    """Convert a line vector into draggable endpoints within the visible page area."""
    vx, vy, x0, y0 = vector
    y_top = img_h * 0.1
    y_btm = img_h * 0.9

    if vy == 0:
        return [float(x0), float(y_top), float(x0), float(y_btm)]

    x_top = ((y_top - y0) * (vx / vy)) + x0
    x_btm = ((y_btm - y0) * (vx / vy)) + x0
    return [float(x_top), float(y_top), float(x_btm), float(y_btm)]
