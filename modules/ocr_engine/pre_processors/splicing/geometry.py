from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Point2D:
    """A standard 2D coordinate."""

    x: float
    y: float

    def to_tuple(self) -> tuple[float, float]:
        """Return this point as an `(x, y)` tuple."""
        return (self.x, self.y)


@dataclass(frozen=True)
class DividerLine:
    """A verified vertical-ish divider line between columns."""

    top: Point2D
    bottom: Point2D

    @property
    def slope(self) -> float:
        """Calculate the dy/dx slope of the line."""
        dx = self.bottom.x - self.top.x
        dy = self.bottom.y - self.top.y
        return dy / dx if dx != 0 else float("inf")

    @property
    def angle_deg(self) -> float:
        """Calculate the angle of the line relative to the vertical axis in degrees."""
        dx = self.bottom.x - self.top.x
        dy = self.bottom.y - self.top.y
        # We want the angle from vertical, so atan2(dx, dy)
        return np.degrees(np.arctan2(dx, dy))


@dataclass
class TransformMatrix:
    """Wrapper for an OpenCV transformation matrix with inversion capability."""

    matrix: np.ndarray  # 3x3 for homography, 2x3 for affine/rotation

    def forward(self, point: Point2D) -> Point2D:
        """Transform a point using the internal matrix."""
        pts = np.array([[point.x, point.y]], dtype=np.float32).reshape(-1, 1, 2)
        if self.matrix.shape == (3, 3):
            dst = cv2.perspectiveTransform(pts, self.matrix)
        else:
            # Affine
            dst = cv2.transform(pts, self.matrix)

        return Point2D(x=float(dst[0][0][0]), y=float(dst[0][0][1]))

    def inverse(self, point: Point2D) -> Point2D:
        """Transform a point back using the inverted matrix."""
        if self.matrix.shape == (3, 3):
            inv_matrix = np.linalg.inv(self.matrix)
            pts = np.array([[point.x, point.y]], dtype=np.float32).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, inv_matrix)
        else:
            inv_matrix = cv2.invertAffineTransform(self.matrix)
            pts = np.array([[point.x, point.y]], dtype=np.float32).reshape(-1, 1, 2)
            dst = cv2.transform(pts, inv_matrix)

        return Point2D(x=float(dst[0][0][0]), y=float(dst[0][0][1]))
