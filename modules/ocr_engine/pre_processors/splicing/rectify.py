from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .geometry import DividerLine, TransformMatrix


class Rectifier:
    """Handles image deskewing and homography warping based on divider lines."""

    @staticmethod
    def get_deskew_matrix(image_size: tuple[int, int], lines: list[DividerLine]) -> TransformMatrix:
        """Calculate a global rotation matrix based on the average divider angle."""
        if not lines:
            return TransformMatrix(matrix=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))

        avg_angle = sum(line.angle_deg for line in lines) / len(lines)
        w, h = image_size
        center = (w / 2, h / 2)

        # We want to rotate by -avg_angle to make lines vertical
        matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
        return TransformMatrix(matrix=matrix)

    @staticmethod
    def apply_transform(image: Image.Image, transform: TransformMatrix) -> Image.Image:
        """Apply a transformation matrix to a PIL Image."""
        # Convert PIL to OpenCV (numpy)
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        w, h = image.size

        if transform.matrix.shape == (3, 3):
            dest = cv2.warpPerspective(cv_img, transform.matrix, (w, h), flags=cv2.INTER_LANCZOS4)
        else:
            dest = cv2.warpAffine(cv_img, transform.matrix, (w, h), flags=cv2.INTER_LANCZOS4)

        return Image.fromarray(cv2.cvtColor(dest, cv2.COLOR_BGR2RGB))

    @staticmethod
    def get_homography_matrix(
        image_size: tuple[int, int], left_line: DividerLine, right_line: DividerLine
    ) -> TransformMatrix:
        """
        Calculate a homography matrix that makes the two dividers vertical and parallel.
        This is a more aggressive rectification than pure rotation.
        """
        _w, h = image_size
        y_top = 0.0
        y_bottom = float(h - 1)

        left_top_x = Rectifier._x_at_y(left_line, y_top)
        left_bottom_x = Rectifier._x_at_y(left_line, y_bottom)
        right_top_x = Rectifier._x_at_y(right_line, y_top)
        right_bottom_x = Rectifier._x_at_y(right_line, y_bottom)

        # Source points based on full-page intersections with divider lines.
        src_pts = np.float32(
            [
                [left_top_x, y_top],
                [right_top_x, y_top],
                [right_bottom_x, y_bottom],
                [left_bottom_x, y_bottom],
            ]
        )

        # Destination points enforce vertical, parallel dividers across full page height.
        avg_left_x = (left_top_x + left_bottom_x) / 2.0
        avg_right_x = (right_top_x + right_bottom_x) / 2.0

        dst_pts = np.float32(
            [
                [avg_left_x, y_top],
                [avg_right_x, y_top],
                [avg_right_x, y_bottom],
                [avg_left_x, y_bottom],
            ]
        )

        matrix = cv2.findHomography(src_pts, dst_pts)[0]
        if matrix is None:
            raise RuntimeError("Failed to compute homography from divider lines.")
        return TransformMatrix(matrix=matrix)

    @staticmethod
    def _x_at_y(line: DividerLine, y_value: float) -> float:
        """Return line-intersection x-coordinate at a given y."""
        x1, y1 = line.top.x, line.top.y
        x2, y2 = line.bottom.x, line.bottom.y
        if y2 == y1:
            return float(x1)
        ratio = (y_value - y1) / (y2 - y1)
        return float(x1 + ratio * (x2 - x1))
