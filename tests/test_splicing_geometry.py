import cv2
import numpy as np
from PIL import Image

from modules.ocr_engine.pre_processors.splicing.cropper import Cropper
from modules.ocr_engine.pre_processors.splicing.geometry import (
    DividerLine,
    Point2D,
    TransformMatrix,
)


def test_point_to_tuple():
    """Point2D should serialize as an `(x, y)` tuple."""
    p = Point2D(10.5, 20.7)
    assert p.to_tuple() == (10.5, 20.7)


def test_divider_angle():
    """Divider angle should be measured relative to vertical axis."""
    # Vertical line
    line_v = DividerLine(Point2D(100, 0), Point2D(100, 1000))
    assert line_v.angle_deg == 0.0

    # Line tilted 45 degrees to the right
    # dx = 10, dy = 10 -> atan2(10, 10) = 45 deg
    line_45 = DividerLine(Point2D(0, 0), Point2D(10, 10))
    assert line_45.angle_deg == 45.0


def test_affine_roundtrip():
    """Affine forward/inverse mapping should round-trip with tiny error."""
    # Create a rotation matrix (rotate 10 degrees around 0,0)
    matrix = cv2.getRotationMatrix2D((0, 0), 10, 1.0)
    transform = TransformMatrix(matrix=matrix)

    p_orig = Point2D(100, 100)
    p_forward = transform.forward(p_orig)
    p_back = transform.inverse(p_forward)

    # Assert within 1px Tolerance (due to float precision)
    assert abs(p_orig.x - p_back.x) < 0.1
    assert abs(p_orig.y - p_back.y) < 0.1


def test_homography_roundtrip():
    """Homography forward/inverse mapping should round-trip with tiny error."""
    # Create a simple perspective warp
    src = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
    dst = np.float32([[10, 10], [90, 0], [110, 110], [0, 90]])
    matrix = cv2.findHomography(src, dst)[0]
    transform = TransformMatrix(matrix=matrix)

    p_orig = Point2D(50, 50)
    p_forward = transform.forward(p_orig)
    p_back = transform.inverse(p_forward)

    assert abs(p_orig.x - p_back.x) < 0.1
    assert abs(p_orig.y - p_back.y) < 0.1


def test_cropper_clamps_bounds_and_offsets():
    """Cropper should clamp negative/overflow cut positions to image bounds."""
    image = Image.new("RGB", (100, 60), "white")
    left = DividerLine(top=Point2D(-20, 0), bottom=Point2D(-20, 59))
    right = DividerLine(top=Point2D(120, 0), bottom=Point2D(120, 59))

    strips, offsets = Cropper.get_column_strips(image, left, right, margin_px=10)

    assert offsets["geez"] == (0, 0)
    assert offsets["amharic"][0] >= 0
    assert offsets["english"][0] >= 0
    assert strips["geez"].size[0] <= 100
    assert strips["amharic"].size[0] <= 100
    assert strips["english"].size[0] <= 100
