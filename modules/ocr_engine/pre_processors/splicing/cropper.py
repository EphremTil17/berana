from __future__ import annotations

import numpy as np
from PIL import Image

from .geometry import DividerLine


class Cropper:
    """Handles the surgical extraction of language columns from rectified images."""

    @staticmethod
    def get_column_strips(
        image: Image.Image,
        left_line: DividerLine,
        right_line: DividerLine,
        margin_px: int = 10,
    ) -> tuple[dict[str, Image.Image], dict[str, tuple[int, int]]]:
        """
        Slices the rectified image into 3 strips: Geez, Amharic, English.
        Assumes the image has already been deskewed such that lines are mostly vertical.
        """
        w, h = image.size

        # We use the average X positions of the dividers as the crop boundaries
        # after rectification.
        lx = (left_line.top.x + left_line.bottom.x) / 2
        rx = (right_line.top.x + right_line.bottom.x) / 2

        # Clamp/normalize boundaries to avoid out-of-bounds crops and negative offsets.
        left_cut = max(0, min(w, round(lx)))
        right_cut = max(0, min(w, round(rx)))
        if right_cut < left_cut:
            left_cut, right_cut = right_cut, left_cut

        geez_x2 = max(0, min(w, left_cut + margin_px))
        amharic_x1 = max(0, min(w, left_cut - margin_px))
        amharic_x2 = max(0, min(w, right_cut + margin_px))
        english_x1 = max(0, min(w, right_cut - margin_px))

        if amharic_x2 < amharic_x1:
            amharic_x1, amharic_x2 = amharic_x2, amharic_x1

        # Geez (Left)
        geez_strip = image.crop((0, 0, geez_x2, h))

        # Amharic (Middle)
        amharic_strip = image.crop((amharic_x1, 0, amharic_x2, h))

        # English (Right)
        english_strip = image.crop((english_x1, 0, w, h))

        strips = {
            "geez": geez_strip,
            "amharic": amharic_strip,
            "english": english_strip,
        }

        offsets = {
            "geez": (0, 0),
            "amharic": (amharic_x1, 0),
            "english": (english_x1, 0),
        }

        return strips, offsets

    @staticmethod
    def get_mask_polygons(
        image_size: tuple[int, int],
        left_line: DividerLine,
        right_line: DividerLine,
    ) -> dict[str, np.ndarray]:
        """
        Calculates the 4-point polygons for each column.
        Useful for more precise masking if simple cropping is too blunt.
        """
        w, h = image_size

        # Geez: [TopLeft, DividerTop, DividerBottom, BottomLeft]
        geez_poly = np.array(
            [
                [0, 0],
                [left_line.top.x, left_line.top.y],
                [left_line.bottom.x, left_line.bottom.y],
                [0, h],
            ],
            dtype=np.int32,
        )

        # Amharic: [DividerLeftTop, DividerRightTop, DividerRightBottom, DividerLeftBottom]
        amharic_poly = np.array(
            [
                [left_line.top.x, left_line.top.y],
                [right_line.top.x, right_line.top.y],
                [right_line.bottom.x, right_line.bottom.y],
                [left_line.bottom.x, left_line.bottom.y],
            ],
            dtype=np.int32,
        )

        # English: [DividerRightTop, TopRight, BottomRight, DividerRightBottom]
        english_poly = np.array(
            [
                [right_line.top.x, right_line.top.y],
                [w, 0],
                [w, h],
                [right_line.bottom.x, right_line.bottom.y],
            ],
            dtype=np.int32,
        )

        return {
            "geez": geez_poly,
            "amharic": amharic_poly,
            "english": english_poly,
        }
