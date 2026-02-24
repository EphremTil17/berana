from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

from schemas.ocr_models import BoundingBox, ColumnBlock, PageLayout, TextLine
from utils.logger import get_logger

if TYPE_CHECKING:
    from surya.detection import DetectionPredictor
    from surya.layout import LayoutPredictor

logger = get_logger("LayoutParser")


def load_predictors(
    include_layout: bool = True,
):
    """Instantiate and return the Surya detection predictor and optional layout predictor."""
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.layout import LayoutPredictor

    logger.info("Loading DetectionPredictor (~1.1GB model)...")
    det_predictor = DetectionPredictor()
    logger.info("DetectionPredictor loaded. ✅")

    if not include_layout:
        return det_predictor, None

    # Layout requires a dedicated FoundationPredictor with the specific Layout checkpoint.
    from surya.settings import settings

    logger.info("Loading LayoutPredictor (~1.5GB model)...")
    layout_foundation = FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    layout_predictor = LayoutPredictor(layout_foundation)
    logger.info("LayoutPredictor loaded. ✅")

    return det_predictor, layout_predictor


def extract_layout_and_boxes(
    image: Image.Image,
    page_number: int,
    det_predictor: DetectionPredictor,
    layout_predictor: LayoutPredictor,
) -> PageLayout:
    """Execute Surya's text detection and layout analysis on a single high-res image.

    This function explicitly does NOT run text recognition (reading characters).
    It only finds where the text is, and groups them into our strict Pydantic columns.

    Args:
        image: The physical PIL Image.
        page_number: The integer representing the source PDF page number.
        det_predictor: The instantiated Surya DetectionPredictor.
        layout_predictor: The instantiated Surya LayoutPredictor.

    Returns:
        PageLayout: The Pydantic representation of the detected text boundaries.
    """
    img_width, img_height = image.size
    logger.debug(f"Running layout analysis on Page {page_number} ({img_width}x{img_height})")

    # Step 1: Detect all text lines on the page (creates raw bounding boxes)
    line_predictions = det_predictor([image])[0]

    # Step 2: Detect structural layout (Headers, Footer, Columns, etc.)
    layout_predictions = layout_predictor([image])[0]

    # Step 3: Map Surya's dynamic outputs into our strict Pydantic "ColumnBlock" geometry
    columns: list[ColumnBlock] = []

    for layout_box in layout_predictions.bboxes:
        raw_coords = layout_box.bbox

        # We only care about layout blocks identified as 'Text' or 'List' by Surya.
        if layout_box.label not in ["Text", "List"]:
            continue

        master_bbox = BoundingBox(coordinates=raw_coords)

        # Find all specific line-level bounding boxes that fit INSIDE this master layout block
        lines: list[TextLine] = []
        for line_pred in line_predictions.bboxes:
            line_coords = line_pred.bbox

            # Simple intersection check: If the center of the line box is inside the layout box
            center_x = (line_coords[0] + line_coords[2]) / 2
            center_y = (line_coords[1] + line_coords[3]) / 2

            if (raw_coords[0] <= center_x <= raw_coords[2]) and (
                raw_coords[1] <= center_y <= raw_coords[3]
            ):
                lines.append(
                    TextLine(
                        text="",
                        bbox=BoundingBox(coordinates=line_coords),
                        confidence=None,
                    )
                )

        if lines:
            block = ColumnBlock(lines=lines, bbox=master_bbox, page_width=img_width)
            columns.append(block)

    logger.info(f"Page {page_number}: Mapped {len(columns)} geographic text columns")

    return PageLayout(
        page_number=page_number,
        image_width=img_width,
        image_height=img_height,
        columns=columns,
        fallback_triggered=False,
        uncertainty_score=0.0,
    )
