import logging

from PIL import Image

# IMPORTANT: Surya uses PyTorch and OpenCV which handles heavy image matrices.
# We must import these carefully.
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection

from schemas.ocr_models import BoundingBox, ColumnBlock, PageLayout, TextLine

logger = logging.getLogger(__name__)


def extract_layout_and_boxes(
    image: Image.Image,
    page_number: int,
    det_model,
    det_processor,
    layout_model,
    layout_processor,
) -> PageLayout:
    """Executes Surya OCR's text detection and layout analysis on a single high-res image.

    This function explicitly does NOT run text recognition (reading characters).
    It only finds where the text is, and groups them into our strict Pydantic columns.

    Args:
        image: The physical PIL Image.
        page_number: The integer representing the source PDF page number.
        det_model: The loaded Surya text detection vision model.
        det_processor: The associated Surya image processor matching detection.
        layout_model: The loaded Surya layout vision model.
        layout_processor: The associated Surya image processor matching layout.

    Returns:
        PageLayout: The Pydantic representation of the detected text boundaries.
    """
    img_width, img_height = image.size
    logger.debug(f"Running layout analysis on Page {page_number} ({img_width}x{img_height})")

    # Step 1: Detect all text lines on the page (creates raw bounding boxes)
    line_predictions = batch_text_detection([image], det_model, det_processor)[0]

    # Step 2: Detect structural layout (Headers, Footer, Columns, etc.)
    # We pass the line predictions in to help the layout engine map blocks logically
    layout_predictions = batch_layout_detection(
        [image], layout_model, layout_processor, line_predictions=[line_predictions]
    )[0]

    # Step 3: Map Surya's dynamic outputs into our strict Pydantic "ColumnBlock" geometry
    columns: list[ColumnBlock] = []

    # Surya returns 'bboxes' formatted as [x1, y1, x2, y2, label]
    for layout_box in layout_predictions.bboxes:
        raw_coords = layout_box.bbox

        # We only care about layout blocks identified as 'Text' or 'List' by Surya.
        # We ignore 'Image', 'Table', 'Page-header', etc. to keep our Liturgical translation clean.
        if layout_box.label not in ["Text", "List"]:
            continue

        master_bbox = BoundingBox(coordinates=raw_coords)

        # Now find all the specific line-level bounding boxes that fit INSIDE this master layout block
        lines: list[TextLine] = []
        for line_pred in line_predictions.bboxes:
            line_coords = line_pred.bbox

            # Simple intersection check: If the center of the line box is inside the layout box
            center_x = (line_coords[0] + line_coords[2]) / 2
            center_y = (line_coords[1] + line_coords[3]) / 2

            if (raw_coords[0] <= center_x <= raw_coords[2]) and (
                raw_coords[1] <= center_y <= raw_coords[3]
            ):
                # Create an empty TextLine that we will fill during the extraction phase later
                lines.append(
                    TextLine(
                        text="",  # Empty for now, this is just layout padding
                        bbox=BoundingBox(coordinates=line_coords),
                        confidence=None,
                    )
                )

        if lines:
            # We instantiate ColumnBlock, which automatically runs @model_validator detect_language_geometrically
            block = ColumnBlock(lines=lines, bbox=master_bbox, page_width=img_width)
            columns.append(block)

    logger.info(f"Page {page_number}: Mapped {len(columns)} geographic text columns")

    return PageLayout(
        page_number=page_number, image_width=img_width, image_height=img_height, columns=columns
    )
