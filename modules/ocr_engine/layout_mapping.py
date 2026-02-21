from schemas.ocr_models import BoundingBox, ColumnBlock, PageLayout, TextLine


def initialize_page_layout(
    page_number: int,
    img_width: int,
    img_height: int,
    fallback_triggered: bool,
    uncertainty: float,
    warnings: list[str],
) -> PageLayout:
    """Create the base PageLayout container with metadata."""
    return PageLayout(
        page_number=page_number,
        image_width=img_width,
        image_height=img_height,
        fallback_triggered=fallback_triggered,
        uncertainty_score=uncertainty,
        warnings=warnings,
    )


def map_boxes_to_columns(
    surya_bboxes: list[list[float]],
    slice_lines: list[int],
    img_width: int,
    img_height: int,
    safety_padding: int = 10,
) -> list[ColumnBlock]:
    """Assign detected text boxes to column blocks using divider lines."""
    columns: list[ColumnBlock] = []
    extents = [0, *slice_lines, img_width]

    for index in range(len(extents) - 1):
        x_start = extents[index]
        x_end = extents[index + 1]
        col_bbox = BoundingBox(coordinates=[x_start, 0, x_end, img_height])
        column = ColumnBlock(bbox=col_bbox, page_width=img_width)

        for box in surya_bboxes:
            center_x = (box[0] + box[2]) / 2
            if (x_start - safety_padding) <= center_x <= (x_end + safety_padding):
                column.lines.append(
                    TextLine(text="", bbox=BoundingBox(coordinates=box), confidence=None)
                )

        if column.lines:
            column.lines.sort(key=lambda line_item: line_item.bbox.y1)
            columns.append(column)

    return columns
