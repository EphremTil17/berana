import pytest
from pydantic import ValidationError

from schemas.ocr_models import BoundingBox, ColumnBlock, LanguageTag, PageLayout, TextLine


def test_bounding_box_init():
    """Verify BoundingBox properties are correctly mapped from list."""
    bbox = BoundingBox(coordinates=[10.0, 20.0, 100.0, 200.0])
    assert bbox.x1 == 10.0
    assert bbox.y1 == 20.0
    assert bbox.x2 == 100.0
    assert bbox.y2 == 200.0
    assert bbox.center_x == 55.0


def test_bounding_box_invalid_length():
    """Verify BoundingBox rejects coordinates that do not have 4 elements."""
    with pytest.raises(ValidationError):
        # Only 3 coordinates
        BoundingBox(coordinates=[10.0, 20.0, 100.0])


def test_text_line_init():
    """Verify TextLine correctly initializes with a BoundingBox and confidence."""
    bbox = BoundingBox(coordinates=[10.0, 20.0, 100.0, 30.0])
    line = TextLine(text="Hello", bbox=bbox, confidence=0.98)
    assert line.text == "Hello"
    assert line.confidence == 0.98
    assert line.bbox.x1 == 10.0


def test_geographic_language_clustering():
    """Testing that columns automatically identify their language based on X position."""
    page_width = 1000.0  # 1000px wide for easy math

    # Left Column (Center X < 333)
    left_bbox = BoundingBox(coordinates=[0.0, 0.0, 300.0, 1000.0])
    assert left_bbox.center_x == 150.0  # (0 + 300) / 2
    left_column = ColumnBlock(bbox=left_bbox, page_width=page_width)
    assert left_column.language == LanguageTag.GEEZ

    # Middle Column (333 <= Center X < 666)
    middle_bbox = BoundingBox(coordinates=[334.0, 0.0, 600.0, 1000.0])
    assert middle_bbox.center_x == 467.0
    middle_column = ColumnBlock(bbox=middle_bbox, page_width=page_width)
    assert middle_column.language == LanguageTag.AMHARIC

    # Right Column (Center X >= 666)
    right_bbox = BoundingBox(coordinates=[667.0, 0.0, 1000.0, 1000.0])
    assert right_bbox.center_x == 833.5
    right_column = ColumnBlock(bbox=right_bbox, page_width=page_width)
    assert right_column.language == LanguageTag.ENGLISH


def test_page_layout_container():
    """Verify PageLayout correctly stores page metadata."""
    page = PageLayout(
        page_number=1,
        image_width=2000.0,
        image_height=3000.0,
        columns=[],
    )
    assert page.page_number == 1
    assert page.image_width == 2000.0
