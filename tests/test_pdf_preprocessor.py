from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from modules.ocr_engine.pre_processors.pdf_to_image import yield_pdf_pages


@pytest.fixture
def mock_pdf_path(tmp_path):
    """Creates a fake empty PDF file to bypass path.exists() check."""
    fake_pdf = tmp_path / "test_doc.pdf"
    fake_pdf.touch()
    return fake_pdf


@patch("pdf2image.pdf2image.pdfinfo_from_path")
@patch("modules.ocr_engine.pre_processors.pdf_to_image.convert_from_path")
def test_generator_chunks_correctly(mock_convert, mock_pdfinfo, mock_pdf_path):
    """Simulates parsing an 8-page PDF with a chunk limit of 3.

    Ensures convert_from_path is called 3 times (pages 1-3, 4-6, 7-8).
    """
    # Mock total pages = 8
    mock_pdfinfo.return_value = {"Pages": "8"}

    # Mock the return values for convert_from_path
    # Chunk 1 (1-3) -> 3 images
    # Chunk 2 (4-6) -> 3 images
    # Chunk 3 (7-8) -> 2 images
    mock_image1 = MagicMock(spec=Image.Image)
    mock_image2 = MagicMock(spec=Image.Image)

    mock_convert.side_effect = [
        [mock_image1, mock_image1, mock_image1],
        [mock_image1, mock_image1, mock_image1],
        [mock_image2, mock_image2],
    ]

    # Execute generator
    generator = yield_pdf_pages(pdf_path=mock_pdf_path, chunk_size=3, dpi=300)

    results = list(generator)

    # We should have yielded 8 items exactly
    assert len(results) == 8

    # Assert output structure is (PageNum, Image)
    assert results[0][0] == 1  # Page 1
    assert results[7][0] == 8  # Page 8
    assert results[0][1] == mock_image1
    assert results[7][1] == mock_image2

    # Verify the chunking logic was hit perfectly
    assert mock_convert.call_count == 3

    # Assert parameters for the last chunk call (pages 7 to 8)
    last_call_args = mock_convert.call_args_list[-1]
    assert last_call_args.kwargs["first_page"] == 7
    assert last_call_args.kwargs["last_page"] == 8
