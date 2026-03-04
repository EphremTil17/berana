import pytest

from modules.ocr_training.normalize import normalize_source_type, normalize_text
from modules.ocr_training.schemas import NormalizedType, SourceRepo


@pytest.mark.parametrize(
    ("raw_type", "expected"),
    [
        ("typed", NormalizedType.TYPED),
        ("synthetic", NormalizedType.SYNTHETIC),
        ("handwritten", NormalizedType.HANDWRITTEN),
        ("hdd", NormalizedType.HANDWRITTEN),
        ("hdd_18", NormalizedType.HANDWRITTEN),
        ("hdd_rand", NormalizedType.HANDWRITTEN),
    ],
)
def test_normalize_source_type(raw_type: str, expected: NormalizedType):
    assert normalize_source_type(raw_type, SourceRepo.FIDEL_DATASET) == expected


def test_normalize_source_type_synthetic_repo_forces_synthetic():
    assert normalize_source_type("anything", SourceRepo.FIDEL_SYNTHETIC) == NormalizedType.SYNTHETIC


def test_normalize_source_type_unsupported():
    with pytest.raises(ValueError, match="Unsupported source type"):
        normalize_source_type("unknown", SourceRepo.FIDEL_DATASET)


def test_normalize_text_collapses_whitespace_and_trims():
    assert normalize_text("  a\n\t b   c  ") == "a b c"
