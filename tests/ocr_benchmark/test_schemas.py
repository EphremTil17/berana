import pytest
from pydantic import ValidationError

from schemas.ocr_benchmark import ColumnKey, DatasetSplit, LangPrompt, LineManifestRow


def test_line_manifest_row_valid():
    row = LineManifestRow(
        line_id="line_001",
        doc_stem="doc_001",
        page_id="page_01",
        column_key=ColumnKey.GEEZ,
        lang_prompt=LangPrompt.GEEZ,
        image_path="output/images/line_001.png",
        split=DatasetSplit.TRAIN,
        gt_text="ሀሁሂሃሄህሆ",
        source_run_dir="/some/run/dir",
    )
    assert row.split == DatasetSplit.TRAIN
    assert row.quality_flag == "ok"
    assert row.schema_version == "1.0"
    abs_path = row.get_absolute_image_path()
    assert abs_path.is_absolute()
    assert str(abs_path).endswith("output/images/line_001.png")


def test_image_path_must_be_relative():
    with pytest.raises(ValidationError, match="MUST be project-relative"):
        LineManifestRow(
            line_id="line_001",
            doc_stem="doc_001",
            page_id="page_01",
            column_key=ColumnKey.GEEZ,
            lang_prompt=LangPrompt.GEEZ,
            image_path="/absolute/path/output/images/line_001.png",
            split=DatasetSplit.TRAIN,
            source_run_dir="/some/run/dir",
        )


def test_invalid_schema_version():
    with pytest.raises(ValidationError, match="String should match pattern"):
        LineManifestRow(
            schema_version="2.0",
            line_id="line_001",
            doc_stem="doc_001",
            page_id="page_01",
            column_key=ColumnKey.GEEZ,
            lang_prompt=LangPrompt.GEEZ,
            image_path="relative.png",
            split=DatasetSplit.TRAIN,
            source_run_dir="/some/run/dir",
        )
