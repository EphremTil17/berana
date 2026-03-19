import tempfile
from pathlib import Path

import pytest

from modules.ocr_benchmark.dataset import read_manifest, validate_split_leakage, write_manifest
from schemas.ocr_benchmark import ColumnKey, DatasetSplit, LangPrompt, LineManifestRow


def mock_row(line_id: str, split: DatasetSplit, page_id: str = "page_01") -> LineManifestRow:
    return LineManifestRow(
        line_id=line_id,
        doc_stem="doc_001",
        page_id=page_id,
        column_key=ColumnKey.GEEZ,
        lang_prompt=LangPrompt.GEEZ,
        image_path=f"output/images/{line_id}.png",
        split=split,
        source_run_dir="/run01",
    )


def test_manifest_io():
    rows = [mock_row("line_01", DatasetSplit.TRAIN), mock_row("line_02", DatasetSplit.HOLDOUT)]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "manifest.jsonl"
        write_manifest(rows, path)

        loaded = read_manifest(path)
        assert len(loaded) == 2
        assert loaded[0].line_id == "line_01"
        assert loaded[1].split == DatasetSplit.HOLDOUT


def test_hard_fail_on_line_leakage():
    rows = [mock_row("line_01", DatasetSplit.TRAIN), mock_row("line_01", DatasetSplit.HOLDOUT)]
    with pytest.raises(ValueError, match="CRITICAL LEAKAGE"):
        validate_split_leakage(rows)


def test_page_leakage_strict_mode():
    rows = [
        mock_row("line_01", DatasetSplit.TRAIN, page_id="page_100"),
        mock_row("line_02", DatasetSplit.HOLDOUT, page_id="page_100"),
    ]
    with pytest.raises(ValueError, match="PAGE LEAKAGE"):
        validate_split_leakage(rows, strict_page_isolation=True)


def test_page_leakage_loose_mode(caplog):
    rows = [
        mock_row("line_01", DatasetSplit.TRAIN, page_id="page_100"),
        mock_row("line_02", DatasetSplit.HOLDOUT, page_id="page_100"),
    ]
    # Should not raise
    stats = validate_split_leakage(rows, strict_page_isolation=False)
    assert stats["train_lines"] == 1
    assert "PAGE LEAKAGE" in caplog.text


def test_low_train_count_warning(caplog):
    rows = [
        mock_row(f"line_{i:03d}", DatasetSplit.TRAIN, page_id=f"page_train_{i}") for i in range(179)
    ]
    # Add valid holdout
    rows.append(mock_row("holdout_01", DatasetSplit.HOLDOUT, page_id="page_holdout"))

    validate_split_leakage(rows, strict_page_isolation=True)
    assert "STRICT MODE DROPPED USABLE LINES" in caplog.text
