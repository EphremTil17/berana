import sqlite3
from pathlib import Path

import pytest

from modules.ocr_engine.pre_processors.splicing.engine import SplicingEngine


def _create_sqlite(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE page_lines (
            page_id TEXT PRIMARY KEY,
            left_json TEXT,
            right_json TEXT,
            verified INTEGER DEFAULT 1
        )
        """
    )
    conn.commit()
    conn.close()


def test_source_prefers_sqlite_when_both_exist(tmp_path: Path):
    """Engine should prefer SQLite source when both source files exist."""
    db_path = tmp_path / "hitl_line_editor.sqlite3"
    json_path = tmp_path / "ocr_column_map.json"
    _create_sqlite(db_path)
    json_path.write_text(
        '{"page_001":{"dividers":{"left":[1,2,3,4],"right":[5,6,7,8]}}}',
        encoding="utf-8",
    )

    engine = SplicingEngine(db_path=db_path, json_map_path=json_path)

    assert engine.source_kind == "sqlite"
    assert engine.source_path == db_path


def test_source_uses_json_map_when_sqlite_missing(tmp_path: Path):
    """Engine should use strict OCR map JSON when SQLite source is missing."""
    db_path = tmp_path / "missing.sqlite3"
    json_path = tmp_path / "ocr_column_map.json"
    json_path.write_text(
        '{"page_001":{"dividers":{"left":[1,2,3,4],"right":[5,6,7,8]}}}',
        encoding="utf-8",
    )

    engine = SplicingEngine(db_path=db_path, json_map_path=json_path)

    assert engine.source_kind == "json_map"
    left, right = engine.get_dividers_for_page("page_001")
    assert left is not None
    assert right is not None


def test_json_map_rejects_unsupported_schema(tmp_path: Path):
    """Engine should reject non-ocr-column-map JSON formats."""
    db_path = tmp_path / "missing.sqlite3"
    json_path = tmp_path / "bad.json"
    json_path.write_text('{"page_001":{"left":[1,2,3,4]}}', encoding="utf-8")

    with pytest.raises(ValueError):
        SplicingEngine(db_path=db_path, json_map_path=json_path)
