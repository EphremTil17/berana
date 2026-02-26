import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tools.hitl_line_editor_app.paths import DB_FILE


@contextmanager
def db_connection(db_file: Path = DB_FILE) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with row access enabled."""
    db_file.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_file)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def ensure_schema(db_file: Path = DB_FILE) -> None:
    """Create required tables for line editor state if missing."""
    with db_connection(db_file) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS page_lines (
                page_id TEXT PRIMARY KEY,
                left_json TEXT,
                right_json TEXT,
                verified INTEGER NOT NULL DEFAULT 0,
                img_w INTEGER NOT NULL,
                img_h INTEGER NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS provenance_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                source_stage TEXT NOT NULL,
                doc_stem TEXT NOT NULL,
                source_run_dir TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def set_provenance(db_file: Path, source_stage: str, doc_stem: str, source_run_dir: str) -> None:
    """Upsert the single allowed provenance record for this database."""
    ensure_schema(db_file)
    with db_connection(db_file) as conn:
        conn.execute(
            """
            INSERT INTO provenance_metadata (id, source_stage, doc_stem, source_run_dir, updated_at)
            VALUES (1, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                source_stage=excluded.source_stage,
                doc_stem=excluded.doc_stem,
                source_run_dir=excluded.source_run_dir,
                updated_at=CURRENT_TIMESTAMP
            """,
            (source_stage, doc_stem, source_run_dir),
        )


def get_provenance(db_file: Path) -> dict[str, str] | None:
    """Retrieve the provenance record, returning None if unmigrated/missing."""
    if not db_file.exists():
        return None
    ensure_schema(db_file)
    with db_connection(db_file) as conn:
        row = conn.execute("SELECT * FROM provenance_metadata WHERE id = 1").fetchone()
        if not row:
            return None
        return dict(row)


def list_page_ids(db_file: Path) -> list[str]:
    """Return all stored page identifiers in deterministic order."""
    ensure_schema(db_file)
    with db_connection(db_file) as conn:
        rows = conn.execute("SELECT page_id FROM page_lines ORDER BY page_id").fetchall()
    return [str(row["page_id"]) for row in rows]
