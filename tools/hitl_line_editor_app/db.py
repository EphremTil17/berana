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
