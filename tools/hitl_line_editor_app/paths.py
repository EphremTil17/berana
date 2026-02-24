from pathlib import Path

# Resolve project root (two levels up from tools/hitl_line_editor_app/paths.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ZIP_DIR = PROJECT_ROOT / "input/layout_dataset"
SOURCE_DIRS = [
    PROJECT_ROOT / "output/layout_inference",
    PROJECT_ROOT / "output/layout_prep",
]
DB_FILE = ZIP_DIR / "hitl_line_editor.sqlite3"
LEGACY_STATE_FILE = ZIP_DIR / "verified_lines_state.json"
TEMPLATE_FILE = Path(__file__).parent / "templates" / "hitl_line_editor.html"
