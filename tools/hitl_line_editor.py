import sys
from pathlib import Path

try:
    from tools.hitl_line_editor_app.app import app, run
except ModuleNotFoundError:
    # Allow running from inside ./tools without requiring PYTHONPATH=.
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from tools.hitl_line_editor_app.app import app, run

__all__ = ["app"]

if __name__ == "__main__":
    run()
