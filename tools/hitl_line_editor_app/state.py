import json
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tools.hitl_line_editor_app.db import db_connection, ensure_schema
from tools.hitl_line_editor_app.geometry import (
    calculate_optimal_slice_vector,
    extract_endpoints_from_vector,
)
from tools.hitl_line_editor_app.paths import DB_FILE, LEGACY_STATE_FILE, SOURCE_DIRS, ZIP_DIR
from utils.logger import get_logger

logger = get_logger("LineEditorState")


def _resolve_image_path(base_name: str, source_dirs: list[Path]) -> Path | None:
    img_filename = f"{base_name}.jpg"
    candidates: list[Path] = []
    for directory in source_dirs:
        for candidate in directory.rglob(img_filename):
            if candidate.exists():
                candidates.append(candidate)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _build_page_data(img: np.ndarray, raw_lines: list[str]) -> dict[str, Any]:
    img_h, img_w = img.shape[:2]
    page_data: dict[str, Any] = {
        "verified": False,
        "left": None,
        "right": None,
        "img_w": img_w,
        "img_h": img_h,
    }

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        class_id = int(parts[0])
        coords = [float(value) for value in parts[1:]]
        points = [
            [int(coords[i] * img_w), int(coords[i + 1] * img_h)] for i in range(0, len(coords), 2)
        ]
        pts_array = np.array(points, np.int32).reshape((-1, 1, 2))
        if len(pts_array) < 3:
            continue

        vector = calculate_optimal_slice_vector(pts_array)
        line_endpoints = extract_endpoints_from_vector(vector, img_h)
        if class_id == 0:
            page_data["left"] = line_endpoints
        else:
            page_data["right"] = line_endpoints

    return page_data


def _extract_zip_pages(zip_path: Path, source_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    page_map: dict[str, dict[str, Any]] = {}

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        for filename in zip_file.namelist():
            if not (filename.startswith("labels/") and filename.endswith(".txt")):
                continue

            base_name = Path(filename).stem
            img_path = _resolve_image_path(base_name, source_dirs)
            if not img_path:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            raw_text = zip_file.read(filename).decode("utf-8")
            lines = raw_text.strip().split("\n")
            page_map[base_name] = _build_page_data(img, lines)

    return page_map


def _db_is_seeded(db_file: Path) -> bool:
    with db_connection(db_file) as conn:
        row = conn.execute("SELECT COUNT(*) AS total FROM page_lines").fetchone()
    return bool(row and row["total"])


def _load_legacy_state(legacy_state_file: Path) -> dict[str, dict[str, Any]]:
    if not legacy_state_file.exists():
        return {}

    with legacy_state_file.open(encoding="utf-8") as file:
        raw_state = json.load(file)

    normalized: dict[str, dict[str, Any]] = {}
    for page_id, page in raw_state.items():
        normalized[page_id] = {
            "verified": bool(page.get("verified", False)),
            "left": page.get("left"),
            "right": page.get("right"),
            "img_w": int(page.get("img_w", 0)),
            "img_h": int(page.get("img_h", 0)),
        }
    return normalized


def _seed_db(pages: dict[str, dict[str, Any]], db_file: Path) -> None:
    with db_connection(db_file) as conn:
        for page_id, page in pages.items():
            conn.execute(
                """
                INSERT OR IGNORE INTO page_lines (page_id, left_json, right_json, verified, img_w, img_h)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    page_id,
                    json.dumps(page["left"]),
                    json.dumps(page["right"]),
                    int(page["verified"]),
                    int(page["img_w"]),
                    int(page["img_h"]),
                ),
            )


def initialize_state(
    zip_dir: Path = ZIP_DIR,
    source_dirs: list[Path] = SOURCE_DIRS,
    db_file: Path = DB_FILE,
    legacy_state_file: Path = LEGACY_STATE_FILE,
) -> None:
    """Initialize SQLite-backed state from Label Studio ZIP exports if DB is empty."""
    ensure_schema(db_file)
    if _db_is_seeded(db_file):
        logger.info("Found existing HITL DB state. Resuming...")
        return

    logger.info("Seeding HITL SQLite state from Label Studio ZIP exports...")
    merged_pages: dict[str, dict[str, Any]] = _load_legacy_state(legacy_state_file)
    if merged_pages:
        logger.info(f"Loaded {len(merged_pages)} pages from legacy JSON state.")

    zip_files = list(zip_dir.glob("*.zip"))
    if not zip_files:
        logger.warning(f"No ZIP files found in {zip_dir}.")

    for zip_path in zip_files:
        logger.info(f"Processing ZIP: {zip_path}")
        extracted = _extract_zip_pages(zip_path, source_dirs)
        for base_name, page_data in extracted.items():
            if base_name in merged_pages and merged_pages[base_name].get("verified", False):
                continue
            merged_pages[base_name] = page_data

    _seed_db(merged_pages, db_file)
    logger.info(f"Seeded HITL DB with {len(merged_pages)} pages.")


def load_state(db_file: Path = DB_FILE) -> dict[str, Any]:
    """Load current HITL state from SQLite."""
    ensure_schema(db_file)
    with db_connection(db_file) as conn:
        rows = conn.execute(
            "SELECT page_id, left_json, right_json, verified, img_w, img_h FROM page_lines ORDER BY page_id"
        ).fetchall()

    state: dict[str, Any] = {}
    for row in rows:
        state[row["page_id"]] = {
            "left": json.loads(row["left_json"]) if row["left_json"] else None,
            "right": json.loads(row["right_json"]) if row["right_json"] else None,
            "verified": bool(row["verified"]),
            "img_w": int(row["img_w"]),
            "img_h": int(row["img_h"]),
        }
    return state


def save_page_state(
    page_id: str,
    left: list[float] | None,
    right: list[float] | None,
    verified: bool,
    db_file: Path = DB_FILE,
) -> bool:
    """Update one page row; returns False if the page does not exist."""
    ensure_schema(db_file)
    with db_connection(db_file) as conn:
        existing = conn.execute("SELECT 1 FROM page_lines WHERE page_id = ?", (page_id,)).fetchone()
        if not existing:
            return False

        conn.execute(
            """
            UPDATE page_lines
            SET left_json = ?, right_json = ?, verified = ?, updated_at = CURRENT_TIMESTAMP
            WHERE page_id = ?
            """,
            (json.dumps(left), json.dumps(right), int(verified), page_id),
        )
    return True
