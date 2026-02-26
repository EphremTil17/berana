import json
import re
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tools.hitl_line_editor_app.db import (
    db_connection,
    ensure_schema,
    get_provenance,
    list_page_ids,
    set_provenance,
)
from tools.hitl_line_editor_app.geometry import (
    calculate_optimal_slice_vector,
    extract_endpoints_from_vector,
)
from tools.hitl_line_editor_app.paths import DB_FILE, LEGACY_STATE_FILE, SOURCE_DIRS, ZIP_DIR
from utils.logger import get_logger

logger = get_logger("LineEditorState")


RUN_DIR_PATTERN = re.compile(r"^(?P<doc_stem>.+)_v\d+$")


def _as_layout_infer_run_dir(img_path: Path) -> Path | None:
    """Return the parent run directory for a layout-infer image path."""
    # Expected: .../output/layout_inference/<doc_stem>_vNN/visuals/page_XXX.jpg
    if img_path.parent.name != "visuals":
        return None

    run_dir = img_path.parent.parent
    if run_dir.parent.name != "layout_inference":
        return None

    if not RUN_DIR_PATTERN.match(run_dir.name):
        return None
    return run_dir


def _resolve_image_path(
    base_name: str,
    source_dirs: list[Path],
    locked_run_dir: Path | None = None,
) -> tuple[Path | None, Path | None]:
    img_filename = f"{base_name}.jpg"
    candidates: list[tuple[Path, Path]] = []
    for directory in source_dirs:
        for candidate in directory.rglob(img_filename):
            run_dir = _as_layout_infer_run_dir(candidate)
            if run_dir is not None:
                candidates.append((candidate, run_dir))

    if locked_run_dir is not None:
        candidates = [pair for pair in candidates if pair[1] == locked_run_dir]

    if not candidates:
        return None, locked_run_dir

    run_dirs = {run_dir for _, run_dir in candidates}
    if len(run_dirs) > 1:
        sample_runs = ", ".join(sorted(str(path) for path in run_dirs)[:5])
        raise ValueError(
            f"Ambiguous image resolution for '{img_filename}'. "
            f"Found multiple layout-infer runs: {sample_runs}"
        )

    selected_run = next(iter(run_dirs))
    selected_image = max(candidates, key=lambda pair: pair[0].stat().st_mtime)[0]
    return selected_image, selected_run


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


def _extract_provenance(img_path: Path) -> dict[str, str] | None:
    run_dir = _as_layout_infer_run_dir(img_path)
    if run_dir is None:
        return None

    match = RUN_DIR_PATTERN.match(run_dir.name)
    if not match:
        return None

    return {
        "source_stage": "layout-infer",
        "doc_stem": match.group("doc_stem"),
        "source_run_dir": str(run_dir.absolute()),
    }


def _extract_zip_pages(
    zip_path: Path, source_dirs: list[Path]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    page_map: dict[str, dict[str, Any]] = {}
    provenances = []
    locked_run_dir: Path | None = None

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        for filename in zip_file.namelist():
            if not (filename.startswith("labels/") and filename.endswith(".txt")):
                continue

            base_name = Path(filename).stem
            img_path, locked_run_dir = _resolve_image_path(
                base_name,
                source_dirs,
                locked_run_dir=locked_run_dir,
            )
            if not img_path:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            raw_text = zip_file.read(filename).decode("utf-8")
            lines = raw_text.strip().split("\n")
            page_map[base_name] = _build_page_data(img, lines)

            prov = _extract_provenance(img_path)
            if prov and prov not in provenances:
                provenances.append(prov)

    return page_map, provenances


def _db_is_seeded(db_file: Path) -> bool:
    with db_connection(db_file) as conn:
        row = conn.execute("SELECT COUNT(*) AS total FROM page_lines").fetchone()
    return bool(row and row["total"])


def _candidate_visual_dirs(source_dirs: list[Path]) -> list[Path]:
    visual_dirs: list[Path] = []
    for directory in source_dirs:
        if not directory.exists():
            continue
        if directory.name == "layout_inference":
            visual_dirs.extend(path for path in directory.glob("*_v*/visuals") if path.is_dir())
            continue
        if directory.name == "visuals" and directory.parent.name == "layout_inference":
            visual_dirs.append(directory)
    return visual_dirs


def _run_coverage_counts(page_ids: list[str], visual_dirs: list[Path]) -> list[tuple[Path, int]]:
    coverage: list[tuple[Path, int]] = []
    for visuals_dir in visual_dirs:
        run_dir = visuals_dir.parent
        count = sum(1 for page_id in page_ids if (visuals_dir / f"{page_id}.jpg").exists())
        if count > 0:
            coverage.append((run_dir, count))
    return coverage


def _infer_seeded_db_provenance(db_file: Path, source_dirs: list[Path]) -> dict[str, str] | None:
    """Best-effort provenance backfill for already-seeded DBs missing metadata."""
    page_ids = list_page_ids(db_file)
    if not page_ids:
        return None

    visual_dirs = _candidate_visual_dirs(source_dirs)
    if not visual_dirs:
        return None

    coverage = _run_coverage_counts(page_ids, visual_dirs)
    if not coverage:
        return None

    max_count = max(count for _, count in coverage)
    winners = [run_dir for run_dir, count in coverage if count == max_count]
    if len(winners) != 1:
        return None
    if max_count != len(page_ids):
        return None

    match = RUN_DIR_PATTERN.match(winners[0].name)
    if not match:
        return None
    return {
        "source_stage": "layout-infer",
        "doc_stem": match.group("doc_stem"),
        "source_run_dir": str(winners[0].absolute()),
    }


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


def _process_zip_files(
    zip_files: list[Path],
    source_dirs: list[Path],
    merged_pages: dict,
) -> dict | None:
    detected_provenance = None

    for zip_path in zip_files:
        logger.info(f"Processing ZIP: {zip_path}")
        extracted, zip_provs = _extract_zip_pages(zip_path, source_dirs)

        for prov in zip_provs:
            if detected_provenance is None:
                detected_provenance = prov
            elif detected_provenance != prov:
                raise ValueError(
                    f"Mixed source runs detected during seed! Initialization failed.\n"
                    f"Found {detected_provenance} and {prov}"
                )

        for base_name, page_data in extracted.items():
            if base_name in merged_pages and merged_pages[base_name].get("verified", False):
                continue
            merged_pages[base_name] = page_data
    return detected_provenance


def initialize_state(
    zip_dir: Path = ZIP_DIR,
    source_dirs: list[Path] = SOURCE_DIRS,
    db_file: Path = DB_FILE,
    legacy_state_file: Path = LEGACY_STATE_FILE,
) -> None:
    """Initialize SQLite-backed state from Label Studio ZIP exports if DB is empty."""
    ensure_schema(db_file)
    if _db_is_seeded(db_file):
        if get_provenance(db_file) is None:
            inferred = _infer_seeded_db_provenance(db_file, source_dirs)
            if inferred is not None:
                set_provenance(
                    db_file,
                    inferred["source_stage"],
                    inferred["doc_stem"],
                    inferred["source_run_dir"],
                )
                logger.info(
                    "Backfilled missing DB provenance: "
                    f"{inferred['doc_stem']} ({inferred['source_run_dir']})"
                )
            else:
                logger.warning(
                    "Existing HITL DB is missing provenance metadata and could not be auto-inferred. "
                    "Finetuner export will fail until provenance is re-seeded."
                )
        logger.info("Found existing HITL DB state. Resuming...")
        return

    logger.info("Seeding HITL SQLite state from Label Studio ZIP exports...")
    merged_pages: dict[str, dict[str, Any]] = _load_legacy_state(legacy_state_file)
    if merged_pages:
        logger.info(f"Loaded {len(merged_pages)} pages from legacy JSON state.")

    zip_files = list(zip_dir.glob("*.zip"))
    if not zip_files:
        logger.warning(f"No ZIP files found in {zip_dir}.")

    detected_provenance = _process_zip_files(zip_files, source_dirs, merged_pages)

    _seed_db(merged_pages, db_file)
    logger.info(f"Seeded HITL DB with {len(merged_pages)} pages.")

    if detected_provenance:
        set_provenance(
            db_file,
            detected_provenance["source_stage"],
            detected_provenance["doc_stem"],
            detected_provenance["source_run_dir"],
        )
        logger.info(
            f"Bound DB provenance to {detected_provenance['doc_stem']} from {detected_provenance['source_stage']}."
        )


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
