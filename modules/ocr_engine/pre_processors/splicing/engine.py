from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from PIL import Image

from utils.logger import get_logger

from .cropper import Cropper
from .geometry import DividerLine, Point2D
from .rectify import Rectifier

log = get_logger("SplicingEngine")
DEFAULT_HITL_DB_PATH = Path("data/layout_dataset/hitl_line_editor.sqlite3")
DEFAULT_JSON_MAP_PATH = Path("output/hitl/ocr_column_map.json")


class SplicingEngine:
    """Orchestrates the full geometry-first splicing pipeline."""

    def __init__(self, db_path: Path | None = None, json_map_path: Path | None = None):
        """Resolve and store divider source (SQLite preferred, JSON-map fallback)."""
        resolved_kind, resolved_path = self._resolve_source(
            db_path=db_path,
            json_map_path=json_map_path,
        )
        self.source_kind = resolved_kind
        self.source_path = resolved_path
        self.db_path = resolved_path if resolved_kind == "sqlite" else None
        self.json_map_path = resolved_path if resolved_kind == "json_map" else None
        self._json_map_cache: dict[str, dict] | None = None

        if self.source_kind == "sqlite":
            log.info(f"Using preferred HITL source: SQLite DB at {self.source_path}")
        else:
            log.info(f"Falling back to JSON map at {self.source_path}")
            log.info(
                "💡 Hint: Using the HITL review tool with the SQLite database is preferred for maximum precision."
            )
            self._json_map_cache = self._load_json_map(self.source_path)

    def _resolve_source(
        self,
        db_path: Path | None,
        json_map_path: Path | None,
    ) -> tuple[str, Path]:
        """Resolve divider source with deterministic precedence."""
        candidate_db = db_path or DEFAULT_HITL_DB_PATH
        candidate_json = json_map_path or DEFAULT_JSON_MAP_PATH
        db_exists = candidate_db.exists()
        json_exists = candidate_json.exists()

        if db_exists:
            return "sqlite", candidate_db
        if json_exists:
            return "json_map", candidate_json

        raise FileNotFoundError(
            f"\n❌ No verified divider source found!\n\n"
            f"PREFERRED: Use the HITL Review Tool to save verified lines to:\n"
            f"-- {candidate_db}\n\n"
            f"FALLBACK: If you want to use a JSON map, paste your export file at:\n"
            f"-- {candidate_json}\n\n"
            f"To generate a JSON map from an existing database, use 'tools/export_hitl_coordinates.py'."
        )

    def _load_json_map(self, json_path: Path) -> dict[str, dict]:
        """Load and validate strict OCR JSON map schema."""
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in divider map '{json_path}': {exc}") from exc

        if not isinstance(raw, dict):
            raise ValueError(
                f"Unsupported divider JSON format in '{json_path}'. Expected top-level object."
            )

        validated: dict[str, dict] = {}
        for page_id, payload in raw.items():
            if not isinstance(page_id, str) or not isinstance(payload, dict):
                raise ValueError(
                    f"Unsupported divider JSON format in '{json_path}'. "
                    "Expected page_id -> object mapping."
                )
            dividers = payload.get("dividers")
            if not isinstance(dividers, dict):
                raise ValueError(
                    f"Unsupported divider JSON format for page '{page_id}'. "
                    "Expected object key 'dividers'."
                )
            left = dividers.get("left")
            right = dividers.get("right")
            if not self._is_valid_line(left) or not self._is_valid_line(right):
                raise ValueError(
                    f"Unsupported divider JSON format for page '{page_id}'. "
                    "Expected 'dividers.left/right' as numeric [x1,y1,x2,y2]."
                )
            validated[page_id] = payload
        return validated

    def _is_valid_line(self, maybe_line) -> bool:
        if not isinstance(maybe_line, list) or len(maybe_line) < 4:
            return False
        return all(isinstance(value, (int, float)) for value in maybe_line[:4])

    def get_dividers_for_page(self, page_id: str) -> tuple[DividerLine | None, DividerLine | None]:
        """Load divider lines from resolved source."""
        if self.source_kind == "sqlite":
            return self._get_dividers_from_sqlite(page_id)
        return self._get_dividers_from_json_map(page_id)

    def _get_dividers_from_sqlite(
        self, page_id: str
    ) -> tuple[DividerLine | None, DividerLine | None]:
        """Load human-verified divider lines from HITL SQLite."""
        if self.db_path is None or not self.db_path.exists():
            log.error(f"Splicing DB not found at {self.db_path}")
            return None, None

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT left_json, right_json
                FROM page_lines
                WHERE page_id = ? AND verified = 1
                """,
                (page_id,),
            ).fetchone()

        if not row:
            return None, None

        def parse_line(js_str: str | None) -> DividerLine | None:
            if not js_str:
                return None
            try:
                data = json.loads(js_str)
            except json.JSONDecodeError:
                return None
            if not isinstance(data, list) or len(data) < 4:
                return None
            return DividerLine(
                top=Point2D(x=data[0], y=data[1]),
                bottom=Point2D(x=data[2], y=data[3]),
            )

        return parse_line(row["left_json"]), parse_line(row["right_json"])

    def _get_dividers_from_json_map(
        self, page_id: str
    ) -> tuple[DividerLine | None, DividerLine | None]:
        """Load divider lines from strict OCR JSON map format."""
        if self._json_map_cache is None:
            return None, None

        row = self._json_map_cache.get(page_id)
        if not row:
            return None, None

        dividers = row.get("dividers", {})
        left = dividers.get("left")
        right = dividers.get("right")
        if not self._is_valid_line(left) or not self._is_valid_line(right):
            return None, None

        return (
            DividerLine(
                top=Point2D(x=float(left[0]), y=float(left[1])),
                bottom=Point2D(x=float(left[2]), y=float(left[3])),
            ),
            DividerLine(
                top=Point2D(x=float(right[0]), y=float(right[1])),
                bottom=Point2D(x=float(right[2]), y=float(right[3])),
            ),
        )

    def splice_page(
        self,
        image: Image.Image,
        page_id: str,
        rectify_mode: str = "rotate+homography",
    ) -> dict:
        """
        Executes the full splicing pipeline for a single page.
        Returns a result dict with strips, transforms, and QC metrics.
        """
        left, right = self.get_dividers_for_page(page_id)

        if not left or not right:
            return {"status": "ERR_MISSING_DIVIDER", "page_id": page_id}

        # Failure Taxonomy check: Crossing dividers
        if self._check_crossing(left, right):
            return {"status": "ERR_CROSSING_DIVIDERS", "page_id": page_id}

        try:
            # 1. Calculate Transform
            if rectify_mode == "rotate+homography":
                transform = Rectifier.get_homography_matrix(image.size, left, right)
            elif rectify_mode == "rotate":
                transform = Rectifier.get_deskew_matrix(image.size, [left, right])
            else:
                return {
                    "status": "ERR_INVALID_RECTIFY_MODE",
                    "page_id": page_id,
                    "error": f"Unsupported rectify mode: {rectify_mode}",
                }

            # 2. Rectify Image
            rectified_img = Rectifier.apply_transform(image, transform)

            # 3. Transform Dividers for Cropping
            # We need the dividers in the "rectified space" to know where to cut accurately
            rect_left = DividerLine(
                top=transform.forward(left.top), bottom=transform.forward(left.bottom)
            )
            rect_right = DividerLine(
                top=transform.forward(right.top), bottom=transform.forward(right.bottom)
            )

            # 4. Crop into strips
            strips, offsets = Cropper.get_column_strips(rectified_img, rect_left, rect_right)

            return {
                "status": "SUCCESS",
                "page_id": page_id,
                "strips": strips,
                "offsets": offsets,
                "transform": transform,
                "dividers": {"left": left, "right": right},
                "qc": {
                    "skew": (left.angle_deg + right.angle_deg) / 2,
                    "rectified_width": rectified_img.size[0],
                },
            }
        except Exception as exc:
            log.error(f"Splicing failed for page {page_id}: {exc}")
            return {"status": "ERR_UNKNOWN", "page_id": page_id, "error": str(exc)}

    def _check_crossing(self, left: DividerLine, right: DividerLine) -> bool:
        """Check if the two lines intersect within the page bounds."""
        return left.top.x >= right.top.x or left.bottom.x >= right.bottom.x
