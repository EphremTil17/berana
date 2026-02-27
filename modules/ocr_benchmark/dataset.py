import hashlib
import json
from pathlib import Path

from schemas.ocr_benchmark import DatasetSplit, LineManifestRow, SplitManifest
from utils.logger import get_logger

logger = get_logger("OCRBenchmarkDataset")


def read_manifest(path: Path) -> list[LineManifestRow]:
    """Read a JSONL benchmark manifest file and return strictly validated rows."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                row = LineManifestRow.model_validate(data)
                rows.append(row)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse or validate manifest row at line {line_num} in {path}: {e}"
                ) from e

    return rows


def write_manifest(rows: list[LineManifestRow], path: Path) -> None:
    """Write validated manifest rows to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json(exclude_none=True) + "\n")


def _stable_json_bytes(payload: object) -> bytes:
    """Return deterministic serialized bytes for hashing."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def compute_records_hash(records: list[dict]) -> str:
    """Compute deterministic hash for a record list."""
    normalized = sorted(records, key=lambda item: str(item.get("line_id", "")))
    return hashlib.sha256(_stable_json_bytes(normalized)).hexdigest()


def write_split_manifest(
    *,
    path: Path,
    dataset_hash: str,
    random_seed: int,
    train_count: int,
    holdout_count: int,
) -> SplitManifest:
    """Write split-freeze metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = SplitManifest(
        dataset_hash=dataset_hash,
        random_seed=random_seed,
        train_count=train_count,
        holdout_count=holdout_count,
    )
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def load_split_manifest(path: Path) -> SplitManifest:
    """Load split-freeze metadata."""
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return SplitManifest.model_validate(data)


def validate_split_leakage(
    rows: list[LineManifestRow], strict_page_isolation: bool = True
) -> dict[str, int]:
    """Ensure absolutely no leakage between train and holdout datasets.

    Hard constraints:
    - No line_id may appear in both train and holdout.
    - If strict_page_isolation=True, no page_id may appear in both train and holdout.

    Returns:
        A dictionary containing counts for train and holdout lines and pages.
    """
    train_lines = set()
    holdout_lines = set()

    train_pages = set()
    holdout_pages = set()

    for row in rows:
        if row.split == DatasetSplit.TRAIN:
            train_lines.add(row.line_id)
            train_pages.add(row.page_id)
        elif row.split == DatasetSplit.HOLDOUT:
            holdout_lines.add(row.line_id)
            holdout_pages.add(row.page_id)

    # Constraint 1: line_id overlap
    line_overlap = train_lines.intersection(holdout_lines)
    if line_overlap:
        raise ValueError(
            f"CRITICAL LEAKAGE: The following line_ids are mixed across train/holdout splits: "
            f"{line_overlap}"
        )

    # Constraint 2: page_id overlap
    page_overlap = train_pages.intersection(holdout_pages)
    if page_overlap:
        msg = (
            "PAGE LEAKAGE: The following page_ids are mixed across train/holdout splits: "
            f"{page_overlap}"
        )
        if strict_page_isolation:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    stats = {
        "train_lines": len(train_lines),
        "holdout_lines": len(holdout_lines),
        "train_pages": len(train_pages),
        "holdout_pages": len(holdout_pages),
    }

    if stats["train_lines"] < 180 and strict_page_isolation:
        logger.warning(
            f"STRICT MODE DROPPED USABLE LINES: Train corpus is underpowered "
            f"({stats['train_lines']} < 180 lines) due to strict_page_isolation=True. "
            f"Consider setting to False if holdout constraints bleed too heavily."
        )

    return stats
