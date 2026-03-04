from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from modules.ocr_training.normalize import build_sample_id, normalize_source_type, normalize_text
from modules.ocr_training.schemas import NormalizedType, SourceRepo, SourceSplit


@dataclass(frozen=True)
class CatalogRow:
    """Normalized catalog row from FIDEL source labels."""

    sample_id: str
    source_repo: SourceRepo
    source_split: SourceSplit
    original_filename: str
    normalized_type: NormalizedType
    text_raw: str
    text_normalized: str


def _read_fidel_dataset_labels(csv_path: Path, split: SourceSplit) -> list[CatalogRow]:
    rows: list[CatalogRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            original_filename = (row.get("image_filename") or "").strip()
            if not original_filename:
                continue
            source_repo = SourceRepo.FIDEL_DATASET
            normalized_type = normalize_source_type(row.get("type"), source_repo)
            text_raw = row.get("line_text") or ""
            rows.append(
                CatalogRow(
                    sample_id=build_sample_id(source_repo, split, original_filename),
                    source_repo=source_repo,
                    source_split=split,
                    original_filename=original_filename,
                    normalized_type=normalized_type,
                    text_raw=text_raw,
                    text_normalized=normalize_text(text_raw),
                )
            )
    return rows


def _read_fidel_synthetic_labels(csv_path: Path) -> list[CatalogRow]:
    rows: list[CatalogRow] = []
    source_repo = SourceRepo.FIDEL_SYNTHETIC
    split = SourceSplit.SYNTHETIC
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            original_filename = (row.get("images") or "").strip()
            if not original_filename:
                continue
            text_raw = row.get("text") or ""
            rows.append(
                CatalogRow(
                    sample_id=build_sample_id(source_repo, split, original_filename),
                    source_repo=source_repo,
                    source_split=split,
                    original_filename=original_filename,
                    normalized_type=NormalizedType.SYNTHETIC,
                    text_raw=text_raw,
                    text_normalized=normalize_text(text_raw),
                )
            )
    return rows


def load_catalog(raw_root: Path) -> list[CatalogRow]:
    """Load and normalize all upstream FIDEL label rows into a single catalog."""
    dataset_root = raw_root / SourceRepo.FIDEL_DATASET.value
    synthetic_root = raw_root / SourceRepo.FIDEL_SYNTHETIC.value

    required = [
        dataset_root / "train_labels.csv",
        dataset_root / "test_labels.csv",
        synthetic_root / "synthetic_labels.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required catalog files: {missing}")

    rows = []
    rows.extend(_read_fidel_dataset_labels(dataset_root / "train_labels.csv", SourceSplit.TRAIN))
    rows.extend(_read_fidel_dataset_labels(dataset_root / "test_labels.csv", SourceSplit.TEST))
    rows.extend(_read_fidel_synthetic_labels(synthetic_root / "synthetic_labels.csv"))
    return rows
