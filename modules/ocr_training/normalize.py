from __future__ import annotations

import re
from pathlib import Path

from modules.ocr_training.schemas import NormalizedType, SourceRepo, SourceSplit

_HANDWRITTEN_ALIASES = {"handwritten", "hdd", "hdd_18", "hdd_rand"}


def normalize_text(text: str) -> str:
    """Normalize transcript text for deterministic manifest behavior."""
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_source_type(raw_type: str | None, source_repo: SourceRepo) -> NormalizedType:
    """Map upstream source type values to canonical normalized types."""
    value = (raw_type or "").strip().lower()
    if source_repo == SourceRepo.FIDEL_SYNTHETIC:
        return NormalizedType.SYNTHETIC
    if value == "typed":
        return NormalizedType.TYPED
    if value == "synthetic":
        return NormalizedType.SYNTHETIC
    if value in _HANDWRITTEN_ALIASES:
        return NormalizedType.HANDWRITTEN
    raise ValueError(f"Unsupported source type value '{raw_type}'.")


def build_sample_id(
    source_repo: SourceRepo,
    source_split: SourceSplit,
    original_filename: str,
) -> str:
    """Build deterministic globally unique sample identifier."""
    return f"{source_repo.value}:{source_split.value}:{Path(original_filename).name}"


def build_extracted_filename(
    source_repo: SourceRepo,
    source_split: SourceSplit,
    original_filename: str,
) -> str:
    """Build deterministic namespaced output filename for extracted assets."""
    name = Path(original_filename).name
    if source_repo == SourceRepo.FIDEL_SYNTHETIC:
        return f"fidel_synthetic__{name}"
    return f"fidel_dataset_{source_split.value}__{name}"


def derive_group_id(source_repo: SourceRepo, original_filename: str) -> str:
    """Derive a coarse grouping key to support strict page-level split isolation."""
    stem = Path(original_filename).stem
    if "_line_" in stem:
        return stem.split("_line_", 1)[0]
    if source_repo == SourceRepo.FIDEL_SYNTHETIC and "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem
