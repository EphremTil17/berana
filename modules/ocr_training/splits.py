from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from modules.ocr_training.normalize import derive_group_id
from modules.ocr_training.schemas import DatasetSplit, SourceSnapshotRow, SplitConfig


def _stable_sort_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _target_counts(total: int, cfg: SplitConfig) -> dict[DatasetSplit, int]:
    train_count = int(total * cfg.train_ratio)
    val_count = int(total * cfg.val_ratio)
    holdout_count = total - train_count - val_count
    return {
        DatasetSplit.TRAIN: train_count,
        DatasetSplit.VAL: val_count,
        DatasetSplit.HOLDOUT: holdout_count,
    }


def assign_splits(rows: list[SourceSnapshotRow], cfg: SplitConfig) -> dict[str, DatasetSplit]:
    """Deterministically assign rows into train/val/holdout splits."""
    if not rows:
        raise ValueError("No rows available for split assignment.")

    targets = _target_counts(len(rows), cfg)
    assignments: dict[str, DatasetSplit] = {}

    if cfg.strict_page_isolation:
        by_group: dict[str, list[SourceSnapshotRow]] = defaultdict(list)
        for row in rows:
            by_group[derive_group_id(row.source_repo, row.original_filename)].append(row)
        group_order = sorted(by_group.keys(), key=lambda g: _stable_sort_key(cfg.seed, g))
        counts = Counter()

        for group in group_order:
            group_rows = by_group[group]
            remaining = {
                split: max(0, targets[split] - counts[split])
                for split in (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.HOLDOUT)
            }
            split = max(remaining, key=lambda key: remaining[key])
            for row in group_rows:
                assignments[row.sample_id] = split
            counts[split] += len(group_rows)
        return assignments

    ordered = sorted(rows, key=lambda r: _stable_sort_key(cfg.seed, r.sample_id))
    idx = 0
    for split in (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.HOLDOUT):
        count = targets[split]
        for row in ordered[idx : idx + count]:
            assignments[row.sample_id] = split
        idx += count
    return assignments


def validate_split_leakage(
    rows: list[SourceSnapshotRow],
    assignments: dict[str, DatasetSplit],
    *,
    strict_page_isolation: bool,
) -> dict[str, int]:
    """Validate line-level and optional page-level split isolation."""
    seen: set[str] = set()
    page_splits: dict[str, set[DatasetSplit]] = defaultdict(set)
    counts = Counter()

    for row in rows:
        split = assignments.get(row.sample_id)
        if split is None:
            raise ValueError(f"Missing split assignment for sample_id={row.sample_id}.")
        if row.sample_id in seen:
            raise ValueError(f"CRITICAL LEAKAGE: duplicate sample_id encountered: {row.sample_id}")
        seen.add(row.sample_id)
        counts[split.value] += 1

        page_key = derive_group_id(row.source_repo, row.original_filename)
        page_splits[page_key].add(split)

    if strict_page_isolation:
        overlaps = [
            page_id
            for page_id, split_set in page_splits.items()
            if len(
                split_set.intersection({DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.HOLDOUT})
            )
            > 1
        ]
        if overlaps:
            raise ValueError(
                "PAGE LEAKAGE: page groups are split across datasets. "
                f"Example groups: {sorted(overlaps)[:10]}"
            )

    return {
        "train": int(counts[DatasetSplit.TRAIN.value]),
        "val": int(counts[DatasetSplit.VAL.value]),
        "holdout": int(counts[DatasetSplit.HOLDOUT.value]),
    }
