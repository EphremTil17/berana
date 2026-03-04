from modules.ocr_training.schemas import (
    DatasetSplit,
    NormalizedType,
    SourceRepo,
    SourceSnapshotRow,
    SourceSplit,
    SplitConfig,
)
from modules.ocr_training.splits import assign_splits, validate_split_leakage


def _row(sample_id: str, filename: str) -> SourceSnapshotRow:
    return SourceSnapshotRow(
        sample_id=sample_id,
        source_repo=SourceRepo.FIDEL_DATASET,
        source_split=SourceSplit.TRAIN,
        original_filename=filename,
        normalized_type=NormalizedType.TYPED,
        text_raw="abc",
        text_normalized="abc",
        image_relpath="input/ocr_training/fidel/extracted/typed/x.png",
        excluded=False,
    )


def test_assign_splits_is_deterministic():
    rows = [_row(f"id_{i}", f"typed_{i}_line_1.png") for i in range(30)]
    cfg = SplitConfig(train_ratio=0.8, val_ratio=0.1, holdout_ratio=0.1, seed=42)
    first = assign_splits(rows, cfg)
    second = assign_splits(rows, cfg)
    assert first == second


def test_validate_split_leakage_counts():
    rows = [_row(f"id_{i}", f"typed_{i}_line_1.png") for i in range(10)]
    cfg = SplitConfig(train_ratio=0.8, val_ratio=0.1, holdout_ratio=0.1, seed=7)
    assignments = assign_splits(rows, cfg)
    stats = validate_split_leakage(rows, assignments, strict_page_isolation=False)
    assert stats["train"] == 8
    assert stats["val"] == 1
    assert stats["holdout"] == 1


def test_validate_split_leakage_page_isolation_fails_when_group_overlaps():
    rows = [_row("a", "typed_1_line_1.png"), _row("b", "typed_1_line_2.png")]
    assignments = {"a": DatasetSplit.TRAIN, "b": DatasetSplit.HOLDOUT}
    try:
        validate_split_leakage(rows, assignments, strict_page_isolation=True)
    except ValueError as exc:
        assert "PAGE LEAKAGE" in str(exc)
    else:
        raise AssertionError("Expected strict page isolation failure")
