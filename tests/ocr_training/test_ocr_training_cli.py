from pathlib import Path

from modules.ocr_training.schemas import TrainMode
from tools.ocr_training import (
    _dataset_run_stem,
    _resolve_train_output_dir,
    _strip_version_suffix,
)


def test_strip_version_suffix_removes_trailing_vnn():
    assert _strip_version_suffix("fidel_typed_synthetic_v01") == "fidel_typed_synthetic"
    assert _strip_version_suffix("fidel_typed_synthetic_auto_v12") == "fidel_typed_synthetic_auto"
    assert _strip_version_suffix("plain_name") == "plain_name"


def test_dataset_run_stem_infers_parent_dataset_name(tmp_path: Path):
    dataset_dir = (
        tmp_path
        / "output"
        / "ocr_training_datasets"
        / "fidel_typed_synthetic_v03"
        / "data"
        / "hf_dataset"
    )
    dataset_dir.mkdir(parents=True)

    assert _dataset_run_stem(dataset_dir) == "fidel_typed_synthetic"


def test_resolve_train_output_dir_defaults_to_versioned_auto_run(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "output" / "ocr_training_runs"
    (base / "fidel_typed_synthetic_auto_v01").mkdir(parents=True)

    resolved = _resolve_train_output_dir(
        dataset_dir=tmp_path
        / "output"
        / "ocr_training_datasets"
        / "fidel_typed_synthetic_v03"
        / "data"
        / "hf_dataset",
        output_dir=None,
        mode=TrainMode.AUTO,
    )

    assert resolved == Path("output/ocr_training_runs/fidel_typed_synthetic_auto_v02")


def test_resolve_train_output_dir_preserves_explicit_override(tmp_path: Path):
    explicit = tmp_path / "custom" / "run_dir"

    resolved = _resolve_train_output_dir(
        dataset_dir=tmp_path / "dataset",
        output_dir=explicit,
        mode=TrainMode.MANUAL,
    )

    assert resolved == explicit
