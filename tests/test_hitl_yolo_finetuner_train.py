import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.hitl_yolo_finetuner_app.train import run_yolo_train


def _write_dataset_yaml(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "manifests").mkdir(parents=True, exist_ok=True)
    (path.parent / "manifests" / "train_pages.txt").write_text("page_001\n", encoding="utf-8")
    (path.parent / "manifests" / "val_pages.txt").write_text("page_002\n", encoding="utf-8")
    path.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: divider_left\n  1: divider_right\n",
        encoding="utf-8",
    )


def test_train_fails_without_latest_export_pointer(tmp_path: Path) -> None:
    """Training must fail when no latest export pointer exists for the requested doc."""
    with pytest.raises(ValueError, match="No latest layout-finetune-export run found"):
        run_yolo_train(
            doc_stem="doc_a",
            epochs=1,
            batch=1,
            imgsz=640,
        )


def test_train_registers_weights_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Training should produce a manifest and register layout-train artifacts/metadata."""
    output_root = tmp_path / "output" / "hitl_finetuner"
    dataset_yaml = output_root / "doc_a_v01" / "data" / "dataset.yaml"
    _write_dataset_yaml(dataset_yaml)

    class FakeYOLO:
        def __init__(self, _model_name: str) -> None:
            pass

        def train(self, **kwargs):  # type: ignore[no-untyped-def]
            save_dir = Path(kwargs["project"]) / kwargs["name"]
            (save_dir / "weights").mkdir(parents=True, exist_ok=True)
            (save_dir / "weights" / "best.pt").write_bytes(b"weights")
            return SimpleNamespace(save_dir=str(save_dir))

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    captured = {}

    def _fake_register_latest_run(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return Path("mock_registry_path")

    monkeypatch.setattr(
        "tools.hitl_yolo_finetuner_app.train.register_latest_run",
        _fake_register_latest_run,
    )
    monkeypatch.setattr(
        "tools.hitl_yolo_finetuner_app.train.load_latest_run",
        lambda stage, doc_stem: {
            "stage": stage,
            "doc_stem": doc_stem,
            "run_dir": str(output_root / "doc_a_v01"),
            "artifacts": {"dataset_yaml": str(dataset_yaml)},
        },
    )

    manifest = run_yolo_train(
        doc_stem="doc_a",
        epochs=2,
        batch=3,
        imgsz=512,
    )

    assert manifest["doc_stem"] == "doc_a"
    assert captured["stage"] == "layout-train"
    assert captured["doc_stem"] == "doc_a"
    assert "weights_best" in captured["artifacts"]
    assert captured["metadata"]["epochs"] == 2
    assert captured["metadata"]["batch"] == 3
    assert captured["metadata"]["imgsz"] == 512


def test_train_preserves_best_weights_on_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl+C should preserve best-so-far checkpoint into run_dir/weights."""
    output_root = tmp_path / "output" / "hitl_finetuner"
    run_dir = output_root / "doc_a_v01"
    dataset_yaml = run_dir / "data" / "dataset.yaml"
    _write_dataset_yaml(dataset_yaml)

    class FakeYOLO:
        def __init__(self, _model_name: str) -> None:
            pass

        def train(self, **kwargs):  # type: ignore[no-untyped-def]
            save_dir = Path(kwargs["project"]) / kwargs["name"]
            (save_dir / "weights").mkdir(parents=True, exist_ok=True)
            (save_dir / "weights" / "best.pt").write_bytes(b"weights")
            raise KeyboardInterrupt

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setattr(
        "tools.hitl_yolo_finetuner_app.train.load_latest_run",
        lambda stage, doc_stem: {
            "stage": stage,
            "doc_stem": doc_stem,
            "run_dir": str(run_dir),
            "artifacts": {"dataset_yaml": str(dataset_yaml)},
        },
    )

    with pytest.raises(KeyboardInterrupt):
        run_yolo_train(
            doc_stem="doc_a",
            epochs=2,
            batch=3,
            imgsz=512,
        )

    assert (run_dir / "weights" / "best.pt").exists()
