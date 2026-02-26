import copy
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from tools.hitl_line_editor_app.db import ensure_schema
from tools.hitl_yolo_finetuner_app.export import export_dataset


def create_synthetic_db(db_file: Path, num_pages: int = 30) -> None:
    """Create a synthetic down-sampled HITL database for testing."""
    ensure_schema(db_file)
    with sqlite3.connect(db_file) as conn:
        for i in range(num_pages):
            left_json = json.dumps([100.0, 50.0, 100.0, 950.0])
            right_json = json.dumps([900.0, 50.0, 900.0, 950.0])
            conn.execute(
                """
                INSERT INTO page_lines (page_id, left_json, right_json, verified, img_w, img_h)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (f"page_{i:03d}", left_json, right_json, 1, 1000, 1000),
            )


def test_export_dataset_integration(tmp_path: Path):
    """Integrate export_dataset across synthetic db, asserting standard outputs."""
    db_file = tmp_path / "synthetic.sqlite3"
    create_synthetic_db(db_file, num_pages=100)

    # Create real images matching DB dimensions
    images_dir = tmp_path / "source_images"
    images_dir.mkdir()
    for i in range(100):
        img = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
        img.save(images_dir / f"page_{i:03d}.jpg")

    out_root = tmp_path / "export_output"

    # Run export
    with (
        patch("tools.hitl_yolo_finetuner_app.export.get_provenance") as mock_prov,
        patch("tools.hitl_yolo_finetuner_app.export.load_latest_run") as mock_run,
        patch("tools.hitl_yolo_finetuner_app.export.register_latest_run"),
    ):
        mock_prov.return_value = {
            "source_stage": "layout-infer",
            "doc_stem": "test_doc",
            "source_run_dir": "/mock/run/dir",
        }
        mock_run.return_value = {
            "run_dir": "/mock/run/dir",
            "artifacts": {"visuals_dir": str(images_dir)},
        }

        report = export_dataset(
            doc_stem="test_doc",
            db_file=db_file,
            output_dir=out_root,
            split_seed=42,
            min_train_pages=20,
            min_val_pages=5,
            copy_mode="copy",  # Safe for tmp_path cross-device
        )

    # next_versioned_dir creates test_doc_v01
    run_dir = out_root / "test_doc_v01"
    out_dir = run_dir / "data"

    assert report["total_exported"] == 100
    assert report["train_count"] >= 20
    assert report["val_count"] >= 5
    assert report["train_count"] + report["val_count"] == 100

    # Check outputs
    assert (out_dir / "dataset.yaml").exists()
    assert (out_dir / "export_report.json").exists()
    assert (out_dir / "manifests" / "train_pages.txt").exists()
    assert (out_dir / "manifests" / "val_pages.txt").exists()

    # Check that labels/images got populated
    train_labels = list((out_dir / "labels" / "train").glob("*.txt"))
    assert len(train_labels) == report["train_count"]

    train_imgs = list((out_dir / "images" / "train").glob("*.jpg"))
    assert len(train_imgs) == report["train_count"]

    # Check the actual YOLO file content format
    sample_label = train_labels[0].read_text()
    assert sample_label.startswith("0 ")
    assert "1 " in sample_label  # both classes present


def test_export_dataset_fails_on_dimension_mismatch(tmp_path: Path):
    """Verified pages must have strict DB/image dimension parity."""
    db_file = tmp_path / "synthetic.sqlite3"
    create_synthetic_db(db_file, num_pages=1)
    images_dir = tmp_path / "source_images"
    images_dir.mkdir()
    # DB says 1000x1000 but image is 900x1000.
    img = Image.new("RGB", (900, 1000), color=(255, 255, 255))
    img.save(images_dir / "page_000.jpg")

    with (
        patch("tools.hitl_yolo_finetuner_app.export.get_provenance") as mock_prov,
        patch("tools.hitl_yolo_finetuner_app.export.load_latest_run") as mock_run,
        pytest.raises(ValueError, match="Dimension mismatch"),
    ):
        mock_prov.return_value = {
            "source_stage": "layout-infer",
            "doc_stem": "test_doc",
            "source_run_dir": "/mock",
        }
        mock_run.return_value = {"run_dir": "/mock", "artifacts": {"visuals_dir": str(images_dir)}}

        export_dataset(
            doc_stem="test_doc",
            db_file=db_file,
            output_dir=tmp_path / "export_output",
            copy_mode="copy",
        )


def test_export_dataset_fails_on_missing_verified_image(tmp_path: Path):
    """Missing image for a verified page must fail loud."""
    db_file = tmp_path / "synthetic.sqlite3"
    create_synthetic_db(db_file, num_pages=1)
    images_dir = tmp_path / "source_images"
    images_dir.mkdir()

    with (
        patch("tools.hitl_yolo_finetuner_app.export.get_provenance") as mock_prov,
        patch("tools.hitl_yolo_finetuner_app.export.load_latest_run") as mock_run,
        pytest.raises(FileNotFoundError, match="missing source image"),
    ):
        mock_prov.return_value = {
            "source_stage": "layout-infer",
            "doc_stem": "test_doc",
            "source_run_dir": "/mock",
        }
        mock_run.return_value = {"run_dir": "/mock", "artifacts": {"visuals_dir": str(images_dir)}}

        export_dataset(
            doc_stem="test_doc",
            db_file=db_file,
            output_dir=tmp_path / "export_output",
            copy_mode="copy",
        )


def test_export_dataset_fails_on_pointer_drift(tmp_path: Path):
    """Reject drift when DB provenance points to an older run and force mode is off."""
    db_file = tmp_path / "synthetic.sqlite3"
    create_synthetic_db(db_file, num_pages=1)

    with (
        patch("tools.hitl_yolo_finetuner_app.export.get_provenance") as mock_prov,
        patch("tools.hitl_yolo_finetuner_app.export.load_latest_run") as mock_run,
        pytest.raises(ValueError, match="Latest pointer drift detected"),
    ):
        # DB thinks v01
        mock_prov.return_value = {
            "source_stage": "layout-infer",
            "doc_stem": "test_doc",
            "source_run_dir": "/mock/test_doc_v01",
        }
        # Registry says latest is v02
        mock_run.return_value = {
            "run_dir": "/mock/test_doc_v02",
            "artifacts": {"visuals_dir": "/mock"},
        }

        export_dataset(
            doc_stem="test_doc",
            db_file=db_file,
            output_dir=tmp_path / "export_output",
        )


def test_export_dataset_fails_on_missing_provenance(tmp_path: Path):
    """Fail fast when DB has no provenance metadata and force mode is off."""
    db_file = tmp_path / "synthetic.sqlite3"
    create_synthetic_db(db_file, num_pages=1)

    with (
        patch("tools.hitl_yolo_finetuner_app.export.load_latest_run") as mock_run,
        pytest.raises(ValueError, match="missing provenance metadata"),
    ):
        mock_run.return_value = {
            "run_dir": "/mock/test_doc_v01",
            "artifacts": {"visuals_dir": "/mock/visuals"},
        }
        export_dataset(
            doc_stem="test_doc",
            db_file=db_file,
            output_dir=tmp_path / "export_output",
        )


def test_export_dataset_force_provenance_rebind_allows_missing_metadata(tmp_path: Path):
    """Allow explicit forced provenance binding for long-lived legacy DBs."""
    db_file = tmp_path / "synthetic.sqlite3"
    create_synthetic_db(db_file, num_pages=1)
    images_dir = tmp_path / "source_images"
    images_dir.mkdir()
    img = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
    img.save(images_dir / "page_000.jpg")

    out_root = tmp_path / "export_output"
    with (
        patch("tools.hitl_yolo_finetuner_app.export.get_provenance", return_value=None),
        patch("tools.hitl_yolo_finetuner_app.export.load_latest_run") as mock_run,
        patch("tools.hitl_yolo_finetuner_app.export.set_provenance") as mock_set_prov,
        patch("tools.hitl_yolo_finetuner_app.export.register_latest_run"),
    ):
        mock_run.return_value = {
            "run_dir": "/mock/test_doc_v01",
            "artifacts": {"visuals_dir": str(images_dir)},
        }
        report = export_dataset(
            doc_stem="test_doc",
            db_file=db_file,
            output_dir=out_root,
            min_train_pages=0,
            min_val_pages=0,
            copy_mode="copy",
            force_provenance=True,
        )

    assert report["total_exported"] == 1
    mock_set_prov.assert_called_once()


def test_export_dataset_fails_on_doc_stem_mismatch(tmp_path: Path):
    """Reject export when DB provenance doc_stem does not match request."""
    db_file = tmp_path / "synthetic.sqlite3"
    create_synthetic_db(db_file, num_pages=1)

    with (
        patch("tools.hitl_yolo_finetuner_app.export.get_provenance") as mock_prov,
        patch("tools.hitl_yolo_finetuner_app.export.load_latest_run") as mock_run,
        pytest.raises(ValueError, match="does not match requested"),
    ):
        mock_prov.return_value = {
            "source_stage": "layout-infer",
            "doc_stem": "test_doc",
            "source_run_dir": "/mock/test_doc_v01",
        }
        mock_run.return_value = {
            "run_dir": "/mock/different_doc_v01",
            "artifacts": {"visuals_dir": "/mock/visuals"},
        }

        export_dataset(
            doc_stem="different_doc",
            db_file=db_file,
            output_dir=tmp_path / "export_output",
        )


def test_export_dataset_idempotency(tmp_path: Path):
    """Export reports should be deterministic for identical inputs/config."""
    db_file = tmp_path / "synthetic.sqlite3"
    create_synthetic_db(db_file, num_pages=100)

    images_dir = tmp_path / "source_images"
    images_dir.mkdir()
    for i in range(100):
        img = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
        img.save(images_dir / f"page_{i:03d}.jpg")

    out_root = tmp_path / "export_output"

    with (
        patch("tools.hitl_yolo_finetuner_app.export.get_provenance") as mock_prov,
        patch("tools.hitl_yolo_finetuner_app.export.load_latest_run") as mock_run,
        patch("tools.hitl_yolo_finetuner_app.export.register_latest_run"),
    ):
        mock_prov.return_value = {
            "source_stage": "layout-infer",
            "doc_stem": "test_doc",
            "source_run_dir": "/mock/run/dir",
        }
        mock_run.return_value = {
            "run_dir": "/mock/run/dir",
            "artifacts": {"visuals_dir": str(images_dir)},
        }

        export_dataset(
            doc_stem="test_doc",
            db_file=db_file,
            output_dir=out_root,
            split_seed=42,
            min_train_pages=20,
            min_val_pages=5,
        )
        export_dataset(
            doc_stem="test_doc",
            db_file=db_file,
            output_dir=out_root,
            split_seed=42,
            min_train_pages=20,
            min_val_pages=5,
        )

    run_1_data = out_root / "test_doc_v01" / "data"
    run_2_data = out_root / "test_doc_v02" / "data"

    manifest_1 = (run_1_data / "export_report.json").read_text()
    manifest_2 = (run_2_data / "export_report.json").read_text()
    report_1 = json.loads(manifest_1)
    report_2 = json.loads(manifest_2)
    report_1_norm = copy.deepcopy(report_1)
    report_2_norm = copy.deepcopy(report_2)
    report_1_norm.pop("run_dir", None)
    report_2_norm.pop("run_dir", None)
    assert report_1_norm == report_2_norm


def test_export_dataset_vertical_clip_filters_shortened_lines(tmp_path: Path):
    """Aggressive clipping should mark candidates invalid and export zero pages in dry-run."""
    db_file = tmp_path / "synthetic.sqlite3"
    create_synthetic_db(db_file, num_pages=1)

    images_dir = tmp_path / "source_images"
    images_dir.mkdir()
    img = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
    img.save(images_dir / "page_000.jpg")

    with (
        patch("tools.hitl_yolo_finetuner_app.export.get_provenance") as mock_prov,
        patch("tools.hitl_yolo_finetuner_app.export.load_latest_run") as mock_run,
    ):
        mock_prov.return_value = {
            "source_stage": "layout-infer",
            "doc_stem": "test_doc",
            "source_run_dir": "/mock/run/dir",
        }
        mock_run.return_value = {
            "run_dir": "/mock/run/dir",
            "artifacts": {"visuals_dir": str(images_dir)},
        }

        report = export_dataset(
            doc_stem="test_doc",
            db_file=db_file,
            output_dir=tmp_path / "export_output",
            min_train_pages=0,
            min_val_pages=0,
            min_line_length_px=100.0,
            clip_top_px=475.0,
            clip_bottom_px=475.0,
            dry_run=True,
            copy_mode="copy",
        )

    assert report["total_exported"] == 0
    assert report["skips"]["invalid_geometry"] == 1


def test_export_dataset_omit_pages_excludes_selected_ids(tmp_path: Path):
    """Omit-pages should filter matching page_### ids before export."""
    db_file = tmp_path / "synthetic.sqlite3"
    create_synthetic_db(db_file, num_pages=3)

    images_dir = tmp_path / "source_images"
    images_dir.mkdir()
    for i in range(3):
        img = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
        img.save(images_dir / f"page_{i:03d}.jpg")

    with (
        patch("tools.hitl_yolo_finetuner_app.export.get_provenance") as mock_prov,
        patch("tools.hitl_yolo_finetuner_app.export.load_latest_run") as mock_run,
    ):
        mock_prov.return_value = {
            "source_stage": "layout-infer",
            "doc_stem": "test_doc",
            "source_run_dir": "/mock/run/dir",
        }
        mock_run.return_value = {
            "run_dir": "/mock/run/dir",
            "artifacts": {"visuals_dir": str(images_dir)},
        }

        report = export_dataset(
            doc_stem="test_doc",
            db_file=db_file,
            output_dir=tmp_path / "export_output",
            omit_pages="1-2",
            min_train_pages=0,
            min_val_pages=0,
            dry_run=True,
            copy_mode="copy",
        )

    assert report["total_exported"] == 1
    assert report["skips"]["omitted"] == 2
