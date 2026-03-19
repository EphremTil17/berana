import csv
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

from config.settings import settings
from modules.ocr_training.fidel_cleanup import cleanup_fidel_extracted
from modules.ocr_training.fidel_extract import extract_fidel
from modules.ocr_training.schemas import SplitConfig
from modules.ocr_training.surya_dataset import build_surya_dataset


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (16, 8), color=(255, 255, 255))
    image.save(path)


def _write_blank_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (64, 24), color=255).save(path)


def _write_text_like_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (128, 32), color=255)
    for x0 in range(8, 120, 20):
        for x in range(x0, x0 + 8):
            for y in range(8, 22):
                image.putpixel((x, y), 0)
    image.save(path)


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _zip_with_members(zip_path: Path, members: list[tuple[str, Path]]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for arcname, src in members:
            zf.write(src, arcname=arcname)
        zf.writestr("__MACOSX/._junk", b"junk")


def test_extract_fidel_excludes_handwritten_and_builds_snapshot():
    base = Path(tempfile.mkdtemp(prefix="ocr_training_test_", dir=settings.BASE_DIR))
    try:
        raw_root = base / "input" / "ocr_training" / "fidel" / "raw"
        extracted_root = base / "input" / "ocr_training" / "fidel" / "extracted"

        ds_root = raw_root / "fidel_dataset"
        syn_root = raw_root / "fidel_synthetic"

        # Source images for zip members
        src = base / "src"
        typed_a = src / "typed_a.png"
        typed_b = src / "typed_b.png"
        synth_a = src / "synth_train.png"
        hand_a = src / "hand_a.png"
        image_0 = src / "image_0_0.png"
        for path in [typed_a, typed_b, synth_a, hand_a, image_0]:
            _write_png(path)

        _write_csv(
            ds_root / "train_labels.csv",
            ["image_filename", "line_text", "type", "writer"],
            [
                ["typed_a.png", "typed train", "typed", ""],
                ["synth_train.png", "synthetic train", "synthetic", ""],
                ["hand_a.png", "hand train", "handwritten", "10"],
            ],
        )
        _write_csv(
            ds_root / "test_labels.csv",
            ["image_filename", "line_text", "type", "writer"],
            [
                ["typed_b.png", "typed test", "typed", ""],
                ["hdd_b.png", "hdd test", "hdd_rand", ""],
            ],
        )
        _write_csv(
            syn_root / "synthetic_labels.csv",
            ["images", "text"],
            [["image_0_0.png", "synthetic repo"]],
        )

        _zip_with_members(
            ds_root / "train.zip",
            [
                ("train/typed_a.png", typed_a),
                ("train/synth_train.png", synth_a),
                ("train/hand_a.png", hand_a),
            ],
        )
        _zip_with_members(
            ds_root / "test.zip",
            [
                ("test/typed_b.png", typed_b),
            ],
        )
        _zip_with_members(
            syn_root / "data.zip",
            [("data/image_0_0.png", image_0)],
        )

        result = extract_fidel(
            raw_root=raw_root,
            extracted_root=extracted_root,
            include_types={"typed", "synthetic"},
            exclude_types={"handwritten", "hdd", "hdd_18", "hdd_rand"},
            allow_missing_rate=0.01,
            workers=1,
            overwrite=False,
            dry_run=False,
        )

        typed_files = sorted((extracted_root / "typed").glob("*.png"))
        synthetic_files = sorted((extracted_root / "synthetic").glob("*.png"))
        assert len(typed_files) == 2
        assert len(synthetic_files) == 2

        source_snapshot = result["source_snapshot"]
        assert isinstance(source_snapshot, Path)
        snapshot_rows = [
            json.loads(line)
            for line in source_snapshot.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        excluded_rows = [row for row in snapshot_rows if row["excluded"]]
        assert any(row["normalized_type"] == "handwritten" for row in excluded_rows)

    finally:
        shutil.rmtree(base)


def test_build_surya_dataset_creates_train_val_holdout_jsonl():
    base = Path(tempfile.mkdtemp(prefix="ocr_training_test_", dir=settings.BASE_DIR))
    try:
        raw_root = base / "input" / "ocr_training" / "fidel" / "raw"
        extracted_root = base / "input" / "ocr_training" / "fidel" / "extracted"
        output_root = base / "output" / "ocr_training_datasets"

        ds_root = raw_root / "fidel_dataset"
        syn_root = raw_root / "fidel_synthetic"

        src = base / "src"
        files = []
        for name in ["typed_1.png", "typed_2.png", "typed_3.png", "typed_4.png", "image_1_0.png"]:
            p = src / name
            _write_png(p)
            files.append(p)

        _write_csv(
            ds_root / "train_labels.csv",
            ["image_filename", "line_text", "type", "writer"],
            [["typed_1.png", "t1", "typed", ""], ["typed_2.png", "t2", "typed", ""]],
        )
        _write_csv(
            ds_root / "test_labels.csv",
            ["image_filename", "line_text", "type", "writer"],
            [["typed_3.png", "t3", "typed", ""], ["typed_4.png", "t4", "typed", ""]],
        )
        _write_csv(
            syn_root / "synthetic_labels.csv",
            ["images", "text"],
            [["image_1_0.png", "s1"]],
        )

        _zip_with_members(
            ds_root / "train.zip",
            [("train/typed_1.png", files[0]), ("train/typed_2.png", files[1])],
        )
        _zip_with_members(
            ds_root / "test.zip",
            [("test/typed_3.png", files[2]), ("test/typed_4.png", files[3])],
        )
        _zip_with_members(syn_root / "data.zip", [("data/image_1_0.png", files[4])])

        extract_fidel(
            raw_root=raw_root,
            extracted_root=extracted_root,
            include_types={"typed", "synthetic"},
            exclude_types={"handwritten"},
            allow_missing_rate=0.01,
            workers=1,
            overwrite=False,
            dry_run=False,
        )

        run_dir = build_surya_dataset(
            extracted_root=extracted_root,
            output_root=output_root,
            dataset_name="fidel_test",
            split_config=SplitConfig(
                train_ratio=0.6,
                val_ratio=0.2,
                holdout_ratio=0.2,
                seed=42,
                strict_page_isolation=False,
            ),
        )

        hf_root = run_dir / "data" / "hf_dataset"
        assert (hf_root / "train.jsonl").exists()
        assert (hf_root / "val.jsonl").exists()
        assert (hf_root / "holdout.jsonl").exists()

        total = 0
        for split_name in ["train", "val", "holdout"]:
            lines = [
                line
                for line in (hf_root / f"{split_name}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            total += len(lines)
        assert total == 5

    finally:
        shutil.rmtree(base)


def test_build_surya_dataset_excludes_suspects_by_default():
    base = Path(tempfile.mkdtemp(prefix="ocr_training_test_", dir=settings.BASE_DIR))
    try:
        extracted_root = base / "input" / "ocr_training" / "fidel" / "extracted"
        manifests_root = (
            base / "input" / "ocr_training" / "fidel" / "manifests" / "source_snapshots"
        )
        output_root = base / "output" / "ocr_training_datasets"
        typed_img = extracted_root / "typed" / "good.png"
        suspect_img = extracted_root / "synthetic" / "suspect.png"
        _write_text_like_image(typed_img)
        _write_blank_image(suspect_img)
        manifests_root.mkdir(parents=True, exist_ok=True)
        (manifests_root / "fidel_sources.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "sample_id": "good",
                            "source_repo": "fidel_dataset",
                            "source_split": "train",
                            "original_filename": "good.png",
                            "normalized_type": "typed",
                            "text_raw": "good",
                            "text_normalized": "good",
                            "image_relpath": os.path.relpath(typed_img, Path.cwd()),
                            "excluded": False,
                            "excluded_reason": None,
                        }
                    ),
                    json.dumps(
                        {
                            "sample_id": "suspect",
                            "source_repo": "fidel_synthetic",
                            "source_split": "synthetic",
                            "original_filename": "suspect.png",
                            "normalized_type": "synthetic",
                            "text_raw": "suspect",
                            "text_normalized": "suspect",
                            "image_relpath": os.path.relpath(suspect_img, Path.cwd()),
                            "excluded": False,
                            "excluded_reason": None,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        cleaned_root = base / "input" / "ocr_training" / "fidel_cleaned"
        cleanup_fidel_extracted(
            extracted_root=extracted_root,
            output_root=cleaned_root,
            workers=1,
        )

        run_dir = build_surya_dataset(
            extracted_root=cleaned_root / "extracted",
            output_root=output_root,
            dataset_name="fidel_cleaned_test",
            split_config=SplitConfig(
                train_ratio=0.5,
                val_ratio=0.25,
                holdout_ratio=0.25,
                seed=42,
                strict_page_isolation=False,
            ),
        )

        records = []
        hf_root = run_dir / "data" / "hf_dataset"
        for split_name in ["train", "val", "holdout"]:
            records.extend(
                json.loads(line)
                for line in (hf_root / f"{split_name}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            )
        assert len(records) == 1
        assert records[0]["text"] == "good"
    finally:
        shutil.rmtree(base)


def test_build_surya_dataset_include_suspect_uses_remaining_review_files():
    base = Path(tempfile.mkdtemp(prefix="ocr_training_test_", dir=settings.BASE_DIR))
    try:
        extracted_root = base / "input" / "ocr_training" / "fidel" / "extracted"
        manifests_root = (
            base / "input" / "ocr_training" / "fidel" / "manifests" / "source_snapshots"
        )
        output_root = base / "output" / "ocr_training_datasets"
        typed_img = extracted_root / "typed" / "good.png"
        suspect_img = extracted_root / "synthetic" / "suspect.png"
        _write_text_like_image(typed_img)
        _write_blank_image(suspect_img)
        manifests_root.mkdir(parents=True, exist_ok=True)
        (manifests_root / "fidel_sources.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "sample_id": "good",
                            "source_repo": "fidel_dataset",
                            "source_split": "train",
                            "original_filename": "good.png",
                            "normalized_type": "typed",
                            "text_raw": "good",
                            "text_normalized": "good",
                            "image_relpath": os.path.relpath(typed_img, Path.cwd()),
                            "excluded": False,
                            "excluded_reason": None,
                        }
                    ),
                    json.dumps(
                        {
                            "sample_id": "suspect",
                            "source_repo": "fidel_synthetic",
                            "source_split": "synthetic",
                            "original_filename": "suspect.png",
                            "normalized_type": "synthetic",
                            "text_raw": "suspect",
                            "text_normalized": "suspect",
                            "image_relpath": os.path.relpath(suspect_img, Path.cwd()),
                            "excluded": False,
                            "excluded_reason": None,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        cleaned_root = base / "input" / "ocr_training" / "fidel_cleaned"
        cleanup_fidel_extracted(
            extracted_root=extracted_root,
            output_root=cleaned_root,
            workers=1,
        )

        run_dir = build_surya_dataset(
            extracted_root=cleaned_root / "extracted",
            output_root=output_root,
            dataset_name="fidel_cleaned_test",
            split_config=SplitConfig(
                train_ratio=0.5,
                val_ratio=0.25,
                holdout_ratio=0.25,
                seed=42,
                strict_page_isolation=False,
            ),
            include_suspect=True,
        )

        records = []
        hf_root = run_dir / "data" / "hf_dataset"
        for split_name in ["train", "val", "holdout"]:
            records.extend(
                json.loads(line)
                for line in (hf_root / f"{split_name}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            )
        assert len(records) == 2
        assert sorted(record["text"] for record in records) == ["good", "suspect"]
    finally:
        shutil.rmtree(base)


def test_build_surya_dataset_include_suspect_respects_pruned_review_folder():
    base = Path(tempfile.mkdtemp(prefix="ocr_training_test_", dir=settings.BASE_DIR))
    try:
        extracted_root = base / "input" / "ocr_training" / "fidel" / "extracted"
        manifests_root = (
            base / "input" / "ocr_training" / "fidel" / "manifests" / "source_snapshots"
        )
        output_root = base / "output" / "ocr_training_datasets"
        typed_img = extracted_root / "typed" / "good.png"
        suspect_img = extracted_root / "synthetic" / "suspect.png"
        _write_text_like_image(typed_img)
        _write_blank_image(suspect_img)
        manifests_root.mkdir(parents=True, exist_ok=True)
        (manifests_root / "fidel_sources.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "sample_id": "good",
                            "source_repo": "fidel_dataset",
                            "source_split": "train",
                            "original_filename": "good.png",
                            "normalized_type": "typed",
                            "text_raw": "good",
                            "text_normalized": "good",
                            "image_relpath": os.path.relpath(typed_img, Path.cwd()),
                            "excluded": False,
                            "excluded_reason": None,
                        }
                    ),
                    json.dumps(
                        {
                            "sample_id": "suspect",
                            "source_repo": "fidel_synthetic",
                            "source_split": "synthetic",
                            "original_filename": "suspect.png",
                            "normalized_type": "synthetic",
                            "text_raw": "suspect",
                            "text_normalized": "suspect",
                            "image_relpath": os.path.relpath(suspect_img, Path.cwd()),
                            "excluded": False,
                            "excluded_reason": None,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        cleaned_root = base / "input" / "ocr_training" / "fidel_cleaned"
        cleanup_fidel_extracted(
            extracted_root=extracted_root,
            output_root=cleaned_root,
            workers=1,
        )

        suspect_review_copy = cleaned_root / "suspect_blank_images" / suspect_img.name
        assert suspect_review_copy.exists()
        suspect_review_copy.unlink()

        run_dir = build_surya_dataset(
            extracted_root=cleaned_root / "extracted",
            output_root=output_root,
            dataset_name="fidel_cleaned_test",
            split_config=SplitConfig(
                train_ratio=0.5,
                val_ratio=0.25,
                holdout_ratio=0.25,
                seed=42,
                strict_page_isolation=False,
            ),
            include_suspect=True,
        )

        records = []
        hf_root = run_dir / "data" / "hf_dataset"
        for split_name in ["train", "val", "holdout"]:
            records.extend(
                json.loads(line)
                for line in (hf_root / f"{split_name}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            )
        assert len(records) == 1
        assert records[0]["text"] == "good"
    finally:
        shutil.rmtree(base)


def test_cleanup_fidel_excludes_rows_from_heuristic_cleanup_dir():
    base = Path(tempfile.mkdtemp(prefix="ocr_training_test_", dir=settings.BASE_DIR))
    try:
        extracted_root = base / "input" / "ocr_training" / "fidel" / "extracted"
        manifests_root = (
            base / "input" / "ocr_training" / "fidel" / "manifests" / "source_snapshots"
        )
        output_root = base / "input" / "ocr_training" / "fidel_cleaned"
        dataset_output_root = base / "output" / "ocr_training_datasets"

        good_img = extracted_root / "typed" / "good.png"
        bad_img = extracted_root / "typed" / "bad.png"
        _write_text_like_image(good_img)
        _write_text_like_image(bad_img)
        manifests_root.mkdir(parents=True, exist_ok=True)
        (manifests_root / "fidel_sources.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "sample_id": "good:typed",
                            "source_repo": "fidel_dataset",
                            "source_split": "train",
                            "original_filename": "good.png",
                            "normalized_type": "typed",
                            "text_raw": "good",
                            "text_normalized": "good",
                            "image_relpath": os.path.relpath(good_img, Path.cwd()),
                            "excluded": False,
                            "excluded_reason": None,
                        }
                    ),
                    json.dumps(
                        {
                            "sample_id": "bad:typed",
                            "source_repo": "fidel_dataset",
                            "source_split": "train",
                            "original_filename": "bad.png",
                            "normalized_type": "typed",
                            "text_raw": "bad",
                            "text_normalized": "bad",
                            "image_relpath": os.path.relpath(bad_img, Path.cwd()),
                            "excluded": False,
                            "excluded_reason": None,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        heuristic_cleanup_dir = base / "heuristics" / "exact_false"
        heuristic_cleanup_dir.mkdir(parents=True, exist_ok=True)
        built_bad_name = "bad__typed__bad.png"
        # sample_id bad:typed -> safe id bad__typed, final built name bad__typed__bad.png
        (heuristic_cleanup_dir / "likely_label_mismatch_predictions.jsonl").write_text(
            json.dumps(
                {
                    "image": str(
                        base
                        / "output"
                        / "ocr_training_datasets"
                        / "dummy"
                        / "data"
                        / "hf_dataset"
                        / "images"
                        / "train"
                        / built_bad_name
                    ),
                    "gt_text": "bad",
                    "pred_text": "wrong",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        summary = cleanup_fidel_extracted(
            extracted_root=extracted_root,
            output_root=output_root,
            workers=1,
            heuristic_cleanup_dir=heuristic_cleanup_dir,
        )

        assert summary["heuristic_excluded_rows"] == 1
        heuristic_by_category = summary["heuristic_excluded_rows_by_category"]
        assert isinstance(heuristic_by_category, dict)
        assert heuristic_by_category["likely_label_mismatch"] == 1

        cleaned_snapshot = (
            output_root / "manifests" / "source_snapshots" / "fidel_sources.jsonl"
        ).read_text(encoding="utf-8")
        assert "heuristic_exclusion_after_fidel_cleanup:likely_label_mismatch" in cleaned_snapshot

        run_dir = build_surya_dataset(
            extracted_root=output_root / "extracted",
            output_root=dataset_output_root,
            dataset_name="fidel_heuristic_cleaned_test",
            split_config=SplitConfig(
                train_ratio=0.5,
                val_ratio=0.25,
                holdout_ratio=0.25,
                seed=42,
                strict_page_isolation=False,
            ),
        )

        records = []
        hf_root = run_dir / "data" / "hf_dataset"
        for split_name in ["train", "val", "holdout"]:
            records.extend(
                json.loads(line)
                for line in (hf_root / f"{split_name}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            )
        assert len(records) == 1
        assert records[0]["text"] == "good"
    finally:
        shutil.rmtree(base)
