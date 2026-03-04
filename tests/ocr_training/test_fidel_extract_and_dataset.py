import csv
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

from config.settings import settings
from modules.ocr_training.fidel_extract import extract_fidel
from modules.ocr_training.schemas import SplitConfig
from modules.ocr_training.surya_dataset import build_surya_dataset


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (16, 8), color=(255, 255, 255))
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

        snapshot_rows = [
            json.loads(line)
            for line in result["source_snapshot"].read_text(encoding="utf-8").splitlines()
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
