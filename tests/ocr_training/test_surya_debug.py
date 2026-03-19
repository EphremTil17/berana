import json
import os
from pathlib import Path
from typing import TypedDict, cast

from PIL import Image, ImageDraw

from modules.ocr_training.fidel_cleanup import cleanup_fidel_extracted
from modules.ocr_training.surya_cleanup import verify_surya_dataset
from modules.ocr_training.surya_debug import BLANK_SIGNATURE_SIZE, extract_exact_false_debug_bundle


class _CategorySummary(TypedDict):
    num_rows: int


class _SignalOverlap(TypedDict):
    structural_only: int


class _ExactFalseSummary(TypedDict):
    num_rows: int
    signal_overlap: _SignalOverlap


class _DebugBundleSummary(TypedDict):
    exact_false: _ExactFalseSummary
    confirmed_blank: _CategorySummary
    suspect_blank: _CategorySummary
    original_summary: _CategorySummary
    summary_excluding_confirmed_blank: _CategorySummary


class _FidelCleanupSummary(TypedDict):
    excluded_rows: int


class _SuryaVerifySummary(TypedDict):
    confirmed_blank_rows: int


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_blank_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (64, 24), color=255).save(path)


def _write_signature_blank_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", BLANK_SIGNATURE_SIZE, color=255).save(path)


def _write_text_like_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (128, 32), color=255)
    draw = ImageDraw.Draw(image)
    for x0 in range(8, 120, 20):
        draw.rectangle((x0, 8, x0 + 8, 22), fill=0)
    image.save(path)


def test_extract_exact_false_debug_bundle_writes_confirmed_and_suspect_artifacts(tmp_path: Path):
    img_text = tmp_path / "images" / "text.png"
    img_blank = tmp_path / "images" / "blank.png"
    _write_text_like_image(img_text)
    _write_blank_image(img_blank)
    predictions_path = tmp_path / "predictions_holdout.jsonl"
    _write_predictions(
        predictions_path,
        [
            {
                "image": str(img_text),
                "gt_text": "abc",
                "pred_text": "abc",
                "cer": 0.0,
                "wer": 0.0,
                "exact": True,
            },
            {
                "image": str(img_blank),
                "gt_text": "xyz",
                "pred_text": "",
                "cer": 1.0,
                "wer": 1.0,
                "exact": False,
            },
            {
                "image": str(img_text),
                "gt_text": "abc",
                "pred_text": "xbc",
                "cer": 0.1,
                "wer": 0.2,
                "exact": False,
            },
        ],
    )

    summary = cast(
        _DebugBundleSummary,
        extract_exact_false_debug_bundle(
            predictions_path=predictions_path,
            output_dir=tmp_path / "debug",
        ),
    )

    assert summary["exact_false"]["num_rows"] == 2
    assert summary["confirmed_blank"]["num_rows"] == 0
    assert summary["suspect_blank"]["num_rows"] == 1
    assert summary["original_summary"]["num_rows"] == 3
    assert summary["summary_excluding_confirmed_blank"]["num_rows"] == 3
    assert (tmp_path / "debug" / "exact_false_predictions.json").exists()
    assert (tmp_path / "debug" / "exact_false_predictions.jsonl").exists()
    assert (tmp_path / "debug" / "confirmed_blank_predictions.jsonl").exists()
    assert (tmp_path / "debug" / "suspect_blank_predictions.jsonl").exists()
    assert (tmp_path / "debug" / "predictions_excluding_confirmed_blank.jsonl").exists()
    assert (tmp_path / "debug" / "images" / img_blank.name).exists()
    assert not (tmp_path / "debug" / "confirmed_blank_images" / img_blank.name).exists()
    assert summary["exact_false"]["signal_overlap"]["structural_only"] == 1


def test_extract_exact_false_debug_bundle_requires_signature_for_confirmed_blank(tmp_path: Path):
    img_blank = tmp_path / "images" / "signature_blank.png"
    _write_signature_blank_image(img_blank)
    predictions_path = tmp_path / "predictions_holdout.jsonl"
    _write_predictions(
        predictions_path,
        [
            {
                "image": str(img_blank),
                "gt_text": "xyz",
                "pred_text": "",
                "cer": 1.0,
                "wer": 1.0,
                "exact": False,
            },
        ],
    )

    summary = cast(
        _DebugBundleSummary,
        extract_exact_false_debug_bundle(
            predictions_path=predictions_path,
            output_dir=tmp_path / "debug",
        ),
    )

    assert summary["confirmed_blank"]["num_rows"] == 1
    assert (tmp_path / "debug" / "confirmed_blank_images" / img_blank.name).exists()


def test_cleanup_fidel_extracted_creates_filtered_snapshot_and_copy(tmp_path: Path):
    extracted_root = tmp_path / "fidel" / "extracted"
    manifests_root = tmp_path / "fidel" / "manifests" / "source_snapshots"
    good_img = extracted_root / "typed" / "good.png"
    bad_img = extracted_root / "synthetic" / "bad.png"
    _write_text_like_image(good_img)
    _write_signature_blank_image(bad_img)
    manifests_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = manifests_root / "fidel_sources.jsonl"
    snapshot_path.write_text(
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
                        "image_relpath": os.path.relpath(good_img, Path.cwd()),
                        "excluded": False,
                        "excluded_reason": None,
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "bad",
                        "source_repo": "fidel_synthetic",
                        "source_split": "synthetic",
                        "original_filename": "bad.png",
                        "normalized_type": "synthetic",
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

    summary = cast(
        _FidelCleanupSummary,
        cleanup_fidel_extracted(
            extracted_root=extracted_root,
            output_root=tmp_path / "cleaned_fidel",
        ),
    )

    cleaned_snapshot = (
        tmp_path / "cleaned_fidel" / "manifests" / "source_snapshots" / "fidel_sources.jsonl"
    ).read_text(encoding="utf-8")
    assert summary["excluded_rows"] == 1
    assert (tmp_path / "cleaned_fidel" / "extracted" / "typed" / good_img.name).exists()
    assert not (tmp_path / "cleaned_fidel" / "extracted" / "synthetic" / bad_img.name).exists()
    assert (tmp_path / "cleaned_fidel" / "excluded_blank_images" / bad_img.name).exists()
    assert "confirmed_blank_after_fidel_cleanup" in cleaned_snapshot


def test_verify_surya_dataset_emits_review_artifacts(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    good_img = dataset_dir / "images" / "train" / "good.png"
    bad_img = dataset_dir / "images" / "train" / "bad.png"
    val_img = dataset_dir / "images" / "val" / "val.png"
    holdout_img = dataset_dir / "images" / "holdout" / "holdout.png"
    _write_text_like_image(good_img)
    _write_signature_blank_image(bad_img)
    _write_text_like_image(val_img)
    _write_text_like_image(holdout_img)
    _write_predictions(
        dataset_dir / "train.jsonl",
        [
            {"image": str(good_img), "text": "good"},
            {"image": str(bad_img), "text": "bad"},
        ],
    )
    _write_predictions(dataset_dir / "val.jsonl", [{"image": str(val_img), "text": "val"}])
    _write_predictions(
        dataset_dir / "holdout.jsonl",
        [{"image": str(holdout_img), "text": "holdout"}],
    )

    summary = cast(
        _SuryaVerifySummary,
        verify_surya_dataset(
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "verify",
        ),
    )

    assert summary["confirmed_blank_rows"] == 1
    assert (tmp_path / "verify" / "confirmed_blank_rows.jsonl").exists()
    assert (tmp_path / "verify" / "confirmed_blank_images" / bad_img.name).exists()
