import json
from pathlib import Path

from PIL import Image, ImageDraw

from modules.ocr_training.failure_analysis import analyze_predictions_failures
from modules.ocr_training.surya_debug import BLANK_SIGNATURE_SIZE


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


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


def test_analyze_predictions_failures_writes_exact_false_bundle(tmp_path: Path):
    text_img = tmp_path / "images" / "text.png"
    blank_img = tmp_path / "images" / "blank.png"
    missing_img = tmp_path / "images" / "missing.png"
    _write_text_like_image(text_img)
    _write_signature_blank_image(blank_img)
    predictions_path = tmp_path / "predictions_train.jsonl"
    _write_predictions(
        predictions_path,
        [
            {
                "image": str(text_img),
                "gt_text": "short good",
                "pred_text": "short good",
                "cer": 0.0,
                "wer": 0.0,
                "exact": True,
            },
            {
                "image": str(blank_img),
                "gt_text": "this should be excluded",
                "pred_text": "",
                "cer": 1.0,
                "wer": 1.0,
                "exact": False,
            },
            {
                "image": str(text_img),
                "gt_text": "a" * 60,
                "pred_text": "b" * 60,
                "cer": 0.95,
                "wer": 1.0,
                "exact": False,
            },
            {
                "image": str(missing_img),
                "gt_text": "missing image row",
                "pred_text": "missing pred row",
                "cer": 0.9,
                "wer": 1.0,
                "exact": False,
            },
        ],
    )

    summary = analyze_predictions_failures(
        predictions_path=predictions_path,
        output_dir=tmp_path / "analysis",
    )

    exact_false_dir = tmp_path / "analysis" / "exact_false"
    assert summary["num_rows"] == 4
    assert summary["exact_false_count"] == 3
    assert summary["exact_false"]["confirmed_blank"] == 1
    assert summary["exact_false"]["likely_label_mismatch"] == 1
    assert summary["exact_false"]["missing_images"] == 1
    assert (exact_false_dir / "exact_false_predictions.jsonl").exists()
    assert (exact_false_dir / "confirmed_blank_predictions.jsonl").exists()
    assert (exact_false_dir / "likely_label_mismatch_predictions.jsonl").exists()
    assert (exact_false_dir / "missing_image_predictions.jsonl").exists()
    assert (exact_false_dir / "images" / text_img.name).exists()
    assert (exact_false_dir / "confirmed_blank_images" / blank_img.name).exists()
