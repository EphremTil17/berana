import json
from pathlib import Path

from PIL import Image

from modules.ocr_training.failure_review import create_failure_review_tasks


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 16), color=255).save(path)


def test_create_failure_review_tasks_unions_sources_and_dedupes(tmp_path: Path):
    exact_false_dir = tmp_path / "analysis" / "exact_false"
    image_path = (
        tmp_path / "output" / "ocr_training_datasets" / "set" / "images" / "train" / "row.png"
    )
    _write_image(image_path)
    row = {
        "image": str(image_path),
        "gt_text": "gt text",
        "pred_text": "pred text",
        "cer": 0.9,
        "wer": 1.0,
        "exact": False,
        "modality": "typed",
        "classification": "text_present",
        "structural_classification": "text_present",
        "blank_reasons": [],
        "image_width": 32,
        "image_height": 16,
        "gt_len": 7,
        "pred_len": 9,
    }
    _write_jsonl(exact_false_dir / "cer_outliers_2std.jsonl", [row])
    _write_jsonl(exact_false_dir / "wer_outliers_2std.jsonl", [row])
    _write_jsonl(exact_false_dir / "likely_label_mismatch_predictions.jsonl", [row])

    summary = create_failure_review_tasks(
        exact_false_dir=exact_false_dir,
        output_dir=tmp_path / "label_studio",
    )

    assert summary["num_tasks"] == 1
    tasks = json.loads(
        (tmp_path / "label_studio" / "ocr_failure_review_tasks.json").read_text(encoding="utf-8")
    )
    assert len(tasks) == 1
    task = tasks[0]
    assert task["data"]["image"].startswith("/data/local-files/?d=")
    assert "cer_outlier_2std" in task["data"]["candidate_sources"]
    assert "wer_outlier_2std" in task["data"]["candidate_sources"]
    assert "likely_label_mismatch" in task["data"]["candidate_sources"]
