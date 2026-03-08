import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from modules.ocr_training.surya_eval import evaluate_surya_checkpoint
from modules.ocr_training.surya_reports import (
    write_confusion_artifacts,
    write_training_history_artifacts,
)


def _write_split(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 8), color=(255, 255, 255)).save(path)


def test_evaluate_surya_checkpoint_batches_inference(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    rows = []
    for index in range(4):
        image_path = dataset_dir / "images" / f"sample_{index}.png"
        _write_png(image_path)
        rows.append({"image": str(image_path), "text": f"text-{index}"})
    _write_split(dataset_dir / "holdout.jsonl", rows)

    call_sizes: list[int] = []

    class DummyPredictor:
        def __init__(self):
            self.disable_tqdm = False

        def __call__(self, images, **kwargs):
            call_sizes.append(len(images))
            return [
                SimpleNamespace(text_lines=[SimpleNamespace(text=f"text-{offset}")])
                for offset, _image in enumerate(images, start=sum(call_sizes[:-1]))
            ]

    summary = evaluate_surya_checkpoint(
        run_key="fidel_typed_synthetic",
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="holdout",
        eval_fraction=1.0,
        eval_batch_size=2,
        max_rows=None,
        seed=42,
        runtime={
            "RecognitionPredictor": lambda _foundation: DummyPredictor(),
            "TaskNames": SimpleNamespace(ocr_with_boxes="ocr"),
        },
        load_surya_eval_predictor=lambda runtime, run_dir: object(),
    )

    assert call_sizes == [2, 2]
    assert summary["num_rows"] == 4
    assert summary["mean_cer"] == 0.0
    assert summary["mean_wer"] == 0.0


def test_write_confusion_artifacts_outputs_top_pairs(tmp_path: Path):
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    artifacts = write_confusion_artifacts(
        eval_dir=eval_dir,
        split="holdout",
        records=[
            {"gt_text": "abc", "pred_text": "axc"},
            {"gt_text": "abc", "pred_text": "axc"},
        ],
    )

    payload = json.loads(artifacts["character_confusions_json"].read_text(encoding="utf-8"))
    assert payload[0] == {"gt": "b", "pred": "x", "count": 2}
    assert artifacts["character_confusions_md"].exists()


def test_write_training_history_artifacts_writes_csv_and_svg(tmp_path: Path):
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    trainer_state = {
        "log_history": [
            {"step": 10, "loss": 1.2, "epoch": 0.1},
            {"step": 20, "eval_loss": 0.4, "eval_cer": 0.3, "eval_wer": 0.5, "epoch": 0.2},
        ]
    }
    (run_dir / "trainer_state.json").write_text(json.dumps(trainer_state), encoding="utf-8")

    artifacts = write_training_history_artifacts(run_dir=run_dir, eval_dir=eval_dir)

    assert artifacts["training_history_csv"].exists()
    assert artifacts["training_curves_svg"].exists()
    assert "Eval CER" in artifacts["training_curves_svg"].read_text(encoding="utf-8")
