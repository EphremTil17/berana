from pathlib import Path

from modules.ocr_training import surya_train


def test_evaluate_surya_checkpoint_wrapper_forwards_extended_eval_args(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr("modules.ocr_training.surya_train.require_surya", lambda: {"runtime": True})

    def _capture_eval(**kwargs):
        captured.update(kwargs)
        return {"mean_cer": 0.1, "mean_wer": 0.2, "num_rows": 10}

    monkeypatch.setattr(
        "modules.ocr_training.surya_train._evaluate_surya_checkpoint", _capture_eval
    )

    summary = surya_train.evaluate_surya_checkpoint(
        run_key="fidel_typed_synthetic",
        run_dir=Path("/tmp/run"),
        dataset_dir=Path("/tmp/dataset"),
        split="holdout",
        eval_fraction=0.25,
        eval_batch_size=16,
        max_rows=500,
        seed=7,
    )

    assert summary["mean_cer"] == 0.1
    assert captured["eval_fraction"] == 0.25
    assert captured["eval_batch_size"] == 16
    assert captured["max_rows"] == 500
    assert captured["seed"] == 7
