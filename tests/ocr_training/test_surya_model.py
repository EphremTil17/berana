from pathlib import Path

from modules.ocr_training.surya_model import resolve_eval_adapter_source


def test_resolve_eval_adapter_source_prefers_best_cer(tmp_path: Path):
    run_dir = tmp_path / "run"
    best_dir = run_dir / "weights" / "best_checkpoint"
    best_dir.mkdir(parents=True)

    resolved = resolve_eval_adapter_source(run_dir, checkpoint_target="best_cer")

    assert resolved == best_dir.resolve()


def test_resolve_eval_adapter_source_uses_best_wer(tmp_path: Path):
    run_dir = tmp_path / "run"
    best_dir = run_dir / "weights" / "best_checkpoint_wer"
    best_dir.mkdir(parents=True)

    resolved = resolve_eval_adapter_source(run_dir, checkpoint_target="best_wer")

    assert resolved == best_dir.resolve()


def test_resolve_eval_adapter_source_uses_latest_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoint-20").mkdir(parents=True)
    (run_dir / "checkpoint-200").mkdir()

    resolved = resolve_eval_adapter_source(run_dir, checkpoint_target="latest")

    assert resolved == (run_dir / "checkpoint-200")
