from pathlib import Path

from modules.ocr_training.checkpointing import (
    resolve_latest_checkpoint,
    update_best_checkpoint_pointer,
    write_resume_state,
)


def test_resolve_latest_checkpoint(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True)
    (out / "checkpoint-10").mkdir()
    (out / "checkpoint-2").mkdir()
    latest = resolve_latest_checkpoint(out)
    assert latest is not None
    assert latest.name == "checkpoint-10"


def test_update_best_checkpoint_pointer_writes_meta_and_link(tmp_path: Path):
    out = tmp_path / "run"
    ckpt = out / "checkpoint-20"
    ckpt.mkdir(parents=True)
    meta_path = update_best_checkpoint_pointer(
        out,
        checkpoint_path=ckpt,
        metric_name="eval_cer",
        metric_value=0.12,
        global_step=20,
    )
    assert meta_path.exists()
    link_path = out / "weights" / "best_checkpoint"
    assert link_path.exists()


def test_write_resume_state(tmp_path: Path):
    out = tmp_path / "run"
    ckpt = out / "checkpoint-7"
    ckpt.mkdir(parents=True)
    state_path = write_resume_state(out, status="interrupted", latest_checkpoint=ckpt)
    assert state_path.exists()
    text = state_path.read_text(encoding="utf-8")
    assert "interrupted" in text
    assert "checkpoint-7" in text
