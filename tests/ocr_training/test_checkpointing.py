from pathlib import Path

from modules.ocr_training.checkpointing import (
    BestCerCheckpointCallback,
    TrainingSignalState,
    install_signal_handlers,
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


def test_resolve_latest_checkpoint_ignores_non_numeric_suffixes(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True)
    (out / "checkpoint-10").mkdir()
    (out / "checkpoint-emergency").mkdir()

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


def test_install_signal_handlers_ignores_child_process_signals(monkeypatch):
    handlers = {}
    warnings: list[str] = []
    state = TrainingSignalState()

    monkeypatch.setattr(
        "modules.ocr_training.checkpointing.signal.signal",
        lambda signum, handler: handlers.setdefault(signum, handler),
    )
    monkeypatch.setattr(
        "modules.ocr_training.checkpointing.logger.warning",
        lambda message, *args: warnings.append(message % args),
    )
    current_pid = {"value": 101}
    monkeypatch.setattr(
        "modules.ocr_training.checkpointing.os.getpid",
        lambda: current_pid["value"],
    )

    install_signal_handlers(state)

    handlers[2](2, None)
    assert state.interrupted is True
    assert len(warnings) == 1

    state.interrupted = False
    current_pid["value"] = 202
    handlers[2](2, None)
    assert state.interrupted is False
    assert len(warnings) == 1


def test_best_checkpoint_callback_updates_pointer_on_save(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True)
    callback = BestCerCheckpointCallback(out, metric_name="eval_cer")
    args = type("Args", (), {"output_dir": str(out)})()
    state = type("State", (), {"global_step": 4000})()
    control = object()

    callback.on_evaluate(args, state, control, metrics={"eval_cer": 0.42})
    (out / "checkpoint-4000").mkdir()
    callback.on_save(args, state, control)

    meta_path = out / "best_model_meta.json"
    assert meta_path.exists()
    text = meta_path.read_text(encoding="utf-8")
    assert "checkpoint-4000" in text
    assert "0.42" in text
