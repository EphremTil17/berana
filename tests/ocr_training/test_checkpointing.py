from pathlib import Path

from modules.ocr_training.checkpointing import (
    BestCerCheckpointCallback,
    BestWerCheckpointCallback,
    PlateauWarningCallback,
    TrainingSignalState,
    TrainingTerminationCoordinator,
    install_signal_handlers,
    observe_training_stop,
    request_training_stop,
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
    (ckpt / "trainer_state.json").write_text("{}", encoding="utf-8")
    meta_path = update_best_checkpoint_pointer(
        out,
        checkpoint_path=ckpt,
        metric_name="eval_cer",
        metric_value=0.12,
        global_step=20,
    )
    assert meta_path.exists()
    best_dir = out / "weights" / "best_checkpoint"
    assert best_dir.exists()
    assert best_dir.is_dir()
    assert (best_dir / "trainer_state.json").exists()


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
    assert "source_checkpoint" in text
    assert "0.42" in text


def test_best_wer_checkpoint_callback_writes_separate_meta(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True)
    callback = BestWerCheckpointCallback(
        out,
        metric_name="eval_wer",
        weights_subdir="best_checkpoint_wer",
        meta_filename="best_wer_model_meta.json",
    )
    args = type("Args", (), {"output_dir": str(out)})()
    state = type("State", (), {"global_step": 4000})()
    control = object()

    callback.on_evaluate(args, state, control, metrics={"eval_wer": 0.25})
    (out / "checkpoint-4000").mkdir()
    callback.on_save(args, state, control)

    meta_path = out / "best_wer_model_meta.json"
    assert meta_path.exists()
    text = meta_path.read_text(encoding="utf-8")
    assert "eval_wer" in text
    assert "0.25" in text
    assert (out / "weights" / "best_checkpoint_wer").exists()


def test_request_and_observe_training_stop_roundtrip(tmp_path: Path):
    coordinator = TrainingTerminationCoordinator(tmp_path)
    source_state = TrainingSignalState()
    peer_state = TrainingSignalState()

    request_training_stop(
        source_state,
        reason="VRAM guard triggered: GPU 1",
        stop_type="runtime",
        save_checkpoint=False,
        coordinator=coordinator,
        rank=1,
    )
    observe_training_stop(peer_state, coordinator=coordinator)

    assert source_state.runtime_error_message == "VRAM guard triggered: GPU 1"
    assert peer_state.stop_requested is True
    assert peer_state.runtime_error_message == "VRAM guard triggered: GPU 1"
    assert peer_state.save_checkpoint_on_stop is False


def test_plateau_warning_callback_warns_without_stopping(monkeypatch, tmp_path: Path):
    warnings: list[str] = []
    callback = PlateauWarningCallback(
        tmp_path,
        min_evals=3,
        patience_evals=3,
        cer_tolerance=0.01,
        wer_tolerance=0.02,
    )
    monkeypatch.setattr(
        "modules.ocr_training.checkpointing.logger.warning",
        lambda message, *args: warnings.append(message % args),
    )
    args = object()
    control = type("Control", (), {"should_training_stop": False})()
    state = type(
        "State",
        (),
        {
            "global_step": 100,
            "log_history": [
                {"step": 20, "loss": 1.1},
                {"step": 40, "loss": 1.0},
                {"step": 60, "loss": 0.9},
            ],
        },
    )()

    callback.on_evaluate(args, state, control, metrics={"eval_cer": 0.30, "eval_wer": 0.50})
    state.global_step = 200
    callback.on_evaluate(args, state, control, metrics={"eval_cer": 0.31, "eval_wer": 0.51})
    state.global_step = 300
    callback.on_evaluate(args, state, control, metrics={"eval_cer": 0.305, "eval_wer": 0.505})
    state.global_step = 400
    callback.on_evaluate(args, state, control, metrics={"eval_cer": 0.304, "eval_wer": 0.506})

    assert control.should_training_stop is False
    assert warnings
    assert "training_history.csv" in warnings[0]
    warnings_path = tmp_path / "evaluation" / "plateau_warnings.jsonl"
    assert warnings_path.exists()
    assert "evals_since_best_cer" in warnings_path.read_text(encoding="utf-8")


def test_plateau_warning_callback_warns_on_sustained_regression(monkeypatch, tmp_path: Path):
    warnings: list[str] = []
    callback = PlateauWarningCallback(tmp_path)
    monkeypatch.setattr(
        "modules.ocr_training.checkpointing.logger.warning",
        lambda message, *args: warnings.append(message % args),
    )
    args = object()
    control = type("Control", (), {"should_training_stop": False})()
    state = type(
        "State",
        (),
        {
            "global_step": 100,
            "log_history": [{"step": 20, "loss": 4.0}, {"step": 40, "loss": 3.0}],
        },
    )()

    eval_points = [
        (100, 0.8192, 4.8460),
        (200, 0.6728, 4.7498),
        (300, 0.7272, 6.5203),
        (400, 0.7359, 8.7998),
        (500, 0.7292, 7.1436),
        (600, 0.7369, 8.7998),
        (700, 0.7315, 7.0180),
        (800, 0.7381, 7.8094),
        (900, 0.7398, 7.2985),
        (1000, 0.7436, 7.9194),
    ]
    for step, cer, wer in eval_points:
        state.global_step = step
        callback.on_evaluate(args, state, control, metrics={"eval_cer": cer, "eval_wer": wer})

    assert warnings
    assert "current CER gap" in warnings[0]
