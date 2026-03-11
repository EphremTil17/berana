from pathlib import Path
from types import SimpleNamespace

from modules.ocr_training.checkpointing import (
    TrainingSignalState,
    TrainingTerminationCoordinator,
    request_training_stop,
)
from modules.ocr_training.runtime.autotune_runner import select_best_candidate
from modules.ocr_training.runtime.candidate_builder import (
    build_training_candidates,
    derive_auto_constraints,
)
from modules.ocr_training.runtime.execution_controller import (
    choose_retry_candidate,
    should_retry_for_memory_pressure,
)
from modules.ocr_training.runtime.hardware_profile import detect_hardware_profile
from modules.ocr_training.runtime.strategy_catalog import strategy_is_auto_admissible
from modules.ocr_training.runtime.telemetry import (
    VramPressureCallback,
    _combined_current_used_memory_mb,
)
from modules.ocr_training.schemas import (
    CandidateResult,
    CandidateStatus,
    FinetuneStrategy,
    HardwareProfile,
    SuryaTrainConfig,
    TrainingCandidate,
    TrainMode,
)
from modules.ocr_training.surya_executor import (
    _resolve_effective_best_metric,
    _safe_save_training_bundle,
    run_training_candidate,
)
from modules.ocr_training.surya_patches import (
    build_eval_cleanup_callback,
    build_eval_interrupt_discard_callback,
    build_interrupt_callback,
)
from modules.ocr_training.surya_planner import run_auto_with_fallback
from modules.ocr_training.surya_training_args import build_training_arguments


def _cuda_stub(*, available: bool, device_count: int = 0):
    return SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: device_count,
    )


def _candidate(
    candidate_id: str,
    *,
    strategy: FinetuneStrategy = FinetuneStrategy.QLORA,
    batch_size: int = 1,
    grad_accum: int = 4,
    workers: int = 8,
    sequence_length: int = 1024,
) -> TrainingCandidate:
    return TrainingCandidate(
        candidate_id=candidate_id,
        finetune_strategy=strategy,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        dataloader_num_workers=workers,
        max_sequence_length=sequence_length,
    )


def test_detect_hardware_profile_cpu_only(monkeypatch):
    monkeypatch.setattr("modules.ocr_training.runtime.hardware_profile.os.cpu_count", lambda: 28)
    monkeypatch.setattr(
        "modules.ocr_training.runtime.hardware_profile._system_ram_mb",
        lambda: 65536,
    )
    torch_stub = SimpleNamespace(cuda=_cuda_stub(available=False))

    profile = detect_hardware_profile(torch_stub)

    assert profile.device_type == "cpu"
    assert profile.cuda_device_count == 0
    assert profile.cpu_count == 28
    assert profile.system_ram_mb == 65536


def test_strategy_auto_admissibility_for_8gb_gpu():
    profile = HardwareProfile(
        device_type="cuda",
        cuda_device_count=1,
        gpu_index=0,
        gpu_name="RTX 3060 Ti",
        total_vram_mb=8192,
        free_vram_mb=4096,
        supports_fp16=True,
        supports_bf16=False,
        cpu_count=28,
    )

    assert strategy_is_auto_admissible(profile, FinetuneStrategy.QLORA) is True
    assert strategy_is_auto_admissible(profile, FinetuneStrategy.LORA) is False
    assert strategy_is_auto_admissible(profile, FinetuneStrategy.FULL) is False


def test_build_training_candidates_respects_auto_constraints():
    profile = HardwareProfile(
        device_type="cuda",
        cuda_device_count=1,
        gpu_index=0,
        gpu_name="RTX 3060 Ti",
        total_vram_mb=8192,
        free_vram_mb=4096,
        supports_fp16=True,
        supports_bf16=False,
        cpu_count=28,
    )
    config = SuryaTrainConfig(
        mode=TrainMode.AUTO,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        dataloader_num_workers=8,
        max_sequence_length=896,
        strategy_allowlist="qlora,lora",
    )

    constraints = derive_auto_constraints(config, profile)
    candidates = build_training_candidates(
        profile=profile,
        config=config,
        constraints=constraints,
    )

    assert candidates
    assert all(candidate.finetune_strategy == FinetuneStrategy.QLORA for candidate in candidates)
    assert max(candidate.per_device_train_batch_size for candidate in candidates) == 2
    assert {candidate.max_sequence_length for candidate in candidates} == {1024}
    assert max(candidate.dataloader_num_workers for candidate in candidates) == 8


def test_build_training_candidates_propagates_allow_ram_spillover():
    profile = HardwareProfile(
        device_type="cuda",
        cuda_device_count=1,
        gpu_index=0,
        gpu_name="RTX 5090",
        total_vram_mb=32768,
        free_vram_mb=30000,
        supports_fp16=True,
        supports_bf16=True,
        cpu_count=28,
    )
    config = SuryaTrainConfig(
        mode=TrainMode.AUTO,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        allow_ram_spillover=False,
    )

    constraints = derive_auto_constraints(config, profile)
    candidates = build_training_candidates(
        profile=profile,
        config=config,
        constraints=constraints,
    )

    assert candidates
    assert all(candidate.allow_ram_spillover is False for candidate in candidates)


def test_select_best_candidate_prefers_highest_throughput(tmp_path: Path):
    candidates = [_candidate("slow"), _candidate("fast", batch_size=2)]
    candidate_results = [
        CandidateResult(
            candidate_id="slow",
            status=CandidateStatus.COMPLETED,
            effective_samples_per_second=1.5,
        ),
        CandidateResult(
            candidate_id="fast",
            status=CandidateStatus.COMPLETED,
            effective_samples_per_second=2.5,
            peak_vram_mb=4096,
        ),
    ]

    selection = select_best_candidate(
        candidates=candidates,
        candidate_results=candidate_results,
        output_dir=tmp_path,
    )

    assert selection.selected_candidate.candidate_id == "fast"
    assert selection.selection_reason == "highest_measured_samples_per_second"
    assert (tmp_path / "selected_training_config.json").exists()


def test_select_best_candidate_prefers_vram_headroom_when_available(tmp_path: Path):
    candidates = [_candidate("near_limit", batch_size=4), _candidate("safer", batch_size=2)]
    candidate_results = [
        CandidateResult(
            candidate_id="near_limit",
            status=CandidateStatus.COMPLETED,
            effective_samples_per_second=5.0,
            peak_vram_mb=7500,
        ),
        CandidateResult(
            candidate_id="safer",
            status=CandidateStatus.COMPLETED,
            effective_samples_per_second=4.5,
            peak_vram_mb=6900,
        ),
    ]

    selection = select_best_candidate(
        candidates=candidates,
        candidate_results=candidate_results,
        output_dir=tmp_path,
        safe_peak_vram_mb=7200,
    )

    assert selection.selected_candidate.candidate_id == "safer"
    assert selection.selection_reason == "highest_measured_samples_per_second_with_vram_headroom"


def test_choose_retry_candidate_prefers_lower_batch_before_lower_sequence():
    current = _candidate("current", batch_size=4, sequence_length=1024)
    lower_batch = _candidate("lower_batch", batch_size=2, sequence_length=1024)
    lower_seq = _candidate("lower_seq", batch_size=4, sequence_length=896)

    retry = choose_retry_candidate(
        current_candidate=current,
        candidate_results=[],
        all_candidates=[current, lower_seq, lower_batch],
        attempted_candidate_ids={"current"},
        reason="VRAM guard triggered",
    )

    assert retry is not None
    assert retry.candidate_id == "lower_batch"


def test_choose_retry_candidate_does_not_fallback_to_lower_sequence():
    current = _candidate("current", batch_size=1, sequence_length=1024)
    lower_seq = _candidate("lower_seq", batch_size=1, sequence_length=896)

    retry = choose_retry_candidate(
        current_candidate=current,
        candidate_results=[],
        all_candidates=[current, lower_seq],
        attempted_candidate_ids={"current"},
        reason="VRAM guard triggered",
    )

    assert retry is None


def test_choose_retry_candidate_uses_benchmark_ranking_for_throughput_shortfall():
    current = _candidate("current", batch_size=2)
    faster = _candidate("faster", batch_size=1)
    slower = _candidate("slower", batch_size=1, workers=4)
    results = [
        CandidateResult(
            candidate_id="slower",
            status=CandidateStatus.COMPLETED,
            effective_samples_per_second=1.0,
        ),
        CandidateResult(
            candidate_id="faster",
            status=CandidateStatus.COMPLETED,
            effective_samples_per_second=2.0,
        ),
    ]

    retry = choose_retry_candidate(
        current_candidate=current,
        candidate_results=results,
        all_candidates=[current, slower, faster],
        attempted_candidate_ids={"current"},
        reason="throughput_shortfall:observed=0.5000,planned=2.0000",
    )

    assert retry is not None
    assert retry.candidate_id == "faster"


def test_build_training_arguments_omits_none_max_steps():
    captured = {}

    def training_arguments_cls(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    build_training_arguments(
        training_arguments_cls=training_arguments_cls,
        output_dir=Path("/tmp/unused"),
        candidate=_candidate("train"),
        eval_enabled=False,
        save_enabled=True,
        compute_metrics_enabled=False,
        max_steps=None,
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
    )

    assert "max_steps" not in captured


def test_resolve_effective_best_metric_uses_authoritative_eval_selection():
    candidate = _candidate("train")

    effective_candidate, metric_name = _resolve_effective_best_metric(
        candidate=candidate,
        compute_metrics=None,
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
    )

    assert metric_name == "eval_cer"
    assert effective_candidate.metric_for_best_model == "eval_loss"
    assert effective_candidate.greater_is_better is False
    assert effective_candidate.load_best_model_at_end is False


def test_should_retry_for_memory_pressure_matches_oom_and_vram_guard():
    assert should_retry_for_memory_pressure("CUDA out of memory") is True
    assert should_retry_for_memory_pressure("VRAM guard triggered: GPU 0") is True
    assert should_retry_for_memory_pressure("throughput_shortfall:observed=1,planned=2") is False


def test_run_auto_with_fallback_keeps_retrying_for_memory_pressure(tmp_path: Path):
    config = SuryaTrainConfig(mode=TrainMode.AUTO, max_replans=0)
    initial = _candidate("b4", batch_size=4)
    retry = _candidate("b2", batch_size=2)
    final = _candidate("b1", batch_size=1)
    calls: list[str] = []
    attempts: list[dict] = []

    def runner(*, selected_candidate, selection_reason, retry_count, planned_samples_per_second):
        calls.append(selected_candidate.candidate_id)
        if selected_candidate.candidate_id != "b1":
            raise RuntimeError("VRAM guard triggered: GPU 0 is using too much memory")
        return {
            "status": "completed",
            "selected_candidate_id": selected_candidate.candidate_id,
        }

    result = run_auto_with_fallback(
        initial_candidate=initial,
        candidate_pool=[initial, retry, final],
        candidate_results=[],
        selection_reason="highest_measured_samples_per_second",
        planned_samples_per_second=1.0,
        config=config,
        output_dir=tmp_path,
        attempts=attempts,
        runner=runner,
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
    )

    assert result["selected_candidate_id"] == "b1"
    assert calls == ["b4", "b2", "b1"]


def test_run_training_candidate_flushes_cuda_cache_before_model_load(tmp_path: Path):
    calls: list[str] = []
    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: calls.append("synchronize"),
            empty_cache=lambda: calls.append("empty_cache"),
            ipc_collect=lambda: calls.append("ipc_collect"),
        )
    )

    class _Trainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def add_callback(self, callback):
            return None

        def train(self, resume_from_checkpoint=None):
            return None

        def save_model(self, output_dir):
            return None

        def save_state(self):
            return None

    runtime = {
        "torch": torch_stub,
        "TrainingArguments": lambda **kwargs: SimpleNamespace(**kwargs),
        "Trainer": _Trainer,
        "TrainerCallback": object,
        "TaskNames": SimpleNamespace(ocr_with_boxes="ocr"),
    }
    candidate = _candidate("b1", batch_size=1)
    processor = SimpleNamespace(save_pretrained=lambda path: None)
    model = SimpleNamespace(save_pretrained=lambda path: None)

    def _load_stack(runtime_arg, checkpoint, config):
        calls.append("load_stack")
        return model, processor, {}

    result = run_training_candidate(
        runtime=runtime,
        run_key="test",
        output_dir=tmp_path,
        config=SuryaTrainConfig(mode=TrainMode.MANUAL),
        candidate=candidate,
        base_checkpoint="checkpoint",
        train_rows=[],
        val_rows=[],
        original_train_count=0,
        attempts=[],
        selection_reason="manual_mode",
        discarded_candidates=0,
        retry_count=0,
        planned_samples_per_second=None,
        mode=TrainMode.MANUAL,
        load_surya_training_stack=_load_stack,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        epoch_logging_callback_cls=lambda: object(),
    )

    assert result["status"] == "completed"
    assert calls[:4] == ["synchronize", "empty_cache", "ipc_collect", "load_stack"]


def test_safe_save_training_bundle_ignores_processor_serialization_errors(tmp_path: Path):
    saved: list[str] = []

    class _Model:
        def save_pretrained(self, output_dir):
            saved.append(f"model:{output_dir}")

    class _Processor:
        def save_pretrained(self, output_dir):
            raise AttributeError("'function' object has no attribute 'save_pretrained'")

    warnings: list[str] = []
    _safe_save_training_bundle(
        model=_Model(),
        processor=_Processor(),
        output_dir=tmp_path,
        logger=SimpleNamespace(warning=lambda message, *args: warnings.append(message % args)),
    )

    assert saved == [f"model:{tmp_path}"]
    assert warnings


def test_run_training_candidate_marks_signal_stop_as_interrupted(tmp_path: Path, monkeypatch):
    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
        )
    )

    class _Trainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def add_callback(self, callback):
            return None

        def train(self, resume_from_checkpoint=None):
            del resume_from_checkpoint
            return None

        def save_state(self):
            return None

    runtime = {
        "torch": torch_stub,
        "TrainingArguments": lambda **kwargs: SimpleNamespace(**kwargs),
        "Trainer": _Trainer,
        "TrainerCallback": object,
        "TaskNames": SimpleNamespace(ocr_with_boxes="ocr"),
    }
    candidate = _candidate("b1", batch_size=1)
    processor = SimpleNamespace(save_pretrained=lambda path: None)
    model = SimpleNamespace(save_pretrained=lambda path: None)

    def _load_stack(runtime_arg, checkpoint, config):
        del runtime_arg, checkpoint, config
        return model, processor, {}

    def _install_signal_handlers(state):
        state.interrupted = True

    monkeypatch.setattr(
        "modules.ocr_training.surya_executor.install_signal_handlers",
        _install_signal_handlers,
    )

    result = run_training_candidate(
        runtime=runtime,
        run_key="test",
        output_dir=tmp_path,
        config=SuryaTrainConfig(mode=TrainMode.MANUAL),
        candidate=candidate,
        base_checkpoint="checkpoint",
        train_rows=[],
        val_rows=[],
        original_train_count=0,
        attempts=[],
        selection_reason="manual_mode",
        discarded_candidates=0,
        retry_count=0,
        planned_samples_per_second=None,
        mode=TrainMode.MANUAL,
        load_surya_training_stack=_load_stack,
        logger=SimpleNamespace(
            info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None
        ),
        epoch_logging_callback_cls=lambda: object(),
    )

    assert result["status"] == "interrupted"
    assert (tmp_path / "resume_state.json").exists()


def test_run_training_candidate_skips_vram_guard_when_ram_spillover_allowed(tmp_path: Path):
    added_callbacks: list[object] = []
    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
        )
    )

    class _Trainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def add_callback(self, callback):
            added_callbacks.append(callback)
            return None

        def train(self, resume_from_checkpoint=None):
            del resume_from_checkpoint
            return None

        def save_state(self):
            return None

    runtime = {
        "torch": torch_stub,
        "TrainingArguments": lambda **kwargs: SimpleNamespace(**kwargs),
        "Trainer": _Trainer,
        "TrainerCallback": object,
        "TaskNames": SimpleNamespace(ocr_with_boxes="ocr"),
    }
    candidate = _candidate("b1", batch_size=1).model_copy(update={"allow_ram_spillover": True})
    processor = SimpleNamespace(save_pretrained=lambda path: None)
    model = SimpleNamespace(save_pretrained=lambda path: None)

    def _load_stack(runtime_arg, checkpoint, config):
        del runtime_arg, checkpoint, config
        return model, processor, {}

    run_training_candidate(
        runtime=runtime,
        run_key="test",
        output_dir=tmp_path,
        config=SuryaTrainConfig(mode=TrainMode.MANUAL),
        candidate=candidate,
        base_checkpoint="checkpoint",
        train_rows=[],
        val_rows=[],
        original_train_count=0,
        attempts=[],
        selection_reason="manual_mode",
        discarded_candidates=0,
        retry_count=0,
        planned_samples_per_second=None,
        mode=TrainMode.MANUAL,
        load_surya_training_stack=_load_stack,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        epoch_logging_callback_cls=lambda: object(),
    )

    assert len(added_callbacks) == 3


def test_combined_current_used_memory_caps_driver_report_to_total():
    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            memory_reserved=lambda: 2048 * 1024 * 1024,
        )
    )
    snapshot = SimpleNamespace(total_memory_mb=8192, used_memory_mb=10186)

    used_memory_mb = _combined_current_used_memory_mb(torch_stub, snapshot)

    assert used_memory_mb == 8192


def test_eval_cleanup_callback_flushes_cuda_cache():
    calls: list[str] = []
    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: calls.append("synchronize"),
            empty_cache=lambda: calls.append("empty_cache"),
            ipc_collect=lambda: calls.append("ipc_collect"),
        )
    )

    callback = build_eval_cleanup_callback(torch_module=torch_stub, callback_base=object)
    control = SimpleNamespace()

    returned_control = callback.on_evaluate(None, None, control)

    assert returned_control is control
    assert calls == ["synchronize", "empty_cache", "ipc_collect"]


def test_vram_pressure_callback_can_request_stop_without_raising(tmp_path: Path):
    state = TrainingSignalState()
    coordinator = TrainingTerminationCoordinator(tmp_path)
    callback = VramPressureCallback(
        callback_base=object,
        torch_module=SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                memory_reserved=lambda: 0,
                get_device_properties=lambda _index: SimpleNamespace(
                    total_memory=8192 * 1024 * 1024
                ),
                current_device=lambda: 0,
            )
        ),
        usage_threshold_ratio=0.9,
        on_trigger=lambda message: request_training_stop(
            state,
            reason=message,
            stop_type="runtime",
            save_checkpoint=False,
            coordinator=coordinator,
            rank=0,
        ),
    ).build()
    control = SimpleNamespace(should_training_stop=False)
    trainer_state = SimpleNamespace(global_step=1)

    from modules.ocr_training.runtime import telemetry as telemetry_module

    snapshot = SimpleNamespace(total_memory_mb=100, used_memory_mb=95, gpu_index=0)
    original = telemetry_module.collect_gpu_memory_snapshot
    telemetry_module.collect_gpu_memory_snapshot = lambda _torch: snapshot
    try:
        returned_control = callback.on_step_end(None, trainer_state, control)
    finally:
        telemetry_module.collect_gpu_memory_snapshot = original

    assert returned_control is control
    assert control.should_training_stop is True
    assert state.runtime_error_message is not None


def test_interrupt_callback_marks_eval_interrupted_on_prediction_step():
    state = TrainingSignalState(interrupted=True)
    callback = build_interrupt_callback(state, object)
    control = SimpleNamespace(should_training_stop=False, should_save=False)

    returned_control = callback.on_prediction_step(None, None, control)

    assert returned_control is control
    assert state.eval_interrupted is True
    assert control.should_training_stop is True
    assert control.should_save is True


def test_eval_interrupt_discard_callback_strips_metrics_and_logs():
    warnings: list[str] = []
    state = TrainingSignalState(interrupted=True, eval_interrupted=True)
    callback = build_eval_interrupt_discard_callback(
        state,
        object,
        SimpleNamespace(warning=lambda message, *args: warnings.append(message % args)),
    )
    control = SimpleNamespace()
    metrics = {
        "eval_loss": 0.2,
        "eval_cer": 0.3,
        "eval_wer": 0.4,
        "eval_exact": 0.0,
    }
    logs = {
        "eval_loss": 0.2,
        "eval_cer": 0.3,
        "eval_wer": 0.4,
        "eval_exact": 0.0,
        "eval_runtime": 1.0,
    }
    trainer_state = SimpleNamespace(global_step=123)

    callback.on_evaluate(None, trainer_state, control, metrics=metrics)
    callback.on_log(None, trainer_state, control, logs=logs)

    assert metrics == {}
    assert logs == {}
    assert state.eval_interrupted is False
    assert len(warnings) >= 1


def test_interrupt_callback_observes_peer_stop_request(tmp_path: Path):
    coordinator = TrainingTerminationCoordinator(tmp_path)
    request_training_stop(
        TrainingSignalState(),
        reason="peer runtime failure",
        stop_type="runtime",
        save_checkpoint=False,
        coordinator=coordinator,
        rank=1,
    )
    state = TrainingSignalState()
    callback = build_interrupt_callback(
        state,
        object,
        termination_coordinator=coordinator,
    )
    control = SimpleNamespace(should_training_stop=False, should_save=False)

    returned_control = callback.on_step_end(None, None, control)

    assert returned_control is control
    assert state.runtime_error_message == "peer runtime failure"
    assert control.should_training_stop is True
    assert control.should_save is False
