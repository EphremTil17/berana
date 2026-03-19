from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from modules.ocr_training import surya_train
from modules.ocr_training.distributed.context import DistributedContext
from modules.ocr_training.schemas import ExecutionBackend, SuryaTrainConfig, TrainMode
from modules.ocr_training.surya_eval import EvaluateSuryaModalitiesSummary

if TYPE_CHECKING:
    from modules.ocr_training.schemas import TrainingCandidate


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
        checkpoint_target="latest",
        checkpoint_path=Path("/tmp/run/checkpoint-500"),
        output_dir=Path("/tmp/run/tool_evaluation"),
    )

    assert summary["mean_cer"] == 0.1
    assert captured["eval_fraction"] == 0.25
    assert captured["eval_batch_size"] == 16
    assert captured["max_rows"] == 500
    assert captured["seed"] == 7
    assert captured["output_dir"] == Path("/tmp/run/tool_evaluation")
    load_predictor = captured["load_surya_eval_predictor"]
    assert callable(load_predictor)


def test_evaluate_surya_modalities_wrapper_forwards_modalities(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr("modules.ocr_training.surya_train.require_surya", lambda: {"runtime": True})

    def _capture_eval(**kwargs):
        captured.update(kwargs)
        return {"modalities": {"typed": {}, "synthetic": {}}}

    monkeypatch.setattr(
        "modules.ocr_training.surya_train._evaluate_surya_modalities", _capture_eval
    )

    summary: EvaluateSuryaModalitiesSummary = surya_train.evaluate_surya_modalities(
        run_key="fidel_typed_synthetic",
        run_dir=Path("/tmp/run"),
        dataset_dir=Path("/tmp/dataset"),
        split="holdout",
        eval_fraction=0.5,
        eval_batch_size=8,
        max_rows=100,
        seed=9,
        modalities=["typed", "synthetic"],
        checkpoint_target="best_wer",
        checkpoint_path=Path("/tmp/run/checkpoint-200"),
        output_dir=Path("/tmp/run/tool_evaluation"),
    )

    assert set(summary["modalities"]) == {"typed", "synthetic"}
    assert captured["modalities"] == ["typed", "synthetic"]
    assert captured["output_dir"] == Path("/tmp/run/tool_evaluation")


def test_run_surya_finetune_barriers_before_destroy_on_interrupt(monkeypatch, tmp_path: Path):
    calls: list[str] = []
    context = DistributedContext(
        execution_backend="ddp",
        ddp_backend="nccl",
        is_distributed=True,
        rank=0,
        local_rank=0,
        world_size=2,
        device="cuda:0",
        is_rank_zero=True,
    )

    monkeypatch.setattr(
        "modules.ocr_training.surya_train.require_surya",
        lambda: {"torch": SimpleNamespace()},
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_train.initialize_distributed_context",
        lambda **kwargs: context,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_train._prepare_train_and_val_rows",
        lambda **kwargs: ([{"id": "train"}], [{"id": "val"}], [{"id": "train"}], [{"id": "val"}]),
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_train.detect_hardware_profile",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_train.resolve_base_checkpoint",
        lambda *args, **kwargs: "checkpoint",
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_train.write_hardware_profile",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_train._load_finetune_meta",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_train._build_training_stack_loader",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_train._log_subset_adjustments",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_train.maybe_barrier",
        lambda **kwargs: calls.append("barrier"),
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_train.destroy_distributed_context",
        lambda **kwargs: calls.append("destroy"),
    )

    def _interrupt_manual_mode(**kwargs):
        calls.append("run")
        return {"status": "interrupted"}

    monkeypatch.setattr("modules.ocr_training.surya_train._run_manual_mode", _interrupt_manual_mode)

    result = surya_train.run_surya_finetune(
        run_key="run",
        dataset_dir=tmp_path / "dataset",
        output_dir=tmp_path / "output",
        config=SuryaTrainConfig(mode=TrainMode.MANUAL, execution_backend=ExecutionBackend.DDP),
    )

    assert result["status"] == "interrupted"
    assert calls == ["run", "barrier", "destroy"]


def test_authoritative_eval_runner_forwards_dataloader_workers(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    class _FakeModel:
        pass

    class _FakeProcessor:
        pass

    class _FakeTrainer:
        def __init__(self, *args, **kwargs):
            self.args = kwargs["args"]
            self.callbacks = []

        def add_callback(self, callback):
            self.callbacks.append(callback)

        def train(self, resume_from_checkpoint=None):
            del resume_from_checkpoint
            return None

    class _FakeRecognitionPredictor:
        def __init__(self, foundation_predictor):
            self.foundation_predictor = foundation_predictor
            self.disable_tqdm = False

    class _FakeTrainingArguments:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    runtime = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: False,
                synchronize=lambda: None,
                empty_cache=lambda: None,
                ipc_collect=lambda: None,
            )
        ),
        "Trainer": _FakeTrainer,
        "TrainingArguments": _FakeTrainingArguments,
        "TaskNames": SimpleNamespace(ocr_with_boxes="ocr_with_boxes"),
        "RecognitionPredictor": _FakeRecognitionPredictor,
        "TrainerCallback": object,
    }

    candidate = SimpleNamespace(
        per_device_train_batch_size=6,
        per_device_eval_batch_size=24,
        gradient_accumulation_steps=2,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_prefetch_factor=2,
        learning_rate=1e-4,
        fp16=False,
        gradient_checkpointing=False,
        finetune_strategy=SimpleNamespace(value="lora"),
        num_train_epochs=1,
        eval_steps=50,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=20,
        verbose_epochs=False,
        allow_ram_spillover=True,
        abort_vram_usage_ratio=0.95,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        execution_backend=SimpleNamespace(value="ddp"),
        max_sequence_length=1024,
        model_copy=lambda update: SimpleNamespace(**{**candidate.__dict__, **update}),
        model_dump=lambda mode="json": {"candidate_id": "manual"},
        candidate_id="manual",
    )

    config = SimpleNamespace(
        eval_fraction=1.0,
        eval_max_rows=2000,
        seed=42,
        resume="none",
        train_fraction=0.05,
        dataloader_num_workers=0,
    )
    config.model_copy = lambda update: SimpleNamespace(**{**config.__dict__, **update})

    monkeypatch.setattr(
        "modules.ocr_training.surya_executor.write_finetune_meta",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_executor.load_finetune_meta",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_executor.resolve_resume_checkpoint",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_executor.LocalSuryaOCRDataset",
        lambda **kwargs: kwargs["rows"],
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_executor.SuryaOCRDataCollator",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_executor.build_training_arguments",
        lambda **kwargs: _FakeTrainingArguments(
            output_dir=str(tmp_path / "run"),
            per_device_eval_batch_size=24,
        ),
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_executor.candidate_to_train_config",
        lambda config, candidate: config,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_executor._attach_training_callbacks",
        lambda **kwargs: captured.setdefault(
            "authoritative_eval_runner", kwargs["authoritative_eval_runner"]
        ),
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_executor._resolve_effective_best_metric",
        lambda **kwargs: (candidate, "eval_wer"),
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_executor.load_surya_eval_predictor",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_executor.evaluate_surya_rows",
        lambda **kwargs: captured.setdefault("eval_kwargs", kwargs) or {"mean_cer": 0.1},
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_executor._safe_save_training_bundle",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_executor.register_completed_finetune",
        lambda **kwargs: {"status": "completed"},
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_executor.write_training_report_bundle",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_executor.write_resume_state",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_executor.install_signal_handlers",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_executor.observe_training_stop",
        lambda *args, **kwargs: None,
    )

    distributed_context = DistributedContext(
        execution_backend="ddp",
        ddp_backend="nccl",
        is_distributed=True,
        rank=0,
        local_rank=0,
        world_size=2,
        device="cuda:0",
        is_rank_zero=True,
    )

    from modules.ocr_training.surya_executor import run_training_candidate

    run_training_candidate(
        runtime=runtime,
        run_key="run",
        output_dir=tmp_path / "run",
        config=config,
        candidate=cast("TrainingCandidate", candidate),
        base_checkpoint="checkpoint",
        train_rows=[{"image": "a", "text": "x"}],
        val_rows=[{"image": "b", "text": "y"}],
        original_train_count=1,
        attempts=[],
        selection_reason="manual",
        discarded_candidates=0,
        retry_count=0,
        planned_samples_per_second=None,
        mode=TrainMode.MANUAL,
        distributed_context=distributed_context,
        load_surya_training_stack=lambda *args, **kwargs: (_FakeModel(), _FakeProcessor(), {}),
        logger=SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None),
        epoch_logging_callback_cls=lambda: object(),
    )

    authoritative_runner = captured["authoritative_eval_runner"]
    assert callable(authoritative_runner)
    authoritative_runner(
        checkpoint_path=tmp_path / "run" / "checkpoint-200",
        state=SimpleNamespace(global_step=200),
    )

    eval_kwargs = captured["eval_kwargs"]
    assert isinstance(eval_kwargs, dict)
    assert eval_kwargs["dataloader_num_workers"] == 0
