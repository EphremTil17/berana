from pathlib import Path
from types import SimpleNamespace

from modules.ocr_training import surya_train
from modules.ocr_training.distributed.context import DistributedContext
from modules.ocr_training.schemas import SuryaTrainConfig, TrainMode


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


def test_evaluate_surya_modalities_wrapper_forwards_modalities(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr("modules.ocr_training.surya_train.require_surya", lambda: {"runtime": True})

    def _capture_eval(**kwargs):
        captured.update(kwargs)
        return {"modalities": {"typed": {}, "synthetic": {}}}

    monkeypatch.setattr(
        "modules.ocr_training.surya_train._evaluate_surya_modalities", _capture_eval
    )

    summary = surya_train.evaluate_surya_modalities(
        run_key="fidel_typed_synthetic",
        run_dir=Path("/tmp/run"),
        dataset_dir=Path("/tmp/dataset"),
        split="holdout",
        eval_fraction=0.5,
        eval_batch_size=8,
        max_rows=100,
        seed=9,
        modalities=["typed", "synthetic"],
    )

    assert set(summary["modalities"]) == {"typed", "synthetic"}
    assert captured["modalities"] == ["typed", "synthetic"]


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
        config=SuryaTrainConfig(mode=TrainMode.MANUAL, execution_backend="ddp"),
    )

    assert result["status"] == "interrupted"
    assert calls == ["run", "barrier", "destroy"]
