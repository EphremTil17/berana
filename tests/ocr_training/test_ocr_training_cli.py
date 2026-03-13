import os
from pathlib import Path

from typer.testing import CliRunner

from modules.ocr_training.schemas import TrainMode
from tools.ocr_training import (
    _apply_training_env_defaults,
    _build_multi_gpu_eval_command,
    _build_multi_gpu_launch_command,
    _dataset_run_stem,
    _merge_pytorch_alloc_conf,
    _normalize_metric_selector,
    _resolve_tool_benchmark_output_dir,
    _resolve_tool_eval_output_dir,
    _resolve_train_output_dir,
    _strip_version_suffix,
    _training_launch_env,
    _visible_cuda_device_count,
    app,
)

runner = CliRunner()


def test_strip_version_suffix_removes_trailing_vnn():
    assert _strip_version_suffix("fidel_typed_synthetic_v01") == "fidel_typed_synthetic"
    assert _strip_version_suffix("fidel_typed_synthetic_auto_v12") == "fidel_typed_synthetic_auto"
    assert _strip_version_suffix("plain_name") == "plain_name"


def test_dataset_run_stem_infers_parent_dataset_name(tmp_path: Path):
    dataset_dir = (
        tmp_path
        / "output"
        / "ocr_training_datasets"
        / "fidel_typed_synthetic_v03"
        / "data"
        / "hf_dataset"
    )
    dataset_dir.mkdir(parents=True)

    assert _dataset_run_stem(dataset_dir) == "fidel_typed_synthetic"


def test_normalize_metric_selector_maps_user_facing_values():
    assert _normalize_metric_selector("cer") == "best_cer"
    assert _normalize_metric_selector("wer") == "best_wer"
    assert _normalize_metric_selector("latest") == "latest"
    assert _normalize_metric_selector("best_cer") == "best_cer"


def test_normalize_metric_selector_rejects_unknown_values():
    try:
        _normalize_metric_selector("foo")
    except Exception as exc:
        assert "--metric must be one of: cer, wer, latest" in str(exc)
    else:
        raise AssertionError("Expected invalid metric selector to raise.")


def test_resolve_train_output_dir_defaults_to_versioned_auto_run(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "output" / "ocr_training_runs"
    (base / "fidel_typed_synthetic_auto_v01").mkdir(parents=True)

    resolved = _resolve_train_output_dir(
        dataset_dir=tmp_path
        / "output"
        / "ocr_training_datasets"
        / "fidel_typed_synthetic_v03"
        / "data"
        / "hf_dataset",
        output_dir=None,
        mode=TrainMode.AUTO,
    )

    assert resolved == Path("output/ocr_training_runs/fidel_typed_synthetic_auto_v02")


def test_resolve_train_output_dir_preserves_explicit_override(tmp_path: Path):
    explicit = tmp_path / "custom" / "run_dir"

    resolved = _resolve_train_output_dir(
        dataset_dir=tmp_path / "dataset",
        output_dir=explicit,
        mode=TrainMode.MANUAL,
    )

    assert resolved == explicit


def test_resolve_tool_eval_output_dir_defaults_to_versioned_run_dir(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "tool_evaluation_v01").mkdir(parents=True)

    resolved = _resolve_tool_eval_output_dir(run_dir=run_dir, output_dir=None)

    assert resolved == run_dir / "tool_evaluation_v02"


def test_resolve_tool_benchmark_output_dir_defaults_to_ocr_benchmark_root(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    run_dir = (
        tmp_path / "output" / "ocr_training_runs" / "fidel_typed_synthetic_5090_lora_evalfix_v01"
    )
    run_dir.mkdir(parents=True)
    (
        tmp_path
        / "output"
        / "ocr_benchmark"
        / "gpu_performance_eval_v01"
        / "fidel_typed_synthetic_5090_lora_evalfix"
    ).mkdir(parents=True)

    resolved = _resolve_tool_benchmark_output_dir(run_dir=run_dir, output_dir=None)

    assert resolved == Path(
        "output/ocr_benchmark/gpu_performance_eval_v02/fidel_typed_synthetic_5090_lora_evalfix"
    )


def test_visible_cuda_device_count_prefers_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2,5")

    assert _visible_cuda_device_count() == 3


def test_build_multi_gpu_launch_command_strips_flag_and_pins_output_dir(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr("tools.ocr_training._torchrun_entrypoint", lambda: ["torchrun"])
    monkeypatch.setattr("tools.ocr_training.sys.argv", ["tools/ocr_training.py", "train-surya"])

    command = _build_multi_gpu_launch_command(
        argv=[
            "train-surya",
            "--dataset-dir",
            "output/data/hf_dataset",
            "--multi-gpu",
            "--mode",
            "auto",
        ],
        nproc_per_node=4,
        resolved_output_dir=tmp_path / "run_dir",
    )

    assert command[:3] == ["torchrun", "--standalone", "--nproc_per_node=4"]
    assert "--multi-gpu" not in command
    assert command[-2:] == ["--output-dir", str(tmp_path / "run_dir")]


def test_build_multi_gpu_eval_command_strips_flag(monkeypatch):
    monkeypatch.setattr("tools.ocr_training._torchrun_entrypoint", lambda: ["torchrun"])
    monkeypatch.setattr("tools.ocr_training.sys.argv", ["tools/ocr_training.py", "evaluate-surya"])

    command = _build_multi_gpu_eval_command(
        argv=[
            "evaluate-surya",
            "--run-dir",
            "output/run",
            "--multi-gpu",
            "--split",
            "holdout",
        ],
        nproc_per_node=2,
    )

    assert command[:3] == ["torchrun", "--standalone", "--nproc_per_node=2"]
    assert "--multi-gpu" not in command


def test_merge_pytorch_alloc_conf_adds_expandable_segments():
    assert _merge_pytorch_alloc_conf(None) == "expandable_segments:True"
    assert (
        _merge_pytorch_alloc_conf("garbage_collection_threshold:0.8")
        == "garbage_collection_threshold:0.8,expandable_segments:True"
    )
    assert (
        _merge_pytorch_alloc_conf("expandable_segments:False,max_split_size_mb:256")
        == "expandable_segments:True,max_split_size_mb:256"
    )


def test_training_launch_env_preserves_existing_env(monkeypatch):
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "max_split_size_mb:256")

    env = _training_launch_env()

    assert env["PYTORCH_ALLOC_CONF"] == "max_split_size_mb:256,expandable_segments:True"


def test_apply_training_env_defaults_sets_allocator(monkeypatch):
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)

    _apply_training_env_defaults()

    assert os.environ["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"


def test_visualize_surya_run_command_generates_report(tmp_path: Path):
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trainer_state.json").write_text(
        '{"log_history": [{"step": 10, "loss": 1.2, "epoch": 0.1}]}',
        encoding="utf-8",
    )

    result = runner.invoke(app, ["visualize-surya-run", "--run-dir", str(run_dir)])

    assert result.exit_code == 0
    assert (eval_dir / "training_report.md").exists()


def test_monitor_surya_run_command_reports_summary(tmp_path: Path):
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trainer_state.json").write_text(
        '{"log_history": [{"step": 10, "loss": 1.2, "epoch": 0.1}, {"step": 20, "eval_cer": 0.3, "eval_wer": 0.5, "epoch": 0.2}]}',
        encoding="utf-8",
    )

    result = runner.invoke(app, ["monitor-surya-run", "--run-dir", str(run_dir)])

    assert result.exit_code == 0
    assert "monitor-surya-run selection_metric" in result.stdout


def test_benchmark_surya_eval_requires_candidate_lists():
    result = runner.invoke(
        app,
        [
            "benchmark-surya-eval",
            "--run-dir",
            "output/ocr_training_runs/fidel_typed_synthetic_5090_lora_evalfix_v01",
            "--dataset-dir",
            "output/ocr_training_datasets/fidel_typed_synthetic_v01/data/hf_dataset",
            "--metric",
            "cer",
            "--candidate-worker-counts",
            "0,4",
        ],
    )

    assert result.exit_code != 0
    assert "--candidate-eval-batch-sizes" in result.output


def test_benchmark_surya_eval_rejects_mismatched_linked_candidate_lengths():
    result = runner.invoke(
        app,
        [
            "benchmark-surya-eval",
            "--run-dir",
            "output/ocr_training_runs/fidel_typed_synthetic_5090_lora_evalfix_v01",
            "--dataset-dir",
            "output/ocr_training_datasets/fidel_typed_synthetic_v01/data/hf_dataset",
            "--metric",
            "cer",
            "--candidate-eval-batch-sizes",
            "4,8,12",
            "--candidate-worker-counts",
            "0,4",
        ],
    )

    assert result.exit_code != 0
