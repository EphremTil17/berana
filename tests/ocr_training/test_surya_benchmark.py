import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from modules.ocr_training.distributed.context import DistributedContext
from modules.ocr_training.surya_benchmark import (
    EvalBenchmarkResult,
    benchmark_surya_eval,
)
from modules.ocr_training.surya_eval import evaluate_surya_checkpoint


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 8), color=(255, 255, 255)).save(path)


def _write_split(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _single_process_context() -> DistributedContext:
    return DistributedContext(
        execution_backend="single",
        ddp_backend=None,
        is_distributed=False,
        rank=0,
        local_rank=0,
        world_size=1,
        device="cpu",
        is_rank_zero=True,
    )


def test_benchmark_surya_eval_writes_artifacts(tmp_path: Path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "benchmark"
    rows = []
    for index in range(4):
        image_path = dataset_dir / "images" / f"sample_{index}.png"
        _write_png(image_path)
        rows.append({"image": str(image_path), "text": f"text-{index}"})
    _write_split(dataset_dir / "holdout.jsonl", rows)

    class DummyPredictor:
        def __init__(self):
            self.disable_tqdm = False
            self._seen = 0

        def __call__(self, images, **kwargs):
            del kwargs
            outputs = []
            for _image in images:
                outputs.append(
                    SimpleNamespace(text_lines=[SimpleNamespace(text=f"text-{self._seen}")])
                )
                self._seen += 1
            return outputs

    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark.require_surya",
        lambda: {
            "torch": None,
            "RecognitionPredictor": lambda _foundation: DummyPredictor(),
            "TaskNames": SimpleNamespace(ocr_with_boxes="ocr"),
        },
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark.initialize_distributed_context",
        lambda **kwargs: _single_process_context(),
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark.destroy_distributed_context",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark.maybe_barrier",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark.load_surya_eval_predictor",
        lambda *args, **kwargs: object(),
    )

    summary = benchmark_surya_eval(
        run_key="fidel_typed_synthetic",
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="holdout",
        eval_fraction=1.0,
        eval_batch_size=2,
        dataloader_num_workers=1,
        max_rows=None,
        seed=42,
        checkpoint_target="best_cer",
        checkpoint_path=None,
        output_dir=output_dir,
        candidate_eval_batch_sizes=None,
        candidate_worker_counts=None,
        max_vram_ratio=0.95,
    )

    assert summary["num_rows"] == 4
    assert summary["eval_batch_size"] == 2
    assert summary["dataloader_num_workers"] == 1
    assert summary["samples_per_second"] > 0.0
    assert (output_dir / "benchmark_summary.json").exists()
    assert (output_dir / "batch_timings.jsonl").exists()
    assert (output_dir / "stage_timings.json").exists()
    assert (output_dir / "benchmark_report.md").exists()
    stage_timings = json.loads((output_dir / "stage_timings.json").read_text(encoding="utf-8"))
    assert "predictor" in stage_timings
    assert "image_load" in stage_timings


def test_benchmark_surya_eval_selects_fastest_valid_candidate(tmp_path: Path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "benchmark"
    rows = [{"image": "/tmp/a.png", "text": "a"}] * 3

    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark.require_surya",
        lambda: {"torch": None},
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark.initialize_distributed_context",
        lambda **kwargs: _single_process_context(),
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark.destroy_distributed_context",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark.maybe_barrier",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark.load_split_rows",
        lambda *args, **kwargs: rows,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark.prepare_eval_rows",
        lambda **kwargs: SimpleNamespace(rows=rows, row_selection_sec=0.1),
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark._hardware_snapshot",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark.load_surya_eval_predictor",
        lambda *args, **kwargs: object(),
    )

    def _candidate_result(*, candidate, base_output_dir, sweep, **kwargs):
        status = "completed"
        reason = None
        sps = 10.0 + candidate.eval_batch_size + candidate.dataloader_num_workers
        if candidate.eval_batch_size == 8:
            status = "rejected_vram"
            reason = "too much vram"
        return EvalBenchmarkResult(
            candidate=candidate,
            status=status,
            reason=reason,
            output_dir=(base_output_dir / candidate.name if sweep else base_output_dir),
            num_rows=3,
            total_wall_time_sec=1.0,
            samples_per_second=sps if status == "completed" else None,
            peak_vram_mb=1000,
            summary={"status": status, "num_rows": 3} if status == "completed" else None,
        )

    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark._run_one_benchmark_candidate",
        _candidate_result,
    )

    summary = benchmark_surya_eval(
        run_key="fidel_typed_synthetic",
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="holdout",
        eval_fraction=1.0,
        eval_batch_size=2,
        dataloader_num_workers=0,
        max_rows=3,
        seed=42,
        checkpoint_target="best_cer",
        checkpoint_path=None,
        output_dir=output_dir,
        candidate_eval_batch_sizes=[2, 4, 8],
        candidate_worker_counts=[0, 4, 8],
        max_vram_ratio=0.95,
    )

    selected = summary["selected_candidate"]
    assert selected["candidate"] == "b4_w4"
    candidate_lines = (
        (output_dir / "candidate_results.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(candidate_lines) == 3
    assert (output_dir / "selected_benchmark_config.json").exists()
    assert (output_dir / "benchmark_report.md").exists()


def test_benchmark_surya_eval_auto_fills_worker_counts_when_omitted(tmp_path: Path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "benchmark"
    rows = [{"image": "/tmp/a.png", "text": "a"}] * 3

    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark.require_surya",
        lambda: {"torch": None},
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark.initialize_distributed_context",
        lambda **kwargs: _single_process_context(),
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark.destroy_distributed_context",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark.maybe_barrier",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark.load_split_rows",
        lambda *args, **kwargs: rows,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark.prepare_eval_rows",
        lambda **kwargs: SimpleNamespace(rows=rows, row_selection_sec=0.1),
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark._hardware_snapshot",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark.load_surya_eval_predictor",
        lambda *args, **kwargs: object(),
    )

    def _candidate_result(*, candidate, base_output_dir, sweep, **kwargs):
        return EvalBenchmarkResult(
            candidate=candidate,
            status="completed",
            reason=None,
            output_dir=(base_output_dir / candidate.name if sweep else base_output_dir),
            num_rows=3,
            total_wall_time_sec=1.0,
            samples_per_second=float(candidate.eval_batch_size),
            peak_vram_mb=1000,
            summary={"status": "completed", "num_rows": 3},
        )

    monkeypatch.setattr(
        "modules.ocr_training.surya_benchmark._run_one_benchmark_candidate",
        _candidate_result,
    )

    summary = benchmark_surya_eval(
        run_key="fidel_typed_synthetic",
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="holdout",
        eval_fraction=1.0,
        eval_batch_size=2,
        dataloader_num_workers=0,
        max_rows=3,
        seed=42,
        checkpoint_target="best_cer",
        checkpoint_path=None,
        output_dir=output_dir,
        candidate_eval_batch_sizes=[2, 4],
        candidate_worker_counts=None,
        max_vram_ratio=0.95,
    )

    selected = summary["selected_candidate"]
    assert selected["candidate"] == "b4_w0"
    candidate_lines = (
        (output_dir / "candidate_results.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(candidate_lines) == 2


def test_evaluate_surya_checkpoint_records_dataloader_workers_in_summary(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    rows = []
    for index in range(4):
        image_path = dataset_dir / "images" / f"sample_{index}.png"
        _write_png(image_path)
        rows.append({"image": str(image_path), "text": f"text-{index}"})
    _write_split(dataset_dir / "holdout.jsonl", rows)

    class DummyPredictor:
        def __init__(self):
            self.disable_tqdm = False
            self._seen = 0

        def __call__(self, images, **kwargs):
            del kwargs
            outputs = []
            for _image in images:
                outputs.append(
                    SimpleNamespace(text_lines=[SimpleNamespace(text=f"text-{self._seen}")])
                )
                self._seen += 1
            return outputs

    summary = evaluate_surya_checkpoint(
        run_key="fidel_typed_synthetic",
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="holdout",
        eval_fraction=1.0,
        eval_batch_size=2,
        dataloader_num_workers=2,
        max_rows=None,
        seed=42,
        runtime={
            "RecognitionPredictor": lambda _foundation: DummyPredictor(),
            "TaskNames": SimpleNamespace(ocr_with_boxes="ocr"),
        },
        load_surya_eval_predictor=lambda runtime, run_dir: object(),
    )

    assert summary["num_rows"] == 4
    payload = json.loads(
        (run_dir / "tool_evaluation" / "summary_holdout.json").read_text(encoding="utf-8")
    )
    assert payload["dataloader_num_workers"] == 2
