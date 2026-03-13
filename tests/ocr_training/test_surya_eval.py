import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from modules.ocr_training.surya_eval import (
    evaluate_surya_checkpoint,
    evaluate_surya_modalities,
    evaluate_surya_rows,
)
from modules.ocr_training.surya_reports import (
    monitor_training_run,
    write_confusion_artifacts,
    write_subset_manifest,
    write_training_history_artifacts,
    write_training_report_bundle,
)


def _write_split(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 8), color=(255, 255, 255)).save(path)


def test_evaluate_surya_checkpoint_batches_inference(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    rows = []
    for index in range(4):
        image_path = dataset_dir / "images" / f"sample_{index}.png"
        _write_png(image_path)
        rows.append({"image": str(image_path), "text": f"text-{index}"})
    _write_split(dataset_dir / "holdout.jsonl", rows)

    call_sizes: list[int] = []

    class DummyPredictor:
        def __init__(self):
            self.disable_tqdm = False

        def __call__(self, images, **kwargs):
            assert all(
                isinstance(image_boxes, list) and len(image_boxes) == 1 and len(image_boxes[0]) == 4
                for image_boxes in kwargs["bboxes"]
            )
            call_sizes.append(len(images))
            return [
                SimpleNamespace(text_lines=[SimpleNamespace(text=f"text-{offset}")])
                for offset, _image in enumerate(images, start=sum(call_sizes[:-1]))
            ]

    summary = evaluate_surya_checkpoint(
        run_key="fidel_typed_synthetic",
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="holdout",
        eval_fraction=1.0,
        eval_batch_size=2,
        max_rows=None,
        seed=42,
        runtime={
            "RecognitionPredictor": lambda _foundation: DummyPredictor(),
            "TaskNames": SimpleNamespace(ocr_with_boxes="ocr"),
        },
        load_surya_eval_predictor=lambda runtime, run_dir: object(),
    )

    assert call_sizes == [2, 2]
    assert summary["num_rows"] == 4
    assert summary["mean_cer"] == 0.0
    assert summary["mean_wer"] == 0.0
    assert (run_dir / "tool_evaluation" / "predictions_holdout.jsonl").exists()
    assert (run_dir / "tool_evaluation" / "summary_holdout.json").exists()
    assert not (run_dir / "tool_evaluation" / "training_history.csv").exists()


def test_evaluate_surya_checkpoint_logs_split_counts(tmp_path: Path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    rows = []
    for index in range(3):
        image_path = dataset_dir / "images" / f"sample_{index}.png"
        _write_png(image_path)
        rows.append({"image": str(image_path), "text": f"text-{index}"})
    _write_split(dataset_dir / "train.jsonl", rows)
    _write_split(dataset_dir / "val.jsonl", rows[:2])
    _write_split(dataset_dir / "holdout.jsonl", rows[:1])

    logged_messages: list[str] = []

    def _capture_info(message, *args, **kwargs):
        logged_messages.append(message % args if args else message)

    monkeypatch.setattr("modules.ocr_training.surya_eval.logger.info", _capture_info)

    evaluate_surya_checkpoint(
        run_key="fidel_typed_synthetic",
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="holdout",
        eval_fraction=1.0,
        eval_batch_size=1,
        max_rows=None,
        seed=42,
        runtime={
            "RecognitionPredictor": lambda _foundation: (
                lambda images, **kwargs: [
                    SimpleNamespace(text_lines=[SimpleNamespace(text="text-0")]) for _ in images
                ]
            ),
            "TaskNames": SimpleNamespace(ocr_with_boxes="ocr"),
        },
        load_surya_eval_predictor=lambda runtime, run_dir: object(),
    )

    assert any(
        "dataset_rows={train:3,val:2,holdout:1}" in message and "selected_rows=1" in message
        for message in logged_messages
    )


def test_evaluate_surya_rows_barriers_after_rank_zero_writes(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    output_dir = run_dir / "tool_evaluation"
    rows = [{"image": str(tmp_path / "fake.png"), "text": "hello"}]

    barrier_calls: list[int] = []

    def _fake_barrier(*, torch_module, context):
        barrier_calls.append(int(context.rank))

    monkeypatch.setattr("modules.ocr_training.surya_eval.maybe_barrier", _fake_barrier)
    monkeypatch.setattr(
        "modules.ocr_training.surya_eval.run_surya_eval_batches",
        lambda **kwargs: SimpleNamespace(
            records=[
                {
                    "image": rows[0]["image"],
                    "gt_text": "hello",
                    "pred_text": "hello",
                    "cer": 0.0,
                    "wer": 0.0,
                    "exact": True,
                }
            ],
            world_size=2,
            batch_timings=[],
        ),
    )

    summary = evaluate_surya_rows(
        run_key="fidel_typed_synthetic",
        run_dir=run_dir,
        rows=rows,
        split="holdout",
        eval_fraction=1.0,
        max_rows=None,
        eval_batch_size=1,
        dataloader_num_workers=0,
        seed=42,
        modality=None,
        predictor=object(),
        runtime={"TaskNames": SimpleNamespace(ocr_with_boxes="ocr")},
        distributed_context=SimpleNamespace(is_distributed=True, is_rank_zero=True, rank=0),
        torch_module=None,
        output_dir=output_dir,
        register_stage=False,
        include_predictions=False,
        include_confusions=False,
        include_report_bundle=False,
    )

    assert summary["num_rows"] == 1
    assert barrier_calls == [0]


def test_write_confusion_artifacts_outputs_top_pairs(tmp_path: Path):
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    artifacts = write_confusion_artifacts(
        eval_dir=eval_dir,
        split="holdout",
        records=[
            {"gt_text": "abc", "pred_text": "axc"},
            {"gt_text": "abc", "pred_text": "axc"},
        ],
    )

    payload = json.loads(artifacts["character_confusions_json"].read_text(encoding="utf-8"))
    assert payload[0] == {"gt": "b", "pred": "x", "count": 2}
    assert artifacts["character_confusions_md"].exists()


def test_write_training_history_artifacts_writes_csv_and_svg(tmp_path: Path):
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    trainer_state = {
        "log_history": [
            {"step": 10, "loss": 1.2, "epoch": 0.1},
            {"step": 20, "eval_loss": 0.4, "eval_cer": 0.3, "eval_wer": 0.5, "epoch": 0.2},
        ]
    }
    (run_dir / "trainer_state.json").write_text(json.dumps(trainer_state), encoding="utf-8")

    artifacts = write_training_history_artifacts(run_dir=run_dir, eval_dir=eval_dir)

    assert artifacts["training_history_csv"].exists()
    assert artifacts["training_curves_svg"].exists()
    assert "Eval CER" in artifacts["training_curves_svg"].read_text(encoding="utf-8")


def test_write_training_history_artifacts_falls_back_to_latest_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "evaluation"
    checkpoint_dir = run_dir / "checkpoint-500"
    eval_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer_state = {
        "log_history": [
            {"step": 10, "loss": 1.2, "epoch": 0.1},
            {"step": 20, "eval_loss": 0.4, "eval_cer": 0.3, "eval_wer": 0.5, "epoch": 0.2},
        ]
    }
    (checkpoint_dir / "trainer_state.json").write_text(json.dumps(trainer_state), encoding="utf-8")

    artifacts = write_training_history_artifacts(run_dir=run_dir, eval_dir=eval_dir)

    assert artifacts["training_history_csv"].exists()
    assert artifacts["training_summary_json"].exists()


def test_write_training_history_artifacts_summary_includes_best_wer_and_fraction_note(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    trainer_state = {
        "log_history": [
            {"step": 10, "loss": 1.2, "epoch": 0.1},
            {"step": 20, "eval_loss": 0.4, "eval_cer": 0.3, "eval_wer": 0.6, "epoch": 0.2},
            {"step": 40, "eval_loss": 0.35, "eval_cer": 0.32, "eval_wer": 0.5, "epoch": 0.4},
        ]
    }
    (run_dir / "trainer_state.json").write_text(json.dumps(trainer_state), encoding="utf-8")
    (run_dir / "finetune_meta.json").write_text(
        json.dumps({"train_fraction": 0.2}, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "best_model_meta.json").write_text(
        json.dumps({"metric_global_step": 20}, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "best_wer_model_meta.json").write_text(
        json.dumps({"metric_global_step": 40}, ensure_ascii=False),
        encoding="utf-8",
    )

    artifacts = write_training_history_artifacts(run_dir=run_dir, eval_dir=eval_dir)

    summary = json.loads(artifacts["training_summary_json"].read_text(encoding="utf-8"))
    svg_text = artifacts["training_curves_svg"].read_text(encoding="utf-8")
    assert summary["best_eval_by_wer"]["step"] == 40
    assert "train_fraction < 1.0" in summary["notes"][0]
    assert "best CER @20" in svg_text
    assert "best WER @40" in svg_text


def test_write_training_history_artifacts_summary_includes_plateau_warning(tmp_path: Path):
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    trainer_state = {
        "log_history": [
            {"step": 20, "eval_loss": 0.4, "eval_cer": 0.3, "eval_wer": 0.6, "epoch": 0.2},
        ]
    }
    (run_dir / "trainer_state.json").write_text(json.dumps(trainer_state), encoding="utf-8")
    (eval_dir / "plateau_warnings.jsonl").write_text(
        json.dumps({"step": 120, "evals_since_best_cer": 4}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    artifacts = write_training_history_artifacts(run_dir=run_dir, eval_dir=eval_dir)

    summary = json.loads(artifacts["training_summary_json"].read_text(encoding="utf-8"))
    assert summary["plateau_warning_count"] == 1
    assert summary["latest_plateau_warning"]["step"] == 120


def test_write_training_history_artifacts_prefers_authoritative_checkpoint_eval_metrics(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    trainer_state = {
        "log_history": [
            {"step": 20, "loss": 1.2, "epoch": 0.1},
            {"step": 20, "eval_loss": 0.4, "epoch": 0.2},
        ]
    }
    (run_dir / "trainer_state.json").write_text(json.dumps(trainer_state), encoding="utf-8")
    (eval_dir / "checkpoint_eval_history.jsonl").write_text(
        json.dumps(
            {
                "step": 20,
                "eval_cer": 0.03,
                "eval_wer": 0.07,
                "eval_exact": 0.75,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = write_training_history_artifacts(run_dir=run_dir, eval_dir=eval_dir)

    csv_text = artifacts["training_history_csv"].read_text(encoding="utf-8")
    summary = json.loads(artifacts["training_summary_json"].read_text(encoding="utf-8"))
    assert "0.03" in csv_text
    assert summary["best_eval_by_cer"]["eval_cer"] == 0.03


def test_write_training_history_artifacts_merges_timing_sidecar(tmp_path: Path):
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    trainer_state = {
        "log_history": [
            {"step": 10, "loss": 1.2, "epoch": 0.1},
            {"step": 20, "eval_loss": 0.4, "eval_cer": 0.3, "eval_wer": 0.6, "epoch": 0.2},
        ]
    }
    (run_dir / "trainer_state.json").write_text(json.dumps(trainer_state), encoding="utf-8")
    (eval_dir / "training_timing.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "step": 10,
                        "event_type": "train",
                        "wall_time_sec": 5.0,
                        "rolling_step_time_sec": 0.25,
                        "eval_runtime_sec": None,
                    }
                ),
                json.dumps(
                    {
                        "step": 20,
                        "event_type": "eval",
                        "wall_time_sec": 12.0,
                        "rolling_step_time_sec": 0.35,
                        "eval_runtime_sec": 7.5,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = write_training_history_artifacts(run_dir=run_dir, eval_dir=eval_dir)

    csv_lines = artifacts["training_history_csv"].read_text(encoding="utf-8").splitlines()
    assert "wall_time_sec" in csv_lines[0]
    assert ",5.0,0.25" in csv_lines[1]
    assert ",7.5," in csv_lines[2]


def test_write_training_history_artifacts_formats_learning_rate_scientifically(tmp_path: Path):
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    trainer_state = {
        "log_history": [
            {
                "step": 10,
                "loss": 1.2,
                "learning_rate": 1.969202197632129e-05,
                "epoch": 0.1,
            }
        ]
    }
    (run_dir / "trainer_state.json").write_text(json.dumps(trainer_state), encoding="utf-8")

    artifacts = write_training_history_artifacts(run_dir=run_dir, eval_dir=eval_dir)

    csv_lines = artifacts["training_history_csv"].read_text(encoding="utf-8").splitlines()
    assert "1.9692e-05" in csv_lines[1]
    summary = json.loads(artifacts["training_summary_json"].read_text(encoding="utf-8"))
    assert summary["latest_train"]["learning_rate"] == "1.9692e-05"


def test_write_training_report_bundle_generates_report_with_confusion_data(tmp_path: Path):
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trainer_state.json").write_text(
        json.dumps(
            {
                "log_history": [
                    {"step": 10, "loss": 1.2, "epoch": 0.1},
                    {
                        "step": 20,
                        "eval_loss": 0.4,
                        "eval_cer": 0.3,
                        "eval_wer": 0.6,
                        "epoch": 0.2,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    predictions_path = eval_dir / "predictions_holdout.jsonl"
    predictions_path.write_text(
        "\n".join(
            [
                json.dumps({"gt_text": "abc", "pred_text": "axc"}, ensure_ascii=False),
                json.dumps({"gt_text": "word", "pred_text": "ward"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = write_training_report_bundle(run_dir=run_dir, split="holdout")

    assert artifacts["training_report_md"].exists()
    assert artifacts["character_confusions_json"].exists()
    assert artifacts["training_curves_png"].exists()
    report_text = artifacts["training_report_md"].read_text(encoding="utf-8")
    assert "confusion artifacts generated" in report_text


def test_write_training_report_bundle_can_skip_training_artifact_copy(tmp_path: Path):
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "evaluation"
    tool_eval_dir = run_dir / "tool_evaluation_v01"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trainer_state.json").write_text(
        json.dumps(
            {
                "log_history": [
                    {"step": 10, "loss": 1.2, "epoch": 0.1},
                    {
                        "step": 20,
                        "eval_loss": 0.4,
                        "eval_cer": 0.3,
                        "eval_wer": 0.6,
                        "epoch": 0.2,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (eval_dir / "training_history.csv").write_text("step,loss\n10,1.2\n", encoding="utf-8")
    (eval_dir / "training_history.jsonl").write_text("{}", encoding="utf-8")
    (eval_dir / "training_curves.svg").write_text("<svg />", encoding="utf-8")
    (eval_dir / "training_curves.png").write_bytes(b"png")
    predictions_path = tool_eval_dir / "predictions_holdout.jsonl"
    tool_eval_dir.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(
        json.dumps({"gt_text": "abc", "pred_text": "abc"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    artifacts = write_training_report_bundle(
        run_dir=run_dir,
        output_dir=tool_eval_dir,
        split="holdout",
        predictions_path=predictions_path,
        include_training_artifacts=False,
    )

    assert artifacts["training_report_md"].exists()
    assert not (tool_eval_dir / "training_history.csv").exists()
    assert not (tool_eval_dir / "training_curves.svg").exists()


def test_write_training_report_bundle_degrades_gracefully_without_confusions(tmp_path: Path):
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trainer_state.json").write_text(
        json.dumps({"log_history": [{"step": 10, "loss": 1.2, "epoch": 0.1}]}),
        encoding="utf-8",
    )

    artifacts = write_training_report_bundle(run_dir=run_dir)

    assert artifacts["training_report_md"].exists()
    report_text = artifacts["training_report_md"].read_text(encoding="utf-8")
    assert "confusion data unavailable" in report_text


def test_monitor_training_run_returns_best_metric_summary(tmp_path: Path):
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trainer_state.json").write_text(
        json.dumps(
            {
                "log_history": [
                    {"step": 10, "loss": 1.2, "epoch": 0.1},
                    {"step": 20, "eval_loss": 0.4, "eval_cer": 0.3, "eval_wer": 0.6, "epoch": 0.2},
                    {
                        "step": 40,
                        "eval_loss": 0.39,
                        "eval_cer": 0.31,
                        "eval_wer": 0.5,
                        "epoch": 0.4,
                    },
                ],
                "best_model_checkpoint": str(run_dir / "checkpoint-20"),
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "finetune_meta.json").write_text(
        json.dumps({"effective_metric_for_best_model": "eval_cer"}),
        encoding="utf-8",
    )
    (run_dir / "best_model_meta.json").write_text(
        json.dumps({"source_checkpoint": str(run_dir / "checkpoint-20"), "checkpoint_step": 20}),
        encoding="utf-8",
    )
    (run_dir / "best_wer_model_meta.json").write_text(
        json.dumps({"source_checkpoint": str(run_dir / "checkpoint-40"), "checkpoint_step": 40}),
        encoding="utf-8",
    )

    summary = monitor_training_run(run_dir)

    assert summary["selection_metric"] == "eval_cer"
    assert summary["best_cer_step"] == 20
    assert summary["best_wer_step"] == 40
    assert summary["evals_since_best_cer"] == 1


def test_write_subset_manifest_persists_rows(tmp_path: Path):
    manifest_path = write_subset_manifest(
        output_path=tmp_path / "manifests" / "eval_subset_manifest.jsonl",
        rows=[{"image": "/tmp/a.png", "text": "abc", "modality": "typed"}],
        split="val",
        seed=42,
        selection="eval_fraction+eval_max_rows",
    )

    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    assert payload["selection"] == "eval_fraction+eval_max_rows"
    assert payload["modality"] == "typed"


def test_evaluate_surya_checkpoint_uses_seeded_sample_for_max_rows(tmp_path: Path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    rows = []
    for index in range(6):
        image_path = dataset_dir / "images" / f"sample_{index}.png"
        _write_png(image_path)
        rows.append({"image": str(image_path), "text": f"text-{index}"})
    _write_split(dataset_dir / "holdout.jsonl", rows)

    sampler_calls: list[tuple[int, int, int]] = []

    def _sample_rows(sample_rows, *, max_rows, seed):
        sampler_calls.append((len(sample_rows), max_rows, seed))
        return list(reversed(sample_rows))[:max_rows]

    monkeypatch.setattr(
        "modules.ocr_training.surya_eval_runtime.deterministic_sample_rows",
        _sample_rows,
    )

    class DummyPredictor:
        def __init__(self):
            self.disable_tqdm = False

        def __call__(self, images, **kwargs):
            del kwargs
            return [SimpleNamespace(text_lines=[SimpleNamespace(text="text")]) for _image in images]

    evaluate_surya_checkpoint(
        run_key="fidel_typed_synthetic",
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="holdout",
        eval_fraction=1.0,
        eval_batch_size=2,
        max_rows=3,
        seed=42,
        runtime={
            "RecognitionPredictor": lambda _foundation: DummyPredictor(),
            "TaskNames": SimpleNamespace(ocr_with_boxes="ocr"),
        },
        load_surya_eval_predictor=lambda runtime, run_dir: object(),
    )

    assert sampler_calls == [(6, 3, 42)]


def test_evaluate_surya_checkpoint_filters_by_modality(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    typed_image = dataset_dir / "images" / "typed" / "typed_0.png"
    synthetic_image = dataset_dir / "images" / "synthetic" / "synthetic_0.png"
    _write_png(typed_image)
    _write_png(synthetic_image)
    _write_split(
        dataset_dir / "holdout.jsonl",
        [
            {"image": str(typed_image), "text": "typed-text"},
            {"image": str(synthetic_image), "text": "synthetic-text"},
        ],
    )

    class DummyPredictor:
        def __init__(self):
            self.disable_tqdm = False

        def __call__(self, images, **kwargs):
            del images, kwargs
            return [SimpleNamespace(text_lines=[SimpleNamespace(text="typed-text")])]

    summary = evaluate_surya_checkpoint(
        run_key="fidel_typed_synthetic",
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="holdout",
        eval_fraction=1.0,
        eval_batch_size=2,
        max_rows=None,
        seed=42,
        modality="typed",
        runtime={
            "RecognitionPredictor": lambda _foundation: DummyPredictor(),
            "TaskNames": SimpleNamespace(ocr_with_boxes="ocr"),
        },
        load_surya_eval_predictor=lambda runtime, run_dir: object(),
    )

    assert summary["num_rows"] == 1
    assert (run_dir / "tool_evaluation" / "summary_holdout_typed.json").exists()


def test_evaluate_surya_modalities_returns_combined_summary(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    typed_image = dataset_dir / "images" / "typed" / "typed_0.png"
    synthetic_image = dataset_dir / "images" / "synthetic" / "synthetic_0.png"
    _write_png(typed_image)
    _write_png(synthetic_image)
    _write_split(
        dataset_dir / "holdout.jsonl",
        [
            {"image": str(typed_image), "text": "typed-text"},
            {"image": str(synthetic_image), "text": "synthetic-text"},
        ],
    )

    class DummyPredictor:
        def __init__(self):
            self.disable_tqdm = False

        def __call__(self, images, **kwargs):
            del kwargs
            outputs = []
            for _image in images:
                outputs.append(SimpleNamespace(text_lines=[SimpleNamespace(text="typed-text")]))
            return outputs

    summary = evaluate_surya_modalities(
        run_key="fidel_typed_synthetic",
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        split="holdout",
        eval_fraction=1.0,
        eval_batch_size=2,
        max_rows=None,
        seed=42,
        modalities=["typed", "synthetic"],
        runtime={
            "RecognitionPredictor": lambda _foundation: DummyPredictor(),
            "TaskNames": SimpleNamespace(ocr_with_boxes="ocr"),
        },
        load_surya_eval_predictor=lambda runtime, run_dir: object(),
    )

    assert set(summary["modalities"]) == {"typed", "synthetic"}
    assert (run_dir / "tool_evaluation" / "summary_holdout_modalities.json").exists()
