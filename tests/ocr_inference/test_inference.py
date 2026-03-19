from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image
from typer.testing import CliRunner

import berana
from modules.cli.common import parse_page_selection
from modules.ocr_inference.inputs import (
    OCRInferenceInputError,
    collect_crop_ocr_tasks,
    iter_source_images,
    resolve_source_artifacts,
    resolve_source_artifacts_optional,
)
from modules.ocr_inference.outputs import write_crop_text_output, write_page_text_output
from modules.ocr_inference.pipeline import run_source_ocr_inference_pipeline
from modules.ocr_inference.schemas import OCRTask, PredictorBundle, SourceArtifacts
from modules.ocr_inference.surya_runtime import build_surya_predictor

runner = CliRunner()


def _create_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 12), color="white").save(path)


def test_collect_crop_ocr_tasks_respects_page_filters(tmp_path: Path):
    img_a = tmp_path / "spliced" / "page_001" / "amharic.png"
    img_b = tmp_path / "spliced" / "page_002" / "geez.png"
    _create_image(img_a)
    _create_image(img_b)
    manifest_path = tmp_path / "cropping_manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "page_id": "page_001",
                    "source_page": 1,
                    "status": "SUCCESS",
                    "strip_paths": {"amharic": str(img_a)},
                },
                {
                    "page_id": "page_002",
                    "source_page": 2,
                    "status": "SUCCESS",
                    "strip_paths": {"geez": str(img_b)},
                },
            ]
        ),
        encoding="utf-8",
    )
    source = SourceArtifacts(
        crop_run_dir=tmp_path / "crop_run",
        cropping_manifest=manifest_path,
        spliced_dir=tmp_path / "spliced",
        crop_registry_pointer={"run_dir": str(tmp_path / "crop_run")},
    )

    tasks = collect_crop_ocr_tasks(
        source_path=tmp_path / "sample.pdf",
        source_artifacts=source,
        selected_pages=[2],
    )

    assert len(tasks) == 1
    assert tasks[0].language == "geez"
    assert tasks[0].page_number == 2


def test_resolve_source_artifacts_requires_crop_columns(monkeypatch):
    monkeypatch.setattr(
        "modules.ocr_inference.inputs.load_latest_run", lambda *args, **kwargs: None
    )

    try:
        resolve_source_artifacts(doc_stem="missing_doc")
    except OCRInferenceInputError as exc:
        assert "Run `berana.py crop-columns" in str(exc)
    else:
        raise AssertionError("Expected missing crop-columns artifacts to fail.")


def test_resolve_source_artifacts_optional_returns_none(monkeypatch):
    monkeypatch.setattr(
        "modules.ocr_inference.inputs.load_latest_run", lambda *args, **kwargs: None
    )

    assert resolve_source_artifacts_optional(doc_stem="missing_doc") is None


def test_parse_page_selection_deduplicates_and_sorts():
    assert parse_page_selection("5-6,2,6,3") == [2, 3, 5, 6]


def test_write_text_outputs_create_expected_layout(tmp_path: Path):
    run_dir = tmp_path / "output" / "sample_v01"
    page_txt = write_page_text_output(run_dir=run_dir, page_number=1, text="ሰላም\nዓለም")
    crop_txt = write_crop_text_output(
        run_dir=run_dir,
        language="amharic",
        page_number=3,
        text="በሰላም",
    )

    assert page_txt.read_text(encoding="utf-8") == "ሰላም\nዓለም"
    assert crop_txt.read_text(encoding="utf-8") == "በሰላም"
    assert page_txt.name == "page_001.txt"
    assert crop_txt == run_dir / "amharic" / "page_003" / "ocr.txt"


def test_iter_source_images_supports_image_input(tmp_path: Path):
    image_path = tmp_path / "sample.png"
    _create_image(image_path)

    rows = list(iter_source_images(source_path=image_path, selected_pages=None))

    assert len(rows) == 1
    assert rows[0][0] == 1
    assert rows[0][1].size == (32, 12)


def test_build_surya_predictor_uses_zero_shot_runtime(monkeypatch):
    foundation_instance = object()
    recognition_instance = SimpleNamespace(disable_tqdm=False)
    foundation_ctor = Mock(return_value=foundation_instance)
    recognition_ctor = Mock(return_value=recognition_instance)
    monkeypatch.setattr(
        "modules.ocr_inference.surya_runtime.require_surya",
        lambda: {
            "FoundationPredictor": foundation_ctor,
            "RecognitionPredictor": recognition_ctor,
        },
    )

    _runtime, bundle = build_surya_predictor(zero_shot=True, checkpoint_dir=None)

    foundation_ctor.assert_called_once_with()
    recognition_ctor.assert_called_once_with(foundation_instance)
    assert bundle.model_info["model_mode"] == "zero_shot"
    assert bundle.predictor.disable_tqdm is True


def test_build_surya_predictor_uses_checkpoint_loader(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "finetune_meta.json").write_text("{}", encoding="utf-8")
    predictor = SimpleNamespace(disable_tqdm=False)
    foundation_predictor = object()
    loader = Mock(return_value=foundation_predictor)
    recognition_ctor = Mock(return_value=predictor)
    monkeypatch.setattr(
        "modules.ocr_inference.surya_runtime.require_surya",
        lambda: {
            "RecognitionPredictor": recognition_ctor,
        },
    )
    monkeypatch.setattr("modules.ocr_inference.surya_runtime.load_surya_eval_predictor", loader)
    monkeypatch.setattr(
        "modules.ocr_inference.surya_runtime.load_finetune_meta",
        lambda _p: {"finetune_strategy": "lora"},
    )

    _runtime, bundle = build_surya_predictor(zero_shot=False, checkpoint_dir=run_dir)

    loader.assert_called_once()
    recognition_ctor.assert_called_once_with(foundation_predictor)
    assert bundle.model_info["model_mode"] == "checkpoint"
    assert bundle.predictor.disable_tqdm is True


def test_run_source_ocr_inference_pipeline_writes_generic_page_outputs(monkeypatch, tmp_path: Path):
    image_path = tmp_path / "sample.png"
    _create_image(image_path)

    class FakePredictor:
        disable_tqdm = True

        def __call__(self, images, **kwargs):
            del images, kwargs
            return [
                SimpleNamespace(
                    text_lines=[
                        SimpleNamespace(
                            text="ሰላም",
                            confidence=0.88,
                            polygon=[[2, 2], [24, 2], [24, 10], [2, 10]],
                            bbox=[2, 2, 24, 10],
                        ),
                        SimpleNamespace(
                            text="ዓለም",
                            confidence=0.85,
                            polygon=[[2, 12], [24, 12], [24, 20], [2, 20]],
                            bbox=[2, 12, 24, 20],
                        ),
                    ]
                )
            ]

    monkeypatch.setattr(
        "modules.ocr_inference.pipeline.resolve_source_artifacts_optional", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "modules.ocr_inference.pipeline.build_surya_predictor",
        lambda **kwargs: (
            {"TaskNames": SimpleNamespace(ocr_with_boxes="ocr_with_boxes")},
            PredictorBundle(
                predictor=FakePredictor(),
                model_info={
                    "model_mode": "checkpoint",
                    "checkpoint_dir": str(tmp_path / "run"),
                    "run_dir": str(tmp_path / "run"),
                },
            ),
        ),
    )
    monkeypatch.setattr(
        "modules.ocr_inference.pipeline.build_surya_detection_predictor", lambda: object()
    )
    register_calls = []
    monkeypatch.setattr(
        "modules.ocr_inference.pipeline.register_latest_run",
        lambda **kwargs: register_calls.append(kwargs),
    )

    run_dir = run_source_ocr_inference_pipeline(
        source_path=image_path,
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "run",
        zero_shot=False,
        eval_batch_size=8,
        dataloader_num_workers=0,
        diagnose=True,
        selected_pages=None,
    )

    assert run_dir.name == "sample_v01"
    assert (run_dir / "page_001.txt").read_text(encoding="utf-8") == "ሰላም\nዓለም"
    assert (run_dir / "images" / "page_001__annotated.png").exists()
    assert (run_dir / "images" / "page_001__annotations.json").exists()
    assert register_calls and register_calls[0]["stage"] == "ocr"
    assert register_calls[0]["metadata"]["used_crop_layout"] is False


def test_run_source_ocr_inference_pipeline_uses_crop_layout_when_available(
    monkeypatch, tmp_path: Path
):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("placeholder", encoding="utf-8")
    image_path = tmp_path / "spliced" / "page_001" / "amharic.png"
    _create_image(image_path)
    tasks = [
        OCRTask(
            doc_stem="sample",
            pdf_path=pdf_path,
            page_id="page_001",
            page_number=1,
            language="amharic",
            image_path=image_path,
            source_page=1,
            ordering_index=0,
            crop_run_dir=tmp_path / "crop_run",
        )
    ]
    source = SourceArtifacts(
        crop_run_dir=tmp_path / "crop_run",
        cropping_manifest=tmp_path / "cropping_manifest.json",
        spliced_dir=tmp_path / "spliced",
        crop_registry_pointer={"run_dir": str(tmp_path / "crop_run")},
    )
    source.cropping_manifest.write_text("[]", encoding="utf-8")

    class FakePredictor:
        disable_tqdm = True

        def __call__(self, images, **kwargs):
            del images, kwargs
            return [
                SimpleNamespace(
                    text_lines=[
                        SimpleNamespace(
                            text="ሰላም",
                            confidence=0.88,
                            polygon=[[2, 2], [24, 2], [24, 10], [2, 10]],
                            bbox=[2, 2, 24, 10],
                        )
                    ]
                )
            ]

    monkeypatch.setattr(
        "modules.ocr_inference.pipeline.resolve_source_artifacts_optional", lambda **kwargs: source
    )
    monkeypatch.setattr(
        "modules.ocr_inference.pipeline.collect_crop_ocr_tasks", lambda **kwargs: tasks
    )
    monkeypatch.setattr(
        "modules.ocr_inference.pipeline.build_surya_predictor",
        lambda **kwargs: (
            {"TaskNames": SimpleNamespace(ocr_with_boxes="ocr_with_boxes")},
            PredictorBundle(
                predictor=FakePredictor(),
                model_info={
                    "model_mode": "checkpoint",
                    "checkpoint_dir": str(tmp_path / "run"),
                    "run_dir": str(tmp_path / "run"),
                },
            ),
        ),
    )
    monkeypatch.setattr(
        "modules.ocr_inference.pipeline.build_surya_detection_predictor", lambda: object()
    )
    register_calls = []
    monkeypatch.setattr(
        "modules.ocr_inference.pipeline.register_latest_run",
        lambda **kwargs: register_calls.append(kwargs),
    )

    run_dir = run_source_ocr_inference_pipeline(
        source_path=pdf_path,
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "run",
        zero_shot=False,
        eval_batch_size=4,
        dataloader_num_workers=0,
        diagnose=True,
        selected_pages=None,
    )

    assert (run_dir / "amharic" / "page_001" / "ocr.txt").read_text(encoding="utf-8") == "ሰላም"
    assert (run_dir / "images" / "amharic" / "page_001" / "amharic__annotated.png").exists()
    assert register_calls[0]["metadata"]["used_crop_layout"] is True


def test_ocr_cli_requires_checkpoint_or_zero_shot():
    result = runner.invoke(berana.app, ["ocr", "--source", "fake.pdf"])

    assert result.exit_code != 0
    assert "Provide --checkpoint-dir or explicitly pass --zero-shot" in result.output


def test_ocr_cli_routes_checkpoint_mode(monkeypatch, tmp_path: Path):
    captured = {}
    monkeypatch.setattr(
        "modules.cli.ocr_commands.ensure_source_exists", lambda source, context_label: Path(source)
    )
    monkeypatch.setattr(
        "modules.ocr_inference.pipeline.run_source_ocr_inference_pipeline",
        lambda **kwargs: captured.update(kwargs),
    )

    result = runner.invoke(
        berana.app,
        [
            "ocr",
            "--source",
            str(tmp_path / "sample.pdf"),
            "--checkpoint-dir",
            str(tmp_path / "run"),
            "--eval-batch-size",
            "24",
            "--diagnose",
            "--pages",
            "1-3,5",
        ],
    )

    assert result.exit_code == 0
    assert captured["checkpoint_dir"] == tmp_path / "run"
    assert captured["zero_shot"] is False
    assert captured["diagnose"] is True
    assert captured["eval_batch_size"] == 24
    assert captured["selected_pages"] == [1, 2, 3, 5]


def test_ocr_cli_routes_zero_shot_mode(monkeypatch, tmp_path: Path):
    captured = {}
    monkeypatch.setattr(
        "modules.cli.ocr_commands.ensure_source_exists", lambda source, context_label: Path(source)
    )
    monkeypatch.setattr(
        "modules.ocr_inference.pipeline.run_source_ocr_inference_pipeline",
        lambda **kwargs: captured.update(kwargs),
    )

    result = runner.invoke(
        berana.app,
        [
            "ocr",
            "--source",
            str(tmp_path / "sample.pdf"),
            "--zero-shot",
        ],
    )

    assert result.exit_code == 0
    assert captured["zero_shot"] is True
    assert captured["checkpoint_dir"] is None


def test_ocr_cli_default_eval_batch_size_is_one(monkeypatch, tmp_path: Path):
    captured = {}
    monkeypatch.setattr(
        "modules.cli.ocr_commands.ensure_source_exists", lambda source, context_label: Path(source)
    )
    monkeypatch.setattr(
        "modules.ocr_inference.pipeline.run_source_ocr_inference_pipeline",
        lambda **kwargs: captured.update(kwargs),
    )

    result = runner.invoke(
        berana.app,
        [
            "ocr",
            "--source",
            str(tmp_path / "sample.pdf"),
            "--zero-shot",
        ],
    )

    assert result.exit_code == 0
    assert captured["eval_batch_size"] == 1


def test_build_surya_detection_predictor_loads_detector(monkeypatch):
    detector = object()
    monkeypatch.setattr("surya.detection.DetectionPredictor", lambda: detector, raising=False)
