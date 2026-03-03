import json
from pathlib import Path

from modules.ocr_benchmark.label_studio_sync import create_import_tasks


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_create_import_tasks_split_filters_and_prefill(monkeypatch, tmp_path):
    crops_path = tmp_path / "candidate_crops.json"
    _write_json(
        crops_path,
        [
            {
                "line_id": "L_train_001",
                "doc_stem": "doc_x",
                "page_id": "page_001",
                "column_key": "geez",
                "image_path": "output/ocr_benchmark/doc_x_v01/prep/images/page_001/geez/L_train_001.png",
                "split": "train",
                "source_run_dir": "output/ocr_benchmark/doc_x_v01",
            },
            {
                "line_id": "L_holdout_001",
                "doc_stem": "doc_x",
                "page_id": "page_002",
                "column_key": "amharic",
                "image_path": "output/ocr_benchmark/doc_x_v01/prep/images/page_002/amharic/L_holdout_001.png",
                "split": "holdout",
                "source_run_dir": "output/ocr_benchmark/doc_x_v01",
            },
        ],
    )

    preds_path = tmp_path / "baseline_predictions.jsonl"
    _write_jsonl(
        preds_path,
        [
            {
                "line_id": "L_holdout_001",
                "raw_pred": "ጽሁፍ",
                "confidence": 0.88,
                "doc_stem": "doc_x",
                "page_id": "page_002",
                "column_key": "amharic",
                "image_path": "output/ocr_benchmark/doc_x_v01/prep/images/page_002/amharic/L_holdout_001.png",
                "split": "holdout",
            }
        ],
    )

    def _mock_resolve_required_input(*, upstream_stage, doc_stem, artifact_key, root_dir=None):
        assert upstream_stage == "ocr-benchmark-prepare"
        assert artifact_key == "crops_metadata"
        assert doc_stem == "doc_x"
        return crops_path

    def _mock_load_latest_run(stage, doc_stem, root_dir=None):
        if stage == "ocr-benchmark-surya-zero" and doc_stem == "doc_x":
            return {"artifacts": {"baseline_predictions_jsonl": str(preds_path)}}
        return None

    monkeypatch.setattr(
        "modules.ocr_benchmark.label_studio_sync.resolve_required_input",
        _mock_resolve_required_input,
    )
    monkeypatch.setattr(
        "modules.ocr_benchmark.label_studio_sync.load_latest_run",
        _mock_load_latest_run,
    )
    monkeypatch.setattr(
        "modules.ocr_benchmark.label_studio_sync._to_label_studio_local_files_url",
        lambda image_path: f"/data/local-files/?d={image_path}",
    )

    out_holdout = tmp_path / "holdout_tasks.json"
    create_import_tasks("doc_x", out_holdout, split="holdout")
    holdout_tasks = json.loads(out_holdout.read_text(encoding="utf-8"))
    assert len(holdout_tasks) == 1
    assert holdout_tasks[0]["data"]["split_hint"] == "holdout"
    assert len(holdout_tasks[0]["predictions"]) == 1

    out_train = tmp_path / "train_tasks.json"
    create_import_tasks("doc_x", out_train, split="train")
    train_tasks = json.loads(out_train.read_text(encoding="utf-8"))
    assert len(train_tasks) == 1
    assert train_tasks[0]["data"]["split_hint"] == "train"
    assert train_tasks[0]["predictions"] == []

    out_all = tmp_path / "all_tasks.json"
    create_import_tasks("doc_x", out_all, split="all")
    all_tasks = json.loads(out_all.read_text(encoding="utf-8"))
    assert len(all_tasks) == 2
