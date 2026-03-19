import json

from modules.ocr_benchmark.coverage import build_annotation_queue
from modules.ocr_benchmark.dataset import write_manifest
from schemas.ocr_benchmark import ColumnKey, DatasetSplit, LangPrompt, LineManifestRow
from schemas.ocr_coverage import CoverageDeficit, CoverageReport, CoverageTier


def test_coverage_queue_prioritizes_deficit_hits(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(
        [
            LineManifestRow(
                line_id="L_labeled",
                doc_stem="doc_q",
                page_id="page_001",
                column_key=ColumnKey.GEEZ,
                lang_prompt=LangPrompt.GEEZ,
                image_path="output/ocr_benchmark/doc_q_v01/prep/images/page_001/geez/L_labeled.png",
                split=DatasetSplit.HOLDOUT,
                gt_text="ሀ",
                source_run_dir="output/ocr_benchmark/doc_q_v01",
            )
        ],
        manifest_path,
    )

    crops_path = tmp_path / "candidate_crops.json"
    crops_path.write_text(
        json.dumps(
            [
                {
                    "line_id": "L_labeled",
                    "image_path": "output/ocr_benchmark/doc_q_v01/prep/images/page_001/geez/L_labeled.png",
                    "column_key": "geez",
                },
                {
                    "line_id": "L_best",
                    "image_path": "output/ocr_benchmark/doc_q_v01/prep/images/page_002/geez/L_best.png",
                    "column_key": "geez",
                },
                {
                    "line_id": "L_low",
                    "image_path": "output/ocr_benchmark/doc_q_v01/prep/images/page_003/amharic/L_low.png",
                    "column_key": "amharic",
                },
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    preds_path = tmp_path / "preds.jsonl"
    with preds_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"line_id": "L_best", "raw_pred": "ሀሀ", "confidence": 0.9}) + "\n")
        f.write(json.dumps({"line_id": "L_low", "raw_pred": "ሀ", "confidence": 0.1}) + "\n")

    def _mock_load_latest_run(stage, doc_stem, root_dir=None):
        if stage == "ocr-benchmark-prepare":
            return {"artifacts": {"crops_metadata": str(crops_path)}}
        if stage == "ocr-benchmark-surya-zero":
            return {"artifacts": {"baseline_predictions_jsonl": str(preds_path)}}
        return None

    report = CoverageReport(
        doc_stem="doc_q",
        manifest_hash="m",
        charset_config_hash="c",
        coverage_status=False,
        split_stats={},
        missing_chars=[],
        under_threshold=[
            CoverageDeficit(
                tier=CoverageTier.HIGH,
                char="ሀ",
                count=0,
                min_required=20,
                deficit=20,
            )
        ],
        recommendations=[],
    )
    coverage_dir = tmp_path / "doc_q_v01" / "coverage"
    coverage_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "modules.ocr_benchmark.coverage.build_coverage_report",
        lambda **kwargs: (report, coverage_dir),
    )
    monkeypatch.setattr("modules.ocr_benchmark.coverage.load_latest_run", _mock_load_latest_run)
    monkeypatch.setattr(
        "modules.ocr_benchmark.coverage.resolve_doc_benchmark_root",
        lambda doc_stem: tmp_path / "doc_q_v01",
    )
    monkeypatch.setattr("modules.ocr_benchmark.coverage.register_latest_run", lambda **kwargs: None)

    out_path = build_annotation_queue(
        doc_stem="doc_q",
        manifest_path=manifest_path,
        charset_config_path=tmp_path / "charset.json",
        max_items=2,
    )
    lines = [json.loads(x) for x in out_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 2
    assert lines[0]["line_id"] == "L_best"
