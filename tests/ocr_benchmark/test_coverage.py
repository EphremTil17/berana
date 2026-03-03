import json
from pathlib import Path

import pytest

from modules.ocr_benchmark.coverage import build_coverage_report, ensure_coverage_gate
from modules.ocr_benchmark.dataset import write_manifest
from schemas.ocr_benchmark import LineManifestRow


def _manifest_rows() -> list[LineManifestRow]:
    return [
        LineManifestRow(
            line_id="L001",
            doc_stem="doc_cov",
            page_id="page_001",
            column_key="geez",
            lang_prompt="<gez>",
            image_path="output/ocr_benchmark/doc_cov_v01/prep/images/page_001/geez/L001.png",
            split="train",
            gt_text="ሀለመ",
            source_run_dir="output/ocr_benchmark/doc_cov_v01",
        ),
        LineManifestRow(
            line_id="L002",
            doc_stem="doc_cov",
            page_id="page_002",
            column_key="amharic",
            lang_prompt="<amh>",
            image_path="output/ocr_benchmark/doc_cov_v01/prep/images/page_002/amharic/L002.png",
            split="holdout",
            gt_text="በተነ",
            source_run_dir="output/ocr_benchmark/doc_cov_v01",
        ),
    ]


def _write_charset_config(path: Path, min_count: int = 1) -> None:
    payload = {
        "schema_version": "1.0",
        "name": "test",
        "description": "test cfg",
        "allowed_scripts": ["Ethiopic"],
        "tiers": {
            "high": {"min_count": min_count, "chars": ["ሀ", "በ"]},
            "medium": {"min_count": min_count, "chars": ["ለ"]},
            "rare": {"min_count": min_count, "chars": ["መ"]},
            "optional": {"min_count": 0, "chars": ["።"]},
        },
        "ignored_chars": [" "],
        "normalization_profile": "ethiopic_v1",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_build_coverage_report_outputs(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(_manifest_rows(), manifest_path)
    charset_path = tmp_path / "charset.json"
    _write_charset_config(charset_path, min_count=1)

    doc_root = tmp_path / "doc_cov_v01"
    doc_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "modules.ocr_benchmark.coverage.resolve_doc_benchmark_root", lambda doc_stem: doc_root
    )
    monkeypatch.setattr("modules.ocr_benchmark.coverage.register_latest_run", lambda **kwargs: None)

    report, out_dir = build_coverage_report(
        doc_stem="doc_cov",
        manifest_path=manifest_path,
        charset_config_path=charset_path,
    )
    assert out_dir == doc_root / "coverage"
    assert (out_dir / "coverage_report.json").exists()
    assert report.coverage_status is True
    assert report.split_stats["train"]["num_rows"] == 1
    assert report.split_stats["holdout"]["num_rows"] == 1


def test_ensure_coverage_gate_enforces(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(_manifest_rows(), manifest_path)
    charset_path = tmp_path / "charset.json"
    _write_charset_config(charset_path, min_count=5)

    doc_root = tmp_path / "doc_cov_v01"
    doc_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "modules.ocr_benchmark.coverage.resolve_doc_benchmark_root", lambda doc_stem: doc_root
    )
    monkeypatch.setattr("modules.ocr_benchmark.coverage.register_latest_run", lambda **kwargs: None)

    with pytest.raises(ValueError, match="Coverage gate failed"):
        ensure_coverage_gate(
            doc_stem="doc_cov",
            manifest_path=manifest_path,
            charset_config_path=charset_path,
            enforce=True,
        )
