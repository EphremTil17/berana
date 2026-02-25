"""Integration tests asserting command-stage chaining and artifact resolution invariants."""

import json
from pathlib import Path

from utils.run_registry import register_latest_run, resolve_required_input


def test_orchestration_stage_chaining(tmp_path: Path):
    """
    Assert that Stage A's output is registered and Stage B can resolve it
    automatically using pure registry pointers (no hardcoded paths passed between them).
    """
    # 1. Stage A (e.g., crop-columns) completes and registers its manifest
    stage_a_dir = tmp_path / "crop_columns_v01"
    stage_a_dir.mkdir(parents=True, exist_ok=True)
    manifest_a = stage_a_dir / "cropping_manifest.json"
    manifest_a.write_text('{"status": "success"}', encoding="utf-8")

    register_latest_run(
        stage="crop-columns",
        doc_stem="test_doc",
        run_dir=stage_a_dir,
        artifacts={"cropping_manifest": str(manifest_a)},
        root_dir=tmp_path / ".registry",
    )

    # 2. Stage B (e.g., ocr) starts up and asks the registry for Stage A's artifact
    # It does not know 'stage_a_dir', it only knows 'test_doc' and 'crop-columns'
    resolved_artifact = resolve_required_input(
        upstream_stage="crop-columns",
        doc_stem="test_doc",
        artifact_key="cropping_manifest",
        root_dir=tmp_path / ".registry",
    )

    assert resolved_artifact == manifest_a
    assert json.loads(resolved_artifact.read_text(encoding="utf-8")) == {"status": "success"}
