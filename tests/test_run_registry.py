from pathlib import Path

import pytest

from utils.run_registry import (
    RegistryCorruptionError,
    load_latest_run,
    next_versioned_dir,
    register_latest_run,
    registry_file,
    resolve_required_input,
)


def test_next_versioned_dir_increments(tmp_path: Path):
    """Version allocation should increment by existing `_vNN` directories."""
    base = tmp_path / "runs"
    (base / "doc_v01").mkdir(parents=True)
    (base / "doc_v02").mkdir(parents=True)

    next_dir = next_versioned_dir(base, "doc")

    assert next_dir.name == "doc_v03"


def test_registry_round_trip_and_resolve_required_artifact(tmp_path: Path):
    """Latest pointer registration should be readable and artifact-resolvable."""
    run_dir = tmp_path / "output" / "ocr_artifacts" / "doc_v01"
    run_dir.mkdir(parents=True)
    manifest = run_dir / "cropping_manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    pointer_path = register_latest_run(
        stage="crop-columns",
        doc_stem="doc",
        run_dir=run_dir,
        artifacts={"cropping_manifest": str(manifest)},
        metadata={"rectify_mode": "rotate+homography"},
        root_dir=tmp_path / "registry",
    )
    assert pointer_path.exists()

    pointer = load_latest_run("crop-columns", "doc", root_dir=tmp_path / "registry")
    assert pointer is not None
    assert pointer["run_dir"] == str(run_dir)

    resolved = resolve_required_input(
        upstream_stage="crop-columns",
        doc_stem="doc",
        artifact_key="cropping_manifest",
        root_dir=tmp_path / "registry",
    )
    assert resolved == manifest


def test_registry_corruption_handling(tmp_path: Path):
    """Malformed or incomplete registry files should raise RegistryCorruptionError."""
    stage = "test-stage"
    doc_stem = "test-doc"
    path = registry_file(stage, doc_stem, root_dir=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text("{malformed json", encoding="utf-8")
    with pytest.raises(RegistryCorruptionError, match="Failed to decode"):
        load_latest_run(stage, doc_stem, root_dir=tmp_path)

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(RegistryCorruptionError, match="invalid shape"):
        load_latest_run(stage, doc_stem, root_dir=tmp_path)

    path.write_text('{"stage": "test-stage"}', encoding="utf-8")
    with pytest.raises(RegistryCorruptionError, match="missing required keys"):
        load_latest_run(stage, doc_stem, root_dir=tmp_path)

    path.write_text(
        (
            '{"schema_version":"0.9","stage":"test-stage","doc_stem":"test-doc",'
            '"run_dir":"/tmp/run","updated_at_utc":"2026-02-25T00:00:00+00:00"}'
        ),
        encoding="utf-8",
    )
    with pytest.raises(RegistryCorruptionError, match="unsupported schema_version"):
        load_latest_run(stage, doc_stem, root_dir=tmp_path)
