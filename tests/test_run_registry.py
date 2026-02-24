from pathlib import Path

from utils.run_registry import (
    load_latest_run,
    next_versioned_dir,
    register_latest_run,
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
