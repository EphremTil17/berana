from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from config.settings import settings

REGISTRY_ROOT = settings.OUTPUT_DIR / ".registry"


class RegistryError(Exception):
    """Base exception for run registry errors."""


class RegistryCorruptionError(RegistryError):
    """Raised when a registry pointer file is malformed or missing required keys."""


def next_versioned_dir(base_dir: Path, doc_stem: str) -> Path:
    """Return next versioned run directory in format `<doc_stem>_vNN`."""
    base_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(doc_stem)}_v(\d+)$")
    latest_version = 0
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            latest_version = max(latest_version, int(match.group(1)))
    return base_dir / f"{doc_stem}_v{latest_version + 1:02d}"


def registry_file(stage: str, doc_stem: str, root_dir: Path | None = None) -> Path:
    """Return canonical latest-pointer file path for a stage+document pair."""
    base = root_dir or REGISTRY_ROOT
    return base / stage / f"{doc_stem}.json"


def register_latest_run(
    *,
    stage: str,
    doc_stem: str,
    run_dir: Path,
    artifacts: dict[str, str] | None = None,
    metadata: dict | None = None,
    root_dir: Path | None = None,
) -> Path:
    """Persist the latest successful run pointer for a stage+document pair."""
    pointer = {
        "schema_version": "1.0",
        "stage": stage,
        "doc_stem": doc_stem,
        "run_dir": str(run_dir),
        "artifacts": artifacts or {},
        "metadata": metadata or {},
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }
    path = registry_file(stage, doc_stem, root_dir=root_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(pointer, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        Path(tmp_path).replace(path)
    except Exception:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()
        raise

    return path


def load_latest_run(stage: str, doc_stem: str, root_dir: Path | None = None) -> dict | None:
    """Load the latest run pointer if it exists."""
    path = registry_file(stage, doc_stem, root_dir=root_dir)
    if not path.exists():
        return None

    try:
        pointer = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryCorruptionError(
            f"Failed to decode registry file {path}. File is corrupted: {exc}"
        ) from exc

    if not isinstance(pointer, dict):
        raise RegistryCorruptionError(
            f"Registry pointer {path} contains invalid shape (expected dict, got {type(pointer).__name__})."
        )

    required_keys = {"schema_version", "stage", "doc_stem", "run_dir", "updated_at_utc"}
    missing = required_keys - set(pointer.keys())
    if missing:
        raise RegistryCorruptionError(
            f"Registry pointer {path} is missing required keys: {missing}"
        )

    if pointer["schema_version"] != "1.0":
        raise RegistryCorruptionError(
            f"Registry pointer {path} has unsupported schema_version "
            f"'{pointer['schema_version']}' (expected '1.0')."
        )

    return pointer


def resolve_required_input(
    *,
    upstream_stage: str,
    doc_stem: str,
    artifact_key: str,
    root_dir: Path | None = None,
) -> Path:
    """Resolve a required upstream artifact from the latest-run pointer."""
    pointer = load_latest_run(upstream_stage, doc_stem, root_dir=root_dir)
    if pointer is None:
        raise FileNotFoundError(
            f"No latest pointer found for stage '{upstream_stage}' and document '{doc_stem}'."
        )

    artifacts = pointer.get("artifacts", {})
    if artifact_key not in artifacts:
        raise KeyError(
            f"Pointer for stage '{upstream_stage}' missing required artifact '{artifact_key}'."
        )

    candidate = Path(artifacts[artifact_key])
    if not candidate.exists():
        raise FileNotFoundError(
            f"Resolved artifact path does not exist for '{artifact_key}': {candidate}"
        )
    return candidate
