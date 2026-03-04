from __future__ import annotations

from pathlib import Path


def validate_berana_gold_inputs(
    *,
    extra_manifest: Path | None,
    extra_images_root: Path | None,
    extra_weight: float,
) -> None:
    """Validate optional Berana gold adapter inputs for future integration.

    Phase 1 intentionally reserves this interface without ingesting rows.
    """
    if extra_weight <= 0.0 or extra_weight >= 1.0:
        raise ValueError("--extra-weight must be between 0 and 1 (exclusive).")
    if (extra_manifest is None) ^ (extra_images_root is None):
        raise ValueError(
            "--extra-manifest and --extra-images-root must be provided together, or omitted."
        )
    if extra_manifest and not extra_manifest.exists():
        raise FileNotFoundError(f"Extra manifest not found: {extra_manifest}")
    if extra_images_root and not extra_images_root.exists():
        raise FileNotFoundError(f"Extra images root not found: {extra_images_root}")
