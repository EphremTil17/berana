from __future__ import annotations

from pathlib import Path

from utils.run_registry import load_latest_run, register_latest_run

STAGE_FIDEL_EXTRACT = "ocr-training-fidel-extract"
STAGE_SURYA_DATASET = "ocr-training-surya-dataset"
STAGE_SURYA_FINETUNE = "ocr-training-surya-finetune"
STAGE_SURYA_EVALUATE = "ocr-training-surya-evaluate"


def register_training_stage(
    *,
    stage: str,
    run_key: str,
    run_dir: Path,
    artifacts: dict[str, str],
    metadata: dict,
) -> Path:
    """Register training stage output in the shared run registry."""
    return register_latest_run(
        stage=stage,
        doc_stem=run_key,
        run_dir=run_dir,
        artifacts=artifacts,
        metadata=metadata,
    )


def load_training_stage(stage: str, run_key: str) -> dict | None:
    """Load a previously registered training stage pointer."""
    return load_latest_run(stage, run_key)
