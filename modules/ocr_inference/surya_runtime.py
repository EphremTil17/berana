from __future__ import annotations

from pathlib import Path
from typing import Any

from modules.ocr_inference.schemas import PredictorBundle
from modules.ocr_training.surya_artifacts import load_finetune_meta
from modules.ocr_training.surya_model import load_surya_eval_predictor, require_surya


def _resolve_checkpoint_source(checkpoint_dir: Path) -> tuple[Path, Path | None]:
    """Resolve a training-run directory plus optional explicit checkpoint path."""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if (checkpoint_dir / "finetune_meta.json").exists():
        return checkpoint_dir, None
    if (
        checkpoint_dir.name.startswith("checkpoint-")
        and (checkpoint_dir.parent / "finetune_meta.json").exists()
    ):
        return checkpoint_dir.parent, checkpoint_dir
    raise ValueError(
        "`--checkpoint-dir` must point to a Surya training run directory containing `finetune_meta.json` "
        "or to one of its `checkpoint-*` subdirectories."
    )


def build_surya_predictor(
    *, zero_shot: bool, checkpoint_dir: Path | None
) -> tuple[dict[str, Any], PredictorBundle]:
    """Create the Surya predictor bundle for zero-shot or fine-tuned checkpoint inference."""
    runtime = require_surya()

    if zero_shot:
        foundation_predictor = runtime["FoundationPredictor"]()
        predictor = runtime["RecognitionPredictor"](foundation_predictor)
        predictor.disable_tqdm = True
        return runtime, PredictorBundle(
            predictor=predictor,
            model_info={
                "model_mode": "zero_shot",
                "checkpoint_dir": None,
                "run_dir": None,
            },
        )

    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required when zero_shot=False")

    run_dir, checkpoint_path = _resolve_checkpoint_source(checkpoint_dir)
    foundation_predictor = load_surya_eval_predictor(
        runtime,
        run_dir,
        load_finetune_meta,
        checkpoint_path=checkpoint_path,
        checkpoint_target="best_cer",
    )
    predictor = runtime["RecognitionPredictor"](foundation_predictor)
    predictor.disable_tqdm = True
    finetune_meta = load_finetune_meta(run_dir) or {}
    return runtime, PredictorBundle(
        predictor=predictor,
        model_info={
            "model_mode": "checkpoint",
            "checkpoint_dir": str(checkpoint_dir),
            "run_dir": str(run_dir),
            "base_checkpoint": finetune_meta.get("base_checkpoint"),
            "finetune_strategy": finetune_meta.get("finetune_strategy"),
        },
    )


def build_surya_detection_predictor():
    """Load Surya line detection predictor for diagnostic overlays."""
    try:
        from surya.detection import DetectionPredictor
    except ImportError as exc:
        raise RuntimeError(
            "Surya detection dependencies are not installed in this environment. "
            "Install `surya-ocr` first."
        ) from exc
    return DetectionPredictor()
