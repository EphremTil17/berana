from __future__ import annotations

import hashlib
import json
from math import ceil
from pathlib import Path

from config.settings import settings
from modules.ocr_training.checkpointing import resolve_latest_checkpoint
from modules.ocr_training.runtime.strategy_catalog import (
    resolve_finetune_strategy as _runtime_resolve_finetune_strategy,
)
from modules.ocr_training.schemas import FinetuneStrategy


def resolve_finetune_strategy(value: str | FinetuneStrategy | None) -> FinetuneStrategy:
    """Normalize one finetuning strategy and preserve a stable local error message."""
    if value is None:
        return FinetuneStrategy.QLORA
    try:
        return _runtime_resolve_finetune_strategy(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported finetune strategy: {value}") from exc


def sanitize_prediction_text(text: str) -> str:
    """Drop markup artifacts from OCR output for clean metric computation."""
    import re

    if not text:
        return ""
    cleaned = re.sub(r"</?([A-Za-z][A-Za-z0-9_-]*)(\s[^>]*)?>", "", text)
    cleaned = cleaned.replace("<>", "")
    cleaned = cleaned.replace("<", "").replace(">", "")
    return " ".join(cleaned.split())


def load_split_rows(dataset_dir: Path, split: str) -> list[dict[str, str]]:
    """Load one local JSONL split into memory."""
    path = dataset_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset split file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in split file: {path}")
    return rows


def _stable_row_key(seed: int, row: dict[str, str]) -> str:
    """Return a deterministic ordering key for one dataset row."""
    image = row.get("image", "")
    text = row.get("text", "")
    payload = f"{seed}:{image}:{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def infer_train_subset_bucket(row: dict[str, str]) -> str:
    """Infer a stable training-source bucket from the image path."""
    image_path = row.get("image", "").lower()
    if (
        "/synthetic/" in image_path
        or "__synthetic__" in image_path
        or "__synth_" in image_path
        or "__train__synth_" in image_path
    ):
        return "synthetic"
    if (
        "/typed/" in image_path
        or "__typed__" in image_path
        or "__typed_" in image_path
        or "__train__typed_" in image_path
    ):
        return "typed"
    return "unknown"


def subset_train_rows(
    rows: list[dict[str, str]],
    *,
    train_fraction: float,
    seed: int,
) -> list[dict[str, str]]:
    """Return a deterministic train-only subset while preserving source mix."""
    if train_fraction >= 1.0:
        return list(rows)

    buckets: dict[str, list[dict[str, str]]] = {
        "typed": [],
        "synthetic": [],
        "unknown": [],
    }
    for row in rows:
        buckets.setdefault(infer_train_subset_bucket(row), []).append(row)

    selected: list[dict[str, str]] = []
    for bucket_name in ("typed", "synthetic", "unknown"):
        bucket_rows = buckets.get(bucket_name, [])
        if not bucket_rows:
            continue
        ordered = sorted(bucket_rows, key=lambda row: _stable_row_key(seed, row))
        target_count = max(1, min(len(ordered), round(len(ordered) * train_fraction)))
        selected.extend(ordered[:target_count])

    return sorted(selected, key=lambda row: _stable_row_key(seed, row))


def resolve_resume_checkpoint(output_dir: Path, resume_mode: str) -> Path | None:
    """Resolve the latest resume checkpoint according to the configured mode."""
    if resume_mode.lower() == "none":
        return None
    resume_state_path = output_dir / "resume_state.json"
    if resume_mode.lower() in {"auto", "latest"} and resume_state_path.exists():
        data = json.loads(resume_state_path.read_text(encoding="utf-8"))
        checkpoint = data.get("latest_checkpoint")
        if checkpoint:
            candidate = Path(checkpoint)
            if candidate.exists():
                return candidate
    return resolve_latest_checkpoint(output_dir)


def relative_to_base(path: Path) -> str:
    """Return a path relative to the project base when possible."""
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(settings.BASE_DIR))
    except ValueError:
        return str(path)


def resolve_save_eval_steps(
    *,
    eval_steps: int,
    save_steps: int,
    load_best_model_at_end: bool,
    logger,
) -> tuple[int, int]:
    """Return Trainer-safe eval/save step values for best-model loading."""
    normalized_eval_steps = max(1, int(eval_steps))
    normalized_save_steps = max(1, int(save_steps))
    if not load_best_model_at_end:
        return normalized_eval_steps, normalized_save_steps
    if normalized_save_steps % normalized_eval_steps == 0:
        return normalized_eval_steps, normalized_save_steps

    multiplier = ceil(normalized_save_steps / normalized_eval_steps)
    adjusted_save_steps = normalized_eval_steps * max(1, multiplier)
    logger.warning(
        "Adjusted save_steps from %d to %d to satisfy load_best_model_at_end "
        "(multiple of eval_steps=%d).",
        normalized_save_steps,
        adjusted_save_steps,
        normalized_eval_steps,
    )
    return normalized_eval_steps, adjusted_save_steps


def bounded_worker_count(requested_workers: int) -> int:
    """Clamp requested dataloader workers to local CPU availability."""
    import os

    cpu_count = os.cpu_count() or 1
    return max(0, min(int(requested_workers), max(0, cpu_count - 1)))
