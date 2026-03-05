from __future__ import annotations

import json
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger("OCRTrainingCheckpointing")

try:
    from transformers import TrainerCallback as _TrainerCallback
except Exception:  # pragma: no cover - runtime environments may not satisfy trainer deps.

    class _TrainerCallback:  # type: ignore[too-many-ancestors]
        """Fallback callback base used when transformers is unavailable."""


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically to avoid corruption on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def resolve_latest_checkpoint(output_dir: Path) -> Path | None:
    """Resolve latest Hugging Face checkpoint directory if present."""
    checkpoints = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]))
    return checkpoints[-1]


def write_resume_state(output_dir: Path, *, status: str, latest_checkpoint: Path | None) -> Path:
    """Persist deterministic resume metadata for interrupted runs."""
    resume_path = output_dir / "resume_state.json"
    atomic_write_json(
        resume_path,
        {
            "schema_version": "1.0",
            "status": status,
            "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
        },
    )
    return resume_path


def update_best_checkpoint_pointer(
    output_dir: Path,
    *,
    checkpoint_path: Path,
    metric_name: str,
    metric_value: float,
    global_step: int,
) -> Path:
    """Update stable best-checkpoint pointer and metadata atomically."""
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    link_path = weights_dir / "best_checkpoint"
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(checkpoint_path)

    meta_path = output_dir / "best_model_meta.json"
    atomic_write_json(
        meta_path,
        {
            "schema_version": "1.0",
            "best_checkpoint": str(checkpoint_path),
            "metric_name": metric_name,
            "metric_value": metric_value,
            "global_step": global_step,
        },
    )
    return meta_path


@dataclass
class TrainingSignalState:
    """Signal-shared mutable state for interruption handling."""

    interrupted: bool = False
    warning_emitted: bool = False


def install_signal_handlers(state: TrainingSignalState) -> None:
    """Install SIGINT/SIGTERM handlers to mark an interrupted training run."""

    def _handler(signum, _frame) -> None:
        state.interrupted = True
        if not state.warning_emitted:
            state.warning_emitted = True
            logger.warning("Received signal %s. Will stop training gracefully.", signum)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


class BestCerCheckpointCallback(_TrainerCallback):
    """Trainer callback that tracks best CER and updates stable checkpoint pointers."""

    def __init__(self, output_dir: Path, metric_name: str = "eval_cer"):
        """Initialize best-checkpoint callback state."""
        self.output_dir = output_dir
        self.metric_name = metric_name
        self.best_value: float | None = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Capture best checkpoint on every evaluation event."""
        if not metrics:
            return control
        metric_value = metrics.get(self.metric_name)
        if metric_value is None:
            return control
        if self.best_value is None or float(metric_value) < float(self.best_value):
            self.best_value = float(metric_value)
            checkpoint_path = resolve_latest_checkpoint(Path(args.output_dir))
            if checkpoint_path is not None:
                update_best_checkpoint_pointer(
                    self.output_dir,
                    checkpoint_path=checkpoint_path,
                    metric_name=self.metric_name,
                    metric_value=float(metric_value),
                    global_step=int(state.global_step),
                )
        return control


class EpochLoggingCallback(_TrainerCallback):
    """Trainer callback that emits concise, epoch-oriented training telemetry."""

    def __init__(self):
        """Initialize per-epoch tracking state."""
        self._epoch_losses: list[float] = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Reset per-epoch running metrics and log epoch start."""
        self._epoch_losses = []
        current_epoch = int(state.epoch or 0) + 1
        total_epochs = int(args.num_train_epochs)
        logger.info("Epoch %d/%d started", current_epoch, total_epochs)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Capture periodic loss logs and emit structured step telemetry."""
        if not logs:
            return control
        if "loss" in logs:
            loss = float(logs["loss"])
            self._epoch_losses.append(loss)
            logger.info(
                "Epoch %.2f step=%d loss=%.5f lr=%s",
                float(state.epoch or 0.0),
                int(state.global_step),
                loss,
                logs.get("learning_rate", "n/a"),
            )
        if "eval_loss" in logs or "eval_cer" in logs or "eval_wer" in logs:
            eval_exact = logs.get("eval_exact", None)
            eval_exact_pct = "n/a" if eval_exact is None else f"{float(eval_exact) * 100.0:.2f}%"
            logger.info(
                "Eval step=%d loss=%s cer=%s wer=%s exact=%s",
                int(state.global_step),
                logs.get("eval_loss", "n/a"),
                logs.get("eval_cer", "n/a"),
                logs.get("eval_wer", "n/a"),
                eval_exact_pct,
            )
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        """Emit end-of-epoch summary using aggregated loss statistics."""
        if not self._epoch_losses:
            logger.info("Epoch %.2f completed (no logged losses).", float(state.epoch or 0.0))
            return control
        mean_loss = sum(self._epoch_losses) / len(self._epoch_losses)
        logger.info(
            "Epoch %.2f completed. batches_logged=%d mean_loss=%.5f min_loss=%.5f max_loss=%.5f",
            float(state.epoch or 0.0),
            len(self._epoch_losses),
            mean_loss,
            min(self._epoch_losses),
            max(self._epoch_losses),
        )
        return control
