from __future__ import annotations

import json
import os
import shutil
import signal
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modules.ocr_training.surya_reports import write_training_history_from_log_history
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
    tmp.write_text(
        json.dumps(_round_artifact_value(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def _round_artifact_value(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if key == "learning_rate":
            return f"{value:.4e}"
        return round(value, 4)
    if isinstance(value, dict):
        return {
            item_key: _round_artifact_value(item, key=item_key) for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_round_artifact_value(item, key=key) for item in value]
    return value


def resolve_latest_checkpoint(output_dir: Path) -> Path | None:
    """Resolve latest Hugging Face checkpoint directory if present."""
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if not suffix.isdigit():
            continue
        checkpoints.append(path)
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda p: int(p.name.removeprefix("checkpoint-")))
    return checkpoints[-1]


def write_resume_state(
    output_dir: Path,
    *,
    status: str,
    latest_checkpoint: Path | None,
    is_rank_zero: bool = True,
) -> Path:
    """Persist deterministic resume metadata for interrupted runs."""
    resume_path = output_dir / "resume_state.json"
    if not is_rank_zero:
        return resume_path
    atomic_write_json(
        resume_path,
        {
            "schema_version": "1.0",
            "status": status,
            "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
        },
    )
    return resume_path


def _checkpoint_eval_history_path(output_dir: Path) -> Path:
    return output_dir / "evaluation" / "checkpoint_eval_history.jsonl"


def _checkpoint_eval_failures_path(output_dir: Path) -> Path:
    return output_dir / "evaluation" / "checkpoint_eval_failures.jsonl"


def append_checkpoint_eval_result(output_dir: Path, payload: dict[str, Any]) -> Path:
    """Append one authoritative checkpoint-eval summary row."""
    path = _checkpoint_eval_history_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_round_artifact_value(payload), ensure_ascii=False) + "\n")
    return path


def load_checkpoint_eval_history(output_dir: Path) -> list[dict[str, Any]]:
    """Load authoritative checkpoint-eval summaries if present."""
    path = _checkpoint_eval_history_path(output_dir)
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except Exception:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def append_checkpoint_eval_failure(output_dir: Path, payload: dict[str, Any]) -> Path:
    """Append one checkpoint-eval failure row for later inspection."""
    path = _checkpoint_eval_failures_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_round_artifact_value(payload), ensure_ascii=False) + "\n")
    return path


def update_best_checkpoint_pointer(
    output_dir: Path,
    *,
    checkpoint_path: Path,
    metric_name: str,
    metric_value: float,
    global_step: int,
    weights_subdir: str = "best_checkpoint",
    meta_filename: str = "best_model_meta.json",
) -> Path:
    """Update stable best-checkpoint pointer and metadata atomically."""
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    best_dir = weights_dir / weights_subdir
    if best_dir.exists() or best_dir.is_symlink():
        if best_dir.is_dir() and not best_dir.is_symlink():
            shutil.rmtree(best_dir)
        else:
            best_dir.unlink()
    shutil.copytree(checkpoint_path, best_dir)

    meta_path = output_dir / meta_filename
    atomic_write_json(
        meta_path,
        {
            "schema_version": "1.0",
            "best_checkpoint": str(best_dir),
            "source_checkpoint": str(checkpoint_path),
            "metric_name": metric_name,
            "metric_value": metric_value,
            "metric_global_step": global_step,
            "checkpoint_step": int(checkpoint_path.name.removeprefix("checkpoint-")),
        },
    )
    return meta_path


@dataclass
class TrainingSignalState:
    """Signal-shared mutable state for interruption handling."""

    interrupted: bool = False
    warning_emitted: bool = False
    eval_interrupted: bool = False
    eval_discard_warning_emitted: bool = False
    stop_requested: bool = False
    stop_reason: str | None = None
    runtime_error_message: str | None = None
    save_checkpoint_on_stop: bool = False


class TrainingTerminationCoordinator:
    """Coordinate stop requests between local DDP ranks via a shared marker file."""

    def __init__(self, output_dir: Path):
        """Bind one coordinator to the shared output directory for a run."""
        self.marker_path = output_dir / "termination_state.json"

    def clear(self) -> None:
        """Remove any stale termination marker from a prior run."""
        with suppress(FileNotFoundError):
            self.marker_path.unlink()

    def request_stop(
        self,
        *,
        reason: str,
        stop_type: str,
        save_checkpoint: bool,
        rank: int,
    ) -> None:
        """Publish a stop request that peer ranks can observe."""
        atomic_write_json(
            self.marker_path,
            {
                "schema_version": "1.0",
                "reason": reason,
                "stop_type": stop_type,
                "save_checkpoint": bool(save_checkpoint),
                "rank": int(rank),
            },
        )

    def read_stop_request(self) -> dict[str, Any] | None:
        """Return the latest stop request payload when present."""
        if not self.marker_path.exists():
            return None
        try:
            return json.loads(self.marker_path.read_text(encoding="utf-8"))
        except Exception:
            return None


def request_training_stop(
    state: TrainingSignalState,
    *,
    reason: str,
    stop_type: str,
    save_checkpoint: bool,
    coordinator: TrainingTerminationCoordinator | None = None,
    rank: int = 0,
) -> None:
    """Record a local stop request and optionally publish it for peer ranks."""
    state.stop_requested = True
    state.stop_reason = reason
    state.save_checkpoint_on_stop = bool(save_checkpoint)
    if stop_type == "signal":
        state.interrupted = True
    else:
        state.runtime_error_message = reason
    if coordinator is not None:
        coordinator.request_stop(
            reason=reason,
            stop_type=stop_type,
            save_checkpoint=save_checkpoint,
            rank=rank,
        )


def observe_training_stop(
    state: TrainingSignalState,
    *,
    coordinator: TrainingTerminationCoordinator | None = None,
) -> None:
    """Refresh local stop state from any published peer-rank stop request."""
    if coordinator is None or state.stop_requested:
        return
    payload = coordinator.read_stop_request()
    if not payload:
        return
    request_training_stop(
        state,
        reason=str(payload.get("reason", "peer_stop_requested")),
        stop_type=str(payload.get("stop_type", "runtime")),
        save_checkpoint=bool(payload.get("save_checkpoint", False)),
        coordinator=None,
        rank=int(payload.get("rank", 0)),
    )


def install_signal_handlers(state: TrainingSignalState) -> None:
    """Install SIGINT/SIGTERM handlers to mark an interrupted training run."""
    owner_pid = os.getpid()

    def _handler(signum, _frame) -> None:
        if os.getpid() != owner_pid:
            return
        state.interrupted = True
        state.stop_requested = True
        state.stop_reason = f"signal:{signum}"
        state.save_checkpoint_on_stop = True
        if not state.warning_emitted:
            state.warning_emitted = True
            logger.warning("Received signal %s. Will stop training gracefully.", signum)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


class BestMetricCheckpointCallback(_TrainerCallback):
    """Trainer callback that tracks one eval metric and updates a stable checkpoint copy."""

    def __init__(
        self,
        output_dir: Path,
        metric_name: str = "eval_cer",
        *,
        weights_subdir: str = "best_checkpoint",
        meta_filename: str = "best_model_meta.json",
    ):
        """Initialize best-checkpoint callback state."""
        self.output_dir = output_dir
        self.metric_name = metric_name
        self.weights_subdir = weights_subdir
        self.meta_filename = meta_filename
        self.best_value: float | None = None
        self._pending_metric_value: float | None = None
        self._pending_global_step: int | None = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Mark improved evaluation metrics for the next save event."""
        if not metrics:
            return control
        metric_value = metrics.get(self.metric_name)
        if metric_value is None:
            return control
        if self.best_value is None or float(metric_value) < float(self.best_value):
            self.best_value = float(metric_value)
            self._pending_metric_value = float(metric_value)
            self._pending_global_step = int(state.global_step)
        return control

    def on_save(self, args, state, control, **kwargs):
        """Update the stable best-checkpoint pointer after the checkpoint exists on disk."""
        if self._pending_metric_value is None or self._pending_global_step is None:
            return control
        checkpoint_path = resolve_latest_checkpoint(Path(args.output_dir))
        if checkpoint_path is None:
            return control
        update_best_checkpoint_pointer(
            self.output_dir,
            checkpoint_path=checkpoint_path,
            metric_name=self.metric_name,
            metric_value=self._pending_metric_value,
            global_step=self._pending_global_step,
            weights_subdir=self.weights_subdir,
            meta_filename=self.meta_filename,
        )
        self._pending_metric_value = None
        self._pending_global_step = None
        return control


class BestCerCheckpointCallback(BestMetricCheckpointCallback):
    """Trainer callback that tracks best CER and updates stable checkpoint pointers."""


class BestWerCheckpointCallback(BestMetricCheckpointCallback):
    """Trainer callback that tracks best WER and updates stable checkpoint pointers."""


class TrainingArtifactsCallback(_TrainerCallback):
    """Trainer callback that keeps run-level history artifacts current during training."""

    def __init__(self, output_dir: Path):
        """Initialize one artifact writer callback bound to a run directory."""
        self.output_dir = output_dir
        self.eval_dir = output_dir / "evaluation"
        self._last_written_step = -1
        self._start_time = time.monotonic()
        self._last_log_time: float | None = None
        self._last_log_step: int | None = None
        self._timing_history: list[dict[str, float | int | str | None]] = []

    def _append_timing_event(self, state, logs: dict[str, Any]) -> None:
        now = time.monotonic()
        global_step = int(state.global_step or 0)
        event_type = "eval" if "eval_cer" in logs or "eval_loss" in logs else "train"
        event = {
            "step": global_step,
            "event_type": event_type,
            "wall_time_sec": round(now - self._start_time, 4),
            "rolling_step_time_sec": None,
            "eval_runtime_sec": logs.get("eval_runtime", logs.get("eval_runtime_sec")),
        }
        if (
            self._last_log_time is not None
            and self._last_log_step is not None
            and global_step > self._last_log_step
        ):
            event["rolling_step_time_sec"] = round(
                (now - self._last_log_time) / max(global_step - self._last_log_step, 1),
                4,
            )
        self._last_log_time = now
        self._last_log_step = global_step
        self._timing_history.append(event)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        timing_path = self.eval_dir / "training_timing.jsonl"
        with timing_path.open("w", encoding="utf-8") as handle:
            for item in self._timing_history:
                handle.write(json.dumps(_round_artifact_value(item), ensure_ascii=False) + "\n")

    def _write_if_needed(self, state, *, force: bool = False) -> None:
        global_step = int(state.global_step or 0)
        if not force and global_step <= self._last_written_step:
            return
        log_history = list(getattr(state, "log_history", None) or [])
        if not log_history:
            return
        write_training_history_from_log_history(
            run_dir=self.output_dir,
            eval_dir=self.eval_dir,
            log_history=log_history,
            include_visuals=force,
        )
        self._last_written_step = global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Refresh training-history artifacts on periodic logs and eval emissions."""
        del args, kwargs
        if not logs:
            return control
        self._append_timing_event(state, logs)
        self._write_if_needed(state)
        return control

    def on_save(self, args, state, control, **kwargs):
        """Force-refresh tracking artifacts after checkpoint saves."""
        del args, kwargs
        self._write_if_needed(state, force=True)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Flush final tracking artifacts at the end of training."""
        del args, kwargs
        self._write_if_needed(state, force=True)
        return control


class PlateauWarningCallback(_TrainerCallback):
    """Warn when evaluation metrics plateau without changing trainer control flow."""

    def __init__(
        self,
        output_dir: Path,
        *,
        min_evals: int = 6,
        patience_evals: int = 8,
        cer_tolerance: float = 0.015,
        wer_tolerance: float = 0.75,
        cer_regression_threshold: float = 0.03,
        wer_regression_threshold: float = 0.5,
    ):
        """Configure a conservative warning-only plateau detector for eval CER/WER."""
        self.output_dir = output_dir
        self.min_evals = max(1, int(min_evals))
        self.patience_evals = max(1, int(patience_evals))
        self.cer_tolerance = float(cer_tolerance)
        self.wer_tolerance = float(wer_tolerance)
        self.cer_regression_threshold = float(cer_regression_threshold)
        self.wer_regression_threshold = float(wer_regression_threshold)
        self._eval_history: list[dict[str, float]] = []
        self._best_cer: float | None = None
        self._best_wer: float | None = None
        self._best_eval_index = -1
        self._last_warning_eval_index = -1
        self._warnings_path = self.output_dir / "evaluation" / "plateau_warnings.jsonl"
        for record in load_checkpoint_eval_history(self.output_dir):
            cer = record.get("eval_cer")
            wer = record.get("eval_wer")
            step = record.get("step")
            if cer is None or wer is None or step is None:
                continue
            self._record_eval_point(step=float(step), cer=float(cer), wer=float(wer))

    def _record_eval_point(
        self, *, step: float, cer: float, wer: float
    ) -> tuple[int, dict[str, float]]:
        current = {
            "step": float(step),
            "cer": float(cer),
            "wer": float(wer),
        }
        self._eval_history.append(current)
        eval_index = len(self._eval_history) - 1
        if self._best_cer is None or current["cer"] < self._best_cer:
            self._best_cer = current["cer"]
            self._best_eval_index = eval_index
        if self._best_wer is None or current["wer"] < self._best_wer:
            self._best_wer = current["wer"]
        return eval_index, current

    def _append_warning_artifact(self, payload: dict[str, Any]) -> None:
        self._warnings_path.parent.mkdir(parents=True, exist_ok=True)
        with self._warnings_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_round_artifact_value(payload), ensure_ascii=False) + "\n")

    def _should_skip_warning(self, *, eval_index: int, evals_since_best: int) -> bool:
        return (
            len(self._eval_history) < self.min_evals
            or evals_since_best < self.patience_evals
            or eval_index <= self._last_warning_eval_index
        )

    def _evaluate_plateau_state(
        self,
        *,
        current: dict[str, float],
        recent: list[dict[str, float]],
    ) -> dict[str, float | bool]:
        cer_values = [item["cer"] for item in recent]
        wer_values = [item["wer"] for item in recent]
        cer_span = max(cer_values) - min(cer_values)
        wer_span = max(wer_values) - min(wer_values)
        best_cer = float(self._best_cer or current["cer"])
        best_wer = float(self._best_wer or current["wer"])
        cer_gap = current["cer"] - best_cer
        wer_gap = current["wer"] - best_wer
        recent_best_cer = min(cer_values)
        recent_best_wer = min(wer_values)
        flat_recent_window = cer_span <= self.cer_tolerance and wer_span <= self.wer_tolerance
        sustained_regression = (
            cer_gap >= self.cer_regression_threshold
            and recent_best_cer >= best_cer + (self.cer_regression_threshold / 2.0)
            and (
                wer_gap >= self.wer_regression_threshold
                or recent_best_wer >= best_wer + (self.wer_regression_threshold / 2.0)
            )
        )
        return {
            "cer_span": cer_span,
            "wer_span": wer_span,
            "best_cer": best_cer,
            "best_wer": best_wer,
            "cer_gap": cer_gap,
            "wer_gap": wer_gap,
            "flat_recent_window": flat_recent_window,
            "sustained_regression": sustained_regression,
        }

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Emit warning-only plateau telemetry after repeated flat evals."""
        del args, kwargs
        if not metrics:
            return control
        cer = metrics.get("eval_cer")
        wer = metrics.get("eval_wer")
        if cer is None or wer is None:
            return control
        train_losses = [
            float(item["loss"])
            for item in list(getattr(state, "log_history", None) or [])
            if item.get("loss") is not None
        ]
        self.observe_authoritative_eval_result(
            step=int(state.global_step or 0),
            cer=float(cer),
            wer=float(wer),
            train_losses=train_losses,
        )
        return control

    def observe_authoritative_eval_result(
        self,
        *,
        step: int,
        cer: float,
        wer: float,
        train_losses: list[float] | None = None,
    ) -> None:
        """Record one authoritative OCR-eval result and emit a warning if the run has plateaued."""
        eval_index, current = self._record_eval_point(step=float(step), cer=cer, wer=wer)
        if eval_index == self._best_eval_index:
            return
        evals_since_best = eval_index - self._best_eval_index
        if self._should_skip_warning(eval_index=eval_index, evals_since_best=evals_since_best):
            return

        recent = self._eval_history[-self.patience_evals :]
        plateau_state = self._evaluate_plateau_state(current=current, recent=recent)
        if not plateau_state["flat_recent_window"] and not plateau_state["sustained_regression"]:
            return

        train_losses = train_losses or []
        train_loss_note = ""
        if len(train_losses) >= 2 and train_losses[-1] < train_losses[-2]:
            train_loss_note = " Train loss is still falling, so this may be overfitting rather than optimization failure."
        warning_payload = {
            "step": int(current["step"]),
            "evals_since_best_cer": int(evals_since_best),
            "recent_cer_span": plateau_state["cer_span"],
            "recent_wer_span": plateau_state["wer_span"],
            "best_cer_so_far": plateau_state["best_cer"],
            "best_wer_so_far": plateau_state["best_wer"],
            "current_cer_gap": plateau_state["cer_gap"],
            "current_wer_gap": plateau_state["wer_gap"],
            "flat_recent_window": plateau_state["flat_recent_window"],
            "sustained_regression": plateau_state["sustained_regression"],
            "train_loss_still_falling": bool(train_loss_note),
            "message": (
                "Plateau warning: no new best CER for "
                f"{evals_since_best} evals; inspect evaluation/training_history.csv and "
                "best checkpoint metadata before continuing."
            ),
        }
        self._append_warning_artifact(warning_payload)
        logger.warning(
            "Plateau warning: no new best CER for %d evals at step %d; current CER gap=%.5f WER gap=%.5f recent CER span=%.5f WER span=%.5f. Inspect %s and %s / %s, then decide whether to interrupt.%s",
            evals_since_best,
            int(current["step"]),
            plateau_state["cer_gap"],
            plateau_state["wer_gap"],
            plateau_state["cer_span"],
            plateau_state["wer_span"],
            self.output_dir / "evaluation" / "training_history.csv",
            self.output_dir / "best_model_meta.json",
            self.output_dir / "best_wer_model_meta.json",
            train_loss_note,
        )
        self._last_warning_eval_index = eval_index
        return None


class AuthoritativeCheckpointEvalCallback(_TrainerCallback):
    """Evaluate saved checkpoints with real OCR inference and update best-metric state."""

    def __init__(
        self,
        output_dir: Path,
        *,
        eval_runner,
        distributed_context,
        torch_module,
        signal_state: TrainingSignalState | None = None,
        termination_coordinator: TrainingTerminationCoordinator | None = None,
        plateau_callback: PlateauWarningCallback | None = None,
    ):
        """Bind checkpoint-eval orchestration to one training run."""
        self.output_dir = output_dir
        self.eval_runner = eval_runner
        self.distributed_context = distributed_context
        self.torch_module = torch_module
        self.signal_state = signal_state
        self.termination_coordinator = termination_coordinator
        self.plateau_callback = plateau_callback
        history = load_checkpoint_eval_history(output_dir)
        self._best_cer = min(
            (float(item["eval_cer"]) for item in history if item.get("eval_cer") is not None),
            default=None,
        )
        self._best_wer = min(
            (float(item["eval_wer"]) for item in history if item.get("eval_wer") is not None),
            default=None,
        )
        self._last_completed_step = max(
            (int(item["step"]) for item in history if item.get("step") is not None),
            default=-1,
        )

    def _sync_stop(self) -> None:
        if self.signal_state is None:
            return
        observe_training_stop(
            self.signal_state,
            coordinator=self.termination_coordinator,
        )

    def _sync_ranks(self) -> None:
        if not self.distributed_context.is_distributed:
            return
        try:
            self.torch_module.distributed.barrier(device_ids=[self.distributed_context.local_rank])
        except TypeError:
            self.torch_module.distributed.barrier()

    def _failure_payload(
        self, *, checkpoint_path: Path, step: int, exc: Exception
    ) -> dict[str, Any]:
        return {
            "step": step,
            "checkpoint_path": str(checkpoint_path),
            "status": "failed",
            "error": str(exc),
        }

    def _record_success(
        self, *, checkpoint_path: Path, checkpoint_step: int, summary: dict[str, Any], state
    ) -> None:
        result_payload = {
            "source": "authoritative_checkpoint_eval",
            "step": checkpoint_step,
            "checkpoint_path": str(checkpoint_path),
            "split": summary.get("split", "val"),
            "num_rows": summary.get("num_rows"),
            "world_size": summary.get("world_size"),
            "eval_fraction": summary.get("eval_fraction"),
            "eval_batch_size": summary.get("eval_batch_size"),
            "max_rows": summary.get("max_rows"),
            "seed": summary.get("seed"),
            "eval_cer": summary.get("mean_cer"),
            "eval_wer": summary.get("mean_wer"),
            "eval_exact": summary.get("exact_rate"),
        }
        log_payload = {
            "eval_cer": result_payload["eval_cer"],
            "eval_wer": result_payload["eval_wer"],
            "eval_exact": result_payload["eval_exact"],
            "checkpoint_step": checkpoint_step,
            "eval_num_rows": summary.get("num_rows"),
        }
        if not hasattr(state, "log_history") or state.log_history is None:
            state.log_history = []
        state.log_history.append(
            {
                "step": checkpoint_step,
                **log_payload,
            }
        )
        append_checkpoint_eval_result(self.output_dir, result_payload)

        current_cer = float(summary["mean_cer"])
        current_wer = float(summary["mean_wer"])
        if self._best_cer is None or current_cer < self._best_cer:
            self._best_cer = current_cer
            update_best_checkpoint_pointer(
                self.output_dir,
                checkpoint_path=checkpoint_path,
                metric_name="eval_cer",
                metric_value=current_cer,
                global_step=checkpoint_step,
            )
        if self._best_wer is None or current_wer < self._best_wer:
            self._best_wer = current_wer
            update_best_checkpoint_pointer(
                self.output_dir,
                checkpoint_path=checkpoint_path,
                metric_name="eval_wer",
                metric_value=current_wer,
                global_step=checkpoint_step,
                weights_subdir="best_checkpoint_wer",
                meta_filename="best_wer_model_meta.json",
            )
        if self.plateau_callback is not None:
            train_losses = [
                float(item["loss"])
                for item in list(getattr(state, "log_history", None) or [])
                if item.get("loss") is not None
            ]
            self.plateau_callback.observe_authoritative_eval_result(
                step=checkpoint_step,
                cer=current_cer,
                wer=current_wer,
                train_losses=train_losses,
            )
        self._last_completed_step = checkpoint_step
        if self.distributed_context.is_rank_zero:
            logger.info("%s", _round_artifact_value(log_payload))

    def on_save(self, args, state, control, **kwargs):
        """Run authoritative OCR evaluation after each saved checkpoint."""
        del kwargs
        self._sync_stop()
        if self.signal_state is not None and self.signal_state.stop_requested:
            control.should_training_stop = True
            return control
        checkpoint_path = resolve_latest_checkpoint(Path(args.output_dir))
        if checkpoint_path is None:
            return control
        checkpoint_step = int(checkpoint_path.name.removeprefix("checkpoint-"))
        if checkpoint_step <= self._last_completed_step:
            return control

        self._sync_ranks()
        try:
            summary = self.eval_runner(checkpoint_path=checkpoint_path, state=state)
        except Exception as exc:
            if self.distributed_context.is_rank_zero:
                append_checkpoint_eval_failure(
                    self.output_dir,
                    self._failure_payload(
                        checkpoint_path=checkpoint_path,
                        step=checkpoint_step,
                        exc=exc,
                    ),
                )
                logger.warning(
                    "Authoritative checkpoint eval failed for %s: %s",
                    checkpoint_path,
                    exc,
                )
            return control
        finally:
            self._sync_ranks()
            self._sync_stop()

        if summary.get("status") != "completed_nonzero_rank":
            self._record_success(
                checkpoint_path=checkpoint_path,
                checkpoint_step=checkpoint_step,
                summary=summary,
                state=state,
            )
        if self.signal_state is not None and self.signal_state.stop_requested:
            control.should_training_stop = True
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
