from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


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


def _align_texts(source: str, target: str) -> list[tuple[str, str]]:
    """Align two strings with edit-distance traceback for confusion analysis."""
    rows = len(source) + 1
    cols = len(target) + 1
    dp = [[0] * cols for _ in range(rows)]
    for row in range(rows):
        dp[row][0] = row
    for col in range(cols):
        dp[0][col] = col
    for row in range(1, rows):
        for col in range(1, cols):
            cost = 0 if source[row - 1] == target[col - 1] else 1
            dp[row][col] = min(
                dp[row - 1][col] + 1,
                dp[row][col - 1] + 1,
                dp[row - 1][col - 1] + cost,
            )

    aligned: list[tuple[str, str]] = []
    row = len(source)
    col = len(target)
    while row > 0 or col > 0:
        if (
            row > 0
            and col > 0
            and dp[row][col]
            == dp[row - 1][col - 1] + (0 if source[row - 1] == target[col - 1] else 1)
        ):
            aligned.append((source[row - 1], target[col - 1]))
            row -= 1
            col -= 1
            continue
        if row > 0 and dp[row][col] == dp[row - 1][col] + 1:
            aligned.append((source[row - 1], "∅"))
            row -= 1
            continue
        aligned.append(("∅", target[col - 1]))
        col -= 1
    aligned.reverse()
    return aligned


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _load_plateau_warnings(run_dir: Path) -> list[dict[str, Any]]:
    return _read_jsonl(run_dir / "evaluation" / "plateau_warnings.jsonl")


def _load_authoritative_eval_history(run_dir: Path) -> list[dict[str, Any]]:
    return _read_jsonl(run_dir / "evaluation" / "checkpoint_eval_history.jsonl")


def _load_checkpoint_eval_failures(run_dir: Path) -> list[dict[str, Any]]:
    return _read_jsonl(run_dir / "evaluation" / "checkpoint_eval_failures.jsonl")


def _latest_checkpoint_step(run_dir: Path) -> int | None:
    from modules.ocr_training.checkpointing import resolve_latest_checkpoint

    latest_checkpoint = resolve_latest_checkpoint(run_dir)
    if latest_checkpoint is None:
        return None
    suffix = latest_checkpoint.name.removeprefix("checkpoint-")
    return int(suffix) if suffix.isdigit() else None


def _trainer_state_payload(run_dir: Path) -> dict[str, Any]:
    trainer_state_path = run_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        from modules.ocr_training.checkpointing import resolve_latest_checkpoint

        latest_checkpoint = resolve_latest_checkpoint(run_dir)
        if latest_checkpoint is not None:
            trainer_state_path = latest_checkpoint / "trainer_state.json"
    if not trainer_state_path.exists():
        return {}
    try:
        payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _chart_points(values: list[tuple[float, float]], width: int, height: int) -> str:
    if not values:
        return ""
    x_values = [point[0] for point in values]
    y_values = [point[1] for point in values]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    x_span = max(max_x - min_x, 1.0)
    y_span = max(max_y - min_y, 1.0)
    coordinates = []
    for x_value, y_value in values:
        x_pos = 40 + ((x_value - min_x) / x_span) * (width - 60)
        y_pos = 20 + (1 - ((y_value - min_y) / y_span)) * (height - 40)
        coordinates.append(f"{x_pos:.1f},{y_pos:.1f}")
    return " ".join(coordinates)


def _render_chart_block(
    *,
    title: str,
    values: list[tuple[float, float]],
    color: str,
    width: int,
    height: int,
    offset_y: int,
    markers: list[dict[str, Any]] | None = None,
) -> str:
    chart_height = height - 20
    polyline = _chart_points(values, width, chart_height)
    if not polyline:
        return ""
    y_values = [point[1] for point in values]
    label_min = min(y_values)
    label_max = max(y_values)
    min_x, max_x = min(point[0] for point in values), max(point[0] for point in values)
    x_span = max(max_x - min_x, 1.0)
    marker_blocks: list[str] = []
    for marker in markers or []:
        step = float(marker.get("step", 0.0))
        x_pos = 40 + ((step - min_x) / x_span) * (width - 60)
        marker_blocks.extend(
            [
                f'<line x1="{x_pos:.1f}" y1="20" x2="{x_pos:.1f}" y2="{chart_height - 20}" stroke="{marker.get("color", "#666666")}" stroke-dasharray="4 3" />',
                f'<text x="{min(x_pos + 4, width - 180):.1f}" y="34" font-size="10" font-family="monospace" fill="{marker.get("color", "#666666")}">{marker.get("label", "marker")}</text>',
            ]
        )
    return "\n".join(
        [
            f'<g transform="translate(0,{offset_y})">',
            f'<text x="20" y="16" font-size="14" font-family="monospace">{title}</text>',
            f'<text x="20" y="{height - 6}" font-size="10" font-family="monospace">min={label_min:.4f} max={label_max:.4f}</text>',
            f'<rect x="40" y="20" width="{width - 60}" height="{chart_height - 40}" fill="none" stroke="#c9ced6" />',
            *marker_blocks,
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{polyline}" />',
            "</g>",
        ]
    )


def _history_rows(
    log_history: list[dict[str, Any]],
    *,
    timing_records: list[dict[str, Any]] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
]:
    """Convert trainer log history into tabular rows and chart points."""
    timing_by_key: dict[tuple[int, str], deque[dict[str, Any]]] = defaultdict(deque)
    for record in timing_records or []:
        step = int(record.get("step", 0))
        event_type = str(record.get("event_type", "train"))
        timing_by_key[(step, event_type)].append(record)
    rows = []
    train_loss_points: list[tuple[float, float]] = []
    eval_loss_points: list[tuple[float, float]] = []
    eval_cer_points: list[tuple[float, float]] = []
    eval_wer_points: list[tuple[float, float]] = []
    for item in log_history:
        step = float(item.get("step", 0))
        event_type = (
            "eval"
            if item.get("eval_cer") is not None or item.get("eval_loss") is not None
            else "train"
        )
        timing_record = (
            timing_by_key[(int(step), event_type)].popleft()
            if timing_by_key[(int(step), event_type)]
            else {}
        )
        rows.append(
            _round_artifact_value(
                {
                    "step": item.get("step"),
                    "epoch": item.get("epoch"),
                    "loss": item.get("loss"),
                    "eval_loss": item.get("eval_loss"),
                    "eval_cer": item.get("eval_cer"),
                    "eval_wer": item.get("eval_wer"),
                    "eval_exact": item.get("eval_exact"),
                    "eval_runtime_sec": timing_record.get(
                        "eval_runtime_sec", item.get("eval_runtime_sec", item.get("eval_runtime"))
                    ),
                    "learning_rate": item.get("learning_rate"),
                    "grad_norm": item.get("grad_norm"),
                    "wall_time_sec": timing_record.get("wall_time_sec", item.get("wall_time_sec")),
                    "rolling_step_time_sec": timing_record.get(
                        "rolling_step_time_sec", item.get("rolling_step_time_sec")
                    ),
                }
            )
        )
        if item.get("loss") is not None:
            train_loss_points.append((step, float(item["loss"])))
        if item.get("eval_loss") is not None:
            eval_loss_points.append((step, float(item["eval_loss"])))
        if item.get("eval_cer") is not None:
            eval_cer_points.append((step, float(item["eval_cer"])))
        if item.get("eval_wer") is not None:
            eval_wer_points.append((step, float(item["eval_wer"])))
    return rows, train_loss_points, eval_loss_points, eval_cer_points, eval_wer_points


def _merge_authoritative_eval_history(
    *,
    log_history: list[dict[str, Any]],
    authoritative_eval_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge authoritative checkpoint-eval metrics into trainer log history by step."""
    merged = [dict(item) for item in log_history]
    for eval_item in authoritative_eval_history:
        step = int(eval_item.get("step", 0))
        payload = {
            "step": step,
            "eval_cer": eval_item.get("eval_cer"),
            "eval_wer": eval_item.get("eval_wer"),
            "eval_exact": eval_item.get("eval_exact"),
        }
        target = None
        for item in merged:
            if int(item.get("step", -1)) == step and item.get("eval_loss") is not None:
                target = item
                break
        if target is None:
            for item in merged:
                if int(item.get("step", -1)) == step:
                    target = item
                    break
        if target is None:
            merged.append(payload)
        else:
            target.update({key: value for key, value in payload.items() if value is not None})
    merged.sort(
        key=lambda item: (
            float(item.get("step", 0)),
            0 if item.get("loss") is not None and item.get("eval_loss") is None else 1,
        )
    )
    return merged


def _load_best_checkpoint_meta(run_dir: Path, name: str) -> dict[str, Any] | None:
    return _read_json(run_dir / name)


def _load_finetune_context(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / "finetune_meta.json") or {}


def _training_summary(run_dir: Path, log_history: list[dict[str, Any]]) -> dict[str, Any]:
    """Return compact run-level training summary derived from trainer log history."""
    eval_rows = [item for item in log_history if item.get("eval_cer") is not None]
    train_rows = [item for item in log_history if item.get("loss") is not None]
    best_eval = min(eval_rows, key=lambda item: float(item["eval_cer"])) if eval_rows else None
    best_eval_wer = min(eval_rows, key=lambda item: float(item["eval_wer"])) if eval_rows else None
    latest_eval = eval_rows[-1] if eval_rows else None
    latest_train = train_rows[-1] if train_rows else None
    finetune_meta = _load_finetune_context(run_dir)
    trainer_state = _trainer_state_payload(run_dir)
    plateau_warnings = _load_plateau_warnings(run_dir)
    checkpoint_eval_failures = _load_checkpoint_eval_failures(run_dir)
    train_fraction = finetune_meta.get("train_fraction")
    selection_metric = str(finetune_meta.get("effective_metric_for_best_model", "eval_cer"))
    cer_meta = _load_best_checkpoint_meta(run_dir, "best_model_meta.json")
    wer_meta = _load_best_checkpoint_meta(run_dir, "best_wer_model_meta.json")
    best_cer_eval_index = (
        eval_rows.index(best_eval) if best_eval is not None and best_eval in eval_rows else None
    )
    evals_since_best_cer = (
        len(eval_rows) - 1 - best_cer_eval_index if best_cer_eval_index is not None else None
    )
    trainer_best_checkpoint = trainer_state.get("best_model_checkpoint")
    cer_best_source = cer_meta.get("source_checkpoint") if cer_meta else None
    notes: list[str] = []
    if isinstance(train_fraction, (float, int)) and float(train_fraction) < 1.0:
        notes.append(
            "train_fraction < 1.0 uses a fixed deterministic subset of train rows for this run; it is not equivalent to full-dataset coverage."
        )
    return {
        "schema_version": "1.0",
        "num_train_logs": len(train_rows),
        "num_eval_logs": len(eval_rows),
        "selection_metric": selection_metric,
        "latest_train": latest_train,
        "latest_eval": latest_eval,
        "best_eval_by_cer": best_eval,
        "best_eval_by_wer": best_eval_wer,
        "best_checkpoint_by_cer": cer_meta,
        "best_checkpoint_by_wer": wer_meta,
        "best_cer_checkpoint_step": cer_meta.get("checkpoint_step") if cer_meta else None,
        "best_wer_checkpoint_step": wer_meta.get("checkpoint_step") if wer_meta else None,
        "latest_checkpoint_step": _latest_checkpoint_step(run_dir),
        "trainer_best_model_checkpoint": trainer_best_checkpoint,
        "load_best_model_at_end_requested": bool(
            finetune_meta.get("selected_training_config", {}).get("load_best_model_at_end", False)
        ),
        "best_model_restored_at_end": (
            str(trainer_best_checkpoint) == str(cer_best_source)
            if trainer_best_checkpoint is not None and cer_best_source is not None
            else None
        ),
        "evals_since_best_cer": evals_since_best_cer,
        "plateau_warning_count": len(plateau_warnings),
        "latest_plateau_warning": plateau_warnings[-1] if plateau_warnings else None,
        "checkpoint_eval_failure_count": len(checkpoint_eval_failures),
        "latest_checkpoint_eval_failure": (
            checkpoint_eval_failures[-1] if checkpoint_eval_failures else None
        ),
        "train_fraction": train_fraction,
        "notes": notes,
    }


def _load_training_summary(run_dir: Path, log_history: list[dict[str, Any]]) -> dict[str, Any]:
    return _training_summary(run_dir, log_history)


def _summary_markers(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cer_meta = _load_best_checkpoint_meta(run_dir, "best_model_meta.json")
    wer_meta = _load_best_checkpoint_meta(run_dir, "best_wer_model_meta.json")
    cer_markers: list[dict[str, Any]] = []
    wer_markers: list[dict[str, Any]] = []
    if cer_meta and cer_meta.get("metric_global_step") is not None:
        cer_markers.append(
            {
                "step": float(cer_meta["metric_global_step"]),
                "label": f"best CER @{int(cer_meta['metric_global_step'])}",
                "color": "#0b6a0b",
            }
        )
    if wer_meta and wer_meta.get("metric_global_step") is not None:
        wer_markers.append(
            {
                "step": float(wer_meta["metric_global_step"]),
                "label": f"best WER @{int(wer_meta['metric_global_step'])}",
                "color": "#8f4e00",
            }
        )
    return cer_markers, wer_markers


def load_training_log_history(run_dir: Path) -> list[dict[str, Any]]:
    """Load trainer log history from the run root or latest checkpoint fallback."""
    trainer_state = _trainer_state_payload(run_dir)
    log_history = trainer_state.get("log_history", [])
    raw_history = log_history if isinstance(log_history, list) else []
    authoritative_eval_history = _load_authoritative_eval_history(run_dir)
    return _merge_authoritative_eval_history(
        log_history=raw_history,
        authoritative_eval_history=authoritative_eval_history,
    )


def _render_training_curves_png(
    *,
    png_path: Path,
    summary: dict[str, Any],
    train_loss_points: list[tuple[float, float]],
    eval_loss_points: list[tuple[float, float]],
    eval_cer_points: list[tuple[float, float]],
    eval_wer_points: list[tuple[float, float]],
) -> Path:
    width = 1200
    height = 980
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, width - 20, 140), fill="#f7f9fc", outline="#c9ced6")
    latest_eval = summary.get("latest_eval") or {}
    best_cer = summary.get("best_eval_by_cer") or {}
    best_wer = summary.get("best_eval_by_wer") or {}
    summary_lines = [
        "OCR Training Report",
        f"selection metric: {summary.get('selection_metric', 'eval_cer')}",
        f"latest eval: step={latest_eval.get('step', 'n/a')} cer={latest_eval.get('eval_cer', 'n/a')} wer={latest_eval.get('eval_wer', 'n/a')}",
        f"best CER: step={best_cer.get('step', 'n/a')} cer={best_cer.get('eval_cer', 'n/a')}",
        f"best WER: step={best_wer.get('step', 'n/a')} wer={best_wer.get('eval_wer', 'n/a')}",
    ]
    for index, line in enumerate(summary_lines):
        draw.text((40, 40 + index * 18), line, fill="black")

    chart_specs = [
        ("Train Loss", train_loss_points, "#0f6cbd"),
        ("Eval Loss", eval_loss_points, "#a4262c"),
        ("Eval CER", eval_cer_points, "#0b6a0b"),
        ("Eval WER", eval_wer_points, "#8f4e00"),
    ]
    chart_top = 180
    chart_height = 170
    chart_width = width - 120
    for index, (title, values, color) in enumerate(chart_specs):
        top = chart_top + index * 190
        left = 60
        right = left + chart_width
        bottom = top + chart_height
        draw.text((left, top - 22), title, fill="black")
        draw.rectangle((left, top, right, bottom), outline="#c9ced6")
        if values:
            min_x = min(point[0] for point in values)
            max_x = max(point[0] for point in values)
            min_y = min(point[1] for point in values)
            max_y = max(point[1] for point in values)
            x_span = max(max_x - min_x, 1.0)
            y_span = max(max_y - min_y, 1.0)
            coordinates = []
            for x_value, y_value in values:
                x_pos = left + ((x_value - min_x) / x_span) * chart_width
                y_pos = top + (1 - ((y_value - min_y) / y_span)) * chart_height
                coordinates.append((x_pos, y_pos))
            if len(coordinates) > 1:
                draw.line(coordinates, fill=color, width=3)
        draw.text((left, bottom + 4), f"points={len(values)}", fill="black")
    image.save(png_path)
    return png_path


def write_subset_manifest(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    split: str,
    seed: int,
    selection: str,
) -> Path:
    """Persist one deterministic sampled row manifest for reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            payload = {
                "index": index,
                "split": split,
                "seed": seed,
                "selection": selection,
                "image": row.get("image"),
                "text": row.get("text"),
                "modality": row.get("modality"),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return output_path


def monitor_training_run(run_dir: Path) -> dict[str, Any]:
    """Return one monitor-friendly summary for an active or completed run."""
    log_history = load_training_log_history(run_dir)
    summary = _load_training_summary(run_dir, log_history)
    latest_eval = summary.get("latest_eval") or {}
    best_cer = summary.get("best_eval_by_cer") or {}
    best_wer = summary.get("best_eval_by_wer") or {}
    return {
        "selection_metric": summary.get("selection_metric"),
        "latest_eval_step": latest_eval.get("step"),
        "latest_eval_cer": latest_eval.get("eval_cer"),
        "latest_eval_wer": latest_eval.get("eval_wer"),
        "best_cer_step": best_cer.get("step"),
        "best_cer_value": best_cer.get("eval_cer"),
        "best_wer_step": best_wer.get("step"),
        "best_wer_value": best_wer.get("eval_wer"),
        "evals_since_best_cer": summary.get("evals_since_best_cer"),
        "training_history_csv": str(run_dir / "evaluation" / "training_history.csv"),
        "best_model_meta": str(run_dir / "best_model_meta.json"),
        "best_wer_model_meta": str(run_dir / "best_wer_model_meta.json"),
    }


def write_training_history_from_log_history(
    *,
    run_dir: Path,
    eval_dir: Path,
    log_history: list[dict[str, Any]],
    include_visuals: bool = True,
) -> dict[str, Path]:
    """Write CSV, SVG, and summary artifacts from one in-memory trainer log history."""
    if not log_history:
        return {}
    merged_log_history = _merge_authoritative_eval_history(
        log_history=log_history,
        authoritative_eval_history=_load_authoritative_eval_history(run_dir),
    )
    timing_records = _read_jsonl(eval_dir / "training_timing.jsonl")
    rows, train_loss_points, eval_loss_points, eval_cer_points, eval_wer_points = _history_rows(
        merged_log_history,
        timing_records=timing_records,
    )
    eval_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_dir / "training_history.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "epoch",
                "loss",
                "eval_loss",
                "eval_cer",
                "eval_wer",
                "eval_exact",
                "eval_runtime_sec",
                "learning_rate",
                "grad_norm",
                "wall_time_sec",
                "rolling_step_time_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = _round_artifact_value(_load_training_summary(run_dir, merged_log_history))
    cer_markers, wer_markers = _summary_markers(run_dir)
    svg_path = eval_dir / "training_curves.svg"
    if include_visuals:
        width = 960
        summary_height = 120
        chart_height = 160
        blocks = [
            _render_chart_block(
                title="Train Loss",
                values=train_loss_points,
                color="#0f6cbd",
                width=width,
                height=chart_height,
                offset_y=summary_height,
            ),
            _render_chart_block(
                title="Eval Loss",
                values=eval_loss_points,
                color="#a4262c",
                width=width,
                height=chart_height,
                offset_y=summary_height + chart_height,
            ),
            _render_chart_block(
                title="Eval CER",
                values=eval_cer_points,
                color="#0b6a0b",
                width=width,
                height=chart_height,
                offset_y=summary_height + chart_height * 2,
                markers=cer_markers,
            ),
            _render_chart_block(
                title="Eval WER",
                values=eval_wer_points,
                color="#8f4e00",
                width=width,
                height=chart_height,
                offset_y=summary_height + chart_height * 3,
                markers=wer_markers,
            ),
        ]
        latest_eval = summary.get("latest_eval") or {}
        best_cer = summary.get("best_eval_by_cer") or {}
        best_wer = summary.get("best_eval_by_wer") or {}
        notes = list(summary.get("notes") or [])
        summary_lines = [
            "Training Summary",
            f"latest eval: step={latest_eval.get('step', 'n/a')} cer={latest_eval.get('eval_cer', 'n/a')} wer={latest_eval.get('eval_wer', 'n/a')}",
            f"best CER: step={best_cer.get('step', 'n/a')} cer={best_cer.get('eval_cer', 'n/a')}",
            f"best WER: step={best_wer.get('step', 'n/a')} wer={best_wer.get('eval_wer', 'n/a')}",
        ]
        summary_lines.extend(notes[:2])
        svg_path.write_text(
            "\n".join(
                [
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{summary_height + chart_height * 4}">',
                    '<rect width="100%" height="100%" fill="#ffffff" />',
                    '<rect x="10" y="10" width="940" height="96" fill="#f7f9fc" stroke="#c9ced6" />',
                    *[
                        f'<text x="24" y="{36 + index * 18}" font-size="13" font-family="monospace">{line}</text>'
                        for index, line in enumerate(summary_lines)
                    ],
                    *[block for block in blocks if block],
                    "</svg>",
                ]
            ),
            encoding="utf-8",
        )
    summary_path = run_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    jsonl_path = eval_dir / "training_history.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for item in _round_artifact_value(merged_log_history):
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    artifacts = {
        "training_history_csv": csv_path,
        "training_history_jsonl": jsonl_path,
        "training_summary_json": summary_path,
    }
    if include_visuals:
        artifacts["training_curves_svg"] = svg_path
    return artifacts


def write_training_history_artifacts(*, run_dir: Path, eval_dir: Path) -> dict[str, Path]:
    """Write CSV and SVG summaries for trainer history if available."""
    log_history = load_training_log_history(run_dir)
    if not log_history:
        return {}
    return write_training_history_from_log_history(
        run_dir=run_dir,
        eval_dir=eval_dir,
        log_history=log_history,
    )


def write_confusion_artifacts(
    *,
    eval_dir: Path,
    split: str,
    records: list[dict[str, Any]],
    top_n: int = 50,
) -> dict[str, Path]:
    """Write OCR-style character confusion summaries from prediction records."""
    confusion_counts: Counter[tuple[str, str]] = Counter()
    for record in records:
        for pred_char, gt_char in _align_texts(record["pred_text"], record["gt_text"]):
            if pred_char != gt_char:
                confusion_counts[(gt_char, pred_char)] += 1
    payload = [
        {"gt": gt_char, "pred": pred_char, "count": count}
        for (gt_char, pred_char), count in confusion_counts.most_common(top_n)
    ]
    json_path = eval_dir / f"character_confusions_{split}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_lines = [
        "# Character Confusions",
        "",
        "| Ground Truth | Prediction | Count |",
        "| --- | --- | ---: |",
    ]
    for item in payload:
        md_lines.append(f"| `{item['gt']}` | `{item['pred']}` | {item['count']} |")
    md_path = eval_dir / f"character_confusions_{split}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return {"character_confusions_json": json_path, "character_confusions_md": md_path}


def _load_prediction_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def write_training_report_bundle(
    *,
    run_dir: Path,
    output_dir: Path | None = None,
    split: str | None = None,
    predictions_path: Path | None = None,
    include_training_artifacts: bool = True,
) -> dict[str, Path]:
    """Generate one read-only run report bundle from existing history and optional predictions."""
    target_dir = output_dir or (run_dir / "evaluation")
    target_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    if include_training_artifacts:
        artifacts = write_training_history_artifacts(run_dir=run_dir, eval_dir=target_dir)
    log_history = load_training_log_history(run_dir)
    summary = _round_artifact_value(_load_training_summary(run_dir, log_history))

    resolved_predictions = predictions_path
    if resolved_predictions is None:
        candidates = sorted((run_dir / "evaluation").glob("predictions_*.jsonl"))
        if split is not None:
            preferred = run_dir / "evaluation" / f"predictions_{split}.jsonl"
            candidates = [preferred] if preferred.exists() else candidates
        resolved_predictions = candidates[0] if candidates else None

    confusion_note = "confusion data unavailable; run explicit evaluation to generate it."
    if resolved_predictions is not None and resolved_predictions.exists():
        inferred_split = split or resolved_predictions.stem.removeprefix("predictions_")
        confusion_artifacts = write_confusion_artifacts(
            eval_dir=target_dir,
            split=inferred_split,
            records=_load_prediction_records(resolved_predictions),
        )
        artifacts.update(confusion_artifacts)
        confusion_note = f"confusion artifacts generated for split `{inferred_split}` from `{resolved_predictions.name}`."

    latest_eval = summary.get("latest_eval") or {}
    best_cer = summary.get("best_eval_by_cer") or {}
    best_wer = summary.get("best_eval_by_wer") or {}
    report_path = target_dir / "training_report.md"
    report_lines = [
        "# OCR Training Report",
        "",
        "## Summary",
        "",
        "| Metric | Step | Value |",
        "| --- | ---: | ---: |",
        f"| Latest CER | {latest_eval.get('step', 'n/a')} | {latest_eval.get('eval_cer', 'n/a')} |",
        f"| Latest WER | {latest_eval.get('step', 'n/a')} | {latest_eval.get('eval_wer', 'n/a')} |",
        f"| Best CER | {best_cer.get('step', 'n/a')} | {best_cer.get('eval_cer', 'n/a')} |",
        f"| Best WER | {best_wer.get('step', 'n/a')} | {best_wer.get('eval_wer', 'n/a')} |",
        "",
        "## Artifacts",
        "",
        f"- training history csv: `{(run_dir / 'evaluation' / 'training_history.csv').name}`",
        f"- training curves svg: `{(run_dir / 'evaluation' / 'training_curves.svg').name}`",
        f"- training curves png: `{(run_dir / 'evaluation' / 'training_curves.png').name}`",
        f"- training history jsonl: `{(run_dir / 'evaluation' / 'training_history.jsonl').name}`",
        f"- {confusion_note}",
    ]
    if summary.get("plateau_warning_count"):
        report_lines.append(f"- plateau warnings emitted: `{summary.get('plateau_warning_count')}`")
    for note in summary.get("notes") or []:
        report_lines.append(f"- note: {note}")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    if include_training_artifacts:
        png_path = target_dir / "training_curves.png"
        rows, train_loss_points, eval_loss_points, eval_cer_points, eval_wer_points = _history_rows(
            log_history
        )
        del rows
        artifacts["training_curves_png"] = _render_training_curves_png(
            png_path=png_path,
            summary=summary,
            train_loss_points=train_loss_points,
            eval_loss_points=eval_loss_points,
            eval_cer_points=eval_cer_points,
            eval_wer_points=eval_wer_points,
        )
    artifacts["training_report_md"] = report_path
    return artifacts
