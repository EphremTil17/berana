from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


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
) -> str:
    chart_height = height - 20
    polyline = _chart_points(values, width, chart_height)
    if not polyline:
        return ""
    y_values = [point[1] for point in values]
    label_min = min(y_values)
    label_max = max(y_values)
    return "\n".join(
        [
            f'<g transform="translate(0,{offset_y})">',
            f'<text x="20" y="16" font-size="14" font-family="monospace">{title}</text>',
            f'<text x="20" y="{height - 6}" font-size="10" font-family="monospace">min={label_min:.4f} max={label_max:.4f}</text>',
            f'<rect x="40" y="20" width="{width - 60}" height="{chart_height - 40}" fill="none" stroke="#c9ced6" />',
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{polyline}" />',
            "</g>",
        ]
    )


def write_training_history_artifacts(*, run_dir: Path, eval_dir: Path) -> dict[str, Path]:
    """Write CSV and SVG summaries for trainer history if available."""
    trainer_state_path = run_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        return {}
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    log_history = trainer_state.get("log_history", [])
    rows = []
    train_loss_points: list[tuple[float, float]] = []
    eval_loss_points: list[tuple[float, float]] = []
    eval_cer_points: list[tuple[float, float]] = []
    eval_wer_points: list[tuple[float, float]] = []
    for item in log_history:
        step = float(item.get("step", 0))
        rows.append(
            {
                "step": item.get("step"),
                "epoch": item.get("epoch"),
                "loss": item.get("loss"),
                "eval_loss": item.get("eval_loss"),
                "eval_cer": item.get("eval_cer"),
                "eval_wer": item.get("eval_wer"),
                "eval_exact": item.get("eval_exact"),
                "learning_rate": item.get("learning_rate"),
                "grad_norm": item.get("grad_norm"),
            }
        )
        if item.get("loss") is not None:
            train_loss_points.append((step, float(item["loss"])))
        if item.get("eval_loss") is not None:
            eval_loss_points.append((step, float(item["eval_loss"])))
        if item.get("eval_cer") is not None:
            eval_cer_points.append((step, float(item["eval_cer"])))
        if item.get("eval_wer") is not None:
            eval_wer_points.append((step, float(item["eval_wer"])))

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
                "learning_rate",
                "grad_norm",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    width = 960
    chart_height = 160
    blocks = [
        _render_chart_block(
            title="Train Loss",
            values=train_loss_points,
            color="#0f6cbd",
            width=width,
            height=chart_height,
            offset_y=0,
        ),
        _render_chart_block(
            title="Eval Loss",
            values=eval_loss_points,
            color="#a4262c",
            width=width,
            height=chart_height,
            offset_y=chart_height,
        ),
        _render_chart_block(
            title="Eval CER",
            values=eval_cer_points,
            color="#0b6a0b",
            width=width,
            height=chart_height,
            offset_y=chart_height * 2,
        ),
        _render_chart_block(
            title="Eval WER",
            values=eval_wer_points,
            color="#8f4e00",
            width=width,
            height=chart_height,
            offset_y=chart_height * 3,
        ),
    ]
    svg_path = eval_dir / "training_curves.svg"
    svg_path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{chart_height * 4}">',
                '<rect width="100%" height="100%" fill="#ffffff" />',
                *[block for block in blocks if block],
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )
    return {"training_history_csv": csv_path, "training_curves_svg": svg_path}


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
