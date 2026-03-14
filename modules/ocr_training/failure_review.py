from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote


def _to_label_studio_local_files_url(image_path: str) -> str:
    image_abs = Path(image_path).resolve()
    try:
        output_idx = image_abs.parts.index("output")
    except ValueError as exc:
        raise ValueError(
            f"Image path must resolve under output/ for Label Studio local-files serving. "
            f"Got '{image_abs}'."
        ) from exc
    output_rel = Path(*image_abs.parts[output_idx + 1 :])
    rel_url_path = str(output_rel).replace("\\", "/")
    return f"/data/local-files/?d={quote(rel_url_path, safe='/')}"


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _task_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (str(row["image"]), str(row["gt_text"]), str(row["pred_text"]))


def create_failure_review_tasks(
    *,
    exact_false_dir: Path,
    output_dir: Path,
    task_file_name: str = "ocr_failure_review_tasks.json",
) -> dict[str, object]:
    """Create one deduplicated Label Studio task file from exact-false review candidates."""
    if not exact_false_dir.exists():
        raise FileNotFoundError(f"Missing exact-false analysis directory: {exact_false_dir}")

    source_files = {
        "cer_outlier_2std": exact_false_dir / "cer_outliers_2std.jsonl",
        "cer_outlier_3std": exact_false_dir / "cer_outliers_3std.jsonl",
        "wer_outlier_2std": exact_false_dir / "wer_outliers_2std.jsonl",
        "wer_outlier_3std": exact_false_dir / "wer_outliers_3std.jsonl",
        "likely_label_mismatch": exact_false_dir / "likely_label_mismatch_predictions.jsonl",
        "likely_artifact": exact_false_dir / "likely_artifact_predictions.jsonl",
        "suspect_blank": exact_false_dir / "suspect_blank_predictions.jsonl",
        "confirmed_blank": exact_false_dir / "confirmed_blank_predictions.jsonl",
    }

    by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    candidate_sources: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    source_counts: dict[str, int] = {}

    for source_name, path in source_files.items():
        rows = _read_jsonl(path)
        source_counts[source_name] = len(rows)
        for row in rows:
            key = _task_key(row)
            if key not in by_key:
                by_key[key] = dict(row)
            candidate_sources[key].add(source_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, object]] = []
    skipped_missing_images: list[dict[str, object]] = []

    for idx, (key, row) in enumerate(
        sorted(
            by_key.items(),
            key=lambda item: (
                -len(candidate_sources[item[0]]),
                -float(item[1].get("cer", 0.0)),
                -float(item[1].get("wer", 0.0)),
                str(item[1].get("image", "")),
            ),
        ),
        start=1,
    ):
        image_path = Path(str(row["image"]))
        if not image_path.exists():
            skipped_missing_images.append(
                {
                    **row,
                    "candidate_sources": sorted(candidate_sources[key]),
                    "skip_reason": "image_missing",
                }
            )
            continue
        task_id = f"ocr_failure_{idx:06d}"
        tasks.append(
            {
                "data": {
                    "image": _to_label_studio_local_files_url(str(image_path)),
                    "task_id": task_id,
                    "image_path": str(image_path),
                    "gt_text": str(row["gt_text"]),
                    "pred_text": str(row["pred_text"]),
                    "corrected_gt_seed": str(row["gt_text"]),
                    "cer": float(row["cer"]),
                    "wer": float(row["wer"]),
                    "exact": bool(row.get("exact")),
                    "modality": row.get("modality"),
                    "candidate_sources": ", ".join(sorted(candidate_sources[key])),
                    "classification": row.get("classification"),
                    "structural_classification": row.get("structural_classification"),
                    "blank_reasons": ", ".join(
                        str(reason) for reason in row.get("blank_reasons", [])
                    ),
                    "image_width": row.get("image_width"),
                    "image_height": row.get("image_height"),
                    "gt_len": row.get("gt_len"),
                    "pred_len": row.get("pred_len"),
                }
            }
        )

    output_json = output_dir / task_file_name
    output_json.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "skipped_missing_images.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in skipped_missing_images)
        + ("\n" if skipped_missing_images else ""),
        encoding="utf-8",
    )
    summary = {
        "exact_false_dir": str(exact_false_dir),
        "output_json": str(output_json),
        "num_tasks": len(tasks),
        "skipped_missing_images": len(skipped_missing_images),
        "source_counts": source_counts,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
