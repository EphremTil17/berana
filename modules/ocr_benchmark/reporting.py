import csv
import json
from pathlib import Path
from statistics import mean

from config.settings import settings
from modules.ocr_benchmark.dataset import DatasetSplit, read_manifest
from modules.ocr_benchmark.metrics import (
    align_chars_levenshtein,
    build_char_confusion_counts,
    calculate_cer_wer,
    normalize_ethiopic_text,
)
from modules.ocr_benchmark.paths import resolve_doc_benchmark_root
from utils.logger import get_logger
from utils.run_registry import load_latest_run

logger = get_logger("OCRBenchmarkReporting")
MODEL_STAGE_MAP = {
    "surya_zero_shot": "ocr-benchmark-surya-zero",
    "trocr_zero_shot": "ocr-benchmark-trocr-zero",
    "surya_finetuned": "ocr-benchmark-surya-finetune",
    "trocr_finetuned": "ocr-benchmark-trocr-finetune",
}


def load_model_predictions(stage: str, doc_stem: str) -> dict[str, str]:
    """Helper to load raw predictions from a known registry stage."""
    pointer = load_latest_run(stage, doc_stem)
    if not pointer:
        return {}

    pred_rel = None
    for k in pointer.get("artifacts", {}):
        if "predictions" in k:
            pred_rel = pointer["artifacts"][k]
            break

    if not pred_rel:
        return {}

    path = settings.BASE_DIR / pred_rel
    if not path.exists():
        logger.warning("Prediction artifact missing on disk for stage=%s path=%s", stage, path)
        return {}
    preds = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            preds[data["line_id"]] = data["raw_pred"]

    return preds


def _aggregate(rows: list[dict]) -> dict[str, float | int]:
    if not rows:
        return {"n": 0, "cer": 1.0, "wer": 1.0, "exact": 0.0}
    cer_vals = [float(r["cer"]) for r in rows]
    wer_vals = [float(r["wer"]) for r in rows]
    return {
        "n": len(rows),
        "cer": mean(cer_vals),
        "wer": mean(wer_vals),
        "exact": sum(1 for r in rows if r["exact_match"]) / len(rows),
    }


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 1.0
    idx = round((len(sorted_values) - 1) * q)
    return sorted_values[idx]


def _build_split_debug(all_gt_rows: list, preds: dict[str, str]) -> tuple[dict, list[dict]]:
    split_debug_rows = []
    for row in all_gt_rows:
        raw_gt = row.gt_text or ""
        raw_pred = preds.get(row.line_id, "")
        norm_gt = normalize_ethiopic_text(raw_gt)
        norm_pred = normalize_ethiopic_text(raw_pred)
        cer, wer, exact = calculate_cer_wer(norm_pred, norm_gt)
        split_debug_rows.append(
            {
                "line_id": row.line_id,
                "split": row.split.value,
                "lang": row.column_key,
                "cer": cer,
                "wer": wer,
                "exact_match": exact,
                "raw_pred": raw_pred,
                "normalized_pred": norm_pred,
                "raw_gt": raw_gt,
                "normalized_gt": norm_gt,
            }
        )

    split_debug = {}
    for split_name in ("holdout", "train", "all"):
        subset = (
            split_debug_rows
            if split_name == "all"
            else [r for r in split_debug_rows if r["split"] == split_name]
        )
        split_debug[split_name] = {
            "overall": _aggregate(subset),
            "geez": _aggregate([r for r in subset if r["lang"] == "geez"]),
            "amharic": _aggregate([r for r in subset if r["lang"] == "amharic"]),
            "worst5": [
                {"line_id": r["line_id"], "cer": r["cer"], "lang": r["lang"]}
                for r in sorted(subset, key=lambda x: x["cer"], reverse=True)[:5]
            ],
            "high_error_gt20_count": sum(1 for r in subset if r["cer"] > 0.20),
        }
    return split_debug, split_debug_rows


def _evaluate_single_model(
    model_key: str, preds: dict[str, str], gt_rows: list, all_gt_rows: list
) -> tuple[dict, dict]:
    metrics = []
    aligned_pairs_all: list[tuple[str | None, str | None]] = []
    geez_cer, amh_cer, geez_wer, amh_wer = [], [], [], []

    for gt_row in gt_rows:
        raw_gt = gt_row.gt_text or ""
        raw_pred = preds.get(gt_row.line_id, "")
        norm_gt = normalize_ethiopic_text(raw_gt)
        norm_pred = normalize_ethiopic_text(raw_pred)

        cer, wer, exact = calculate_cer_wer(norm_pred, norm_gt)
        aligned_pairs_all.extend(align_chars_levenshtein(norm_pred, norm_gt))
        metrics.append(
            {
                "line_id": gt_row.line_id,
                "lang": gt_row.column_key,
                "cer": cer,
                "wer": wer,
                "exact_match": exact,
                "raw_pred": raw_pred,
                "normalized_pred": norm_pred,
                "raw_gt": raw_gt,
                "normalized_gt": norm_gt,
            }
        )
        if gt_row.column_key == "geez":
            geez_cer.append(cer)
            geez_wer.append(wer)
        elif gt_row.column_key == "amharic":
            amh_cer.append(cer)
            amh_wer.append(wer)

    overall_cers = geez_cer + amh_cer
    overall_wers = geez_wer + amh_wer
    if not overall_cers:
        raise ValueError(f"No scored rows for model={model_key}.")

    sorted_cers = sorted(overall_cers)
    split_debug, split_debug_rows = _build_split_debug(all_gt_rows, preds)
    model_result = {
        "mean_cer": mean(overall_cers),
        "overall_wer": mean(overall_wers) if overall_wers else 1.0,
        "geez_mean_cer": mean(geez_cer) if geez_cer else 1.0,
        "amharic_mean_cer": mean(amh_cer) if amh_cer else 1.0,
        "worst_language_cer": max(
            mean(geez_cer) if geez_cer else 1.0,
            mean(amh_cer) if amh_cer else 1.0,
        ),
        "cer_p50": _quantile(sorted_cers, 0.50),
        "cer_p75": _quantile(sorted_cers, 0.75),
        "cer_p90": _quantile(sorted_cers, 0.90),
        "details": metrics,
        "split_debug": split_debug,
        "split_debug_rows": split_debug_rows,
    }
    return model_result, build_char_confusion_counts(aligned_pairs_all)


def _write_scores_csv(out_dir: Path, results: dict[str, dict]) -> None:
    with (out_dir / "final_scores.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "model",
                "overall_cer",
                "overall_wer",
                "geez_cer",
                "amharic_cer",
                "worst_language_cer",
                "cer_p50",
                "cer_p75",
                "cer_p90",
            ],
        )
        writer.writeheader()
        for model_key, model_data in results.items():
            writer.writerow(
                {
                    "model": model_key,
                    "overall_cer": model_data["mean_cer"],
                    "overall_wer": model_data["overall_wer"],
                    "geez_cer": model_data["geez_mean_cer"],
                    "amharic_cer": model_data["amharic_mean_cer"],
                    "worst_language_cer": model_data["worst_language_cer"],
                    "cer_p50": model_data["cer_p50"],
                    "cer_p75": model_data["cer_p75"],
                    "cer_p90": model_data["cer_p90"],
                }
            )


def _write_final_report(
    out_dir: Path,
    decision: dict,
    production_ready: bool,
    skipped_models: dict[str, str],
    results: dict[str, dict],
) -> None:
    with (out_dir / "final_report.md").open("w", encoding="utf-8") as f:
        f.write("# Benchmark Final Report\n\n")
        f.write(f"**Winner:** {decision['winner']}\n")
        f.write(f"**Production Ready (<15% CER):** {'Yes' if production_ready else 'No'}\n\n")
        if skipped_models:
            f.write("## Skipped Models\n")
            for model_key, reason in skipped_models.items():
                f.write(f"- {model_key}: {reason}\n")
            f.write("\n")
        for m, m_data in results.items():
            f.write(f"### {m}\n")
            f.write(f"- Overall CER: {m_data['mean_cer']:.2%}\n")
            f.write(f"- Overall WER: {m_data['overall_wer']:.2%}\n")
            f.write(f"- Ge'ez CER: {m_data['geez_mean_cer']:.2%}\n")
            f.write(f"- Amharic CER: {m_data['amharic_mean_cer']:.2%}\n")
            f.write(
                f"- CER quantiles: p50={m_data['cer_p50']:.2%}, "
                f"p75={m_data['cer_p75']:.2%}, p90={m_data['cer_p90']:.2%}\n\n"
            )
            split_debug = m_data.get("split_debug", {})
            if split_debug:
                f.write("#### Split Comparison (Train vs Holdout)\n")
                for split_name in ("holdout", "train", "all"):
                    block = split_debug.get(split_name)
                    if not block:
                        continue
                    ov, gz, am = block["overall"], block["geez"], block["amharic"]
                    f.write(
                        f"- {split_name.upper()} overall: n={ov['n']} CER={ov['cer']:.2%} "
                        f"WER={ov['wer']:.2%} Exact={ov['exact']:.2%}\n"
                    )
                    f.write(
                        f"  geez: n={gz['n']} CER={gz['cer']:.2%} "
                        f"WER={gz['wer']:.2%} Exact={gz['exact']:.2%}\n"
                    )
                    f.write(
                        f"  amharic: n={am['n']} CER={am['cer']:.2%} "
                        f"WER={am['wer']:.2%} Exact={am['exact']:.2%}\n"
                    )
                    f.write(
                        "  worst5: "
                        + ", ".join(
                            f"{w['line_id']}({w['lang']} {w['cer']:.2%})" for w in block["worst5"]
                        )
                        + "\n"
                    )
                    f.write(f"  high-error(>20% CER): {block['high_error_gt20_count']}\n")
                f.write("\n")


def _write_confusion_and_debug_artifacts(
    out_dir: Path,
    results: dict[str, dict],
    confusion_by_model: dict[str, dict[str, dict[str, int]]],
) -> None:
    for model_key, matrix in confusion_by_model.items():
        confusion_path = out_dir / f"char_confusion_{model_key}.json"
        with confusion_path.open("w", encoding="utf-8") as f:
            json.dump(matrix, f, ensure_ascii=False, indent=2)

        flat_pairs: list[tuple[str, str, int]] = []
        for gt_char, pred_map in matrix.items():
            for pred_char, count in pred_map.items():
                if gt_char != pred_char:
                    flat_pairs.append((gt_char, pred_char, int(count)))
        flat_pairs.sort(key=lambda x: x[2], reverse=True)
        top_conf_path = out_dir / f"top_confusions_{model_key}.csv"
        with top_conf_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gt_char", "pred_char", "count"])
            for gt_char, pred_char, count in flat_pairs[:200]:
                writer.writerow([gt_char, pred_char, count])

    for model_key, m_data in results.items():
        details = m_data.get("details", [])
        if details:
            debug_csv = out_dir / f"line_debug_{model_key}.csv"
            with debug_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "line_id",
                        "lang",
                        "cer",
                        "wer",
                        "exact_match",
                        "raw_pred",
                        "normalized_pred",
                        "raw_gt",
                        "normalized_gt",
                    ],
                )
                writer.writeheader()
                for row in details:
                    writer.writerow(row)

        split_csv = out_dir / f"line_debug_{model_key}_all_splits.csv"
        with split_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "line_id",
                    "split",
                    "lang",
                    "cer",
                    "wer",
                    "exact_match",
                    "raw_pred",
                    "normalized_pred",
                    "raw_gt",
                    "normalized_gt",
                ],
            )
            writer.writeheader()
            for row in m_data.get("split_debug_rows", []):
                writer.writerow(row)


def evaluate_models(
    doc_stem: str,
    cer_ceiling_threshold: float = 0.15,
    manifest_path: Path | None = None,
    require_all_models: bool = True,
) -> Path:
    """
    Generate final benchmark report evaluating models natively.
    Applies Acceptance Gate CER rule.
    """
    dataset_path = manifest_path or (settings.INPUT_DIR / "ocr_benchmark" / "line_manifest.jsonl")
    if not dataset_path.exists():
        raise FileNotFoundError("Canonical GT dataset manifest not found.")

    all_gt_rows = [
        r
        for r in read_manifest(dataset_path)
        if r.doc_stem == doc_stem and (r.gt_text or "").strip()
    ]
    gt_rows = [r for r in all_gt_rows if r.split == DatasetSplit.HOLDOUT]
    if not gt_rows:
        raise ValueError("No HOLDOUT rows found in the canonical dataset manifest.")

    results = {}
    confusion_by_model: dict[str, dict[str, dict[str, int]]] = {}
    skipped_models: dict[str, str] = {}
    for model_key, stage in MODEL_STAGE_MAP.items():
        preds = load_model_predictions(stage, doc_stem)
        if not preds:
            msg = f"Missing or empty predictions for required model stage '{stage}' ({model_key})."
            if require_all_models:
                raise ValueError(msg)
            logger.warning(f"Skipping {model_key} - {msg}")
            skipped_models[model_key] = msg
            continue
        model_result, confusion = _evaluate_single_model(model_key, preds, gt_rows, all_gt_rows)
        results[model_key] = model_result
        confusion_by_model[model_key] = confusion

    if not results:
        raise ValueError("No models had predictions available to evaluate.")

    best_model = min(
        results.keys(), key=lambda k: (results[k]["mean_cer"], results[k]["worst_language_cer"])
    )
    best_cer = results[best_model]["mean_cer"]

    production_ready = best_cer <= cer_ceiling_threshold

    decision = {
        "winner": best_model,
        "winning_mean_cer": best_cer,
        "production_ready": production_ready,
        "msg": (
            "Production Ready"
            if production_ready
            else f"FAILED: CER {best_cer:.2%} > Ceiling {cer_ceiling_threshold:.2%}"
        ),
        "all_scores": {k: v["mean_cer"] for k, v in results.items()},
    }

    doc_root = resolve_doc_benchmark_root(doc_stem)
    out_dir = doc_root / "final_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_scores_csv(out_dir, results)
    _write_final_report(out_dir, decision, production_ready, skipped_models, results)
    _write_confusion_and_debug_artifacts(out_dir, results, confusion_by_model)

    # Write decision artifact last to avoid partial decision states.
    with (out_dir / "final_decision.json").open("w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)

    logger.info(
        f"Evaluation complete. Winner: {best_model} (CER={best_cer:.2%}) - "
        f"Production Ready: {production_ready}"
    )
    return out_dir
