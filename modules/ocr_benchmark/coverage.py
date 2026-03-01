import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from config.settings import settings
from modules.ocr_benchmark.dataset import read_manifest
from modules.ocr_benchmark.metrics import normalize_ethiopic_text
from modules.ocr_benchmark.paths import resolve_doc_benchmark_root
from schemas.ocr_coverage import (
    CoverageDeficit,
    CoverageReport,
    CoverageTier,
    EthiopicCharsetConfig,
    QueueItem,
)
from utils.logger import get_logger
from utils.run_registry import load_latest_run, register_latest_run

logger = get_logger("OCRBenchmarkCoverage")


def _artifact_path(path: Path) -> str:
    try:
        return str(path.relative_to(settings.BASE_DIR))
    except ValueError:
        return str(path)


def _sha256_json(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def load_charset_config(path: Path) -> EthiopicCharsetConfig:
    """Load and validate charset configuration from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Charset config not found at {path}. "
            "Create it from input/ocr_benchmark/config/ethiopic_charset.v1.json."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return EthiopicCharsetConfig.model_validate(data)


def _manifest_hash(rows: list[dict]) -> str:
    return _sha256_json(sorted(rows, key=lambda r: r["line_id"]))


def _charset_hash(cfg: EthiopicCharsetConfig) -> str:
    return _sha256_json(cfg.model_dump(mode="json"))


def _char_counts(texts: list[str], ignored: set[str]) -> Counter:
    counts: Counter = Counter()
    for text in texts:
        normalized = normalize_ethiopic_text(text)
        for ch in normalized:
            if ch in ignored:
                continue
            counts[ch] += 1
    return counts


def _split_texts(rows: list[dict]) -> dict[str, list[str]]:
    split_map: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if not str(row.get("gt_text", "")).strip():
            continue
        split_map[row["split"]].append(row["gt_text"])
        split_map["all"].append(row["gt_text"])
    return split_map


def _evaluate_deficits(
    counts: Counter, cfg: EthiopicCharsetConfig
) -> tuple[list[str], list[CoverageDeficit]]:
    missing: set[str] = set()
    deficits: list[CoverageDeficit] = []
    for tier, tier_cfg in cfg.tiers.items():
        for ch in tier_cfg.chars:
            c = int(counts.get(ch, 0))
            if c == 0:
                missing.add(ch)
            if c < tier_cfg.min_count:
                deficits.append(
                    CoverageDeficit(
                        tier=tier,
                        char=ch,
                        count=c,
                        min_required=tier_cfg.min_count,
                        deficit=tier_cfg.min_count - c,
                    )
                )
    deficits.sort(key=lambda d: (d.tier.value, -d.deficit, d.char))
    return sorted(missing), deficits


def build_coverage_report(
    *,
    doc_stem: str,
    manifest_path: Path,
    charset_config_path: Path,
) -> tuple[CoverageReport, Path]:
    """Build and persist a deterministic coverage report for a single document manifest."""
    rows = [r for r in read_manifest(manifest_path) if r.doc_stem == doc_stem]
    if not rows:
        raise ValueError(f"No rows found for doc_stem '{doc_stem}' in {manifest_path}.")

    cfg = load_charset_config(charset_config_path)
    if sum(len(tier_cfg.chars) for tier_cfg in cfg.tiers.values()) == 0:
        raise ValueError(
            "Charset config has no tier character lists (declaration-only config). "
            "Use a policy config with tier chars for coverage gating."
        )
    ignored = set(cfg.ignored_chars)

    row_payload = [r.model_dump(mode="json", exclude_none=True) for r in rows]
    split_texts = _split_texts(row_payload)
    split_counts = {split: _char_counts(texts, ignored) for split, texts in split_texts.items()}
    all_counts = split_counts.get("all", Counter())

    missing, deficits = _evaluate_deficits(all_counts, cfg)
    blocking_deficits = [d for d in deficits if d.tier != CoverageTier.OPTIONAL]
    status = len(blocking_deficits) == 0
    recommendations: list[str] = []
    if blocking_deficits:
        top = blocking_deficits[:10]
        recommendations.append(
            "Annotate additional lines containing under-covered characters: "
            + ", ".join(f"{x.char}(+{x.deficit})" for x in top)
        )
    if "train" not in split_texts or len(split_texts["train"]) == 0:
        recommendations.append("No train GT text present. Export and annotate the train split.")

    report = CoverageReport(
        doc_stem=doc_stem,
        manifest_hash=_manifest_hash(row_payload),
        charset_config_hash=_charset_hash(cfg),
        coverage_status=status,
        split_stats={
            split: {
                "num_rows": len([r for r in row_payload if r["split"] == split])
                if split != "all"
                else len(row_payload),
                "num_unique_chars": len(split_counts.get(split, Counter())),
                "num_total_chars": int(sum(split_counts.get(split, Counter()).values())),
            }
            for split in ("train", "holdout", "all")
        },
        missing_chars=missing,
        under_threshold=deficits,
        recommendations=recommendations,
    )

    doc_root = resolve_doc_benchmark_root(doc_stem)
    out_dir = doc_root / "coverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "coverage_report.json"
    md_path = out_dir / "coverage_report.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# OCR Coverage Report\n\n")
        f.write(f"- Doc: `{doc_stem}`\n")
        f.write(f"- Coverage Status: `{report.coverage_status}`\n")
        f.write(f"- Missing Chars: `{len(report.missing_chars)}`\n")
        f.write(f"- Under-threshold Entries: `{len(report.under_threshold)}`\n\n")
        f.write("## Split Stats\n")
        for split, stats in report.split_stats.items():
            f.write(
                f"- {split}: rows={stats['num_rows']} "
                f"unique_chars={stats['num_unique_chars']} "
                f"total_chars={stats['num_total_chars']}\n"
            )
        f.write("\n## Top Deficits\n")
        for item in report.under_threshold[:25]:
            f.write(
                f"- {item.char} ({item.tier.value}): "
                f"{item.count}/{item.min_required} deficit={item.deficit}\n"
            )
        f.write("\n## Recommendations\n")
        for rec in report.recommendations:
            f.write(f"- {rec}\n")

    register_latest_run(
        stage="ocr-benchmark-coverage-report",
        doc_stem=doc_stem,
        run_dir=out_dir,
        artifacts={
            "coverage_report_json": _artifact_path(json_path),
            "coverage_report_md": _artifact_path(md_path),
        },
        metadata={
            "manifest_hash": report.manifest_hash,
            "charset_config_hash": report.charset_config_hash,
            "coverage_status": report.coverage_status,
        },
    )
    logger.info("Coverage report written to %s", out_dir)
    return report, out_dir


def ensure_coverage_gate(
    *,
    doc_stem: str,
    manifest_path: Path,
    charset_config_path: Path,
    enforce: bool,
) -> tuple[CoverageReport, Path]:
    """Enforce coverage policy before finetuning and optionally hard-fail when unmet."""
    report, out_dir = build_coverage_report(
        doc_stem=doc_stem,
        manifest_path=manifest_path,
        charset_config_path=charset_config_path,
    )
    if enforce and not report.coverage_status:
        raise ValueError(
            "Coverage gate failed: unmet minimum character coverage. "
            f"See report: {out_dir / 'coverage_report.md'}"
        )
    return report, out_dir


def _load_prepare_candidates(doc_stem: str) -> list[dict]:
    prepare_pointer = load_latest_run("ocr-benchmark-prepare", doc_stem)
    if not prepare_pointer:
        raise FileNotFoundError(f"Missing prepare pointer for {doc_stem}. Run prepare-lines first.")
    crops_path = settings.BASE_DIR / prepare_pointer["artifacts"]["crops_metadata"]
    candidate_rows = json.loads(crops_path.read_text(encoding="utf-8"))
    if not candidate_rows:
        raise ValueError("No candidate rows found in prepare artifacts.")
    return candidate_rows


def _load_zero_shot_predictions(doc_stem: str) -> dict[str, dict]:
    preds: dict[str, dict] = {}
    zero_pointer = load_latest_run("ocr-benchmark-surya-zero", doc_stem)
    if not zero_pointer:
        return preds
    preds_path = settings.BASE_DIR / zero_pointer["artifacts"]["baseline_predictions_jsonl"]
    with preds_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            preds[item["line_id"]] = item
    return preds


def _score_queue_item(
    row: dict,
    labeled_ids: set[str],
    preds: dict[str, dict],
    deficit_map: dict[str, int],
) -> QueueItem | None:
    line_id = row["line_id"]
    if line_id in labeled_ids:
        return None

    pred = preds.get(line_id, {})
    pred_text = normalize_ethiopic_text(str(pred.get("raw_pred", "")))
    confidence = pred.get("confidence")
    hits = sorted({ch for ch in pred_text if ch in deficit_map})
    if not hits:
        return None

    gain = float(sum(deficit_map[ch] for ch in hits))
    conf_penalty = (1.0 - float(confidence)) if isinstance(confidence, (int, float)) else 0.2
    score = gain - conf_penalty
    return QueueItem(
        line_id=line_id,
        image_path=row["image_path"],
        column_key=row["column_key"],
        pred_text=pred.get("raw_pred"),
        score=score,
        target_chars_hit=hits,
        confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
        reasons=[
            f"deficit_gain={gain:.2f}",
            f"confidence_penalty={conf_penalty:.2f}",
        ],
    )


def build_annotation_queue(
    *,
    doc_stem: str,
    manifest_path: Path,
    charset_config_path: Path,
    max_items: int = 200,
) -> Path:
    """Build ranked annotation queue candidates for unresolved coverage deficits."""
    report, _ = build_coverage_report(
        doc_stem=doc_stem,
        manifest_path=manifest_path,
        charset_config_path=charset_config_path,
    )
    candidate_rows = _load_prepare_candidates(doc_stem)
    preds = _load_zero_shot_predictions(doc_stem)

    manifest_rows = [r for r in read_manifest(manifest_path) if r.doc_stem == doc_stem]
    labeled_ids = {r.line_id for r in manifest_rows}
    deficit_map = {
        d.char: d.deficit for d in report.under_threshold if d.tier != CoverageTier.OPTIONAL
    }

    queue_items: list[QueueItem] = []
    for row in candidate_rows:
        item = _score_queue_item(row, labeled_ids, preds, deficit_map)
        if item is not None:
            queue_items.append(item)

    queue_items.sort(key=lambda x: (-x.score, x.line_id))
    queue_items = queue_items[:max_items]
    if not queue_items:
        raise ValueError(
            "No queue candidates could be ranked from current deficits and available predictions. "
            "Try labeling more data or regenerate baseline predictions."
        )

    doc_root = resolve_doc_benchmark_root(doc_stem)
    out_dir = doc_root / "coverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "annotation_queue.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for item in queue_items:
            f.write(item.model_dump_json(exclude_none=True) + "\n")

    register_latest_run(
        stage="ocr-benchmark-coverage-queue",
        doc_stem=doc_stem,
        run_dir=out_dir,
        artifacts={"annotation_queue_jsonl": _artifact_path(out_path)},
        metadata={"max_items": max_items, "num_items": len(queue_items)},
    )
    logger.info("Coverage queue generated with %d items at %s", len(queue_items), out_path)
    return out_path
