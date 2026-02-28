import json
import statistics
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from PIL import Image
from surya.detection import DetectionPredictor
from tqdm import tqdm

from config.settings import settings
from modules.ocr_benchmark.dataset import (
    compute_records_hash,
    load_split_manifest,
    write_split_manifest,
)
from modules.ocr_benchmark.paths import create_new_doc_benchmark_root
from utils.logger import get_logger
from utils.run_registry import register_latest_run, resolve_required_input

logger = get_logger("OCRBenchmarkPrepare")
CANONICAL_BENCHMARK_LANG_ORDER = ("geez", "amharic")


def _run_detection_quietly(predictor: DetectionPredictor, img: Image.Image):
    """
    Suppress noisy per-call tqdm output from Surya internals so we can expose one
    concise pipeline-level progress bar.
    """
    sink = StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return predictor([img])[0]


def _deterministic_split_assignment(
    crops_metadata: list[dict],
    train_count: int = 200,
    holdout_count: int = 50,
    seed: int = 42,
    include_languages: list[str] | None = None,
) -> list[dict]:
    """
    Enforce split freeze directly during candidate generation to ensure pre-annotation
    and all downstream stages agree on train/holdout assignment before humans even see it.
    """
    import random

    random.seed(seed)

    # Randomly shuffle all candidates
    random.shuffle(crops_metadata)

    language_targets: dict[str, int]
    include_set = set(include_languages or ["geez", "amharic"])
    if include_set == {"geez", "amharic"}:
        language_targets = {"geez": 25, "amharic": 25}
    elif include_set == {"geez"}:
        language_targets = {"geez": holdout_count}
    elif include_set == {"amharic"}:
        language_targets = {"amharic": holdout_count}
    else:
        raise ValueError(
            f"Unsupported language selection for split assignment: {sorted(include_set)}. "
            "Use geez and/or amharic."
        )

    holdout_counts = {lang: 0 for lang in language_targets}
    assigned_train = 0

    keep_rows: list[dict] = []
    for meta in crops_metadata:
        lang = meta["column_key"]
        if lang in language_targets and holdout_counts[lang] < language_targets[lang]:
            meta["split"] = "holdout"
            holdout_counts[lang] += 1
            keep_rows.append(meta)
        elif assigned_train < train_count:
            meta["split"] = "train"
            assigned_train += 1
            keep_rows.append(meta)
        else:
            meta["split"] = "exclude"  # Overflow lines beyond 250 pilot limits

    missing_holdout = {
        lang: target - holdout_counts[lang]
        for lang, target in language_targets.items()
        if holdout_counts[lang] < target
    }
    if missing_holdout:
        raise ValueError(
            "Insufficient holdout rows for selected benchmark languages. "
            f"Missing targets: {missing_holdout}."
        )
    if assigned_train < train_count:
        raise ValueError(
            f"Insufficient training rows for pilot. Need {train_count}, got {assigned_train}."
        )
    if len(keep_rows) != (train_count + holdout_count):
        raise ValueError(
            "Split assignment produced unexpected row count: "
            f"{len(keep_rows)} != {train_count + holdout_count}."
        )

    return keep_rows


def _line_height(bbox: list[float]) -> float:
    return float(max(1.0, bbox[3] - bbox[1]))


def _line_width(bbox: list[float]) -> float:
    return float(max(1.0, bbox[2] - bbox[0]))


def _sort_boxes_reading_order(boxes: list, img_w: int) -> tuple[list[list[float]], float]:
    """
    Sort detected boxes by reading order using y-centroid grouping and x-min ordering.
    Returns sorted bboxes and median height used for tolerance rules.
    """
    raw_bboxes = [list(box.bbox) for box in boxes]
    if not raw_bboxes:
        return [], 1.0

    heights = [_line_height(bbox) for bbox in raw_bboxes]
    median_h = statistics.median(heights)
    y_tolerance = max(4.0, median_h * 0.55)

    raw_bboxes.sort(key=lambda bbox: ((bbox[1] + bbox[3]) / 2.0, bbox[0]))
    grouped_rows: list[list[list[float]]] = []
    for bbox in raw_bboxes:
        cy = (bbox[1] + bbox[3]) / 2.0
        if not grouped_rows:
            grouped_rows.append([bbox])
            continue
        row = grouped_rows[-1]
        row_cy = statistics.median([((item[1] + item[3]) / 2.0) for item in row])
        if abs(cy - row_cy) <= y_tolerance:
            row.append(bbox)
        else:
            grouped_rows.append([bbox])

    sorted_bboxes: list[list[float]] = []
    for row in grouped_rows:
        row.sort(key=lambda bbox: bbox[0])
        sorted_bboxes.extend(row)

    return sorted_bboxes, median_h


def _classify_quality_flag(bbox: list[float], median_h: float, img_w: int) -> str:
    """Apply heuristics to route suspicious detections for human review."""
    height = _line_height(bbox)
    width = _line_width(bbox)
    center_x = (bbox[0] + bbox[2]) / 2.0

    if height > (1.8 * median_h):
        return "merged_line"
    if width < max(8.0, median_h * 0.30) and abs(center_x - (img_w / 2.0)) <= (img_w * 0.12):
        return "structural_marker"
    return "ok"


def _enforce_frozen_split(doc_stem: str, split_manifest_path: Path) -> None:
    """Enforce immutable split hash for a document after the first successful freeze."""
    freeze_root = settings.INPUT_DIR / "ocr_benchmark" / "frozen_splits"
    frozen_path = freeze_root / f"{doc_stem}.json"
    freeze_root.mkdir(parents=True, exist_ok=True)

    current = load_split_manifest(split_manifest_path)
    if frozen_path.exists():
        frozen = load_split_manifest(frozen_path)
        if frozen.dataset_hash != current.dataset_hash:
            raise ValueError(
                "Split freeze violation: current split hash differs from frozen hash for "
                f"{doc_stem}.\nFrozen: {frozen.dataset_hash}\nCurrent: {current.dataset_hash}\n"
                f"If this is intentional, explicitly review and replace {frozen_path}."
            )
        return

    frozen_path.write_text(split_manifest_path.read_text(encoding="utf-8"), encoding="utf-8")


def _stable_split_records(records: list[dict]) -> list[dict]:
    """
    Build a run-invariant representation for split freeze hashing.
    Excludes path fields that naturally change across versioned runs.
    """
    stable = []
    for row in records:
        stable.append(
            {
                "line_id": row["line_id"],
                "doc_stem": row["doc_stem"],
                "page_id": row["page_id"],
                "column_key": row["column_key"],
                "quality_flag": row["quality_flag"],
                "split": row["split"],
            }
        )
    return stable


def _normalize_requested_languages(include_languages: list[str] | None) -> list[str]:
    requested_languages = [
        lang.lower() for lang in (include_languages or list(CANONICAL_BENCHMARK_LANG_ORDER))
    ]
    allowed_languages = set(CANONICAL_BENCHMARK_LANG_ORDER)
    invalid_languages = [lang for lang in requested_languages if lang not in allowed_languages]
    if invalid_languages:
        raise ValueError(
            f"Unsupported --languages values: {invalid_languages}. Allowed values: geez, amharic."
        )
    include_set = set(requested_languages)
    return [lang for lang in CANONICAL_BENCHMARK_LANG_ORDER if lang in include_set]


def _resolve_spliced_dir(doc_stem: str) -> Path:
    try:
        return resolve_required_input(
            upstream_stage="crop-columns",
            doc_stem=doc_stem,
            artifact_key="spliced_dir",
        )
    except (FileNotFoundError, KeyError) as exc:
        raise FileNotFoundError(
            "No crop-columns artifacts found for this document. "
            "Run crop first for the page range you want to benchmark, "
            "then re-run prepare-lines.\n\n"
            "Example:\n"
            "python berana.py crop-columns --pdf-path input/raw_pdfs/<doc>.pdf "
            "--start-page 1 --end-page 40"
        ) from exc


def _collect_strip_tasks(
    spliced_dir: Path,
    normalized_languages: list[str],
    limit_pages: int | None,
) -> list[tuple[str, Path, str]]:
    pages = sorted([p for p in spliced_dir.iterdir() if p.is_dir()])
    if limit_pages:
        pages = pages[:limit_pages]

    strip_tasks: list[tuple[str, Path, str]] = []
    for page_dir in pages:
        page_id = page_dir.name
        for col_img_path in sorted(page_dir.glob("*.png")):
            lang_key = col_img_path.stem
            if lang_key not in normalized_languages:
                continue
            strip_tasks.append((page_id, col_img_path, lang_key))
    return strip_tasks


def _process_strip_task(
    *,
    task: tuple[str, Path, str],
    predictor: DetectionPredictor,
    images_out_dir: Path,
    doc_root: Path,
    doc_stem: str,
    crops_metadata: list[dict],
) -> int:
    page_id, col_img_path, lang_key = task
    try:
        img = Image.open(col_img_path).convert("RGB")
    except Exception as exc:
        logger.warning("Failed to open %s: %s", col_img_path, exc)
        return 0

    try:
        res = _run_detection_quietly(predictor, img)
    except Exception as exc:
        logger.warning("Detection failed for %s: %s", col_img_path, exc)
        return 0

    sorted_bboxes, median_h = _sort_boxes_reading_order(res.bboxes, img.width)
    generated = 0
    for i, bbox in enumerate(sorted_bboxes):
        x1, y1, x2, y2 = bbox
        quality_flag = _classify_quality_flag(bbox, median_h, img.width)
        if quality_flag == "structural_marker":
            continue

        pad = 2
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img.width, x2 + pad)
        y2 = min(img.height, y2 + pad)

        line_img = img.crop((x1, y1, x2, y2))
        line_id = f"{page_id}_{lang_key}_L{i:03d}"
        page_lang_dir = images_out_dir / page_id / lang_key
        page_lang_dir.mkdir(parents=True, exist_ok=True)
        rel_img_path = str((page_lang_dir / f"{line_id}.png").relative_to(settings.BASE_DIR))
        abs_out_path = settings.BASE_DIR / rel_img_path
        line_img.save(abs_out_path, format="PNG")

        crops_metadata.append(
            {
                "line_id": line_id,
                "doc_stem": doc_stem,
                "page_id": page_id,
                "column_key": lang_key,
                "image_path": rel_img_path,
                "quality_flag": quality_flag,
                "source_run_dir": str(doc_root.relative_to(settings.BASE_DIR)),
            }
        )
        generated += 1
    return generated


def _write_prepare_outputs(
    *,
    run_dir: Path,
    doc_root: Path,
    doc_stem: str,
    crops_metadata: list[dict],
    refresh_frozen_split: bool,
) -> None:
    stable_split_hash = compute_records_hash(_stable_split_records(crops_metadata))

    meta_path = run_dir / "candidate_crops.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(crops_metadata, f, indent=2)

    split_manifest_path = run_dir / "split_manifest.json"
    train_count = sum(1 for row in crops_metadata if row["split"] == "train")
    holdout_count = sum(1 for row in crops_metadata if row["split"] == "holdout")
    write_split_manifest(
        path=split_manifest_path,
        dataset_hash=stable_split_hash,
        random_seed=42,
        train_count=train_count,
        holdout_count=holdout_count,
    )
    if refresh_frozen_split:
        freeze_root = settings.INPUT_DIR / "ocr_benchmark" / "frozen_splits"
        freeze_path = freeze_root / f"{doc_stem}.json"
        freeze_root.mkdir(parents=True, exist_ok=True)
        freeze_path.write_text(split_manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
        logger.warning("Refreshed frozen split hash for %s at %s", doc_stem, freeze_path)
    else:
        try:
            _enforce_frozen_split(doc_stem=doc_stem, split_manifest_path=split_manifest_path)
        except ValueError as exc:
            raise ValueError(
                f"{exc}\n\nIf you intentionally changed selection/heuristics, rerun with "
                f"'--refresh-freeze' once."
            ) from exc

    register_latest_run(
        stage="ocr-benchmark-prepare",
        doc_stem=doc_stem,
        run_dir=run_dir,
        artifacts={
            "crops_metadata": str(meta_path.relative_to(settings.BASE_DIR)),
            "split_manifest": str(split_manifest_path.relative_to(settings.BASE_DIR)),
        },
        metadata={
            "doc_root": str(doc_root.relative_to(settings.BASE_DIR)),
            "dataset_hash": stable_split_hash,
            "random_seed": 42,
            "train_count": train_count,
            "holdout_count": holdout_count,
            "detection_strategy": "strip_surya_detection",
            "reading_order_sort": "y_centroid_tolerance_then_xmin",
        },
    )


def generate_candidate_lines(
    doc_stem: str,
    limit_pages: int | None = None,
    include_languages: list[str] | None = None,
    refresh_frozen_split: bool = False,
) -> Path:
    """
    Extract line crops using Surya DetectionPredictor from the latest `crop-columns` output.
    Returns the versioned run directory path.
    """
    normalized_languages = _normalize_requested_languages(include_languages)
    spliced_dir = _resolve_spliced_dir(doc_stem)

    # Setup document-root and stage directory
    doc_root = create_new_doc_benchmark_root(doc_stem)
    run_dir = doc_root / "prep"
    run_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir = run_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Surya DetectionPredictor for doc {doc_stem}...")
    predictor = DetectionPredictor()

    strip_tasks = _collect_strip_tasks(spliced_dir, normalized_languages, limit_pages)

    if not strip_tasks:
        raise ValueError(
            "No benchmark strip images found for selected languages. "
            "Confirm crop-columns artifacts exist and include geez/amharic columns."
        )

    generated_lines = 0
    crops_metadata = []
    strip_tasks_by_lang: dict[str, list[tuple[str, Path, str]]] = {
        lang: [] for lang in normalized_languages
    }
    for task in strip_tasks:
        strip_tasks_by_lang[task[2]].append(task)

    for lang_key in normalized_languages:
        lang_tasks = strip_tasks_by_lang.get(lang_key, [])
        if not lang_tasks:
            continue
        logger.info(
            "Benchmark Prepare start: doc=%s language=%s strips=%d",
            doc_stem,
            lang_key,
            len(lang_tasks),
        )
        progress = tqdm(
            total=len(lang_tasks),
            desc=f"Benchmark Prepare ({lang_key})",
            unit="strip",
            dynamic_ncols=True,
        )
        for page_id, col_img_path, lang_key in lang_tasks:
            generated_lines += _process_strip_task(
                task=(page_id, col_img_path, lang_key),
                predictor=predictor,
                images_out_dir=images_out_dir,
                doc_root=doc_root,
                doc_stem=doc_stem,
                crops_metadata=crops_metadata,
            )
            progress.update(1)
        progress.close()

    logger.info(f"Extracted {generated_lines} candidate lines to {images_out_dir}")

    # Enforce deterministic splits before persisting
    crops_metadata = _deterministic_split_assignment(
        crops_metadata,
        include_languages=normalized_languages,
    )
    logger.info(f"Frozen dataset split: {len(crops_metadata)} lines retained for pilot.")

    _write_prepare_outputs(
        run_dir=run_dir,
        doc_root=doc_root,
        doc_stem=doc_stem,
        crops_metadata=crops_metadata,
        refresh_frozen_split=refresh_frozen_split,
    )

    return run_dir
