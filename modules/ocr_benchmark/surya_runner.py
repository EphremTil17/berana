import json
import re
from pathlib import Path

import surya
from PIL import Image
from surya.common.surya.schema import TaskNames
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from tqdm import tqdm

from config.settings import settings
from modules.ocr_benchmark.dataset import DatasetSplit, compute_records_hash
from modules.ocr_benchmark.paths import resolve_doc_benchmark_root
from utils.logger import get_logger
from utils.run_registry import load_latest_run, register_latest_run

logger = get_logger("OCRBenchmarkSurya")
TAG_FILTER_LIST = [
    "p",
    "li",
    "ul",
    "ol",
    "table",
    "td",
    "tr",
    "th",
    "tbody",
    "pre",
    "b",
    "strong",
    "i",
    "em",
    "u",
    "span",
    "div",
    "br",
    "sup",
    "sub",
]


def _sanitize_prediction_text(text: str) -> str:
    """Drop markup artifacts that are not valid Ethiopic transcription content."""
    if not text:
        return ""
    cleaned = re.sub(r"</?([A-Za-z][A-Za-z0-9_-]*)(\s[^>]*)?>", "", text)
    cleaned = cleaned.replace("<>", "")
    cleaned = cleaned.replace("<", "").replace(">", "")
    return " ".join(cleaned.split())


def run_zero_shot_baseline(doc_stem: str, split: str = "all") -> Path:
    """Run Surya Zero-Shot baseline on extracted candidate line crops."""
    split = split.lower().strip()
    if split not in {"train", "holdout", "all"}:
        raise ValueError(f"Unsupported split '{split}'. Use train|holdout|all.")

    prepare_pointer = load_latest_run("ocr-benchmark-prepare", doc_stem)
    if not prepare_pointer:
        raise FileNotFoundError(f"Missing pipeline prepare stage for {doc_stem}")

    crops_meta_rel = prepare_pointer["artifacts"]["crops_metadata"]
    crops_meta_abs = settings.BASE_DIR / crops_meta_rel

    with crops_meta_abs.open("r", encoding="utf-8") as f:
        crops = json.load(f)

    doc_root = resolve_doc_benchmark_root(doc_stem)
    run_dir = doc_root / "surya_zero_shot"
    run_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = run_dir / f"baseline_predictions_{split}.jsonl"

    logger.info("Loading Surya FoundationPredictor + RecognitionPredictor...")
    foundation_predictor = FoundationPredictor()
    predictor = RecognitionPredictor(foundation_predictor)
    predictor.disable_tqdm = True

    predictions = []

    if split == "all":
        target_crops = [
            crop
            for crop in crops
            if crop.get("split") in {DatasetSplit.TRAIN.value, DatasetSplit.HOLDOUT.value}
        ]
    else:
        target_crops = [crop for crop in crops if crop.get("split") == split]

    logger.info("Running zero-shot recognition on %d %s line crops...", len(target_crops), split)
    progress = tqdm(
        total=len(target_crops),
        desc="Surya Zero-Shot",
        unit="line",
        dynamic_ncols=True,
    )
    for crop in target_crops:
        img_path = settings.BASE_DIR / crop["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            progress.update(1)
            continue

        # Pass one full-image bbox since each input is already a single line crop.
        res = predictor(
            [img],
            task_names=[TaskNames.ocr_with_boxes],
            bboxes=[[[0, 0, img.width, img.height]]],
            math_mode=False,
            drop_repeated_text=True,
            filter_tag_list=TAG_FILTER_LIST,
        )[0]

        pred_text = _sanitize_prediction_text(res.text_lines[0].text) if res.text_lines else ""
        confidence = res.text_lines[0].confidence if res.text_lines else 0.0

        pred_record = {
            "line_id": crop["line_id"],
            "doc_stem": crop["doc_stem"],
            "page_id": crop["page_id"],
            "column_key": crop["column_key"],
            "image_path": crop["image_path"],
            "source_run_dir": crop.get("source_run_dir"),
            "split": crop["split"],  # CRITICAL FIX: Consume deterministic split from prepare phase
            "raw_pred": pred_text,
            "confidence": confidence,
        }
        predictions.append(pred_record)
        progress.update(1)
    progress.close()

    with out_jsonl.open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    pred_hash = compute_records_hash(predictions)
    input_hash = compute_records_hash(crops)
    register_latest_run(
        stage="ocr-benchmark-surya-zero",
        doc_stem=doc_stem,
        run_dir=run_dir,
        artifacts={"baseline_predictions_jsonl": str(out_jsonl.relative_to(settings.BASE_DIR))},
        metadata={
            "doc_root": str(doc_root.relative_to(settings.BASE_DIR)),
            "model_name": "surya-recognition",
            "surya_version": lock_surya_version(),
            "split_policy": split,
            "num_predictions": len(predictions),
            "input_records_hash": input_hash,
            "predictions_hash": pred_hash,
        },
    )

    return run_dir


def lock_surya_version() -> str:
    """Helper to pin the exact Surya version running for finetuning reproducibility."""
    try:
        return surya.__version__
    except AttributeError:
        from importlib.metadata import version

        return version("surya-ocr")
