import json
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from config.settings import settings
from modules.ocr_benchmark.dataset import DatasetSplit, compute_records_hash
from modules.ocr_benchmark.paths import resolve_doc_benchmark_root
from utils.logger import get_logger
from utils.run_registry import load_latest_run, register_latest_run

logger = get_logger("OCRBenchmarkTrOCR")
TROCR_STAGE1_CHECKPOINT = "microsoft/trocr-base-stage1"


def _sanitize_prediction_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"</?([A-Za-z][A-Za-z0-9_-]*)(\\s[^>]*)?>", "", text)
    cleaned = cleaned.replace("<>", "")
    cleaned = cleaned.replace("<", "").replace(">", "")
    return " ".join(cleaned.split())


def run_zero_shot_baseline(doc_stem: str, split: str = "all") -> Path:
    """
    Run TrOCR zero-shot baseline on extracted candidate line crops.
    Defaults to `all` so the same artifact can seed both holdout and train tasks.
    """
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

    if split == "all":
        target_crops = [
            crop
            for crop in crops
            if crop.get("split") in {DatasetSplit.TRAIN.value, DatasetSplit.HOLDOUT.value}
        ]
    else:
        target_crops = [crop for crop in crops if crop.get("split") == split]

    doc_root = resolve_doc_benchmark_root(doc_stem)
    run_dir = doc_root / "trocr_zero_shot"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = run_dir / f"baseline_predictions_{split}.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading TrOCR processor+model for zero-shot baseline on device=%s ...", device)
    logger.warning(
        "TrOCR zero-shot is being run from stage1 backbone (%s). "
        "This is a structural baseline only; Ethiopic-quality results require finetuning.",
        TROCR_STAGE1_CHECKPOINT,
    )
    processor = TrOCRProcessor.from_pretrained(TROCR_STAGE1_CHECKPOINT)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_STAGE1_CHECKPOINT).to(device)  # type: ignore[assignment]
    model.eval()

    predictions = []
    logger.info("Running TrOCR zero-shot on %d %s line crops...", len(target_crops), split)
    progress = tqdm(
        total=len(target_crops),
        desc="TrOCR Zero-Shot",
        unit="line",
        dynamic_ncols=True,
    )
    for crop in target_crops:
        img_path = settings.BASE_DIR / crop["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            logger.warning("Failed to load image %s: %s", img_path, exc)
            progress.update(1)
            continue

        try:
            pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)  # type: ignore[attr-defined]
            with torch.no_grad():
                generated_ids = model.generate(  # type: ignore[call-overload]
                    pixel_values,
                    max_new_tokens=128,
                    do_sample=False,
                )
            raw_pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            pred_text = _sanitize_prediction_text(raw_pred)
        except Exception as exc:
            logger.warning("TrOCR generation failed for %s: %s", img_path, exc)
            pred_text = ""

        predictions.append(
            {
                "line_id": crop["line_id"],
                "doc_stem": crop["doc_stem"],
                "page_id": crop["page_id"],
                "column_key": crop["column_key"],
                "image_path": crop["image_path"],
                "source_run_dir": crop.get("source_run_dir"),
                "split": crop["split"],
                "raw_pred": pred_text,
                "confidence": None,
            }
        )
        progress.update(1)
    progress.close()

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    register_latest_run(
        stage="ocr-benchmark-trocr-zero",
        doc_stem=doc_stem,
        run_dir=run_dir,
        artifacts={"baseline_predictions_jsonl": str(out_jsonl.relative_to(settings.BASE_DIR))},
        metadata={
            "doc_root": str(doc_root.relative_to(settings.BASE_DIR)),
            "model_name": TROCR_STAGE1_CHECKPOINT,
            "split_policy": split,
            "num_predictions": len(predictions),
            "input_records_hash": compute_records_hash(crops),
            "predictions_hash": compute_records_hash(predictions),
        },
    )
    return run_dir
