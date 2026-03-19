import json
from pathlib import Path

import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
from transformers.optimization import get_linear_schedule_with_warmup

from config.settings import settings
from modules.ocr_benchmark.coverage import ensure_coverage_gate
from modules.ocr_benchmark.dataset import DatasetSplit, compute_records_hash, read_manifest
from modules.ocr_benchmark.paths import resolve_doc_benchmark_root
from modules.ocr_benchmark.trocr_runner import TROCR_STAGE1_CHECKPOINT, _sanitize_prediction_text
from utils.logger import get_logger
from utils.run_registry import register_latest_run

logger = get_logger("OCRBenchmarkTrOCRFinetune")
PROMPT_TOKENS = ["<gez>", "<amh>"]


def _build_ethiopic_tokenizer(
    text: str = "በስመ አብ ወወልድ ወመንፈስ ቅዱስ፥ አሐዱ አምላክ አሜን።",
) -> PreTrainedTokenizerBase:
    """
    Execute the required TrOCR Ethiopic Tokenizer Roundtrip Preflight.
    Without this XLM-RoBERTa step, TrOCR fails silently on Amharic/Ge'ez text.
    """
    logger.info("Running TrOCR Ethiopic Tokenizer Preflight...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load xlm-roberta-base tokenizer. "
            "If your environment lacks SentencePiece, install it with "
            "`pip install sentencepiece` and rerun."
        ) from exc
    tokenizer.add_special_tokens({"additional_special_tokens": PROMPT_TOKENS})

    ids = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.unk_token_id in ids:
        raise ValueError(
            "TR_OCR PREFLIGHT FAILED: XLM-R produced <unk> tokens for valid Ethiopic script. "
            "Finetuning will be completely defective."
        )

    decoded = tokenizer.decode(ids).replace(" ", "").replace(" ", "").strip()
    clean_text = text.replace(" ", "").replace(" ", "").strip()
    if clean_text != decoded:
        raise ValueError(
            f"TR_OCR PREFLIGHT FAILED: Roundtrip decoded text drifted from GT.\n"
            f"Orig: {clean_text}\nDeco: {decoded}"
        )

    logger.info("Tokenizer roundtrip passed. TrOCR embedding resize is functionally safe.")
    return tokenizer


def preflight_tokenizer_check(
    text: str = "በስመ አብ ወወልድ ወመንፈስ ቅዱስ፥ አሐዱ አምላክ አሜን።",
) -> PreTrainedTokenizerBase:
    """Public preflight for CLI/tests."""
    return _build_ethiopic_tokenizer(text=text)


def initialize_trocr_ethiopic() -> tuple[VisionEncoderDecoderModel, PreTrainedTokenizerBase]:
    """Initialize TrOCR and resize decoder embeddings for Ethiopic-capable tokenizer."""
    tokenizer = _build_ethiopic_tokenizer()
    logger.info("Loading TrOCR stage1 backbone model...")
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_STAGE1_CHECKPOINT)

    logger.info("Resizing TrOCR decoder embeddings to XLM-R vocab size=%d...", len(tokenizer))
    model.decoder.resize_token_embeddings(len(tokenizer))  # type: ignore[reportOptionalMemberAccess]

    model.config.decoder_start_token_id = tokenizer.bos_token_id  # type: ignore[reportAttributeAccessIssue]
    model.config.pad_token_id = tokenizer.pad_token_id  # type: ignore[reportAttributeAccessIssue]
    model.config.eos_token_id = tokenizer.eos_token_id  # type: ignore[reportAttributeAccessIssue]
    model.config.vocab_size = model.config.decoder.vocab_size  # type: ignore[reportOptionalMemberAccess]

    return model, tokenizer  # type: ignore[reportReturnType]


class OCRLineDataset(Dataset):
    """TrOCR finetune dataset backed by benchmark manifest rows."""

    def __init__(
        self,
        rows,
        *,
        processor: TrOCRProcessor,
        tokenizer: PreTrainedTokenizerBase,
        max_target_length: int = 192,
    ):
        """Initialize dataset with manifest rows, processor, and target tokenization controls."""
        self.rows = rows
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        """Return number of rows."""
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        """Return model-ready sample for a single row index."""
        row = self.rows[idx]
        image_path = settings.BASE_DIR / row.image_path
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)  # type: ignore[reportCallIssue]

        prompt = row.lang_prompt.value
        target_text = f"{prompt} {row.gt_text}"
        tokenized = self.tokenizer(
            target_text,
            add_special_tokens=True,
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = tokenized["input_ids"].squeeze(0)  # type: ignore[reportAttributeAccessIssue]
        return {"pixel_values": pixel_values, "labels": labels}


def _collate_batch(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    max_len = max(item["labels"].shape[0] for item in batch)
    labels = torch.full((len(batch), max_len), fill_value=-100, dtype=torch.long)
    for i, item in enumerate(batch):
        seq = item["labels"].clone()
        seq[seq == pad_token_id] = -100
        labels[i, : seq.shape[0]] = seq
    return {"pixel_values": pixel_values, "labels": labels}


def run_trocr_finetune(
    doc_stem: str,
    dataset_manifest: Path,
    *,
    enforce_coverage: bool = True,
    charset_config_path: Path | None = None,
    epochs: int = 15,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_target_length: int = 192,
    seed: int = 42,
) -> Path:
    """Train TrOCR on TRAIN split and emit HOLDOUT predictions for evaluation."""
    if not dataset_manifest.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {dataset_manifest}")

    doc_root = resolve_doc_benchmark_root(doc_stem)
    run_dir = doc_root / "trocr_finetune"
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model, tokenizer = initialize_trocr_ethiopic()
    processor = TrOCRProcessor.from_pretrained(TROCR_STAGE1_CHECKPOINT)
    processor.tokenizer = tokenizer  # type: ignore[reportAttributeAccessIssue]

    charset_cfg = charset_config_path or (
        settings.INPUT_DIR / "ocr_benchmark" / "config" / "ethiopic_charset.v1.json"
    )
    coverage_report, coverage_out_dir = ensure_coverage_gate(
        doc_stem=doc_stem,
        manifest_path=dataset_manifest,
        charset_config_path=charset_cfg,
        enforce=enforce_coverage,
    )

    rows = [row for row in read_manifest(dataset_manifest) if row.doc_stem == doc_stem]
    holdout_rows = [row for row in rows if row.split == DatasetSplit.HOLDOUT]
    if not holdout_rows:
        raise ValueError(f"No holdout rows found for doc_stem '{doc_stem}' in {dataset_manifest}.")

    train_rows = [
        row for row in rows if row.split == DatasetSplit.TRAIN and (row.gt_text or "").strip()
    ]
    if not train_rows:
        raise ValueError(f"No train rows with non-empty gt_text found for doc_stem '{doc_stem}'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # type: ignore[reportArgumentType]
    model.train()

    train_dataset = OCRLineDataset(
        train_rows,
        processor=processor,
        tokenizer=tokenizer,
        max_target_length=max_target_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate_batch(b, tokenizer.pad_token_id),  # type: ignore[reportArgumentType]
    )

    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(
        "Starting TrOCR finetuning doc=%s train_rows=%d holdout_rows=%d epochs=%d batch_size=%d",
        doc_stem,
        len(train_rows),
        len(holdout_rows),
        epochs,
        batch_size,
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress = tqdm(
            total=len(train_loader),
            desc=f"TrOCR Finetune E{epoch + 1}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.item())
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}")
        progress.close()
        mean_loss = epoch_loss / max(1, len(train_loader))
        logger.info("Epoch %d/%d complete mean_loss=%.5f", epoch + 1, epochs, mean_loss)

    model_dir = run_dir / "checkpoint_final"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)

    model.eval()
    out_jsonl = run_dir / "finetuned_predictions.jsonl"
    predictions: list[dict] = []

    logger.info("Running holdout inference with finetuned TrOCR on %d lines...", len(holdout_rows))
    infer_progress = tqdm(
        total=len(holdout_rows),
        desc="TrOCR Holdout Inference",
        unit="line",
        dynamic_ncols=True,
    )
    for row in holdout_rows:
        image_path = settings.BASE_DIR / row.image_path
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)  # type: ignore[reportCallIssue]
        forced_id = tokenizer.convert_tokens_to_ids(row.lang_prompt.value)  # type: ignore[reportCallIssue]
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=max_target_length,
                do_sample=False,
                forced_bos_token_id=forced_id,
            )
        raw_pred = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]  # type: ignore[reportArgumentType]
        pred_text = _sanitize_prediction_text(raw_pred)
        predictions.append(
            {
                "line_id": row.line_id,
                "doc_stem": row.doc_stem,
                "page_id": row.page_id,
                "column_key": row.column_key.value,
                "image_path": row.image_path,
                "split": row.split.value,
                "raw_pred": pred_text,
                "confidence": None,
            }
        )
        infer_progress.update(1)
    infer_progress.close()

    with out_jsonl.open("w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    train_hash = compute_records_hash([row.model_dump(mode="json") for row in train_rows])
    holdout_hash = compute_records_hash([row.model_dump(mode="json") for row in holdout_rows])
    pred_hash = compute_records_hash(predictions)

    register_latest_run(
        stage="ocr-benchmark-trocr-finetune",
        doc_stem=doc_stem,
        run_dir=run_dir,
        artifacts={
            "finetuned_predictions_jsonl": str(out_jsonl.relative_to(settings.BASE_DIR)),
            "checkpoint_dir": str(model_dir.relative_to(settings.BASE_DIR)),
        },
        metadata={
            "doc_root": str(doc_root.relative_to(settings.BASE_DIR)),
            "model_name": f"{TROCR_STAGE1_CHECKPOINT} + xlm-roberta-base tokenizer",
            "train_count": len(train_rows),
            "holdout_count": len(holdout_rows),
            "dataset_manifest": str(dataset_manifest),
            "status": "completed",
            "preflight_tokenizer": "passed_with_prompt_tokens",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "warmup_ratio": warmup_ratio,
            "max_target_length": max_target_length,
            "seed": seed,
            "train_records_hash": train_hash,
            "holdout_records_hash": holdout_hash,
            "predictions_hash": pred_hash,
            "num_predictions": len(predictions),
            "coverage_enforced": enforce_coverage,
            "coverage_status": coverage_report.coverage_status,
            "coverage_report": str(coverage_out_dir / "coverage_report.json"),
        },
    )
    logger.info(
        "TrOCR finetuning completed doc=%s checkpoint=%s predictions=%s",
        doc_stem,
        model_dir,
        out_jsonl,
    )
    return run_dir
