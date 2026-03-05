from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def build_script_prefixed_text(text: str, runtime: dict[str, Any]) -> str:
    """Return OCR ground-truth text with Surya script-prefix tokens."""
    scripts = runtime["get_top_scripts"](text)
    mapping = runtime["SCRIPT_TOKEN_MAPPING"]
    prefix = "".join(mapping.get(script, "") for script in scripts)
    return f"{prefix}{text}"


def build_surya_training_sample(
    *,
    processor,
    row: dict[str, str],
    runtime: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, tuple[int, int]]]:
    """Build one Surya OCR training sample and return image-shape metadata."""
    with Image.open(row["image"]) as image:
        rgb_image = image.convert("RGB")
        original_size = rgb_image.size
        image_np = processor.scale_to_fit(
            np.asarray(rgb_image, dtype=np.float32),
            max_size=(1024, 512),
        )

    gt_text = build_script_prefixed_text(row["text"], runtime)
    sample = {
        "task": runtime["TaskNames"].ocr_with_boxes,
        "inputs": [
            runtime["ImageInput"](type="image", image=image_np, rotated=False),
            runtime["TextInput"](type="text", text=""),
            runtime["TextInput"](type="text", text=gt_text),
        ],
    }
    image_meta = {
        "original_size": original_size,
        "processed_size": (int(image_np.shape[1]), int(image_np.shape[0])),
    }
    return sample, image_meta


class LocalSuryaOCRDataset:
    """Torch dataset for locally materialized Surya OCR JSONL records."""

    def __init__(self, *, processor, rows: list[dict[str, str]], runtime: dict[str, Any]):
        """Initialize dataset with processor, row payloads, and Surya runtime helpers."""
        self.processor = processor
        self.rows = rows
        self.runtime = runtime

    def __len__(self) -> int:
        """Return number of records in the local split."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one Surya-formatted OCR training sample."""
        row = self.rows[index]
        sample, _image_meta = build_surya_training_sample(
            processor=self.processor,
            row=row,
            runtime=self.runtime,
        )
        return sample


class SuryaOCRDataCollator:
    """Data collator aligned with Surya finetuning script conventions."""

    def __init__(self, processor, max_sequence_length: int | None, task_name):
        """Initialize collator with processor, optional sequence cap, and fixed task name."""
        self.processor = processor
        self.max_sequence_length = max_sequence_length
        self.task_name = task_name

    def __call__(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        """Batch and mask labels for supervised autoregressive OCR training."""
        batch = self.processor(inputs, padding_side="right")
        if self.max_sequence_length is not None:
            batch["input_ids"] = batch["input_ids"][:, : self.max_sequence_length]
            batch["attention_mask"] = batch["attention_mask"][:, : self.max_sequence_length]
            batch["position_ids"] = batch["position_ids"][:, : self.max_sequence_length]

        lm_labels = batch["input_ids"].clone()
        skip_mask = (
            (lm_labels == self.processor.pad_token_id)
            | (lm_labels == self.processor.eoi_token_id)
            | (lm_labels == self.processor.image_token_id)
        )
        bos_token = None
        bos_map = getattr(self.processor, "bos_token_id", None)
        if isinstance(bos_map, dict):
            bos_token = bos_map.get(self.task_name)
        if bos_token is not None:
            skip_mask = skip_mask | (lm_labels == bos_token)

        lm_labels[skip_mask] = -100
        batch["labels"] = lm_labels
        batch["cache_position"] = batch["position_ids"].clone()
        return batch
