from __future__ import annotations

import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image
from tqdm import tqdm

from modules.ocr_benchmark.metrics import calculate_cer_wer_paper
from modules.ocr_training.surya_common import (
    deterministic_sample_rows,
    infer_row_modality,
    sanitize_prediction_text,
    subset_rows,
)

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


@dataclass(slots=True)
class EvalBatchTiming:
    """Per-batch timing breakdown for explicit Surya evaluation."""

    batch_index: int
    batch_size: int
    image_load_sec: float
    batch_prep_sec: float
    predictor_sec: float
    decode_sec: float
    metric_sec: float
    total_sec: float
    running_samples_per_second: float


@dataclass(slots=True)
class PreparedEvalRows:
    """Prepared split rows plus the deterministic subsetting time."""

    rows: list[dict[str, str]]
    row_selection_sec: float


@dataclass(slots=True)
class EvalRunArtifacts:
    """Prediction records and optional batch timings from one eval run."""

    records: list[dict[str, Any]]
    world_size: int
    batch_timings: list[EvalBatchTiming]


def _chunk_rows(rows: list[dict[str, str]], batch_size: int) -> list[list[dict[str, str]]]:
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def _shard_rows_for_rank(
    rows: list[dict[str, str]],
    *,
    rank: int,
    world_size: int,
) -> list[dict[str, str]]:
    if world_size <= 1:
        return rows
    return [row for index, row in enumerate(rows) if index % world_size == rank]


def prepare_eval_rows(
    *,
    rows: list[dict[str, str]],
    modality: str | None,
    eval_fraction: float,
    max_rows: int | None,
    seed: int,
) -> PreparedEvalRows:
    """Apply deterministic modality/fraction/max-row selection for evaluation."""
    started_at = perf_counter()
    selected_rows = rows
    if modality is not None:
        normalized_modality = modality.strip().lower()
        selected_rows = [
            row for row in selected_rows if infer_row_modality(row) == normalized_modality
        ]
    if eval_fraction < 1.0:
        selected_rows = subset_rows(selected_rows, fraction=eval_fraction, seed=seed)
    if max_rows is not None and len(selected_rows) > max_rows:
        selected_rows = deterministic_sample_rows(selected_rows, max_rows=max_rows, seed=seed)
    return PreparedEvalRows(
        rows=selected_rows,
        row_selection_sec=perf_counter() - started_at,
    )


def build_in_memory_eval_predictor(*, runtime, model, processor):
    """Build a RecognitionPredictor around the current in-memory training model."""
    torch = runtime["torch"]
    foundation_cls = runtime["FoundationPredictor"]
    foundation_predictor = foundation_cls.__new__(foundation_cls)
    foundation_predictor.model = model
    foundation_predictor.processor = processor
    foundation_predictor.prompt_queue = deque()
    foundation_predictor.batch_prompt_mapping = None
    foundation_predictor.kv_cache = None
    foundation_predictor.beacon_token_interval = model.config.beacon_token_interval
    foundation_predictor.device_pad_token = torch.tensor(
        processor.pad_token_id,
        device=model.device,
        dtype=torch.long,
    )
    foundation_predictor.device_beacon_token = torch.tensor(
        processor.beacon_token_id,
        device=model.device,
        dtype=torch.long,
    )
    foundation_predictor.special_token_ids = torch.tensor(
        [model.config.image_token_id, *model.config.register_token_ids],
        device=model.device,
    )
    foundation_predictor.pad_to_multiple = None
    foundation_predictor._disable_tqdm = False
    predictor = runtime["RecognitionPredictor"](foundation_predictor)
    predictor.disable_tqdm = True
    return predictor


def _load_eval_image(row: dict[str, str]) -> tuple[Image.Image, list[int]]:
    image_path = Path(row["image"])
    with Image.open(image_path) as image:
        converted = image.convert("RGB")
    return converted, [0, 0, converted.width, converted.height]


def _gather_object(torch_module, payload: Any, world_size: int) -> list[Any]:
    gathered_payload = [None] * world_size
    torch_module.distributed.all_gather_object(gathered_payload, payload)
    return gathered_payload


def run_surya_eval_batches(
    *,
    rows: list[dict[str, str]],
    split: str,
    eval_batch_size: int,
    predictor,
    runtime,
    dataloader_num_workers: int = 0,
    distributed_context=None,
    torch_module=None,
    collect_batch_timings: bool = False,
    progress_desc: str | None = None,
) -> EvalRunArtifacts:
    """Run OCR inference for one deterministic row list and optionally collect batch timings."""
    rank = int(getattr(distributed_context, "rank", 0) if distributed_context else 0)
    world_size = int(getattr(distributed_context, "world_size", 1) if distributed_context else 1)
    local_rows = _shard_rows_for_rank(rows, rank=rank, world_size=world_size)

    records: list[dict[str, Any]] = []
    batch_timings: list[EvalBatchTiming] = []
    wall_started_at = perf_counter()
    progress = tqdm(
        _chunk_rows(local_rows, max(1, eval_batch_size)),
        desc=progress_desc or f"Evaluate {split}",
        unit="batch",
        dynamic_ncols=True,
        disable=bool(distributed_context and not distributed_context.is_rank_zero),
    )
    processed_rows = 0
    executor = (
        ThreadPoolExecutor(max_workers=dataloader_num_workers)
        if dataloader_num_workers > 0
        else None
    )
    try:
        for batch_index, row_batch in enumerate(progress):
            batch_started_at = perf_counter()

            load_started_at = perf_counter()
            if executor is not None:
                loaded_batch = list(executor.map(_load_eval_image, row_batch))
            else:
                loaded_batch = [_load_eval_image(row) for row in row_batch]
            image_load_sec = perf_counter() - load_started_at

            prep_started_at = perf_counter()
            images = [item[0] for item in loaded_batch]
            # RecognitionPredictor expects one list of boxes per image, even when
            # we are evaluating a single full-image OCR box for each sample.
            bboxes = [[item[1]] for item in loaded_batch]
            task_names = [runtime["TaskNames"].ocr_with_boxes] * len(images)
            batch_prep_sec = perf_counter() - prep_started_at

            predictor_started_at = perf_counter()
            results = predictor(
                images,
                task_names=task_names,
                bboxes=bboxes,
                math_mode=False,
                drop_repeated_text=True,
                filter_tag_list=TAG_FILTER_LIST,
            )
            predictor_sec = perf_counter() - predictor_started_at

            decode_started_at = perf_counter()
            decoded_predictions = [
                sanitize_prediction_text(result.text_lines[0].text) if result.text_lines else ""
                for result in results
            ]
            decode_sec = perf_counter() - decode_started_at

            metric_started_at = perf_counter()
            for row, raw_pred in zip(row_batch, decoded_predictions, strict=False):
                gt_text = row["text"]
                cer, wer, exact = calculate_cer_wer_paper(raw_pred, gt_text)
                records.append(
                    {
                        "image": row["image"],
                        "gt_text": gt_text,
                        "pred_text": raw_pred,
                        "cer": cer,
                        "wer": wer,
                        "exact": exact,
                    }
                )
            metric_sec = perf_counter() - metric_started_at

            processed_rows += len(row_batch)
            total_sec = perf_counter() - batch_started_at
            running_sps = processed_rows / max(1e-9, perf_counter() - wall_started_at)
            progress.set_postfix(
                rows=processed_rows,
                workers=dataloader_num_workers,
                sps=f"{running_sps:.2f}",
            )
            if collect_batch_timings:
                batch_timings.append(
                    EvalBatchTiming(
                        batch_index=batch_index,
                        batch_size=len(row_batch),
                        image_load_sec=image_load_sec,
                        batch_prep_sec=batch_prep_sec,
                        predictor_sec=predictor_sec,
                        decode_sec=decode_sec,
                        metric_sec=metric_sec,
                        total_sec=total_sec,
                        running_samples_per_second=running_sps,
                    )
                )
    finally:
        progress.close()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    if distributed_context and distributed_context.is_distributed:
        gathered_records = _gather_object(torch_module, records, distributed_context.world_size)
        gathered_timings = _gather_object(
            torch_module,
            [asdict(timing) for timing in batch_timings],
            distributed_context.world_size,
        )
        if not distributed_context.is_rank_zero:
            return EvalRunArtifacts(records=[], world_size=world_size, batch_timings=[])
        records = [record for chunk in gathered_records if chunk for record in chunk]
        batch_timings = [
            EvalBatchTiming(**timing) for chunk in gathered_timings if chunk for timing in chunk
        ]
    return EvalRunArtifacts(records=records, world_size=world_size, batch_timings=batch_timings)


def write_batch_timings_jsonl(output_path: Path, batch_timings: list[EvalBatchTiming]) -> Path:
    """Persist one JSONL line per timed batch."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for timing in batch_timings:
            handle.write(json.dumps(asdict(timing), ensure_ascii=False) + "\n")
    return output_path


def reset_eval_peak_vram(torch_module, *, distributed_context=None) -> None:
    """Best-effort reset of CUDA peak-memory stats before one eval/benchmark run."""
    if torch_module is None or not torch_module.cuda.is_available():
        return
    device = getattr(distributed_context, "device", None)
    try:
        if device is None:
            torch_module.cuda.reset_peak_memory_stats()
        else:
            torch_module.cuda.reset_peak_memory_stats(device)
    except Exception:
        return


def collect_eval_peak_vram_mb(torch_module, *, distributed_context=None) -> int | None:
    """Return the maximum reserved CUDA memory observed during one eval run."""
    if torch_module is None or not torch_module.cuda.is_available():
        return None
    device = getattr(distributed_context, "device", None)
    try:
        reserved_bytes = (
            torch_module.cuda.max_memory_reserved()
            if device is None
            else torch_module.cuda.max_memory_reserved(device)
        )
    except Exception:
        return None
    peak_mb = int(reserved_bytes // (1024 * 1024))
    if not distributed_context or not distributed_context.is_distributed:
        return peak_mb
    gathered = _gather_object(torch_module, peak_mb, distributed_context.world_size)
    if not distributed_context.is_rank_zero:
        return None
    return max(int(value or 0) for value in gathered)


def maybe_sync_cuda(torch_module) -> None:
    """Synchronize CUDA timing points when available."""
    if torch_module is None or not torch_module.cuda.is_available():
        return
    with suppress(Exception):
        torch_module.cuda.synchronize()
