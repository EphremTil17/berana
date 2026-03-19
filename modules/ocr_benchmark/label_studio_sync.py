import json
import random
from pathlib import Path
from urllib.parse import quote

from config.settings import settings
from modules.ocr_benchmark.dataset import (
    LineManifestRow,
    read_manifest,
    validate_split_leakage,
    write_manifest,
)
from schemas.ocr_benchmark import ColumnKey, DatasetSplit, LangPrompt
from utils.logger import get_logger
from utils.run_registry import load_latest_run, resolve_required_input

logger = get_logger("OCRBenchmarkLabelStudio")


def _to_label_studio_local_files_url(image_path: str) -> str:
    """
    Build Label Studio local-files URL from a project-relative image path.
    Mirrors layout-infer contract: path must be output-root-relative.
    """
    image_abs = (settings.BASE_DIR / image_path).resolve()
    output_root = settings.OUTPUT_DIR.resolve()
    try:
        output_rel = image_abs.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(
            f"Image path must resolve under output root for Label Studio local-files serving. "
            f"Got '{image_abs}', output root is '{output_root}'."
        ) from exc

    rel_url_path = str(output_rel).replace("\\", "/")
    return f"/data/local-files/?d={quote(rel_url_path, safe='/')}"


def create_import_tasks(
    doc_stem: str,
    output_json: Path,
    split: str = "holdout",
    random_seed: int = 42,
) -> None:
    """
    Create Label Studio tasks from benchmark candidate crops, optionally injecting
    Surya Zero-Shot pre-predictions when available.
    """
    split = split.lower().strip()
    if split not in {"train", "holdout", "all"}:
        raise ValueError(f"Unsupported split value '{split}'. Use train|holdout|all.")

    crops_meta_path = resolve_required_input(
        upstream_stage="ocr-benchmark-prepare",
        doc_stem=doc_stem,
        artifact_key="crops_metadata",
    )
    crops = json.loads(crops_meta_path.read_text(encoding="utf-8"))

    predicted_by_line: dict[str, dict] = {}
    zero_pointer = load_latest_run("ocr-benchmark-surya-zero", doc_stem)
    if zero_pointer:
        predictions_rel = zero_pointer["artifacts"]["baseline_predictions_jsonl"]
        predictions_path = settings.BASE_DIR / predictions_rel
        with predictions_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                predicted_by_line[data["line_id"]] = data
    else:
        logger.warning(
            "Zero-shot predictions not found for '%s'. "
            "Tasks will be exported without pre-annotations.",
            doc_stem,
        )

    ls_tasks = []
    for row in crops:
        row_split = row.get("split")
        if row_split == "exclude":
            continue
        if split != "all" and row_split != split:
            continue
        ls_url = _to_label_studio_local_files_url(row["image_path"])
        pred = predicted_by_line.get(row["line_id"])
        predictions_payload = []
        if pred is not None:
            predictions_payload.append(
                {
                    "model_version": "surya-zero-shot",
                    "score": pred.get("confidence", 0.0),
                    "result": [
                        {
                            "from_name": "transcription",
                            "to_name": "image",
                            "type": "textarea",
                            "value": {"text": [pred.get("raw_pred", "")]},
                        }
                    ],
                }
            )

        ls_tasks.append(
            {
                "data": {
                    "image": ls_url,
                    "line_id": row["line_id"],
                    "doc_stem": row["doc_stem"],
                    "page_id": row["page_id"],
                    "split_hint": row_split,
                    "lang": row["column_key"],
                    "source_run_dir": row.get("source_run_dir"),
                    # Bind directly to TextArea value so annotators can edit prefilled text.
                    "transcription_seed": pred.get("raw_pred", "") if pred is not None else "",
                },
                "predictions": predictions_payload,
            }
        )

    # Randomize to prevent annotator fatigue scaling predictably across splits
    random.seed(random_seed)
    random.shuffle(ls_tasks)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(ls_tasks, f, indent=2)

    if ls_tasks:
        first_url = ls_tasks[0]["data"]["image"]
        rel_part = first_url.split("?d=", 1)[-1]
        rel_image_root = str(Path(rel_part).parent.parent.parent).replace("\\", "/")
        ls_abs_path = f"/label-studio/files/{rel_image_root}"
        logger.info(
            "Label Studio Local Files path for this OCR run: %s",
            ls_abs_path,
        )
        logger.info("After setting the path, click Sync, then import %s", output_json)
    logger.info(
        "Generated %d Label Studio tasks (split=%s) at %s",
        len(ls_tasks),
        split,
        output_json,
    )


def _load_existing_manifest_rows(output_manifest: Path) -> dict[str, LineManifestRow]:
    existing_rows: dict[str, LineManifestRow] = {}
    if not output_manifest.exists():
        return existing_rows
    try:
        for row in read_manifest(output_manifest):
            existing_rows[row.line_id] = row
    except Exception as exc:
        logger.warning("Could not load existing manifest for deduplication: %s", exc)
    return existing_rows


def _extract_task_data(task: dict) -> dict:
    return task.get("data", task)


def _extract_gt_text(task: dict) -> str | None:
    annotations = task.get("annotations", [])
    if annotations:
        latest_annotation = annotations[-1]
        try:
            return latest_annotation["result"][0]["value"]["text"][0]
        except (KeyError, IndexError):
            return None
    return task.get("transcription")


def _resolve_lang_prompt(lang: str | None) -> str | None:
    if lang == "geez":
        return "<gez>"
    if lang == "amharic":
        return "<amh>"
    return None


def _resolve_column_key(lang: str | None) -> ColumnKey | None:
    if lang is None:
        return None
    if lang not in {"geez", "amharic"}:
        return None
    return ColumnKey(lang)


def _resolve_lang_prompt_enum(lang: str | None) -> LangPrompt | None:
    lang_prompt = _resolve_lang_prompt(lang)
    if lang_prompt is None:
        return None
    return LangPrompt(lang_prompt)


def _resolve_project_rel_image_path(image_url: str | None) -> str | None:
    if not image_url or "?d=" not in image_url:
        return None
    return f"output/{image_url.split('?d=')[-1]}"


def _parse_task_to_manifest_row(task: dict) -> LineManifestRow | None:
    task_data = _extract_task_data(task)
    line_id = task_data.get("line_id")
    if not line_id:
        return None

    gt_text = _extract_gt_text(task)
    if gt_text is None or not str(gt_text).strip():
        return None

    lang = task_data.get("lang")
    column_key = _resolve_column_key(lang)
    lang_prompt = _resolve_lang_prompt_enum(lang)
    if lang_prompt is None:
        if lang is None:
            return None
        raise ValueError(
            f"Unexpected language key found in Label Studio export: '{lang}'. "
            "Expected one of {'geez', 'amharic'}."
        )
    if column_key is None:
        return None

    project_rel_image_path = _resolve_project_rel_image_path(task_data.get("image"))
    if project_rel_image_path is None:
        return None

    split_hint = task_data.get("split_hint")
    if split_hint not in {DatasetSplit.TRAIN.value, DatasetSplit.HOLDOUT.value}:
        return None

    doc_stem = task_data.get("doc_stem")
    page_id = task_data.get("page_id")
    if not doc_stem or not page_id:
        return None

    return LineManifestRow(
        line_id=line_id,
        doc_stem=doc_stem,
        page_id=page_id,
        column_key=column_key,
        lang_prompt=lang_prompt,
        image_path=project_rel_image_path,
        split=DatasetSplit(split_hint),
        gt_text=str(gt_text),
        source_run_dir=_resolve_source_run_dir(project_rel_image_path, task_data),
    )


def _resolve_source_run_dir(image_path: str, task_data: dict) -> str:
    source_run_dir = task_data.get("source_run_dir")
    if isinstance(source_run_dir, str) and source_run_dir.strip():
        return source_run_dir
    marker = "/images/"
    idx = image_path.find(marker)
    if idx > 0:
        return image_path[:idx]
    return str(Path(image_path).parent)


def parse_export(
    export_json: Path,
    output_manifest: Path,
    *,
    strict_page_isolation: bool = False,
) -> None:
    """
    Parse a Label Studio export and append to the canonical benchmark manifest.
    Uses strict idempotency based on line_id.
    """
    with export_json.open("r", encoding="utf-8") as f:
        ls_data = json.load(f)

    # Idempotency deduplication over any existing manifest
    existing_rows = _load_existing_manifest_rows(output_manifest)

    updates_count = 0
    skipped_incomplete = 0
    conflict_overwrites = 0

    for task in ls_data:
        row = _parse_task_to_manifest_row(task)
        if row is None:
            skipped_incomplete += 1
            continue
        if row.line_id in existing_rows:
            conflict_overwrites += 1
            logger.warning(
                "Conflict on line_id '%s': overwriting existing manifest row.",
                row.line_id,
            )
        existing_rows[row.line_id] = row
        updates_count += 1

    final_rows = list(existing_rows.values())

    # Audit leakage constraints on export integration:
    # - line leakage is always hard-fail
    # - page leakage is configurable (default False for benchmark line-level splits)
    try:
        validate_split_leakage(final_rows, strict_page_isolation=strict_page_isolation)
    except ValueError as e:
        logger.error(f"Leakage validation failed after JSON export import: {e}")
        raise

    write_manifest(final_rows, output_manifest)
    logger.info(
        f"Imported {updates_count} annotations. "
        f"Manifest now contains {len(final_rows)} total rows. "
        f"Conflicts overwritten={conflict_overwrites}, incomplete skipped={skipped_incomplete}."
    )
