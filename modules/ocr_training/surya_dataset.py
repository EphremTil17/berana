from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from tqdm import tqdm

from config.settings import settings
from modules.ocr_benchmark.dataset import compute_records_hash
from modules.ocr_training.registry import STAGE_SURYA_DATASET, register_training_stage
from modules.ocr_training.schemas import DatasetSplit, SourceSnapshotRow, SplitConfig
from modules.ocr_training.splits import assign_splits, validate_split_leakage
from utils.logger import get_logger
from utils.run_registry import next_versioned_dir

logger = get_logger("OCRTrainingSuryaDataset")


def _relative_to_base(path: Path) -> str:
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(settings.BASE_DIR))
    except ValueError:
        return str(path)


def _load_source_snapshot(snapshot_path: Path) -> list[SourceSnapshotRow]:
    rows: list[SourceSnapshotRow] = []
    with snapshot_path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            try:
                rows.append(SourceSnapshotRow.model_validate(data))
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse source snapshot row line={line_num} path={snapshot_path}: {exc}"
                ) from exc
    return rows


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "existing"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _safe_sample_filename(sample_id: str, original_name: str) -> str:
    safe_id = sample_id.replace(":", "__").replace("/", "_")
    return f"{safe_id}__{Path(original_name).name}"


def _snapshot_path_from_extracted_root(extracted_root: Path) -> Path:
    return extracted_root.parent / "manifests" / "source_snapshots" / "fidel_sources.jsonl"


def _safe_review_filename(sample_id: str, original_name: str) -> str:
    safe_id = sample_id.replace(":", "__").replace("/", "_")
    return f"{safe_id}__{Path(original_name).name}"


def _suspect_review_sets(extracted_root: Path) -> tuple[set[str], set[str]]:
    suspect_dir = extracted_root.parent / "suspect_blank_images"
    review_manifest = extracted_root.parent / "blank_cleanup_review" / "suspect_rows.jsonl"
    if not review_manifest.exists():
        return set(), set()

    remaining_names = (
        {path.name for path in suspect_dir.iterdir() if path.is_file()}
        if suspect_dir.exists()
        else set()
    )

    all_suspect_ids: set[str] = set()
    included_ids: set[str] = set()
    for line in review_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row["sample_id"])
        original_filename = str(row["original_filename"])
        image_relpath = str(row.get("image_relpath") or "")
        all_suspect_ids.add(sample_id)
        review_name = _safe_review_filename(sample_id, original_filename)
        if review_name in remaining_names or Path(image_relpath).name in remaining_names:
            included_ids.add(sample_id)
    return all_suspect_ids, included_ids


def build_surya_dataset(
    *,
    extracted_root: Path,
    output_root: Path,
    dataset_name: str,
    split_config: SplitConfig,
    extra_manifest: Path | None = None,
    extra_images_root: Path | None = None,
    extra_weight: float = 0.30,
    include_suspect: bool = False,
) -> Path:
    """Build deterministic local Surya-compatible dataset with train/val/holdout splits."""
    snapshot_path = _snapshot_path_from_extracted_root(extracted_root)
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"Source snapshot not found: {snapshot_path}. Run extract-fidel first."
        )

    if extra_manifest or extra_images_root:
        logger.warning(
            "Berana gold adapter is scaffold-only in phase 1. "
            "Extra manifest/images options are accepted but not yet ingested."
        )

    rows = _load_source_snapshot(snapshot_path)
    all_suspect_ids, suspect_inclusion_ids = _suspect_review_sets(extracted_root)
    included_rows = [
        row
        for row in rows
        if not row.excluded
        and row.image_relpath
        and (
            (not include_suspect and row.sample_id not in all_suspect_ids)
            or (
                include_suspect
                and (row.sample_id not in all_suspect_ids or row.sample_id in suspect_inclusion_ids)
            )
        )
    ]
    if not included_rows:
        raise ValueError("No included rows available to build Surya dataset.")

    assignments = assign_splits(included_rows, split_config)
    split_counts = validate_split_leakage(
        included_rows,
        assignments,
        strict_page_isolation=split_config.strict_page_isolation,
    )

    run_dir = next_versioned_dir(output_root, dataset_name)
    hf_root = run_dir / "data" / "hf_dataset"
    manifest_root = run_dir / "data" / "manifests"
    meta_root = run_dir / "meta"
    for split in (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.HOLDOUT):
        (hf_root / "images" / split.value).mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    split_json_files = {
        DatasetSplit.TRAIN: (hf_root / "train.jsonl").open("w", encoding="utf-8"),
        DatasetSplit.VAL: (hf_root / "val.jsonl").open("w", encoding="utf-8"),
        DatasetSplit.HOLDOUT: (hf_root / "holdout.jsonl").open("w", encoding="utf-8"),
    }
    sidecar_path = manifest_root / "dataset_rows.jsonl"
    sidecar_handle = sidecar_path.open("w", encoding="utf-8")

    link_mode_counts = {"hardlink": 0, "copy": 0, "existing": 0}
    hash_rows: list[dict] = []

    try:
        progress = tqdm(
            included_rows,
            desc="Build Surya dataset",
            unit="sample",
            dynamic_ncols=True,
        )
        for row in progress:
            split = assignments[row.sample_id]
            source_image = settings.BASE_DIR / (row.image_relpath or "")
            if not source_image.exists():
                raise FileNotFoundError(
                    f"Missing extracted image referenced by snapshot: {source_image}"
                )

            target_name = _safe_sample_filename(row.sample_id, row.original_filename)
            target_image = hf_root / "images" / split.value / target_name
            link_mode = _link_or_copy(source_image, target_image)
            link_mode_counts[link_mode] += 1

            record = {
                "image": str(target_image),
                "text": row.text_normalized,
            }
            split_json_files[split].write(json.dumps(record, ensure_ascii=False) + "\n")

            sidecar = {
                "sample_id": row.sample_id,
                "source_repo": row.source_repo.value,
                "source_split": row.source_split.value,
                "normalized_type": row.normalized_type.value,
                "original_filename": row.original_filename,
                "split": split.value,
                "image": str(target_image),
                "text": row.text_normalized,
            }
            sidecar_handle.write(json.dumps(sidecar, ensure_ascii=False) + "\n")
            hash_rows.append(sidecar)
            progress.set_postfix(
                {
                    "hardlink": link_mode_counts["hardlink"],
                    "copy": link_mode_counts["copy"],
                    "existing": link_mode_counts["existing"],
                },
                refresh=False,
            )
        progress.close()
    finally:
        sidecar_handle.close()
        for handle in split_json_files.values():
            handle.close()

    dataset_hash = compute_records_hash(hash_rows)
    split_manifest_path = manifest_root / "split_manifest.json"
    split_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_name": dataset_name,
                "dataset_hash": dataset_hash,
                "seed": split_config.seed,
                "ratios": {
                    "train": split_config.train_ratio,
                    "val": split_config.val_ratio,
                    "holdout": split_config.holdout_ratio,
                },
                "strict_page_isolation": split_config.strict_page_isolation,
                "counts": split_counts,
                "suspect_review_rows": len(all_suspect_ids),
                "suspect_included_rows": len(suspect_inclusion_ids) if include_suspect else 0,
                "include_suspect": include_suspect,
                "extra_manifest": str(extra_manifest) if extra_manifest else None,
                "extra_images_root": str(extra_images_root) if extra_images_root else None,
                "extra_weight": extra_weight,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    config_path = meta_root / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_name": dataset_name,
                "split_config": split_config.model_dump(mode="json"),
                "source_snapshot": _relative_to_base(snapshot_path),
                "link_mode_counts": link_mode_counts,
                "suspect_review_rows": len(all_suspect_ids),
                "suspect_included_rows": len(suspect_inclusion_ids) if include_suspect else 0,
                "include_suspect": include_suspect,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    register_training_stage(
        stage=STAGE_SURYA_DATASET,
        run_key=dataset_name,
        run_dir=run_dir,
        artifacts={
            "train_jsonl": _relative_to_base(hf_root / "train.jsonl"),
            "val_jsonl": _relative_to_base(hf_root / "val.jsonl"),
            "holdout_jsonl": _relative_to_base(hf_root / "holdout.jsonl"),
            "dataset_rows": _relative_to_base(sidecar_path),
            "split_manifest": _relative_to_base(split_manifest_path),
        },
        metadata={
            "status": "completed",
            "dataset_name": dataset_name,
            "dataset_hash": dataset_hash,
            "counts": split_counts,
            "link_mode_counts": link_mode_counts,
            "suspect_review_rows": len(all_suspect_ids),
            "suspect_included_rows": len(suspect_inclusion_ids) if include_suspect else 0,
            "include_suspect": include_suspect,
        },
    )

    logger.info(
        "Surya dataset build complete run_dir=%s train=%d val=%d holdout=%d",
        run_dir,
        split_counts["train"],
        split_counts["val"],
        split_counts["holdout"],
    )
    return run_dir
