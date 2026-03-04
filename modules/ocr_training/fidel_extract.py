from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from config.settings import settings
from modules.ocr_benchmark.dataset import compute_records_hash
from modules.ocr_training.fidel_catalog import CatalogRow, load_catalog
from modules.ocr_training.normalize import build_extracted_filename
from modules.ocr_training.registry import STAGE_FIDEL_EXTRACT, register_training_stage
from modules.ocr_training.schemas import (
    ExtractionSummary,
    NormalizedType,
    SourceRepo,
    SourceSnapshotRow,
    SourceSplit,
)
from utils.logger import get_logger

logger = get_logger("OCRTrainingFidelExtract")


@dataclass(frozen=True)
class ArchiveSpec:
    """Archive metadata for streaming extraction."""

    source_repo: SourceRepo
    source_split: SourceSplit
    zip_path: Path
    content_prefix: str


@dataclass
class ExtractionCounters:
    """Mutable counters produced during archive extraction."""

    extracted_new: int = 0
    extracted_existing: int = 0
    unknown_entries: int = 0
    skipped_macosx_entries: int = 0


def _normalize_filter_values(values: set[str]) -> set[NormalizedType]:
    mapping = {
        "typed": NormalizedType.TYPED,
        "synthetic": NormalizedType.SYNTHETIC,
        "handwritten": NormalizedType.HANDWRITTEN,
        "hdd": NormalizedType.HANDWRITTEN,
        "hdd_18": NormalizedType.HANDWRITTEN,
        "hdd_rand": NormalizedType.HANDWRITTEN,
    }
    normalized: set[NormalizedType] = set()
    for value in values:
        key = value.strip().lower()
        if key not in mapping:
            raise ValueError(f"Unsupported type filter value: '{value}'.")
        normalized.add(mapping[key])
    return normalized


def _archive_specs(raw_root: Path) -> list[ArchiveSpec]:
    dataset_root = raw_root / SourceRepo.FIDEL_DATASET.value
    synthetic_root = raw_root / SourceRepo.FIDEL_SYNTHETIC.value
    specs = [
        ArchiveSpec(
            SourceRepo.FIDEL_DATASET, SourceSplit.TRAIN, dataset_root / "train.zip", "train/"
        ),
        ArchiveSpec(SourceRepo.FIDEL_DATASET, SourceSplit.TEST, dataset_root / "test.zip", "test/"),
        ArchiveSpec(
            SourceRepo.FIDEL_SYNTHETIC,
            SourceSplit.SYNTHETIC,
            synthetic_root / "data.zip",
            "data/",
        ),
    ]
    missing = [str(spec.zip_path) for spec in specs if not spec.zip_path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required archives: {missing}")
    return specs


def _build_lookup(
    catalog: list[CatalogRow],
) -> dict[tuple[SourceRepo, SourceSplit, str], CatalogRow]:
    lookup = {}
    for row in catalog:
        key = (row.source_repo, row.source_split, row.original_filename)
        if key in lookup:
            raise ValueError(f"Duplicate catalog key encountered: {key}")
        lookup[key] = row
    return lookup


def _relative_to_base(path: Path) -> str:
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(settings.BASE_DIR))
    except ValueError:
        return str(path)


def _write_stream_to_file(src, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_suffix(dst.suffix + ".tmp")
    with tmp_path.open("wb") as out:
        shutil.copyfileobj(src, out)
    tmp_path.replace(dst)


def _resolve_expected_ids(
    catalog: list[CatalogRow],
    *,
    effective_include: set[NormalizedType],
    excluded_types: set[NormalizedType],
) -> set[str]:
    return {
        row.sample_id
        for row in catalog
        if row.normalized_type in effective_include and row.normalized_type not in excluded_types
    }


def _count_types(
    catalog: list[CatalogRow], expected_ids: set[str]
) -> tuple[dict[str, int], dict[str, int]]:
    included: dict[str, int] = {}
    excluded: dict[str, int] = {}
    for row in catalog:
        target = included if row.sample_id in expected_ids else excluded
        key = row.normalized_type.value
        target[key] = target.get(key, 0) + 1
    return included, excluded


def _extract_archive_entry(
    *,
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    spec: ArchiveSpec,
    extracted_root: Path,
    lookup: dict[tuple[SourceRepo, SourceSplit, str], CatalogRow],
    expected_ids: set[str],
    extracted_relpath_by_id: dict[str, str],
    seen_ids: set[str],
    counters: ExtractionCounters,
    overwrite: bool,
    dry_run: bool,
) -> None:
    name = info.filename
    if info.is_dir():
        return
    if name.startswith("__MACOSX/"):
        counters.skipped_macosx_entries += 1
        return
    if not name.startswith(spec.content_prefix) or not name.lower().endswith(".png"):
        return

    original_name = Path(name).name
    row = lookup.get((spec.source_repo, spec.source_split, original_name))
    if row is None:
        counters.unknown_entries += 1
        return
    if row.sample_id not in expected_ids:
        return

    dest_name = build_extracted_filename(row.source_repo, row.source_split, row.original_filename)
    dest_path = extracted_root / row.normalized_type.value / dest_name
    relpath = _relative_to_base(dest_path)

    if dest_path.exists():
        if dest_path.stat().st_size == info.file_size:
            counters.extracted_existing += 1
            extracted_relpath_by_id[row.sample_id] = relpath
            seen_ids.add(row.sample_id)
            return
        if not overwrite:
            raise ValueError(
                "Existing file size mismatch for extraction target and overwrite disabled: "
                f"{dest_path}"
            )

    if not dry_run:
        with zf.open(info, "r") as src_stream:
            _write_stream_to_file(src_stream, dest_path)
    counters.extracted_new += 1
    extracted_relpath_by_id[row.sample_id] = relpath
    seen_ids.add(row.sample_id)


def _extract_archives(
    *,
    specs: list[ArchiveSpec],
    extracted_root: Path,
    lookup: dict[tuple[SourceRepo, SourceSplit, str], CatalogRow],
    expected_ids: set[str],
    overwrite: bool,
    dry_run: bool,
) -> tuple[dict[str, str], set[str], ExtractionCounters]:
    extracted_relpath_by_id: dict[str, str] = {}
    seen_ids: set[str] = set()
    counters = ExtractionCounters()

    for spec in specs:
        logger.info("Scanning archive: %s", spec.zip_path)
        with zipfile.ZipFile(spec.zip_path, "r") as zf:
            archive_entries = zf.infolist()
            progress = tqdm(
                archive_entries,
                desc=f"Extract {spec.zip_path.name}",
                unit="entry",
                dynamic_ncols=True,
            )
            for info in progress:
                _extract_archive_entry(
                    zf=zf,
                    info=info,
                    spec=spec,
                    extracted_root=extracted_root,
                    lookup=lookup,
                    expected_ids=expected_ids,
                    extracted_relpath_by_id=extracted_relpath_by_id,
                    seen_ids=seen_ids,
                    counters=counters,
                    overwrite=overwrite,
                    dry_run=dry_run,
                )
                progress.set_postfix(
                    {
                        "new": counters.extracted_new,
                        "existing": counters.extracted_existing,
                        "unknown": counters.unknown_entries,
                    },
                    refresh=False,
                )
            progress.close()

    return extracted_relpath_by_id, seen_ids, counters


def _write_source_snapshot(
    *,
    catalog: list[CatalogRow],
    expected_ids: set[str],
    extracted_relpath_by_id: dict[str, str],
    source_snapshot_path: Path,
) -> tuple[list[dict], Path]:
    records_for_hash: list[dict] = []
    with source_snapshot_path.open("w", encoding="utf-8") as handle:
        progress = tqdm(
            catalog,
            desc="Write source snapshot",
            unit="row",
            dynamic_ncols=True,
        )
        for row in progress:
            include = row.sample_id in expected_ids
            snapshot = SourceSnapshotRow(
                sample_id=row.sample_id,
                source_repo=row.source_repo,
                source_split=row.source_split,
                original_filename=row.original_filename,
                normalized_type=row.normalized_type,
                text_raw=row.text_raw,
                text_normalized=row.text_normalized,
                image_relpath=extracted_relpath_by_id.get(row.sample_id),
                excluded=not include,
                excluded_reason=None if include else f"filtered_type:{row.normalized_type.value}",
            )
            handle.write(snapshot.model_dump_json(exclude_none=True) + "\n")
            records_for_hash.append(snapshot.model_dump(mode="json", exclude_none=True))
            progress.set_postfix({"written": len(records_for_hash)}, refresh=False)
        progress.close()
    return records_for_hash, source_snapshot_path


def _build_summary(
    *,
    expected_ids: set[str],
    seen_ids: set[str],
    counters: ExtractionCounters,
    type_counts_included: dict[str, int],
    type_counts_excluded: dict[str, int],
) -> ExtractionSummary:
    missing_expected = len(expected_ids - seen_ids)
    missing_rate = (missing_expected / len(expected_ids)) if expected_ids else 0.0
    return ExtractionSummary(
        expected_included=len(expected_ids),
        extracted_new=counters.extracted_new,
        extracted_existing=counters.extracted_existing,
        missing_expected=missing_expected,
        missing_rate=missing_rate,
        unknown_archive_entries=counters.unknown_entries,
        skipped_macosx_entries=counters.skipped_macosx_entries,
        included_type_counts=type_counts_included,
        excluded_type_counts=type_counts_excluded,
    )


def _write_reports(
    *,
    base_root: Path,
    records_for_hash: list[dict],
    summary: ExtractionSummary,
) -> tuple[Path, Path]:
    provenance_root = base_root / "provenance"
    provenance_root.mkdir(parents=True, exist_ok=True)

    extract_report_path = provenance_root / "extract_report.json"
    extract_report_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    hash_path = provenance_root / "source_snapshot_hash.txt"
    hash_path.write_text(compute_records_hash(records_for_hash), encoding="utf-8")
    return extract_report_path, hash_path


def _register_stage(
    *,
    raw_root: Path,
    extracted_root: Path,
    source_snapshot_path: Path,
    extract_report_path: Path,
    hash_path: Path,
    include_types: set[NormalizedType],
    exclude_types: set[NormalizedType],
    summary: ExtractionSummary,
) -> None:
    metadata = {
        "status": "completed",
        "raw_root": _relative_to_base(raw_root),
        "extracted_root": _relative_to_base(extracted_root),
        "include_types": sorted(t.value for t in include_types),
        "exclude_types": sorted(t.value for t in exclude_types),
        "expected_included": summary.expected_included,
        "extracted_new": summary.extracted_new,
        "extracted_existing": summary.extracted_existing,
        "missing_expected": summary.missing_expected,
        "missing_rate": summary.missing_rate,
        "source_snapshot_hash": hash_path.read_text(encoding="utf-8").strip(),
    }
    register_training_stage(
        stage=STAGE_FIDEL_EXTRACT,
        run_key="fidel",
        run_dir=extracted_root,
        artifacts={
            "source_snapshot": _relative_to_base(source_snapshot_path),
            "extract_report": _relative_to_base(extract_report_path),
            "source_snapshot_hash": _relative_to_base(hash_path),
        },
        metadata=metadata,
    )


def _prepare_output_dirs(extracted_root: Path, effective_include: set[NormalizedType]) -> None:
    extracted_root.mkdir(parents=True, exist_ok=True)
    for output_type in effective_include:
        (extracted_root / output_type.value).mkdir(parents=True, exist_ok=True)


def _validate_workers(workers: int) -> None:
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    if workers != 1:
        logger.warning(
            "--workers is currently reserved for future parallel extraction optimizations. "
            "Proceeding with deterministic single-stream zip reads."
        )


def extract_fidel(
    *,
    raw_root: Path,
    extracted_root: Path,
    include_types: set[str],
    exclude_types: set[str],
    allow_missing_rate: float,
    workers: int,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Path | dict]:
    """Extract FIDEL archives into canonical typed/synthetic buckets."""
    _validate_workers(workers)

    include_norm = _normalize_filter_values(include_types)
    exclude_norm = _normalize_filter_values(exclude_types)
    effective_include = include_norm - exclude_norm
    if not effective_include:
        raise ValueError("Effective include set is empty after exclude filters.")

    _prepare_output_dirs(extracted_root, effective_include)
    catalog = load_catalog(raw_root)
    expected_ids = _resolve_expected_ids(
        catalog,
        effective_include=effective_include,
        excluded_types=exclude_norm,
    )
    lookup = _build_lookup(catalog)
    specs = _archive_specs(raw_root)
    type_counts_included, type_counts_excluded = _count_types(catalog, expected_ids)

    extracted_relpath_by_id, seen_ids, counters = _extract_archives(
        specs=specs,
        extracted_root=extracted_root,
        lookup=lookup,
        expected_ids=expected_ids,
        overwrite=overwrite,
        dry_run=dry_run,
    )

    summary = _build_summary(
        expected_ids=expected_ids,
        seen_ids=seen_ids,
        counters=counters,
        type_counts_included=type_counts_included,
        type_counts_excluded=type_counts_excluded,
    )
    if summary.missing_rate > allow_missing_rate:
        raise ValueError(
            "Missing-rate threshold exceeded for extracted samples. "
            f"missing={summary.missing_expected}, expected={summary.expected_included}, "
            f"missing_rate={summary.missing_rate:.6f}, allow_missing_rate={allow_missing_rate:.6f}"
        )

    base_root = extracted_root.parent
    source_snapshot_root = base_root / "manifests" / "source_snapshots"
    source_snapshot_root.mkdir(parents=True, exist_ok=True)
    source_snapshot_path = source_snapshot_root / "fidel_sources.jsonl"

    records_for_hash, source_snapshot_path = _write_source_snapshot(
        catalog=catalog,
        expected_ids=expected_ids,
        extracted_relpath_by_id=extracted_relpath_by_id,
        source_snapshot_path=source_snapshot_path,
    )
    extract_report_path, hash_path = _write_reports(
        base_root=base_root,
        records_for_hash=records_for_hash,
        summary=summary,
    )
    _register_stage(
        raw_root=raw_root,
        extracted_root=extracted_root,
        source_snapshot_path=source_snapshot_path,
        extract_report_path=extract_report_path,
        hash_path=hash_path,
        include_types=effective_include,
        exclude_types=exclude_norm,
        summary=summary,
    )

    logger.info(
        "FIDEL extraction complete expected=%d new=%d existing=%d missing=%d",
        summary.expected_included,
        summary.extracted_new,
        summary.extracted_existing,
        summary.missing_expected,
    )
    return {
        "source_snapshot": source_snapshot_path,
        "extract_report": extract_report_path,
        "hash_path": hash_path,
        "summary": summary.model_dump(mode="json"),
    }
