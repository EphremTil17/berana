from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from modules.ocr_training.failure_analysis import load_heuristic_exclusion_index
from modules.ocr_training.schemas import SourceSnapshotRow
from modules.ocr_training.surya_debug import audit_image_blankness


def _snapshot_path_from_extracted_root(extracted_root: Path) -> Path:
    """Resolve the source snapshot path associated with one extracted root."""
    return extracted_root.parent / "manifests" / "source_snapshots" / "fidel_sources.jsonl"


def _read_snapshot_rows(snapshot_path: Path) -> list[SourceSnapshotRow]:
    """Load one source snapshot manifest."""
    rows: list[SourceSnapshotRow] = []
    for line in snapshot_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(SourceSnapshotRow.model_validate_json(line))
    return rows


def _write_snapshot_rows(path: Path, rows: list[SourceSnapshotRow]) -> None:
    """Write one source snapshot manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(row.model_dump_json() for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _copy_or_link_file(source: Path, destination: Path) -> None:
    """Materialize one file at the destination, preferring hard links."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _safe_built_filename(sample_id: str, original_name: str) -> str:
    """Mirror the build-stage dataset filename for one source row."""
    safe_id = sample_id.replace(":", "__").replace("/", "_")
    return f"{safe_id}__{Path(original_name).name}"


def _copy_tree(source_root: Path, destination_root: Path) -> None:
    """Clone one directory tree using hard links where possible."""
    progress = tqdm(
        sorted(source_root.rglob("*")),
        desc="Clone extracted tree",
        unit="path",
        dynamic_ncols=True,
    )
    for source_path in progress:
        relative = source_path.relative_to(source_root)
        destination_path = destination_root / relative
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
            continue
        _copy_or_link_file(source_path, destination_path)
    progress.close()


def _resolve_snapshot_image(path_str: str) -> Path:
    """Resolve one project-relative snapshot image path to an absolute filesystem path."""
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _audit_image_path(image_path: Path) -> dict[str, object]:
    """Run one blank-image audit for a resolved source image path."""
    return audit_image_blankness(image_path)


def _audit_result_iterator(
    *,
    auditable_rows: list[tuple[SourceSnapshotRow, Path]],
    workers: int,
):
    """Yield blank-image audits for the provided rows, optionally using a bounded pool."""
    if workers == 1:
        return (_audit_image_path(image_path) for _row, image_path in auditable_rows), None
    executor = ThreadPoolExecutor(max_workers=workers)
    iterator = executor.map(
        _audit_image_path,
        (image_path for _row, image_path in auditable_rows),
    )
    return iterator, executor


def _build_cleaned_row(
    *,
    row: SourceSnapshotRow,
    cleaned_image: Path,
) -> SourceSnapshotRow:
    """Create one cleaned source snapshot row with the rewritten image path."""
    return row.model_copy(
        update={
            "image_relpath": os.path.relpath(cleaned_image, Path.cwd()),
        }
    )


def _build_excluded_row(*, row: SourceSnapshotRow) -> SourceSnapshotRow:
    """Create one excluded source snapshot row for confirmed blank images."""
    return row.model_copy(
        update={
            "excluded": True,
            "excluded_reason": "confirmed_blank_after_fidel_cleanup",
        }
    )


def _build_heuristic_excluded_row(
    *,
    row: SourceSnapshotRow,
    categories: set[str],
) -> SourceSnapshotRow:
    """Create one excluded source snapshot row for heuristic analysis matches."""
    ordered = ",".join(sorted(categories))
    return row.model_copy(
        update={
            "excluded": True,
            "excluded_reason": f"heuristic_exclusion_after_fidel_cleanup:{ordered}",
        }
    )


def _update_cleanup_progress(
    progress: tqdm,
    *,
    workers: int,
    blank_excluded_rows: int,
    heuristic_excluded_rows: int,
    suspect_rows: int,
) -> None:
    """Refresh one cleanup progress bar postfix."""
    progress.set_postfix(
        {
            "workers": workers,
            "blank": blank_excluded_rows,
            "heur": heuristic_excluded_rows,
            "suspect": suspect_rows,
        },
        refresh=False,
    )


def _handle_blank_exclusion(
    *,
    row: SourceSnapshotRow,
    source_image: Path,
    cleaned_image: Path,
    enriched: dict[str, object],
    excluded_rows: list[dict[str, object]],
    excluded_by_type: Counter[str],
    excluded_dir: Path,
    cleaned_rows: list[SourceSnapshotRow],
) -> None:
    """Exclude one confirmed blank row from the cleaned snapshot."""
    excluded_rows.append(enriched)
    excluded_by_type[row.normalized_type.value] += 1
    _copy_or_link_file(source_image, excluded_dir / source_image.name)
    if cleaned_image.exists():
        cleaned_image.unlink()
    cleaned_rows.append(_build_excluded_row(row=row))


def _handle_heuristic_exclusion(
    *,
    row: SourceSnapshotRow,
    source_image: Path,
    cleaned_image: Path,
    enriched: dict[str, object],
    heuristic_categories: set[str],
    excluded_rows: list[dict[str, object]],
    heuristic_excluded_by_type: Counter[str],
    heuristic_counts: Counter[str],
    heuristic_dir: Path,
    cleaned_rows: list[SourceSnapshotRow],
) -> None:
    """Exclude one row matched by the exact-false heuristic analysis bundle."""
    enriched["heuristic_categories"] = sorted(heuristic_categories)
    excluded_rows.append(enriched)
    heuristic_excluded_by_type[row.normalized_type.value] += 1
    for category in heuristic_categories:
        heuristic_counts[category] += 1
    _copy_or_link_file(source_image, heuristic_dir / source_image.name)
    if cleaned_image.exists():
        cleaned_image.unlink()
    cleaned_rows.append(_build_heuristic_excluded_row(row=row, categories=heuristic_categories))


def cleanup_fidel_extracted(
    *,
    extracted_root: Path,
    output_root: Path,
    workers: int = 8,
    heuristic_cleanup_dir: Path | None = None,
) -> dict[str, object]:
    """Create one cleaned extracted-root copy and filtered source snapshot manifest."""
    snapshot_path = _snapshot_path_from_extracted_root(extracted_root)
    if not extracted_root.exists():
        raise FileNotFoundError(f"Missing extracted root: {extracted_root}")
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing source snapshot: {snapshot_path}")

    cleaned_extracted_root = output_root / "extracted"
    review_root = output_root / "blank_cleanup_review"
    excluded_dir = output_root / "excluded_blank_images"
    heuristic_dir = output_root / "excluded_heuristic_images"
    suspect_dir = output_root / "suspect_blank_images"
    cleaned_extracted_root.mkdir(parents=True, exist_ok=True)
    review_root.mkdir(parents=True, exist_ok=True)
    excluded_dir.mkdir(parents=True, exist_ok=True)
    heuristic_dir.mkdir(parents=True, exist_ok=True)
    suspect_dir.mkdir(parents=True, exist_ok=True)

    _copy_tree(extracted_root, cleaned_extracted_root)
    rows = _read_snapshot_rows(snapshot_path)
    cleaned_rows: list[SourceSnapshotRow] = []
    excluded_rows: list[dict[str, object]] = []
    suspect_rows: list[dict[str, object]] = []
    counts_by_type: Counter[str] = Counter()
    excluded_by_type: Counter[str] = Counter()
    heuristic_excluded_by_type: Counter[str] = Counter()
    suspect_by_type: Counter[str] = Counter()
    blank_excluded_rows = 0
    heuristic_excluded_rows = 0
    normalized_workers = max(1, int(workers))
    auditable_rows: list[tuple[SourceSnapshotRow, Path]] = []
    heuristic_index = (
        load_heuristic_exclusion_index(heuristic_cleanup_dir)
        if heuristic_cleanup_dir is not None
        else None
    )
    heuristic_counts: Counter[str] = Counter()

    for row in rows:
        if row.excluded or not row.image_relpath:
            cleaned_rows.append(row)
            continue
        auditable_rows.append((row, _resolve_snapshot_image(row.image_relpath)))

    audit_results, executor = _audit_result_iterator(
        auditable_rows=auditable_rows,
        workers=normalized_workers,
    )

    progress = tqdm(
        auditable_rows,
        desc="Cleanup FIDEL extracted",
        unit="row",
        dynamic_ncols=True,
    )
    try:
        for (row, source_image), audit in zip(progress, audit_results, strict=False):
            enriched = {
                **row.model_dump(mode="json"),
                **audit,
            }
            counts_by_type[row.normalized_type.value] += 1
            relative_image = source_image.relative_to(extracted_root.resolve())
            cleaned_image = cleaned_extracted_root / relative_image
            if audit["classification"] == "confirmed_blank" and row.text_normalized.strip():
                _handle_blank_exclusion(
                    row=row,
                    source_image=source_image,
                    cleaned_image=cleaned_image,
                    enriched=enriched,
                    excluded_rows=excluded_rows,
                    excluded_by_type=excluded_by_type,
                    excluded_dir=excluded_dir,
                    cleaned_rows=cleaned_rows,
                )
                blank_excluded_rows += 1
                _update_cleanup_progress(
                    progress,
                    workers=normalized_workers,
                    blank_excluded_rows=blank_excluded_rows,
                    heuristic_excluded_rows=heuristic_excluded_rows,
                    suspect_rows=len(suspect_rows),
                )
                continue
            built_name = _safe_built_filename(row.sample_id, row.original_filename)
            heuristic_categories = (
                heuristic_index["by_basename"].get(built_name, set()) if heuristic_index else set()
            )
            if heuristic_categories and row.text_normalized.strip():
                _handle_heuristic_exclusion(
                    row=row,
                    source_image=source_image,
                    cleaned_image=cleaned_image,
                    enriched=enriched,
                    heuristic_categories=heuristic_categories,
                    excluded_rows=excluded_rows,
                    heuristic_excluded_by_type=heuristic_excluded_by_type,
                    heuristic_counts=heuristic_counts,
                    heuristic_dir=heuristic_dir,
                    cleaned_rows=cleaned_rows,
                )
                heuristic_excluded_rows += 1
                _update_cleanup_progress(
                    progress,
                    workers=normalized_workers,
                    blank_excluded_rows=blank_excluded_rows,
                    heuristic_excluded_rows=heuristic_excluded_rows,
                    suspect_rows=len(suspect_rows),
                )
                continue
            if audit["classification"] == "suspect_blank" and row.text_normalized.strip():
                suspect_rows.append(enriched)
                suspect_by_type[row.normalized_type.value] += 1
                _copy_or_link_file(source_image, suspect_dir / source_image.name)
            cleaned_rows.append(_build_cleaned_row(row=row, cleaned_image=cleaned_image))
            _update_cleanup_progress(
                progress,
                workers=normalized_workers,
                blank_excluded_rows=blank_excluded_rows,
                heuristic_excluded_rows=heuristic_excluded_rows,
                suspect_rows=len(suspect_rows),
            )
    finally:
        progress.close()
        if normalized_workers != 1:
            executor.shutdown(wait=True)

    _write_snapshot_rows(
        output_root / "manifests" / "source_snapshots" / "fidel_sources.jsonl",
        cleaned_rows,
    )
    (review_root / "summary.json").write_text(
        json.dumps(
            {
                "source_extracted_root": str(extracted_root),
                "cleaned_extracted_root": str(cleaned_extracted_root),
                "included_rows_by_type": dict(counts_by_type),
                "excluded_rows": len(excluded_rows),
                "blank_excluded_rows": blank_excluded_rows,
                "blank_excluded_rows_by_type": dict(excluded_by_type),
                "heuristic_cleanup_dir": str(heuristic_cleanup_dir)
                if heuristic_cleanup_dir is not None
                else None,
                "heuristic_excluded_rows": heuristic_excluded_rows,
                "heuristic_excluded_rows_by_type": dict(heuristic_excluded_by_type),
                "heuristic_excluded_rows_by_category": dict(heuristic_counts),
                "suspect_rows": len(suspect_rows),
                "suspect_rows_by_type": dict(suspect_by_type),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (review_root / "excluded_rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in excluded_rows)
        + ("\n" if excluded_rows else ""),
        encoding="utf-8",
    )
    (review_root / "suspect_rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in suspect_rows)
        + ("\n" if suspect_rows else ""),
        encoding="utf-8",
    )
    return {
        "cleaned_extracted_root": str(cleaned_extracted_root),
        "excluded_rows": len(excluded_rows),
        "blank_excluded_rows": blank_excluded_rows,
        "suspect_rows": len(suspect_rows),
        "heuristic_excluded_rows": heuristic_excluded_rows,
        "heuristic_excluded_rows_by_category": dict(heuristic_counts),
    }
