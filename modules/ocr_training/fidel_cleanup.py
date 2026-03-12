from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

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


def cleanup_fidel_extracted(
    *,
    extracted_root: Path,
    output_root: Path,
    workers: int = 8,
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
    suspect_dir = output_root / "suspect_blank_images"
    cleaned_extracted_root.mkdir(parents=True, exist_ok=True)
    review_root.mkdir(parents=True, exist_ok=True)
    excluded_dir.mkdir(parents=True, exist_ok=True)
    suspect_dir.mkdir(parents=True, exist_ok=True)

    _copy_tree(extracted_root, cleaned_extracted_root)
    rows = _read_snapshot_rows(snapshot_path)
    cleaned_rows: list[SourceSnapshotRow] = []
    excluded_rows: list[dict[str, object]] = []
    suspect_rows: list[dict[str, object]] = []
    counts_by_type: Counter[str] = Counter()
    excluded_by_type: Counter[str] = Counter()
    suspect_by_type: Counter[str] = Counter()
    normalized_workers = max(1, int(workers))
    auditable_rows: list[tuple[SourceSnapshotRow, Path]] = []

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
                excluded_rows.append(enriched)
                excluded_by_type[row.normalized_type.value] += 1
                _copy_or_link_file(source_image, excluded_dir / source_image.name)
                if cleaned_image.exists():
                    cleaned_image.unlink()
                cleaned_rows.append(_build_excluded_row(row=row))
                progress.set_postfix(
                    {
                        "workers": normalized_workers,
                        "excluded": len(excluded_rows),
                        "suspect": len(suspect_rows),
                    },
                    refresh=False,
                )
                continue
            if audit["classification"] == "suspect_blank" and row.text_normalized.strip():
                suspect_rows.append(enriched)
                suspect_by_type[row.normalized_type.value] += 1
                _copy_or_link_file(source_image, suspect_dir / source_image.name)
            cleaned_rows.append(_build_cleaned_row(row=row, cleaned_image=cleaned_image))
            progress.set_postfix(
                {
                    "workers": normalized_workers,
                    "excluded": len(excluded_rows),
                    "suspect": len(suspect_rows),
                },
                refresh=False,
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
                "excluded_rows_by_type": dict(excluded_by_type),
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
        "suspect_rows": len(suspect_rows),
    }
