from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from modules.ocr_training.surya_debug import audit_image_blankness


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Load one JSONL manifest."""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write one JSONL manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _copy_review_image(source: Path, destination_dir: Path) -> None:
    """Copy one reviewed image into the requested review directory."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if destination.exists():
        return
    destination.write_bytes(source.read_bytes())


def verify_surya_dataset(
    *,
    dataset_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Audit one built hf_dataset for blank-image contradictions without modifying it."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_dir}")

    splits = ("train", "val", "holdout")
    for split in splits:
        split_path = dataset_dir / f"{split}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing dataset split file: {split_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    confirmed_dir = output_dir / "confirmed_blank_images"
    suspect_dir = output_dir / "suspect_blank_images"
    confirmed_rows: list[dict[str, object]] = []
    suspect_rows: list[dict[str, object]] = []
    counts_by_split: Counter[str] = Counter()
    confirmed_by_split: Counter[str] = Counter()
    suspect_by_split: Counter[str] = Counter()

    for split in splits:
        split_rows = _read_jsonl(dataset_dir / f"{split}.jsonl")
        progress = tqdm(
            split_rows,
            desc=f"Verify Surya dataset {split}",
            unit="row",
            dynamic_ncols=True,
        )
        for row in progress:
            image_path = Path(str(row["image"]))
            audit = audit_image_blankness(image_path)
            enriched = {
                **row,
                "split": split,
                **audit,
            }
            counts_by_split[split] += 1
            if audit["classification"] == "confirmed_blank":
                confirmed_rows.append(enriched)
                confirmed_by_split[split] += 1
                _copy_review_image(image_path, confirmed_dir)
            elif audit["classification"] == "suspect_blank":
                suspect_rows.append(enriched)
                suspect_by_split[split] += 1
                _copy_review_image(image_path, suspect_dir)
            progress.set_postfix(
                {
                    "confirmed": len(confirmed_rows),
                    "suspect": len(suspect_rows),
                },
                refresh=False,
            )
        progress.close()

    summary = {
        "dataset_dir": str(dataset_dir),
        "verified_rows": sum(counts_by_split.values()),
        "counts_by_split": dict(counts_by_split),
        "confirmed_blank_rows": len(confirmed_rows),
        "confirmed_blank_by_split": dict(confirmed_by_split),
        "suspect_blank_rows": len(suspect_rows),
        "suspect_blank_by_split": dict(suspect_by_split),
    }
    _write_jsonl(output_dir / "confirmed_blank_rows.jsonl", confirmed_rows)
    _write_jsonl(output_dir / "suspect_blank_rows.jsonl", suspect_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
