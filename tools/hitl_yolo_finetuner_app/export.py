import hashlib
import json
import math
import re
import shutil
from pathlib import Path

from PIL import Image

from tools.hitl_line_editor_app.db import get_provenance, set_provenance
from tools.hitl_line_editor_app.state import load_state
from tools.hitl_yolo_finetuner_app.geometry import apply_vertical_clip, process_line_to_polygon
from utils.logger import get_logger
from utils.run_registry import load_latest_run, next_versioned_dir, register_latest_run

log = get_logger("YOLOExporter")


def parse_omit_pages(omit_pages: str | None) -> set[int]:
    """Parse page filters like '1,2,5-8' into a set of page numbers."""
    parsed: set[int] = set()
    if not omit_pages:
        return parsed
    for part in omit_pages.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_page, end_page = map(int, chunk.split("-", maxsplit=1))
            parsed.update(range(start_page, end_page + 1))
        else:
            parsed.add(int(chunk))
    return parsed


def _page_num_from_id(page_id: str) -> int | None:
    match = re.search(r"(\d+)$", page_id)
    if not match:
        return None
    return int(match.group(1))


def get_split(page_id: str, seed: int) -> str:
    """Deterministically assign 'train' (80%) or 'val' (20%) based on hashed page_id."""
    hash_input = f"{page_id}_{seed}".encode()
    hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
    return "train" if hash_val % 100 < 80 else "val"


def calculate_tilt_deg(line: list[float]) -> float:
    """Calculate the tilt of a line in degrees from vertical."""
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    if dy == 0:
        return 90.0
    angle_rad = math.atan(abs(dx) / abs(dy))
    return math.degrees(angle_rad)


def calculate_length(line: list[float]) -> float:
    """Calculate the length of the line."""
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx * dx + dy * dy)


def write_yolo_label_file(
    out_path: Path, left: list[float], right: list[float], img_w: int, img_h: int, width_px: float
) -> bool:
    """Write normalizing YOLO polygon format. Returns True if valid labels written."""
    poly_left = process_line_to_polygon(left, img_w, img_h, width_px)
    poly_right = process_line_to_polygon(right, img_w, img_h, width_px)

    lines = []
    if poly_left:
        lines.append(f"0 {' '.join(f'{v:.6f}' for v in poly_left)}")
    if poly_right:
        lines.append(f"1 {' '.join(f'{v:.6f}' for v in poly_right)}")

    if not lines:
        return False

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def sync_image(src: Path, dst: Path, copy_mode: str) -> bool:
    """Sync source image to dest using hardlink, copy, or symlink."""
    if not src.exists():
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    try:
        if copy_mode == "hardlink":
            dst.hardlink_to(src)
        elif copy_mode == "symlink":
            dst.symlink_to(src.absolute())
        else:
            shutil.copy2(src, dst)
        return True
    except OSError:
        # Fallback to copy if hardlink/symlink fails (e.g., cross-device)
        try:
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            log.warning(f"Failed to sync image {src} -> {dst}: {e}")
            return False


def _collect_candidates(
    state: dict,
    images_source_dir: Path,
    max_tilt_deg: float,
    min_line_length_px: float,
    clip_top_px: float,
    clip_bottom_px: float,
    omit_pages: set[int],
) -> tuple[list, dict]:
    candidates = []
    skips = {"unverified": 0, "omitted": 0, "missing_lines": 0, "invalid_geometry": 0}

    for page_id, page in state.items():
        page_num = _page_num_from_id(page_id)
        if page_num is not None and page_num in omit_pages:
            skips["omitted"] += 1
            continue

        if not page.get("verified"):
            skips["unverified"] += 1
            continue

        left = page.get("left")
        right = page.get("right")
        img_w = page.get("img_w", 0)
        img_h = page.get("img_h", 0)

        if not left or not right or img_w <= 0 or img_h <= 0:
            skips["missing_lines"] += 1
            continue

        clipped_left = apply_vertical_clip(
            left, img_h, clip_top_px=clip_top_px, clip_bottom_px=clip_bottom_px
        )
        clipped_right = apply_vertical_clip(
            right, img_h, clip_top_px=clip_top_px, clip_bottom_px=clip_bottom_px
        )
        if not clipped_left or not clipped_right:
            skips["invalid_geometry"] += 1
            continue

        if (
            calculate_tilt_deg(clipped_left) > max_tilt_deg
            or calculate_tilt_deg(clipped_right) > max_tilt_deg
        ):
            skips["invalid_geometry"] += 1
            continue

        if (
            calculate_length(clipped_left) < min_line_length_px
            or calculate_length(clipped_right) < min_line_length_px
        ):
            skips["invalid_geometry"] += 1
            continue

        src_img = images_source_dir / f"{page_id}.jpg"
        if not src_img.exists():
            raise FileNotFoundError(f"Verified page '{page_id}' is missing source image: {src_img}")

        with Image.open(src_img) as image:
            actual_w, actual_h = image.size
        if actual_w != img_w or actual_h != img_h:
            raise ValueError(
                f"Dimension mismatch for page '{page_id}': "
                f"db=({img_w}x{img_h}) image=({actual_w}x{actual_h})"
            )

        candidates.append((page_id, clipped_left, clipped_right, img_w, img_h, src_img))
    return candidates, skips


def _commit_split_to_disk(
    split_name: str, pages: list, output_dir: Path, line_width_px: float, copy_mode: str
) -> list[str]:
    (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    written_ids = []
    for page_id, left, right, img_w, img_h, src_img in pages:
        lbl_path = output_dir / "labels" / split_name / f"{page_id}.txt"
        img_dst = output_dir / "images" / split_name / f"{page_id}.jpg"
        if write_yolo_label_file(lbl_path, left, right, img_w, img_h, line_width_px):
            if not sync_image(src_img, img_dst, copy_mode):
                raise RuntimeError(
                    f"Failed to sync image for page '{page_id}' from {src_img} to {img_dst}"
                )
            written_ids.append(page_id)
    return written_ids


def _write_dataset_files(
    data_dir: Path,
    train_pages: list,
    val_pages: list,
    line_width_px: float,
    copy_mode: str,
    min_train_pages: int,
    min_val_pages: int,
) -> tuple[int, int]:
    """Commit YOLO output structure to isolated data directory."""
    data_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = data_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    train_ids = _commit_split_to_disk("train", train_pages, data_dir, line_width_px, copy_mode)
    val_ids = _commit_split_to_disk("val", val_pages, data_dir, line_width_px, copy_mode)

    if len(train_ids) < min_train_pages:
        raise ValueError(f"Insufficient written train pages: {len(train_ids)} < {min_train_pages}")
    if len(val_ids) < min_val_pages:
        raise ValueError(f"Insufficient written val pages: {len(val_ids)} < {min_val_pages}")

    (manifests_dir / "train_pages.txt").write_text("\n".join(train_ids) + "\n")
    (manifests_dir / "val_pages.txt").write_text("\n".join(val_ids) + "\n")

    return len(train_ids), len(val_ids)


def _latest_layout_infer_pointer(doc_stem: str) -> dict:
    latest_run = load_latest_run("layout-infer", doc_stem)
    if not latest_run:
        raise FileNotFoundError(f"Missing registry pointer for layout-infer / {doc_stem}")
    if "visuals_dir" not in latest_run.get("artifacts", {}):
        raise KeyError(
            f"Registry pointer {latest_run['run_dir']} missing required 'visuals_dir' artifact."
        )
    return latest_run


def _rebind_provenance_to_latest(
    db_file: Path, doc_stem: str, latest_run: dict, reason: str
) -> Path:
    set_provenance(
        db_file=db_file,
        source_stage="layout-infer",
        doc_stem=doc_stem,
        source_run_dir=latest_run["run_dir"],
    )
    log.warning(f"FORCED_PROVENANCE_REBIND: {reason}. Rebound to {latest_run['run_dir']}.")
    return Path(latest_run["artifacts"]["visuals_dir"])


def _validate_provenance(db_file: Path, doc_stem: str, force_provenance: bool = False) -> Path:
    latest_run = _latest_layout_infer_pointer(doc_stem)

    prov = get_provenance(db_file)
    if not prov:
        if force_provenance:
            return _rebind_provenance_to_latest(
                db_file=db_file,
                doc_stem=doc_stem,
                latest_run=latest_run,
                reason=(f"Database had no provenance metadata for doc '{doc_stem}'"),
            )
        raise ValueError(
            f"Database {db_file} missing provenance metadata. "
            "Re-initialize DB from Label Studio exports to lock provenance, "
            "or pass --force-provenance to explicitly bind this DB to the current "
            "layout-infer run for the provided --doc-stem."
        )

    if prov["doc_stem"] != doc_stem:
        if force_provenance:
            return _rebind_provenance_to_latest(
                db_file=db_file,
                doc_stem=doc_stem,
                latest_run=latest_run,
                reason=(f"Database doc-stem mismatch ('{prov['doc_stem']}' -> '{doc_stem}')"),
            )
        raise ValueError(
            f"Database provenance doc '{prov['doc_stem']}' does not match requested '{doc_stem}'"
        )

    if prov["source_stage"] != "layout-infer" and not force_provenance:
        raise ValueError(
            f"Database provenance source_stage is '{prov['source_stage']}', expected 'layout-infer'. "
            "Use --force-provenance to rebind to current layout-infer run."
        )
    if prov["source_stage"] != "layout-infer" and force_provenance:
        return _rebind_provenance_to_latest(
            db_file=db_file,
            doc_stem=doc_stem,
            latest_run=latest_run,
            reason=(f"Database source_stage mismatch ('{prov['source_stage']}' -> 'layout-infer')"),
        )

    if prov["source_run_dir"] != latest_run["run_dir"]:
        if force_provenance:
            return _rebind_provenance_to_latest(
                db_file=db_file,
                doc_stem=doc_stem,
                latest_run=latest_run,
                reason=(
                    "Pointer drift overridden "
                    f"('{prov['source_run_dir']}' -> '{latest_run['run_dir']}')"
                ),
            )
        raise ValueError(
            f"Latest pointer drift detected: DB provenance points to {prov['source_run_dir']} "
            f"but registry latest is {latest_run['run_dir']}. "
            "Export aborted to maintain reproducible linkage."
        )

    return Path(latest_run["artifacts"]["visuals_dir"])


def export_dataset(
    doc_stem: str,
    db_file: Path,
    output_dir: Path,
    split_seed: int = 42,
    line_width_px: float = 4.0,
    max_tilt_deg: float = 15.0,
    min_line_length_px: float = 100.0,
    min_train_pages: int = 20,
    min_val_pages: int = 5,
    copy_mode: str = "hardlink",
    dry_run: bool = False,
    force_provenance: bool = False,
    clip_top_px: float = 0.0,
    clip_bottom_px: float = 0.0,
    omit_pages: str | None = None,
) -> dict:
    """Harvest verified rows and convert to YOLOv8-seg folder structure via registry provenance."""
    images_source_dir = _validate_provenance(db_file, doc_stem, force_provenance=force_provenance)

    state = load_state(db_file)
    if not state:
        raise ValueError(f"No state found in {db_file}.")

    omitted = parse_omit_pages(omit_pages)

    candidates, skips = _collect_candidates(
        state,
        images_source_dir,
        max_tilt_deg,
        min_line_length_px,
        clip_top_px,
        clip_bottom_px,
        omitted,
    )

    # Assign splits
    train_pages = []
    val_pages = []
    for candidate in candidates:
        if get_split(candidate[0], split_seed) == "train":
            train_pages.append(candidate)
        else:
            val_pages.append(candidate)

    log.info(
        "Export Scan Summary | "
        f"doc={doc_stem} verified_scanned={len(state)} "
        f"eligible={len(candidates)} skips(unverified={skips['unverified']}, omitted={skips['omitted']}, "
        f"missing_lines={skips['missing_lines']}, invalid_geometry={skips['invalid_geometry']})"
    )
    log.info(
        "Export Split Plan | "
        f"train={len(train_pages)} val={len(val_pages)} "
        f"ratio={len(train_pages)}/{max(len(candidates), 1)}"
    )

    # Floor Checks
    if len(train_pages) < min_train_pages:
        raise ValueError(f"Insufficient train pages: {len(train_pages)} < {min_train_pages}")
    if len(val_pages) < min_val_pages:
        raise ValueError(f"Insufficient val pages: {len(val_pages)} < {min_val_pages}")

    report = {
        "status": "success",
        "total_verified_scanned": len(candidates)
        + skips["missing_lines"]
        + skips["invalid_geometry"],
        "total_exported": len(candidates),
        "train_count": len(train_pages),
        "val_count": len(val_pages),
        "skips": skips,
        "config": {
            "seed": split_seed,
            "line_width_px": line_width_px,
            "max_tilt_deg": max_tilt_deg,
            "min_line_length_px": min_line_length_px,
            "clip_top_px": clip_top_px,
            "clip_bottom_px": clip_bottom_px,
            "omit_pages": sorted(omitted),
        },
    }

    if dry_run:
        return report

    # Create deterministic run directory (Atomicity: cleanup on fail)
    run_dir = next_versioned_dir(output_dir, doc_stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        written_train, written_val = _write_dataset_files(
            data_dir,
            train_pages,
            val_pages,
            line_width_px,
            copy_mode,
            min_train_pages,
            min_val_pages,
        )

        report["total_exported"] = written_train + written_val
        report["train_count"] = written_train
        report["val_count"] = written_val
        report["run_dir"] = str(run_dir)
        (data_dir / "export_report.json").write_text(json.dumps(report, indent=2))
        log.info(
            "Export Write Summary | "
            f"run_dir={run_dir} total_exported={report['total_exported']} "
            f"train={report['train_count']} val={report['val_count']}"
        )

        yaml_content = f"""path: {data_dir.absolute()}
train: images/train
val: images/val

names:
  0: divider_left
  1: divider_right
"""
        dataset_yaml_path = data_dir / "dataset.yaml"
        dataset_yaml_path.write_text(yaml_content)

        # Register the export stage
        register_latest_run(
            stage="layout-finetune-export",
            doc_stem=doc_stem,
            run_dir=run_dir,
            artifacts={
                "dataset_yaml": str(dataset_yaml_path),
                "export_report": str(data_dir / "export_report.json"),
                "provenance": str(db_file),
            },
            metadata={
                "seed": split_seed,
                "line_width_px": line_width_px,
                "max_tilt_deg": max_tilt_deg,
                "min_line_length_px": min_line_length_px,
                "clip_top_px": clip_top_px,
                "clip_bottom_px": clip_bottom_px,
                "omit_pages": sorted(omitted),
                "train_count": written_train,
                "val_count": written_val,
            },
        )
        return report

    except Exception:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        raise
