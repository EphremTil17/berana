import random
import shutil
import zipfile
from pathlib import Path

from utils.logger import get_logger

logger = get_logger("IngestLabels")


def _reset_temp_dir(temp_dir: Path) -> None:
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)


def _resolve_source_image(img_name: str, source_images_dirs: list[Path]) -> Path | None:
    for src_dir in source_images_dirs:
        candidate = src_dir / img_name
        if candidate.exists():
            return candidate
    return None


def _copy_split_files(
    split_files: list[Path],
    split_name: str,
    final_ds_dir: Path,
    source_images_dirs: list[Path],
) -> None:
    for label_path in split_files:
        img_name = label_path.stem + ".jpg"
        source_img_path = _resolve_source_image(img_name, source_images_dirs)
        if not source_img_path:
            logger.warning(
                f"Labeled image {img_name} not found in any source dirs: {source_images_dirs}. Skipping."
            )
            continue

        dest_label_path = final_ds_dir / "labels" / split_name / label_path.name
        shutil.copy(label_path, dest_label_path)
        dest_img_path = final_ds_dir / "images" / split_name / img_name
        shutil.copy(source_img_path, dest_img_path)


def _build_dataset_dirs(final_ds_dir: Path) -> None:
    if final_ds_dir.exists():
        shutil.rmtree(final_ds_dir)
    for split in ["train", "val"]:
        (final_ds_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (final_ds_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def ingest_yolo_dataset(
    zip_path: Path, output_dir: Path, source_images_dirs: list[Path], val_split: float = 0.2
):
    """Ingests a raw Label Studio YOLO zip export and formats it perfectly for ultralytics."""
    logger.info(f"Ingesting raw labels from {zip_path}...")

    # Create extraction dir
    temp_dir = output_dir / ".temp_extract"
    _reset_temp_dir(temp_dir)

    # Unpack Zip securely using python
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    labels_dir = temp_dir / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError("Zip file does not contain a 'labels' directory.")

    # Get all .txt labels
    label_files = list(labels_dir.glob("*.txt"))
    logger.info(f"Found {len(label_files)} labeled pages.")

    # Shuffle consistently for reproducibility
    random.seed(42)
    label_files.sort()  # Sort first so shuffle is deterministic
    random.shuffle(label_files)

    num_val = max(1, int(len(label_files) * val_split))
    val_files = label_files[:num_val]
    train_files = label_files[num_val:]

    # Setup ultralytics directories
    final_ds_dir = output_dir / "yolo_dataset"
    _build_dataset_dirs(final_ds_dir)

    logger.info(f"Splitting data into {len(train_files)} train, {len(val_files)} val.")
    _copy_split_files(train_files, "train", final_ds_dir, source_images_dirs)
    _copy_split_files(val_files, "val", final_ds_dir, source_images_dirs)

    # Recover classes from classes.txt or notes.json
    classes_path = temp_dir / "classes.txt"
    classes = ["divider_left", "divider_right"]
    if classes_path.exists():
        classes = [
            line.strip() for line in classes_path.read_text().strip().split("\n") if line.strip()
        ]

    # Generate mapping YAML
    yaml_path = final_ds_dir / "dataset.yaml"

    class_def = "\n".join([f"  {i}: {c}" for i, c in enumerate(classes)])

    yaml_content = f"""# Ultralytics YOLOv8 Configuration
path: {final_ds_dir.absolute()}
train: images/train
val: images/val

names:
{class_def}
"""
    yaml_path.write_text(yaml_content)
    logger.info(f"Generated ultralytics dataset.yaml at {yaml_path}")

    # Cleanup
    shutil.rmtree(temp_dir)
    logger.info("Ingestion completed successfully.")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-images", required=True, type=Path, nargs="+")

    args = parser.parse_args()
    ingest_yolo_dataset(args.zip_file, args.output_dir, args.source_images)
    sys.exit(0)
