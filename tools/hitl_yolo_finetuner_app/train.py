import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from utils.logger import get_logger
from utils.run_registry import load_latest_run, register_latest_run

log = get_logger("YOLOTrain")


def _compute_native_max_imgsz(dataset_yaml: Path) -> int:
    """Read the source train dataset images and return max dimension bounded to nearest 32px multiple constraint for YOLO."""
    import yaml
    from PIL import Image

    with dataset_yaml.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    train_rel_path = data.get("train", "images/train")
    train_dir = dataset_yaml.parent / train_rel_path

    max_dim = 1024
    if train_dir.exists():
        for img_path in train_dir.glob("*.jpg"):
            with Image.open(img_path) as img:
                max_dim = max(max_dim, *img.size)

    # YOLO requires imgsz to be a multiple of 32
    return ((max_dim + 31) // 32) * 32


def _generate_signature(dataset_yaml: Path, epochs: int, batch: int, imgsz: int, model: str) -> str:
    """Generate a deterministic signature of dataset logic to prevent duplicate work."""
    manifests_dir = dataset_yaml.parent / "manifests"
    data = {"epochs": epochs, "batch": batch, "imgsz": imgsz, "model": model}

    if dataset_yaml.exists():
        data["yaml"] = dataset_yaml.read_text()
    if (manifests_dir / "train_pages.txt").exists():
        data["train_pages"] = (manifests_dir / "train_pages.txt").read_text()
    if (manifests_dir / "val_pages.txt").exists():
        data["val_pages"] = (manifests_dir / "val_pages.txt").read_text()

    hash_input = json.dumps(data, sort_keys=True).encode()
    return hashlib.md5(hash_input).hexdigest()


def _check_duplicate_signature(output_dir: Path, doc_stem: str, signature: str) -> None:
    """Iterate previous finetuner runs for this doc and warn if signature matches."""
    if not output_dir.exists():
        return
    for run_dir in output_dir.glob(f"{doc_stem}_v*"):
        sig_file = run_dir / "meta" / "signature.json"
        if sig_file.exists():
            try:
                prev_sig = json.loads(sig_file.read_text())["signature"]
                if prev_sig == signature:
                    log.warning(
                        f"⚠️ HIGH_SIGNAL_WARNING: Duplicate dataset/config signature detected from previous run: {run_dir.name}"
                    )
            except Exception:
                pass


def _copy_best_weights(tmp_dir: Path, weights_dir: Path) -> Path | None:
    best_pt = tmp_dir / "hitl_finetune" / "weights" / "best.pt"
    if not best_pt.exists():
        return None
    final_weights = weights_dir / "best.pt"
    shutil.copy2(best_pt, final_weights)
    return final_weights


def _resolve_dataset_yaml(latest_export: dict, doc_stem: str) -> Path:
    dataset_yaml_value = latest_export.get("artifacts", {}).get("dataset_yaml")
    if not dataset_yaml_value:
        raise KeyError(
            f"Latest layout-finetune-export pointer for '{doc_stem}' missing 'dataset_yaml' artifact."
        )
    dataset_yaml = Path(dataset_yaml_value)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {dataset_yaml}")
    return dataset_yaml


def _write_signature(meta_dir: Path, signature: str) -> Path:
    signature_path = meta_dir / "signature.json"
    signature_path.write_text(json.dumps({"signature": signature}, indent=2))
    return signature_path


def _run_ultralytics_train(
    *,
    yolo_cls: object,
    model: str,
    dataset_yaml: Path,
    epochs: int,
    batch: int,
    imgsz: int,
    tmp_dir: Path,
) -> Path:
    model_instance = yolo_cls(model)
    log.info(
        "Training config | "
        f"model={model} epochs={epochs} batch={batch} imgsz={imgsz} "
        f"dataset={dataset_yaml}"
    )
    log.info(f"Training artifacts temp dir: {tmp_dir / 'hitl_finetune'}")
    results = model_instance.train(
        data=str(dataset_yaml.absolute()),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(tmp_dir.absolute()),
        name="hitl_finetune",
        exist_ok=False,
    )
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Ultralytics did not produce best.pt at {best_pt}")
    return best_pt


def _register_training_run(
    *,
    doc_stem: str,
    run_dir: Path,
    signature_path: Path,
    final_weights: Path,
    manifest_path: Path,
    signature: str,
    dataset_yaml: Path,
    epochs: int,
    batch: int,
    imgsz: int,
    model: str,
) -> None:
    register_latest_run(
        stage="layout-train",
        doc_stem=doc_stem,
        run_dir=run_dir,
        artifacts={
            "training_manifest": str(manifest_path),
            "weights_best": str(final_weights),
            "signature": str(signature_path),
        },
        metadata={
            "signature": signature,
            "source_export_yaml": str(dataset_yaml),
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "model": model,
        },
    )


def run_yolo_train(
    doc_stem: str,
    epochs: int = 100,
    batch: int = 4,
    imgsz: int | None = None,
    model: str = "yolov8s-seg.pt",
) -> dict:
    """Trigger Ultralytics training asynchronously within the locked closed loop."""
    latest_export = load_latest_run("layout-finetune-export", doc_stem)
    if not latest_export:
        raise ValueError(f"No latest layout-finetune-export run found for '{doc_stem}'.")
    dataset_yaml = _resolve_dataset_yaml(latest_export, doc_stem)

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "Ultralytics is not installed. Install via `pip install ultralytics` to run training."
        ) from e

    if imgsz is None:
        imgsz = _compute_native_max_imgsz(dataset_yaml)
        log.info(f"Dynamically resolved native maximum resolution for imgsz: {imgsz}")

    signature = _generate_signature(dataset_yaml, epochs, batch, imgsz, model)
    _check_duplicate_signature(Path(latest_export["run_dir"]).parent, doc_stem, signature)

    run_dir = Path(latest_export["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = run_dir / "meta"
    weights_dir = run_dir / "weights"
    tmp_dir = run_dir / "artifacts" / "tmp"

    for d in [meta_dir, weights_dir, tmp_dir]:
        d.mkdir(parents=True, exist_ok=True)
    signature_path = _write_signature(meta_dir, signature)

    try:
        best_pt = _run_ultralytics_train(
            yolo_cls=YOLO,
            model=model,
            dataset_yaml=dataset_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            tmp_dir=tmp_dir,
        )
        log.info(f"Best checkpoint produced at {best_pt}")
        final_weights = _copy_best_weights(tmp_dir, weights_dir)
        if final_weights is None:
            raise FileNotFoundError(f"Ultralytics produced best.pt but copy failed from {best_pt}")
        log.info(f"Copied final best checkpoint to {final_weights}")

        manifest_path = meta_dir / "training_manifest.json"
        manifest = {
            "doc_stem": doc_stem,
            "signature": signature,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "model": model,
            "source_export_yaml": str(dataset_yaml),
            "updated_at_utc": datetime.now(UTC).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        _register_training_run(
            doc_stem=doc_stem,
            run_dir=run_dir,
            signature_path=signature_path,
            final_weights=final_weights,
            manifest_path=manifest_path,
            signature=signature,
            dataset_yaml=dataset_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            model=model,
        )

        log.info(f"✅ Training completed. Isolated model at {final_weights}")
        return manifest

    except KeyboardInterrupt:
        final_weights = _copy_best_weights(tmp_dir, weights_dir)
        if final_weights:
            log.warning(
                f"Training interrupted (Ctrl+C). Preserved best-so-far checkpoint at {final_weights}"
            )
        else:
            log.warning("Training interrupted (Ctrl+C) before any best checkpoint was produced.")
        raise

    except Exception:
        # Never delete run_dir here because it is the export-owned directory.
        raise
