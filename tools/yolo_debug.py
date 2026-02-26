import hashlib
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_checkpoint(path: Path):
    # PyTorch 2.6+ defaults to weights_only=True, which breaks many YOLO checkpoints.
    # For local trusted checkpoints, fall back to full pickle load.
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def get_yolo_info(path: str | Path):
    """Return key metadata and fingerprint fields for a YOLO checkpoint."""
    p = Path(path)
    ckpt = _load_checkpoint(p)

    train_args = ckpt.get("train_args", {}) if isinstance(ckpt, dict) else {}
    info = {
        "Path": str(p.resolve()),
        "Exists": p.exists(),
        "File Size (MB)": round(p.stat().st_size / (1024 * 1024), 2),
        "SHA256": _sha256(p)[:16],
        "Epochs": ckpt.get("epoch", "N/A"),
        "Date": ckpt.get("date", "N/A"),
        "Best Fitness": ckpt.get("fitness", "N/A"),
        "Image Size": train_args.get("imgsz", "N/A"),
        "Optimizer": train_args.get("optimizer", "N/A"),
        "Model": train_args.get("model", "N/A"),
    }

    return info


current_path = Path("models/layout/weights/berana_yolov8_divider_v13.pt")

print(f"--- Metadata for {current_path} ---")
print(get_yolo_info(current_path))
