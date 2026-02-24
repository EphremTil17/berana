from pathlib import Path

import numpy as np
from PIL import Image

from utils.logger import get_logger

logger = get_logger("YOLOEngine")


class LayoutSegmentationEngine:
    """Handles YOLOv8-Segmentation inference for column dividers."""

    def __init__(self, model_path: str | Path | None = None):
        """Initialize the YOLO model from explicit path or best local fallback weights."""
        from ultralytics import YOLO

        if model_path is None:
            # First try our tracked, organized model
            organized_model_path = Path("models/layout/weights/berana_yolov8_divider_v13.pt")
            if organized_model_path.exists():
                model_path = organized_model_path
            else:
                # Auto-discovery fallback: Look for local training runs
                weights_globs = list(Path("runs").rglob("**/weights/best.pt"))
                if weights_globs:
                    # Get the one most recently modified
                    weights_globs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    model_path = weights_globs[0]

        if model_path and Path(model_path).exists():
            self.model = YOLO(str(model_path))
            logger.info(f"YOLO Layout Engine initialized with: {model_path}")
        else:
            self.model = None
            logger.warning(
                "YOLO Layout Engine started WITHOUT model weights. Placeholder fallback will be used."
            )

    def predict_dividers(self, img: Image.Image, conf: float = 0.2) -> list[dict]:
        """
        Runs inference to find divider_left and divider_right.
        Returns list of segments/masks.
        """
        if self.model is None:
            return []

        # Run inference (strictly on GPU if available)
        results = self.model(img, conf=conf, verbose=False)[0]

        dividers = []
        if results.masks is not None:
            logger.debug(f"YOLO found {len(results.masks)} candidate masks.")
            for mask, box in zip(results.masks.xy, results.boxes, strict=False):
                cls_name = results.names[int(box.cls)]
                dividers.append(
                    {
                        "class": cls_name,
                        "points": mask.tolist(),  # List of (x, y) coordinates for the polygon
                        "confidence": float(box.conf),
                    }
                )
                logger.debug(f" - Found {cls_name} with conf {float(box.conf):.3f}")
        else:
            logger.debug("YOLO found ZERO masks.")

        return dividers

    def generate_auto_labels(self, img: Image.Image, page_num: int, filename: str) -> dict:
        """
        Generates a Label Studio 'Prediction' format for the image.
        Used for Active Learning (Phase 2).
        """
        # 1. Be highly permissive at the base level to capture faint lines
        raw_dividers = self.predict_dividers(img, conf=0.15)
        img_w, img_h = img.size

        # 2. Dynamic Sort & Filter (NMS-style clustering)
        # Only keep the HIGHEST confidence mask per class
        best_dividers = {}
        for div in raw_dividers:
            cls_name = div["class"]
            # If we don't have this class yet, or this one is higher confidence, keep it.
            if (
                cls_name not in best_dividers
                or div["confidence"] > best_dividers[cls_name]["confidence"]
            ):
                best_dividers[cls_name] = div

        # 3. Convert back to list for processing
        dividers = list(best_dividers.values())

        results = []
        for _i, div in enumerate(dividers):
            # Normalize points to 0-100 for Label Studio
            points = []
            for px, py in div["points"]:
                points.append([(px / img_w) * 100, (py / img_h) * 100])

            results.append(
                {
                    "original_width": img_w,
                    "original_height": img_h,
                    "image_rotation": 0,
                    "value": {"points": points, "polygonlabels": [div["class"]]},
                    "from_name": "label",
                    "to_name": "image",
                    "type": "polygonlabels",
                    "score": div["confidence"],
                }
            )

        # Dynamic Evaluation (Warnings for Manual Review)
        warnings = []
        if len(dividers) > 2:
            warnings.append(f"⚠️ MULTIPLE COLUMNS: Found {len(dividers)} dividers. Verify layout.")
        elif len(dividers) < 2:
            warnings.append(
                f"⚠️ MISSING COLUMNS: Found {len(dividers)} dividers. Draw missing ones."
            )

        avg_conf = sum(div["confidence"] for div in dividers) / len(dividers) if dividers else 0
        if avg_conf < 0.6:
            warnings.append(f"⚠️ LOW CONFIDENCE: Avg model certainty is only {avg_conf:.1%}.")

        status_text = (
            "\\n".join(warnings)
            if warnings
            else "✅ AI Confident: Standard Triple-Column detected."
        )

        return {
            "data": {
                "image": f"/data/local-files/?d=visuals/layout_auto/{filename}",
                "page_num": page_num,
                "status": status_text,
            },
            "annotations": [{"result": results}],
        }


def find_yolo_slices(img: Image.Image, dividers: list[dict]) -> list[int]:
    """Translates raw YOLO masks into vertical slice X-coordinates."""
    _img_w, _ = img.size

    # 1. Group by class
    left_dividers = [d for d in dividers if d["class"] == "divider_left"]
    right_dividers = [d for d in dividers if d["class"] == "divider_right"]

    slices = []

    # Take the median X of the most confident divider_left
    if left_dividers:
        best_left = max(left_dividers, key=lambda x: x["confidence"])
        points = np.array(best_left["points"])
        slices.append(int(np.median(points[:, 0])))

    # Take the median X of the most confident divider_right
    if right_dividers:
        best_right = max(right_dividers, key=lambda x: x["confidence"])
        points = np.array(best_right["points"])
        slices.append(int(np.median(points[:, 0])))

    # Ensure they are sorted
    slices.sort()

    return slices


def export_images_for_labeling(
    pdf_path: Path, output_dir: Path, num_pages: int | None = None, dpi: int = 300
):
    """
    Converts PDF pages to images and saves them to the mapped Docker volume.
    User will connect them via Label Studio's 'Cloud Storage -> Local Files' UI.
    """
    from modules.ocr_engine.pre_processors.pdf_to_image import yield_pdf_pages

    # We use the existing Label Studio volume mount path: output/visuals
    # so that the Docker container can read them natively via its /berana_data mount.
    ls_img_dir = Path("output/visuals/layout_training")
    try:
        ls_img_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            "Cannot write to 'output/visuals/layout_training'. "
            "This path is often owned by root after Docker volume writes. "
            "Fix ownership, then retry, e.g.: "
            "'sudo chown -R $USER:$USER output/visuals output'."
        ) from exc

    if num_pages is None:
        logger.info(f"Exporting all pages from {pdf_path} to {ls_img_dir} for Label Studio...")
    else:
        logger.info(f"Exporting {num_pages} pages to {ls_img_dir} for Label Studio...")

    count = 0
    for page_num, img in yield_pdf_pages(pdf_path, dpi=dpi, end_page=num_pages):
        filename = f"page_{page_num:03d}.jpg"
        dest = ls_img_dir / filename
        img.save(dest, "JPEG", quality=95)
        count += 1

    logger.info(f"Export Complete: {count} images saved to {ls_img_dir}")
