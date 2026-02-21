import os

from schemas.ocr_models import ColumnBlock

# Label Studio host — override via LABEL_STUDIO_HOST env var for non-local deployments.
_LS_HOST = os.getenv("LABEL_STUDIO_HOST", "http://localhost:8080")


def generate_label_studio_task(
    image_filename: str, img_width: int, img_height: int, columns: list[ColumnBlock]
) -> dict:
    """Takes the strict ColumnBlock schemas and formats them into a Label Studio UI Task.

    Label Studio uses absolute pixels relative to percentages for bounding boxes.
    This creates the JSON task required for the `/api/import` endpoint or bulk file import.

    Args:
        image_filename: The local filename (e.g., 'page_001.jpg') hosted in the volume.
        img_width: Absolute width of the original image.
        img_height: Absolute height of the original image.
        columns: Our perfectly sliced Verse Grid columns.

    Returns:
        dict: The serialized JSON Task representation for Label Studio.
    """
    # Label Studio v1.x requires a FULLY-QUALIFIED absolute URL in the task JSON.
    # The browser fetches the image directly; a relative path like /data/local-files/...
    # is not resolved correctly and triggers "There was an issue loading URL from $ocr".
    # The LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT inside the container is /berana_data,
    # so ?d=visuals/<file> resolves to /berana_data/visuals/<file> inside the container.
    image_url = f"/data/local-files/?d=visuals/{image_filename}"

    task_skeleton = {
        "data": {"ocr": image_url},
        "predictions": [{"model_version": "berana-grid-slicer-v1", "result": []}],
    }

    results = task_skeleton["predictions"][0]["result"]
    id_counter = 1

    for block in columns:
        for line in block.lines:
            coords = line.bbox.coordinates

            # Label Studio requires exactly this mathematical conversion structure:
            # x, y, width, height as PERCENTAGES of the original image
            x1 = coords[0]
            y1 = coords[1]
            x2 = coords[2]
            y2 = coords[3]

            width_px = x2 - x1
            height_px = y2 - y1

            results.append(
                {
                    "id": f"bbox_{id_counter}",
                    "type": "rectanglelabels",
                    "value": {
                        "x": (x1 / img_width) * 100,
                        "y": (y1 / img_height) * 100,
                        "width": (width_px / img_width) * 100,
                        "height": (height_px / img_height) * 100,
                        "rotation": 0,
                        "rectanglelabels": [block.language.value.upper()],
                    },
                    "origin": "manual",
                    "to_name": "image",
                    "from_name": "label",
                    "image_rotation": 0,
                    "original_width": img_width,
                    "original_height": img_height,
                }
            )
            id_counter += 1

    return task_skeleton
