import json
from pathlib import Path


def write_json_output(records: list[dict], output_path: Path) -> None:
    """Persist records to JSON with UTF-8 encoding."""
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
