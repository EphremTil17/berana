from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class OCRTask:
    """One OCR inference task derived from crop-columns artifacts."""

    doc_stem: str
    pdf_path: Path
    page_id: str
    page_number: int
    language: str
    image_path: Path
    source_page: int
    ordering_index: int
    crop_run_dir: Path


@dataclass(slots=True)
class OCRLine:
    """One OCR line emitted by Surya in reading order."""

    line_index: int
    text: str
    bbox: list[float]
    polygon: list[list[float]]
    confidence: float | None


@dataclass(slots=True)
class OCRPrediction:
    """One OCR prediction enriched with provenance metadata."""

    doc_stem: str
    pdf_path: str
    page_id: str
    page_number: int
    language: str
    image_path: str
    source_page: int
    ordering_index: int
    model_mode: str
    checkpoint_dir: str | None
    recognized_text: str
    confidence: float | None


@dataclass(slots=True)
class PredictorBundle:
    """Loaded OCR predictor plus structured model metadata."""

    predictor: Any
    model_info: dict[str, Any]


@dataclass(slots=True)
class SourceArtifacts:
    """Upstream artifact references used to build one inference run."""

    crop_run_dir: Path
    cropping_manifest: Path
    spliced_dir: Path
    crop_registry_pointer: dict[str, Any]
