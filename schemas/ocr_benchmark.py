from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config.settings import settings


class BenchmarkStage(StrEnum):
    """Canonical registry stage names for the OCR benchmark pipeline."""

    PREPARE = "ocr-benchmark-prepare"
    COVERAGE_REPORT = "ocr-benchmark-coverage-report"
    COVERAGE_QUEUE = "ocr-benchmark-coverage-queue"
    SURYA_ZERO = "ocr-benchmark-surya-zero"
    SURYA_FINETUNE = "ocr-benchmark-surya-finetune"
    TROCR_FINETUNE = "ocr-benchmark-trocr-finetune"


class DatasetSplit(StrEnum):
    """Supported dataset splits for the benchmark."""

    TRAIN = "train"
    HOLDOUT = "holdout"


class ColumnKey(StrEnum):
    """Supported Ethiopic benchmark language columns."""

    GEEZ = "geez"
    AMHARIC = "amharic"


class LangPrompt(StrEnum):
    """Decoder language prompts for conditioned generation."""

    GEEZ = "<gez>"
    AMHARIC = "<amh>"


class QualityFlag(StrEnum):
    """Data quality routing labels for line crops."""

    OK = "ok"
    MERGED_LINE = "merged_line"
    STRUCTURAL_MARKER = "structural_marker"
    UNCERTAIN = "uncertain"


class LineManifestRow(BaseModel):
    """Pydantic model representing a single row in the JSONL line manifest."""

    schema_version: str = Field(default="1.0", pattern="^1\\.0$")
    line_id: str
    doc_stem: str
    page_id: str
    column_key: ColumnKey
    lang_prompt: LangPrompt
    image_path: str
    split: DatasetSplit
    gt_text: str | None = None
    quality_flag: QualityFlag = Field(default=QualityFlag.OK)
    source_run_dir: str

    model_config = ConfigDict(extra="forbid")

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: str) -> str:
        """Enforce that image_path is strictly project-relative."""
        if Path(v).is_absolute():
            raise ValueError(
                f"image_path MUST be project-relative to ensure portability. Got absolute path: {v}"
            )
        return v

    def get_absolute_image_path(self) -> Path:
        """Resolve the project-relative image path to an absolute Path."""
        return settings.BASE_DIR / self.image_path


class EvalMetrics(BaseModel):
    """Schema for prediction evaluation metrics."""

    cer: float
    wer: float
    exact_match: bool
    raw_pred: str
    normalized_pred: str
    raw_gt: str
    normalized_gt: str

    model_config = ConfigDict(extra="forbid")


class SplitManifest(BaseModel):
    """Schema defining the dataset freeze state."""

    schema_version: str = Field(default="1.0", pattern="^1\\.0$")
    dataset_hash: str
    random_seed: int
    train_count: int
    holdout_count: int

    model_config = ConfigDict(extra="forbid")
