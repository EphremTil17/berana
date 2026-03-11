from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceRepo(StrEnum):
    """Canonical upstream source repository identifiers."""

    FIDEL_DATASET = "fidel_dataset"
    FIDEL_SYNTHETIC = "fidel_synthetic"


class SourceSplit(StrEnum):
    """Source-level split labels from upstream datasets."""

    TRAIN = "train"
    TEST = "test"
    SYNTHETIC = "synthetic"


class NormalizedType(StrEnum):
    """Normalized text-rendering type used by training pipelines."""

    TYPED = "typed"
    SYNTHETIC = "synthetic"
    HANDWRITTEN = "handwritten"


class DatasetSplit(StrEnum):
    """Model development split labels."""

    TRAIN = "train"
    VAL = "val"
    HOLDOUT = "holdout"


class TrainMode(StrEnum):
    """Supported training planner modes."""

    AUTO = "auto"
    MANUAL = "manual"


class ExecutionBackend(StrEnum):
    """Supported single-node execution backends."""

    AUTO = "auto"
    SINGLE = "single"
    DDP = "ddp"


class FinetuneStrategy(StrEnum):
    """Supported Surya finetuning strategies."""

    QLORA = "qlora"
    LORA = "lora"
    FULL = "full"


class CandidateStatus(StrEnum):
    """Candidate benchmark lifecycle states."""

    COMPLETED = "completed"
    OOM = "oom"
    VRAM_GUARD = "vram_guard"
    INVALID = "invalid"
    ERROR = "error"


class SourceSnapshotRow(BaseModel):
    """Single normalized record emitted by the FIDEL extraction stage."""

    schema_version: str = Field(default="1.0", pattern=r"^1\.0$")
    sample_id: str
    source_repo: SourceRepo
    source_split: SourceSplit
    original_filename: str
    normalized_type: NormalizedType
    text_raw: str
    text_normalized: str
    image_relpath: str | None = None
    excluded: bool = False
    excluded_reason: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("image_relpath")
    @classmethod
    def validate_image_relpath(cls, value: str | None) -> str | None:
        """Ensure persisted image paths remain relative for portability."""
        if value is None:
            return None
        if Path(value).is_absolute():
            raise ValueError("image_relpath must be relative to project root.")
        return value


class SplitConfig(BaseModel):
    """Deterministic split allocation configuration."""

    train_ratio: float = 0.80
    val_ratio: float = 0.10
    holdout_ratio: float = 0.10
    seed: int = 42
    strict_page_isolation: bool = False

    model_config = ConfigDict(extra="forbid")

    @field_validator("holdout_ratio")
    @classmethod
    def validate_ratios(cls, _value: float, info):
        """Validate that split ratios sum to 1.0 exactly within tolerance."""
        values = info.data
        train_ratio = float(values.get("train_ratio", 0.0))
        val_ratio = float(values.get("val_ratio", 0.0))
        holdout_ratio = float(_value)
        total = train_ratio + val_ratio + holdout_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "Split ratios must sum to 1.0. "
                f"Got train={train_ratio}, val={val_ratio}, holdout={holdout_ratio}."
            )
        return _value


class SuryaTrainConfig(BaseModel):
    """Surya finetuning runtime configuration."""

    mode: TrainMode = TrainMode.AUTO
    execution_backend: ExecutionBackend = ExecutionBackend.AUTO
    ddp_backend: str = "nccl"
    distributed_world_size: int = 1
    seed: int = 42
    train_fraction: float = 1.0
    eval_fraction: float = 1.0
    eval_max_rows: int | None = None
    planning_budget_minutes: int = 3
    target_vram_utilization: float = 0.9375
    strategy_allowlist: list[FinetuneStrategy] = Field(
        default_factory=lambda: [FinetuneStrategy.QLORA, FinetuneStrategy.LORA]
    )
    throughput_metric: str = "samples_per_second"
    warmup_steps_per_candidate: int = 10
    measure_steps_per_candidate: int = 20
    max_replans: int = 1
    finetune_strategy: FinetuneStrategy | None = None
    per_device_train_batch_size: int | None = None
    per_device_eval_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    dataloader_num_workers: int | None = None
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 2
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    fp16: bool = True
    gradient_checkpointing: bool = True
    max_sequence_length: int | None = None
    num_train_epochs: int = 8
    learning_rate: float = 2e-5
    eval_steps: int | None = None
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 4
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "cer"
    greater_is_better: bool = False
    resume: str = "auto"
    verbose_epochs: bool = True
    foreign_vram_threshold_ratio: float = 0.10
    abort_vram_usage_ratio: float = 0.97
    allow_ram_spillover: bool = True

    model_config = ConfigDict(extra="forbid")

    @field_validator("strategy_allowlist", mode="before")
    @classmethod
    def normalize_strategy_allowlist(cls, value):
        """Normalize strategy allowlists from strings or enum values."""
        if value is None:
            return [FinetuneStrategy.QLORA, FinetuneStrategy.LORA]
        if isinstance(value, str):
            parts = [part.strip().lower() for part in value.split(",") if part.strip()]
            return [FinetuneStrategy(part) for part in parts]
        return [
            item if isinstance(item, FinetuneStrategy) else FinetuneStrategy(str(item).lower())
            for item in value
        ]

    @field_validator("throughput_metric")
    @classmethod
    def validate_throughput_metric(cls, value: str) -> str:
        """Validate the currently supported throughput scoring metric."""
        metric = value.strip().lower()
        if metric != "samples_per_second":
            raise ValueError("Only `samples_per_second` throughput scoring is currently supported.")
        return metric

    @field_validator("train_fraction")
    @classmethod
    def validate_train_fraction(cls, value: float) -> float:
        """Require a positive train fraction no greater than 1.0."""
        normalized = float(value)
        if normalized <= 0.0 or normalized > 1.0:
            raise ValueError("train_fraction must be > 0.0 and <= 1.0.")
        return normalized

    @field_validator("eval_fraction")
    @classmethod
    def validate_eval_fraction(cls, value: float) -> float:
        """Require a positive evaluation fraction no greater than 1.0."""
        normalized = float(value)
        if normalized <= 0.0 or normalized > 1.0:
            raise ValueError("eval_fraction must be > 0.0 and <= 1.0.")
        return normalized

    @field_validator("eval_max_rows")
    @classmethod
    def validate_eval_max_rows(cls, value: int | None) -> int | None:
        """Require eval_max_rows to be positive when provided."""
        if value is not None and int(value) < 1:
            raise ValueError("eval_max_rows must be >= 1 when provided.")
        return value


class HardwareProfile(BaseModel):
    """Normalized single-host hardware profile used by the adaptive planner."""

    schema_version: str = Field(default="1.0", pattern=r"^1\.0$")
    device_type: str
    cuda_device_count: int
    execution_backend: str | None = None
    distributed_world_size: int = 1
    gpu_index: int | None = None
    gpu_name: str | None = None
    gpu_uuid: str | None = None
    total_vram_mb: int | None = None
    used_vram_mb: int | None = None
    free_vram_mb: int | None = None
    compute_capability: str | None = None
    supports_fp16: bool = False
    supports_bf16: bool = False
    cpu_count: int = 1
    system_ram_mb: int | None = None
    output_root: str | None = None
    foreign_processes: list[dict[str, str | int]] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class TrainingCandidate(BaseModel):
    """Concrete trainable candidate produced by manual materialization or auto planning."""

    schema_version: str = Field(default="1.0", pattern=r"^1\.0$")
    candidate_id: str
    execution_backend: ExecutionBackend = ExecutionBackend.SINGLE
    world_size: int = 1
    effective_global_batch_size: int = 1
    finetune_strategy: FinetuneStrategy
    per_device_train_batch_size: int
    per_device_eval_batch_size: int | None = None
    gradient_accumulation_steps: int
    dataloader_num_workers: int
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 2
    fp16: bool = True
    gradient_checkpointing: bool = True
    max_sequence_length: int = 1024
    num_train_epochs: float = 8
    learning_rate: float = 2e-5
    eval_steps: int | None = None
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 4
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "cer"
    greater_is_better: bool = False
    verbose_epochs: bool = True
    foreign_vram_threshold_ratio: float = 0.10
    abort_vram_usage_ratio: float = 0.9375
    allow_ram_spillover: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    planning_notes: list[str] = Field(default_factory=list)
    expected_samples_per_second: float | None = None

    model_config = ConfigDict(extra="forbid")


class CandidateResult(BaseModel):
    """Measured benchmark result for one candidate."""

    schema_version: str = Field(default="1.0", pattern=r"^1\.0$")
    candidate_id: str
    execution_backend: str | None = None
    world_size: int = 1
    effective_global_batch_size: int | None = None
    status: CandidateStatus
    effective_samples_per_second: float | None = None
    optimizer_step_seconds: float | None = None
    peak_vram_mb: int | None = None
    average_loss: float | None = None
    invalid_gradients: bool = False
    reason: str | None = None
    measured_steps: int = 0
    warmup_steps: int = 0

    model_config = ConfigDict(extra="forbid")


class ExtractionSummary(BaseModel):
    """Extraction stage summary report payload."""

    schema_version: str = Field(default="1.0", pattern=r"^1\.0$")
    expected_included: int
    extracted_new: int
    extracted_existing: int
    missing_expected: int
    missing_rate: float
    unknown_archive_entries: int
    skipped_macosx_entries: int
    included_type_counts: dict[str, int]
    excluded_type_counts: dict[str, int]

    model_config = ConfigDict(extra="forbid")
