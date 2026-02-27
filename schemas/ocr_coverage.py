from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CoverageTier(StrEnum):
    """Tier bucket used by coverage policy thresholds."""

    HIGH = "high"
    MEDIUM = "medium"
    RARE = "rare"
    OPTIONAL = "optional"


class CharsetTierConfig(BaseModel):
    """Threshold definition for a single coverage tier."""

    min_count: int = Field(ge=0)
    chars: list[str]

    model_config = ConfigDict(extra="forbid")

    @field_validator("chars")
    @classmethod
    def validate_chars(cls, chars: list[str]) -> list[str]:
        """Validate tier character list integrity."""
        if len(chars) != len(set(chars)):
            raise ValueError("Tier chars contain duplicates.")
        for ch in chars:
            if len(ch) != 1:
                raise ValueError("Each charset entry must be a single Unicode character.")
        return chars


class EthiopicFamilyForm(BaseModel):
    """Single order form in an Ethiopic family."""

    order: int = Field(ge=0, le=7)
    char: str
    codepoint: str

    model_config = ConfigDict(extra="forbid")


class EthiopicFamily(BaseModel):
    """Grouped Ethiopic family with ordered forms."""

    family_base_char: str
    family_base_codepoint: str
    forms: list[EthiopicFamilyForm]

    model_config = ConfigDict(extra="forbid")


class EthiopicCharsetConfig(BaseModel):
    """Canonical charset declaration/policy used by coverage tooling."""

    schema_version: Literal["1.0"] = "1.0"
    name: str
    description: str
    allowed_scripts: list[str]
    unicode_blocks: list[str] = []
    tiers: dict[CoverageTier, CharsetTierConfig]
    ignored_chars: list[str] = []
    normalization_profile: str = "ethiopic_v1"
    # Compact family storage: base codepoint -> 8-slot form string.
    # Example: "1200": "ሀሁሂሃሄህሆሇ"
    family_grid: dict[str, str] = {}
    families: list["EthiopicFamily"] = []
    order_labels: dict[str, str] = {
        "0": "order_0",
        "1": "order_1",
        "2": "order_2",
        "3": "order_3",
        "4": "order_4",
        "5": "order_5",
        "6": "order_6",
        "7": "order_7",
    }

    model_config = ConfigDict(extra="forbid")

    @field_validator("ignored_chars")
    @classmethod
    def validate_ignored_chars(cls, chars: list[str]) -> list[str]:
        """Validate ignored character list integrity."""
        if len(chars) != len(set(chars)):
            raise ValueError("ignored_chars contains duplicates.")
        for ch in chars:
            if len(ch) != 1:
                raise ValueError("ignored_chars entries must be single Unicode characters.")
        return chars

    @field_validator("family_grid")
    @classmethod
    def validate_family_grid(cls, grid: dict[str, str]) -> dict[str, str]:
        """Validate compact family-grid declaration encoding."""
        for key, value in grid.items():
            if len(key) != 4:
                raise ValueError("family_grid keys must be 4-hex codepoint strings (e.g., '1200').")
            int(key, 16)  # raises ValueError if not hex
            if len(value) != 8:
                raise ValueError(
                    "family_grid values must be 8-char form strings (use '_' for missing slots)."
                )
        return grid


class CoverageDeficit(BaseModel):
    """Deficit entry for a single character against its tier threshold."""

    tier: CoverageTier
    char: str
    count: int
    min_required: int
    deficit: int

    model_config = ConfigDict(extra="forbid")


class CoverageReport(BaseModel):
    """Structured output for benchmark coverage analysis."""

    schema_version: Literal["1.0"] = "1.0"
    doc_stem: str
    manifest_hash: str
    charset_config_hash: str
    coverage_status: bool
    split_stats: dict[str, dict[str, int]]
    missing_chars: list[str]
    under_threshold: list[CoverageDeficit]
    recommendations: list[str]

    model_config = ConfigDict(extra="forbid")


class QueueItem(BaseModel):
    """Candidate line recommendation for annotation queueing."""

    schema_version: Literal["1.0"] = "1.0"
    line_id: str
    image_path: str
    column_key: str
    pred_text: str | None = None
    score: float
    target_chars_hit: list[str]
    confidence: float | None = None
    reasons: list[str]

    model_config = ConfigDict(extra="forbid")
