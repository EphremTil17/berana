from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for the Berana pipeline.

    Utilizes Pydantic BaseSettings to fail fast and loudly on missing or
    invalid configurations.
    """

    # Project Root Directory Modeled dynamically
    BASE_DIR: Path = Path(__file__).parent.parent

    # Model Configuration
    MODEL_REPO: str = Field(default="bullerwins/translategemma-12b-it-GGUF")
    MODEL_FILE: str = Field(default="translategemma-12b-it-Q8_0.gguf")
    MODELS_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "models")

    # Pipeline I/O Directories
    INPUT_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "input")
    OUTPUT_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "output")
    GLOSSARY_FILE: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "input" / "glossary.json"
    )

    # Data Directories
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    RAW_PDF_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "raw_pdfs"
    )
    PROCESSED_IMAGES_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "processed_images"
    )
    EXTRACTED_JSON_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "extracted_json"
    )

    # Load from environment variables and an optional .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


# Initialize central settings instance
settings = Settings()

# Post-init validation: Create standard data and model dirs if missing
DIRS_TO_CREATE = [
    settings.MODELS_DIR,
    settings.INPUT_DIR,
    settings.OUTPUT_DIR,
    settings.RAW_PDF_DIR,
    settings.PROCESSED_IMAGES_DIR,
    settings.EXTRACTED_JSON_DIR,
]
for _dir in DIRS_TO_CREATE:
    _dir.mkdir(parents=True, exist_ok=True)
