# Changelog

All notable changes to the **Berana** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-19

### Added
- **Environment Orchestrator**: Implemented `setup.sh`, a dynamic bash script for isolated, CUDA-accelerated virtual environment creation.
- **Quality Enforcement**: Integrated `ruff` in `pyproject.toml` with "Structuralist" standards (McCabe complexity = 10, Google docstrings, Pathlib enforcement).
- **Git Quality Hooks**: Established `.pre-commit-config.yaml` to automate linting and formatting before every commit.
- **Dependency Manifest**: Created `requirements.txt` synchronized with `surya-ocr` 0.17.1 internal pins and modern 2026 stable baselines.
- **Configuration Layer**: Integrated Pydantic `BaseSettings` in `config/settings.py` for strictly-typed, fail-fast environment management.
- **CLI Orchestrator**: Established Typer-based entry point in `berana.py` for automated pipeline routing.
- **Benchmarking Engine**: Added `evals/benchmark_translation.py` for assessing LLM liturgical translation fidelity.
- **Technical Documentation**: Produced `RESEARCH_REPORT.md` justifying Surya-OCR selection and CUDA 12.x target architecture.

### Changed
- **Structural Refactoring**: Migrated entire codebase ('evals', 'config', 'main') to pure `pathlib` implementation, removing legacy `os` pathing.
- **Nomenclature Synchronization**: Standardized project branding to **Berana** across all source files, configurations, and documentation.
- **Pathing Logic**: Refactored setup logic to use dynamic project root detection and absolute-relative executable paths (Rule 10 Compliance).

### Fixed
- **Logger Reliability**: Patched `utils/logger.py` to fix docstring inconsistencies and ensure `ClassVar` safety for mutable class attributes.

### Security
- **Environment Isolation**: Enforced strict virtual environment boundaries to prevent system-level Python package contamination.
## [0.2.0] - 2026-02-20

### Added
- **Geometric Data Contracts**: Implemented `schemas/ocr_models.py` with strict Pydantic v2 models for `BoundingBox`, `TextLine`, and `ColumnBlock`.
- **Automated Language Identification**: Engineered a geometric clustering algorithm that automatically tags text columns as Ge'ez, Amharic, or English based on relative horizontal coordinates.
- **Memory-Safe PDF Ingestion**: Implemented `modules/ocr_engine/pre_processors/pdf_to_image.py` utilizing a Python generator pattern to chunk large PDFs (50 pages/chunk), strictly adhering to the 8GB VRAM/64GB RAM constraints.
- **Validation Suite**: Established `tests/test_ocr_schemas.py` and `tests/test_pdf_preprocessor.py` for comprehensive unit testing of the ingestion logic.

### Changed
- **Architectural Refinement**: Formalized the "Typed Schema" hand-off pattern between the future OCR engine and Translator modules to maintain structuralist decoupling.
- **Enumerated Typings**: Migrated `LanguageTag` to use `enum.StrEnum` for improved runtime performance and serialization reliability.

### Fixed
- **Docstring Compliance**: Manually resolved 37+ `ruff` linting violations related to summary line spacing (D205) and multi-line summary formatting (D212) across the new module implementations.
- **Mock Scope resolution**: Fixed Pytest path resolution issues in `test_pdf_preprocessor.py` by scoping patches to the internal `pdf2image` implementation rather than the local wrapper.
