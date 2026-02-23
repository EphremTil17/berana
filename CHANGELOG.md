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

## [0.3.0] - 2026-02-22
Tag: `v0.3.0` (`30929bb`)

### Added
- **Phase 2 OCR Routing Foundation**: Implemented the Surya-focused OCR routing baseline and iterative CLI integration path used as the hand-off point for later layout and HITL expansion.
- **Iterative Pipeline Entry Points**: Established the precursor command structure that enabled subsequent modular extraction, layout, and verification workflows.

### Changed
- **Pipeline Direction**: Formalized the transition from pure OCR flow toward layout-aware processing by preparing the architecture for dedicated layout and verification layers.

### Fixed
- **Dependency Resolution**: Stabilized key dependency interactions to keep the OCR baseline operational while the next modular phases were introduced.

## [0.4.0] - 2026-02-21
Tag: `v0.4.0` (`c8c085d`)

### Added
- **Layout Core Engine**: Added `modules/ocr_engine/layout/yolo_engine.py` and `modules/ocr_engine/layout/column_engine.py` to establish divider-centric layout analysis for triple-column manuscripts.
- **Layout Mapping Layer**: Added `modules/ocr_engine/layout_mapping.py` to convert raw divider and bounding data into deterministic column assignments.
- **Output Serialization Module**: Added `modules/ocr_engine/output_writer.py` to isolate persistence from orchestration and keep data flow modular.
- **Label Studio Task Adapter**: Added `utils/label_studio_adapter.py` for direct conversion of structured layout outputs into importable Label Studio task JSON.
- **Model Artifact**: Added trained divider model `models/layout/weights/berana_yolov8_divider_v13.pt` to make layout analysis immediately executable.

### Changed
- **Layout Parsing Contracts**: Updated `modules/ocr_engine/layout_parser.py` and `schemas/ocr_models.py` to align strict typed layout structures with downstream OCR and review layers.

## [0.5.0] - 2026-02-22
Tag: `v0.5.0` (`a8fb6b8`)

### Added
- **CLI Command Modules**: Added `modules/cli/common.py`, `modules/cli/layout_commands.py`, `modules/cli/ocr_commands.py`, and `modules/cli/layout_infer_runtime.py` for explicit command separation and runtime control.
- **Runtime Inference Controls**: Added adaptive chunking, cache-aware page reuse, target-page planning, and prefetch iteration for scalable layout inference on long PDFs.
- **Inference Telemetry**: Added progress/coverage/confidence summaries in `layout-infer` to expose operational quality metrics during batch inference.

### Changed
- **Top-Level Orchestration**: Refactored `berana.py` to lazy-load heavy dependencies and reduce CLI startup overhead.
- **PDF Conversion Pipeline**: Enhanced `modules/ocr_engine/pre_processors/pdf_to_image.py` with `start_page` bounds and threaded conversion controls.
- **OCR Routing**: Updated `modules/ocr_engine/orchestrator.py` and `modules/ocr_engine/extractor.py` to improve modular delegation and recognition-line mapping safety.
- **Benchmark Startup Path**: Updated `evals/benchmark_translation.py` import usage to reduce unnecessary startup load.

## [0.6.0] - 2026-02-23
Tag: `v0.6.0` (`3a11788`)

### Added
- **Standalone HITL Web Tool**: Added `tools/hitl_line_editor.py` and the modular app package `tools/hitl_line_editor_app/` (`app.py`, `db.py`, `state.py`, `geometry.py`, `export.py`, `paths.py`, template HTML).
- **OCR-Ready HITL Export**: Added `tools/export_hitl_coordinates.py` to transform verified divider state into explicit OCR coordinate payloads.
- **Labeling Utility Scripts**: Added `tools/ingest_labels.py` and `tools/debug_extractor.py` for dataset ingestion and geometric verification debugging.
- **Label Studio Runtime Assets**: Added `tools/label_studio/docker-compose.yaml`, `tools/label_studio/project_ui.xml`, and `tools/label_studio/setup_label_studio.sh`.
- **Environment Templates**: Added `tools/label_studio/.env.example` and `tools/label_studio/data/.env.example` to support reproducible local setup without committing runtime secrets.

### Changed
- **Setup and Dependencies**: Updated `setup.sh`, `requirements.txt`, `pyproject.toml`, and `.gitignore` to support the HITL tooling and reproducible local operations.

## [0.6.1] - 2026-02-23
Tag: `v0.6.1` (`bfd8ebe`)

### Added
- **Research Layout Report**: Added `docs/research/layout_analysis_report.md` to capture analysis outcomes and implementation decisions for research traceability.
- **HITL Methodology Guide**: Added `docs/research/hitl_methodology.md` with procedural guidance for inference, manual verification, and extraction planning.
- **Label Studio Workflow Guide**: Added `tools/label_studio/README.md` with exact local-files storage setup, JSON import method, and export format requirements.

### Changed
- **Main Project Documentation**: Updated `README.md` with current inference defaults, output locations, and operational instructions.
- **Release Narrative**: Updated `CHANGELOG.md` chronology to align entries with commit-tag history.
