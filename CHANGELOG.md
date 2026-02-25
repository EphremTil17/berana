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

## [0.7.0] - 2026-02-23
Tag: `v0.7.0` (implied)

### Added
- **Geometric Splicing Engine**: Engineered a physics-verified "Geometry-First" cropping system in `modules/ocr_engine/pre_processors/splicing/` utilizing Homography and Rotation for sub-pixel deskewing.
- **Precision Extraction Pipeline**: Implemented the first industrial-grade extraction path that consumes human-verified HITL coordinates to produce vertically-aligned language strips.
- **Inverse Coordinate Remapping**: Developed a reversible transformation engine to map OCR results from rectified "strip-space" back to non-linear "page-space" coordinates.
- **Run Registry & Stage Chaining**: Introduced `utils/run_registry.py` for immutable run histories (`doc_vNN`) and `.registry` pointers, enabling automatic upstream artifact resolution.
- **High-Resolution Pipeline Verification**: Added a comprehensive test suite including `test_splicing_geometry.py`, `test_splicing_source_resolution.py`, and `test_run_registry.py`.
- **Pipeline Activity Scaffolding**: Added `ocr` and `ocr-train` commands with standardized pagination and manifest-driven persistence.

### Changed
- **Modular Orchestration Facade**: Refactored `orchestrator.py` into a thin facade with lazy function-level routing to specialized modules in `modules/ocr_engine/pipelines/`.
- **CLI Signal-to-Noise Optimization**: Replaced per-page log spam with descriptive `tqdm` progress bars and concise post-run quality summaries.
- **Command Nomenclature Refinement**: Renamed `extract-text` to `crop-columns` to accurately reflect its geometric-pass responsibilities in the multi-stage pipeline.
- **Deterministic Source Discovery**: Implemented auto-discovery in `SplicingEngine` with strict precedence (SQLite Preferred > JSON Fallback).

### Fixed
- **Pathing Robustness**: Hardened registry and output pathing to use absolute project-root detection, ensuring pipeline stability across varying execution environments.
- **Memory Management (Rule 5)**: Optimized the PDF ingestion loop to ensure VRAM/RAM stability during high-resolution Homography calculations.

## [0.8.0] - 2026-02-24

### Added
- **Canonical Diagnostics Stage**: Promoted `layout-diagnostics` as the single visual diagnostics workflow and replaced legacy PoC naming with `modules/ocr_engine/pipelines/diagnostics.py`.
- **UI Confidence Banner**: Added explicit low-confidence rule display in `tools/label_studio/project_ui.xml`.
- **Session Handoff Chronicle**: Appended a detailed 2026-02-24 engineering handoff in `.git_exclude/project_chronicle.md`.

### Changed
- **Input Contract Migration**: Migrated source assets from `data/*` to `input/*` (`input/raw_pdfs`, `input/layout_dataset`) and updated all code defaults/docs to match.
- **Label Studio Path Contract**: Standardized task image URLs to output-root-relative local-files references and aligned storage documentation in `tools/label_studio/README.md`.
- **CLI Surface Cleanup**: Removed `ingest` and old `poc-slicer` command exposure from `berana.py`; retained thin orchestration through modern command modules.
- **Pipeline Boundaries**: Removed overlapping OCR-smoke/ingest execution path and consolidated diagnostics/cropping responsibilities into dedicated modules.
- **Threading Throughput**: Increased default PDF render worker allocation in `modules/ocr_engine/pre_processors/pdf_to_image.py` for improved chunk conversion throughput.
- **Confidence Policy**: Lowered low-confidence threshold from 0.60 to 0.30 across layout inference warnings and summaries.
- **Documentation Scope**: Simplified top-level README operational detail and delegated step-by-step Label Studio operations to the dedicated secondary README.

### Fixed
- **HITL Geometry Scalar Crash**: Fixed `cv2.fitLine` scalar conversion in `tools/hitl_line_editor_app/geometry.py` by flattening OpenCV `(4,1)` vectors before casting.
- **Label Studio XML Parsing Error**: Escaped comparison symbol in `project_ui.xml` (`&lt;`) to avoid setup parse failures.
- **Label Studio Image Import Resolution**: Resolved repeated `$image` loading failures by enforcing deterministic local-files URL semantics and matching storage guidance.
- **HITL DB Migration Integrity**: Corrected nested `layout_dataset` move side effect and restored canonical verified DB state at `input/layout_dataset/hitl_line_editor.sqlite3`.

### Removed
- **Obsolete Ingest Pipeline**: Deleted `modules/ocr_engine/pipelines/ingest.py` after diagnostics/cropping decoupling.
- **Unused Label Studio Adapter**: Removed `utils/label_studio_adapter.py` after eliminating dead callsites.

## [0.8.1] - 2026-02-25

### Changed
- **Registry Hardening**: Upgraded `utils/run_registry.py` with atomic latest-pointer writes (tmp file + flush + `fsync` + atomic replace), strict `"schema_version": "1.0"` enforcement, and loud corruption failures via `RegistryCorruptionError`.
- **CLI Runtime Standardization**: Introduced `modules/cli/runtime.py` and refactored OCR CLI commands to use a shared `execute_pipeline` wrapper for consistent validation, error handling, and exit semantics.
- **OCR Scaffold Manifest Contract**: Standardized OCR inference scaffold mode metadata to canonical `"ocr"` naming in `modules/ocr_engine/pipelines/inference.py`.
- **Documentation Synchronization**: Updated `README.md` and release notes to remove `ocr-infer` compatibility references and align command surface with current runtime.

### Added
- **Validation Coverage**: Added `tests/test_cli_runtime.py` for standardized CLI failure exit handling and `tests/test_orchestration_chaining.py` for registry-based stage chaining invariants.
- **Registry Corruption Assertions**: Expanded `tests/test_run_registry.py` to validate malformed JSON, invalid payload shape, missing required keys, and unsupported schema versions.

### Removed
- **Legacy Phase 2 OCR Modules**: Deleted obsolete `modules/ocr_engine/layout_parser.py`, `modules/ocr_engine/layout_mapping.py`, and `modules/ocr_engine/extractor.py` no longer used by the active staged pipeline.
- **Deprecated CLI Alias**: Removed `ocr-infer` command exposure from `berana.py` and `modules/cli/ocr_commands.py` to maintain a single canonical OCR command path.
