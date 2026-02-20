# 📜 Berana-Trans: Liturgical Ge'ez Translation & OCR Pipeline

## 1. Project Abstract
**Berana-Trans** is an advanced, research-grade Machine Learning and Data Engineering pipeline designed to extract, translate, and benchmark low-resource Ethiopian liturgical texts (Ge'ez and Amharic) into formal English.

The primary objective of this repository is to process complex, triple-column PDF manuscripts, digitize them using layout-aware OCR, and prepare "Gold Standard" datasets for benchmarking against Large Language Models (specifically `TranslateGemma-12b-it-GGUF`).

## 2. The Crucial Engineering Decision: OCR Library
We have explicitly selected **Surya OCR (v0.17.1)** over Tesseract.
* **The Problem:** Liturgical PDFs feature dense, triple-column layouts (Ge'ez | Amharic | English). Standard OCR reads horizontally, destroying the semantic pairings.
* **The Solution:** Surya provides state-of-the-art native Layout Analysis and Line-Level Bounding Box extraction. It identifies discrete columns before applying text recognition, ensuring zero cross-column data bleeding.

## 3. Tech Stack & Dependencies (2026 Standards)
* **OCR & Layout Analysis:** `surya-ocr == 0.17.1` (Requires PyTorch)
* **LLM Inference (Local GPU):** `llama-cpp-python == 0.3.16` (Compiled with `GGML_CUDA=on` for RTX 3060 Ti)
* **API / Backend (Future-proofing):** `fastapi == 0.129.0`
* **Image Processing:** `pdf2image`, `opencv-python` (for pre-processing deskew/binarization)
* **Environment:** Python 3.10+, 64GB System RAM, Nvidia RTX 3060 Ti (8GB VRAM).

## 4. Architectural Philosophy: The "Tree-Branch" Pattern
This codebase adheres to a strict **Modular Monolith** architecture.
* **Rule of Modularity:** No monolithic code. Maximum 250 lines per file.
* **Orchestrator Pattern:** Top-level files (e.g., `berana.py`, `modules/ocr_engine/orchestrator.py`) contain **zero** processing logic. They are strictly routers/callers that pass explicitly typed DataClasses between deeper modules.
* **Progressive Complexity:** Logic becomes more verbose and specialized the deeper you traverse into the `/modules` directory.

## 5. Directory Structure
```text
berana/
├── berana.py                   # Top-level orchestrator CLI
├── config/                     # Configuration management
│   └── settings.py             # Global variables, paths, and model configs
├── input/                      # 📥 Volumetric Input (PDFs, text chunks)
│   └── glossary.json           # Liturgical term preservation mapping
├── output/                     # 📤 Pipeline Output (JSON structures, Markdown translations)
├── data/                       # Local data storage (Git ignored)
│   ├── raw_pdfs/               # Original triple-column scans
│   ├── processed_images/       # Binarized, high-res images ready for OCR
│   └── extracted_json/         # Final parsed column data mapped to JSON
├── modules/                    # Core Business Logic
│   ├── ocr_engine/             # --- THE OCR COMPONENT ---
│   │   ├── orchestrator.py     # Component-level orchestrator for OCR pipeline
│   │   ├── pre_processors/     # Image deskewing and cropping
│   │   ├── layout_parser/      # Surya layout analysis (Column isolation)
│   │   └── extractors/         # Surya text recognition mapped to bounding boxes
│   ├── auditor/                # SLM error-flagging component (Future)
│   └── translator/             # TranslateGemma implementation (Future)
└── utils/
    └── logger.py               # Standardized logging

## 6. System Requirements & Setup Guide

This project is built for researchers and developers to accurately reproduce the Ge'ez LLM and OCR pipeline.

### Prerequisites (Linux / WSL2)
1. **NVIDIA GPU** with at least 8GB VRAM (e.g., RTX 3060 Ti).
2. **NVIDIA Drivers** and **CUDA Toolkit** installed.
   To install the CUDA compiler alongside standard drivers on Ubuntu/WSL run:
   ```bash
   sudo apt update && sudo apt install -y nvidia-cuda-toolkit
   ```
2.1 Check your CUDA version and Nvidia-smi output:
   ```bash
   nvidia-smi
   ```
3. **Python 3.10+**.
4. **nvidia-smi** should be visible in your terminal, confirming GPU linkage.

### Installation (Automated Setup)
We have provided a bash script to automatically configure a reproducible python virtual environment, install the correct dependencies, and heavily compile `llama-cpp-python` with `GGML_CUDA=on` to guarantee your hardware acceleration is effectively utilized.

1. Clone this repository.
2. Make the setup script executable and run it:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
3. Once finished, activate the environment:
   ```bash
   source .venv/bin/activate
   ```

## 7. Execution & Orchestration (Volumetric Pipeline)

The project uses a heavily validated Typer CLI. The entry point (`berana.py`) acts as a **logic-free orchestrator**, routing commands to the specialized modules. Ensure your virtual environment is active before running commands.

**View all available sub-commands (Auto-documented):**
```bash
python berana.py --help
```

### The Three Phases of Berana
1. **Ingest (Active Development):** Extracts structural JSON from triple-column PDFs using Surya OCR.
   ```bash
   python berana.py ingest --pdf-path "/path/to/manuscript.pdf"
   ```
2. **Pipeline (Future):** The "Gold Standard" execution. Chains Ingest ➡️ Translation.
   ```bash
   python berana.py pipeline
   ```
3. **Benchmark (Functional Simulator):** Tests the LLM's liturgical translation fidelity by pulling raw text from the `input/` directory, resolving the `glossary.json`, and exporting markdown to the `output/` directory.
   ```bash
   python berana.py benchmark-translation --temp 0.0 --ctx 8192
   ```
