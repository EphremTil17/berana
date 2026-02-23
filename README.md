# 📜 Berana: Liturgical Ge'ez Translation & OCR Pipeline

## 1. Project Abstract
**Berana-Trans** is an advanced, research-grade Machine Learning and Data Engineering pipeline designed to extract, translate, and benchmark low-resource Ethiopian liturgical texts (Ge'ez and Amharic) into formal English.

The primary objective of this repository is to process complex, triple-column PDF manuscripts, digitize them using layout-aware OCR, and prepare "Gold Standard" datasets for benchmarking against Large Language Models (specifically `TranslateGemma-12b-it-GGUF`).

## 2. The Crucial Engineering Decision: OCR Library
We have explicitly selected **Surya OCR (v0.17.1)** over Tesseract.
* **The Problem:** Liturgical PDFs feature dense, triple-column layouts (Ge'ez | Amharic | English). Standard OCR reads horizontally, destroying the semantic pairings.
* **The Solution:** Surya provides state-of-the-art native Layout Analysis and Line-Level Bounding Box extraction. It identifies discrete columns before applying text recognition, ensuring zero cross-column data bleeding.

## 3. Tech Stack & Dependencies (Dev Environment)
* **OCR & Layout Analysis:** `surya-ocr >= 0.17.1` (Class-based Predictor API)
* **GPU Compute:** PyTorch `2.10.0+cu130` (CUDA 13.0 native)
* **Vision Transformers:** `transformers >= 4.48.0, < 5.0.0` (Surya incompatible with 5.x)
* **LLM Inference (Local GPU):** `llama-cpp-python == 0.3.16` (Compiled with `GGML_CUDA=on`)
* **API / Backend (Future-proofing):** `fastapi >= 0.129.0`
* **Image Processing:** `pdf2image`, `opencv-python-headless` (for pre-processing deskew/binarization)
* **Environment:** Python 3.10+, 64GB System RAM, NVIDIA RTX 3060 Ti GPU (8GB-VRAM), CUDA 13.0 Toolkit.

### Known Dependency Constraints
| Package | Constraint | Reason |
|---------|-----------|--------|
| `torch` | `==2.10.0+cu130` | Must match system CUDA 13.0 toolkit. Install via PyTorch cu130 index. |
| `transformers` | `>=4.48.0,<5.0.0` | Surya 0.17.x crashes on transformers 5.x (`SuryaDecoderConfig.pad_token_id` removed). |
| `huggingface-hub` | `>=0.36.0,<1.0.0` | Required by transformers 4.x; the 1.x series is only compatible with transformers 5.x. |
| `poppler-utils` | System package | Required by `pdf2image` for PDF page counting. Install: `sudo apt install poppler-utils` |

## 4. Directory Structure

### Folder Modularity and Architecture

```text
berana/
├── config/                     # Typed runtime configuration and environment settings
├── docs/
│   └── research/               # Methodology notes, layout analysis, and research artifacts
├── evals/                      # Translation benchmark runners and evaluation support
├── input/                      # Source text assets, glossary, and user-provided inputs
├── models/
│   └── layout/
│       └── weights/            # Trained divider model weights for layout inference
├── modules/                    # Core modular business logic (thin CLI/orchestrator delegates here)
│   ├── cli/                    # Command routing, shared CLI utilities, and runtime planning
│   └── ocr_engine/
│       ├── layout/             # Divider detection and column geometry engines
│       └── pre_processors/     # PDF/image preparation and memory-safe preprocessing
├── schemas/                    # Typed data contracts shared across pipeline stages
├── tests/                      # Unit tests for schemas, preprocessors, and pipeline behavior
├── tools/                      # Standalone operational tools (HITL, export, debug, dataset ingestion)
│   ├── hitl_line_editor_app/
│   │   └── templates/          # Web UI templates for human-in-the-loop verification
│   └── label_studio/           # Label Studio stack, project UI template, and workflow docs
└── utils/                      # Cross-cutting helpers (logging, adapters, shared helpers)
```

## 5. System Requirements & Setup Guide

This project is built for researchers and developers to accurately reproduce the Ge'ez LLM and OCR pipeline.

### Prerequisites (Linux / WSL2)
1. **NVIDIA GPU** with at least 8GB VRAM (e.g., RTX 3060 Ti).
2. **NVIDIA Driver** 580.76+ and **CUDA Toolkit 13.0**:
   ```bash
   # Install CUDA 13.0 Toolkit on Ubuntu 24.04
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get install -y cuda-toolkit-13-0
   ```
3. **System Dependencies:**
   ```bash
   sudo apt install -y poppler-utils
   ```
4. **Python 3.10+**.
5. Verify GPU linkage:
   ```bash
   nvidia-smi           # Should show Driver 580.97+ and CUDA 13.0
   /usr/local/cuda/bin/nvcc --version   # Should show release 13.0
   ```

### Installation (Automated Setup)
We have provided a bash script to automatically configure a reproducible python virtual environment, install the correct dependencies, and heavily compile `llama-cpp-python` with `GGML_CUDA=on` to guarantee your hardware acceleration is effectively utilized.

1. Clone this repository.
2. Make the setup script executable and run it:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
3. **Install PyTorch with CUDA 13.0** (must use the official PyTorch index):
   ```bash
   source .venv/bin/activate
   pip install torch==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130
   ```

## 6. Execution & Orchestration (Volumetric Pipeline)

The project uses a heavily validated Typer CLI. The entry point (`berana.py`) acts as a **logic-free orchestrator**, routing commands to the specialized modules. Ensure your virtual environment is active before running commands.

**View all available sub-commands (Auto-documented):**
```bash
python berana.py --help
```

### Sub-Command Help
Each command has its own isolated help menu:
```bash
python berana.py ingest --help
python berana.py benchmark-translation --help
```

### The Three Phases of Berana
1. **Ingest (Active Development):** Extracts structural JSON from triple-column PDFs using Surya OCR.
   ```bash
   # Full ingest
   python berana.py ingest --pdf-path "data/raw_pdfs/manuscript.pdf"

   # Grid-slice only (exports Label Studio JSON for visual verification)
   python berana.py ingest \
       --pdf-path "data/raw_pdfs/manuscript.pdf" \
       --slice-only \
       --max-pages 50 \
       --omit-pages "1-8"
   ```
   **CLI Flags:**
   | Flag | Type | Description |
   |------|------|-------------|
   | `--pdf-path` | TEXT (required) | Path to the source liturgical PDF |
   | `--chunk-size` | INT (default: 50) | Pages loaded into RAM per batch |
   | `--dpi` | INT (default: 300) | Image processing resolution |
   | `--slice-only` | FLAG | Skip OCR, export Label Studio visualization JSON |
   | `--omit-pages` | TEXT | Pages to skip. Supports ranges: `"1,2,5-8"` |
   | `--max-pages` | INT | Process only N pages (useful for testing) |

2. **Pipeline (Future):** The "Gold Standard" execution. Chains Ingest ➡️ Translation.
   ```bash
   python berana.py pipeline
   ```
3. **Benchmark (Functional Simulator):** Tests the LLM's liturgical translation fidelity by pulling raw text from the `input/` directory, resolving the `glossary.json`, and exporting markdown to the `output/` directory.
   ```bash
   python berana.py benchmark-translation --temp 0.0 --ctx 8192
   ```


## 7. Surya OCR Model Sizes (VRAM Reference)

When the pipeline runs for the first time, Surya will automatically download models from HuggingFace Hub:

| Model | Approximate Size | Purpose |
|-------|-----------------|---------|
| `DetectionPredictor` | ~1.1 GB | Finds text line bounding boxes |
| `LayoutPredictor` (Foundation) | ~1.5 GB | Identifies structural layout (columns, headers) |
| `RecognitionPredictor` (Foundation) | ~1.8 GB | Reads actual characters from detected regions |

> **Note:** When using `--slice-only`, the RecognitionPredictor is **not loaded**, saving ~1.8GB of VRAM.

## 8. Layout Automation & HITL Verification (The "Gutter Vision" Pipeline)

To solve the "Semantic Bleeding" problem where traditional generic OCR bleeds Amharic into English, Berana employs a custom **YOLOv8-Segmentation** layout engine coupled with an SQLite-backed HITL (Human-In-The-Loop) Web Editor.

This decoupled approach ensures our mathematical coordinate map (The "Brain") translates perfectly to our human-verified layout strategy (The "Soul").

### Phase 1: AI Layout Inference
Predict column trajectories (gutters) over the entire document utilizing our specialized model (`models/layout/weights/berana_yolov8_divider_v13.pt`).

```bash
# 1. Prepare raw samples from a complex PDF
python berana.py layout-prep --pdf-path data/raw_pdfs/manuscript.pdf --num-pages 20

# 2. Run vision inference to generate bounding masks across the entire manuscript
python berana.py layout-infer --pdf-path data/raw_pdfs/manuscript.pdf --start-page 0 --num-pages 456
```

By default, auto-label tasks are written to:
`output/layout_auto/auto_labels_tasks.json`

Import that JSON file into Label Studio for manual verification.
For exact UI steps (Local Files storage path, JSON import mode, and YOLO export mode),
see `tools/label_studio/README.md` section **"Exact Label Studio Workflow (Import/Export Contract)"**.

### Phase 2: Human Verification (The HITL Editor)

`tools/hitl_line_editor.py`

Because ancient manuscripts are physically crooked/tilted, bounding boxes fail. We utilize a **Line of Best Fit** vector engine that allows the researcher to pivot the detected AI columns linearly.

> Read more: [HITL Line Editor Research Doc](docs/research/hitl_methodology.md)

```bash
# Start the local isolated SQLite verification server
PYTHONPATH=. .venv/bin/python tools/hitl_line_editor.py
```
*(Open your browser to `http://localhost:8000`)*

*   **Database:** Modifies `data/layout_dataset/hitl_line_editor.sqlite3`.
*   **Geometric Contract:** Records absolute `[x1, y1, x2, y2]` array properties for surgical cropping.

### Phase 3: Text Extraction (Upcoming)

```bash
# Take the SQLite vectors, execute Affine perspective warping, and run Surya OCR
python berana.py extract-text --pdf-path data/raw_pdfs/manuscript.pdf
```
*(Produces final `document_structural.json` output).*

---

## 9. Data Safety & Local Developer Backups

The HITL Pipeline utilizes an ACID-compliant local SQLite database to prevent catastrophic loss during multi-hour review sessions.

All proprietary tracking/state occurs locally inside `.git_exclude/` to prevent publishing workflow data to research repositories.

**Daily Snapshot Command:**
```bash
# Back up your verification progress (both SQLite and JSON states) without hitting Version Control
mkdir -p .git_exclude/backup
cp data/layout_dataset/hitl_line_editor.sqlite3 .git_exclude/backup/hitl_line_editor_backup.sqlite3
cp data/layout_dataset/verified_lines_state.json .git_exclude/backup/verified_lines_state_backup.json
```
---

## 10. Performance Metrics & Weights

Our `berana_yolov8_divider_v13.pt` small-segmenter (11.7M params) was achieved via active learning and mosaic augmentations across over 100 epochs, early stopping at ~0.270 hours training time on a single RTX 3060 Ti.

> **Want to train your own custom layout model?** Check out the dedicated [Custom Model Training Guide](tools/label_studio/README.md) for instructions on setting up Label Studio, active learning, and YOLOv8-seg configuration.

**Validation Results (Mask Level):**
*   **Mask mAP50:** `0.946`
*   **Precision:** `0.862` / **Recall:** `0.880`
*   `divider_left` mAP50: `0.924`
*   `divider_right` mAP50: `0.968`

This extraordinarily high fidelity restricts human intervention required in Phase 2 to roughly **1 page out of every 7**, typically related to massive physical tears in the original parchment.

## 11. Architectural Philosophy: The "Tree-Branch" Pattern
This codebase adheres to a strict **Modular Monolith** architecture.
* **Rule of Modularity:** No monolithic code. Maximum 250 lines per file.
* **Orchestrator Pattern:** Top-level files (e.g., `berana.py`, `modules/ocr_engine/orchestrator.py`) contain **zero** processing logic. They are strictly routers/callers that pass explicitly typed DataClasses between deeper modules.
* **Progressive Complexity:** Logic becomes more verbose and specialized the deeper you traverse into the `/modules` directory.
