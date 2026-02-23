# Phase 3: Human-In-The-Loop (HITL) Layout & OCR Architecture

## Core Philosophy
We have completely rejected the "black-box" monolithic pipeline. Instead, we are adopting a strictly gated, 2-Step verification process. This ensures that the OCR engine **only** processes 100% accurate, human-verified column boundaries.

By avoiding blind OCR on AI-generated layouts, we guarantee zero data contamination (e.g., Ge'ez bleeding into Amharic) and build an infinitely expanding ground-truth dataset for future training.

---

## Step 1: Vision Inference & Manual Review (The Safety Net)
**Goal:** Generate "smart overlays" using our trained YOLOv8 model, then let the human expert perform a rapid QC (Quality Control) pass.

1. **Mass AI Inference:**
   - Run the updated `layout-infer` command over the remaining ~360 pages of the document.
   - *Command:* `python berana.py layout-infer --pdf-path data/raw_pdfs/doc_001.Triple.pdf --start-page 101 --num-pages 360`
2. **Review Environment (Label Studio):**
   - Import the generated JSON into Label Studio.
   - Default file: `output/layout_auto/auto_labels_tasks.json`
   - Perform human review. Since the model has been highly trained, ~90% of the pages will only require a single click ("Submit").
   - For edge cases (e.g., missing dividers or skewed pages), physically drag or redraw the polygons.
3. **The Gold Standard Export:**
   - Once all 460 pages are confirmed perfectly sliced, export the final, human-verified annotations as a YOLO ZIP file.

---

## Step 2: The "Extract-Text" Pipeline (Precision OCR)
**Goal:** Ingest the verified ZIP file, physically (or logically) isolate the columns based ONLY on human-confirmed coordinates, and feed them into the OCR character recognition.

1. **Delete the Monolith:** Remove the previous `run_pipeline` command.
2. **Create the Decoupled Command:**
   - Build a new command: `berana.py extract-text --pdf-path doc_001.pdf --verified-labels final_export.zip`
3. **Execution Flow inside `extract-text`:**
   - **Ingest:** Unzip the Label Studio export and parse the exact `[x1, y1, x2, y2]` bounding coordinates for every single page.
   - **Isolate:** Iterate through the PDF. Use the exact human coordinates to draw the 3 logical column zones.
   - **Detect:** Run Surya Text Detection over the page to find individual text boundaries.
   - **Map & Filter:** If a text box falls in Column 1, it is strictly assigned to Ge'ez. If it falls between the dividers, it is Amharic.
   - **Recognize:** Pass these strictly separated and validated clusters into Surya's Text Recognition engine.
   - **Output:** Generate the pristine `doc_001_structural.json` containing perfect translations grouped by language column.

---

## Step 3: OCR Engine Fine-Tuning (Preparation for Phase 4)
Because we took the time to build this layout safety net, we can now shift all our focus to the AI's ability to read Ethiopic characters.
- *Next Milestone:* Generating robust OCR training data (pairing cropped images of Ge'ez/Amharic text with their true digital text).
