# HITL YOLO Finetuner

This tool exports YOLOv8 segmentation training data directly from the verified HITL SQLite state and can optionally preview and train.

- Entry point: `tools/hitl_yolo_finetuner.py`
- Source of truth: `input/layout_dataset/hitl_line_editor.sqlite3`
- Label model: `divider_left -> 0`, `divider_right -> 1`

## Commands

### 1) Export Dataset

Builds a YOLO dataset from verified pages using deterministic 80/20 split and Run Registry resolution.

```bash
./.venv/bin/python tools/hitl_yolo_finetuner.py export \
  --pdf-path input/manuscript.pdf \
  --db-file input/layout_dataset/hitl_line_editor.sqlite3 \
  --output-dir output/hitl_finetuner
```

Useful flags:

- `--split-seed 42`: deterministic train/val assignment.
- `--line-width-px 4.0`: line extrusion thickness in pixels.
- `--max-tilt-deg 15.0`: skip extreme line tilt.
- `--min-line-length-px 100.0`: skip short/noisy lines.
- `--min-train-pages 20`, `--min-val-pages 5`: fail loud if split floors are not met.
- `--copy-mode hardlink|copy|symlink`: image sync strategy.
- `--dry-run`: print report only, no files written.

### 2) Preview

Renders generated polygons on source images for QA safely isolated from real data runs.

```bash
./.venv/bin/python tools/hitl_yolo_finetuner.py preview \
  --pdf-path input/manuscript.pdf \
  --db-file input/layout_dataset/hitl_line_editor.sqlite3 \
  --output-dir output/hitl_finetuner \
  --line-width-px 4.0 \
  --max-pages 20
```

Preview outputs are written cleanly matching standard layout directories:

- `output/hitl_finetuner/<doc_stem>_vNN/visuals/*_overlay.jpg`

### 3) Train

Runs Ultralytics training against generated dataset and registers the weights in the Run Registry.

```bash
./.venv/bin/python tools/hitl_yolo_finetuner.py train \
  --pdf-path input/manuscript.pdf \
  --output-dir output/hitl_finetuner \
  --epochs 100 \
  --batch 4
```

### 4) Run (Export + Train)

Convenience wrapper to sequentially run export and train for a completely closed loop.

```bash
./.venv/bin/python tools/hitl_yolo_finetuner.py run --pdf-path input/manuscript.pdf
```

## Output Structure

After export, a versioned directory is created (e.g. `output/hitl_finetuner/manuscript_v01`):

```text
output/hitl_finetuner/<doc_stem>_vNN/
├── data/
│   ├── dataset.yaml
│   ├── export_report.json
│   ├── manifests/
│   │   ├── train_pages.txt
│   │   └── val_pages.txt
│   ├── images/
│   └── labels/
├── meta/
│   ├── signature.json
│   └── training_manifest.json
└── weights/
    └── best.pt
```

## Validation and Failure Rules

Export is strict and enforces Closed-Loop Provenance. It will hard-fail on:

- database missing provenance metadata (`provenance_metadata` table),
- `--doc-stem` mismatch with database locked provenance,
- "pointer drift" (the registry latest run does not match the DB's source run),
- missing source image for any verified page,
- DB/image dimension mismatch (`img_w`/`img_h` vs actual image size),
- insufficient split floors (`min_train_pages` / `min_val_pages`),
- empty/invalid DB state.

Each line is converted from `[x1, y1, x2, y2]` into a clamped normalized polygon:

- `[x1, y1, x2, y2, x3, y3, x4, y4]` in `[0.0, 1.0]`.
