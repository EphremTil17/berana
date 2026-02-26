# HITL Verification Tool Guide

This guide covers the HITL line-editor workflow used to verify and correct divider lines before downstream cropping/OCR stages.

- Entry point: `tools/hitl_line_editor.py`
- Web app module: `tools/hitl_line_editor_app/`
- Source of truth DB: `input/layout_dataset/hitl_line_editor.sqlite3`

## Purpose

The tool lets you review each page and adjust divider geometry as explicit endpoint lines:

- `left`: `[x1, y1, x2, y2]`
- `right`: `[x1, y1, x2, y2]`
- `verified`: boolean flag per page

The database now enforces strict **Closed-Loop Provenance**. On initialization, the database locks onto the `layout-infer` run directory that the Label Studio ZIP images originated from. This ensures that downstream finetuning can deterministically source from identical, registry-tracked inputs.

## Run

From project root:

```bash
source .venv/bin/activate
PYTHONPATH=. .venv/bin/python tools/hitl_line_editor.py
```

Open:

- `http://localhost:8000`

## Data Contract

Persistent state is stored in:

- `input/layout_dataset/hitl_line_editor.sqlite3`

The `page_lines` table stores:

- `page_id`
- `left_json`
- `right_json`
- `verified`
- `img_w`
- `img_h`

A secondary table `provenance_metadata` (singleton row `id=1`) locks the lineage:
- `source_stage` (e.g. `layout-infer`)
- `doc_stem` (e.g. `manuscript`)
- `source_run_dir` (the absolute `/output` directory)

## Typical Workflow

1. Generate candidate pages and/or auto-labels (`layout-prep`, `layout-infer`).
2. Open the HITL editor and adjust divider lines.
3. Mark pages as verified.
4. Use verified DB state downstream:
   - `crop-columns` pipeline
   - HITL YOLO finetuner dataset export/training

## Related Tool Docs

- Label Studio operations: `tools/label_studio/README.md`
- HITL SQLite -> YOLO finetuning: `tools/hitl_yolo_finetuner_app/README.md`
