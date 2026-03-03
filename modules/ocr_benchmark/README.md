# OCR Benchmark (Pilot 50+200) - Full Runbook

This runbook is the end-to-end benchmark flow, including Label Studio setup and annotation loops.

## 1. Preconditions
1. Crop-columns artifacts must already exist for the target PDF:
   - `python berana.py crop-columns --pdf-path input/raw_pdfs/<doc>.pdf`
2. OCR benchmark commands are run from project root with `.venv` active.
3. Canonical manifest path:
   - `input/ocr_benchmark/line_manifest.jsonl`

## 2. Stage 1 - Prepare Benchmark Line Candidates
```bash
python berana.py ocr-benchmark prepare-lines \
  --pdf-path input/raw_pdfs/<doc>.pdf \
  --languages geez,amharic
```
What it does:
- extracts line crops from latest column-crop run,
- assigns deterministic split (`train`/`holdout`),
- enforces split freeze integrity.

## 3. Stage 2 - Generate Zero-Shot Baseline
```bash
python berana.py ocr-benchmark run-surya-baseline --pdf-path input/raw_pdfs/<doc>.pdf
```
What it does:
- generates baseline OCR predictions used for analysis and pre-annotation hints.

## 4. Stage 3 - Label Studio Setup for OCR Benchmark
1. Start Label Studio:
```bash
cd tools/label_studio
docker compose up -d
```
2. In Label Studio project settings, set labeling template XML to:
   - `tools/label_studio/ocr_benchmark_project_ui.xml`
3. Configure Local Files source path:
   - `/label-studio/files/ocr_benchmark/<doc_stem>_vNN/prep/images`
4. Click Sync after saving the Local Files source.

## 5. Stage 4 - Holdout Annotation Loop
1. Export holdout tasks:
```bash
python berana.py ocr-benchmark make-ls-tasks \
  --pdf-path input/raw_pdfs/<doc>.pdf \
  --split holdout
```
2. In Label Studio:
- import generated holdout task JSON,
- correct transcription text in the OCR benchmark UI.
3. Export annotations (both formats are supported):
- full task JSON export, or
- minimum/flattened JSON export.
4. Import annotation export into canonical manifest:
```bash
python berana.py ocr-benchmark import-ls-export \
  --export-json input/ocr_benchmark/<doc>_holdout_export.json \
  --manifest input/ocr_benchmark/line_manifest.jsonl
```

## 6. Stage 5 - Train Annotation Loop
1. Export train tasks:
```bash
python berana.py ocr-benchmark make-ls-tasks \
  --pdf-path input/raw_pdfs/<doc>.pdf \
  --split train
```
2. In Label Studio:
- import train task JSON,
- complete train transcriptions.
3. Import train export back to manifest:
```bash
python berana.py ocr-benchmark import-ls-export \
  --export-json input/ocr_benchmark/<doc>_train_export.json \
  --manifest input/ocr_benchmark/line_manifest.jsonl
```

## 7. Stage 6 - Character Coverage Governance
Coverage policy config (tiered gate) currently uses:
- `input/ocr_benchmark/config/ethiopic_charset.v1.json`

Unicode declaration-only config (compact family grid) is:
- `input/ocr_benchmark/config/ethiopic_charset.v2.unicode.json`

Run coverage diagnostics:
```bash
python berana.py ocr-benchmark coverage-report \
  --manifest input/ocr_benchmark/line_manifest.jsonl \
  --doc-stem <doc_stem> \
  --charset-config input/ocr_benchmark/config/ethiopic_charset.v1.json
```

Generate targeted queue when coverage is insufficient:
```bash
python berana.py ocr-benchmark coverage-queue \
  --pdf-path input/raw_pdfs/<doc>.pdf \
  --manifest input/ocr_benchmark/line_manifest.jsonl \
  --charset-config input/ocr_benchmark/config/ethiopic_charset.v1.json \
  --max-items 200
```

## 8. Stage 7 - Finetune + Evaluate
Finetune commands are coverage-gated by default:
```bash
python berana.py ocr-benchmark run-surya-finetuning \
  --pdf-path input/raw_pdfs/<doc>.pdf \
  --manifest input/ocr_benchmark/line_manifest.jsonl

python berana.py ocr-benchmark run-trocr-finetuning \
  --pdf-path input/raw_pdfs/<doc>.pdf \
  --manifest input/ocr_benchmark/line_manifest.jsonl
```

Run final benchmark evaluation:
```bash
python berana.py ocr-benchmark evaluate \
  --pdf-path input/raw_pdfs/<doc>.pdf \
  --manifest input/ocr_benchmark/line_manifest.jsonl
```

## 9. Stage 8 - Post-Labeling Auto Pipeline
If both holdout and train exports are complete, you can run the post-labeling pipeline in one command:
```bash
python berana.py ocr-benchmark auto \
  --pdf-path input/raw_pdfs/<doc>.pdf \
  --manifest input/ocr_benchmark/line_manifest.jsonl \
  --charset-config input/ocr_benchmark/config/ethiopic_charset.v1.json
```

What it does:
- imports provided exports (or reuses existing manifest rows),
- runs coverage report,
- optionally builds queue,
- optionally runs finetune,
- runs evaluation.

## 10. Character Ingestion & Generation Commands
Generate compact Unicode declaration config:
```bash
python berana.py ocr-benchmark generate-charset-config \
  --output-path input/ocr_benchmark/config/ethiopic_charset.v2.unicode.json \
  --source wiktionary \
  --declaration-only
```

Generate explicit tier-char policy config (for gating):
```bash
python berana.py ocr-benchmark generate-charset-config \
  --output-path input/ocr_benchmark/config/ethiopic_charset.policy.json \
  --source wiktionary \
  --include-tier-charlists \
  --manifest input/ocr_benchmark/line_manifest.jsonl \
  --pdf-path input/raw_pdfs/<doc>.pdf
```

## 11. Operational Notes
- If split freeze fails, rerun `prepare-lines` with `--refresh-freeze` only when you intentionally changed split strategy or candidate selection.
- If Label Studio image loading fails, verify Local Files root points to `/label-studio/files/ocr_benchmark/<doc_stem>_vNN/prep/images`.
- If evaluation says a model is missing, run only available models with `--allow-missing-models` on `evaluate`.

## 12. Workflow Links
- Label Studio setup for layout + OCR tasks: `tools/label_studio/README.md`
- HITL verification workflow: `tools/hitl_line_editor_app/README.md`
- HITL SQLite-driven finetuner path: `tools/hitl_yolo_finetuner_app/README.md`
