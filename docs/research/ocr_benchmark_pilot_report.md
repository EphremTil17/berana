# OCR Benchmark Pilot: From Ad-Hoc OCR Attempts to a Controlled Research Workflow

## 1. Why We Built This Benchmark in the First Place
The OCR benchmarking effort did not start as a neat research program. It started from operational pain.
We had OCR outputs, we had model candidates, and we had pages that looked clean enough to read manually, but we did not have a trustworthy method for answering the only question that matters: which OCR path should we invest in for production on Ethiopic manuscripts?

At first glance, the answer looked simple. Run Surya, run TrOCR, compare outputs, then choose the better one.
In practice, that approach failed immediately because the comparisons were unstable.
Small changes in crop quality, split composition, export format, or annotation completeness could move outcomes dramatically.
When a benchmark can be changed accidentally by pathing, random order, or inconsistent data contracts, model differences become noise.

The purpose of this project phase was therefore not to chase the best score in a single run.
The purpose was to build a benchmark system that forces methodological discipline:
if two runs claim different conclusions, we can identify exactly why.
If a model appears to improve, we can prove the data and rules were held constant.
If a model fails, we can diagnose where and how it fails rather than merely observing a low average CER.

That is the true objective of this report: documenting how we transformed OCR benchmarking from scripts into a reproducible research pipeline.

## 2. The Domain Reality: Why Ethiopic OCR Needs Extra Control
The documents in scope are liturgical, multi-column, and physically inconsistent.
Column geometry is already challenging in earlier project phases, and OCR inherits those imperfections.
By the time text reaches recognition, we are still dealing with skew artifacts, variable stroke thickness, bleed-through, blur, and line fragments.

Unlike modern printed corpora, this domain also amplifies script-level issues.
Ge'ez and Amharic share visual families but diverge in usage and frequency.
Rare character forms can appear sparsely, making random sampling a weak strategy for robust training readiness.
A 200-line dataset can look numerically sufficient while still under-representing critical forms.

This leads to the central methodological constraint for benchmark design:
a line quota is necessary, but not sufficient.
We need explicit character coverage governance.

## 3. What We Decided to Measure and Why
From the start, we separated two categories of outputs.
First are outcome metrics such as CER and WER.
Second are process guarantees such as split integrity, schema validity, and coverage sufficiency.

Traditional benchmarking often focuses only on first-category metrics.
That is risky here, because a strong metric from a weak process is not scientifically meaningful.
A model can “win” because the holdout leaked into training, because malformed exports were silently dropped, or because Unicode variants created fake penalties.

For this reason, the pilot was structured around a staged question sequence:
1. Can we produce deterministic and auditable benchmark data?
2. Can we move annotations reliably through Label Studio and back into canonical manifests?
3. Can we prove training readiness through explicit coverage policy?
4. Only then: can we compare models fairly?

The stack was built to enforce this order.

## 4. Architectural Direction: One Benchmark Surface, Clear Stage Boundaries
We intentionally kept the benchmark under the main Berana CLI while isolating implementation under `modules/ocr_benchmark`.
This preserved project structure: Berana remains the top-level orchestrator, while OCR benchmarking behaves as a sub-application with its own stage lifecycle.

A key architectural principle was document-scoped artifact roots.
Every stage writes under:
`output/ocr_benchmark/<doc_stem_vNN>/...`

This pattern matches the broader project convention and keeps provenance discoverable.
The stage directories (`prep`, `surya_zero_shot`, `coverage`, `final_eval`, and others) are not just storage.
They represent the chronological scientific state of the experiment.

Run registry pointers were integrated so later stages can resolve the latest valid upstream artifacts without manual path hunting.
That removes human error and keeps command ergonomics practical.

## 5. The Chronological Workflow We Implemented

### 5.1 Stage A: Prepare Lines as the Entry Gate
The benchmark begins with line preparation from existing crop-columns outputs.
This was a deliberate choice.
If we allow arbitrary image inputs at benchmark time, reproducibility is immediately compromised.

`prepare-lines` resolves the source document stem, locates validated column-crop artifacts, extracts candidate lines, and stores both images and candidate metadata.
The image layout now follows a self-explanatory hierarchy by page and language.
This was important not only for human navigation but also for debugging segmentation quality by local region.

During this stage we apply deterministic split assignment.
The pilot target is 200 train and 50 holdout.
When the split is first created, we persist a split freeze artifact with a hash.
Later re-runs compare against that frozen hash.
If the hash drifts, we block continuation unless explicitly reviewed.

This behavior initially felt strict, but it is exactly what protects benchmark validity.
A benchmark that silently changes split composition is not a benchmark.

### 5.2 Stage B: Surya Zero-Shot Baseline as Annotation Accelerator
After preparation, we run Surya zero-shot baseline over benchmark lines.
The baseline is stored as structured predictions, not transient console output.

This stage serves two purposes.
First, it gives us the first model checkpoint in the comparison matrix.
Second, it seeds annotation tasks with pre-annotations so human reviewers correct instead of typing from scratch.
For Ethiopic scripts, that correction flow significantly improves throughput.

We documented and observed a recurring failure mode here: wrong-script or noisy symbol output on low-quality crops.
This confirmed that zero-shot cannot be trusted as final truth, but is still valuable as a draft.

### 5.3 Stage C: Split-Aware Label Studio Tasking
Originally task export was holdout-centric.
That was useful for quick checks, but incomplete for full benchmark lifecycle.
We extended `make-ls-tasks` with `--split train|holdout|all`.
This change looks small from CLI perspective, but it closes a major operational gap: a single command now supports the entire annotation schedule.

Default output filenames now encode split, reducing accidental mix-ups.
Task payloads include benchmark metadata fields (`line_id`, `doc_stem`, `page_id`, `lang`, and split hints) so exports can be merged back deterministically.

A practical integration issue surfaced with Label Studio local-file serving.
The JSON itself was not the root problem.
The root problem was path mapping under Label Studio’s local storage root.
We aligned benchmark task image paths with the project’s working local-files strategy and documented this in the tool docs.

### 5.4 Stage D: Robust Import Back to Canonical Manifest
Annotation is useful only if imported safely.
Real-world Label Studio exports vary by format and user settings.
Some include nested `data`; minimum exports flatten fields at top level.

`import-ls-export` was hardened to support both formats.
Instead of crashing on first malformed item, the importer now logs precise skips and continues where safe.
This is essential for interrupted sessions and partial exports.

The canonical destination remains the JSONL line manifest.
This keeps data governance centralized and makes downstream modules independent from Label Studio format details.

### 5.5 Stage E: Coverage Governance as a First-Class Decision Gate
At this point, the benchmark could run.
But it still lacked a formal answer to a key research question:
is the train split alphabetically representative enough to justify finetuning?

We implemented coverage governance through three linked components:
1. authoritative charset config in repo,
2. deterministic coverage reporting,
3. actionable annotation queue generation.

The charset policy config (initially v1 during early pilot wiring, now standardized to v3 policy in operations) defines required character tiers and thresholds.
The coverage report computes deficits by split and overall.
The queue module converts deficits into ranked candidate lines so annotation can close gaps deliberately.

This changed the annotation strategy from passive accumulation to deficit-driven curation.
That is a major methodological upgrade.

### 5.6 Stage F: Finetuning Commands With Coverage Enforcement
Coverage governance has no value if it remains advisory.
We therefore added enforcement hooks in finetune commands with default hard-fail behavior.

`run-surya-finetuning` and `run-trocr-finetuning` now check coverage status before proceeding.
Operators can override with `--no-enforce-coverage`, but the default path is scientifically conservative.

This protects the project from wasting GPU cycles on data that does not meet baseline representational requirements.
It also prevents accidental “success narratives” from under-covered training sets.

### 5.7 Stage G: Evaluation Beyond Averages
The reporting stack still computes CER/WER as primary metrics.
However, we added alignment-backed character confusion diagnostics using Levenshtein backtrace.

This enables explicit counting of substitutions, insertions, and deletions with pseudo-symbol channels (`<INS>`, `<DEL>`).
The result is not just a score, but a diagnosis surface.
When the model fails, we can identify whether failure is mainly substitution drift, insertion noise, or deletion collapse.

This is especially important for Ethiopic OCR, where seemingly small confusions can map to semantically different forms.

## 6. What We Observed During Pilot Operation
The pilot produced several clear observations.
First, line extraction quality is a dominant variable.
Tiny or clipped crops generate unstable predictions regardless of model family.
This is not an OCR-language issue alone; it is an upstream extraction quality issue.

Second, zero-shot outputs frequently include non-Ethiopic artifacts on difficult samples.
These errors are expected for domain mismatch and reinforce the need for correction-based annotation.

Third, split discipline creates friction early but saves substantial debugging time later.
The split freeze “violation” messages that initially felt inconvenient turned out to be exactly the protection needed to prevent invisible benchmark drift.

Fourth, structured artifact paths and registry pointers reduce cognitive overhead.
Operators no longer need to remember ad-hoc directories.
This materially improves repeatability under active development.

## 7. Why the TrOCR Tokenizer Constraint Remains Central
A recurring technical concern in planning was the TrOCR tokenizer trap.
Default Latin-centric tokenizers are not acceptable for Ethiopic-heavy tasks.
This pilot keeps that constraint explicit in architecture and readiness logic.

The benchmark infrastructure is now prepared to enforce that TrOCR runs include valid tokenizer preflight assumptions in implementation.
Without that, TrOCR comparisons are not meaningful.
This is not a minor detail; it is a validity prerequisite.

## 8. Limitations of the Current Phase
The current phase now has an asymmetric state rather than an entirely incomplete state.
TrOCR finetuning is implemented end-to-end in code and produces checkpoint plus holdout predictions.
Surya finetuning remains scaffolded but intentionally not yet implemented in this repository.
That asymmetry matters because it changes how results should be interpreted.

When `evaluate` is run in strict mode, missing Surya-finetuned artifacts are treated as a hard failure.
When `evaluate` is run with `--allow-missing-models`, the pipeline produces a valid comparative report over available models.
This behavior is correct, but it means operational intent must be explicit at command time.

Coverage policy remains versioned and intentionally adjustable.
As corpus understanding evolves, charset thresholds and inclusion sets will evolve as policy updates.
That should be treated as controlled methodology iteration rather than benchmark instability, provided the policy file used for any reported run is recorded.

## 9. Character Ingestion and Generation Methodology
Character handling was deliberately separated into declaration and policy responsibilities.
This separation prevents a common failure mode where Unicode truth and training thresholds get mixed into one brittle file.

The declaration side is the Unicode-oriented Ethiopic inventory, generated and structured to support coordinate-style retrieval by family and order.
This serves as the script reference layer.
It is not intended to encode experiment-specific scarcity thresholds.

The policy side is the benchmark gate layer.
It maps expected characters into threshold tiers and drives `coverage-report` and finetune enforcement behavior.
In this project state, the actively used policy for operations is `input/ocr_benchmark/config/ethiopic_charset.v3.policy.json`.
Earlier policy files remain as history artifacts, not canonical active policy.

This distinction gives the team two important controls:
first, a stable source-of-truth declaration that can be regenerated;
second, a tunable but explicit coverage gate aligned to current annotation goals.

## 10. Recommended Next Execution Sequence
The next cycle should continue as a closed loop with one planned branch:
use Surya as the primary investment path for full comparative quality improvement, while TrOCR remains a secondary research branch unless implementation strategy changes.

Operational priority order should be:
1. keep holdout immutable and continue expanding train annotations;
2. run coverage closure against the active v3 policy;
3. run model stages and evaluate with explicit strictness mode;
4. archive run artifacts and update research conclusions only from frozen-report outputs.

This order protects against misleading progress.
Running finetune before coverage closure can still be useful for diagnostics, but those runs should be labeled exploratory and excluded from decision claims.

## 11. Scientific Position at the End of This Pilot
The pilot now has both infrastructural and numeric outcomes.
The infrastructure objective was achieved: the team now has a reproducible benchmarking system with enforceable contracts and traceable artifacts.
The numeric outcome from the latest documented run is also clear:
Surya zero-shot remains the winning available model on this manuscript subset.

From the current `final_eval` artifacts:
- `surya_zero_shot` overall CER is approximately 5.90%;
- `trocr_zero_shot` overall CER is approximately 102.62%;
- `trocr_finetuned` overall CER is approximately 90.10%.

This does not prove TrOCR is intrinsically unsuitable for Ethiopic OCR.
It proves that the current TrOCR finetune implementation and data regime are not yet competitive.
The immediate research conclusion is therefore methodological and strategic:
production effort should prioritize Surya finetuning with expanded and curated data, while TrOCR should be treated as a secondary track requiring architectural rework before fair comparison.

## 12. Operational Appendix: Canonical Command Flow
This appendix is intentionally procedural and mirrors the exact interaction pattern used in operations: run a command, perform a Label Studio action when required, then continue.

Step 1: prepare benchmark line candidates.
```bash
python berana.py ocr-benchmark prepare-lines --pdf-path input/raw_pdfs/doc_001.Triple.pdf --languages geez,amharic
```

Step 2: run Surya zero-shot baseline to seed pre-annotations.
```bash
python berana.py ocr-benchmark run-surya-baseline --pdf-path input/raw_pdfs/doc_001.Triple.pdf --split all
```

Step 3: generate holdout tasks for Label Studio.
```bash
python berana.py ocr-benchmark make-ls-tasks --pdf-path input/raw_pdfs/doc_001.Triple.pdf --split holdout
```

Step 4: in Label Studio, import holdout tasks and annotate.
Use the OCR benchmark XML config, complete all holdout tasks by correcting transcription text, then export JSON (minimum JSON export is supported).

Step 5: import holdout export back to canonical manifest.
```bash
python berana.py ocr-benchmark import-ls-export \
  --export-json input/ocr_benchmark/doc_001.Triple_v01_ocr_label_studio_holdout.json \
  --manifest input/ocr_benchmark/line_manifest.jsonl \
  --allow-page-overlap
```

Step 6: generate train tasks for Label Studio.
```bash
python berana.py ocr-benchmark make-ls-tasks --pdf-path input/raw_pdfs/doc_001.Triple.pdf --split train
```

Step 7: in Label Studio, import train tasks and annotate.
Use the same OCR benchmark project config, complete train tasks, then export JSON (minimum JSON export supported).

Step 8: import train export to update canonical manifest.
```bash
python berana.py ocr-benchmark import-ls-export \
  --export-json input/ocr_benchmark/doc_001.Triple_v01_ocr_label_studio_train.json \
  --manifest input/ocr_benchmark/line_manifest.jsonl \
  --allow-page-overlap
```

Step 9: run coverage analysis using active policy.
```bash
python berana.py ocr-benchmark coverage-report \
  --manifest input/ocr_benchmark/line_manifest.jsonl \
  --charset-config input/ocr_benchmark/config/ethiopic_charset.v3.policy.json \
  --doc-stem doc_001.Triple
```

Step 10: generate annotation queue for deficit closure.
```bash
python berana.py ocr-benchmark coverage-queue \
  --pdf-path input/raw_pdfs/doc_001.Triple.pdf \
  --manifest input/ocr_benchmark/line_manifest.jsonl \
  --charset-config input/ocr_benchmark/config/ethiopic_charset.v3.policy.json \
  --max-items 200
```

Step 11: run TrOCR finetuning (implemented path).
```bash
python berana.py ocr-benchmark run-trocr-finetuning \
  --pdf-path input/raw_pdfs/doc_001.Triple.pdf \
  --manifest input/ocr_benchmark/line_manifest.jsonl \
  --charset-config input/ocr_benchmark/config/ethiopic_charset.v3.policy.json \
  --no-enforce-coverage
```

Step 12: evaluate available models without requiring Surya-finetuned artifacts.
```bash
python berana.py ocr-benchmark evaluate \
  --pdf-path input/raw_pdfs/doc_001.Triple.pdf \
  --manifest input/ocr_benchmark/line_manifest.jsonl \
  --allow-missing-models
```

Step 13: review final artifacts in this order.
First read `final_report.md`, then inspect `final_scores.csv`, then inspect per-model debug files (`line_debug_*.csv` and `top_confusions_*.csv`) before making model direction decisions.

## 13. Label Studio Instruction Set (Benchmark Transcription)
The benchmark loop requires a dedicated OCR transcription project template.
Use `tools/label_studio/ocr_benchmark_project_ui.xml` as the labeling interface.
Do not reuse layout labeling XML; the tags and expected task payloads differ.

Set Local Files storage root to the benchmark image path under Label Studio document root:
`/label-studio/files/ocr_benchmark/<doc_stem>_vNN/prep/images`
Then run storage sync before task import.

For task import, use JSON files produced by `make-ls-tasks`.
Annotate by correcting the transcription field only.
Do not edit metadata fields (`line_id`, `page_id`, `lang`, `split_hint`) because importer logic depends on them.

Export format can be full JSON or minimum JSON.
Importer supports both and merges idempotently into `input/ocr_benchmark/line_manifest.jsonl`.
After each import, run `coverage-report` before continuing to any finetuning stage.

## 14. Closing Note
This report now records a complete checkpoint of the pilot: architecture hardening, operational workflow, and measured model outcomes.
The key strategic outcome is not ambiguous.
For this corpus and current implementation state, Surya is the production-leading path.
Future TrOCR work remains possible, but should be pursued as a separate R&D branch with explicit architectural changes before re-entering the main comparative decision track.
