# OCR Training Tooling

This directory documents the OCR training workflow implemented by [`tools/ocr_training.py`](../ocr_training.py) and the supporting runtime under `modules/ocr_training/`.

The current system is built around:

- FIDEL typed + synthetic extraction
- deterministic Surya-compatible dataset generation
- manual or auto-planned Surya finetuning
- single-GPU and single-node multi-GPU execution
- CER-primary best-checkpoint tracking with side-by-side WER tracking
- live run monitoring artifacts written during training
- automatic final report generation on completion or interrupt

If you need environment setup from scratch, read [setup.md](./setup.md).

Related docs:

- root project overview: [../../README.md](../../README.md)
- OCR training setup and canonical command guide: [setup.md](./setup.md)

## Design Goals

The OCR training stack is meant to behave more like a mature training framework than a one-off script:

- training should be reproducible
- eval subsets should be deterministic within a run
- best checkpoints should survive checkpoint pruning
- runs should emit enough live telemetry to decide whether to continue
- multi-GPU interruption and failure paths should not strand worker processes
- post-run reports should be created automatically

## Core Workflow

The end-to-end flow is:

1. Extract raw FIDEL assets.
2. Run `cleanup-fidel` to produce a cleaned extracted tree and review buckets.
   Optional: pass `--heuristic-cleanup-dir` to fold full-eval failure analysis exclusions into the cleaned snapshot before dataset build.
3. Build the Surya dataset from the cleaned extracted tree.
4. Optionally run `verify-surya-dataset` on the built `hf_dataset`.
5. Inspect batch/token geometry if needed.
6. Train with `train-surya`.
7. Let the runtime auto-write live reports during training.
8. Let the runtime auto-generate final report assets on completion or interrupt.
9. Run explicit evaluation or benchmarking against the saved run/checkpoints.

## Main Commands

### 1. Extract FIDEL typed + synthetic data

```bash
PYTHONPATH=. .venv/bin/python tools/ocr_training.py extract-fidel \
  --raw-root input/ocr_training/fidel/raw \
  --extracted-root input/ocr_training/fidel/extracted
```

Default behavior:

- includes `typed,synthetic`
- excludes `handwritten,hdd,hdd_18,hdd_rand`

### 2. Build the Surya dataset

```bash
PYTHONPATH=. .venv/bin/python tools/ocr_training.py build-surya-dataset \
  --extracted-root input/ocr_training/fidel_cleaned/extracted \
  --output-root output/ocr_training_datasets \
  --dataset-name fidel_typed_synthetic_clean
```

This creates a versioned dataset run with:

- `train.jsonl`
- `val.jsonl`
- `holdout.jsonl`
- split manifests
- row manifests under the dataset run

Suspect review semantics:

- by default, `build-surya-dataset` excludes confirmed blanks and all suspect rows from the cleaned review manifest
- `--include-suspect` re-includes only the suspect review copies still left in `suspect_blank_images/`
- deleting a review-copy from `suspect_blank_images/` keeps that row excluded

### 2a. Clean extracted FIDEL assets

```bash
PYTHONPATH=. .venv/bin/python tools/ocr_training.py cleanup-fidel \
  --extracted-root input/ocr_training/fidel/extracted \
  --output-root input/ocr_training/fidel_cleaned
```

This stage:

- clones the extracted tree into a cleaned root
- excludes high-confidence blank rows from the cleaned snapshot
- emits `excluded_blank_images/` and `suspect_blank_images/` review buckets
- writes cleanup review manifests under `blank_cleanup_review/`

### 2b. Verify the built Surya dataset

```bash
PYTHONPATH=. .venv/bin/python tools/ocr_training.py verify-surya-dataset \
  --dataset-dir output/ocr_training_datasets/fidel_typed_synthetic_clean_v01/data/hf_dataset \
  --output-dir output/ocr_training_datasets/fidel_typed_synthetic_clean_v01/data/hf_dataset_verification
```

This is a post-build audit. It does not rewrite the dataset; it emits confirmed/suspect review artifacts so you can confirm that known bad blank-image rows did not survive dataset generation.

### 3. Inspect token and batch geometry

```bash
PYTHONPATH=. .venv/bin/python tools/ocr_training.py inspect-surya-dataset \
  --dataset-dir output/ocr_training_datasets/fidel_typed_synthetic_clean_v01/data/hf_dataset
```

Use this before large runs when tuning:

- sequence length
- batch size
- grad accumulation
- truncation risk

### 4. Train Surya

Example full-data one-epoch manual LoRA run:

```bash
PYTHONPATH=. .venv/bin/python tools/ocr_training.py train-surya \
  --dataset-dir output/ocr_training_datasets/fidel_typed_synthetic_clean_v01/data/hf_dataset \
  --output-dir output/ocr_training_runs/fidel_typed_synthetic_clean_5090_lora_full1ep_v01 \
  --mode manual \
  --finetune-strategy lora \
  --per-device-train-batch-size 6 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --dataloader-num-workers 8 \
  --max-sequence-length 1024 \
  --no-gradient-checkpointing \
  --eval-save-steps 100 \
  --eval-max-rows 1000 \
  --resume none \
  --logging-steps 20 \
  --verbose-epochs \
  --train-fraction 1.0 \
  --num-train-epochs 1 \
  --multi-gpu
```

Important notes:

- `--multi-gpu` automatically relaunches under `torchrun`
- allocator defaults now automatically enable `expandable_segments:True`
- `--eval-save-steps` is the single cadence flag for validation OCR eval plus checkpoint saves
- `--ram-spillover` is allowed by default
- `--resume none` means start fresh even if that output directory already has checkpoints

### 5. Evaluate a trained run

Standard eval:

```bash
PYTHONPATH=. .venv/bin/python tools/ocr_training.py evaluate-surya \
  --run-dir output/ocr_training_runs/fidel_typed_synthetic_clean_5090_lora_full1ep_v01 \
  --dataset-dir output/ocr_training_datasets/fidel_typed_synthetic_clean_v01/data/hf_dataset \
  --metric cer \
  --split holdout
```

Per-modality eval:

```bash
PYTHONPATH=. .venv/bin/python tools/ocr_training.py evaluate-surya-modalities \
  --run-dir output/ocr_training_runs/fidel_typed_synthetic_clean_5090_lora_full1ep_v01 \
  --dataset-dir output/ocr_training_datasets/fidel_typed_synthetic_clean_v01/data/hf_dataset \
  --split holdout \
  --metric cer \
  --modalities typed,synthetic
```

### 6. Benchmark explicit Surya evaluation throughput

```bash
PYTHONPATH=. .venv/bin/python tools/ocr_training.py benchmark-surya-eval \
  --run-dir output/ocr_training_runs/fidel_typed_synthetic_clean_5090_lora_full1ep_v01 \
  --dataset-dir output/ocr_training_datasets/fidel_typed_synthetic_clean_v01/data/hf_dataset \
  --metric cer \
  --split holdout \
  --eval-fraction 0.01 \
  --candidate-eval-batch-sizes 8,16,24,32
```

Benchmark notes:

- `--run-dir` is required and should point to the root training run directory that contains saved checkpoints and `weights/`
- `--metric cer|wer|latest` selects which saved model to benchmark by default
- `--checkpoint-path` overrides metric-based selection and benchmarks one explicit checkpoint directory
- `--candidate-eval-batch-sizes` is required for the benchmark sweep
- `--candidate-worker-counts` is optional and linked positionally one-to-one with batch sizes
- if `--candidate-worker-counts` is omitted, every benchmark candidate runs sequentially with `0` workers
- benchmark artifacts are written under `output/ocr_benchmark/gpu_performance_eval_vNN/<run-stem>/` unless `--output-dir` is set

Benchmark artifacts include:

- `benchmark_summary.json`
- `batch_timings.jsonl`
- `stage_timings.json`
- `candidate_results.jsonl`
- `selected_benchmark_config.json`
- `benchmark_report.md`

### 6. Optional manual utilities

These exist for inspection and regeneration, but they are not required for the normal training workflow:

```bash
PYTHONPATH=. .venv/bin/python tools/ocr_training.py monitor-surya-run --run-dir output/ocr_training_runs/...
PYTHONPATH=. .venv/bin/python tools/ocr_training.py visualize-surya-run --run-dir output/ocr_training_runs/...
```

Normal workflow does not require you to remember these, because the live and final artifacts are generated automatically.

## What Is Automated Now

### During training

Rank 0 automatically writes:

- `evaluation/training_history.csv`
- `evaluation/training_history.jsonl`
- `evaluation/training_curves.svg`
- `training_summary.json`

These are refreshed during the run. They are meant to answer:

- Are loss, CER, and WER still improving?
- How many evals since best CER?
- Did train loss keep falling while CER/WER flattened?
- Which checkpoint is currently best by CER and by WER?

### On completion or interrupt

Rank 0 automatically generates or refreshes:

- `evaluation/training_report.md`
- `evaluation/training_curves.png`
- `evaluation/training_curves.svg`
- `training_summary.json`

The goal is that one run command produces both live monitoring artifacts and a final human-readable report bundle.

## Run Artifacts

### Best-checkpoint artifacts

Primary best metric is CER.

Saved artifacts:

- `weights/best_checkpoint/`
- `best_model_meta.json`

Side-by-side WER tracking:

- `weights/best_checkpoint_wer/`
- `best_wer_model_meta.json`

These are copied stable checkpoints, not fragile pointers to prunable `checkpoint-*` folders.

### Live history artifacts

- `evaluation/training_history.csv`
- `evaluation/training_history.jsonl`
- `evaluation/training_curves.svg`
- `training_summary.json`

CSV columns include:

- `step`
- `epoch`
- `loss`
- `eval_loss`
- `eval_cer`
- `eval_wer`
- `eval_exact`
- `eval_runtime_sec`
- `learning_rate`
- `grad_norm`
- `wall_time_sec`
- `rolling_step_time_sec`

### Reproducibility manifests

Training now writes row manifests so subsetted runs can be reconstructed:

- `manifests/eval_subset_manifest.jsonl`
- `manifests/train_subset_manifest.jsonl` when `train_fraction < 1.0`

This is especially important because:

- train subsetting is deterministic
- eval subsetting is deterministic
- `eval_max_rows` is now a seeded sample, not a top-row slice

## Determinism and Sampling

### Train fraction

`train_fraction < 1.0` currently means:

- one deterministic subset is chosen at run start
- the same train subset is reused for all epochs in that run

This is not equivalent to whole-dataset training. It is a fixed repeated subset unless you start a new run or change the seed.

### Eval fraction and eval max rows

Current eval behavior:

- validation rows are deterministically subsetted with `eval_fraction`
- if `eval_max_rows` is set, a deterministic seeded sample is taken
- the same eval subset is reused throughout the run

This is intentional. It keeps CER/WER curves comparable across steps and epochs.

## Evaluation Policy

### During training

Training-time evaluation is designed for progress tracking, not final paper claims.

Use it to answer:

- Is the run improving?
- Did CER stop improving?
- Is WER diverging from CER?

### After training

For paper-style comparison, use explicit evaluation on:

- `typed`
- `synthetic`

That is why `evaluate-surya-modalities` exists. It separates the two modalities rather than relying on a mixed validation aggregate.

## Metric Policy

### Best checkpoint selection

Official best checkpoint:

- `eval_cer`

Side-by-side tracked metric:

- `eval_wer`

This keeps model selection simple and defensible while preserving WER visibility.

### Text normalization

The OCR evaluation normalization now does the following before CER/WER:

- NFC Unicode normalization
- leading/trailing whitespace trim
- punctuation normalization to spaces
- internal whitespace collapse
- Ethiopic equivalence mapping for known confusing variants

This is closer to the stated paper protocol than the earlier implementation, which did not sufficiently normalize punctuation.

## Practical Stop Guidance

The system does not auto-stop on plateau.

Instead it:

- writes live CSV and summaries
- tracks `evals_since_best_cer`
- emits rank-0 warning logs when CER and WER plateau conservatively

This is deliberate. It warns the user without taking control away during expensive training.

## Problems We Hit and How They Were Fixed

### 1. Multi-GPU interrupt teardown left noisy NCCL/TCPStore shutdown and orphaned workers

Observed problem:

- Ctrl-C on DDP runs could leave residual workers
- one GPU could stay pinned at `100%`
- teardown produced repeated TCPStore/NCCL heartbeat warnings

Fixes:

- strengthened distributed stop coordination across ranks
- synchronized teardown more carefully before process-group destruction
- made device selection explicit in distributed setup and barriers
- improved interrupt and peer-stop propagation

### 2. Near-VRAM training behavior was unstable

Observed problem:

- large LoRA configs near VRAM limits would either hard OOM or wedge under DDP
- single-GPU behavior and multi-GPU behavior differed

Fixes:

- enabled allocator `expandable_segments:True` automatically in training launches
- exposed spillover policy via `--ram-spillover/--no-ram-spillover`
- preserved early guard behavior when spillover is explicitly disabled

### 3. Best checkpoint tracking was fragile

Observed problem:

- metadata could point to pruned checkpoints
- best checkpoint references were not stable enough for long runs

Fixes:

- stable copied best checkpoint under `weights/best_checkpoint`
- separate stable copied WER-best checkpoint under `weights/best_checkpoint_wer`
- root metadata files for both CER and WER bests

### 4. Training observability was too weak

Observed problem:

- no YOLO-style live metric artifact path
- user had to inspect `trainer_state.json` by hand

Fixes:

- live CSV/JSONL/SVG summaries during training
- automatic final report generation
- automatic final PNG generation
- live monitor summary support
- manifests for deterministic subset reproducibility

### 5. Eval subset behavior was biased

Observed problem:

- `eval_max_rows` used the first `N` rows, which could bias metrics

Fix:

- replaced head slicing with deterministic seeded sampling

### 6. Console eval logs were hard to read

Observed problem:

- eval metrics were logged in one dense horizontal line
- progress bar output made the result visually noisy

Fix:

- eval logging now emits a structured multi-line block through the logger

## Recommended Practice Before a Serious Run

Use a new output directory for each major experiment.

Preferred order:

1. `extract-fidel`
2. `build-surya-dataset`
3. `inspect-surya-dataset`
4. `train-surya`
5. review auto-generated CSV/report artifacts
6. run `evaluate-surya-modalities` for final typed/synthetic comparison

## Current Limitations

- training-time eval does not generate full confusion matrices unless prediction records exist
- modality-specific evaluation is still explicit post-run work, not silently inserted into every training run
- `train_fraction` still means fixed subset reuse, not rotating exclusive shards
- OCR metric normalization is improved and defensible, but not intended as an exact claim of paper-source parity

## Files To Know

- CLI: [`tools/ocr_training.py`](../ocr_training.py)
- setup guide: [setup.md](./setup.md)
- executor/runtime: [`modules/ocr_training/surya_executor.py`](../../modules/ocr_training/surya_executor.py)
- reporting: [`modules/ocr_training/surya_reports.py`](../../modules/ocr_training/surya_reports.py)
- checkpointing callbacks: [`modules/ocr_training/checkpointing.py`](../../modules/ocr_training/checkpointing.py)
- train wrapper: [`modules/ocr_training/surya_train.py`](../../modules/ocr_training/surya_train.py)
- explicit evaluation: [`modules/ocr_training/surya_eval.py`](../../modules/ocr_training/surya_eval.py)
