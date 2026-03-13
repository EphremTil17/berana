# OCR Training Setup Guide

This guide describes how to set up OCR training for the FIDEL typed + synthetic dataset from scratch on this repository.

It covers the practical path that was needed to get the training stack stable:

- system prerequisites
- NVIDIA and CUDA checks
- compiler and toolkit checks
- Python environment creation
- dataset download, extraction, cleanup, and build
- training launch
- what artifacts to expect while training
- what to do when a run finishes or is interrupted

## 1. System Prerequisites

The OCR training stack expects:

- Linux or WSL2 with NVIDIA GPU access
- recent NVIDIA driver
- CUDA toolkit installed and visible to the shell
- Python 3.10+
- `g++` available
- `poppler-utils` available

Recommended checks:

```bash
nvidia-smi
/usr/local/cuda/bin/nvcc --version
g++ --version
python3 --version
pdfinfo -v
```

If `nvidia-smi` does not work, stop there and fix the driver/runtime first.

## 2. Install Core System Packages

On Ubuntu or Ubuntu-based environments:

```bash
sudo apt-get update
sudo apt-get install -y build-essential g++ poppler-utils
```

If CUDA 13.0 toolkit is missing:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0
```

Then verify:

```bash
nvidia-smi
/usr/local/cuda/bin/nvcc --version
```

## 3. Bootstrap the Repository Environment

From the repo root:

```bash
chmod +x setup.sh
./setup.sh
```

What `setup.sh` does:

- creates `.venv`
- upgrades `pip`, `setuptools`, and `wheel`
- installs PyTorch CUDA 13.0 wheels from the official PyTorch cu130 index
- installs `requirements.txt`
- builds or reuses a cached CUDA wheel for `llama-cpp-python`
- installs pre-commit hooks when `.git` is present

After that:

```bash
source .venv/bin/activate
```

Quick verification:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 4. Download Raw FIDEL Data

The simplest current path is to download the source datasets directly from Hugging Face into:

```text
input/ocr_training/fidel/raw/
```

If `huggingface-cli` is not already available after `setup.sh`, verify the environment first before installing anything else. In the normal repo setup it should already be installed through the pinned requirements.

Login:

```bash
huggingface-cli login
```

Create the raw directories:

```bash
mkdir -p input/ocr_training/fidel/raw/fidel_dataset
mkdir -p input/ocr_training/fidel/raw/fidel_synthetic
```

Download the two upstream datasets:

```bash
huggingface-cli download upanzi/fidel-dataset \
  --repo-type dataset \
  --local-dir input/ocr_training/fidel/raw/fidel_dataset
```

```bash
huggingface-cli download upanzi/Fidel-synthetic \
  --repo-type dataset \
  --local-dir input/ocr_training/fidel/raw/fidel_synthetic
```

Expected source layout is handled by the extraction tool. The normal extraction flow includes:

- typed
- synthetic

and excludes handwritten data by default.

## 5. Extract the FIDEL Sources

```bash
PYTHONPATH=. python tools/ocr_training.py extract-fidel \
  --raw-root input/ocr_training/fidel/raw \
  --extracted-root input/ocr_training/fidel/extracted
```

Default behavior:

- include `typed,synthetic`
- exclude `handwritten,hdd,hdd_18,hdd_rand`

This is intentional if you are targeting the typed + synthetic regime.

## 6. Clean the Extracted Sources Before Dataset Build

Run cleanup on the extracted assets before building the Surya dataset:

```bash
PYTHONPATH=. python tools/ocr_training.py cleanup-fidel \
  --extracted-root input/ocr_training/fidel/extracted \
  --output-root input/ocr_training/fidel_cleaned
```

This creates:

- a cleaned extracted tree under `input/ocr_training/fidel_cleaned/extracted`
- a rewritten source snapshot manifest under `input/ocr_training/fidel_cleaned/manifests/source_snapshots/`
- review dumps:
  - `excluded_blank_images/`
  - `suspect_blank_images/`
  - `blank_cleanup_review/`

Important current behavior:

- confirmed blanks are always excluded
- suspect rows are excluded by default at dataset build time
- if you manually prune `suspect_blank_images/` and later pass `--include-suspect`, only the suspect review copies still left in that folder are re-included

## 7. Build the OCR Training Dataset

```bash
PYTHONPATH=. python tools/ocr_training.py build-surya-dataset \
  --extracted-root input/ocr_training/fidel_cleaned/extracted \
  --output-root output/ocr_training_datasets \
  --dataset-name fidel_typed_synthetic_clean
```

This writes a versioned dataset run such as:

```text
output/ocr_training_datasets/fidel_typed_synthetic_clean_v01/
```

Important contents:

- `data/hf_dataset/train.jsonl`
- `data/hf_dataset/val.jsonl`
- `data/hf_dataset/holdout.jsonl`
- dataset manifests under the run directory

If you want to re-include reviewed suspects that still remain in `suspect_blank_images/`, use:

```bash
PYTHONPATH=. python tools/ocr_training.py build-surya-dataset \
  --extracted-root input/ocr_training/fidel_cleaned/extracted \
  --output-root output/ocr_training_datasets \
  --dataset-name fidel_typed_synthetic_clean \
  --include-suspect
```

## 8. Verify the Built Dataset

```bash
PYTHONPATH=. python tools/ocr_training.py verify-surya-dataset \
  --dataset-dir output/ocr_training_datasets/fidel_typed_synthetic_clean_v01/data/hf_dataset \
  --output-dir output/ocr_training_datasets/fidel_typed_synthetic_clean_v01/data/hf_dataset_verification
```

This does not rewrite the built dataset. It is a sanity check that emits review artifacts for any remaining blank-image contradictions.

## 9. Inspect the Dataset Before Training

This is recommended before expensive runs:

```bash
PYTHONPATH=. python tools/ocr_training.py inspect-surya-dataset \
  --dataset-dir output/ocr_training_datasets/fidel_typed_synthetic_clean_v01/data/hf_dataset
```

Use this to sanity-check:

- token lengths
- truncation pressure
- batch geometry
- likely safe sequence-length / batch combinations

## 10. Choose a Training Strategy

### Manual mode

Use manual mode when you want explicit control of:

- batch size
- eval batch size
- grad accumulation
- LoRA vs QLoRA
- worker count
- sequence length

### Auto mode

Use auto mode when you want the planner to benchmark admissible candidates and choose a configuration.

## 11. Recommended First Full-Coverage Test Run

One epoch over the full train split is often a better baseline than repeating a tiny train fraction many times.

Example:

```bash
PYTHONPATH=. python tools/ocr_training.py train-surya \
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
  --eval-steps 100 \
  --eval-max-rows 1000 \
  --save-steps 500 \
  --resume none \
  --logging-steps 20 \
  --verbose-epochs \
  --train-fraction 1.0 \
  --num-train-epochs 1 \
  --multi-gpu
```

## 12. What the Runtime Does Automatically

You do not need to remember extra commands for the core workflow.

### During training

The runtime automatically writes:

- `evaluation/training_history.csv`
- `evaluation/training_history.jsonl`
- `evaluation/training_curves.svg`
- `training_summary.json`

### On completion or interrupt

The runtime automatically writes or refreshes:

- `evaluation/training_report.md`
- `evaluation/training_curves.png`
- `evaluation/training_curves.svg`

### Best checkpoint tracking

The runtime automatically tracks:

- best CER checkpoint:
  - `weights/best_checkpoint/`
  - `best_model_meta.json`
- best WER checkpoint:
  - `weights/best_checkpoint_wer/`
  - `best_wer_model_meta.json`

### Reproducibility manifests

The runtime automatically writes:

- `manifests/eval_subset_manifest.jsonl`
- `manifests/train_subset_manifest.jsonl` when `train_fraction < 1.0`

## 13. Important Current Runtime Behavior

### Allocator setup

You do not need to manually export `PYTORCH_ALLOC_CONF=expandable_segments:True`.

The training CLI now applies that automatically.

### Eval sampling

Training evals are deterministic within a run.

That means:

- the eval subset is chosen once
- `eval_max_rows` now uses a seeded sample, not top rows
- the same eval subset is reused across epochs

### Train fraction

`train_fraction < 1.0` still means a fixed deterministic train subset for that run.

It does not rotate shards across epochs.

### RAM spillover

Spillover is currently allowed by default.

You can force the old guard behavior with:

```bash
--no-ram-spillover
```

## 14. Monitoring a Live Run

The main live artifact is:

```text
evaluation/training_history.csv
```

This is the easiest file to inspect in Excel, Python, or a notebook while the run is active.

Useful fields:

- `loss`
- `eval_loss`
- `eval_cer`
- `eval_wer`
- `eval_runtime_sec`
- `wall_time_sec`
- `rolling_step_time_sec`

The runtime also logs:

- best CER so far
- best WER so far
- evals since best CER

## 15. Final Evaluation for Paper-Style Comparison

Use explicit evaluation after training finishes.

Standard mixed evaluation:

```bash
PYTHONPATH=. python tools/ocr_training.py evaluate-surya \
  --run-dir output/ocr_training_runs/fidel_typed_synthetic_clean_5090_lora_full1ep_v01 \
  --dataset-dir output/ocr_training_datasets/fidel_typed_synthetic_clean_v01/data/hf_dataset \
  --split holdout
```

Typed + synthetic separate evaluation:

```bash
PYTHONPATH=. python tools/ocr_training.py evaluate-surya-modalities \
  --run-dir output/ocr_training_runs/fidel_typed_synthetic_clean_5090_lora_full1ep_v01 \
  --dataset-dir output/ocr_training_datasets/fidel_typed_synthetic_clean_v01/data/hf_dataset \
  --split holdout \
  --modalities typed,synthetic
```

This is the command you should use when comparing against typed/synthetic rows reported in the dataset paper.

## 16. Common Problems We Hit

### Multi-GPU interrupt left GPUs pinned

Observed:

- Ctrl-C caused noisy NCCL/TCPStore teardown
- stale worker processes could remain on one GPU

Fixes now in place:

- coordinated stop propagation
- stronger teardown handling
- explicit distributed device selection

### Near-OOM runs wedged or hard-OOMed

Observed:

- high-memory runs behaved differently across hosts and between single-GPU and DDP

Fixes now in place:

- allocator expandable segments enabled by default
- spillover policy surfaced explicitly
- live artifacts and warnings added so plateaued runs are easier to terminate manually

### Best checkpoint tracking was misleading

Observed:

- best metadata could point to pruned checkpoints

Fixes now in place:

- best checkpoints are copied into stable `weights/` directories

### Eval subset was biased

Observed:

- `eval_max_rows` used the first rows

Fix now in place:

- seeded deterministic sampling

## 17. Before You Launch a Real Run

Checklist:

1. `nvidia-smi` works.
2. `nvcc --version` works.
3. `g++ --version` works.
4. `.venv` exists and activates.
5. `torch.cuda.is_available()` is `True`.
6. raw FIDEL data is downloaded.
7. raw FIDEL data is extracted.
8. extracted FIDEL data is cleaned.
9. Surya dataset is built and optionally verified.
10. `inspect-surya-dataset` looks sane.
11. you are using a fresh output directory.
12. you know whether you want:
   - full-data one-epoch baseline
   - small fixed-fraction exploratory run
   - auto planner vs manual mode

## 18. Minimal End-to-End Sequence

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate

huggingface-cli login
mkdir -p input/ocr_training/fidel/raw/fidel_dataset
mkdir -p input/ocr_training/fidel/raw/fidel_synthetic

huggingface-cli download upanzi/fidel-dataset \
  --repo-type dataset \
  --local-dir input/ocr_training/fidel/raw/fidel_dataset

huggingface-cli download upanzi/Fidel-synthetic \
  --repo-type dataset \
  --local-dir input/ocr_training/fidel/raw/fidel_synthetic

PYTHONPATH=. python tools/ocr_training.py extract-fidel \
  --raw-root input/ocr_training/fidel/raw \
  --extracted-root input/ocr_training/fidel/extracted

PYTHONPATH=. python tools/ocr_training.py cleanup-fidel \
  --extracted-root input/ocr_training/fidel/extracted \
  --output-root input/ocr_training/fidel_cleaned

PYTHONPATH=. python tools/ocr_training.py build-surya-dataset \
  --extracted-root input/ocr_training/fidel_cleaned/extracted \
  --output-root output/ocr_training_datasets \
  --dataset-name fidel_typed_synthetic_clean

PYTHONPATH=. python tools/ocr_training.py verify-surya-dataset \
  --dataset-dir output/ocr_training_datasets/fidel_typed_synthetic_clean_v01/data/hf_dataset \
  --output-dir output/ocr_training_datasets/fidel_typed_synthetic_clean_v01/data/hf_dataset_verification

PYTHONPATH=. python tools/ocr_training.py inspect-surya-dataset \
  --dataset-dir output/ocr_training_datasets/fidel_typed_synthetic_clean_v01/data/hf_dataset

PYTHONPATH=. python tools/ocr_training.py train-surya \
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
  --eval-steps 100 \
  --eval-max-rows 1000 \
  --save-steps 500 \
  --resume none \
  --logging-steps 20 \
  --verbose-epochs \
  --train-fraction 1.0 \
  --num-train-epochs 1 \
  --multi-gpu

PYTHONPATH=. python tools/ocr_training.py evaluate-surya-modalities \
  --run-dir output/ocr_training_runs/fidel_typed_synthetic_clean_5090_lora_full1ep_v01 \
  --dataset-dir output/ocr_training_datasets/fidel_typed_synthetic_clean_v01/data/hf_dataset \
  --split holdout \
  --modalities typed,synthetic
```
