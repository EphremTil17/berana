# OCR Training Tool (FIDEL + Surya)

This file documents the `tools/ocr_training.py` CLI surface. The implementation lives in
`modules/ocr_training/`.

Standalone training workflow entrypoint:

```bash
PYTHONPATH=. .venv/bin/python tools/ocr_training.py --help
```

## Commands

1. Extract FIDEL typed+synthetic assets:
```bash
PYTHONPATH=. .venv/bin/python tools/ocr_training.py extract-fidel \
  --raw-root input/ocr_training/fidel/raw \
  --extracted-root input/ocr_training/fidel/extracted
```

2. Build deterministic Surya dataset:
```bash
PYTHONPATH=. .venv/bin/python tools/ocr_training.py build-surya-dataset \
  --extracted-root input/ocr_training/fidel/extracted \
  --output-root output/ocr_training_datasets \
  --dataset-name fidel_typed_synthetic
```

3. Train Surya with the adaptive planner (`auto` mode by default):
```bash
PYTHONPATH=. .venv/bin/python tools/ocr_training.py train-surya \
  --dataset-dir output/ocr_training_datasets/fidel_typed_synthetic_v01/data/hf_dataset \
  --train-fraction 0.10 \
  --mode auto \
  --logging-steps 20 \
  --verbose-epochs
```

4. Inspect token lengths, truncation pressure, and batch geometry before training:
```bash
PYTHONPATH=. .venv/bin/python tools/ocr_training.py inspect-surya-dataset \
  --dataset-dir output/ocr_training_datasets/fidel_typed_synthetic_v01/data/hf_dataset
```

5. Run explicit manual-mode training when you need fixed settings:
```bash
PYTHONPATH=. .venv/bin/python tools/ocr_training.py train-surya \
  --dataset-dir output/ocr_training_datasets/fidel_typed_synthetic_v01/data/hf_dataset \
  --mode manual \
  --finetune-strategy qlora \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --dataloader-num-workers 8
```

6. Run single-node multi-GPU training with the same command shape:
```bash
PYTHONPATH=. .venv/bin/python tools/ocr_training.py train-surya \
  --dataset-dir output/ocr_training_datasets/fidel_typed_synthetic_v01/data/hf_dataset \
  --mode auto \
  --multi-gpu
```

7. Evaluate holdout CER/WER:
```bash
PYTHONPATH=. .venv/bin/python tools/ocr_training.py evaluate-surya \
  --run-dir output/ocr_training_runs/fidel_typed_synthetic_v01 \
  --dataset-dir output/ocr_training_datasets/fidel_typed_synthetic_v01/data/hf_dataset \
  --split holdout
```

## Notes

- Handwritten sources are excluded by default.
- Split policy defaults to `train/val/holdout = 80/10/10`.
- Berana gold adapter flags are reserved and validated, but ingestion is phase-2.
- Progress bars are shown for extraction, dataset build, and evaluation.
- Training uses Hugging Face step progress plus epoch-level verbose logging (enabled by default).
- Evaluation logging includes CER, WER, and exact-match percentage.
- When `--load-best-model-at-end` is enabled, `save_steps` is auto-adjusted to a multiple of `eval_steps` if needed.
- Training now runs a GPU preflight check and aborts if foreign GPU consumers already occupy more than `10%` of VRAM.
- Training also stops on sustained extreme VRAM usage instead of continuing into likely shared-memory fallback territory.
- Dataloader defaults now use worker processes, pinned memory, and persistent workers for better host-to-GPU throughput.
- `train-surya` now defaults to adaptive `auto` mode and benchmarks admissible `QLoRA`/`LoRA` candidates before real training.
- If `--output-dir` is omitted, `train-surya` now auto-allocates a versioned run directory under `output/ocr_training_runs/`, for example `fidel_typed_synthetic_auto_v01`.
- In `auto` mode, low-level knobs such as batch size, grad accumulation, sequence length, and dataloader workers act as ceilings rather than fixed values.
- `--train-fraction` subsets only the `train` split at load time; `val` and `holdout` remain full.
- `inspect-surya-dataset` writes a JSON inspection report under the dataset run's `inspection/` directory.
- `inspect-surya-dataset` defaults to the `train` split, inspects `1024` deterministic rows, and reports local 8GB-friendly batch/grad-accum geometry by default.
- `full` finetune is manual-only; auto mode never selects it.
- Completed runs save adapter/base-model metadata so resume and `evaluate-surya` load the correct stack automatically.
- Multi-GPU v1 is single-node only and expects a homogeneous CUDA host.
- `--multi-gpu` relaunches the same CLI under `torchrun` automatically, using all visible local CUDA devices.
- `--execution-backend` and `--ddp-backend` remain available as advanced overrides, but normal runs should use the unified `--multi-gpu` flag.
- In multi-GPU mode, batch flags remain per-rank. Effective global batch is:
  `per_device_train_batch_size * gradient_accumulation_steps * gpu_count`.
- In multi-GPU auto mode, the planner evaluates per-rank settings, records effective global batch size in the selected config, and prefers configurations that stay within per-rank VRAM headroom instead of pushing into shared system memory.
- In multi-GPU mode, only rank 0 writes run metadata, best-checkpoint pointers, reports, and final root artifacts.
- Training-time eval remains available in multi-GPU mode, but `evaluate-surya` itself still runs as a standalone single-process command.
- Docker templates must provide a shared writable workspace for all ranks plus NCCL-compatible CUDA drivers.
- Adaptive runs persist:
  - `hardware_profile.json`
  - `autotune_plan.json`
  - `candidate_results.jsonl`
  - `selected_training_config.json`
