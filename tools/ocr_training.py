from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from modules.ocr_training.adapters.berana_gold import validate_berana_gold_inputs
from modules.ocr_training.distributed.context import torchrun_is_active
from modules.ocr_training.fidel_extract import extract_fidel
from modules.ocr_training.schemas import SplitConfig, SuryaTrainConfig, TrainMode
from modules.ocr_training.surya_dataset import build_surya_dataset
from modules.ocr_training.surya_inspect import inspect_surya_dataset
from modules.ocr_training.surya_reports import monitor_training_run, write_training_report_bundle
from modules.ocr_training.surya_train import (
    evaluate_surya_checkpoint,
    evaluate_surya_modalities,
    run_surya_finetune,
)
from utils.logger import get_logger
from utils.run_registry import next_versioned_dir

app = typer.Typer(help="Standalone OCR training toolchain (FIDEL + Surya).", no_args_is_help=True)
log = get_logger("OCRTrainingCLI")


def _csv_to_set(values: str) -> set[str]:
    return {item.strip().lower() for item in values.split(",") if item.strip()}


def _csv_to_int_list(values: str) -> list[int]:
    parsed = [int(item.strip()) for item in values.split(",") if item.strip()]
    if not parsed:
        raise typer.BadParameter("Expected at least one integer value.")
    return parsed


def _strip_version_suffix(name: str) -> str:
    """Strip a trailing `_vNN` suffix from a versioned run directory name."""
    return re.sub(r"_v\d+$", "", name.strip())


def _dataset_run_stem(dataset_dir: Path) -> str:
    """Infer the dataset run stem from a generated `hf_dataset` directory path."""
    normalized = dataset_dir.resolve()
    if normalized.name == "hf_dataset" and normalized.parent.name == "data":
        return _strip_version_suffix(normalized.parent.parent.name)
    return _strip_version_suffix(normalized.name)


def _resolve_train_output_dir(
    *,
    dataset_dir: Path,
    output_dir: Path | None,
    mode: TrainMode,
) -> Path:
    """Resolve the explicit or auto-versioned training output directory."""
    if output_dir is not None:
        return output_dir
    dataset_stem = _dataset_run_stem(dataset_dir)
    run_stem = f"{dataset_stem}_{mode.value}"
    return next_versioned_dir(Path("output/ocr_training_runs"), run_stem)


def _resolve_tool_eval_output_dir(*, run_dir: Path, output_dir: Path | None) -> Path:
    """Resolve the explicit or auto-versioned output directory for explicit tool evaluation."""
    if output_dir is not None:
        return output_dir
    return next_versioned_dir(run_dir, "tool_evaluation")


def _cli_is_rank_zero() -> bool:
    rank = os.environ.get("RANK", "0").strip()
    return not rank.isdigit() or int(rank) == 0


def _visible_cuda_device_count() -> int:
    """Infer the number of CUDA devices visible to the current process."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        tokens = [token.strip() for token in visible_devices.split(",") if token.strip()]
        if tokens:
            return len(tokens)
    try:
        import torch
    except ModuleNotFoundError:
        return 0
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def _torchrun_entrypoint() -> list[str]:
    """Return the launcher prefix for a torchrun-style distributed execution."""
    torchrun_path = shutil.which("torchrun")
    if torchrun_path:
        return [torchrun_path]
    return [sys.executable, "-m", "torch.distributed.run"]


def _merge_pytorch_alloc_conf(existing: str | None) -> str:
    """Ensure the allocator config enables expandable segments without dropping user settings."""
    if existing is None or not existing.strip():
        return "expandable_segments:True"
    entries: list[str] = []
    seen_expandable = False
    for token in existing.split(","):
        normalized = token.strip()
        if not normalized:
            continue
        key, _, _value = normalized.partition(":")
        if key.strip() == "expandable_segments":
            entries.append("expandable_segments:True")
            seen_expandable = True
            continue
        entries.append(normalized)
    if not seen_expandable:
        entries.append("expandable_segments:True")
    return ",".join(entries)


def _training_launch_env() -> dict[str, str]:
    """Return the launcher environment with allocator settings suited for high-pressure training."""
    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = _merge_pytorch_alloc_conf(env.get("PYTORCH_ALLOC_CONF"))
    return env


def _apply_training_env_defaults() -> None:
    """Apply allocator defaults to the current process before training imports initialize CUDA."""
    os.environ["PYTORCH_ALLOC_CONF"] = _merge_pytorch_alloc_conf(
        os.environ.get("PYTORCH_ALLOC_CONF")
    )


def _strip_flag(argv: list[str], flag: str) -> list[str]:
    """Return argv with one boolean flag removed."""
    return [value for value in argv if value != flag]


def _ensure_option(argv: list[str], *, option: str, value: str) -> list[str]:
    """Append one CLI option/value pair if the option is not already present."""
    for token in argv:
        if token == option or token.startswith(f"{option}="):
            return argv
    return [*argv, option, value]


def _build_multi_gpu_launch_command(
    *,
    argv: list[str],
    nproc_per_node: int,
    resolved_output_dir: Path,
) -> list[str]:
    """Construct the internal torchrun relaunch command for multi-GPU training."""
    forwarded = _strip_flag(_strip_flag(argv, "--multi-gpu"), "--no-multi-gpu")
    forwarded = _ensure_option(
        forwarded,
        option="--output-dir",
        value=str(resolved_output_dir),
    )
    return [
        *_torchrun_entrypoint(),
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        str(Path(sys.argv[0]).resolve()),
        *forwarded,
    ]


def _build_multi_gpu_eval_command(
    *,
    argv: list[str],
    nproc_per_node: int,
) -> list[str]:
    """Construct the internal torchrun relaunch command for multi-GPU evaluation."""
    forwarded = _strip_flag(_strip_flag(argv, "--multi-gpu"), "--single-gpu")
    return [
        *_torchrun_entrypoint(),
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        str(Path(sys.argv[0]).resolve()),
        *forwarded,
    ]


def _maybe_relaunch_multi_gpu(
    *,
    multi_gpu: bool,
    execution_backend: str,
    resolved_output_dir: Path,
) -> None:
    """Relaunch the current train command under torchrun when requested."""
    if not multi_gpu or torchrun_is_active():
        return
    normalized_backend = execution_backend.strip().lower()
    if normalized_backend == "single":
        raise typer.BadParameter("`--multi-gpu` is incompatible with `--execution-backend single`.")
    nproc_per_node = _visible_cuda_device_count()
    if nproc_per_node < 2:
        raise typer.BadParameter(
            "`--multi-gpu` requires at least 2 visible CUDA devices on this host."
        )
    command = _build_multi_gpu_launch_command(
        argv=sys.argv[1:],
        nproc_per_node=nproc_per_node,
        resolved_output_dir=resolved_output_dir,
    )
    if _cli_is_rank_zero():
        log.info(
            "Launching multi-GPU Surya training with %s ranks via: %s",
            nproc_per_node,
            shlex.join(command),
        )
    completed = subprocess.run(command, check=False, env=_training_launch_env())
    raise typer.Exit(code=completed.returncode)


def _maybe_relaunch_multi_gpu_eval(*, multi_gpu: bool) -> None:
    """Relaunch the current evaluation command under torchrun when requested."""
    if not multi_gpu or torchrun_is_active():
        return
    nproc_per_node = _visible_cuda_device_count()
    if nproc_per_node < 2:
        raise typer.BadParameter(
            "`--multi-gpu` requires at least 2 visible CUDA devices on this host."
        )
    command = _build_multi_gpu_eval_command(argv=sys.argv[1:], nproc_per_node=nproc_per_node)
    if _cli_is_rank_zero():
        log.info(
            "Launching multi-GPU Surya evaluation with %s ranks via: %s",
            nproc_per_node,
            shlex.join(command),
        )
    completed = subprocess.run(command, check=False, env=_training_launch_env())
    raise typer.Exit(code=completed.returncode)


@app.command("extract-fidel")
def cli_extract_fidel(
    raw_root: Annotated[
        Path,
        typer.Option(
            "--raw-root", help="Root containing fidel_dataset/ and fidel_synthetic/ raw assets."
        ),
    ] = Path("input/ocr_training/fidel/raw"),
    extracted_root: Annotated[
        Path,
        typer.Option("--extracted-root", help="Output extracted root path."),
    ] = Path("input/ocr_training/fidel/extracted"),
    include_types: Annotated[
        str,
        typer.Option("--include-types", help="Comma-separated include types."),
    ] = "typed,synthetic",
    exclude_types: Annotated[
        str,
        typer.Option("--exclude-types", help="Comma-separated exclude types."),
    ] = "handwritten,hdd,hdd_18,hdd_rand",
    allow_missing_rate: Annotated[
        float,
        typer.Option("--allow-missing-rate", help="Allowed max missing-rate before hard failure."),
    ] = 0.005,
    workers: Annotated[
        int,
        typer.Option("--workers", help="Reserved worker count for future extraction parallelism."),
    ] = 8,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite/--no-overwrite", help="Allow overwriting existing mismatched targets."
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run/--no-dry-run", help="Plan extraction without writing files."),
    ] = False,
):
    """Stream-extract FIDEL archives into canonical typed/synthetic buckets."""
    log.info("Starting FIDEL extraction raw_root=%s extracted_root=%s", raw_root, extracted_root)
    try:
        result = extract_fidel(
            raw_root=raw_root,
            extracted_root=extracted_root,
            include_types=_csv_to_set(include_types),
            exclude_types=_csv_to_set(exclude_types),
            allow_missing_rate=allow_missing_rate,
            workers=workers,
            overwrite=overwrite,
            dry_run=dry_run,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("extract-fidel failed: %s", exc)
        raise typer.Exit(code=1) from exc

    log.info("extract-fidel complete snapshot=%s", result["source_snapshot"])


@app.command("build-surya-dataset")
def cli_build_surya_dataset(
    extracted_root: Annotated[
        Path,
        typer.Option("--extracted-root", help="Extracted data root path."),
    ] = Path("input/ocr_training/fidel/extracted"),
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Output root for versioned dataset runs."),
    ] = Path("output/ocr_training_datasets"),
    dataset_name: Annotated[
        str,
        typer.Option("--dataset-name", help="Dataset run key/stem for versioned output."),
    ] = "fidel_typed_synthetic",
    train_ratio: Annotated[float, typer.Option("--train-ratio", help="Train split ratio.")] = 0.80,
    val_ratio: Annotated[float, typer.Option("--val-ratio", help="Validation split ratio.")] = 0.10,
    holdout_ratio: Annotated[
        float,
        typer.Option("--holdout-ratio", help="Holdout split ratio."),
    ] = 0.10,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic split seed.")] = 42,
    strict_page_isolation: Annotated[
        bool,
        typer.Option(
            "--strict-page-isolation/--allow-page-overlap",
            help="Disallow cross-split page-group overlap.",
        ),
    ] = False,
    extra_manifest: Annotated[
        Path | None,
        typer.Option(
            "--extra-manifest", help="Optional Berana gold adapter manifest path (scaffolded)."
        ),
    ] = None,
    extra_images_root: Annotated[
        Path | None,
        typer.Option("--extra-images-root", help="Optional Berana gold images root (scaffolded)."),
    ] = None,
    extra_weight: Annotated[
        float,
        typer.Option("--extra-weight", help="Optional Berana gold mix weight (scaffolded)."),
    ] = 0.30,
):
    """Build deterministic Surya-compatible local dataset artifacts."""
    split_config = SplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        holdout_ratio=holdout_ratio,
        seed=seed,
        strict_page_isolation=strict_page_isolation,
    )

    try:
        validate_berana_gold_inputs(
            extra_manifest=extra_manifest,
            extra_images_root=extra_images_root,
            extra_weight=extra_weight,
        )
        run_dir = build_surya_dataset(
            extracted_root=extracted_root,
            output_root=output_root,
            dataset_name=dataset_name,
            split_config=split_config,
            extra_manifest=extra_manifest,
            extra_images_root=extra_images_root,
            extra_weight=extra_weight,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("build-surya-dataset failed: %s", exc)
        raise typer.Exit(code=1) from exc

    log.info("build-surya-dataset complete run_dir=%s", run_dir)


@app.command("train-surya")
def cli_train_surya(
    dataset_dir: Annotated[
        Path,
        typer.Option("--dataset-dir", help="Path to generated hf_dataset directory."),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir", help="Training output run directory. Defaults to a versioned auto path."
        ),
    ] = None,
    run_key: Annotated[
        str,
        typer.Option("--run-key", help="Registry run key/stem."),
    ] = "fidel_typed_synthetic",
    pretrained_checkpoint_path: Annotated[
        str,
        typer.Option("--pretrained-checkpoint-path", help="Optional Surya checkpoint init path."),
    ] = "",
    mode: Annotated[
        str,
        typer.Option("--mode", help="auto|manual adaptive planner mode."),
    ] = "auto",
    seed: Annotated[
        int,
        typer.Option("--seed", help="Deterministic seed for train subsetting and planning."),
    ] = 42,
    train_fraction: Annotated[
        float,
        typer.Option(
            "--train-fraction",
            help="Deterministic fraction of the train split to use at load time.",
        ),
    ] = 1.0,
    eval_fraction: Annotated[
        float,
        typer.Option(
            "--eval-fraction",
            help="Deterministic fraction of the validation split to use during training evaluation.",
        ),
    ] = 1.0,
    eval_max_rows: Annotated[
        int | None,
        typer.Option(
            "--eval-max-rows",
            help="Optional cap on validation rows used during training evaluation after subsetting.",
        ),
    ] = None,
    planning_budget_minutes: Annotated[
        int,
        typer.Option("--planning-budget-minutes"),
    ] = 3,
    target_vram_utilization: Annotated[
        float,
        typer.Option("--target-vram-utilization"),
    ] = 0.9375,
    strategy_allowlist: Annotated[
        str,
        typer.Option("--strategy-allowlist", help="Comma-separated adaptive strategy allowlist."),
    ] = "qlora,lora",
    max_replans: Annotated[
        int,
        typer.Option("--max-replans"),
    ] = 1,
    multi_gpu: Annotated[
        bool,
        typer.Option(
            "--multi-gpu/--single-gpu",
            help="Use all visible local GPUs and relaunch under torchrun automatically.",
        ),
    ] = False,
    execution_backend: Annotated[
        str,
        typer.Option("--execution-backend", help="Advanced override: auto|single|ddp"),
    ] = "auto",
    ddp_backend: Annotated[
        str,
        typer.Option("--ddp-backend", help="Advanced override for distributed backend: nccl|gloo"),
    ] = "nccl",
    per_device_train_batch_size: Annotated[
        int | None,
        typer.Option("--per-device-train-batch-size"),
    ] = None,
    per_device_eval_batch_size: Annotated[
        int | None,
        typer.Option("--per-device-eval-batch-size"),
    ] = None,
    gradient_accumulation_steps: Annotated[
        int | None,
        typer.Option("--gradient-accumulation-steps"),
    ] = None,
    finetune_strategy: Annotated[
        str | None,
        typer.Option("--finetune-strategy", help="qlora|lora|full"),
    ] = None,
    lora_rank: Annotated[
        int,
        typer.Option("--lora-rank"),
    ] = 16,
    lora_alpha: Annotated[
        int,
        typer.Option("--lora-alpha"),
    ] = 32,
    lora_dropout: Annotated[
        float,
        typer.Option("--lora-dropout"),
    ] = 0.05,
    dataloader_num_workers: Annotated[
        int | None,
        typer.Option("--dataloader-num-workers"),
    ] = None,
    dataloader_pin_memory: Annotated[
        bool,
        typer.Option("--dataloader-pin-memory/--no-dataloader-pin-memory"),
    ] = True,
    dataloader_persistent_workers: Annotated[
        bool,
        typer.Option("--dataloader-persistent-workers/--no-dataloader-persistent-workers"),
    ] = True,
    dataloader_prefetch_factor: Annotated[
        int,
        typer.Option("--dataloader-prefetch-factor"),
    ] = 2,
    fp16: Annotated[bool, typer.Option("--fp16/--no-fp16")] = True,
    gradient_checkpointing: Annotated[
        bool,
        typer.Option("--gradient-checkpointing/--no-gradient-checkpointing"),
    ] = True,
    max_sequence_length: Annotated[int | None, typer.Option("--max-sequence-length")] = None,
    num_train_epochs: Annotated[float, typer.Option("--num-train-epochs")] = 8,
    learning_rate: Annotated[float, typer.Option("--learning-rate")] = 2e-5,
    eval_steps: Annotated[int | None, typer.Option("--eval-steps")] = None,
    logging_steps: Annotated[int, typer.Option("--logging-steps")] = 10,
    save_steps: Annotated[int, typer.Option("--save-steps")] = 500,
    save_total_limit: Annotated[int, typer.Option("--save-total-limit")] = 4,
    load_best_model_at_end: Annotated[
        bool,
        typer.Option("--load-best-model-at-end/--no-load-best-model-at-end"),
    ] = True,
    metric_for_best_model: Annotated[str, typer.Option("--metric-for-best-model")] = "cer",
    greater_is_better: Annotated[
        bool, typer.Option("--greater-is-better/--lower-is-better")
    ] = False,
    resume: Annotated[str, typer.Option("--resume", help="auto|latest|none")] = "auto",
    foreign_vram_threshold_ratio: Annotated[
        float,
        typer.Option("--foreign-vram-threshold-ratio"),
    ] = 0.10,
    abort_vram_usage_ratio: Annotated[
        float,
        typer.Option("--abort-vram-usage-ratio"),
    ] = 0.97,
    allow_ram_spillover: Annotated[
        bool,
        typer.Option(
            "--ram-spillover/--no-ram-spillover",
            help="Allow GPU workloads to spill into shared/system memory instead of aborting at the VRAM guard threshold.",
        ),
    ] = True,
    verbose_epochs: Annotated[
        bool,
        typer.Option(
            "--verbose-epochs/--quiet-epochs",
            help="Emit epoch-level verbose training logs.",
        ),
    ] = True,
):
    """Run Surya finetuning with manual control or adaptive hardware-aware planning."""
    try:
        normalized_mode = TrainMode(mode.strip().lower())
    except ValueError as exc:
        raise typer.BadParameter("--mode must be one of: auto, manual") from exc
    if normalized_mode == TrainMode.AUTO and finetune_strategy == "full":
        raise typer.BadParameter("`full` finetuning is manual-only. Use `--mode manual`.")
    if normalized_mode == TrainMode.AUTO and _cli_is_rank_zero():
        log.info(
            "Auto mode treats batch, grad accumulation, sequence length, and worker flags as planner ceilings."
        )
    resolved_output_dir = _resolve_train_output_dir(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        mode=normalized_mode,
    )
    if output_dir is None and _cli_is_rank_zero():
        log.info("Resolved training output_dir=%s", resolved_output_dir)
    _apply_training_env_defaults()
    _maybe_relaunch_multi_gpu(
        multi_gpu=multi_gpu,
        execution_backend=execution_backend,
        resolved_output_dir=resolved_output_dir,
    )

    cfg = SuryaTrainConfig(
        mode=normalized_mode,
        seed=seed,
        train_fraction=train_fraction,
        eval_fraction=eval_fraction,
        eval_max_rows=eval_max_rows,
        planning_budget_minutes=planning_budget_minutes,
        target_vram_utilization=target_vram_utilization,
        strategy_allowlist=strategy_allowlist,
        max_replans=max_replans,
        execution_backend=execution_backend,
        ddp_backend=ddp_backend,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        finetune_strategy=finetune_strategy,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        dataloader_persistent_workers=dataloader_persistent_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
        max_sequence_length=max_sequence_length,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        resume=resume,
        verbose_epochs=verbose_epochs,
        foreign_vram_threshold_ratio=foreign_vram_threshold_ratio,
        abort_vram_usage_ratio=abort_vram_usage_ratio,
        allow_ram_spillover=allow_ram_spillover,
    )

    try:
        result = run_surya_finetune(
            run_key=run_key,
            dataset_dir=dataset_dir,
            output_dir=resolved_output_dir,
            config=cfg,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("train-surya failed: %s", exc)
        raise typer.Exit(code=1) from exc

    if _cli_is_rank_zero():
        log.info("train-surya complete status=%s", result["status"])


@app.command("evaluate-surya")
def cli_evaluate_surya(
    run_dir: Annotated[
        Path,
        typer.Option("--run-dir", help="Directory containing finetuned Surya checkpoints."),
    ],
    dataset_dir: Annotated[
        Path,
        typer.Option("--dataset-dir", help="Path to generated hf_dataset directory."),
    ],
    run_key: Annotated[
        str,
        typer.Option("--run-key", help="Registry run key/stem."),
    ] = "fidel_typed_synthetic",
    split: Annotated[str, typer.Option("--split", help="Dataset split to evaluate.")] = "holdout",
    eval_fraction: Annotated[
        float,
        typer.Option(
            "--eval-fraction", help="Deterministic fraction of the requested split to evaluate."
        ),
    ] = 1.0,
    eval_batch_size: Annotated[
        int,
        typer.Option("--eval-batch-size", help="Batch size for Surya inference during evaluation."),
    ] = 8,
    max_rows: Annotated[
        int | None,
        typer.Option("--max-rows", help="Optional cap on evaluated rows after split subsetting."),
    ] = None,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Deterministic seed for evaluation subsetting."),
    ] = 42,
    checkpoint_target: Annotated[
        str,
        typer.Option(
            "--checkpoint-target",
            help="Which checkpoint to evaluate: best_cer, best_wer, or latest.",
        ),
    ] = "best_cer",
    checkpoint_path: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint-path",
            help="Optional explicit checkpoint/adapter directory to evaluate.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Optional output directory for evaluation artifacts. Defaults to <run-dir>/tool_evaluation.",
        ),
    ] = None,
    multi_gpu: Annotated[
        bool,
        typer.Option(
            "--multi-gpu/--single-gpu",
            help="Use all visible local GPUs and relaunch under torchrun automatically.",
        ),
    ] = False,
):
    """Evaluate Surya checkpoint on untouched split and emit CER/WER summary artifacts."""
    split = split.strip().lower()
    if split not in {"holdout", "val", "train"}:
        raise typer.BadParameter("--split must be one of: train, val, holdout")
    if not 0 < eval_fraction <= 1.0:
        raise typer.BadParameter("--eval-fraction must be in the interval (0, 1].")
    if eval_batch_size < 1:
        raise typer.BadParameter("--eval-batch-size must be >= 1.")
    if max_rows is not None and max_rows < 1:
        raise typer.BadParameter("--max-rows must be >= 1 when provided.")
    checkpoint_target = checkpoint_target.strip().lower()
    if checkpoint_target not in {"best_cer", "best_wer", "latest"}:
        raise typer.BadParameter("--checkpoint-target must be one of: best_cer, best_wer, latest")

    resolved_output_dir = _resolve_tool_eval_output_dir(run_dir=run_dir, output_dir=output_dir)
    if output_dir is None and _cli_is_rank_zero():
        log.info("Resolved tool evaluation output_dir=%s", resolved_output_dir)

    _apply_training_env_defaults()
    _maybe_relaunch_multi_gpu_eval(multi_gpu=multi_gpu)

    try:
        summary = evaluate_surya_checkpoint(
            run_key=run_key,
            run_dir=run_dir,
            dataset_dir=dataset_dir,
            split=split,
            eval_fraction=eval_fraction,
            eval_batch_size=eval_batch_size,
            max_rows=max_rows,
            seed=seed,
            checkpoint_target=checkpoint_target,
            checkpoint_path=checkpoint_path,
            output_dir=resolved_output_dir,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("evaluate-surya failed: %s", exc)
        raise typer.Exit(code=1) from exc

    if summary.get("status") == "completed_nonzero_rank":
        return
    log.info(
        "evaluate-surya complete split=%s rows=%d cer=%.4f wer=%.4f",
        split,
        summary["num_rows"],
        summary["mean_cer"],
        summary["mean_wer"],
    )


@app.command("evaluate-surya-modalities")
def cli_evaluate_surya_modalities(
    run_dir: Annotated[
        Path,
        typer.Option("--run-dir", help="Directory containing finetuned Surya checkpoints."),
    ],
    dataset_dir: Annotated[
        Path,
        typer.Option("--dataset-dir", help="Path to generated hf_dataset directory."),
    ],
    run_key: Annotated[
        str,
        typer.Option("--run-key", help="Registry run key/stem."),
    ] = "fidel_typed_synthetic",
    split: Annotated[str, typer.Option("--split", help="Dataset split to evaluate.")] = "holdout",
    eval_fraction: Annotated[
        float,
        typer.Option("--eval-fraction", help="Deterministic fraction of the requested split."),
    ] = 1.0,
    eval_batch_size: Annotated[
        int,
        typer.Option("--eval-batch-size", help="Batch size for Surya inference during evaluation."),
    ] = 8,
    max_rows: Annotated[
        int | None,
        typer.Option("--max-rows", help="Optional cap on evaluated rows after split subsetting."),
    ] = None,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Deterministic seed for evaluation subsetting."),
    ] = 42,
    checkpoint_target: Annotated[
        str,
        typer.Option(
            "--checkpoint-target",
            help="Which checkpoint to evaluate: best_cer, best_wer, or latest.",
        ),
    ] = "best_cer",
    checkpoint_path: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint-path",
            help="Optional explicit checkpoint/adapter directory to evaluate.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Optional output directory for evaluation artifacts. Defaults to <run-dir>/tool_evaluation.",
        ),
    ] = None,
    multi_gpu: Annotated[
        bool,
        typer.Option(
            "--multi-gpu/--single-gpu",
            help="Use all visible local GPUs and relaunch under torchrun automatically.",
        ),
    ] = False,
    modalities: Annotated[
        str,
        typer.Option(
            "--modalities",
            help="Comma-separated modalities to evaluate independently, typically typed,synthetic.",
        ),
    ] = "typed,synthetic",
):
    """Evaluate a run separately across typed/synthetic modalities."""
    checkpoint_target = checkpoint_target.strip().lower()
    if checkpoint_target not in {"best_cer", "best_wer", "latest"}:
        raise typer.BadParameter("--checkpoint-target must be one of: best_cer, best_wer, latest")
    resolved_output_dir = _resolve_tool_eval_output_dir(run_dir=run_dir, output_dir=output_dir)
    if output_dir is None and _cli_is_rank_zero():
        log.info("Resolved tool evaluation output_dir=%s", resolved_output_dir)
    _apply_training_env_defaults()
    _maybe_relaunch_multi_gpu_eval(multi_gpu=multi_gpu)

    try:
        summary = evaluate_surya_modalities(
            run_key=run_key,
            run_dir=run_dir,
            dataset_dir=dataset_dir,
            split=split.strip().lower(),
            eval_fraction=eval_fraction,
            eval_batch_size=eval_batch_size,
            max_rows=max_rows,
            seed=seed,
            modalities=sorted(_csv_to_set(modalities)),
            checkpoint_target=checkpoint_target,
            checkpoint_path=checkpoint_path,
            output_dir=resolved_output_dir,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("evaluate-surya-modalities failed: %s", exc)
        raise typer.Exit(code=1) from exc

    if summary.get("status") == "completed_nonzero_rank":
        return
    log.info(
        "evaluate-surya-modalities complete split=%s modalities=%s",
        split,
        ",".join(sorted(summary["modalities"])),
    )


@app.command("visualize-surya-run")
def cli_visualize_surya_run(
    run_dir: Annotated[
        Path,
        typer.Option(
            "--run-dir", help="Directory containing one finished or active Surya training run."
        ),
    ],
    split: Annotated[
        str | None,
        typer.Option(
            "--split",
            help="Optional evaluated split name used to resolve predictions_<split>.jsonl for confusion artifacts.",
        ),
    ] = None,
    predictions_path: Annotated[
        Path | None,
        typer.Option(
            "--predictions-path",
            help="Optional explicit predictions_*.jsonl path to use for confusion artifacts.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Optional output directory override for generated report artifacts. Defaults to <run-dir>/evaluation.",
        ),
    ] = None,
):
    """Generate a read-only report bundle with training curves and optional confusion artifacts."""
    try:
        artifacts = write_training_report_bundle(
            run_dir=run_dir,
            output_dir=output_dir,
            split=split.strip().lower() if split else None,
            predictions_path=predictions_path,
        )
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        log.error("visualize-surya-run failed: %s", exc)
        raise typer.Exit(code=1) from exc

    log.info(
        "visualize-surya-run complete report=%s curves=%s",
        artifacts.get("training_report_md"),
        artifacts.get("training_curves_svg"),
    )


@app.command("monitor-surya-run")
def cli_monitor_surya_run(
    run_dir: Annotated[
        Path,
        typer.Option("--run-dir", help="Directory containing one active or completed Surya run."),
    ],
):
    """Print a concise live monitor summary from training artifacts."""
    try:
        summary = monitor_training_run(run_dir)
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        log.error("monitor-surya-run failed: %s", exc)
        raise typer.Exit(code=1) from exc

    log.info(
        "monitor-surya-run selection_metric=%s latest_eval_step=%s latest_cer=%s latest_wer=%s best_cer=%s@%s best_wer=%s@%s evals_since_best_cer=%s csv=%s",
        summary.get("selection_metric"),
        summary.get("latest_eval_step"),
        summary.get("latest_eval_cer"),
        summary.get("latest_eval_wer"),
        summary.get("best_cer_value"),
        summary.get("best_cer_step"),
        summary.get("best_wer_value"),
        summary.get("best_wer_step"),
        summary.get("evals_since_best_cer"),
        summary.get("training_history_csv"),
    )
    typer.echo(
        "monitor-surya-run "
        f"selection_metric={summary.get('selection_metric')} "
        f"latest_eval_step={summary.get('latest_eval_step')} "
        f"latest_cer={summary.get('latest_eval_cer')} "
        f"latest_wer={summary.get('latest_eval_wer')} "
        f"best_cer={summary.get('best_cer_value')}@{summary.get('best_cer_step')} "
        f"best_wer={summary.get('best_wer_value')}@{summary.get('best_wer_step')} "
        f"evals_since_best_cer={summary.get('evals_since_best_cer')}"
    )


@app.command("inspect-surya-dataset")
def cli_inspect_surya_dataset(
    dataset_dir: Annotated[
        Path,
        typer.Option("--dataset-dir", help="Path to generated hf_dataset directory."),
    ],
    split: Annotated[
        str,
        typer.Option("--split", help="Dataset split to inspect."),
    ] = "train",
    sample_size: Annotated[
        int,
        typer.Option(
            "--sample-size", help="Deterministic number of rows to inspect; 0 means full split."
        ),
    ] = 1024,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Deterministic seed for row sampling."),
    ] = 42,
    train_fraction: Annotated[
        float,
        typer.Option("--train-fraction", help="Optional train-only load-time fraction to inspect."),
    ] = 1.0,
    max_sequence_lengths: Annotated[
        str,
        typer.Option(
            "--max-sequence-lengths", help="Comma-separated analytical sequence caps to measure."
        ),
    ] = "1024,896,768",
    per_device_batch_sizes: Annotated[
        str,
        typer.Option(
            "--per-device-batch-sizes", help="Comma-separated micro-batch sizes for step geometry."
        ),
    ] = "1,2",
    gradient_accumulation_steps: Annotated[
        str,
        typer.Option(
            "--gradient-accumulation-steps",
            help="Comma-separated grad-accum values for step geometry.",
        ),
    ] = "4,8",
    pretrained_checkpoint_path: Annotated[
        str,
        typer.Option("--pretrained-checkpoint-path", help="Optional Surya checkpoint init path."),
    ] = "",
):
    """Inspect Surya token pressure, truncation risk, and batch geometry for one split."""
    normalized_split = split.strip().lower()
    if normalized_split not in {"train", "val", "holdout"}:
        raise typer.BadParameter("--split must be one of: train, val, holdout")

    effective_max_sequence_lengths = _csv_to_int_list(max_sequence_lengths)
    effective_batch_sizes = _csv_to_int_list(per_device_batch_sizes)
    effective_grad_accum = _csv_to_int_list(gradient_accumulation_steps)
    log.info(
        "inspect-surya-dataset params dataset_dir=%s split=%s sample_size=%d seed=%d "
        "train_fraction=%.4f max_sequence_lengths=%s per_device_batch_sizes=%s "
        "gradient_accumulation_steps=%s pretrained_checkpoint_path=%s",
        dataset_dir,
        normalized_split,
        sample_size,
        seed,
        train_fraction,
        effective_max_sequence_lengths,
        effective_batch_sizes,
        effective_grad_accum,
        pretrained_checkpoint_path or "<default>",
    )

    try:
        result = inspect_surya_dataset(
            dataset_dir=dataset_dir,
            split=normalized_split,
            sample_size=sample_size,
            seed=seed,
            train_fraction=train_fraction,
            max_sequence_lengths=effective_max_sequence_lengths,
            per_device_batch_sizes=effective_batch_sizes,
            gradient_accumulation_steps=effective_grad_accum,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("inspect-surya-dataset failed: %s", exc)
        raise typer.Exit(code=1) from exc

    report = result["report"]
    token_lengths = report["token_lengths"]
    log.info(
        "inspect-surya-dataset complete split=%s inspected_rows=%d p95_tokens=%d max_tokens=%d report=%s",
        normalized_split,
        report["inspected_rows"],
        token_lengths["p95"],
        token_lengths["max"],
        result["report_path"],
    )


if __name__ == "__main__":
    app()
