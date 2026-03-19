from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from modules.ocr_training.distributed import (
    destroy_distributed_context,
    initialize_distributed_context,
    maybe_barrier,
)
from modules.ocr_training.runtime.hardware_profile import detect_hardware_profile
from modules.ocr_training.surya_artifacts import load_finetune_meta
from modules.ocr_training.surya_common import infer_row_modality, load_split_rows, relative_to_base
from modules.ocr_training.surya_eval_runtime import (
    EvalBatchTiming,
    collect_eval_peak_vram_mb,
    maybe_sync_cuda,
    prepare_eval_rows,
    reset_eval_peak_vram,
    run_surya_eval_batches,
    write_batch_timings_jsonl,
)
from modules.ocr_training.surya_model import load_surya_eval_predictor, require_surya
from modules.ocr_training.surya_reports import write_subset_manifest
from utils.logger import get_logger

logger = get_logger("OCRTrainingSuryaBenchmark")


@dataclass(frozen=True)
class EvalBenchmarkCandidate:
    """One eval benchmark candidate configuration."""

    eval_batch_size: int
    dataloader_num_workers: int

    @property
    def name(self) -> str:
        """Return the stable candidate label used in output paths and logs."""
        return f"b{self.eval_batch_size}_w{self.dataloader_num_workers}"


@dataclass(slots=True)
class EvalBenchmarkResult:
    """Measured benchmark result for one eval candidate."""

    candidate: EvalBenchmarkCandidate
    status: str
    reason: str | None
    output_dir: Path
    num_rows: int
    total_wall_time_sec: float | None
    samples_per_second: float | None
    peak_vram_mb: int | None
    summary: dict[str, Any] | None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stage_summary(
    batch_timings: list[EvalBatchTiming], *, row_selection_sec: float, artifact_write_sec: float
) -> dict[str, dict[str, float | int | None]]:
    stage_map = {
        "row_selection": [row_selection_sec],
        "image_load": [timing.image_load_sec for timing in batch_timings],
        "batch_prep": [timing.batch_prep_sec for timing in batch_timings],
        "predictor": [timing.predictor_sec for timing in batch_timings],
        "decode": [timing.decode_sec for timing in batch_timings],
        "metric": [timing.metric_sec for timing in batch_timings],
        "artifact_write": [artifact_write_sec],
    }
    summary: dict[str, dict[str, float | int | None]] = {}
    for stage_name, values in stage_map.items():
        clean_values = [float(value) for value in values if value is not None]
        total_sec = float(sum(clean_values)) if clean_values else 0.0
        summary[stage_name] = {
            "count": len(clean_values),
            "total_sec": total_sec,
            "mean_sec": float(mean(clean_values)) if clean_values else None,
            "p50_sec": _percentile(clean_values, 0.50),
            "p95_sec": _percentile(clean_values, 0.95),
        }
    return summary


def _hardware_snapshot(runtime, *, distributed_context) -> dict[str, Any] | None:
    torch = runtime.get("torch")
    if torch is None:
        return None
    return detect_hardware_profile(
        torch,
        execution_backend="ddp" if distributed_context.is_distributed else "single",
        distributed_world_size=int(distributed_context.world_size),
    ).model_dump(mode="json")


def _candidate_output_dir(
    base_dir: Path, candidate: EvalBenchmarkCandidate, *, sweep: bool
) -> Path:
    return base_dir / candidate.name if sweep else base_dir


def _write_json(output_path: Path, payload: dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _write_benchmark_report(
    output_path: Path, summary: dict[str, Any], stage_timings: dict[str, Any]
) -> Path:
    lines = [
        "# Surya Eval Benchmark Report",
        "",
        f"- Split: `{summary['split']}`",
        f"- Rows: `{summary['num_rows']}`",
        f"- Eval Fraction: `{summary['eval_fraction']:.4f}`",
        f"- Eval Batch Size: `{summary['eval_batch_size']}`",
        f"- Dataloader Workers: `{summary['dataloader_num_workers']}`",
        f"- Max Rows: `{summary['max_rows']}`",
        f"- Seed: `{summary['seed']}`",
        f"- Samples/sec: `{summary['samples_per_second']:.4f}`",
        f"- Peak VRAM MiB: `{summary['peak_vram_mb']}`",
        f"- Decode Share: `{summary['decode_share']:.4f}`",
        f"- Metric Share: `{summary['metric_share']:.4f}`",
        f"- Write Share: `{summary['write_share']:.4f}`",
        "",
        "## Stage Timings",
        "",
        "| Stage | Count | Total (s) | Mean (s) | P50 (s) | P95 (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage_name, metrics in stage_timings.items():
        lines.append(
            "| {stage} | {count} | {total:.4f} | {mean_val} | {p50_val} | {p95_val} |".format(
                stage=stage_name,
                count=metrics["count"],
                total=float(metrics["total_sec"] or 0.0),
                mean_val=(
                    "n/a" if metrics["mean_sec"] is None else f"{float(metrics['mean_sec']):.4f}"
                ),
                p50_val=(
                    "n/a" if metrics["p50_sec"] is None else f"{float(metrics['p50_sec']):.4f}"
                ),
                p95_val=(
                    "n/a" if metrics["p95_sec"] is None else f"{float(metrics['p95_sec']):.4f}"
                ),
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _summarize_records(records: list[dict[str, Any]]) -> tuple[float, float, float]:
    if not records:
        return 1.0, 1.0, 0.0
    mean_cer = float(mean(record["cer"] for record in records))
    mean_wer = float(mean(record["wer"] for record in records))
    exact_rate = float(mean(1.0 if record["exact"] else 0.0 for record in records))
    return mean_cer, mean_wer, exact_rate


def _resolve_benchmark_candidates(
    *,
    candidate_eval_batch_sizes: list[int] | None,
    candidate_worker_counts: list[int] | None,
) -> list[EvalBenchmarkCandidate]:
    """Return linked benchmark candidates for eval tuning."""
    batch_sizes = list(dict.fromkeys(candidate_eval_batch_sizes or []))
    if not batch_sizes:
        raise ValueError("At least one candidate eval batch size is required.")

    if candidate_worker_counts is None:
        worker_counts = [0] * len(batch_sizes)
    else:
        worker_counts = candidate_worker_counts
        if len(worker_counts) != len(batch_sizes):
            raise ValueError(
                "candidate worker counts must match candidate eval batch sizes one-to-one."
            )

    return [
        EvalBenchmarkCandidate(eval_batch_size=batch_size, dataloader_num_workers=worker_count)
        for batch_size, worker_count in zip(batch_sizes, worker_counts, strict=True)
    ]


def _select_benchmark_winner(
    results: list[EvalBenchmarkResult],
) -> tuple[EvalBenchmarkResult | None, dict[str, Any] | None]:
    """Choose the fastest completed candidate and serialize its public summary."""
    valid_results = [
        result
        for result in results
        if result.status == "completed" and result.samples_per_second is not None
    ]
    winner = (
        max(valid_results, key=lambda result: float(result.samples_per_second))  # type: ignore[arg-type]
        if valid_results
        else None
    )
    if winner is None:
        return None, None
    return winner, {
        "candidate": winner.candidate.name,
        "eval_batch_size": winner.candidate.eval_batch_size,
        "dataloader_num_workers": winner.candidate.dataloader_num_workers,
        "samples_per_second": winner.samples_per_second,
        "peak_vram_mb": winner.peak_vram_mb,
        "output_dir": relative_to_base(winner.output_dir),
    }


def _write_candidate_results(output_dir: Path, results: list[EvalBenchmarkResult]) -> Path:
    """Persist one JSONL line per benchmark candidate result."""
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_results_path = output_dir / "candidate_results.jsonl"
    with candidate_results_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    {
                        "candidate": result.candidate.name,
                        "eval_batch_size": result.candidate.eval_batch_size,
                        "dataloader_num_workers": result.candidate.dataloader_num_workers,
                        "status": result.status,
                        "reason": result.reason,
                        "num_rows": result.num_rows,
                        "samples_per_second": result.samples_per_second,
                        "peak_vram_mb": result.peak_vram_mb,
                        "output_dir": relative_to_base(result.output_dir),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return candidate_results_path


def _write_single_benchmark_summary(
    *,
    output_dir: Path,
    winner: EvalBenchmarkResult | None,
    selected_payload: dict[str, Any] | None,
    run_key: str,
    run_dir: Path,
    split: str,
    eval_fraction: float,
    eval_batch_size: int,
    dataloader_num_workers: int,
    max_rows: int | None,
    seed: int,
    checkpoint_target: str,
    checkpoint_path: Path | None,
    max_vram_ratio: float,
    candidate_results_path: Path,
) -> dict[str, Any]:
    """Persist the single-run benchmark summary without clobbering stage reports."""
    summary = (
        winner.summary
        if winner is not None and winner.summary is not None
        else {
            "status": "no_valid_candidate",
            "split": split,
            "num_rows": 0,
            "eval_fraction": eval_fraction,
            "eval_batch_size": eval_batch_size,
            "dataloader_num_workers": dataloader_num_workers,
            "max_rows": max_rows,
            "seed": seed,
        }
    )
    summary = {
        **summary,
        "run_key": run_key,
        "run_dir": str(run_dir),
        "candidate_results": relative_to_base(candidate_results_path),
        "selected_candidate": selected_payload,
        "checkpoint_target": checkpoint_target,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "max_vram_ratio": max_vram_ratio,
    }
    _write_json(output_dir / "benchmark_summary.json", summary)
    return summary


def _write_sweep_benchmark_summary(
    *,
    output_dir: Path,
    run_key: str,
    run_dir: Path,
    split: str,
    prepared_rows,
    eval_fraction: float,
    max_rows: int | None,
    seed: int,
    checkpoint_target: str,
    checkpoint_path: Path | None,
    max_vram_ratio: float,
    results: list[EvalBenchmarkResult],
    selected_payload: dict[str, Any] | None,
    candidate_results_path: Path,
) -> dict[str, Any]:
    """Persist the sweep summary and winner table for multi-candidate runs."""
    root_summary = {
        "status": "completed" if selected_payload is not None else "no_valid_candidate",
        "run_key": run_key,
        "run_dir": str(run_dir),
        "split": split,
        "num_rows": len(prepared_rows.rows),
        "candidate_count": len(results),
        "candidate_results": relative_to_base(candidate_results_path),
        "selected_candidate": selected_payload,
        "eval_fraction": eval_fraction,
        "max_rows": max_rows,
        "seed": seed,
        "checkpoint_target": checkpoint_target,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "max_vram_ratio": max_vram_ratio,
    }
    _write_json(output_dir / "benchmark_summary.json", root_summary)
    report_lines = [
        "# Surya Eval Benchmark Sweep",
        "",
        f"- Run Key: `{run_key}`",
        f"- Split: `{split}`",
        f"- Rows: `{len(prepared_rows.rows)}`",
        f"- Eval Fraction: `{eval_fraction:.4f}`",
        f"- Max Rows: `{max_rows}`",
        f"- Seed: `{seed}`",
        f"- Candidate Count: `{len(results)}`",
        "",
        "## Candidates",
        "",
        "| Candidate | Batch | Workers | Status | Samples/s | Peak VRAM MiB | Reason |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for result in results:
        report_lines.append(
            "| {candidate} | {batch} | {workers} | {status} | {sps} | {peak_vram} | {reason} |".format(
                candidate=result.candidate.name,
                batch=result.candidate.eval_batch_size,
                workers=result.candidate.dataloader_num_workers,
                status=result.status,
                sps=(
                    "n/a"
                    if result.samples_per_second is None
                    else f"{result.samples_per_second:.4f}"
                ),
                peak_vram=("n/a" if result.peak_vram_mb is None else result.peak_vram_mb),
                reason=result.reason or "",
            )
        )
    if selected_payload is not None:
        report_lines.extend(
            [
                "",
                "## Winner",
                "",
                f"- Candidate: `{selected_payload['candidate']}`",
                f"- Batch: `{selected_payload['eval_batch_size']}`",
                f"- Workers: `{selected_payload['dataloader_num_workers']}`",
                f"- Samples/sec: `{selected_payload['samples_per_second']:.4f}`",
                f"- Peak VRAM MiB: `{selected_payload['peak_vram_mb']}`",
            ]
        )
    (output_dir / "benchmark_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return root_summary


def _run_one_benchmark_candidate(
    *,
    candidate: EvalBenchmarkCandidate,
    base_output_dir: Path,
    sweep: bool,
    runtime,
    foundation_predictor,
    prepared_rows,
    split: str,
    eval_fraction: float,
    max_rows: int | None,
    seed: int,
    distributed_context,
    torch_module,
    max_vram_ratio: float,
    hardware_snapshot: dict[str, Any] | None,
) -> EvalBenchmarkResult:
    candidate_output_dir = _candidate_output_dir(base_output_dir, candidate, sweep=sweep)
    candidate_output_dir.mkdir(parents=True, exist_ok=True)

    predictor = runtime["RecognitionPredictor"](foundation_predictor)
    predictor.disable_tqdm = True

    subset_manifest_path = write_subset_manifest(
        output_path=candidate_output_dir / "benchmark_subset_manifest.jsonl",
        rows=[{**row, "modality": infer_row_modality(row)} for row in prepared_rows.rows],
        split=split,
        seed=seed,
        selection="eval_fraction+eval_max_rows",
    )

    maybe_sync_cuda(torch_module)
    reset_eval_peak_vram(torch_module, distributed_context=distributed_context)
    benchmark_started_at = perf_counter()
    eval_artifacts = run_surya_eval_batches(
        rows=prepared_rows.rows,
        split=split,
        eval_batch_size=candidate.eval_batch_size,
        predictor=predictor,
        runtime=runtime,
        dataloader_num_workers=candidate.dataloader_num_workers,
        distributed_context=distributed_context,
        torch_module=torch_module,
        collect_batch_timings=True,
        progress_desc=f"Benchmark {candidate.name}",
    )
    maybe_sync_cuda(torch_module)
    inference_wall_time_sec = perf_counter() - benchmark_started_at
    peak_vram_mb = collect_eval_peak_vram_mb(torch_module, distributed_context=distributed_context)

    if distributed_context.is_distributed and not distributed_context.is_rank_zero:
        return EvalBenchmarkResult(
            candidate=candidate,
            status="completed_nonzero_rank",
            reason=None,
            output_dir=candidate_output_dir,
            num_rows=len(eval_artifacts.records),
            total_wall_time_sec=None,
            samples_per_second=None,
            peak_vram_mb=None,
            summary=None,
        )

    write_started_at = perf_counter()
    batch_timings_path = write_batch_timings_jsonl(
        candidate_output_dir / "batch_timings.jsonl",
        eval_artifacts.batch_timings,
    )
    artifact_write_sec = perf_counter() - write_started_at

    stage_timings = _stage_summary(
        eval_artifacts.batch_timings,
        row_selection_sec=prepared_rows.row_selection_sec,
        artifact_write_sec=artifact_write_sec,
    )
    total_wall_time_sec = (
        prepared_rows.row_selection_sec + inference_wall_time_sec + artifact_write_sec
    )
    samples_per_second = len(prepared_rows.rows) / max(total_wall_time_sec, 1e-9)
    total_runtime_denom = max(total_wall_time_sec, 1e-9)
    mean_cer, mean_wer, exact_rate = _summarize_records(eval_artifacts.records)
    summary_payload = {
        "status": "completed",
        "split": split,
        "num_rows": len(prepared_rows.rows),
        "world_size": int(eval_artifacts.world_size),
        "eval_fraction": eval_fraction,
        "eval_batch_size": candidate.eval_batch_size,
        "dataloader_num_workers": candidate.dataloader_num_workers,
        "max_rows": max_rows,
        "seed": seed,
        "row_selection_sec": prepared_rows.row_selection_sec,
        "inference_wall_time_sec": inference_wall_time_sec,
        "artifact_write_sec": artifact_write_sec,
        "total_wall_time_sec": total_wall_time_sec,
        "samples_per_second": samples_per_second,
        "peak_vram_mb": peak_vram_mb,
        "mean_cer": mean_cer,
        "mean_wer": mean_wer,
        "exact_rate": exact_rate,
        "decode_share": float(stage_timings["decode"]["total_sec"] or 0.0) / total_runtime_denom,
        "metric_share": float(stage_timings["metric"]["total_sec"] or 0.0) / total_runtime_denom,
        "write_share": float(stage_timings["artifact_write"]["total_sec"] or 0.0)
        / total_runtime_denom,
        "subset_manifest": relative_to_base(subset_manifest_path),
        "batch_timings": relative_to_base(batch_timings_path),
    }

    if hardware_snapshot is not None:
        _write_json(candidate_output_dir / "hardware_profile.json", hardware_snapshot)
    stage_timings_path = _write_json(candidate_output_dir / "stage_timings.json", stage_timings)
    summary_path = _write_json(candidate_output_dir / "benchmark_summary.json", summary_payload)
    report_path = _write_benchmark_report(
        candidate_output_dir / "benchmark_report.md",
        summary_payload,
        stage_timings,
    )
    summary_payload.update(
        {
            "stage_timings": relative_to_base(stage_timings_path),
            "summary_path": relative_to_base(summary_path),
            "report_path": relative_to_base(report_path),
        }
    )

    status = "completed"
    reason = None
    total_vram_mb = hardware_snapshot.get("total_vram_mb") if hardware_snapshot else None
    if peak_vram_mb is not None and total_vram_mb:
        peak_ratio = peak_vram_mb / max(1, int(total_vram_mb))
        summary_payload["peak_vram_ratio"] = peak_ratio
        if peak_ratio > max_vram_ratio:
            status = "rejected_vram"
            reason = f"peak_vram_ratio {peak_ratio:.4f} exceeded configured max_vram_ratio {max_vram_ratio:.4f}"
    summary_payload["status"] = status
    summary_payload["reason"] = reason
    _write_json(candidate_output_dir / "benchmark_summary.json", summary_payload)

    return EvalBenchmarkResult(
        candidate=candidate,
        status=status,
        reason=reason,
        output_dir=candidate_output_dir,
        num_rows=len(prepared_rows.rows),
        total_wall_time_sec=total_wall_time_sec,
        samples_per_second=samples_per_second,
        peak_vram_mb=peak_vram_mb,
        summary=summary_payload,
    )


def benchmark_surya_eval(
    *,
    run_key: str,
    run_dir: Path,
    dataset_dir: Path,
    split: str,
    eval_fraction: float = 1.0,
    eval_batch_size: int = 8,
    dataloader_num_workers: int = 0,
    max_rows: int | None = None,
    seed: int = 42,
    checkpoint_target: str = "best_cer",
    checkpoint_path: Path | None = None,
    output_dir: Path,
    candidate_eval_batch_sizes: list[int] | None = None,
    candidate_worker_counts: list[int] | None = None,
    max_vram_ratio: float = 0.95,
) -> dict[str, Any]:
    """Benchmark the explicit Surya evaluation pipeline on one prepared dataset split."""
    runtime = require_surya()
    torch_module = runtime.get("torch")
    distributed_context = initialize_distributed_context(
        torch_module=torch_module,
        requested_backend="auto",
        ddp_backend="nccl",
    )
    try:
        prepared_rows = prepare_eval_rows(
            rows=load_split_rows(dataset_dir, split),
            modality=None,
            eval_fraction=eval_fraction,
            max_rows=max_rows,
            seed=seed,
        )
        hardware_snapshot = _hardware_snapshot(
            runtime,
            distributed_context=distributed_context,
        )
        foundation_predictor = load_surya_eval_predictor(
            runtime,
            run_dir,
            load_finetune_meta,
            checkpoint_target=checkpoint_target,
            checkpoint_path=checkpoint_path,
        )

        candidates = _resolve_benchmark_candidates(
            candidate_eval_batch_sizes=candidate_eval_batch_sizes or [eval_batch_size],
            candidate_worker_counts=(
                [dataloader_num_workers]
                if candidate_worker_counts is None and candidate_eval_batch_sizes is None
                else candidate_worker_counts
            ),
        )
        sweep = len(candidates) > 1

        if distributed_context.is_rank_zero:
            logger.info(
                "Benchmarking Surya eval split=%s rows=%d candidates=%d batch_sizes=%s workers=%s",
                split,
                len(prepared_rows.rows),
                len(candidates),
                sorted({candidate.eval_batch_size for candidate in candidates}),
                sorted({candidate.dataloader_num_workers for candidate in candidates}),
            )

        results: list[EvalBenchmarkResult] = []
        for candidate in candidates:
            if distributed_context.is_rank_zero:
                logger.info(
                    "Benchmarking eval candidate=%s batch=%d workers=%d",
                    candidate.name,
                    candidate.eval_batch_size,
                    candidate.dataloader_num_workers,
                )
            try:
                result = _run_one_benchmark_candidate(
                    candidate=candidate,
                    base_output_dir=output_dir,
                    sweep=sweep,
                    runtime=runtime,
                    foundation_predictor=foundation_predictor,
                    prepared_rows=prepared_rows,
                    split=split,
                    eval_fraction=eval_fraction,
                    max_rows=max_rows,
                    seed=seed,
                    distributed_context=distributed_context,
                    torch_module=torch_module,
                    max_vram_ratio=max_vram_ratio,
                    hardware_snapshot=hardware_snapshot,
                )
            except (RuntimeError, ValueError, OSError) as exc:
                result = EvalBenchmarkResult(
                    candidate=candidate,
                    status="failed",
                    reason=str(exc),
                    output_dir=_candidate_output_dir(output_dir, candidate, sweep=sweep),
                    num_rows=len(prepared_rows.rows),
                    total_wall_time_sec=None,
                    samples_per_second=None,
                    peak_vram_mb=None,
                    summary=None,
                )
            results.append(result)
            if distributed_context.is_rank_zero:
                logger.info(
                    "Benchmark result candidate=%s status=%s throughput=%s peak_vram=%sMiB reason=%s",
                    candidate.name,
                    result.status,
                    (
                        f"{result.samples_per_second:.4f}"
                        if result.samples_per_second is not None
                        else "n/a"
                    ),
                    result.peak_vram_mb if result.peak_vram_mb is not None else "n/a",
                    result.reason or "none",
                )

        if distributed_context.is_distributed and not distributed_context.is_rank_zero:
            return {"status": "completed_nonzero_rank", "candidate_count": len(candidates)}

        if hardware_snapshot is not None:
            _write_json(output_dir / "hardware_profile.json", hardware_snapshot)
        candidate_results_path = _write_candidate_results(output_dir, results)
        winner, selected_payload = _select_benchmark_winner(results)
        if selected_payload is not None:
            _write_json(output_dir / "selected_benchmark_config.json", selected_payload)

        if not sweep:
            return _write_single_benchmark_summary(
                output_dir=output_dir,
                winner=winner,
                selected_payload=selected_payload,
                run_key=run_key,
                run_dir=run_dir,
                split=split,
                eval_fraction=eval_fraction,
                eval_batch_size=eval_batch_size,
                dataloader_num_workers=dataloader_num_workers,
                max_rows=max_rows,
                seed=seed,
                checkpoint_target=checkpoint_target,
                checkpoint_path=checkpoint_path,
                max_vram_ratio=max_vram_ratio,
                candidate_results_path=candidate_results_path,
            )
        return _write_sweep_benchmark_summary(
            output_dir=output_dir,
            run_key=run_key,
            run_dir=run_dir,
            split=split,
            prepared_rows=prepared_rows,
            eval_fraction=eval_fraction,
            max_rows=max_rows,
            seed=seed,
            checkpoint_target=checkpoint_target,
            checkpoint_path=checkpoint_path,
            max_vram_ratio=max_vram_ratio,
            results=results,
            selected_payload=selected_payload,
            candidate_results_path=candidate_results_path,
        )
    finally:
        with suppress(Exception):
            maybe_barrier(torch_module=torch_module, context=distributed_context)
        destroy_distributed_context(torch_module=torch_module, context=distributed_context)
