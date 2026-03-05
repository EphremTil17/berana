from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from config.settings import settings
from modules.ocr_training.schemas import HardwareProfile
from utils.logger import get_logger

logger = get_logger("OCRTrainingHardwareProfile")


@dataclass(frozen=True)
class GpuProcessUsage:
    """Single process currently consuming GPU framebuffer memory."""

    pid: int
    process_name: str
    used_memory_mb: int


@dataclass(frozen=True)
class GpuMemorySnapshot:
    """Framebuffer snapshot for one selected NVIDIA GPU."""

    gpu_index: int
    gpu_uuid: str
    gpu_name: str
    total_memory_mb: int
    used_memory_mb: int
    processes: tuple[GpuProcessUsage, ...]


def _split_csv_line(line: str, expected_fields: int) -> list[str]:
    """Split a simple `nvidia-smi` CSV line into stripped fields."""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < expected_fields:
        raise ValueError(f"Expected at least {expected_fields} fields, got {len(parts)}: {line}")
    return parts


def _run_nvidia_smi_query(*query_args: str) -> list[str]:
    """Run an `nvidia-smi` query and return non-empty output lines."""
    try:
        result = subprocess.run(
            ["nvidia-smi", *query_args],
            capture_output=True,
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _detect_selected_gpu_index(torch_module) -> int:
    """Resolve the active GPU index from CUDA visibility or torch state."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices:
        first_token = visible_devices.split(",")[0].strip()
        if first_token.isdigit():
            return int(first_token)
    return int(torch_module.cuda.current_device())


def _foreign_gpu_processes(
    snapshot: GpuMemorySnapshot, current_pid: int
) -> tuple[GpuProcessUsage, ...]:
    """Return GPU processes that do not belong to the current Python process."""
    return tuple(process for process in snapshot.processes if process.pid != current_pid)


def _format_gpu_process_list(processes: tuple[GpuProcessUsage, ...]) -> str:
    """Render compact process diagnostics for error messages."""
    if not processes:
        return "unattributed GPU consumers"
    return "; ".join(
        f"pid={process.pid} name={process.process_name} vram={process.used_memory_mb}MiB"
        for process in processes
    )


def _system_ram_mb() -> int | None:
    """Best-effort system RAM detection without extra dependencies."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None
    return int((page_size * phys_pages) // (1024 * 1024))


def collect_gpu_memory_snapshot(torch_module) -> GpuMemorySnapshot | None:
    """Collect current framebuffer usage for the selected GPU."""
    if not torch_module.cuda.is_available():
        return None

    selected_index = _detect_selected_gpu_index(torch_module)
    summary_lines = _run_nvidia_smi_query(
        "--query-gpu=index,uuid,name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    )
    if not summary_lines:
        return None

    selected_uuid = ""
    gpu_name = "unknown"
    total_memory_mb = 0
    used_memory_mb = 0
    for line in summary_lines:
        index_str, uuid, name, total_str, used_str = _split_csv_line(line, 5)[:5]
        if int(index_str) != selected_index:
            continue
        selected_uuid = uuid
        gpu_name = name
        total_memory_mb = int(total_str)
        used_memory_mb = int(used_str)
        break

    if not selected_uuid:
        index_str, selected_uuid, gpu_name, total_str, used_str = _split_csv_line(
            summary_lines[0], 5
        )[:5]
        selected_index = int(index_str)
        total_memory_mb = int(total_str)
        used_memory_mb = int(used_str)

    processes: list[GpuProcessUsage] = []
    process_lines = _run_nvidia_smi_query(
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    )
    for line in process_lines:
        gpu_uuid, pid_str, process_name, used_memory_str = _split_csv_line(line, 4)[:4]
        if gpu_uuid != selected_uuid:
            continue
        if not pid_str.isdigit() or not used_memory_str.isdigit():
            continue
        processes.append(
            GpuProcessUsage(
                pid=int(pid_str),
                process_name=process_name,
                used_memory_mb=int(used_memory_str),
            )
        )

    return GpuMemorySnapshot(
        gpu_index=selected_index,
        gpu_uuid=selected_uuid,
        gpu_name=gpu_name,
        total_memory_mb=total_memory_mb,
        used_memory_mb=used_memory_mb,
        processes=tuple(processes),
    )


def enforce_single_gpu(torch_module) -> None:
    """Reject multi-GPU environments in adaptive mode v1."""
    if not torch_module.cuda.is_available():
        return
    if int(torch_module.cuda.device_count()) > 1:
        raise RuntimeError(
            "Adaptive Surya training v1 supports exactly one visible CUDA device. "
            "Restrict visibility to one GPU and retry."
        )


def enforce_gpu_preflight(torch_module, foreign_usage_threshold_ratio: float) -> None:
    """Abort early if other applications already occupy too much GPU memory."""
    snapshot = collect_gpu_memory_snapshot(torch_module)
    if snapshot is None or snapshot.total_memory_mb <= 0:
        logger.warning("Skipping GPU preflight check because `nvidia-smi` data was unavailable.")
        return

    current_pid = os.getpid()
    foreign_processes = _foreign_gpu_processes(snapshot, current_pid)
    foreign_used_memory_mb = sum(process.used_memory_mb for process in foreign_processes)
    used_ratio = snapshot.used_memory_mb / snapshot.total_memory_mb
    foreign_ratio = foreign_used_memory_mb / snapshot.total_memory_mb

    if used_ratio < foreign_usage_threshold_ratio and foreign_ratio < foreign_usage_threshold_ratio:
        if foreign_processes:
            logger.warning(
                "Other GPU processes detected but below threshold: %s",
                _format_gpu_process_list(foreign_processes),
            )
        return

    threshold_percent = foreign_usage_threshold_ratio * 100.0
    process_details = _format_gpu_process_list(foreign_processes)
    raise RuntimeError(
        "GPU preflight blocked: other applications are already using too much VRAM on "
        f"GPU {snapshot.gpu_index}. Total in-use={snapshot.used_memory_mb}MiB/"
        f"{snapshot.total_memory_mb}MiB ({used_ratio:.1%}); foreign-use={foreign_used_memory_mb}MiB "
        f"({foreign_ratio:.1%}). Threshold={threshold_percent:.1f}%. Terminate those applications "
        f"and retry. Detected processes: {process_details}."
    )


def detect_hardware_profile(torch_module) -> HardwareProfile:
    """Create a normalized single-host hardware profile."""
    cpu_count = os.cpu_count() or 1
    if not torch_module.cuda.is_available():
        return HardwareProfile(
            device_type="cpu",
            cuda_device_count=0,
            cpu_count=cpu_count,
            system_ram_mb=_system_ram_mb(),
            output_root=str(settings.OUTPUT_DIR),
        )

    snapshot = collect_gpu_memory_snapshot(torch_module)
    selected_index = _detect_selected_gpu_index(torch_module)
    capability = torch_module.cuda.get_device_capability(selected_index)
    device_name = torch_module.cuda.get_device_name(selected_index)
    foreign_processes: list[dict[str, str | int]] = []
    if snapshot is not None:
        foreign_processes = [
            {
                "pid": process.pid,
                "process_name": process.process_name,
                "used_memory_mb": process.used_memory_mb,
            }
            for process in _foreign_gpu_processes(snapshot, os.getpid())
        ]

    total_vram_mb = snapshot.total_memory_mb if snapshot else None
    used_vram_mb = snapshot.used_memory_mb if snapshot else None
    free_vram_mb = None
    if total_vram_mb is not None and used_vram_mb is not None:
        free_vram_mb = max(0, total_vram_mb - used_vram_mb)

    return HardwareProfile(
        device_type="cuda",
        cuda_device_count=int(torch_module.cuda.device_count()),
        gpu_index=selected_index,
        gpu_name=snapshot.gpu_name if snapshot else device_name,
        gpu_uuid=snapshot.gpu_uuid if snapshot else None,
        total_vram_mb=total_vram_mb,
        used_vram_mb=used_vram_mb,
        free_vram_mb=free_vram_mb,
        compute_capability=f"{capability[0]}.{capability[1]}",
        supports_fp16=True,
        supports_bf16=bool(torch_module.cuda.is_bf16_supported(including_emulation=False)),
        cpu_count=cpu_count,
        system_ram_mb=_system_ram_mb(),
        output_root=str(settings.OUTPUT_DIR),
        foreign_processes=foreign_processes,
    )
