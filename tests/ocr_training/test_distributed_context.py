from pathlib import Path
from types import SimpleNamespace

import pytest

from modules.ocr_training.distributed.context import (
    initialize_distributed_context,
    resolve_execution_backend,
)
from modules.ocr_training.runtime.candidate_builder import (
    build_training_candidates,
    derive_auto_constraints,
)
from modules.ocr_training.runtime.hardware_profile import detect_hardware_profile
from modules.ocr_training.schemas import (
    ExecutionBackend,
    FinetuneStrategy,
    HardwareProfile,
    SuryaTrainConfig,
    TrainMode,
)
from modules.ocr_training.surya_artifacts import write_finetune_meta, write_hardware_profile


def test_resolve_execution_backend_auto_rejects_multi_gpu_without_torchrun():
    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
    )

    with pytest.raises(RuntimeError, match=r"multi-gpu|torchrun"):
        resolve_execution_backend(torch_module=torch_stub, requested_backend="auto")


def test_initialize_distributed_context_sets_local_rank(monkeypatch):
    calls: list[str] = []
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            set_device=lambda index: calls.append(f"set_device:{index}"),
        ),
        distributed=SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: False,
            init_process_group=lambda backend, device_id=None: calls.append(
                f"init:{backend}:{device_id}"
            ),
        ),
    )

    context = initialize_distributed_context(
        torch_module=torch_stub,
        requested_backend="ddp",
        ddp_backend="nccl",
    )

    assert context.is_distributed is True
    assert context.rank == 1
    assert context.local_rank == 1
    assert context.world_size == 2
    assert context.device == "cuda:1"
    assert calls == ["set_device:1", "init:nccl:1"]


def test_maybe_barrier_prefers_local_device_id():
    calls: list[object] = []
    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        distributed=SimpleNamespace(
            is_initialized=lambda: True,
            barrier=lambda device_ids=None: calls.append(device_ids),
        ),
    )

    from modules.ocr_training.distributed.context import DistributedContext, maybe_barrier

    maybe_barrier(
        torch_module=torch_stub,
        context=DistributedContext(
            execution_backend="ddp",
            ddp_backend="nccl",
            is_distributed=True,
            rank=1,
            local_rank=1,
            world_size=2,
            device="cuda:1",
            is_rank_zero=False,
        ),
    )

    assert calls == [[1]]


def test_build_training_candidates_tracks_world_size_in_global_batch():
    profile = HardwareProfile(
        device_type="cuda",
        cuda_device_count=2,
        execution_backend="ddp",
        distributed_world_size=2,
        gpu_index=0,
        gpu_name="2x RTX 5090",
        total_vram_mb=32768,
        free_vram_mb=30000,
        supports_fp16=True,
        supports_bf16=True,
        cpu_count=64,
    )
    config = SuryaTrainConfig(
        mode=TrainMode.AUTO,
        execution_backend=ExecutionBackend.DDP,
        distributed_world_size=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        strategy_allowlist=[FinetuneStrategy.QLORA, FinetuneStrategy.LORA],
    )

    constraints = derive_auto_constraints(config, profile)
    candidates = build_training_candidates(
        profile=profile,
        config=config,
        constraints=constraints,
    )

    assert candidates
    assert all(candidate.execution_backend.value == "ddp" for candidate in candidates)
    assert all(candidate.world_size == 2 for candidate in candidates)
    assert all(
        candidate.effective_global_batch_size
        == candidate.per_device_train_batch_size * candidate.gradient_accumulation_steps * 2
        for candidate in candidates
    )


def test_rank_zero_artifact_writes_can_be_suppressed(tmp_path: Path):
    profile = HardwareProfile(device_type="cpu", cuda_device_count=0)

    assert write_hardware_profile(tmp_path, profile, is_rank_zero=False) is None
    assert write_finetune_meta(tmp_path, {"schema_version": "1.0"}, is_rank_zero=False) is None
    assert not (tmp_path / "hardware_profile.json").exists()
    assert not (tmp_path / "finetune_meta.json").exists()


def test_detect_hardware_profile_carries_backend_metadata(monkeypatch):
    monkeypatch.setattr("modules.ocr_training.runtime.hardware_profile.os.cpu_count", lambda: 16)
    torch_stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

    profile = detect_hardware_profile(
        torch_stub,
        execution_backend="ddp",
        distributed_world_size=4,
    )

    assert profile.execution_backend == "ddp"
    assert profile.distributed_world_size == 4
