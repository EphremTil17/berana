from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch.nn as nn

from modules.ocr_training.runtime.hardware_profile import (
    GpuMemorySnapshot,
    GpuProcessUsage,
    _split_csv_line,
    enforce_gpu_preflight,
)
from modules.ocr_training.runtime.telemetry import (
    _combined_current_used_memory_mb,
    _combined_peak_used_memory_mb,
)
from modules.ocr_training.schemas import SuryaTrainConfig
from modules.ocr_training.surya_artifacts import load_finetune_meta, write_finetune_meta
from modules.ocr_training.surya_common import (
    deterministic_sample_rows,
    infer_row_modality,
    infer_train_subset_bucket,
    resolve_eval_save_steps,
    resolve_finetune_strategy,
    subset_train_rows,
)
from modules.ocr_training.surya_model import find_lora_target_modules
from modules.ocr_training.surya_patches import (
    build_preprocess_logits_for_metrics,
    compute_metrics_factory,
)
from modules.ocr_training.surya_train import _prepare_train_and_val_rows
from modules.ocr_training.surya_training_args import build_training_arguments


def test_resolve_eval_save_steps_normalizes_shared_cadence():
    eval_steps, save_steps = resolve_eval_save_steps(
        eval_save_steps=500,
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
    )
    assert eval_steps == 500
    assert save_steps == 500


def test_resolve_eval_save_steps_requires_positive_values():
    eval_steps, save_steps = resolve_eval_save_steps(
        eval_save_steps=1,
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
    )
    assert eval_steps == 1
    assert save_steps == 1


def test_build_training_arguments_disables_eval_when_omitted_but_keeps_saving():
    candidate = SimpleNamespace(
        metric_for_best_model="cer",
        eval_steps=None,
        save_steps=500,
        load_best_model_at_end=True,
        dataloader_num_workers=4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=None,
        gradient_accumulation_steps=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
        learning_rate=2e-5,
        fp16=True,
        gradient_checkpointing=False,
        finetune_strategy=SimpleNamespace(value="qlora"),
        num_train_epochs=1,
        save_total_limit=4,
        greater_is_better=False,
        logging_steps=20,
    )

    args = build_training_arguments(
        training_arguments_cls=lambda **kwargs: kwargs,
        output_dir=Path("/tmp/out"),
        candidate=candidate,
        eval_enabled=False,
        save_enabled=True,
        compute_metrics_enabled=False,
        max_steps=None,
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
    )

    assert args["eval_strategy"] == "no"
    assert args["eval_steps"] is None
    assert args["save_strategy"] == "steps"
    assert args["save_steps"] == 500
    assert args["load_best_model_at_end"] is False


def test_build_training_arguments_sets_eval_batch_and_accumulation():
    candidate = SimpleNamespace(
        metric_for_best_model="cer",
        eval_steps=500,
        save_steps=500,
        load_best_model_at_end=True,
        dataloader_num_workers=4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=None,
        gradient_accumulation_steps=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
        learning_rate=2e-5,
        fp16=True,
        gradient_checkpointing=False,
        finetune_strategy=SimpleNamespace(value="qlora"),
        num_train_epochs=1,
        save_total_limit=4,
        greater_is_better=False,
        logging_steps=20,
    )

    args = build_training_arguments(
        training_arguments_cls=lambda **kwargs: kwargs,
        output_dir=Path("/tmp/out"),
        candidate=candidate,
        eval_enabled=True,
        save_enabled=True,
        compute_metrics_enabled=True,
        max_steps=None,
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
    )

    assert args["per_device_eval_batch_size"] == 1
    assert args["eval_accumulation_steps"] == 1
    assert args["prediction_loss_only"] is False
    assert args["save_steps"] == 500


def test_build_training_arguments_omits_hf_best_metric_when_authoritative_eval_owns_selection():
    candidate = SimpleNamespace(
        metric_for_best_model="wer",
        eval_steps=500,
        save_steps=500,
        load_best_model_at_end=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_prefetch_factor=2,
        learning_rate=2e-5,
        fp16=True,
        gradient_checkpointing=False,
        finetune_strategy=SimpleNamespace(value="lora"),
        num_train_epochs=1,
        save_total_limit=4,
        greater_is_better=False,
        logging_steps=20,
    )

    args = build_training_arguments(
        training_arguments_cls=lambda **kwargs: kwargs,
        output_dir=Path("/tmp/out"),
        candidate=candidate,
        eval_enabled=True,
        save_enabled=True,
        compute_metrics_enabled=False,
        max_steps=None,
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
    )

    assert args["load_best_model_at_end"] is False
    assert args["metric_for_best_model"] is None
    assert args["greater_is_better"] is None


def test_build_training_arguments_aligns_save_steps_with_eval_steps_for_authoritative_eval():
    candidate = SimpleNamespace(
        metric_for_best_model="wer",
        eval_steps=50,
        save_steps=200,
        load_best_model_at_end=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_prefetch_factor=2,
        learning_rate=2e-5,
        fp16=True,
        gradient_checkpointing=False,
        finetune_strategy=SimpleNamespace(value="lora"),
        num_train_epochs=1,
        save_total_limit=4,
        greater_is_better=False,
        logging_steps=20,
    )

    args = build_training_arguments(
        training_arguments_cls=lambda **kwargs: kwargs,
        output_dir=Path("/tmp/out"),
        candidate=candidate,
        eval_enabled=True,
        save_enabled=True,
        compute_metrics_enabled=False,
        max_steps=None,
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
    )

    assert args["eval_steps"] == 50
    assert args["save_steps"] == 50


def test_surya_train_config_allows_small_eval_subset():
    config = SuryaTrainConfig(eval_fraction=0.1, eval_max_rows=100)

    assert config.eval_fraction == 0.1
    assert config.eval_max_rows == 100


def test_surya_train_config_rejects_invalid_eval_max_rows():
    with pytest.raises(ValueError, match="eval_max_rows"):
        SuryaTrainConfig(eval_max_rows=0)


def test_deterministic_sample_rows_is_seeded_and_capped():
    rows = [{"image": f"/tmp/{index}.png", "text": f"text-{index}"} for index in range(8)]

    sampled_a = deterministic_sample_rows(rows, max_rows=3, seed=42)
    sampled_b = deterministic_sample_rows(rows, max_rows=3, seed=42)
    sampled_c = deterministic_sample_rows(rows, max_rows=3, seed=7)

    assert sampled_a == sampled_b
    assert len(sampled_a) == 3
    assert sampled_a != rows[:3]
    assert sampled_a != sampled_c


def test_prepare_train_and_val_rows_uses_seeded_sample_for_eval_max_rows(
    monkeypatch, tmp_path: Path
):
    train_rows = [
        {"image": f"/tmp/train_{index}.png", "text": f"train-{index}"} for index in range(4)
    ]
    val_rows = [{"image": f"/tmp/val_{index}.png", "text": f"val-{index}"} for index in range(7)]
    sampler_calls: list[tuple[int, int, int]] = []

    monkeypatch.setattr(
        "modules.ocr_training.surya_train.load_split_rows",
        lambda dataset_dir, split: train_rows if split == "train" else val_rows,
    )
    monkeypatch.setattr(
        "modules.ocr_training.surya_train.deterministic_sample_rows",
        lambda rows, *, max_rows, seed: (
            sampler_calls.append((len(rows), max_rows, seed)) or list(reversed(rows))[:max_rows]
        ),
    )

    _original_train_rows, _original_val_rows, _train_subset, sampled_val_rows = (
        _prepare_train_and_val_rows(
            dataset_dir=tmp_path / "dataset",
            config=SuryaTrainConfig(eval_fraction=1.0, eval_max_rows=3, seed=42),
        )
    )

    assert sampler_calls == [(7, 3, 42)]
    assert sampled_val_rows == list(reversed(val_rows))[:3]


def test_infer_row_modality_matches_typed_and_synthetic_paths():
    assert infer_row_modality({"image": "/tmp/typed/example.png"}) == "typed"
    assert infer_row_modality({"image": "/tmp/synthetic/example.png"}) == "synthetic"


def test_compute_metrics_factory_uses_surya_ocr_tokenizer():
    class _Tokenizer:
        pad_token_id = None

        def batch_decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            outputs = []
            for row in ids:
                outputs.append(" ".join(str(int(token)) for token in row if int(token) != 0))
            return outputs

    processor = SimpleNamespace(ocr_tokenizer=_Tokenizer(), pad_token_id=0)
    compute_metrics = compute_metrics_factory(processor)

    metrics = compute_metrics(
        SimpleNamespace(
            predictions=np.array([[1, 2, 3], [4, 5, 0]]),
            label_ids=np.array([[1, 2, 3], [4, 5, -100]]),
        )
    )

    assert metrics["cer"] == 0.0
    assert metrics["wer"] == 0.0
    assert metrics["exact"] == 1.0


def test_compute_metrics_factory_normalizes_negative_unsigned_like_labels():
    class _Tokenizer:
        pad_token_id = None

        def batch_decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            outputs = []
            for row in ids:
                outputs.append(" ".join(str(int(token)) for token in row if int(token) != 0))
            return outputs

    processor = SimpleNamespace(ocr_tokenizer=_Tokenizer(), pad_token_id=0)
    compute_metrics = compute_metrics_factory(processor)

    metrics = compute_metrics(
        SimpleNamespace(
            predictions=np.array([[1, 2, 3]], dtype=np.int64),
            label_ids=np.array([[1, -100, -1]], dtype=np.int64),
        )
    )

    assert metrics["cer"] >= 0.0
    assert metrics["wer"] >= 0.0


def test_compute_metrics_factory_normalizes_negative_prediction_ids():
    class _Tokenizer:
        pad_token_id = None
        vocab_size = 32

        def batch_decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            outputs = []
            for row in ids:
                outputs.append(" ".join(str(int(token)) for token in row if int(token) != 0))
            return outputs

    processor = SimpleNamespace(ocr_tokenizer=_Tokenizer(), pad_token_id=0)
    compute_metrics = compute_metrics_factory(processor)

    metrics = compute_metrics(
        SimpleNamespace(
            predictions=np.array([[1, -5, 2]], dtype=np.int64),
            label_ids=np.array([[1, 0, 2]], dtype=np.int64),
        )
    )

    assert metrics["cer"] >= 0.0
    assert metrics["wer"] >= 0.0


def test_build_preprocess_logits_for_metrics_argmaxes_sequence_logits():
    preprocess = build_preprocess_logits_for_metrics()
    logits = np.array(
        [
            [[0.1, 0.9], [0.8, 0.2]],
            [[0.7, 0.3], [0.4, 0.6]],
        ]
    )

    result = preprocess(logits, labels=None)

    assert result.tolist() == [[1, 0], [0, 1]]


def test_split_csv_line_parses_stripped_fields():
    parts = _split_csv_line("0, GPU-123, 8192, 512", 4)
    assert parts == ["0", "GPU-123", "8192", "512"]


def test_enforce_gpu_preflight_allows_low_foreign_usage(monkeypatch):
    snapshot = GpuMemorySnapshot(
        gpu_index=0,
        gpu_uuid="GPU-123",
        gpu_name="RTX 3060 Ti",
        total_memory_mb=8192,
        used_memory_mb=512,
        processes=(GpuProcessUsage(pid=99999, process_name="python", used_memory_mb=512),),
    )

    monkeypatch.setattr(
        "modules.ocr_training.runtime.hardware_profile.collect_gpu_memory_snapshot",
        lambda _torch: snapshot,
    )
    monkeypatch.setattr("modules.ocr_training.runtime.hardware_profile.os.getpid", lambda: 12345)

    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
    )

    enforce_gpu_preflight(torch_stub, foreign_usage_threshold_ratio=0.10)


def test_enforce_gpu_preflight_blocks_high_foreign_usage(monkeypatch):
    snapshot = GpuMemorySnapshot(
        gpu_index=0,
        gpu_uuid="GPU-123",
        gpu_name="RTX 3060 Ti",
        total_memory_mb=8192,
        used_memory_mb=1400,
        processes=(
            GpuProcessUsage(pid=22222, process_name="python", used_memory_mb=1200),
            GpuProcessUsage(pid=12345, process_name="python", used_memory_mb=200),
        ),
    )

    monkeypatch.setattr(
        "modules.ocr_training.runtime.hardware_profile.collect_gpu_memory_snapshot",
        lambda _torch: snapshot,
    )
    monkeypatch.setattr("modules.ocr_training.runtime.hardware_profile.os.getpid", lambda: 12345)

    with pytest.raises(RuntimeError, match="GPU preflight blocked"):
        enforce_gpu_preflight(
            SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
            foreign_usage_threshold_ratio=0.10,
        )


def test_combined_current_used_memory_mb_ignores_peak_reserved(monkeypatch):
    snapshot = GpuMemorySnapshot(
        gpu_index=0,
        gpu_uuid="GPU-123",
        gpu_name="RTX 3060 Ti",
        total_memory_mb=8192,
        used_memory_mb=4500,
        processes=(),
    )
    monkeypatch.setattr(
        "modules.ocr_training.runtime.telemetry._torch_reserved_mb",
        lambda _torch, *, peak: 8476 if peak else 4300,
    )

    used_mb = _combined_current_used_memory_mb(SimpleNamespace(), snapshot)

    assert used_mb == 4500


def test_combined_peak_used_memory_mb_includes_peak_reserved(monkeypatch):
    snapshot = GpuMemorySnapshot(
        gpu_index=0,
        gpu_uuid="GPU-123",
        gpu_name="RTX 3060 Ti",
        total_memory_mb=8192,
        used_memory_mb=4500,
        processes=(),
    )
    monkeypatch.setattr(
        "modules.ocr_training.runtime.telemetry._torch_reserved_mb",
        lambda _torch, *, peak: 8476 if peak else 4300,
    )

    used_mb = _combined_peak_used_memory_mb(SimpleNamespace(), snapshot)

    assert used_mb == 8476


def test_resolve_finetune_strategy_normalizes_and_validates():
    assert resolve_finetune_strategy("QLoRA") == "qlora"
    with pytest.raises(ValueError, match="Unsupported finetune strategy"):
        resolve_finetune_strategy("adapter++")


def test_write_and_load_finetune_meta_roundtrip(tmp_path):
    payload = {
        "schema_version": "1.0",
        "finetune_strategy": "qlora",
        "base_checkpoint": "datalab-to/surya",
    }
    write_finetune_meta(tmp_path, payload)
    assert load_finetune_meta(tmp_path) == payload


class _VisionAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 4)
        self.proj = nn.Linear(4, 4)


class _VisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _VisionAttention()


class _DecoderAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.o_proj = nn.Linear(4, 4)


class _DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _DecoderAttention()


class _SuryaLikeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = nn.Module()
        self.vision_encoder.blocks = nn.ModuleList([_VisionBlock()])
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([_DecoderLayer()])


def test_find_lora_target_modules_returns_attention_layers():
    targets = find_lora_target_modules(_SuryaLikeModule())
    assert "vision_encoder.blocks.0.attn.qkv" in targets
    assert "vision_encoder.blocks.0.attn.proj" in targets
    assert "decoder.layers.0.self_attn.q_proj" in targets
    assert "decoder.layers.0.self_attn.o_proj" in targets


def test_infer_train_subset_bucket_reads_image_path_markers():
    assert infer_train_subset_bucket({"image": "/tmp/typed/example.png"}) == "typed"
    assert (
        infer_train_subset_bucket(
            {
                "image": (
                    "output/ocr_training_datasets/fidel_typed_synthetic_v01/data/hf_dataset/"
                    "images/train/fidel_dataset__train__synth_image_0_0.png__synth_image_0_0.png"
                )
            }
        )
        == "synthetic"
    )
    assert (
        infer_train_subset_bucket(
            {
                "image": (
                    "output/ocr_training_datasets/fidel_typed_synthetic_v01/data/hf_dataset/"
                    "images/train/fidel_dataset__train__typed_3642_line_1.png__typed_3642_line_1.png"
                )
            }
        )
        == "typed"
    )
    assert infer_train_subset_bucket({"image": "/tmp/synthetic/example.png"}) == "synthetic"
    assert infer_train_subset_bucket({"image": "/tmp/other/example.png"}) == "unknown"


def test_subset_train_rows_is_deterministic_and_preserves_mix():
    rows = [
        {"image": f"/tmp/typed/sample_{index}.png", "text": f"typed-{index}"} for index in range(10)
    ] + [
        {"image": f"/tmp/synthetic/sample_{index}.png", "text": f"synthetic-{index}"}
        for index in range(10)
    ]

    first = subset_train_rows(rows, train_fraction=0.3, seed=42)
    second = subset_train_rows(rows, train_fraction=0.3, seed=42)

    assert first == second
    assert len(first) == 6
    assert sum(infer_train_subset_bucket(row) == "typed" for row in first) == 3
    assert sum(infer_train_subset_bucket(row) == "synthetic" for row in first) == 3
