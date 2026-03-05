from __future__ import annotations

from typing import Any

from modules.ocr_training.schemas import SuryaTrainConfig
from modules.ocr_training.surya_common import resolve_finetune_strategy
from modules.ocr_training.surya_patches import (
    patch_surya_checkpoint_inputs,
    patch_surya_forward_for_trainer,
)


def require_surya() -> dict[str, Any]:
    """Load Surya runtime dependencies only when training/evaluation is invoked."""
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from surya.common.surya import SuryaModel, SuryaXLAModel
        from surya.common.surya.config import SuryaModelConfig
        from surya.common.surya.processor.schema import ImageInput, TextInput
        from surya.common.surya.schema import TaskNames
        from surya.common.util import (
            SCRIPT_TOKEN_MAPPING,
            get_top_scripts,
            is_flash_attn_2_supported,
        )
        from surya.foundation import FoundationPredictor
        from surya.foundation.loader import FoundationModelLoader
        from surya.recognition import RecognitionPredictor
        from transformers import BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments
        from transformers.utils import is_flash_attn_2_available

        return {
            "torch": torch,
            "LoraConfig": LoraConfig,
            "PeftModel": PeftModel,
            "get_peft_model": get_peft_model,
            "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
            "ImageInput": ImageInput,
            "TextInput": TextInput,
            "SuryaModelConfig": SuryaModelConfig,
            "SuryaModel": SuryaModel,
            "SuryaXLAModel": SuryaXLAModel,
            "TaskNames": TaskNames,
            "SCRIPT_TOKEN_MAPPING": SCRIPT_TOKEN_MAPPING,
            "get_top_scripts": get_top_scripts,
            "is_flash_attn_2_supported": is_flash_attn_2_supported,
            "FoundationPredictor": FoundationPredictor,
            "FoundationModelLoader": FoundationModelLoader,
            "RecognitionPredictor": RecognitionPredictor,
            "BitsAndBytesConfig": BitsAndBytesConfig,
            "Trainer": Trainer,
            "TrainerCallback": TrainerCallback,
            "TrainingArguments": TrainingArguments,
            "is_flash_attn_2_available": is_flash_attn_2_available,
        }
    except ImportError as exc:
        raise RuntimeError(
            "Surya training dependencies are not installed in this environment. "
            "Install `surya-ocr`, `transformers`, and CUDA-enabled torch first."
        ) from exc


def resolve_base_checkpoint(runtime: dict[str, Any], pretrained_checkpoint_path: str) -> str:
    """Resolve the effective base checkpoint path for Surya loading."""
    loader = runtime["FoundationModelLoader"](pretrained_checkpoint_path or None)
    return str(loader.checkpoint)


def resolve_device_label(torch_module, detect_selected_gpu_index) -> str:
    """Resolve the concrete device label used for model placement."""
    if torch_module.cuda.is_available():
        return f"cuda:{detect_selected_gpu_index(torch_module)}"
    return "cpu"


def resolve_model_dtype(torch_module):
    """Choose the practical dense-model dtype for this hardware."""
    if not torch_module.cuda.is_available():
        return torch_module.float32
    if torch_module.cuda.is_bf16_supported(including_emulation=False):
        return torch_module.bfloat16
    return torch_module.float16


def resolve_adapter_base_name(model, fallback_checkpoint: str) -> str:
    """Resolve the base-model identifier PEFT should record for this loaded model."""
    model_name = getattr(model, "name_or_path", None)
    if isinstance(model_name, str) and model_name.strip():
        return model_name
    config_name = getattr(getattr(model, "config", None), "_name_or_path", None)
    if isinstance(config_name, str) and config_name.strip():
        return config_name
    return fallback_checkpoint


def build_surya_model_config(
    runtime: dict[str, Any],
    *,
    checkpoint: str,
    device_label: str,
    attention_implementation: str | None = None,
):
    """Build a Surya config with the same attention-selection rules as the upstream loader."""
    config = runtime["SuryaModelConfig"].from_pretrained(checkpoint)
    is_cuda = device_label.startswith("cuda")
    if attention_implementation is not None:
        chosen_attention = attention_implementation
    elif runtime["is_flash_attn_2_available"]() and runtime["is_flash_attn_2_supported"](
        device_label
    ):
        chosen_attention = "flash_attention_2"
    else:
        chosen_attention = "sdpa"

    config.decoder._attn_implementation = chosen_attention
    config.vision_encoder._attn_implementation = chosen_attention
    config._attn_implementation_autoset = True
    config.vision_encoder._attn_implementation_autoset = True
    config.decoder._attn_implementation_autoset = True
    if not is_cuda:
        config.decoder._attn_implementation = "sdpa"
        config.vision_encoder._attn_implementation = "sdpa"
    return config


def find_lora_target_modules(model) -> list[str]:
    """Return concrete attention projection module names suitable for LoRA injection."""
    target_suffixes = (
        "attn.qkv",
        "attn.proj",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    )
    target_modules = sorted(
        name for name, _module in model.named_modules() if name.endswith(target_suffixes)
    )
    if not target_modules:
        raise RuntimeError("No compatible Surya attention projection layers were found for LoRA.")
    return target_modules


def log_trainable_parameter_summary(model, logger) -> None:
    """Emit trainable-parameter totals for the active training strategy."""
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    ratio = 0.0 if total == 0 else (trainable / total) * 100.0
    logger.info("Trainable parameters: %d / %d (%.4f%%)", trainable, total, ratio)


def coerce_model_dtype(model, torch_module):
    """Keep dense full-finetune model weights in FP32 for stable AMP scaling."""
    if torch_module.cuda.is_available():
        model = model.to(dtype=torch_module.float32)
    return model


def load_surya_training_stack(
    runtime: dict[str, Any],
    *,
    checkpoint: str,
    config: SuryaTrainConfig,
    detect_selected_gpu_index,
    logger,
    attention_implementation: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load processor and model, then apply the requested training strategy."""
    torch_module = runtime["torch"]
    strategy = resolve_finetune_strategy(config.finetune_strategy)
    device_label = resolve_device_label(torch_module, detect_selected_gpu_index)
    processor = runtime["FoundationModelLoader"](checkpoint).processor()
    if strategy == "qlora" and not torch_module.cuda.is_available():
        raise RuntimeError("QLoRA requires CUDA in this implementation.")

    model_config = build_surya_model_config(
        runtime,
        checkpoint=checkpoint,
        device_label=device_label,
        attention_implementation=attention_implementation,
    )
    if strategy == "qlora":
        quant_config = runtime["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_module.float16,
        )
        model = runtime["SuryaModel"].from_pretrained(
            checkpoint,
            config=model_config,
            quantization_config=quant_config,
            device_map={"": detect_selected_gpu_index(torch_module)},
            ignore_mismatched_sizes=True,
        )
        target_modules = find_lora_target_modules(model)
        model = runtime["prepare_model_for_kbit_training"](
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
        patch_surya_forward_for_trainer(model, torch_module, logger)
        if config.gradient_checkpointing:
            patch_surya_checkpoint_inputs(model, logger)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model = runtime["get_peft_model"](
            model,
            runtime["LoraConfig"](
                base_model_name_or_path=resolve_adapter_base_name(model, checkpoint),
                inference_mode=False,
                r=int(config.lora_rank),
                lora_alpha=int(config.lora_alpha),
                lora_dropout=float(config.lora_dropout),
                bias="none",
                target_modules=target_modules,
            ),
        )
        model.train()
        metadata = {
            "schema_version": "1.0",
            "finetune_strategy": strategy,
            "base_checkpoint": checkpoint,
            "target_modules": target_modules,
            "quantization": "4bit-nf4",
        }
        log_trainable_parameter_summary(model, logger)
        return model, processor, metadata

    model = (
        runtime["SuryaModel"]
        .from_pretrained(
            checkpoint,
            dtype=resolve_model_dtype(torch_module),
            config=model_config,
            ignore_mismatched_sizes=True,
        )
        .to(device_label)
    )
    model = model.eval()
    metadata = {
        "schema_version": "1.0",
        "finetune_strategy": strategy,
        "base_checkpoint": checkpoint,
        "quantization": None,
    }
    if strategy == "full":
        model = coerce_model_dtype(model, torch_module)
        patch_surya_forward_for_trainer(model, torch_module, logger)
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        metadata["target_modules"] = []
        return model, processor, metadata

    target_modules = find_lora_target_modules(model)
    patch_surya_forward_for_trainer(model, torch_module, logger)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        patch_surya_checkpoint_inputs(model, logger)
    model.enable_input_require_grads()
    model = runtime["get_peft_model"](
        model,
        runtime["LoraConfig"](
            base_model_name_or_path=resolve_adapter_base_name(model, checkpoint),
            inference_mode=False,
            r=int(config.lora_rank),
            lora_alpha=int(config.lora_alpha),
            lora_dropout=float(config.lora_dropout),
            bias="none",
            target_modules=target_modules,
        ),
    )
    model.train()
    metadata["target_modules"] = target_modules
    log_trainable_parameter_summary(model, logger)
    return model, processor, metadata


def load_surya_eval_predictor(runtime: dict[str, Any], run_dir, load_finetune_meta):
    """Load a FoundationPredictor-compatible object for evaluation."""
    finetune_meta = load_finetune_meta(run_dir)
    if not finetune_meta:
        return runtime["FoundationPredictor"](checkpoint=str(run_dir))

    base_checkpoint = str(finetune_meta["base_checkpoint"])
    strategy = resolve_finetune_strategy(str(finetune_meta["finetune_strategy"]))
    foundation_predictor = runtime["FoundationPredictor"](checkpoint=base_checkpoint)
    if strategy == "full":
        return foundation_predictor

    adapter_source = run_dir
    best_checkpoint_link = run_dir / "weights" / "best_checkpoint"
    if best_checkpoint_link.exists():
        adapter_source = best_checkpoint_link.resolve()
    foundation_predictor.model = runtime["PeftModel"].from_pretrained(
        foundation_predictor.model,
        str(adapter_source),
        is_trainable=False,
    )
    foundation_predictor.model.eval()
    return foundation_predictor
