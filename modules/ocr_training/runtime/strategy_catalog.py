from __future__ import annotations

from modules.ocr_training.schemas import FinetuneStrategy, HardwareProfile

MANUAL_DEFAULTS = {
    "finetune_strategy": FinetuneStrategy.QLORA,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "dataloader_num_workers": 8,
    "max_sequence_length": 1024,
}


def resolve_finetune_strategy(value: str | FinetuneStrategy) -> FinetuneStrategy:
    """Normalize one finetuning strategy value."""
    if isinstance(value, FinetuneStrategy):
        return value
    return FinetuneStrategy(str(value).strip().lower())


def resolve_strategy_allowlist(values: list[str | FinetuneStrategy]) -> list[FinetuneStrategy]:
    """Normalize and de-duplicate a planner strategy allowlist."""
    normalized: list[FinetuneStrategy] = []
    for value in values:
        strategy = resolve_finetune_strategy(value)
        if strategy not in normalized:
            normalized.append(strategy)
    return normalized


def strategy_is_auto_admissible(profile: HardwareProfile, strategy: FinetuneStrategy) -> bool:
    """Return whether auto mode may benchmark a strategy on this host."""
    if strategy == FinetuneStrategy.FULL:
        return False
    if strategy == FinetuneStrategy.QLORA:
        return profile.device_type == "cuda"
    if strategy != FinetuneStrategy.LORA:
        return False
    if profile.device_type != "cuda":
        return False
    if profile.total_vram_mb is None:
        return False
    return profile.total_vram_mb >= 16384 and profile.supports_fp16


def preferred_auto_strategies(
    profile: HardwareProfile,
    allowlist: list[FinetuneStrategy],
) -> list[FinetuneStrategy]:
    """Choose the adapter strategy set auto mode should actually benchmark."""
    normalized = [
        strategy for strategy in allowlist if strategy_is_auto_admissible(profile, strategy)
    ]
    if not normalized:
        return []
    if FinetuneStrategy.LORA in normalized and (profile.total_vram_mb or 0) >= 24576:
        return [FinetuneStrategy.LORA]
    if FinetuneStrategy.QLORA in normalized:
        return [FinetuneStrategy.QLORA]
    return normalized[:1]
