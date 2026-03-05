from __future__ import annotations

from statistics import mean

import numpy as np

from modules.ocr_benchmark.metrics import calculate_cer_wer, normalize_ethiopic_text
from modules.ocr_training.checkpointing import TrainingSignalState


def build_interrupt_callback(signal_state: TrainingSignalState, callback_base):
    """Build a TrainerCallback instance that stops on signal interruption."""

    class _InterruptAwareCallback(callback_base):
        def on_step_end(self, args, state, control, **kwargs):
            if signal_state.interrupted:
                control.should_training_stop = True
                control.should_save = True
            return control

    return _InterruptAwareCallback()


def patch_surya_forward_for_trainer(model, torch_module, logger) -> None:
    """Patch Surya forward to auto-build image embeddings from tiles for Trainer calls."""
    if getattr(model, "_berana_forward_patch_v1", False):
        return
    original_forward = model.forward

    def _patched_forward(*args, **kwargs):
        image_embeddings = kwargs.get("image_embeddings")
        image_tiles = kwargs.get("image_tiles")
        grid_thw = kwargs.get("grid_thw")
        input_ids = kwargs.get("input_ids")
        if image_embeddings is None and image_tiles is not None and grid_thw is not None:
            has_image_tokens = bool(
                input_ids is not None and (input_ids == model.config.image_token_id).any().item()
            )
            if has_image_tokens:
                valid_batch_size = kwargs.get("valid_batch_size")
                if valid_batch_size is None and input_ids is not None:
                    valid_batch_size = torch_module.tensor(
                        input_ids.shape[0],
                        device=model.device,
                    )
                max_batch_size = kwargs.get("max_batch_size")
                if max_batch_size is None and input_ids is not None:
                    max_batch_size = int(input_ids.shape[0])
                encoder_chunk_size = int(kwargs.get("encoder_chunk_size", 32768))
                kwargs["image_embeddings"] = model.get_image_embeddings(
                    pixel_values=image_tiles,
                    grid_thw=grid_thw,
                    encoder_chunk_size=encoder_chunk_size,
                    valid_batch_size=valid_batch_size,
                    max_batch_size=max_batch_size,
                )
        return original_forward(*args, **kwargs)

    model.forward = _patched_forward
    model._berana_forward_patch_v1 = True
    logger.info("Applied Surya forward patch: auto image_embeddings from image_tiles/grid_thw.")


def patch_surya_checkpoint_inputs(model, logger) -> None:
    """Force vision patch embeddings to require grad for checkpointed adapter training."""
    if getattr(model, "_berana_checkpoint_inputs_patch_v1", False):
        return
    patch_embed = getattr(getattr(model, "vision_encoder", None), "patch_embed", None)
    if patch_embed is None or not hasattr(patch_embed, "register_forward_hook"):
        return

    def _make_outputs_require_grad(_module, _inputs, output):
        if isinstance(output, tuple):
            return tuple(
                item.requires_grad_(True) if hasattr(item, "requires_grad_") else item
                for item in output
            )
        if hasattr(output, "requires_grad_"):
            output.requires_grad_(True)
        return output

    model._berana_checkpoint_inputs_hook_v1 = patch_embed.register_forward_hook(
        _make_outputs_require_grad
    )
    model._berana_checkpoint_inputs_patch_v1 = True
    logger.info("Applied Surya checkpoint patch: vision patch embeddings now require grad.")


def compute_metrics_factory(processor):
    """Build CER/WER compute_metrics callable for Hugging Face Trainer."""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return None

    def _decode(ids):
        return tokenizer.batch_decode(ids, skip_special_tokens=True)

    def compute_metrics(eval_pred) -> dict[str, float]:
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        pred_ids = np.argmax(predictions, axis=-1)
        labels_for_decode = labels.copy()
        labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id

        decoded_preds = _decode(pred_ids)
        decoded_labels = _decode(labels_for_decode)
        cer_scores = []
        wer_scores = []
        exact_scores = []
        for pred, gt in zip(decoded_preds, decoded_labels, strict=False):
            norm_pred = normalize_ethiopic_text(pred)
            norm_gt = normalize_ethiopic_text(gt)
            cer, wer, exact = calculate_cer_wer(norm_pred, norm_gt)
            cer_scores.append(cer)
            wer_scores.append(wer)
            exact_scores.append(1.0 if exact else 0.0)

        return {
            "cer": float(mean(cer_scores)) if cer_scores else 1.0,
            "wer": float(mean(wer_scores)) if wer_scores else 1.0,
            "exact": float(mean(exact_scores)) if exact_scores else 0.0,
        }

    return compute_metrics
