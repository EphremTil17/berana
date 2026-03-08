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

        def on_prediction_step(self, args, state, control, **kwargs):
            del args, state, kwargs
            if signal_state.interrupted:
                signal_state.eval_interrupted = True
                control.should_training_stop = True
                control.should_save = True
            return control

    return _InterruptAwareCallback()


def _warn_discarded_eval(signal_state: TrainingSignalState, logger, step: int) -> None:
    if signal_state.eval_discard_warning_emitted:
        return
    signal_state.eval_discard_warning_emitted = True
    logger.warning(
        "Discarding eval metrics at step %d because interrupt was received during evaluation.",
        step,
    )


def _strip_eval_payload(payload: dict[str, float | int] | None) -> bool:
    if not payload:
        return False
    removed = False
    for key in list(payload):
        if key.startswith("eval_"):
            payload.pop(key, None)
            removed = True
    return removed


def build_eval_interrupt_discard_callback(signal_state: TrainingSignalState, callback_base, logger):
    """Build a callback that discards eval metrics if a signal arrived during eval."""

    class _EvalInterruptDiscardCallback(callback_base):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            del args, kwargs
            if not signal_state.eval_interrupted or not metrics:
                return control
            if _strip_eval_payload(metrics):
                _warn_discarded_eval(signal_state, logger, int(state.global_step or 0))
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            del args, kwargs
            if not signal_state.eval_interrupted or not logs:
                return control
            if _strip_eval_payload(logs):
                _warn_discarded_eval(signal_state, logger, int(state.global_step or 0))
                signal_state.eval_interrupted = False
                signal_state.eval_discard_warning_emitted = False
            return control

    return _EvalInterruptDiscardCallback()


def build_eval_cleanup_callback(*, torch_module, callback_base):
    """Build a callback that drops CUDA cache immediately after evaluation."""

    class _EvalCleanupCallback(callback_base):
        def on_evaluate(self, args, state, control, **kwargs):
            if not torch_module.cuda.is_available():
                return control
            try:
                torch_module.cuda.synchronize()
            except Exception:
                return control
            try:
                torch_module.cuda.empty_cache()
            except Exception:
                return control
            try:
                torch_module.cuda.ipc_collect()
            except Exception:
                return control
            return control

    return _EvalCleanupCallback()


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


def resolve_metrics_tokenizer(processor):
    """Resolve the text tokenizer used for OCR metric decoding."""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    return getattr(processor, "ocr_tokenizer", None)


def build_preprocess_logits_for_metrics():
    """Compress eval logits into token IDs before Trainer stores them."""

    def preprocess_logits_for_metrics(logits, labels):
        del labels
        if isinstance(logits, tuple):
            logits = logits[0]
        dim = getattr(logits, "dim", None)
        if callable(dim) and int(dim()) >= 3:
            return logits.argmax(dim=-1)
        ndim = getattr(logits, "ndim", None)
        if ndim is not None and int(ndim) >= 3:
            return np.argmax(logits, axis=-1)
        return logits

    return preprocess_logits_for_metrics


def compute_metrics_factory(processor):
    """Build CER/WER compute_metrics callable for Hugging Face Trainer."""
    tokenizer = resolve_metrics_tokenizer(processor)
    if tokenizer is None:
        return None
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(processor, "pad_token_id", 0)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        vocab_size = getattr(processor, "tokenizer_vocab_size", None)

    def _sanitize_decode_ids(ids):
        sanitized = np.array(ids, dtype=np.int64, copy=True)
        sanitized[sanitized < 0] = int(pad_token_id)
        if vocab_size is not None:
            sanitized[sanitized >= int(vocab_size)] = int(pad_token_id)
        return sanitized

    def _decode(ids):
        return tokenizer.batch_decode(_sanitize_decode_ids(ids), skip_special_tokens=True)

    def compute_metrics(eval_pred) -> dict[str, float]:
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions_array = np.asarray(predictions)
        pred_ids = (
            np.argmax(predictions_array, axis=-1)
            if predictions_array.ndim >= 3
            else predictions_array
        )
        pred_ids = _sanitize_decode_ids(pred_ids)
        labels_for_decode = np.array(labels, dtype=np.int64, copy=True)
        labels_for_decode[labels_for_decode == -100] = int(pad_token_id)
        labels_for_decode = _sanitize_decode_ids(labels_for_decode)

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
