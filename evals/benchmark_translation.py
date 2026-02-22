import json
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from config.settings import settings
from utils.logger import get_logger

log = get_logger("Benchmark")


def download_model(repo_id: str, filename: str) -> Path:
    """Downloads the specified model from Hugging Face if it does not already exist.

    Args:
        repo_id (str): The Hugging Face repository ID.
        filename (str): The specific GGUF filename to download.

    Returns:
        Path: The absolute path to the downloaded model.
    """
    log.info(f"Downloading model {filename} from {repo_id}...")
    model_path = hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=str(settings.MODELS_DIR)
    )
    return Path(model_path)


def run_benchmark(
    model_path: Path,
    n_gpu_layers: int,
    n_ctx: int = 2048,
    temperature: float = 0.0,
) -> None:
    """Initializes LLM and runs benchmarks on all Ge'ez text files in resources.

    Args:
        model_path (Path): Path to the GGUF model file.
        n_gpu_layers (int): Number of layers to offload to the GPU.
        n_ctx (int, optional): Size of the context window. Defaults to 2048.
        temperature (float, optional): Inference temperature. Defaults to 0.0.
    """
    model_path_str = str(model_path)
    log.info(f"Using model {model_path_str}")
    log.info(f"Context window set to {n_ctx}")
    log.info(f"n_gpu_layers set to {n_gpu_layers}")

    # Load model
    try:
        log.info("Initializing Llama engine (This may take a moment to compile kernels)...")
        llm = Llama(
            model_path=model_path_str,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,  # Disabled to reduce log bloat
        )
        log.success("Model loaded and engine ready.")
    except Exception as e:
        log.error(f"Error loading model: {e}")
        return

    # Check for BLAS/CUDA
    # llama-cpp-python doesn't expose a direct 'blas' attribute easily,
    # but the verbose output shows it.
    # We can also check llm.model.n_gpu_layers if it worked as expected.

    input_files = [
        f for f in settings.INPUT_DIR.iterdir() if f.is_file() and f.suffix in (".txt", ".gez")
    ]
    if not input_files:
        log.warning(f"No input files found in {settings.INPUT_DIR}.")
        return

    for filepath in input_files:
        filename = filepath.name
        with filepath.open(encoding="utf-8") as f:
            text = f.read()

        log.info(f"Translating {filename}...")

        # TranslateGemma Specific Logic
        # User Instructions:
        # - Strict Roles: Use only "User" and "Assistant" roles.
        # - Input Formatting: The prompt must include the source and target language codes.
        # - Content Mapping: For text-only benchmarking, the user role content
        #   must be an iterable containing a mapping with type: 'text',
        #   source_lang_code: 'gez', target_lang_code: 'en', and the actual text.

        # Based on these instructions, we simulate the chat completion call structure.
        # TranslateGemma usually expects a specific prompt template.

        # We use create_chat_completion or manually build the prompt if create_chat_completion
        # doesn't handle "User"/"Assistant" capitalization as needed.
        # Most GGUF chat templates handle this, but the user was very specific.

        start_time = time.time()
        # Using create_chat_completion to allow the model's chat template to work,
        # but manually specifying the prompt if the user's "Strict Roles" requirement
        # implies a non-standard template.

        # Manual prompt construction:
        # <start_of_turn>user\n[{"type": "text", "source_lang_code": "gez",
        # "target_lang_code": "en", "text": "..."}]<end_of_turn>\n<start_of_turn>model\n
        # But the user said "Use only User and Assistant roles".
        # This might refer to the role names in the messages list.

        # We'll use the prompt format suggested by the Gemma/TranslateGemma docs.

        # Load glossary if available
        glossary_text = ""
        if settings.GLOSSARY_FILE.exists():
            try:
                with settings.GLOSSARY_FILE.open(encoding="utf-8") as g:
                    glossary_data = json.load(g)
                    if glossary_data:
                        glossary_text = "Liturgical Glossary (Preserve these terms):\n"
                        for gez, en in glossary_data.items():
                            glossary_text += f"- {gez}: {en}\n"
                        glossary_text += "\n"
                log.info(f"Loaded {len(glossary_data)} terms from glossary.")
            except Exception as e:
                log.warning(f"Could not load glossary: {e}")

        # Optimization Strategy implementation:
        expert_persona = (
            "Persona: Expert liturgical translator of the EOTC.\n"
            "Style: Formal, solemn English (Thee/Thou), preserve Ge'ez syntax.\n\n"
            f"{glossary_text}"
            "Instruction: For the provided text, produce exactly two sections:\n"
            "1. Word-for-word Breakdown: A list mapping each word.\n"
            "2. Liturgical Translation: A cohesive English translation.\n\n"
            "Example One-Shot:\n"
            "Source (gez): ጸጋ እግዚአብሔር ምስሌክሙ።\n"
            "Breakdown:\n- ጸጋ: Grace\n- እግዚአብሔር: of God\n- ምስሌክሙ: be with you\n"
            "Translation: The grace of God be with you.\n\n"
        )

        # The "Two Blank Lines" Rule: Prepending two newlines to the actual text.
        full_text_input = f"{expert_persona}Text to translate:\n\n\n{text}"

        content_mapping = {
            "type": "text",
            "source_lang_code": "gez",
            "target_lang_code": "en",
            "text": full_text_input,
        }

        # Leading the Assistant prompt with a prefix to trigger the breakdown first
        prompt_with_mapping = f"User: {[content_mapping]!s}\nAssistant: Breakdown:\n"

        log.info(f"Translating {filename} using expert persona (T={temperature})...")

        start_time = time.time()
        output = llm(
            prompt_with_mapping,
            max_tokens=n_ctx,
            temperature=temperature,
            top_p=0.95,
            stop=["User:", "Assistant:", "<end_of_turn>", "<bos>", "<pad>"],
            echo=False,
        )
        end_time = time.time()

        translated_text = output["choices"][0]["text"].strip()

        # Post-processing: Remove any potential model-hallucinated headers from the start
        noise_prefixes = [
            "Breakdown:",
            "Word-for-word Breakdown:",
            "Translation:",
            "Liturgical Translation:",
            "Assistant:",
        ]
        for prefix in noise_prefixes:
            if translated_text.lower().startswith(prefix.lower()):
                translated_text = translated_text[len(prefix) :].strip()

        duration = end_time - start_time
        tokens = output["usage"]["completion_tokens"]
        tps = tokens / duration if duration > 0 else 0

        log.success(f"Translation of {filename} complete in {duration:.2f}s ({tps:.2f} tokens/s)")

        # Save as .md
        output_filename = filepath.with_suffix(".md").name
        output_path = settings.OUTPUT_DIR / output_filename
        with output_path.open("w", encoding="utf-8") as f:
            f.write(f"# Benchmarking Report: {filename}\n\n")
            f.write("**Source Language:** Ge'ez (gez)\n")
            f.write("**Target Language:** English (en)\n")
            f.write(
                f"**Stats:** {duration:.2f}s, {tps:.2f} t/s, Temp: {temperature}, "
                f"Context: {n_ctx}\n\n"
            )
            f.write("## Original Ge'ez\n")
            f.write("```\n" + text + "\n```\n\n")
            f.write("## Word-for-word & Liturgical Translation\n")
            f.write(translated_text + "\n")


def run_cli_entrypoint(n_gpu_layers: int, n_ctx: int, temperature: float) -> None:
    """Entry point for the Typer CLI to trigger benchmarking logic.

    Args:
        n_gpu_layers (int): The number of tensor layers to offload to the GPU.
        n_ctx (int): The maximum context window size in tokens.
        temperature (float): Generation temperature.
    """
    model_path = settings.MODELS_DIR / settings.MODEL_FILE

    if model_path.exists():
        log.success(f"Model found locally: {model_path} (Skipping download)")
    else:
        log.info(f"Model not found at {model_path}. Starting ingestion...")
        model_path = download_model(settings.MODEL_REPO, settings.MODEL_FILE)

    import sys

    run_benchmark(
        model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, temperature=temperature
    )
    sys.exit(0)
