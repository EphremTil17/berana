from pathlib import Path

from config.settings import settings
from utils.run_registry import load_latest_run

_STAGE_DIR_NAMES = {
    "prep",
    "coverage",
    "surya_zero_shot",
    "trocr_zero_shot",
    "surya_finetune",
    "trocr_finetune",
    "final_eval",
}


def _doc_root_from_run_dir(run_dir: Path) -> Path:
    """Resolve canonical document benchmark root from a stage run directory."""
    if run_dir.name in _STAGE_DIR_NAMES:
        return run_dir.parent
    return run_dir


def resolve_doc_benchmark_root(doc_stem: str) -> Path:
    """Resolve canonical benchmark root for a document from latest registry pointers."""
    stages = (
        "ocr-benchmark-prepare",
        "ocr-benchmark-coverage-report",
        "ocr-benchmark-coverage-queue",
        "ocr-benchmark-surya-zero",
        "ocr-benchmark-trocr-zero",
        "ocr-benchmark-surya-finetune",
        "ocr-benchmark-trocr-finetune",
    )
    for stage in stages:
        pointer = load_latest_run(stage, doc_stem)
        if not pointer:
            continue
        run_dir = Path(pointer["run_dir"])
        return _doc_root_from_run_dir(run_dir)

    raise FileNotFoundError(
        f"No OCR benchmark run root found for '{doc_stem}'. "
        "Run 'ocr-benchmark prepare-lines' first."
    )


def create_new_doc_benchmark_root(doc_stem: str) -> Path:
    """Allocate next document-level benchmark root in `<doc_stem>_vNN` format."""
    from utils.run_registry import next_versioned_dir

    return next_versioned_dir(settings.OUTPUT_DIR / "ocr_benchmark", doc_stem)
