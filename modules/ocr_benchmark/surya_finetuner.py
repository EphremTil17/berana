from pathlib import Path

from config.settings import settings
from modules.ocr_benchmark.coverage import ensure_coverage_gate
from modules.ocr_benchmark.dataset import DatasetSplit, read_manifest
from modules.ocr_benchmark.paths import resolve_doc_benchmark_root
from modules.ocr_benchmark.surya_runner import lock_surya_version
from utils.logger import get_logger
from utils.run_registry import register_latest_run

logger = get_logger("OCRBenchmarkSuryaFinetune")


def run_surya_finetune(
    doc_stem: str,
    dataset_manifest: Path,
    *,
    enforce_coverage: bool = True,
    charset_config_path: Path | None = None,
) -> Path:
    """
    Surya finetune scaffolding.
    This stage currently validates prerequisites and records metadata, then exits.
    """
    surya_version = lock_surya_version()
    if not surya_version.startswith("0.17"):
        raise ValueError(
            f"CRITICAL: Found Surya v{surya_version}. The benchmark is strictly pinned "
            f"to v0.17.x to assure API/finetune-script consistency. Aborting."
        )

    doc_root = resolve_doc_benchmark_root(doc_stem)
    run_dir = doc_root / "surya_finetune"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_manifest.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {dataset_manifest}")

    charset_cfg = charset_config_path or (
        settings.INPUT_DIR / "ocr_benchmark" / "config" / "ethiopic_charset.v1.json"
    )
    coverage_report, coverage_out_dir = ensure_coverage_gate(
        doc_stem=doc_stem,
        manifest_path=dataset_manifest,
        charset_config_path=charset_cfg,
        enforce=enforce_coverage,
    )

    rows = [row for row in read_manifest(dataset_manifest) if row.doc_stem == doc_stem]
    holdout_rows = [row for row in rows if row.split == DatasetSplit.HOLDOUT]
    if not holdout_rows:
        raise ValueError(f"No holdout rows found for doc_stem '{doc_stem}' in {dataset_manifest}.")
    train_rows = [row for row in rows if row.split == DatasetSplit.TRAIN]

    register_latest_run(
        stage="ocr-benchmark-surya-finetune",
        doc_stem=doc_stem,
        run_dir=run_dir,
        artifacts={},
        metadata={
            "doc_root": str(doc_root.relative_to(settings.BASE_DIR)),
            "model_name": "surya-recognition",
            "surya_version": surya_version,
            "train_count": len(train_rows),
            "holdout_count": len(holdout_rows),
            "dataset_manifest": str(dataset_manifest),
            "status": "not_implemented",
            "coverage_enforced": enforce_coverage,
            "coverage_status": coverage_report.coverage_status,
            "coverage_report": str(coverage_out_dir / "coverage_report.json"),
        },
    )
    raise NotImplementedError("Surya finetuning pipeline is scaffolded but not implemented yet.")
