from pathlib import Path
from typing import Annotated

import typer

from modules.cli.common import ensure_pdf_exists
from utils.logger import get_logger

app = typer.Typer(help="OCR Benchmark pilot commands.", no_args_is_help=True)
log = get_logger("BenchmarkCLI")


def _doc_stem_from_pdf(pdf_path: Path, context_label: str) -> str:
    """Resolve and validate source PDF, returning canonical doc stem."""
    source_pdf = ensure_pdf_exists(str(pdf_path), context_label=context_label)
    return source_pdf.stem


@app.command()
def prepare_lines(
    pdf_path: Annotated[Path, typer.Option("--pdf-path", help="Path to the source PDF.")],
    languages: Annotated[
        str,
        typer.Option(
            "--languages",
            help="Comma-separated benchmark languages to include: geez,amharic",
        ),
    ] = "geez,amharic",
    refresh_freeze: Annotated[
        bool,
        typer.Option(
            "--refresh-freeze",
            help="Overwrite existing frozen split hash for this document.",
        ),
    ] = False,
):
    """Extract line crops from layout splicing for benchmarking."""
    from modules.ocr_benchmark.prepare import generate_candidate_lines

    doc_stem = _doc_stem_from_pdf(pdf_path, context_label="Benchmark prepare-lines failed")
    selected_languages = [item.strip().lower() for item in languages.split(",") if item.strip()]
    log.info(
        "Preparing benchmark lines for doc=%s languages=%s refresh_freeze=%s",
        doc_stem,
        selected_languages,
        refresh_freeze,
    )
    try:
        run_dir = generate_candidate_lines(
            doc_stem,
            include_languages=selected_languages,
            refresh_frozen_split=refresh_freeze,
        )
    except (FileNotFoundError, ValueError) as exc:
        log.error("prepare-lines failed for doc=%s: %s", doc_stem, exc)
        raise typer.Exit(code=1) from exc
    log.info("prepare-lines completed for doc=%s run_dir=%s", doc_stem, run_dir)


@app.command()
def run_surya_baseline(
    pdf_path: Annotated[Path, typer.Option("--pdf-path", help="Path to the source PDF.")],
    split: Annotated[
        str,
        typer.Option(
            "--split",
            help="Which split to run zero-shot on: train|holdout|all. Default=all.",
        ),
    ] = "all",
):
    """Run Surya Zero-Shot baseline (required before Label Studio export)."""
    from modules.ocr_benchmark.surya_runner import run_zero_shot_baseline

    doc_stem = _doc_stem_from_pdf(pdf_path, context_label="Benchmark Surya baseline failed")
    log.info("Running Surya baseline for doc=%s split=%s", doc_stem, split)
    try:
        run_dir = run_zero_shot_baseline(doc_stem, split=split)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("Surya baseline failed for doc=%s: %s", doc_stem, exc)
        raise typer.Exit(code=1) from exc
    log.info("Surya baseline completed for doc=%s run_dir=%s", doc_stem, run_dir)


@app.command()
def run_trocr_baseline(
    pdf_path: Annotated[Path, typer.Option("--pdf-path", help="Path to the source PDF.")],
):
    """Run TrOCR Zero-Shot baseline for benchmark comparison."""
    from modules.ocr_benchmark.trocr_runner import run_zero_shot_baseline

    doc_stem = _doc_stem_from_pdf(pdf_path, context_label="Benchmark TrOCR baseline failed")
    log.info("Running TrOCR baseline for doc=%s split=all", doc_stem)
    try:
        run_dir = run_zero_shot_baseline(doc_stem, split="all")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("TrOCR baseline failed for doc=%s: %s", doc_stem, exc)
        raise typer.Exit(code=1) from exc
    log.info("TrOCR baseline completed for doc=%s run_dir=%s", doc_stem, run_dir)


@app.command()
def zero_shot_bakeoff(
    pdf_path: Annotated[Path, typer.Option("--pdf-path", help="Path to the source PDF.")],
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="Canonical GT Line Manifest JSONL path."),
    ] = Path("input/ocr_benchmark/line_manifest.jsonl"),
):
    """One-command baseline bakeoff: run Surya+TrOCR zero-shot and evaluate."""
    from modules.ocr_benchmark.reporting import evaluate_models
    from modules.ocr_benchmark.surya_runner import run_zero_shot_baseline as run_surya_zero
    from modules.ocr_benchmark.trocr_runner import run_zero_shot_baseline as run_trocr_zero

    doc_stem = _doc_stem_from_pdf(pdf_path, context_label="Benchmark zero-shot-bakeoff failed")
    log.info("Running zero-shot bakeoff for doc=%s", doc_stem)
    try:
        run_surya_zero(doc_stem, split="all")
        run_trocr_zero(doc_stem, split="all")
        out_dir = evaluate_models(
            doc_stem=doc_stem,
            manifest_path=manifest,
            require_all_models=False,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("zero-shot-bakeoff failed for doc=%s: %s", doc_stem, exc)
        raise typer.Exit(code=1) from exc
    log.info("Zero-shot bakeoff completed for doc=%s final_eval=%s", doc_stem, out_dir)


@app.command()
def make_ls_tasks(
    pdf_path: Annotated[Path, typer.Option("--pdf-path", help="Path to the source PDF.")],
    split: Annotated[
        str,
        typer.Option("--split", help="Dataset split to export: train|holdout|all."),
    ] = "holdout",
    output_json: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Output JSON task path. "
                "Defaults to output/ocr_benchmark/<doc_stem_vNN>/label_studio/"
                "ocr_transcription_tasks_<split>.json "
                "for the active benchmark run."
            )
        ),
    ] = None,
):
    """Generate Label Studio text-transcription tasks populated with pre-annotations."""
    from modules.ocr_benchmark.label_studio_sync import create_import_tasks
    from modules.ocr_benchmark.paths import resolve_doc_benchmark_root

    doc_stem = _doc_stem_from_pdf(pdf_path, context_label="Benchmark make-ls-tasks failed")
    log.info("Generating Label Studio tasks for doc=%s", doc_stem)
    if output_json is None:
        output_json = (
            resolve_doc_benchmark_root(doc_stem)
            / "label_studio"
            / f"ocr_transcription_tasks_{split.lower()}.json"
        )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        create_import_tasks(doc_stem, output_json, split=split)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("make-ls-tasks failed for doc=%s: %s", doc_stem, exc)
        raise typer.Exit(code=1) from exc
    log.info("Label Studio tasks generated for doc=%s output=%s", doc_stem, output_json)


@app.command()
def import_ls_export(
    export_json: Annotated[
        Path,
        typer.Option(help="Label Studio exported JSON annotations path."),
    ],
    manifest: Annotated[
        Path,
        typer.Option(help="Target Line Manifest JSONL path."),
    ],
    strict_page_isolation: Annotated[
        bool,
        typer.Option(
            "--strict-page-isolation/--allow-page-overlap",
            help=(
                "Enforce page-level split isolation during import. "
                "Line-level leakage is always blocked."
            ),
        ),
    ] = False,
):
    """Idempotently import verified Label Studio annotations into canonical GT manifest."""
    from modules.ocr_benchmark.label_studio_sync import parse_export

    log.info("Importing Label Studio export=%s manifest=%s", export_json, manifest)
    try:
        parse_export(
            export_json,
            manifest,
            strict_page_isolation=strict_page_isolation,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("import-ls-export failed export=%s manifest=%s: %s", export_json, manifest, exc)
        raise typer.Exit(code=1) from exc
    log.info("Label Studio import complete manifest=%s", manifest)


@app.command()
def run_trocr_finetuning(
    pdf_path: Annotated[Path, typer.Option("--pdf-path", help="Path to the source PDF.")],
    manifest: Annotated[
        Path,
        typer.Option(help="Canonical GT Line Manifest JSONL path."),
    ],
    enforce_coverage: Annotated[
        bool,
        typer.Option(
            "--enforce-coverage/--no-enforce-coverage",
            help="Hard fail finetuning when coverage thresholds are unmet.",
        ),
    ] = True,
    charset_config: Annotated[
        Path,
        typer.Option("--charset-config", help="Path to Ethiopic charset coverage config."),
    ] = Path("input/ocr_benchmark/config/ethiopic_charset.v1.json"),
):
    """Run TrOCR finetuning, automatically executing tokenizer preflights."""
    from modules.ocr_benchmark.trocr_finetuner import run_trocr_finetune

    doc_stem = _doc_stem_from_pdf(pdf_path, context_label="Benchmark TrOCR finetuning failed")
    log.info("Running TrOCR finetuning for doc=%s manifest=%s", doc_stem, manifest)
    try:
        run_dir = run_trocr_finetune(
            doc_stem,
            manifest,
            enforce_coverage=enforce_coverage,
            charset_config_path=charset_config,
        )
    except (NotImplementedError, FileNotFoundError, ValueError) as exc:
        log.error("TrOCR finetuning failed for doc=%s: %s", doc_stem, exc)
        raise typer.Exit(code=1) from exc
    log.info("TrOCR finetuning completed for doc=%s run_dir=%s", doc_stem, run_dir)


@app.command()
def run_surya_finetuning(
    pdf_path: Annotated[Path, typer.Option("--pdf-path", help="Path to the source PDF.")],
    manifest: Annotated[
        Path,
        typer.Option(help="Canonical GT Line Manifest JSONL path."),
    ],
    enforce_coverage: Annotated[
        bool,
        typer.Option(
            "--enforce-coverage/--no-enforce-coverage",
            help="Hard fail finetuning when coverage thresholds are unmet.",
        ),
    ] = True,
    charset_config: Annotated[
        Path,
        typer.Option("--charset-config", help="Path to Ethiopic charset coverage config."),
    ] = Path("input/ocr_benchmark/config/ethiopic_charset.v1.json"),
):
    """Run Surya finetuning stage."""
    from modules.ocr_benchmark.surya_finetuner import run_surya_finetune

    doc_stem = _doc_stem_from_pdf(pdf_path, context_label="Benchmark Surya finetuning failed")
    log.info("Running Surya finetuning for doc=%s manifest=%s", doc_stem, manifest)
    try:
        run_dir = run_surya_finetune(
            doc_stem,
            manifest,
            enforce_coverage=enforce_coverage,
            charset_config_path=charset_config,
        )
    except (NotImplementedError, FileNotFoundError, ValueError) as exc:
        log.error("Surya finetuning failed for doc=%s: %s", doc_stem, exc)
        raise typer.Exit(code=1) from exc
    log.info("Surya finetuning completed for doc=%s run_dir=%s", doc_stem, run_dir)


@app.command()
def evaluate(
    pdf_path: Annotated[Path, typer.Option("--pdf-path", help="Path to the source PDF.")],
    manifest: Annotated[
        Path,
        typer.Option(help="Canonical GT Line Manifest JSONL path."),
    ] = Path("input/ocr_benchmark/line_manifest.jsonl"),
    require_all_models: Annotated[
        bool,
        typer.Option(
            "--require-all-models/--allow-missing-models",
            help=(
                "Require every benchmark model stage to exist. "
                "Use --allow-missing-models to evaluate available models only."
            ),
        ),
    ] = True,
):
    """Evaluate predictions against canonical GT dataset, generating decision payloads."""
    from modules.ocr_benchmark.reporting import evaluate_models

    doc_stem = _doc_stem_from_pdf(pdf_path, context_label="Benchmark evaluation failed")
    log.info(
        "Evaluating benchmark models for doc=%s manifest=%s require_all_models=%s",
        doc_stem,
        manifest,
        require_all_models,
    )
    try:
        out_dir = evaluate_models(
            doc_stem,
            manifest_path=manifest,
            require_all_models=require_all_models,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("evaluate failed for doc=%s: %s", doc_stem, exc)
        raise typer.Exit(code=1) from exc
    log.info("Benchmark evaluation completed for doc=%s out_dir=%s", doc_stem, out_dir)


@app.command()
def coverage_report(
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="Canonical GT Line Manifest JSONL path."),
    ] = Path("input/ocr_benchmark/line_manifest.jsonl"),
    charset_config: Annotated[
        Path,
        typer.Option("--charset-config", help="Path to Ethiopic charset coverage config."),
    ] = Path("input/ocr_benchmark/config/ethiopic_charset.v1.json"),
    doc_stem: Annotated[
        str | None,
        typer.Option(
            "--doc-stem",
            help="Optional document stem filter. If omitted, must resolve to a single doc in manifest.",
        ),
    ] = None,
):
    """Generate deterministic character coverage report and artifacts."""
    from modules.ocr_benchmark.coverage import build_coverage_report
    from modules.ocr_benchmark.dataset import read_manifest

    if doc_stem is None:
        rows = read_manifest(manifest)
        stems = sorted({row.doc_stem for row in rows})
        if len(stems) != 1:
            raise typer.BadParameter(
                f"--doc-stem is required because manifest has {len(stems)} doc stems: {stems}"
            )
        doc_stem = stems[0]

    log.info(
        "Generating coverage report for doc=%s manifest=%s charset=%s",
        doc_stem,
        manifest,
        charset_config,
    )
    try:
        report, out_dir = build_coverage_report(
            doc_stem=doc_stem,
            manifest_path=manifest,
            charset_config_path=charset_config,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("coverage-report failed for doc=%s: %s", doc_stem, exc)
        raise typer.Exit(code=1) from exc
    log.info(
        "Coverage report complete for doc=%s status=%s out_dir=%s",
        doc_stem,
        report.coverage_status,
        out_dir,
    )


@app.command()
def coverage_queue(
    pdf_path: Annotated[Path, typer.Option("--pdf-path", help="Path to the source PDF.")],
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="Canonical GT Line Manifest JSONL path."),
    ] = Path("input/ocr_benchmark/line_manifest.jsonl"),
    charset_config: Annotated[
        Path,
        typer.Option("--charset-config", help="Path to Ethiopic charset coverage config."),
    ] = Path("input/ocr_benchmark/config/ethiopic_charset.v1.json"),
    max_items: Annotated[
        int,
        typer.Option("--max-items", help="Maximum queue items to emit."),
    ] = 200,
):
    """Create ranked annotation queue for under-covered characters."""
    from modules.ocr_benchmark.coverage import build_annotation_queue

    doc_stem = _doc_stem_from_pdf(pdf_path, context_label="Benchmark coverage-queue failed")
    log.info(
        "Generating coverage queue for doc=%s manifest=%s max_items=%d",
        doc_stem,
        manifest,
        max_items,
    )
    try:
        out_path = build_annotation_queue(
            doc_stem=doc_stem,
            manifest_path=manifest,
            charset_config_path=charset_config,
            max_items=max_items,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("coverage-queue failed for doc=%s: %s", doc_stem, exc)
        raise typer.Exit(code=1) from exc
    log.info("Coverage queue generated for doc=%s at %s", doc_stem, out_path)


@app.command()
def generate_charset_config(
    output_path: Annotated[
        Path,
        typer.Option("--output-path", help="Destination charset config JSON path."),
    ] = Path("input/ocr_benchmark/config/ethiopic_charset.v2.unicode.json"),
    manifest: Annotated[
        Path | None,
        typer.Option(
            "--manifest",
            help=(
                "Optional manifest JSONL for data-driven tier assignment. "
                "If omitted, all Ethiopic letters are emitted as optional."
            ),
        ),
    ] = None,
    pdf_path: Annotated[
        Path | None,
        typer.Option(
            "--pdf-path",
            help="Optional source PDF to derive doc stem filter when manifest is provided.",
        ),
    ] = None,
    high_min_count: Annotated[
        int,
        typer.Option("--high-min-count", help="Minimum required count for high-tier coverage."),
    ] = 20,
    medium_min_count: Annotated[
        int,
        typer.Option("--medium-min-count", help="Minimum required count for medium-tier coverage."),
    ] = 10,
    rare_min_count: Annotated[
        int,
        typer.Option("--rare-min-count", help="Minimum required count for rare-tier coverage."),
    ] = 5,
    source: Annotated[
        str,
        typer.Option("--source", help="Character source: unicode|wiktionary."),
    ] = "unicode",
    wiktionary_url: Annotated[
        str,
        typer.Option(
            "--wiktionary-url",
            help="Wiktionary page URL used when --source wiktionary.",
        ),
    ] = "https://en.wiktionary.org/wiki/Appendix:Unicode/Ethiopic",
    declaration_only: Annotated[
        bool,
        typer.Option(
            "--declaration-only/--include-tier-charlists",
            help=(
                "Declaration-only mode keeps config compact and Unicode-driven. "
                "Use --include-tier-charlists to emit explicit tier character lists."
            ),
        ),
    ] = True,
):
    """Generate Ethiopic charset config from Unicode ranges and optional manifest frequencies."""
    from modules.ocr_benchmark.charset_builder import generate_unicode_charset_config

    doc_stem: str | None = None
    if pdf_path is not None:
        doc_stem = _doc_stem_from_pdf(
            pdf_path, context_label="Benchmark generate-charset-config failed"
        )

    log.info(
        "Generating Unicode charset config output=%s manifest=%s doc_stem=%s",
        output_path,
        manifest,
        doc_stem,
    )
    try:
        cfg = generate_unicode_charset_config(
            output_path=output_path,
            manifest_path=manifest,
            doc_stem=doc_stem,
            high_min_count=high_min_count,
            medium_min_count=medium_min_count,
            rare_min_count=rare_min_count,
            source=source.lower(),
            wiktionary_url=wiktionary_url,
            declaration_only=declaration_only,
        )
    except (FileNotFoundError, ValueError) as exc:
        log.error("generate-charset-config failed: %s", exc)
        raise typer.Exit(code=1) from exc

    tier_counts = {getattr(k, "value", str(k)): len(v.chars) for k, v in cfg.tiers.items()}
    log.info(
        "Charset config generated: high=%d medium=%d rare=%d optional=%d path=%s",
        tier_counts.get("high", 0),
        tier_counts.get("medium", 0),
        tier_counts.get("rare", 0),
        tier_counts.get("optional", 0),
        output_path,
    )


@app.command("auto")
def finalize_after_labeling(
    pdf_path: Annotated[Path, typer.Option("--pdf-path", help="Path to the source PDF.")],
    holdout_export_json: Annotated[
        Path | None,
        typer.Option(
            "--holdout-export-json", help="Optional Label Studio export JSON for holdout split."
        ),
    ] = None,
    train_export_json: Annotated[
        Path | None,
        typer.Option(
            "--train-export-json", help="Optional Label Studio export JSON for train split."
        ),
    ] = None,
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="Canonical GT Line Manifest JSONL path."),
    ] = Path("input/ocr_benchmark/line_manifest.jsonl"),
    charset_config: Annotated[
        Path,
        typer.Option("--charset-config", help="Coverage policy charset config path."),
    ] = Path("input/ocr_benchmark/config/ethiopic_charset.v1.json"),
    run_queue: Annotated[
        bool,
        typer.Option(
            "--run-queue/--no-run-queue",
            help="Generate coverage queue after coverage report.",
        ),
    ] = True,
    run_finetune: Annotated[
        bool,
        typer.Option(
            "--run-finetune/--no-run-finetune",
            help="Attempt finetuning stages (currently scaffolded).",
        ),
    ] = False,
):
    """
    One-shot post-Label Studio pipeline:
    import exports -> coverage report -> optional queue -> optional finetune -> evaluate.
    """
    from modules.ocr_benchmark.coverage import build_annotation_queue, build_coverage_report
    from modules.ocr_benchmark.label_studio_sync import parse_export
    from modules.ocr_benchmark.reporting import evaluate_models
    from modules.ocr_benchmark.surya_finetuner import run_surya_finetune
    from modules.ocr_benchmark.trocr_finetuner import run_trocr_finetune

    doc_stem = _doc_stem_from_pdf(
        pdf_path, context_label="Benchmark finalize-after-labeling failed"
    )
    log.info("Finalize pipeline started for doc=%s", doc_stem)

    if holdout_export_json is not None:
        log.info("Importing holdout export: %s", holdout_export_json)
        parse_export(holdout_export_json, manifest)
    if train_export_json is not None:
        log.info("Importing train export: %s", train_export_json)
        parse_export(train_export_json, manifest)
    if holdout_export_json is None and train_export_json is None:
        log.info("No export JSONs provided; reusing existing manifest data at %s", manifest)

    report, coverage_dir = build_coverage_report(
        doc_stem=doc_stem,
        manifest_path=manifest,
        charset_config_path=charset_config,
    )
    log.info("Coverage status=%s report_dir=%s", report.coverage_status, coverage_dir)

    if run_queue:
        try:
            queue_path = build_annotation_queue(
                doc_stem=doc_stem,
                manifest_path=manifest,
                charset_config_path=charset_config,
                max_items=200,
            )
            log.info("Coverage queue generated: %s", queue_path)
        except ValueError as exc:
            log.warning("Coverage queue skipped: %s", exc)

    if run_finetune:
        try:
            run_surya_finetune(
                doc_stem,
                manifest,
                enforce_coverage=True,
                charset_config_path=charset_config,
            )
        except NotImplementedError as exc:
            log.warning("Surya finetune scaffold not implemented: %s", exc)
        try:
            run_trocr_finetune(
                doc_stem,
                manifest,
                enforce_coverage=True,
                charset_config_path=charset_config,
            )
        except NotImplementedError as exc:
            log.warning("TrOCR finetune scaffold not implemented: %s", exc)

    # Finetune predictions are not always present; evaluate available models only.
    out_dir = evaluate_models(
        doc_stem=doc_stem,
        manifest_path=manifest,
        require_all_models=False,
    )
    log.info("Finalize pipeline completed for doc=%s final_eval=%s", doc_stem, out_dir)
