import typer

from evals import benchmark_translation
from utils.logger import get_logger

log = get_logger("Orchestrator")
app = typer.Typer(
    help="Berana: Liturgical Ge'ez Translation & OCR Pipeline Orchestrator", no_args_is_help=True
)


@app.command(name="benchmark-translation")
def run_benchmark_translation(
    gpu_layers: int = typer.Option(
        20, "--gpu-layers", help="Number of layers to offload to GPU. (8GB VRAM: 20)."
    ),
    ctx: int = typer.Option(8192, "--ctx", help="Context window size."),
    temp: float = typer.Option(
        0.0, "--temp", help="Temperature for generation. (Liturgical fidelity: 0.0)."
    ),
) -> None:
    """Run translation benchmarking on evaluation datasets."""
    log.info("Initializing translation benchmark evaluation...")
    benchmark_translation.run_cli_entrypoint(n_gpu_layers=gpu_layers, n_ctx=ctx, temperature=temp)


@app.command(name="ingest")
def run_ingest(
    pdf_path: str = typer.Argument(..., help="Path to the source liturgical PDF."),
    chunk_size: int = typer.Option(
        50, "--chunk-size", help="How many pages to load into RAM at once."
    ),
    dpi: int = typer.Option(300, "--dpi", help="Image processing resolution."),
) -> None:
    """Run OCR and Layout Analysis on a raw PDF, saving Geographic Language bounds to JSON."""
    from pathlib import Path

    from modules.ocr_engine.orchestrator import process_pdf_to_structural_json

    source_path = Path(pdf_path)
    output_dir = Path("output")

    if not source_path.exists():
        log.error(f"Ingest Failed: Source PDF '{source_path}' does not exist.")
        raise typer.Exit(code=1)

    log.info(f"Ingesting PDF from {source_path} into structural JSON... 🚀")

    try:
        final_file = process_pdf_to_structural_json(
            pdf_path=source_path,
            output_dir=output_dir,
            chunk_size=chunk_size,
            dpi=dpi,
        )
        log.info(f"✅ Ingestion Complete. Structural Map saved to: {final_file}")
    except Exception as e:
        log.error(f"❌ OCR Pipeline failed: {e}")
        raise typer.Exit(code=1) from e


@app.command(name="pipeline")
def run_pipeline() -> None:
    """Run full volumetric pipeline: PDF ingest -> Translation -> Export (Future)."""
    log.info("Starting Volumetric Translation Pipeline... (Not Implemented)")


if __name__ == "__main__":
    app()
