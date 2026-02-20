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
) -> None:
    """Run OCR and Layout Analysis on a raw PDF (Future)."""
    log.info(f"Ingesting PDF from {pdf_path} into structural JSON... (Not Implemented)")


@app.command(name="pipeline")
def run_pipeline() -> None:
    """Run full volumetric pipeline: PDF ingest -> Translation -> Export (Future)."""
    log.info("Starting Volumetric Translation Pipeline... (Not Implemented)")


if __name__ == "__main__":
    app()
