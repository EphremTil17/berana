import sys

import typer

from modules.cli.layout_commands import run_layout_infer, run_layout_prep, run_train_layout
from modules.cli.ocr_commands import (
    run_crop_columns,
    run_ingest,
    run_ocr,
    run_ocr_infer,
    run_ocr_train,
    run_poc_slicer,
)
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
    from evals import benchmark_translation

    log.info("Initializing translation benchmark evaluation...")
    benchmark_translation.run_cli_entrypoint(n_gpu_layers=gpu_layers, n_ctx=ctx, temperature=temp)


app.command(name="ingest")(run_ingest)
app.command(name="poc-slicer")(run_poc_slicer)
app.command(name="layout-prep")(run_layout_prep)
app.command(name="train-layout")(run_train_layout)
app.command(name="layout-infer")(run_layout_infer)
app.command(name="crop-columns")(run_crop_columns)
app.command(name="ocr")(run_ocr)
app.command(name="ocr-train")(run_ocr_train)
app.command(name="ocr-infer")(run_ocr_infer)


if __name__ == "__main__":
    app()
    sys.exit(0)
