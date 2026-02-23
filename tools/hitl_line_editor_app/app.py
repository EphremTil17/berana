from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from tools.hitl_line_editor_app.paths import SOURCE_DIRS, TEMPLATE_FILE
from tools.hitl_line_editor_app.state import initialize_state, load_state, save_page_state
from utils.logger import get_logger

logger = get_logger("LineEditor")


class SaveRequest(BaseModel):
    """Request payload for saving adjusted divider lines for a page."""

    left: list[float] | None
    right: list[float] | None
    verified: bool


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize persistent state once during app startup."""
    initialize_state()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/api/state")
def get_state() -> dict:
    """Return full line-verification state for the editor UI."""
    return load_state()


@app.post("/api/save/{page_id}")
def save_page(page_id: str, req: SaveRequest) -> dict[str, str]:
    """Persist updated divider lines for one page."""
    saved = save_page_state(page_id=page_id, left=req.left, right=req.right, verified=req.verified)
    if not saved:
        raise HTTPException(status_code=404, detail="Page not found")
    return {"status": "ok"}


@app.get("/api/image/{page_id}")
def get_image(page_id: str) -> FileResponse:
    """Serve page image from known source directories."""
    img_filename = f"{page_id}.jpg"
    for directory in SOURCE_DIRS:
        candidate = directory / img_filename
        if candidate.exists():
            return FileResponse(candidate)

    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/", response_class=HTMLResponse)
def serve_ui() -> str:
    """Serve the editor frontend template."""
    return TEMPLATE_FILE.read_text(encoding="utf-8")


def run() -> None:
    """Run the HITL line editor development server."""
    logger.info("Starting Hitl Line Editor on http://localhost:8000")
    uvicorn.run("tools.hitl_line_editor:app", host="0.0.0.0", port=8000, reload=True)
