"""
main.py – FastAPI application entry point.

Provides:
  GET  /          → serves index.html
  WS   /ws        → real-time voice chat
  GET  /health    → health check
  GET  /history/{session_id} → last 20 messages from memory
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# ── Load .env FIRST so env vars are available everywhere ─────────────────────
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Ensure our package root is on sys.path ───────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from memory import init_db, get_recent_messages, get_all_sessions
from websocket_handler import handle_websocket

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── Lifespan (replaces deprecated on_event) ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup logic, then yield, then shutdown logic."""
    init_db()
    logger.info("Database initialised.")
    audio_dir = Path(__file__).parent.parent / "audio_cache"
    audio_dir.mkdir(exist_ok=True)
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key.startswith("gsk_"):
        logger.info("GROQ_API_KEY loaded ✓")
    else:
        logger.warning("GROQ_API_KEY not set or invalid – LLM/STT will use fallback")
    yield  # app runs here
    # (cleanup on shutdown if needed)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Aria Voice Chat", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (frontend) ───────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/history/{session_id}")
async def history(session_id: str, n: int = 20) -> JSONResponse:
    """Return recent messages for a given session."""
    msgs = get_recent_messages(n=n, session_id=session_id)
    return JSONResponse({"session_id": session_id, "messages": msgs})


@app.get("/sessions")
async def sessions() -> JSONResponse:
    return JSONResponse({"sessions": get_all_sessions()})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await handle_websocket(ws)


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
