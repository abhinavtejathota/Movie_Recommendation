"""FastAPI backend + static frontend for movie recommendations.

Run:  python app.py
Open: http://127.0.0.1:8000
"""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from recommender import DATA_PATH, MovieRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("movie-rec")

ROOT = Path(__file__).parent
STATIC = ROOT / "static"

engine: MovieRecommender | None = None
engine_lock = asyncio.Lock()
shutting_down = False


async def get_engine() -> MovieRecommender:
    global engine
    if shutting_down:
        raise HTTPException(status_code=503, detail="Server is shutting down")
    if engine is not None and engine.ready:
        return engine
    async with engine_lock:
        if engine is None:
            if not DATA_PATH.exists():
                raise HTTPException(
                    status_code=503,
                    detail="movies_features.csv missing. Run model_selection.ipynb first.",
                )
            log.info("Loading recommender (first load builds cache)...")
            engine = await asyncio.to_thread(MovieRecommender, DATA_PATH, True)
            log.info("Recommender ready: %s movies", len(engine.df))
        return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, shutting_down
    shutting_down = False
    # Warm-load in background so /health works immediately
    async def _warm():
        try:
            await get_engine()
        except Exception as exc:
            log.warning("Warm load failed: %s", exc)

    warm_task = asyncio.create_task(_warm())
    yield
    shutting_down = True
    warm_task.cancel()
    try:
        await warm_task
    except asyncio.CancelledError:
        pass
    engine = None
    log.info("Shutdown complete")


app = FastAPI(title="Movie Mood Recommender", lifespan=lifespan)


class MoodRequest(BaseModel):
    mood: str = Field(..., min_length=1)
    language: str | None = None
    count: int = Field(10, ge=1, le=30)
    min_rating: float = Field(0.0, ge=0.0, le=10.0)


class SimilarRequest(BaseModel):
    title: str = Field(..., min_length=1)
    count: int = Field(10, ge=1, le=30)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    ready = engine is not None and getattr(engine, "ready", False)
    return {
        "status": "ok" if ready else "starting",
        "ready": ready,
        "shutting_down": shutting_down,
        "data_exists": DATA_PATH.exists(),
    }


@app.get("/api/info")
async def info(request: Request) -> dict[str, Any]:
    if await request.is_disconnected():
        raise HTTPException(499, "Client disconnected")
    rec = await get_engine()
    return rec.info()


@app.post("/api/recommend/mood")
async def recommend_mood(body: MoodRequest, request: Request) -> dict[str, Any]:
    if await request.is_disconnected():
        raise HTTPException(499, "Client disconnected")
    rec = await get_engine()
    lang = None if not body.language or body.language.lower() == "any" else body.language.lower()

    def _run():
        return rec.recommend_by_mood(
            body.mood, n=body.count, language=lang, min_rating=body.min_rating
        )

    mapped, rows = await asyncio.to_thread(_run)
    if await request.is_disconnected():
        return JSONResponse({"mapped": mapped, "results": [], "cancelled": True})
    return {"mapped": mapped, "results": rows, "cancelled": False}


@app.post("/api/recommend/similar")
async def recommend_similar(body: SimilarRequest, request: Request) -> dict[str, Any]:
    if await request.is_disconnected():
        raise HTTPException(499, "Client disconnected")
    rec = await get_engine()

    rows = await asyncio.to_thread(rec.recommend_similar, body.title, body.count)
    if await request.is_disconnected():
        return JSONResponse({"results": [], "cancelled": True})
    return {"results": rows, "cancelled": False}


@app.get("/api/suggest")
async def suggest(
    request: Request,
    q: str = Query("", min_length=0),
    limit: int = Query(8, ge=1, le=20),
) -> dict[str, Any]:
    if not q.strip():
        return {"suggestions": []}
    if await request.is_disconnected():
        raise HTTPException(499, "Client disconnected")
    rec = await get_engine()
    suggestions = await asyncio.to_thread(rec.search_titles, q, limit)
    return {"suggestions": suggestions}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main() -> None:
    import uvicorn

    def _handle_signal(signum, frame):
        global shutting_down
        shutting_down = True
        log.info("Signal %s received — finishing current work then exiting", signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):
            pass

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8101,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
