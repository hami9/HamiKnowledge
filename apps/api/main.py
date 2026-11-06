"""Main FastAPI application entrypoint for HamiKnowledge."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from apps.api.routers import agent, budget, documents, query, search, workspaces


def create_app() -> FastAPI:
    app = FastAPI(
        title="HamiKnowledge API",
        version="0.1.0",
        description="Multi-user RAG knowledge platform with cited answers and tenant isolation",
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoints
    @app.get("/healthz", tags=["health"])
    def healthz() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/readyz", tags=["health"])
    def readyz() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/", include_in_schema=False)
    def index() -> Response:
        index_file = Path(__file__).resolve().parent.parent / "web" / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return HTMLResponse("<h1>HamiKnowledge API</h1>")

    # Include routers
    app.include_router(workspaces.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    app.include_router(query.router)
    app.include_router(agent.router)
    app.include_router(budget.router)

    return app


app = create_app()
