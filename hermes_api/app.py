"""FastAPI application factory."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hermes_api.routers import chat, config, health, memory, models, sessions, skills, tools


def create_app() -> FastAPI:
    app = FastAPI(
        title="Hermes Agent API",
        description="REST + WebSocket API for Hermes Agent web dashboard",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # CORS — allow React dev servers and custom origins
    origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    custom = os.getenv("HERMES_API_CORS_ORIGINS")
    if custom:
        origins.extend(o.strip() for o in custom.split(",") if o.strip())

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routers
    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
    app.include_router(config.router, prefix="/api/config", tags=["config"])
    app.include_router(tools.router, prefix="/api/tools", tags=["tools"])
    app.include_router(memory.router, prefix="/api/memory", tags=["memory"])
    app.include_router(skills.router, prefix="/api/skills", tags=["skills"])
    app.include_router(models.router, prefix="/api/models", tags=["models"])

    # Serve React static build if available (production mode)
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir() and (static_dir / "index.html").exists():
        from fastapi.staticfiles import StaticFiles
        from starlette.responses import FileResponse

        # Static assets (JS, CSS, images)
        if (static_dir / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets")

        @app.get("/")
        async def serve_index():
            return FileResponse(static_dir / "index.html")

        # SPA fallback — only for non-API paths
        @app.get("/{path:path}")
        async def serve_spa(path: str):
            if path.startswith("api"):
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Not found")
            file_path = static_dir / path
            if file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(static_dir / "index.html")

    return app
