"""FastAPI application factory for Cascade Studio.

The backend wraps cascade pipeline operations as HTTP endpoints and also
serves the bundled static frontend. Designed to be launched via the
`cascade ui` CLI command, but can also be run directly with uvicorn for
development.

Run for development:
    uvicorn cascade.studio.server:app --reload --port 8000

Run via CLI (production):
    cascade ui
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .. import __version__
from ..exceptions import CascadeError
from .core.config import StudioSettings, get_settings


logger = logging.getLogger(__name__)


# Where the pre-built frontend static files live. Populated at package
# install time. If the directory is empty, the server falls back to a
# minimal "coming soon" page (covered by static/index.html).
STATIC_DIR = Path(__file__).parent / "static"


def _require_fastapi():
    """Lazy import so cascade-agent works without the [studio] extra."""
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:
        raise CascadeError(
            "Cascade Studio dependencies are not installed. Run:\n"
            "  pip install cascade-agent[studio]"
        ) from exc


def create_app(settings: Optional[StudioSettings] = None):
    """Build the FastAPI application.

    Args:
        settings: Optional explicit settings (mostly for tests). Falls
            back to env-var-driven defaults.

    Returns:
        A FastAPI app instance ready to serve.

    Raises:
        CascadeError: If the [studio] extra is not installed.
    """
    _require_fastapi()

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles

    settings = settings or get_settings()

    app = FastAPI(
        title="Cascade Studio",
        description="Web dashboard for the Cascade SDLC accelerator",
        version=__version__,
        docs_url="/api/docs",  # under /api/ to avoid colliding with the frontend
        redoc_url=None,
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ---- API routes ----
    from .api import health

    app.include_router(health.router, tags=["health"])

    # Future feature routes mount here:
    # from .api import projects, stories, builds
    # app.include_router(projects.router, prefix="/api/projects", tags=["projects"])

    # ---- Static frontend ----
    _mount_static_frontend(app, STATIC_DIR)

    return app


def _mount_static_frontend(app, static_dir: Path) -> None:
    """Serve the bundled Next.js frontend.

    Mounts the static directory at /, plus an SPA fallback that returns
    index.html for unmatched non-API routes (so client-side navigation
    works after a refresh).

    If the static directory is missing or empty, returns a simple
    "coming soon" message instead of failing.
    """
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from starlette.exceptions import HTTPException

    index_path = static_dir / "index.html"

    if not index_path.exists():
        # Nothing bundled. Surface a clear message so users understand.
        @app.get("/", response_class=HTMLResponse)
        def _missing_frontend():
            return _FALLBACK_HTML
        return

    # Mount the directory at /, but we still need SPA fallback for
    # routes like /projects/123 that should serve index.html.
    @app.get("/", response_class=HTMLResponse)
    def _root():
        return FileResponse(index_path)

    # Mount /_next/ and other asset paths as static files
    app.mount(
        "/_next",
        StaticFiles(directory=static_dir / "_next") if (static_dir / "_next").exists()
        else StaticFiles(directory=static_dir),
        name="next-assets",
    )

    # Catch-all for SPA routes: any non-API path that doesn't match an
    # actual file falls back to index.html
    @app.get("/{full_path:path}", response_class=HTMLResponse)
    def _spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        candidate = static_dir / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index_path)


_FALLBACK_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Cascade Studio</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #05070d; color: #e5e7eb;
           margin: 0; min-height: 100vh; display: grid; place-items: center; }
    main { max-width: 36rem; padding: 2rem; }
    h1 { font-size: 2rem; background: linear-gradient(120deg, #22d3ee, #a78bfa, #e879f9);
         -webkit-background-clip: text; background-clip: text; color: transparent; }
    p { line-height: 1.6; color: #94a3b8; }
    code { background: #1e293b; padding: 0.1em 0.4em; border-radius: 4px; color: #22d3ee; }
  </style>
</head>
<body>
  <main>
    <h1>Cascade Studio</h1>
    <p>The frontend isn't bundled in this installation. The Studio backend is
    running (API available at <code>/api/docs</code>), but the UI files are
    missing from <code>cascade/studio/static/</code>.</p>
    <p>This usually means you're running cascade-agent from source without
    building the frontend. See the README for development instructions.</p>
  </main>
</body>
</html>
"""


# Module-level app for `uvicorn cascade.studio.server:app`
app = create_app()
