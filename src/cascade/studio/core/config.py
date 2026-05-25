"""Studio server settings.

Loaded from environment variables (prefixed STUDIO_) or .env files. Single
source of truth for ports, hosts, CORS, database URLs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field


try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _PYDANTIC_SETTINGS_AVAILABLE = True
except ImportError:
    _PYDANTIC_SETTINGS_AVAILABLE = False

    # Fallback: a plain Pydantic model with no env var loading. Used only
    # when cascade-agent is imported without the [studio] extra; tests
    # and the CLI both ensure pydantic_settings is present before this
    # code path is hit.
    from pydantic import BaseModel

    class BaseSettings(BaseModel):
        pass

    def SettingsConfigDict(**kwargs):
        return None


class StudioSettings(BaseSettings):
    """Runtime config for the Studio backend."""

    if _PYDANTIC_SETTINGS_AVAILABLE:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_prefix="STUDIO_",
            extra="ignore",
        )

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # For dev: Next.js dev server runs on :3000 and needs CORS. In bundled
    # mode the frontend is same-origin so CORS isn't strictly needed.
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # Where to look for Cascade-enabled repos in the user's workspace
    workspace_root: Optional[Path] = None


@lru_cache
def get_settings() -> StudioSettings:
    """Cached settings accessor. FastAPI Depends() injects this."""
    return StudioSettings()
