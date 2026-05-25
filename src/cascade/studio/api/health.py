"""Health and metadata endpoints. The frontend uses these to discover
which providers / languages / VCS / trackers are supported."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from ... import __version__


if TYPE_CHECKING:
    from fastapi import APIRouter


def _get_router():
    """Lazy import to allow loading this module without fastapi."""
    from fastapi import APIRouter
    return APIRouter()


router = _get_router()


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check used by the frontend and any uptime monitors."""
    return HealthResponse(status="ok", version=__version__)


class MetadataResponse(BaseModel):
    name: str
    version: str
    cascade_supports: dict[str, list[str]]


@router.get("/api/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    """Surface what this Cascade installation can reach.

    The frontend uses this to decide what provider options to show in
    config forms, what languages to offer in dropdowns, etc.
    """
    from ...issue_sources import SUPPORTED_ISSUE_SOURCES
    from ...languages import PROFILES
    from ...llm import SUPPORTED_PROVIDERS as LLM_PROVIDERS
    from ...vcs import SUPPORTED_VCS_PROVIDERS as VCS_PROVIDERS

    return MetadataResponse(
        name="Cascade Studio",
        version=__version__,
        cascade_supports={
            "languages": sorted(PROFILES.keys()),
            "llm_providers": list(LLM_PROVIDERS),
            "vcs_providers": list(VCS_PROVIDERS),
            "issue_sources": list(SUPPORTED_ISSUE_SOURCES),
        },
    )
