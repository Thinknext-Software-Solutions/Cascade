"""/api/projects endpoints: list projects, list/show one project."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ...exceptions import CascadeError
from ..core.config import StudioSettings, get_settings
from ..models.api import ProjectDetail, ProjectSummary, StoryBatchSummary
from ..services.projects import discover_projects, get_project


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from fastapi import APIRouter


def _get_router():
    from fastapi import APIRouter
    return APIRouter()


router = _get_router()


def _to_summary(p) -> ProjectSummary:
    return ProjectSummary(
        id=p.id,
        name=p.name,
        path=str(p.absolute_path),
        language=p.language,
        transcript_count=p.transcript_count,
        story_batch_count=p.story_batch_count,
    )


def _project_workspace_root(settings: StudioSettings) -> Path:
    """Resolve the directory Studio scans for projects."""
    if settings.workspace_root is not None:
        return settings.workspace_root
    return Path.cwd()


def _build_batch_summaries(project_path: Path) -> list[StoryBatchSummary]:
    """Walk project_path/stories/ and summarize each batch."""
    from ...io import read_story_batch

    stories_dir = project_path / "stories"
    if not stories_dir.is_dir():
        return []

    summaries: list[StoryBatchSummary] = []
    for yaml_path in sorted(stories_dir.glob("*.yaml"), reverse=True):
        try:
            batch = read_story_batch(yaml_path)
        except Exception as exc:
            logger.warning(
                "batch.summary.skip",
                extra={"file": str(yaml_path), "reason": str(exc)},
            )
            continue
        counts = _count_statuses(batch.stories)
        summaries.append(
            StoryBatchSummary(
                id=yaml_path.stem,
                file_name=yaml_path.name,
                meeting_id=batch.meeting_id,
                extracted_at=batch.extracted_at,
                extractor_model=batch.extractor_model,
                story_count=len(batch.stories),
                approved_count=counts["approved"],
                rejected_count=counts["rejected"],
                edited_count=counts["edited"],
                pending_count=counts["pending"],
            )
        )
    return summaries


def _count_statuses(stories) -> dict[str, int]:
    from ...schemas import StoryStatus

    out = {"approved": 0, "rejected": 0, "edited": 0, "pending": 0}
    for s in stories:
        if s.status == StoryStatus.APPROVED:
            out["approved"] += 1
        elif s.status == StoryStatus.REJECTED:
            out["rejected"] += 1
        elif s.status == StoryStatus.EDITED:
            out["edited"] += 1
        else:
            out["pending"] += 1
    return out


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------


def setup_routes():
    """Build and return the router. Called by server.py at app construction."""
    from fastapi import APIRouter, Depends, HTTPException

    r = APIRouter(prefix="/api/projects", tags=["projects"])

    @r.get("", response_model=list[ProjectSummary])
    def list_projects(settings: StudioSettings = Depends(get_settings)):
        """List every Cascade-enabled project under the configured workspace."""
        try:
            root = _project_workspace_root(settings)
            projects = discover_projects(root)
        except CascadeError as exc:
            raise HTTPException(status_code=400, detail=exc.message)
        return [_to_summary(p) for p in projects]

    @r.get("/{project_id}", response_model=ProjectDetail)
    def get_project_detail(
        project_id: str, settings: StudioSettings = Depends(get_settings)
    ):
        """Return one project plus its story batches."""
        root = _project_workspace_root(settings)
        project = get_project(project_id, root)
        if project is None:
            raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
        batches = _build_batch_summaries(project.absolute_path)
        return ProjectDetail(project=_to_summary(project), batches=batches)

    return r


router = setup_routes()
