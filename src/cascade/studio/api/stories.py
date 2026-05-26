"""/api/projects/{id}/batches endpoints: read story batches, apply decisions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ...exceptions import CascadeError
from ...io import read_story_batch, write_story_batch
from ...schemas import Story, StoryStatus
from ..core.config import StudioSettings, get_settings
from ..models.api import (
    AcceptanceCriterionView,
    StoryBatchDetail,
    StoryBatchSummary,
    StoryDecisionRequest,
    StoryDecisionResponse,
    StoryView,
)
from ..services.projects import get_project
from .projects import _build_batch_summaries, _count_statuses


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from fastapi import APIRouter


def _story_to_view(story: Story) -> StoryView:
    return StoryView(
        id=story.id,
        title=story.title,
        description=story.description,
        acceptance_criteria=[
            AcceptanceCriterionView(given=ac.given, when=ac.when, then=ac.then)
            for ac in story.acceptance_criteria
        ],
        size=story.size.value,
        confidence=story.confidence,
        status=story.status.value,
        notes=story.notes,
        source_meeting_id=story.source_meeting_id,
    )


def _batch_summary(batch_yaml: Path, batch) -> StoryBatchSummary:
    counts = _count_statuses(batch.stories)
    return StoryBatchSummary(
        id=batch_yaml.stem,
        file_name=batch_yaml.name,
        meeting_id=batch.meeting_id,
        extracted_at=batch.extracted_at,
        extractor_model=batch.extractor_model,
        story_count=len(batch.stories),
        approved_count=counts["approved"],
        rejected_count=counts["rejected"],
        edited_count=counts["edited"],
        pending_count=counts["pending"],
    )


def _resolve_batch_yaml(
    project_id: str, batch_id: str, settings: StudioSettings
) -> Path:
    """Find the YAML file for one batch in one project. Raises HTTPException
    on 404."""
    from fastapi import HTTPException

    root = settings.workspace_root or Path.cwd()
    project = get_project(project_id, root)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    yaml_path = project.absolute_path / "stories" / f"{batch_id}.yaml"
    if not yaml_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Story batch '{batch_id}' not found in project '{project.name}'",
        )
    return yaml_path


def setup_routes():
    """Build and return the router."""
    from fastapi import APIRouter, Depends, HTTPException

    r = APIRouter(prefix="/api/projects", tags=["stories"])

    @r.get("/{project_id}/batches", response_model=list[StoryBatchSummary])
    def list_batches(
        project_id: str, settings: StudioSettings = Depends(get_settings)
    ):
        """List all story batches for one project."""
        root = settings.workspace_root or Path.cwd()
        project = get_project(project_id, root)
        if project is None:
            raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
        return _build_batch_summaries(project.absolute_path)

    @r.get(
        "/{project_id}/batches/{batch_id}",
        response_model=StoryBatchDetail,
    )
    def get_batch(
        project_id: str,
        batch_id: str,
        settings: StudioSettings = Depends(get_settings),
    ):
        """Get one batch with every story (for the review board)."""
        yaml_path = _resolve_batch_yaml(project_id, batch_id, settings)
        try:
            batch = read_story_batch(yaml_path)
        except CascadeError as exc:
            raise HTTPException(status_code=500, detail=exc.message)
        return StoryBatchDetail(
            batch=_batch_summary(yaml_path, batch),
            stories=[_story_to_view(s) for s in batch.stories],
        )

    @r.post(
        "/{project_id}/batches/{batch_id}/decisions",
        response_model=StoryDecisionResponse,
    )
    def apply_decision(
        project_id: str,
        batch_id: str,
        decision: StoryDecisionRequest,
        settings: StudioSettings = Depends(get_settings),
    ):
        """Apply one accept/reject/reset decision and persist it to the YAML.

        This is the heart of the review board: a click in the UI turns into
        a status update on the underlying YAML, ready for cascade build.
        """
        yaml_path = _resolve_batch_yaml(project_id, batch_id, settings)
        try:
            new_status = decision.status()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        try:
            batch = read_story_batch(yaml_path)
        except CascadeError as exc:
            raise HTTPException(status_code=500, detail=exc.message)

        # Find and update the story
        idx = None
        for i, s in enumerate(batch.stories):
            if s.id == decision.story_id:
                idx = i
                break
        if idx is None:
            raise HTTPException(
                status_code=404,
                detail=f"Story '{decision.story_id}' not in batch '{batch_id}'",
            )

        updated_stories = list(batch.stories)
        updated_stories[idx] = batch.stories[idx].model_copy(
            update={"status": new_status}
        )
        updated_batch = batch.model_copy(update={"stories": updated_stories})
        write_story_batch(updated_batch, yaml_path)

        return StoryDecisionResponse(
            story_id=decision.story_id,
            new_status=new_status.value,
            batch=_batch_summary(yaml_path, updated_batch),
        )

    return r


router = setup_routes()
