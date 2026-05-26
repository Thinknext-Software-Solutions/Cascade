"""Tests for cascade.studio project discovery + projects API + stories API."""

from __future__ import annotations

from pathlib import Path

import pytest

# Skip the whole module if [studio] isn't installed
pytest.importorskip("fastapi")


from fastapi.testclient import TestClient

from cascade.io import write_story_batch
from cascade.schemas import (
    AcceptanceCriterion,
    Story,
    StoryBatch,
    StorySize,
    StoryStatus,
)
from cascade.studio.core.config import StudioSettings, get_settings
from cascade.studio.server import create_app
from cascade.studio.services.projects import (
    discover_projects,
    get_project,
    project_id_for_path,
)


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


def _write_project(root: Path, name: str, language: str | None = "python") -> Path:
    """Create a minimal cascade-enabled project directory."""
    project = root / name
    project.mkdir()
    (project / "cascade.yaml").write_text("version: 1\n")
    if language == "python":
        (project / "pyproject.toml").write_text("[project]\nname='x'\nversion='0.0.1'\n")
    elif language == "go":
        (project / "go.mod").write_text("module x\n")
    return project


def _make_story(idx: int, status: StoryStatus = StoryStatus.EXTRACTED) -> Story:
    return Story(
        id=f"story-{idx:03d}",
        title=f"Story {idx} title goes here",
        description=f"As a user, I want feature {idx} so that benefit {idx}.",
        acceptance_criteria=[
            AcceptanceCriterion(given="g", when="w", then=f"outcome {idx}")
        ],
        size=StorySize.S,
        confidence=80,
        source_meeting_id="m-1",
        status=status,
    )


def _make_batch(n: int = 3, statuses: list[StoryStatus] | None = None) -> StoryBatch:
    stories = []
    for i in range(1, n + 1):
        status = statuses[i - 1] if statuses else StoryStatus.EXTRACTED
        stories.append(_make_story(i, status))
    return StoryBatch(
        meeting_id="m-1",
        extractor_model="claude-test",
        extractor_version="0.0.1",
        stories=stories,
    )


@pytest.fixture
def workspace_with_projects(tmp_path: Path) -> Path:
    """A workspace with two cascade-enabled projects."""
    _write_project(tmp_path, "alpha", language="python")
    _write_project(tmp_path, "bravo", language="go")
    # Non-cascade dir should be ignored
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "README.md").write_text("# not a cascade project")
    return tmp_path


@pytest.fixture
def client_for_workspace(workspace_with_projects: Path):
    """A TestClient whose workspace_root is the populated workspace."""

    def override():
        return StudioSettings(workspace_root=workspace_with_projects)

    app = create_app(settings=StudioSettings(workspace_root=workspace_with_projects))
    app.dependency_overrides[get_settings] = override
    return TestClient(app)


# ----------------------------------------------------------------------------
# Discovery service
# ----------------------------------------------------------------------------


class TestDiscovery:
    def test_project_id_stable(self, tmp_path):
        # Same path -> same ID across calls
        id1 = project_id_for_path(tmp_path)
        id2 = project_id_for_path(tmp_path)
        assert id1 == id2

    def test_finds_top_level_projects(self, workspace_with_projects):
        found = discover_projects(workspace_with_projects)
        names = {p.name for p in found}
        assert names == {"alpha", "bravo"}

    def test_detects_language(self, workspace_with_projects):
        found = {p.name: p for p in discover_projects(workspace_with_projects)}
        assert found["alpha"].language == "python"
        assert found["bravo"].language == "go"

    def test_sorted_by_name(self, workspace_with_projects):
        found = discover_projects(workspace_with_projects)
        names = [p.name for p in found]
        assert names == sorted(names)

    def test_skips_vendored_dirs(self, tmp_path):
        # cascade.yaml inside node_modules should be ignored
        (tmp_path / "node_modules" / "weird-pkg").mkdir(parents=True)
        (tmp_path / "node_modules" / "weird-pkg" / "cascade.yaml").write_text("v: 1")
        _write_project(tmp_path, "real-project")
        found = discover_projects(tmp_path)
        names = {p.name for p in found}
        assert "real-project" in names
        assert "weird-pkg" not in names

    def test_counts_transcripts_and_stories(self, tmp_path):
        project = _write_project(tmp_path, "with-content")
        (project / "transcripts").mkdir()
        (project / "transcripts" / "a.yaml").write_text("x")
        (project / "transcripts" / "b.yaml").write_text("x")
        (project / "stories").mkdir()
        (project / "stories" / "1.yaml").write_text("x")

        found = discover_projects(tmp_path)
        assert found[0].transcript_count == 2
        assert found[0].story_batch_count == 1

    def test_no_projects_returns_empty(self, tmp_path):
        assert discover_projects(tmp_path) == []

    def test_missing_workspace_raises_with_hint(self, tmp_path):
        from cascade.exceptions import CascadeError

        with pytest.raises(CascadeError) as info:
            discover_projects(tmp_path / "does-not-exist")
        assert "does not exist" in info.value.message
        assert info.value.hints  # has actionable suggestions

    def test_get_project_by_id(self, workspace_with_projects):
        all_projects = discover_projects(workspace_with_projects)
        target = all_projects[0]
        found = get_project(target.id, workspace_with_projects)
        assert found is not None
        assert found.name == target.name

    def test_get_project_unknown_id(self, workspace_with_projects):
        assert get_project("nonexistent12", workspace_with_projects) is None


# ----------------------------------------------------------------------------
# /api/projects endpoints
# ----------------------------------------------------------------------------


class TestListProjects:
    def test_lists_all_projects(self, client_for_workspace):
        resp = client_for_workspace.get("/api/projects")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 2
        names = {p["name"] for p in body}
        assert names == {"alpha", "bravo"}
        # Every project has an id, path, language
        for p in body:
            assert p["id"]
            assert p["path"]
            assert p["language"] in {"python", "go", None}

    def test_empty_workspace_returns_empty_list(self, tmp_path):
        app = create_app(settings=StudioSettings(workspace_root=tmp_path))
        client = TestClient(app)
        resp = client.get("/api/projects")
        assert resp.status_code == 200
        assert resp.json() == []


class TestGetProject:
    def test_returns_detail_for_known_id(self, client_for_workspace):
        resp = client_for_workspace.get("/api/projects")
        first_id = resp.json()[0]["id"]
        detail_resp = client_for_workspace.get(f"/api/projects/{first_id}")
        assert detail_resp.status_code == 200
        body = detail_resp.json()
        assert body["project"]["id"] == first_id
        assert "batches" in body

    def test_404_for_unknown_id(self, client_for_workspace):
        resp = client_for_workspace.get("/api/projects/deadbeef0000")
        assert resp.status_code == 404


# ----------------------------------------------------------------------------
# /api/projects/{id}/batches endpoints
# ----------------------------------------------------------------------------


class TestBatchesEndpoints:
    def test_lists_batches_with_counts(self, workspace_with_projects):
        # Add a story batch to alpha
        alpha = workspace_with_projects / "alpha"
        batch = _make_batch(
            n=4,
            statuses=[
                StoryStatus.APPROVED,
                StoryStatus.REJECTED,
                StoryStatus.EXTRACTED,
                StoryStatus.EXTRACTED,
            ],
        )
        write_story_batch(batch, alpha / "stories" / "meeting-001.yaml")

        app = create_app(
            settings=StudioSettings(workspace_root=workspace_with_projects)
        )
        client = TestClient(app)
        project_id = project_id_for_path(alpha)

        resp = client.get(f"/api/projects/{project_id}/batches")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        b = body[0]
        assert b["story_count"] == 4
        assert b["approved_count"] == 1
        assert b["rejected_count"] == 1
        assert b["pending_count"] == 2

    def test_get_batch_returns_stories(self, workspace_with_projects):
        alpha = workspace_with_projects / "alpha"
        batch = _make_batch(n=2)
        write_story_batch(batch, alpha / "stories" / "meeting-002.yaml")

        app = create_app(
            settings=StudioSettings(workspace_root=workspace_with_projects)
        )
        client = TestClient(app)
        project_id = project_id_for_path(alpha)

        resp = client.get(f"/api/projects/{project_id}/batches/meeting-002")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["stories"]) == 2
        assert body["stories"][0]["title"].startswith("Story 1")
        # Story fields surface
        assert body["stories"][0]["status"] == "extracted"
        assert body["stories"][0]["confidence"] == 80
        assert len(body["stories"][0]["acceptance_criteria"]) == 1

    def test_get_batch_404(self, workspace_with_projects):
        alpha = workspace_with_projects / "alpha"
        app = create_app(
            settings=StudioSettings(workspace_root=workspace_with_projects)
        )
        client = TestClient(app)
        project_id = project_id_for_path(alpha)
        resp = client.get(f"/api/projects/{project_id}/batches/missing")
        assert resp.status_code == 404

    def test_apply_decision_persists_to_yaml(self, workspace_with_projects):
        from cascade.io import read_story_batch

        alpha = workspace_with_projects / "alpha"
        batch = _make_batch(n=2)
        batch_path = alpha / "stories" / "meeting-003.yaml"
        write_story_batch(batch, batch_path)

        app = create_app(
            settings=StudioSettings(workspace_root=workspace_with_projects)
        )
        client = TestClient(app)
        project_id = project_id_for_path(alpha)

        resp = client.post(
            f"/api/projects/{project_id}/batches/meeting-003/decisions",
            json={"story_id": "story-001", "decision": "approve"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["new_status"] == "approved"
        assert body["batch"]["approved_count"] == 1

        # File on disk reflects the decision
        reloaded = read_story_batch(batch_path)
        assert reloaded.stories[0].status == StoryStatus.APPROVED
        assert reloaded.stories[1].status == StoryStatus.EXTRACTED

    def test_apply_decision_unknown_story(self, workspace_with_projects):
        alpha = workspace_with_projects / "alpha"
        batch = _make_batch(n=1)
        write_story_batch(batch, alpha / "stories" / "meeting-004.yaml")

        app = create_app(
            settings=StudioSettings(workspace_root=workspace_with_projects)
        )
        client = TestClient(app)
        project_id = project_id_for_path(alpha)

        resp = client.post(
            f"/api/projects/{project_id}/batches/meeting-004/decisions",
            json={"story_id": "not-there", "decision": "approve"},
        )
        assert resp.status_code == 404

    def test_apply_decision_invalid_value(self, workspace_with_projects):
        alpha = workspace_with_projects / "alpha"
        batch = _make_batch(n=1)
        write_story_batch(batch, alpha / "stories" / "meeting-005.yaml")

        app = create_app(
            settings=StudioSettings(workspace_root=workspace_with_projects)
        )
        client = TestClient(app)
        project_id = project_id_for_path(alpha)

        resp = client.post(
            f"/api/projects/{project_id}/batches/meeting-005/decisions",
            json={"story_id": "story-001", "decision": "obliterate"},
        )
        assert resp.status_code == 400

    def test_reset_decision_returns_to_extracted(self, workspace_with_projects):
        from cascade.io import read_story_batch

        alpha = workspace_with_projects / "alpha"
        batch = _make_batch(n=1, statuses=[StoryStatus.APPROVED])
        batch_path = alpha / "stories" / "meeting-006.yaml"
        write_story_batch(batch, batch_path)

        app = create_app(
            settings=StudioSettings(workspace_root=workspace_with_projects)
        )
        client = TestClient(app)
        project_id = project_id_for_path(alpha)

        resp = client.post(
            f"/api/projects/{project_id}/batches/meeting-006/decisions",
            json={"story_id": "story-001", "decision": "reset"},
        )
        assert resp.status_code == 200
        reloaded = read_story_batch(batch_path)
        assert reloaded.stories[0].status == StoryStatus.EXTRACTED
