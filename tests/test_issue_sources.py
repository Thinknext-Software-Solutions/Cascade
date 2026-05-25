"""Tests for cascade.issue_sources (parsing + factory + prompt helper).

We don't make real API calls; the per-provider classes are exercised via
unit-level checks on identifier parsing. Real-API integration tests live
in tests/integration/ and require env vars.
"""

from __future__ import annotations

import pytest

from cascade.exceptions import CascadeError
from cascade.issue_sources import (
    SUPPORTED_ISSUE_SOURCES,
    IssueRef,
    build_issue_source,
    parse_issue_ref,
    story_from_prompt,
)
from cascade.schemas import StoryStatus


# --------- parse_issue_ref ----------


class TestParseIssueRef:
    def test_jira(self):
        r = parse_issue_ref("jira:PROJ-123")
        assert r == IssueRef(provider="jira", identifier="PROJ-123", raw="jira:PROJ-123")

    def test_github(self):
        r = parse_issue_ref("github:owner/repo#42")
        assert r.provider == "github"
        assert r.identifier == "owner/repo#42"

    def test_azure(self):
        r = parse_issue_ref("azure_devops:org/proj/123")
        assert r.provider == "azure_devops"

    def test_case_insensitive_provider(self):
        r = parse_issue_ref("Jira:PROJ-1")
        assert r.provider == "jira"

    def test_strips_whitespace(self):
        r = parse_issue_ref("jira: PROJ-1 ")
        assert r.identifier == "PROJ-1"

    def test_missing_colon_raises(self):
        with pytest.raises(CascadeError, match="<provider>:<id>"):
            parse_issue_ref("PROJ-123")

    def test_unknown_provider_raises(self):
        with pytest.raises(CascadeError, match="Unknown issue source"):
            parse_issue_ref("trello:CARD-1")

    def test_empty_identifier_raises(self):
        with pytest.raises(CascadeError, match="Empty issue identifier"):
            parse_issue_ref("jira:")

    def test_all_supported_providers(self):
        for p in SUPPORTED_ISSUE_SOURCES:
            r = parse_issue_ref(f"{p}:abc-1")
            assert r.provider == p


# --------- build_issue_source ----------


class TestBuildIssueSource:
    def test_unknown_provider(self):
        with pytest.raises(CascadeError, match="Unknown issue source"):
            build_issue_source(provider="trello", token="t")

    # Real provider construction requires the SDKs; skip those here.
    # They're exercised in test_user_config / per-provider integration suites.


# --------- story_from_prompt ----------


class TestStoryFromPrompt:
    def test_basic_prompt_becomes_story(self):
        story = story_from_prompt("Add cursor-based pagination to /api/users")
        assert story.title == "Add cursor-based pagination to /api/users"
        assert story.status == StoryStatus.APPROVED
        assert len(story.acceptance_criteria) >= 1
        assert story.source_meeting_id.startswith("adhoc-prompt:")

    def test_multiline_prompt_uses_first_line_as_title(self):
        story = story_from_prompt(
            "Add user logout endpoint\n\nMust support POST /api/logout returning 204."
        )
        assert story.title == "Add user logout endpoint"
        assert "POST /api/logout" in story.description

    def test_empty_prompt_raises(self):
        with pytest.raises(CascadeError, match="empty"):
            story_from_prompt("")

    def test_whitespace_prompt_raises(self):
        with pytest.raises(CascadeError, match="empty"):
            story_from_prompt("   \n  \t ")

    def test_long_title_truncated(self):
        prompt = "x" * 500
        story = story_from_prompt(prompt)
        assert len(story.title) == 140

    def test_unique_ids_across_prompts(self):
        import time

        s1 = story_from_prompt("first prompt")
        time.sleep(1.01)  # ensure timestamp ticks
        s2 = story_from_prompt("second prompt")
        assert s1.id != s2.id
