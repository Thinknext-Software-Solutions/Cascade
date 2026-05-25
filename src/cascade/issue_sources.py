"""Issue tracker abstractions: GitHub Issues, Jira, Azure Boards, Linear, GitLab Issues.

Each issue source can fetch a single issue by its tracker-native identifier
and convert it into a `Story` ready to flow through the pipeline. This is
the "ongoing development" entry point -- when work originates from a ticket,
not a meeting.

The identifier format is `<provider>:<id>`, e.g.:
    github:owner/repo#42
    jira:PROJ-123
    azure_devops:org/project/123
    linear:ENG-456
    gitlab:owner/repo#42
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Protocol

from .exceptions import CascadeError
from .schemas import AcceptanceCriterion, Story, StorySize, StoryStatus


logger = logging.getLogger(__name__)


SUPPORTED_ISSUE_SOURCES: tuple[str, ...] = (
    "github",
    "jira",
    "azure_devops",
    "linear",
    "gitlab",
)


@dataclass(frozen=True)
class IssueRef:
    """Parsed identifier of an issue across any tracker."""

    provider: str
    identifier: str  # provider-specific identifier (e.g. "PROJ-123")
    raw: str  # the full input string


def parse_issue_ref(identifier: str) -> IssueRef:
    """Parse a `<provider>:<id>` identifier into an IssueRef.

    Examples:
        github:owner/repo#42
        jira:PROJ-123
        azure_devops:org/project/123
        linear:ENG-456
    """
    if ":" not in identifier:
        raise CascadeError(
            f"Issue identifier must be '<provider>:<id>', got: {identifier!r}. "
            f"Supported providers: {', '.join(SUPPORTED_ISSUE_SOURCES)}."
        )
    provider, rest = identifier.split(":", 1)
    provider = provider.lower().strip()
    rest = rest.strip()
    if provider not in SUPPORTED_ISSUE_SOURCES:
        raise CascadeError(
            f"Unknown issue source '{provider}'. Supported: {', '.join(SUPPORTED_ISSUE_SOURCES)}."
        )
    if not rest:
        raise CascadeError(f"Empty issue identifier after provider: {identifier!r}")
    return IssueRef(provider=provider, identifier=rest, raw=identifier)


class IssueSource(Protocol):
    """Interface a tracker integration must satisfy."""

    @property
    def provider_name(self) -> str: ...

    def fetch_story(self, identifier: str) -> Story:
        """Fetch an issue from this tracker and convert it to a Story."""


def _safe_story_id(provider: str, identifier: str) -> str:
    """Stable Story ID derived from an issue's provider+identifier."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", identifier).strip("-")
    return f"story-{provider}-{slug}"[:80]


def _default_acceptance(description: str) -> list[AcceptanceCriterion]:
    """When the tracker issue doesn't have structured AC, synthesize a
    single placeholder so Story validation passes. Reviewers should refine
    these before approving the story for build."""
    return [
        AcceptanceCriterion(
            given="the system in its current state",
            when="the change described in the issue is implemented",
            then="the issue's acceptance criteria as documented are satisfied",
        )
    ]


# ----------------------------------------------------------------------------
# GitHub Issues
# ----------------------------------------------------------------------------


class GitHubIssueSource:
    """Fetches issues from GitHub via PyGithub.

    Identifier format: `owner/repo#number` (e.g. `myorg/myrepo#42`).
    """

    def __init__(self, *, token: str, base_url: Optional[str] = None):
        try:
            from github import Github
        except ImportError as exc:  # pragma: no cover
            raise CascadeError("PyGithub not installed. Run: pip install PyGithub") from exc
        kwargs: dict = {}
        if base_url:
            kwargs["base_url"] = base_url
        self._gh = Github(token, **kwargs)

    @property
    def provider_name(self) -> str:
        return "github"

    def fetch_story(self, identifier: str) -> Story:
        m = re.match(r"^([^/]+)/([^#]+)#(\d+)$", identifier)
        if not m:
            raise CascadeError(
                f"GitHub issue identifier must be 'owner/repo#N', got: {identifier!r}"
            )
        owner, repo, number = m.group(1), m.group(2), int(m.group(3))
        try:
            repo_obj = self._gh.get_repo(f"{owner}/{repo}")
            issue = repo_obj.get_issue(number=number)
        except Exception as exc:
            raise CascadeError(
                f"Failed to fetch GitHub issue {identifier}: {exc}"
            ) from exc
        return Story(
            id=_safe_story_id("github", identifier),
            title=(issue.title or "Untitled issue")[:140],
            description=issue.body or "(no description provided in the issue)",
            acceptance_criteria=_default_acceptance(issue.body or ""),
            size=StorySize.M,
            confidence=70,
            source_meeting_id=f"github-issue:{identifier}",
            notes=f"Imported from GitHub issue {issue.html_url}",
            status=StoryStatus.APPROVED,  # tracker tickets are pre-approved by definition
        )


# ----------------------------------------------------------------------------
# Jira
# ----------------------------------------------------------------------------


class JiraIssueSource:
    """Fetches issues from Jira via the atlassian-python-api library.

    Identifier format: `PROJECTKEY-NNN` (e.g. `PROJ-123`).
    """

    def __init__(self, *, token: str, base_url: str, user: str):
        if not base_url:
            raise CascadeError("Jira requires base_url (e.g. https://acme.atlassian.net)")
        if not user:
            raise CascadeError("Jira requires user (email for cloud, username for server)")
        try:
            from atlassian import Jira  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise CascadeError(
                "atlassian-python-api not installed. "
                "Run: pip install atlassian-python-api"
            ) from exc
        self._jira = Jira(url=base_url, username=user, password=token, cloud=True)

    @property
    def provider_name(self) -> str:
        return "jira"

    def fetch_story(self, identifier: str) -> Story:
        try:
            issue = self._jira.issue(identifier)
        except Exception as exc:
            raise CascadeError(
                f"Failed to fetch Jira issue {identifier}: {exc}"
            ) from exc

        fields = issue.get("fields", {}) if isinstance(issue, dict) else {}
        summary = fields.get("summary", "(no summary)")
        description = fields.get("description") or "(no description)"
        # Jira descriptions may be Atlassian Document Format (a dict); flatten.
        if isinstance(description, dict):
            description = _flatten_adf(description) or "(no description)"

        return Story(
            id=_safe_story_id("jira", identifier),
            title=summary[:140],
            description=str(description),
            acceptance_criteria=_default_acceptance(str(description)),
            size=StorySize.M,
            confidence=70,
            source_meeting_id=f"jira:{identifier}",
            notes=f"Imported from Jira {identifier}",
            status=StoryStatus.APPROVED,
        )


def _flatten_adf(node) -> str:
    """Best-effort flatten of Atlassian Document Format to plain text."""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "\n".join(_flatten_adf(n) for n in node)
    if isinstance(node, dict):
        text = node.get("text", "")
        children = node.get("content", [])
        return text + (("\n" + _flatten_adf(children)) if children else "")
    return ""


# ----------------------------------------------------------------------------
# Linear
# ----------------------------------------------------------------------------


class LinearIssueSource:
    """Fetches issues from Linear via GraphQL.

    Identifier format: `TEAM-NNN` (e.g. `ENG-456`).
    """

    GRAPHQL_URL = "https://api.linear.app/graphql"

    def __init__(self, *, token: str, base_url: Optional[str] = None):
        try:
            import requests  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise CascadeError("requests not installed. Run: pip install requests") from exc
        self._requests = requests
        self._token = token
        self._url = (base_url or self.GRAPHQL_URL).rstrip("/")

    @property
    def provider_name(self) -> str:
        return "linear"

    def fetch_story(self, identifier: str) -> Story:
        # Linear's issue() query takes the team-prefixed identifier directly
        query = """
        query($id: String!) {
          issue(id: $id) {
            identifier
            title
            description
            url
          }
        }
        """
        try:
            resp = self._requests.post(
                self._url,
                json={"query": query, "variables": {"id": identifier}},
                headers={"Authorization": self._token, "Content-Type": "application/json"},
                timeout=60,
            )
        except Exception as exc:
            raise CascadeError(f"Linear API call failed: {exc}") from exc
        if not resp.ok:
            raise CascadeError(
                f"Failed to fetch Linear issue {identifier}: HTTP {resp.status_code} {resp.text[:300]}"
            )
        data = resp.json().get("data", {}).get("issue")
        if not data:
            raise CascadeError(f"Linear issue {identifier!r} not found")
        return Story(
            id=_safe_story_id("linear", identifier),
            title=(data.get("title") or "Untitled")[:140],
            description=data.get("description") or "(no description)",
            acceptance_criteria=_default_acceptance(data.get("description") or ""),
            size=StorySize.M,
            confidence=70,
            source_meeting_id=f"linear:{identifier}",
            notes=f"Imported from Linear {data.get('url', identifier)}",
            status=StoryStatus.APPROVED,
        )


# ----------------------------------------------------------------------------
# Azure DevOps Boards
# ----------------------------------------------------------------------------


class AzureBoardsIssueSource:
    """Fetches work items from Azure DevOps Boards via REST.

    Identifier format: `org/project/NNN` (e.g. `myorg/myproj/123`).
    """

    API_VERSION = "7.1-preview.3"
    DEFAULT_BASE_URL = "https://dev.azure.com"

    def __init__(self, *, token: str, base_url: Optional[str] = None):
        try:
            import requests  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise CascadeError("requests not installed. Run: pip install requests") from exc
        self._requests = requests
        self._token = token
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")

    @property
    def provider_name(self) -> str:
        return "azure_devops"

    def fetch_story(self, identifier: str) -> Story:
        m = re.match(r"^([^/]+)/([^/]+)/(\d+)$", identifier)
        if not m:
            raise CascadeError(
                f"Azure Boards identifier must be 'org/project/N', got: {identifier!r}"
            )
        org, project, work_id = m.group(1), m.group(2), m.group(3)
        url = (
            f"{self._base_url}/{org}/{project}/_apis/wit/workitems/{work_id}"
            f"?api-version={self.API_VERSION}"
        )
        import base64

        headers = {
            "Authorization": "Basic "
            + base64.b64encode(f":{self._token}".encode("utf-8")).decode("ascii"),
        }
        try:
            resp = self._requests.get(url, headers=headers, timeout=60)
        except Exception as exc:
            raise CascadeError(f"Azure Boards API call failed: {exc}") from exc
        if not resp.ok:
            raise CascadeError(
                f"Failed to fetch Azure work item {identifier}: HTTP {resp.status_code}"
            )
        data = resp.json()
        fields = data.get("fields", {})
        title = fields.get("System.Title", "Untitled work item")
        description = fields.get("System.Description", "") or fields.get(
            "Microsoft.VSTS.Common.AcceptanceCriteria", ""
        )
        return Story(
            id=_safe_story_id("azuredev", identifier),
            title=title[:140],
            description=description or "(no description)",
            acceptance_criteria=_default_acceptance(description or ""),
            size=StorySize.M,
            confidence=70,
            source_meeting_id=f"azure_devops:{identifier}",
            notes=f"Imported from Azure Boards work item {identifier}",
            status=StoryStatus.APPROVED,
        )


# ----------------------------------------------------------------------------
# GitLab Issues
# ----------------------------------------------------------------------------


class GitLabIssueSource:
    """Fetches issues from GitLab via python-gitlab.

    Identifier format: `group/project#NNN` (e.g. `myteam/myrepo#42`).
    """

    def __init__(self, *, token: str, base_url: Optional[str] = None):
        try:
            import gitlab  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise CascadeError(
                "python-gitlab not installed. Run: pip install python-gitlab"
            ) from exc
        self._gl = gitlab.Gitlab(
            url=base_url or "https://gitlab.com",
            private_token=token,
        )

    @property
    def provider_name(self) -> str:
        return "gitlab"

    def fetch_story(self, identifier: str) -> Story:
        m = re.match(r"^(.+)#(\d+)$", identifier)
        if not m:
            raise CascadeError(
                f"GitLab issue identifier must be 'group/project#N', got: {identifier!r}"
            )
        project_path, iid = m.group(1), int(m.group(2))
        try:
            project = self._gl.projects.get(project_path)
            issue = project.issues.get(iid)
        except Exception as exc:
            raise CascadeError(
                f"Failed to fetch GitLab issue {identifier}: {exc}"
            ) from exc
        return Story(
            id=_safe_story_id("gitlab", identifier),
            title=(issue.title or "Untitled")[:140],
            description=issue.description or "(no description)",
            acceptance_criteria=_default_acceptance(issue.description or ""),
            size=StorySize.M,
            confidence=70,
            source_meeting_id=f"gitlab:{identifier}",
            notes=f"Imported from GitLab {issue.web_url}",
            status=StoryStatus.APPROVED,
        )


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------


def build_issue_source(
    *,
    provider: str,
    token: str,
    base_url: Optional[str] = None,
    user: Optional[str] = None,
) -> IssueSource:
    """Construct an IssueSource implementation by provider name."""
    p = provider.lower()
    if p == "github":
        return GitHubIssueSource(token=token, base_url=base_url)
    if p == "jira":
        return JiraIssueSource(token=token, base_url=base_url or "", user=user or "")
    if p == "linear":
        return LinearIssueSource(token=token, base_url=base_url)
    if p == "azure_devops":
        return AzureBoardsIssueSource(token=token, base_url=base_url)
    if p == "gitlab":
        return GitLabIssueSource(token=token, base_url=base_url)
    raise CascadeError(
        f"Unknown issue source '{provider}'. Supported: {', '.join(SUPPORTED_ISSUE_SOURCES)}."
    )


def story_from_prompt(prompt: str) -> Story:
    """Build a Story directly from a free-text prompt (the `cascade prompt` path).

    Used when there's no meeting and no ticket -- just a developer typing
    "build me X". The Story flows through plan/code/test/PR like any other.
    """
    if not prompt or not prompt.strip():
        raise CascadeError("Prompt is empty")
    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    title = prompt.strip().split("\n", 1)[0][:140]
    return Story(
        id=f"story-prompt-{now}",
        title=title,
        description=prompt.strip(),
        acceptance_criteria=_default_acceptance(prompt),
        size=StorySize.M,
        confidence=80,
        source_meeting_id=f"adhoc-prompt:{now}",
        notes="Ad-hoc story generated from a direct prompt (no meeting/ticket source).",
        status=StoryStatus.APPROVED,
    )
