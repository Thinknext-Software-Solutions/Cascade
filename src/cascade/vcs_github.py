"""GitHub VCS provider implementation (PyGithub)."""

from __future__ import annotations

from typing import Optional

from .exceptions import CascadeRepoError
from .plan_schemas import PullRequestRef
from .vcs import RepoIdentity


class GitHubVCSProvider:
    """VCSProvider implementation backed by PyGithub."""

    def __init__(self, *, token: str, base_url: Optional[str] = None):
        try:
            from github import Github
        except ImportError as exc:  # pragma: no cover
            raise CascadeRepoError(
                "PyGithub not installed. Run: pip install PyGithub"
            ) from exc
        kwargs: dict = {}
        if base_url:
            kwargs["base_url"] = base_url
        self._gh = Github(token, **kwargs)

    @property
    def provider_name(self) -> str:
        return "github"

    def open_pull_request(
        self,
        *,
        identity: RepoIdentity,
        head: str,
        base: str,
        title: str,
        body: str,
    ) -> PullRequestRef:
        try:
            repo = self._gh.get_repo(f"{identity.owner}/{identity.repo}")
            pr = repo.create_pull(title=title, body=body, head=head, base=base)
        except Exception as exc:
            raise CascadeRepoError(
                f"Failed to open GitHub PR for {identity.owner}/{identity.repo}: {exc}"
            ) from exc
        return PullRequestRef(
            number=pr.number,
            url=pr.html_url,
            branch=head,
            title=title,
        )
