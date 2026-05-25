"""GitLab VCS provider implementation (python-gitlab)."""

from __future__ import annotations

from typing import Optional

from .exceptions import CascadeRepoError
from .plan_schemas import PullRequestRef
from .vcs import RepoIdentity


class GitLabVCSProvider:
    """VCSProvider implementation backed by python-gitlab.

    Supports both gitlab.com and self-hosted GitLab via base_url.
    GitLab calls them 'Merge Requests' but we still return them as
    PullRequestRef for a uniform downstream interface.
    """

    def __init__(self, *, token: str, base_url: Optional[str] = None):
        try:
            import gitlab  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise CascadeRepoError(
                "python-gitlab not installed. Run: pip install python-gitlab"
            ) from exc
        self._gl = gitlab.Gitlab(
            url=base_url or "https://gitlab.com",
            private_token=token,
        )

    @property
    def provider_name(self) -> str:
        return "gitlab"

    def open_pull_request(
        self,
        *,
        identity: RepoIdentity,
        head: str,
        base: str,
        title: str,
        body: str,
    ) -> PullRequestRef:
        project_path = f"{identity.owner}/{identity.repo}"
        try:
            project = self._gl.projects.get(project_path)
            mr = project.mergerequests.create(
                {
                    "source_branch": head,
                    "target_branch": base,
                    "title": title,
                    "description": body,
                }
            )
        except Exception as exc:
            raise CascadeRepoError(
                f"Failed to open GitLab MR for {project_path}: {exc}"
            ) from exc
        return PullRequestRef(
            number=mr.iid,
            url=mr.web_url,
            branch=head,
            title=title,
        )
