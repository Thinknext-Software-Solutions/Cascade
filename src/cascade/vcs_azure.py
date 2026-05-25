"""Azure DevOps Repos VCS provider implementation (REST API).

Uses direct REST calls. The azure-devops Python SDK is large and slow to
import; for the small surface we need (creating PRs) the REST API is
cleaner.

Authentication: Personal Access Token (PAT) sent as HTTP Basic auth with
an empty username.
"""

from __future__ import annotations

import base64
from typing import Optional

from .exceptions import CascadeRepoError
from .plan_schemas import PullRequestRef
from .vcs import RepoIdentity


class AzureDevOpsVCSProvider:
    """VCSProvider implementation for Azure DevOps Repos.

    RepoIdentity.owner is formatted as 'org/project' (the parser sets this
    automatically). RepoIdentity.repo is the repository name.

    Uses Azure DevOps REST API v7.1.
    """

    API_VERSION = "7.1-preview.1"
    DEFAULT_BASE_URL = "https://dev.azure.com"

    def __init__(self, *, token: str, base_url: Optional[str] = None):
        try:
            import requests  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise CascadeRepoError(
                "requests not installed. Run: pip install requests"
            ) from exc
        self._requests = requests
        self._token = token
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")

    @property
    def provider_name(self) -> str:
        return "azure_devops"

    def open_pull_request(
        self,
        *,
        identity: RepoIdentity,
        head: str,
        base: str,
        title: str,
        body: str,
    ) -> PullRequestRef:
        # owner is "org/project" per the parser
        if "/" not in identity.owner:
            raise CascadeRepoError(
                f"Azure DevOps identity.owner must be 'org/project', got: "
                f"{identity.owner!r}"
            )
        org, project = identity.owner.split("/", 1)
        url = (
            f"{self._base_url}/{org}/{project}/_apis/git/repositories/"
            f"{identity.repo}/pullrequests?api-version={self.API_VERSION}"
        )
        payload = {
            "sourceRefName": f"refs/heads/{head}",
            "targetRefName": f"refs/heads/{base}",
            "title": title,
            "description": body,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Basic " + base64.b64encode(
                f":{self._token}".encode("utf-8")
            ).decode("ascii"),
        }
        try:
            resp = self._requests.post(url, json=payload, headers=headers, timeout=60)
        except Exception as exc:
            raise CascadeRepoError(
                f"Azure DevOps API call failed: {exc}"
            ) from exc

        if not resp.ok:
            raise CascadeRepoError(
                f"Failed to open Azure DevOps PR for {identity.owner}/{identity.repo}: "
                f"HTTP {resp.status_code} {resp.text[:300]}"
            )

        data = resp.json()
        pr_number = data["pullRequestId"]
        # Construct the human-facing URL; Azure doesn't return it in this response shape consistently
        web_url = (
            f"{self._base_url}/{org}/{project}/_git/{identity.repo}/pullrequest/{pr_number}"
        )
        return PullRequestRef(
            number=pr_number,
            url=web_url,
            branch=head,
            title=title,
        )
