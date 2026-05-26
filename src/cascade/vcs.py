"""VCS provider abstraction: GitHub, GitLab, Bitbucket, Azure DevOps.

Each VCS provider implements `VCSProvider`. The protocol covers what the
build pipeline needs: parsing a repo identifier, opening a PR/MR, and
posting a comment back to the PR.

Local git operations (branch, commit, push) stay in `repo.py` -- they're
the same regardless of which hosting service the remote points to.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, Protocol

from .exceptions import CascadeRepoError
from .plan_schemas import PullRequestRef


logger = logging.getLogger(__name__)


SUPPORTED_VCS_PROVIDERS: tuple[str, ...] = (
    "github",
    "gitlab",
    "bitbucket",
    "azure_devops",
)


@dataclass(frozen=True)
class RepoIdentity:
    """Parsed identity of a remote repo."""

    provider: str  # "github" | "gitlab" | "bitbucket" | "azure_devops"
    owner: str  # org/user (or org/project for Azure DevOps)
    repo: str
    base_url: Optional[str] = None  # for self-hosted instances


class VCSProvider(Protocol):
    """Minimal interface the pipeline needs from a VCS host."""

    @property
    def provider_name(self) -> str: ...

    def open_pull_request(
        self,
        *,
        identity: RepoIdentity,
        head: str,
        base: str,
        title: str,
        body: str,
    ) -> PullRequestRef: ...


# ----------------------------------------------------------------------------
# Origin URL parsing -- detects which provider this repo belongs to
# ----------------------------------------------------------------------------


def parse_remote_url(url: str) -> RepoIdentity:
    """Parse a git remote URL and return the RepoIdentity.

    Supports the four major VCS providers in their HTTPS and SSH forms.

    Raises:
        CascadeRepoError: If the URL doesn't match any known provider.
    """
    # GitHub
    m = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if m:
        return RepoIdentity(provider="github", owner=m.group(1), repo=m.group(2))
    m = re.match(r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if m:
        return RepoIdentity(provider="github", owner=m.group(1), repo=m.group(2))

    # GitLab (gitlab.com or self-hosted)
    m = re.match(r"https://([^/]*gitlab[^/]*)/(.+?)/([^/]+?)(?:\.git)?/?$", url)
    if m:
        host, owner_path, repo = m.group(1), m.group(2), m.group(3)
        base_url = f"https://{host}" if host != "gitlab.com" else None
        return RepoIdentity(
            provider="gitlab", owner=owner_path, repo=repo, base_url=base_url
        )
    m = re.match(r"git@([^:]*gitlab[^:]*):(.+?)/([^/]+?)(?:\.git)?/?$", url)
    if m:
        host, owner_path, repo = m.group(1), m.group(2), m.group(3)
        base_url = f"https://{host}" if host != "gitlab.com" else None
        return RepoIdentity(
            provider="gitlab", owner=owner_path, repo=repo, base_url=base_url
        )

    # Bitbucket
    m = re.match(r"https://bitbucket\.org/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if m:
        return RepoIdentity(provider="bitbucket", owner=m.group(1), repo=m.group(2))
    m = re.match(r"git@bitbucket\.org:([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if m:
        return RepoIdentity(provider="bitbucket", owner=m.group(1), repo=m.group(2))

    # Azure DevOps  -- forms:
    #   https://dev.azure.com/{org}/{project}/_git/{repo}
    #   https://{org}@dev.azure.com/{org}/{project}/_git/{repo}
    #   git@ssh.dev.azure.com:v3/{org}/{project}/{repo}
    m = re.match(
        r"https://(?:[^@]+@)?dev\.azure\.com/([^/]+)/([^/]+)/_git/([^/]+?)/?$", url
    )
    if m:
        owner = f"{m.group(1)}/{m.group(2)}"  # org/project
        return RepoIdentity(provider="azure_devops", owner=owner, repo=m.group(3))
    m = re.match(r"git@ssh\.dev\.azure\.com:v3/([^/]+)/([^/]+)/([^/]+?)/?$", url)
    if m:
        owner = f"{m.group(1)}/{m.group(2)}"
        return RepoIdentity(provider="azure_devops", owner=owner, repo=m.group(3))

    raise CascadeRepoError(
        f"Could not identify VCS provider from origin URL: {url!r}",
        hint=[
            "Cascade supports: GitHub, GitLab (cloud + self-hosted), Bitbucket Cloud, Azure DevOps",
            "Your origin URL doesn't match any of those patterns",
            "If you're on a self-hosted Git server we don't recognize, open an issue requesting support",
            "As a workaround: run cascade build with --no-pr to skip the PR step entirely",
        ],
        learn_more="https://github.com/Thinknext-Software-Solutions/Cascade/issues",
    )


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------


def build_vcs_provider(
    *,
    provider: str,
    token: str,
    base_url: Optional[str] = None,
) -> VCSProvider:
    """Construct a VCSProvider implementation for the named host.

    Raises:
        CascadeRepoError: Unknown provider or initialization failure.
    """
    p = provider.lower()
    if p == "github":
        from .vcs_github import GitHubVCSProvider

        return GitHubVCSProvider(token=token, base_url=base_url)
    if p == "gitlab":
        from .vcs_gitlab import GitLabVCSProvider

        return GitLabVCSProvider(token=token, base_url=base_url)
    if p == "bitbucket":
        from .vcs_bitbucket import BitbucketVCSProvider

        return BitbucketVCSProvider(token=token, base_url=base_url)
    if p == "azure_devops":
        from .vcs_azure import AzureDevOpsVCSProvider

        return AzureDevOpsVCSProvider(token=token, base_url=base_url)
    raise CascadeRepoError(
        f"Unknown VCS provider '{provider}'. Supported: {', '.join(SUPPORTED_VCS_PROVIDERS)}."
    )
