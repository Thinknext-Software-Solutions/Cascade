"""Bitbucket Cloud VCS provider implementation (REST API).

We use direct REST calls via the `requests` library rather than a heavier
SDK. Bitbucket's API is small enough that this is simpler.
"""

from __future__ import annotations

import logging
from typing import Optional

from .exceptions import CascadeRepoError
from .plan_schemas import PullRequestRef
from .vcs import RepoIdentity


logger = logging.getLogger(__name__)


class BitbucketVCSProvider:
    """VCSProvider implementation for Bitbucket Cloud (bitbucket.org).

    Uses Bitbucket REST API v2.0 with HTTP basic auth using a username +
    App Password. Server/Datacenter (self-hosted) is NOT supported in v0.1
    -- the API surface differs significantly.
    """

    DEFAULT_BASE_URL = "https://api.bitbucket.org/2.0"

    def __init__(self, *, token: str, base_url: Optional[str] = None):
        try:
            import requests  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise CascadeRepoError(
                "requests not installed. Run: pip install requests"
            ) from exc
        # token here is the Bitbucket App Password OR an API token. We pass
        # it as a Bearer token; users with username:password should encode
        # as "user:pass" in the token field.
        self._requests = requests
        self._token = token
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")

    @property
    def provider_name(self) -> str:
        return "bitbucket"

    def open_pull_request(
        self,
        *,
        identity: RepoIdentity,
        head: str,
        base: str,
        title: str,
        body: str,
    ) -> PullRequestRef:
        url = f"{self._base_url}/repositories/{identity.owner}/{identity.repo}/pullrequests"
        payload = {
            "title": title,
            "description": body,
            "source": {"branch": {"name": head}},
            "destination": {"branch": {"name": base}},
        }
        headers = self._auth_headers()
        try:
            resp = self._requests.post(url, json=payload, headers=headers, timeout=60)
        except Exception as exc:
            raise CascadeRepoError(
                f"Bitbucket API call failed: {exc}"
            ) from exc

        if not resp.ok:
            raise CascadeRepoError(
                f"Failed to open Bitbucket PR for {identity.owner}/{identity.repo}: "
                f"HTTP {resp.status_code} {resp.text[:300]}"
            )

        data = resp.json()
        return PullRequestRef(
            number=data["id"],
            url=data["links"]["html"]["href"],
            branch=head,
            title=title,
        )

    def _auth_headers(self) -> dict:
        # Bitbucket supports both Bearer tokens (newer API tokens) and
        # HTTP Basic Auth with App Passwords. We default to Bearer; if
        # the token contains a colon we treat it as "user:password" basic auth.
        if ":" in self._token:
            import base64

            encoded = base64.b64encode(self._token.encode("utf-8")).decode("ascii")
            return {"Authorization": f"Basic {encoded}"}
        return {"Authorization": f"Bearer {self._token}"}
