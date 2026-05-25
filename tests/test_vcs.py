"""Tests for cascade.vcs (URL parsing, provider factory, GitHub impl)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cascade.exceptions import CascadeRepoError
from cascade.plan_schemas import PullRequestRef
from cascade.vcs import RepoIdentity, build_vcs_provider, parse_remote_url


# --------- parse_remote_url ----------


class TestParseRemoteUrl:
    def test_github_https(self):
        r = parse_remote_url("https://github.com/foo/bar.git")
        assert r.provider == "github"
        assert r.owner == "foo"
        assert r.repo == "bar"
        assert r.base_url is None

    def test_github_ssh(self):
        r = parse_remote_url("git@github.com:foo/bar.git")
        assert (r.provider, r.owner, r.repo) == ("github", "foo", "bar")

    def test_github_no_dot_git(self):
        r = parse_remote_url("https://github.com/foo/bar")
        assert r.repo == "bar"

    def test_gitlab_cloud(self):
        r = parse_remote_url("https://gitlab.com/group/subgroup/repo.git")
        assert r.provider == "gitlab"
        assert r.owner == "group/subgroup"
        assert r.repo == "repo"
        assert r.base_url is None  # gitlab.com is default

    def test_gitlab_self_hosted(self):
        r = parse_remote_url("https://gitlab.acme.io/team/proj.git")
        assert r.provider == "gitlab"
        assert r.owner == "team"
        assert r.repo == "proj"
        assert r.base_url == "https://gitlab.acme.io"

    def test_bitbucket_cloud(self):
        r = parse_remote_url("https://bitbucket.org/team/repo.git")
        assert r.provider == "bitbucket"
        assert (r.owner, r.repo) == ("team", "repo")

    def test_bitbucket_ssh(self):
        r = parse_remote_url("git@bitbucket.org:team/repo.git")
        assert r.provider == "bitbucket"

    def test_azure_devops_https(self):
        r = parse_remote_url(
            "https://dev.azure.com/myorg/myproj/_git/myrepo"
        )
        assert r.provider == "azure_devops"
        assert r.owner == "myorg/myproj"
        assert r.repo == "myrepo"

    def test_azure_devops_https_with_user(self):
        r = parse_remote_url(
            "https://myorg@dev.azure.com/myorg/myproj/_git/myrepo"
        )
        assert r.owner == "myorg/myproj"
        assert r.repo == "myrepo"

    def test_azure_devops_ssh(self):
        r = parse_remote_url(
            "git@ssh.dev.azure.com:v3/myorg/myproj/myrepo"
        )
        assert r.provider == "azure_devops"
        assert r.owner == "myorg/myproj"
        assert r.repo == "myrepo"

    def test_unknown_provider_raises(self):
        with pytest.raises(CascadeRepoError, match="Could not identify"):
            parse_remote_url("https://sourcehut.org/foo/bar")


# --------- build_vcs_provider ----------


class TestBuildVCSProvider:
    def test_github_factory(self):
        with patch("cascade.vcs_github.GitHubVCSProvider") as mock:
            mock.return_value = MagicMock(provider_name="github")
            client = build_vcs_provider(provider="github", token="t")
            mock.assert_called_once_with(token="t", base_url=None)

    def test_gitlab_factory(self):
        with patch("cascade.vcs_gitlab.GitLabVCSProvider") as mock:
            mock.return_value = MagicMock(provider_name="gitlab")
            build_vcs_provider(provider="gitlab", token="t", base_url="https://x.io")
            mock.assert_called_once_with(token="t", base_url="https://x.io")

    def test_unknown_provider_raises(self):
        with pytest.raises(CascadeRepoError, match="Unknown VCS provider"):
            build_vcs_provider(provider="sourcehut", token="t")


# --------- GitHubVCSProvider (mocked PyGithub) ----------


class TestGitHubVCSProvider:
    def test_open_pull_request(self):
        from cascade import vcs_github

        with patch.object(vcs_github, "Github", create=True) as mock_gh_class:
            # Patch the import inside the constructor
            with patch.dict(
                "sys.modules",
                {"github": MagicMock(Github=mock_gh_class)},
            ):
                from cascade.vcs_github import GitHubVCSProvider

                mock_pr = MagicMock(number=99, html_url="https://github.com/o/r/pull/99")
                mock_gh_class.return_value.get_repo.return_value.create_pull.return_value = mock_pr
                provider = GitHubVCSProvider(token="t")
                ref = provider.open_pull_request(
                    identity=RepoIdentity(provider="github", owner="o", repo="r"),
                    head="feature",
                    base="main",
                    title="t",
                    body="b",
                )
                assert isinstance(ref, PullRequestRef)
                assert ref.number == 99
                assert ref.url == "https://github.com/o/r/pull/99"
