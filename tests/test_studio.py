"""Tests for cascade.studio (Studio web dashboard wiring).

Covers:
- create_app returns a working FastAPI instance when [studio] is installed
- /api/health returns ok + version
- /api/metadata surfaces the supported providers/languages/etc.
- Static frontend mounts at / and returns the index.html
- The CLI `cascade ui` command exists with the expected options
"""

from __future__ import annotations

import pytest

# Skip the whole module if [studio] isn't installed
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")


from fastapi.testclient import TestClient

from cascade.studio.server import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


class TestHealth:
    def test_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body


class TestMetadata:
    def test_returns_supported_providers(self, client):
        resp = client.get("/api/metadata")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "Cascade Studio"
        supports = body["cascade_supports"]
        assert "anthropic" in supports["llm_providers"]
        assert "openai" in supports["llm_providers"]
        assert "github" in supports["vcs_providers"]
        assert "python" in supports["languages"]
        assert "jira" in supports["issue_sources"]


class TestStaticFrontend:
    def test_root_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        # Could be the bundled index.html or the placeholder; both have
        # Cascade Studio in the title.
        body = resp.text.lower()
        assert "cascade studio" in body
        assert "html" in body  # responds with HTML, not JSON


class TestOpenAPI:
    def test_docs_available_at_api_docs(self, client):
        resp = client.get("/api/docs")
        assert resp.status_code == 200


class TestCLICommand:
    def test_ui_command_registered(self):
        from cascade.cli import cli

        assert "ui" in cli.commands

    def test_ui_command_options(self):
        from cascade.cli import cli

        ui_cmd = cli.commands["ui"]
        option_names = {opt.name for opt in ui_cmd.params}
        assert "host" in option_names
        assert "port" in option_names
        assert "no_browser" in option_names

    def test_ui_command_help_text(self):
        from click.testing import CliRunner

        from cascade.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["ui", "--help"])
        assert result.exit_code == 0
        assert "Cascade Studio" in result.output
        assert "[studio]" in result.output
