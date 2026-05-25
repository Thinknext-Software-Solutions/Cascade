"""Tests for the cascade CLI.

Focused on argument handling, file outputs, and error paths -- not on the
underlying pipeline behavior (those have their own unit tests).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from cascade.cli import cli
from cascade.io import read_story_batch, write_story_batch, write_transcript


class TestVersion:
    def test_version_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "cascade" in result.output.lower()


class TestInit:
    def test_creates_cascade_yaml_and_memory_files(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            cwd = Path.cwd()
            assert (cwd / "cascade.yaml").exists()
            assert (cwd / "team-memory" / "conventions.md").exists()
            assert (cwd / "team-memory" / "decisions.md").exists()
            assert (cwd / "team-memory" / "glossary.md").exists()
            assert (cwd / "team-memory" / "constraints.md").exists()
            assert (cwd / "team-memory" / "prior-work.md").exists()

    def test_does_not_overwrite_existing_without_force(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("cascade.yaml").write_text("custom: content")
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert "exists" in result.output
            assert Path("cascade.yaml").read_text() == "custom: content"

    def test_force_overwrites(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("cascade.yaml").write_text("custom: content")
            result = runner.invoke(cli, ["init", "--force"])
            assert result.exit_code == 0
            assert "wrote" in result.output
            assert "version: 1" in Path("cascade.yaml").read_text()


class TestReview:
    def test_lists_stories(self, tmp_path, sample_story_batch):
        batch_path = tmp_path / "batch.yaml"
        write_story_batch(sample_story_batch, batch_path)
        runner = CliRunner()
        # review now delegates to the interactive review_batch; mock it
        # so the test doesn't try to read from stdin.
        with patch("cascade.cli.review_batch", return_value=(sample_story_batch, None)):
            result = runner.invoke(cli, ["review", str(batch_path)])
        assert result.exit_code == 0
        assert sample_story_batch.meeting_id in result.output
        assert "claude-opus-4-7" in result.output

    def test_missing_file_errors(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["review", str(tmp_path / "nope.yaml")])
        assert result.exit_code != 0


class TestStatus:
    def test_runs_in_empty_dir(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["status"])
            assert result.exit_code == 0
            assert "transcripts" in result.output

    def test_counts_files(self, tmp_path, sample_transcript, sample_story_batch):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            cwd = Path.cwd()
            write_transcript(sample_transcript, cwd / "transcripts" / "t.yaml")
            write_story_batch(sample_story_batch, cwd / "stories" / "s.yaml")
            result = runner.invoke(cli, ["status"])
            assert "transcripts:   1" in result.output
            assert "story batches: 1" in result.output


class TestBuild:
    def test_no_approved_stories_returns_zero(self, tmp_path, sample_story_batch):
        batch_path = tmp_path / "batch.yaml"
        # Default sample has status=EXTRACTED, not APPROVED
        write_story_batch(sample_story_batch, batch_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["build", str(batch_path)])
        assert result.exit_code == 0
        assert "No approved stories" in result.output


class TestExtract:
    def test_calls_pipeline_with_mocked_llm(self, tmp_path, sample_transcript, sample_story_batch):
        from cascade.extractor import ExtractionResult
        from cascade.llm import LLMUsage

        transcript_path = tmp_path / "t.yaml"
        write_transcript(sample_transcript, transcript_path)

        usage = LLMUsage(input_tokens=42, output_tokens=84, model="m", provider="p")
        fake_result = ExtractionResult(batch=sample_story_batch, usage=usage)

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as cwd_str:
            cwd = Path(cwd_str)
            # Copy transcript into the isolated cwd so the path is relative-safe
            new_transcript = cwd / "t.yaml"
            write_transcript(sample_transcript, new_transcript)

            with patch("cascade.cli.build_client_from_credentials") as mock_build, patch(
                "cascade.cli.extract_stories", return_value=fake_result
            ), patch("cascade.cli.resolve_llm_credentials") as mock_creds:
                mock_creds.return_value = object()
                mock_build.return_value = object()
                result = runner.invoke(cli, ["extract", str(new_transcript)])

            assert result.exit_code == 0, result.output
            output_path = cwd / "stories" / f"{sample_transcript.meeting_id}.yaml"
            assert output_path.exists()
            loaded = read_story_batch(output_path)
            assert loaded.meeting_id == sample_story_batch.meeting_id
