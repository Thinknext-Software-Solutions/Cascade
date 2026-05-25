"""Tests for cascade.tester."""

from __future__ import annotations

from pathlib import Path

import pytest

from cascade.exceptions import CascadeError
from cascade.languages import LanguageProfile, PYTHON
from cascade.tester import install_dependencies, run_tests


def _make_script(tmp_path: Path, name: str, body: str, *, mode=0o755) -> Path:
    """Create a shell script and return its path."""
    p = tmp_path / name
    p.write_text(body)
    p.chmod(mode)
    return p


def _profile_pointing_to(script: Path) -> LanguageProfile:
    """A LanguageProfile whose test_command runs the given script."""
    return LanguageProfile(
        name="fake",
        display_name="Fake",
        file_extensions=(".x",),
        source_dir_default="src",
        test_dir_default="tests",
        test_file_glob="*.x",
        test_command=(str(script),),
        install_command=None,
        type_check_command=None,
        detection_files=(),
    )


class TestRunTests:
    def test_passing_command(self, tmp_path):
        script = _make_script(tmp_path, "ok.sh", "#!/bin/sh\necho pytest passed: 7 tests\nexit 0\n")
        profile = _profile_pointing_to(script)
        result = run_tests(tmp_path, profile)
        assert result.passed is True
        assert result.exit_code == 0
        assert "passed" in result.stdout.lower()
        assert result.duration_seconds >= 0

    def test_failing_command(self, tmp_path):
        script = _make_script(tmp_path, "fail.sh", "#!/bin/sh\necho test failed: 3\nexit 1\n")
        profile = _profile_pointing_to(script)
        result = run_tests(tmp_path, profile)
        assert result.passed is False
        assert result.exit_code == 1

    def test_summary_extracts_pass_line(self, tmp_path):
        script = _make_script(
            tmp_path,
            "ok.sh",
            "#!/bin/sh\necho '==== 42 passed in 0.18s ===='\nexit 0\n",
        )
        profile = _profile_pointing_to(script)
        result = run_tests(tmp_path, profile)
        assert "42 passed" in result.summary

    def test_override_command_replaces_profile(self, tmp_path):
        ok = _make_script(tmp_path, "ok.sh", "#!/bin/sh\nexit 0\n")
        fail_profile = _profile_pointing_to(
            _make_script(tmp_path, "fail.sh", "#!/bin/sh\nexit 1\n")
        )
        result = run_tests(tmp_path, fail_profile, override_command=[str(ok)])
        assert result.passed is True

    def test_missing_executable_raises(self, tmp_path):
        profile = LanguageProfile(
            name="missing",
            display_name="Missing",
            file_extensions=(".x",),
            source_dir_default="src",
            test_dir_default="tests",
            test_file_glob="*",
            test_command=("definitely-not-on-path-12345",),
            install_command=None,
            type_check_command=None,
            detection_files=(),
        )
        with pytest.raises(CascadeError, match="not found on PATH"):
            run_tests(tmp_path, profile)

    def test_timeout_returns_failed_result(self, tmp_path):
        script = _make_script(tmp_path, "slow.sh", "#!/bin/sh\nsleep 2\nexit 0\n")
        profile = _profile_pointing_to(script)
        result = run_tests(tmp_path, profile, timeout_seconds=0.2)
        assert result.passed is False
        assert "timed out" in result.summary

    def test_empty_command_raises(self, tmp_path):
        profile = LanguageProfile(
            name="empty",
            display_name="Empty",
            file_extensions=(".x",),
            source_dir_default="src",
            test_dir_default="tests",
            test_file_glob="*",
            test_command=(),
            install_command=None,
            type_check_command=None,
            detection_files=(),
        )
        with pytest.raises(CascadeError, match="No test command"):
            run_tests(tmp_path, profile)


class TestInstallDependencies:
    def test_skipped_when_no_install_command(self, tmp_path):
        profile = LanguageProfile(
            name="x",
            display_name="X",
            file_extensions=(".x",),
            source_dir_default="src",
            test_dir_default="tests",
            test_file_glob="*",
            test_command=("true",),
            install_command=None,
            type_check_command=None,
            detection_files=(),
        )
        result = install_dependencies(tmp_path, profile)
        assert result.passed is True
        assert result.summary == "skipped"

    def test_runs_command(self, tmp_path):
        script = _make_script(tmp_path, "inst.sh", "#!/bin/sh\nexit 0\n")
        profile = LanguageProfile(
            name="x",
            display_name="X",
            file_extensions=(".x",),
            source_dir_default="src",
            test_dir_default="tests",
            test_file_glob="*",
            test_command=("true",),
            install_command=(str(script),),
            type_check_command=None,
            detection_files=(),
        )
        result = install_dependencies(tmp_path, profile)
        assert result.passed is True
        assert "install ok" in result.summary
