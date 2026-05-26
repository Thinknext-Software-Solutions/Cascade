"""Tests for cascade.languages."""

from __future__ import annotations

from pathlib import Path

import pytest

from cascade.exceptions import CascadeError
from cascade.languages import (
    GO,
    JAVA,
    JAVASCRIPT,
    PROFILES,
    PYTHON,
    RUBY,
    RUST,
    TYPESCRIPT,
    detect_language,
    get_profile,
    resolve_language,
)


# --------- Profile registry ----------


class TestProfileRegistry:
    def test_all_built_in_languages_registered(self):
        expected = {"python", "typescript", "javascript", "go", "rust", "java", "ruby", "csharp"}
        assert expected.issubset(PROFILES.keys())

    def test_profile_keys_match_names(self):
        for key, profile in PROFILES.items():
            assert key == profile.name

    def test_all_profiles_have_test_command(self):
        for profile in PROFILES.values():
            assert profile.test_command, f"{profile.name} missing test_command"
            assert isinstance(profile.test_command, tuple)


# --------- get_profile ----------


class TestGetProfile:
    def test_canonical_lookup(self):
        assert get_profile("python") is PYTHON

    def test_case_insensitive(self):
        assert get_profile("Python") is PYTHON
        assert get_profile("PYTHON") is PYTHON
        assert get_profile(" python ") is PYTHON

    def test_unknown_raises_with_help(self):
        with pytest.raises(CascadeError, match="Unknown language"):
            get_profile("cobol")


# --------- detect_language ----------


class TestDetect:
    def test_python_via_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'")
        assert detect_language(tmp_path) is PYTHON

    def test_python_via_setup_py(self, tmp_path):
        (tmp_path / "setup.py").write_text("from setuptools import setup; setup()")
        assert detect_language(tmp_path) is PYTHON

    def test_typescript_wins_over_javascript(self, tmp_path):
        (tmp_path / "package.json").write_text("{}")
        (tmp_path / "tsconfig.json").write_text("{}")
        # Both could match but TS has higher detection_priority
        assert detect_language(tmp_path) is TYPESCRIPT

    def test_javascript_only_when_no_tsconfig(self, tmp_path):
        (tmp_path / "package.json").write_text("{}")
        assert detect_language(tmp_path) is JAVASCRIPT

    def test_go(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/x")
        assert detect_language(tmp_path) is GO

    def test_rust(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'")
        assert detect_language(tmp_path) is RUST

    def test_java_via_pom(self, tmp_path):
        (tmp_path / "pom.xml").write_text("<project/>")
        assert detect_language(tmp_path) is JAVA

    def test_java_via_gradle(self, tmp_path):
        (tmp_path / "build.gradle").write_text("plugins {}")
        assert detect_language(tmp_path) is JAVA

    def test_ruby_via_gemfile(self, tmp_path):
        (tmp_path / "Gemfile").write_text("source 'https://rubygems.org'")
        assert detect_language(tmp_path) is RUBY

    def test_csharp_via_csproj_glob(self, tmp_path):
        (tmp_path / "MyApp.csproj").write_text("<Project/>")
        detected = detect_language(tmp_path)
        assert detected is not None
        assert detected.name == "csharp"

    def test_no_markers_returns_none(self, tmp_path):
        assert detect_language(tmp_path) is None

    def test_nonexistent_dir_returns_none(self, tmp_path):
        assert detect_language(tmp_path / "missing") is None

    def test_polyglot_picks_highest_priority(self, tmp_path):
        # Python + Go in the same repo. Go has detection_priority=90, Python=60.
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'")
        (tmp_path / "go.mod").write_text("module x")
        assert detect_language(tmp_path) is GO

    def test_detection_is_non_recursive(self, tmp_path):
        # A marker file in a subdirectory should NOT be detected.
        sub = tmp_path / "vendor" / "some-pkg"
        sub.mkdir(parents=True)
        (sub / "pyproject.toml").write_text("[project]\nname='x'")
        assert detect_language(tmp_path) is None


# --------- resolve_language ----------


class TestResolve:
    def test_explicit_override_wins(self, tmp_path):
        # Even though we detect Python, explicit Go wins.
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'")
        assert resolve_language(tmp_path, configured_name="go") is GO

    def test_falls_back_to_detection(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'")
        assert resolve_language(tmp_path) is RUST

    def test_raises_when_no_signal(self, tmp_path):
        with pytest.raises(CascadeError, match="Could not detect"):
            resolve_language(tmp_path)

    def test_invalid_explicit_name_raises(self, tmp_path):
        with pytest.raises(CascadeError, match="Unknown language"):
            resolve_language(tmp_path, configured_name="cobol")
