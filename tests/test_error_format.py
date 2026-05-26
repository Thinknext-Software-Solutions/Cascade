"""Tests for cascade.error_format and the enriched CascadeError hierarchy."""

from __future__ import annotations

import pytest

from cascade.error_format import echo_error, format_error
from cascade.exceptions import (
    CascadeConfigError,
    CascadeError,
    CascadeLLMError,
    CascadeRepoError,
)


# --------- Enriched exceptions ----------


class TestCascadeError:
    def test_message_only(self):
        exc = CascadeError("something broke")
        assert exc.message == "something broke"
        assert str(exc) == "something broke"
        assert exc.hint is None
        assert exc.hints == []
        assert exc.learn_more is None

    def test_single_string_hint(self):
        exc = CascadeError("oops", hint="try this")
        assert exc.hint == "try this"
        assert exc.hints == ["try this"]

    def test_list_of_hints(self):
        exc = CascadeError("oops", hint=["first", "second", "third"])
        assert exc.hints == ["first", "second", "third"]

    def test_learn_more(self):
        exc = CascadeError("oops", learn_more="cascade doctor")
        assert exc.learn_more == "cascade doctor"

    def test_all_optional(self):
        exc = CascadeError(
            "oops",
            hint=["try X", "try Y"],
            learn_more="https://example.com/docs",
        )
        assert exc.hints == ["try X", "try Y"]
        assert exc.learn_more == "https://example.com/docs"

    def test_subclasses_inherit_features(self):
        exc = CascadeConfigError(
            "bad config",
            hint="run cascade init",
            learn_more="cascade doctor",
        )
        assert isinstance(exc, CascadeError)
        assert exc.message == "bad config"
        assert exc.hints == ["run cascade init"]


# --------- format_error rendering ----------


class TestFormatErrorBasics:
    def test_message_only_renders_single_line(self):
        exc = CascadeError("plain error")
        out = format_error(exc, use_color=False)
        assert "error:" in out
        assert "plain error" in out
        # No hints section
        assert "How to fix" not in out
        assert "Learn more" not in out

    def test_single_hint_appears(self):
        exc = CascadeError("plain error", hint="try running X")
        out = format_error(exc, use_color=False)
        assert "How to fix:" in out
        assert "try running X" in out

    def test_multiple_hints_all_appear(self):
        exc = CascadeError("plain error", hint=["first option", "second option", "third option"])
        out = format_error(exc, use_color=False)
        assert "first option" in out
        assert "second option" in out
        assert "third option" in out

    def test_learn_more_appears(self):
        exc = CascadeError("plain error", learn_more="cascade doctor")
        out = format_error(exc, use_color=False)
        assert "Learn more:" in out
        assert "cascade doctor" in out

    def test_explanation_appears_when_passed(self):
        exc = CascadeError("plain error")
        out = format_error(
            exc, explanation="This happened during story X.", use_color=False
        )
        assert "What this means:" in out
        assert "This happened during story X." in out

    def test_multiline_hint_indents_continuation(self):
        exc = CascadeError("oops", hint="first line\nsecond line\nthird line")
        out = format_error(exc, use_color=False)
        # The first line is bulleted; continuation lines are indented under it
        lines = out.splitlines()
        first_line_idx = next(i for i, l in enumerate(lines) if "first line" in l)
        # Subsequent lines should be present in the same area
        assert "second line" in lines[first_line_idx + 1]
        assert "third line" in lines[first_line_idx + 2]

    def test_no_color_strips_ansi(self):
        exc = CascadeError("plain", hint="try X", learn_more="cascade doctor")
        out = format_error(exc, use_color=False)
        # No ANSI escape sequences
        assert "\x1b[" not in out


class TestFormatErrorOrder:
    def test_sections_appear_in_expected_order(self):
        exc = CascadeError(
            "oops",
            hint=["fix 1"],
            learn_more="cascade doctor",
        )
        out = format_error(
            exc, explanation="context here", use_color=False
        )
        msg_idx = out.index("oops")
        means_idx = out.index("What this means")
        fix_idx = out.index("How to fix")
        learn_idx = out.index("Learn more")
        assert msg_idx < means_idx < fix_idx < learn_idx


# --------- echo_error ----------


def test_echo_error_writes_to_stderr(capsys):
    exc = CascadeError("oops", hint="try X")
    echo_error(exc)
    captured = capsys.readouterr()
    assert "oops" in captured.err
    assert "try X" in captured.err
    # nothing on stdout
    assert captured.out == ""


# --------- Integration: enriched raise sites ----------


class TestUserConfigHints:
    def test_no_api_key_includes_actionable_hints(self, monkeypatch):
        from cascade.user_config import UserConfig, resolve_llm_credentials

        for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.delenv(env, raising=False)
        cfg = UserConfig()

        with pytest.raises(CascadeConfigError) as info:
            resolve_llm_credentials(user_config=cfg, provider="anthropic")
        exc = info.value
        assert exc.hints, "Expected actionable hints"
        # Should mention the configure command
        joined = "\n".join(exc.hints)
        assert "cascade configure llm anthropic" in joined
        # Anthropic specifically should suggest claude_code as alternative
        assert "claude_code" in joined
        # And ollama
        assert "ollama" in joined.lower()
        # learn_more should point to doctor
        assert exc.learn_more == "cascade doctor"

    def test_no_vcs_token_includes_token_url(self, monkeypatch):
        from cascade.user_config import UserConfig, resolve_vcs_credentials

        for env in ("GITHUB_TOKEN", "GH_TOKEN"):
            monkeypatch.delenv(env, raising=False)
        cfg = UserConfig()

        with pytest.raises(CascadeConfigError) as info:
            resolve_vcs_credentials(user_config=cfg, provider="github")
        exc = info.value
        joined = "\n".join(exc.hints)
        assert "github.com/settings/tokens" in joined
        # Suggests --no-pr as workaround
        assert "--no-pr" in joined


class TestLanguageDetectionHints:
    def test_undetected_includes_actionable_hints(self, tmp_path):
        from cascade.exceptions import CascadeError as _CE
        from cascade.languages import resolve_language

        with pytest.raises(_CE) as info:
            resolve_language(tmp_path)
        exc = info.value
        joined = "\n".join(exc.hints)
        assert "cascade.yaml" in joined
        assert "--language" in joined
        assert "python" in joined  # lists supported languages
        assert exc.learn_more == "cascade doctor"


class TestRepoCleanHints:
    def test_dirty_tree_includes_recovery_options(self, tmp_path):
        from unittest.mock import MagicMock

        from cascade.repo import _check_clean

        res = MagicMock(stdout=" M dirty.txt\n")
        with pytest.raises(CascadeRepoError) as info:
            _check_clean(res)
        exc = info.value
        joined = "\n".join(exc.hints)
        # The three standard recovery options
        assert "commit" in joined.lower()
        assert "stash" in joined.lower()
        assert "restore" in joined.lower() or "discard" in joined.lower()
