"""Tests for cascade.progress: ProgressReporter implementations."""

from __future__ import annotations

import pytest

from cascade.progress import (
    NoopProgress,
    ProgressEvent,
    RecordingProgress,
    RichProgress,
)


class TestNoopProgress:
    def test_methods_do_nothing(self):
        p = NoopProgress()
        p.start("plan", "starting")
        p.update("plan", "halfway")
        p.succeed("plan", "done")
        p.fail("plan", "broke")

    def test_stage_context_manager_swallows_nothing(self):
        p = NoopProgress()
        with p.stage("plan", "starting") as handle:
            handle.update("halfway")
            handle.succeed("done")

    def test_stage_context_manager_reraises(self):
        p = NoopProgress()
        with pytest.raises(ValueError):
            with p.stage("plan", "starting"):
                raise ValueError("boom")


class TestRecordingProgress:
    def test_records_direct_calls(self):
        p = RecordingProgress()
        p.start("plan", "starting")
        p.update("plan", "halfway")
        p.succeed("plan", "done")
        assert p.events == [
            ProgressEvent("start", "plan", "starting"),
            ProgressEvent("update", "plan", "halfway"),
            ProgressEvent("succeed", "plan", "done"),
        ]

    def test_stage_context_auto_succeeds_on_clean_exit(self):
        p = RecordingProgress()
        with p.stage("plan", "starting"):
            pass
        assert p.stages_started() == ["plan"]
        assert p.stages_succeeded() == ["plan"]
        assert p.stages_failed() == []

    def test_stage_context_auto_fails_on_exception(self):
        p = RecordingProgress()
        with pytest.raises(ValueError):
            with p.stage("plan", "starting"):
                raise ValueError("boom")
        assert p.stages_started() == ["plan"]
        assert p.stages_succeeded() == []
        assert p.stages_failed() == ["plan"]

    def test_explicit_succeed_inside_stage_is_not_double_reported(self):
        p = RecordingProgress()
        with p.stage("plan", "starting") as handle:
            handle.succeed("custom message")
        # Should record exactly one succeed (the explicit one), not two
        assert p.stages_succeeded() == ["plan"]
        succeed_events = [e for e in p.events if e.kind == "succeed"]
        assert len(succeed_events) == 1
        assert succeed_events[0].message == "custom message"

    def test_explicit_fail_inside_stage_is_not_double_reported(self):
        p = RecordingProgress()
        with p.stage("plan", "starting") as handle:
            handle.fail("could not plan")
        # Clean exit AFTER explicit fail should NOT also auto-succeed
        assert p.stages_failed() == ["plan"]
        assert p.stages_succeeded() == []

    def test_multiple_stages_recorded_in_order(self):
        p = RecordingProgress()
        with p.stage("plan", "planning"):
            pass
        with p.stage("code", "coding"):
            pass
        with p.stage("test", "testing"):
            pass
        assert p.stages_started() == ["plan", "code", "test"]
        assert p.stages_succeeded() == ["plan", "code", "test"]


class TestRichProgress:
    """RichProgress is largely a render layer; we just verify it does not
    crash and that it degrades gracefully when rich is unavailable."""

    def test_disabled_mode_is_silent(self):
        p = RichProgress(enabled=False)
        p.start("plan", "starting")
        p.update("plan", "halfway")
        p.succeed("plan", "done")
        p.fail("plan", "broke")
        with p.stage("plan", "starting") as handle:
            handle.update("u")
            handle.succeed("s")

    def test_stage_reraises_exceptions(self):
        p = RichProgress(enabled=False)
        with pytest.raises(ValueError):
            with p.stage("plan", "starting"):
                raise ValueError("boom")

    def test_console_capture_does_not_error(self, capsys):
        """Sanity check that calling with a real (enabled) console doesn't blow up."""
        pytest.importorskip("rich")
        p = RichProgress(enabled=True)
        with p.stage("plan", "starting") as handle:
            handle.succeed("done")
        # We aren't asserting on stderr content (rich rendering is brittle to
        # assert on); just that nothing raised.
