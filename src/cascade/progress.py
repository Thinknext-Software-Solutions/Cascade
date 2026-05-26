"""Progress reporting during long-running pipeline stages.

The pipeline calls into a ProgressReporter at the start and end of each
stage. Default implementation is NoopProgress (silent; used by tests and
library callers). The CLI uses RichProgress for animated spinners and
checkmarks during interactive runs.

This decouples 'what stage is happening' from 'how to display it', so:
- Tests can use RecordingProgress to assert which stages fired
- The CLI can use rich-based spinners
- A future web UI could subscribe to the same events to stream to the browser
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional, Protocol


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------------


class ProgressReporter(Protocol):
    """Pipeline stages call into this to report what they're doing.

    Implementations decide whether to render spinners, log lines, post to
    a websocket, etc. The pipeline stays display-agnostic.
    """

    def start(self, stage: str, message: str) -> None:
        """A stage is beginning. May render a spinner."""
        ...

    def update(self, stage: str, message: str) -> None:
        """Update the message for an in-flight stage (without ending it)."""
        ...

    def succeed(self, stage: str, message: str) -> None:
        """The stage completed successfully. May render a checkmark."""
        ...

    def fail(self, stage: str, message: str) -> None:
        """The stage failed. May render an X mark."""
        ...

    @contextmanager
    def stage(self, name: str, start_message: str) -> Iterator["StageHandle"]:
        """Context-manager convenience. Yields a StageHandle the caller can
        update() / succeed() / fail() on, with default succeed on clean exit
        and fail on raise."""
        ...


class StageHandle(Protocol):
    """Returned by ProgressReporter.stage() context manager."""

    def update(self, message: str) -> None: ...
    def succeed(self, message: str) -> None: ...
    def fail(self, message: str) -> None: ...


# ----------------------------------------------------------------------------
# Noop (default; silent; used by tests and library callers)
# ----------------------------------------------------------------------------


class _NoopStageHandle:
    def update(self, message: str) -> None:
        pass

    def succeed(self, message: str) -> None:
        pass

    def fail(self, message: str) -> None:
        pass


class NoopProgress:
    """Reporter that does nothing. Default for library callers and tests
    that don't care about progress events."""

    def start(self, stage: str, message: str) -> None:
        pass

    def update(self, stage: str, message: str) -> None:
        pass

    def succeed(self, stage: str, message: str) -> None:
        pass

    def fail(self, stage: str, message: str) -> None:
        pass

    @contextmanager
    def stage(self, name: str, start_message: str) -> Iterator[StageHandle]:
        yield _NoopStageHandle()


# ----------------------------------------------------------------------------
# Recording (for tests)
# ----------------------------------------------------------------------------


@dataclass
class ProgressEvent:
    """One reported event."""

    kind: str  # "start" | "update" | "succeed" | "fail"
    stage: str
    message: str


@dataclass
class RecordingProgress:
    """Reporter that records every event. Tests inspect .events."""

    events: list[ProgressEvent] = field(default_factory=list)

    def start(self, stage: str, message: str) -> None:
        self.events.append(ProgressEvent("start", stage, message))

    def update(self, stage: str, message: str) -> None:
        self.events.append(ProgressEvent("update", stage, message))

    def succeed(self, stage: str, message: str) -> None:
        self.events.append(ProgressEvent("succeed", stage, message))

    def fail(self, stage: str, message: str) -> None:
        self.events.append(ProgressEvent("fail", stage, message))

    @contextmanager
    def stage(self, name: str, start_message: str) -> Iterator[StageHandle]:
        self.start(name, start_message)
        outer = self

        class _Handle:
            def update(self, message: str) -> None:
                outer.update(name, message)

            def succeed(self, message: str) -> None:
                outer.succeed(name, message)

            def fail(self, message: str) -> None:
                outer.fail(name, message)

        handle = _Handle()
        succeeded = False
        try:
            yield handle
            succeeded = True
        except BaseException:
            self.fail(name, "failed")
            raise
        finally:
            if succeeded and not _last_event_terminal(self.events, name):
                self.succeed(name, "done")

    def stages_started(self) -> list[str]:
        return [e.stage for e in self.events if e.kind == "start"]

    def stages_succeeded(self) -> list[str]:
        return [e.stage for e in self.events if e.kind == "succeed"]

    def stages_failed(self) -> list[str]:
        return [e.stage for e in self.events if e.kind == "fail"]


def _last_event_terminal(events: list[ProgressEvent], stage: str) -> bool:
    """True if the most recent event for `stage` was succeed/fail (so the
    context manager doesn't double-emit succeed)."""
    for e in reversed(events):
        if e.stage == stage and e.kind in {"succeed", "fail"}:
            return True
        if e.stage == stage and e.kind == "start":
            return False
    return False


# ----------------------------------------------------------------------------
# Rich-based (for interactive CLI)
# ----------------------------------------------------------------------------


class RichProgress:
    """Animated reporter for the CLI. Uses rich.console.Status for the
    in-flight spinner and prints success/fail lines as stages finish.

    Falls back to a plain printer if rich isn't installed (rich is a base
    dependency of cascade-agent so this is rare).
    """

    def __init__(self, *, console=None, enabled: bool = True):
        self._enabled = enabled
        self._console = console
        self._current_stage: Optional[str] = None
        self._current_status_cm = None  # rich.status.Status context manager
        if enabled and console is None:
            try:
                from rich.console import Console

                self._console = Console(stderr=True)
            except ImportError:
                self._enabled = False
                self._console = None

    def _stop_current(self) -> None:
        if self._current_status_cm is not None:
            try:
                self._current_status_cm.stop()
            except Exception:
                pass
            self._current_status_cm = None
        self._current_stage = None

    def start(self, stage: str, message: str) -> None:
        if not self._enabled or self._console is None:
            return
        # Stop any prior status before starting a new one
        self._stop_current()
        self._current_stage = stage
        try:
            from rich.status import Status

            self._current_status_cm = Status(
                f"[cyan]{stage}[/cyan]: {message}",
                spinner="dots",
                console=self._console,
            )
            self._current_status_cm.start()
        except ImportError:
            self._console.print(f"  [cyan]{stage}[/cyan]: {message}...")

    def update(self, stage: str, message: str) -> None:
        if not self._enabled or self._current_status_cm is None:
            return
        try:
            self._current_status_cm.update(f"[cyan]{stage}[/cyan]: {message}")
        except Exception:
            pass

    def succeed(self, stage: str, message: str) -> None:
        if not self._enabled or self._console is None:
            return
        self._stop_current()
        self._console.print(f"  [green]✓[/green] [cyan]{stage}[/cyan]: {message}")

    def fail(self, stage: str, message: str) -> None:
        if not self._enabled or self._console is None:
            return
        self._stop_current()
        self._console.print(f"  [red]✗[/red] [cyan]{stage}[/cyan]: {message}")

    @contextmanager
    def stage(self, name: str, start_message: str) -> Iterator[StageHandle]:
        self.start(name, start_message)
        reporter = self

        class _Handle:
            _terminated = False

            def update(self, message: str) -> None:
                reporter.update(name, message)

            def succeed(self, message: str) -> None:
                self._terminated = True
                reporter.succeed(name, message)

            def fail(self, message: str) -> None:
                self._terminated = True
                reporter.fail(name, message)

        handle = _Handle()
        try:
            yield handle
        except BaseException:
            if not handle._terminated:
                reporter.fail(name, "failed")
            raise
        else:
            if not handle._terminated:
                reporter.succeed(name, "done")
