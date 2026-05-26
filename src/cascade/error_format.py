"""Render CascadeError instances as multi-line, user-facing text.

Used by every CLI command's exception handler so error output is
consistent across the entire CLI. Inspired by Rust's compiler diagnostics
and modern CLIs like `cargo`, `gh`, and `homebrew`.

Output shape:

    error: <one-line summary>

      What this means:
        <optional explanation paragraph>

      How to fix:
        - <suggestion 1>
        - <suggestion 2>

      Learn more: <command or URL>

The 'What this means' and 'Learn more' sections are only included when
the exception carries that metadata. Bare exceptions still render
cleanly with just the error line.
"""

from __future__ import annotations

from typing import Optional

import click

from .exceptions import CascadeError


def format_error(
    exc: CascadeError,
    *,
    explanation: Optional[str] = None,
    use_color: bool = True,
) -> str:
    """Render a CascadeError as multi-line CLI text.

    Args:
        exc: The exception to format.
        explanation: Optional one-line context to insert under 'What this
            means'. Useful when the CLI command can add context the
            exception itself doesn't know (e.g., "while building story X").
        use_color: When True, applies click styling.

    Returns:
        The formatted multi-line string. Caller is responsible for
        click.echo / sys.stderr writing.
    """
    parts: list[str] = []

    error_label = click.style("error:", fg="red", bold=True) if use_color else "error:"
    parts.append(f"{error_label} {exc.message}")

    if explanation:
        parts.append("")
        parts.append(f"  {_section('What this means:', use_color)}")
        parts.append(f"    {explanation}")

    hints = exc.hints
    if hints:
        parts.append("")
        parts.append(f"  {_section('How to fix:', use_color)}")
        for h in hints:
            # Split multi-line hints sensibly: first line bulleted, rest indented
            hint_lines = h.split("\n")
            parts.append(f"    {_bullet(use_color)} {hint_lines[0]}")
            for extra in hint_lines[1:]:
                parts.append(f"      {extra}")

    if exc.learn_more:
        parts.append("")
        learn_text = _style("Learn more:", "cyan", use_color)
        parts.append(f"  {learn_text} {exc.learn_more}")

    return "\n".join(parts)


def echo_error(
    exc: CascadeError,
    *,
    explanation: Optional[str] = None,
) -> None:
    """Convenience: format the error and write to stderr in one call."""
    click.echo(format_error(exc, explanation=explanation), err=True)


def _section(label: str, use_color: bool) -> str:
    if use_color:
        return click.style(label, fg="white", bold=True)
    return label


def _bullet(use_color: bool) -> str:
    """The bullet character used for hint lists.

    `*` over `•` for terminals that don't render Unicode well.
    """
    return click.style("*", fg="cyan", bold=True) if use_color else "*"


def _style(text: str, color: str, use_color: bool) -> str:
    return click.style(text, fg=color) if use_color else text
