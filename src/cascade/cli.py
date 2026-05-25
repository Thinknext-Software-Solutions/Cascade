"""Command-line interface for Cascade.

Wires together the pipeline stages into the user-facing commands described
in the README. Each command is intentionally thin -- it parses args,
calls the appropriate module, prints results. Business logic lives in
the modules, not here.

Commands (v0.1):
    cascade init             Scaffold team-memory/ and cascade.yaml
    cascade ingest <file>    Transcribe audio/video -> transcripts/*.yaml  (NOT YET)
    cascade extract <file>   Transcript -> stories YAML for review
    cascade review <file>    Interactive review of extracted stories         (STUB)
    cascade build <file>     Approved stories -> code + tests + PR           (STUB)
    cascade status           Show pipeline state for the current repo       (STUB)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from . import __version__
from .config import DEFAULT_CONFIG_FILENAME, load_config
from .exceptions import CascadeError
from .extractor import extract_stories
from .io import read_story_batch, read_transcript, write_story_batch
from .llm import build_client
from .memory import KNOWN_MEMORY_FILES, TeamMemory


# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        stream=sys.stderr,
    )


# ----------------------------------------------------------------------------
# Root group
# ----------------------------------------------------------------------------


@click.group()
@click.version_option(__version__, prog_name="cascade")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug-level logs.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Cascade: turn a team meeting into shipped code."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# ----------------------------------------------------------------------------
# init
# ----------------------------------------------------------------------------


@cli.command()
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing files if present.",
)
def init(force: bool) -> None:
    """Scaffold cascade.yaml and team-memory/ in the current directory."""
    root = Path.cwd()

    # 1. cascade.yaml
    config_path = root / DEFAULT_CONFIG_FILENAME
    if config_path.exists() and not force:
        click.echo(f"  exists  {config_path.name} (use --force to overwrite)")
    else:
        config_path.write_text(_DEFAULT_CASCADE_YAML)
        click.echo(f"  wrote   {config_path.name}")

    # 2. team-memory/ directory
    mem_dir = root / "team-memory"
    mem_dir.mkdir(exist_ok=True)
    for name in KNOWN_MEMORY_FILES:
        p = mem_dir / name
        if p.exists() and not force:
            click.echo(f"  exists  team-memory/{name} (use --force to overwrite)")
            continue
        p.write_text(_starter_content_for(name))
        click.echo(f"  wrote   team-memory/{name}")

    click.echo()
    click.echo("Cascade initialized. Next steps:")
    click.echo("  1. Edit team-memory/*.md with your team's real conventions and decisions.")
    click.echo("  2. Set the ANTHROPIC_API_KEY environment variable.")
    click.echo("  3. Run: cascade extract <transcript.txt> to try the extractor.")


# ----------------------------------------------------------------------------
# extract
# ----------------------------------------------------------------------------


@cli.command()
@click.argument(
    "transcript_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write the extracted stories YAML. Defaults to "
    "stories/{meeting_id}.yaml relative to the current directory.",
)
@click.option(
    "--model",
    default=None,
    help="Override LLM model from cascade.yaml.",
)
def extract(transcript_path: Path, output: Path | None, model: str | None) -> None:
    """Read a transcript YAML, extract stories, write a YAML batch for review."""
    try:
        config = load_config()
        transcript = read_transcript(transcript_path)
        memory = TeamMemory.load(Path.cwd() / config.memory.path)
        llm = build_client(config.agent.provider, model=model or config.agent.model)

        click.echo(
            f"  extracting from {transcript.meeting_id} "
            f"({len(transcript.turns)} turns, {len(memory.non_empty_files)} memory files)"
        )

        result = extract_stories(
            transcript=transcript,
            llm=llm,
            memory=memory,
            memory_char_budget=config.memory.max_chars_per_call,
            temperature=config.agent.temperature,
        )

        out_path = output or (
            Path.cwd() / "stories" / f"{transcript.meeting_id}.yaml"
        )
        write_story_batch(result.batch, out_path)

        click.echo(
            f"  extracted {len(result.batch.stories)} stories -> {out_path}"
        )
        click.echo(
            f"  tokens: in={result.usage.input_tokens} out={result.usage.output_tokens}"
        )
    except CascadeError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)


# ----------------------------------------------------------------------------
# review (stub for v0.1 follow-up)
# ----------------------------------------------------------------------------


@cli.command()
@click.argument(
    "batch_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def review(batch_path: Path) -> None:
    """Interactively review extracted stories. (Stub -- prints summary only in this build.)"""
    try:
        batch = read_story_batch(batch_path)
    except CascadeError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Story batch: {batch_path}")
    click.echo(f"  meeting:   {batch.meeting_id}")
    click.echo(f"  extracted: {batch.extracted_at.isoformat()}")
    click.echo(f"  model:     {batch.extractor_model}")
    click.echo(f"  stories:   {len(batch.stories)}")
    click.echo()
    for i, s in enumerate(batch.stories, 1):
        click.echo(f"  {i:2d}. [{s.size.value}] [{s.confidence:3d}] {s.title}")
    click.echo()
    click.echo("(Interactive review UI lands in the next build. For now: edit the YAML")
    click.echo("directly to set each story's status to 'approved' or 'rejected'.)")


# ----------------------------------------------------------------------------
# build (stub)
# ----------------------------------------------------------------------------


@cli.command(name="build")
@click.argument(
    "batch_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--story",
    "story_index",
    type=int,
    default=None,
    help="1-based index of a single story to build. Defaults to all approved.",
)
def build(batch_path: Path, story_index: int | None) -> None:
    """Build code and tests for approved stories, then open PRs. (Stub.)"""
    try:
        batch = read_story_batch(batch_path)
    except CascadeError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)

    approved = batch.approved()
    if not approved:
        click.echo("No approved stories in this batch. Run `cascade review` first.")
        sys.exit(0)

    click.echo(f"  {len(approved)} approved story/ies in {batch_path}")
    for s in approved:
        click.echo(f"    - [{s.id}] {s.title}")
    click.echo()
    click.echo("(Build pipeline -- planner, coder, tester, PR-opener -- lands in the next")
    click.echo("milestone. Approved stories are validated here but not yet acted on.)")


# ----------------------------------------------------------------------------
# status (stub)
# ----------------------------------------------------------------------------


@cli.command()
def status() -> None:
    """Show the pipeline state for the current repo. (Stub.)"""
    root = Path.cwd()

    stories_dir = root / "stories"
    transcripts_dir = root / "transcripts"

    n_transcripts = (
        len(list(transcripts_dir.glob("*.yaml"))) if transcripts_dir.exists() else 0
    )
    n_story_batches = (
        len(list(stories_dir.glob("*.yaml"))) if stories_dir.exists() else 0
    )

    click.echo(f"  transcripts:   {n_transcripts}")
    click.echo(f"  story batches: {n_story_batches}")
    click.echo()
    click.echo("(Detailed pipeline state -- per-story progress, open PRs, costs -- lands")
    click.echo("with the build pipeline.)")


# ----------------------------------------------------------------------------
# Starter file content
# ----------------------------------------------------------------------------


_DEFAULT_CASCADE_YAML = """\
version: 1

agent:
  provider: anthropic
  model: claude-opus-4-7
  max_iterations: 1
  temperature: 0.2

memory:
  path: team-memory
  max_chars_per_call: 20000

paths:
  allowed:
    - src/**
    - tests/**
    - docs/**
  disallowed:
    - .github/**
    - migrations/**
    - cascade.yaml
    - team-memory/**

test_command: pytest
"""


_STARTER_TEMPLATES: dict[str, str] = {
    "conventions.md": (
        "# Conventions\n\n"
        "> Tell your team's AI sessions what your team's conventions are.\n"
        "> Be specific. Examples beat abstractions.\n"
    ),
    "decisions.md": (
        "# Decisions\n\n"
        "> Architectural decisions made and why. ADR-style log.\n"
    ),
    "constraints.md": (
        "# Constraints\n\n"
        "> Non-functional requirements (performance, security, compliance).\n"
    ),
    "glossary.md": (
        "# Glossary\n\n"
        "> Domain-specific terms used in this codebase.\n"
    ),
    "prior-work.md": (
        "# Prior Work\n\n"
        "> Brief summaries of recently shipped stories. Helps the AI avoid duplicates.\n"
    ),
}


def _starter_content_for(filename: str) -> str:
    return _STARTER_TEMPLATES.get(filename, f"# {filename}\n")


def main() -> None:
    """Entry point referenced from pyproject.toml."""
    cli(obj={})


if __name__ == "__main__":
    main()
