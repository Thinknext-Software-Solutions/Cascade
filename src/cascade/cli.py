"""Command-line interface for Cascade.

Wires together the pipeline stages into the user-facing commands described
in the README. Each command is intentionally thin -- it parses args,
calls the appropriate module, prints results. Business logic lives in
the modules, not here.

Commands:
    cascade init                Scaffold team-memory/ and cascade.yaml
    cascade configure ...       Set up credentials (LLM/VCS/issue trackers)
    cascade ingest <file>       Transcribe audio/video -> transcripts/*.yaml
    cascade extract <file>      Transcript -> stories YAML for review
    cascade review <file>       Interactive review of extracted stories
    cascade prompt "<text>"     Build directly from an ad-hoc prompt
    cascade ticket <provider:id> Build from a tracker ticket
    cascade build <file>        Approved stories -> code + tests + PR
    cascade status              Show pipeline state for the current repo
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from . import __version__
from .config import DEFAULT_CONFIG_FILENAME, load_config
from .cost import CostTracker, format_cost
from .demo import run_demo
from .doctor import CheckStatus, run_doctor, summarize
from .exceptions import CascadeError
from .extractor import extract_stories
from .init_seed import seed_team_memory
from .io import read_story_batch, read_transcript, write_story_batch
from .issue_sources import (
    SUPPORTED_ISSUE_SOURCES,
    build_issue_source,
    parse_issue_ref,
    story_from_prompt,
)
from .languages import resolve_language
from .llm import SUPPORTED_PROVIDERS as SUPPORTED_LLM_PROVIDERS, build_client_from_credentials
from .memory import KNOWN_MEMORY_FILES, TeamMemory
from .pipeline import build_story
from .repo import PyGithubClient
from .review import review_batch
from .transcribe import SUPPORTED_BACKENDS as SUPPORTED_TRANSCRIBE_BACKENDS, transcribe_file
from .user_config import (
    IssueSourceConfig,
    LLMProviderConfig,
    UserConfig,
    VCSProviderConfig,
    config_path,
    load_user_config,
    mask_secret,
    resolve_issue_credentials,
    resolve_llm_credentials,
    resolve_vcs_credentials,
    save_user_config,
)
from .vcs import SUPPORTED_VCS_PROVIDERS


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
@click.option(
    "--no-seed",
    is_flag=True,
    help="Use bare templates instead of language-aware seeded content.",
)
def init(force: bool, no_seed: bool) -> None:
    """Scaffold cascade.yaml and team-memory/ in the current directory.

    By default, team-memory files are seeded with sensible defaults based
    on the detected language and any existing ADR documentation in the
    repo. Use --no-seed for bare templates instead.
    """
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

    if no_seed:
        seeded = None
    else:
        try:
            seeded = seed_team_memory(root)
        except Exception as exc:
            click.echo(f"  note: smart seeding failed ({exc}); using bare templates")
            seeded = None

    for name in KNOWN_MEMORY_FILES:
        p = mem_dir / name
        if p.exists() and not force:
            click.echo(f"  exists  team-memory/{name} (use --force to overwrite)")
            continue
        content = seeded[name] if seeded else _starter_content_for(name)
        p.write_text(content)
        annotation = " (seeded)" if seeded else ""
        click.echo(f"  wrote   team-memory/{name}{annotation}")

    click.echo()
    click.echo("Cascade initialized. Next steps:")
    click.echo("  1. Run `cascade doctor` to verify your setup.")
    click.echo("  2. Edit team-memory/*.md with your team's real conventions.")
    click.echo("  3. Configure an LLM: `cascade configure llm anthropic --key ...`")
    click.echo("  4. Run `cascade try` to verify the full pipeline works end-to-end.")


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
        user_cfg = load_user_config()
        transcript = read_transcript(transcript_path)
        memory = TeamMemory.load(Path.cwd() / config.memory.path)
        llm_creds = resolve_llm_credentials(
            user_config=user_cfg,
            provider=config.agent.provider,
            model_override=model or config.agent.model,
        )
        llm = build_client_from_credentials(llm_creds)

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
            f"  cost: {format_cost(result.usage.estimated_cost_usd)} "
            f"({result.usage.input_tokens:,} in / {result.usage.output_tokens:,} out tokens, "
            f"{result.usage.provider}/{result.usage.model})"
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
    # Hand off to the interactive review loop
    try:
        review_batch(batch_path)
    except CascadeError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)


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
@click.option(
    "--language",
    default=None,
    help="Override the language profile (e.g. python, typescript, go, rust). "
    "Defaults to auto-detection or cascade.yaml.",
)
@click.option(
    "--base-branch",
    default="main",
    show_default=True,
    help="Branch the new work branches off.",
)
@click.option(
    "--no-pr",
    is_flag=True,
    help="Generate code, run tests, and commit locally -- but do NOT push "
    "or open a PR. Useful for dry runs.",
)
@click.option(
    "--repo-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Path to the target repo (where code is generated). Defaults to "
    "the current working directory.",
)
@click.option(
    "--max-cost",
    type=float,
    default=None,
    help="Maximum cumulative LLM cost (USD) before aborting between stories. "
    "Cascade checks the running total after each story; if it exceeds this "
    "threshold, remaining stories are skipped. No effect on free providers "
    "(claude_code, ollama).",
)
def build(
    batch_path: Path,
    story_index: int | None,
    language: str | None,
    base_branch: str,
    no_pr: bool,
    repo_root: Path | None,
    max_cost: float | None,
) -> None:
    """Build code and tests for approved stories, then open PRs.

    Runs the full pipeline per approved story: plan -> code -> apply -> test
    -> commit -> push -> PR. Stops on first failure with an actionable error.
    """
    try:
        batch = read_story_batch(batch_path)
    except CascadeError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)

    approved = batch.approved()
    if not approved:
        click.echo("No approved stories in this batch. Run `cascade review` first.")
        sys.exit(0)

    if story_index is not None:
        if story_index < 1 or story_index > len(approved):
            click.echo(
                f"error: --story {story_index} out of range "
                f"(1..{len(approved)})",
                err=True,
            )
            sys.exit(1)
        stories_to_build = [approved[story_index - 1]]
    else:
        stories_to_build = approved

    target_root = (repo_root or Path.cwd()).resolve()

    try:
        config = load_config()
        user_cfg = load_user_config()
        language_profile = resolve_language(target_root, configured_name=language or config.language)
        memory = TeamMemory.load(target_root / config.memory.path)
        llm_creds = resolve_llm_credentials(
            user_config=user_cfg,
            provider=config.agent.provider,
            model_override=config.agent.model,
        )
        llm = build_client_from_credentials(llm_creds)

        github_client = None
        if not no_pr:
            vcs_creds = resolve_vcs_credentials(user_config=user_cfg, provider="github")
            github_client = PyGithubClient(token=vcs_creds.token)
    except CascadeError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)

    click.echo(
        f"  language: {language_profile.display_name}  "
        f"({len(memory.non_empty_files)} memory files loaded)"
    )
    click.echo(f"  stories to build: {len(stories_to_build)}")
    click.echo()

    cumulative_cost = 0.0
    cumulative_calls = 0
    cumulative_in = 0
    cumulative_out = 0
    stories_built = 0
    aborted_for_cost = False

    for s in stories_to_build:
        if max_cost is not None and cumulative_cost >= max_cost:
            click.echo(
                f"  --max-cost {format_cost(max_cost)} reached. "
                f"Skipping remaining {len(stories_to_build) - stories_built} story/ies.",
                err=True,
            )
            aborted_for_cost = True
            break

        click.echo(f"==> [{s.id}] {s.title}")
        try:
            result = build_story(
                story=s,
                repo_root=target_root,
                llm=llm,
                language=language_profile,
                memory=memory,
                github_client=github_client,
                base_branch=base_branch,
                test_override_command=(
                    config.test_command.split() if config.test_command else None
                ),
                push_and_open_pr=not no_pr,
            )
        except CascadeError as exc:
            click.echo(f"  failed: {exc}", err=True)
            sys.exit(1)

        _print_build_result(result, no_pr=no_pr)
        cumulative_cost += result.total_llm_cost_usd
        cumulative_in += result.total_input_tokens
        cumulative_out += result.total_output_tokens
        cumulative_calls += 2  # plan + code per story
        stories_built += 1

    # Session summary
    if stories_built > 1 or (stories_built >= 1 and max_cost is not None):
        click.echo("=" * 60)
        click.echo(
            f"  session: {stories_built} stor{'y' if stories_built == 1 else 'ies'} built, "
            f"{cumulative_calls} LLM calls, {format_cost(cumulative_cost)}"
        )
        click.echo("=" * 60)

    if aborted_for_cost:
        sys.exit(2)


def _print_build_result(result, *, no_pr: bool) -> None:
    """Print a per-story build summary."""
    click.echo(f"  branch:  {result.branch}")
    click.echo(f"  commit:  {result.commit_sha}")
    click.echo(
        f"  install: {'ok' if result.install_result.passed else 'failed'}"
    )
    click.echo(
        f"  tests:   {'passed' if result.test_result.passed else 'FAILED'} "
        f"({result.test_result.summary})"
    )
    if no_pr or result.pull_request is None:
        click.echo("  PR:      (skipped --no-pr)")
    else:
        click.echo(f"  PR:      #{result.pull_request.number}  {result.pull_request.url}")
    click.echo(
        f"  cost:    {format_cost(result.total_llm_cost_usd)} "
        f"({result.total_input_tokens:,} in / {result.total_output_tokens:,} out tokens)"
    )
    click.echo()


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


# ----------------------------------------------------------------------------
# ingest
# ----------------------------------------------------------------------------


@cli.command()
@click.argument(
    "source",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write the transcript YAML. Defaults to "
    "transcripts/{meeting_id}.yaml.",
)
@click.option(
    "--backend",
    type=click.Choice(list(SUPPORTED_TRANSCRIBE_BACKENDS), case_sensitive=False),
    default="auto",
    show_default=True,
    help="Transcription backend.",
)
@click.option(
    "--model",
    default="base",
    show_default=True,
    help="Whisper model size for local backends (tiny/base/small/medium/large).",
)
@click.option(
    "--language",
    default=None,
    help="ISO 639-1 language code (e.g. 'en') to force, or omit for auto-detect.",
)
@click.option(
    "--no-diarization",
    is_flag=True,
    help="Skip speaker diarization (faster, single 'Speaker' label).",
)
@click.option(
    "--meeting-id",
    default=None,
    help="Override the meeting ID. Defaults to a slug derived from the file name.",
)
def ingest(
    source: Path,
    output: Path | None,
    backend: str,
    model: str,
    language: str | None,
    no_diarization: bool,
    meeting_id: str | None,
) -> None:
    """Transcribe audio/video into a structured transcript YAML.

    Example: cascade ingest recordings/standup.mp3
    """
    try:
        user_cfg = load_user_config()
        # If using the openai-api backend, pull the OpenAI key from the
        # standard credentials resolver.
        openai_key = None
        if backend == "openai-api":
            llm_creds = resolve_llm_credentials(
                user_config=user_cfg, provider="openai"
            )
            openai_key = llm_creds.api_key

        click.echo(f"  transcribing {source}  (backend={backend}, model={model})")
        result = transcribe_file(
            source,
            backend=backend,
            model=model,
            language=language,
            enable_diarization=not no_diarization,
            openai_api_key=openai_key,
            meeting_id=meeting_id,
        )

        out_path = output or (
            Path.cwd() / "transcripts" / f"{result.transcript.meeting_id}.yaml"
        )
        from .io import write_transcript

        write_transcript(result.transcript, out_path)

        click.echo(
            f"  transcribed {len(result.transcript.turns)} turns "
            f"({len(result.transcript.speakers)} speakers, "
            f"{result.transcript.duration_seconds:.1f}s) -> {out_path}"
        )
        if not result.diarization_used and not no_diarization:
            click.echo("  (diarization unavailable; all turns labeled 'Speaker')")
    except CascadeError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)


# ----------------------------------------------------------------------------
# configure
# ----------------------------------------------------------------------------


@cli.group()
def configure() -> None:
    """Set up credentials and defaults at ~/.config/cascade/config.yaml."""


@configure.command("show")
def configure_show() -> None:
    """Show the effective user config (with secrets masked)."""
    cfg = load_user_config()
    click.echo(f"Config file: {config_path()}")
    click.echo(f"Defaults:")
    click.echo(f"  llm_provider:   {cfg.defaults.llm_provider}")
    click.echo(f"  vcs_provider:   {cfg.defaults.vcs_provider}")
    click.echo(f"  issue_provider: {cfg.defaults.issue_provider or '(none)'}")
    click.echo()
    if cfg.llm_providers:
        click.echo("LLM providers:")
        for name, p in cfg.llm_providers.items():
            click.echo(
                f"  {name}: key={mask_secret(p.api_key)} "
                f"model={p.default_model or '(default)'} "
                f"base_url={p.base_url or '(default)'}"
            )
        click.echo()
    if cfg.vcs_providers:
        click.echo("VCS providers:")
        for name, p in cfg.vcs_providers.items():
            click.echo(
                f"  {name}: token={mask_secret(p.token)} "
                f"base_url={p.base_url or '(default)'} "
                f"org={p.organization or '(n/a)'}"
            )
        click.echo()
    if cfg.issue_sources:
        click.echo("Issue sources:")
        for name, p in cfg.issue_sources.items():
            click.echo(
                f"  {name}: token={mask_secret(p.token)} "
                f"base_url={p.base_url or '(default)'} "
                f"user={p.user or '(n/a)'}"
            )


@configure.command("llm")
@click.argument(
    "provider",
    type=click.Choice(list(SUPPORTED_LLM_PROVIDERS), case_sensitive=False),
)
@click.option("--key", default=None, help="API key for this provider.")
@click.option("--model", default=None, help="Default model identifier.")
@click.option("--base-url", default=None, help="Override the API base URL.")
@click.option("--set-default", is_flag=True, help="Make this the default LLM provider.")
def configure_llm(
    provider: str, key: str | None, model: str | None, base_url: str | None, set_default: bool
) -> None:
    """Configure an LLM provider (e.g. `cascade configure llm openai --key sk-...`)."""
    cfg = load_user_config()
    existing = cfg.llm_providers.get(provider.lower(), LLMProviderConfig())
    updated = LLMProviderConfig(
        api_key=key if key is not None else existing.api_key,
        default_model=model if model is not None else existing.default_model,
        base_url=base_url if base_url is not None else existing.base_url,
    )
    new_providers = dict(cfg.llm_providers)
    new_providers[provider.lower()] = updated
    new_defaults = cfg.defaults.model_copy(
        update={"llm_provider": provider.lower()} if set_default else {}
    )
    new_cfg = cfg.model_copy(update={"llm_providers": new_providers, "defaults": new_defaults})
    save_user_config(new_cfg)
    click.echo(f"Updated LLM provider '{provider}'.")
    if set_default:
        click.echo(f"Set '{provider}' as the default LLM provider.")


@configure.command("vcs")
@click.argument(
    "provider",
    type=click.Choice(list(SUPPORTED_VCS_PROVIDERS), case_sensitive=False),
)
@click.option("--token", default=None, help="Access token.")
@click.option("--base-url", default=None, help="Override the API base URL.")
@click.option("--organization", default=None, help="Organization name (Azure DevOps).")
@click.option("--set-default", is_flag=True, help="Make this the default VCS provider.")
def configure_vcs(
    provider: str,
    token: str | None,
    base_url: str | None,
    organization: str | None,
    set_default: bool,
) -> None:
    """Configure a VCS provider (e.g. `cascade configure vcs gitlab --token ...`)."""
    cfg = load_user_config()
    existing = cfg.vcs_providers.get(provider.lower(), VCSProviderConfig())
    updated = VCSProviderConfig(
        token=token if token is not None else existing.token,
        base_url=base_url if base_url is not None else existing.base_url,
        organization=organization if organization is not None else existing.organization,
    )
    new_providers = dict(cfg.vcs_providers)
    new_providers[provider.lower()] = updated
    new_defaults = cfg.defaults.model_copy(
        update={"vcs_provider": provider.lower()} if set_default else {}
    )
    new_cfg = cfg.model_copy(update={"vcs_providers": new_providers, "defaults": new_defaults})
    save_user_config(new_cfg)
    click.echo(f"Updated VCS provider '{provider}'.")
    if set_default:
        click.echo(f"Set '{provider}' as the default VCS provider.")


@configure.command("issue")
@click.argument(
    "provider",
    type=click.Choice(list(SUPPORTED_ISSUE_SOURCES), case_sensitive=False),
)
@click.option("--token", default=None, help="Access token.")
@click.option("--base-url", default=None, help="API base URL (required for Jira).")
@click.option("--user", default=None, help="Username/email (required for Jira).")
@click.option("--set-default", is_flag=True, help="Make this the default issue source.")
def configure_issue(
    provider: str,
    token: str | None,
    base_url: str | None,
    user: str | None,
    set_default: bool,
) -> None:
    """Configure an issue tracker (e.g. `cascade configure issue jira --url ... --user ... --token ...`)."""
    cfg = load_user_config()
    existing = cfg.issue_sources.get(provider.lower(), IssueSourceConfig())
    updated = IssueSourceConfig(
        token=token if token is not None else existing.token,
        base_url=base_url if base_url is not None else existing.base_url,
        user=user if user is not None else existing.user,
    )
    new_sources = dict(cfg.issue_sources)
    new_sources[provider.lower()] = updated
    new_defaults = cfg.defaults.model_copy(
        update={"issue_provider": provider.lower()} if set_default else {}
    )
    new_cfg = cfg.model_copy(update={"issue_sources": new_sources, "defaults": new_defaults})
    save_user_config(new_cfg)
    click.echo(f"Updated issue source '{provider}'.")
    if set_default:
        click.echo(f"Set '{provider}' as the default issue source.")


# ----------------------------------------------------------------------------
# prompt -- direct-prompt entry point
# ----------------------------------------------------------------------------


@cli.command()
@click.argument("text")
@click.option(
    "--language",
    default=None,
    help="Override the language profile.",
)
@click.option(
    "--base-branch",
    default="main",
    show_default=True,
    help="Branch the new work branches off.",
)
@click.option("--no-pr", is_flag=True, help="Skip push + PR opening.")
@click.option(
    "--repo-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
)
def prompt(
    text: str,
    language: str | None,
    base_branch: str,
    no_pr: bool,
    repo_root: Path | None,
) -> None:
    """Build directly from an ad-hoc text prompt (no meeting, no ticket).

    Example: cascade prompt "Add cursor-based pagination to /api/users"
    """
    target_root = (repo_root or Path.cwd()).resolve()
    try:
        story = story_from_prompt(text)
        config = load_config()
        user_cfg = load_user_config()
        language_profile = resolve_language(
            target_root, configured_name=language or config.language
        )
        memory = TeamMemory.load(target_root / config.memory.path)
        llm_creds = resolve_llm_credentials(
            user_config=user_cfg,
            provider=config.agent.provider,
            model_override=config.agent.model,
        )
        llm = build_client_from_credentials(llm_creds)

        github_client = None
        if not no_pr:
            vcs_creds = resolve_vcs_credentials(user_config=user_cfg, provider="github")
            github_client = PyGithubClient(token=vcs_creds.token)

        click.echo(f"==> [{story.id}] {story.title}")
        result = build_story(
            story=story,
            repo_root=target_root,
            llm=llm,
            language=language_profile,
            memory=memory,
            github_client=github_client,
            base_branch=base_branch,
            test_override_command=(
                config.test_command.split() if config.test_command else None
            ),
            push_and_open_pr=not no_pr,
        )
        _print_build_result(result, no_pr=no_pr)
    except CascadeError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)


# ----------------------------------------------------------------------------
# ticket -- issue-tracker entry point
# ----------------------------------------------------------------------------


@cli.command()
@click.argument("identifier")
@click.option(
    "--language",
    default=None,
    help="Override the language profile.",
)
@click.option(
    "--base-branch",
    default="main",
    show_default=True,
)
@click.option("--no-pr", is_flag=True, help="Skip push + PR opening.")
@click.option(
    "--repo-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
)
def ticket(
    identifier: str,
    language: str | None,
    base_branch: str,
    no_pr: bool,
    repo_root: Path | None,
) -> None:
    """Build from a tracker ticket: `cascade ticket <provider>:<id>`.

    Examples:
        cascade ticket github:myorg/myrepo#42
        cascade ticket jira:PROJ-123
        cascade ticket linear:ENG-456
        cascade ticket azure_devops:org/proj/123
    """
    target_root = (repo_root or Path.cwd()).resolve()
    try:
        ref = parse_issue_ref(identifier)
        user_cfg = load_user_config()
        issue_creds = resolve_issue_credentials(
            user_config=user_cfg, provider=ref.provider
        )
        source = build_issue_source(
            provider=issue_creds.provider,
            token=issue_creds.token,
            base_url=issue_creds.base_url,
            user=issue_creds.user,
        )
        story = source.fetch_story(ref.identifier)

        config = load_config()
        language_profile = resolve_language(
            target_root, configured_name=language or config.language
        )
        memory = TeamMemory.load(target_root / config.memory.path)
        llm_creds = resolve_llm_credentials(
            user_config=user_cfg,
            provider=config.agent.provider,
            model_override=config.agent.model,
        )
        llm = build_client_from_credentials(llm_creds)

        github_client = None
        if not no_pr:
            vcs_creds = resolve_vcs_credentials(user_config=user_cfg, provider="github")
            github_client = PyGithubClient(token=vcs_creds.token)

        click.echo(f"==> [{story.id}] {story.title}  (from {identifier})")
        result = build_story(
            story=story,
            repo_root=target_root,
            llm=llm,
            language=language_profile,
            memory=memory,
            github_client=github_client,
            base_branch=base_branch,
            test_override_command=(
                config.test_command.split() if config.test_command else None
            ),
            push_and_open_pr=not no_pr,
        )
        _print_build_result(result, no_pr=no_pr)
    except CascadeError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)


# ----------------------------------------------------------------------------
# doctor -- health check
# ----------------------------------------------------------------------------


_STATUS_ICONS = {
    CheckStatus.OK: ("✓", "green"),
    CheckStatus.WARN: ("!", "yellow"),
    CheckStatus.FAIL: ("✗", "red"),
    CheckStatus.SKIP: ("-", None),
}


@cli.command()
@click.option(
    "--repo-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Repo to check. Defaults to the current directory.",
)
def doctor(repo_root: Path | None) -> None:
    """Health-check the Cascade installation and report status of every component.

    Walks through a series of checks (Python version, optional extras, git,
    cascade.yaml, team memory, LLM and VCS credentials, test runner) and
    reports each as OK / WARN / FAIL with actionable suggestions for any
    issues found.
    """
    target = (repo_root or Path.cwd()).resolve()
    results = run_doctor(target)

    click.echo()
    click.echo(f"Cascade {__version__} on {target}")
    click.echo("=" * 60)
    click.echo()

    for r in results:
        icon, color = _STATUS_ICONS[r.status]
        prefix = click.style(f"[{icon}]", fg=color) if color else f"[{icon}]"
        click.echo(f"  {prefix}  {r.name:24s}  {r.message}")
        if r.suggestion and r.status != CheckStatus.OK:
            click.echo(f"        {click.style('hint:', fg='cyan')} {r.suggestion}")

    ok, warn, fail, skip = summarize(results)
    click.echo()
    click.echo("=" * 60)
    parts = []
    if ok:
        parts.append(click.style(f"{ok} ok", fg="green"))
    if warn:
        parts.append(click.style(f"{warn} warning", fg="yellow"))
    if fail:
        parts.append(click.style(f"{fail} failed", fg="red"))
    if skip:
        parts.append(f"{skip} skipped")
    click.echo("  " + ", ".join(parts))
    click.echo()

    if fail:
        click.echo(click.style("  One or more checks failed.", fg="red"))
        click.echo("  Fix the issues above before running cascade build.")
        sys.exit(1)
    elif warn:
        click.echo(click.style("  Some checks reported warnings.", fg="yellow"))
        click.echo("  You can run cascade build, but quality may suffer.")
        sys.exit(0)
    else:
        click.echo(click.style("  All checks passed. You're ready to go.", fg="green"))
        click.echo("  Try: cascade try   (a risk-free end-to-end pipeline test)")
        sys.exit(0)


# ----------------------------------------------------------------------------
# try -- risk-free end-to-end verification
# ----------------------------------------------------------------------------


@cli.command(name="try")
@click.option(
    "--keep-workspace",
    is_flag=True,
    help="Don't delete the temp directory after the run. Useful for inspecting "
    "what Cascade generated.",
)
def try_command(keep_workspace: bool) -> None:
    """Run a built-in toy story end-to-end to verify the pipeline works.

    Creates a disposable Python project in a temp directory, asks Cascade
    to add a tiny hello() function (and a test), runs the test, and
    reports the result. Nothing is pushed; nothing touches your real repo.

    This is the recommended thing to run right after `cascade configure`
    to know that your setup will work before you bet on it.
    """
    try:
        user_cfg = load_user_config()
        cfg = load_config()
        llm_creds = resolve_llm_credentials(
            user_config=user_cfg,
            provider=cfg.agent.provider,
            model_override=cfg.agent.model,
        )
        llm = build_client_from_credentials(llm_creds)
    except CascadeError as exc:
        click.echo(f"error: {exc}", err=True)
        click.echo("  hint: run `cascade doctor` to find what's missing", err=True)
        sys.exit(1)

    click.echo()
    click.echo(f"Running cascade try with {llm.provider_name} / {llm.model}")
    click.echo("-" * 60)

    def _on_stage(stage: str, message: str) -> None:
        click.echo(f"  {click.style(stage:=stage[:8].ljust(8), fg='cyan')}  {message}")

    try:
        result = run_demo(
            llm=llm,
            keep_workspace=keep_workspace,
            on_stage=_on_stage,
        )
    except CascadeError as exc:
        click.echo()
        click.echo(click.style(f"  FAILED: {exc}", fg="red"), err=True)
        click.echo("  hint: run `cascade doctor` to debug your setup", err=True)
        sys.exit(1)

    click.echo()
    click.echo("-" * 60)
    if result.success:
        click.echo(click.style("  cascade try PASSED.", fg="green"))
        click.echo()
        click.echo(f"  Plan: {result.plan.summary}")
        click.echo(f"  Files generated: {len(result.code_change.files)}")
        click.echo(f"  Tests: {result.test_result.summary}")
        click.echo(
            f"  Cost: {format_cost(result.total_llm_cost_usd)} "
            f"({result.total_input_tokens:,} in / {result.total_output_tokens:,} out tokens)"
        )
        click.echo()
        click.echo("  Your installation is working end-to-end.")
        click.echo("  Try a real one: cascade prompt \"Add a /health endpoint\"")
    else:
        click.echo(click.style("  cascade try FAILED at the test stage.", fg="red"))
        click.echo()
        click.echo(f"  Plan: {result.plan.summary}")
        click.echo(f"  Files generated: {len(result.code_change.files)}")
        click.echo(f"  Test summary: {result.test_result.summary}")
        click.echo()
        click.echo("  The LLM produced code but it didn't pass the test.")
        click.echo("  This often means the LLM you chose isn't producing")
        click.echo("  high-enough-quality output for Cascade's expectations.")
        click.echo("  Try a different model or provider.")
        if keep_workspace:
            click.echo()
            click.echo(f"  Workspace preserved at: {result.workspace}")
        sys.exit(1)


# ----------------------------------------------------------------------------
# ui -- launch Cascade Studio (web dashboard)
# ----------------------------------------------------------------------------


@cli.command()
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Host to bind. Use 0.0.0.0 to allow connections from other machines.",
)
@click.option(
    "--port",
    default=8000,
    show_default=True,
    type=int,
    help="Port to bind.",
)
@click.option(
    "--no-browser",
    is_flag=True,
    help="Don't open a browser window after the server starts.",
)
def ui(host: str, port: int, no_browser: bool) -> None:
    """Launch Cascade Studio (the web dashboard).

    Requires the [studio] extra:
        pip install cascade-agent[studio]

    The dashboard runs locally at http://HOST:PORT and auto-opens in
    your default browser unless --no-browser is set.
    """
    try:
        from .studio.server import create_app
    except CascadeError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)
    except ImportError as exc:
        click.echo(
            "error: Cascade Studio dependencies are not installed.\n"
            "  pip install cascade-agent[studio]",
            err=True,
        )
        click.echo(f"  (underlying: {exc})", err=True)
        sys.exit(1)

    import threading
    import time
    import webbrowser

    try:
        import uvicorn  # type: ignore
    except ImportError:
        click.echo(
            "error: uvicorn not installed.  pip install cascade-agent[studio]",
            err=True,
        )
        sys.exit(1)

    url = f"http://{host}:{port}"
    click.echo(f"  Cascade Studio starting at {url}")
    click.echo("  Press Ctrl+C to stop.")
    click.echo()

    if not no_browser:
        # Open the browser shortly after the server starts. Best effort;
        # if the browser can't be opened (headless env), just continue.
        def _open():
            time.sleep(1.0)
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=_open, daemon=True).start()

    # Build the app instance and hand it to uvicorn. We pass the factory
    # via dotted path so uvicorn handles its own lifecycle cleanly.
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


def main() -> None:
    """Entry point referenced from pyproject.toml."""
    cli(obj={})


if __name__ == "__main__":
    main()
