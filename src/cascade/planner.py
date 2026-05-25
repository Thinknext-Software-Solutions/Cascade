"""Planner stage: approved Story -> Plan.

Reads the story, team memory, language profile, and repo summary, then
asks the LLM to produce a file-level implementation plan. The Plan is
language-aware and respects the team's accumulated knowledge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from . import __version__
from .exceptions import CascadeError, CascadeLLMError
from .languages import LanguageProfile
from .llm import LLMClient, LLMUsage
from .memory import TeamMemory
from .plan_schemas import Plan
from .repo_scan import RepoSummary
from .schemas import Story


logger = logging.getLogger(__name__)


PLANNER_SYSTEM_PROMPT = """You are a senior software architect designing the
smallest correct implementation for one user story.

Core principles:
1. Honor the team's existing conventions, decisions, and constraints. The
   team memory is authoritative -- never propose a path the team has
   already rejected.
2. Match the language's idioms exactly (see the language guidance below).
3. Touch the minimum number of files necessary. Surgical changes win.
4. For each file change, write the INTENT in plain English -- NOT the code.
   The coder stage produces the actual code from the intent.
5. Flag risks honestly (security, data migration, breaking changes,
   touching well-trafficked code paths).
6. Document deliberate out-of-scope choices to prevent scope creep.
7. Order the file plans by dependency: if file A imports file B, list B first.

Output discipline:
- Use the provided tool to return a structured Plan.
- File paths are repo-relative, using forward slashes.
- Actions are exactly one of: create, modify, delete.
- Plans with zero file changes are invalid -- if there's truly nothing to
  do, the story should have been rejected at review time, not approved.
"""


@dataclass(frozen=True)
class PlanResult:
    plan: Plan
    usage: LLMUsage


def build_planner_user_prompt(
    *,
    story: Story,
    language: LanguageProfile,
    team_memory_context: str,
    repo_summary_text: str,
) -> str:
    """Compose the user-side prompt for the planner."""
    parts: list[str] = []

    if team_memory_context.strip():
        parts.append(team_memory_context)
        parts.append("\n---\n\n")

    parts.append(f"# Language: {language.display_name}\n\n")
    if language.notes_for_llm:
        parts.append(f"Language guidance:\n{language.notes_for_llm}\n\n")
    parts.append(f"Source location convention: {language.source_dir_default}/\n")
    parts.append(f"Test location convention: {language.test_dir_default}/\n")
    parts.append(f"Test file naming: {language.test_file_glob}\n\n---\n\n")

    if repo_summary_text.strip():
        parts.append(repo_summary_text)
        parts.append("\n\n---\n\n")

    parts.append("# Story to implement\n\n")
    parts.append(story.as_text())
    parts.append("\n\n---\n\n")

    parts.append(
        f"Produce the Plan to implement this story. Set story_id to "
        f"`{story.id}`. Order file plans by dependency. Be specific about "
        "intents; the next stage produces code from your intents."
    )

    return "".join(parts)


def plan_story(
    *,
    story: Story,
    llm: LLMClient,
    language: LanguageProfile,
    memory: Optional[TeamMemory] = None,
    repo_summary: Optional[RepoSummary] = None,
    memory_char_budget: int = 20_000,
    max_output_tokens: int = 8192,
    temperature: float = 0.2,
) -> PlanResult:
    """Generate an implementation Plan for an approved story.

    Args:
        story: The approved story.
        llm: LLM client.
        language: Resolved language profile for the repo.
        memory: Optional TeamMemory; if provided, included as grounding context.
        repo_summary: Optional RepoSummary; if provided, included to ground
            the planner in the actual repo structure.

    Returns:
        PlanResult with the validated Plan and token usage.

    Raises:
        CascadeError: If the LLM fails or returns a plan that violates
            higher-level invariants (e.g. story_id mismatch we can't fix,
            zero files after Pydantic validation -- which shouldn't happen
            but we double-check).
    """
    memory_context = (
        memory.as_llm_context(max_chars=memory_char_budget)
        if memory is not None
        else ""
    )
    repo_text = repo_summary.as_text() if repo_summary is not None else ""

    user_prompt = build_planner_user_prompt(
        story=story,
        language=language,
        team_memory_context=memory_context,
        repo_summary_text=repo_text,
    )

    logger.info(
        "planner.start",
        extra={
            "story_id": story.id,
            "language": language.name,
            "memory_chars": len(memory_context),
            "repo_summary_chars": len(repo_text),
            "model": llm.model,
        },
    )

    try:
        response = llm.structured_call(
            system=PLANNER_SYSTEM_PROMPT,
            user=user_prompt,
            schema=Plan,
            max_tokens=max_output_tokens,
            temperature=temperature,
        )
    except CascadeLLMError as exc:
        raise CascadeError(f"Planner LLM call failed for story {story.id}: {exc}") from exc

    plan = response.parsed
    if plan.story_id != story.id:
        logger.warning(
            "planner.story_id_mismatch",
            extra={"asked": story.id, "returned": plan.story_id},
        )
        plan = plan.model_copy(update={"story_id": story.id})

    logger.info(
        "planner.done",
        extra={
            "story_id": story.id,
            "file_count": len(plan.files),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    )

    return PlanResult(plan=plan, usage=response.usage)
