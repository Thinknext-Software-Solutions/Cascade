"""Coder stage: Plan -> CodeChange.

Reads the plan, story, team memory, language profile, and (when modifying
existing files) the current file contents, then asks the LLM to produce
the full file contents for every planned change.

Why full-file generation rather than diffs in v0.1:
- More reliable: diffs require the LLM to match existing lines exactly,
  which it gets wrong frequently
- Easier to apply: we just overwrite the file (or create it)
- Easier to review: the human sees the final state, not patch hunks
- Trade-off: more tokens. Acceptable for files under ~2000 lines.

For files larger than the in-prompt budget we either skip them (and the
planner is wrong to have included them in a single plan) or chunk -- v0.2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .exceptions import CascadeError, CascadeLLMError
from .languages import LanguageProfile
from .llm import LLMClient, LLMUsage
from .memory import TeamMemory
from .plan_schemas import CodeChange, FileAction, Plan
from .schemas import Story


logger = logging.getLogger(__name__)


CODER_SYSTEM_PROMPT = """You are a senior software engineer implementing
one well-scoped plan.

Core principles:
1. Match the team's conventions exactly (see team memory and language notes).
2. Write the COMPLETE file contents for every create/modify action. Do not
   produce diffs, fragments, or placeholders like "... rest of the file
   unchanged ..." -- the system overwrites the entire file with your output.
3. For 'modify' actions, the CURRENT contents are provided. Preserve all
   unrelated code; only change what the plan requires.
4. Implement tests as part of the same change. Test files are listed in the
   plan; produce them with real assertions, not placeholders.
5. Idiomatic code in the target language. Type hints / type annotations
   where the language supports them.
6. No commented-out code, no TODOs, no dead branches. The PR description
   captures known limitations; the code itself ships clean.
7. Explain WHY each file change matters in the `reason` field. This goes
   into the PR description.

Output discipline:
- Use the provided tool to return a structured CodeChange.
- One FileChange per file in the plan; same order.
- For 'delete' actions, content must be null/omitted.
- For 'create' and 'modify', content is the full file body as a string.
"""


@dataclass(frozen=True)
class CodeResult:
    change: CodeChange
    usage: LLMUsage


def build_coder_user_prompt(
    *,
    story: Story,
    plan: Plan,
    language: LanguageProfile,
    team_memory_context: str,
    current_file_contents: dict[str, str],
) -> str:
    """Compose the user-side prompt for the coder.

    Args:
        story: The story being implemented.
        plan: The planner's output.
        language: Resolved language profile.
        team_memory_context: Pre-formatted memory excerpts.
        current_file_contents: Map of repo-relative path -> current file body
            for every 'modify' or 'delete' action in the plan. Missing
            entries for create actions are normal.
    """
    parts: list[str] = []

    if team_memory_context.strip():
        parts.append(team_memory_context)
        parts.append("\n---\n\n")

    parts.append(f"# Language: {language.display_name}\n\n")
    if language.notes_for_llm:
        parts.append(f"Language guidance:\n{language.notes_for_llm}\n\n")
    parts.append("---\n\n")

    parts.append("# Story\n\n")
    parts.append(story.as_text())
    parts.append("\n\n---\n\n")

    parts.append("# Plan\n\n")
    parts.append(f"Summary: {plan.summary}\n\n")
    if plan.risks:
        parts.append("Risks:\n")
        for r in plan.risks:
            parts.append(f"- {r}\n")
        parts.append("\n")
    if plan.out_of_scope:
        parts.append("Out of scope:\n")
        for o in plan.out_of_scope:
            parts.append(f"- {o}\n")
        parts.append("\n")
    parts.append("Files to change:\n")
    for fp in plan.files:
        parts.append(f"  - [{fp.action.value}] {fp.path}\n")
        parts.append(f"      Intent: {fp.intent}\n")
        if fp.references:
            parts.append(f"      References: {', '.join(fp.references)}\n")
    parts.append("\n---\n\n")

    if current_file_contents:
        parts.append("# Current file contents (for modify/delete actions)\n\n")
        for path, content in current_file_contents.items():
            parts.append(f"## {path}\n\n")
            parts.append("```\n")
            parts.append(content)
            parts.append("\n```\n\n")
        parts.append("---\n\n")

    parts.append(
        f"Produce the CodeChange to implement the plan. Set story_id to "
        f"`{story.id}` and plan_summary to the summary above. Generate the "
        "FULL contents for each create/modify file. For delete actions, "
        "omit content. Match the team's conventions and the language's idioms."
    )

    return "".join(parts)


def read_existing_files(
    repo_root: Path,
    plan: Plan,
    *,
    max_bytes_per_file: int = 200_000,
) -> dict[str, str]:
    """Read current contents for any 'modify' or 'delete' files in the plan.

    Files that don't exist (because the planner over-claimed them as
    'modify' when they should have been 'create') are silently skipped.
    Files larger than max_bytes_per_file are skipped and logged.
    """
    out: dict[str, str] = {}
    for fp in plan.files:
        if fp.action not in (FileAction.MODIFY, FileAction.DELETE):
            continue
        path = repo_root / fp.path
        if not path.exists():
            logger.warning(
                "coder.modify_target_missing",
                extra={"path": fp.path, "action": fp.action.value},
            )
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > max_bytes_per_file:
            logger.warning(
                "coder.file_too_large",
                extra={"path": fp.path, "size": size, "limit": max_bytes_per_file},
            )
            continue
        try:
            out[fp.path] = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "coder.read_failed", extra={"path": fp.path, "error": str(exc)}
            )
    return out


def generate_code(
    *,
    story: Story,
    plan: Plan,
    llm: LLMClient,
    language: LanguageProfile,
    repo_root: Path,
    memory: Optional[TeamMemory] = None,
    memory_char_budget: int = 15_000,
    max_output_tokens: int = 16_384,
    temperature: float = 0.2,
) -> CodeResult:
    """Generate a CodeChange that implements a Plan.

    Args:
        story: The story being implemented.
        plan: The planner's output.
        llm: LLM client.
        language: Resolved language profile.
        repo_root: Repository root for reading existing-file contents.
        memory: Optional team memory.

    Returns:
        CodeResult with the validated CodeChange and token usage.

    Raises:
        CascadeError: If the LLM fails, or the resulting CodeChange disagrees
            with the plan in a way we can't reconcile (e.g. files in the
            code change that weren't in the plan).
    """
    if not plan.files:
        raise CascadeError(
            f"Cannot generate code for story {story.id}: plan has zero files"
        )

    current_contents = read_existing_files(repo_root, plan)
    memory_context = (
        memory.as_llm_context(max_chars=memory_char_budget)
        if memory is not None
        else ""
    )
    user_prompt = build_coder_user_prompt(
        story=story,
        plan=plan,
        language=language,
        team_memory_context=memory_context,
        current_file_contents=current_contents,
    )

    logger.info(
        "coder.start",
        extra={
            "story_id": story.id,
            "plan_files": len(plan.files),
            "loaded_existing": len(current_contents),
            "model": llm.model,
        },
    )

    try:
        response = llm.structured_call(
            system=CODER_SYSTEM_PROMPT,
            user=user_prompt,
            schema=CodeChange,
            max_tokens=max_output_tokens,
            temperature=temperature,
        )
    except CascadeLLMError as exc:
        raise CascadeError(f"Coder LLM call failed for story {story.id}: {exc}") from exc

    change = response.parsed

    if change.story_id != story.id:
        change = change.model_copy(update={"story_id": story.id})

    _validate_change_against_plan(plan=plan, change=change)

    logger.info(
        "coder.done",
        extra={
            "story_id": story.id,
            "files_produced": len(change.files),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    )

    return CodeResult(change=change, usage=response.usage)


def _validate_change_against_plan(*, plan: Plan, change: CodeChange) -> None:
    """Sanity-check that the LLM's CodeChange is consistent with the Plan.

    Checks:
    - Every file path in the change appears in the plan (no surprise files)
    - Every create/modify FileChange has non-null content
    - Every delete FileChange has null content
    """
    plan_paths = {fp.path for fp in plan.files}
    plan_actions = {fp.path: fp.action for fp in plan.files}

    for fc in change.files:
        if fc.path not in plan_paths:
            raise CascadeError(
                f"Coder produced file '{fc.path}' that wasn't in the plan. "
                f"This is a coder hallucination; rejecting the change."
            )
        if fc.action != plan_actions[fc.path]:
            raise CascadeError(
                f"Coder action for '{fc.path}' ({fc.action.value}) does not "
                f"match plan action ({plan_actions[fc.path].value})."
            )
        if fc.action in (FileAction.CREATE, FileAction.MODIFY):
            if fc.content is None or not fc.content.strip():
                raise CascadeError(
                    f"Coder produced empty content for '{fc.path}' "
                    f"(action={fc.action.value})."
                )
        elif fc.action == FileAction.DELETE:
            if fc.content is not None:
                # Tolerable; we just ignore content for deletes. Log it.
                logger.warning(
                    "coder.delete_with_content",
                    extra={"path": fc.path},
                )
