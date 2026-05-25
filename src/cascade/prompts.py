"""Prompt templates for Cascade's LLM stages.

Kept in a separate module so prompts can be reviewed, A/B tested, and
evolved without touching pipeline logic. Each prompt is a function so
team memory and other dynamic context can be injected cleanly.

Style guide for prompts in this module:
- System prompt: defines role, principles, output discipline
- User prompt: task input plus relevant context
- No magic phrases or jailbreaks
- Explicit about what to skip / ignore
- Always specifies the exact structured-output target
"""

from __future__ import annotations


EXTRACTOR_SYSTEM_PROMPT = """You are a senior product manager and technical lead.
Your job is to read a team meeting transcript and extract well-formed user stories
that capture work the team has agreed should be done.

Core principles:
1. Only extract stories the team actually decided to pursue. Ignore brainstorming
   that didn't resolve, status updates with no action, and off-topic discussion.
2. Respect prior decisions: if the team has already shipped something similar
   (see team memory below), do not create a duplicate story. Note the overlap
   in the story's `notes` field instead.
3. Match the team's terminology exactly. If they have a glossary, use those terms.
4. Stories must be concrete enough that an engineer could start work without
   asking what they mean. Vague stories get a low confidence score.
5. Each story must have at least one acceptance criterion in Given/When/Then form.
6. Use the source_turn_indices to point to which turns in the transcript
   informed the story -- this lets reviewers trace your reasoning.
7. Confidence scoring guide:
   - 90+: explicit decision with clear scope and constraints
   - 70-89: clear intent, some scoping ambiguity
   - 50-69: rough direction agreed, significant gaps
   - Below 50: do NOT extract; the conversation isn't decided enough

Output discipline:
- Use the provided tool to return a structured StoryBatch.
- Story IDs follow the pattern: story-{YYYY-MM-DD}-{NNN} where NNN is a
  zero-padded 3-digit counter starting at 001 for this meeting.
- Titles must be imperative ("Add X", "Fix Y", "Refactor Z"), under 140 chars.
- Sizes: XS (<1hr), S (1-4hr), M (4-16hr), L (1-3d), XL (3d+, suggest splitting in notes).
"""


def build_extractor_user_prompt(
    *,
    transcript_text: str,
    team_memory_context: str,
    meeting_id: str,
    meeting_date: str,
) -> str:
    """Compose the user-side prompt for the story extractor.

    Args:
        transcript_text: The formatted transcript (e.g. MeetingTranscript.as_text()).
        team_memory_context: Pre-formatted team memory excerpts (may be empty).
        meeting_id: Stable ID for this meeting.
        meeting_date: Date in YYYY-MM-DD format (used in story ID prefix).

    Returns:
        A single string ready to pass as the user message.
    """
    parts: list[str] = []

    if team_memory_context.strip():
        parts.append(team_memory_context)
        parts.append("\n---\n\n")
    else:
        parts.append(
            "# Team memory\n\n"
            "(No team memory provided. Stories will be extracted using only "
            "the transcript -- quality may be lower than with team context.)\n\n"
            "---\n\n"
        )

    parts.append(f"# Meeting metadata\n\n")
    parts.append(f"- meeting_id: `{meeting_id}`\n")
    parts.append(f"- date: `{meeting_date}`\n")
    parts.append(f"- story ID prefix: `story-{meeting_date}-`\n\n")
    parts.append("---\n\n")

    parts.append("# Transcript\n\n")
    parts.append(transcript_text)
    parts.append("\n\n---\n\n")

    parts.append(
        "Extract the user stories. Use the provided tool. Set "
        f"`meeting_id` to `{meeting_id}`. Set source_turn_indices for each "
        "story to point to the transcript turn numbers (0-indexed) that "
        "informed it. If the transcript contains nothing extractable, return "
        "a StoryBatch with an empty stories list."
    )

    return "".join(parts)
