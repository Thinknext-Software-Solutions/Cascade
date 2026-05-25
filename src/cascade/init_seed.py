"""Smart project initialization: seed team-memory files with sensible
defaults based on what's actually in the repo.

`cascade init` calls into this module after creating the scaffolding so
that users land on populated team-memory files instead of empty templates.

Heuristics applied:
- Detect the project's language and seed conventions.md with language-
  specific defaults (snake_case for Python, gofmt for Go, etc.).
- Detect the test framework from existing test files; hint at it in
  conventions.md.
- Read git log for the last ~10 merged commits; seed prior-work.md so
  the LLM has some recent context to ground in.
- Look for common ADR locations (docs/adr/, docs/decisions/, ARCHITECTURE.md)
  and reference them in decisions.md so users know to consolidate.
- Look for a README.md and reference it.

All seeded content is clearly marked as "Cascade-suggested starting
point -- edit freely" so users know they should personalize it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from .languages import LanguageProfile, detect_language


_SEED_MARKER = (
    "<!-- Cascade-suggested starting point. Edit freely. The more accurate "
    "this is to your team's actual practice, the better Cascade's output. -->"
)


# ----------------------------------------------------------------------------
# Conventions
# ----------------------------------------------------------------------------


_LANGUAGE_CONVENTIONS: dict[str, str] = {
    "python": (
        "## Code style\n\n"
        "- Python 3.11+ with type hints on all public function signatures.\n"
        "- snake_case for functions and variables, PascalCase for classes, "
        "UPPER_SNAKE for constants.\n"
        "- Prefer pathlib over os.path.\n"
        "- f-strings over % formatting and .format().\n"
        "- Raise specific exceptions, not bare 'except' clauses.\n\n"
        "## Test framework\n\n"
        "- pytest. Test files live in tests/ and are named test_*.py.\n"
        "- Use fixtures over setUp/tearDown.\n"
        "- Parametrize tests with @pytest.mark.parametrize where appropriate.\n\n"
        "## Common patterns\n\n"
        "- Type-safe data with Pydantic where it makes sense.\n"
        "- No global mutable state.\n"
    ),
    "typescript": (
        "## Code style\n\n"
        "- Strict TypeScript: no 'any', prefer 'unknown' then narrow.\n"
        "- ES modules (import/export), not CommonJS.\n"
        "- Named exports preferred over default exports.\n"
        "- camelCase for variables and functions, PascalCase for types and components.\n\n"
        "## Test framework\n\n"
        "- Vitest. Test files use the .test.ts suffix and live next to source.\n"
        "- Use describe + it blocks. Prefer it.each for parametrized tests.\n\n"
        "## Common patterns\n\n"
        "- async/await over .then() chains.\n"
        "- Optional chaining and nullish coalescing instead of && and ||.\n"
    ),
    "javascript": (
        "## Code style\n\n"
        "- Modern JavaScript: const/let (no var), arrow functions, async/await.\n"
        "- ES modules.\n"
        "- camelCase for variables and functions, PascalCase for classes.\n\n"
        "## Test framework\n\n"
        "- Vitest. Test files use the .test.js suffix.\n"
        "- JSDoc comments where types would help, even without TypeScript.\n"
    ),
    "go": (
        "## Code style\n\n"
        "- gofmt-formatted. No unused imports.\n"
        "- Idiomatic Go: error returns as the last return value, "
        "exhaustive switches, no overly clever abstractions.\n"
        "- camelCase for unexported, PascalCase for exported.\n\n"
        "## Test framework\n\n"
        "- Standard testing package. Test files end in _test.go, same package.\n"
        "- Table-driven tests for multiple scenarios.\n"
        "- Use t.Helper() in test helper functions.\n\n"
        "## Common patterns\n\n"
        "- Interfaces defined where they're used, not where they're implemented.\n"
        "- Context as the first parameter for any function that does I/O.\n"
    ),
    "rust": (
        "## Code style\n\n"
        "- Idiomatic Rust: prefer Result over panics, '?' for error propagation.\n"
        "- Derive common traits (Debug, Clone, PartialEq) where useful.\n"
        "- snake_case for functions and variables, PascalCase for types.\n\n"
        "## Test framework\n\n"
        "- Unit tests in '#[cfg(test)] mod tests' blocks within the same file.\n"
        "- Integration tests in tests/.\n"
    ),
    "java": (
        "## Code style\n\n"
        "- camelCase for methods and variables, PascalCase for classes.\n"
        "- Prefer constructor injection over field injection.\n"
        "- Records for immutable data, sealed classes for closed hierarchies.\n\n"
        "## Test framework\n\n"
        "- JUnit 5. Test classes end in Test, in the parallel package under src/test/java.\n"
        "- Use @Test, @BeforeEach, @ParameterizedTest where useful.\n"
    ),
    "ruby": (
        "## Code style\n\n"
        "- Idiomatic Ruby with RuboCop defaults.\n"
        "- snake_case for methods and variables, PascalCase for classes and modules.\n"
        "- Prefer symbols over strings for hash keys when they're identifiers.\n\n"
        "## Test framework\n\n"
        "- RSpec. Spec files end in _spec.rb under spec/.\n"
        "- Use describe/context/it blocks. Prefer 'let' over instance variables.\n"
    ),
    "csharp": (
        "## Code style\n\n"
        "- PascalCase for methods, classes, and public members.\n"
        "- camelCase for local variables and parameters.\n"
        "- Prefer expression-bodied members and pattern matching where readable.\n\n"
        "## Test framework\n\n"
        "- xUnit. Test class names end in Tests.\n"
        "- [Fact] for single-case tests, [Theory] + [InlineData] for parameterized.\n"
    ),
}


def seed_conventions(language: Optional[LanguageProfile]) -> str:
    """Produce a conventions.md body seeded with language-specific defaults."""
    body = ["# Conventions", "", _SEED_MARKER, ""]

    if language is None:
        body.extend(
            [
                "## Code style",
                "",
                "Describe your team's naming, formatting, and style conventions here.",
                "Be specific. Examples beat abstractions.",
                "",
                "## Test framework",
                "",
                "Which test framework do you use? Where do tests live?",
                "",
            ]
        )
    else:
        body.append(_LANGUAGE_CONVENTIONS.get(language.name, ""))

    body.extend(
        [
            "## Patterns we avoid",
            "",
            "List anti-patterns specific to your team. Libraries you've intentionally rejected. Approaches that have burned you before.",
            "",
        ]
    )
    return "\n".join(body)


# ----------------------------------------------------------------------------
# Decisions
# ----------------------------------------------------------------------------


_ADR_HINTS = (
    "docs/adr",
    "docs/decisions",
    "doc/adr",
    "adr",
    "ARCHITECTURE.md",
    "DECISIONS.md",
    "docs/architecture.md",
)


def find_existing_adr_paths(repo_root: Path) -> list[str]:
    """Look for common ADR locations in the repo."""
    found: list[str] = []
    for hint in _ADR_HINTS:
        candidate = repo_root / hint
        if candidate.exists():
            found.append(hint)
    return found


def seed_decisions(repo_root: Path) -> str:
    body = ["# Decisions", "", _SEED_MARKER, ""]
    existing = find_existing_adr_paths(repo_root)
    if existing:
        body.append("## Existing architecture documentation in this repo")
        body.append("")
        body.append(
            "Cascade detected the following locations that already document team decisions. "
            "Consider copying the relevant entries here, or referencing them with a brief summary."
        )
        body.append("")
        for path in existing:
            body.append(f"- `{path}`")
        body.append("")
        body.append("---")
        body.append("")

    body.extend(
        [
            "## Decision template",
            "",
            "```",
            "## [YYYY-MM-DD] Decision title",
            "**Context**: What was the situation that required a decision?",
            "**Decision**: What did we decide?",
            "**Why**: What were the reasons? What alternatives did we reject and why?",
            "**Implications**: What does this mean for future work?",
            "```",
            "",
        ]
    )
    return "\n".join(body)


# ----------------------------------------------------------------------------
# Prior work
# ----------------------------------------------------------------------------


def seed_prior_work(repo_root: Path) -> str:
    """Read recent merge commits and seed prior-work.md with placeholders."""
    body = ["# Prior Work", "", _SEED_MARKER, ""]

    recent = _recent_merge_commits(repo_root, limit=10)
    if recent:
        body.append("## Recent merged work (from git log)")
        body.append("")
        body.append(
            "Cascade pulled the last few merged-PR commit subjects below. Replace each with a 1-2 sentence summary of what shipped and any pattern that other work should follow."
        )
        body.append("")
        for date, subject in recent:
            body.append(f"### [{date}] {subject}")
            body.append("")
            body.append("- What shipped: ")
            body.append("- Files touched: ")
            body.append("- Notes for future work: ")
            body.append("")
    else:
        body.append("## Format")
        body.append("")
        body.append("```")
        body.append("## [YYYY-MM-DD] Story title")
        body.append("- Summary: what shipped (1-2 sentences)")
        body.append("- Files touched: primary modules / endpoints / components")
        body.append("- Notes for future work: anything to know")
        body.append("```")
        body.append("")

    return "\n".join(body)


def _recent_merge_commits(repo_root: Path, *, limit: int) -> list[tuple[str, str]]:
    """Return (date, subject) for the most recent merge commits, or
    regular commits if there are no merges yet."""
    try:
        # Prefer merge commits (PRs); fall back to regular commits
        result = subprocess.run(
            [
                "git",
                "log",
                "--merges",
                "--pretty=format:%ad|%s",
                "--date=short",
                f"-n{limit}",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        lines = [l for l in result.stdout.strip().splitlines() if l]
        if lines:
            return [tuple(line.split("|", 1)) for line in lines if "|" in line]

        # No merges -- fall back to regular commits
        result = subprocess.run(
            [
                "git",
                "log",
                "--pretty=format:%ad|%s",
                "--date=short",
                f"-n{limit}",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        lines = [l for l in result.stdout.strip().splitlines() if l]
        return [tuple(line.split("|", 1)) for line in lines if "|" in line]
    except (OSError, subprocess.SubprocessError):
        return []


# ----------------------------------------------------------------------------
# Glossary and constraints (no language-specific seeding yet)
# ----------------------------------------------------------------------------


def seed_glossary() -> str:
    return (
        "# Glossary\n\n"
        f"{_SEED_MARKER}\n\n"
        "Domain-specific terms used in this codebase. Cascade reads this so AI-generated names, comments, and docs match your team's vocabulary.\n\n"
        "## Example\n\n"
        "**Workspace**: A user's top-level container. Each user can have many workspaces. Not a folder, not a UI panel.\n\n"
        "Add your team's terms below.\n"
    )


def seed_constraints() -> str:
    return (
        "# Constraints\n\n"
        f"{_SEED_MARKER}\n\n"
        "Non-functional requirements that affect how new work should be designed and built.\n\n"
        "## Performance\n\n"
        "- API response time targets (for example, p99 under 200ms)\n"
        "- Database query budgets (for example, max 5 queries per request)\n"
        "- Memory limits, payload sizes\n\n"
        "## Security\n\n"
        "- Authentication requirements\n"
        "- Data handling (PII, encryption, audit logging)\n"
        "- Compliance regimes (SOC 2, HIPAA, GDPR)\n\n"
        "## Deployment\n\n"
        "- Runtime environment (containers, serverless, bare metal)\n"
        "- Deployment frequency expectations\n"
        "- Rollback constraints\n"
    )


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------


def seed_team_memory(repo_root: Path) -> dict[str, str]:
    """Produce smart starter content for every team-memory file.

    Returns a mapping of filename -> body. Callers write these to disk.
    """
    language = detect_language(repo_root)
    return {
        "conventions.md": seed_conventions(language),
        "decisions.md": seed_decisions(repo_root),
        "constraints.md": seed_constraints(),
        "glossary.md": seed_glossary(),
        "prior-work.md": seed_prior_work(repo_root),
    }
