# Team Memory

This directory is the heart of Cascade. It's the shared substrate every AI stage reads from when extracting stories, planning code, generating tests, or opening PRs.

Without this, Cascade is just another AI tool. With it, Cascade *knows what your team knows*.

## Files

| File | Purpose |
|---|---|
| `conventions.md` | Coding style, naming patterns, file layout |
| `decisions.md` | Architectural decisions and *why* (ADR-style log) |
| `glossary.md` | Domain-specific terms unique to this product/codebase |
| `prior-work.md` | Summaries of recently-shipped stories (avoid duplicates) |
| `constraints.md` | Performance budgets, security requirements, deployment limits |

## How it works

Every Cascade stage reads relevant excerpts from these files into the LLM context. So when the LLM extracts a story from a meeting, plans code, or writes tests, it's grounded in *your team's accumulated knowledge*, not generic best practices.

## How to maintain

Update these as a team:
- After architecture meetings → add to `decisions.md`
- When you introduce a new convention → add to `conventions.md`
- When you ship something new → summarize in `prior-work.md`
- When jargon emerges → add to `glossary.md`

**v0.1**: manual updates. Crude but real.
**v0.2+**: Cascade can suggest updates based on processed meetings and merged PRs.

---

For the full vision, see [Cascade README](../README.md).
