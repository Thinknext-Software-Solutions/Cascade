# Contributing to Cascade

Thanks for being interested in Cascade. Contributions of all sizes welcome.

## Quick start

```bash
git clone https://github.com/thinknext/cascade
cd cascade
uv sync --extra dev
uv run pytest
```

## What we welcome

- **Bug reports** — open an issue with repro steps
- **Small PRs** — typo fixes, doc improvements, test additions: open directly
- **Larger PRs** — open a Discussion first to align on direction
- **Recipes** — concrete examples of using Cascade in your workflow
- **Prompt improvements** — story extraction quality is the highest-leverage area
- **Eval contributions** — benchmark examples for any stage

## What we're cautious about

- v0.2+ features in v0.1 PRs (we're scope-disciplined; see roadmap)
- Heavy framework dependencies (LangChain, LangGraph)
- Provider-specific code paths (we keep the LLM provider configurable)
- UI work in v0.1 (CLI-only until v1.0)

## Code style

- Python: Black + Ruff (both run in CI)
- Type hints required for public functions
- Docstrings: short, focused, explain *why* not *what*
- Tests required for new functionality

## Commit messages

Conventional commits:
- `feat:` new capability
- `fix:` bug fix
- `docs:` documentation
- `refactor:` code restructure, no behavior change
- `test:` tests added/changed
- `chore:` build / tooling / deps

## SLA

- First response on issues: within **5 business days**
- First response on PRs: within **3 business days**

Tag the maintainers if we miss this — life happens.

## Code of Conduct

By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

---

Questions? Open a Discussion or email hello@thinknextsoftware.com.
