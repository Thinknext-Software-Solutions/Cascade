# Cascade users

A living document that grounds product and engineering decisions in the real people we're building for. When in doubt about a feature or a polish detail, ask: which persona benefits, and how much?

Personas are illustrative composites, not real customers. They evolve as we learn from actual users.

---

## Sarita - Senior backend engineer at a 50-person SaaS

**Stack**: Python (FastAPI), PostgreSQL, GitHub, Jira, Anthropic API key (personal).
**Wants**: Run `cascade ticket jira:PROJ-123` and get a PR she can review in 10 minutes instead of building it herself in 4 hours.
**Frustrations today**:
- Empty team-memory files after `cascade init`. Doesn't know what "good" looks like.
- No way to verify everything is wired correctly before her first real run.
- Worried about token cost on long Jira tickets; no estimator before the call.

---

## Marcus - TypeScript / React frontend at a 200-person org

**Stack**: TypeScript, Next.js, self-hosted GitLab, corporate GitHub Copilot subscription, no Anthropic budget.
**Wants**: Use Cascade with his Copilot subscription. Open MRs (not PRs) on the self-hosted GitLab.
**Frustrations today**:
- No `copilot` LLM provider yet (planned v0.2).
- Has to configure GitLab base URL by hand. Documentation could be sharper for self-hosted setups.
- When something fails, error messages are stack traces, not actionable hints.

---

## Priya - Go solo founder

**Stack**: Go, Cloudflare Workers, GitHub, Linear, Claude Code subscription.
**Wants**: Drop in `cascade configure llm claude_code` and never see another API key prompt.
**Frustrations today**:
- Claude Code SDK is wrapped, but reliability of structured output isn't proven yet (no real-world data).
- Linear ticket import produces a story with one placeholder acceptance criterion; she'd like richer extraction from Linear's description markdown.
- No way to see what Cascade is currently doing during a long run; needs streaming progress.

---

## David - Senior Java engineer at a Fortune 500

**Stack**: Java/Spring, Azure DevOps Repos + Boards, Azure OpenAI behind a corporate gateway.
**Wants**: Demonstrate Cascade end-to-end to his architecture review board before requesting budget.
**Frustrations today**:
- Cannot demo without an LLM key. Internal Ollama / vLLM offers a path but needs proof it produces acceptable output.
- Azure DevOps token scopes are confusing; needs sharper docs on minimum required permissions.
- Compliance team will ask: where exactly does source code travel? Needs the security model spelled out.

---

## Yuki - Rust developer building a SaaS, no budget

**Stack**: Rust (Axum), Postgres, GitHub, no money for paid LLMs.
**Wants**: Run Cascade fully locally with Ollama against a 70B model, accept the slower speed.
**Frustrations today**:
- 70B model on a 24GB GPU is borderline; needs guidance on which models actually produce usable Cascade output.
- No way to test "is my Ollama setup good enough?" without running a real build.
- Wants a cost calculator that says "zero" so she has receipts for her decision.

---

## Emma - Engineering manager at a 100-person company

**Stack**: doesn't write code; uses Slack, Notion, Google Meet, GitHub Enterprise.
**Wants**: Sit in on planning meetings, record them, drag the file into Cascade Studio, and route approved stories to the right engineer.
**Frustrations today**:
- The CLI is intimidating for her use case. She needs Studio v0.1+ features (project picker, story review board, "assign to engineer" workflow).
- Meeting audio quality varies; needs Cascade to handle noisy meetings gracefully.
- Wants to share the extracted stories with engineers in a link, not a YAML file.

---

## What all six personas have in common

Three friction points show up across every story:

1. **Verification anxiety.** "I installed it. Did it actually work?" Nobody trusts a new tool with their real repo until they've seen it succeed on something safe.
2. **Empty team-memory.** Cascade tells users team-memory is important, then leaves them with template files. Self-defeating.
3. **Cost uncertainty.** Token costs are real money for the LLM providers. Users need to see estimates before clicking go.

These three are what `cascade doctor`, smart `cascade init`, and `cascade try` were built to address.

---

## How to use this document

When proposing a new feature:

- Name the persona(s) who benefit and by how much.
- Estimate the friction the feature removes (in minutes of confusion saved, or commands avoided).
- If the feature serves zero personas above, question whether it's worth building.

When triaging bugs:

- Bugs that block a persona on first run beat bugs that affect power users.
- Bugs in the "Bring your own AI" path (Claude Code, Ollama) beat bugs in the API-key path because they're our differentiation.

This document is intentionally short. Add a persona only when a real user pattern emerges that none of the existing six capture.
