# Cascade

> Turn a team meeting into shipped code. Recording in, working tested pull request out — humans approving at every gate.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)](#roadmap)
[![Built by ThinkNext](https://img.shields.io/badge/built%20by-ThinkNext-22d3ee.svg)](https://thinknextsoftware.com)

> **Status**: Pre-alpha, building in public. Soft launch target: **2026-09-15**. Star + watch to follow along, or [join the early beta list](mailto:hello@thinknextsoftware.com?subject=Cascade%20beta).

---

## What is Cascade?

Cascade takes a **team meeting recording** and runs it through a structured pipeline:

```
Meeting recording  →  Stories  →  [human review]  →  Code + Tests  →  PR
```

Every stage is grounded in your **team's shared memory** — the conventions, decisions, glossary, and constraints your team has accumulated. So the AI knows what *your team* knows, not just what's in the codebase.

It's self-hosted, OSS, uses your own LLM key, and your code never leaves your infrastructure.

---

## Why Cascade exists

Cascade solves two real problems that today's AI dev tools ignore:

### 1. The chasm between "the team talked about it" and "code shipped"

Teams discuss work in meetings, Slack threads, and impromptu calls. Then a single engineer has to manually translate that conversation into stories, then specs, then code, then tests. Information leaks at every transition.

Cascade closes the loop. The output of a meeting *is* the input to the build pipeline.

### 2. AI sessions are siloed by developer

When teams use Copilot, Cursor, or Claude individually:
- Each developer's AI session is isolated
- Architectural decisions made last week aren't in anyone's AI context this week
- New devs start from zero with their AI
- Senior knowledge doesn't compound across the team

**Teams collaborate in human channels — but the AI side of the modern dev workflow is single-player.** Cascade makes it team-shared.

---

## How it works

```
┌──────────────────────────────────────────────────────────────────┐
│                       TEAM MEMORY LAYER                           │
│  conventions · decisions · glossary · prior work · constraints    │
│                     (read by every stage)                          │
└──────────────────────────────────────────────────────────────────┘
       ▲    ▲     ▲      ▲      ▲      ▲      ▲      ▲
       │    │     │      │      │      │      │      │
  ┌────┴┐ ┌─┴┐ ┌──┴──┐ ┌─┴───┐ ┌┴───┐ ┌┴───┐ ┌┴───┐ ┌─┴────┐
  │Audio│→│TX│→│Story│→│REVW │→│PLAN│→│CODE│→│TEST│→│PR    │
  │ /Vid│ │  │ │     │ │     │ │    │ │    │ │    │ │      │
  └─────┘ └──┘ └─────┘ └─────┘ └────┘ └────┘ └────┘ └──────┘
                          │
                    HUMAN GATES HERE
                  (approve / edit / reject
                   each story before code gen)
```

| Stage | What it does |
|---|---|
| **Ingest** | Accepts audio, video, or text. Routes to transcription if needed. |
| **Transcribe** | Local Whisper transcription with speaker diarization (Speaker A/B/C). |
| **Stories** | LLM extracts structured user stories with acceptance criteria, informed by team memory. |
| **Review** | Human approves, edits, or rejects each story (CLI-interactive in v0.1). |
| **Plan** | For each approved story: file list, approach, dependencies. |
| **Code** | Generates code + tests on a new branch. |
| **Test** | Runs tests. Iterates once if failures. |
| **PR** | Opens GitHub PR linked back to the original meeting timestamp and story. |

---

## Quick start

> ⚠️ Pre-alpha. The commands below show the *intended* developer experience. Pieces work today; end-to-end pipeline lands by 2026-09-15.

```bash
# Install (eventually)
pip install cascade-ai

# Initialize team memory in your repo
cascade init
$EDITOR team-memory/conventions.md
$EDITOR team-memory/decisions.md
$EDITOR team-memory/glossary.md

# Process a meeting recording
cascade ingest standup-2026-09-12.mp4
cascade extract transcripts/2026-09-12.txt
cascade review stories/2026-09-12.yaml      # interactive

# Build approved stories
cascade build stories/2026-09-12-approved.yaml --story 1
# → creates branch, generates code + tests, opens PR
```

Configuration via `cascade.yaml`:

```yaml
version: 1
agent:
  model: claude-opus-4-7
  max_iterations: 1

memory:
  path: team-memory/

paths:
  allowed:
    - src/**
    - tests/**
  disallowed:
    - .github/**

test_command: pytest
```

---

## The team memory layer

This is what makes Cascade different from "another agent."

A `team-memory/` directory in your repo holds structured markdown files that every Cascade stage reads as context:

```
team-memory/
├── conventions.md      # coding style, naming, file layout
├── decisions.md        # ADR-style log: what we chose and why
├── glossary.md         # domain terms specific to this product
├── prior-work.md       # summaries of past stories shipped
└── constraints.md      # performance budgets, security requirements
```

When Cascade extracts stories from a meeting, generates code, or writes tests, it knows:
- "Our team uses snake_case for Python, camelCase for TypeScript"
- "We chose Postgres over MongoDB last quarter — don't suggest MongoDB"
- "A 'workspace' in our app is what other tools call a 'project'"
- "We already shipped pagination on /api/users — don't redo it"
- "Response times under 200ms p99"

**v0.1**: plain markdown, manually updated by the team. Crude but real.

**v0.2+**: vector store + smart retrieval. Auto-updates from processed meetings.

---

## Security model

Same security position as Relay — built for teams that can't or won't send code to a SaaS:

- ✅ Runs entirely on your infrastructure (your machine, your CI, your servers)
- ✅ Uses your own LLM API key
- ✅ Code never leaves your network
- ✅ All file changes on a new branch, never main
- ✅ **Cascade never merges** — every PR requires human approval
- ✅ Cascade only writes to allowed paths in `cascade.yaml`
- ✅ Cannot modify `.github/` or its own config

---

## Comparison vs alternatives

| | Cursor | Devin | Aider | Linear AI | **Cascade** |
|---|---|---|---|---|---|
| Input modality | typed prompts | typed prompts | typed prompts | typed text | **meeting recordings** |
| Autonomous (no constant driving) | ❌ | ✅ | ❌ | partial | ✅ |
| Team-shared memory layer | ❌ | ❌ | ❌ | partial | ✅ |
| Open source | ❌ | ❌ | ✅ | ❌ | ✅ |
| Self-hosted (your infra) | ✅ | ❌ | ✅ | ❌ | ✅ |
| Source never leaves your org | ✅ | ❌ | ✅ | ❌ | ✅ |

The combination of *meeting-as-input* + *team-memory substrate* + *OSS self-hosted* is the wedge. No tool checks all three boxes.

---

## Roadmap

| Version | Capabilities | Target |
|---|---|---|
| **v0.1** (MVP) | Audio/text input · Python only · Anthropic LLM · CLI-driven · single-story-at-a-time · markdown team memory | **2026-09-15** |
| **v0.2** | TypeScript support · multi-story batch · OpenAI provider · vector-store team memory · cost monitoring | Q4 2026 |
| **v0.3** | Multi-meeting collation · Slack/Linear/Notion integrations · real-time meeting capture · auto-updating team memory | Q1 2027 |
| **v1.0** | Web UI · GitLab/Bitbucket · self-hosted LLM (Ollama, vLLM) · multi-agent specialists | Mid 2027 |

---

## Sister project: Relay

For teams that already have well-scoped issues and don't need the meeting-extraction front-end, the simpler [Relay](https://github.com/Thinknext-Software-Solutions/Relay) project is a focused Issue→PR agent. Cascade is the broader vision; Relay is one specific entry point into the build pipeline.

---

## FAQ

**Q: Why start from meetings? Why not from a typed prompt?**
A: Because that's where the actual context lives. By the time someone types a prompt, they've already filtered, compressed, and edited the original conversation. Cascade preserves the full context.

**Q: What about meeting privacy?**
A: Everything runs locally by default — Whisper transcription is local, LLM calls go to your provider with your key. No cloud relay.

**Q: How accurate is the transcription?**
A: Whisper's `medium` model gets ~95% word accuracy on clean audio. Quality degrades with poor audio, heavy accents, or domain jargon. Cascade flags low-confidence sections for review.

**Q: What if the story extraction is wrong?**
A: Humans review every story before any code is generated. That's the entire point of the design — Cascade extracts, humans approve, then code happens.

**Q: How much will it cost to run?**
A: You pay your own LLM costs. Rough estimate for a 30-min meeting → 5 stories shipped: $10-$30 in Claude API calls plus a few cents in compute. Cost monitoring lands in v0.2.

**Q: Can I use this on a private repo?**
A: Yes. That's the primary use case.

**Q: What languages will Cascade support?**
A: v0.1 = Python only. v0.2 adds TypeScript. v1.0 expands further based on community demand.

---

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md).

For non-trivial changes, open a GitHub Discussion first to align on direction.

---

## License

MIT. See [LICENSE](LICENSE).

---

## About

Cascade is built and maintained by [ThinkNext Software](https://thinknextsoftware.com) — an AI-augmented engineering and staffing firm. We use AI at every step of our own SDLC, and we ship the tools we use ourselves.

If Cascade is helpful to your team, consider [working with us](https://thinknextsoftware.com#contact) on your next project.

Follow along: [@ThinkNextHQ](https://twitter.com/ThinkNextHQ) · [LinkedIn](https://linkedin.com/company/thinknextsoftware) · [Blog](https://thinknextsoftware.com/blog)
