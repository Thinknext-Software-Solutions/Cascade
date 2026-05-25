# Cascade

> An open-source AI agent that takes a meeting recording, a tracker ticket, or a one-line prompt — and ships a tested pull request. Self-hosted. Your LLM key. Your code never leaves your org.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)](#roadmap)
[![Built by ThinkNext](https://img.shields.io/badge/built%20by-ThinkNext-22d3ee.svg)](https://thinknextsoftware.com)

> **Status**: Pre-alpha, building in public. Star + watch to follow along, or [join the early beta list](mailto:hello@thinknextsoftware.com?subject=Cascade%20beta).

## What Cascade is

Three on-ramps. One pipeline. A pull request at the end.

```
                       INPUT                                    OUTPUT
                       =====                                    ======
                       meeting recording (.mp3/.mp4)            ┐
                       tracker ticket                  ┐        │   plan → code → test
                         (Jira / Linear /              │═══════→│════════════════════→  PR on
                          GitHub Issues /              │        │                          GitHub /
                          Azure Boards /               │        │   LLM:                  GitLab /
                          GitLab Issues)               │        │   Anthropic /            Bitbucket /
                       ad-hoc prompt                   │        │   OpenAI /               Azure DevOps
                                                       ┘        │   Google Gemini /
                                                                │   Claude Code (no API key) /
                                                                │   Ollama / vLLM (self-hosted)
                                                                ┘
```

**Why it exists**: most AI dev tools assume a single starting point (your IDE), a single LLM (theirs), a single VCS (GitHub), and a single source of work (you typing). Real engineering teams aren't shaped like that. Cascade meets you where you already work.

## Differentiators

- **Self-hosted everything.** Code stays in your CI runners. Your LLM key. Your VCS. No SaaS in the path.
- **Bring your own AI subscription.** Already pay for Claude Code? Use the Claude Code SDK — no separate API key.
- **Polyglot from day one.** Python, TypeScript, JavaScript, Go, Rust, Java, Ruby, C#. New languages added by appending to a registry.
- **Human at every gate.** Cascade plans, codes, and tests — humans review and approve every merge. Designed for trust, not autonomy theatre.
- **Team memory baked in.** A `team-memory/` directory of conventions, decisions, glossary, and prior work that every AI stage reads as grounding context.

## Quick start

```bash
# 1. Install
pip install cascade-agent                  # base
pip install cascade-agent[all]             # + OpenAI, Google, Claude Code, GitLab, Jira

# 2. Configure once (creds saved to ~/.config/cascade/config.yaml, chmod 0600)
cascade configure llm anthropic --key sk-ant-xxx --set-default
cascade configure vcs github --token ghp-xxx
# Or use a local Claude Code subscription instead of an API key:
cascade configure llm claude_code --set-default

# 3. Use it -- pick any entry point
cascade prompt "Add cursor pagination to /api/users with ?limit and ?after"
cascade ticket jira:PROJ-123
cascade ticket github:myorg/myrepo#42
cascade ingest recordings/standup.mp3        # produces transcripts/*.yaml
cascade extract transcripts/standup.yaml     # produces stories/*.yaml
cascade review stories/standup.yaml          # interactive accept/edit/reject
cascade build stories/standup.yaml           # plan → code → test → PR
```

## How it works

```
┌────────────────────────────────────────────────────────────────────┐
│                       TEAM MEMORY LAYER                             │
│  (conventions · decisions · glossary · prior work · constraints)    │
└────────────────────────────────────────────────────────────────────┘
            ▲    ▲    ▲    ▲    ▲    ▲    ▲    ▲
            │    │    │    │    │    │    │    │  every stage reads
            │    │    │    │    │    │    │    │  from + writes to
            │    │    │    │    │    │    │    │
┌───────┐ ┌─┴┐ ┌─┴┐ ┌──┴───┐ ┌┴─┐ ┌┴─┐ ┌┴─┐ ┌┴─────┐
│ input │→│IN│→│TX│→│Stories│→│RV│→│PL│→│CD│→│PR Open│
└───────┘ └──┘ └──┘ └───────┘ └──┘ └──┘ └──┘ └───────┘
              ingest  extract  review plan code   PR
                                       │
                                       │
                            HUMAN APPROVES EACH GATE
                            (story review + final PR review)
```

| Stage | Module | What it does |
|---|---|---|
| Ingest | `transcribe.py` | Audio/video → text via Whisper (local, faster-whisper, or OpenAI API) |
| Transcribe | (same) | Optional speaker diarization via pyannote |
| Extract | `extractor.py` | Transcript → structured user stories with Given/When/Then acceptance criteria |
| Review | `review.py` | Interactive accept / edit / reject / skip per story |
| Plan | `planner.py` | Approved story → file-level implementation plan with risks + out-of-scope |
| Code | `coder.py` | Plan → full file contents (modify/create/delete); language-aware |
| Test | `tester.py` | Run the language's test command; capture results |
| Repo | `repo.py` + `vcs*.py` | Branch, commit, push, open PR |

## Provider matrix

### LLM providers

| Provider | API key needed? | Notes |
|---|---|---|
| Anthropic Claude | Yes | Default. Uses tool-use for structured output. |
| OpenAI | Yes | Structured Outputs via `response_format json_schema`. Works for Azure OpenAI / OpenRouter / vLLM via `--base-url`. |
| Google Gemini | Yes | `response_schema` with Pydantic model. |
| Claude Code SDK | **No** | Uses your local Claude Code subscription. Zero-setup. |
| Ollama / vLLM | No | Local self-hosted models via OpenAI-compatible API. |

### VCS providers

| Provider | Self-hosted supported? |
|---|---|
| GitHub | Yes (GitHub Enterprise via `--base-url`) |
| GitLab | Yes (cloud + self-hosted) |
| Bitbucket Cloud | Cloud only in v0.1 |
| Azure DevOps Repos | Yes |

### Issue trackers (for `cascade ticket`)

GitHub Issues · Jira (Cloud + Server) · Linear · Azure DevOps Boards · GitLab Issues

### Languages

Python · TypeScript · JavaScript · Go · Rust · Java · Ruby · C#

Adding a new language = add a `LanguageProfile` entry. See [`languages.py`](src/cascade/languages.py).

## Security model

Cascade's pipeline preserves these invariants. Each one is a non-goal of the design that, if violated, is treated as a bug:

- Cascade never merges PRs; humans always approve before merge.
- Cascade only writes to paths matching `paths.allowed` in `cascade.yaml`, minus anything in `paths.disallowed` (deny wins).
- Cascade never modifies `.github/`, `cascade.yaml`, or `team-memory/` (configurable but disallowed by default).
- Cascade only executes the configured `test_command` and `git` shell commands — no arbitrary shell access.
- Source code, transcripts, and meeting recordings stay on the local machine and the configured LLM provider; nothing else.
- User credentials are stored at `~/.config/cascade/config.yaml` with mode `0600`.

See [SECURITY.md](SECURITY.md) for the full threat model and how to report vulnerabilities.

## Configuration

Three layers, highest wins:

1. **CLI flags** (per-call): `--language go`, `--model claude-opus-4-7`
2. **Project config** at `./cascade.yaml`: per-repo settings, language override, path allowlists. See [`cascade.yaml.example`](cascade.yaml.example).
3. **User config** at `~/.config/cascade/config.yaml`: credentials and personal defaults. Managed via `cascade configure`.
4. **Environment variables** (fallback): `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GITHUB_TOKEN`, `JIRA_API_TOKEN`, etc.

## Roadmap

| Version | Target | Highlights |
|---|---|---|
| **v0.1** (tech preview) | 2026-09-15 | Foundation + all 5 LLM providers + all 4 VCS providers + all 5 issue trackers + ingest/review/build pipeline |
| v0.2 | 2026-11-15 | Whisper API quality bar, vector-store team memory (RAG over embeddings), Copilot CLI provider, multi-story batch build |
| v0.3 | 2027-01-15 | Real-time meeting capture, Slack/Teams source, multi-repo coordination |
| v1.0 | 2027-04-15 | Web UI for review, fine-tuned routing, GA |

## Contributing

We welcome contributions of all sizes. See [CONTRIBUTING.md](CONTRIBUTING.md).

Maintainer SLA: issues responded to within **5 business days**, PRs within **3 business days**.

## License

[MIT](LICENSE). Use freely, commercially, anywhere.

---

Built by [ThinkNext Software Solutions](https://thinknextsoftware.com). Have a question? [hello@thinknextsoftware.com](mailto:hello@thinknextsoftware.com).
