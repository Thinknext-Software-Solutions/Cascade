# Cascade

An open-source AI agent that takes a meeting recording, a tracker ticket, or a one-line prompt, and ships a tested pull request. Self-hosted. Uses your LLM key. Your code never leaves your org.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)](#roadmap)
[![Built by ThinkNext](https://img.shields.io/badge/built%20by-ThinkNext-22d3ee.svg)](https://thinknextsoftware.com)

> **Status**: pre-alpha, building in public. Star and watch to follow along, or [join the early beta list](mailto:hello@thinknextsoftware.com?subject=Cascade%20beta).

## Why Cascade exists

Most AI dev tools assume your team works exactly the way they expect. You write code in their IDE. You pay for their LLM. You host on the VCS they support. You start work by typing into a chat window.

Real engineering teams don't look like that. Some discuss requirements in standups, others work from Jira tickets, others just have a senior dev send a Slack message. Some teams have a corporate Copilot subscription and no Anthropic budget. Some teams can't legally send code to a SaaS at all. Some teams use GitLab. Some use Azure DevOps. Some still use Bitbucket.

Cascade was built to meet teams where they already work. Bring your own LLM. Use your existing tracker. Keep your code on your own infrastructure. Pick whichever input mode fits how you actually capture requirements.

## How it works

There are three on-ramps and one pipeline:

```
                       INPUT                                    OUTPUT
                       =====                                    ======
                       meeting recording (.mp3/.mp4)            ┐
                       tracker ticket                  ┐        │   plan, code, test
                         (Jira / Linear /              │═══════>│════════════════════>  PR on
                          GitHub Issues /              │        │                          GitHub /
                          Azure Boards /               │        │   LLM:                  GitLab /
                          GitLab Issues)               │        │   Anthropic /            Bitbucket /
                       ad-hoc prompt                   │        │   OpenAI /               Azure DevOps
                                                       ┘        │   Google Gemini /
                                                                │   Claude Code (no API key) /
                                                                │   Ollama / vLLM (self-hosted)
                                                                ┘
```

A meeting recording, a ticket, or a typed prompt all turn into the same thing: a `Story`. Each story moves through the same pipeline. A human reviews the extracted stories before any code gets generated. A human reviews the PR before any code gets merged. Cascade does the work in between.

## What makes it different

A handful of choices set Cascade apart from the rest of the agent landscape:

1. Everything runs on your machine or your CI. Cascade doesn't ship your code anywhere except to the LLM provider you configured.
2. You can use it with no API key if you already have a Claude Code subscription. The Claude Code SDK becomes the LLM transport, so the marginal cost is zero.
3. It's polyglot from day one. Python, TypeScript, JavaScript, Go, Rust, Java, Ruby, and C# are supported. Adding a ninth language is one entry in a registry.
4. Humans approve every gate. Cascade is built for trust, not for autonomy theatre. Engineers stay in the loop.
5. Team memory is a first-class concept. A `team-memory/` directory captures your conventions, decisions, glossary, and prior work. Every AI stage reads from it.

## Quick start

```bash
pip install cascade-agent              # base install
pip install cascade-agent[all]         # adds optional providers + Studio web dashboard

cascade init                            # scaffold cascade.yaml and team-memory/

cascade configure llm anthropic --key sk-ant-xxx --set-default
# Or skip the key entirely if you have Claude Code installed:
cascade configure llm claude_code --set-default

cascade configure vcs github --token ghp-xxx

# Pick whichever entry point matches how the work showed up:
cascade prompt "Add cursor pagination to /api/users with ?limit and ?after"
cascade ticket jira:PROJ-123
cascade ticket github:myorg/myrepo#42
cascade ingest recordings/standup.mp3       # writes transcripts/*.yaml
cascade extract transcripts/standup.yaml    # writes stories/*.yaml
cascade review stories/standup.yaml         # interactive accept / edit / reject
cascade build stories/standup.yaml          # plan, code, test, PR

# Prefer a web dashboard?
cascade ui                                  # opens http://localhost:8000
```

## Cascade Studio (the web dashboard)

Cascade ships with a web UI that surfaces the same operations as the CLI in a friendlier interface: visual story review, build history, provider config forms, and a team-memory editor with markdown preview.

```bash
pip install cascade-agent[studio]      # adds FastAPI + uvicorn
cascade ui                              # starts at http://localhost:8000
```

The dashboard runs locally. No remote service, no auth required for single-user mode, your code never leaves your machine. The frontend ships pre-built inside the pip package; no Node.js install needed at runtime.

Studio is in early development; the frontend source lives at [Thinknext-Software-Solutions/Cascade-Studio](https://github.com/Thinknext-Software-Solutions/Cascade-Studio) and the bundled UI updates with every `cascade-agent` release.

Credentials live at `~/.config/cascade/config.yaml` (mode 0600). Run `cascade configure show` to see what's set, with secrets masked.

## The pipeline in detail

```
+-------------------------------------------------------------------+
|                       TEAM MEMORY LAYER                            |
| (conventions, decisions, glossary, prior work, constraints)        |
+-------------------------------------------------------------------+
            ^    ^    ^    ^    ^    ^    ^    ^
            |    |    |    |    |    |    |    |  every stage
            |    |    |    |    |    |    |    |  reads memory
            |    |    |    |    |    |    |    |
+---------+ ++  ++  +-+----+ ++ ++ ++ +-+-----+
|  input  |->|IN|->|TX|->|stories|->|RV|->|PL|->|CD|->|PR open|
+---------+ +--+ +--+ +-------+ +--+ +--+ +--+ +-------+
            ingest transcribe extract review plan code   PR
                                          |
                            HUMAN APPROVES EACH GATE
                            (story review + final PR review)
```

| Stage | Module | What it does |
|---|---|---|
| Ingest | `transcribe.py` | Audio or video to text via Whisper. Three backends: faster-whisper, openai-whisper, OpenAI API. |
| Transcribe | (same) | Optional speaker diarization via pyannote, so each turn knows who said it. |
| Extract | `extractor.py` | Transcript into structured user stories with Given/When/Then acceptance criteria. |
| Review | `review.py` | Interactive accept / edit / reject / skip per story. Edit opens the story YAML in your `$EDITOR`. |
| Plan | `planner.py` | Approved story into a file-level implementation plan, with risks and explicit out-of-scope notes. |
| Code | `coder.py` | Plan into full file contents (modify, create, or delete). Language-aware, conventions-aware. |
| Test | `tester.py` | Run the language's test command. Capture pass/fail, output, duration. |
| Repo | `repo.py` + `vcs*.py` | Create branch, apply changes, commit, push, open PR. |

## Providers

### LLM providers

| Provider | API key needed? | Notes |
|---|---|---|
| Anthropic Claude | Yes | Default. Uses tool-use for structured output. |
| OpenAI | Yes | Structured Outputs via `response_format json_schema`. Works with Azure OpenAI, OpenRouter, or vLLM via `--base-url`. |
| Google Gemini | Yes | `response_schema` with Pydantic models. |
| Claude Code SDK | No | Uses your local Claude Code subscription. Zero-setup. |
| Ollama / vLLM | No | Local self-hosted models via OpenAI-compatible API. |

### VCS providers

| Provider | Self-hosted supported? |
|---|---|
| GitHub | Yes (GitHub Enterprise via `--base-url`) |
| GitLab | Yes (cloud and self-hosted) |
| Bitbucket Cloud | Cloud only in v0.1 |
| Azure DevOps Repos | Yes |

### Issue trackers (for `cascade ticket`)

GitHub Issues, Jira (Cloud and Server), Linear, Azure DevOps Boards, GitLab Issues.

### Languages

Python, TypeScript, JavaScript, Go, Rust, Java, Ruby, C#.

Adding a new language is a single entry in [`languages.py`](src/cascade/languages.py). Each entry captures the file extensions, default source and test directories, the test command, the install command, and any language-specific guidance to pass to the LLM.

## Security model

Cascade preserves a small set of invariants. Each one is a non-goal of the design. If any of them is violated, that's a bug, and we treat it as a security issue.

- Cascade never merges PRs. Humans always approve before merge.
- Cascade only writes to paths matching `paths.allowed` in `cascade.yaml`, minus anything in `paths.disallowed`. Deny wins over allow.
- Cascade never modifies `.github/`, `cascade.yaml`, or `team-memory/` by default.
- Cascade only executes the configured `test_command` and `git` shell commands. No arbitrary shell access.
- Source code, transcripts, and meeting recordings stay on the local machine and the configured LLM provider. Nothing else.
- User credentials at `~/.config/cascade/config.yaml` are stored with mode 0600.

See [SECURITY.md](SECURITY.md) for the full threat model and how to report a vulnerability.

## Configuration

Four layers, highest wins:

1. **CLI flags** for per-call overrides like `--language go` or `--model claude-opus-4-7`.
2. **Project config** at `./cascade.yaml` for per-repo settings, language overrides, and path allowlists. See [`cascade.yaml.example`](cascade.yaml.example).
3. **User config** at `~/.config/cascade/config.yaml` for credentials and personal defaults. Managed via `cascade configure`.
4. **Environment variables** as a fallback. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GITHUB_TOKEN`, `JIRA_API_TOKEN`, and so on.

## Roadmap

| Version | Target | Highlights |
|---|---|---|
| **v0.1** (tech preview) | 2026-09-15 | Foundation, all 5 LLM providers, all 4 VCS providers, all 5 issue trackers, ingest / review / build pipeline |
| v0.2 | 2026-11-15 | Quality bar via real-world dogfooding, vector-store team memory (RAG over embeddings), Copilot CLI provider, multi-story batch build |
| v0.3 | 2027-01-15 | Real-time meeting capture, Slack and Teams as sources, multi-repo coordination |
| v1.0 | 2027-04-15 | Web UI for review, fine-tuned routing, GA |

## Contributing

Contributions of all sizes welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

Maintainer response targets: issues within 5 business days, PRs within 3.

## License

[MIT](LICENSE). Use freely, commercially, anywhere.

---

Built by [ThinkNext Software Solutions](https://thinknextsoftware.com). Questions? [hello@thinknextsoftware.com](mailto:hello@thinknextsoftware.com).
