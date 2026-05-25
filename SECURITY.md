# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| latest v0.x | ✅ |
| older | ❌ — upgrade |

## Reporting a vulnerability

**Do NOT open a public GitHub issue for security vulnerabilities.**

Email **security@thinknextsoftware.com** with:
- Description
- Repro steps
- Impact assessment
- Your contact

Acknowledgment within **2 business days**. Remediation timeline within **5 business days**.

## Security invariants

Cascade's design relies on these. Violations are vulnerabilities:

- Cascade never merges PRs
- Cascade never modifies `.github/`
- Cascade never modifies its own `cascade.yaml`
- Cascade only runs `pytest` and `git` shell commands
- Cascade only writes to paths allowed by `cascade.yaml`
- Source code, transcripts, and meeting recordings never leave the local machine or your LLM provider
- Cascade never accesses external APIs beyond the configured LLM
- Cascade never reads files outside the repo root + transcripts directory

## Out of scope

Not vulnerabilities:
- LLM produces low-quality code (humans approve PRs)
- High LLM costs (configure caps in v0.2)
- Third-party dependency issues (report upstream)
- Whisper transcription errors on poor audio
