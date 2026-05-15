# Security Policy

software-satellite-lab is a local-first evidence layer for AI-assisted software work. Its default public demo path is designed to run without an API key, without network calls, and without uploading raw source code.

## Supported Scope

Security reports should focus on:

- evidence that is incorrectly treated as positive support
- private source, secrets, or private design notes appearing in public artifacts
- pack manifests bypassing the declared permission model
- release checks that silently skip a required gate
- transcript or agent-claim output being trusted without source-linked artifact evidence

## Reporting

Please open a private security advisory when possible. If that is not available, open a minimal issue that describes the affected command, artifact path, and expected impact without pasting secrets or proprietary code.

For suspected privacy leaks, include:

- the command that produced the artifact
- the generated artifact path
- whether the leak is raw source, secret-shaped text, transcript content, or metadata
- the smallest redacted excerpt needed to reproduce the issue

## Non-Goals

This project does not claim to guarantee secure software. Signed evidence and release reports prove the scope of checks performed; they do not prove that a system is bug-free.
