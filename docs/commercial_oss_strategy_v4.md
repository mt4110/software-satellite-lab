# Commercial OSS Strategy v4

This is the public roadmap pointer for the M18-M30 product wave.

## Core Decision

software-satellite-lab should become a local-first, zero-trust evidence protocol for AI-assisted software work. It should not become another coding agent, PR reviewer, or LLM observability dashboard.

## Product Wedge

The project records and verifies software-work evidence:

- source-linked evidence for AI-assisted code work
- signed, scoped proof of checks performed
- zero-trust handling of agent transcripts
- local-first failure memory
- sanitized failure signatures
- team-shareable software-work memory without raw code upload

## M18-M30 Roadmap

| Milestone | Focus |
|---|---|
| M18 | v0.1 hardening, strict release gate acceleration, public contribution surface |
| M19 | pain validation and paid-pilot gate |
| M20 | signed evidence pack protocol |
| M21 | passive capture hooks and wrappers |
| M22 | sanitized failure signature protocol |
| M23 | local immunity engine |
| M24 | self-hosted team memory registry |
| M25 | agent transcript firewall |
| M26 | resilience replay and local chaos sandbox |
| M27 | static audit reports and CI integration |
| M28 | interop and standardization preparation |
| M29 | commercial alpha packaging |
| M30 | beta release and selling motion |

## Non-Negotiable Rules

- No default telemetry.
- No automatic global upload.
- No raw code in shared failure signatures.
- No transcript claim becomes support without artifact evidence.
- No remote signature becomes positive support without local validation.
- No absolute security-guarantee wording.
- No cloud-only product direction.
- No feature that breaks the fresh-clone, no-API demo.

## M18 Entry Gate

M18 starts by making the release gate inspectable and fast:

```bash
python3 scripts/satlab.py release check --strict --profile
python3 scripts/satlab.py release check --strict --only benchmarks
python3 scripts/satlab.py release check --strict --only tests --timeout 180
```

Signed evidence and failure-memory features should wait until this baseline is stable.
