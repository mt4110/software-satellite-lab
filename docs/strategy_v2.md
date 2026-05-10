# Strategy v2: AI Coding Flight Recorder

`software-satellite-lab` is a local-first AI Coding Flight Recorder.

It records, compares, recalls, and curates source-linked evidence from AI-assisted software work.

Longer definition:

```text
software-satellite-lab is a local-first, file-first software-work evidence ledger for AI-assisted development.
It makes agent runs, reviews, failures, repairs, backend comparisons, and human verdicts inspectable, reusable, and safe to inspect for future learning preparation.
```

## What It Is

- local-first evidence ledger
- failure-memory system
- source-linked recall layer
- backend / proposal comparison record
- human verdict capture system
- curation / learning-candidate inspection layer
- safe declarative Satellite Evidence Pack system

## What It Is Not

- not a coding agent
- not an IDE assistant
- not a provider hub
- not a PR-review replacement
- not a general LLM observability platform
- not an MCP replacement
- not an Agent Skills marketplace
- not an auto-training loop

## First Public Problem Statement

```text
AI coding tools generate more work than humans can remember or audit.
software-satellite-lab remembers which AI-assisted changes passed, failed, were rejected, or were accepted, with source-linked evidence.
```

## First User Sentence

```text
For developers using multiple AI coding tools who need to stop repeated failures and keep local, inspectable evidence of reviews, repairs, comparisons, and human decisions.
```

## Core Loop

```text
input patch / output / failure
  -> normalize software_work_event
  -> recall similar failures and decisions
  -> produce risk note
  -> compare candidate outputs if present
  -> require human verdict
  -> write evidence record
  -> show curation / learning-candidate inspection
```

## Product Guardrail

If a feature does not improve source-linked evidence, failure memory, human verdict capture, or comparison reuse, it does not belong in the first product wave.
