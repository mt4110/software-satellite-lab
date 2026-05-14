# Evidence Pack Contributor Guide

Satellite Evidence Packs are declarative, local-first bundles for turning software-work artifacts into inspectable evidence. This guide is for external contributors who want to add or improve packs without needing private design notes, API keys, network calls, or generated training artifacts.

## Contribution Boundary

Good pack contributions stay inside this boundary:

- local files only in the default path
- public fixtures only
- no API key requirement
- no network calls
- no executable runtime in the pack manifest
- no secret, credential, or environment-variable access
- no repo mutation from pack execution
- no install scripts or auto-update hooks
- no trainable export artifacts
- support-gated output for any decision-facing evidence

If a contribution needs to cross one of those lines, it is not an Evidence Pack v1 contribution. Write a design note first and keep the manifest out of the runnable path.

## Good First Contributions

- Add a small benchmark fixture to an existing built-in pack.
- Add a new report section that is backed by an existing core transform.
- Improve pack metadata, summaries, or examples.
- Add a `software_work_event` example to the public gallery.
- Propose an additive schema note in `docs/schema_changelog_and_compatibility.md`.

## Before You Edit

Read these files first:

- `docs/satellite_evidence_pack_contract.md`
- `docs/software_work_event_schema_notes.md`
- `docs/schema_changelog_and_compatibility.md`
- `schemas/satellite_evidence_pack_v1.schema.json`
- `templates/failure-memory-pack.satellite.yaml`
- `templates/agent-session-pack.satellite.yaml`

The quickest orientation command is:

```bash
python3 scripts/satlab.py pack list --builtin
```

## Draft Loop

Start from a built-in template:

```bash
python3 scripts/satlab.py pack scaffold --kind failure-memory --output scratch/failure-memory-pack.satellite.yaml
```

Edit the manifest, then run the local checks:

```bash
python3 scripts/satlab.py pack audit scratch/failure-memory-pack.satellite.yaml --strict
python3 scripts/satlab.py pack test scratch/failure-memory-pack.satellite.yaml --strict
```

When the manifest is stable enough for review, write a lockfile:

```bash
python3 scripts/satlab.py pack lock scratch/failure-memory-pack.satellite.yaml
```

Lockfiles are mutation detectors. Refresh the lock only after intentional manifest changes.

## Manifest Checklist

Every Evidence Pack v1 manifest must include:

- `schema_name: software-satellite-evidence-pack-v1`
- `schema_version: 1`
- `metadata` with a stable `pack_id`, `version`, `title`, and `summary`
- `input_kinds` from the allowlist
- `artifact_policy` with selected public roots, refused links, and blocked missing sources
- `recall_policy` that requires source references
- `support_policy` that requires `evidence_support_v1`
- `report_sections` sourced from known report inputs
- `benchmark_fixtures` with expected support outcomes
- `redaction_policy` set to best-effort local redaction
- `output_schema_refs` from the allowlist
- `core_transform_refs` from the allowlist

Unknown fields are blocked in strict audit mode. That is intentional: readers should be able to inspect a pack without guessing what a field might do.

## Fixture Rules

Fixtures should be synthetic or explicitly public. Keep them short and boring enough to review by eye.

Use fixture text for input data only. Do not put executable behavior, URLs, credential-shaped tokens, install instructions, or environment-variable access into policy fields. If a fixture needs a redaction probe, keep it in a dedicated test fixture and verify that reports redact it.

## Schema Changes

Most schema work should be additive:

- adding optional fields under an existing object is usually compatible
- adding a new enum value can be compatible only when old readers fail closed
- renaming or removing a required field is breaking
- changing support semantics is breaking even if the JSON shape is unchanged

Record proposed schema changes in `docs/schema_changelog_and_compatibility.md`. Include the affected schema, compatibility class, reader impact, migration note, and a fixture or example path.

## Review Checklist

Before opening a change, run:

```bash
python3 scripts/satlab.py pack audit templates/failure-memory-pack.satellite.yaml --strict
python3 scripts/satlab.py pack audit templates/agent-session-pack.satellite.yaml --strict
python3 scripts/satlab.py pack test templates/failure-memory-pack.satellite.yaml --strict
python3 scripts/satlab.py pack test templates/agent-session-pack.satellite.yaml --strict
python3 scripts/satlab.py schema coverage --format md
python3 scripts/satlab.py research pack --output artifacts/research_pack
```

For code or schema changes, also run the focused unit tests for the touched module.

## What To Include

A strong contribution includes:

- the manifest or fixture change
- the local command output summary
- any lockfile update, when intentional
- docs or gallery examples for new concepts
- a note explaining why the change remains local-first and file-first

Keep attribution out of generated text unless explicitly requested. The artifact should describe what changed and why it is inspectable.
