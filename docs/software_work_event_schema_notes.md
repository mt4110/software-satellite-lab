# Software Work Event Schema Notes

These notes prepare the minimal open schema surface for research review. The goal is standardization pressure, not premature protocol freeze.

## software_work_event

### Required fields

- `schema_name`
- `schema_version`
- `event_id`
- `event_kind`
- `recorded_at_utc`
- `workspace`
- `session`
- `outcome`
- `content`
- `source_refs`
- `tags`

### Optional fields

- `target_paths`
- `quality_status`
- `execution_status`
- `artifact_vault_refs`
- `note`

### Compatibility policy

Additive fields are allowed while the record stays file-first and keeps source references durable. Renaming required fields needs a new schema version and a migration note.

### Privacy considerations

Events may point at local source files or logs, so portable exports should carry artifact references and redacted excerpts rather than raw private code by default.

### Examples

```json
{
  "schema_name": "software-satellite-event",
  "schema_version": 1,
  "event_id": "event_demo_patch",
  "event_kind": "patch_input",
  "recorded_at_utc": "2026-05-14T00:00:00+00:00",
  "workspace": {"workspace_id": "public-fixture"},
  "session": {"session_id": "demo", "surface": "cli"},
  "outcome": {"status": "needs_review"},
  "content": {"notes": ["public fixture patch"]},
  "source_refs": {},
  "tags": ["fixture"]
}
```

### Known gaps

There is not yet a standalone JSON Schema file for this event shape; the runtime contract and event-contract report are the current source of truth.

## artifact_ref

### Required fields

- `schema_name`
- `schema_version`
- `artifact_id`
- `kind`
- `original_path`
- `vault_path`
- `repo_relative_path`
- `sha256`
- `size_bytes`
- `mime_hint`
- `source_state`
- `capture_state`
- `git`
- `redaction`
- `report_excerpt`
- `captured_at_utc`

### Optional fields

The JSON Schema currently marks all top-level fields as required. Future additive metadata should remain nested under a versioned extension field or trigger a schema version bump.

### Compatibility policy

Artifact refs are content-addressed records. Existing checksums and capture states must remain stable across readers; new artifact kinds should be additive enum changes.

### Privacy considerations

The ref can include local paths and excerpts. Excerpts use best-effort redaction and must not be treated as upload-safe without review.

### Examples

```json
{
  "schema_name": "software-satellite-artifact-ref",
  "schema_version": 1,
  "artifact_id": "artifact_demo",
  "kind": "review_note",
  "original_path": "examples/demo.md",
  "vault_path": null,
  "repo_relative_path": "examples/demo.md",
  "sha256": null,
  "size_bytes": null,
  "mime_hint": "text/markdown",
  "source_state": "present",
  "capture_state": "ref_only",
  "git": {"commit": null, "blob_sha": null, "dirty_tree": false},
  "redaction": {"applied": false, "secret_like_tokens": 0, "long_lines_truncated": 0, "binary_bytes_refused": 0, "rules_version": 1},
  "report_excerpt": {"text": "public fixture", "best_effort_redaction": true, "never_upload_notice": "review before upload"},
  "captured_at_utc": "2026-05-14T00:00:00+00:00"
}
```

### Known gaps

The top-level schema is intentionally strict, but extension guidance for new artifact kinds needs a changelog once external contributors appear.

## human_verdict

### Required fields

- `schema_name`
- `schema_version`
- `workspace_id`
- `recorded_at_utc`
- `event_id`
- `verdict`
- `reason`
- `signal`

### Optional fields

- `target_event_id`
- `relation_kind`
- `follow_up`
- `recall_usefulness`
- `paths`
- `snapshot`
- `learning_preview`

### Compatibility policy

Verdict labels are a small controlled vocabulary. New labels must map to an evaluation signal kind and preserve old verdict meanings.

### Privacy considerations

The human rationale can contain project-sensitive judgement. Exports should keep the rationale concise and avoid copying private code or customer text.

### Examples

```json
{
  "schema_name": "software-satellite-human-verdict-record",
  "schema_version": 1,
  "workspace_id": "public-fixture",
  "recorded_at_utc": "2026-05-14T00:00:00+00:00",
  "event_id": "event_demo_patch",
  "verdict": "reject",
  "reason": "The recalled evidence is missing its source file.",
  "signal": {"signal_kind": "rejection"}
}
```

### Known gaps

There is not yet a dedicated JSON Schema file for verdict records; the CLI writer and evaluation signal schema define the runtime contract.

## evidence_support_result

### Required fields

- `schema_name`
- `schema_version`
- `event_id`
- `support_class`
- `can_support_decision`
- `support_polarity`
- `blockers`
- `warnings`
- `artifact_refs`
- `active_review_excluded`
- `checked_at_utc`

### Optional fields

The current JSON Schema is strict at the top level. Future explanatory fields should be additive in a new schema version or nested under a typed diagnostics object.

### Compatibility policy

Support classes are policy-bearing values. A reader must treat unknown classes as non-supporting until explicitly upgraded.

### Privacy considerations

Support results should reference artifact ids and blocker reasons, not reproduce full source artifacts.

### Examples

```json
{
  "schema_name": "software-satellite-evidence-support-result",
  "schema_version": 1,
  "event_id": "event_prior_failure",
  "support_class": "negative_prior",
  "can_support_decision": true,
  "support_polarity": "risk",
  "blockers": [],
  "warnings": [],
  "artifact_refs": ["artifact_demo"],
  "active_review_excluded": false,
  "checked_at_utc": "2026-05-14T00:00:00+00:00"
}
```

### Known gaps

The schema does not yet describe target identity or freshness diagnostics; those live in derived graph metadata.

## review_memory_fixture

### Required fields

- `schema_name`
- `schema_version`
- `suite_id`
- `suite_kind`
- `fixtures`

### Optional fields

- `requested_polarity`
- `candidate_events`
- `expected`
- fixture-level diagnostic fields such as `category`, `description`, and source behavior metadata

### Compatibility policy

Fixture categories are additive when old benchmark semantics remain unchanged. Breaking fixture expectation semantics requires a suite version bump.

### Privacy considerations

Research fixtures must be synthetic or explicitly public dogfood artifacts. Secret-shaped strings are allowed only as redaction probes.

### Examples

```json
{
  "schema_name": "software-satellite-review-memory-fixture-suite",
  "schema_version": 1,
  "suite_id": "synthetic-public",
  "suite_kind": "synthetic",
  "fixtures": [
    {
      "fixture_id": "no_prior",
      "category": "no_prior_evidence",
      "description": "No source-linked prior should be promoted.",
      "query": "review cache patch",
      "review_started_at_utc": "2026-05-14T00:00:00+00:00",
      "current_event": {"event_id": "event_current", "target_paths": ["app.py"]},
      "expected": {"no_strong_evidence": true}
    }
  ]
}
```

### Known gaps

Fixture coverage is still synthetic-heavy. More public, consented dogfood cases are needed before claiming external benchmark strength.

## agent_session_bundle

### Required fields

- `schema_name`
- `schema_version`
- `agent_label`
- `task`
- `artifacts`
- `privacy`

### Optional fields

- `session_started_at_utc`
- `session_finished_at_utc`
- `declared_claims`
- task details such as `goal`
- artifact metadata beyond `kind` and `path`

### Compatibility policy

New agent labels and artifact kinds should be additive. Unknown labels must normalize to `unknown` rather than enabling live provider behavior.

### Privacy considerations

Bundles may contain transcripts, diffs, and user text. Export is controlled by the `privacy.export_allowed` flag and should default conservative.

### Examples

```json
{
  "schema_name": "software-satellite-agent-session-bundle",
  "schema_version": 1,
  "agent_label": "generic",
  "task": {"title": "Review local patch"},
  "artifacts": [{"kind": "diff", "path": "examples/agent_session_bundles/files/generic.diff"}],
  "privacy": {"contains_private_code": false, "contains_user_text": false, "export_allowed": true}
}
```

### Known gaps

The bundle normalizes files, not live integrations. Provider-specific session metadata remains intentionally out of scope.

## satellite_evidence_pack_v1

### Required fields

- `schema_name`
- `schema_version`
- `artifact_policy`
- `benchmark_fixtures`
- `core_transform_refs`
- `input_kinds`
- `metadata`
- `output_schema_refs`
- `recall_policy`
- `redaction_policy`
- `report_sections`
- `support_policy`

### Optional fields

`metadata.license` is optional. The top-level manifest is otherwise strict to keep pack behavior auditable.

### Compatibility policy

Evidence Pack v1 allows declarative manifest changes only inside known fields. New core transforms, input kinds, or output schemas require allowlist and schema updates.

### Privacy considerations

Packs may read only selected local roots, refuse links, block missing sources, avoid network calls, and keep learning export disabled.

### Examples

```json
{
  "schema_name": "software-satellite-evidence-pack-v1",
  "schema_version": 1,
  "metadata": {"pack_id": "public-demo-pack", "version": "1.0.0", "title": "Public Demo Pack", "summary": "Inspect public fixtures."},
  "input_kinds": ["failure"],
  "artifact_policy": {"selected_roots": ["examples/review_memory_benchmark"], "path_mode": "selected_roots_only", "link_policy": "refuse_links", "missing_source_policy": "block"},
  "recall_policy": {"mode": "failure_memory", "max_items": 5, "require_source_refs": true},
  "support_policy": {"required_kernel": "evidence_support_v1", "requested_polarity": "risk", "require_source_linked_evidence": true, "require_support_kernel": true},
  "report_sections": [{"id": "summary", "title": "Summary", "source": "support_kernel"}],
  "benchmark_fixtures": [{"id": "fixture", "input_kind": "failure", "artifact_name": "failure_log", "fixture_text": "public fixture", "requested_polarity": "risk", "expected_support_class": "negative_prior", "expected_can_support_decision": true}],
  "redaction_policy": {"mode": "best_effort_local", "token_handling": "redact_known_tokens"},
  "output_schema_refs": ["evidence_support.schema.json"],
  "core_transform_refs": ["artifact_capture", "support_gate"]
}
```

### Known gaps

There is no arbitrary command runtime, marketplace, network tier, or trainable export path in v1. Those are deferred until audit and consent models are stronger.
