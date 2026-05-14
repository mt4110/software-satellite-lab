# Design Backlog Hard-Mode Closure

This public checklist tracks M10-M17 hard-mode and issue-backlog closure status.
It expands only the remaining implementation slices identified in this pass, and it is intentionally file-first without depending on private design notes.

## Invariants

- Local-first and file-first default paths.
- No API key required for default demos, checks, or fixtures.
- No network calls in default paths.
- Public fixtures only.
- No trainable export artifacts.
- No provider-hub behavior.

## Status Summary

| Area | Implemented baseline | Remaining backlog |
| --- | --- | --- |
| M10 Artifact Vault + Support Kernel | `scripts/artifact_vault.py`, `scripts/evidence_support.py`, `artifact capture`, `artifact inspect`, `evidence support`, symlink/path traversal refusal, content-addressed vault objects | `artifact gc --dry-run`; support policy JSON registry and summary report |
| M11 Derived Evidence Graph + Lint | `evidence graph`, `evidence lint`, `evidence trace --why-blocked`, `evidence impact`, graph schema validation, target fingerprints | graph diff between two snapshots |
| M12 Adversarial Review Memory Benchmark | `review eval`, `review miss-report`, `review benchmark --spartan`, synthetic/adversarial fixtures, miss taxonomy, dogfood metrics slot | review fixture generator; dogfood importer; benchmark trend report; eval compare; target-fingerprint ablation |
| M13 Cross-Agent Intake | `intake agent-session`, `intake pr-bundle`, agent-session bundle fixtures, redaction and no-network tests | no additional hard-mode closure item identified in this pass |
| M14 Evidence Pack v1 Policy Kernel | `pack audit`, `pack test`, `pack lock`, `pack scaffold`, `pack list --builtin`, strict denied-field checks, built-in packs | pack fixture generator; minimal `pack explain`; schema docs generated from JSON schema |
| M15 Backend Adoption Dossier | `backend dossier`, strict benchmark-report consumption, insufficient-evidence behavior, schema validation, markdown/JSON output | dossier comparison across two runs; adoption dossier fixture in public demo |
| M16 Release Candidate + Demand Gate | `release check`, `release demo --no-api`, `demand gate`, public demo walkthrough and fixture-backed gates | no additional hard-mode closure item identified in this pass |
| M17 Research Reproducibility + Standardization Prep | `research pack`, `research reproduce`, `schema coverage`, contributor guide, research reproducibility docs | no additional hard-mode closure item identified in this pass |

## PR Split

Only milestones with remaining implementation slices are expanded below. Milestones marked as having no additional closure item in the status summary are intentionally not repeated as PRs.

### PR 1: M10 Artifact GC Dry Run

Goal: add a non-destructive inventory command for stale vault objects.

Scope:

- Add artifact-vault inventory helpers that read `artifacts/vault/refs/*.json` and `artifacts/vault/objects/**`.
- Add `python3 scripts/satlab.py artifact gc --dry-run --format text|json`.
- Report referenced objects, unreferenced objects, missing objects, malformed refs, and reclaimable byte count.
- Keep deletion out of scope.
- Add tests in `tests/test_artifact_vault.py`.

Acceptance:

- Dry run never removes files.
- Malformed refs are reported but do not crash the whole scan.
- Output is deterministic for fixture roots.

### PR 2: M10 Support Policy Registry

Goal: make support-kernel rules inspectable without changing the default decision behavior.

Scope:

- Add a public JSON registry, for example `configs/evidence_support_policies/v1.json`.
- Add a schema or validator for support classes, polarities, blocker reasons, and decision requirements.
- Add `python3 scripts/satlab.py evidence policy --format text|json`.
- Include a compact summary report that maps each support class to allowed decision use.
- Add tests in `tests/test_evidence_support.py`.

Acceptance:

- Registry mirrors the current hard-coded support behavior.
- Default `evidence support` still works without reading network or private files.
- Invalid registry examples fail validation in tests.

### PR 3: M11 Evidence Graph Diff

Goal: compare two derived graph snapshots without treating either snapshot as source of truth.

Scope:

- Add `diff_evidence_graph_snapshots(before, after)`.
- Add `python3 scripts/satlab.py evidence graph-diff --before before.json --after after.json --format md|json`.
- Compare nodes, edges, support class changes, causal validity changes, and decision-support changes.
- Add tests in `tests/test_evidence_graph.py`.

Acceptance:

- Diff output is stable and sorted.
- Added, removed, and changed graph records are visible.
- The command reads snapshots only and does not rebuild or mutate event logs.

### PR 4: M12 Review Fixture Generator

Goal: reduce hand-written fixture drift while keeping fixtures public and deterministic.

Scope:

- Add a generator for a minimal review-memory fixture suite from a small JSON seed.
- Add `python3 scripts/satlab.py review fixture-generate --output <path>`.
- Support synthetic fixture categories only in this PR.
- Add generated-fixture validation tests.

Acceptance:

- Generated suite passes `review eval` when using the default safe seed.
- Generated content has no private paths, no credentials, and no network references.

### PR 5: M12 Eval Compare And Trend Report

Goal: make benchmark movement reviewable across runs.

Scope:

- Add `python3 scripts/satlab.py review eval --compare previous.json current.json`.
- Add a trend report over local `artifacts/review_memory_benchmark/<workspace>/runs/*.json`.
- Compare gate status, result digest, fixture pass/fail changes, critical false support, Recall@5, no-evidence honesty, and miss reasons.
- Add tests in `tests/test_review_memory_eval.py`.

Acceptance:

- Compare works on two JSON files without needing latest artifacts.
- Trend report handles zero or one run gracefully.
- No private data is included in report output.

### PR 6: M12 Dogfood Import Stub

Goal: convert local dogfood verdict artifacts into a dogfood fixture suite shape without pretending the suite is sufficient.

Scope:

- Add importer from local validation/review artifacts into `suite_kind: dogfood`.
- Add `python3 scripts/satlab.py review dogfood-import --output <path>`.
- Mark low-volume imports with `needs_20_runs` status.
- Add tests with public fixture ledgers only.

Acceptance:

- Importer does not read private docs or external services.
- Missing source artifacts are represented as blocked fixture candidates.
- Holdout policy metadata remains visible.

### PR 7: M12 Target-Fingerprint Ablation

Goal: measure path-only target identity versus path-plus-hunk identity.

Scope:

- Add an ablation mode to review-memory eval.
- Report path-only and path-plus-hunk metrics side by side.
- Keep ranking and support decisions deterministic.
- Add focused tests around self-recall and weak-match fixtures.

Acceptance:

- Ablation report cannot change default benchmark pass/fail behavior.
- Report identifies any fixture whose result changes under the alternate fingerprint mode.

### PR 8: M14 Pack Explain

Goal: give contributors a small, readable explanation of why a pack passes or is blocked.

Scope:

- Add `python3 scripts/satlab.py pack explain <pack> --format md|json`.
- Summarize manifest identity, selected roots, allowed transforms, denied-field checks, lock status, support policy, and fixture expectations.
- Reuse existing audit/test internals.
- Add tests in `tests/test_evidence_pack_v1.py`.

Acceptance:

- Explain is read-only.
- Explain works for valid built-in packs and draft blocked packs.
- Output does not include raw secret-like values from denied evidence.

### PR 9: M14 Pack Fixture Generator

Goal: make safe Evidence Pack v1 benchmark fixtures easier to create without adding executable pack behavior.

Scope:

- Add a generator for safe benchmark fixture entries inside a pack manifest.
- Add tests in `tests/test_evidence_pack_v1.py`.

Acceptance:

- Fixture generator emits public, local-only fixture text.
- Generated fixture entries pass the existing pack audit and test runner.
- No executable/plugin semantics are introduced.

### PR 10: M14 Pack Schema Docs

Goal: make the Evidence Pack v1 JSON schema readable as public contributor documentation.

Scope:

- Add schema-doc generation from `schemas/satellite_evidence_pack_v1.schema.json`.
- Write generated docs under `docs/` and keep them deterministic.
- Add tests that fail when generated docs drift from the schema.

Acceptance:

- Generated docs are reproducible.
- Generated docs do not add policy rules beyond the JSON schema and audit kernel.
- The docs path is public and does not depend on generated artifact directories.

### PR 11: M15 Dossier Compare

Goal: compare adoption dossiers across two runs without re-running backend checks.

Scope:

- Add `python3 scripts/satlab.py backend dossier-compare --before before.json --after after.json --format md|json`.
- Compare recommendation, blockers, warnings, benchmark gate, lint gate, source-linked outcomes, cost/latency metadata, and rollback state.
- Add tests in `tests/test_backend_adoption_dossier.py`.

Acceptance:

- Diff is stable and sorted.
- Recommendation changes are called out first.
- Compare reads local JSON only and performs no provider calls.

### PR 12: M15 Adoption Dossier Public Demo Fixture

Goal: make backend adoption dossier behavior visible in the no-provider public demo path.

Scope:

- Add a public adoption-dossier fixture under `examples/`.
- Extend `docs/public_demo_walkthrough.md` with a dossier command and expected short output.
- Optionally add a release-demo check that verifies the fixture command runs without API keys.
- Add focused tests around fixture readability and command exit status.

Acceptance:

- Public demo still runs with `release demo --no-api`.
- Fixture contains no private docs, private paths, API keys, or network references.
- The demo shows both an adoptable and an insufficient-evidence case, or clearly links to the insufficient-evidence fixture.

## Recommended Order

1. PR 1 and PR 2 close M10 safety/reporting gaps.
2. PR 3 closes the only M11 hard-mode gap.
3. PR 4 can land independently before or after PR 5 because the generator output is validated by the existing benchmark runner.
4. PR 5 should land before PR 6 and PR 7 because compare/trend output gives the dogfood and ablation work a review surface.
5. PR 8 should land before PR 9 and PR 10 so fixture and schema docs can point to the same explanation model.
6. PR 11 should land before PR 12 so the public demo can include a comparison story if needed.

## Out Of Scope For This Closure

- Enabling destructive artifact deletion.
- Pulling dogfood data from network services.
- Adding provider-hub integrations.
- Promoting any review, pack, or dossier artifact into trainable exports.
- Changing default backend behavior.
