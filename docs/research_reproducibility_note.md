# Research Reproducibility Note

M17 packages the project as a local, file-first research artifact that an external reviewer can inspect without private notes, API keys, network calls, or trainable export files.

## Reproduce

```bash
python3 scripts/satlab.py research pack --output artifacts/research_pack
python3 scripts/satlab.py research reproduce --pack artifacts/research_pack/latest
python3 scripts/satlab.py schema coverage --format md
```

The research pack writes a timestamped run directory and refreshes `artifacts/research_pack/latest` as the reproducible entrypoint.

## Pack Contents

- `README.md`: one-sentence project summary plus guardrails.
- `demo_artifacts/`: public demo docs and demand-gate fixture.
- `benchmark_fixtures/`: public review-memory fixture suite and built-in Evidence Pack manifests.
- `contributor_materials/`: Evidence Pack contributor guide, schema compatibility matrix, and public `software_work_event` gallery.
- `benchmark_results.json`: stable summary of deterministic benchmark outcomes.
- `evidence_graph_snapshot.json`: derived graph snapshot for the public fixture scope.
- `evidence_lint_report.md`: local evidence lint result.
- `pack_audit_report.md`: strict audit summary for built-in Evidence Pack v1 templates.
- `release_check_report.md`: static release-candidate gate report.
- `schema_coverage_report.md`: standardization coverage for the core open schema candidates.
- `limitations.md`: reproducibility limits and preserved research questions.
- `checksums.json`: deterministic SHA-256 manifest for generated contents.

## Exit Gate

The reproduction command reports these fields:

```text
research_pack_reproduces = true
private_doc_dependency_count = 0
schema_coverage_core >= 0.90
benchmark_results_included = true
limitations_included = true
no_trainable_export = true
```

## Boundary

The pack intentionally includes public fixtures and summarized reports only. It does not include local user artifacts, private planning notes, provider credentials, live integration output, or supervised-training export material.

## Release Artifact Preparation

For a future GitHub or Zenodo release, attach the generated `artifacts/research_pack/latest` directory as a static archive only after `research reproduce` passes locally. Do not publish automatically from the CLI.
