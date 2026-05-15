# Contributing

Thanks for helping make software-satellite-lab sturdier.

This project is not a coding agent. It is a local-first evidence layer around AI-assisted software work. Contributions should make evidence easier to inspect, harder to overclaim, and safer to share.

## Ground Rules

- Keep default workflows local-first and file-first.
- Do not require API keys for public demo paths.
- Do not add default telemetry, raw code upload, training export, or hidden background execution.
- Do not let transcript claims become positive support without source-linked artifact evidence.
- Keep pack manifests declarative; arbitrary code execution, network access, secrets access, and repo writes remain denied by default.
- Add or update tests for every new evidence rule, gate, schema, or CLI behavior.

## Local Checks

```bash
PYTHONPATH=scripts python3 -m unittest tests.test_release_candidate_checks
PYTHONPATH=scripts python3 -m py_compile scripts/*.py tests/*.py
python3 scripts/satlab.py release check --docs --no-write
```

For the stricter public baseline:

```bash
python3 scripts/satlab.py release check --strict --profile --no-write
```

## Evidence Pack Contributions

Evidence Pack changes should include:

- a manifest update
- a strict audit expectation
- benchmark fixtures when behavior changes
- a note explaining why the pack output cannot bypass the support kernel

Start with `docs/evidence_pack_contributor_guide.md`.

## Roadmap

The current public roadmap pointer is `docs/commercial_oss_strategy_v4.md`.
