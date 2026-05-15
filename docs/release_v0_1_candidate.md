# v0.1 Release Candidate

This is the first OSS release candidate for a local-first AI Coding Flight Recorder.

It turns AI-assisted software work into source-linked evidence: local artifacts, review verdicts, recall results, benchmark checks, and demand validation reports that can be inspected without provider accounts.

## What This Is

- A file-first evidence ledger for AI-assisted software work.
- A reproducible demo of failure-memory review, evidence support, pack policy audit, and demand-gate boundaries.
- A local CLI surface for release checks:

```bash
python3 scripts/satlab.py release check --strict
python3 scripts/satlab.py release check --strict --profile
python3 scripts/satlab.py release check --strict --only benchmarks
python3 scripts/satlab.py release check --strict --only tests --timeout 180
python3 scripts/satlab.py release demo --no-api
python3 scripts/satlab.py validation report --write --format md
python3 scripts/satlab.py demand gate --format md
```

## What This Is Not

- It is not a coding agent runtime.
- It is not a model training pipeline.
- It is not a cloud service.
- It is not a promise that synthetic fixture demand equals real market demand.

## Quickstart

English:

```bash
git clone <repo-url>
cd software-satellite-lab
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/satlab.py release demo --no-api
```

日本語:

```bash
git clone <repo-url>
cd software-satellite-lab
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/satlab.py release demo --no-api
```

No API key required. The default demo uses public fixtures and local files only.

## Demo Workflow

The public demo path is intentionally narrow:

1. Run release candidate checks.
2. Run deterministic review-memory benchmarks.
3. Run Evidence Lint.
4. Audit built-in Evidence Pack v1 manifests in strict mode.
5. Run a fixture-backed demand gate to prove the pass/fail boundary.
6. Read the generated Markdown report.

The demo does not call a provider, fetch remote assets, or read private notes.

## Evidence Support Invariants

- The active review subject is excluded from prior evidence.
- Missing, modified, future, contradictory, weak, and unverified evidence cannot become positive support.
- Human verdicts stay explicit and source-linked.
- Pack execution is declarative and policy-gated; arbitrary pack code is not enabled.
- Learning-candidate inspection is preview-only and does not write trainable export artifacts.

## Benchmark Summary

Release checks run:

- deterministic evidence-gated review benchmark,
- adversarial review-memory benchmark,
- Evidence Lint strict gate,
- Evidence Pack v1 strict audits,
- redaction fixtures,
- the public-demo default test gate under a timeout when `--strict` is used.

The benchmark report is generated during the check, so stale benchmark evidence is not accepted as a release pass.

## Strict Gate Slices

The strict release check can be split when a contributor needs a faster or more focused signal:

```bash
python3 scripts/satlab.py release check --docs
python3 scripts/satlab.py release check --benchmarks
python3 scripts/satlab.py release check --packs
python3 scripts/satlab.py release check --tests --timeout 180
```

Use `--profile` to include timing, slow test, and slow pack-audit details in the release report.

## Demand Validation Summary

The `demand gate` command enforces the v0.1 release-message boundary:

- dogfood review sessions >= 20,
- dogfood agent-session intakes >= 5,
- external technical-user inspections >= 3,
- external fresh-clone demo attempts >= 1,
- useful recall at 5 >= 0.30,
- critical false support count = 0,
- verdict capture median seconds <= 30,
- external exact-pain recognition >= 2,
- external wants-to-try >= 1,
- fresh-clone demo minutes <= 15.

The public demo includes a synthetic fixture so the gate can be demonstrated without private data. Real release messaging should use local dogfood and external-user records, not the fixture.

## Security/Privacy Caveats

- Release reports redact common secret-shaped tokens in rendered excerpts, but this is a guardrail, not a data-loss-prevention system.
- Local artifacts can contain private code or user text. Do not upload generated artifact directories without review.
- The default demo requires no API key and performs no network calls.
- The CLI does not require private design notes to run or understand the demo.

## Known Limitations

- Recall is lexical and structured first; vector search is intentionally out of scope for this candidate.
- Demand validation is only as strong as the recorded dogfood and external-user evidence.
- Redaction is best-effort and should not be treated as a guarantee.
- The release candidate does not include hosted UX polish; the CLI is the canonical public surface.

## Next Milestone

M18 hardens this release candidate into a stable public baseline: named strict gates, timing/profile output, issue templates, package metadata, and a contribution/security surface that does not depend on private design notes.
