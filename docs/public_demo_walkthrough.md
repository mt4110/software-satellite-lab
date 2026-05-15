# Public Demo Walkthrough

This walkthrough is screenshot-free and text-only so it works in a fresh clone, terminal, or CI log.

## Setup

```bash
git clone <repo-url>
cd software-satellite-lab
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## No-Provider Demo

```bash
.venv/bin/python scripts/satlab.py release demo --no-api
```

Expected shape:

```text
# Public Demo Walkthrough
- No API key required.
- No network calls in the default path.
- No private design notes required.
- No trainable export artifacts are written.
```

The command writes Markdown and JSON reports under `artifacts/release_candidate/<workspace>/demo/`.

## Strict Release Check

```bash
.venv/bin/python scripts/satlab.py release check --strict
```

The strict check runs release documentation checks, review-memory benchmarks, Evidence Lint, pack audits, redaction checks, the fixture-backed demand and paid-pilot gates, and the public-demo default test gate under a timeout.

## Demand Gate

Fixture demo:

```bash
.venv/bin/python scripts/satlab.py demand gate --fixture-metrics examples/demand_gate/release_candidate_fixture.json --format md
```

Real local evidence:

```bash
.venv/bin/python scripts/satlab.py validation report --write --format md
.venv/bin/python scripts/satlab.py demand gate --format md
```

The fixture is only for demonstrating the gate. A public claim should be based on real dogfood review sessions, agent-session intakes, external technical-user inspections, and a timed fresh-clone demo attempt.

## Paid-Pilot Gate

Fixture demo:

```bash
.venv/bin/python scripts/satlab.py pilot report --fixture-records examples/pilot_evidence/passing_gate_records.jsonl --format md
```

Real local evidence:

```bash
.venv/bin/python scripts/satlab.py pilot record-interview --help
.venv/bin/python scripts/satlab.py pilot record-demo --help
.venv/bin/python scripts/satlab.py pilot record-loi --help
.venv/bin/python scripts/satlab.py pilot report --format md
```

The fixture is only for demonstrating the report boundary. A paid-pilot decision should be based on real discovery calls, hands-on demos, and LOIs or paid-pilot commitments.

## Demo Transcript

```text
$ .venv/bin/python scripts/satlab.py release demo --no-api
# Public Demo Walkthrough

This is a local, file-first release candidate demo. It uses public fixtures only.

## Guardrails

- No API key required.
- No network calls in the default path.
- No private design notes required.
- No trainable export artifacts are written.

## Transcript

1. Release checks finish with `pass`.
2. Demand gate fixture finishes with `pass`.
3. Paid-pilot gate fixture finishes with `pass`.
4. The operator reads the generated Markdown and only treats real demand and pilot records as decision evidence.
```

## Notes For Reviewers

- The default path does not fetch remote assets.
- The default path does not read private design notes.
- The default path does not create training data or fine-tuning exports.
- Generated reports are local artifacts and should be reviewed before sharing.
