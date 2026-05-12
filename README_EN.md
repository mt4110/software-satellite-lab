# software-satellite-lab

[日本語 README](./README.md)

A local-first AI Coding Flight Recorder.

It records reviews, failures, repairs, comparisons, and human verdicts from AI-assisted software work as source-linked evidence that can be inspected, compared, recalled, and safely considered for future learning-candidate inspection.

It is not a coding agent. It is the evidence layer around coding agents.

## Goals

- remember why AI coding work failed, passed, was accepted, or was rejected
- recall similar failures and human verdicts
- preserve backend and proposal comparisons with source paths
- keep learning-candidate inspection preview-only and human-gated
- inspect evidence from any agent or backend without becoming another agent runtime

## What It Builds

The core target is not a single model or an agent runtime.
The target is a local software-work evidence ledger for AI-assisted development.

The current design is organized around five functions:

1. event and artifact capture
2. memory and retrieval
3. failure-memory review
4. backend and proposal comparison
5. human-gated learning-candidate inspection

## Design Stance

- local-first
- file-first
- simple infrastructure first
- evidence before training
- swappable model backends
- inspectable outputs over hidden automation

## Storage Stance

The initial storage direction is:

- local files as durable truth
- `SQLite` as the first indexed memory layer
- lexical and structured retrieval before vector-heavy expansion
- heavier infrastructure only when real pressure appears

## Current Base Assets

This repo already contains reusable foundations:

- shared runtime and session management
- file-based workspace and session state
- schema-versioned artifacts
- capability services
- a thin local UI
- an evaluation harness
- a software-work event normalization layer
- a SQLite memory-index foundation

These are treated as baseline assets for the redesign, not as throwaway work.

## Documents

- `README.md`: Japanese overview
- `README_EN.md`: English overview
- `PLAN.md`: redesign blueprint and milestones
- `docs/strategy_v2.md`: v2 strategy around the AI Coding Flight Recorder wedge
- `docs/satellite_evidence_pack_contract.md`: safety contract for Satellite Evidence Packs
- `docs/failure_memory_review_demo.md`: first failure-memory review demo spec
- `docs/demand_validation_demo_kit.md`: demand-validation and dogfood measurement kit after the public demo
- `docs/recall_context_builder_design.md`: first implementation design for Recall / Context Builder
- `docs/recall_hit_quality_loop.md`: hit-quality visualization and lightweight evaluation loop for Recall
- `docs/learning_finetune_prep_design.md`: M7 preview-only dataset candidate design for Learning and Fine-Tune Prep
- `docs/evidence_gated_git_review_workbench_design.md`: M9 Evidence-Gated Git Review Workbench strategy hardening and two-week implementation design

## Evidence-Gated Git Review

```bash
python3 scripts/satlab.py review git --base origin/main --head HEAD
python3 scripts/satlab.py review verdict --from-latest --decision needs_fix --rationale "Needs one more verification step" --recall-usefulness useful
python3 scripts/satlab.py review benchmark
```

- the active patch is excluded from prior evidence
- weak, missing-source, and contradictory evidence can be shown, but never as positive support
- the benchmark runs without API keys and checks self-recall, no-prior-evidence, missing-source, and secret redaction

## Git Rules

- use `feat/<topic>` for feature branches
- avoid tool-name prefixes in branch names, PR titles, and commit messages

## Newly Added Foundations

- `scripts/software_work_events.py`
  - normalizes workspace and session records into software-work events
- `scripts/memory_index.py`
  - local `SQLite FTS5` memory index foundation
- `scripts/rebuild_memory_index.py`
  - rebuild command for the event log and memory index
- `scripts/evaluation_loop.py`
  - foundation for evaluation signals, comparisons, curation filters, typed learning review queues, learning dataset previews, and evaluation snapshots
- `scripts/run_evaluation_loop.py`
  - CLI for writing evaluation snapshots and preview-only artifacts
- `scripts/agent_lane.py`
  - M5 foundation for file-first bounded software task/run schemas, tool traces, and verification outcomes
- `scripts/run_agent_lane.py`
  - CLI for recording patch-plan-verify tasks and connecting verification traces to software-work events and the evaluation loop
- `scripts/backend_swap.py`
  - M6 foundation for file-first backend adapter configs, capability metadata, compatibility checks, and side-by-side harness runs
- `scripts/run_backend_swap.py`
  - CLI for running two or more backend configs through the same agent-lane workflow and connecting results to the memory index and evaluation comparisons
- `scripts/dogfood_workflows.py`
  - preview-only workflow layer that connects recall, evaluation, and curation preview for small repeated software-work loops
- `scripts/run_dogfood_workflow.py`
  - CLI launcher for patch review, proposal comparison, prior-failure recall, decision explanation, and resolved-work curation preview
- `scripts/satellite_pack.py`
  - foundation for Satellite Evidence Pack manifest loading, v0 schema validation, and permission audit artifacts
- `scripts/demand_validation.py`
  - public-demo validation kit that records dogfood runs, external interviews, and clone-to-demo timing into local file ledgers and reports
- `scripts/satlab.py`
  - thin CLI for `event ingest`, `recall failure`, `compare proposals`, `verdict`, `report latest`, `learning inspect --preview-only`, `pack inspect|audit|run review-risk-pack`, and `validation`

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Common Commands

```bash
.venv/bin/python scripts/doctor.py
.venv/bin/python scripts/fetch_demo_assets.py
PYTHONPATH=scripts .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
PYTHONPATH=scripts .venv/bin/python -m py_compile scripts/*.py tests/*.py
.venv/bin/python scripts/rebuild_memory_index.py
.venv/bin/python scripts/run_recall_demo.py --list-requests
.venv/bin/python scripts/run_recall_demo.py --eval --miss-report
.venv/bin/python scripts/run_recall_demo.py --request-index 1
.venv/bin/python scripts/run_evaluation_loop.py
.venv/bin/python scripts/run_evaluation_loop.py --accept-candidate --source-event-id <event-id> --rationale "Accepted after review." --curation-preview
.venv/bin/python scripts/run_evaluation_loop.py --mark-review-resolved --source-event-id <event-id> --review-id <review-id> --resolution-summary "Review thread closed." --curation-preview
.venv/bin/python scripts/run_evaluation_loop.py --curation-preview --curation-state ready --curation-reason review_resolved --curation-limit 10
.venv/bin/python scripts/run_evaluation_loop.py --curation-preview --learning-preview --learning-limit 10
.venv/bin/python scripts/run_dogfood_workflow.py --workflow-kind review_patch --source-event-id <event-id> --query "review patch risk and verification"
.venv/bin/python scripts/run_dogfood_workflow.py --workflow-kind compare_proposals --candidate-event-id <event-a> --candidate-event-id <event-b> --winner-event-id <event-a>
.venv/bin/python scripts/run_dogfood_workflow.py --workflow-kind resolved_work_curation_preview --curation-reason review_resolved --curation-limit 10
.venv/bin/python scripts/run_agent_lane.py --task-title "Patch smoke" --goal "Run a bounded patch-plan-verify loop." --plan-step "Inspect scope." --plan-step "Run verification." --verification-command ".venv/bin/python -m unittest tests.test_agent_lane"
.venv/bin/python scripts/run_backend_swap.py --list-backends
.venv/bin/python scripts/run_backend_swap.py --task-title "Backend swap smoke" --goal "Run the same workflow across local backend configs." --plan-step "Load backend config." --plan-step "Run verification." --verification-command ".venv/bin/python -m unittest tests.test_backend_swap"
.venv/bin/python scripts/satlab.py event ingest --patch changes.diff --note "Patch review input"
.venv/bin/python scripts/satlab.py recall failure --query "patch risk similar failure"
.venv/bin/python scripts/satlab.py pack run review-risk-pack --patch changes.diff
.venv/bin/python scripts/satlab.py compare proposals --candidate proposal-a.md --candidate proposal-b.md --verdict winner --winner-candidate 1 --rationale "Proposal A preserves source evidence."
.venv/bin/python scripts/satlab.py verdict reject --event <event-id> --reason "Repeats prior missing-source bug"
.venv/bin/python scripts/satlab.py report latest --format md
.venv/bin/python scripts/satlab.py learning inspect --preview-only
.venv/bin/python scripts/satlab.py validation template --output-dir artifacts/demand_validation_notes
.venv/bin/python scripts/satlab.py validation record-run --event <event-id> --useful-recall yes --critical-false-evidence-count 0 --verdict-capture-seconds 20 --notes-file artifacts/demand_validation_notes/dogfood_run_notes.md
.venv/bin/python scripts/satlab.py validation record-interview --participant user-1 --recognized-pain yes --wants-to-try yes --notes-file artifacts/demand_validation_notes/external_user_interview.md
.venv/bin/python scripts/satlab.py validation record-setup --clone-to-demo-minutes 12 --notes-file artifacts/demand_validation_notes/setup_timing.md
.venv/bin/python scripts/satlab.py validation report --write --format md
.venv/bin/python scripts/satlab.py pack inspect templates/review-risk-pack.satellite.yaml
.venv/bin/python scripts/satlab.py pack audit templates/review-risk-pack.satellite.yaml
.venv/bin/python scripts/run_public_demo_checks.py
.venv/bin/python scripts/run_local_ui.py
.venv/bin/python scripts/run_capability_matrix.py --smoke
```
