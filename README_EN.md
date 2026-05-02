# software-satellite-lab

[日本語 README](./README.md)

A software-engineering satellite-system lab.

This repository is not centered on building a brand-new frontier model first.
Its immediate purpose is to build an outer intelligence layer that makes real software work better.

## Goals

- strengthen review work
- improve design and proposal quality
- support bounded agent execution
- preserve evaluation loops
- enable cross-session memory
- prepare for future backend swapping and learning pipelines

## What It Builds

The core target is not a single model.
The target is a satellite system for software development.

The current design is organized around five functions:

1. event and artifact capture
2. memory and retrieval
3. review and proposal support
4. agent execution
5. evaluation and learning preparation

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
- `docs/recall_context_builder_design.md`: first implementation design for Recall / Context Builder
- `docs/recall_hit_quality_loop.md`: hit-quality visualization and lightweight evaluation loop for Recall
- `docs/learning_finetune_prep_design.md`: M7 preview-only dataset candidate design for Learning and Fine-Tune Prep

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
  - foundation for evaluation signals, comparisons, curation filters, learning dataset previews, and evaluation snapshots
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
.venv/bin/python scripts/run_evaluation_loop.py --record-signal --signal-kind acceptance --source-event-id <event-id>
.venv/bin/python scripts/run_evaluation_loop.py --record-signal --signal-kind review_resolved --source-event-id <event-id> --review-id <review-id> --resolution-summary "Review thread closed." --curation-preview
.venv/bin/python scripts/run_evaluation_loop.py --curation-preview --curation-state ready --curation-reason review_resolved --curation-limit 10
.venv/bin/python scripts/run_evaluation_loop.py --curation-preview --learning-preview --learning-limit 10
.venv/bin/python scripts/run_agent_lane.py --task-title "Patch smoke" --goal "Run a bounded patch-plan-verify loop." --plan-step "Inspect scope." --plan-step "Run verification." --verification-command ".venv/bin/python -m unittest tests.test_agent_lane"
.venv/bin/python scripts/run_backend_swap.py --list-backends
.venv/bin/python scripts/run_backend_swap.py --task-title "Backend swap smoke" --goal "Run the same workflow across local backend configs." --plan-step "Load backend config." --plan-step "Run verification." --verification-command ".venv/bin/python -m unittest tests.test_backend_swap"
.venv/bin/python scripts/run_local_ui.py
.venv/bin/python scripts/run_capability_matrix.py --smoke
```
