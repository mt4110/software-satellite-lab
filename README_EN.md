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

These are treated as baseline assets for the redesign, not as throwaway work.

## Documents

- `README.md`: Japanese overview
- `README_EN.md`: English overview
- `PLAN.md`: redesign blueprint and milestones

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
.venv/bin/python scripts/run_local_ui.py
.venv/bin/python scripts/run_capability_matrix.py --smoke
```
