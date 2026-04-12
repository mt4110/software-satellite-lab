# Lab Redesign Plan

This document is the new design blueprint for `software-satellite-lab`.
It replaces the older phase-centric story that centered the repo around a capability lab and thin local UI milestones.

## 1. Purpose

`software-satellite-lab` now aims to become a software-development satellite system built around a swappable AI core.

The short-term product goal is:

- make software creation materially better
- specialize in review, design, proposal, agent execution, evaluation, and memory
- collect operational data with clear outcome signals
- use that data later for selective fine-tuning and model evolution

The long-term research goal is:

- build a learning loop that can support dataset curation, LoRA / SFT, model comparison, and eventual independent model development under human supervision

The ordering matters:

1. useful software-work system first
2. trustworthy data and evaluation second
3. model training and model evolution third

## 2. Non-Goals

The following are intentionally out of scope for the first redesign wave:

- a Docker-first architecture
- a Postgres-first architecture
- cloud orchestration
- multi-user tenancy
- autonomous self-improving model training without human gates
- broad benchmark chasing detached from software-work usefulness
- replacing the current repo with a web stack, DB stack, or service mesh

If one of those becomes necessary later, it must be justified by observed limits, not by habit.

## 3. Core Functions

The redesigned lab should be organized around five core modules.

### 3.1 Event and Artifact Layer

Purpose:

- record prompts, context, tool actions, outputs, tests, review comments, acceptance or rejection signals, and follow-up fixes

Must provide:

- append-friendly records
- inspectable local files
- stable schema for later indexing
- clear links between artifact, session, model, and outcome

Likely inputs:

- current artifact payloads
- workspace and session manifests
- review notes
- patch outcomes
- test results

### 3.2 Memory and Retrieval Layer

Purpose:

- search prior sessions, artifacts, failures, fixes, design notes, code references, and accepted patterns across sessions

Must provide:

- structured filters
- lexical search
- later optional vector recall
- low-latency local use

Planned storage stance:

- source of truth remains file-based
- first index layer is local `SQLite`
- `FTS5` is the first retrieval engine
- vector search stays optional and secondary

### 3.3 Proposal and Review Layer

Purpose:

- specialize the system for software-engineering judgment rather than generic chat

Must provide:

- design proposal support
- code review support
- implementation planning
- failure analysis
- repair suggestions grounded in prior evidence

Success signal:

- proposals and reviews become more consistent, faster, and easier to verify

### 3.4 Agent Execution Layer

Purpose:

- run bounded software tasks with tools while keeping the work inspectable

Must provide:

- task boundaries
- tool-use traces
- patch or action logs
- explicit success, failure, and retry outcomes

Constraint:

- the system must remain debuggable by humans
- hidden magic is a liability here

### 3.5 Evaluation and Learning-Prep Layer

Purpose:

- turn runtime behavior into reusable evidence for selection and later training

Must provide:

- pass or fail outcomes
- regression detection
- proposal acceptance or rejection signals
- review resolution signals
- dataset curation hooks for future LoRA / SFT work

Constraint:

- weak or noisy data must be filtered before it reaches any training path

## 4. Swappable Boundaries

The redesign should keep the following boundaries explicit.

### 4.1 Backend Adapter Boundary

The AI backend must be replaceable.

The adapter contract should eventually cover:

- text generation
- optional tool call formatting
- optional embeddings
- model metadata
- runtime capability flags

The rest of the repo should not depend on one backend's prompt quirks more than necessary.

### 4.2 Storage Boundary

The file layer remains the durable record.
Indexes are allowed to be rebuildable.

That means:

- artifacts and session manifests are durable
- SQLite index state can be rebuilt from source files
- retrieval must degrade gracefully if the index is stale

### 4.3 Evaluation Boundary

Evaluation must not be fused into one UI or one model.

It should stay usable from:

- CLI
- local UI
- batch jobs
- future training-prep jobs

### 4.4 Agent Boundary

Agent execution must stay bounded and inspectable.

That means:

- explicit tasks
- explicit tools
- explicit outputs
- explicit stop conditions
- explicit review or verification outcomes

### 4.5 Learning Boundary

Training and fine-tuning must remain downstream consumers of curated evidence.

That means:

- no direct training on raw logs by default
- no promotion to training data without quality filters
- no silent feedback loop that retrains on its own output without review

## 5. Milestones

The redesign roadmap below is ordered by usefulness and dependency, not by glamour.

### M0. Doc Reset and Boundary Freeze

Goal:

- make the repo documentation match the new direction

Done when:

- top-level docs describe the satellite-system goal
- old confusing direction-setting docs are removed or rewritten
- future work is framed around software work, memory, evaluation, and backend swapability

### M1. Software-Work Event Schema

Goal:

- define a normalized local record for software work across sessions

Deliverables:

- event schema for prompts, tool actions, outputs, review comments, tests, and outcomes
- artifact linkage rules
- migration note from current workspace and artifact files

Validation:

- sample sessions can be converted into normalized event records
- records preserve enough detail for later retrieval and evaluation

Current baseline:

- `scripts/software_work_events.py` now provides the first event normalization layer over existing workspace/session entries

### M2. Local Memory Index

Goal:

- add a local indexed memory layer without changing the file-first source of truth

Deliverables:

- SQLite database under a repo-local path
- ingest job from artifacts and session manifests
- `FTS5` search over prompts, outputs, notes, paths, and outcomes

Validation:

- rebuild index from files
- basic retrieval latency is acceptable on local hardware
- stale index behavior is explicit

Current baseline:

- `scripts/memory_index.py` and `scripts/rebuild_memory_index.py` now provide the first rebuildable local `SQLite FTS5` index over software-work events

### M3. Recall and Context Builder

Goal:

- assemble useful context for review, design, and proposal tasks

Deliverables:

- retrieval API
- ranking or filtering policy
- context assembly rules for:
  - review
  - design
  - implementation proposal
  - failure analysis

Validation:

- shadow runs on existing repo artifacts
- retrieval quality measured with a small hand-labeled query set

### M4. Evaluation Loop

Goal:

- capture which outputs actually helped and which did not

Deliverables:

- acceptance and rejection signals
- test pass and fail signals
- fix-after-failure linkage
- comparison views for competing proposals or model outputs

Validation:

- enough signal exists to answer:
  - what got accepted
  - what failed
  - what was repaired
  - which patterns recur

### M5. Agent Lane for Software Tasks

Goal:

- support bounded agent tasks with inspectable traces

Deliverables:

- task schema
- tool trace capture
- result summary
- verification step

Validation:

- the lab can execute at least a narrow patch-plan-verify loop
- failures are recorded as first-class outcomes

### M6. Backend Swap Harness

Goal:

- make the system survive model replacement

Deliverables:

- backend adapter interface
- capability metadata
- compatibility checks
- side-by-side comparison path for different backends

Validation:

- at least two backend configurations can run through the same outer workflow with bounded special casing

### M7. Learning and Fine-Tune Prep

Goal:

- prepare high-quality training inputs instead of dumping raw logs into a trainer

Deliverables:

- curation filters
- export format for supervised examples
- candidate LoRA / SFT dataset generation
- comparison and adoption checklist

Validation:

- generated datasets are traceable to accepted or high-value evidence
- rejected or noisy examples are filtered out by default

### M8. Human-Gated Model Evolution

Goal:

- reach a supervised loop for dataset selection, training, evaluation, and adoption

Deliverables:

- training job recipes
- comparison reports
- promotion criteria
- rollback path

Validation:

- a candidate model can be trained, evaluated, compared, and either adopted or rejected with clear evidence

## Immediate Operating Rule

When in doubt:

- preserve file-first truth
- choose simpler local infrastructure
- keep the backend swappable
- record evidence
- delay training until the evidence is worth learning from
