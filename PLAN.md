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
- capability matrix results can be backfilled into the same software-work event shape for recall and evaluation

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
- the index includes capability matrix events and `pass_definition`, which makes contract-based recall evaluable without a separate store

### M3. Recall and Context Builder

Goal:

- assemble useful context for review, design, proposal, and failure-analysis tasks

Deliverables:

- retrieval API
- ranking or filtering policy
- context assembly rules for:
  - review
  - design
  - proposal
  - failure analysis

Validation:

- shadow runs on existing repo artifacts
- source-hit coverage measured on a prepared real-data request set
- CLI and Local UI can inspect the same recall snapshots

Current baseline:

- `scripts/recall_context.py` implements the current `RecallRequest` to `ContextBundle` path
- supported task kinds are `review`, `design`, `proposal`, and `failure_analysis`
- ranking is lexical and rule-based, with explainable reason tags rather than a learned re-ranker
- context assembly groups selected candidates into stable blocks and trims them against a character budget
- `source_event_id` support records whether the expected source was selected and why it missed when it was not selected
- pass-definition requests use phrase-first retrieval and same-pass-definition grouping
- pinned event ids can be injected for manual compare and source-hit diagnosis
- `docs/recall_context_builder_design.md` is the implementation-following design note for this milestone
- `docs/recall_hit_quality_loop.md` defines the lightweight source-hit evaluation loop used to tune M3

Current validation:

- `scripts/prepare_recall_real_data.py` builds prepared real-data requests from workspace and capability-matrix events
- `scripts/run_recall_demo.py` lists requests, evaluates source-hit coverage, reports misses, and writes reusable snapshots
- Local UI reads the same recall dataset and evaluation summary for request selection, miss diagnosis, manual recall, and pin compare
- unit tests cover request normalization, ranking, budget trimming, source evaluation, pass-definition grouping, dataset generation, runner behavior, and Local UI recall helpers

M3 checkpoint scope:

- keep M3 focused on local, explainable, lexical recall
- treat source-hit measurement as the M3 quality loop, not as the full M4 evaluation system
- finish by keeping docs, tests, CLI reports, and Local UI recall flows consistent with the current implementation

Deferred from M3:

- semantic/vector fallback
- learned re-ranking
- token-aware budgeting
- answer-usefulness judging
- large dashboard or external search infrastructure

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

Current baseline:

- M3 already includes a narrow source-hit evaluation loop for Recall quality
- `scripts/evaluation_loop.py` defines the first broader, file-first evaluation signal layer
- acceptance, rejection, test pass, and test fail use one local signal schema under `artifacts/evaluation/<workspace>/signals.jsonl`
- review-resolution uses the same signal schema via `review_resolved` / `review_unresolved`; resolved review can satisfy the human-selection gate when a test pass is present, while unresolved review blocks curation
- test pass/fail signals can be derived from software-work events, including capability matrix validation and Local UI artifact validation
- failure to repair/follow-up linkage is explicit through `relation_kind` and `target_event_id`, avoiding noisy automatic repair guesses
- `comparisons.jsonl` stores local comparison records for competing outputs without mixing them into pass/fail signals
- evaluation snapshots derive curation candidates as `ready`, `needs_review`, or `blocked`; training export remains deliberately downstream and gated
- curation export preview writes `artifacts/evaluation/<workspace>/curation/preview-latest.json` and run artifacts as `preview_only`, listing candidate decisions, filters, adoption checklist state, and required next steps without writing downstream training data
- `scripts/run_evaluation_loop.py` writes the same reusable snapshot that the Local UI Evaluation tab displays; both paths support curation filters for state, export decision, and reason
- the Local UI Evaluation tab can record minimal `review_resolved` / `review_unresolved` signals against selected curation candidates
- richer multi-backend comparison UX and final export policy definition are intentionally deferred to M6 / M7 rather than reopened inside M4

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

Current baseline:

- `scripts/agent_lane.py` defines a file-first task/run schema under `artifacts/agent_lane/<workspace>/`
- agent tasks preserve bounded scope, plan steps, verification commands, acceptance criteria, and pass definition
- agent runs capture plan-step traces and verification command traces with stdout/stderr excerpts, duration, exit status, timeout state, and result summary
- `scripts/run_agent_lane.py` records a narrow patch-plan-verify task, runs verification without a shell, writes a run artifact, and rebuilds the software-work event index by default
- agent runs are normalized into `agent_task_run` software-work events, so successful verification derives `test_pass` and failed verification derives `test_fail` in the M4 evaluation snapshot
- failed verification is kept as a first-class outcome through `quality_status=fail`, `execution_status=failed`, quality checks, and pending-failure curation state

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

- curation filters for preview-only export
- final export policy and supervised example format
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
