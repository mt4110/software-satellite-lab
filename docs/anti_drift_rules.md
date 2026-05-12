# Anti-Drift Rules

These rules keep the first product wave focused on the AI Coding Flight Recorder wedge.

The product may grow later, but the first wave must stay narrow enough that a stranger can
describe it in one sentence:

```text
It remembers AI coding work with evidence.
```

## Core Rule

Before adding a feature, answer this question:

```text
Does this make AI-assisted software work easier to remember, compare, inspect, or reuse with source-linked evidence?
```

If the answer is no, do not build it in the first product wave.

If the answer is yes, the feature still must pass the boundary checks below.

## First Wave Boundaries

| Boundary | Allowed in the first wave | Drift to reject |
|---|---|---|
| Product category | Local-first AI Coding Flight Recorder / software-work evidence ledger | Coding agent, IDE assistant, provider hub, marketplace, dashboard SaaS |
| Primary value | Source-linked evidence, failure memory, comparison reuse, human verdict capture | Generic automation, generic observability, generic chat, task execution spectacle |
| Storage | File-first artifacts with rebuildable local indexes | Cloud-first source of truth, required hosted database, opaque remote state |
| Recall | Lexical / structured recall that preserves source paths and confidence limits | Vector search as a default requirement, unsupported semantic claims, unverifiable recall |
| Review | Evidence-gated risk notes and prior-failure memory | Automated PR-review replacement, self-certifying bug finding |
| Comparison | Candidate/backend comparison records with human rationale | Provider-count competition, live provider brokerage, benchmark chasing detached from user work |
| Learning | Preview-only learning-candidate inspection after quality filters | Training export, auto fine-tune, raw-log training, silent self-improvement loop |
| Packs | Declarative Satellite Evidence Packs using core-owned transforms | Arbitrary code plugins, marketplace install, network packs, command packs in v0 |
| UI | Thin inspection surfaces and reports | Complex dashboard, live collaboration suite, full IDE surface |

## Required Gate

A first-wave feature is acceptable only when all statements are true:

- It writes or improves durable evidence records.
- It preserves source artifact paths whenever it refers to prior work.
- It separates current work from prior evidence.
- It can represent weak, missing-source, contradictory, rejected, or unresolved evidence without promoting it.
- It keeps human verdicts explicit when acceptance, rejection, comparison winner, or learning eligibility matters.
- It works locally without a required cloud service.
- It keeps indexes rebuildable from file-first source artifacts.
- It does not require arbitrary pack code, network access, secrets access, repo writes, background daemons, or remote install.
- It does not produce training data, start a training job, or mark data as training-ready.

If one of these statements is false, the feature needs a smaller scope or belongs after demand validation.

## Evidence Quality Rules

Evidence is not positive support unless it has a source path and passes the relevant gate.

First-wave reports and recall outputs must distinguish:

- positive source-linked evidence
- weak matches
- missing-source records
- contradictory evidence
- current-review subject material
- unresolved work
- rejected work
- failed verification
- human-pinned notes

Weak, missing-source, contradictory, current-subject, unresolved, rejected, or failed records may be shown for context, but must not be rendered as proof that a proposed change is safe or correct.

## Learning Boundary

Learning preparation stays inspection-only in the first product wave.

Allowed:

- curation preview
- blocked reasons
- exclusion reasons
- human-selected candidate lists
- metadata-only dry-runs
- evidence summaries for future policy review

Denied:

- trainable JSONL export
- model fine-tune launch
- automatic candidate promotion
- raw log export as training data
- treating a comparison winner as training-ready without source paths, verification, and human rationale

## Satellite Evidence Pack Boundary

Satellite Evidence Packs are declarative evidence workflows, not plugins.

Allowed in v0:

- YAML or JSON manifests
- prompt / instruction templates
- recall policies
- evaluation criteria
- read-only widgets
- schema-defined outputs
- core-owned transforms

Denied in v0:

- Python, JavaScript, shell, or other executable pack content
- network access
- secrets access
- repo writes
- background daemons
- remote install or auto-update
- marketplace distribution
- live model or backend adapter packs

## Drift Traps

Reject or defer a proposal when it mainly argues for:

- more agent autonomy instead of better evidence capture
- more providers instead of better comparison records
- more UI surface instead of clearer source-linked reports
- more dashboards instead of stronger recall and verdict loops
- more benchmarks instead of dogfood usefulness
- more training machinery before curated evidence exists
- more pack freedom before the permission model is proven
- cloud or multi-user infrastructure before local workflows are validated

The useful question is not "could this be powerful?" but:

```text
Will this make the next human judgment better because prior AI-assisted software work is inspectable?
```

## Review Checklist

Use this checklist for issues, PRs, design notes, and milestone proposals:

```text
[ ] Names the source-linked evidence it creates or improves.
[ ] Names the human decision it helps: accept, reject, needs_fix, compare, recall, curate, or exclude.
[ ] Preserves source artifact paths.
[ ] Does not treat current input as prior evidence.
[ ] Handles missing, weak, contradictory, rejected, failed, and unresolved evidence honestly.
[ ] Keeps local-first and file-first behavior.
[ ] Keeps learning-candidate inspection preview-only.
[ ] Does not add arbitrary pack execution or forbidden v0 permissions.
[ ] Does not reposition the project as a coding agent, IDE assistant, provider hub, marketplace, or dashboard SaaS.
```

## Defer Until After Demand Validation

The following may become valid later, but are outside the first product wave:

- IDE extension
- marketplace
- executable packs
- live multi-provider integration
- cloud or multi-user service
- vector search as a required default
- training export
- automatic fine-tuning
- complex dashboard

Each deferred item must be reintroduced with evidence from real dogfood or user demand, not because it is a familiar software category.
