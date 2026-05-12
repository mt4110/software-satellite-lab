# M9 Strategy Hardening Review — Evidence-Gated Git Review Workbench

Date: 2026-05-12

## Bottom Line

I am **not** 100% confident in the original M9 strategy as written.

The direction is strong, but the first draft had enough loopholes that I would not ship it unchanged. The largest issue is that it could be interpreted as a thin wrapper around the existing `review-risk-pack`, and a local smoke run showed a concrete failure mode: the current patch input can be recalled as if it were prior evidence unless the implementation explicitly excludes the active review event.

After hardening, the recommended strategy becomes:

> Build a Git-native, evidence-gated review workbench whose primary value is not AI bug finding, but source-linked memory of prior failures, repairs, human verdicts, and recall usefulness.

This is still M9, but the emphasis changes from "one-command report" to **anti-false-evidence review memory**.

## Confidence After Hardening

| Scope | Confidence | Rationale |
|---|---:|---|
| Two-week implementation feasibility | 90% | Current repo already has event normalization, memory index, recall, verdict, and validation infrastructure. |
| OSS strategic direction | 86% | Differentiation is real if the project avoids competing directly with PR reviewers and observability suites. |
| External adoption without further polish | 68% | Current repo has no releases, no install package, and 0 stars/forks at review time. Adoption requires a much sharper quickstart. |
| Revised M9 as next milestone | 89% | Best next step because it compresses M0-M8 into user-visible value and exposes measurable demand. |

I would not claim factual 120% certainty. The better engineering standard is: **all known critical loopholes are either closed or moved behind an explicit gate.**

## Falsification Loop

### Loop 0 — Original Claim

Original strategy:

```text
git diff / test log / note
  -> software-work event
  -> prior failure recall
  -> evidence-grounded risk report
  -> human verdict
  -> reusable memory and evaluation signal
```

Initial confidence: 78%.

Reason: strategically coherent, but under-specified around false recall, cold start, adoption, and competitive overlap.

### Loop 1 — Repo and Competitor Check

Findings:

1. The repository is already framed as a local-first AI Coding Flight Recorder / evidence layer, not a coding agent.
2. Existing functionality already supports patch ingest, failure recall, review-risk reports, verdicts, evaluation signals, and validation metrics.
3. The competitive market is crowded across three fronts:
   - AI coding agents: Claude Code, GitHub Copilot cloud agent, OpenHands, Aider, SWE-agent.
   - LLM / agent observability: Langfuse, LangSmith, MLflow Tracing, Arize Phoenix, Braintrust.
   - PR review automation: Qodo, CodeRabbit, PR-Agent, Cursor Bugbot-like workflows.
4. A local smoke run of `satlab.py pack run review-risk-pack` produced a report, but recalled the newly ingested patch input itself as the top recalled evidence. That means M9 needs active-event exclusion and temporal evidence gating before it can be trusted.

Confidence after Loop 1: 71% for the original draft, because the strategy was directionally right but not safe enough.

### Loop 2 — Loophole Closure

Critical changes:

1. M9 must not be "another review-risk-pack wrapper." It must add Git-native intake, active event exclusion, test-log parsing, source integrity, report schema, and benchmark fixtures.
2. The report must distinguish:
   - `source_linked_prior`
   - `current_review_subject`
   - `weak_match`
   - `manual_pin`
   - `missing_source`
   - `contradictory`
3. No current review subject may appear in Top Prior Evidence.
4. No missing-source, weak, or contradictory evidence may be rendered as positive support.
5. The cold-start path must be first-class: when no prior evidence exists, the tool should say so and produce a useful capture/verdict workflow rather than pretending to have recall value.
6. M9 must ship with a benchmark harness that measures false evidence, useful recall, source path completeness, and verdict friction.

Confidence after Loop 2: 86%.

### Loop 3 — Revised Strategy

Revised name:

```text
M9: Evidence-Gated Git Review Workbench
```

Revised one-sentence purpose:

> Turn a real git diff into a source-linked review-memory artifact that helps a human determine whether the change repeats known failures, lacks verification, or deserves a verdict — without pretending that weak recall is evidence.

Final confidence: 89% for M9 as the next milestone.

## Critical Loopholes and Fixes

| # | Loophole | Severity | Proper Fix | Validation Gate |
|---:|---|---|---|---|
| 1 | Current patch can be recalled as prior evidence | Critical | Add `exclude_event_ids`, `temporal_role=current_review_subject`, and recall-after-ingest guard | Top Prior Evidence never contains active review event |
| 2 | M9 may duplicate existing `review-risk-pack` | Critical | Define M9 as Git intake + test-log parser + evidence classes + benchmark + verdict usefulness loop | New tests fail if command only proxies pack output |
| 3 | Cold start produces no value | High | First-class no-prior-evidence report, demo seed fixture, and capture/verdict guidance | Fresh repo still produces useful summary and next action |
| 4 | False recall can mislead reviewers | Critical | Confidence classes, source contract checks, contradiction flags, usefulness feedback | Critical false evidence count must be 0 |
| 5 | FTS lexical search misses semantic similarity | Medium | Add structured diff features first: paths, extensions, symbols, hunk fingerprints, status, test names | Useful recall >= 30% before vector search is considered |
| 6 | Competes with PR review automation | High | Do not market as automated code review. Market as prior-failure memory and evidence ledger | README says it does not post PR comments or replace review |
| 7 | Competes with observability platforms | High | Do not chase generic traces. Keep outcome-linked software-work evidence as the unit | Report emphasizes source paths, human verdicts, test outcomes |
| 8 | Secrets in diffs/test logs | Critical | Default redaction pass, max-size caps, binary refusal, explicit raw-capture override | Fixture with fake secret is redacted |
| 9 | Source file may change after capture | High | Store diff/test-log hash, repo commit metadata, base/head refs, source artifact snapshot path | Report has stable hash and commit/ref metadata |
| 10 | Huge diffs can overwhelm report | Medium | Bounded parser, changed-file cap, hunk cap, explicit truncation metadata | Large diff fixture returns bounded report with warnings |
| 11 | Deleted/binary/submodule diffs break parser | Medium | Classify unsupported diff components and keep them out of positive evidence | Edge fixtures pass |
| 12 | Verdict command friction remains too high | Medium | Add `review verdict --from-latest`, required rationale, optional follow-up | Median verdict capture under 30 seconds |
| 13 | Learning boundary can drift | High | Explicit `training_export_ready=false` and no JSONL write path in M9 | Tests assert no training export artifact is created |
| 14 | Full test suite appears slow/flaky | Medium | Split default, slow, and live tests; public demo check must be deterministic | Default CI completes under fixed timeout |
| 15 | No release/install path | High | Add minimal `python -m` or `pipx` install story after core command stabilizes | New user can run demo in under 10-15 minutes |
| 16 | Competitors can copy one-command report | High | Make moat the accumulated source-linked verdict ledger, not report formatting | Demo shows value improves with prior verdict history |
| 17 | Research story not visible to OSS users | Medium | Publish metrics: useful recall, false evidence, source completeness, verdict friction | `review benchmark` writes metric artifact |
| 18 | Human verdicts may be biased/noisy | Medium | Capture rationale, decision class, reviewer label optional, contradiction checks | No verdict becomes learning-eligible without positive gate |
| 19 | OTel/observability standards may absorb the space | Medium | Treat OTel as future import/export layer, not core identity | M9 does not depend on OTel |
| 20 | M9 still lacks external demand proof | Critical | Use demand validation gates: dogfood runs, external user interviews, clone-to-demo time | Do not start M10 until demand gate passes |

## Revised Two-Week Design

### Week 1 — Trustworthy Git Review Core

#### Day 1 — Active Event and Prior Evidence Separation

Implement:

- `current_review_subject` marker
- `exclude_event_ids` in recall path
- temporal evidence gate: only events recorded before current review can count as prior evidence
- tests proving current input is never listed as prior evidence

Exit gate:

```text
Top Prior Evidence must not include the active review event.
```

#### Day 2 — Git-Native Intake

Implement:

- `scripts/git_work_intake.py`
- `satlab review git --base <ref> --head <ref>`
- base/head commit capture
- changed files, diff stats, hunk count
- dirty tree warning
- bounded diff capture

Exit gate:

```text
Works on a real git repo and writes stable JSON without network or model credentials.
```

#### Day 3 — Test Log and Source Integrity

Implement:

- optional `--test-log`
- test status extraction: pass/fail/unknown
- command/excerpt/hash capture
- source artifact snapshot references
- redaction of common secret patterns

Exit gate:

```text
Fake API key fixture is redacted; source hash is present.
```

#### Day 4 — Evidence-Gated Report

Implement:

- `scripts/evidence_review_workbench.py`
- Markdown and JSON report generated from same artifact
- evidence classes:
  - `source_linked_prior`
  - `current_review_subject`
  - `weak_match`
  - `manual_pin`
  - `missing_source`
  - `contradictory`
- no-prior-evidence path

Exit gate:

```text
Report never presents weak/missing/contradictory evidence as positive support.
```

#### Day 5 — Human Verdict and Recall Usefulness

Implement:

- `satlab review verdict --from-latest`
- decisions: `accept`, `reject`, `needs_fix`, `needs_more_evidence`
- rationale required
- follow-up optional
- recall usefulness label: `useful`, `irrelevant`, `misleading`, `not_checked`

Exit gate:

```text
Verdict writes evaluation signal and usefulness feedback in one command.
```

### Week 2 — Benchmark, Demo, and OSS Readiness

#### Day 6 — Review Benchmark Fixtures

Add fixtures:

1. no prior evidence
2. true prior failure match
3. misleading lexical match
4. missing source event
5. current-event self-match attempt
6. huge diff
7. binary diff
8. deleted file
9. fake secret in test log
10. contradictory prior verdicts

Exit gate:

```text
review benchmark reports 0 critical false evidence.
```

#### Day 7 — Public Demo Check

Implement:

- deterministic demo command
- sample patch fixture
- sample prior failure fixture
- sample output report
- no API key requirement

Exit gate:

```text
Fresh clone can run the demo without live model credentials.
```

#### Day 8 — Quickstart and Positioning

Update:

- README quickstart
- README_EN quickstart
- docs/M9 design
- competitor positioning section

Message:

```text
Not an automated code reviewer.
Not an LLM observability suite.
A local evidence ledger for repeated AI coding failures and human verdicts.
```

Exit gate:

```text
A first-time reader understands the product in 60 seconds.
```

#### Day 9 — CI/Test Determinism

Implement:

- default test subset
- slow/live marker or split
- `run_public_demo_checks.py` includes M9
- py_compile gate
- no-training-export assertion

Exit gate:

```text
Default test command completes under fixed timeout.
```

#### Day 10 — Stabilization and Release Candidate

Implement:

- sample outputs
- known limitations
- issue templates for dogfood feedback
- M10 gate conditions

Exit gate:

```text
Do not advance to M10 unless M9 demand validation passes.
```

## Revised M9 Success Criteria

M9 is successful only if all are true:

1. A user can run one command on a git diff or fixture without API keys.
2. The report separates current subject from prior evidence.
3. Self-recall is impossible.
4. Missing-source evidence is visible but never positive.
5. Weak matches are labeled weak.
6. No-prior-evidence is handled honestly.
7. Human verdict is captured in one command.
8. Recall usefulness is captured.
9. Benchmark includes misleading evidence fixtures.
10. Public demo check is deterministic.
11. No training export occurs.
12. README positions the tool away from agents, PR reviewers, and generic observability.

## Strategic Recommendation

Keep M9, but rename and harden it:

```text
From: M9 Git-Native Evidence Workbench
To:   M9 Evidence-Gated Git Review Workbench
```

The phrase "Git-native" is useful but not unique. The phrase "Evidence-gated" expresses the real moat: the system should be trusted because it refuses to treat weak or self-generated recall as evidence.

## Stop Conditions

Do not continue M9 implementation if any of these remain true after Week 1:

- current event can appear as prior evidence
- report has no no-prior-evidence path
- missing-source event appears as positive support
- verdict command requires manual event ID lookup
- public demo needs API keys

Do not start M10 if any of these remain true after Week 2:

- useful recall has not been measured
- critical false evidence is nonzero
- clone-to-demo takes too long
- no external user has recognized the pain
- README still reads like a generic AI coding tool

## Final Position

The new strategy is not "build a better code reviewer."

The new strategy is:

```text
AI coding agents generate more code.
PR review automation generates more comments.
LLM observability tools generate more traces.

software-satellite-lab should remember which software-work evidence actually mattered.
```

That is the defensible OSS lane.
