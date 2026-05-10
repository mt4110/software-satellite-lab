# Review Risk Report Example

## Patch

- Source: `examples/patches/risky-change.diff`
- Summary: AI-generated patch changes a review-risk path without preserving the source artifact reference.

## Recalled Failure Memory

| Rank | Prior outcome | Evidence path | Why it matters |
|---:|---|---|---|
| 1 | rejected | `artifacts/event_logs/local-default.jsonl#missing-source-2026-05` | Prior review rejected an output that claimed success without a durable source artifact path. |
| 2 | fixed | `artifacts/dogfood_workflows/local-default/runs/example-repair.json` | Repair added explicit source refs before the result could be considered for learning-candidate inspection. |

## Risk Note

This patch is risky because it appears to repeat a prior missing-source pattern. The next action should be to add or restore the source artifact path before any positive evidence or learning-candidate inspection state is recorded.

## Human Verdict

```json
{
  "verdict": "needs_review",
  "reason": "Source path preservation is not yet demonstrated.",
  "human_gate_required": true
}
```

## Learning-Candidate State

```json
{
  "state": "blocked",
  "blocked_reason": "missing_source",
  "preview_only": true
}
```

## What This Demonstrates

- source-linked failure memory
- negative evidence preservation
- human verdict before positive signal
- learning-candidate inspection stays preview-only
