# Backend / Model Adoption Dossier

## 1. Decision Question

Should `{candidate_backend}` be adopted for `{workflow_kind}` against baseline `{baseline_backend}`, based on source-linked software-work outcomes?

## 2. Scope of Adoption

- Workflow kind: `{workflow_kind}`
- Repo scope: `{repo_scope}`
- Risk scope: `{risk_scope}`

## 3. Baseline and Candidate Metadata

| Role | Backend | Model | Compatibility |
| --- | --- | --- | --- |
| candidate | `{candidate_backend}` | `{candidate_model}` | `{candidate_compatibility}` |
| baseline | `{baseline_backend}` | `{baseline_model}` | `{baseline_compatibility}` |

## 4. Source-Linked Task Outcomes

| Role | Comparison | Backend | Event | Positive Support | Risk Support | Source Artifact Refs |
| --- | --- | --- | --- | --- | --- | --- |
| candidate | `{candidate_comparison_role}` | `{candidate_backend}` | `{candidate_event_id}` | `{candidate_positive_support}` | `{candidate_risk_support}` | `{candidate_source_refs}` |
| baseline | `{baseline_comparison_role}` | `{baseline_backend}` | `{baseline_event_id}` | `{baseline_positive_support}` | `{baseline_risk_support}` | `{baseline_source_refs}` |

## 5. Failures and Regressions

List negative evidence, failed benchmark gates, unresolved contradictions, regressions, and source-linked risk outcomes. Negative evidence must stay visible even when the recommendation is adopt.

## 6. Human Verdicts and Rationale

- Comparison outcome: `{comparison_outcome}`
- Winner event: `{winner_event_id}`
- Human rationale: `{human_rationale}`

## 7. Evidence Support Summary

```json
{evidence_support_summary}
```

## 8. Cost / Latency Metadata

```json
{cost_latency_metadata}
```

## 9. Rollback Plan

- Present: `{rollback_present}`
- Source: `{rollback_source}`
- Plan: `{rollback_plan}`

## 10. Recommendation

- Recommendation: `{recommendation}`
- Rationale: `{recommendation_rationale}`
- Blockers: `{blockers}`
- Warnings: `{warnings}`

## Exit Gate

```json
{exit_gate}
```
