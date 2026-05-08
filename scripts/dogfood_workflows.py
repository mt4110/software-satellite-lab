#!/usr/bin/env python3
from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Iterable as IterableABC
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from evaluation_loop import (
    build_curation_export_preview,
    build_evaluation_snapshot,
    curation_export_preview_latest_path,
    curation_export_preview_run_path,
    evaluation_comparison_log_path,
    evaluation_signal_log_path,
    format_curation_export_preview_report,
    normalize_curation_preview_filters,
    read_evaluation_comparisons,
    read_evaluation_signals,
    record_curation_export_preview,
    record_evaluation_snapshot,
)
from gemma_runtime import repo_root, timestamp_slug, timestamp_utc, write_json
from memory_index import MemoryIndex, rebuild_memory_index
from recall_context import DEFAULT_CONTEXT_BUDGET_CHARS, DEFAULT_LIMIT, build_context_bundle
from software_work_events import read_event_log
from workspace_state import DEFAULT_WORKSPACE_ID


DOGFOOD_WORKFLOW_PREVIEW_SCHEMA_NAME = "software-satellite-dogfood-workflow-preview"
DOGFOOD_WORKFLOW_PREVIEW_SCHEMA_VERSION = 1

WORKFLOW_REVIEW_PATCH = "review_patch"
WORKFLOW_COMPARE_PROPOSALS = "compare_proposals"
WORKFLOW_RECALL_PRIOR_FAILURE = "recall_prior_failure"
WORKFLOW_EXPLAIN_DECISION = "explain_decision"
WORKFLOW_RESOLVED_WORK_CURATION_PREVIEW = "resolved_work_curation_preview"

DOGFOOD_WORKFLOW_KINDS = (
    WORKFLOW_REVIEW_PATCH,
    WORKFLOW_COMPARE_PROPOSALS,
    WORKFLOW_RECALL_PRIOR_FAILURE,
    WORKFLOW_EXPLAIN_DECISION,
    WORKFLOW_RESOLVED_WORK_CURATION_PREVIEW,
)

DOGFOOD_WORKFLOW_SPECS: dict[str, dict[str, Any]] = {
    WORKFLOW_REVIEW_PATCH: {
        "label": "Review a patch",
        "recall_task_kind": "review",
        "default_query": "review patch risk regression tests prior similar failures",
        "intent": (
            "Use local recall and evaluation evidence to inspect a small patch before recording "
            "a human review signal."
        ),
        "operator_steps": [
            "Inspect recalled accepted work, prior failures, related files, and open risks.",
            "Run or inspect verification outside this preview if the patch is not already test-backed.",
            "Record review_resolved or review_unresolved only after a human decision.",
        ],
    },
    WORKFLOW_COMPARE_PROPOSALS: {
        "label": "Compare two proposals",
        "recall_task_kind": "proposal",
        "default_query": "compare implementation proposals tradeoffs tests adoption risk",
        "intent": "Keep two or more proposals side by side before a human records a comparison winner or follow-up.",
        "operator_steps": [
            "Inspect each candidate event and the recalled context around the shared task.",
            "Choose winner_selected only when the decision is explicit.",
            "Leave the comparison as needs_follow_up when the evidence is still thin.",
        ],
        "default_criteria": ["test evidence", "review outcome", "source trace durability", "implementation risk"],
    },
    WORKFLOW_RECALL_PRIOR_FAILURE: {
        "label": "Recall prior similar failure",
        "recall_task_kind": "failure_analysis",
        "default_query": "prior similar failure repair follow-up unresolved regression",
        "intent": "Pull prior failures and repair patterns into a small failure-analysis loop.",
        "operator_steps": [
            "Inspect failed or blocked recalled events before proposing a repair.",
            "Link repaired work to the failure with relation_kind=repairs or follow_up_for after the fix is verified.",
            "Keep unresolved failures visible instead of silently treating them as positive examples.",
        ],
    },
    WORKFLOW_EXPLAIN_DECISION: {
        "label": "Explain accepted or rejected work",
        "recall_task_kind": "design",
        "default_query": "why accepted rejected decision rationale review signal evidence",
        "intent": "Explain the latest accepted or rejected state from file-first signals and nearby context.",
        "operator_steps": [
            "Read the latest human-selection signal before older stale signals.",
            "Check review and test signals that support or contradict the decision.",
            "Use the explanation as inspection context, not as a new decision record.",
        ],
    },
    WORKFLOW_RESOLVED_WORK_CURATION_PREVIEW: {
        "label": "Curation preview from resolved work",
        "recall_task_kind": "review",
        "default_query": "resolved review curation preview adoption checklist source trace",
        "intent": "Produce a preview-only curation artifact filtered around resolved review work.",
        "operator_steps": [
            "Inspect review_resolved candidates and their adoption checklist.",
            "Confirm source artifacts, test signals, and human gates before any downstream export.",
            "Stop at preview-only curation; do not write JSONL or start training.",
        ],
    },
}


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def dogfood_workflow_root(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return _resolve_root(root) / "artifacts" / "dogfood_workflows" / workspace_id


def dogfood_workflow_preview_latest_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return dogfood_workflow_root(workspace_id=workspace_id, root=root) / "preview-latest.json"


def dogfood_workflow_preview_run_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return (
        dogfood_workflow_root(workspace_id=workspace_id, root=root)
        / "runs"
        / f"{timestamp_slug()}-dogfood-workflow-preview.json"
    )


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _mapping_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        cleaned = _clean_text(value)
        return [cleaned] if cleaned is not None else []
    if not isinstance(value, IterableABC) or isinstance(value, Mapping):
        return []
    cleaned_values: list[str] = []
    seen: set[str] = set()
    for item in value:
        cleaned = _clean_text(item)
        if cleaned is None or cleaned in seen:
            continue
        seen.add(cleaned)
        cleaned_values.append(cleaned)
    return cleaned_values


def _normalize_workflow_kind(workflow_kind: str) -> str:
    normalized = (_clean_text(workflow_kind) or "").lower().replace("-", "_")
    if normalized not in DOGFOOD_WORKFLOW_KINDS:
        raise ValueError(f"Unsupported dogfood workflow kind `{workflow_kind}`.")
    return normalized


def _event_label(event: Mapping[str, Any] | None, *, fallback: str) -> str:
    payload = _mapping_dict(event)
    content = _mapping_dict(payload.get("content"))
    return (
        _clean_text(content.get("prompt"))
        or _clean_text(content.get("output_text"))
        or fallback
    )


def _compact_event(event_id: str, event: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = _mapping_dict(event)
    session = _mapping_dict(payload.get("session"))
    outcome = _mapping_dict(payload.get("outcome"))
    content = _mapping_dict(payload.get("content"))
    source_refs = _mapping_dict(payload.get("source_refs"))
    artifact_ref = _mapping_dict(source_refs.get("artifact_ref"))
    return {
        "event_id": event_id,
        "present": bool(payload),
        "label": _event_label(payload, fallback=event_id),
        "event_kind": _clean_text(payload.get("event_kind")),
        "recorded_at_utc": _clean_text(payload.get("recorded_at_utc")),
        "session_surface": _clean_text(session.get("surface")),
        "session_mode": _clean_text(session.get("mode")),
        "status": _clean_text(outcome.get("status")),
        "quality_status": _clean_text(outcome.get("quality_status")),
        "execution_status": _clean_text(outcome.get("execution_status")),
        "artifact_path": _clean_text(artifact_ref.get("artifact_path")),
        "prompt_excerpt": _clean_text(content.get("prompt")),
    }


def _events_by_id_from_index_summary(index_summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    event_log_path = Path(str(index_summary["event_log_path"]))
    event_log = read_event_log(event_log_path)
    return {
        str(event.get("event_id")): dict(event)
        for event in event_log.get("events") or []
        if isinstance(event, Mapping) and _clean_text(event.get("event_id")) is not None
    }


def _source_event_id_from_signal(signal: Mapping[str, Any]) -> str | None:
    return _clean_text(_mapping_dict(signal.get("source")).get("source_event_id"))


def _target_event_id_from_signal(signal: Mapping[str, Any]) -> str | None:
    relation = _mapping_dict(signal.get("relation"))
    return _clean_text(relation.get("target_event_id"))


def _compact_signal(signal: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping_dict(signal.get("source"))
    evidence = _mapping_dict(signal.get("evidence"))
    relation = _mapping_dict(signal.get("relation"))
    return {
        "signal_id": _clean_text(signal.get("signal_id")),
        "signal_kind": _clean_text(signal.get("signal_kind")),
        "polarity": _clean_text(signal.get("polarity")),
        "origin": _clean_text(signal.get("origin")),
        "recorded_at_utc": _clean_text(signal.get("recorded_at_utc")),
        "source_event_id": _clean_text(source.get("source_event_id")),
        "target_event_id": _clean_text(relation.get("target_event_id")),
        "relation_kind": _clean_text(relation.get("relation_kind")),
        "status": _clean_text(source.get("status")),
        "quality_status": _clean_text(source.get("quality_status")),
        "execution_status": _clean_text(source.get("execution_status")),
        "prompt_excerpt": _clean_text(source.get("prompt_excerpt")),
        "rationale": _clean_text(evidence.get("rationale")),
        "decision_summary": _clean_text(evidence.get("decision_summary")),
        "resolution_summary": _clean_text(evidence.get("resolution_summary")),
        "failure_summary": _clean_text(evidence.get("failure_summary")),
        "review_id": _clean_text(evidence.get("review_id")),
        "review_url": _clean_text(evidence.get("review_url")),
        "test_name": _clean_text(evidence.get("test_name")),
        "validation_command": _clean_text(evidence.get("validation_command")),
    }


def _compact_comparison(comparison: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [
        _clean_text(item.get("event_id"))
        for item in comparison.get("candidates") or []
        if isinstance(item, Mapping)
    ]
    return {
        "comparison_id": _clean_text(comparison.get("comparison_id")),
        "recorded_at_utc": _clean_text(comparison.get("recorded_at_utc")),
        "outcome": _clean_text(comparison.get("outcome")),
        "winner_event_id": _clean_text(comparison.get("winner_event_id")),
        "candidate_event_ids": [item for item in candidates if item is not None],
        "task_label": _clean_text(comparison.get("task_label")),
        "criteria": _string_list(comparison.get("criteria")),
        "rationale": _clean_text(comparison.get("rationale")),
    }


def _related_signals(
    signals: Iterable[Mapping[str, Any]],
    *,
    event_ids: set[str],
) -> list[dict[str, Any]]:
    related = []
    for signal in signals:
        source_event_id = _source_event_id_from_signal(signal)
        target_event_id = _target_event_id_from_signal(signal)
        if not event_ids or source_event_id in event_ids or target_event_id in event_ids:
            related.append(_compact_signal(signal))
    related.sort(
        key=lambda item: (
            str(item.get("recorded_at_utc") or ""),
            str(item.get("signal_id") or ""),
        ),
        reverse=True,
    )
    return related


def _related_comparisons(
    comparisons: Iterable[Mapping[str, Any]],
    *,
    event_ids: set[str],
) -> list[dict[str, Any]]:
    related = []
    for comparison in comparisons:
        compact = _compact_comparison(comparison)
        candidate_ids = set(_string_list(compact.get("candidate_event_ids")))
        winner_event_id = _clean_text(compact.get("winner_event_id"))
        if not event_ids or candidate_ids & event_ids or winner_event_id in event_ids:
            related.append(compact)
    related.sort(
        key=lambda item: (
            str(item.get("recorded_at_utc") or ""),
            str(item.get("comparison_id") or ""),
        ),
        reverse=True,
    )
    return related


def _latest_decision_signal(
    signals: Iterable[Mapping[str, Any]],
    *,
    source_event_id: str | None,
) -> dict[str, Any] | None:
    decision_signals = [
        _compact_signal(signal)
        for signal in signals
        if _clean_text(signal.get("signal_kind")) in {"acceptance", "rejection"}
        if source_event_id is None or _source_event_id_from_signal(signal) == source_event_id
    ]
    decision_signals.sort(
        key=lambda item: (
            str(item.get("recorded_at_utc") or ""),
            str(item.get("signal_id") or ""),
        ),
        reverse=True,
    )
    return decision_signals[0] if decision_signals else None


def _decision_explanation(
    *,
    signals: Iterable[Mapping[str, Any]],
    source_event_id: str | None,
) -> dict[str, Any]:
    latest = _latest_decision_signal(signals, source_event_id=source_event_id)
    if latest is None:
        return {
            "source_event_id": source_event_id,
            "decision_state": "unknown",
            "latest_signal_kind": None,
            "explanation": "No acceptance or rejection signal is available for this workflow scope.",
            "evidence": {},
        }

    signal_kind = _clean_text(latest.get("signal_kind"))
    decision_state = "accepted" if signal_kind == "acceptance" else "rejected"
    evidence_text = (
        _clean_text(latest.get("decision_summary"))
        or _clean_text(latest.get("rationale"))
        or _clean_text(latest.get("resolution_summary"))
        or "The latest human-selection signal does not include a detailed rationale."
    )
    resolved_source_event_id = _clean_text(latest.get("source_event_id")) or source_event_id
    return {
        "source_event_id": resolved_source_event_id,
        "decision_state": decision_state,
        "latest_signal_kind": signal_kind,
        "recorded_at_utc": _clean_text(latest.get("recorded_at_utc")),
        "explanation": f"Latest human-selection signal marks this work as {decision_state}: {evidence_text}",
        "evidence": latest,
    }


def _query_from_scope(
    *,
    workflow_kind: str,
    query_text: str | None,
    events_by_id: Mapping[str, Mapping[str, Any]],
    source_event_id: str | None,
    target_event_id: str | None,
    candidate_event_ids: list[str],
) -> str:
    explicit_query = _clean_text(query_text)
    if explicit_query is not None:
        return explicit_query

    spec = DOGFOOD_WORKFLOW_SPECS[workflow_kind]
    labels = []
    for event_id in [source_event_id, target_event_id, *candidate_event_ids]:
        if event_id is None:
            continue
        labels.append(_event_label(events_by_id.get(event_id), fallback=event_id))
    if labels:
        return " ".join([str(spec["default_query"]), *labels])
    return str(spec["default_query"])


def _workflow_event_ids(
    *,
    source_event_id: str | None,
    target_event_id: str | None,
    candidate_event_ids: Iterable[str],
    winner_event_id: str | None,
) -> set[str]:
    return {
        event_id
        for event_id in [
            source_event_id,
            target_event_id,
            winner_event_id,
            *_string_list(candidate_event_ids),
        ]
        if event_id
    }


def _recall_request_payload(
    *,
    workflow_kind: str,
    query_text: str,
    source_event_id: str | None,
    candidate_event_ids: list[str],
    file_hints: list[str],
    limit: int,
    context_budget_chars: int,
) -> dict[str, Any]:
    spec = DOGFOOD_WORKFLOW_SPECS[workflow_kind]
    pinned_event_ids = candidate_event_ids if workflow_kind == WORKFLOW_COMPARE_PROPOSALS else []
    payload: dict[str, Any] = {
        "task_kind": spec["recall_task_kind"],
        "query_text": query_text,
        "request_basis": f"dogfood workflow: {spec['label']}",
        "file_hints": file_hints,
        "pinned_event_ids": pinned_event_ids,
        "limit": limit,
        "context_budget_chars": context_budget_chars,
    }
    if source_event_id is not None:
        payload["source_event_id"] = source_event_id
    return payload


def _recall_summary(bundle: Mapping[str, Any]) -> dict[str, Any]:
    source_evaluation = _mapping_dict(bundle.get("source_evaluation"))
    return {
        "task_kind": _clean_text(bundle.get("task_kind")),
        "query_text": _clean_text(bundle.get("query_text")),
        "selected_count": int(bundle.get("selected_count") or 0),
        "omitted_count": int(bundle.get("omitted_count") or 0),
        "pinned_event_ids": _string_list(bundle.get("pinned_event_ids")),
        "source_event_id": _clean_text(source_evaluation.get("source_event_id")),
        "source_selected": source_evaluation.get("source_selected"),
        "source_rank": source_evaluation.get("source_rank"),
        "miss_reason": _clean_text(source_evaluation.get("miss_reason")),
        "top_selected_event_ids": [
            _clean_text(item.get("event_id"))
            for item in bundle.get("selected_candidates") or []
            if isinstance(item, Mapping)
            if _clean_text(item.get("event_id")) is not None
        ][:5],
    }


def _comparison_preview(
    *,
    candidate_event_ids: list[str],
    winner_event_id: str | None,
    events_by_id: Mapping[str, Mapping[str, Any]],
    criteria: Iterable[str] | None = None,
) -> dict[str, Any]:
    normalized_candidates = _string_list(candidate_event_ids)
    winner = _clean_text(winner_event_id)
    ready = len(normalized_candidates) >= 2
    missing_candidate_event_ids = [
        event_id
        for event_id in normalized_candidates
        if event_id not in events_by_id
    ]
    winner_in_candidates = winner is None or winner in normalized_candidates
    ready_to_record = bool(ready and winner_in_candidates and not missing_candidate_event_ids)
    outcome = "winner_selected" if ready_to_record and winner is not None else "needs_follow_up"
    comparison_criteria = _string_list(criteria) or list(
        DOGFOOD_WORKFLOW_SPECS[WORKFLOW_COMPARE_PROPOSALS]["default_criteria"]
    )
    argv = [
        ".venv/bin/python",
        "scripts/run_evaluation_loop.py",
        "--record-comparison",
    ]
    for event_id in normalized_candidates:
        argv.extend(["--candidate-event-id", event_id])
    if winner is not None and ready_to_record:
        argv.extend(["--winner-event-id", winner])
    argv.extend(["--comparison-outcome", outcome, "--curation-preview"])
    return {
        "ready_to_record": ready_to_record,
        "outcome": outcome,
        "winner_event_id": winner,
        "candidate_count": len(normalized_candidates),
        "missing_candidate_event_ids": missing_candidate_event_ids,
        "candidates": [
            _compact_event(event_id, events_by_id.get(event_id))
            for event_id in normalized_candidates
        ],
        "criteria": comparison_criteria,
        "record_command_preview": {
            "description": "Preview only; run intentionally after human comparison.",
            "argv": argv if ready_to_record else [],
            "blocked": not ready_to_record,
        },
        "blocking_reasons": [
            reason
            for reason, blocked in (
                ("needs_at_least_two_candidates", len(normalized_candidates) < 2),
                ("winner_event_id_is_not_a_candidate", not winner_in_candidates),
                ("candidate_event_id_missing_from_index", bool(missing_candidate_event_ids)),
            )
            if blocked
        ],
    }


def _signal_action_templates(
    *,
    workflow_kind: str,
    source_event_id: str | None,
    target_event_id: str | None,
) -> list[dict[str, Any]]:
    event_id = source_event_id or "<event-id>"
    templates: list[dict[str, Any]] = []
    if workflow_kind in {WORKFLOW_REVIEW_PATCH, WORKFLOW_RECALL_PRIOR_FAILURE}:
        templates.extend(
            [
                {
                    "label": "mark_review_resolved",
                    "argv": [
                        ".venv/bin/python",
                        "scripts/run_evaluation_loop.py",
                        "--mark-review-resolved",
                        "--source-event-id",
                        event_id,
                        "--resolution-summary",
                        "<human summary>",
                        "--curation-preview",
                    ],
                },
                {
                    "label": "mark_review_unresolved",
                    "argv": [
                        ".venv/bin/python",
                        "scripts/run_evaluation_loop.py",
                        "--mark-review-unresolved",
                        "--source-event-id",
                        event_id,
                        "--resolution-summary",
                        "<human summary>",
                        "--curation-preview",
                    ],
                },
            ]
        )
    if workflow_kind == WORKFLOW_RECALL_PRIOR_FAILURE and target_event_id is not None:
        templates.append(
            {
                "label": "record_repair_link",
                "argv": [
                    ".venv/bin/python",
                    "scripts/run_evaluation_loop.py",
                    "--accept-candidate",
                    "--source-event-id",
                    event_id,
                    "--target-event-id",
                    target_event_id,
                    "--relation-kind",
                    "repairs",
                    "--rationale",
                    "<repair evidence>",
                    "--curation-preview",
                ],
            }
        )
    if workflow_kind == WORKFLOW_EXPLAIN_DECISION:
        templates.extend(
            [
                {
                    "label": "accept_after_review",
                    "argv": [
                        ".venv/bin/python",
                        "scripts/run_evaluation_loop.py",
                        "--accept-candidate",
                        "--source-event-id",
                        event_id,
                        "--rationale",
                        "<decision summary>",
                        "--curation-preview",
                    ],
                },
                {
                    "label": "reject_after_review",
                    "argv": [
                        ".venv/bin/python",
                        "scripts/run_evaluation_loop.py",
                        "--reject-candidate",
                        "--source-event-id",
                        event_id,
                        "--rationale",
                        "<decision summary>",
                        "--curation-preview",
                    ],
                },
            ]
        )
    return templates


def _default_next_actions(
    *,
    workflow_kind: str,
    source_event_id: str | None,
    candidate_event_ids: list[str],
    decision: Mapping[str, Any],
    curation_preview: Mapping[str, Any] | None,
) -> list[str]:
    if workflow_kind == WORKFLOW_COMPARE_PROPOSALS:
        if len(candidate_event_ids) < 2:
            return ["add_at_least_two_candidate_event_ids", "inspect_recall_context"]
        return ["inspect_candidate_evidence", "record_comparison_only_after_human_choice"]
    if workflow_kind == WORKFLOW_EXPLAIN_DECISION:
        if decision.get("decision_state") == "unknown":
            return ["select_source_event_with_acceptance_or_rejection", "inspect_related_signals"]
        return ["inspect_latest_decision_rationale", "check_for_stale_contradicting_signals"]
    if workflow_kind == WORKFLOW_RESOLVED_WORK_CURATION_PREVIEW:
        counts = _mapping_dict((curation_preview or {}).get("counts"))
        if int(counts.get("previewed_candidate_count") or 0) <= 0:
            return ["record_review_resolved_and_test_pass_first", "rerun_curation_preview"]
        return ["inspect_curation_candidates", "confirm_policy_separately_before_any_export"]
    if workflow_kind == WORKFLOW_RECALL_PRIOR_FAILURE:
        return ["inspect_prior_failures", "link_repair_after_verification"]
    if source_event_id is None:
        return ["select_patch_event_id", "inspect_recall_context", "record_review_signal_after_human_decision"]
    return ["inspect_recall_context", "run_or_check_verification", "record_review_signal_after_human_decision"]


def _snapshot_summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    counts = _mapping_dict(snapshot.get("counts"))
    return {
        "generated_at_utc": _clean_text(snapshot.get("generated_at_utc")),
        "event_count": int(snapshot.get("event_count") or 0),
        "signal_count": int(snapshot.get("signal_count") or 0),
        "counts": {
            key: counts.get(key)
            for key in (
                "acceptance",
                "rejection",
                "review_resolved",
                "review_unresolved",
                "test_pass",
                "test_fail",
                "pending_failures",
                "comparisons",
                "comparison_winners",
                "curation_ready",
                "curation_needs_review",
                "curation_blocked",
            )
        },
    }


def _default_curation_filters(
    *,
    workflow_kind: str,
    curation_filters: Mapping[str, Any] | None,
) -> dict[str, Any]:
    raw_filters = _mapping_dict(curation_filters)
    if workflow_kind == WORKFLOW_RESOLVED_WORK_CURATION_PREVIEW:
        raw_filters = {
            **(
                {"reasons": ["review_resolved"]}
                if not raw_filters.get("reasons") and not raw_filters.get("reason")
                else {}
            ),
            **raw_filters,
        }
    return normalize_curation_preview_filters(raw_filters)


def build_dogfood_workflow_preview(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    workflow_kind: str,
    query_text: str | None = None,
    source_event_id: str | None = None,
    target_event_id: str | None = None,
    candidate_event_ids: Iterable[str] | None = None,
    winner_event_id: str | None = None,
    file_hints: Iterable[str] | None = None,
    criteria: Iterable[str] | None = None,
    curation_filters: Mapping[str, Any] | None = None,
    limit: int = DEFAULT_LIMIT,
    context_budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS,
    snapshot: Mapping[str, Any] | None = None,
    index_summary: Mapping[str, Any] | None = None,
    curation_preview: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    normalized_workflow_kind = _normalize_workflow_kind(workflow_kind)
    spec = DOGFOOD_WORKFLOW_SPECS[normalized_workflow_kind]
    normalized_source_event_id = _clean_text(source_event_id)
    normalized_target_event_id = _clean_text(target_event_id)
    normalized_candidate_event_ids = _string_list(candidate_event_ids or [])
    normalized_winner_event_id = _clean_text(winner_event_id)
    normalized_file_hints = _string_list(file_hints or [])

    resolved_index_summary = (
        copy.deepcopy(dict(index_summary))
        if isinstance(index_summary, Mapping)
        else rebuild_memory_index(root=resolved_root, workspace_id=workspace_id)
    )
    events_by_id = _events_by_id_from_index_summary(resolved_index_summary)
    source_event_ids = _workflow_event_ids(
        source_event_id=normalized_source_event_id,
        target_event_id=normalized_target_event_id,
        candidate_event_ids=normalized_candidate_event_ids,
        winner_event_id=normalized_winner_event_id,
    )
    resolved_snapshot = (
        copy.deepcopy(dict(snapshot))
        if isinstance(snapshot, Mapping)
        else build_evaluation_snapshot(
            root=resolved_root,
            workspace_id=workspace_id,
            index_summary=resolved_index_summary,
        )
    )
    signals = read_evaluation_signals(
        evaluation_signal_log_path(workspace_id=workspace_id, root=resolved_root)
    )
    comparisons = read_evaluation_comparisons(
        evaluation_comparison_log_path(workspace_id=workspace_id, root=resolved_root)
    )
    decision = _decision_explanation(
        signals=signals,
        source_event_id=normalized_source_event_id,
    )
    if normalized_workflow_kind == WORKFLOW_EXPLAIN_DECISION and normalized_source_event_id is None:
        normalized_source_event_id = _clean_text(decision.get("source_event_id"))
        source_event_ids = _workflow_event_ids(
            source_event_id=normalized_source_event_id,
            target_event_id=normalized_target_event_id,
            candidate_event_ids=normalized_candidate_event_ids,
            winner_event_id=normalized_winner_event_id,
        )

    resolved_query = _query_from_scope(
        workflow_kind=normalized_workflow_kind,
        query_text=query_text,
        events_by_id=events_by_id,
        source_event_id=normalized_source_event_id,
        target_event_id=normalized_target_event_id,
        candidate_event_ids=normalized_candidate_event_ids,
    )
    recall_request = _recall_request_payload(
        workflow_kind=normalized_workflow_kind,
        query_text=resolved_query,
        source_event_id=normalized_source_event_id,
        candidate_event_ids=normalized_candidate_event_ids,
        file_hints=normalized_file_hints,
        limit=max(1, int(limit or DEFAULT_LIMIT)),
        context_budget_chars=max(1, int(context_budget_chars or DEFAULT_CONTEXT_BUDGET_CHARS)),
    )
    recall_bundle = build_context_bundle(
        recall_request,
        root=resolved_root,
        workspace_id=workspace_id,
        index=MemoryIndex(Path(str(resolved_index_summary["index_path"]))),
    )
    related_signals = _related_signals(signals, event_ids=source_event_ids)[:12]
    related_comparisons = _related_comparisons(comparisons, event_ids=source_event_ids)[:12]
    comparison_preview = (
        _comparison_preview(
            candidate_event_ids=normalized_candidate_event_ids,
            winner_event_id=normalized_winner_event_id,
            events_by_id=events_by_id,
            criteria=criteria,
        )
        if normalized_workflow_kind == WORKFLOW_COMPARE_PROPOSALS
        else None
    )
    if isinstance(curation_preview, Mapping):
        compact_curation_preview = copy.deepcopy(dict(curation_preview))
    elif normalized_workflow_kind == WORKFLOW_RESOLVED_WORK_CURATION_PREVIEW:
        compact_curation_preview = build_curation_export_preview(
            resolved_snapshot,
            workspace_id=workspace_id,
            filters=_default_curation_filters(
                workflow_kind=normalized_workflow_kind,
                curation_filters=curation_filters,
            ),
        )
    else:
        compact_curation_preview = None
    next_actions = _default_next_actions(
        workflow_kind=normalized_workflow_kind,
        source_event_id=normalized_source_event_id,
        candidate_event_ids=normalized_candidate_event_ids,
        decision=decision,
        curation_preview=compact_curation_preview,
    )
    signal_counts = Counter(str(item.get("signal_kind") or "") for item in related_signals)
    return {
        "schema_name": DOGFOOD_WORKFLOW_PREVIEW_SCHEMA_NAME,
        "schema_version": DOGFOOD_WORKFLOW_PREVIEW_SCHEMA_VERSION,
        "workflow_id": f"{workspace_id}:dogfood-workflow:{timestamp_slug()}:{uuid4().hex[:8]}",
        "workspace_id": workspace_id,
        "workflow_kind": normalized_workflow_kind,
        "label": spec["label"],
        "generated_at_utc": timestamp_utc(),
        "export_mode": "preview_only",
        "human_gate_required": True,
        "training_export_ready": False,
        "intent": spec["intent"],
        "scope": {
            "source_event_id": normalized_source_event_id,
            "target_event_id": normalized_target_event_id,
            "candidate_event_ids": normalized_candidate_event_ids,
            "winner_event_id": normalized_winner_event_id,
            "file_hints": normalized_file_hints,
            "events": [
                _compact_event(event_id, events_by_id.get(event_id))
                for event_id in sorted(source_event_ids)
            ],
        },
        "recall": {
            "request": recall_request,
            "summary": _recall_summary(recall_bundle),
            "bundle": recall_bundle,
        },
        "evaluation": {
            "snapshot": _snapshot_summary(resolved_snapshot),
            "related_signal_count": len(related_signals),
            "related_signal_kind_counts": dict(sorted(signal_counts.items())),
            "related_signals": related_signals,
            "related_comparisons": related_comparisons,
            "decision_explanation": decision,
        },
        "comparison_preview": comparison_preview,
        "curation_preview": compact_curation_preview,
        "operator_steps": list(spec["operator_steps"]),
        "next_actions": next_actions,
        "signal_action_templates": _signal_action_templates(
            workflow_kind=normalized_workflow_kind,
            source_event_id=normalized_source_event_id,
            target_event_id=normalized_target_event_id,
        ),
        "guardrails": {
            "preview_only": True,
            "records_signals": False,
            "records_comparisons": False,
            "writes_training_data": False,
            "starts_training_job": False,
            "requires_human_to_record_selection_or_review": True,
        },
        "paths": {
            "index_path": str(resolved_index_summary["index_path"]),
            "event_log_path": str(resolved_index_summary["event_log_path"]),
        },
    }


def record_dogfood_workflow_preview(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    workflow_kind: str,
    query_text: str | None = None,
    source_event_id: str | None = None,
    target_event_id: str | None = None,
    candidate_event_ids: Iterable[str] | None = None,
    winner_event_id: str | None = None,
    file_hints: Iterable[str] | None = None,
    criteria: Iterable[str] | None = None,
    curation_filters: Mapping[str, Any] | None = None,
    limit: int = DEFAULT_LIMIT,
    context_budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS,
) -> tuple[dict[str, Any], Path, Path]:
    resolved_root = _resolve_root(root)
    normalized_workflow_kind = _normalize_workflow_kind(workflow_kind)
    index_summary = rebuild_memory_index(root=resolved_root, workspace_id=workspace_id)
    snapshot, snapshot_latest_path, snapshot_run_path = record_evaluation_snapshot(
        root=resolved_root,
        workspace_id=workspace_id,
        index_summary=index_summary,
    )
    curation_preview: dict[str, Any] | None = None
    if normalized_workflow_kind == WORKFLOW_RESOLVED_WORK_CURATION_PREVIEW:
        filters = _default_curation_filters(
            workflow_kind=normalized_workflow_kind,
            curation_filters=curation_filters,
        )
        curation_preview, curation_latest_path, curation_run_path = record_curation_export_preview(
            root=resolved_root,
            workspace_id=workspace_id,
            snapshot=snapshot,
            filters=filters,
        )
    else:
        curation_latest_path = curation_export_preview_latest_path(workspace_id=workspace_id, root=resolved_root)
        curation_run_path = curation_export_preview_run_path(workspace_id=workspace_id, root=resolved_root)

    preview = build_dogfood_workflow_preview(
        root=resolved_root,
        workspace_id=workspace_id,
        workflow_kind=normalized_workflow_kind,
        query_text=query_text,
        source_event_id=source_event_id,
        target_event_id=target_event_id,
        candidate_event_ids=candidate_event_ids,
        winner_event_id=winner_event_id,
        file_hints=file_hints,
        criteria=criteria,
        curation_filters=curation_filters,
        limit=limit,
        context_budget_chars=context_budget_chars,
        snapshot=snapshot,
        index_summary=index_summary,
        curation_preview=curation_preview,
    )
    latest_path = dogfood_workflow_preview_latest_path(workspace_id=workspace_id, root=resolved_root)
    run_path = dogfood_workflow_preview_run_path(workspace_id=workspace_id, root=resolved_root)
    preview["paths"].update(
        {
            "workflow_preview_latest_path": str(latest_path),
            "workflow_preview_run_path": str(run_path),
            "snapshot_latest_path": str(snapshot_latest_path),
            "snapshot_run_path": str(snapshot_run_path),
        }
    )
    if curation_preview is not None:
        preview["paths"].update(
            {
                "curation_preview_latest_path": str(curation_latest_path),
                "curation_preview_run_path": str(curation_run_path),
            }
        )
    write_json(run_path, preview)
    write_json(latest_path, preview)
    return preview, latest_path, run_path


def _format_counts(counts: Mapping[str, Any], keys: Iterable[str]) -> str:
    parts = []
    for key in keys:
        value = counts.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    return "; ".join(parts) if parts else "n/a"


def format_dogfood_workflow_preview_report(preview: Mapping[str, Any]) -> str:
    workflow_kind = _clean_text(preview.get("workflow_kind")) or "unknown"
    label = _clean_text(preview.get("label")) or workflow_kind
    recall = _mapping_dict(preview.get("recall"))
    recall_summary = _mapping_dict(recall.get("summary"))
    evaluation = _mapping_dict(preview.get("evaluation"))
    snapshot = _mapping_dict(evaluation.get("snapshot"))
    snapshot_counts = _mapping_dict(snapshot.get("counts"))
    guardrails = _mapping_dict(preview.get("guardrails"))
    lines = [
        f"Dogfood workflow preview: {label}",
        f"kind: {workflow_kind}",
        "mode: preview_only / human-gated",
        f"intent: {_clean_text(preview.get('intent')) or 'n/a'}",
        "",
        "Recall:",
        (
            f"task={recall_summary.get('task_kind') or 'n/a'} "
            f"selected={int(recall_summary.get('selected_count') or 0)} "
            f"omitted={int(recall_summary.get('omitted_count') or 0)} "
            f"source_selected={recall_summary.get('source_selected')}"
        ),
        "",
        "Evaluation:",
        (
            f"events={int(snapshot.get('event_count') or 0)} "
            f"signals={int(snapshot.get('signal_count') or 0)} "
            + _format_counts(
                snapshot_counts,
                (
                    "acceptance",
                    "rejection",
                    "review_resolved",
                    "review_unresolved",
                    "test_pass",
                    "test_fail",
                    "curation_ready",
                    "curation_blocked",
                ),
            )
        ),
    ]
    decision = _mapping_dict(evaluation.get("decision_explanation"))
    if decision:
        lines.extend(["", "Decision:", _clean_text(decision.get("explanation")) or "n/a"])
    comparison_preview = preview.get("comparison_preview")
    if isinstance(comparison_preview, Mapping):
        lines.extend(
            [
                "",
                "Comparison preview:",
                (
                    f"candidates={int(comparison_preview.get('candidate_count') or 0)} "
                    f"outcome={comparison_preview.get('outcome') or 'n/a'} "
                    f"ready_to_record={bool(comparison_preview.get('ready_to_record'))}"
                ),
            ]
        )
        blocking_reasons = _string_list(comparison_preview.get("blocking_reasons"))
        if blocking_reasons:
            lines.append("blocking: " + ", ".join(blocking_reasons))
    curation_preview = preview.get("curation_preview")
    if isinstance(curation_preview, Mapping):
        lines.extend(["", format_curation_export_preview_report(curation_preview)])
    next_actions = _string_list(preview.get("next_actions"))
    if next_actions:
        lines.extend(["", "Next actions:", *[f"- {item}" for item in next_actions]])
    signal_templates = [
        item
        for item in preview.get("signal_action_templates") or []
        if isinstance(item, Mapping)
    ]
    if signal_templates:
        lines.extend(["", "Signal action templates:"])
        for item in signal_templates:
            argv = " ".join(str(part) for part in item.get("argv") or [])
            lines.append(f"- {item.get('label') or 'action'}: {argv}")
    lines.extend(
        [
            "",
            "Guardrails:",
            (
                f"signals_recorded={bool(guardrails.get('records_signals'))}; "
                f"comparisons_recorded={bool(guardrails.get('records_comparisons'))}; "
                f"training_data_written={bool(guardrails.get('writes_training_data'))}; "
                f"training_job_started={bool(guardrails.get('starts_training_job'))}"
            ),
        ]
    )
    paths = _mapping_dict(preview.get("paths"))
    workflow_run_path = _clean_text(paths.get("workflow_preview_run_path"))
    if workflow_run_path is not None:
        lines.extend(["", f"Workflow preview written: {workflow_run_path}"])
    return "\n".join(lines)
