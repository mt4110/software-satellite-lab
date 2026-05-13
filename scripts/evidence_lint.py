#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from evidence_graph import (
    EVIDENCE_GRAPH_SCHEMA_NAME,
    HARD_BLOCKING_SUPPORT_CLASSES,
    build_evidence_graph,
    validate_evidence_graph_snapshot,
)
from gemma_runtime import timestamp_utc
from workspace_state import DEFAULT_WORKSPACE_ID


EVIDENCE_LINT_SCHEMA_NAME = "software-satellite-evidence-lint-report"
EVIDENCE_LINT_SCHEMA_VERSION = 1

def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _mapping_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _clean_text(item)
        if text is None or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _coerce_utc_datetime(value: Any) -> datetime | None:
    text = _clean_text(value)
    if text is None:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _issue_id(rule_id: str, *, node_id: str | None = None, edge_id: str | None = None, detail: str | None = None) -> str:
    key = json.dumps([rule_id, node_id, edge_id, detail], sort_keys=True, ensure_ascii=False)
    return f"issue_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}"


def _issue(
    issues: list[dict[str, Any]],
    *,
    severity: str,
    rule_id: str,
    message: str,
    node_id: str | None = None,
    edge_id: str | None = None,
    source_id: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    issues.append(
        {
            "issue_id": _issue_id(rule_id, node_id=node_id, edge_id=edge_id, detail=source_id or message),
            "severity": severity,
            "rule_id": rule_id,
            "message": message,
            "node_id": node_id,
            "edge_id": edge_id,
            "source_id": source_id,
            "details": copy.deepcopy(dict(details or {})),
        }
    )


def _positive_support_requested(node: Mapping[str, Any]) -> bool:
    return bool(node.get("can_support_decision")) or _clean_text(node.get("polarity")) == "positive"


def _node_created_by_pack(node: Mapping[str, Any]) -> bool:
    metadata = _mapping_dict(node.get("metadata"))
    return (
        _clean_text(metadata.get("created_by")) == "pack"
        or _clean_text(metadata.get("source_kind")) == "pack_output"
        or bool(metadata.get("pack_run")) and _clean_text(metadata.get("created_by")) == "pack"
    )


def _node_has_graph_blockers(node: Mapping[str, Any]) -> bool:
    metadata = _mapping_dict(node.get("metadata"))
    return bool(
        _string_list(metadata.get("blocked_reasons"))
        or _string_list(metadata.get("excluded_by"))
        or _clean_text(metadata.get("blocked_reason"))
        or _clean_text(node.get("support_class")) in HARD_BLOCKING_SUPPORT_CLASSES
    )


def _learning_candidate_promoted(node: Mapping[str, Any]) -> bool:
    metadata = _mapping_dict(node.get("metadata"))
    curation = _mapping_dict(metadata.get("curation"))
    lifecycle = _mapping_dict(metadata.get("lifecycle_summary"))
    return (
        bool(node.get("can_support_decision"))
        or _clean_text(metadata.get("promotion_state")) in {"promoted", "selected", "approved"}
        or (
            _clean_text(curation.get("export_decision")) == "include_when_approved"
            and bool(metadata.get("eligible_for_supervised_candidate"))
            and _clean_text(lifecycle.get("policy_state")) == "confirmed"
        )
    )


def _edge_promotes_prior_support(edge: Mapping[str, Any]) -> bool:
    return (
        bool(edge.get("can_support_decision"))
        or (
            _clean_text(edge.get("strength")) in {"strong", "manual_pin"}
            and _clean_text(edge.get("relation_kind")) in {"recalls", "selects", "evaluates"}
        )
    )


def _node_is_benchmark_report(node: Mapping[str, Any]) -> bool:
    metadata = _mapping_dict(node.get("metadata"))
    source_kind = (_clean_text(metadata.get("source_kind")) or "").lower()
    origin = (_clean_text(metadata.get("origin")) or "").lower()
    source_id = (_clean_text(node.get("source_id")) or "").lower()
    return (
        "benchmark" in source_kind
        or "benchmark" in origin
        or "benchmark" in source_id
        or _clean_text(metadata.get("created_by")) == "benchmark_fixture"
    )


def _is_stale(created_at_utc: Any, *, generated_at_utc: str, max_age_days: int = 7) -> bool:
    created_at = _coerce_utc_datetime(created_at_utc)
    generated_at = _coerce_utc_datetime(generated_at_utc)
    if created_at is None or generated_at is None:
        return False
    return (generated_at - created_at).total_seconds() > max_age_days * 24 * 60 * 60


def build_evidence_lint_report(
    graph: Mapping[str, Any] | None = None,
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    strict: bool = False,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    generated_at = generated_at_utc or (
        _clean_text(graph.get("generated_at_utc")) if isinstance(graph, Mapping) else None
    ) or timestamp_utc()
    resolved_graph = (
        dict(graph)
        if isinstance(graph, Mapping)
        else build_evidence_graph(
            root=root,
            workspace_id=workspace_id,
            generated_at_utc=generated_at,
        )
    )
    issues: list[dict[str, Any]] = []

    for schema_issue in validate_evidence_graph_snapshot(resolved_graph):
        _issue(
            issues,
            severity="hard_fail",
            rule_id="graph_schema_invalid",
            message=schema_issue.get("message") or "Graph schema validation failed.",
            source_id=schema_issue.get("path"),
            details=schema_issue,
        )

    nodes = [dict(node) for node in resolved_graph.get("nodes") or [] if isinstance(node, Mapping)]
    edges = [dict(edge) for edge in resolved_graph.get("edges") or [] if isinstance(edge, Mapping)]
    node_by_id = {
        _clean_text(node.get("node_id")): node
        for node in nodes
        if _clean_text(node.get("node_id")) is not None
    }
    incoming_edges: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        to_node_id = _clean_text(edge.get("to_node_id"))
        if to_node_id is not None:
            incoming_edges.setdefault(to_node_id, []).append(edge)

    for node in nodes:
        node_id = _clean_text(node.get("node_id"))
        source_id = _clean_text(node.get("source_id"))
        support_class = _clean_text(node.get("support_class")) or "unknown"
        polarity = _clean_text(node.get("polarity")) or "unknown"
        metadata = _mapping_dict(node.get("metadata"))
        node_kind = _clean_text(node.get("node_kind")) or "unknown"

        if support_class == "missing_source" and _positive_support_requested(node):
            _issue(
                issues,
                severity="hard_fail",
                rule_id="missing_source_positive_support",
                message="Missing-source evidence is marked as positive decision support.",
                node_id=node_id,
                source_id=source_id,
            )
        if support_class == "modified_source" and _positive_support_requested(node):
            _issue(
                issues,
                severity="hard_fail",
                rule_id="modified_source_positive_support",
                message="Modified-source evidence is marked as positive decision support.",
                node_id=node_id,
                source_id=source_id,
            )
        if support_class == "current_review_subject" and _positive_support_requested(node):
            _issue(
                issues,
                severity="hard_fail",
                rule_id="current_review_subject_prior_support",
                message="Current review subject is promoted as prior support.",
                node_id=node_id,
                source_id=source_id,
            )
        if support_class == "future_evidence" and _positive_support_requested(node):
            _issue(
                issues,
                severity="hard_fail",
                rule_id="future_evidence_prior_support",
                message="Future evidence is marked as prior decision support.",
                node_id=node_id,
                source_id=source_id,
            )
        if support_class == "unverified_agent_claim" and _positive_support_requested(node):
            _issue(
                issues,
                severity="hard_fail",
                rule_id="unverified_agent_claim_positive_support",
                message="Unverified agent claim is marked as positive support.",
                node_id=node_id,
                source_id=source_id,
            )
        if support_class == "contradictory" and _positive_support_requested(node):
            _issue(
                issues,
                severity="hard_fail",
                rule_id="contradictory_verdicts_promoted",
                message="Contradictory evidence is promoted without a resolving signal.",
                node_id=node_id,
                source_id=source_id,
            )
        if node_kind == "event" and bool(node.get("can_support_decision")) and polarity == "positive":
            contradiction_edges = [
                edge
                for edge in incoming_edges.get(node_id or "", [])
                if edge.get("relation_kind") == "contradicts"
                and edge.get("causal_validity") != "invalid"
            ]
            for edge in contradiction_edges:
                _issue(
                    issues,
                    severity="hard_fail",
                    rule_id="contradictory_verdicts_promoted",
                    message="Positive event support has an unresolved contradiction edge.",
                    node_id=node_id,
                    edge_id=_clean_text(edge.get("edge_id")),
                    source_id=source_id,
                    details={"edge_explanation": _clean_text(edge.get("explanation"))},
                )
        if node_kind == "learning_candidate" and _learning_candidate_promoted(node) and _node_has_graph_blockers(node):
            _issue(
                issues,
                severity="hard_fail",
                rule_id="learning_candidate_promoted_with_graph_blockers",
                message="Learning candidate is promoted while graph blockers are still present.",
                node_id=node_id,
                source_id=source_id,
                details={
                    "blocked_reasons": _string_list(metadata.get("blocked_reasons")),
                    "queue_state": _clean_text(metadata.get("queue_state")),
                },
            )
        if node_kind == "learning_candidate" and (
            support_class == "missing_source" or _clean_text(metadata.get("queue_state")) == "missing_source"
        ):
            _issue(
                issues,
                severity="hard_fail",
                rule_id="learning_candidate_missing_source",
                message="Learning candidate is missing its source evidence.",
                node_id=node_id,
                source_id=source_id,
                details={
                    "blocked_reasons": _string_list(metadata.get("blocked_reasons")),
                    "queue_state": _clean_text(metadata.get("queue_state")),
                },
            )
        if _node_created_by_pack(node) and bool(node.get("can_support_decision")):
            _issue(
                issues,
                severity="hard_fail",
                rule_id="pack_output_bypasses_support_kernel",
                message="Pack-created output is marked as decision support without going through the support kernel.",
                node_id=node_id,
                source_id=source_id,
            )

        if node_kind in {"event", "review", "learning_candidate"} and _clean_text(node.get("target_fingerprint")) is None:
            _issue(
                issues,
                severity="warning",
                rule_id="missing_target_fingerprint",
                message="Evidence has no target fingerprint.",
                node_id=node_id,
                source_id=source_id,
            )
        if node_kind == "artifact" and bool(metadata.get("report_excerpt_fully_redacted")):
            _issue(
                issues,
                severity="warning",
                rule_id="source_excerpt_fully_redacted",
                message="Source excerpt is fully redacted.",
                node_id=node_id,
                source_id=source_id,
            )
        if support_class == "manual_pin_diagnostic":
            _issue(
                issues,
                severity="warning",
                rule_id="manual_pin_without_source_strength",
                message="Manual pin remains diagnostic and does not provide source strength.",
                node_id=node_id,
                source_id=source_id,
            )
        if _node_is_benchmark_report(node) and _is_stale(
            node.get("created_at_utc"),
            generated_at_utc=generated_at,
        ):
            _issue(
                issues,
                severity="warning",
                rule_id="stale_benchmark_report",
                message="Benchmark-derived evidence is older than the freshness window.",
                node_id=node_id,
                source_id=source_id,
                details={"max_age_days": 7, "created_at_utc": _clean_text(node.get("created_at_utc"))},
            )

    for edge in edges:
        edge_id = _clean_text(edge.get("edge_id"))
        support_class = _clean_text(edge.get("support_class")) or "unknown"
        relation_kind = _clean_text(edge.get("relation_kind")) or "unknown"
        from_node = node_by_id.get(_clean_text(edge.get("from_node_id")))
        to_node = node_by_id.get(_clean_text(edge.get("to_node_id")))

        if support_class == "missing_source" and _edge_promotes_prior_support(edge):
            _issue(
                issues,
                severity="hard_fail",
                rule_id="missing_source_positive_support",
                message="Missing-source relation is promoted as prior support.",
                edge_id=edge_id,
                source_id=_clean_text(edge.get("to_node_id")),
            )
        if support_class == "modified_source" and _edge_promotes_prior_support(edge):
            _issue(
                issues,
                severity="hard_fail",
                rule_id="modified_source_positive_support",
                message="Modified-source relation is promoted as prior support.",
                edge_id=edge_id,
            )
        if support_class == "current_review_subject" and _edge_promotes_prior_support(edge):
            _issue(
                issues,
                severity="hard_fail",
                rule_id="current_review_subject_prior_support",
                message="Current review subject relation is promoted as prior support.",
                edge_id=edge_id,
            )
        if support_class == "future_evidence" and _edge_promotes_prior_support(edge):
            _issue(
                issues,
                severity="hard_fail",
                rule_id="future_evidence_prior_support",
                message="Future evidence relation is promoted as prior support.",
                edge_id=edge_id,
            )
        if support_class == "unverified_agent_claim" and _edge_promotes_prior_support(edge):
            _issue(
                issues,
                severity="hard_fail",
                rule_id="unverified_agent_claim_positive_support",
                message="Unverified claim relation is promoted as prior support.",
                edge_id=edge_id,
            )
        if relation_kind == "recalls" and edge.get("causal_validity") == "invalid" and _edge_promotes_prior_support(edge):
            explanation = _clean_text(edge.get("explanation")) or ""
            if "future_evidence" in explanation:
                rule_id = "future_evidence_prior_support"
                message = "Recall edge uses future evidence as prior support."
            else:
                rule_id = "current_review_subject_prior_support"
                message = "Recall edge uses the current subject or matching target as prior support."
            _issue(
                issues,
                severity="hard_fail",
                rule_id=rule_id,
                message=message,
                edge_id=edge_id,
                details={"explanation": explanation},
            )
        if edge.get("created_by") == "pack" and _edge_promotes_prior_support(edge):
            target_can_support = bool(_mapping_dict(to_node).get("can_support_decision"))
            if target_can_support or relation_kind in {"selects", "evaluates"}:
                _issue(
                    issues,
                    severity="hard_fail",
                    rule_id="pack_output_bypasses_support_kernel",
                    message="Pack-created relation attempts to promote decision-bearing output.",
                    edge_id=edge_id,
                    source_id=_clean_text(_mapping_dict(to_node).get("source_id")),
                )
        if edge.get("strength") == "manual_pin" and support_class not in {"source_linked_prior", "negative_prior"}:
            _issue(
                issues,
                severity="warning",
                rule_id="manual_pin_without_source_strength",
                message="Manual pin edge does not have source-backed strength.",
                edge_id=edge_id,
                details={"support_class": support_class},
            )

    recall_nodes = [node for node in nodes if node.get("node_kind") == "recall"]
    for node in recall_nodes:
        metadata = _mapping_dict(node.get("metadata"))
        if not metadata.get("has_human_usefulness_signal"):
            _issue(
                issues,
                severity="warning",
                rule_id="missing_human_usefulness_signal",
                message="Recall evidence has no recorded human usefulness signal.",
                node_id=_clean_text(node.get("node_id")),
                source_id=_clean_text(node.get("source_id")),
            )

    false_recall_counter: Counter[str] = Counter()
    for node in nodes:
        if node.get("node_kind") != "signal":
            continue
        evidence = _mapping_dict(_mapping_dict(node.get("metadata")).get("evidence"))
        usefulness = _clean_text(evidence.get("recall_usefulness"))
        if usefulness in {"irrelevant", "misleading"}:
            false_recall_counter[_clean_text(_mapping_dict(node.get("metadata")).get("source_event_id")) or "unknown"] += 1
    for event_id, count in sorted(false_recall_counter.items()):
        if count >= 2:
            _issue(
                issues,
                severity="warning",
                rule_id="repeated_false_recall_pattern",
                message="Repeated false recall pattern detected.",
                source_id=event_id,
                details={"false_recall_count": count},
            )

    hard_fail_count = sum(1 for issue in issues if issue.get("severity") == "hard_fail")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")
    rule_counts = Counter(str(issue.get("rule_id") or "unknown") for issue in issues)
    report = {
        "schema_name": EVIDENCE_LINT_SCHEMA_NAME,
        "schema_version": EVIDENCE_LINT_SCHEMA_VERSION,
        "workspace_id": _clean_text(resolved_graph.get("workspace_id")) or workspace_id,
        "generated_at_utc": generated_at,
        "strict": bool(strict),
        "verdict": "fail" if hard_fail_count else "pass",
        "hard_fail_count": hard_fail_count,
        "warning_count": warning_count,
        "issues": sorted(
            issues,
            key=lambda issue: (
                0 if issue.get("severity") == "hard_fail" else 1,
                str(issue.get("rule_id") or ""),
                str(issue.get("source_id") or ""),
                str(issue.get("issue_id") or ""),
            ),
        ),
        "counts": {
            "rules": {key: int(value) for key, value in sorted(rule_counts.items())},
            "graph_node_count": len(nodes),
            "graph_edge_count": len(edges),
            "learning_preview_graph_blocker_count": int(
                _mapping_dict(resolved_graph.get("counts")).get("learning_preview_graph_blocker_count") or 0
            ),
        },
        "exit_gate": {
            "evidence_lint_strict_passes": hard_fail_count == 0,
            "support_kernel_decisions_match_graph_nodes": bool(
                _mapping_dict(resolved_graph.get("counts")).get("support_kernel_decisions_match_graph_nodes")
            ),
            "learning_preview_graph_blocker_count_is_visible": "learning_preview_graph_blocker_count"
            in _mapping_dict(resolved_graph.get("counts")),
        },
        "graph_schema_name": _clean_text(resolved_graph.get("schema_name")) or EVIDENCE_GRAPH_SCHEMA_NAME,
        "graph_digest": _clean_text(resolved_graph.get("graph_digest")),
    }
    return report


def format_evidence_lint_report(report: Mapping[str, Any]) -> str:
    lines = [
        "Evidence lint",
        f"Workspace: {_clean_text(report.get('workspace_id')) or DEFAULT_WORKSPACE_ID}",
        f"Verdict: {_clean_text(report.get('verdict')) or 'unknown'}",
        f"Hard failures: {int(report.get('hard_fail_count') or 0)}",
        f"Warnings: {int(report.get('warning_count') or 0)}",
    ]
    issues = [issue for issue in report.get("issues") or [] if isinstance(issue, Mapping)]
    if issues:
        lines.extend(("", "Issues:"))
        for issue in issues[:30]:
            target = _clean_text(issue.get("node_id")) or _clean_text(issue.get("edge_id")) or _clean_text(issue.get("source_id")) or "graph"
            lines.append(
                f"- [{issue.get('severity')}] {issue.get('rule_id')} {target}: {issue.get('message')}"
            )
    else:
        lines.extend(("", "No hard-fail evidence graph issues found."))
    return "\n".join(lines)
