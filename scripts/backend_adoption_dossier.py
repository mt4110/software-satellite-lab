#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluation_loop import (
    EVALUATION_COMPARISON_SCHEMA_NAME,
    evaluation_comparison_log_path,
    read_evaluation_comparisons,
    software_work_events_by_id,
)
from evidence_lint import build_evidence_lint_report
from evidence_support import build_evidence_support_result
from gemma_runtime import repo_root, timestamp_utc
from workspace_state import DEFAULT_WORKSPACE_ID


BACKEND_ADOPTION_DOSSIER_SCHEMA_NAME = "software-satellite-backend-adoption-dossier"
BACKEND_ADOPTION_DOSSIER_SCHEMA_VERSION = 1

WORKFLOW_KINDS = {
    "review_git",
    "agent_session_intake",
    "pack_report",
    "backend_compare",
    "learning_inspection",
}
REPO_SCOPES = {"current_repo", "fixture_only", "dogfood_only"}
RISK_SCOPES = {"experimental", "default_candidate", "default_backend"}
RECOMMENDATIONS = {"adopt", "reject", "experiment_only", "insufficient_evidence"}

SOURCE_BLOCKERS = {
    "event_not_found",
    "missing_source",
    "modified_source",
    "binary_source_refused",
    "oversize_source",
    "outside_workspace",
    "symlink_refused",
    "vault_checksum_mismatch",
}
FAILURE_STATUSES = {"failed", "fail", "blocked", "error", "timeout", "rejected", "reject", "needs_fix"}
FAILURE_QUALITY = {"fail", "failed", "quality_fail"}
FAILURE_EXECUTION = {"fail", "failed", "blocked", "error", "timeout"}
CONTRADICTION_RULES = {"contradictory_verdicts_promoted"}


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _mapping_dict(value: Any) -> dict[str, Any]:
    return copy.deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _clean_text(item)
        if text is None or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _normalize_choice(value: str | None, *, allowed: set[str], default: str, field_name: str) -> str:
    normalized = (_clean_text(value) or default).lower().replace("-", "_")
    if normalized not in allowed:
        raise ValueError(f"Unsupported {field_name} `{value}`.")
    return normalized


def _stable_dossier_id(
    *,
    workspace_id: str,
    comparison_id: str | None,
    candidate_selector: str | None,
    baseline_selector: str | None,
    scope: Mapping[str, Any],
) -> str:
    key = json.dumps(
        {
            "comparison_id": comparison_id,
            "candidate_selector": candidate_selector,
            "baseline_selector": baseline_selector,
            "scope": dict(scope),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"{workspace_id}:backend-adoption-dossier:{digest}"


def _comparison_candidates(comparison: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(dict(item))
        for item in comparison.get("candidates") or []
        if isinstance(item, Mapping)
    ]


def _candidate_metadata(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping_dict(candidate.get("backend_metadata"))


def _candidate_backend_id(candidate: Mapping[str, Any]) -> str | None:
    return _clean_text(_candidate_metadata(candidate).get("backend_id"))


def _candidate_model_id(candidate: Mapping[str, Any]) -> str | None:
    return _clean_text(_candidate_metadata(candidate).get("model_id"))


def _candidate_event_id(candidate: Mapping[str, Any]) -> str | None:
    return _clean_text(candidate.get("event_id"))


def _candidate_matches(candidate: Mapping[str, Any], selector: str | None) -> bool:
    cleaned = _clean_text(selector)
    if cleaned is None:
        return False
    values = {
        _candidate_event_id(candidate),
        _candidate_backend_id(candidate),
        _candidate_model_id(candidate),
        _clean_text(_candidate_metadata(candidate).get("display_name")),
    }
    return cleaned in {value for value in values if value is not None}


def _comparison_role(candidate: Mapping[str, Any], comparison: Mapping[str, Any]) -> str:
    event_id = _candidate_event_id(candidate)
    winner_event_id = _clean_text(comparison.get("winner_event_id"))
    outcome = _clean_text(comparison.get("outcome"))
    if outcome == "winner_selected":
        return "winner" if event_id is not None and event_id == winner_event_id else "loser"
    if outcome == "tie":
        return "tie"
    return "candidate"


def _select_adoption_candidate(
    candidates: list[dict[str, Any]],
    *,
    comparison: Mapping[str, Any],
    selector: str | None,
) -> dict[str, Any] | None:
    if selector is not None:
        return next((candidate for candidate in candidates if _candidate_matches(candidate, selector)), None)
    winner_event_id = _clean_text(comparison.get("winner_event_id"))
    if winner_event_id is not None:
        return next((candidate for candidate in candidates if _candidate_event_id(candidate) == winner_event_id), None)
    return candidates[0] if candidates else None


def _select_baseline_candidate(
    candidates: list[dict[str, Any]],
    *,
    selected_candidate: Mapping[str, Any] | None,
    selector: str | None,
) -> dict[str, Any] | None:
    if selector is not None:
        return next((candidate for candidate in candidates if _candidate_matches(candidate, selector)), None)
    selected_event_id = _candidate_event_id(selected_candidate or {})
    for candidate in candidates:
        if _candidate_event_id(candidate) != selected_event_id:
            return candidate
    return None


def _event_outcome(event: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = _mapping_dict(event)
    outcome = _mapping_dict(payload.get("outcome"))
    content = _mapping_dict(payload.get("content"))
    options = _mapping_dict(content.get("options"))
    session = _mapping_dict(payload.get("session"))
    return {
        "status": _clean_text(outcome.get("status")),
        "quality_status": _clean_text(outcome.get("quality_status")) or _clean_text(options.get("quality_status")),
        "execution_status": _clean_text(outcome.get("execution_status")) or _clean_text(options.get("execution_status")),
        "event_kind": _clean_text(payload.get("event_kind")),
        "session_surface": _clean_text(session.get("surface")),
        "output_excerpt": (_clean_text(content.get("output_text")) or "")[:500],
        "notes": _string_list(content.get("notes")),
    }


def _artifact_ref_summaries(event: Mapping[str, Any] | None, support: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = _mapping_dict(event)
    source_refs = _mapping_dict(payload.get("source_refs"))
    content = _mapping_dict(payload.get("content"))
    options = _mapping_dict(content.get("options"))
    refs: list[dict[str, Any]] = []

    for raw in (
        source_refs.get("artifact_vault_refs"),
        source_refs.get("artifact_refs"),
        options.get("artifact_vault_refs"),
        payload.get("artifact_vault_refs"),
    ):
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, Mapping):
                    refs.append(_source_ref_summary(item))
        elif isinstance(raw, Mapping):
            refs.append(_source_ref_summary(raw))

    legacy_artifact = _mapping_dict(source_refs.get("artifact_ref"))
    if legacy_artifact:
        refs.append(
            {
                "artifact_id": _clean_text(legacy_artifact.get("entry_id")),
                "kind": _clean_text(legacy_artifact.get("artifact_kind")),
                "path": _clean_text(legacy_artifact.get("artifact_path")),
                "repo_relative_path": _clean_text(legacy_artifact.get("artifact_workspace_relative_path")),
                "source_state": "legacy_ref",
                "capture_state": "not_vaulted",
            }
        )

    support_refs = set(_string_list(support.get("artifact_refs")))
    if support_refs:
        seen_ids = {_clean_text(ref.get("artifact_id")) for ref in refs}
        for artifact_id in sorted(support_refs - {value for value in seen_ids if value is not None}):
            refs.append({"artifact_id": artifact_id})

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ref in refs:
        key = json.dumps(ref, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped


def _source_ref_summary(ref: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "artifact_id": _clean_text(ref.get("artifact_id")),
            "kind": _clean_text(ref.get("kind")),
            "path": _clean_text(ref.get("original_path")),
            "repo_relative_path": _clean_text(ref.get("repo_relative_path")),
            "sha256": _clean_text(ref.get("sha256")),
            "source_state": _clean_text(ref.get("source_state")),
            "capture_state": _clean_text(ref.get("capture_state")),
        }.items()
        if value is not None
    }


def _source_blocked(support: Mapping[str, Any]) -> bool:
    support_class = _clean_text(support.get("support_class"))
    blockers = set(_string_list(support.get("blockers")))
    return bool(blockers & SOURCE_BLOCKERS) or support_class in {"missing_source", "modified_source"}


def _has_source_link(outcome: Mapping[str, Any]) -> bool:
    support = _mapping_dict(outcome.get("positive_support"))
    risk_support = _mapping_dict(outcome.get("risk_support"))
    return not (_source_blocked(support) and _source_blocked(risk_support))


def _is_failure_or_regression(outcome: Mapping[str, Any]) -> bool:
    event_outcome = _mapping_dict(outcome.get("event_outcome"))
    positive_support = _mapping_dict(outcome.get("positive_support"))
    risk_support = _mapping_dict(outcome.get("risk_support"))
    status = (_clean_text(event_outcome.get("status")) or "").lower()
    quality = (_clean_text(event_outcome.get("quality_status")) or "").lower()
    execution = (_clean_text(event_outcome.get("execution_status")) or "").lower()
    return (
        status in FAILURE_STATUSES
        or quality in FAILURE_QUALITY
        or execution in FAILURE_EXECUTION
        or bool(risk_support.get("can_support_decision"))
        or _clean_text(positive_support.get("support_class")) == "contradictory"
        or _clean_text(risk_support.get("support_class")) == "contradictory"
    )


def _outcome_record(
    candidate: Mapping[str, Any],
    *,
    comparison: Mapping[str, Any],
    role: str,
    event: Mapping[str, Any] | None,
    root: Path,
    workspace_id: str,
) -> dict[str, Any]:
    event_id = _candidate_event_id(candidate) or "missing-event"
    positive_support = build_evidence_support_result(
        event_id,
        event=event,
        requested_polarity="positive",
        workspace_id=workspace_id,
        root=root,
    )
    risk_support = build_evidence_support_result(
        event_id,
        event=event,
        requested_polarity="risk",
        workspace_id=workspace_id,
        root=root,
    )
    metadata = _candidate_metadata(candidate)
    support_for_refs = positive_support if not _source_blocked(positive_support) else risk_support
    record = {
        "role": role,
        "comparison_role": _comparison_role(candidate, comparison),
        "event_id": _candidate_event_id(candidate),
        "backend_id": _clean_text(metadata.get("backend_id")),
        "model_id": _clean_text(metadata.get("model_id")),
        "display_name": _clean_text(metadata.get("display_name")),
        "adapter_kind": _clean_text(metadata.get("adapter_kind")),
        "compatibility_status": _clean_text(metadata.get("compatibility_status"))
        or _clean_text(_mapping_dict(metadata.get("compatibility")).get("status")),
        "event_outcome": _event_outcome(event),
        "positive_support": positive_support,
        "risk_support": risk_support,
        "source_artifact_refs": _artifact_ref_summaries(event, support_for_refs),
    }
    record["source_linked"] = _has_source_link(record)
    record["failure_or_regression"] = _is_failure_or_regression(record)
    return record


def _metadata_for_role(outcome: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(outcome, Mapping):
        return {}
    metadata = {
        "backend_id": _clean_text(outcome.get("backend_id")),
        "model_id": _clean_text(outcome.get("model_id")),
        "display_name": _clean_text(outcome.get("display_name")),
        "adapter_kind": _clean_text(outcome.get("adapter_kind")),
        "compatibility_status": _clean_text(outcome.get("compatibility_status")),
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _has_human_rationale(comparison: Mapping[str, Any]) -> bool:
    return _clean_text(comparison.get("rationale")) is not None


def _rollback_plan(
    *,
    explicit_plan: str | None,
    baseline_backend: str | None,
    candidate_backend: str | None,
) -> dict[str, Any]:
    if explicit_plan is not None:
        cleaned = _clean_text(explicit_plan)
        return {
            "present": cleaned is not None,
            "source": "provided" if cleaned is not None else "absent",
            "plan": cleaned,
        }
    baseline = _clean_text(baseline_backend)
    candidate = _clean_text(candidate_backend)
    if baseline is None:
        return {"present": False, "source": "absent", "plan": None}
    plan = (
        f"Keep `{baseline}` available as the rollback backend. "
        f"If `{candidate or 'candidate'}` adoption causes regressions, restore `{baseline}` for this scoped workflow "
        "and rerun the same source-linked comparison before trying again."
    )
    return {"present": True, "source": "derived_from_baseline", "plan": plan}


def _benchmark_gate(benchmark_report: Mapping[str, Any] | None, *, strict: bool) -> dict[str, Any]:
    report = _mapping_dict(benchmark_report)
    if not report:
        return {
            "available": False,
            "passed": not strict,
            "strict_required": bool(strict),
            "critical_false_support_count": 0,
            "source": None,
        }
    counts = _mapping_dict(report.get("counts"))
    critical_false_count = int(
        report.get("critical_false_support_count")
        or report.get("critical_false_evidence_count")
        or report.get("false_support_count")
        or counts.get("critical_false_support")
        or counts.get("critical_false_evidence")
        or counts.get("false_support")
        or 0
    )
    raw_passed = report.get("passed")
    verdict = _clean_text(report.get("verdict"))
    status = _clean_text(report.get("status"))
    passed = bool(raw_passed) if isinstance(raw_passed, bool) else verdict in {"pass", "passed"} or status in {"pass", "passed"}
    if critical_false_count > 0:
        passed = False
    return {
        "available": True,
        "passed": passed,
        "strict_required": bool(strict),
        "critical_false_support_count": critical_false_count,
        "source": _clean_text(report.get("schema_name")) or _clean_text(report.get("report_path")),
    }


def _read_benchmark_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Benchmark report `{path}` must contain a JSON object.")
    report = copy.deepcopy(dict(payload))
    report.setdefault("report_path", str(path))
    return report


def _lint_gate(
    *,
    root: Path,
    workspace_id: str,
    strict: bool,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"available": False, "verdict": "not_run", "hard_fail_count": 0, "issues": []}
    report = build_evidence_lint_report(root=root, workspace_id=workspace_id, strict=strict)
    issues = [
        {
            "severity": _clean_text(issue.get("severity")),
            "rule_id": _clean_text(issue.get("rule_id")),
            "message": _clean_text(issue.get("message")),
            "source_id": _clean_text(issue.get("source_id")),
        }
        for issue in report.get("issues") or []
        if isinstance(issue, Mapping)
    ]
    return {
        "available": True,
        "verdict": _clean_text(report.get("verdict")) or "unknown",
        "hard_fail_count": int(report.get("hard_fail_count") or 0),
        "issues": issues[:30],
    }


def _rule_evaluation(
    *,
    comparison: Mapping[str, Any],
    candidate_outcome: Mapping[str, Any] | None,
    baseline_outcome: Mapping[str, Any] | None,
    rollback: Mapping[str, Any],
    benchmark_gate: Mapping[str, Any],
    lint_gate: Mapping[str, Any],
    adoption_scope: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    candidate = _mapping_dict(candidate_outcome)
    baseline = _mapping_dict(baseline_outcome)
    comparison_outcome = _clean_text(comparison.get("outcome"))
    candidate_role = _clean_text(candidate.get("comparison_role"))
    candidate_positive = _mapping_dict(candidate.get("positive_support"))
    candidate_risk = _mapping_dict(candidate.get("risk_support"))
    baseline_positive = _mapping_dict(baseline.get("positive_support"))
    baseline_risk = _mapping_dict(baseline.get("risk_support"))

    if not candidate:
        blockers.append("candidate_not_found")
    if not baseline:
        blockers.append("baseline_not_found")

    if candidate and not _has_source_link(candidate):
        blockers.append("source_evidence_missing")
    if baseline and not _has_source_link(baseline):
        blockers.append("baseline_source_evidence_missing")

    if candidate and not (
        candidate_positive.get("can_support_decision")
        and _clean_text(candidate_positive.get("support_polarity")) == "positive"
    ):
        blockers.append("candidate_positive_support_missing")

    if _clean_text(candidate_positive.get("support_class")) == "unverified_agent_claim" or "unverified_agent_claim" in _string_list(
        candidate_positive.get("blockers")
    ):
        blockers.append("candidate_only_wins_by_agent_self_claim")

    if _clean_text(candidate_positive.get("support_class")) == "contradictory" or _clean_text(
        candidate_risk.get("support_class")
    ) == "contradictory":
        blockers.append("unresolved_contradictions")

    if not _has_human_rationale(comparison):
        blockers.append("human_rationale_absent")
    if not rollback.get("present"):
        blockers.append("rollback_path_absent")
    if int(benchmark_gate.get("critical_false_support_count") or 0) > 0:
        blockers.append("critical_false_support_present")
    if benchmark_gate.get("strict_required") and not benchmark_gate.get("available"):
        blockers.append("benchmark_gate_missing")
    elif benchmark_gate.get("passed") is False:
        blockers.append("benchmark_gate_failed")
    if int(lint_gate.get("hard_fail_count") or 0) > 0:
        rules = {
            _clean_text(issue.get("rule_id"))
            for issue in lint_gate.get("issues") or []
            if isinstance(issue, Mapping)
        }
        if rules & CONTRADICTION_RULES:
            blockers.append("unresolved_contradictions")
        warnings.append("evidence_lint_hard_failures_present")

    if comparison_outcome == "winner_selected" and candidate_role == "loser":
        blockers.append("candidate_lost_to_baseline")
    if comparison_outcome in {"needs_follow_up", None}:
        warnings.append("comparison_needs_follow_up")
    if comparison_outcome == "tie":
        warnings.append("comparison_tie")
    if baseline and (
        baseline_positive.get("can_support_decision") is False
        and baseline_risk.get("can_support_decision") is False
    ):
        warnings.append("baseline_has_no_decision_support")
    if candidate.get("failure_or_regression"):
        warnings.append("candidate_negative_evidence_visible")
    if _clean_text(adoption_scope.get("risk_scope")) == "default_backend":
        warnings.append("default_backend_switch_requires_separate_approval")

    blockers = sorted(set(blockers))
    warnings = sorted(set(warnings))
    return {
        "blockers": blockers,
        "warnings": warnings,
        "rules": {
            "source_evidence_required": True,
            "critical_false_support_blocks_adoption": True,
            "benchmark_gate_required_when_available": True,
            "agent_self_claim_ignored": True,
            "rollback_plan_required": True,
            "human_rationale_required": True,
            "unresolved_contradictions_block_adoption": True,
            "default_backend_switch_not_silent": True,
        },
    }


def _recommendation(
    *,
    rule_evaluation: Mapping[str, Any],
    comparison: Mapping[str, Any],
    candidate_outcome: Mapping[str, Any] | None,
    adoption_scope: Mapping[str, Any],
) -> dict[str, Any]:
    blockers = set(_string_list(rule_evaluation.get("blockers")))
    warnings = set(_string_list(rule_evaluation.get("warnings")))
    candidate_role = _clean_text(_mapping_dict(candidate_outcome).get("comparison_role"))
    comparison_outcome = _clean_text(comparison.get("outcome"))

    if "candidate_lost_to_baseline" in blockers:
        value = "reject"
        rationale = "The candidate lost the scoped comparison against the selected baseline."
    elif blockers & {"critical_false_support_present", "benchmark_gate_failed", "unresolved_contradictions"}:
        value = "reject"
        rationale = "Hard negative evidence or failed gates make adoption unsafe for this scope."
    elif blockers:
        value = "insufficient_evidence"
        rationale = "Required source, rationale, rollback, or positive support evidence is missing."
    elif comparison_outcome == "tie" or "comparison_tie" in warnings:
        value = "experiment_only"
        rationale = "The scoped comparison is tied, so the candidate can only remain experimental."
    elif comparison_outcome == "needs_follow_up" or "comparison_needs_follow_up" in warnings:
        value = "insufficient_evidence"
        rationale = "The comparison requires follow-up before an adoption recommendation."
    elif _clean_text(adoption_scope.get("risk_scope")) == "default_backend":
        value = "experiment_only"
        rationale = "The evidence may support experimentation, but default-backend switching requires a separate explicit approval."
    elif candidate_role == "winner":
        value = "adopt"
        rationale = "The candidate won the scoped, source-linked comparison and no adoption blocker fired."
    else:
        value = "experiment_only"
        rationale = "The evidence is source-linked, but the candidate has not clearly won enough for adoption."

    return {
        "value": value,
        "rationale": rationale,
        "allowed_values": sorted(RECOMMENDATIONS),
    }


def _support_summary(outcomes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    support_classes: Counter[str] = Counter()
    support_polarities: Counter[str] = Counter()
    blockers: Counter[str] = Counter()
    for outcome in outcomes:
        for support_key in ("positive_support", "risk_support"):
            support = _mapping_dict(outcome.get(support_key))
            support_classes[_clean_text(support.get("support_class")) or "unknown"] += 1
            support_polarities[_clean_text(support.get("support_polarity")) or "none"] += 1
            blockers.update(_string_list(support.get("blockers")))
    return {
        "support_class_counts": dict(sorted(support_classes.items())),
        "support_polarity_counts": dict(sorted(support_polarities.items())),
        "blocker_counts": dict(sorted((key, int(value)) for key, value in blockers.items())),
    }


def _cost_latency_metadata(outcome: Mapping[str, Any] | None, *, source_candidate: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(outcome, Mapping):
        return {}
    metadata = _candidate_metadata(source_candidate or {})
    raw_metadata = _mapping_dict(metadata.get("metadata"))
    limits = _mapping_dict(metadata.get("limits"))
    interesting_keys = ("cost", "price", "latency", "duration", "tokens", "seconds", "rate")
    filtered_metadata = {
        key: copy.deepcopy(value)
        for key, value in raw_metadata.items()
        if any(token in key.lower() for token in interesting_keys)
    }
    filtered_limits = {
        key: copy.deepcopy(value)
        for key, value in limits.items()
        if any(token in key.lower() for token in interesting_keys + ("context", "output"))
    }
    return {
        key: value
        for key, value in {
            "backend_id": _clean_text(outcome.get("backend_id")),
            "model_id": _clean_text(outcome.get("model_id")),
            "metadata": filtered_metadata,
            "limits": filtered_limits,
        }.items()
        if value not in (None, {}, [])
    }


def build_backend_adoption_dossier(
    comparison: Mapping[str, Any],
    *,
    events_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    candidate_backend: str | None = None,
    baseline_backend: str | None = None,
    workflow_kind: str = "backend_compare",
    repo_scope: str = "current_repo",
    risk_scope: str = "experimental",
    rollback_plan: str | None = None,
    benchmark_report: Mapping[str, Any] | None = None,
    strict: bool = False,
    run_evidence_lint: bool = False,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    resolved_workspace_id = _clean_text(comparison.get("workspace_id")) or workspace_id
    adoption_scope = {
        "workflow_kind": _normalize_choice(workflow_kind, allowed=WORKFLOW_KINDS, default="backend_compare", field_name="workflow_kind"),
        "repo_scope": _normalize_choice(repo_scope, allowed=REPO_SCOPES, default="current_repo", field_name="repo_scope"),
        "risk_scope": _normalize_choice(risk_scope, allowed=RISK_SCOPES, default="experimental", field_name="risk_scope"),
    }

    candidates = _comparison_candidates(comparison)
    selected_candidate = _select_adoption_candidate(
        candidates,
        comparison=comparison,
        selector=candidate_backend,
    )
    selected_baseline = _select_baseline_candidate(
        candidates,
        selected_candidate=selected_candidate,
        selector=baseline_backend,
    )
    events = {
        str(key): copy.deepcopy(dict(value))
        for key, value in (events_by_id or {}).items()
        if isinstance(value, Mapping)
    }

    outcome_records: list[dict[str, Any]] = []
    for candidate in candidates:
        event_id = _candidate_event_id(candidate)
        role = "other"
        if selected_candidate is not None and event_id == _candidate_event_id(selected_candidate):
            role = "candidate"
        elif selected_baseline is not None and event_id == _candidate_event_id(selected_baseline):
            role = "baseline"
        outcome_records.append(
            _outcome_record(
                candidate,
                comparison=comparison,
                role=role,
                event=events.get(event_id or ""),
                root=resolved_root,
                workspace_id=resolved_workspace_id,
            )
        )

    candidate_outcome = next((item for item in outcome_records if item.get("role") == "candidate"), None)
    baseline_outcome = next((item for item in outcome_records if item.get("role") == "baseline"), None)
    rollback = _rollback_plan(
        explicit_plan=rollback_plan,
        baseline_backend=(
            _clean_text(baseline_backend)
            or _clean_text(_mapping_dict(baseline_outcome).get("backend_id"))
            or _clean_text(_mapping_dict(baseline_outcome).get("model_id"))
        ),
        candidate_backend=(
            _clean_text(candidate_backend)
            or _clean_text(_mapping_dict(candidate_outcome).get("backend_id"))
            or _clean_text(_mapping_dict(candidate_outcome).get("model_id"))
        ),
    )
    benchmark = _benchmark_gate(benchmark_report, strict=strict)
    lint = _lint_gate(root=resolved_root, workspace_id=resolved_workspace_id, strict=strict, enabled=run_evidence_lint)
    rules = _rule_evaluation(
        comparison=comparison,
        candidate_outcome=candidate_outcome,
        baseline_outcome=baseline_outcome,
        rollback=rollback,
        benchmark_gate=benchmark,
        lint_gate=lint,
        adoption_scope=adoption_scope,
    )
    recommendation = _recommendation(
        rule_evaluation=rules,
        comparison=comparison,
        candidate_outcome=candidate_outcome,
        adoption_scope=adoption_scope,
    )
    failures = [
        item
        for item in outcome_records
        if item.get("failure_or_regression")
        or _clean_text(_mapping_dict(item.get("positive_support")).get("support_class")) in {"contradictory", "unverified_agent_claim"}
    ]
    comparison_id = _clean_text(comparison.get("comparison_id"))
    candidate_label = (
        _clean_text(_mapping_dict(candidate_outcome).get("backend_id"))
        or _clean_text(_mapping_dict(candidate_outcome).get("model_id"))
        or _clean_text(candidate_backend)
        or "candidate"
    )
    baseline_label = (
        _clean_text(_mapping_dict(baseline_outcome).get("backend_id"))
        or _clean_text(_mapping_dict(baseline_outcome).get("model_id"))
        or _clean_text(baseline_backend)
        or "baseline"
    )
    dossier = {
        "schema_name": BACKEND_ADOPTION_DOSSIER_SCHEMA_NAME,
        "schema_version": BACKEND_ADOPTION_DOSSIER_SCHEMA_VERSION,
        "dossier_id": _stable_dossier_id(
            workspace_id=resolved_workspace_id,
            comparison_id=comparison_id,
            candidate_selector=candidate_backend,
            baseline_selector=baseline_backend,
            scope=adoption_scope,
        ),
        "workspace_id": resolved_workspace_id,
        "generated_at_utc": generated_at_utc or timestamp_utc(),
        "comparison_id": comparison_id,
        "decision_question": (
            f"Should `{candidate_label}` be adopted for `{adoption_scope['workflow_kind']}` "
            f"against baseline `{baseline_label}`, based on source-linked software-work outcomes?"
        ),
        "adoption_scope": adoption_scope,
        "baseline_metadata": _metadata_for_role(baseline_outcome),
        "candidate_metadata": _metadata_for_role(candidate_outcome),
        "source_linked_task_outcomes": outcome_records,
        "failures_and_regressions": failures,
        "human_verdicts_and_rationale": {
            "comparison_outcome": _clean_text(comparison.get("outcome")),
            "winner_event_id": _clean_text(comparison.get("winner_event_id")),
            "rationale": _clean_text(comparison.get("rationale")),
            "criteria": _string_list(comparison.get("criteria")),
            "human_rationale_present": _has_human_rationale(comparison),
        },
        "evidence_support_summary": _support_summary(outcome_records),
        "cost_latency_metadata": {
            "candidate": _cost_latency_metadata(candidate_outcome, source_candidate=selected_candidate),
            "baseline": _cost_latency_metadata(baseline_outcome, source_candidate=selected_baseline),
        },
        "rollback_plan": rollback,
        "benchmark_gate": benchmark,
        "evidence_lint_gate": lint,
        "rule_evaluation": rules,
        "recommendation": recommendation,
        "exit_gate": {
            "adoption_without_source_blocked": True,
            "adoption_without_human_rationale_blocked": True,
            "adoption_without_rollback_blocked": True,
            "negative_evidence_visible": True,
            "no_live_api_required": True,
        },
        "notes": [
            "This dossier is scoped to the named workflow and repository scope; it is not a global model ranking.",
            "Provider hubs and live API calls are not used; the dossier reads local comparison and evidence files only.",
        ],
    }
    return dossier


def _comparison_by_id(
    comparison_id: str,
    *,
    root: Path,
    workspace_id: str,
) -> dict[str, Any]:
    comparisons = read_evaluation_comparisons(
        evaluation_comparison_log_path(workspace_id=workspace_id, root=root)
    )
    for comparison in comparisons:
        if _clean_text(comparison.get("comparison_id")) == comparison_id:
            return comparison
    raise ValueError(f"Evaluation comparison `{comparison_id}` was not found.")


def build_backend_adoption_dossier_from_comparison_id(
    comparison_id: str,
    *,
    candidate_backend: str | None = None,
    baseline_backend: str | None = None,
    workflow_kind: str = "backend_compare",
    repo_scope: str = "current_repo",
    risk_scope: str = "experimental",
    rollback_plan: str | None = None,
    benchmark_report: Mapping[str, Any] | None = None,
    benchmark_report_path: Path | None = None,
    strict: bool = False,
    run_evidence_lint: bool = False,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    comparison = _comparison_by_id(comparison_id, root=resolved_root, workspace_id=workspace_id)
    events = software_work_events_by_id(root=resolved_root, workspace_id=workspace_id)
    resolved_benchmark_report = benchmark_report or _read_benchmark_report(benchmark_report_path)
    return build_backend_adoption_dossier(
        comparison,
        events_by_id=events,
        candidate_backend=candidate_backend,
        baseline_backend=baseline_backend,
        workflow_kind=workflow_kind,
        repo_scope=repo_scope,
        risk_scope=risk_scope,
        rollback_plan=rollback_plan,
        benchmark_report=resolved_benchmark_report,
        strict=strict,
        run_evidence_lint=run_evidence_lint,
        workspace_id=workspace_id,
        root=resolved_root,
    )


def _failure_memory_root(*, workspace_id: str, root: Path) -> Path:
    return root / "artifacts" / "failure_memory" / workspace_id


def _read_json_mapping(path: Path, *, strict: bool = False) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        if strict:
            raise ValueError(f"Review metadata `{path}` must be a readable JSON object.") from exc
        return None
    if not isinstance(payload, Mapping):
        if strict:
            raise ValueError(f"Review metadata `{path}` must contain a JSON object.")
        return None
    return copy.deepcopy(dict(payload))


def _find_review_metadata(
    review_run_id: str,
    *,
    root: Path,
    workspace_id: str,
) -> tuple[dict[str, Any] | None, Path | None]:
    cleaned = _clean_text(review_run_id)
    if cleaned is None:
        return None, None
    candidate_path = Path(cleaned).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = root / candidate_path
    if candidate_path.exists():
        payload = _read_json_mapping(candidate_path, strict=True)
        return payload, candidate_path

    reports_root = _failure_memory_root(workspace_id=workspace_id, root=root) / "reports"
    paths = [reports_root / "latest.json", *sorted((reports_root / "runs").glob("*.json"))]
    for path in paths:
        payload = _read_json_mapping(path)
        if payload is None:
            continue
        if cleaned in {
            _clean_text(payload.get("event_id")),
            _clean_text(payload.get("review_run_id")),
            path.stem,
            path.name,
        }:
            return payload, path
    return None, None


def _comparison_includes_event(comparison: Mapping[str, Any], event_id: str) -> bool:
    return event_id in {
        _candidate_event_id(candidate)
        for candidate in _comparison_candidates(comparison)
    }


def _latest_comparison_for_review(
    *,
    review_metadata: Mapping[str, Any] | None,
    root: Path,
    workspace_id: str,
    candidate_backend: str | None,
    baseline_backend: str | None,
) -> dict[str, Any] | None:
    comparisons = read_evaluation_comparisons(
        evaluation_comparison_log_path(workspace_id=workspace_id, root=root)
    )
    if not comparisons:
        return None
    review_event_id = _clean_text(_mapping_dict(review_metadata).get("event_id"))
    sorted_comparisons = sorted(
        comparisons,
        key=lambda item: (_clean_text(item.get("recorded_at_utc")) or "", _clean_text(item.get("comparison_id")) or ""),
        reverse=True,
    )
    scoped_comparisons = sorted_comparisons
    if review_event_id is not None:
        scoped_comparisons = [
            comparison
            for comparison in sorted_comparisons
            if _comparison_includes_event(comparison, review_event_id)
        ]
        if not scoped_comparisons:
            return None
    if candidate_backend or baseline_backend:
        for comparison in scoped_comparisons:
            candidates = _comparison_candidates(comparison)
            if candidate_backend and not any(_candidate_matches(candidate, candidate_backend) for candidate in candidates):
                continue
            if baseline_backend and not any(_candidate_matches(candidate, baseline_backend) for candidate in candidates):
                continue
            return comparison
        return None
    if review_event_id is not None:
        return scoped_comparisons[0]
    return None


def build_backend_adoption_dossier_from_review(
    review_run_id: str,
    *,
    candidate_backend: str | None = None,
    baseline_backend: str | None = None,
    workflow_kind: str = "review_git",
    repo_scope: str = "current_repo",
    risk_scope: str = "experimental",
    rollback_plan: str | None = None,
    benchmark_report: Mapping[str, Any] | None = None,
    benchmark_report_path: Path | None = None,
    strict: bool = False,
    run_evidence_lint: bool = False,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    review_metadata, metadata_path = _find_review_metadata(
        review_run_id,
        root=resolved_root,
        workspace_id=workspace_id,
    )
    comparison = (
        _latest_comparison_for_review(
            review_metadata=review_metadata,
            root=resolved_root,
            workspace_id=workspace_id,
            candidate_backend=candidate_backend,
            baseline_backend=baseline_backend,
        )
        if review_metadata is not None
        else None
    )
    resolved_benchmark_report = benchmark_report or _read_benchmark_report(benchmark_report_path)
    if comparison is None:
        comparison = {
            "schema_name": EVALUATION_COMPARISON_SCHEMA_NAME,
            "schema_version": 1,
            "comparison_id": f"review:{_clean_text(review_run_id) or 'unknown'}",
            "workspace_id": workspace_id,
            "recorded_at_utc": _clean_text(_mapping_dict(review_metadata).get("generated_at_utc")),
            "origin": "review_metadata",
            "task_label": "review backend adoption",
            "outcome": "needs_follow_up",
            "winner_event_id": None,
            "candidate_count": 0,
            "candidates": [],
            "criteria": [],
            "rationale": None,
            "tags": ["review", "backend_adoption"],
        }
    dossier = build_backend_adoption_dossier(
        comparison,
        events_by_id=software_work_events_by_id(root=resolved_root, workspace_id=workspace_id),
        candidate_backend=candidate_backend,
        baseline_backend=baseline_backend,
        workflow_kind=workflow_kind,
        repo_scope=repo_scope,
        risk_scope=risk_scope,
        rollback_plan=rollback_plan,
        benchmark_report=resolved_benchmark_report,
        strict=strict,
        run_evidence_lint=run_evidence_lint,
        workspace_id=workspace_id,
        root=resolved_root,
    )
    dossier["review_source"] = {
        "review_run_id": _clean_text(review_run_id),
        "metadata_path": str(metadata_path) if metadata_path is not None else None,
        "metadata_found": review_metadata is not None,
    }
    return dossier


def _markdown_cell(value: Any) -> str:
    text = _clean_text(value)
    if text is None:
        return "n/a"
    return text.replace("|", "\\|").replace("\n", " ")


def _refs_text(refs: Iterable[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for ref in refs:
        artifact_id = _clean_text(ref.get("artifact_id"))
        path = _clean_text(ref.get("repo_relative_path")) or _clean_text(ref.get("path"))
        source_state = _clean_text(ref.get("source_state"))
        if artifact_id and path:
            parts.append(f"{artifact_id} ({path}; {source_state or 'source'})")
        elif artifact_id:
            parts.append(artifact_id)
        elif path:
            parts.append(path)
    return "; ".join(parts) or "none"


def format_backend_adoption_dossier_markdown(dossier: Mapping[str, Any]) -> str:
    recommendation = _mapping_dict(dossier.get("recommendation"))
    scope = _mapping_dict(dossier.get("adoption_scope"))
    rule_evaluation = _mapping_dict(dossier.get("rule_evaluation"))
    rollback = _mapping_dict(dossier.get("rollback_plan"))
    human = _mapping_dict(dossier.get("human_verdicts_and_rationale"))
    benchmark = _mapping_dict(dossier.get("benchmark_gate"))
    lines = [
        "# Backend / Model Adoption Dossier",
        "",
        "## 1. Decision Question",
        "",
        _clean_text(dossier.get("decision_question")) or "n/a",
        "",
        "## 2. Scope of Adoption",
        "",
        f"- Workflow kind: `{_markdown_cell(scope.get('workflow_kind'))}`",
        f"- Repo scope: `{_markdown_cell(scope.get('repo_scope'))}`",
        f"- Risk scope: `{_markdown_cell(scope.get('risk_scope'))}`",
        "",
        "## 3. Baseline and Candidate Metadata",
        "",
        "| Role | Backend | Model | Compatibility |",
        "| --- | --- | --- | --- |",
    ]
    for role, metadata_key in (("candidate", "candidate_metadata"), ("baseline", "baseline_metadata")):
        metadata = _mapping_dict(dossier.get(metadata_key))
        lines.append(
            f"| {role} | `{_markdown_cell(metadata.get('backend_id'))}` | "
            f"`{_markdown_cell(metadata.get('model_id'))}` | `{_markdown_cell(metadata.get('compatibility_status'))}` |"
        )

    lines.extend(
        [
            "",
            "## 4. Source-Linked Task Outcomes",
            "",
            "| Role | Comparison | Backend | Event | Positive Support | Risk Support | Source Artifact Refs |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for outcome in dossier.get("source_linked_task_outcomes") or []:
        if not isinstance(outcome, Mapping):
            continue
        positive = _mapping_dict(outcome.get("positive_support"))
        risk = _mapping_dict(outcome.get("risk_support"))
        backend = _clean_text(outcome.get("backend_id")) or _clean_text(outcome.get("model_id")) or "n/a"
        lines.append(
            "| "
            f"{_markdown_cell(outcome.get('role'))} | "
            f"{_markdown_cell(outcome.get('comparison_role'))} | "
            f"`{_markdown_cell(backend)}` | "
            f"`{_markdown_cell(outcome.get('event_id'))}` | "
            f"{_markdown_cell(positive.get('support_class'))} | "
            f"{_markdown_cell(risk.get('support_class'))} | "
            f"{_markdown_cell(_refs_text(outcome.get('source_artifact_refs') or []))} |"
        )

    lines.extend(["", "## 5. Failures and Regressions", ""])
    failures = [item for item in dossier.get("failures_and_regressions") or [] if isinstance(item, Mapping)]
    if failures:
        for failure in failures:
            event_outcome = _mapping_dict(failure.get("event_outcome"))
            backend = _clean_text(failure.get("backend_id")) or _clean_text(failure.get("model_id")) or "n/a"
            lines.append(
                "- "
                f"`{backend}` event=`{_markdown_cell(failure.get('event_id'))}` "
                f"status=`{_markdown_cell(event_outcome.get('status'))}` "
                f"quality=`{_markdown_cell(event_outcome.get('quality_status'))}` "
                f"refs={_refs_text(failure.get('source_artifact_refs') or [])}"
            )
    else:
        lines.append("- No source-linked negative evidence found for the selected candidates.")

    lines.extend(
        [
            "",
            "## 6. Human Verdicts and Rationale",
            "",
            f"- Comparison outcome: `{_markdown_cell(human.get('comparison_outcome'))}`",
            f"- Winner event: `{_markdown_cell(human.get('winner_event_id'))}`",
            f"- Rationale present: `{str(bool(human.get('human_rationale_present'))).lower()}`",
            f"- Rationale: {_markdown_cell(human.get('rationale'))}",
            "",
            "## 7. Evidence Support Summary",
            "",
            "```json",
            json.dumps(dossier.get("evidence_support_summary") or {}, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## 8. Cost / Latency Metadata",
            "",
            "```json",
            json.dumps(dossier.get("cost_latency_metadata") or {}, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## 9. Rollback Plan",
            "",
            f"- Present: `{str(bool(rollback.get('present'))).lower()}`",
            f"- Source: `{_markdown_cell(rollback.get('source'))}`",
            f"- Plan: {_markdown_cell(rollback.get('plan'))}",
            "",
            "## 10. Recommendation",
            "",
            f"- Recommendation: `{_markdown_cell(recommendation.get('value'))}`",
            f"- Rationale: {_markdown_cell(recommendation.get('rationale'))}",
            f"- Blockers: {', '.join('`' + item + '`' for item in _string_list(rule_evaluation.get('blockers'))) or '`none`'}",
            f"- Warnings: {', '.join('`' + item + '`' for item in _string_list(rule_evaluation.get('warnings'))) or '`none`'}",
            "- Benchmark gate: "
            f"passed=`{str(bool(benchmark.get('passed'))).lower()}` "
            f"available=`{str(bool(benchmark.get('available'))).lower()}`",
            "",
            "## Exit Gate",
            "",
            "```json",
            json.dumps(dossier.get("exit_gate") or {}, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def validate_backend_adoption_dossier(dossier: Mapping[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    required_top = {
        "schema_name",
        "schema_version",
        "dossier_id",
        "workspace_id",
        "generated_at_utc",
        "decision_question",
        "adoption_scope",
        "baseline_metadata",
        "candidate_metadata",
        "source_linked_task_outcomes",
        "failures_and_regressions",
        "human_verdicts_and_rationale",
        "evidence_support_summary",
        "cost_latency_metadata",
        "rollback_plan",
        "benchmark_gate",
        "evidence_lint_gate",
        "rule_evaluation",
        "recommendation",
        "exit_gate",
        "notes",
    }
    for key in sorted(required_top):
        if key not in dossier:
            issues.append({"path": f"$.{key}", "message": "Missing required dossier field."})
    if dossier.get("schema_name") != BACKEND_ADOPTION_DOSSIER_SCHEMA_NAME:
        issues.append({"path": "$.schema_name", "message": "Unexpected schema name."})
    if dossier.get("schema_version") != BACKEND_ADOPTION_DOSSIER_SCHEMA_VERSION:
        issues.append({"path": "$.schema_version", "message": "Unsupported schema version."})
    scope = _mapping_dict(dossier.get("adoption_scope"))
    if _clean_text(scope.get("workflow_kind")) not in WORKFLOW_KINDS:
        issues.append({"path": "$.adoption_scope.workflow_kind", "message": "Unsupported workflow_kind."})
    if _clean_text(scope.get("repo_scope")) not in REPO_SCOPES:
        issues.append({"path": "$.adoption_scope.repo_scope", "message": "Unsupported repo_scope."})
    if _clean_text(scope.get("risk_scope")) not in RISK_SCOPES:
        issues.append({"path": "$.adoption_scope.risk_scope", "message": "Unsupported risk_scope."})
    recommendation = _mapping_dict(dossier.get("recommendation"))
    if _clean_text(recommendation.get("value")) not in RECOMMENDATIONS:
        issues.append({"path": "$.recommendation.value", "message": "Unsupported recommendation value."})
    rule_evaluation = _mapping_dict(dossier.get("rule_evaluation"))
    if not isinstance(rule_evaluation.get("blockers"), list):
        issues.append({"path": "$.rule_evaluation.blockers", "message": "Expected an array."})
    if not isinstance(rule_evaluation.get("warnings"), list):
        issues.append({"path": "$.rule_evaluation.warnings", "message": "Expected an array."})
    if not isinstance(rule_evaluation.get("rules"), Mapping):
        issues.append({"path": "$.rule_evaluation.rules", "message": "Expected an object."})
    if not isinstance(dossier.get("source_linked_task_outcomes"), list):
        issues.append({"path": "$.source_linked_task_outcomes", "message": "Expected an array."})
    exit_gate = _mapping_dict(dossier.get("exit_gate"))
    true_exit_gates = {
        "adoption_without_source_blocked",
        "adoption_without_human_rationale_blocked",
        "adoption_without_rollback_blocked",
        "negative_evidence_visible",
        "no_live_api_required",
    }
    for key in (
        "adoption_without_source_blocked",
        "adoption_without_human_rationale_blocked",
        "adoption_without_rollback_blocked",
        "negative_evidence_visible",
        "no_live_api_required",
    ):
        if not isinstance(exit_gate.get(key), bool):
            issues.append({"path": f"$.exit_gate.{key}", "message": "Expected a boolean."})
        elif key in true_exit_gates and exit_gate.get(key) is not True:
            issues.append({"path": f"$.exit_gate.{key}", "message": "Expected true."})
    return issues
