#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from artifact_vault import artifact_ref_object_verified, resolve_vault_object_path
from evidence_support import artifact_refs_from_event, build_evidence_support_result
from evaluation_loop import (
    evaluation_comparison_log_path,
    evaluation_signal_log_path,
    learning_dataset_preview_latest_path,
    read_evaluation_comparisons,
    read_evaluation_signals,
)
from gemma_runtime import repo_root, timestamp_utc
from memory_index import default_memory_index_path
from software_work_events import (
    build_event_contract_report,
    iter_agent_session_intake_events,
    iter_agent_lane_events,
    iter_capability_matrix_events,
    iter_workspace_events,
    read_event_log,
    workspace_event_log_path,
)
from workspace_state import DEFAULT_WORKSPACE_ID


EVIDENCE_GRAPH_SCHEMA_NAME = "software-satellite-derived-evidence-graph"
EVIDENCE_GRAPH_SCHEMA_VERSION = 1
DETERMINISTIC_SUPPORT_CHECKED_AT_UTC = "1970-01-01T00:00:00+00:00"

NODE_KINDS = {
    "artifact",
    "event",
    "signal",
    "recall",
    "review",
    "comparison",
    "learning_candidate",
    "backend_candidate",
}
RELATION_KINDS = {
    "derives_from",
    "recalls",
    "repairs",
    "contradicts",
    "evaluates",
    "selects",
    "rejects",
    "supersedes",
    "excludes",
    "uses_artifact",
}
EDGE_STRENGTHS = {"strong", "weak", "manual_pin", "diagnostic_only"}
CAUSAL_VALIDITIES = {"valid", "invalid", "unknown"}
EDGE_CREATORS = {"core", "human", "pack", "benchmark_fixture"}
SUPPORT_CLASSES = {
    "source_linked_prior",
    "negative_prior",
    "current_review_subject",
    "future_evidence",
    "missing_source",
    "modified_source",
    "weak_match",
    "contradictory",
    "manual_pin_diagnostic",
    "unverified_agent_claim",
    "unknown",
}
POLARITIES = {"positive", "negative", "risk", "neutral", "unknown"}
QUALITY_STATUSES = {
    "verified",
    "human_accepted",
    "human_rejected",
    "failed",
    "unresolved",
    "blocked",
    "unknown",
}

POSITIVE_SIGNAL_KINDS = {"acceptance", "review_resolved", "test_pass", "export_policy_confirmed"}
NEGATIVE_SIGNAL_KINDS = {"rejection", "review_unresolved", "test_fail"}
HARD_BLOCKING_SUPPORT_CLASSES = {
    "missing_source",
    "modified_source",
    "current_review_subject",
    "future_evidence",
    "unverified_agent_claim",
    "contradictory",
}
ISSUE_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


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


def _dedupe(values: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_text(value)
        if cleaned is None or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def _stable_digest(value: Any, *, length: int = 16) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _stable_node_id(kind: str, source_id: str | None) -> str:
    return f"node_{kind}_{_stable_digest([kind, source_id or 'unknown'])}"


def _stable_edge_id(
    relation_kind: str,
    from_node_id: str,
    to_node_id: str,
    source_id: str | None = None,
) -> str:
    return f"edge_{relation_kind}_{_stable_digest([relation_kind, from_node_id, to_node_id, source_id])}"


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


def _timestamp_sort_key(value: Any) -> tuple[bool, float, str]:
    text = _clean_text(value)
    if text is None:
        return (False, 0.0, "")
    parsed = _coerce_utc_datetime(text)
    if parsed is None:
        return (False, 0.0, text)
    return (True, parsed.timestamp(), text)


def _event_id(event: Mapping[str, Any]) -> str | None:
    return _clean_text(event.get("event_id"))


def _event_node_id(event_id: str) -> str:
    return _stable_node_id("event", event_id)


def _signal_source_event_id(signal: Mapping[str, Any]) -> str | None:
    source = _mapping_dict(signal.get("source"))
    return _clean_text(source.get("source_event_id")) or _clean_text(signal.get("source_event_id"))


def _signal_target_event_id(signal: Mapping[str, Any]) -> str | None:
    relation = _mapping_dict(signal.get("relation"))
    return _clean_text(relation.get("target_event_id")) or _clean_text(signal.get("target_event_id"))


def _signal_relation_kind(signal: Mapping[str, Any]) -> str | None:
    relation = _mapping_dict(signal.get("relation"))
    return _clean_text(relation.get("relation_kind")) or _clean_text(signal.get("relation_kind"))


def _normalize_polarity(value: Any) -> str:
    text = (_clean_text(value) or "").lower().replace("-", "_")
    if text in {"positive", "negative", "risk"}:
        return text
    if text in {"diagnostic", "none", "neutral"}:
        return "neutral"
    return "unknown"


def _quality_from_status(status: Any, *, fallback: str = "unknown") -> str:
    text = (_clean_text(status) or "").lower().replace("-", "_")
    if text in {"verified", "pass", "passed", "quality_pass", "ok", "resolved"}:
        return "verified"
    if text in {"accepted", "accept", "human_accepted"}:
        return "human_accepted"
    if text in {"rejected", "reject", "human_rejected"}:
        return "human_rejected"
    if text in {"fail", "failed", "quality_fail", "error"}:
        return "failed"
    if text in {"blocked"}:
        return "blocked"
    if text in {"needs_review", "needs_more_evidence", "pending", "unresolved"}:
        return "unresolved"
    return fallback


def _event_quality_status(event: Mapping[str, Any], support: Mapping[str, Any]) -> str:
    outcome = _mapping_dict(event.get("outcome"))
    content = _mapping_dict(event.get("content"))
    options = _mapping_dict(content.get("options"))
    for value in (
        outcome.get("quality_status"),
        options.get("quality_status"),
        outcome.get("status"),
        event.get("status"),
    ):
        quality = _quality_from_status(value)
        if quality != "unknown":
            return quality
    if support.get("can_support_decision"):
        return "verified"
    return "unknown"


def _signal_quality_status(signal: Mapping[str, Any]) -> str:
    signal_kind = _clean_text(signal.get("signal_kind"))
    if signal_kind in {"acceptance", "review_resolved", "export_policy_confirmed"}:
        return "human_accepted"
    if signal_kind in {"rejection", "review_unresolved"}:
        return "human_rejected"
    if signal_kind == "test_pass":
        return "verified"
    if signal_kind == "test_fail":
        return "failed"
    return "unknown"


def _signal_polarity(signal: Mapping[str, Any]) -> str:
    signal_kind = _clean_text(signal.get("signal_kind"))
    if signal_kind in POSITIVE_SIGNAL_KINDS:
        return "positive"
    if signal_kind in NEGATIVE_SIGNAL_KINDS:
        return "negative" if signal_kind != "review_unresolved" else "risk"
    return _normalize_polarity(signal.get("polarity"))


def _source_created_by(origin: Any, tags: Iterable[str] | None = None) -> str:
    origin_text = (_clean_text(origin) or "").lower()
    tag_set = {tag.lower() for tag in tags or []}
    if "benchmark" in origin_text or "benchmark_fixture" in tag_set:
        return "benchmark_fixture"
    if origin_text in {"satlab_cli", "manual", "human"} or "human-verdict" in tag_set:
        return "human"
    if "pack" in origin_text:
        return "pack"
    return "core"


def _normalize_repo_path(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None or text == "/dev/null":
        return None
    normalized = text.replace("\\", "/").strip("\"'")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    for prefix in ("a/", "b/"):
        if normalized.startswith(prefix):
            normalized = normalized[2:]
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized.strip("/") or None


def _path_is_inside_root(path: Path, *, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _read_ref_text(ref: Mapping[str, Any], *, root: Path, limit: int = 400_000) -> str | None:
    object_path = resolve_vault_object_path(ref, root=root)
    candidates: list[Path] = []
    if object_path is not None:
        candidates.append(object_path)
    original = _clean_text(ref.get("original_path"))
    if original is not None:
        path = Path(original).expanduser()
        if not path.is_absolute():
            path = root / path
        candidates.append(path)
    for path in candidates:
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            resolved = path
        if not _path_is_inside_root(resolved, root=root):
            continue
        try:
            if not resolved.is_file():
                continue
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        return text[:limit]
    return None


def _parse_patch_identity(text: str) -> tuple[list[str], list[str]]:
    paths: list[str] = []
    hunks: list[str] = []
    current_file: str | None = None
    seen_paths: set[str] = set()
    seen_hunks: set[str] = set()
    for line in text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                candidate = _normalize_repo_path(parts[3])
                if candidate is not None:
                    current_file = candidate
                    if candidate not in seen_paths:
                        seen_paths.add(candidate)
                        paths.append(candidate)
            continue
        if line.startswith("+++ ") or line.startswith("--- "):
            candidate = _normalize_repo_path(line[4:].strip())
            if candidate is not None:
                current_file = candidate
                if candidate not in seen_paths:
                    seen_paths.add(candidate)
                    paths.append(candidate)
            continue
        if line.startswith("@@"):
            normalized = re.sub(r"\s+", " ", line.strip())
            hunk = f"{current_file or 'unknown'} {normalized}"
            if hunk not in seen_hunks:
                seen_hunks.add(hunk)
                hunks.append(hunk)
    return paths, hunks


def _issue_keys_from_event(event: Mapping[str, Any]) -> list[str]:
    content = _mapping_dict(event.get("content"))
    options = _mapping_dict(content.get("options"))
    haystack = " ".join(
        str(part)
        for part in (
            event.get("event_id"),
            content.get("prompt"),
            content.get("resolved_user_prompt"),
            content.get("output_text"),
            options.get("issue_key"),
            options.get("task_key"),
            " ".join(_string_list(event.get("tags"))),
            " ".join(_string_list(content.get("notes"))),
        )
        if part is not None
    )
    return _dedupe(match.group(0) for match in ISSUE_KEY_RE.finditer(haystack))


def target_identity_for_event(event: Mapping[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    refs = artifact_refs_from_event(event, root=resolved_root)
    content = _mapping_dict(event.get("content"))
    options = _mapping_dict(content.get("options"))
    changed_files: list[str] = []
    hunk_headers: list[str] = []
    source_paths: list[str] = []

    for value in (
        options.get("file_hints"),
        options.get("changed_files"),
        _mapping_dict(options.get("input_summary")).get("changed_files"),
    ):
        for path in _string_list(value):
            normalized = _normalize_repo_path(path)
            if normalized is not None:
                changed_files.append(normalized)

    for ref in refs:
        for key in ("repo_relative_path", "original_path"):
            normalized = _normalize_repo_path(ref.get(key))
            if normalized is not None:
                source_paths.append(normalized)
        text = _read_ref_text(ref, root=resolved_root)
        if text is None:
            continue
        parsed_paths, parsed_hunks = _parse_patch_identity(text)
        changed_files.extend(parsed_paths)
        hunk_headers.extend(parsed_hunks)

    changed_files = sorted(_dedupe(changed_files))
    hunk_headers = sorted(_dedupe(hunk_headers))
    issue_keys = sorted(_issue_keys_from_event(event))
    if not changed_files and not hunk_headers and not issue_keys:
        return {
            "target_fingerprint": None,
            "changed_files": [],
            "hunk_headers": [],
            "issue_keys": issue_keys,
            "source_paths": sorted(_dedupe(source_paths)),
        }

    fingerprint_source = {
        "changed_files": changed_files,
        "hunk_headers": hunk_headers,
        "issue_keys": issue_keys,
    }
    return {
        "target_fingerprint": f"target_{_stable_digest(fingerprint_source, length=20)}",
        "changed_files": changed_files,
        "hunk_headers": hunk_headers,
        "issue_keys": issue_keys,
        "source_paths": sorted(_dedupe(source_paths)),
    }


def _artifact_quality_and_support(ref: Mapping[str, Any], *, root: Path) -> tuple[str, str, dict[str, Any]]:
    verified, reason = artifact_ref_object_verified(ref, root=root)
    source_state = _clean_text(ref.get("source_state"))
    capture_state = _clean_text(ref.get("capture_state"))
    support_class = "source_linked_prior" if verified else "missing_source"
    if reason == "vault_checksum_mismatch":
        support_class = "modified_source"
    elif source_state in {"missing", "outside_workspace", "symlink_refused", "binary_refused"}:
        support_class = "missing_source"
    quality = "verified" if verified else "blocked" if support_class in {"missing_source", "modified_source"} else "unknown"
    excerpt = _mapping_dict(ref.get("report_excerpt"))
    excerpt_text = _clean_text(excerpt.get("text")) or ""
    fully_redacted = bool(excerpt_text) and _clean_text(excerpt_text.replace("[REDACTED]", "")) is None
    return quality, support_class, {
        "object_verified": verified,
        "verification_reason": reason,
        "source_state": source_state,
        "capture_state": capture_state,
        "redaction": copy.deepcopy(_mapping_dict(ref.get("redaction"))),
        "report_excerpt_fully_redacted": fully_redacted,
        "repo_relative_path": _clean_text(ref.get("repo_relative_path")),
        "original_path": _clean_text(ref.get("original_path")),
        "sha256": _clean_text(ref.get("sha256")),
    }


def _node(
    *,
    node_kind: str,
    source_id: str,
    support_class: str = "unknown",
    can_support_decision: bool = False,
    polarity: str = "unknown",
    quality_status: str = "unknown",
    target_fingerprint: str | None = None,
    created_at_utc: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "node_id": _stable_node_id(node_kind, source_id),
        "node_kind": node_kind,
        "source_id": source_id,
        "support_class": support_class if support_class in SUPPORT_CLASSES else "unknown",
        "can_support_decision": bool(can_support_decision),
        "polarity": polarity if polarity in POLARITIES else "unknown",
        "quality_status": quality_status if quality_status in QUALITY_STATUSES else "unknown",
        "target_fingerprint": target_fingerprint,
        "created_at_utc": created_at_utc,
        "metadata": copy.deepcopy(dict(metadata or {})),
    }


def _edge(
    *,
    from_node_id: str,
    to_node_id: str,
    relation_kind: str,
    source_id: str | None = None,
    strength: str = "strong",
    causal_validity: str = "valid",
    created_by: str = "core",
    explanation: str = "",
    metadata: Mapping[str, Any] | None = None,
    support_class: str | None = None,
    can_support_decision: bool | None = None,
    polarity: str | None = None,
) -> dict[str, Any]:
    edge_id = _stable_edge_id(relation_kind, from_node_id, to_node_id, source_id)
    payload = {
        "edge_id": edge_id,
        "from_node_id": from_node_id,
        "to_node_id": to_node_id,
        "relation_kind": relation_kind if relation_kind in RELATION_KINDS else "derives_from",
        "strength": strength if strength in EDGE_STRENGTHS else "weak",
        "causal_validity": causal_validity if causal_validity in CAUSAL_VALIDITIES else "unknown",
        "created_by": created_by if created_by in EDGE_CREATORS else "core",
        "explanation": explanation,
        "metadata": copy.deepcopy(dict(metadata or {})),
    }
    if support_class is not None:
        payload["support_class"] = support_class if support_class in SUPPORT_CLASSES else "unknown"
    if can_support_decision is not None:
        payload["can_support_decision"] = bool(can_support_decision)
    if polarity is not None:
        payload["polarity"] = polarity if polarity in POLARITIES else "unknown"
    return payload


def _add_node(nodes: dict[str, dict[str, Any]], node: Mapping[str, Any]) -> None:
    nodes[str(node["node_id"])] = copy.deepcopy(dict(node))


def _add_edge(edges: dict[str, dict[str, Any]], edge: Mapping[str, Any]) -> None:
    edges[str(edge["edge_id"])] = copy.deepcopy(dict(edge))


def _read_json_mapping(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _dedup_paths(paths: Iterable[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            resolved = path.absolute()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _load_recall_artifacts(*, root: Path, workspace_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    recall_root = root / "artifacts" / "failure_memory" / workspace_id / "recall"
    paths = _dedup_paths([recall_root / "latest.json", *sorted((recall_root / "runs").glob("*.json"))])
    recalls: list[dict[str, Any]] = []
    source_paths: list[str] = []
    for path in paths:
        payload = _read_json_mapping(path)
        if payload is None or payload.get("schema_name") != "software-satellite-failure-memory-recall":
            continue
        payload = copy.deepcopy(payload)
        payload.setdefault("_source_path", str(path))
        recalls.append(payload)
        source_paths.append(str(path))
    return recalls, source_paths


def _load_learning_previews(*, root: Path, workspace_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    latest_path = learning_dataset_preview_latest_path(workspace_id=workspace_id, root=root)
    paths: list[Path]
    if latest_path.is_file():
        paths = [latest_path]
    else:
        paths = sorted((root / "artifacts" / "evaluation" / workspace_id / "learning" / "runs").glob("*-learning-preview.json"))
    previews: list[dict[str, Any]] = []
    source_paths: list[str] = []
    for path in _dedup_paths(paths):
        payload = _read_json_mapping(path)
        if payload is None or payload.get("schema_name") != "software-satellite-learning-dataset-preview":
            continue
        payload = copy.deepcopy(payload)
        payload.setdefault("_source_path", str(path))
        previews.append(payload)
        source_paths.append(str(path))
    return previews, source_paths


def _load_pack_audits(*, root: Path, workspace_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    audit_root = root / "artifacts" / "satellite_evidence_packs" / workspace_id / "audits"
    paths = _dedup_paths([audit_root / "latest.json", *sorted((audit_root / "runs").glob("*.json"))])
    audits: list[dict[str, Any]] = []
    source_paths: list[str] = []
    for path in paths:
        payload = _read_json_mapping(path)
        if payload is None or payload.get("schema_name") != "software-satellite-evidence-pack-audit":
            continue
        payload = copy.deepcopy(payload)
        payload.setdefault("_source_path", str(path))
        audits.append(payload)
        source_paths.append(str(path))
    return audits, source_paths


def _load_review_reports(*, root: Path, workspace_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    report_root = root / "artifacts" / "failure_memory" / workspace_id / "reports"
    paths = _dedup_paths([report_root / "latest.json", *sorted((report_root / "runs").glob("*-review-risk-report.json"))])
    reports: list[dict[str, Any]] = []
    source_paths: list[str] = []
    for path in paths:
        payload = _read_json_mapping(path)
        if payload is None or payload.get("schema_name") != "software-satellite-review-risk-report":
            continue
        payload = copy.deepcopy(payload)
        payload.setdefault("_source_path", str(path))
        reports.append(payload)
        source_paths.append(str(path))
    return reports, source_paths


def _load_events_read_only(*, root: Path, workspace_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_log_path = workspace_event_log_path(workspace_id=workspace_id, root=root)
    if event_log_path.is_file():
        event_log = read_event_log(event_log_path)
        events = [
            dict(event)
            for event in event_log.get("events") or []
            if isinstance(event, Mapping)
        ]
        return events, {
            "event_source": "existing_event_log",
            "event_log_path": str(event_log_path),
            "event_log_event_count": len(events),
            "event_log_generated_at_utc": _clean_text(event_log.get("generated_at_utc")),
        }

    capability_matrix_errors: list[dict[str, str]] = []
    agent_lane_errors: list[dict[str, str]] = []
    agent_session_intake_errors: list[dict[str, str]] = []
    workspace_events = iter_workspace_events(root=root, workspace_id=workspace_id)
    capability_events = iter_capability_matrix_events(
        root=root,
        workspace_id=workspace_id,
        errors=capability_matrix_errors,
    )
    agent_events = iter_agent_lane_events(
        root=root,
        workspace_id=workspace_id,
        errors=agent_lane_errors,
    )
    agent_session_intake_events = iter_agent_session_intake_events(
        root=root,
        workspace_id=workspace_id,
        errors=agent_session_intake_errors,
    )
    events = sorted(
        [*workspace_events, *capability_events, *agent_events, *agent_session_intake_events],
        key=lambda item: (
            str(item.get("recorded_at_utc") or ""),
            str(item.get("event_id") or ""),
        ),
    )
    return events, {
        "event_source": "source_manifests_read_only",
        "event_log_path": str(event_log_path),
        "event_log_event_count": 0,
        "workspace_event_count": len(workspace_events),
        "capability_event_count": len(capability_events),
        "capability_matrix_error_count": len(capability_matrix_errors),
        "capability_matrix_errors": capability_matrix_errors,
        "agent_lane_event_count": len(agent_events),
        "agent_lane_error_count": len(agent_lane_errors),
        "agent_lane_errors": agent_lane_errors,
        "agent_session_intake_event_count": len(agent_session_intake_events),
        "agent_session_intake_error_count": len(agent_session_intake_errors),
        "agent_session_intake_errors": agent_session_intake_errors,
    }


def _load_default_sources(*, root: Path, workspace_id: str) -> dict[str, Any]:
    events, event_source_summary = _load_events_read_only(root=root, workspace_id=workspace_id)
    event_log_path = Path(str(event_source_summary["event_log_path"]))
    index_path = default_memory_index_path(workspace_id=workspace_id, root=root)
    event_contract = build_event_contract_report(events, root=root)
    signal_log_path = evaluation_signal_log_path(workspace_id=workspace_id, root=root)
    comparison_log_path = evaluation_comparison_log_path(workspace_id=workspace_id, root=root)
    recalls, recall_paths = _load_recall_artifacts(root=root, workspace_id=workspace_id)
    learning_previews, learning_paths = _load_learning_previews(root=root, workspace_id=workspace_id)
    pack_audits, pack_paths = _load_pack_audits(root=root, workspace_id=workspace_id)
    review_reports, review_paths = _load_review_reports(root=root, workspace_id=workspace_id)
    source_paths = {
        "event_log_path": str(event_log_path),
        "index_path": str(index_path),
        "signal_log_path": str(signal_log_path),
        "comparison_log_path": str(comparison_log_path),
        "recall_paths": recall_paths,
        "learning_preview_paths": learning_paths,
        "pack_audit_paths": pack_paths,
        "review_report_paths": review_paths,
    }
    index_summary = {
        "workspace_id": workspace_id,
        "event_count": len(events),
        "indexed_count": None,
        "event_log_path": str(event_log_path),
        "index_path": str(index_path),
        "read_only": True,
        **event_source_summary,
        "event_contract": event_contract,
    }
    return {
        "events": events,
        "signals": read_evaluation_signals(signal_log_path),
        "comparisons": read_evaluation_comparisons(comparison_log_path),
        "recalls": recalls,
        "learning_previews": learning_previews,
        "pack_audits": pack_audits,
        "review_reports": review_reports,
        "source_paths": source_paths,
        "index_summary": index_summary,
    }


def _recall_source_id(recall: Mapping[str, Any]) -> str:
    paths = _mapping_dict(recall.get("paths"))
    return (
        _clean_text(paths.get("recall_run_path"))
        or _clean_text(recall.get("_source_path"))
        or f"recall:{_stable_digest(recall, length=20)}"
    )


def _comparison_source_id(comparison: Mapping[str, Any]) -> str:
    return _clean_text(comparison.get("comparison_id")) or f"comparison:{_stable_digest(comparison, length=20)}"


def _signal_source_id(signal: Mapping[str, Any]) -> str:
    return _clean_text(signal.get("signal_id")) or f"signal:{_stable_digest(signal, length=20)}"


def _candidate_event_ids_from_comparison(comparison: Mapping[str, Any]) -> list[str]:
    return [
        event_id
        for candidate in comparison.get("candidates") or []
        if isinstance(candidate, Mapping)
        if (event_id := _clean_text(candidate.get("event_id"))) is not None
    ]


def _candidate_backend_metadata(candidate: Mapping[str, Any]) -> dict[str, Any]:
    backend = _mapping_dict(candidate.get("backend_metadata"))
    return {
        key: value
        for key, value in {
            "backend_id": _clean_text(backend.get("backend_id")),
            "display_name": _clean_text(backend.get("display_name")),
            "adapter_kind": _clean_text(backend.get("adapter_kind")),
            "model_id": _clean_text(backend.get("model_id")),
            "compatibility_status": _clean_text(backend.get("compatibility_status")),
        }.items()
        if value is not None
    }


def _support_validity(support: Mapping[str, Any]) -> tuple[str, str]:
    support_class = _clean_text(support.get("support_class")) or "unknown"
    if support.get("can_support_decision"):
        return "valid", "strong"
    if support_class in {"current_review_subject", "future_evidence"}:
        return "invalid", "diagnostic_only"
    if support_class in HARD_BLOCKING_SUPPORT_CLASSES:
        return "invalid", "diagnostic_only"
    return "unknown", "diagnostic_only"


def _support_kernel_for_graph_metadata(support: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(support))
    payload.pop("checked_at_utc", None)
    return payload


def _event_recorded_at(events_by_id: Mapping[str, Mapping[str, Any]], event_id: str | None) -> str | None:
    if event_id is None:
        return None
    return _clean_text(_mapping_dict(events_by_id.get(event_id)).get("recorded_at_utc"))


def _recall_edge_validity(
    *,
    source_event_id: str | None,
    candidate_event_id: str,
    events_by_id: Mapping[str, Mapping[str, Any]],
    event_targets: Mapping[str, str | None],
    support: Mapping[str, Any],
) -> tuple[str, str, str]:
    support_validity, strength = _support_validity(support)
    explanation_parts = [
        f"support_class={_clean_text(support.get('support_class')) or 'unknown'}",
        f"can_support={bool(support.get('can_support_decision'))}",
    ]
    validity = support_validity
    if source_event_id and source_event_id == candidate_event_id:
        validity = "invalid"
        strength = "diagnostic_only"
        explanation_parts.append("self_recall")
    source_fingerprint = event_targets.get(source_event_id or "")
    candidate_fingerprint = event_targets.get(candidate_event_id)
    if source_fingerprint and candidate_fingerprint and source_fingerprint == candidate_fingerprint:
        validity = "invalid"
        strength = "diagnostic_only"
        explanation_parts.append("target_fingerprint_match")
    source_time = _coerce_utc_datetime(_event_recorded_at(events_by_id, source_event_id))
    candidate_time = _coerce_utc_datetime(_event_recorded_at(events_by_id, candidate_event_id))
    if source_time is not None and candidate_time is not None and candidate_time >= source_time:
        validity = "invalid"
        strength = "diagnostic_only"
        explanation_parts.append("future_evidence")
    return validity, strength, "; ".join(explanation_parts)


def build_evidence_graph(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    events: Iterable[Mapping[str, Any]] | None = None,
    signals: Iterable[Mapping[str, Any]] | None = None,
    comparisons: Iterable[Mapping[str, Any]] | None = None,
    recalls: Iterable[Mapping[str, Any]] | None = None,
    learning_previews: Iterable[Mapping[str, Any]] | None = None,
    pack_audits: Iterable[Mapping[str, Any]] | None = None,
    review_reports: Iterable[Mapping[str, Any]] | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    generated_at = generated_at_utc or timestamp_utc()
    if events is None:
        loaded = _load_default_sources(root=resolved_root, workspace_id=workspace_id)
        source_events = loaded["events"]
        source_signals = loaded["signals"] if signals is None else [dict(item) for item in signals]
        source_comparisons = loaded["comparisons"] if comparisons is None else [dict(item) for item in comparisons]
        source_recalls = loaded["recalls"] if recalls is None else [dict(item) for item in recalls]
        source_learning_previews = loaded["learning_previews"] if learning_previews is None else [dict(item) for item in learning_previews]
        source_pack_audits = loaded["pack_audits"] if pack_audits is None else [dict(item) for item in pack_audits]
        source_review_reports = loaded["review_reports"] if review_reports is None else [dict(item) for item in review_reports]
        source_paths = copy.deepcopy(loaded["source_paths"])
        index_summary = copy.deepcopy(loaded["index_summary"])
    else:
        source_events = [dict(item) for item in events]
        source_signals = [dict(item) for item in signals or []]
        source_comparisons = [dict(item) for item in comparisons or []]
        source_recalls = [dict(item) for item in recalls or []]
        source_learning_previews = [dict(item) for item in learning_previews or []]
        source_pack_audits = [dict(item) for item in pack_audits or []]
        source_review_reports = [dict(item) for item in review_reports or []]
        source_paths = {}
        index_summary = None

    events_by_id = {
        event_id: dict(event)
        for event in source_events
        if (event_id := _event_id(event)) is not None
    }
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    event_targets: dict[str, str | None] = {}
    signal_nodes_by_id: dict[str, str] = {}

    for event_id, event in sorted(
        events_by_id.items(),
        key=lambda item: (_timestamp_sort_key(item[1].get("recorded_at_utc")), item[0]),
    ):
        refs = artifact_refs_from_event(event, root=resolved_root)
        target_identity = target_identity_for_event(event, root=resolved_root)
        support = build_evidence_support_result(
            event_id,
            event=event,
            root=resolved_root,
            checked_at_utc=DETERMINISTIC_SUPPORT_CHECKED_AT_UTC,
        )
        event_targets[event_id] = _clean_text(target_identity.get("target_fingerprint"))
        support_polarity = _normalize_polarity(support.get("support_polarity"))
        event_node = _node(
            node_kind="event",
            source_id=event_id,
            support_class=_clean_text(support.get("support_class")) or "unknown",
            can_support_decision=bool(support.get("can_support_decision")),
            polarity=support_polarity,
            quality_status=_event_quality_status(event, support),
            target_fingerprint=_clean_text(target_identity.get("target_fingerprint")),
            created_at_utc=_clean_text(event.get("recorded_at_utc")),
            metadata={
                "support_kernel": _support_kernel_for_graph_metadata(support),
                "support_kernel_checked_at_policy": "omitted_for_deterministic_derived_graph",
                "target_identity": target_identity,
                "changed_files": target_identity.get("changed_files") or [],
                "hunk_headers": target_identity.get("hunk_headers") or [],
                "issue_keys": target_identity.get("issue_keys") or [],
                "source_paths": target_identity.get("source_paths") or [],
                "artifact_refs": [ref.get("artifact_id") for ref in refs if ref.get("artifact_id")],
                "event_kind": _clean_text(event.get("event_kind")),
                "workspace_id": _clean_text(_mapping_dict(event.get("workspace")).get("workspace_id")),
            },
        )
        _add_node(nodes, event_node)
        for ref in refs:
            artifact_id = _clean_text(ref.get("artifact_id")) or f"artifact:{_stable_digest(ref, length=20)}"
            quality, support_class, artifact_metadata = _artifact_quality_and_support(ref, root=resolved_root)
            artifact_node = _node(
                node_kind="artifact",
                source_id=artifact_id,
                support_class=support_class,
                can_support_decision=False,
                polarity="neutral",
                quality_status=quality,
                target_fingerprint=_clean_text(target_identity.get("target_fingerprint")),
                created_at_utc=_clean_text(ref.get("captured_at_utc")),
                metadata=artifact_metadata,
            )
            _add_node(nodes, artifact_node)
            _add_edge(
                edges,
                _edge(
                    from_node_id=artifact_node["node_id"],
                    to_node_id=event_node["node_id"],
                    relation_kind="uses_artifact",
                    source_id=artifact_id,
                    causal_validity="valid" if artifact_metadata.get("object_verified") else "unknown",
                    strength="strong" if artifact_metadata.get("object_verified") else "diagnostic_only",
                    explanation="Event is source-linked to this artifact ref.",
                    metadata={"artifact_id": artifact_id, "event_id": event_id},
                ),
            )

    for signal in sorted(
        source_signals,
        key=lambda item: (_timestamp_sort_key(item.get("recorded_at_utc")), _signal_source_id(item)),
    ):
        signal_id = _signal_source_id(signal)
        source_event_id = _signal_source_event_id(signal)
        signal_node = _node(
            node_kind="signal",
            source_id=signal_id,
            support_class="source_linked_prior" if source_event_id in events_by_id else "missing_source",
            can_support_decision=False,
            polarity=_signal_polarity(signal),
            quality_status=_signal_quality_status(signal),
            target_fingerprint=event_targets.get(source_event_id or ""),
            created_at_utc=_clean_text(signal.get("recorded_at_utc")),
            metadata={
                "signal_kind": _clean_text(signal.get("signal_kind")),
                "origin": _clean_text(signal.get("origin")),
                "source_event_id": source_event_id,
                "target_event_id": _signal_target_event_id(signal),
                "relation_kind": _signal_relation_kind(signal),
                "evidence": copy.deepcopy(_mapping_dict(signal.get("evidence"))),
                "tags": _string_list(signal.get("tags")),
            },
        )
        _add_node(nodes, signal_node)
        signal_nodes_by_id[signal_id] = signal_node["node_id"]
        created_by = _source_created_by(signal.get("origin"), _string_list(signal.get("tags")))
        if source_event_id in events_by_id:
            _add_edge(
                edges,
                _edge(
                    from_node_id=_event_node_id(source_event_id),
                    to_node_id=signal_node["node_id"],
                    relation_kind="evaluates",
                    source_id=signal_id,
                    created_by=created_by,
                    explanation="Evaluation signal records a verdict or test signal for the event.",
                    metadata={"signal_kind": _clean_text(signal.get("signal_kind")), "source_event_id": source_event_id},
                ),
            )
        target_event_id = _signal_target_event_id(signal)
        relation_kind = _signal_relation_kind(signal)
        if target_event_id in events_by_id and relation_kind:
            graph_relation = "repairs" if relation_kind in {"repairs", "follow_up_for"} else "derives_from"
            _add_edge(
                edges,
                _edge(
                    from_node_id=signal_node["node_id"],
                    to_node_id=_event_node_id(target_event_id),
                    relation_kind=graph_relation,
                    source_id=f"{signal_id}:{target_event_id}",
                    created_by=created_by,
                    explanation=f"Signal relation `{relation_kind}` targets this event.",
                    metadata={"relation_kind": relation_kind, "source_event_id": source_event_id, "target_event_id": target_event_id},
                ),
            )

    signals_by_event: dict[str, list[dict[str, Any]]] = {}
    for signal in source_signals:
        source_event_id = _signal_source_event_id(signal)
        if source_event_id is None:
            continue
        signals_by_event.setdefault(source_event_id, []).append(dict(signal))
    for event_id, event_signals in sorted(signals_by_event.items()):
        positives = [signal for signal in event_signals if _clean_text(signal.get("signal_kind")) in POSITIVE_SIGNAL_KINDS]
        negatives = [signal for signal in event_signals if _clean_text(signal.get("signal_kind")) in NEGATIVE_SIGNAL_KINDS]
        if not positives or not negatives:
            continue
        relevant = sorted(
            [*positives, *negatives],
            key=lambda item: (_timestamp_sort_key(item.get("recorded_at_utc")), _signal_source_id(item)),
            reverse=True,
        )
        latest_signal = relevant[0]
        latest_is_negative = _clean_text(latest_signal.get("signal_kind")) in NEGATIVE_SIGNAL_KINDS
        contradiction_validity = "valid" if latest_is_negative else "invalid"
        contradiction_strength = "strong" if latest_is_negative else "diagnostic_only"
        contradiction_explanation = (
            "A later negative verdict or test signal contradicts earlier positive evidence."
            if latest_is_negative
            else "A newer positive verdict or test signal resolves this earlier negative signal; contradiction is diagnostic history."
        )
        negatives.sort(key=lambda item: (_timestamp_sort_key(item.get("recorded_at_utc")), _signal_source_id(item)), reverse=True)
        latest_negative = negatives[0]
        negative_node_id = signal_nodes_by_id.get(_signal_source_id(latest_negative))
        if negative_node_id is None:
            continue
        for positive in positives:
            positive_node_id = signal_nodes_by_id.get(_signal_source_id(positive))
            if positive_node_id is None:
                continue
            _add_edge(
                edges,
                _edge(
                    from_node_id=negative_node_id,
                    to_node_id=positive_node_id,
                    relation_kind="contradicts",
                    source_id=f"{_signal_source_id(latest_negative)}:{_signal_source_id(positive)}",
                    created_by=_source_created_by(latest_negative.get("origin"), _string_list(latest_negative.get("tags"))),
                    causal_validity=contradiction_validity,
                    strength=contradiction_strength,
                    explanation=contradiction_explanation,
                    metadata={"event_id": event_id},
                ),
            )
        if event_id in events_by_id:
            _add_edge(
                edges,
                _edge(
                    from_node_id=negative_node_id,
                    to_node_id=_event_node_id(event_id),
                    relation_kind="contradicts",
                    source_id=f"{_signal_source_id(latest_negative)}:{event_id}",
                    created_by=_source_created_by(latest_negative.get("origin"), _string_list(latest_negative.get("tags"))),
                    causal_validity=contradiction_validity,
                    strength=contradiction_strength,
                    explanation=(
                        "This event has contradictory verdict history; positive support must not be promoted without resolution."
                        if latest_is_negative
                        else "This event had contradictory verdict history, but the latest signal is positive or resolved."
                    ),
                    metadata={"event_id": event_id},
                ),
            )

    for comparison in sorted(
        source_comparisons,
        key=lambda item: (_timestamp_sort_key(item.get("recorded_at_utc")), _comparison_source_id(item)),
    ):
        comparison_id = _comparison_source_id(comparison)
        candidate_event_ids = _candidate_event_ids_from_comparison(comparison)
        winner_event_id = _clean_text(comparison.get("winner_event_id"))
        outcome = _clean_text(comparison.get("outcome"))
        comparison_node = _node(
            node_kind="comparison",
            source_id=comparison_id,
            support_class="source_linked_prior" if candidate_event_ids else "missing_source",
            can_support_decision=False,
            polarity="positive" if outcome == "winner_selected" else "neutral",
            quality_status="human_accepted" if outcome == "winner_selected" else "unresolved",
            created_at_utc=_clean_text(comparison.get("recorded_at_utc")),
            metadata={
                "outcome": outcome,
                "winner_event_id": winner_event_id,
                "candidate_event_ids": candidate_event_ids,
                "origin": _clean_text(comparison.get("origin")),
                "task_label": _clean_text(comparison.get("task_label")),
                "rationale": _clean_text(comparison.get("rationale")),
            },
        )
        _add_node(nodes, comparison_node)
        created_by = _source_created_by(comparison.get("origin"), _string_list(comparison.get("tags")))
        for candidate in comparison.get("candidates") or []:
            if not isinstance(candidate, Mapping):
                continue
            candidate_event_id = _clean_text(candidate.get("event_id"))
            if candidate_event_id not in events_by_id:
                continue
            _add_edge(
                edges,
                _edge(
                    from_node_id=_event_node_id(candidate_event_id),
                    to_node_id=comparison_node["node_id"],
                    relation_kind="evaluates",
                    source_id=f"{comparison_id}:{candidate_event_id}",
                    created_by=created_by,
                    explanation="Comparison evaluates this candidate event.",
                    metadata={"comparison_id": comparison_id, "event_id": candidate_event_id},
                ),
            )
            relation = "selects" if outcome == "winner_selected" and candidate_event_id == winner_event_id else "rejects" if outcome == "winner_selected" else "evaluates"
            _add_edge(
                edges,
                _edge(
                    from_node_id=comparison_node["node_id"],
                    to_node_id=_event_node_id(candidate_event_id),
                    relation_kind=relation,
                    source_id=f"{comparison_id}:{relation}:{candidate_event_id}",
                    created_by=created_by,
                    explanation=f"Comparison outcome marks this candidate as `{relation}`.",
                    metadata={"comparison_id": comparison_id, "event_id": candidate_event_id, "outcome": outcome},
                ),
            )
            backend_metadata = _candidate_backend_metadata(candidate)
            backend_id = backend_metadata.get("backend_id") or backend_metadata.get("model_id")
            if backend_id:
                backend_node = _node(
                    node_kind="backend_candidate",
                    source_id=str(backend_id),
                    support_class="source_linked_prior",
                    can_support_decision=False,
                    polarity="neutral",
                    quality_status=_quality_from_status(backend_metadata.get("compatibility_status")),
                    created_at_utc=_clean_text(comparison.get("recorded_at_utc")),
                    metadata=backend_metadata,
                )
                _add_node(nodes, backend_node)
                _add_edge(
                    edges,
                    _edge(
                        from_node_id=_event_node_id(candidate_event_id),
                        to_node_id=backend_node["node_id"],
                        relation_kind="uses_artifact",
                        source_id=f"{comparison_id}:{candidate_event_id}:{backend_id}",
                        created_by=created_by,
                        explanation="Candidate event records backend or model metadata.",
                        metadata={"comparison_id": comparison_id, "event_id": candidate_event_id},
                    ),
                )

    for recall in sorted(
        source_recalls,
        key=lambda item: (_timestamp_sort_key(item.get("generated_at_utc")), _recall_source_id(item)),
    ):
        recall_source_id = _recall_source_id(recall)
        request = _mapping_dict(recall.get("request"))
        bundle = _mapping_dict(recall.get("bundle"))
        source_event_id = _clean_text(request.get("source_event_id"))
        recall_node = _node(
            node_kind="recall",
            source_id=recall_source_id,
            support_class="source_linked_prior",
            can_support_decision=False,
            polarity="neutral",
            quality_status="unresolved",
            target_fingerprint=event_targets.get(source_event_id or ""),
            created_at_utc=_clean_text(recall.get("generated_at_utc")),
            metadata={
                "source_event_id": source_event_id,
                "query_text": _clean_text(request.get("query_text")),
                "file_hints": _string_list(request.get("file_hints")),
                "selected_count": bundle.get("selected_count"),
                "risk_note": copy.deepcopy(_mapping_dict(recall.get("risk_note"))),
                "has_human_usefulness_signal": False,
            },
        )
        _add_node(nodes, recall_node)
        if source_event_id in events_by_id:
            _add_edge(
                edges,
                _edge(
                    from_node_id=_event_node_id(source_event_id),
                    to_node_id=recall_node["node_id"],
                    relation_kind="derives_from",
                    source_id=f"{recall_source_id}:{source_event_id}",
                    explanation="Recall request derives from the active source event.",
                    metadata={"source_event_id": source_event_id},
                ),
            )
        for row in bundle.get("selected_candidates") or []:
            if not isinstance(row, Mapping):
                continue
            candidate_event_id = _clean_text(row.get("event_id"))
            if candidate_event_id not in events_by_id:
                continue
            review_started_at = _clean_text(request.get("recorded_before_utc")) or _event_recorded_at(events_by_id, source_event_id)
            support = build_evidence_support_result(
                candidate_event_id,
                event=events_by_id[candidate_event_id],
                review_started_at=review_started_at,
                active_subject=source_event_id,
                root=resolved_root,
                checked_at_utc=DETERMINISTIC_SUPPORT_CHECKED_AT_UTC,
            )
            validity, strength, explanation = _recall_edge_validity(
                source_event_id=source_event_id,
                candidate_event_id=candidate_event_id,
                events_by_id=events_by_id,
                event_targets=event_targets,
                support=support,
            )
            edge_can_support_decision = bool(support.get("can_support_decision")) and validity == "valid"
            if "pinned" in {reason.lower() for reason in _string_list(row.get("reasons"))}:
                strength = "manual_pin" if validity == "valid" else "diagnostic_only"
            _add_edge(
                edges,
                _edge(
                    from_node_id=recall_node["node_id"],
                    to_node_id=_event_node_id(candidate_event_id),
                    relation_kind="recalls",
                    source_id=f"{recall_source_id}:{candidate_event_id}",
                    strength=strength,
                    causal_validity=validity,
                    explanation=explanation,
                    metadata={
                        "source_event_id": source_event_id,
                        "candidate_event_id": candidate_event_id,
                        "score": row.get("score"),
                        "reasons": _string_list(row.get("reasons")),
                        "event_contract_status": _clean_text(row.get("event_contract_status")),
                        "source_artifact_status": _clean_text(row.get("source_artifact_status")),
                        "support_kernel": _support_kernel_for_graph_metadata(support),
                        "support_kernel_checked_at_policy": "omitted_for_deterministic_derived_graph",
                    },
                    support_class=_clean_text(support.get("support_class")) or "unknown",
                    can_support_decision=edge_can_support_decision,
                    polarity=_normalize_polarity(support.get("support_polarity")),
                ),
            )

    for preview in sorted(
        source_learning_previews,
        key=lambda item: (_timestamp_sort_key(item.get("generated_at_utc")), _clean_text(item.get("_source_path")) or ""),
    ):
        preview_source = _clean_text(preview.get("_source_path")) or _clean_text(preview.get("source_curation_preview_path")) or f"learning:{_stable_digest(preview, length=20)}"
        for item in preview.get("review_queue") or []:
            if not isinstance(item, Mapping):
                continue
            queue_item_id = _clean_text(item.get("queue_item_id")) or f"{preview_source}:{_stable_digest(item, length=12)}"
            event_id = _clean_text(item.get("event_id"))
            blocked_reasons = _string_list(item.get("blocked_reasons"))
            queue_state = _clean_text(item.get("queue_state")) or "needs_review"
            learning_node = _node(
                node_kind="learning_candidate",
                source_id=queue_item_id,
                support_class="missing_source" if queue_state == "missing_source" else "unknown",
                can_support_decision=False,
                polarity="neutral",
                quality_status="blocked" if blocked_reasons or queue_state in {"blocked", "missing_source", "missing_supervised_text"} else "unresolved",
                target_fingerprint=event_targets.get(event_id or ""),
                created_at_utc=_clean_text(preview.get("generated_at_utc")),
                metadata={
                    "preview_source": preview_source,
                    "event_id": event_id,
                    "queue_state": queue_state,
                    "blocked_reason": _clean_text(item.get("blocked_reason")),
                    "blocked_reasons": blocked_reasons,
                    "eligible_for_supervised_candidate": bool(item.get("eligible_for_supervised_candidate")),
                    "preview_only": True,
                    "next_action": _clean_text(item.get("next_action")),
                    "curation": copy.deepcopy(_mapping_dict(item.get("curation"))),
                    "lifecycle_summary": copy.deepcopy(_mapping_dict(item.get("lifecycle_summary"))),
                },
            )
            _add_node(nodes, learning_node)
            if event_id in events_by_id:
                _add_edge(
                    edges,
                    _edge(
                        from_node_id=_event_node_id(event_id),
                        to_node_id=learning_node["node_id"],
                        relation_kind="derives_from",
                        source_id=f"{queue_item_id}:{event_id}",
                        strength="diagnostic_only",
                        explanation="Learning candidate is preview-only and derived from this event.",
                        metadata={"event_id": event_id, "queue_state": queue_state, "blocked_reasons": blocked_reasons},
                    ),
                )

    for audit in sorted(
        source_pack_audits,
        key=lambda item: (_timestamp_sort_key(item.get("audited_at_utc")), _clean_text(item.get("audit_id")) or ""),
    ):
        audit_id = _clean_text(audit.get("audit_id")) or f"pack-audit:{_stable_digest(audit, length=20)}"
        verdict = _clean_text(audit.get("verdict"))
        audit_node = _node(
            node_kind="review",
            source_id=audit_id,
            support_class="source_linked_prior",
            can_support_decision=False,
            polarity="risk" if verdict == "block" else "neutral",
            quality_status="blocked" if verdict == "block" else "unresolved" if verdict == "needs_review" else "verified",
            created_at_utc=_clean_text(audit.get("audited_at_utc")),
            metadata={
                "source_kind": "pack_audit",
                "created_by": "core",
                "pack_name": _clean_text(audit.get("pack_name")),
                "verdict": verdict,
                "blocked_reasons": _string_list(audit.get("blocked_reasons")),
                "review_reasons": _string_list(audit.get("review_reasons")),
                "source_paths": _string_list(audit.get("source_paths")),
            },
        )
        _add_node(nodes, audit_node)

    for report in sorted(
        source_review_reports,
        key=lambda item: (_timestamp_sort_key(item.get("generated_at_utc")), _clean_text(item.get("_source_path")) or ""),
    ):
        source_id = _clean_text(report.get("_source_path")) or f"review-report:{_stable_digest(report, length=20)}"
        report_event_id = _clean_text(report.get("event_id"))
        created_by = "pack" if _mapping_dict(report.get("pack_run")) else "core"
        review_node = _node(
            node_kind="review",
            source_id=source_id,
            support_class="source_linked_prior" if report_event_id in events_by_id else "unknown",
            can_support_decision=False,
            polarity="risk" if _mapping_dict(report.get("risk_note")).get("level") == "block" else "neutral",
            quality_status="unresolved",
            target_fingerprint=event_targets.get(report_event_id or ""),
            created_at_utc=_clean_text(report.get("generated_at_utc")),
            metadata={
                "source_kind": "review_risk_report",
                "created_by": created_by,
                "event_id": report_event_id,
                "risk_note": copy.deepcopy(_mapping_dict(report.get("risk_note"))),
                "pack_run": copy.deepcopy(_mapping_dict(report.get("pack_run"))),
                "learning_preview_counts": copy.deepcopy(_mapping_dict(report.get("learning_preview_counts"))),
            },
        )
        _add_node(nodes, review_node)
        if report_event_id in events_by_id:
            _add_edge(
                edges,
                _edge(
                    from_node_id=_event_node_id(report_event_id),
                    to_node_id=review_node["node_id"],
                    relation_kind="evaluates",
                    source_id=f"{source_id}:{report_event_id}",
                    created_by=created_by,
                    strength="diagnostic_only",
                    explanation="Review report is derived from the event and cannot replace source evidence.",
                    metadata={"event_id": report_event_id},
                ),
            )

    sorted_nodes = sorted(nodes.values(), key=lambda item: (item["node_kind"], item["source_id"], item["node_id"]))
    sorted_edges = sorted(edges.values(), key=lambda item: (item["relation_kind"], item["from_node_id"], item["to_node_id"], item["edge_id"]))
    node_counts = Counter(node["node_kind"] for node in sorted_nodes)
    edge_counts = Counter(edge["relation_kind"] for edge in sorted_edges)
    invalid_edges = [edge for edge in sorted_edges if edge.get("causal_validity") == "invalid"]
    learning_blocker_count = sum(
        len(_string_list(_mapping_dict(node.get("metadata")).get("blocked_reasons")))
        for node in sorted_nodes
        if node.get("node_kind") == "learning_candidate"
    )
    support_match_count = sum(
        1
        for node in sorted_nodes
        if node.get("node_kind") == "event"
        and _mapping_dict(node.get("metadata")).get("support_kernel", {}).get("can_support_decision")
        == node.get("can_support_decision")
    )
    event_node_count = int(node_counts.get("event", 0))
    graph_core = {
        "nodes": sorted_nodes,
        "edges": sorted_edges,
        "source_paths": source_paths,
    }
    return {
        "schema_name": EVIDENCE_GRAPH_SCHEMA_NAME,
        "schema_version": EVIDENCE_GRAPH_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "generated_at_utc": generated_at,
        "derived": True,
        "rebuildable": True,
        "source_of_truth_policy": {
            "graph_is_derived": True,
            "does_not_edit_event_logs": True,
            "does_not_edit_artifact_refs": True,
            "does_not_edit_verdicts": True,
            "does_not_edit_comparison_records": True,
        },
        "target_identity_model": {
            "algorithm": "sha256(normalized changed file paths + hunk headers + optional issue/task key)",
            "version": 1,
        },
        "source_paths": source_paths,
        "counts": {
            "node_count": len(sorted_nodes),
            "edge_count": len(sorted_edges),
            "nodes_by_kind": {key: int(value) for key, value in sorted(node_counts.items())},
            "edges_by_relation": {key: int(value) for key, value in sorted(edge_counts.items())},
            "invalid_edge_count": len(invalid_edges),
            "support_kernel_event_node_count": event_node_count,
            "support_kernel_decision_match_count": support_match_count,
            "support_kernel_decisions_match_graph_nodes": support_match_count == event_node_count,
            "learning_preview_graph_blocker_count": learning_blocker_count,
        },
        "index_summary": index_summary,
        "graph_digest": _stable_digest(graph_core, length=32),
        "nodes": sorted_nodes,
        "edges": sorted_edges,
    }


def validate_evidence_graph_snapshot(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if graph.get("schema_name") != EVIDENCE_GRAPH_SCHEMA_NAME:
        issues.append({"path": "$.schema_name", "message": "Unexpected evidence graph schema name."})
    if graph.get("schema_version") != EVIDENCE_GRAPH_SCHEMA_VERSION:
        issues.append({"path": "$.schema_version", "message": "Unsupported evidence graph schema version."})
    if _clean_text(graph.get("workspace_id")) is None:
        issues.append({"path": "$.workspace_id", "message": "Workspace id is required."})
    if _clean_text(graph.get("generated_at_utc")) is None:
        issues.append({"path": "$.generated_at_utc", "message": "Graph generation timestamp is required."})
    if graph.get("derived") is not True:
        issues.append({"path": "$.derived", "message": "Evidence graph must be marked derived."})
    if graph.get("rebuildable") is not True:
        issues.append({"path": "$.rebuildable", "message": "Evidence graph must be marked rebuildable."})
    if not isinstance(graph.get("target_identity_model"), Mapping):
        issues.append({"path": "$.target_identity_model", "message": "Target identity model is required."})
    if not isinstance(graph.get("counts"), Mapping):
        issues.append({"path": "$.counts", "message": "Graph counts must be an object."})
    if _clean_text(graph.get("graph_digest")) is None:
        issues.append({"path": "$.graph_digest", "message": "Graph digest is required."})
    policy = _mapping_dict(graph.get("source_of_truth_policy"))
    for key in (
        "graph_is_derived",
        "does_not_edit_event_logs",
        "does_not_edit_artifact_refs",
        "does_not_edit_verdicts",
        "does_not_edit_comparison_records",
    ):
        if policy.get(key) is not True:
            issues.append(
                {
                    "path": f"$.source_of_truth_policy.{key}",
                    "message": "Source-of-truth policy flag must be true.",
                }
            )
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list):
        issues.append({"path": "$.nodes", "message": "Graph nodes must be an array."})
        nodes = []
    if not isinstance(edges, list):
        issues.append({"path": "$.edges", "message": "Graph edges must be an array."})
        edges = []
    node_ids: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            issues.append({"path": f"$.nodes[{index}]", "message": "Node must be an object."})
            continue
        node_id = _clean_text(node.get("node_id"))
        node_kind = _clean_text(node.get("node_kind"))
        source_id = _clean_text(node.get("source_id"))
        if node_id is None:
            issues.append({"path": f"$.nodes[{index}].node_id", "message": "Node id is required."})
        elif node_id in node_ids:
            issues.append({"path": f"$.nodes[{index}].node_id", "message": "Node id must be unique."})
        else:
            node_ids.add(node_id)
        if source_id is None:
            issues.append({"path": f"$.nodes[{index}].source_id", "message": "Node source id is required."})
        if node_kind not in NODE_KINDS:
            issues.append({"path": f"$.nodes[{index}].node_kind", "message": "Unsupported node kind."})
        if _clean_text(node.get("support_class")) not in SUPPORT_CLASSES:
            issues.append({"path": f"$.nodes[{index}].support_class", "message": "Unsupported support class."})
        if not isinstance(node.get("can_support_decision"), bool):
            issues.append(
                {
                    "path": f"$.nodes[{index}].can_support_decision",
                    "message": "Decision support flag must be a boolean.",
                }
            )
        if _clean_text(node.get("polarity")) not in POLARITIES:
            issues.append({"path": f"$.nodes[{index}].polarity", "message": "Unsupported polarity."})
        if _clean_text(node.get("quality_status")) not in QUALITY_STATUSES:
            issues.append({"path": f"$.nodes[{index}].quality_status", "message": "Unsupported quality status."})
        if "created_at_utc" not in node:
            issues.append(
                {
                    "path": f"$.nodes[{index}].created_at_utc",
                    "message": "Node creation timestamp is required.",
                }
            )
        elif node.get("created_at_utc") is not None and not isinstance(node.get("created_at_utc"), str):
            issues.append(
                {
                    "path": f"$.nodes[{index}].created_at_utc",
                    "message": "Node creation timestamp must be a string or null.",
                }
            )
        if (
            "target_fingerprint" in node
            and node.get("target_fingerprint") is not None
            and not isinstance(node.get("target_fingerprint"), str)
        ):
            issues.append(
                {
                    "path": f"$.nodes[{index}].target_fingerprint",
                    "message": "Target fingerprint must be a string or null.",
                }
            )
        if "metadata" in node and not isinstance(node.get("metadata"), Mapping):
            issues.append({"path": f"$.nodes[{index}].metadata", "message": "Node metadata must be an object."})
    edge_ids: set[str] = set()
    for index, edge in enumerate(edges):
        if not isinstance(edge, Mapping):
            issues.append({"path": f"$.edges[{index}]", "message": "Edge must be an object."})
            continue
        edge_id = _clean_text(edge.get("edge_id"))
        from_node_id = _clean_text(edge.get("from_node_id"))
        to_node_id = _clean_text(edge.get("to_node_id"))
        if edge_id is None:
            issues.append({"path": f"$.edges[{index}].edge_id", "message": "Edge id is required."})
        elif edge_id in edge_ids:
            issues.append({"path": f"$.edges[{index}].edge_id", "message": "Edge id must be unique."})
        else:
            edge_ids.add(edge_id)
        if from_node_id is None:
            issues.append({"path": f"$.edges[{index}].from_node_id", "message": "Edge source node id is required."})
        elif from_node_id not in node_ids:
            issues.append({"path": f"$.edges[{index}].from_node_id", "message": "Edge source node is missing."})
        if to_node_id is None:
            issues.append({"path": f"$.edges[{index}].to_node_id", "message": "Edge target node id is required."})
        elif to_node_id not in node_ids:
            issues.append({"path": f"$.edges[{index}].to_node_id", "message": "Edge target node is missing."})
        if _clean_text(edge.get("relation_kind")) not in RELATION_KINDS:
            issues.append({"path": f"$.edges[{index}].relation_kind", "message": "Unsupported relation kind."})
        if _clean_text(edge.get("strength")) not in EDGE_STRENGTHS:
            issues.append({"path": f"$.edges[{index}].strength", "message": "Unsupported edge strength."})
        if _clean_text(edge.get("causal_validity")) not in CAUSAL_VALIDITIES:
            issues.append({"path": f"$.edges[{index}].causal_validity", "message": "Unsupported causal validity."})
        if _clean_text(edge.get("created_by")) not in EDGE_CREATORS:
            issues.append({"path": f"$.edges[{index}].created_by", "message": "Unsupported edge creator."})
        if not isinstance(edge.get("explanation"), str):
            issues.append({"path": f"$.edges[{index}].explanation", "message": "Edge explanation is required."})
        if "metadata" in edge and not isinstance(edge.get("metadata"), Mapping):
            issues.append({"path": f"$.edges[{index}].metadata", "message": "Edge metadata must be an object."})
    return issues


def _node_label(node: Mapping[str, Any]) -> str:
    return f"{node.get('node_kind')}:{node.get('source_id')}"


def format_evidence_graph_markdown(graph: Mapping[str, Any]) -> str:
    counts = _mapping_dict(graph.get("counts"))
    nodes_by_kind = _mapping_dict(counts.get("nodes_by_kind"))
    edges_by_relation = _mapping_dict(counts.get("edges_by_relation"))
    lines = [
        "# Derived Evidence Graph",
        "",
        f"Workspace: {_clean_text(graph.get('workspace_id')) or DEFAULT_WORKSPACE_ID}",
        f"Derived/rebuildable: {str(bool(graph.get('derived') and graph.get('rebuildable'))).lower()}",
        f"Graph digest: {_clean_text(graph.get('graph_digest')) or 'n/a'}",
        "",
        "## Counts",
        "",
        f"- Nodes: {int(counts.get('node_count') or 0)}",
        f"- Edges: {int(counts.get('edge_count') or 0)}",
        f"- Invalid edges: {int(counts.get('invalid_edge_count') or 0)}",
        f"- Learning graph blockers: {int(counts.get('learning_preview_graph_blocker_count') or 0)}",
        f"- Support kernel match: {str(bool(counts.get('support_kernel_decisions_match_graph_nodes'))).lower()}",
    ]
    if nodes_by_kind:
        lines.extend(("", "Nodes by kind:"))
        for key, value in sorted(nodes_by_kind.items()):
            lines.append(f"- {key}: {int(value)}")
    if edges_by_relation:
        lines.extend(("", "Edges by relation:"))
        for key, value in sorted(edges_by_relation.items()):
            lines.append(f"- {key}: {int(value)}")

    invalid_edges = [
        edge
        for edge in graph.get("edges") or []
        if isinstance(edge, Mapping) and edge.get("causal_validity") == "invalid"
    ]
    if invalid_edges:
        node_by_id = {node.get("node_id"): node for node in graph.get("nodes") or [] if isinstance(node, Mapping)}
        lines.extend(("", "## Invalid Edges", ""))
        for edge in invalid_edges[:12]:
            from_node = _mapping_dict(node_by_id.get(edge.get("from_node_id")))
            to_node = _mapping_dict(node_by_id.get(edge.get("to_node_id")))
            lines.append(
                f"- {edge.get('relation_kind')}: {_node_label(from_node)} -> {_node_label(to_node)} "
                f"({edge.get('explanation') or 'no explanation'})"
            )
    return "\n".join(lines) + "\n"


def build_evidence_trace(
    event_id: str,
    *,
    graph: Mapping[str, Any] | None = None,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    why_blocked: bool = False,
) -> dict[str, Any]:
    resolved_graph = dict(graph) if isinstance(graph, Mapping) else build_evidence_graph(root=root, workspace_id=workspace_id)
    nodes = [dict(node) for node in resolved_graph.get("nodes") or [] if isinstance(node, Mapping)]
    edges = [dict(edge) for edge in resolved_graph.get("edges") or [] if isinstance(edge, Mapping)]
    event_node = next(
        (
            node
            for node in nodes
            if node.get("node_kind") == "event" and _clean_text(node.get("source_id")) == event_id
        ),
        None,
    )
    if event_node is None:
        return {
            "schema_name": "software-satellite-evidence-trace",
            "schema_version": 1,
            "workspace_id": _clean_text(resolved_graph.get("workspace_id")) or workspace_id,
            "event_id": event_id,
            "found": False,
            "message": "Event was not present in the derived graph.",
        }
    node_by_id = {node["node_id"]: node for node in nodes if _clean_text(node.get("node_id")) is not None}
    related_edges = [
        edge
        for edge in edges
        if edge.get("from_node_id") == event_node.get("node_id") or edge.get("to_node_id") == event_node.get("node_id")
    ]
    related_nodes = []
    for edge in related_edges:
        other_id = edge.get("to_node_id") if edge.get("from_node_id") == event_node.get("node_id") else edge.get("from_node_id")
        other = node_by_id.get(other_id)
        if other is not None:
            related_nodes.append(other)
    support = _mapping_dict(_mapping_dict(event_node.get("metadata")).get("support_kernel"))
    blockers = _string_list(support.get("blockers"))
    warnings = _string_list(support.get("warnings"))
    return {
        "schema_name": "software-satellite-evidence-trace",
        "schema_version": 1,
        "workspace_id": _clean_text(resolved_graph.get("workspace_id")) or workspace_id,
        "event_id": event_id,
        "found": True,
        "event_node": event_node,
        "support": {
            "support_class": _clean_text(event_node.get("support_class")) or "unknown",
            "can_support_decision": bool(event_node.get("can_support_decision")),
            "polarity": _clean_text(event_node.get("polarity")) or "unknown",
            "blockers": blockers,
            "warnings": warnings,
        },
        "why_blocked": blockers if why_blocked else [],
        "related_edges": sorted(related_edges, key=lambda edge: (str(edge.get("relation_kind") or ""), str(edge.get("edge_id") or ""))),
        "related_nodes": sorted(related_nodes, key=lambda node: (str(node.get("node_kind") or ""), str(node.get("source_id") or ""))),
    }


def format_evidence_trace_markdown(trace: Mapping[str, Any]) -> str:
    event_id = _clean_text(trace.get("event_id")) or "unknown"
    if not trace.get("found"):
        return f"# Evidence Trace\n\nEvent: {event_id}\n\n{trace.get('message') or 'Not found.'}\n"
    support = _mapping_dict(trace.get("support"))
    lines = [
        "# Evidence Trace",
        "",
        f"Event: {event_id}",
        f"Support class: {_clean_text(support.get('support_class')) or 'unknown'}",
        f"Can support decision: {'yes' if support.get('can_support_decision') else 'no'}",
        f"Polarity: {_clean_text(support.get('polarity')) or 'unknown'}",
        f"Blockers: {', '.join(_string_list(support.get('blockers'))) or 'none'}",
        f"Warnings: {', '.join(_string_list(support.get('warnings'))) or 'none'}",
    ]
    why_blocked = _string_list(trace.get("why_blocked"))
    if why_blocked:
        lines.extend(("", "## Why Blocked", ""))
        for blocker in why_blocked:
            lines.append(f"- {blocker}")
    related_edges = [edge for edge in trace.get("related_edges") or [] if isinstance(edge, Mapping)]
    related_nodes = {
        node.get("node_id"): node
        for node in trace.get("related_nodes") or []
        if isinstance(node, Mapping)
    }
    if related_edges:
        lines.extend(("", "## Relations", ""))
        event_node_id = _clean_text(_mapping_dict(trace.get("event_node")).get("node_id"))
        for edge in related_edges[:20]:
            from_node_id = edge.get("from_node_id")
            to_node_id = edge.get("to_node_id")
            if event_node_id is not None and from_node_id == event_node_id:
                other_id = to_node_id
            elif event_node_id is not None and to_node_id == event_node_id:
                other_id = from_node_id
            else:
                other_id = to_node_id
            fallback_id = from_node_id if other_id == to_node_id else to_node_id
            other = _mapping_dict(related_nodes.get(other_id)) or _mapping_dict(related_nodes.get(fallback_id))
            support_class = _clean_text(edge.get("support_class"))
            support_suffix = f"; support={support_class}" if support_class else ""
            lines.append(
                f"- {edge.get('relation_kind')} {edge.get('causal_validity')} "
                f"{_node_label(other)}{support_suffix}: {edge.get('explanation') or ''}"
            )
    return "\n".join(lines) + "\n"


def _path_matches(candidate: str, requested: str) -> bool:
    normalized_candidate = _normalize_repo_path(candidate)
    normalized_requested = _normalize_repo_path(requested)
    if normalized_candidate is None or normalized_requested is None:
        return False
    return (
        normalized_candidate == normalized_requested
        or normalized_candidate.endswith("/" + normalized_requested)
        or normalized_requested.endswith("/" + normalized_candidate)
    )


def build_evidence_impact_report(
    path: str | Path,
    *,
    graph: Mapping[str, Any] | None = None,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
) -> dict[str, Any]:
    requested = str(path)
    resolved_graph = dict(graph) if isinstance(graph, Mapping) else build_evidence_graph(root=root, workspace_id=workspace_id)
    nodes = [dict(node) for node in resolved_graph.get("nodes") or [] if isinstance(node, Mapping)]
    edges = [dict(edge) for edge in resolved_graph.get("edges") or [] if isinstance(edge, Mapping)]
    affected_events: list[dict[str, Any]] = []
    affected_node_ids: set[str] = set()
    for node in nodes:
        if node.get("node_kind") != "event":
            continue
        metadata = _mapping_dict(node.get("metadata"))
        candidates = [
            *_string_list(metadata.get("changed_files")),
            *_string_list(metadata.get("source_paths")),
            *_string_list(_mapping_dict(metadata.get("target_identity")).get("changed_files")),
            *_string_list(_mapping_dict(metadata.get("target_identity")).get("source_paths")),
        ]
        if any(_path_matches(candidate, requested) for candidate in candidates):
            affected_node_ids.add(str(node.get("node_id")))
            affected_events.append(
                {
                    "event_id": _clean_text(node.get("source_id")),
                    "support_class": _clean_text(node.get("support_class")),
                    "can_support_decision": bool(node.get("can_support_decision")),
                    "polarity": _clean_text(node.get("polarity")),
                    "target_fingerprint": _clean_text(node.get("target_fingerprint")),
                    "changed_files": _string_list(metadata.get("changed_files")),
                }
            )
    related_edges = [
        edge
        for edge in edges
        if edge.get("from_node_id") in affected_node_ids or edge.get("to_node_id") in affected_node_ids
    ]
    return {
        "schema_name": "software-satellite-evidence-impact-report",
        "schema_version": 1,
        "workspace_id": _clean_text(resolved_graph.get("workspace_id")) or workspace_id,
        "path": requested,
        "affected_event_count": len(affected_events),
        "affected_events": sorted(affected_events, key=lambda item: str(item.get("event_id") or "")),
        "related_edge_count": len(related_edges),
        "related_edges": sorted(related_edges, key=lambda edge: (str(edge.get("relation_kind") or ""), str(edge.get("edge_id") or "")))[:50],
    }


def format_evidence_impact_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Evidence Impact",
        "",
        f"Path: {_clean_text(report.get('path')) or 'n/a'}",
        f"Affected events: {int(report.get('affected_event_count') or 0)}",
    ]
    events = [item for item in report.get("affected_events") or [] if isinstance(item, Mapping)]
    if events:
        lines.extend(("", "## Events", ""))
        for item in events:
            can_support = "yes" if item.get("can_support_decision") else "no"
            changed = ", ".join(_string_list(item.get("changed_files"))) or "n/a"
            lines.append(
                f"- {item.get('event_id')} support={item.get('support_class')} "
                f"can_support={can_support} files={changed}"
            )
    edges = [item for item in report.get("related_edges") or [] if isinstance(item, Mapping)]
    if edges:
        lines.extend(("", "## Related Relations", ""))
        for edge in edges[:12]:
            lines.append(f"- {edge.get('relation_kind')} {edge.get('causal_validity')}: {edge.get('explanation') or ''}")
    return "\n".join(lines) + "\n"
