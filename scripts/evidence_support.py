#!/usr/bin/env python3
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from artifact_vault import (
    ARTIFACT_REF_SCHEMA_NAME,
    artifact_ref_object_verified,
    load_artifact_ref,
    resolve_vault_object_path,
)
from gemma_runtime import repo_root, timestamp_utc
from memory_index import MemoryIndex, rebuild_memory_index
from workspace_state import DEFAULT_WORKSPACE_ID


EVIDENCE_SUPPORT_SCHEMA_NAME = "software-satellite-evidence-support-result"
EVIDENCE_SUPPORT_SCHEMA_VERSION = 1

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
SUPPORT_POLARITIES = {"positive", "negative", "risk", "diagnostic", "none"}

POSITIVE_STATUSES = {"accepted", "accept", "pass", "passed", "resolved", "winner_selected"}
NEGATIVE_STATUSES = {"failed", "quality_fail", "blocked", "error", "rejected", "reject", "needs_fix"}
UNRESOLVED_STATUSES = {"needs_review", "needs_more_evidence", "unresolved", "pending"}
FAILURE_EVIDENCE_TYPES = {"test_fail", "rejected", "failure", "blocker", "unresolved"}
POSITIVE_EVIDENCE_TYPES = {"test_pass", "accepted", "human_acceptance", "verification_pass"}


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _mapping_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any, *, lowercase: bool = False) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _clean_text(item)
        if text is None:
            continue
        if lowercase:
            text = text.lower()
        if text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        if text is None or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


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


def _file_sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _resolve_path(path_text: str | None, *, root: Path) -> Path | None:
    text = _clean_text(path_text)
    if text is None:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return path.absolute()


def _run_git(root: Path, args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _verified_git_ref(ref: Mapping[str, Any], *, root: Path) -> bool:
    git = _mapping_dict(ref.get("git"))
    commit = _clean_text(git.get("commit"))
    blob_sha = _clean_text(git.get("blob_sha"))
    repo_relative_path = _clean_text(ref.get("repo_relative_path"))
    if commit is None or blob_sha is None or repo_relative_path is None:
        return False
    if _run_git(root, ["rev-parse", "--is-inside-work-tree"]) != "true":
        return False
    resolved_blob = _run_git(root, ["rev-parse", f"{commit}:{repo_relative_path}"])
    return bool(resolved_blob and resolved_blob.lower() == blob_sha.lower())


def _load_ref_if_needed(value: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    if value.get("schema_name") == ARTIFACT_REF_SCHEMA_NAME:
        return copy.deepcopy(dict(value))
    artifact_id = _clean_text(value.get("artifact_id"))
    if artifact_id is None:
        return copy.deepcopy(dict(value))
    try:
        return load_artifact_ref(artifact_id, root=root)
    except (OSError, ValueError, json.JSONDecodeError):
        return copy.deepcopy(dict(value))


def _append_ref_values(values: list[dict[str, Any]], raw: Any, *, root: Path) -> None:
    if isinstance(raw, Mapping):
        candidate = _load_ref_if_needed(raw, root=root)
        if candidate.get("schema_name") == ARTIFACT_REF_SCHEMA_NAME or candidate.get("artifact_id"):
            values.append(candidate)
        return
    if isinstance(raw, list):
        for item in raw:
            _append_ref_values(values, item, root=root)


def artifact_refs_from_event(event: Mapping[str, Any], *, root: Path | None = None) -> list[dict[str, Any]]:
    resolved_root = _resolve_root(root)
    refs: list[dict[str, Any]] = []
    source_refs = _mapping_dict(event.get("source_refs"))
    content = _mapping_dict(event.get("content"))
    options = _mapping_dict(content.get("options"))
    for raw in (
        event.get("artifact_vault_ref"),
        event.get("artifact_vault_refs"),
        event.get("artifact_refs"),
        source_refs.get("artifact_vault_ref"),
        source_refs.get("artifact_vault_refs"),
        source_refs.get("artifact_refs"),
        options.get("artifact_vault_ref"),
        options.get("artifact_vault_refs"),
    ):
        _append_ref_values(refs, raw, root=resolved_root)

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ref in refs:
        key = _clean_text(ref.get("artifact_id")) or json.dumps(ref, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped


def _artifact_source_modified(ref: Mapping[str, Any], *, root: Path) -> bool:
    sha256 = _clean_text(ref.get("sha256"))
    if sha256 is None:
        return False
    path = _resolve_path(_clean_text(ref.get("original_path")), root=root)
    if path is None or not path.is_file():
        return False
    current = _file_sha256(path)
    return bool(current and current.lower() != sha256.lower())


def _valid_source_refs(
    refs: list[dict[str, Any]],
    *,
    root: Path,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    valid: list[dict[str, Any]] = []
    blockers: list[str] = []
    warnings: list[str] = []
    for ref in refs:
        verified, reason = artifact_ref_object_verified(ref, root=root)
        git_verified = _verified_git_ref(ref, root=root)
        if verified or git_verified:
            valid.append(ref)
            if git_verified and not verified:
                warnings.append("verified_git_ref_without_vault_object")
            if _artifact_source_modified(ref, root=root):
                warnings.append("original_source_modified_after_capture")
            continue
        source_state = _clean_text(ref.get("source_state"))
        capture_state = _clean_text(ref.get("capture_state"))
        if source_state == "missing":
            blockers.append("missing_source")
        elif source_state == "binary_refused":
            blockers.append("binary_source_refused")
        elif source_state == "oversize":
            blockers.append("oversize_source")
        elif source_state == "outside_workspace":
            blockers.append("outside_workspace")
        elif source_state == "symlink_refused":
            blockers.append("symlink_refused")
        elif reason == "vault_checksum_mismatch":
            blockers.append("modified_source")
        elif capture_state != "captured":
            blockers.append("missing_source")
        elif reason:
            blockers.append(reason)
    return valid, _dedupe(blockers), _dedupe(warnings)


def _event_from_index(
    event_id: str,
    *,
    root: Path,
    workspace_id: str,
) -> dict[str, Any] | None:
    summary = rebuild_memory_index(root=root, workspace_id=workspace_id)
    index = MemoryIndex(Path(summary["index_path"]))
    row = index.get_event(event_id)
    if not row:
        return None
    try:
        payload = json.loads(str(row.get("payload_json") or "{}"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _event_status(event: Mapping[str, Any]) -> str | None:
    outcome = _mapping_dict(event.get("outcome"))
    return _clean_text(outcome.get("status")) or _clean_text(event.get("status"))


def _quality_status(event: Mapping[str, Any]) -> str | None:
    outcome = _mapping_dict(event.get("outcome"))
    content = _mapping_dict(event.get("content"))
    options = _mapping_dict(content.get("options"))
    return _clean_text(outcome.get("quality_status")) or _clean_text(options.get("quality_status"))


def _execution_status(event: Mapping[str, Any]) -> str | None:
    outcome = _mapping_dict(event.get("outcome"))
    content = _mapping_dict(event.get("content"))
    options = _mapping_dict(content.get("options"))
    return _clean_text(outcome.get("execution_status")) or _clean_text(options.get("execution_status"))


def _event_terms(event: Mapping[str, Any]) -> set[str]:
    content = _mapping_dict(event.get("content"))
    options = _mapping_dict(content.get("options"))
    source_refs = _mapping_dict(event.get("source_refs"))
    terms = []
    terms.extend(_string_list(event.get("tags"), lowercase=True))
    terms.extend(_string_list(event.get("reasons"), lowercase=True))
    terms.extend(_string_list(content.get("notes"), lowercase=True))
    terms.extend(_string_list(options.get("reasons"), lowercase=True))
    terms.extend(_string_list(source_refs.get("reasons"), lowercase=True))
    return {term.replace("-", "_") for term in terms}


def _evidence_types(event: Mapping[str, Any]) -> set[str]:
    content = _mapping_dict(event.get("content"))
    options = _mapping_dict(content.get("options"))
    terms = set(_string_list(event.get("evidence_types"), lowercase=True))
    terms.update(_string_list(options.get("evidence_types"), lowercase=True))
    terms.update(_string_list(event.get("support_evidence_types"), lowercase=True))
    return {term.replace("-", "_") for term in terms}


def _has_quality_check_signal(event: Mapping[str, Any]) -> bool:
    content = _mapping_dict(event.get("content"))
    options = _mapping_dict(content.get("options"))
    checks = options.get("quality_checks")
    if not isinstance(checks, list):
        return False
    return any(isinstance(check, Mapping) and isinstance(check.get("pass"), bool) for check in checks)


def _has_human_verdict(event: Mapping[str, Any]) -> bool:
    content = _mapping_dict(event.get("content"))
    options = _mapping_dict(content.get("options"))
    terms = _event_terms(event)
    return (
        "human_verdict" in terms
        or "human_gate" in terms
        or bool(_mapping_dict(options.get("human_verdict")))
        or bool(_mapping_dict(event.get("human_verdict")))
    )


def _has_positive_signal(event: Mapping[str, Any]) -> bool:
    status = (_event_status(event) or "").lower()
    quality = (_quality_status(event) or "").lower()
    execution = (_execution_status(event) or "").lower()
    evidence_types = _evidence_types(event)
    terms = _event_terms(event)
    return (
        status in POSITIVE_STATUSES
        or quality == "pass"
        or execution in {"pass", "passed"}
        or bool(POSITIVE_EVIDENCE_TYPES & evidence_types)
        or "accepted_signal" in terms
        or "positive_signal" in terms
    )


def _has_negative_or_risk_signal(event: Mapping[str, Any]) -> bool:
    status = (_event_status(event) or "").lower()
    quality = (_quality_status(event) or "").lower()
    execution = (_execution_status(event) or "").lower()
    evidence_types = _evidence_types(event)
    terms = _event_terms(event)
    return (
        status in NEGATIVE_STATUSES
        or status in UNRESOLVED_STATUSES
        or quality == "fail"
        or execution in NEGATIVE_STATUSES
        or bool(FAILURE_EVIDENCE_TYPES & evidence_types)
        or "negative_signal" in terms
        or "failure_signal" in terms
        or "risk_signal" in terms
        or "blocker" in terms
    )


def _is_unresolved(event: Mapping[str, Any]) -> bool:
    status = (_event_status(event) or "").lower()
    return status in UNRESOLVED_STATUSES or "unresolved" in _event_terms(event)


def _is_weak_match(event: Mapping[str, Any]) -> bool:
    content = _mapping_dict(event.get("content"))
    options = _mapping_dict(content.get("options"))
    support = _mapping_dict(event.get("support_context"))
    terms = _event_terms(event)
    return (
        "weak_match" in terms
        or "text_only_match" in terms
        or _clean_text(options.get("match_strength")) == "weak"
        or _clean_text(support.get("match_strength")) == "weak"
    )


def _is_manual_pin(event: Mapping[str, Any]) -> bool:
    terms = _event_terms(event)
    return "pinned" in terms or "manual_pin" in terms


def _is_contradictory(event: Mapping[str, Any]) -> bool:
    evidence_types = _evidence_types(event)
    status = (_event_status(event) or "").lower()
    terms = _event_terms(event)
    return (
        status == "mixed"
        or "contradictory" in terms
        or {"test_fail", "test_pass"}.issubset(evidence_types)
        or {"accepted", "rejected"}.issubset(evidence_types)
    )


def _is_agent_claim(event: Mapping[str, Any]) -> bool:
    session = _mapping_dict(event.get("session"))
    surface = _clean_text(session.get("surface"))
    event_kind = _clean_text(event.get("event_kind"))
    terms = _event_terms(event)
    return (
        surface == "agent_lane"
        or event_kind == "agent_task_run"
        or "agent_claim" in terms
        or "agent_transcript_claim" in terms
    )


def _is_unverified_agent_claim(event: Mapping[str, Any]) -> bool:
    if not _is_agent_claim(event):
        return False
    quality = (_quality_status(event) or "").lower()
    return not (
        quality in {"pass", "fail"}
        or _has_quality_check_signal(event)
        or _has_human_verdict(event)
        or bool(_evidence_types(event))
    )


def _path_fingerprints(value: Any, *, root: Path) -> set[str]:
    text = _clean_text(value)
    if text is None:
        return set()
    fingerprints = {text}
    resolved_path = _resolve_path(text, root=root)
    if resolved_path is not None:
        fingerprints.add(str(resolved_path))
        try:
            fingerprints.add(str(resolved_path.relative_to(root)))
        except ValueError:
            pass
    return fingerprints


def _event_fingerprints(event: Mapping[str, Any], refs: list[dict[str, Any]], *, root: Path) -> set[str]:
    fingerprints = {_clean_text(event.get("event_id")) or ""}
    source_refs = _mapping_dict(event.get("source_refs"))
    legacy_artifact_ref = _mapping_dict(source_refs.get("artifact_ref"))
    fingerprints.update(_path_fingerprints(legacy_artifact_ref.get("artifact_path"), root=root))
    for value in (legacy_artifact_ref.get("artifact_sha256"), legacy_artifact_ref.get("sha256")):
        text = _clean_text(value)
        if text:
            fingerprints.add(text)
    for ref in refs:
        fingerprints.update(_path_fingerprints(ref.get("original_path"), root=root))
        fingerprints.update(_path_fingerprints(ref.get("repo_relative_path"), root=root))
        for value in (ref.get("artifact_id"), ref.get("sha256"), _mapping_dict(ref.get("git")).get("blob_sha")):
            text = _clean_text(value)
            if text:
                fingerprints.add(text)
        object_path = resolve_vault_object_path(ref, root=root)
        if object_path is not None:
            fingerprints.add(str(object_path))
    return {value for value in fingerprints if value}


def _active_subject_matches(
    active_subject: str | None,
    *,
    event: Mapping[str, Any],
    refs: list[dict[str, Any]],
    root: Path,
) -> bool:
    subject = _clean_text(active_subject)
    if subject is None:
        return False
    subject_fingerprints = _path_fingerprints(subject, root=root)
    subject_fingerprints.add(subject)
    return bool(subject_fingerprints & _event_fingerprints(event, refs, root=root))


def _infer_requested_polarity(event: Mapping[str, Any]) -> str:
    if _has_negative_or_risk_signal(event) and not _has_positive_signal(event):
        return "risk"
    if _has_positive_signal(event):
        return "positive"
    return "none"


def _normalized_polarity(value: str | None, event: Mapping[str, Any]) -> str:
    normalized = (value or "").strip().lower().replace("-", "_")
    if normalized in SUPPORT_POLARITIES:
        return normalized
    return _infer_requested_polarity(event)


def _base_result(event_id: str, *, checked_at_utc: str | None) -> dict[str, Any]:
    return {
        "schema_name": EVIDENCE_SUPPORT_SCHEMA_NAME,
        "schema_version": EVIDENCE_SUPPORT_SCHEMA_VERSION,
        "event_id": event_id,
        "support_class": "unknown",
        "can_support_decision": False,
        "support_polarity": "none",
        "blockers": [],
        "warnings": [],
        "artifact_refs": [],
        "active_review_excluded": False,
        "checked_at_utc": checked_at_utc or timestamp_utc(),
    }


def build_evidence_support_result(
    event_id: str,
    *,
    event: Mapping[str, Any] | None = None,
    review_started_at: str | None = None,
    active_subject: str | None = None,
    requested_polarity: str | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    checked_at_utc: str | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    resolved_event = copy.deepcopy(dict(event)) if isinstance(event, Mapping) else _event_from_index(
        event_id,
        root=resolved_root,
        workspace_id=workspace_id,
    )
    result = _base_result(event_id, checked_at_utc=checked_at_utc)
    if resolved_event is None:
        result["blockers"] = ["event_not_found"]
        return result

    refs = artifact_refs_from_event(resolved_event, root=resolved_root)
    valid_refs, source_blockers, source_warnings = _valid_source_refs(refs, root=resolved_root)
    result["artifact_refs"] = sorted(
        _clean_text(ref.get("artifact_id")) or ""
        for ref in refs
        if _clean_text(ref.get("artifact_id"))
    )
    result["warnings"] = _dedupe(source_warnings)
    requested = _normalized_polarity(requested_polarity, resolved_event)

    review_start = _coerce_utc_datetime(review_started_at)
    recorded_at = _coerce_utc_datetime(resolved_event.get("recorded_at_utc"))
    if _active_subject_matches(active_subject, event=resolved_event, refs=refs, root=resolved_root):
        result.update(
            {
                "support_class": "current_review_subject",
                "support_polarity": "none",
                "active_review_excluded": True,
                "blockers": ["current_review_subject"],
            }
        )
        return result
    if review_start is not None:
        if recorded_at is None or recorded_at >= review_start:
            result.update(
                {
                    "support_class": "future_evidence",
                    "support_polarity": "none",
                    "blockers": ["future_evidence"],
                }
            )
            return result

    if not valid_refs:
        blockers = source_blockers or ["missing_source"]
        support_class = "modified_source" if "modified_source" in blockers or "vault_checksum_mismatch" in blockers else "missing_source"
        if _is_manual_pin(resolved_event):
            result["warnings"] = _dedupe([*result["warnings"], "manual_pin_diagnostic"])
        result.update(
            {
                "support_class": support_class,
                "support_polarity": "none",
                "blockers": _dedupe(blockers),
            }
        )
        return result

    if _is_manual_pin(resolved_event) and requested == "diagnostic":
        result.update(
            {
                "support_class": "manual_pin_diagnostic",
                "support_polarity": "diagnostic",
                "blockers": ["manual_pin_diagnostic"],
            }
        )
        return result
    if _is_weak_match(resolved_event):
        result.update(
            {
                "support_class": "weak_match",
                "support_polarity": "none",
                "blockers": ["weak_match"],
            }
        )
        return result
    if _is_unverified_agent_claim(resolved_event):
        result.update(
            {
                "support_class": "unverified_agent_claim",
                "support_polarity": "none",
                "blockers": ["unverified_agent_claim"],
            }
        )
        return result
    if _is_contradictory(resolved_event):
        result.update(
            {
                "support_class": "contradictory",
                "support_polarity": "none",
                "blockers": ["contradictory"],
            }
        )
        return result

    positive = _has_positive_signal(resolved_event)
    negative_or_risk = _has_negative_or_risk_signal(resolved_event)
    unresolved = _is_unresolved(resolved_event)
    if requested == "positive":
        if not positive:
            result.update(
                {
                    "support_class": "unknown",
                    "support_polarity": "none",
                    "blockers": ["missing_positive_signal"],
                }
            )
            return result
        result.update(
            {
                "support_class": "source_linked_prior",
                "can_support_decision": True,
                "support_polarity": "positive",
            }
        )
        return result

    if requested in {"negative", "risk"}:
        if not negative_or_risk:
            result.update(
                {
                    "support_class": "unknown",
                    "support_polarity": "none",
                    "blockers": ["missing_negative_or_risk_signal"],
                }
            )
            return result
        result.update(
            {
                "support_class": "negative_prior",
                "can_support_decision": True,
                "support_polarity": "risk" if unresolved or requested == "risk" else "negative",
            }
        )
        return result

    if positive:
        result.update(
            {
                "support_class": "source_linked_prior",
                "can_support_decision": True,
                "support_polarity": "positive",
            }
        )
        return result
    if negative_or_risk:
        result.update(
            {
                "support_class": "negative_prior",
                "can_support_decision": True,
                "support_polarity": "risk" if unresolved else "negative",
            }
        )
        return result

    result["blockers"] = ["missing_verification_signal"]
    return result


def format_evidence_support_result(result: Mapping[str, Any]) -> str:
    can_support = "yes" if result.get("can_support_decision") else "no"
    blockers = ", ".join(_string_list(result.get("blockers"))) or "none"
    warnings = ", ".join(_string_list(result.get("warnings"))) or "none"
    refs = ", ".join(_string_list(result.get("artifact_refs"))) or "none"
    return "\n".join(
        [
            "Evidence support",
            f"Event: {_clean_text(result.get('event_id')) or 'unknown'}",
            f"Class: {_clean_text(result.get('support_class')) or 'unknown'}",
            f"Can support decision: {can_support}",
            f"Polarity: {_clean_text(result.get('support_polarity')) or 'none'}",
            f"Blockers: {blockers}",
            f"Warnings: {warnings}",
            f"Artifact refs: {refs}",
        ]
    )
