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
SUPPORT_POLICY_SCHEMA_NAME = "software-satellite-evidence-support-policy-registry"
SUPPORT_POLICY_REPORT_SCHEMA_NAME = "software-satellite-evidence-support-policy-report"
SUPPORT_POLICY_SCHEMA_VERSION = 1
SUPPORT_POLICY_ID = "evidence_support_v1"
DEFAULT_SUPPORT_POLICY_REGISTRY_PATH = Path("configs/evidence_support_policies/v1.json")

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
SUPPORT_BLOCKER_REASONS = {
    "binary_source_refused",
    "contradictory",
    "current_review_subject",
    "event_not_found",
    "future_evidence",
    "manual_pin_diagnostic",
    "missing_negative_or_risk_signal",
    "missing_positive_signal",
    "missing_source",
    "missing_verification_signal",
    "missing_vault_object",
    "modified_source",
    "noncanonical_vault_object_path",
    "not_artifact_ref",
    "not_captured",
    "outside_workspace",
    "oversize_source",
    "symlink_refused",
    "unverified_agent_claim",
    "vault_checksum_mismatch",
    "vault_object_outside_vault",
    "weak_match",
}
SUPPORT_WARNING_REASONS = {
    "manual_pin_diagnostic",
    "original_source_modified_after_capture",
    "verified_git_ref_without_vault_object",
}
SUPPORT_DECISION_REQUIREMENTS = {
    "valid_source_ref",
    "prior_to_review",
    "not_current_subject",
    "not_weak_match",
    "not_contradictory",
    "not_unverified_agent_claim",
    "positive_signal_required",
    "negative_or_risk_signal_required",
}
SUPPORT_CLASS_POLICY = {
    "source_linked_prior": {
        "can_support_decision": True,
        "allowed_decision_polarities": ["positive"],
        "default_blockers": [],
    },
    "negative_prior": {
        "can_support_decision": True,
        "allowed_decision_polarities": ["negative", "risk"],
        "default_blockers": [],
    },
    "current_review_subject": {
        "can_support_decision": False,
        "allowed_decision_polarities": [],
        "default_blockers": ["current_review_subject"],
    },
    "future_evidence": {
        "can_support_decision": False,
        "allowed_decision_polarities": [],
        "default_blockers": ["future_evidence"],
    },
    "missing_source": {
        "can_support_decision": False,
        "allowed_decision_polarities": [],
        "default_blockers": [
            "binary_source_refused",
            "missing_source",
            "missing_vault_object",
            "noncanonical_vault_object_path",
            "not_artifact_ref",
            "not_captured",
            "outside_workspace",
            "oversize_source",
            "symlink_refused",
            "vault_object_outside_vault",
        ],
    },
    "modified_source": {
        "can_support_decision": False,
        "allowed_decision_polarities": [],
        "default_blockers": ["modified_source", "vault_checksum_mismatch"],
    },
    "weak_match": {
        "can_support_decision": False,
        "allowed_decision_polarities": [],
        "default_blockers": ["weak_match"],
    },
    "contradictory": {
        "can_support_decision": False,
        "allowed_decision_polarities": [],
        "default_blockers": ["contradictory"],
    },
    "manual_pin_diagnostic": {
        "can_support_decision": False,
        "allowed_decision_polarities": [],
        "default_blockers": ["manual_pin_diagnostic"],
    },
    "unverified_agent_claim": {
        "can_support_decision": False,
        "allowed_decision_polarities": [],
        "default_blockers": ["unverified_agent_claim"],
    },
    "unknown": {
        "can_support_decision": False,
        "allowed_decision_polarities": [],
        "default_blockers": [
            "event_not_found",
            "missing_negative_or_risk_signal",
            "missing_positive_signal",
            "missing_verification_signal",
        ],
    },
}


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def _workspace_path_text(path: Path, *, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except (OSError, RuntimeError, ValueError):
        return str(path)


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


def _policy_string_values(
    issues: list[dict[str, Any]],
    value: Any,
    *,
    path: str,
) -> list[str]:
    if not isinstance(value, list):
        issues.append(_policy_issue(path, "Expected a list of strings.", actual=type(value).__name__))
        return []

    values: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for index, item in enumerate(value):
        text = _clean_text(item)
        if text is None:
            issues.append(_policy_issue(f"{path}[{index}]", "Expected a non-empty string."))
            continue
        if text in seen:
            duplicates.add(text)
        seen.add(text)
        values.append(text)
    if duplicates:
        issues.append(_policy_issue(path, "Values must be unique.", actual=sorted(duplicates)))
    return values


def _policy_issue(path: str, message: str, *, actual: Any = None) -> dict[str, Any]:
    issue: dict[str, Any] = {"path": path, "message": message}
    if actual is not None:
        issue["actual"] = actual
    return issue


def _append_unknown_key_issues(
    issues: list[dict[str, Any]],
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    path: str,
) -> None:
    for key in sorted(str(item) for item in value.keys() if str(item) not in allowed):
        issues.append(_policy_issue(f"{path}.{key}", "Unknown support policy registry field."))


def support_policy_registry_path(root: Path | None = None) -> Path:
    return _resolve_root(root) / DEFAULT_SUPPORT_POLICY_REGISTRY_PATH


def _resolve_policy_path(policy_path: str | Path | None, *, root: Path) -> Path:
    candidate = Path(policy_path).expanduser() if policy_path is not None else DEFAULT_SUPPORT_POLICY_REGISTRY_PATH
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def validate_support_policy_registry(policy: Any) -> list[dict[str, Any]]:
    if not isinstance(policy, Mapping):
        return [_policy_issue("$", "Support policy registry must be a JSON object.", actual=type(policy).__name__)]

    issues: list[dict[str, Any]] = []
    _append_unknown_key_issues(
        issues,
        policy,
        allowed={
            "schema_name",
            "schema_version",
            "policy_id",
            "support_kernel_schema",
            "support_polarities",
            "decision_requirements",
            "support_classes",
            "blocker_reasons",
            "warning_reasons",
            "signals",
            "default_paths",
        },
        path="$",
    )
    if policy.get("schema_name") != SUPPORT_POLICY_SCHEMA_NAME:
        issues.append(
            _policy_issue(
                "$.schema_name",
                f"Expected schema_name {SUPPORT_POLICY_SCHEMA_NAME}.",
                actual=policy.get("schema_name"),
            )
        )
    if policy.get("schema_version") != SUPPORT_POLICY_SCHEMA_VERSION:
        issues.append(
            _policy_issue(
                "$.schema_version",
                "Unsupported support policy registry schema version.",
                actual=policy.get("schema_version"),
            )
        )
    if policy.get("policy_id") != SUPPORT_POLICY_ID:
        issues.append(
            _policy_issue(
                "$.policy_id",
                f"Expected policy_id {SUPPORT_POLICY_ID}.",
                actual=policy.get("policy_id"),
            )
        )
    if policy.get("support_kernel_schema") != EVIDENCE_SUPPORT_SCHEMA_NAME:
        issues.append(
            _policy_issue(
                "$.support_kernel_schema",
                f"Expected support kernel schema {EVIDENCE_SUPPORT_SCHEMA_NAME}.",
                actual=policy.get("support_kernel_schema"),
            )
        )

    polarity_set = set(_policy_string_values(issues, policy.get("support_polarities"), path="$.support_polarities"))
    if polarity_set != SUPPORT_POLARITIES:
        issues.append(
            _policy_issue(
                "$.support_polarities",
                "Support polarities must mirror the runtime support kernel.",
                actual=sorted(polarity_set),
            )
        )

    requirement_ids: set[str] = set()
    seen_requirement_ids: set[str] = set()
    requirements = policy.get("decision_requirements")
    if not isinstance(requirements, list):
        issues.append(_policy_issue("$.decision_requirements", "Decision requirements must be a list."))
    else:
        for index, item in enumerate(requirements):
            if not isinstance(item, Mapping):
                issues.append(
                    _policy_issue(
                        f"$.decision_requirements[{index}]",
                        "Decision requirement must be an object.",
                    )
                )
                continue
            _append_unknown_key_issues(
                issues,
                item,
                allowed={"requirement_id", "summary"},
                path=f"$.decision_requirements[{index}]",
            )
            requirement_id = _clean_text(item.get("requirement_id"))
            if requirement_id is None:
                issues.append(
                    _policy_issue(
                        f"$.decision_requirements[{index}].requirement_id",
                        "Decision requirement id is required.",
                    )
                )
                continue
            if requirement_id in seen_requirement_ids:
                issues.append(
                    _policy_issue(
                        f"$.decision_requirements[{index}].requirement_id",
                        "Duplicate decision requirement id.",
                        actual=requirement_id,
                    )
                )
            seen_requirement_ids.add(requirement_id)
            requirement_ids.add(requirement_id)
            if not _clean_text(item.get("summary")):
                issues.append(
                    _policy_issue(
                        f"$.decision_requirements[{index}].summary",
                        "Decision requirement summary is required.",
                    )
                )
    if requirement_ids != SUPPORT_DECISION_REQUIREMENTS:
        issues.append(
            _policy_issue(
                "$.decision_requirements",
                "Decision requirement ids must mirror the runtime support kernel.",
                actual=sorted(requirement_ids),
            )
        )

    class_entries: dict[str, Mapping[str, Any]] = {}
    classes = policy.get("support_classes")
    if not isinstance(classes, list):
        issues.append(_policy_issue("$.support_classes", "Support classes must be a list."))
    else:
        for index, item in enumerate(classes):
            if not isinstance(item, Mapping):
                issues.append(_policy_issue(f"$.support_classes[{index}]", "Support class policy must be an object."))
                continue
            _append_unknown_key_issues(
                issues,
                item,
                allowed={
                    "support_class",
                    "can_support_decision",
                    "allowed_decision_polarities",
                    "default_blockers",
                    "summary",
                },
                path=f"$.support_classes[{index}]",
            )
            support_class = _clean_text(item.get("support_class"))
            if support_class is None:
                issues.append(_policy_issue(f"$.support_classes[{index}].support_class", "Support class is required."))
                continue
            if support_class in class_entries:
                issues.append(
                    _policy_issue(
                        f"$.support_classes[{index}].support_class",
                        "Duplicate support class.",
                        actual=support_class,
                    )
                )
                continue
            class_entries[support_class] = item
            if support_class not in SUPPORT_CLASSES:
                issues.append(
                    _policy_issue(
                        f"$.support_classes[{index}].support_class",
                        "Unknown support class.",
                        actual=support_class,
                    )
                )
                continue
            expected = SUPPORT_CLASS_POLICY[support_class]
            if item.get("can_support_decision") is not expected["can_support_decision"]:
                issues.append(
                    _policy_issue(
                        f"$.support_classes[{index}].can_support_decision",
                        "Support class decision capability must mirror the runtime support kernel.",
                        actual=item.get("can_support_decision"),
                    )
                )
            allowed = _policy_string_values(
                issues,
                item.get("allowed_decision_polarities"),
                path=f"$.support_classes[{index}].allowed_decision_polarities",
            )
            if set(allowed) != set(expected["allowed_decision_polarities"]):
                issues.append(
                    _policy_issue(
                        f"$.support_classes[{index}].allowed_decision_polarities",
                        "Allowed polarities must mirror the runtime support kernel.",
                        actual=allowed,
                    )
                )
            unknown_allowed = set(allowed) - SUPPORT_POLARITIES
            if unknown_allowed:
                issues.append(
                    _policy_issue(
                        f"$.support_classes[{index}].allowed_decision_polarities",
                        "Allowed polarities include values outside the support polarity set.",
                        actual=sorted(unknown_allowed),
                    )
                )
            blockers = _policy_string_values(
                issues,
                item.get("default_blockers"),
                path=f"$.support_classes[{index}].default_blockers",
            )
            if set(blockers) != set(expected["default_blockers"]):
                issues.append(
                    _policy_issue(
                        f"$.support_classes[{index}].default_blockers",
                        "Default blockers must mirror the runtime support kernel.",
                        actual=blockers,
                    )
                )
            unknown_blockers = set(blockers) - SUPPORT_BLOCKER_REASONS
            if unknown_blockers:
                issues.append(
                    _policy_issue(
                        f"$.support_classes[{index}].default_blockers",
                        "Default blockers include unknown reasons.",
                        actual=sorted(unknown_blockers),
                    )
                )
            if not _clean_text(item.get("summary")):
                issues.append(
                    _policy_issue(
                        f"$.support_classes[{index}].summary",
                        "Support class summary is required.",
                    )
                )
    if set(class_entries) != SUPPORT_CLASSES:
        issues.append(
            _policy_issue(
                "$.support_classes",
                "Support classes must mirror the runtime support kernel.",
                actual=sorted(class_entries),
            )
        )

    blocker_set = set(_policy_string_values(issues, policy.get("blocker_reasons"), path="$.blocker_reasons"))
    if blocker_set != SUPPORT_BLOCKER_REASONS:
        issues.append(
            _policy_issue(
                "$.blocker_reasons",
                "Blocker reasons must mirror the runtime support kernel.",
                actual=sorted(blocker_set),
            )
        )
    warning_set = set(_policy_string_values(issues, policy.get("warning_reasons"), path="$.warning_reasons"))
    if warning_set != SUPPORT_WARNING_REASONS:
        issues.append(
            _policy_issue(
                "$.warning_reasons",
                "Warning reasons must mirror the runtime support kernel.",
                actual=sorted(warning_set),
            )
        )

    signals = _mapping_dict(policy.get("signals"))
    _append_unknown_key_issues(
        issues,
        signals,
        allowed={
            "positive_statuses",
            "negative_statuses",
            "unresolved_statuses",
            "positive_evidence_types",
            "failure_evidence_types",
        },
        path="$.signals",
    )
    expected_signal_sets = {
        "positive_statuses": POSITIVE_STATUSES,
        "negative_statuses": NEGATIVE_STATUSES,
        "unresolved_statuses": UNRESOLVED_STATUSES,
        "positive_evidence_types": POSITIVE_EVIDENCE_TYPES,
        "failure_evidence_types": FAILURE_EVIDENCE_TYPES,
    }
    for key, expected_values in expected_signal_sets.items():
        actual_values = set(_policy_string_values(issues, signals.get(key), path=f"$.signals.{key}"))
        if actual_values != expected_values:
            issues.append(
                _policy_issue(
                    f"$.signals.{key}",
                    "Signal values must mirror the runtime support kernel.",
                    actual=sorted(actual_values),
                )
            )

    default_paths = _mapping_dict(policy.get("default_paths"))
    _append_unknown_key_issues(
        issues,
        default_paths,
        allowed={"network_calls", "private_docs_required", "api_key_required", "trainable_export_artifacts"},
        path="$.default_paths",
    )
    for key in ("network_calls", "private_docs_required", "api_key_required", "trainable_export_artifacts"):
        if default_paths.get(key) is not False:
            issues.append(
                _policy_issue(
                    f"$.default_paths.{key}",
                    "Default policy path must remain disabled.",
                    actual=default_paths.get(key),
                )
            )

    return issues


def load_support_policy_registry(
    policy_path: str | Path | None = None,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    resolved_path = _resolve_policy_path(policy_path, root=resolved_root)
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid support policy registry JSON in `{resolved_path}`: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read support policy registry `{resolved_path}`: {exc}") from exc
    issues = validate_support_policy_registry(payload)
    if issues:
        issue_text = "; ".join(f"{issue['path']}: {issue['message']}" for issue in issues[:5])
        if len(issues) > 5:
            issue_text += f"; ... {len(issues) - 5} more"
        raise ValueError(f"Invalid support policy registry `{resolved_path}`: {issue_text}")
    return copy.deepcopy(dict(payload))


def build_support_policy_report(
    policy_path: str | Path | None = None,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    resolved_path = _resolve_policy_path(policy_path, root=resolved_root)
    registry = load_support_policy_registry(resolved_path, root=resolved_root)
    support_classes = []
    for item in registry.get("support_classes") or []:
        if not isinstance(item, Mapping):
            continue
        allowed = _string_list(item.get("allowed_decision_polarities"))
        support_classes.append(
            {
                "support_class": _clean_text(item.get("support_class")) or "unknown",
                "can_support_decision": bool(item.get("can_support_decision")),
                "allowed_decision_polarities": allowed,
                "decision_use": ", ".join(allowed) if allowed else "diagnostic_only",
                "default_blockers": _string_list(item.get("default_blockers")),
                "summary": _clean_text(item.get("summary")) or "",
            }
        )
    return {
        "schema_name": SUPPORT_POLICY_REPORT_SCHEMA_NAME,
        "schema_version": SUPPORT_POLICY_SCHEMA_VERSION,
        "policy_id": registry.get("policy_id"),
        "registry_path": _workspace_path_text(resolved_path, root=resolved_root),
        "registry_valid": True,
        "validation_issues": [],
        "support_kernel_schema": registry.get("support_kernel_schema"),
        "support_polarities": _string_list(registry.get("support_polarities")),
        "decision_requirements": copy.deepcopy(list(registry.get("decision_requirements") or [])),
        "support_classes": support_classes,
        "blocker_reasons": _string_list(registry.get("blocker_reasons")),
        "warning_reasons": _string_list(registry.get("warning_reasons")),
        "default_paths": copy.deepcopy(_mapping_dict(registry.get("default_paths"))),
    }


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


def format_support_policy_report(report: Mapping[str, Any]) -> str:
    lines = [
        "Evidence support policy",
        f"Policy: {_clean_text(report.get('policy_id')) or SUPPORT_POLICY_ID}",
        f"Registry: {_clean_text(report.get('registry_path')) or str(DEFAULT_SUPPORT_POLICY_REGISTRY_PATH)}",
        f"Registry valid: {'yes' if report.get('registry_valid') else 'no'}",
        "",
        "Support classes:",
    ]
    for item in report.get("support_classes") or []:
        if not isinstance(item, Mapping):
            continue
        support_class = _clean_text(item.get("support_class")) or "unknown"
        can_support = "yes" if item.get("can_support_decision") else "no"
        allowed = ", ".join(_string_list(item.get("allowed_decision_polarities"))) or "none"
        blockers = ", ".join(_string_list(item.get("default_blockers"))) or "none"
        lines.append(
            f"- {support_class}: can_support={can_support}; "
            f"allowed={allowed}; blockers={blockers}"
        )
    requirements = [
        item
        for item in report.get("decision_requirements") or []
        if isinstance(item, Mapping) and _clean_text(item.get("requirement_id"))
    ]
    if requirements:
        lines.extend(["", "Decision requirements:"])
        for item in requirements:
            lines.append(
                f"- {_clean_text(item.get('requirement_id'))}: "
                f"{_clean_text(item.get('summary')) or ''}"
            )
    return "\n".join(lines)
