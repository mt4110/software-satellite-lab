#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from agent_claims import (
    claim_counts,
    claim_evidence_types,
    claim_quality_status,
    extract_claims_from_transcript,
    log_indicates_tests_failed,
    log_indicates_tests_passed,
    normalize_declared_claims,
    verify_claims_against_artifacts,
    build_claim_software_work_event,
)
from artifact_vault import artifact_ref_object_verified, capture_artifact, resolve_vault_object_path
from evidence_support import build_evidence_support_result
from gemma_runtime import repo_root, timestamp_slug, timestamp_utc, write_json
from memory_index import rebuild_memory_index
from software_work_events import build_event_record
from workspace_state import DEFAULT_WORKSPACE_ID


AGENT_SESSION_BUNDLE_SCHEMA_NAME = "software-satellite-agent-session-bundle"
AGENT_SESSION_BUNDLE_SCHEMA_VERSION = 1
AGENT_SESSION_INTAKE_RESULT_SCHEMA_NAME = "software-satellite-agent-session-intake-result"
AGENT_SESSION_INTAKE_RESULT_SCHEMA_VERSION = 1

SUPPORTED_AGENT_LABELS = (
    "generic",
    "claude_code",
    "copilot",
    "aider",
    "openhands",
    "manual",
    "unknown",
)
AGENT_LABELS = set(SUPPORTED_AGENT_LABELS)

BUNDLE_ARTIFACT_KINDS = {
    "diff",
    "patch",
    "transcript",
    "test_log",
    "ci_log",
    "command_log",
    "review",
    "review_note",
    "human_verdict",
    "source_comparison",
    "unknown",
}
BUNDLE_KIND_TO_VAULT_KIND = {
    "diff": "patch",
    "patch": "patch",
    "transcript": "transcript",
    "test_log": "test_log",
    "ci_log": "ci_log",
    "command_log": "command_log",
    "review": "review_note",
    "review_note": "review_note",
    "human_verdict": "review_note",
    "source_comparison": "candidate_output",
    "unknown": "unknown",
}

DEFAULT_TRANSCRIPT_CLAIM_READ_CHARS = 64 * 1024
DEFAULT_MAX_CAPTURE_BYTES = 2 * 1024 * 1024
DEFAULT_REPORT_EXCERPT_CHARS = 1200
DEFAULT_VERIFICATION_LOG_READ_CHARS = 256 * 1024


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _mapping_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _stable_digest(value: Any, *, length: int = 16) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _sanitize_id_part(value: str | None, *, fallback: str) -> str:
    text = (value or "").strip().lower().replace("_", "-")
    cleaned = "".join(char if char.isalnum() or char == "-" else "-" for char in text).strip("-")
    return cleaned or fallback


def _normalize_agent_label(value: Any) -> str:
    text = (_clean_text(value) or "unknown").lower().replace("-", "_")
    return text if text in AGENT_LABELS else "unknown"


def _normalize_bundle_kind(value: Any) -> str:
    text = (_clean_text(value) or "unknown").lower().replace("-", "_")
    return text if text in BUNDLE_ARTIFACT_KINDS else "unknown"


def _resolve_path(path: str | Path, *, root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        return candidate.resolve()
    except (OSError, RuntimeError):
        return candidate.absolute()


def _workspace_relative(path: Path, *, root: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(root))
    except (OSError, RuntimeError, ValueError):
        return None


def _path_inside_root(path: Path, *, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _read_text_for_claims(path: str | Path, *, root: Path, max_chars: int) -> tuple[str | None, bool, str | None]:
    resolved = _resolve_path(path, root=root)
    if not _path_inside_root(resolved, root=root):
        return None, False, "outside_workspace"
    if resolved.is_symlink():
        return None, False, "symlink_refused"
    if not resolved.is_file():
        return None, False, "missing"
    try:
        with resolved.open("r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(max_chars + 1)
    except OSError:
        return None, False, "unreadable"
    capped = len(text) > max_chars
    return text[:max_chars], capped, None


def agent_session_intake_root(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return _resolve_root(root) / "artifacts" / "agent_session_intake" / workspace_id


def agent_session_intake_latest_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return agent_session_intake_root(workspace_id=workspace_id, root=root) / "latest.json"


def agent_session_intake_run_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return (
        agent_session_intake_root(workspace_id=workspace_id, root=root)
        / "runs"
        / f"{timestamp_slug()}-agent-session-intake.json"
    )


def _default_privacy() -> dict[str, Any]:
    return {
        "contains_private_code": True,
        "contains_user_text": True,
        "export_allowed": False,
    }


def _normalize_privacy(value: Any) -> dict[str, Any]:
    privacy = _default_privacy()
    if isinstance(value, Mapping):
        privacy.update(copy.deepcopy(dict(value)))
    privacy["contains_private_code"] = bool(privacy.get("contains_private_code"))
    privacy["contains_user_text"] = bool(privacy.get("contains_user_text"))
    privacy["export_allowed"] = bool(privacy.get("export_allowed"))
    return privacy


def _normalize_task(value: Any, *, fallback_title: str, fallback_goal: str | None = None) -> dict[str, Any]:
    task = copy.deepcopy(dict(value)) if isinstance(value, Mapping) else {}
    title = _clean_text(task.get("title")) or fallback_title
    goal = _clean_text(task.get("goal")) or fallback_goal or title
    task["title"] = title
    task["goal"] = goal
    return task


def _normalize_artifacts(value: Any) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for index, item in enumerate(_list_of_mappings(value)):
        path_text = _clean_text(item.get("path"))
        if path_text is None:
            continue
        bundle_kind = _normalize_bundle_kind(item.get("kind"))
        artifact = copy.deepcopy(item)
        artifact["kind"] = bundle_kind
        artifact["path"] = path_text
        artifact["vault_kind"] = BUNDLE_KIND_TO_VAULT_KIND.get(bundle_kind, "unknown")
        artifact["artifact_index"] = index
        artifacts.append(artifact)
    return artifacts


def validate_agent_session_bundle(bundle: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    if bundle.get("schema_name") != AGENT_SESSION_BUNDLE_SCHEMA_NAME:
        issues.append("schema_name must be software-satellite-agent-session-bundle")
    if bundle.get("schema_version") != AGENT_SESSION_BUNDLE_SCHEMA_VERSION:
        issues.append("schema_version must be 1")
    agent_label = _clean_text(bundle.get("agent_label"))
    normalized_agent_label = (agent_label or "").lower().replace("-", "_")
    if agent_label is None:
        issues.append("agent_label is required")
    elif normalized_agent_label not in AGENT_LABELS:
        issues.append("agent_label must be one of the supported metadata labels")
    if not _list_of_mappings(bundle.get("artifacts")):
        issues.append("artifacts must contain at least one local file reference")
    task = _mapping_dict(bundle.get("task"))
    if _clean_text(task.get("title")) is None:
        issues.append("task.title is required")
    privacy = _mapping_dict(bundle.get("privacy"))
    if not privacy:
        issues.append("privacy metadata is required")
    elif "export_allowed" not in privacy:
        issues.append("privacy.export_allowed is required")
    return issues


def normalize_agent_session_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    issues = validate_agent_session_bundle(bundle)
    if issues:
        raise ValueError("Invalid agent session bundle: " + "; ".join(issues))
    agent_label = _normalize_agent_label(bundle.get("agent_label"))
    normalized = copy.deepcopy(dict(bundle))
    normalized["schema_name"] = AGENT_SESSION_BUNDLE_SCHEMA_NAME
    normalized["schema_version"] = AGENT_SESSION_BUNDLE_SCHEMA_VERSION
    normalized["agent_label"] = agent_label
    normalized["session_started_at_utc"] = _clean_text(bundle.get("session_started_at_utc")) or timestamp_utc()
    normalized["session_finished_at_utc"] = _clean_text(bundle.get("session_finished_at_utc")) or normalized["session_started_at_utc"]
    normalized["task"] = _normalize_task(bundle.get("task"), fallback_title="Patch session")
    normalized["artifacts"] = _normalize_artifacts(bundle.get("artifacts"))
    normalized["declared_claims"] = _list_of_mappings(bundle.get("declared_claims"))
    normalized["privacy"] = _normalize_privacy(bundle.get("privacy"))
    issues = validate_agent_session_bundle(normalized)
    if issues:
        raise ValueError("Invalid agent session bundle: " + "; ".join(issues))
    return normalized


def load_agent_session_bundle(path: str | Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    bundle_path = _resolve_path(path, root=resolved_root)
    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Agent session bundle is not readable: `{bundle_path}`.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Agent session bundle is not valid JSON: `{bundle_path}`.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Agent session bundle must be a JSON object: `{bundle_path}`.")
    normalized = normalize_agent_session_bundle(payload)
    normalized["_bundle_path"] = str(bundle_path)
    normalized["_bundle_workspace_relative_path"] = _workspace_relative(bundle_path, root=resolved_root)
    return normalized


def build_agent_session_bundle(
    *,
    agent_label: str = "generic",
    diff: Path | None = None,
    transcript: Path | None = None,
    test_log: Path | None = None,
    ci_log: Path | None = None,
    note: str | None = None,
    title: str | None = None,
    goal: str | None = None,
) -> dict[str, Any]:
    artifacts: list[dict[str, str]] = []
    if diff is not None:
        artifacts.append({"kind": "diff", "path": str(diff)})
    if transcript is not None:
        artifacts.append({"kind": "transcript", "path": str(transcript)})
    if test_log is not None:
        artifacts.append({"kind": "test_log", "path": str(test_log)})
    if ci_log is not None:
        artifacts.append({"kind": "ci_log", "path": str(ci_log)})
    task_title = title or "Patch session"
    task_goal = goal or note or task_title
    return normalize_agent_session_bundle(
        {
            "schema_name": AGENT_SESSION_BUNDLE_SCHEMA_NAME,
            "schema_version": AGENT_SESSION_BUNDLE_SCHEMA_VERSION,
            "agent_label": agent_label,
            "session_started_at_utc": timestamp_utc(),
            "session_finished_at_utc": timestamp_utc(),
            "task": {"title": task_title, "goal": task_goal},
            "artifacts": artifacts,
            "declared_claims": [],
            "privacy": _default_privacy(),
            "note": _clean_text(note),
        }
    )


def build_pr_bundle(
    *,
    diff: Path | None = None,
    review: Path | None = None,
    ci_log: Path | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    artifacts: list[dict[str, str]] = []
    if diff is not None:
        artifacts.append({"kind": "diff", "path": str(diff)})
    if review is not None:
        artifacts.append({"kind": "review_note", "path": str(review)})
    if ci_log is not None:
        artifacts.append({"kind": "ci_log", "path": str(ci_log)})
    return normalize_agent_session_bundle(
        {
            "schema_name": AGENT_SESSION_BUNDLE_SCHEMA_NAME,
            "schema_version": AGENT_SESSION_BUNDLE_SCHEMA_VERSION,
            "agent_label": "manual",
            "session_started_at_utc": timestamp_utc(),
            "session_finished_at_utc": timestamp_utc(),
            "task": {
                "title": "PR bundle",
                "goal": note or "Ingest local PR diff, review, and CI log files.",
            },
            "artifacts": artifacts,
            "declared_claims": [],
            "privacy": _default_privacy(),
            "note": _clean_text(note),
        }
    )


def _capture_bundle_artifacts(
    bundle: Mapping[str, Any],
    *,
    root: Path,
    max_capture_bytes: int,
    report_excerpt_chars: int,
) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    bundle_path = _clean_text(bundle.get("_bundle_path"))
    if bundle_path is not None:
        bundle_ref = capture_artifact(
            bundle_path,
            kind="source_file",
            root=root,
            max_capture_bytes=max_capture_bytes,
            report_excerpt_chars=report_excerpt_chars,
        )
        captured.append(
            {
                "bundle_kind": "bundle_manifest",
                "vault_kind": "source_file",
                "path": bundle_path,
                "artifact_ref": bundle_ref,
            }
        )
    for artifact in _list_of_mappings(bundle.get("artifacts")):
        ref = capture_artifact(
            str(artifact["path"]),
            kind=str(artifact.get("vault_kind") or "unknown"),
            root=root,
            max_capture_bytes=max_capture_bytes,
            report_excerpt_chars=report_excerpt_chars,
        )
        captured.append(
            {
                "bundle_kind": artifact.get("kind"),
                "vault_kind": artifact.get("vault_kind"),
                "path": artifact.get("path"),
                "artifact_ref": ref,
            }
        )
    return captured


def _refs_from_captured(captured_artifacts: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for item in captured_artifacts:
        ref = item.get("artifact_ref")
        if isinstance(ref, Mapping):
            refs.append(copy.deepcopy(dict(ref)))
    return refs


def _captured_by_bundle_kind(captured_artifacts: Iterable[Mapping[str, Any]], *kinds: str) -> list[dict[str, Any]]:
    wanted = set(kinds)
    return [
        copy.deepcopy(dict(item))
        for item in captured_artifacts
        if _clean_text(item.get("bundle_kind")) in wanted
    ]


def _extract_bundle_claims(
    bundle: Mapping[str, Any],
    captured_artifacts: Iterable[Mapping[str, Any]],
    *,
    root: Path,
    transcript_claim_read_chars: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    claims = normalize_declared_claims(bundle.get("declared_claims"))
    diagnostics: list[str] = []

    for item in _captured_by_bundle_kind(captured_artifacts, "transcript", "review", "review_note"):
        ref = _mapping_dict(item.get("artifact_ref"))
        path = _clean_text(item.get("path"))
        source = _clean_text(item.get("bundle_kind")) or "transcript"
        if path is None:
            continue
        text, capped, read_issue = _read_text_for_claims(
            path,
            root=root,
            max_chars=transcript_claim_read_chars,
        )
        if read_issue is not None:
            diagnostics.append(f"{source}_claim_read_{read_issue}")
            continue
        if capped:
            diagnostics.append(f"{source}_claim_read_capped")
        claims.extend(
            extract_claims_from_transcript(
                text or "",
                source=source,
                source_artifact_id=_clean_text(ref.get("artifact_id")),
            )
        )
    verified = verify_claims_against_artifacts(claims, _refs_from_captured(captured_artifacts), root=root)
    return verified, diagnostics


def _bundle_session_key(bundle: Mapping[str, Any]) -> str:
    explicit = _clean_text(bundle.get("session_id"))
    if explicit is not None:
        return _sanitize_id_part(explicit, fallback="session")
    digest_input = {
        "agent_label": bundle.get("agent_label"),
        "started": bundle.get("session_started_at_utc"),
        "finished": bundle.get("session_finished_at_utc"),
        "task": bundle.get("task"),
        "artifacts": [
            {
                "kind": item.get("kind"),
                "path": item.get("path"),
            }
            for item in _list_of_mappings(bundle.get("artifacts"))
        ],
    }
    return _stable_digest(digest_input, length=20)


def _artifact_diagnostics(bundle: Mapping[str, Any], captured_artifacts: list[dict[str, Any]]) -> list[str]:
    diagnostics: list[str] = []
    diff_items = _captured_by_bundle_kind(captured_artifacts, "diff", "patch")
    captured_diff_items = [
        item
        for item in diff_items
        if _mapping_dict(item.get("artifact_ref")).get("capture_state") == "captured"
    ]
    if not captured_diff_items:
        diagnostics.append("missing_diff")
    for item in captured_artifacts:
        bundle_kind = _clean_text(item.get("bundle_kind")) or "unknown"
        ref = _mapping_dict(item.get("artifact_ref"))
        if ref.get("capture_state") != "captured":
            diagnostics.append(f"{bundle_kind}_{_clean_text(ref.get('source_state')) or 'not_captured'}")
    privacy = _mapping_dict(bundle.get("privacy"))
    if privacy.get("export_allowed"):
        diagnostics.append("declared_export_allowed_ignored")
    return diagnostics


def _read_verified_ref_text(
    ref: Mapping[str, Any],
    *,
    root: Path,
    max_chars: int = DEFAULT_VERIFICATION_LOG_READ_CHARS,
) -> str | None:
    verified, _reason = artifact_ref_object_verified(ref, root=root)
    if not verified:
        return None
    object_path = resolve_vault_object_path(ref, root=root)
    if object_path is None or not object_path.is_file():
        return None
    try:
        with object_path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(max_chars)
    except OSError:
        return None


def _verification_log_signal(captured_artifacts: list[dict[str, Any]], *, root: Path) -> dict[str, Any]:
    signals: list[dict[str, str | None]] = []
    for item in captured_artifacts:
        bundle_kind = _clean_text(item.get("bundle_kind"))
        if bundle_kind not in {"test_log", "ci_log"}:
            continue
        ref = _mapping_dict(item.get("artifact_ref"))
        text = _read_verified_ref_text(ref, root=root)
        if text is None:
            continue
        if log_indicates_tests_failed(text):
            signals.append(
                {
                    "artifact_id": _clean_text(ref.get("artifact_id")),
                    "kind": bundle_kind,
                    "signal": "test_fail",
                }
            )
        elif log_indicates_tests_passed(text):
            signals.append(
                {
                    "artifact_id": _clean_text(ref.get("artifact_id")),
                    "kind": bundle_kind,
                    "signal": "test_pass",
                }
            )

    signal_values = {signal["signal"] for signal in signals}
    if "test_fail" in signal_values:
        return {
            "quality_status": "fail",
            "evidence_types": ["test_fail"],
            "signals": signals,
        }
    if "test_pass" in signal_values:
        return {
            "quality_status": "pass",
            "evidence_types": ["test_pass"],
            "signals": signals,
        }
    return {"quality_status": None, "evidence_types": [], "signals": []}


def _dedupe_texts(values: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        if text is None or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _combined_quality_status(
    *,
    claim_status: str | None,
    log_status: str | None,
) -> str | None:
    if "fail" in {claim_status, log_status}:
        return "fail"
    if "pass" in {claim_status, log_status}:
        return "pass"
    return None


def _status_from_intake(*, diagnostics: list[str], quality_status: str | None) -> str:
    if "missing_diff" in diagnostics:
        return "diagnostic"
    if quality_status == "fail":
        return "failed"
    if quality_status == "pass":
        return "verified"
    return "needs_review"


def _build_session_event(
    *,
    bundle: Mapping[str, Any],
    captured_artifacts: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    diagnostics: list[str],
    verification_log_signal: Mapping[str, Any],
    workspace_id: str,
) -> dict[str, Any]:
    agent_label = _clean_text(bundle.get("agent_label")) or "unknown"
    session_key = _bundle_session_key(bundle)
    refs = _refs_from_captured(captured_artifacts)
    primary_ref = next(
        (ref for ref in refs if _clean_text(ref.get("kind")) == "patch"),
        refs[0] if refs else None,
    )
    counts = claim_counts(claims)
    claims_can_support_patch = "missing_diff" not in diagnostics
    claim_status = claim_quality_status(claims)
    log_status = _clean_text(verification_log_signal.get("quality_status"))
    quality_status = (
        _combined_quality_status(claim_status=claim_status, log_status=log_status)
        if claims_can_support_patch
        else None
    )
    evidence_types = (
        _dedupe_texts(
            [
                *claim_evidence_types(claims),
                *[
                    str(item)
                    for item in verification_log_signal.get("evidence_types") or []
                    if isinstance(item, str)
                ],
            ]
        )
        if claims_can_support_patch
        else []
    )
    status = _status_from_intake(diagnostics=diagnostics, quality_status=quality_status)
    task = _mapping_dict(bundle.get("task"))
    privacy = _mapping_dict(bundle.get("privacy"))
    options: dict[str, Any] = {
        "workflow": "agent_session_intake",
        "validation_mode": "file_first_agent_session_intake",
        "claim_scope": "Agent session claims require local verification artifacts before decision support.",
        "pass_definition": "Only local captured artifacts and linked logs can verify an agent transcript claim.",
        "agent_label": agent_label,
        "session_key": session_key,
        "artifact_vault_refs": refs,
        "captured_artifacts": copy.deepcopy(captured_artifacts),
        "claim_counts": counts,
        "claims": copy.deepcopy(claims),
        "privacy": copy.deepcopy(privacy),
        "export_policy": {
            "declared_export_allowed": bool(privacy.get("export_allowed")),
            "effective_export_allowed": False,
            "no_api_calls": True,
            "no_cloud_sync": True,
            "no_provider_hub": True,
            "no_vector_export": True,
            "no_training_export": True,
        },
        "diagnostics": list(diagnostics),
    }
    if verification_log_signal.get("signals"):
        options["verification_log_signal"] = copy.deepcopy(dict(verification_log_signal))
    if not claims_can_support_patch:
        options["verified_claims_held_for_patch_evidence"] = True
    if quality_status is not None:
        options["quality_status"] = quality_status
    if evidence_types:
        options["evidence_types"] = evidence_types
        options["quality_checks"] = [
            {
                "name": "verified_agent_claim_has_local_evidence",
                "pass": True,
                "detail": ", ".join(evidence_types),
            }
        ]

    notes = ["agent_file_first_intake", "m13"]
    if counts["total"]:
        notes.append("agent_claim")
    if counts["unverified_agent_claim"]:
        notes.append("agent_transcript_claim")
    if diagnostics:
        notes.extend(diagnostics[:6])

    source_refs: dict[str, Any] = {
        "artifact_vault_refs": refs,
        "bundle_path": _clean_text(bundle.get("_bundle_path")),
        "bundle_workspace_relative_path": _clean_text(bundle.get("_bundle_workspace_relative_path")),
    }
    if primary_ref is not None:
        source_refs["artifact_ref"] = {
            "entry_id": session_key,
            "artifact_kind": _clean_text(primary_ref.get("kind")) or "unknown",
            "action": "agent_session_intake",
            "status": status,
            "recorded_at_utc": _clean_text(bundle.get("session_finished_at_utc")) or timestamp_utc(),
            "artifact_path": _clean_text(primary_ref.get("original_path")),
            "artifact_workspace_relative_path": _clean_text(primary_ref.get("repo_relative_path")),
            "artifact_sha256": _clean_text(primary_ref.get("sha256")),
        }

    return build_event_record(
        event_id=f"{workspace_id}:agent-session-intake:{_sanitize_id_part(agent_label, fallback='unknown')}:{session_key}",
        event_kind="agent_session_intake",
        recorded_at_utc=_clean_text(bundle.get("session_finished_at_utc")) or timestamp_utc(),
        workspace={"workspace_id": workspace_id},
        session={
            "session_id": f"agent-session-intake-{session_key}",
            "surface": "chat",
            "mode": "agent_session_intake",
            "title": _clean_text(task.get("title")) or "Agent session intake",
            "selected_model_id": None,
            "session_manifest_path": None,
        },
        outcome={
            "status": status,
            "quality_status": quality_status,
            "execution_status": status,
        },
        content={
            "prompt": _clean_text(task.get("goal")) or _clean_text(task.get("title")),
            "system_prompt": None,
            "resolved_user_prompt": _clean_text(task.get("goal")),
            "output_text": (
                f"Ingested {agent_label} file-first session with "
                f"{len(refs)} artifacts, {counts['verified_signal']} verified claims, "
                f"and {counts['unverified_agent_claim']} unverified claims."
            ),
            "notes": notes,
            "options": options,
        },
        source_refs=source_refs,
        tags=["agent_file_first_intake", "m13", agent_label, status, *notes],
    )


def _unverified_claim_positive_support_count(
    *,
    claims: list[dict[str, Any]],
    refs: list[dict[str, Any]],
    workspace_id: str,
    agent_label: str,
    session_id: str,
    root: Path,
) -> int:
    count = 0
    for claim in claims:
        if _clean_text(claim.get("verification_state")) == "verified_signal":
            continue
        event = build_claim_software_work_event(
            claim,
            artifact_refs=refs,
            workspace_id=workspace_id,
            session_id=session_id,
            agent_label=agent_label,
        )
        support = build_evidence_support_result(
            event["event_id"],
            event=event,
            requested_polarity="positive",
            root=root,
        )
        if support.get("can_support_decision"):
            count += 1
    return count


def _secret_redaction_fixture_failures(refs: Iterable[Mapping[str, Any]]) -> int:
    failures = 0
    for ref in refs:
        redaction = _mapping_dict(ref.get("redaction"))
        excerpt = _mapping_dict(ref.get("report_excerpt")).get("text")
        if int(redaction.get("secret_like_tokens") or 0) > 0 and "[REDACTED]" not in str(excerpt or ""):
            failures += 1
    return failures


def _build_exit_gate(
    *,
    bundle: Mapping[str, Any],
    claims: list[dict[str, Any]],
    refs: list[dict[str, Any]],
    event: Mapping[str, Any],
    workspace_id: str,
    root: Path,
) -> dict[str, Any]:
    session = _mapping_dict(event.get("session"))
    agent_label = _clean_text(bundle.get("agent_label")) or "unknown"
    metrics = {
        "fixture_bundles_normalized": 1,
        "agent_labels_represented": 1,
        "unverified_agent_claim_positive_support_count": _unverified_claim_positive_support_count(
            claims=claims,
            refs=refs,
            workspace_id=workspace_id,
            agent_label=agent_label,
            session_id=_clean_text(session.get("session_id")) or "agent-session-intake",
            root=root,
        ),
        "secret_redaction_fixture_failures": _secret_redaction_fixture_failures(refs),
        "network_call_count": 0,
    }
    metrics["passed"] = (
        metrics["unverified_agent_claim_positive_support_count"] == 0
        and metrics["secret_redaction_fixture_failures"] == 0
        and metrics["network_call_count"] == 0
    )
    return metrics


def ingest_agent_session_bundle(
    bundle: Mapping[str, Any],
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    refresh_index: bool = True,
    write_latest: bool = True,
    max_capture_bytes: int = DEFAULT_MAX_CAPTURE_BYTES,
    report_excerpt_chars: int = DEFAULT_REPORT_EXCERPT_CHARS,
    transcript_claim_read_chars: int = DEFAULT_TRANSCRIPT_CLAIM_READ_CHARS,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    normalized_bundle = normalize_agent_session_bundle(bundle)
    for key in ("_bundle_path", "_bundle_workspace_relative_path"):
        if key in bundle:
            normalized_bundle[key] = copy.deepcopy(bundle[key])

    captured_artifacts = _capture_bundle_artifacts(
        normalized_bundle,
        root=resolved_root,
        max_capture_bytes=max_capture_bytes,
        report_excerpt_chars=report_excerpt_chars,
    )
    artifact_diagnostics = _artifact_diagnostics(normalized_bundle, captured_artifacts)
    claims, claim_diagnostics = _extract_bundle_claims(
        normalized_bundle,
        captured_artifacts,
        root=resolved_root,
        transcript_claim_read_chars=transcript_claim_read_chars,
    )
    diagnostics = [*artifact_diagnostics, *claim_diagnostics]
    log_signal = _verification_log_signal(captured_artifacts, root=resolved_root)
    refs = _refs_from_captured(captured_artifacts)
    event = _build_session_event(
        bundle=normalized_bundle,
        captured_artifacts=captured_artifacts,
        claims=claims,
        diagnostics=diagnostics,
        verification_log_signal=log_signal,
        workspace_id=workspace_id,
    )
    exit_gate = _build_exit_gate(
        bundle=normalized_bundle,
        claims=claims,
        refs=refs,
        event=event,
        workspace_id=workspace_id,
        root=resolved_root,
    )

    run_path = agent_session_intake_run_path(workspace_id=workspace_id, root=resolved_root)
    latest_path = agent_session_intake_latest_path(workspace_id=workspace_id, root=resolved_root)
    result = {
        "schema_name": AGENT_SESSION_INTAKE_RESULT_SCHEMA_NAME,
        "schema_version": AGENT_SESSION_INTAKE_RESULT_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "recorded_at_utc": timestamp_utc(),
        "bundle": normalized_bundle,
        "software_work_event": event,
        "captured_artifacts": captured_artifacts,
        "claims": claims,
        "claim_counts": claim_counts(claims),
        "diagnostics": diagnostics,
        "privacy": copy.deepcopy(normalized_bundle.get("privacy")),
        "export_policy": {
            "declared_export_allowed": bool(_mapping_dict(normalized_bundle.get("privacy")).get("export_allowed")),
            "effective_export_allowed": False,
            "no_api_calls": True,
            "no_cloud_sync": True,
            "no_provider_hub": True,
            "no_vector_export": True,
            "no_training_export": True,
        },
        "exit_gate": exit_gate,
        "paths": {
            "run_path": str(run_path),
            "latest_path": str(latest_path) if write_latest else None,
        },
    }
    write_json(run_path, result)
    if write_latest:
        write_json(latest_path, result)
    result["index_summary"] = rebuild_memory_index(root=resolved_root, workspace_id=workspace_id) if refresh_index else None
    return result


def ingest_agent_session_bundle_path(
    path: str | Path,
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    bundle = load_agent_session_bundle(path, root=root)
    return ingest_agent_session_bundle(bundle, workspace_id=workspace_id, root=root, **kwargs)


def aggregate_intake_exit_gate(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    result_list = [dict(result) for result in results if isinstance(result, Mapping)]
    labels = {
        _clean_text(_mapping_dict(result.get("bundle")).get("agent_label")) or "unknown"
        for result in result_list
    }
    metrics = {
        "fixture_bundles_normalized": len(result_list),
        "agent_labels_represented": len(labels),
        "unverified_agent_claim_positive_support_count": sum(
            int(_mapping_dict(result.get("exit_gate")).get("unverified_agent_claim_positive_support_count") or 0)
            for result in result_list
        ),
        "secret_redaction_fixture_failures": sum(
            int(_mapping_dict(result.get("exit_gate")).get("secret_redaction_fixture_failures") or 0)
            for result in result_list
        ),
        "network_call_count": sum(
            int(_mapping_dict(result.get("exit_gate")).get("network_call_count") or 0)
            for result in result_list
        ),
    }
    metrics["passed"] = (
        metrics["fixture_bundles_normalized"] >= 5
        and metrics["agent_labels_represented"] >= 3
        and metrics["unverified_agent_claim_positive_support_count"] == 0
        and metrics["secret_redaction_fixture_failures"] == 0
        and metrics["network_call_count"] == 0
    )
    return metrics


def format_agent_session_intake_result(result: Mapping[str, Any]) -> str:
    bundle = _mapping_dict(result.get("bundle"))
    event = _mapping_dict(result.get("software_work_event"))
    outcome = _mapping_dict(event.get("outcome"))
    counts = _mapping_dict(result.get("claim_counts"))
    diagnostics = [str(item) for item in result.get("diagnostics") or [] if item]
    lines = [
        "Agent session intake",
        f"Agent label: {_clean_text(bundle.get('agent_label')) or 'unknown'}",
        f"Event: {_clean_text(event.get('event_id')) or 'unknown'}",
        f"Status: {_clean_text(outcome.get('status')) or 'unknown'}",
        (
            "Claims: "
            f"verified={int(counts.get('verified_signal') or 0)}, "
            f"unverified={int(counts.get('unverified_agent_claim') or 0)}"
        ),
        "Export: disabled; local files only.",
    ]
    if diagnostics:
        lines.append("Diagnostics: " + ", ".join(diagnostics))
    lines.append("")
    lines.append("Redacted artifact excerpts:")
    for item in result.get("captured_artifacts") or []:
        if not isinstance(item, Mapping):
            continue
        ref = _mapping_dict(item.get("artifact_ref"))
        excerpt = _mapping_dict(ref.get("report_excerpt")).get("text")
        lines.append(f"- {_clean_text(item.get('bundle_kind')) or 'unknown'}: {_clean_text(ref.get('artifact_id')) or 'n/a'}")
        if excerpt:
            lines.append(str(excerpt))
    return "\n".join(lines)
