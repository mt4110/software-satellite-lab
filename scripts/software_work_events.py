#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from agent_lane import agent_run_log_path, read_agent_runs
from artifact_schema import read_artifact
from gemma_runtime import repo_root, timestamp_utc
from workspace_state import (
    DEFAULT_WORKSPACE_ID,
    WorkspaceSessionStore,
    read_session_manifest,
    read_workspace_manifest,
    session_manifest_path,
    workspace_manifest_path,
    workspace_dir,
)


EVENT_SCHEMA_NAME = "software-satellite-event"
EVENT_SCHEMA_VERSION = 1
EVENT_LOG_SCHEMA_NAME = "software-satellite-event-log"
EVENT_LOG_SCHEMA_VERSION = 1
EVENT_CONTRACT_CHECK_SCHEMA_NAME = "software-satellite-event-contract-check"
EVENT_CONTRACT_CHECK_SCHEMA_VERSION = 1
EVENT_CONTRACT_REPORT_SCHEMA_NAME = "software-satellite-event-contract-report"
EVENT_CONTRACT_REPORT_SCHEMA_VERSION = 1


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def event_logs_root(root: Path | None = None) -> Path:
    return _resolve_root(root) / "artifacts" / "event_logs"


def workspace_event_log_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return event_logs_root(root) / f"{workspace_id}.jsonl"


def capability_matrix_root(root: Path | None = None) -> Path:
    return _resolve_root(root) / "artifacts" / "capability_matrix"


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _clean_string_list(value: Any) -> list[str]:
    cleaned: list[str] = []
    if not isinstance(value, list):
        return cleaned
    for item in value:
        normalized = _clean_text(item)
        if normalized is not None:
            cleaned.append(normalized)
    return cleaned


def _clean_quality_checks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    checks: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue

        check: dict[str, Any] = {}
        name = _clean_text(item.get("name"))
        passed = item.get("pass")
        detail = _clean_text(item.get("detail"))
        if name is None or not isinstance(passed, bool):
            continue
        check["name"] = name
        check["pass"] = passed
        if detail is not None:
            check["detail"] = detail
        checks.append(check)
    return checks


def _quality_check_verdict(value: bool) -> str:
    return "pass" if value else "fail"


def _workspace_relative_path(path: Path, *, root: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(root))
    except (OSError, RuntimeError):
        try:
            return str(path.absolute().relative_to(root))
        except ValueError:
            return None
    except ValueError:
        return None


def _mapping_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _resolve_source_artifact_path(path_text: str, *, root: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return path.absolute()


def build_event_contract_check(
    event: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Check the minimum file-first contract for one software-work event."""
    resolved_root = _resolve_root(root)
    event_id = _clean_text(event.get("event_id"))
    event_kind = _clean_text(event.get("event_kind"))
    outcome = _mapping_dict(event.get("outcome"))
    status = _clean_text(outcome.get("status"))
    source_refs = _mapping_dict(event.get("source_refs"))
    artifact_ref = _mapping_dict(source_refs.get("artifact_ref"))
    artifact_path = _clean_text(artifact_ref.get("artifact_path"))
    schema_issues: list[str] = []
    source_reasons: list[str] = []
    resolved_artifact_path: Path | None = None
    workspace_relative_path: str | None = None
    durability_status = "missing_path"
    readability_status = "missing_path"

    if event_id is None:
        schema_issues.append("missing_event_id")
    if event_kind is None:
        schema_issues.append("missing_event_kind")
    if status is None:
        schema_issues.append("missing_status")

    if artifact_path is None:
        source_reasons.append("missing_source_artifact_path")
    else:
        resolved_artifact_path = _resolve_source_artifact_path(artifact_path, root=resolved_root)
        workspace_relative_path = _workspace_relative_path(resolved_artifact_path, root=resolved_root)
        if workspace_relative_path is None:
            durability_status = "outside_workspace"
            readability_status = "not_checked"
            source_reasons.append("source_artifact_outside_workspace")
        else:
            durability_status = "durable"

            if not resolved_artifact_path.exists():
                readability_status = "missing"
                source_reasons.append("source_artifact_missing")
            elif not resolved_artifact_path.is_file():
                readability_status = "unreadable"
                source_reasons.append("source_artifact_not_file")
            else:
                try:
                    with resolved_artifact_path.open("rb") as handle:
                        handle.read(1)
                except OSError:
                    readability_status = "unreadable"
                    source_reasons.append("source_artifact_unreadable")
                else:
                    readability_status = "readable"

    source_status = "readable" if not source_reasons else "missing_source"
    if source_status == "missing_source":
        contract_status = "missing_source"
    elif schema_issues:
        contract_status = "invalid_event_contract"
    else:
        contract_status = "ok"

    return {
        "schema_name": EVENT_CONTRACT_CHECK_SCHEMA_NAME,
        "schema_version": EVENT_CONTRACT_CHECK_SCHEMA_VERSION,
        "event_id": event_id,
        "event_kind": event_kind,
        "status": status,
        "contract_status": contract_status,
        "schema_issues": schema_issues,
        "source_artifact": {
            "source_status": source_status,
            "durability_status": durability_status,
            "readability_status": readability_status,
            "reasons": source_reasons,
            "artifact_path": artifact_path,
            "resolved_artifact_path": str(resolved_artifact_path) if resolved_artifact_path is not None else None,
            "artifact_workspace_relative_path": workspace_relative_path,
        },
    }


def build_event_contract_report(
    events: Iterable[Mapping[str, Any]],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    checks = [
        build_event_contract_check(event, root=root)
        for event in events
        if isinstance(event, Mapping)
    ]
    status_counts: dict[str, int] = {}
    source_status_counts: dict[str, int] = {}
    issue_counts: dict[str, int] = {}
    for check in checks:
        contract_status = _clean_text(check.get("contract_status")) or "unknown"
        status_counts[contract_status] = status_counts.get(contract_status, 0) + 1
        source_artifact = _mapping_dict(check.get("source_artifact"))
        source_status = _clean_text(source_artifact.get("source_status")) or "unknown"
        source_status_counts[source_status] = source_status_counts.get(source_status, 0) + 1
        for issue in [
            *_clean_string_list(check.get("schema_issues")),
            *_clean_string_list(source_artifact.get("reasons")),
        ]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    failed_checks = [
        check
        for check in checks
        if _clean_text(check.get("contract_status")) != "ok"
    ]
    return {
        "schema_name": EVENT_CONTRACT_REPORT_SCHEMA_NAME,
        "schema_version": EVENT_CONTRACT_REPORT_SCHEMA_VERSION,
        "checked_event_count": len(checks),
        "failed_event_count": len(failed_checks),
        "status_counts": dict(sorted(status_counts.items())),
        "source_status_counts": dict(sorted(source_status_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "failed_checks": failed_checks,
    }


def _capability_surface(result: dict[str, Any]) -> str:
    artifact_kind = _clean_text(result.get("artifact_kind"))
    if artifact_kind == "vision":
        return "vision"
    if artifact_kind == "audio":
        return "audio"
    if artifact_kind == "thinking":
        return "thinking"
    if artifact_kind == "text":
        return "chat"
    return "evaluation"


def _artifact_validation_context(artifact_ref: dict[str, Any]) -> dict[str, Any]:
    artifact_path = _clean_text(artifact_ref.get("artifact_path"))
    if artifact_path is None:
        return {}
    path = Path(artifact_path).expanduser()
    if not path.exists():
        return {}
    try:
        payload = read_artifact(path)
    except Exception:
        return {}
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        return {}

    context: dict[str, Any] = {}
    for key in (
        "validation_mode",
        "claim_scope",
        "pass_definition",
        "quality_status",
        "execution_status",
    ):
        value = _clean_text(validation.get(key))
        if value is not None:
            context[key] = value
    quality_checks = _clean_quality_checks(validation.get("quality_checks"))
    if quality_checks:
        context["quality_checks"] = quality_checks
    quality_notes = _clean_string_list(validation.get("quality_notes"))
    if quality_notes:
        context["quality_notes"] = quality_notes
    return context


def _capability_attached_assets(result: dict[str, Any], *, root: Path) -> list[dict[str, Any]]:
    attached_assets: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    asset_used = _clean_text(result.get("asset_used"))
    if asset_used:
        resolved = Path(asset_used).expanduser()
        seen_paths.add(str(resolved))
        attached_assets.append(
            {
                "role": "primary_input",
                "kind": "file",
                "path": str(resolved),
                "workspace_relative_path": _workspace_relative_path(resolved, root=root),
            }
        )

    for lineage in result.get("preprocessing_lineage") or []:
        if not isinstance(lineage, dict):
            continue
        source_path = _clean_text(lineage.get("source_path"))
        if not source_path:
            continue
        resolved = Path(source_path).expanduser()
        path_key = str(resolved)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        attached_assets.append(
            {
                "role": "preprocessing_source",
                "kind": "file",
                "path": path_key,
                "workspace_relative_path": _workspace_relative_path(resolved, root=root),
            }
        )
    return attached_assets


def build_event_from_capability_matrix_result(
    *,
    root: Path | None,
    workspace_id: str,
    matrix_artifact_path: Path,
    matrix_payload: dict[str, Any],
    result: dict[str, Any],
    result_index: int,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    capability = _clean_text(result.get("capability")) or f"result-{result_index + 1}"
    phase = _clean_text(result.get("phase"))
    status = _clean_text(result.get("status")) or _clean_text(result.get("result"))
    quality_status = _clean_text(result.get("quality_status"))
    claim_scope = _clean_text(result.get("claim_scope"))
    validation_command = _clean_text(result.get("validation_command"))
    quality_checks = _clean_quality_checks(result.get("quality_checks"))
    output_preview = _clean_text(result.get("output_preview"))
    blocker = result.get("blocker") if isinstance(result.get("blocker"), dict) else {}
    blocker_message = _clean_text(blocker.get("message"))
    runtime = matrix_payload.get("runtime") if isinstance(matrix_payload.get("runtime"), dict) else {}
    recorded_at = _clean_text(matrix_payload.get("timestamp_utc")) or timestamp_utc()
    session_id = f"capability-matrix:{matrix_artifact_path.stem}"
    matrix_row_ref = f"{matrix_artifact_path.stem}:row-{result_index + 1}:{capability}"
    artifact_path_text = _clean_text(result.get("artifact_path"))
    artifact_path = Path(artifact_path_text).expanduser() if artifact_path_text else matrix_artifact_path
    attached_assets = _capability_attached_assets(result, root=resolved_root)

    notes = _clean_string_list(result.get("notes"))
    notes.extend(_clean_string_list(result.get("quality_notes")))
    if quality_status:
        notes.append(f"quality_status: {quality_status}")
    validation_mode = _clean_text(result.get("validation_mode"))
    if validation_mode:
        notes.append(f"validation_mode: {validation_mode}")
    execution_status = _clean_text(result.get("execution_status"))
    if execution_status:
        notes.append(f"execution_status: {execution_status}")
    if blocker_message:
        notes.append(blocker_message)
    for item in quality_checks:
        name = _clean_text(item.get("name")) or "quality_check"
        detail = _clean_text(item.get("detail"))
        verdict = _quality_check_verdict(item.get("pass"))
        if detail:
            notes.append(f"{name}: {verdict} - {detail}")
        else:
            notes.append(f"{name}: {verdict}")

    prompt_parts = [capability]
    if claim_scope:
        prompt_parts.append(claim_scope)
    if validation_command:
        prompt_parts.append(validation_command)

    workspace_manifest = workspace_manifest_path(
        workspace_id=workspace_id,
        root=resolved_root,
    )
    workspace_record = {
        "workspace_id": workspace_id,
        "workspace_manifest_path": str(workspace_manifest),
    }
    session_record = {
        "session_id": session_id,
        "surface": _capability_surface(result),
        "mode": phase or "capability-matrix",
        "title": "Capability Matrix",
        "selected_model_id": _clean_text(result.get("model_used"))
        or _clean_text(runtime.get("model_id")),
        "session_manifest_path": None,
    }
    source_refs = {
        "artifact_ref": {
            "entry_id": matrix_row_ref,
            "artifact_kind": _clean_text(result.get("artifact_kind")) or "artifact",
            "action": "capability_matrix",
            "status": status,
            "recorded_at_utc": recorded_at,
            "artifact_path": str(artifact_path),
            "artifact_workspace_relative_path": _workspace_relative_path(artifact_path, root=resolved_root),
        },
        "attached_assets": attached_assets,
        "matrix_artifact_ref": {
            "artifact_path": str(matrix_artifact_path),
            "artifact_workspace_relative_path": _workspace_relative_path(matrix_artifact_path, root=resolved_root),
        },
    }
    tags = [
        tag
        for tag in (
            "capability_matrix",
            session_record["surface"],
            phase,
            capability,
            status,
            quality_status,
            _clean_text(result.get("runtime_backend")),
        )
        if isinstance(tag, str) and tag
    ]
    content = {
        "prompt": " | ".join(part for part in prompt_parts if part),
        "system_prompt": None,
        "resolved_user_prompt": validation_command or claim_scope,
        "output_text": output_preview or blocker_message or f"{capability} run completed.",
        "notes": notes,
        "options": {
            "phase": phase,
            "quality_status": quality_status,
            "quality_checks": quality_checks,
            "validation_mode": validation_mode,
            "validation_command": validation_command,
            "claim_scope": claim_scope,
            "pass_definition": _clean_text(result.get("pass_definition")),
            "matrix_artifact_path": str(matrix_artifact_path),
        },
    }
    return build_event_record(
        event_id=f"{workspace_id}:{session_id}:row-{result_index + 1}:{capability}",
        event_kind="capability_result",
        recorded_at_utc=recorded_at,
        workspace=workspace_record,
        session=session_record,
        outcome={
            "status": status,
            "quality_status": quality_status,
            "execution_status": execution_status,
        },
        content=content,
        source_refs=source_refs,
        tags=tags,
    )


def build_event_record(
    *,
    event_id: str,
    event_kind: str,
    recorded_at_utc: str | None,
    workspace: dict[str, Any],
    session: dict[str, Any],
    outcome: dict[str, Any] | None = None,
    content: dict[str, Any] | None = None,
    source_refs: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_name": EVENT_SCHEMA_NAME,
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": event_id,
        "event_kind": event_kind,
        "recorded_at_utc": recorded_at_utc or timestamp_utc(),
        "workspace": copy.deepcopy(workspace),
        "session": copy.deepcopy(session),
        "outcome": copy.deepcopy(outcome or {}),
        "content": copy.deepcopy(content or {}),
        "source_refs": copy.deepcopy(source_refs or {}),
        "tags": list(tags or []),
    }


def _agent_run_event_status(run_status: str | None) -> str | None:
    if run_status == "succeeded":
        return "ok"
    return run_status


def _agent_run_event_id(*, workspace_id: str, run_id: str) -> str:
    run_id_digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
    safe_run_id = "".join(
        char if char.isalnum() or char in ("-", "_") else "-"
        for char in run_id
    ).strip("-") or "agent-run"
    return f"{workspace_id}:agent-lane-run:{safe_run_id[:64]}:{run_id_digest}"


def _deduplicated_notes(*values: str | None) -> list[str]:
    notes: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        if text is None or text in seen:
            continue
        seen.add(text)
        notes.append(text)
    return notes


def build_event_from_agent_run(
    *,
    root: Path | None,
    workspace_id: str,
    run: dict[str, Any],
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    run_id = _clean_text(run.get("run_id")) or "agent-run"
    task = run.get("task_snapshot") if isinstance(run.get("task_snapshot"), dict) else {}
    outcome = run.get("outcome") if isinstance(run.get("outcome"), dict) else {}
    paths = run.get("paths") if isinstance(run.get("paths"), dict) else {}
    backend = run.get("backend") if isinstance(run.get("backend"), dict) else {}
    compatibility = run.get("compatibility") if isinstance(run.get("compatibility"), dict) else {}
    verification = task.get("verification") if isinstance(task.get("verification"), dict) else {}
    commands = [
        command.get("command")
        for command in verification.get("commands") or []
        if isinstance(command, dict) and _clean_text(command.get("command")) is not None
    ]
    quality_checks = _clean_quality_checks(outcome.get("quality_checks"))
    result_summary = _clean_text(outcome.get("result_summary"))
    failure_summary = _clean_text(outcome.get("failure_summary"))
    run_artifact_path_text = _clean_text(paths.get("run_artifact_path"))
    run_artifact_path = Path(run_artifact_path_text).expanduser() if run_artifact_path_text else None
    recorded_at = _clean_text(run.get("completed_at_utc")) or _clean_text(run.get("started_at_utc")) or timestamp_utc()
    run_status = _clean_text(run.get("status"))
    event_status = _agent_run_event_status(run_status)

    workspace_manifest = workspace_manifest_path(workspace_id=workspace_id, root=resolved_root)
    workspace_record = {
        "workspace_id": workspace_id,
        "workspace_manifest_path": str(workspace_manifest),
    }
    task_id = _clean_text(run.get("task_id")) or _clean_text(task.get("task_id")) or run_id
    task_kind = _clean_text(task.get("task_kind")) or "patch_plan_verify"
    backend_id = _clean_text(backend.get("backend_id")) or _clean_text(outcome.get("backend_id"))
    adapter_kind = _clean_text(backend.get("adapter_kind"))
    model_id = _clean_text(backend.get("model_id")) or _clean_text(outcome.get("model_id"))
    backend_capabilities = backend.get("capabilities") if isinstance(backend.get("capabilities"), dict) else {}
    session_record = {
        "session_id": task_id,
        "surface": "agent_lane",
        "mode": task_kind,
        "title": _clean_text(task.get("title")),
        "selected_model_id": model_id,
        "session_manifest_path": None,
    }
    options = {
        "validation_mode": "agent_lane",
        "validation_command": "\n".join(str(command) for command in commands if command),
        "claim_scope": _clean_text(task.get("title")),
        "pass_definition": _clean_text(verification.get("pass_definition")),
        "quality_status": _clean_text(outcome.get("quality_status")),
        "execution_status": _clean_text(outcome.get("execution_status")),
        "quality_checks": quality_checks,
        "agent_run_status": run_status,
        "agent_task_id": task_id,
        "agent_run_id": run_id,
        "tool_trace_count": len(run.get("tool_traces") or []),
        "backend_id": backend_id,
        "backend_adapter_kind": adapter_kind,
        "backend_display_name": _clean_text(backend.get("display_name")),
        "model_id": model_id,
        "backend_capabilities": copy.deepcopy(backend_capabilities),
        "backend_compatibility_status": _clean_text(compatibility.get("status")),
    }
    if event_status == "failed":
        notes = _deduplicated_notes(failure_summary, result_summary)
    else:
        notes = _deduplicated_notes(result_summary, failure_summary)
    source_refs = {
        "artifact_ref": {
            "entry_id": run_id,
            "artifact_kind": "agent_run",
            "action": "agent_lane",
            "status": run_status,
            "recorded_at_utc": recorded_at,
            "artifact_path": str(run_artifact_path) if run_artifact_path is not None else None,
            "artifact_workspace_relative_path": (
                _workspace_relative_path(run_artifact_path, root=resolved_root)
                if run_artifact_path is not None
                else None
            ),
        },
        "agent_task_ref": {
            "task_id": task_id,
            "task_kind": task_kind,
        },
    }
    if backend_id is not None or model_id is not None:
        source_refs["backend_ref"] = {
            "backend_id": backend_id,
            "adapter_kind": adapter_kind,
            "model_id": model_id,
            "capabilities": copy.deepcopy(backend_capabilities),
            "compatibility_status": _clean_text(compatibility.get("status")),
        }
    tags = [
        tag
        for tag in (
            "agent_lane",
            "m5",
            "m6" if backend_id is not None else None,
            "backend_swap" if backend_id is not None else None,
            task_kind,
            run_status,
            _clean_text(outcome.get("quality_status")),
            _clean_text(outcome.get("execution_status")),
            backend_id,
            adapter_kind,
            model_id,
        )
        if isinstance(tag, str) and tag
    ]
    return build_event_record(
        event_id=_agent_run_event_id(workspace_id=workspace_id, run_id=run_id),
        event_kind="agent_task_run",
        recorded_at_utc=recorded_at,
        workspace=workspace_record,
        session=session_record,
        outcome={
            "status": event_status,
            "quality_status": _clean_text(outcome.get("quality_status")),
            "execution_status": _clean_text(outcome.get("execution_status")),
            "backend_id": backend_id,
            "model_id": model_id,
        },
        content={
            "prompt": _clean_text(task.get("goal")),
            "system_prompt": None,
            "resolved_user_prompt": _clean_text(task.get("goal")),
            "output_text": result_summary,
            "notes": notes,
            "options": options,
        },
        source_refs=source_refs,
        tags=tags,
    )


def build_event_log_payload(
    *,
    workspace_id: str,
    events: list[dict[str, Any]],
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_name": EVENT_LOG_SCHEMA_NAME,
        "schema_version": EVENT_LOG_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "generated_at_utc": generated_at_utc or timestamp_utc(),
        "event_count": len(events),
        "events": copy.deepcopy(events),
    }


def write_event_log(path: Path, events: Iterable[dict[str, Any]], *, workspace_id: str) -> dict[str, Any]:
    normalized_events = [copy.deepcopy(event) for event in events]
    payload = build_event_log_payload(workspace_id=workspace_id, events=normalized_events)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        header = {
            "schema_name": payload["schema_name"],
            "schema_version": payload["schema_version"],
            "workspace_id": payload["workspace_id"],
            "generated_at_utc": payload["generated_at_utc"],
            "event_count": payload["event_count"],
        }
        handle.write(json.dumps(header, ensure_ascii=False) + "\n")
        for event in normalized_events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return payload


def read_event_log(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        raise ValueError(f"Event log `{path}` was empty.")

    header = json.loads(lines[0])
    if header.get("schema_name") != EVENT_LOG_SCHEMA_NAME:
        raise ValueError(f"Unexpected event log schema name in `{path}`.")
    if header.get("schema_version") != EVENT_LOG_SCHEMA_VERSION:
        raise ValueError(f"Unsupported event log schema version in `{path}`.")

    events = [json.loads(line) for line in lines[1:]]
    for event in events:
        if event.get("schema_name") != EVENT_SCHEMA_NAME:
            raise ValueError(f"Unexpected event schema name in `{path}`.")
        if event.get("schema_version") != EVENT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported event schema version in `{path}`.")

    payload = dict(header)
    payload["events"] = events
    return payload


def build_event_from_session_entry(
    *,
    root: Path | None,
    workspace_id: str,
    session: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    session_id = str(session.get("session_id") or "")
    session_surface = str(session.get("surface") or "")
    session_mode = _clean_text(session.get("current_mode"))
    model_id = _clean_text(entry.get("model_id") or session.get("selected_model_id"))
    recorded_at = _clean_text(entry.get("recorded_at_utc"))
    entry_id = _clean_text(entry.get("entry_id")) or f"{session_id}:{recorded_at or 'unknown'}"
    event_kind = _clean_text(entry.get("entry_kind")) or "session_event"

    artifact_ref = copy.deepcopy(entry.get("artifact_ref") or {})
    attached_assets = copy.deepcopy(entry.get("attached_assets") or [])
    notes = _clean_string_list(entry.get("notes"))
    validation_context = _artifact_validation_context(artifact_ref)
    notes.extend(_clean_string_list(validation_context.get("quality_notes")))

    session_manifest = session_manifest_path(
        session_id=session_id,
        workspace_id=workspace_id,
        root=resolved_root,
    )
    workspace_manifest = workspace_manifest_path(
        workspace_id=workspace_id,
        root=resolved_root,
    )

    workspace_record = {
        "workspace_id": workspace_id,
        "workspace_manifest_path": str(workspace_manifest),
    }
    session_record = {
        "session_id": session_id,
        "surface": session_surface,
        "mode": session_mode,
        "title": _clean_text(session.get("title")),
        "selected_model_id": _clean_text(session.get("selected_model_id")),
        "session_manifest_path": str(session_manifest),
    }
    outcome = {
        "status": _clean_text(entry.get("status")),
        "quality_status": _clean_text(validation_context.get("quality_status")),
        "execution_status": _clean_text(validation_context.get("execution_status"))
        or _clean_text(entry.get("status")),
        "message_count_before_turn": entry.get("message_count_before_turn"),
        "message_count_after_turn": entry.get("message_count_after_turn"),
    }
    entry_options = entry.get("options")
    options = copy.deepcopy(entry_options) if isinstance(entry_options, dict) else {}
    for key in (
        "validation_mode",
        "claim_scope",
        "pass_definition",
        "quality_status",
        "execution_status",
        "quality_checks",
    ):
        if key in validation_context and key not in options:
            options[key] = copy.deepcopy(validation_context[key])
    for key in (
        "validation_mode",
        "claim_scope",
        "pass_definition",
        "quality_status",
        "execution_status",
    ):
        if key in options:
            value = _clean_text(options.get(key))
            if value is None:
                options.pop(key, None)
            else:
                options[key] = value
    if "quality_checks" in options:
        quality_checks = _clean_quality_checks(options.get("quality_checks"))
        if quality_checks:
            options["quality_checks"] = quality_checks
        else:
            options.pop("quality_checks", None)
    content = {
        "prompt": _clean_text(entry.get("prompt")),
        "system_prompt": _clean_text(entry.get("system_prompt")),
        "resolved_user_prompt": _clean_text(entry.get("resolved_user_prompt")),
        "output_text": _clean_text(entry.get("output_text")),
        "notes": notes,
        "options": options,
    }
    source_refs = {
        "artifact_ref": artifact_ref,
        "attached_assets": attached_assets,
    }
    tags = [
        tag
        for tag in (
            session_surface,
            session_mode,
            _clean_text(outcome.get("status")),
            event_kind,
            model_id,
        )
        if isinstance(tag, str) and tag
    ]
    return build_event_record(
        event_id=f"{workspace_id}:{session_id}:{entry_id}",
        event_kind=event_kind,
        recorded_at_utc=recorded_at,
        workspace=workspace_record,
        session=session_record,
        outcome=outcome,
        content=content,
        source_refs=source_refs,
        tags=tags,
    )


def iter_workspace_events(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
) -> list[dict[str, Any]]:
    resolved_root = _resolve_root(root)
    manifest_path = workspace_manifest_path(workspace_id=workspace_id, root=resolved_root)
    if not manifest_path.exists():
        return []

    workspace = read_workspace_manifest(manifest_path)
    events: list[dict[str, Any]] = []
    seen_session_ids: set[str] = set()
    workspace_root = workspace_dir(workspace_id=workspace_id, root=resolved_root)

    for summary in workspace.get("sessions") or []:
        session_id = _clean_text(summary.get("session_id"))
        manifest_relative_path = _clean_text(summary.get("manifest_path"))
        if session_id is None or manifest_relative_path is None or session_id in seen_session_ids:
            continue
        seen_session_ids.add(session_id)
        session_path = workspace_root / manifest_relative_path
        if not session_path.exists():
            continue
        session = read_session_manifest(session_path)
        for entry in session.get("entries") or []:
            if not isinstance(entry, dict):
                continue
            events.append(
                build_event_from_session_entry(
                    root=resolved_root,
                    workspace_id=workspace_id,
                    session=session,
                    entry=entry,
                )
            )

    events.sort(key=lambda event: str(event.get("recorded_at_utc") or ""))
    return events


def iter_capability_matrix_events(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    errors: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    resolved_root = _resolve_root(root)
    matrix_root = capability_matrix_root(resolved_root)
    if not matrix_root.exists():
        return []

    events: list[dict[str, Any]] = []
    for path in sorted(matrix_root.glob("*.json")):
        try:
            payload = read_artifact(path)
        except Exception as exc:
            if errors is not None:
                errors.append(
                    {
                        "path": str(path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            continue
        if payload.get("artifact_kind") != "capability_matrix":
            continue
        for index, result in enumerate(payload.get("results") or []):
            if not isinstance(result, dict):
                continue
            events.append(
                build_event_from_capability_matrix_result(
                    root=resolved_root,
                    workspace_id=workspace_id,
                    matrix_artifact_path=path,
                    matrix_payload=payload,
                    result=result,
                    result_index=index,
                )
            )

    events.sort(key=lambda item: str(item.get("recorded_at_utc") or ""))
    return events


def iter_agent_lane_events(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    errors: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    resolved_root = _resolve_root(root)
    run_log = agent_run_log_path(workspace_id=workspace_id, root=resolved_root)
    try:
        runs = read_agent_runs(run_log)
    except Exception as exc:
        if errors is not None:
            errors.append(
                {
                    "path": str(run_log),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        return []

    events = [
        build_event_from_agent_run(
            root=resolved_root,
            workspace_id=workspace_id,
            run=run,
        )
        for run in runs
    ]
    events.sort(key=lambda item: str(item.get("recorded_at_utc") or ""))
    return events


def rebuild_workspace_event_log(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    output_path: Path | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    events = iter_workspace_events(root=resolved_root, workspace_id=workspace_id)
    path = output_path or workspace_event_log_path(workspace_id=workspace_id, root=resolved_root)
    payload = write_event_log(path, events, workspace_id=workspace_id)
    payload["path"] = str(path)
    return payload


def build_workspace_event_snapshot(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    store = WorkspaceSessionStore(root=resolved_root, workspace_id=workspace_id)
    diagnostics = store.diagnostics()
    events = iter_workspace_events(root=resolved_root, workspace_id=workspace_id)
    return {
        "workspace_id": workspace_id,
        "workspace_manifest_path": diagnostics["workspace_manifest_path"],
        "event_count": len(events),
        "events": events,
    }
