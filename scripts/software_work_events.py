#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

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


def _workspace_relative_path(path: Path, *, root: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return None


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
    output_preview = _clean_text(result.get("output_preview"))
    blocker = result.get("blocker") if isinstance(result.get("blocker"), dict) else {}
    blocker_message = _clean_text(blocker.get("message"))
    runtime = matrix_payload.get("runtime") if isinstance(matrix_payload.get("runtime"), dict) else {}
    recorded_at = _clean_text(matrix_payload.get("timestamp_utc")) or timestamp_utc()
    session_id = f"capability-matrix:{matrix_artifact_path.stem}"
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
    for item in result.get("quality_checks") or []:
        if not isinstance(item, dict):
            continue
        name = _clean_text(item.get("name")) or "quality_check"
        detail = _clean_text(item.get("detail"))
        passed = item.get("pass")
        if detail:
            notes.append(f"{name}: {'pass' if passed else 'fail'} - {detail}")
        else:
            notes.append(f"{name}: {'pass' if passed else 'fail'}")

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
            "entry_id": f"{matrix_artifact_path.stem}:{capability}",
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
            "validation_mode": validation_mode,
            "claim_scope": claim_scope,
            "pass_definition": _clean_text(result.get("pass_definition")),
            "matrix_artifact_path": str(matrix_artifact_path),
        },
    }
    return build_event_record(
        event_id=f"{workspace_id}:{session_id}:{matrix_artifact_path.stem}:{capability}",
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
        "message_count_before_turn": entry.get("message_count_before_turn"),
        "message_count_after_turn": entry.get("message_count_after_turn"),
    }
    content = {
        "prompt": _clean_text(entry.get("prompt")),
        "system_prompt": _clean_text(entry.get("system_prompt")),
        "resolved_user_prompt": _clean_text(entry.get("resolved_user_prompt")),
        "output_text": _clean_text(entry.get("output_text")),
        "notes": notes,
        "options": copy.deepcopy(entry.get("options") or {}),
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
) -> list[dict[str, Any]]:
    resolved_root = _resolve_root(root)
    matrix_root = capability_matrix_root(resolved_root)
    if not matrix_root.exists():
        return []

    events: list[dict[str, Any]] = []
    for path in sorted(matrix_root.glob("*.json")):
        try:
            payload = read_artifact(path)
        except Exception:
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
