#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

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
