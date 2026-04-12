#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from gemma_runtime import repo_root, timestamp_slug, timestamp_utc, write_json


WORKSPACE_SCHEMA_NAME = "gemma-lab-workspace"
SESSION_SCHEMA_NAME = "gemma-lab-session"
WORKSPACE_SCHEMA_VERSION = 1
SESSION_SCHEMA_VERSION = 1
DEFAULT_WORKSPACE_ID = "local-default"
SESSION_SURFACES = ("chat", "vision", "audio", "thinking")
DEFAULT_SESSION_IDS = {
    "chat": "chat-main",
    "vision": "vision-main",
    "audio": "audio-main",
    "thinking": "thinking-main",
}
DEFAULT_SESSION_TITLES = {
    "chat": "Chat Session",
    "vision": "Vision Session",
    "audio": "Audio Session",
    "thinking": "Thinking Session",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".aac"}


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def workspaces_root(root: Path | None = None) -> Path:
    return _resolve_root(root) / "artifacts" / "workspaces"


def workspace_dir(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return workspaces_root(root) / workspace_id


def workspace_manifest_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return workspace_dir(workspace_id=workspace_id, root=root) / "workspace.json"


def session_manifest_path(
    *,
    session_id: str,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return workspace_dir(workspace_id=workspace_id, root=root) / "sessions" / f"{session_id}.json"


def _validate_surface(surface: str) -> str:
    normalized = (surface or "").strip().lower()
    if normalized not in SESSION_SURFACES:
        raise ValueError(f"Unsupported workspace session surface `{surface}`.")
    return normalized


def _path_reference(path: str | Path | None, *, root: Path) -> dict[str, Any]:
    if path in (None, ""):
        return {
            "path": None,
            "workspace_relative_path": None,
        }

    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (root / resolved).resolve()
    else:
        resolved = resolved.resolve()

    reference = {"path": str(resolved)}
    try:
        reference["workspace_relative_path"] = str(resolved.relative_to(root))
    except ValueError:
        reference["workspace_relative_path"] = None
    return reference


def detect_asset_kind(path: str | Path | None) -> str | None:
    if path in (None, ""):
        return None
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    return "file"


def build_workspace_manifest(
    *,
    workspace_id: str,
    root_path: Path,
    created_at_utc: str | None = None,
    updated_at_utc: str | None = None,
    selected_model_id: str | None = None,
    active_session_ids: dict[str, str] | None = None,
    sessions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    timestamp = created_at_utc or timestamp_utc()
    return {
        "schema_name": WORKSPACE_SCHEMA_NAME,
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "root_path": str(root_path),
        "created_at_utc": timestamp,
        "updated_at_utc": updated_at_utc or timestamp,
        "selected_model_id": selected_model_id,
        "active_session_ids": copy.deepcopy(active_session_ids or {}),
        "sessions": copy.deepcopy(sessions or []),
    }


def build_session_manifest(
    *,
    workspace_id: str,
    session_id: str,
    surface: str,
    title: str,
    model_id: str | None = None,
    mode: str | None = None,
    created_at_utc: str | None = None,
    updated_at_utc: str | None = None,
    attached_assets: list[dict[str, Any]] | None = None,
    artifact_refs: list[dict[str, Any]] | None = None,
    entries: list[dict[str, Any]] | None = None,
    history_for_next_turn: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_surface = _validate_surface(surface)
    timestamp = created_at_utc or timestamp_utc()
    return {
        "schema_name": SESSION_SCHEMA_NAME,
        "schema_version": SESSION_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "session_id": session_id,
        "surface": normalized_surface,
        "title": title,
        "created_at_utc": timestamp,
        "updated_at_utc": updated_at_utc or timestamp,
        "selected_model_id": model_id,
        "current_mode": mode,
        "attached_assets": copy.deepcopy(attached_assets or []),
        "artifact_refs": copy.deepcopy(artifact_refs or []),
        "entries": copy.deepcopy(entries or []),
        "history_for_next_turn": copy.deepcopy(history_for_next_turn or []),
    }


def _read_schema(
    path: Path,
    *,
    expected_name: str,
    expected_version: int,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_name") != expected_name:
        raise ValueError(f"Unexpected schema name in `{path}`.")
    if payload.get("schema_version") != expected_version:
        raise ValueError(f"Unsupported schema version in `{path}`.")
    return payload


def read_workspace_manifest(path: Path) -> dict[str, Any]:
    return _read_schema(
        path,
        expected_name=WORKSPACE_SCHEMA_NAME,
        expected_version=WORKSPACE_SCHEMA_VERSION,
    )


def read_session_manifest(path: Path) -> dict[str, Any]:
    return _read_schema(
        path,
        expected_name=SESSION_SCHEMA_NAME,
        expected_version=SESSION_SCHEMA_VERSION,
    )


def _normalize_chat_messages(messages: Any) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if not isinstance(messages, list):
        return normalized
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = item.get("content")
        if not role or not isinstance(content, str) or not content.strip():
            continue
        normalized.append({"role": role, "content": content.strip()})
    return normalized


class WorkspaceSessionStore:
    def __init__(
        self,
        *,
        root: Path | None = None,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> None:
        self.root = _resolve_root(root)
        self.workspace_id = workspace_id
        self._workspace_dir = workspace_dir(workspace_id=workspace_id, root=self.root)
        self._workspace_manifest_path = workspace_manifest_path(
            workspace_id=workspace_id,
            root=self.root,
        )

    @property
    def manifest_path(self) -> Path:
        return self._workspace_manifest_path

    def ensure_workspace(self, *, selected_model_id: str | None = None) -> dict[str, Any]:
        if self._workspace_manifest_path.exists():
            workspace = read_workspace_manifest(self._workspace_manifest_path)
        else:
            workspace = build_workspace_manifest(
                workspace_id=self.workspace_id,
                root_path=self.root,
                selected_model_id=selected_model_id,
            )
            self._write_workspace(workspace)
            return workspace

        if selected_model_id and workspace.get("selected_model_id") != selected_model_id:
            workspace["selected_model_id"] = selected_model_id
            workspace["updated_at_utc"] = timestamp_utc()
            self._write_workspace(workspace)
        return workspace

    def selected_model_id(self) -> str | None:
        workspace = self.ensure_workspace()
        model_id = workspace.get("selected_model_id")
        if isinstance(model_id, str) and model_id.strip():
            return model_id.strip()
        return None

    def set_selected_model(self, model_id: str) -> dict[str, Any]:
        workspace = self.ensure_workspace()
        normalized_model_id = (model_id or "").strip() or None
        workspace["selected_model_id"] = normalized_model_id
        workspace["updated_at_utc"] = timestamp_utc()
        self._write_workspace(workspace)
        return workspace

    def active_session(self, surface: str) -> dict[str, Any] | None:
        normalized_surface = _validate_surface(surface)
        workspace = self.ensure_workspace()
        session_id = (workspace.get("active_session_ids") or {}).get(normalized_surface)
        if not isinstance(session_id, str) or not session_id.strip():
            return None
        path = session_manifest_path(
            session_id=session_id,
            workspace_id=self.workspace_id,
            root=self.root,
        )
        if not path.exists():
            return None
        return read_session_manifest(path)

    def workspace_sessions(self) -> list[dict[str, Any]]:
        workspace = self.ensure_workspace()
        return copy.deepcopy(workspace.get("sessions") or [])

    def ensure_session(
        self,
        *,
        surface: str,
        model_id: str | None,
        mode: str | None,
        title: str | None = None,
    ) -> dict[str, Any]:
        normalized_surface = _validate_surface(surface)
        workspace = self.ensure_workspace(selected_model_id=model_id)
        session_id = (workspace.get("active_session_ids") or {}).get(normalized_surface)
        if not isinstance(session_id, str) or not session_id.strip():
            session_id = DEFAULT_SESSION_IDS[normalized_surface]

        path = session_manifest_path(
            session_id=session_id,
            workspace_id=self.workspace_id,
            root=self.root,
        )
        if path.exists():
            session = read_session_manifest(path)
        else:
            session = build_session_manifest(
                workspace_id=self.workspace_id,
                session_id=session_id,
                surface=normalized_surface,
                title=title or DEFAULT_SESSION_TITLES[normalized_surface],
                model_id=model_id,
                mode=mode,
            )

        session["title"] = title or session.get("title") or DEFAULT_SESSION_TITLES[normalized_surface]
        session["selected_model_id"] = model_id
        session["current_mode"] = mode
        session["updated_at_utc"] = timestamp_utc()
        self._write_session(session)
        self._sync_workspace_session_index(workspace, session)
        return session

    def chat_messages_for_next_turn(
        self,
        *,
        model_id: str | None,
        system_prompt: str,
    ) -> list[dict[str, str]]:
        session = self.ensure_session(
            surface="chat",
            model_id=model_id,
            mode="chat",
        )
        history = _normalize_chat_messages(session.get("history_for_next_turn"))
        expected_system_message = {"role": "system", "content": system_prompt}
        if not history or history[0] != expected_system_message:
            return [expected_system_message]
        return history

    def record_chat_turn(
        self,
        *,
        model_id: str | None,
        status: str,
        artifact_path: Path,
        prompt: str,
        system_prompt: str,
        resolved_user_prompt: str,
        output_text: str,
        base_messages: list[dict[str, Any]],
        notes: list[str] | None = None,
    ) -> dict[str, Any]:
        session = self.ensure_session(
            surface="chat",
            model_id=model_id,
            mode="chat",
        )
        recorded_at = timestamp_utc()
        entry_id = self._entry_id("chat-turn", session)
        artifact_ref = self._build_artifact_ref(
            artifact_path=artifact_path,
            artifact_kind="text",
            action="chat",
            status=status,
            entry_id=entry_id,
            recorded_at_utc=recorded_at,
        )
        entry = {
            "entry_id": entry_id,
            "recorded_at_utc": recorded_at,
            "entry_kind": "chat_turn",
            "mode": "chat",
            "status": status,
            "model_id": model_id,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "resolved_user_prompt": resolved_user_prompt,
            "output_text": output_text,
            "artifact_ref": copy.deepcopy(artifact_ref),
            "notes": list(notes or []),
            "message_count_before_turn": len(_normalize_chat_messages(base_messages)),
        }
        session["artifact_refs"].append(artifact_ref)
        session["entries"].append(entry)
        if status == "ok":
            next_history = _normalize_chat_messages(base_messages) + [
                {"role": "user", "content": resolved_user_prompt},
                {"role": "assistant", "content": output_text},
            ]
            session["history_for_next_turn"] = next_history
            entry["message_count_after_turn"] = len(next_history)
        else:
            session["history_for_next_turn"] = _normalize_chat_messages(base_messages)
            entry["message_count_after_turn"] = len(session["history_for_next_turn"])
        session["selected_model_id"] = model_id
        session["current_mode"] = "chat"
        session["updated_at_utc"] = recorded_at
        return self._save_session_and_workspace(session)

    def record_session_run(
        self,
        *,
        surface: str,
        model_id: str | None,
        mode: str | None,
        artifact_kind: str,
        artifact_path: Path,
        status: str,
        prompt: str | None,
        system_prompt: str | None,
        resolved_user_prompt: str | None,
        output_text: str,
        attachments: list[dict[str, Any]] | None = None,
        notes: list[str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_surface = _validate_surface(surface)
        session = self.ensure_session(
            surface=normalized_surface,
            model_id=model_id,
            mode=mode,
        )
        recorded_at = timestamp_utc()
        entry_id = self._entry_id(normalized_surface, session)
        attached_assets = self._build_attached_assets(attachments, recorded_at_utc=recorded_at)
        artifact_ref = self._build_artifact_ref(
            artifact_path=artifact_path,
            artifact_kind=artifact_kind,
            action=normalized_surface,
            status=status,
            entry_id=entry_id,
            recorded_at_utc=recorded_at,
        )
        entry = {
            "entry_id": entry_id,
            "recorded_at_utc": recorded_at,
            "entry_kind": f"{normalized_surface}_run",
            "mode": mode,
            "status": status,
            "model_id": model_id,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "resolved_user_prompt": resolved_user_prompt,
            "output_text": output_text,
            "attached_assets": copy.deepcopy(attached_assets),
            "artifact_ref": copy.deepcopy(artifact_ref),
            "notes": list(notes or []),
            "options": copy.deepcopy(options or {}),
        }
        session["attached_assets"] = attached_assets
        session["artifact_refs"].append(artifact_ref)
        session["entries"].append(entry)
        session["selected_model_id"] = model_id
        session["current_mode"] = mode
        session["updated_at_utc"] = recorded_at
        return self._save_session_and_workspace(session)

    def diagnostics(self) -> dict[str, Any]:
        workspace = self.ensure_workspace()
        return {
            "workspace_id": workspace["workspace_id"],
            "workspace_manifest_path": str(self._workspace_manifest_path),
            "workspace_state_root": str(self._workspace_dir),
            "selected_model_id": workspace.get("selected_model_id"),
            "active_session_ids": copy.deepcopy(workspace.get("active_session_ids") or {}),
            "sessions": copy.deepcopy(workspace.get("sessions") or []),
        }

    def _entry_id(self, suffix: str, session: dict[str, Any]) -> str:
        return f"{timestamp_slug()}-{suffix}-{len(session.get('entries') or []) + 1:04d}"

    def _write_workspace(self, workspace: dict[str, Any]) -> None:
        write_json(self._workspace_manifest_path, workspace)

    def _write_session(self, session: dict[str, Any]) -> None:
        path = session_manifest_path(
            session_id=str(session["session_id"]),
            workspace_id=self.workspace_id,
            root=self.root,
        )
        write_json(path, session)

    def _save_session_and_workspace(self, session: dict[str, Any]) -> dict[str, Any]:
        workspace = self.ensure_workspace(selected_model_id=session.get("selected_model_id"))
        self._write_session(session)
        self._sync_workspace_session_index(workspace, session)
        return session

    def _sync_workspace_session_index(
        self,
        workspace: dict[str, Any],
        session: dict[str, Any],
    ) -> None:
        surface = _validate_surface(str(session["surface"]))
        latest_artifact_ref: dict[str, Any] | None = None
        latest_artifact_path: str | None = None
        artifact_refs = list(session.get("artifact_refs") or [])
        if artifact_refs:
            latest_artifact_ref = dict(artifact_refs[-1])
            latest_artifact_path = latest_artifact_ref.get("artifact_path")
        workspace.setdefault("active_session_ids", {})
        workspace["active_session_ids"][surface] = str(session["session_id"])
        workspace["selected_model_id"] = session.get("selected_model_id") or workspace.get("selected_model_id")

        summary = {
            "session_id": session["session_id"],
            "surface": session["surface"],
            "title": session["title"],
            "manifest_path": str(
                session_manifest_path(
                    session_id=str(session["session_id"]),
                    workspace_id=self.workspace_id,
                    root=self.root,
                ).relative_to(self._workspace_dir)
            ),
            "selected_model_id": session.get("selected_model_id"),
            "current_mode": session.get("current_mode"),
            "updated_at_utc": session.get("updated_at_utc"),
            "entry_count": len(session.get("entries") or []),
            "artifact_count": len(session.get("artifact_refs") or []),
            "latest_artifact_path": latest_artifact_path,
            "latest_artifact_workspace_relative_path": (
                latest_artifact_ref.get("artifact_workspace_relative_path")
                if latest_artifact_ref is not None
                else None
            ),
            "latest_artifact_status": (
                latest_artifact_ref.get("status") if latest_artifact_ref is not None else None
            ),
        }

        sessions = [
            item
            for item in copy.deepcopy(workspace.get("sessions") or [])
            if item.get("session_id") != session["session_id"]
        ]
        sessions.append(summary)
        sessions.sort(key=lambda item: str(item.get("updated_at_utc") or ""), reverse=True)
        workspace["sessions"] = sessions
        workspace["updated_at_utc"] = timestamp_utc()
        self._write_workspace(workspace)

    def _build_attached_assets(
        self,
        attachments: list[dict[str, Any]] | None,
        *,
        recorded_at_utc: str,
    ) -> list[dict[str, Any]]:
        attached_assets: list[dict[str, Any]] = []
        for attachment in attachments or []:
            if not isinstance(attachment, dict):
                continue
            path = attachment.get("path")
            reference = _path_reference(path, root=self.root)
            if reference["path"] is None:
                continue
            attached_assets.append(
                {
                    "role": str(attachment.get("role") or "attachment"),
                    "kind": str(attachment.get("kind") or detect_asset_kind(path) or "file"),
                    "path": reference["path"],
                    "workspace_relative_path": reference["workspace_relative_path"],
                    "added_at_utc": recorded_at_utc,
                }
            )
        return attached_assets

    def _build_artifact_ref(
        self,
        *,
        artifact_path: Path,
        artifact_kind: str,
        action: str,
        status: str,
        entry_id: str,
        recorded_at_utc: str,
    ) -> dict[str, Any]:
        reference = _path_reference(artifact_path, root=self.root)
        return {
            "entry_id": entry_id,
            "artifact_kind": artifact_kind,
            "action": action,
            "status": status,
            "recorded_at_utc": recorded_at_utc,
            "artifact_path": reference["path"],
            "artifact_workspace_relative_path": reference["workspace_relative_path"],
        }
