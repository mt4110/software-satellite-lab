#!/usr/bin/env python3
from __future__ import annotations

from contextlib import contextmanager
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator

from gemma_runtime import repo_root
from software_work_events import (
    build_event_contract_report,
    iter_agent_lane_events,
    iter_capability_matrix_events,
    iter_workspace_events,
    write_event_log,
    workspace_event_log_path,
)
from workspace_state import DEFAULT_WORKSPACE_ID


INDEX_SCHEMA_VERSION = 3


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _quality_check_verdict(value: bool) -> str:
    return "pass" if value else "fail"


def _quality_check_terms(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    terms: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        check_name = _clean_text(item.get("name"))
        passed = item.get("pass")
        if check_name is None or not isinstance(passed, bool):
            continue
        check_detail = _clean_text(item.get("detail"))
        check_verdict = _quality_check_verdict(passed)
        term = " ".join(
            part
            for part in (check_name, check_verdict, check_detail)
            if part
        )
        if term:
            terms.append(term)
    return terms


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def memory_index_root(root: Path | None = None) -> Path:
    return _resolve_root(root) / "artifacts" / "indexes"


def default_memory_index_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return memory_index_root(root) / f"{workspace_id}-software-memory-v1.sqlite3"


def _event_row(event: dict[str, Any]) -> dict[str, Any]:
    session = _dict_or_empty(event.get("session"))
    content = _dict_or_empty(event.get("content"))
    options = _dict_or_empty(content.get("options"))
    outcome = _dict_or_empty(event.get("outcome"))
    workspace = _dict_or_empty(event.get("workspace"))
    source_refs = _dict_or_empty(event.get("source_refs"))
    artifact_ref = _dict_or_empty(source_refs.get("artifact_ref"))
    notes = content.get("notes") or []
    quality_status = _clean_text(outcome.get("quality_status"))
    execution_status = _clean_text(outcome.get("execution_status"))
    evaluation_terms = [
        quality_status,
        execution_status,
        _clean_text(options.get("validation_mode")),
        _clean_text(options.get("validation_command")),
        _clean_text(options.get("claim_scope")),
        _clean_text(options.get("backend_id")),
        _clean_text(options.get("backend_adapter_kind")),
        _clean_text(options.get("backend_compatibility_status")),
    ]
    evaluation_terms.extend(_quality_check_terms(options.get("quality_checks")))
    return {
        "event_id": str(event.get("event_id") or ""),
        "recorded_at_utc": event.get("recorded_at_utc"),
        "workspace_id": workspace.get("workspace_id"),
        "session_id": session.get("session_id"),
        "session_surface": session.get("surface"),
        "session_mode": session.get("mode"),
        "model_id": session.get("selected_model_id") or outcome.get("model_id"),
        "event_kind": event.get("event_kind"),
        "status": outcome.get("status"),
        "quality_status": quality_status,
        "execution_status": execution_status,
        "prompt": content.get("prompt"),
        "output_text": content.get("output_text"),
        "notes_text": "\n".join(item for item in notes if isinstance(item, str)),
        "pass_definition": _clean_text(options.get("pass_definition")),
        "evaluation_signal_text": "\n".join(item for item in evaluation_terms if item),
        "artifact_path": artifact_ref.get("artifact_path"),
        "payload_json": json.dumps(event, ensure_ascii=False),
    }


class MemoryIndex:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _schema_version(self, connection: sqlite3.Connection) -> int | None:
        row = connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            return None
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return None

    def ensure_schema(self) -> None:
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL;")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            if self._schema_version(connection) != INDEX_SCHEMA_VERSION:
                # The index is a rebuildable cache sourced from manifests and event logs.
                connection.executescript(
                    """
                    DROP TABLE IF EXISTS events_fts;
                    DROP TABLE IF EXISTS events;
                    """
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    recorded_at_utc TEXT,
                    workspace_id TEXT,
                    session_id TEXT,
                    session_surface TEXT,
                    session_mode TEXT,
                    model_id TEXT,
                    event_kind TEXT,
                    status TEXT,
                    quality_status TEXT,
                    execution_status TEXT,
                    prompt TEXT,
                    output_text TEXT,
                    notes_text TEXT,
                    pass_definition TEXT,
                    evaluation_signal_text TEXT,
                    artifact_path TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
                    event_id UNINDEXED,
                    prompt,
                    output_text,
                    notes_text,
                    pass_definition,
                    artifact_path,
                    event_kind,
                    session_surface,
                    session_mode,
                    model_id,
                    status,
                    quality_status,
                    execution_status,
                    evaluation_signal_text,
                    tokenize = 'unicode61'
                );
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(INDEX_SCHEMA_VERSION),),
            )

    def clear(self) -> None:
        self.ensure_schema()
        with self.connection() as connection:
            connection.execute("DELETE FROM events")
            connection.execute("DELETE FROM events_fts")

    def index_events(self, events: list[dict[str, Any]]) -> int:
        self.ensure_schema()
        with self.connection() as connection:
            connection.execute("DELETE FROM events")
            connection.execute("DELETE FROM events_fts")
            for event in events:
                row = _event_row(event)
                connection.execute(
                    """
                    INSERT OR REPLACE INTO events (
                        event_id,
                        recorded_at_utc,
                        workspace_id,
                        session_id,
                        session_surface,
                        session_mode,
                        model_id,
                        event_kind,
                        status,
                        quality_status,
                        execution_status,
                        prompt,
                        output_text,
                        notes_text,
                        pass_definition,
                        evaluation_signal_text,
                        artifact_path,
                        payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["event_id"],
                        row["recorded_at_utc"],
                        row["workspace_id"],
                        row["session_id"],
                        row["session_surface"],
                        row["session_mode"],
                        row["model_id"],
                        row["event_kind"],
                        row["status"],
                        row["quality_status"],
                        row["execution_status"],
                        row["prompt"],
                        row["output_text"],
                        row["notes_text"],
                        row["pass_definition"],
                        row["evaluation_signal_text"],
                        row["artifact_path"],
                        row["payload_json"],
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO events_fts (
                        event_id,
                        prompt,
                        output_text,
                        notes_text,
                        pass_definition,
                        artifact_path,
                        event_kind,
                        session_surface,
                        session_mode,
                        model_id,
                        status,
                        quality_status,
                        execution_status,
                        evaluation_signal_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["event_id"],
                        row["prompt"],
                        row["output_text"],
                        row["notes_text"],
                        row["pass_definition"],
                        row["artifact_path"],
                        row["event_kind"],
                        row["session_surface"],
                        row["session_mode"],
                        row["model_id"],
                        row["status"],
                        row["quality_status"],
                        row["execution_status"],
                        row["evaluation_signal_text"],
                    ),
                )
        return len(events)

    def search(
        self,
        query: str | None = None,
        *,
        limit: int = 10,
        surface: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        self.ensure_schema()
        if limit < 1:
            return []

        filters: list[str] = []
        parameters: list[Any] = []
        if surface:
            filters.append("e.session_surface = ?")
            parameters.append(surface)
        if status:
            filters.append("e.status = ?")
            parameters.append(status)
        filter_sql = ""
        if filters:
            filter_sql = " AND " + " AND ".join(filters)

        with self.connection() as connection:
            if query and query.strip():
                sql = f"""
                    SELECT
                        e.event_id,
                        e.recorded_at_utc,
                        e.workspace_id,
                        e.session_id,
                        e.session_surface,
                        e.session_mode,
                        e.model_id,
                        e.event_kind,
                        e.status,
                        e.quality_status,
                        e.execution_status,
                        e.prompt,
                        e.output_text,
                        e.notes_text,
                        e.pass_definition,
                        e.evaluation_signal_text,
                        e.artifact_path,
                        bm25(events_fts) AS score
                    FROM events_fts
                    JOIN events e ON e.event_id = events_fts.event_id
                    WHERE events_fts MATCH ?{filter_sql}
                    ORDER BY score, e.recorded_at_utc DESC, e.event_id DESC
                    LIMIT ?
                """
                rows = connection.execute(sql, [query.strip(), *parameters, limit]).fetchall()
            else:
                sql = f"""
                    SELECT
                        e.event_id,
                        e.recorded_at_utc,
                        e.workspace_id,
                        e.session_id,
                        e.session_surface,
                        e.session_mode,
                        e.model_id,
                        e.event_kind,
                        e.status,
                        e.quality_status,
                        e.execution_status,
                        e.prompt,
                        e.output_text,
                        e.notes_text,
                        e.pass_definition,
                        e.evaluation_signal_text,
                        e.artifact_path,
                        NULL AS score
                    FROM events e
                    WHERE 1 = 1{filter_sql}
                    ORDER BY e.recorded_at_utc DESC, e.event_id DESC
                    LIMIT ?
                """
                rows = connection.execute(sql, [*parameters, limit]).fetchall()
        return [dict(row) for row in rows]

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        self.ensure_schema()
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    event_id,
                    recorded_at_utc,
                    workspace_id,
                    session_id,
                    session_surface,
                    session_mode,
                    model_id,
                    event_kind,
                    status,
                    quality_status,
                    execution_status,
                    prompt,
                    output_text,
                    notes_text,
                    pass_definition,
                    evaluation_signal_text,
                    artifact_path,
                    payload_json
                FROM events
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(row)


def rebuild_memory_index(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    index_path: Path | None = None,
    event_log_path: Path | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    workspace_events = iter_workspace_events(root=resolved_root, workspace_id=workspace_id)
    capability_matrix_errors: list[dict[str, str]] = []
    matrix_events = iter_capability_matrix_events(
        root=resolved_root,
        workspace_id=workspace_id,
        errors=capability_matrix_errors,
    )
    agent_lane_errors: list[dict[str, str]] = []
    agent_lane_events = iter_agent_lane_events(
        root=resolved_root,
        workspace_id=workspace_id,
        errors=agent_lane_errors,
    )
    events = sorted(
        [*workspace_events, *matrix_events, *agent_lane_events],
        key=lambda item: (
            str(item.get("recorded_at_utc") or ""),
            str(item.get("event_id") or ""),
        ),
    )
    target_event_log_path = event_log_path or workspace_event_log_path(
        workspace_id=workspace_id,
        root=resolved_root,
    )
    log_payload = write_event_log(target_event_log_path, events, workspace_id=workspace_id)
    log_payload["path"] = str(target_event_log_path)
    event_contract = build_event_contract_report(events, root=resolved_root)
    target_index_path = index_path or default_memory_index_path(workspace_id=workspace_id, root=resolved_root)
    index = MemoryIndex(target_index_path)
    indexed_count = index.index_events(events)
    return {
        "workspace_id": workspace_id,
        "workspace_event_count": len(workspace_events),
        "capability_event_count": len(matrix_events),
        "capability_matrix_error_count": len(capability_matrix_errors),
        "capability_matrix_errors": capability_matrix_errors,
        "agent_lane_event_count": len(agent_lane_events),
        "agent_lane_error_count": len(agent_lane_errors),
        "agent_lane_errors": agent_lane_errors,
        "event_count": len(events),
        "indexed_count": indexed_count,
        "event_log_path": log_payload["path"],
        "index_path": str(target_index_path),
        "event_contract": event_contract,
    }
