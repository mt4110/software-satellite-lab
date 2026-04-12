#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import tkinter as tk
    from tkinter import filedialog, scrolledtext, ttk
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local Python build
    tk = None  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]
    scrolledtext = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    _TK_IMPORT_ERROR = exc
else:
    _TK_IMPORT_ERROR = None

from artifact_schema import (
    build_artifact_payload,
    build_prompt_record,
    build_runtime_record,
    collect_asset_lineage,
    normalize_device_info,
    read_artifact,
    write_artifact,
)
from audio_service import (
    DEFAULT_TARGET_LANGUAGE,
    default_output_path as default_audio_output_path,
    resolve_prompt as resolve_audio_prompt,
    resolve_system_prompt as resolve_audio_system_prompt,
    run_audio_mode,
    serialize_audio_record,
)
from doctor import assets_summary, probe_optional_modules, probe_torch, probe_transformers
from gemma_core import CancellationSignal, SessionManager
from gemma_runtime import (
    UserFacingError,
    default_thinking_artifact_path,
    repo_root,
    resolve_audio_model_selection,
    resolve_model_id,
    WarmupProgress,
    WARMUP_PHASE_ATTACH_MODEL,
    WARMUP_PHASE_LOAD_MODEL,
    WARMUP_PHASE_LOAD_PROCESSOR,
    WARMUP_PHASE_PRIME_TOKEN,
    WARMUP_PHASE_SESSION_READY,
)
from text_service import (
    DEFAULT_PROMPTS as TEXT_DEFAULT_PROMPTS,
    GENERATION_SETTINGS as TEXT_GENERATION_SETTINGS,
    build_user_prompt as build_text_user_prompt,
    default_output_path as default_text_output_path,
    resolve_system_prompt as resolve_text_system_prompt,
    run_text_task,
)
from thinking_service import (
    TEXT_FOLLOW_UP,
    TEXT_PROMPT,
    TEXT_SYSTEM_PROMPT,
    TOOL_PROMPT,
    TOOL_SYSTEM_PROMPT,
    run_thinking_session,
    warm_thinking_session,
)
from vision_service import (
    DEFAULT_PROMPTS as VISION_DEFAULT_PROMPTS,
    GENERATION_SETTINGS as VISION_GENERATION_SETTINGS,
    build_user_prompt as build_vision_user_prompt,
    default_output_path as default_vision_output_path,
    resolve_prompt as resolve_vision_prompt,
    resolve_system_prompt as resolve_vision_system_prompt,
    run_vision_mode,
    serialize_input_records,
)
from workspace_state import WorkspaceSessionStore
from workspace_state import (
    SESSION_SURFACES,
    read_session_manifest,
    read_workspace_manifest,
    session_manifest_path,
)


CHAT_BACKEND = "gemma-live-text"
VISION_BACKEND = "gemma-live-vision"
AUDIO_BACKEND = "gemma-live-audio"
THINKING_BACKEND = "gemma-live-thinking"
PREWARM_BACKEND = "gemma-runtime-prewarm"
DEFAULT_UI_TITLE = "Gemma 4 Capability Lab"
DEFAULT_MODEL_CHOICES = [
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
]
CHAT_OUTPUT_LIMIT = 12000
JOB_STATE_QUEUED = "queued"
JOB_STATE_RUNNING = "running"
JOB_STATE_COMPLETED = "completed"
JOB_STATE_FAILED = "failed"
JOB_STATE_CANCELLED = "cancelled"
JOB_POLL_INTERVAL_MS = 120
STARTUP_PREWARM_GRACE_MS = 900
STATUS_ANIMATION_INTERVAL_MS = 240
PREWARM_ACTION = "prewarm"
PREWARM_PHASE_LABELS = {
    WARMUP_PHASE_LOAD_PROCESSOR: "Loading processor assets",
    WARMUP_PHASE_LOAD_MODEL: "Loading model weights",
    WARMUP_PHASE_ATTACH_MODEL: "Attaching model to Apple Metal",
    WARMUP_PHASE_SESSION_READY: "Shared session ready",
    WARMUP_PHASE_PRIME_TOKEN: "Priming first thinking token",
}
PREWARM_DEFERRED_CANCEL_PHASES = {
    WARMUP_PHASE_LOAD_PROCESSOR,
    WARMUP_PHASE_LOAD_MODEL,
    WARMUP_PHASE_ATTACH_MODEL,
}
DEFAULT_ACTION_TIMEOUT_SECONDS = {
    "chat": 120.0,
    "vision": 240.0,
    "audio": 240.0,
    "thinking": 300.0,
    PREWARM_ACTION: 180.0,
}
TK_PREVIEWABLE_IMAGE_SUFFIXES = {".png", ".gif", ".ppm", ".pgm"}
PREVIEW_MAX_WIDTH = 320
PREVIEW_MAX_HEIGHT = 220
JobProgressValue = str | WarmupProgress
JobProgressCallback = Callable[[JobProgressValue], None]


def normalize_job_progress(update: JobProgressValue) -> tuple[str | None, str]:
    if isinstance(update, WarmupProgress):
        return update.phase, update.message.strip()
    return None, str(update).strip()


@dataclass
class UiActionResult:
    action: str
    status: str
    output_text: str
    artifact_path: Path
    model_id: str | None
    backend: str
    device_label: str | None
    dtype_name: str | None
    debug_text: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class UiJobSnapshot:
    job_id: int
    action: str
    state: str
    submitted_at: float
    started_at: float | None
    finished_at: float | None
    timeout_seconds: float | None
    cancel_requested: bool
    deadline_exceeded: bool
    progress_phase: str | None
    message: str
    result: UiActionResult | None


@dataclass
class _UiJobRecord:
    job_id: int
    action: str
    work: Callable[[CancellationSignal, JobProgressCallback], UiActionResult]
    timeout_seconds: float | None
    submitted_at: float
    cancel_signal: CancellationSignal = field(default_factory=CancellationSignal)
    started_at: float | None = None
    finished_at: float | None = None
    state: str = JOB_STATE_QUEUED
    cancel_requested: bool = False
    deadline_exceeded: bool = False
    progress_phase: str | None = None
    message: str = ""
    result: UiActionResult | None = None


class LocalUiJobRunner:
    def __init__(self) -> None:
        self._jobs: dict[int, _UiJobRecord] = {}
        self._job_order: list[int] = []
        self._events: deque[UiJobSnapshot] = deque()
        self._queue: queue.Queue[int | None] = queue.Queue()
        self._lock = threading.Lock()
        self._next_job_id = 1
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="gemma-local-ui-worker",
            daemon=True,
        )
        self._worker.start()

    def submit(
        self,
        *,
        action: str,
        work: Callable[[CancellationSignal, JobProgressCallback], UiActionResult],
        timeout_seconds: float | None,
        message: str,
    ) -> UiJobSnapshot:
        now = time.monotonic()
        with self._lock:
            job_id = self._next_job_id
            self._next_job_id += 1
            record = _UiJobRecord(
                job_id=job_id,
                action=action,
                work=work,
                timeout_seconds=timeout_seconds,
                submitted_at=now,
                message=message,
            )
            self._jobs[job_id] = record
            self._job_order.append(job_id)
            snapshot = self._snapshot_locked(record)
            self._events.append(snapshot)
        self._queue.put(job_id)
        return snapshot

    def has_pending_work(self) -> bool:
        with self._lock:
            return any(
                record.state in {JOB_STATE_QUEUED, JOB_STATE_RUNNING}
                for record in self._jobs.values()
            )

    def active_job(self) -> UiJobSnapshot | None:
        with self._lock:
            for state in (JOB_STATE_RUNNING, JOB_STATE_QUEUED):
                for job_id in self._job_order:
                    record = self._jobs.get(job_id)
                    if record is not None and record.state == state:
                        return self._snapshot_locked(record)
        return None

    def cancel_active(self) -> UiJobSnapshot | None:
        with self._lock:
            target: _UiJobRecord | None = None
            for state in (JOB_STATE_RUNNING, JOB_STATE_QUEUED):
                for job_id in self._job_order:
                    record = self._jobs.get(job_id)
                    if record is not None and record.state == state:
                        target = record
                        break
                if target is not None:
                    break

            if target is None:
                return None
            return self._cancel_locked(target)

    def cancel(self, job_id: int) -> UiJobSnapshot | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            return self._cancel_locked(record)

    def _cancel_locked(self, target: _UiJobRecord) -> UiJobSnapshot | None:
        if target.state not in {JOB_STATE_QUEUED, JOB_STATE_RUNNING}:
            return None
        now = time.monotonic()
        target.cancel_requested = True
        target.cancel_signal.request_cancel()
        if target.state == JOB_STATE_QUEUED:
            target.state = JOB_STATE_CANCELLED
            target.finished_at = now
            target.message = f"{target.action} job cancelled before it started."
        snapshot = self._snapshot_locked(target)
        self._events.append(snapshot)
        return snapshot

    def expire_timeouts(self) -> list[UiJobSnapshot]:
        now = time.monotonic()
        expired: list[UiJobSnapshot] = []
        with self._lock:
            for job_id in self._job_order:
                record = self._jobs.get(job_id)
                if record is None or record.timeout_seconds is None or record.deadline_exceeded:
                    continue

                if record.state == JOB_STATE_QUEUED and now - record.submitted_at > record.timeout_seconds:
                    record.deadline_exceeded = True
                    record.state = JOB_STATE_FAILED
                    record.finished_at = now
                    record.message = (
                        f"{record.action} job expired before it started "
                        f"(timeout {record.timeout_seconds:.0f}s)."
                    )
                    snapshot = self._snapshot_locked(record)
                    self._events.append(snapshot)
                    expired.append(snapshot)
                    continue

                if (
                    record.state == JOB_STATE_RUNNING
                    and record.started_at is not None
                    and now - record.started_at > record.timeout_seconds
                ):
                    record.deadline_exceeded = True
                    record.message = (
                        f"{record.action} exceeded the {record.timeout_seconds:.0f}s timeout. "
                        "The UI will discard the result when the current run returns."
                    )
                    snapshot = self._snapshot_locked(record)
                    self._events.append(snapshot)
                    expired.append(snapshot)
        return expired

    def pop_events(self) -> list[UiJobSnapshot]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events

    def close(self) -> None:
        self._queue.put(None)
        self._worker.join(timeout=0.2)

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return

            with self._lock:
                record = self._jobs.get(job_id)
                if record is None:
                    continue
                if record.state == JOB_STATE_CANCELLED:
                    continue
                if record.state == JOB_STATE_FAILED and record.deadline_exceeded:
                    continue
                record.state = JOB_STATE_RUNNING
                record.started_at = time.monotonic()
                record.message = f"Running {record.action} on the shared core..."
                self._events.append(self._snapshot_locked(record))

            result: UiActionResult | None = None
            error_message: str | None = None
            progress_callback = self._build_progress_callback(job_id)
            try:
                result = record.work(record.cancel_signal, progress_callback)
            except Exception as exc:  # pragma: no cover - defensive guard around worker task
                error_message = f"{type(exc).__name__}: {exc}"

            finished_at = time.monotonic()
            with self._lock:
                latest = self._jobs.get(job_id)
                if latest is None:
                    continue
                latest.finished_at = finished_at

                if latest.cancel_requested:
                    latest.state = JOB_STATE_CANCELLED
                    latest.result = None
                    latest.message = (
                        f"{latest.action} cancellation completed. "
                        "The finished result was discarded at a safe boundary."
                    )
                elif latest.deadline_exceeded or (
                    latest.timeout_seconds is not None
                    and latest.started_at is not None
                    and finished_at - latest.started_at > latest.timeout_seconds
                ):
                    latest.deadline_exceeded = True
                    latest.state = JOB_STATE_FAILED
                    latest.result = None
                    latest.message = (
                        f"{latest.action} timed out after {latest.timeout_seconds:.0f}s. "
                        "The finished result was discarded."
                    )
                elif error_message is not None:
                    latest.state = JOB_STATE_FAILED
                    latest.result = None
                    latest.message = error_message
                else:
                    latest.state = JOB_STATE_COMPLETED
                    latest.result = result
                    latest.message = f"{latest.action} completed."

                self._events.append(self._snapshot_locked(latest))

    def _build_progress_callback(self, job_id: int) -> JobProgressCallback:
        def report_progress(update: JobProgressValue) -> None:
            with self._lock:
                latest = self._jobs.get(job_id)
                if latest is None or latest.state != JOB_STATE_RUNNING:
                    return
                if latest.cancel_requested or latest.deadline_exceeded:
                    return
                progress_phase, progress_message = normalize_job_progress(update)
                latest.progress_phase = progress_phase
                latest.message = progress_message or latest.message
                self._events.append(self._snapshot_locked(latest))

        return report_progress

    def _snapshot_locked(self, record: _UiJobRecord) -> UiJobSnapshot:
        return UiJobSnapshot(
            job_id=record.job_id,
            action=record.action,
            state=record.state,
            submitted_at=record.submitted_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            timeout_seconds=record.timeout_seconds,
            cancel_requested=record.cancel_requested,
            deadline_exceeded=record.deadline_exceeded,
            progress_phase=record.progress_phase,
            message=record.message,
            result=record.result,
        )


@dataclass
class UiJobBinding:
    output_widget: Any | None
    debug_widget: Any | None = None


def pretty_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)


def build_thinking_debug_report(result: dict[str, Any], enabled: bool) -> str:
    if not enabled:
        return ""

    lines: list[str] = []
    if "turns" in result:
        for index, turn in enumerate(result["turns"], start=1):
            assistant = turn.get("assistant") or {}
            thinking = str(assistant.get("thinking") or "").strip()
            if thinking:
                lines.append(f"Turn {index} thinking:\n{thinking}")
    if "iterations" in result:
        for index, iteration in enumerate(result["iterations"], start=1):
            assistant = iteration.get("assistant") or {}
            thinking = str(assistant.get("thinking") or "").strip()
            if thinking:
                lines.append(f"Iteration {index} thinking:\n{thinking}")

            tool_results = iteration.get("tool_results") or []
            if tool_results:
                lines.append(f"Iteration {index} tool trace:\n{pretty_json({'tool_results': tool_results})}")

    return "\n\n".join(lines).strip()


def supports_second_vision_input(mode: str) -> bool:
    return mode == "compare"


def ui_execution_status(status: str) -> str:
    normalized = (status or "").strip().lower()
    if normalized == "ok":
        return "ok"
    if normalized == "blocked":
        return "blocked"
    return "failed"


def ui_quality_status(status: str, *, validated: bool) -> str:
    normalized = (status or "").strip().lower()
    if normalized == "ok":
        return "pass" if validated else "not_validated"
    if normalized == "blocked":
        return "not_run"
    return "fail"


def build_ui_validation_record(
    *,
    status: str,
    validation_mode: str,
    claim_scope: str,
    pass_definition: str,
    quality_notes: list[str] | None = None,
    quality_checks: list[dict[str, Any]] | None = None,
    validated: bool = False,
) -> dict[str, Any]:
    return {
        "validation_mode": validation_mode,
        "claim_scope": claim_scope,
        "pass_definition": pass_definition,
        "execution_status": ui_execution_status(status),
        "quality_status": ui_quality_status(status, validated=validated),
        "quality_checks": list(quality_checks or []),
        "quality_notes": list(quality_notes or []),
    }


def summarize_text_preview(text: str | None, *, limit: int = 180) -> str:
    compact = " ".join((text or "").strip().split())
    if not compact:
        return "n/a"
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def build_capability_badges(model_id: str | None) -> list[str]:
    resolved_model_id = (model_id or "").strip() or "selected model"
    return [
        f"{resolved_model_id}",
        "Text live",
        "Image/PDF live",
        "Audio live + translate pipeline",
        "Video proxy only",
        "Long-context local 16k / external 96k",
    ]


def build_capability_badges_text(model_id: str | None) -> str:
    return "Capabilities: " + "  |  ".join(build_capability_badges(model_id))


def latest_artifact_ref(session: dict[str, Any] | None) -> dict[str, Any] | None:
    artifact_refs = list((session or {}).get("artifact_refs") or [])
    if not artifact_refs:
        return None
    return artifact_refs[-1]


def summarize_lineage_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "asset_kind": item.get("asset_kind"),
        "transform": item.get("transform"),
        "source_path": item.get("source_path"),
        "resolved_path": item.get("resolved_path"),
    }


def is_tk_previewable_image(path: str | Path | None) -> bool:
    if path in (None, ""):
        return False
    return Path(path).suffix.lower() in TK_PREVIEWABLE_IMAGE_SUFFIXES


def select_preview_image_path(payload: dict[str, Any]) -> str | None:
    seen: set[str] = set()

    def candidate(path: str | Path | None) -> str | None:
        if not is_tk_previewable_image(path):
            return None
        resolved = str(Path(path).expanduser().resolve())
        if resolved in seen:
            return None
        seen.add(resolved)
        if not Path(resolved).exists():
            return None
        return resolved

    lineage = list(((payload.get("assets") or {}).get("lineage")) or [])
    for item in lineage:
        if not isinstance(item, dict):
            continue
        for key in ("resolved_path", "source_path"):
            resolved = candidate(item.get(key))
            if resolved is not None:
                return resolved

    inputs = list(payload.get("inputs") or [])
    for item in inputs:
        if not isinstance(item, dict):
            continue
        for key in ("resolved_path", "source_path"):
            resolved = candidate(item.get(key))
            if resolved is not None:
                return resolved

    singular_input = payload.get("input")
    if isinstance(singular_input, dict):
        for key in ("resolved_path", "source_path"):
            resolved = candidate(singular_input.get(key))
            if resolved is not None:
                return resolved

    return None


def build_artifact_trace_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(payload.get("runtime") or {})
    validation = dict(payload.get("validation") or {})
    lineage = list(((payload.get("assets") or {}).get("lineage")) or [])
    blocker = payload.get("blocker") or {}
    output_preview = (
        payload.get("output_text")
        or payload.get("final_answer")
        or blocker.get("message")
        or summarize_text_preview(json.dumps(payload.get("pipeline") or {}, ensure_ascii=False))
    )
    return {
        "artifact_path": str(path),
        "artifact_kind": payload.get("artifact_kind"),
        "status": payload.get("status"),
        "timestamp_utc": payload.get("timestamp_utc"),
        "runtime_backend": runtime.get("backend"),
        "model_id": runtime.get("model_id") or payload.get("model_id"),
        "device_label": ((runtime.get("device") or {}).get("label")),
        "validation_mode": validation.get("validation_mode"),
        "claim_scope": validation.get("claim_scope"),
        "pass_definition": validation.get("pass_definition"),
        "execution_status": validation.get("execution_status"),
        "quality_status": validation.get("quality_status"),
        "quality_checks": list(validation.get("quality_checks") or []),
        "quality_notes": list(validation.get("quality_notes") or []),
        "lineage_count": len(lineage),
        "lineage_preview": [summarize_lineage_item(item) for item in lineage[:6] if isinstance(item, dict)],
        "preview_image_path": select_preview_image_path(payload),
        "output_preview": summarize_text_preview(output_preview),
        "tool_trace_present": bool(payload.get("iterations")),
        "raw_thinking_saved_to_artifact": bool(payload.get("raw_thinking_saved_to_artifact")),
        "debug_default_hidden": bool(payload.get("debug_default_hidden")),
    }


def summarize_validation_triplet(
    summary: dict[str, Any] | None,
    *,
    artifact_error: str | None = None,
    default: str = "n/a",
) -> str:
    if artifact_error:
        if "FileNotFoundError" in artifact_error:
            return "stale artifact"
        return "artifact unreadable"
    if summary is None:
        return default
    return (
        f"{summary.get('validation_mode') or 'n/a'} / "
        f"{summary.get('execution_status') or 'n/a'} / "
        f"{summary.get('quality_status') or 'n/a'}"
    )


def summarize_workspace_path(path: str | None, *, root: Path) -> str:
    if not path:
        return "n/a"
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (root / resolved).resolve()
    else:
        resolved = resolved.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def build_session_history_report(snapshot: dict[str, Any]) -> str:
    lines: list[str] = [
        "Workspace",
        f"workspace_manifest: {snapshot.get('workspace_manifest_path') or 'n/a'}",
        f"selected_model: {snapshot.get('selected_model_id') or 'n/a'}",
        f"surface: {snapshot.get('surface') or 'n/a'}",
        "",
    ]
    session_summary = snapshot.get("selected_session_summary") or snapshot.get("active_session_summary")
    session_label = "Selected session" if snapshot.get("selected_session_summary") is not None else "Active session"
    if session_summary is None:
        lines.append(session_label)
        if snapshot.get("recent_sessions"):
            lines.append("No session is selected yet. Choose one from Recent Sessions.")
        else:
            lines.append("No recorded session has been indexed in this workspace yet.")
        return "\n".join(lines)

    lines.extend(
        [
            session_label,
            (
                f"manifest: {snapshot.get('selected_session_manifest_path') or snapshot.get('active_session_manifest_path') or 'n/a'}"
            ),
            (
                f"id={session_summary.get('session_id')} "
                f"surface={session_summary.get('surface') or snapshot.get('surface') or 'n/a'} "
                f"title={session_summary.get('title')} "
                f"mode={session_summary.get('current_mode') or 'n/a'} "
                f"updated={session_summary.get('updated_at_utc') or 'n/a'}"
            ),
            (
                f"entries={session_summary.get('entry_count', 0)} "
                f"artifacts={session_summary.get('artifact_count', 0)} "
                f"last_artifact={session_summary.get('latest_artifact_path') or 'n/a'}"
            ),
            "",
            "Recent entries",
        ]
    )

    if snapshot.get("selected_session_error"):
        lines.append(f"session_state: {snapshot['selected_session_error']}")
        lines.append("")

    recent_entries = list(snapshot.get("recent_entries") or [])
    if not recent_entries:
        lines.append("No entries have been recorded yet.")
        return "\n".join(lines)

    selected_entry_id = snapshot.get("selected_entry_id")
    for entry in recent_entries:
        marker = " [selected]" if selected_entry_id and entry.get("entry_id") == selected_entry_id else ""
        lines.append(
            (
                f"- {entry.get('recorded_at_utc') or 'n/a'} "
                f"[{entry.get('status') or 'n/a'}] "
                f"mode={entry.get('mode') or 'n/a'} "
                f"artifact={entry.get('artifact_path') or 'n/a'} "
                f"validation={entry.get('validation_summary') or 'n/a'}{marker}"
            )
        )
        lines.append(f"  preview: {entry.get('output_preview') or 'n/a'}")
        attached_assets = list(entry.get("attached_assets") or [])
        if attached_assets:
            lines.append(f"  assets: {', '.join(attached_assets)}")
        if entry.get("artifact_error"):
            lines.append(f"  artifact_state: {entry['artifact_error']}")

    return "\n".join(lines)


def build_artifact_browser_report(snapshot: dict[str, Any], *, include_raw: bool) -> str:
    artifact_path = snapshot.get("artifact_path")
    if artifact_path is None:
        return (
            "Artifact viewer\n"
            "No artifact is selected yet. Choose a recorded session or entry, run something first, or paste a JSON artifact path."
        )

    if snapshot.get("artifact_error"):
        return (
            "Artifact viewer\n"
            f"path: {artifact_path}\n"
            f"error: {snapshot['artifact_error']}"
        )

    summary = snapshot.get("artifact_summary") or {}
    lines = [
        "Artifact viewer",
        f"path: {artifact_path}",
        f"source: {snapshot.get('artifact_source') or 'n/a'}",
        f"device: {summary.get('device_label') or 'n/a'}",
        f"claim_scope: {summary.get('claim_scope') or 'n/a'}",
        f"pass_definition: {summary.get('pass_definition') or 'n/a'}",
        f"lineage_count: {summary.get('lineage_count', 0)}",
        f"output_preview: {summary.get('output_preview') or 'n/a'}",
    ]

    quality_notes = list(summary.get("quality_notes") or [])
    if quality_notes:
        lines.append("")
        lines.append("Quality notes")
        for note in quality_notes:
            lines.append(f"- {note}")

    quality_checks = list(summary.get("quality_checks") or [])
    if quality_checks:
        lines.append("")
        lines.append("Quality checks")
        for check in quality_checks[:8]:
            verdict = "PASS" if check.get("pass") else "FAIL"
            lines.append(f"- {verdict} {check.get('name')}: {check.get('detail')}")

    lineage_preview = list(summary.get("lineage_preview") or [])
    if lineage_preview:
        lines.append("")
        lines.append("Preprocessing lineage")
        for item in lineage_preview:
            lines.append(
                f"- {item.get('asset_kind') or 'n/a'} / {item.get('transform') or 'n/a'}: "
                f"{item.get('source_path') or 'n/a'} -> {item.get('resolved_path') or 'n/a'}"
            )

    if summary.get("tool_trace_present") or summary.get("raw_thinking_saved_to_artifact"):
        lines.append("")
        lines.append("Trace flags")
        if summary.get("tool_trace_present"):
            lines.append("- tool trace is stored in this artifact")
        if summary.get("raw_thinking_saved_to_artifact"):
            visibility = "default hidden" if summary.get("debug_default_hidden") else "visible"
            lines.append(f"- raw thinking saved to artifact ({visibility})")

    if include_raw and snapshot.get("raw_artifact") is not None:
        lines.append("")
        lines.append("Raw artifact JSON")
        lines.append(pretty_json(snapshot["raw_artifact"]))

    return "\n".join(lines)


def apply_artifact_overview_fields(
    snapshot: dict[str, Any],
    *,
    status_var: tk.StringVar | Any,
    validation_var: tk.StringVar | Any,
    backend_var: tk.StringVar | Any,
    model_var: tk.StringVar | Any,
    timestamp_var: tk.StringVar | Any,
) -> None:
    field_map = dict(build_artifact_overview_fields(snapshot))
    status_var.set(field_map.get("Status", "n/a"))
    validation_var.set(field_map.get("Validation", "n/a"))
    backend_var.set(field_map.get("Backend", "n/a"))
    model_var.set(field_map.get("Model", "n/a"))
    timestamp_var.set(field_map.get("Timestamp", "n/a"))


def apply_entry_compare_fields(
    snapshot: dict[str, Any],
    *,
    previous_var: tk.StringVar | Any,
    selected_var: tk.StringVar | Any,
    change_var: tk.StringVar | Any,
) -> None:
    field_map = dict(build_entry_compare_fields(snapshot))
    previous_var.set(field_map.get("Previous", "n/a"))
    selected_var.set(field_map.get("Selected", "n/a"))
    change_var.set(field_map.get("Change", "n/a"))


def build_artifact_overview_fields(snapshot: dict[str, Any]) -> list[tuple[str, str]]:
    artifact_path = snapshot.get("artifact_path")
    artifact_error = snapshot.get("artifact_error")
    summary = snapshot.get("artifact_summary") or {}
    if artifact_path is None:
        return [
            ("Status", "No artifact selected"),
            ("Validation", "Choose an artifact"),
            ("Backend", "n/a"),
            ("Model", "n/a"),
            ("Timestamp", "n/a"),
        ]
    if artifact_error:
        return [
            ("Status", "Artifact unreadable"),
            ("Validation", "Unavailable"),
            ("Backend", "n/a"),
            ("Model", "n/a"),
            ("Timestamp", "n/a"),
        ]

    artifact_kind = summary.get("artifact_kind") or "artifact"
    return [
        ("Status", f"{artifact_kind} / {summary.get('status') or 'n/a'}"),
        ("Validation", summarize_validation_triplet(summary)),
        ("Backend", summary.get("runtime_backend") or "n/a"),
        ("Model", summary.get("model_id") or "n/a"),
        ("Timestamp", summary.get("timestamp_utc") or "n/a"),
    ]


def _compare_entry_summary_text(entry: dict[str, Any] | None, *, empty: str) -> str:
    if entry is None:
        return empty

    lines = [
        f"{entry.get('recorded_at_utc') or 'n/a'}",
        f"mode={entry.get('mode') or 'n/a'}  status={entry.get('status') or 'n/a'}",
        f"validation={entry.get('validation_summary') or 'n/a'}",
        f"preview={entry.get('output_preview') or 'n/a'}",
    ]
    return "\n".join(lines)


def build_entry_compare_fields(snapshot: dict[str, Any]) -> list[tuple[str, str]]:
    selected_entry = snapshot.get("selected_entry_summary")
    previous_entry = snapshot.get("previous_entry_summary")

    previous_text = _compare_entry_summary_text(
        previous_entry,
        empty="No older artifact is available in this session.",
    )
    selected_text = _compare_entry_summary_text(
        selected_entry,
        empty="Choose a recorded artifact to compare.",
    )

    if selected_entry is None:
        change_text = "Select a recorded artifact to compare against the prior evidence."
    elif previous_entry is None:
        change_text = "This artifact is the oldest visible entry in the selected session."
    else:
        change_lines: list[str] = []
        for label, previous_value, selected_value in (
            ("status", previous_entry.get("status"), selected_entry.get("status")),
            ("validation", previous_entry.get("validation_summary"), selected_entry.get("validation_summary")),
            ("mode", previous_entry.get("mode"), selected_entry.get("mode")),
        ):
            if previous_value == selected_value:
                change_lines.append(f"{label}: unchanged")
            else:
                change_lines.append(
                    f"{label}: {previous_value or 'n/a'} -> {selected_value or 'n/a'}"
                )

        if previous_entry.get("output_preview") == selected_entry.get("output_preview"):
            change_lines.append("output: preview unchanged")
        else:
            change_lines.append("output: preview changed")
        change_text = "\n".join(change_lines)

    return [
        ("Previous", previous_text),
        ("Selected", selected_text),
        ("Change", change_text),
    ]


class LocalUiController:
    def __init__(
        self,
        *,
        session_manager: SessionManager | None = None,
        workspace_store: WorkspaceSessionStore | None = None,
    ) -> None:
        self.session_manager = session_manager or SessionManager()
        self.workspace_store = workspace_store or WorkspaceSessionStore()
        self.selected_model_id = self.workspace_store.selected_model_id() or resolve_model_id()
        self.workspace_store.set_selected_model(self.selected_model_id)
        self.last_result: UiActionResult | None = None
        self.last_prewarm: dict[str, Any] | None = None

    def close(self) -> None:
        self.session_manager.close_all()

    def set_model_id(self, model_id: str) -> str:
        normalized = (model_id or "").strip() or resolve_model_id()
        if normalized != self.selected_model_id:
            self.session_manager.close_all()
            self.selected_model_id = normalized
        self.workspace_store.set_selected_model(self.selected_model_id)
        return self.selected_model_id

    def resolved_audio_model(self) -> tuple[str, str]:
        return resolve_audio_model_selection(base_model_id=self.selected_model_id)

    def has_cached_selected_text_session(self) -> bool:
        return any(
            key.session_kind == "text" and key.model_id == self.selected_model_id
            for key in self.session_manager.cached_keys()
        )

    def prewarm_selected_model(
        self,
        *,
        cancellation_signal: CancellationSignal | None = None,
        progress_callback: JobProgressCallback | None = None,
    ) -> UiActionResult:
        prewarm = warm_thinking_session(
            model_id=self.selected_model_id,
            session_manager=self.session_manager,
            cancellation_signal=cancellation_signal,
            progress_callback=progress_callback,
        )
        self.last_prewarm = {
            "model_id": prewarm["model_id"],
            "device_label": prewarm["device_info"]["label"],
            "dtype_name": prewarm["device_info"]["dtype_name"],
            "elapsed_seconds": prewarm["elapsed_seconds"],
            "primed_text": prewarm["primed_text"],
        }
        return UiActionResult(
            action=PREWARM_ACTION,
            status="ok",
            output_text=(
                f"Warmup ready in {prewarm['elapsed_seconds']:.3f}s. "
                f"Shared thinking session primed on {prewarm['device_info']['label']}."
            ),
            artifact_path=repo_root() / "artifacts",
            model_id=prewarm["model_id"],
            backend=PREWARM_BACKEND,
            device_label=prewarm["device_info"]["label"],
            dtype_name=prewarm["device_info"]["dtype_name"],
            notes=[f"primed_text={prewarm['primed_text']!r}"],
        )

    def run_chat(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        cancellation_signal: CancellationSignal | None = None,
    ) -> UiActionResult:
        resolved_prompt = (prompt or "").strip() or TEXT_DEFAULT_PROMPTS["chat"]
        resolved_system_prompt = resolve_text_system_prompt("chat", system_prompt)
        user_prompt = build_text_user_prompt("chat", resolved_prompt)
        artifact_path = default_text_output_path("chat")
        status = "ok"
        blocker_message: str | None = None
        task_result: dict[str, Any] | None = None
        output_text = ""
        base_messages = self.workspace_store.chat_messages_for_next_turn(
            model_id=self.selected_model_id,
            system_prompt=resolved_system_prompt,
        )
        messages_to_run = list(base_messages) + [{"role": "user", "content": user_prompt}]

        try:
            task_result = run_text_task(
                task="chat",
                prompt=prompt,
                system_prompt=system_prompt,
                messages=messages_to_run,
                model_id=self.selected_model_id,
                session_manager=self.session_manager,
                cancellation_signal=cancellation_signal,
            )
            output_text = str(task_result["output_text"])
        except UserFacingError as exc:
            status = "blocked"
            blocker_message = str(exc)
            output_text = blocker_message
        except Exception as exc:
            status = "failed"
            blocker_message = f"{type(exc).__name__}: {exc}"
            output_text = blocker_message

        runtime = build_runtime_record(
            backend=CHAT_BACKEND,
            model_id=task_result["model_id"] if task_result is not None else self.selected_model_id,
            device_info=task_result["device_info"] if task_result is not None else "unresolved",
            elapsed_seconds=task_result["elapsed_seconds"] if task_result is not None else None,
        )
        prompts = build_prompt_record(
            system_prompt=task_result["system_prompt"] if task_result is not None else resolved_system_prompt,
            prompt=task_result["prompt"] if task_result is not None else resolved_prompt,
            resolved_user_prompt=task_result["resolved_user_prompt"] if task_result is not None else user_prompt,
        )
        validation = build_ui_validation_record(
            status=status,
            validation_mode="live",
            claim_scope="user-driven local UI text/chat execution",
            pass_definition=(
                "This local UI run records runtime details and the raw artifact path, but it does not by itself close a deterministic capability claim."
            ),
            quality_notes=[
                "For deterministic pass/fail capability claims, use the capability matrix on the recorded fixture inputs.",
            ],
            validated=False,
        )
        payload = build_artifact_payload(
            artifact_kind="text",
            status=status,
            runtime=runtime,
            prompts=prompts,
            blocker_message=blocker_message,
            extra={
                "task": "chat",
                "ui_surface": "local-ui",
                "validation": validation,
                "model_id": task_result["model_id"] if task_result is not None else self.selected_model_id,
                "device": runtime["device"]["label"],
                "dtype": runtime["device"]["dtype"],
                "generation_settings": task_result["generation_settings"] if task_result is not None else dict(TEXT_GENERATION_SETTINGS["chat"]),
                "system_prompt": task_result["system_prompt"] if task_result is not None else resolved_system_prompt,
                "prompt": task_result["prompt"] if task_result is not None else resolved_prompt,
                "resolved_user_prompt": task_result["resolved_user_prompt"] if task_result is not None else user_prompt,
                "elapsed_seconds": round(task_result["elapsed_seconds"], 3) if task_result is not None else None,
                "output_text": output_text if status == "ok" else None,
            },
        )
        write_artifact(artifact_path, payload)
        self.workspace_store.record_chat_turn(
            model_id=runtime["model_id"],
            status=status,
            artifact_path=artifact_path,
            prompt=task_result["prompt"] if task_result is not None else resolved_prompt,
            system_prompt=task_result["system_prompt"] if task_result is not None else resolved_system_prompt,
            resolved_user_prompt=task_result["resolved_user_prompt"] if task_result is not None else user_prompt,
            output_text=output_text,
            base_messages=base_messages,
            notes=[blocker_message] if blocker_message else [],
        )
        return self._finalize_result(
            UiActionResult(
                action="chat",
                status=status,
                output_text=output_text or "",
                artifact_path=artifact_path,
                model_id=runtime["model_id"],
                backend=CHAT_BACKEND,
                device_label=runtime["device"]["label"],
                dtype_name=runtime["device"]["dtype"],
            )
        )

    def run_vision(
        self,
        *,
        mode: str,
        inputs: list[str],
        prompt: str | None = None,
        system_prompt: str | None = None,
        max_pages: int = 4,
        cancellation_signal: CancellationSignal | None = None,
    ) -> UiActionResult:
        resolved_prompt = resolve_vision_prompt(mode, prompt)
        resolved_system_prompt = resolve_vision_system_prompt(mode, system_prompt)
        artifact_path = default_vision_output_path(mode)
        user_prompt: str | None = None
        records: list[dict[str, Any]] = []
        status = "ok"
        blocker_message: str | None = None
        mode_result: dict[str, Any] | None = None
        output_text = ""

        try:
            mode_result = run_vision_mode(
                mode=mode,
                inputs=[item for item in inputs if item.strip()],
                prompt=prompt,
                system_prompt=system_prompt,
                max_pages=max_pages,
                model_id=self.selected_model_id,
                session_manager=self.session_manager,
                cancellation_signal=cancellation_signal,
            )
            records = list(mode_result["records"])
            user_prompt = str(mode_result["resolved_user_prompt"])
            output_text = str(mode_result["output_text"])
        except UserFacingError as exc:
            status = "blocked"
            blocker_message = str(exc)
            output_text = blocker_message
        except Exception as exc:
            status = "failed"
            blocker_message = f"{type(exc).__name__}: {exc}"
            output_text = blocker_message

        runtime = build_runtime_record(
            backend=VISION_BACKEND,
            model_id=mode_result["model_id"] if mode_result is not None else self.selected_model_id,
            device_info=mode_result["device_info"] if mode_result is not None else "unresolved",
            elapsed_seconds=mode_result["elapsed_seconds"] if mode_result is not None else None,
        )
        prompts = build_prompt_record(
            system_prompt=mode_result["system_prompt"] if mode_result is not None else resolved_system_prompt,
            prompt=mode_result["prompt"] if mode_result is not None else resolved_prompt,
            resolved_user_prompt=mode_result["resolved_user_prompt"] if mode_result is not None else user_prompt,
        )
        validation = build_ui_validation_record(
            status=status,
            validation_mode="live",
            claim_scope="user-driven local UI image/pdf execution",
            pass_definition=(
                "This local UI run records the raw artifact plus preprocessing lineage, but it does not auto-claim deterministic fixture validation for arbitrary user inputs."
            ),
            quality_notes=[
                "Preprocessing lineage is recorded here so you can audit how the visible asset was normalized or rasterized.",
                "For deterministic pass/fail capability claims, use the capability matrix on the known local fixtures.",
            ],
            validated=False,
        )
        payload = build_artifact_payload(
            artifact_kind="vision",
            status=status,
            runtime=runtime,
            prompts=prompts,
            asset_lineage=collect_asset_lineage(records),
            blocker_message=blocker_message,
            extra={
                "mode": mode,
                "ui_surface": "local-ui",
                "validation": validation,
                "model_id": mode_result["model_id"] if mode_result is not None else self.selected_model_id,
                "device": runtime["device"]["label"],
                "dtype": runtime["device"]["dtype"],
                "generation_settings": mode_result["generation_settings"] if mode_result is not None else dict(VISION_GENERATION_SETTINGS[mode]),
                "system_prompt": mode_result["system_prompt"] if mode_result is not None else resolved_system_prompt,
                "prompt": mode_result["prompt"] if mode_result is not None else resolved_prompt,
                "resolved_user_prompt": mode_result["resolved_user_prompt"] if mode_result is not None else user_prompt,
                "inputs": serialize_input_records(records),
                "elapsed_seconds": round(mode_result["elapsed_seconds"], 3) if mode_result is not None else None,
                "output_text": output_text if status == "ok" else None,
            },
        )
        write_artifact(artifact_path, payload)
        attachments = []
        if inputs and inputs[0].strip():
            attachments.append({"role": "primary_input", "path": inputs[0]})
        if len(inputs) > 1 and inputs[1].strip():
            attachments.append({"role": "secondary_input", "path": inputs[1]})
        self.workspace_store.record_session_run(
            surface="vision",
            model_id=runtime["model_id"],
            mode=mode,
            artifact_kind="vision",
            artifact_path=artifact_path,
            status=status,
            prompt=mode_result["prompt"] if mode_result is not None else resolved_prompt,
            system_prompt=mode_result["system_prompt"] if mode_result is not None else resolved_system_prompt,
            resolved_user_prompt=mode_result["resolved_user_prompt"] if mode_result is not None else user_prompt,
            output_text=output_text,
            attachments=attachments,
            notes=[blocker_message] if blocker_message else [],
            options={"max_pages": max_pages},
        )
        return self._finalize_result(
            UiActionResult(
                action="vision",
                status=status,
                output_text=output_text or "",
                artifact_path=artifact_path,
                model_id=runtime["model_id"],
                backend=VISION_BACKEND,
                device_label=runtime["device"]["label"],
                dtype_name=runtime["device"]["dtype"],
            )
        )

    def run_audio(
        self,
        *,
        mode: str,
        input_path: str,
        target_language: str = DEFAULT_TARGET_LANGUAGE,
        prompt: str | None = None,
        system_prompt: str | None = None,
        cancellation_signal: CancellationSignal | None = None,
    ) -> UiActionResult:
        artifact_path = default_audio_output_path(mode)
        resolved_prompt = resolve_audio_prompt(mode, prompt, target_language)
        resolved_system_prompt = resolve_audio_system_prompt(mode, system_prompt)
        audio_model_id, model_source = self.resolved_audio_model()
        status = "ok"
        blocker_message: str | None = None
        mode_result: dict[str, Any] | None = None
        output_text = ""
        notes: list[str] = []

        try:
            mode_result = run_audio_mode(
                mode=mode,
                input_path=Path(input_path),
                target_language=target_language,
                prompt=prompt,
                system_prompt=system_prompt,
                base_model_id=self.selected_model_id,
                session_manager=self.session_manager,
                cancellation_signal=cancellation_signal,
            )
            output_text = str(mode_result["output_text"])
            notes = list(mode_result["validation"].get("quality_notes") or [])
        except UserFacingError as exc:
            status = "blocked"
            blocker_message = str(exc)
            output_text = blocker_message
        except Exception as exc:
            status = "failed"
            blocker_message = f"{type(exc).__name__}: {exc}"
            output_text = blocker_message

        runtime = build_runtime_record(
            backend=AUDIO_BACKEND,
            model_id=mode_result["model_id"] if mode_result is not None else audio_model_id,
            device_info=mode_result["device_info"] if mode_result is not None else "unresolved",
            elapsed_seconds=mode_result["elapsed_seconds"] if mode_result is not None else None,
            extra={
                "base_model_id": mode_result["base_model_id"] if mode_result is not None else self.selected_model_id,
                "model_id_source": mode_result["model_id_source"] if mode_result is not None else model_source,
            },
        )
        prompts = build_prompt_record(
            system_prompt=mode_result["system_prompt"] if mode_result is not None else resolved_system_prompt,
            prompt=mode_result["prompt"] if mode_result is not None else resolved_prompt,
            resolved_user_prompt=mode_result["prompt"] if mode_result is not None else resolved_prompt,
        )
        validation = mode_result["validation"] if mode_result is not None else build_ui_validation_record(
            status=status,
            validation_mode="pipeline" if mode == "translate" else "live",
            claim_scope="user-driven local UI audio execution",
            pass_definition=(
                "This local UI run records the audio artifact and any pipeline details, but a blocked or failed entry does not count as a validated claim."
            ),
            quality_notes=[],
            validated=False,
        )
        payload = build_artifact_payload(
            artifact_kind="audio",
            status=status,
            runtime=runtime,
            prompts=prompts,
            asset_lineage=collect_asset_lineage(mode_result["record"] if mode_result is not None else None),
            blocker_message=blocker_message,
            extra={
                "mode": mode,
                "ui_surface": "local-ui",
                "validation": validation,
                "base_model_id": mode_result["base_model_id"] if mode_result is not None else self.selected_model_id,
                "model_id": mode_result["model_id"] if mode_result is not None else audio_model_id,
                "model_id_source": mode_result["model_id_source"] if mode_result is not None else model_source,
                "device": runtime["device"]["label"],
                "dtype": runtime["device"]["dtype"],
                "generation_settings": mode_result["generation_settings"] if mode_result is not None else None,
                "target_language": mode_result["target_language"] if mode_result is not None else target_language,
                "system_prompt": mode_result["system_prompt"] if mode_result is not None else resolved_system_prompt,
                "prompt": mode_result["prompt"] if mode_result is not None else resolved_prompt,
                "resolved_user_prompt": mode_result["prompt"] if mode_result is not None else resolved_prompt,
                "input": serialize_audio_record(mode_result["record"]) if mode_result is not None else {"source_path": str(Path(input_path).expanduser())},
                "elapsed_seconds": round(mode_result["elapsed_seconds"], 3) if mode_result is not None else None,
                "output_text": mode_result["output_text"] if mode_result is not None and status == "ok" else None,
                "pipeline": mode_result["pipeline"] if mode_result is not None else None,
            },
        )
        write_artifact(artifact_path, payload)
        session_notes = list(notes)
        if blocker_message:
            session_notes.append(blocker_message)
        self.workspace_store.record_session_run(
            surface="audio",
            model_id=runtime["model_id"],
            mode=mode,
            artifact_kind="audio",
            artifact_path=artifact_path,
            status=status,
            prompt=mode_result["prompt"] if mode_result is not None else resolved_prompt,
            system_prompt=mode_result["system_prompt"] if mode_result is not None else resolved_system_prompt,
            resolved_user_prompt=mode_result["prompt"] if mode_result is not None else resolved_prompt,
            output_text=output_text,
            attachments=[{"role": "audio_input", "path": input_path}],
            notes=session_notes,
            options={"target_language": mode_result["target_language"] if mode_result is not None else target_language},
        )
        return self._finalize_result(
            UiActionResult(
                action="audio",
                status=status,
                output_text=output_text,
                artifact_path=artifact_path,
                model_id=runtime["model_id"],
                backend=AUDIO_BACKEND,
                device_label=runtime["device"]["label"],
                dtype_name=runtime["device"]["dtype"],
                notes=notes,
            )
        )

    def run_thinking(
        self,
        *,
        mode: str,
        prompt: str | None = None,
        follow_up: str | None = None,
        system_prompt: str | None = None,
        include_debug: bool = False,
        max_tool_iterations: int = 3,
        cancellation_signal: CancellationSignal | None = None,
    ) -> UiActionResult:
        artifact_path = default_thinking_artifact_path(mode)
        resolved_system_prompt = system_prompt or (TEXT_SYSTEM_PROMPT if mode == "text" else TOOL_SYSTEM_PROMPT)
        first_prompt = (prompt or (TEXT_PROMPT if mode == "text" else TOOL_PROMPT)).strip()
        follow_up_prompt = (follow_up or TEXT_FOLLOW_UP).strip()
        status = "ok"
        blocker_message: str | None = None
        device_info: dict[str, Any] | str = "unresolved"
        result: dict[str, Any] | None = None
        output_text = ""

        try:
            result = run_thinking_session(
                mode=mode,
                system_prompt=system_prompt,
                prompt=prompt,
                follow_up=follow_up,
                show_thinking=False,
                max_tool_iterations=max_tool_iterations,
                model_id=self.selected_model_id,
                session_manager=self.session_manager,
                cancellation_signal=cancellation_signal,
            )
            device_info = result["device_info"]
            output_text = str(result["final_answer"]).strip()
        except UserFacingError as exc:
            status = "blocked"
            blocker_message = str(exc)
            output_text = blocker_message
        except Exception as exc:
            status = "failed"
            blocker_message = f"{type(exc).__name__}: {exc}"
            output_text = blocker_message

        runtime = build_runtime_record(
            backend=THINKING_BACKEND,
            model_id=result["model_id"] if result is not None else self.selected_model_id,
            device_info=device_info,
            elapsed_seconds=result["elapsed_seconds"] if result is not None else None,
        )
        prompts = build_prompt_record(
            system_prompt=result["system_prompt"] if result is not None else resolved_system_prompt,
            prompt=result["first_prompt"] if result is not None else first_prompt,
            resolved_user_prompt=result["first_prompt"] if result is not None else first_prompt,
            extra={
                "follow_up_prompt": result["follow_up_prompt"] if result is not None else follow_up_prompt if mode == "text" else None,
            },
        )
        payload_extra = {
            "mode": mode,
            "ui_surface": "local-ui",
            "model_id": result["model_id"] if result is not None else self.selected_model_id,
            "device": runtime["device"]["label"],
            "dtype": runtime["device"]["dtype"],
            "stdout_behavior": "final_answer_only",
            "raw_thinking_saved_to_artifact": True,
            "debug_default_hidden": True,
        }
        if result is not None:
            sanitized_result = dict(result)
            sanitized_result["device_info"] = normalize_device_info(result.get("device_info"))
            payload_extra.update(sanitized_result)
        payload = build_artifact_payload(
            artifact_kind="thinking",
            status=status,
            runtime=runtime,
            prompts=prompts,
            blocker_message=blocker_message,
            extra={
                "validation": build_ui_validation_record(
                    status=status,
                    validation_mode="live",
                    claim_scope="user-driven local UI thinking/tool execution",
                    pass_definition=(
                        "This local UI run records the visible answer plus optional raw thinking in the artifact, but it does not by itself close a deterministic capability claim."
                    ),
                    quality_notes=[
                        "Raw thinking may be stored in the artifact, but it stays hidden in the UI unless you explicitly open the artifact view.",
                    ],
                    validated=False,
                ),
                **payload_extra,
            },
        )
        write_artifact(artifact_path, payload)
        self.workspace_store.record_session_run(
            surface="thinking",
            model_id=runtime["model_id"],
            mode=mode,
            artifact_kind="thinking",
            artifact_path=artifact_path,
            status=status,
            prompt=result["first_prompt"] if result is not None else first_prompt,
            system_prompt=result["system_prompt"] if result is not None else resolved_system_prompt,
            resolved_user_prompt=result["first_prompt"] if result is not None else first_prompt,
            output_text=output_text,
            notes=[blocker_message] if blocker_message else [],
            options={
                "follow_up_prompt": result["follow_up_prompt"] if result is not None else follow_up_prompt if mode == "text" else None,
                "include_debug": include_debug,
            },
        )
        return self._finalize_result(
            UiActionResult(
                action="thinking",
                status=status,
                output_text=output_text,
                artifact_path=artifact_path,
                model_id=runtime["model_id"],
                backend=THINKING_BACKEND,
                device_label=runtime["device"]["label"],
                dtype_name=runtime["device"]["dtype"],
                debug_text=build_thinking_debug_report(result or {}, enabled=include_debug),
            )
        )

    def collect_diagnostics(self) -> dict[str, Any]:
        audio_model_id, audio_model_source = self.resolved_audio_model()
        cached_sessions = [
            {
                "session_kind": key.session_kind,
                "model_id": key.model_id,
                "device_class": key.device_class,
            }
            for key in self.session_manager.cached_keys()
        ]
        diagnostics = {
            "workspace": str(repo_root()),
            "python_executable": sys.executable,
            "python_version": sys.version.replace("\n", " "),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "selected_model_id": self.selected_model_id,
            "resolved_audio_model_id": audio_model_id,
            "resolved_audio_model_source": audio_model_source,
            "hf_token_configured": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
            "artifact_roots": {
                "text": str((repo_root() / "artifacts" / "text").resolve()),
                "vision": str((repo_root() / "artifacts" / "vision").resolve()),
                "audio": str((repo_root() / "artifacts" / "audio").resolve()),
                "thinking": str((repo_root() / "artifacts" / "thinking").resolve()),
            },
            "cached_sessions": cached_sessions,
            "torch": probe_torch(),
            "transformers": probe_transformers(),
            "optional_modules": probe_optional_modules(),
            "assets": assets_summary(repo_root()),
            "last_action": {
                "action": self.last_result.action if self.last_result is not None else None,
                "status": self.last_result.status if self.last_result is not None else None,
                "artifact_path": str(self.last_result.artifact_path) if self.last_result is not None else None,
            },
            "last_prewarm": self.last_prewarm,
            "workspace_state": self.workspace_store.diagnostics(),
        }
        return diagnostics

    def capability_badges_text(self) -> str:
        return build_capability_badges_text(self.selected_model_id)

    def latest_artifact_path(self, *, surface: str) -> str | None:
        session = self.workspace_store.active_session(surface)
        artifact_ref = latest_artifact_ref(session)
        if artifact_ref is None:
            return None
        artifact_path = artifact_ref.get("artifact_path")
        if not isinstance(artifact_path, str) or not artifact_path.strip():
            return None
        return artifact_path

    def _artifact_trace_from_path(
        self,
        artifact_path: str | None,
        *,
        cache: dict[str, dict[str, Any]],
    ) -> tuple[str | None, dict[str, Any] | None, dict[str, Any] | None, str | None]:
        if not isinstance(artifact_path, str) or not artifact_path.strip():
            return None, None, None, None

        resolved_artifact_path = str(Path(artifact_path).expanduser().resolve())
        cached = cache.get(resolved_artifact_path)
        if cached is None:
            raw_artifact: dict[str, Any] | None = None
            artifact_summary: dict[str, Any] | None = None
            artifact_error: str | None = None
            try:
                artifact_file = Path(resolved_artifact_path)
                raw_artifact = read_artifact(artifact_file)
                artifact_summary = build_artifact_trace_summary(artifact_file, raw_artifact)
            except Exception as exc:
                artifact_error = f"{type(exc).__name__}: {exc}"
            cached = {
                "raw_artifact": raw_artifact,
                "artifact_summary": artifact_summary,
                "artifact_error": artifact_error,
            }
            cache[resolved_artifact_path] = cached

        return (
            resolved_artifact_path,
            cached["raw_artifact"],
            cached["artifact_summary"],
            cached["artifact_error"],
        )

    def _session_manifest_from_summary(self, summary: dict[str, Any]) -> Path | None:
        manifest_path = summary.get("manifest_path")
        if not isinstance(manifest_path, str) or not manifest_path.strip():
            return None
        resolved = Path(manifest_path).expanduser()
        if not resolved.is_absolute():
            resolved = self.workspace_store.manifest_path.parent / resolved
        return resolved.resolve()

    def _build_recent_session_rows(
        self,
        *,
        workspace_payload: dict[str, Any],
        artifact_cache: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        active_session_ids = dict(workspace_payload.get("active_session_ids") or {})
        for summary in list(workspace_payload.get("sessions") or []):
            if not isinstance(summary, dict):
                continue

            manifest_path = self._session_manifest_from_summary(summary)
            session_error: str | None = None
            session_state = "ok"
            if manifest_path is None:
                session_state = "manifest_missing"
                session_error = "Session manifest path is missing from the workspace index."
            elif not manifest_path.exists():
                session_state = "stale_manifest"
                session_error = f"Session manifest is missing: {manifest_path}"

            latest_artifact_path = summary.get("latest_artifact_path")
            resolved_artifact_path, _raw, artifact_summary, artifact_error = self._artifact_trace_from_path(
                latest_artifact_path,
                cache=artifact_cache,
            )
            rows.append(
                {
                    "session_id": summary.get("session_id"),
                    "surface": summary.get("surface"),
                    "title": summary.get("title"),
                    "current_mode": summary.get("current_mode"),
                    "updated_at_utc": summary.get("updated_at_utc"),
                    "entry_count": summary.get("entry_count", 0),
                    "artifact_count": summary.get("artifact_count", 0),
                    "manifest_path": str(manifest_path) if manifest_path is not None else None,
                    "manifest_display": (
                        summarize_workspace_path(str(manifest_path), root=self.workspace_store.root)
                        if manifest_path is not None
                        else "n/a"
                    ),
                    "session_state": session_state,
                    "session_error": session_error,
                    "is_active_surface": active_session_ids.get(summary.get("surface")) == summary.get("session_id"),
                    "latest_artifact_path": resolved_artifact_path,
                    "latest_artifact_display": summarize_workspace_path(
                        resolved_artifact_path or latest_artifact_path,
                        root=self.workspace_store.root,
                    ),
                    "latest_validation_summary": summarize_validation_triplet(
                        artifact_summary,
                        artifact_error=artifact_error,
                        default="no artifact yet",
                    ),
                    "latest_validation_mode": (
                        artifact_summary.get("validation_mode") if artifact_summary is not None else None
                    ),
                    "latest_execution_status": (
                        artifact_summary.get("execution_status") if artifact_summary is not None else None
                    ),
                    "latest_quality_status": (
                        artifact_summary.get("quality_status") if artifact_summary is not None else None
                    ),
                    "latest_artifact_error": artifact_error,
                }
            )
        return rows

    def _build_recent_entry_rows(
        self,
        *,
        session_payload: dict[str, Any],
        artifact_cache: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        entries = list(session_payload.get("entries") or [])
        for entry in reversed(entries[-12:]):
            if not isinstance(entry, dict):
                continue
            artifact_ref = dict(entry.get("artifact_ref") or {})
            artifact_path = artifact_ref.get("artifact_path")
            resolved_artifact_path, _raw, artifact_summary, artifact_error = self._artifact_trace_from_path(
                artifact_path,
                cache=artifact_cache,
            )
            attached_assets = [
                str(asset.get("path"))
                for asset in list(entry.get("attached_assets") or [])
                if isinstance(asset, dict) and asset.get("path")
            ]
            rows.append(
                {
                    "entry_id": entry.get("entry_id"),
                    "recorded_at_utc": entry.get("recorded_at_utc"),
                    "mode": entry.get("mode"),
                    "status": entry.get("status"),
                    "artifact_path": resolved_artifact_path,
                    "artifact_display": summarize_workspace_path(
                        resolved_artifact_path or artifact_path,
                        root=self.workspace_store.root,
                    ),
                    "attached_assets": attached_assets,
                    "output_preview": summarize_text_preview(str(entry.get("output_text") or "")),
                    "validation_summary": summarize_validation_triplet(
                        artifact_summary,
                        artifact_error=artifact_error,
                        default="no artifact yet",
                    ),
                    "validation_mode": (
                        artifact_summary.get("validation_mode") if artifact_summary is not None else None
                    ),
                    "execution_status": (
                        artifact_summary.get("execution_status") if artifact_summary is not None else None
                    ),
                    "quality_status": (
                        artifact_summary.get("quality_status") if artifact_summary is not None else None
                    ),
                    "artifact_error": artifact_error,
                }
            )
        return rows

    def collect_forensics(
        self,
        *,
        surface: str,
        artifact_path: str | None = None,
        session_id: str | None = None,
        entry_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_surface = (surface or "").strip().lower()
        if normalized_surface not in SESSION_SURFACES:
            raise UserFacingError(f"Unsupported workspace session surface `{surface}`.")

        workspace_payload = read_workspace_manifest(self.workspace_store.manifest_path)
        artifact_cache: dict[str, dict[str, Any]] = {}
        recent_sessions = self._build_recent_session_rows(
            workspace_payload=workspace_payload,
            artifact_cache=artifact_cache,
        )
        active_session_id = dict(workspace_payload.get("active_session_ids") or {}).get(normalized_surface)
        active_session_row = next(
            (
                row
                for row in recent_sessions
                if row.get("surface") == normalized_surface and row.get("session_id") == active_session_id
            ),
            None,
        )

        selected_session_row = None
        if session_id:
            selected_session_row = next(
                (row for row in recent_sessions if row.get("session_id") == session_id),
                None,
            )
        if selected_session_row is None:
            selected_session_row = active_session_row
        if selected_session_row is None:
            selected_session_row = next(
                (row for row in recent_sessions if row.get("surface") == normalized_surface),
                None,
            )

        selected_session_summary: dict[str, Any] | None = None
        selected_session_manifest_path: str | None = None
        selected_session_error: str | None = None
        active_session_summary: dict[str, Any] | None = None
        recent_entries: list[dict[str, Any]] = []
        selected_entry_id: str | None = None
        selected_entry_error: str | None = None
        selected_entry_summary: dict[str, Any] | None = None
        previous_entry_summary: dict[str, Any] | None = None

        if active_session_row is not None:
            active_session_summary = {
                "session_id": active_session_row.get("session_id"),
                "surface": active_session_row.get("surface"),
                "title": active_session_row.get("title"),
                "current_mode": active_session_row.get("current_mode"),
                "updated_at_utc": active_session_row.get("updated_at_utc"),
                "entry_count": active_session_row.get("entry_count", 0),
                "artifact_count": active_session_row.get("artifact_count", 0),
                "latest_artifact_path": active_session_row.get("latest_artifact_path"),
            }

        session_payload: dict[str, Any] | None = None
        if selected_session_row is not None:
            selected_session_summary = {
                "session_id": selected_session_row.get("session_id"),
                "surface": selected_session_row.get("surface"),
                "title": selected_session_row.get("title"),
                "current_mode": selected_session_row.get("current_mode"),
                "updated_at_utc": selected_session_row.get("updated_at_utc"),
                "entry_count": selected_session_row.get("entry_count", 0),
                "artifact_count": selected_session_row.get("artifact_count", 0),
                "latest_artifact_path": selected_session_row.get("latest_artifact_path"),
            }
            selected_session_manifest_path = selected_session_row.get("manifest_path")
            selected_session_error = selected_session_row.get("session_error")
            if selected_session_manifest_path is not None and selected_session_error is None:
                try:
                    session_payload = read_session_manifest(Path(selected_session_manifest_path))
                except Exception as exc:
                    selected_session_error = f"{type(exc).__name__}: {exc}"
            if session_payload is not None:
                recent_entries = self._build_recent_entry_rows(
                    session_payload=session_payload,
                    artifact_cache=artifact_cache,
                )
                if entry_id:
                    selected_entry = next(
                        (item for item in recent_entries if item.get("entry_id") == entry_id),
                        None,
                    )
                    if selected_entry is None:
                        selected_entry_error = (
                            f"Entry `{entry_id}` is no longer present in session "
                            f"`{selected_session_summary.get('session_id')}`."
                        )
                    else:
                        selected_entry_id = str(selected_entry.get("entry_id"))
                elif recent_entries:
                    selected_entry_id = str(recent_entries[0].get("entry_id"))
                if selected_entry_id is not None:
                    for index, item in enumerate(recent_entries):
                        if item.get("entry_id") == selected_entry_id:
                            selected_entry_summary = item
                            if index + 1 < len(recent_entries):
                                previous_entry_summary = recent_entries[index + 1]
                            break
        elif session_id:
            selected_session_error = f"Session `{session_id}` is no longer indexed in this workspace."

        resolved_artifact_path: str | None = None
        artifact_source: str | None = None
        if artifact_path and artifact_path.strip():
            resolved_artifact_path = str(Path(artifact_path).expanduser().resolve())
            artifact_source = "override"
        elif selected_entry_id is not None:
            selected_entry = next(
                (item for item in recent_entries if item.get("entry_id") == selected_entry_id),
                None,
            )
            if selected_entry is not None and selected_entry.get("artifact_path"):
                resolved_artifact_path = str(selected_entry["artifact_path"])
                artifact_source = "selected_entry"
        elif selected_session_row is not None and selected_session_row.get("latest_artifact_path"):
            resolved_artifact_path = str(selected_session_row["latest_artifact_path"])
            artifact_source = "selected_session_latest"
        else:
            resolved_artifact_path = active_session_row.get("latest_artifact_path") if active_session_row is not None else None
            artifact_source = "surface_latest" if resolved_artifact_path else None
            if resolved_artifact_path is None and self.last_result is not None:
                resolved_artifact_path = str(self.last_result.artifact_path.resolve())
                artifact_source = "last_result"

        raw_artifact: dict[str, Any] | None = None
        artifact_summary: dict[str, Any] | None = None
        artifact_error: str | None = None
        if resolved_artifact_path is not None:
            (
                resolved_artifact_path,
                raw_artifact,
                artifact_summary,
                artifact_error,
            ) = self._artifact_trace_from_path(
                resolved_artifact_path,
                cache=artifact_cache,
            )

        return {
            "selected_model_id": self.selected_model_id,
            "capability_badges": build_capability_badges(self.selected_model_id),
            "workspace_manifest_path": str(self.workspace_store.manifest_path.resolve()),
            "workspace_updated_at_utc": workspace_payload.get("updated_at_utc"),
            "surface": normalized_surface,
            "active_session_manifest_path": active_session_row.get("manifest_path") if active_session_row is not None else None,
            "active_session_summary": active_session_summary,
            "selected_session_id": selected_session_summary.get("session_id") if selected_session_summary is not None else None,
            "selected_session_manifest_path": selected_session_manifest_path,
            "selected_session_summary": selected_session_summary,
            "selected_session_error": selected_session_error,
            "selected_entry_id": selected_entry_id,
            "selected_entry_error": selected_entry_error,
            "selected_entry_summary": selected_entry_summary,
            "previous_entry_summary": previous_entry_summary,
            "recent_sessions": recent_sessions,
            "recent_entries": recent_entries,
            "artifact_source": artifact_source,
            "artifact_path": resolved_artifact_path,
            "artifact_summary": artifact_summary,
            "artifact_error": artifact_error,
            "raw_artifact": raw_artifact,
        }

    def _finalize_result(self, result: UiActionResult) -> UiActionResult:
        self.last_result = result
        return result


class LocalUiApp:
    def __init__(self, root: tk.Tk, *, controller: LocalUiController | None = None) -> None:
        self.root = root
        self.controller = controller or LocalUiController()
        self.job_runner = LocalUiJobRunner()
        self._job_bindings: dict[int, UiJobBinding] = {}
        self.root.title(DEFAULT_UI_TITLE)
        self.root.geometry("1180x860")
        self.root.minsize(980, 720)
        self.root.configure(background="#EEF2F6")

        self.model_var = tk.StringVar(value=self.controller.selected_model_id)
        self.audio_model_var = tk.StringVar()
        self.capability_badges_var = tk.StringVar(value=self.controller.capability_badges_text())
        self.status_var = tk.StringVar(value="Ready. Local-only, single-user, shared-core UI.")
        self.backend_var = tk.StringVar(value="Backend: idle")
        self.device_var = tk.StringVar(value="Device: unresolved")
        self.artifact_var = tk.StringVar(value="Artifact: not written yet")
        self.hint_var = tk.StringVar(value="")
        self._animated_status_job_id: int | None = None
        self._animated_status_base: str = ""
        self._animated_status_frame: int = 0
        self._prewarm_hint_job_id: int | None = None
        self._startup_prewarm_after_id: str | int | None = None
        self._startup_prewarm_pending: bool = False
        self._startup_prewarm_reason: str = ""
        self._startup_prewarm_needs_idle_retry: bool = False

        self.chat_prompt = tk.StringVar(value=TEXT_DEFAULT_PROMPTS["chat"])
        self.chat_system = tk.StringVar(value="")

        self.vision_mode = tk.StringVar(value="caption")
        self.vision_input_one = tk.StringVar(value=str((repo_root() / "assets" / "images" / "sample.png").resolve()))
        self.vision_input_two = tk.StringVar(value=str((repo_root() / "assets" / "images" / "sample_compare.png").resolve()))
        self.vision_attachment_var = tk.StringVar(value="")
        self.vision_prompt = tk.StringVar(value=VISION_DEFAULT_PROMPTS["caption"])
        self.vision_system = tk.StringVar(value="")
        self.vision_max_pages = tk.IntVar(value=4)

        self.audio_mode = tk.StringVar(value="transcribe")
        self.audio_input = tk.StringVar(value=str((repo_root() / "assets" / "audio" / "sample_audio.wav").resolve()))
        self.audio_attachment_var = tk.StringVar(value="")
        self.audio_target_language = tk.StringVar(value=DEFAULT_TARGET_LANGUAGE)
        self.audio_prompt = tk.StringVar(value="")
        self.audio_system = tk.StringVar(value="")

        self.thinking_mode = tk.StringVar(value="text")
        self.thinking_prompt = tk.StringVar(value=TEXT_PROMPT)
        self.thinking_follow_up = tk.StringVar(value=TEXT_FOLLOW_UP)
        self.thinking_system = tk.StringVar(value="")
        self.debug_enabled = tk.BooleanVar(value=False)

        self.forensics_surface = tk.StringVar(value="chat")
        self.forensics_artifact_path = tk.StringVar(value="")
        self.forensics_show_raw = tk.BooleanVar(value=False)
        self.forensics_sessions_state = tk.StringVar(value="")
        self.forensics_entries_state = tk.StringVar(value="")
        self.forensics_quality_state = tk.StringVar(value="")
        self.forensics_lineage_state = tk.StringVar(value="")
        self.forensics_preview_state = tk.StringVar(value="")
        self.forensics_artifact_status_var = tk.StringVar(value="No artifact selected")
        self.forensics_artifact_validation_var = tk.StringVar(value="Choose an artifact")
        self.forensics_artifact_backend_var = tk.StringVar(value="n/a")
        self.forensics_artifact_model_var = tk.StringVar(value="n/a")
        self.forensics_artifact_timestamp_var = tk.StringVar(value="n/a")
        self.forensics_compare_previous_var = tk.StringVar(value="No older artifact is available in this session.")
        self.forensics_compare_selected_var = tk.StringVar(value="Choose a recorded artifact to compare.")
        self.forensics_compare_change_var = tk.StringVar(
            value="Select a recorded artifact to compare against the prior evidence."
        )
        self._forensics_selected_session_id: str | None = None
        self._forensics_selected_entry_id: str | None = None
        self._forensics_navigation_refreshing = False
        self._forensics_session_rows: dict[str, dict[str, Any]] = {}
        self._forensics_entry_rows: dict[str, dict[str, Any]] = {}
        self._forensics_preview_image: Any | None = None

        self.vision_input_one.trace_add("write", lambda *_args: self._update_vision_attachment_summary())
        self.vision_input_two.trace_add("write", lambda *_args: self._update_vision_attachment_summary())
        self.audio_input.trace_add("write", lambda *_args: self._update_audio_attachment_summary())

        self._configure_styles()
        self._build_layout()
        self._set_action_controls_state(False)
        self._refresh_audio_model()
        self._update_vision_attachment_summary()
        self._update_audio_attachment_summary()
        self.refresh_diagnostics()
        self.refresh_forensics()
        self.root.after(JOB_POLL_INTERVAL_MS, self._poll_job_events)
        self.root.after(STATUS_ANIMATION_INTERVAL_MS, self._tick_status_animation)
        self._schedule_startup_prewarm(reason="launch")

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("aqua")
        except tk.TclError:
            pass

        style.configure("Lab.TFrame", background="#EEF2F6")
        style.configure("Card.TFrame", background="#F8FAFC", relief="flat")
        style.configure("Lab.TLabel", background="#EEF2F6", foreground="#1F2937", font=("SF Pro Text", 11))
        style.configure("Title.TLabel", background="#EEF2F6", foreground="#111827", font=("SF Pro Display", 20, "bold"))
        style.configure("Subtitle.TLabel", background="#EEF2F6", foreground="#475569", font=("SF Pro Text", 11))
        style.configure("Section.TLabel", background="#F8FAFC", foreground="#111827", font=("SF Pro Text", 11, "bold"))
        style.configure("MetricCard.TFrame", background="#FFFFFF", relief="flat")
        style.configure("MetricTitle.TLabel", background="#FFFFFF", foreground="#64748B", font=("SF Pro Text", 10))
        style.configure("MetricValue.TLabel", background="#FFFFFF", foreground="#111827", font=("SF Pro Text", 12, "bold"))
        style.configure("MetricBody.TLabel", background="#FFFFFF", foreground="#1F2937", font=("SF Pro Text", 11))
        style.configure("Lab.TButton", font=("SF Pro Text", 11))
        style.configure("Lab.TNotebook", background="#EEF2F6", borderwidth=0)
        style.configure("Lab.TNotebook.Tab", font=("SF Pro Text", 11), padding=(16, 8))

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, style="Lab.TFrame", padding=18)
        outer.grid(sticky="nsew")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(1, weight=1)

        header = ttk.Frame(outer, style="Lab.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        header.grid_columnconfigure(1, weight=1)

        title_block = ttk.Frame(header, style="Lab.TFrame")
        title_block.grid(row=0, column=0, sticky="w", padx=(0, 18))
        ttk.Label(title_block, text=DEFAULT_UI_TITLE, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            title_block,
            text="Thin local UI over the shared core. Single-lane background worker, no subprocess runners, no hidden thinking by default.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(
            title_block,
            textvariable=self.capability_badges_var,
            style="Subtitle.TLabel",
            wraplength=700,
        ).grid(row=2, column=0, sticky="w", pady=(6, 0))

        controls = ttk.Frame(header, style="Lab.TFrame")
        controls.grid(row=0, column=1, sticky="e")
        ttk.Label(controls, text="Model", style="Lab.TLabel").grid(row=0, column=0, sticky="w")
        self.model_combo = ttk.Combobox(
            controls,
            textvariable=self.model_var,
            values=self._model_choices(),
            width=34,
        )
        self.model_combo.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self.apply_model_button = ttk.Button(controls, text="Apply", style="Lab.TButton", command=self.apply_model)
        self.apply_model_button.grid(row=1, column=1, sticky="ew")
        self.cancel_button = ttk.Button(controls, text="Cancel Job", style="Lab.TButton", command=self.cancel_active_job)
        self.cancel_button.grid(row=1, column=2, sticky="ew", padx=(8, 0))
        ttk.Label(controls, textvariable=self.audio_model_var, style="Subtitle.TLabel").grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        notebook = ttk.Notebook(outer, style="Lab.TNotebook")
        notebook.grid(row=1, column=0, sticky="nsew")
        outer.grid_rowconfigure(1, weight=1)

        self.chat_tab = self._build_chat_tab(notebook)
        self.vision_tab = self._build_vision_tab(notebook)
        self.audio_tab = self._build_audio_tab(notebook)
        self.thinking_tab = self._build_thinking_tab(notebook)
        self.forensics_tab = self._build_forensics_tab(notebook)
        self.diagnostics_tab = self._build_diagnostics_tab(notebook)

        footer = ttk.Frame(outer, style="Lab.TFrame", padding=(0, 12, 0, 0))
        footer.grid(row=2, column=0, sticky="ew")
        footer.grid_columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var, style="Lab.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(footer, textvariable=self.backend_var, style="Subtitle.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(footer, textvariable=self.device_var, style="Subtitle.TLabel").grid(row=2, column=0, sticky="w", pady=(2, 0))
        ttk.Label(footer, textvariable=self.artifact_var, style="Subtitle.TLabel", wraplength=1040).grid(row=3, column=0, sticky="w", pady=(2, 0))
        ttk.Label(footer, textvariable=self.hint_var, style="Subtitle.TLabel", wraplength=1040).grid(row=4, column=0, sticky="w", pady=(2, 0))

    def _build_chat_tab(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, style="Card.TFrame", padding=18)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(6, weight=1)
        notebook.add(frame, text="Chat")

        ttk.Label(frame, text="Prompt", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.chat_prompt_widget = self._multiline_entry(frame, row=1, initial=self.chat_prompt.get(), height=6)

        ttk.Label(frame, text="System Prompt Override", style="Section.TLabel").grid(row=2, column=0, sticky="w", pady=(12, 0))
        self.chat_system_entry = ttk.Entry(frame, textvariable=self.chat_system)
        self.chat_system_entry.grid(row=3, column=0, sticky="ew")

        self.run_chat_button = ttk.Button(frame, text="Run Chat", style="Lab.TButton", command=self.run_chat)
        self.run_chat_button.grid(row=4, column=0, sticky="w", pady=(14, 12))

        ttk.Label(frame, text="Output", style="Section.TLabel").grid(row=5, column=0, sticky="nw")
        self.chat_output = self._readonly_text(frame, row=6)
        return frame

    def _build_vision_tab(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, style="Card.TFrame", padding=18)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(9, weight=1)
        notebook.add(frame, text="Vision / PDF")

        ttk.Label(frame, text="Mode", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        mode_combo = ttk.Combobox(
            frame,
            textvariable=self.vision_mode,
            values=["caption", "vqa", "ocr", "compare", "pdf-summary"],
            width=20,
            state="readonly",
        )
        mode_combo.grid(row=0, column=1, sticky="w")
        mode_combo.bind("<<ComboboxSelected>>", self._on_vision_mode_change)

        ttk.Label(frame, text="Primary Input", style="Section.TLabel").grid(row=1, column=0, sticky="w", pady=(12, 0))
        self._path_row(frame, row=1, variable=self.vision_input_one, browse_command=self.browse_vision_primary)

        ttk.Label(frame, text="Second Input (compare only)", style="Section.TLabel").grid(row=2, column=0, sticky="w", pady=(12, 0))
        self.vision_second_path_entry, self.vision_second_path_button = self._path_row(
            frame,
            row=2,
            variable=self.vision_input_two,
            browse_command=self.browse_vision_secondary,
        )
        ttk.Label(frame, textvariable=self.vision_attachment_var, style="Subtitle.TLabel", wraplength=760).grid(
            row=3,
            column=1,
            columnspan=2,
            sticky="w",
            pady=(6, 0),
        )

        ttk.Label(frame, text="Prompt Override", style="Section.TLabel").grid(row=4, column=0, sticky="w", pady=(12, 0))
        self.vision_prompt_entry = ttk.Entry(frame, textvariable=self.vision_prompt)
        self.vision_prompt_entry.grid(row=4, column=1, sticky="ew")

        ttk.Label(frame, text="System Prompt Override", style="Section.TLabel").grid(row=5, column=0, sticky="w", pady=(12, 0))
        self.vision_system_entry = ttk.Entry(frame, textvariable=self.vision_system)
        self.vision_system_entry.grid(row=5, column=1, sticky="ew")

        ttk.Label(frame, text="Max PDF Pages", style="Section.TLabel").grid(row=6, column=0, sticky="w", pady=(12, 0))
        ttk.Spinbox(frame, from_=1, to=12, textvariable=self.vision_max_pages, width=6).grid(row=6, column=1, sticky="w")

        self.run_vision_button = ttk.Button(frame, text="Run Vision / PDF", style="Lab.TButton", command=self.run_vision)
        self.run_vision_button.grid(row=7, column=0, columnspan=2, sticky="w", pady=(14, 12))

        ttk.Label(frame, text="Output", style="Section.TLabel").grid(row=8, column=0, sticky="w")
        self.vision_output = self._readonly_text(frame, row=9, columnspan=2)
        self._on_vision_mode_change()
        return frame

    def _build_audio_tab(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, style="Card.TFrame", padding=18)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(8, weight=1)
        notebook.add(frame, text="Audio")

        ttk.Label(frame, text="Mode", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            frame,
            textvariable=self.audio_mode,
            values=["transcribe", "translate", "summarize"],
            width=20,
            state="readonly",
        ).grid(row=0, column=1, sticky="w")

        ttk.Label(frame, text="Audio Input", style="Section.TLabel").grid(row=1, column=0, sticky="w", pady=(12, 0))
        self._path_row(frame, row=1, variable=self.audio_input, browse_command=self.browse_audio)
        ttk.Label(frame, textvariable=self.audio_attachment_var, style="Subtitle.TLabel", wraplength=760).grid(
            row=2,
            column=1,
            columnspan=2,
            sticky="w",
            pady=(6, 0),
        )

        ttk.Label(frame, text="Target Language", style="Section.TLabel").grid(row=3, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(frame, textvariable=self.audio_target_language).grid(row=3, column=1, sticky="ew")

        ttk.Label(frame, text="Prompt Override", style="Section.TLabel").grid(row=4, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(frame, textvariable=self.audio_prompt).grid(row=4, column=1, sticky="ew")

        ttk.Label(frame, text="System Prompt Override", style="Section.TLabel").grid(row=5, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(frame, textvariable=self.audio_system).grid(row=5, column=1, sticky="ew")

        self.run_audio_button = ttk.Button(frame, text="Run Audio", style="Lab.TButton", command=self.run_audio)
        self.run_audio_button.grid(row=6, column=0, columnspan=2, sticky="w", pady=(14, 12))

        ttk.Label(frame, text="Output", style="Section.TLabel").grid(row=7, column=0, sticky="w")
        self.audio_output = self._readonly_text(frame, row=8, columnspan=2)
        return frame

    def _build_thinking_tab(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, style="Card.TFrame", padding=18)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(7, weight=1)
        frame.grid_rowconfigure(9, weight=1)
        notebook.add(frame, text="Tools / Thinking / Debug")

        ttk.Label(frame, text="Mode", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        thinking_mode_combo = ttk.Combobox(
            frame,
            textvariable=self.thinking_mode,
            values=["text", "tool"],
            width=20,
            state="readonly",
        )
        thinking_mode_combo.grid(row=0, column=1, sticky="w")
        thinking_mode_combo.bind("<<ComboboxSelected>>", self._on_thinking_mode_change)

        ttk.Label(frame, text="Prompt", style="Section.TLabel").grid(row=1, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(frame, textvariable=self.thinking_prompt).grid(row=1, column=1, sticky="ew")

        ttk.Label(frame, text="Follow-up (text mode only)", style="Section.TLabel").grid(row=2, column=0, sticky="w", pady=(12, 0))
        self.follow_up_entry = ttk.Entry(frame, textvariable=self.thinking_follow_up)
        self.follow_up_entry.grid(row=2, column=1, sticky="ew")

        ttk.Label(frame, text="System Prompt Override", style="Section.TLabel").grid(row=3, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(frame, textvariable=self.thinking_system).grid(row=3, column=1, sticky="ew")

        ttk.Checkbutton(
            frame,
            text="Show raw thinking and tool trace",
            variable=self.debug_enabled,
            command=self._toggle_debug_panel,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(12, 0))

        self.run_thinking_button = ttk.Button(frame, text="Run Thinking / Tool Mode", style="Lab.TButton", command=self.run_thinking)
        self.run_thinking_button.grid(row=5, column=0, columnspan=2, sticky="w", pady=(14, 12))

        ttk.Label(frame, text="Visible Output", style="Section.TLabel").grid(row=6, column=0, sticky="w")
        self.thinking_output = self._readonly_text(frame, row=7, columnspan=2)

        self.debug_label = ttk.Label(frame, text="Debug Trace", style="Section.TLabel")
        self.debug_output = self._readonly_text(frame, row=9, columnspan=2, height=10)
        self._on_thinking_mode_change()
        self._toggle_debug_panel()
        return frame

    def _build_forensics_tab(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, style="Card.TFrame", padding=18)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(10, weight=1)
        frame.grid_rowconfigure(16, weight=1)
        frame.grid_rowconfigure(19, weight=1)
        frame.grid_rowconfigure(22, weight=1)
        notebook.add(frame, text="History / Forensics")

        ttk.Label(frame, text="Surface", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        surface_combo = ttk.Combobox(
            frame,
            textvariable=self.forensics_surface,
            values=list(SESSION_SURFACES),
            width=18,
            state="readonly",
        )
        surface_combo.grid(row=0, column=1, sticky="w")
        surface_combo.bind("<<ComboboxSelected>>", self._on_forensics_surface_change)

        ttk.Label(frame, text="Recent Sessions", style="Section.TLabel").grid(row=1, column=0, sticky="w", pady=(12, 0))
        ttk.Label(
            frame,
            textvariable=self.forensics_sessions_state,
            style="Subtitle.TLabel",
            wraplength=920,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.forensics_sessions_tree = self._build_treeview(
            frame,
            row=3,
            columns=("session", "surface", "updated", "artifact", "validation"),
            headings=("Session", "Surface", "Updated", "Latest Artifact", "Validation"),
            columnspan=3,
            height=5,
        )
        self.forensics_sessions_tree.bind("<<TreeviewSelect>>", self._on_forensics_session_selected)

        ttk.Label(frame, text="Recorded Artifacts", style="Section.TLabel").grid(row=4, column=0, sticky="w", pady=(12, 0))
        ttk.Label(
            frame,
            textvariable=self.forensics_entries_state,
            style="Subtitle.TLabel",
            wraplength=920,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.forensics_entries_tree = self._build_treeview(
            frame,
            row=6,
            columns=("recorded", "mode", "status", "artifact", "validation"),
            headings=("Recorded", "Mode", "Status", "Artifact", "Validation"),
            columnspan=3,
            height=6,
        )
        self.forensics_entries_tree.bind("<<TreeviewSelect>>", self._on_forensics_entry_selected)

        ttk.Label(frame, text="Artifact Path Override", style="Section.TLabel").grid(row=7, column=0, sticky="w", pady=(12, 0))
        self.forensics_artifact_entry, self.forensics_browse_button = self._path_row(
            frame,
            row=7,
            variable=self.forensics_artifact_path,
            browse_command=self.browse_artifact,
        )

        controls = ttk.Frame(frame, style="Card.TFrame")
        controls.grid(row=8, column=1, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Button(controls, text="Use Surface Latest", style="Lab.TButton", command=self.use_surface_latest_artifact).grid(row=0, column=0, sticky="w")
        ttk.Button(controls, text="Use Last Result", style="Lab.TButton", command=self.use_last_result_artifact).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Button(controls, text="Refresh Forensics", style="Lab.TButton", command=self.refresh_forensics).grid(row=0, column=2, sticky="w", padx=(8, 0))

        ttk.Checkbutton(
            controls,
            text="Show raw artifact JSON",
            variable=self.forensics_show_raw,
            command=self.refresh_forensics,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(12, 0))

        ttk.Label(frame, text="Session Trail", style="Section.TLabel").grid(row=9, column=0, sticky="w", pady=(12, 0))
        self.forensics_history_output = self._readonly_text(frame, row=10, columnspan=3, height=11)

        ttk.Label(frame, text="Artifact Summary", style="Section.TLabel").grid(row=11, column=0, sticky="w", pady=(12, 0))
        summary_frame = ttk.Frame(frame, style="Card.TFrame")
        summary_frame.grid(row=12, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        for column in range(5):
            summary_frame.grid_columnconfigure(column, weight=1)
        self._build_metric_card(summary_frame, column=0, title="Status", variable=self.forensics_artifact_status_var)
        self._build_metric_card(summary_frame, column=1, title="Validation", variable=self.forensics_artifact_validation_var)
        self._build_metric_card(summary_frame, column=2, title="Backend", variable=self.forensics_artifact_backend_var)
        self._build_metric_card(summary_frame, column=3, title="Model", variable=self.forensics_artifact_model_var)
        self._build_metric_card(summary_frame, column=4, title="Timestamp", variable=self.forensics_artifact_timestamp_var)

        ttk.Label(frame, text="Compare Strip", style="Section.TLabel").grid(row=13, column=0, sticky="w", pady=(12, 0))
        compare_frame = ttk.Frame(frame, style="Card.TFrame")
        compare_frame.grid(row=14, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        for column in range(3):
            compare_frame.grid_columnconfigure(column, weight=1)
        self._build_compare_card(compare_frame, column=0, title="Previous", variable=self.forensics_compare_previous_var)
        self._build_compare_card(compare_frame, column=1, title="Selected", variable=self.forensics_compare_selected_var)
        self._build_compare_card(compare_frame, column=2, title="Change", variable=self.forensics_compare_change_var)

        ttk.Label(frame, text="Artifact Viewer", style="Section.TLabel").grid(row=15, column=0, sticky="w", pady=(12, 0))
        self.forensics_output = self._readonly_text(frame, row=16, columnspan=3, height=7)

        ttk.Label(frame, text="Quality Checks", style="Section.TLabel").grid(row=17, column=0, sticky="w", pady=(12, 0))
        ttk.Label(
            frame,
            textvariable=self.forensics_quality_state,
            style="Subtitle.TLabel",
            wraplength=920,
        ).grid(row=18, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.forensics_quality_tree = self._build_treeview(
            frame,
            row=19,
            columns=("verdict", "name", "detail"),
            headings=("Verdict", "Check", "Detail"),
            columnspan=3,
            height=4,
        )

        ttk.Label(frame, text="Preprocessing Lineage", style="Section.TLabel").grid(row=20, column=0, sticky="w", pady=(12, 0))
        ttk.Label(
            frame,
            textvariable=self.forensics_lineage_state,
            style="Subtitle.TLabel",
            wraplength=920,
        ).grid(row=21, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.forensics_lineage_tree = self._build_treeview(
            frame,
            row=22,
            columns=("asset_kind", "transform", "source", "resolved"),
            headings=("Kind", "Transform", "Source", "Resolved"),
            columnspan=2,
            height=4,
        )

        preview_frame = ttk.Frame(frame, style="Card.TFrame")
        preview_frame.grid(row=22, column=2, sticky="nsew", padx=(12, 0), pady=(6, 0))
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(2, weight=1)
        ttk.Label(preview_frame, text="Small Preview", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            preview_frame,
            textvariable=self.forensics_preview_state,
            style="Subtitle.TLabel",
            wraplength=300,
        ).grid(row=1, column=0, sticky="w", pady=(4, 8))
        self.forensics_preview_label = tk.Label(
            preview_frame,
            text="No preview available.",
            anchor="center",
            justify="center",
            background="#FFFFFF",
            foreground="#475569",
            relief="flat",
            borderwidth=1,
            padx=12,
            pady=12,
        )
        self.forensics_preview_label.grid(row=2, column=0, sticky="nsew")
        return frame

    def _build_diagnostics_tab(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, style="Card.TFrame", padding=18)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(3, weight=1)
        notebook.add(frame, text="Diagnostics")

        ttk.Label(frame, text="Scope", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            frame,
            text="Local-only diagnostics for runtime health, cached sessions, asset paths, and dependency visibility.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 10))
        self.refresh_diagnostics_button = ttk.Button(frame, text="Refresh Diagnostics", style="Lab.TButton", command=self.refresh_diagnostics)
        self.refresh_diagnostics_button.grid(row=2, column=0, sticky="w", pady=(0, 10))
        self.diagnostics_output = self._readonly_text(frame, row=3, height=26)
        return frame

    def _multiline_entry(self, parent: ttk.Frame, *, row: int, initial: str, height: int) -> scrolledtext.ScrolledText:
        widget = scrolledtext.ScrolledText(
            parent,
            height=height,
            wrap=tk.WORD,
            font=("SF Pro Text", 11),
            background="#FFFFFF",
            foreground="#111827",
            insertbackground="#111827",
            relief="flat",
            borderwidth=1,
        )
        widget.grid(row=row, column=0, sticky="nsew", pady=(6, 0))
        widget.insert("1.0", initial)
        return widget

    def _readonly_text(
        self,
        parent: ttk.Frame,
        *,
        row: int,
        columnspan: int = 1,
        height: int = 16,
    ) -> scrolledtext.ScrolledText:
        widget = scrolledtext.ScrolledText(
            parent,
            height=height,
            wrap=tk.WORD,
            font=("SF Pro Text", 11),
            background="#FFFFFF",
            foreground="#111827",
            relief="flat",
            borderwidth=1,
        )
        widget.grid(row=row, column=0, columnspan=columnspan, sticky="nsew", pady=(6, 0))
        widget.configure(state=tk.DISABLED)
        return widget

    def _build_treeview(
        self,
        parent: ttk.Frame,
        *,
        row: int,
        columns: tuple[str, ...],
        headings: tuple[str, ...],
        columnspan: int = 1,
        height: int = 5,
    ) -> ttk.Treeview:
        wrapper = ttk.Frame(parent, style="Card.TFrame")
        wrapper.grid(row=row, column=0, columnspan=columnspan, sticky="nsew", pady=(6, 0))
        wrapper.grid_columnconfigure(0, weight=1)
        wrapper.grid_rowconfigure(0, weight=1)

        tree = ttk.Treeview(
            wrapper,
            columns=columns,
            show="headings",
            height=height,
            selectmode="browse",
        )
        for column, heading in zip(columns, headings):
            anchor = "w"
            width = 120
            if column in {"artifact", "validation"}:
                width = 220
            elif column in {"detail", "resolved"}:
                width = 280
            elif column == "source":
                width = 240
            elif column in {"session", "recorded"}:
                width = 180
            elif column == "updated":
                width = 170
            tree.heading(column, text=heading)
            tree.column(column, anchor=anchor, width=width, stretch=True)

        scrollbar = ttk.Scrollbar(wrapper, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        return tree

    def _build_metric_card(
        self,
        parent: ttk.Frame,
        *,
        column: int,
        title: str,
        variable: tk.StringVar,
    ) -> ttk.Frame:
        card = ttk.Frame(parent, style="MetricCard.TFrame", padding=(12, 10))
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0))
        ttk.Label(card, text=title, style="MetricTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            card,
            textvariable=variable,
            style="MetricValue.TLabel",
            wraplength=180,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        return card

    def _build_compare_card(
        self,
        parent: ttk.Frame,
        *,
        column: int,
        title: str,
        variable: tk.StringVar,
    ) -> ttk.Frame:
        card = ttk.Frame(parent, style="MetricCard.TFrame", padding=(12, 10))
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0))
        ttk.Label(card, text=title, style="MetricTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            card,
            textvariable=variable,
            style="MetricBody.TLabel",
            wraplength=290,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        return card

    def _path_row(
        self,
        parent: ttk.Frame,
        *,
        row: int,
        variable: tk.StringVar,
        browse_command: Any,
    ) -> tuple[ttk.Entry, ttk.Button]:
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=(12, 0), padx=(0, 8))
        button = ttk.Button(parent, text="Browse", style="Lab.TButton", command=browse_command)
        button.grid(row=row, column=2, sticky="ew", pady=(12, 0))
        return entry, button

    def _model_choices(self) -> list[str]:
        choices = list(DEFAULT_MODEL_CHOICES)
        current = resolve_model_id()
        if current not in choices:
            choices.insert(0, current)
        return choices

    def _set_output(self, widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text[:CHAT_OUTPUT_LIMIT])
        widget.configure(state=tk.DISABLED)

    def _on_vision_mode_change(self, *_args: object) -> None:
        mode = self.vision_mode.get()
        self.vision_prompt.set(VISION_DEFAULT_PROMPTS[mode])
        enabled = supports_second_vision_input(mode)
        state = "normal" if enabled else "disabled"
        self.vision_second_path_entry.configure(state=state)
        self.vision_second_path_button.configure(state=state)
        self._update_vision_attachment_summary()

    def _on_thinking_mode_change(self, *_args: object) -> None:
        mode = self.thinking_mode.get()
        self.thinking_prompt.set(TEXT_PROMPT if mode == "text" else TOOL_PROMPT)
        self.follow_up_entry.configure(state="normal" if mode == "text" else "disabled")

    def _toggle_debug_panel(self) -> None:
        if self.debug_enabled.get():
            self.debug_label.grid(row=8, column=0, sticky="w", pady=(12, 0))
            self.debug_output.grid()
        else:
            self.debug_label.grid_remove()
            self.debug_output.grid_remove()

    def _refresh_audio_model(self) -> None:
        model_id, source = self.controller.resolved_audio_model()
        self.audio_model_var.set(f"Audio resolves to {model_id} ({source}).")
        self.capability_badges_var.set(self.controller.capability_badges_text())

    def _update_vision_attachment_summary(self) -> None:
        paths = [self.vision_input_one.get().strip()]
        if supports_second_vision_input(self.vision_mode.get()):
            paths.append(self.vision_input_two.get().strip())
        resolved = [path for path in paths if path]
        if not resolved:
            self.vision_attachment_var.set("Attached assets: none selected.")
            return
        labels = [Path(path).name for path in resolved]
        self.vision_attachment_var.set(f"Attached assets: {', '.join(labels)}")

    def _update_audio_attachment_summary(self) -> None:
        path = self.audio_input.get().strip()
        if not path:
            self.audio_attachment_var.set("Attached audio: none selected.")
            return
        self.audio_attachment_var.set(f"Attached audio: {Path(path).name}")

    def _on_forensics_surface_change(self, _event: object | None = None) -> None:
        self._forensics_selected_session_id = None
        self._forensics_selected_entry_id = None
        self.refresh_forensics()

    def _on_forensics_session_selected(self, _event: object | None = None) -> None:
        if self._forensics_navigation_refreshing:
            return
        selection = self.forensics_sessions_tree.selection()
        if not selection:
            return
        session_id = str(selection[0])
        row = self._forensics_session_rows.get(session_id)
        self._forensics_selected_session_id = session_id
        self._forensics_selected_entry_id = None
        if row is not None and row.get("surface"):
            self.forensics_surface.set(str(row["surface"]))
        self.forensics_artifact_path.set("")
        self.refresh_forensics()

    def _on_forensics_entry_selected(self, _event: object | None = None) -> None:
        if self._forensics_navigation_refreshing:
            return
        selection = self.forensics_entries_tree.selection()
        if not selection:
            return
        self._forensics_selected_entry_id = str(selection[0])
        self.forensics_artifact_path.set("")
        self.refresh_forensics()

    def _populate_forensics_navigation(self, snapshot: dict[str, Any]) -> None:
        recent_sessions = list(snapshot.get("recent_sessions") or [])
        recent_entries = list(snapshot.get("recent_entries") or [])
        self._forensics_selected_session_id = snapshot.get("selected_session_id")
        self._forensics_selected_entry_id = snapshot.get("selected_entry_id")
        self._forensics_session_rows = {
            str(row["session_id"]): row
            for row in recent_sessions
            if row.get("session_id")
        }
        self._forensics_entry_rows = {
            str(row["entry_id"]): row
            for row in recent_entries
            if row.get("entry_id")
        }

        if not recent_sessions:
            self.forensics_sessions_state.set(
                "No recorded sessions yet. Run Chat, Vision / PDF, Audio, or Thinking once to seed forensics."
            )
        elif snapshot.get("selected_session_error"):
            self.forensics_sessions_state.set(snapshot["selected_session_error"])
        else:
            self.forensics_sessions_state.set(
                "Choose a recent session to inspect its latest artifact and recorded run trail."
            )

        if snapshot.get("selected_entry_error"):
            self.forensics_entries_state.set(snapshot["selected_entry_error"])
        elif snapshot.get("selected_session_summary") is None:
            self.forensics_entries_state.set("Select a recent session to load its recorded artifacts.")
        elif not recent_entries:
            self.forensics_entries_state.set("This session has no recorded artifacts yet.")
        else:
            self.forensics_entries_state.set("Choose a recorded artifact to open it without pasting a path.")

        self._forensics_navigation_refreshing = True
        try:
            session_children = self.forensics_sessions_tree.get_children()
            if session_children:
                self.forensics_sessions_tree.delete(*session_children)
            for row in recent_sessions:
                session_id = row.get("session_id")
                if not session_id:
                    continue
                label = str(row.get("title") or session_id)
                if row.get("title") and row.get("session_id") and row["title"] != row["session_id"]:
                    label = f"{row['title']} ({row['session_id']})"
                self.forensics_sessions_tree.insert(
                    "",
                    "end",
                    iid=str(session_id),
                    values=(
                        label,
                        row.get("surface") or "n/a",
                        row.get("updated_at_utc") or "n/a",
                        row.get("latest_artifact_display") or "n/a",
                        row.get("latest_validation_summary") or "n/a",
                    ),
                )
            if (
                self._forensics_selected_session_id is not None
                and self.forensics_sessions_tree.exists(self._forensics_selected_session_id)
            ):
                self.forensics_sessions_tree.selection_set(self._forensics_selected_session_id)
                self.forensics_sessions_tree.focus(self._forensics_selected_session_id)

            entry_children = self.forensics_entries_tree.get_children()
            if entry_children:
                self.forensics_entries_tree.delete(*entry_children)
            for row in recent_entries:
                entry_id = row.get("entry_id")
                if not entry_id:
                    continue
                self.forensics_entries_tree.insert(
                    "",
                    "end",
                    iid=str(entry_id),
                    values=(
                        row.get("recorded_at_utc") or "n/a",
                        row.get("mode") or "n/a",
                        row.get("status") or "n/a",
                        row.get("artifact_display") or "n/a",
                        row.get("validation_summary") or "n/a",
                    ),
                )
            if (
                self._forensics_selected_entry_id is not None
                and self.forensics_entries_tree.exists(self._forensics_selected_entry_id)
            ):
                self.forensics_entries_tree.selection_set(self._forensics_selected_entry_id)
                self.forensics_entries_tree.focus(self._forensics_selected_entry_id)
        finally:
            self._forensics_navigation_refreshing = False

    def _populate_quality_checks(self, snapshot: dict[str, Any]) -> None:
        summary = snapshot.get("artifact_summary") or {}
        quality_checks = list(summary.get("quality_checks") or [])
        if snapshot.get("artifact_error"):
            self.forensics_quality_state.set("Artifact could not be read, so quality checks are unavailable.")
        elif not snapshot.get("artifact_path"):
            self.forensics_quality_state.set("Select an artifact to inspect its validation checks.")
        elif not quality_checks:
            self.forensics_quality_state.set("No explicit quality checks were recorded for this artifact.")
        else:
            self.forensics_quality_state.set("Recorded quality checks come directly from the artifact validation record.")

        children = self.forensics_quality_tree.get_children()
        if children:
            self.forensics_quality_tree.delete(*children)
        for index, check in enumerate(quality_checks, start=1):
            verdict = "PASS" if check.get("pass") else "FAIL"
            self.forensics_quality_tree.insert(
                "",
                "end",
                iid=f"quality-{index}",
                values=(
                    verdict,
                    check.get("name") or "n/a",
                    check.get("detail") or "n/a",
                ),
            )

    def _populate_lineage(self, snapshot: dict[str, Any]) -> None:
        summary = snapshot.get("artifact_summary") or {}
        lineage = list(summary.get("lineage_preview") or [])
        if snapshot.get("artifact_error"):
            self.forensics_lineage_state.set("Artifact could not be read, so preprocessing lineage is unavailable.")
        elif not snapshot.get("artifact_path"):
            self.forensics_lineage_state.set("Select an artifact to inspect its preprocessing lineage.")
        elif not lineage:
            self.forensics_lineage_state.set("No preprocessing lineage was recorded for this artifact.")
        else:
            self.forensics_lineage_state.set("Lineage rows show the recorded source and resolved files used by this artifact.")

        children = self.forensics_lineage_tree.get_children()
        if children:
            self.forensics_lineage_tree.delete(*children)
        for index, item in enumerate(lineage, start=1):
            self.forensics_lineage_tree.insert(
                "",
                "end",
                iid=f"lineage-{index}",
                values=(
                    item.get("asset_kind") or "n/a",
                    item.get("transform") or "n/a",
                    summarize_workspace_path(item.get("source_path"), root=self.controller.workspace_store.root),
                    summarize_workspace_path(item.get("resolved_path"), root=self.controller.workspace_store.root),
                ),
            )

    def _clear_preview_image(self, *, message: str) -> None:
        self._forensics_preview_image = None
        self.forensics_preview_label.configure(image="", text=message)

    def _populate_preview_image(self, snapshot: dict[str, Any]) -> None:
        summary = snapshot.get("artifact_summary") or {}
        preview_path = summary.get("preview_image_path")
        if snapshot.get("artifact_error"):
            self.forensics_preview_state.set("Artifact could not be read, so no preview can be shown.")
            self._clear_preview_image(message="Preview unavailable.")
            return
        if not snapshot.get("artifact_path"):
            self.forensics_preview_state.set("Select an artifact to inspect a small preview when one is available.")
            self._clear_preview_image(message="No preview available.")
            return
        if not preview_path:
            self.forensics_preview_state.set("This artifact does not point to a previewable image file.")
            self._clear_preview_image(message="No preview available.")
            return

        preview_file = Path(str(preview_path)).expanduser().resolve()
        self.forensics_preview_state.set(
            f"Preview source: {summarize_workspace_path(str(preview_file), root=self.controller.workspace_store.root)}"
        )
        try:
            image = tk.PhotoImage(file=str(preview_file))
            width = image.width()
            height = image.height()
            scale = max(
                1,
                (width + PREVIEW_MAX_WIDTH - 1) // PREVIEW_MAX_WIDTH,
                (height + PREVIEW_MAX_HEIGHT - 1) // PREVIEW_MAX_HEIGHT,
            )
            if scale > 1:
                image = image.subsample(scale, scale)
            self._forensics_preview_image = image
            self.forensics_preview_label.configure(image=image, text="")
        except tk.TclError as exc:
            self.forensics_preview_state.set(
                f"Preview source: {summarize_workspace_path(str(preview_file), root=self.controller.workspace_store.root)}"
            )
            self._clear_preview_image(message=f"Preview failed to load ({type(exc).__name__}).")

    def _populate_artifact_overview(self, snapshot: dict[str, Any]) -> None:
        apply_artifact_overview_fields(
            snapshot,
            status_var=self.forensics_artifact_status_var,
            validation_var=self.forensics_artifact_validation_var,
            backend_var=self.forensics_artifact_backend_var,
            model_var=self.forensics_artifact_model_var,
            timestamp_var=self.forensics_artifact_timestamp_var,
        )

    def _populate_compare_strip(self, snapshot: dict[str, Any]) -> None:
        apply_entry_compare_fields(
            snapshot,
            previous_var=self.forensics_compare_previous_var,
            selected_var=self.forensics_compare_selected_var,
            change_var=self.forensics_compare_change_var,
        )

    def _set_busy(self, message: str) -> None:
        self.status_var.set(message)
        self.root.configure(cursor="watch")
        self._set_action_controls_state(True)
        self.root.update_idletasks()

    def _clear_busy(self) -> None:
        self.root.configure(cursor="")
        self._set_action_controls_state(False)
        self.root.update_idletasks()

    def _set_action_controls_state(self, busy: bool) -> None:
        run_state = "disabled" if busy else "normal"
        cancel_state = "normal" if busy else "disabled"
        self.model_combo.configure(state="disabled" if busy else "normal")
        self.apply_model_button.configure(state=run_state)
        self.cancel_button.configure(state=cancel_state)
        self.run_chat_button.configure(state=run_state)
        self.run_vision_button.configure(state=run_state)
        self.run_audio_button.configure(state=run_state)
        self.run_thinking_button.configure(state=run_state)
        self.refresh_diagnostics_button.configure(state=run_state)

    def _begin_status_animation(self, *, job_id: int, base_message: str) -> None:
        normalized = (base_message or "").strip().rstrip(".")
        self._animated_status_job_id = job_id
        self._animated_status_base = normalized or "Working"
        self._animated_status_frame = 0
        self.status_var.set(self._animated_status_base)

    def _clear_status_animation(self, *, job_id: int | None = None) -> None:
        if job_id is not None and self._animated_status_job_id != job_id:
            return
        self._animated_status_job_id = None
        self._animated_status_base = ""
        self._animated_status_frame = 0

    def _tick_status_animation(self) -> None:
        if self._animated_status_job_id is not None and self._animated_status_base:
            suffix = ("", ".", "..", "...")[self._animated_status_frame % 4]
            self.status_var.set(f"{self._animated_status_base}{suffix}")
            self._animated_status_frame += 1
        self.root.after(STATUS_ANIMATION_INTERVAL_MS, self._tick_status_animation)

    def _begin_prewarm_hints(self, *, job_id: int) -> None:
        self._prewarm_hint_job_id = job_id

    def _clear_prewarm_hints(self, *, job_id: int | None = None, final_hint: str = "") -> None:
        if job_id is not None and self._prewarm_hint_job_id != job_id:
            return
        self._prewarm_hint_job_id = None
        self.hint_var.set(final_hint)

    def _prewarm_phase_label(self, snapshot: UiJobSnapshot) -> str:
        if snapshot.progress_phase == WARMUP_PHASE_ATTACH_MODEL and snapshot.message.startswith("Placing model on "):
            return snapshot.message
        return PREWARM_PHASE_LABELS.get(snapshot.progress_phase or "", snapshot.message or "Running prewarm on the shared core")

    def _prewarm_running_status(self, snapshot: UiJobSnapshot) -> str:
        label = self._prewarm_phase_label(snapshot)
        if snapshot.cancel_requested:
            if snapshot.progress_phase == WARMUP_PHASE_PRIME_TOKEN:
                return f"{label}. Cancel accepted; stop should take effect quickly."
            if snapshot.progress_phase in PREWARM_DEFERRED_CANCEL_PHASES:
                return f"{label}. Cancel accepted; lane release waits for the current step to return."
            return f"{label}. Cancel accepted."
        return label

    def _prewarm_hint_for_snapshot(self, snapshot: UiJobSnapshot) -> str:
        phase = snapshot.progress_phase
        if snapshot.cancel_requested:
            if phase == WARMUP_PHASE_PRIME_TOKEN:
                return (
                    "Cancel accepted. Immediate effect is expected here because the prime decode "
                    "checks the stop signal."
                )
            if phase in PREWARM_DEFERRED_CANCEL_PHASES:
                return (
                    "Cancel accepted. Immediate effect is not expected here; the worker lane clears "
                    "after the current library call returns."
                )
            return "Cancel accepted. The worker lane will clear at the next safe boundary."

        if phase == WARMUP_PHASE_LOAD_PROCESSOR:
            return (
                "Processor assets are loading now. Cancel is accepted, but the worker lane stays busy "
                "until the loader returns."
            )
        if phase == WARMUP_PHASE_LOAD_MODEL:
            return (
                "Model weights are loading now. Cancel is accepted, but the lane cannot clear until "
                "that load call returns."
            )
        if phase == WARMUP_PHASE_ATTACH_MODEL:
            return (
                "The model is being attached to the runtime device. Cancel is accepted, but lane "
                "release is deferred until that attach step finishes."
            )
        if phase == WARMUP_PHASE_PRIME_TOKEN:
            return (
                "The first-token prime decode is active. Cancel is accepted here, and lane release "
                "should be comparatively fast."
            )
        if phase == WARMUP_PHASE_SESSION_READY:
            return "Shared session is ready. The tiny first-token prime is about to start."
        return "Warmup is running on the same shared-core path as real work."

    def _startup_prewarm_pending_hint(self, *, reason: str) -> str:
        if reason == "launch":
            return "Warmup will start once launch settles unless you run something first."
        return "Warmup was deferred. It will retry once the lane stays idle for a moment."

    def _startup_prewarm_started_reason(self, *, reason: str) -> str:
        if reason == "launch":
            return "launch settled"
        return "the lane returned to idle"

    def _clear_startup_prewarm_pending(self) -> None:
        self._startup_prewarm_after_id = None
        self._startup_prewarm_pending = False
        self._startup_prewarm_reason = ""

    def _schedule_startup_prewarm(self, *, reason: str) -> bool:
        if self._startup_prewarm_pending or self.job_runner.has_pending_work():
            return False
        if self.controller.has_cached_selected_text_session():
            if self._startup_prewarm_needs_idle_retry:
                self.hint_var.set("")
            self._startup_prewarm_needs_idle_retry = False
            return False

        self._startup_prewarm_pending = True
        self._startup_prewarm_reason = reason
        self._startup_prewarm_needs_idle_retry = False
        self._startup_prewarm_after_id = self.root.after(
            STARTUP_PREWARM_GRACE_MS,
            self._start_startup_prewarm,
        )
        self.hint_var.set(self._startup_prewarm_pending_hint(reason=reason))
        return True

    def _cancel_pending_startup_prewarm(self) -> bool:
        after_id = self._startup_prewarm_after_id
        if not self._startup_prewarm_pending or after_id is None:
            return False
        try:
            self.root.after_cancel(after_id)
        except tk.TclError:
            pass
        self._clear_startup_prewarm_pending()
        self._startup_prewarm_needs_idle_retry = True
        self.hint_var.set("Startup warmup deferred so your action can start first.")
        return True

    def _resume_startup_prewarm_if_needed(self) -> bool:
        if not self._startup_prewarm_needs_idle_retry:
            return False
        if self.job_runner.has_pending_work():
            return False
        if self.controller.has_cached_selected_text_session():
            self._startup_prewarm_needs_idle_retry = False
            self.hint_var.set("")
            return False
        return self._schedule_startup_prewarm(reason="idle")

    def _apply_result(self, result: UiActionResult, *, output_widget: scrolledtext.ScrolledText, debug_widget: scrolledtext.ScrolledText | None = None) -> None:
        self._set_output(output_widget, result.output_text)
        if debug_widget is not None:
            self._set_output(debug_widget, result.debug_text if self.debug_enabled.get() else "")

        self.status_var.set(f"{result.action}: {result.status}")
        self.backend_var.set(f"Backend: {result.backend}")
        self.device_var.set(f"Device: {result.device_label or 'unresolved'} / {result.dtype_name or 'unknown'}")
        self.artifact_var.set(f"Artifact: {result.artifact_path}")
        if result.action in SESSION_SURFACES:
            self.forensics_surface.set(result.action)
        self._forensics_selected_session_id = None
        self._forensics_selected_entry_id = None
        self.forensics_artifact_path.set(str(result.artifact_path.resolve()))
        self.refresh_forensics()

    def _submit_job(
        self,
        *,
        action: str,
        running_message: str,
        output_widget: scrolledtext.ScrolledText | None,
        work: Callable[[CancellationSignal, JobProgressCallback], UiActionResult],
        debug_widget: scrolledtext.ScrolledText | None = None,
    ) -> None:
        if action != PREWARM_ACTION:
            self._cancel_pending_startup_prewarm()
        active_job = self.job_runner.active_job()
        if active_job is not None:
            self.status_var.set(
                f"{active_job.action} is already {active_job.state}. "
                "This UI stays single-lane; wait for it or cancel it first."
            )
            return

        snapshot = self.job_runner.submit(
            action=action,
            work=work,
            timeout_seconds=DEFAULT_ACTION_TIMEOUT_SECONDS.get(action),
            message=running_message,
        )
        if output_widget is not None or debug_widget is not None:
            self._job_bindings[snapshot.job_id] = UiJobBinding(
                output_widget=output_widget,
                debug_widget=debug_widget,
            )
        self._set_busy(running_message)

    def _submit_prewarm(self, *, reason: str) -> None:
        self._startup_prewarm_needs_idle_retry = False
        if self.job_runner.active_job() is not None:
            return
        if self.controller.has_cached_selected_text_session():
            self.hint_var.set("Warmup skipped. The selected model already has a cached shared session.")
            return

        self._submit_job(
            action=PREWARM_ACTION,
            running_message=f"Starting background warmup for {self.controller.selected_model_id}",
            output_widget=None,
            work=lambda cancellation_signal, report_progress: self.controller.prewarm_selected_model(
                cancellation_signal=cancellation_signal,
                progress_callback=report_progress,
            ),
        )
        self.hint_var.set(f"Warmup started after {reason}.")

    def _start_startup_prewarm(self) -> None:
        reason = self._startup_prewarm_reason or "launch"
        self._clear_startup_prewarm_pending()
        self._submit_prewarm(reason=self._startup_prewarm_started_reason(reason=reason))

    def _poll_job_events(self) -> None:
        self.job_runner.expire_timeouts()
        for snapshot in self.job_runner.pop_events():
            self._handle_job_snapshot(snapshot)
        self.root.after(JOB_POLL_INTERVAL_MS, self._poll_job_events)

    def _handle_job_snapshot(self, snapshot: UiJobSnapshot) -> None:
        binding = self._job_bindings.get(snapshot.job_id)
        is_prewarm = snapshot.action == PREWARM_ACTION

        if snapshot.state == JOB_STATE_QUEUED:
            self._clear_status_animation(job_id=snapshot.job_id)
            self.status_var.set(snapshot.message or f"{snapshot.action} queued.")
            self.backend_var.set("Backend: queued")
            if is_prewarm:
                self._begin_prewarm_hints(job_id=snapshot.job_id)
            return

        if snapshot.state == JOB_STATE_RUNNING:
            if is_prewarm and not snapshot.cancel_requested and not snapshot.deadline_exceeded:
                self._begin_status_animation(
                    job_id=snapshot.job_id,
                    base_message=self._prewarm_phase_label(snapshot),
                )
                self._begin_prewarm_hints(job_id=snapshot.job_id)
            elif is_prewarm:
                self._clear_status_animation(job_id=snapshot.job_id)
            else:
                self._clear_status_animation(job_id=snapshot.job_id)
            if snapshot.cancel_requested and is_prewarm:
                self.status_var.set(self._prewarm_running_status(snapshot))
                self.hint_var.set(self._prewarm_hint_for_snapshot(snapshot))
            elif snapshot.cancel_requested:
                self.status_var.set(
                    f"{snapshot.action} cancellation requested. "
                    "The current run will stop at the next safe boundary."
                )
            elif snapshot.deadline_exceeded:
                self.status_var.set(snapshot.message)
                if is_prewarm:
                    self.hint_var.set(
                        "Warmup exceeded its timeout. The lane will clear once the current underlying call returns."
                    )
            elif is_prewarm:
                self.status_var.set(self._prewarm_running_status(snapshot))
                self.hint_var.set(self._prewarm_hint_for_snapshot(snapshot))
            elif not is_prewarm:
                self.status_var.set(snapshot.message or f"Running {snapshot.action}...")
            self.backend_var.set(f"Backend: worker/{snapshot.action}")
            return

        self._clear_status_animation(job_id=snapshot.job_id)

        if snapshot.state == JOB_STATE_COMPLETED and snapshot.result is not None and binding is not None:
            self._apply_result(
                snapshot.result,
                output_widget=binding.output_widget,
                debug_widget=binding.debug_widget,
            )
        elif snapshot.state == JOB_STATE_COMPLETED and snapshot.result is not None and is_prewarm:
            self.status_var.set(snapshot.result.output_text)
            self.backend_var.set(f"Backend: worker/{snapshot.action}")
            self.device_var.set(
                f"Device: {snapshot.result.device_label or 'unresolved'} / {snapshot.result.dtype_name or 'unknown'}"
            )
            self.hint_var.set("Warmup complete. The first real thinking turn should start noticeably faster.")
        elif binding is not None:
            self._set_output(binding.output_widget, snapshot.message)
            if binding.debug_widget is not None:
                self._set_output(binding.debug_widget, "")
            self.status_var.set(snapshot.message)
            self.backend_var.set(f"Backend: worker/{snapshot.action}")
        elif is_prewarm:
            self.status_var.set(snapshot.message)
            self.backend_var.set(f"Backend: worker/{snapshot.action}")
            if snapshot.state == JOB_STATE_CANCELLED:
                if snapshot.progress_phase == WARMUP_PHASE_PRIME_TOKEN:
                    self.hint_var.set(
                        "Warmup cancelled during the prime decode. The lane should have cleared quickly, "
                        "but the next first-run action may still pay some cold-start cost."
                    )
                else:
                    self.hint_var.set("Warmup cancelled. The next first-run action may pay the cold-start cost.")
            elif snapshot.state == JOB_STATE_FAILED:
                self.hint_var.set("Warmup failed. Diagnostics can help confirm the runtime state.")

        if snapshot.state in {JOB_STATE_COMPLETED, JOB_STATE_FAILED, JOB_STATE_CANCELLED}:
            self._job_bindings.pop(snapshot.job_id, None)
            self._clear_prewarm_hints(job_id=snapshot.job_id, final_hint=self.hint_var.get())
            if not self.job_runner.has_pending_work():
                self._clear_busy()
                if is_prewarm:
                    self.refresh_diagnostics()
                else:
                    self._resume_startup_prewarm_if_needed()

    def apply_model(self) -> None:
        if self.job_runner.has_pending_work():
            self.status_var.set("Finish or cancel the current job before changing models.")
            return
        self._cancel_pending_startup_prewarm()
        selected = self.model_var.get().strip() or self.controller.selected_model_id
        changed = selected != self.controller.selected_model_id
        self._set_busy("Applying model selection...")
        try:
            selected = self.controller.set_model_id(self.model_var.get())
            self.model_var.set(selected)
            self._refresh_audio_model()
            self.status_var.set(f"Selected model: {selected}")
            self.refresh_diagnostics()
            self.refresh_forensics()
        finally:
            self._clear_busy()
        if changed:
            self._startup_prewarm_needs_idle_retry = False
            self._submit_prewarm(reason="model change")
        else:
            self._resume_startup_prewarm_if_needed()

    def cancel_active_job(self) -> None:
        snapshot = self.job_runner.cancel_active()
        if snapshot is None:
            self.status_var.set("No queued or running job to cancel.")
            return
        if snapshot.state == JOB_STATE_RUNNING and snapshot.action == PREWARM_ACTION:
            self.status_var.set(self._prewarm_running_status(snapshot))
            self.hint_var.set(self._prewarm_hint_for_snapshot(snapshot))
            return
        if snapshot.state == JOB_STATE_RUNNING:
            self.status_var.set(
                f"{snapshot.action} cancellation requested. "
                "The current run will stop at the next safe boundary."
            )
            return
        self.status_var.set(snapshot.message)

    def browse_vision_primary(self) -> None:
        self._browse_path(self.vision_input_one, filetypes=[("Images and PDFs", "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff *.pdf")])

    def browse_vision_secondary(self) -> None:
        self._browse_path(self.vision_input_two, filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff")])

    def browse_audio(self) -> None:
        self._browse_path(self.audio_input, filetypes=[("Audio", "*.wav *.mp3 *.m4a *.flac *.aac")])

    def browse_artifact(self) -> None:
        self._browse_path(self.forensics_artifact_path, filetypes=[("Artifact JSON", "*.json")])

    def use_surface_latest_artifact(self) -> None:
        self._forensics_selected_session_id = None
        self._forensics_selected_entry_id = None
        latest = self.controller.latest_artifact_path(surface=self.forensics_surface.get())
        self.forensics_artifact_path.set(latest or "")
        self.refresh_forensics()

    def use_last_result_artifact(self) -> None:
        if self.controller.last_result is None:
            self.status_var.set("No completed action is available yet.")
            return
        self._forensics_selected_entry_id = None
        self.forensics_artifact_path.set(str(self.controller.last_result.artifact_path.resolve()))
        self.refresh_forensics()

    def _browse_path(self, variable: tk.StringVar, *, filetypes: list[tuple[str, str]]) -> None:
        selected = filedialog.askopenfilename(
            title="Choose a local file",
            initialdir=str(repo_root()),
            filetypes=filetypes,
        )
        if selected:
            variable.set(selected)

    def run_chat(self) -> None:
        prompt = self.chat_prompt_widget.get("1.0", tk.END).strip()
        system_prompt = self.chat_system.get().strip() or None
        self._submit_job(
            action="chat",
            running_message="Running chat on the shared core...",
            output_widget=self.chat_output,
            work=lambda cancellation_signal, _report_progress: self.controller.run_chat(
                prompt=prompt,
                system_prompt=system_prompt,
                cancellation_signal=cancellation_signal,
            ),
        )

    def run_vision(self) -> None:
        mode = self.vision_mode.get()
        inputs = [self.vision_input_one.get()]
        if supports_second_vision_input(mode):
            inputs.append(self.vision_input_two.get())
        prompt = self.vision_prompt.get().strip() or None
        system_prompt = self.vision_system.get().strip() or None
        max_pages = int(self.vision_max_pages.get())
        self._submit_job(
            action="vision",
            running_message="Running vision/PDF on the shared core...",
            output_widget=self.vision_output,
            work=lambda cancellation_signal, _report_progress: self.controller.run_vision(
                mode=mode,
                inputs=list(inputs),
                prompt=prompt,
                system_prompt=system_prompt,
                max_pages=max_pages,
                cancellation_signal=cancellation_signal,
            ),
        )

    def run_audio(self) -> None:
        mode = self.audio_mode.get()
        input_path = self.audio_input.get().strip()
        target_language = self.audio_target_language.get().strip() or DEFAULT_TARGET_LANGUAGE
        prompt = self.audio_prompt.get().strip() or None
        system_prompt = self.audio_system.get().strip() or None
        self._submit_job(
            action="audio",
            running_message="Running audio on the shared core...",
            output_widget=self.audio_output,
            work=lambda cancellation_signal, _report_progress: self.controller.run_audio(
                mode=mode,
                input_path=input_path,
                target_language=target_language,
                prompt=prompt,
                system_prompt=system_prompt,
                cancellation_signal=cancellation_signal,
            ),
        )

    def run_thinking(self) -> None:
        mode = self.thinking_mode.get()
        prompt = self.thinking_prompt.get().strip() or None
        follow_up = self.thinking_follow_up.get().strip() or None
        system_prompt = self.thinking_system.get().strip() or None
        include_debug = self.debug_enabled.get()
        self._submit_job(
            action="thinking",
            running_message="Running thinking/tool mode on the shared core...",
            output_widget=self.thinking_output,
            debug_widget=self.debug_output,
            work=lambda cancellation_signal, _report_progress: self.controller.run_thinking(
                mode=mode,
                prompt=prompt,
                follow_up=follow_up,
                system_prompt=system_prompt,
                include_debug=include_debug,
                cancellation_signal=cancellation_signal,
            ),
        )

    def refresh_diagnostics(self) -> None:
        if self.job_runner.has_pending_work():
            self.status_var.set("Diagnostics refresh is disabled while a worker job is running.")
            return
        self._cancel_pending_startup_prewarm()
        diagnostics = self.controller.collect_diagnostics()
        self._set_output(self.diagnostics_output, pretty_json(diagnostics))
        self._resume_startup_prewarm_if_needed()

    def refresh_forensics(self) -> None:
        artifact_override = self.forensics_artifact_path.get().strip() or None
        snapshot = self.controller.collect_forensics(
            surface=self.forensics_surface.get(),
            artifact_path=artifact_override,
            session_id=self._forensics_selected_session_id,
            entry_id=self._forensics_selected_entry_id,
        )
        self.capability_badges_var.set(build_capability_badges_text(snapshot.get("selected_model_id")))
        self._populate_forensics_navigation(snapshot)
        self._populate_artifact_overview(snapshot)
        self._populate_compare_strip(snapshot)
        self._set_output(self.forensics_history_output, build_session_history_report(snapshot))
        self._set_output(
            self.forensics_output,
            build_artifact_browser_report(snapshot, include_raw=self.forensics_show_raw.get()),
        )
        self._populate_quality_checks(snapshot)
        self._populate_lineage(snapshot)
        self._populate_preview_image(snapshot)

    def close(self) -> None:
        self.job_runner.cancel_active()
        self.job_runner.close()
        if not self.job_runner.has_pending_work():
            self.controller.close()
def main() -> int:
    if _TK_IMPORT_ERROR is not None:
        raise UserFacingError(
            "Tkinter is not available in this Python build. Install a Python distribution with Tk support to use the local UI."
        ) from _TK_IMPORT_ERROR
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - depends on local GUI availability
        raise UserFacingError(
            "Failed to start the local Tk UI. Confirm that a local desktop session is available."
        ) from exc

    app = LocalUiApp(root)
    try:
        root.mainloop()
    finally:
        app.close()
        try:
            root.destroy()
        except tk.TclError:
            pass
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserFacingError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
