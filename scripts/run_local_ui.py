#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import queue
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

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
from evaluation_loop import (
    CURATION_EXPORT_DECISIONS,
    CURATION_STATES,
    LEARNING_LIFECYCLE_STATES,
    format_curation_export_preview_report,
    format_evaluation_snapshot_report,
    human_selected_candidates_latest_path,
    jsonl_training_export_dry_run_latest_path,
    learning_candidate_diff_summary_latest_path,
    learning_dataset_preview_latest_path,
    record_curation_export_preview,
    record_evaluation_snapshot,
    record_review_resolution_signal,
)
from gemma_core import CancellationSignal, SessionManager
from gemma_runtime import (
    UserFacingError,
    default_thinking_artifact_path,
    repo_root,
    resolve_audio_model_selection,
    resolve_model_id,
    timestamp_slug,
    write_json,
    WarmupProgress,
    WARMUP_PHASE_ATTACH_MODEL,
    WARMUP_PHASE_LOAD_MODEL,
    WARMUP_PHASE_LOAD_PROCESSOR,
    WARMUP_PHASE_PRIME_TOKEN,
    WARMUP_PHASE_SESSION_READY,
)
from recall_context import DEFAULT_CONTEXT_BUDGET_CHARS, DEFAULT_LIMIT, TASK_KINDS, build_context_bundle
from run_recall_demo import (
    BASELINE_REQUEST_VARIANT,
    MAX_REQUESTS as RECALL_MAX_REQUESTS,
    build_bundle_report,
    compare_evaluation_summaries,
    dataset_bundle_path,
    default_dataset_path,
    evaluate_dataset,
    ensure_recall_dataset,
    format_evaluation_report,
    format_miss_report,
    load_latest_evaluation_summary,
    read_bundle,
    record_evaluation_summary,
    request_catalog,
    select_dataset_request,
    sync_dataset_with_evaluation,
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
    DEFAULT_WORKSPACE_ID,
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
RECALL_MANUAL_LIMIT_MIN = 1
RECALL_MANUAL_LIMIT_MAX = 32
RECALL_MANUAL_CONTEXT_BUDGET_MIN = 1200
RECALL_MANUAL_CONTEXT_BUDGET_MAX = 20000
RECALL_TOP_SELECTED_COMPARE_LIMIT = 3
RECALL_WINNER_CHIP_ORDER = ("winner_only", "shared", "source_only", "pending")
RECALL_WINNER_CHIP_COLORS = {
    "winner_only": {"background": "#E8F5EC", "foreground": "#166534"},
    "shared": {"background": "#EDF2F7", "foreground": "#475569"},
    "source_only": {"background": "#FFF4E5", "foreground": "#B45309"},
    "pending": {"background": "#EEF2F7", "foreground": "#64748B"},
}
LEARNING_CANDIDATE_REVIEW_ARTIFACTS = (
    ("learning_preview", "Learning preview", learning_dataset_preview_latest_path),
    ("human_selected", "Human-selected", human_selected_candidates_latest_path),
    ("jsonl_dry_run", "JSONL dry-run", jsonl_training_export_dry_run_latest_path),
    ("candidate_diff", "Candidate diff", learning_candidate_diff_summary_latest_path),
)
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


def coerce_ui_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        normalized = default
    if normalized < minimum:
        normalized = minimum
    if maximum is not None and normalized > maximum:
        normalized = maximum
    return normalized


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


def apply_recall_compare_fields(
    comparison: Mapping[str, Any] | None,
    *,
    before_var: tk.StringVar | Any,
    after_var: tk.StringVar | Any,
    change_var: tk.StringVar | Any,
) -> None:
    field_map = dict(build_recall_compare_fields(comparison))
    before_var.set(field_map.get("Before", "n/a"))
    after_var.set(field_map.get("After", "n/a"))
    change_var.set(field_map.get("Change", "n/a"))


def apply_recall_eval_compare_fields(
    comparison: Mapping[str, Any] | None,
    *,
    previous_var: tk.StringVar | Any,
    current_var: tk.StringVar | Any,
    change_var: tk.StringVar | Any,
) -> None:
    field_map = dict(build_recall_eval_compare_fields(comparison))
    previous_var.set(field_map.get("Previous", "n/a"))
    current_var.set(field_map.get("Current", "n/a"))
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


def _recall_compare_default_fields() -> list[tuple[str, str]]:
    return [
        ("Before", "Pin a candidate to keep the unpinned bundle here as the baseline."),
        ("After", "Run recall with one or more pins to capture the pinned bundle here."),
        (
            "Change",
            (
                "Source rank, miss reason, selected candidates, budget, and source vs winners "
                "differences appear here."
            ),
        ),
    ]


def _recall_bundle_source_evaluation(bundle: Mapping[str, Any]) -> dict[str, Any]:
    payload = bundle.get("source_evaluation")
    return dict(payload) if isinstance(payload, Mapping) else {}


def _recall_group_members_text(
    row: Mapping[str, Any],
    *,
    count_key: str = "group_member_count",
    labels_key: str = "group_member_labels",
    limit: int = 3,
) -> str:
    group_member_count = coerce_ui_int(
        row.get(count_key),
        default=0,
        minimum=0,
    )
    if group_member_count <= 1:
        return ""
    labels = [
        summarize_text_preview(item, limit=44)
        for item in (row.get(labels_key) or [])
        if isinstance(item, str) and item.strip()
    ][: min(limit, group_member_count)]
    if not labels:
        return str(group_member_count)
    more = group_member_count - len(labels)
    suffix = f"; +{more} more" if more > 0 else ""
    return f"{group_member_count}: {'; '.join(labels)}{suffix}"


def _recall_selected_candidate_descriptors(
    bundle: Mapping[str, Any],
) -> list[tuple[str, str]]:
    descriptors: list[tuple[str, str]] = []
    for index, item in enumerate(bundle.get("selected_candidates") or [], start=1):
        if not isinstance(item, Mapping):
            continue
        event_id = str(item.get("event_id") or f"candidate-{index}")
        label = summarize_text_preview(
            str(item.get("prompt_excerpt") or item.get("event_id") or "n/a"),
            limit=56,
        )
        group_members = _recall_group_members_text(item, limit=2)
        if group_members:
            label = summarize_text_preview(f"{label} (group {group_members})", limit=96)
        descriptors.append((event_id, label))
    return descriptors


def _recall_selected_candidates_text(bundle: Mapping[str, Any], *, limit: int = 3) -> str:
    descriptors = _recall_selected_candidate_descriptors(bundle)
    if not descriptors:
        return "none"
    labels = [label for _, label in descriptors[:limit]]
    more = len(descriptors) - len(labels)
    if more > 0:
        labels.append(f"+{more} more")
    return "; ".join(labels)


def _recall_bundle_source_rank_text(bundle: Mapping[str, Any]) -> str:
    source_evaluation = _recall_bundle_source_evaluation(bundle)
    source_rank = source_evaluation.get("source_rank")
    if source_rank is None:
        return "n/a"
    return str(int(source_rank or 0))


def _recall_bundle_miss_reason_text(bundle: Mapping[str, Any]) -> str:
    source_evaluation = _recall_bundle_source_evaluation(bundle)
    if not source_evaluation:
        return "n/a"
    miss_reason = str(source_evaluation.get("miss_reason") or "").strip()
    if miss_reason:
        return miss_reason
    if source_evaluation.get("source_selected"):
        return "selected"
    return "none"


def _recall_bundle_budget_text(bundle: Mapping[str, Any]) -> str:
    budget = dict(bundle.get("budget") or {})
    context_budget_chars = int(budget.get("context_budget_chars") or 0)
    used_chars = int(budget.get("used_chars") or 0)
    effective_context_budget_chars = int(budget.get("effective_context_budget_chars") or 0)
    summary = f"{used_chars} / {context_budget_chars} chars"
    if effective_context_budget_chars and effective_context_budget_chars != context_budget_chars:
        return f"{summary} (eff {effective_context_budget_chars})"
    return summary


def _recall_winner_labels(
    source_evaluation: Mapping[str, Any],
    *,
    limit: int = 2,
) -> list[str]:
    winners: list[str] = []
    for item in (source_evaluation.get("top_selected") or [])[:limit]:
        if not isinstance(item, Mapping):
            continue
        winners.append(
            summarize_text_preview(
                str(item.get("prompt_excerpt") or item.get("event_id") or "n/a"),
                limit=52,
            )
        )
    return winners


def _recall_top_selected_rows(row: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(row, Mapping):
        return []
    return [
        dict(item)
        for item in (row.get("top_selected") or [])
        if isinstance(item, Mapping)
    ]


def _recall_primary_winner_row(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    winners = _recall_top_selected_rows(row)
    return winners[0] if winners else None


def recall_row_has_forensics_navigation(row: Mapping[str, Any] | None) -> bool:
    if not isinstance(row, Mapping):
        return False
    artifact_path = str(row.get("artifact_path") or "").strip()
    session_id = str(row.get("session_id") or "").strip()
    event_id = str(row.get("event_id") or "").strip()
    return bool(artifact_path or (session_id and event_id))


def _parse_workspace_event_id(event_id: str | None) -> tuple[str, str, str] | None:
    normalized = str(event_id or "").strip()
    if not normalized:
        return None
    parts = [part.strip() for part in normalized.split(":", 2)]
    if len(parts) != 3 or not all(parts):
        return None
    return parts[0], parts[1], parts[2]


def _infer_surface_from_session_id(session_id: str | None) -> str | None:
    normalized = str(session_id or "").strip().lower()
    if not normalized:
        return None
    prefix = normalized.split("-", 1)[0]
    return prefix if prefix in SESSION_SURFACES else None


def resolve_recall_forensics_row(
    row: Mapping[str, Any] | None,
    *,
    root: Path,
    current_workspace_id: str,
) -> dict[str, Any] | None:
    if not isinstance(row, Mapping):
        return None

    event_id = str(row.get("event_id") or "").strip()
    session_id = str(row.get("session_id") or "").strip()
    session_surface = str(row.get("session_surface") or "").strip().lower()
    artifact_path = str(row.get("artifact_path") or "").strip()
    workspace_id = str(row.get("workspace_id") or "").strip()

    parsed_event = _parse_workspace_event_id(event_id)
    if parsed_event is not None:
        parsed_workspace_id, parsed_session_id, parsed_entry_id = parsed_event
        workspace_id = workspace_id or parsed_workspace_id
        session_id = session_id or parsed_session_id
        event_id = parsed_entry_id

    if session_id and (not artifact_path or session_surface not in SESSION_SURFACES):
        manifest_workspace_id = workspace_id or current_workspace_id or DEFAULT_WORKSPACE_ID
        manifest_path = session_manifest_path(
            session_id=session_id,
            workspace_id=manifest_workspace_id,
            root=root,
        )
        if manifest_path.exists():
            try:
                session_payload = read_session_manifest(manifest_path)
            except Exception:
                session_payload = None
            if isinstance(session_payload, Mapping):
                if session_surface not in SESSION_SURFACES:
                    session_surface = str(session_payload.get("surface") or "").strip().lower()
                if not artifact_path and event_id:
                    selected_entry = next(
                        (
                            item
                            for item in (session_payload.get("entries") or [])
                            if isinstance(item, Mapping) and str(item.get("entry_id") or "").strip() == event_id
                        ),
                        None,
                    )
                    if isinstance(selected_entry, Mapping):
                        artifact_ref = dict(selected_entry.get("artifact_ref") or {})
                        artifact_path = str(artifact_ref.get("artifact_path") or "").strip()

    if session_surface not in SESSION_SURFACES:
        inferred_surface = _infer_surface_from_session_id(session_id)
        if inferred_surface is not None:
            session_surface = inferred_surface

    if not artifact_path and not (session_id and event_id):
        return None

    return {
        **dict(row),
        "workspace_id": workspace_id or current_workspace_id,
        "event_id": event_id,
        "session_id": session_id,
        "session_surface": session_surface if session_surface in SESSION_SURFACES else None,
        "artifact_path": artifact_path,
    }


def build_recall_eval_source_navigation_row(
    row: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(row, Mapping):
        return None

    source_event_id = str(row.get("source_event_id") or "").strip()
    source_entry_id = str(row.get("source_entry_id") or "").strip()
    source_session_id = str(row.get("source_session_id") or "").strip()
    source_session_surface = str(row.get("source_session_surface") or "").strip().lower()
    source_artifact_path = str(row.get("source_artifact_path") or "").strip()
    source_workspace_id = str(row.get("source_workspace_id") or "").strip()
    if not any((source_event_id, source_entry_id, source_session_id, source_artifact_path)):
        return None

    return {
        "workspace_id": source_workspace_id,
        "event_id": source_entry_id or source_event_id,
        "session_id": source_session_id,
        "session_surface": source_session_surface,
        "artifact_path": source_artifact_path,
        "prompt_excerpt": str(
            row.get("source_prompt_excerpt") or row.get("query_text") or source_event_id or "source"
        ).strip(),
    }


def _recall_bundle_source_vs_winners_text(bundle: Mapping[str, Any]) -> str:
    source_evaluation = _recall_bundle_source_evaluation(bundle)
    if not source_evaluation:
        return "n/a"

    source_label = summarize_text_preview(
        str(
            source_evaluation.get("source_prompt_excerpt")
            or source_evaluation.get("source_event_id")
            or "source"
        ),
        limit=52,
    )
    winner_labels = _recall_winner_labels(source_evaluation)
    winners_text = "; ".join(winner_labels) if winner_labels else "none"
    if source_evaluation.get("source_selected"):
        if source_evaluation.get("source_selected_via_group"):
            group_members = _recall_group_members_text(
                source_evaluation,
                count_key="source_group_member_count",
                labels_key="source_group_member_labels",
                limit=2,
            )
            grouped_by = str(source_evaluation.get("source_grouped_by") or "selected group")
            group_text = group_members or str(
                coerce_ui_int(
                    source_evaluation.get("source_group_member_count"),
                    default=1,
                    minimum=1,
                )
            )
            return f"{source_label} joined {grouped_by} ({group_text}) with {winners_text}"
        return f"{source_label} made the bundle with {winners_text}"
    if winner_labels:
        return f"{source_label} lost to {winners_text}"
    return f"{source_label} missed without recorded winners"


def _recall_bundle_compare_text(
    bundle: Mapping[str, Any] | None,
    *,
    empty: str,
) -> str:
    if bundle is None:
        return empty
    return "\n".join(
        [
            f"source rank={_recall_bundle_source_rank_text(bundle)}",
            f"miss reason={_recall_bundle_miss_reason_text(bundle)}",
            f"selected candidates={_recall_selected_candidates_text(bundle)}",
            f"budget={_recall_bundle_budget_text(bundle)}",
            f"source vs winners={_recall_bundle_source_vs_winners_text(bundle)}",
        ]
    )


def _recall_change_line(label: str, before_value: str, after_value: str) -> str:
    if before_value == after_value:
        return f"{label}: unchanged ({before_value})"
    return f"{label}: {before_value} -> {after_value}"


def _recall_selected_candidates_change_text(
    before_bundle: Mapping[str, Any],
    after_bundle: Mapping[str, Any],
) -> str:
    before_descriptors = _recall_selected_candidate_descriptors(before_bundle)
    after_descriptors = _recall_selected_candidate_descriptors(after_bundle)
    before_event_ids = [event_id for event_id, _label in before_descriptors]
    after_event_ids = [event_id for event_id, _label in after_descriptors]
    if before_event_ids == after_event_ids:
        return f"selected candidates: unchanged ({_recall_selected_candidates_text(before_bundle)})"

    before_labels = {event_id: label for event_id, label in before_descriptors}
    after_labels = {event_id: label for event_id, label in after_descriptors}
    added = [after_labels[event_id] for event_id in after_event_ids if event_id not in before_labels]
    removed = [before_labels[event_id] for event_id in before_event_ids if event_id not in after_labels]
    parts: list[str] = []
    if added:
        parts.append(f"+ {'; '.join(added[:2])}")
    if removed:
        parts.append(f"- {'; '.join(removed[:2])}")
    if parts:
        return "selected candidates: " + " ".join(parts)
    return (
        "selected candidates: "
        f"{_recall_selected_candidates_text(before_bundle)} -> {_recall_selected_candidates_text(after_bundle)}"
    )


def _recall_budget_change_text(
    before_bundle: Mapping[str, Any],
    after_bundle: Mapping[str, Any],
) -> str:
    before_budget = dict(before_bundle.get("budget") or {})
    after_budget = dict(after_bundle.get("budget") or {})
    before_text = _recall_bundle_budget_text(before_bundle)
    after_text = _recall_bundle_budget_text(after_bundle)
    if before_text == after_text:
        return f"budget: unchanged ({before_text})"
    delta_used_chars = int(after_budget.get("used_chars") or 0) - int(before_budget.get("used_chars") or 0)
    delta_suffix = f" ({delta_used_chars:+d} used)" if delta_used_chars else ""
    return f"budget: {before_text} -> {after_text}{delta_suffix}"


def build_recall_compare_fields(
    comparison: Mapping[str, Any] | None,
) -> list[tuple[str, str]]:
    if not isinstance(comparison, Mapping):
        return _recall_compare_default_fields()
    before_bundle = comparison.get("before_bundle")
    after_bundle = comparison.get("after_bundle")
    if not isinstance(before_bundle, Mapping) or not isinstance(after_bundle, Mapping):
        return _recall_compare_default_fields()

    before_text = _recall_bundle_compare_text(
        before_bundle,
        empty="Unpinned bundle not available.",
    )
    after_text = _recall_bundle_compare_text(
        after_bundle,
        empty="Pinned bundle not available.",
    )
    change_text = "\n".join(
        [
            _recall_change_line(
                "source rank",
                _recall_bundle_source_rank_text(before_bundle),
                _recall_bundle_source_rank_text(after_bundle),
            ),
            _recall_change_line(
                "miss reason",
                _recall_bundle_miss_reason_text(before_bundle),
                _recall_bundle_miss_reason_text(after_bundle),
            ),
            _recall_selected_candidates_change_text(before_bundle, after_bundle),
            _recall_budget_change_text(before_bundle, after_bundle),
            _recall_change_line(
                "source vs winners",
                _recall_bundle_source_vs_winners_text(before_bundle),
                _recall_bundle_source_vs_winners_text(after_bundle),
            ),
        ]
    )
    return [
        ("Before", before_text),
        ("After", after_text),
        ("Change", change_text),
    ]


def _recall_eval_compare_default_fields() -> list[tuple[str, str]]:
    return [
        ("Previous", "Run evaluation once to capture the previous hit-quality snapshot."),
        ("Current", "Run evaluation to score source-hit recovery across the prepared requests."),
        ("Change", "Hit, miss, hit-rate, variant, and miss-reason changes appear here."),
    ]


def _recall_eval_variant_summary_text(summary: Mapping[str, Any], *, limit: int = 2) -> str:
    variants = dict(summary.get("variants") or {})
    if not variants:
        return "n/a"
    labels: list[str] = []
    for variant, payload in list(sorted(variants.items()))[:limit]:
        if not isinstance(payload, Mapping):
            continue
        labels.append(
            f"{variant} {int(payload.get('source_hits') or 0)}/{int(payload.get('request_count') or 0)} "
            f"({float(payload.get('hit_rate') or 0.0):.3f})"
        )
    if not labels:
        return "n/a"
    more = len(variants) - len(labels)
    if more > 0:
        labels.append(f"+{more} more")
    return "; ".join(labels)


def _recall_eval_miss_reason_summary_text(summary: Mapping[str, Any], *, limit: int = 2) -> str:
    miss_reason_counts = dict(summary.get("miss_reason_counts") or {})
    if not miss_reason_counts:
        return "none"
    labels = [f"{miss_reason} {count}" for miss_reason, count in list(sorted(miss_reason_counts.items()))[:limit]]
    more = len(miss_reason_counts) - len(labels)
    if more > 0:
        labels.append(f"+{more} more")
    return "; ".join(labels)


def _recall_eval_summary_text(
    summary: Mapping[str, Any] | None,
    *,
    empty: str,
) -> str:
    if not isinstance(summary, Mapping):
        return empty
    return "\n".join(
        [
            f"requests={int(summary.get('request_count') or 0)}",
            f"source hits={int(summary.get('source_hits') or 0)}",
            f"source misses={int(summary.get('source_misses') or 0)}",
            f"hit rate={float(summary.get('hit_rate') or 0.0):.3f}",
            f"variants={_recall_eval_variant_summary_text(summary)}",
            f"miss reasons={_recall_eval_miss_reason_summary_text(summary)}",
        ]
    )


def _recall_eval_change_line(label: str, before_value: int | float, after_value: int | float, *, digits: int = 0) -> str:
    if digits:
        before_text = f"{float(before_value):.{digits}f}"
        after_text = f"{float(after_value):.{digits}f}"
        delta = float(after_value) - float(before_value)
        delta_text = f"{delta:+.{digits}f}"
    else:
        before_text = str(int(before_value))
        after_text = str(int(after_value))
        delta_text = f"{int(after_value) - int(before_value):+d}"
    if before_text == after_text:
        return f"{label}: unchanged ({before_text})"
    return f"{label}: {before_text} -> {after_text} ({delta_text})"


def _recall_eval_variant_change_text(
    before_summary: Mapping[str, Any],
    after_summary: Mapping[str, Any],
) -> str:
    delta = dict(compare_evaluation_summaries(before_summary, after_summary).get("variants") or {})
    if not delta:
        return f"variants: unchanged ({_recall_eval_variant_summary_text(after_summary)})"
    parts: list[str] = []
    for variant, payload in list(sorted(delta.items()))[:2]:
        if not isinstance(payload, Mapping):
            continue
        parts.append(
            f"{variant} hits {int(payload.get('source_hits_delta') or 0):+d}, "
            f"rate {float(payload.get('hit_rate_delta') or 0.0):+.3f}"
        )
    return "variants: " + "; ".join(parts) if parts else "variants: unchanged"


def _recall_eval_miss_reason_change_text(
    before_summary: Mapping[str, Any],
    after_summary: Mapping[str, Any],
) -> str:
    delta = dict(compare_evaluation_summaries(before_summary, after_summary).get("miss_reason_counts") or {})
    if not delta:
        return f"miss reasons: unchanged ({_recall_eval_miss_reason_summary_text(after_summary)})"
    labels = [f"{miss_reason} {count:+d}" for miss_reason, count in list(sorted(delta.items()))[:3]]
    return "miss reasons: " + "; ".join(labels)


def build_recall_eval_compare_fields(
    comparison: Mapping[str, Any] | None,
) -> list[tuple[str, str]]:
    if not isinstance(comparison, Mapping):
        return _recall_eval_compare_default_fields()
    before_summary = comparison.get("before_summary")
    after_summary = comparison.get("after_summary")
    if not isinstance(after_summary, Mapping):
        return _recall_eval_compare_default_fields()

    previous_text = _recall_eval_summary_text(
        before_summary if isinstance(before_summary, Mapping) else None,
        empty="No earlier evaluation snapshot is available yet.",
    )
    current_text = _recall_eval_summary_text(
        after_summary,
        empty="Evaluation summary unavailable.",
    )
    if not isinstance(before_summary, Mapping):
        return [
            ("Previous", previous_text),
            ("Current", current_text),
            ("Change", "Run evaluation again after a ranking tweak to see the delta here."),
        ]

    change_text = "\n".join(
        [
            _recall_eval_change_line(
                "source hits",
                int(before_summary.get("source_hits") or 0),
                int(after_summary.get("source_hits") or 0),
            ),
            _recall_eval_change_line(
                "source misses",
                int(before_summary.get("source_misses") or 0),
                int(after_summary.get("source_misses") or 0),
            ),
            _recall_eval_change_line(
                "hit rate",
                float(before_summary.get("hit_rate") or 0.0),
                float(after_summary.get("hit_rate") or 0.0),
                digits=3,
            ),
            _recall_eval_variant_change_text(before_summary, after_summary),
            _recall_eval_miss_reason_change_text(before_summary, after_summary),
        ]
    )
    return [
        ("Previous", previous_text),
        ("Current", current_text),
        ("Change", change_text),
    ]


def build_recall_eval_state(
    summary: Mapping[str, Any] | None,
    *,
    dataset_path: str | None,
    root: Path,
    empty: str = "Run evaluation to score source-hit recovery across the prepared requests.",
) -> str:
    if not isinstance(summary, Mapping):
        return empty
    dataset_display = summarize_workspace_path(dataset_path, root=root) if dataset_path else "n/a"
    evaluated_at_utc = str(summary.get("evaluated_at_utc") or "").strip()
    state = (
        f"{int(summary.get('request_count') or 0)} requests from {dataset_display}. "
        f"hit rate {float(summary.get('hit_rate') or 0.0):.3f}, "
        f"misses {int(summary.get('source_misses') or 0)}."
    )
    if evaluated_at_utc:
        state += f" Evaluated {evaluated_at_utc}."
    return state


def build_recall_eval_miss_summary(
    row: Mapping[str, Any] | None,
    *,
    empty: str = "Select an evaluation miss to sync it with the prepared request list.",
) -> str:
    if row is None:
        return empty
    header = (
        f"request {row.get('index') or 'n/a'}: {row.get('task_kind') or 'n/a'}  "
        f"reason={row.get('miss_reason') or 'unknown'}  "
        f"rank={row.get('source_rank') if row.get('source_rank') is not None else 'n/a'}"
    )
    request_variant = str(row.get("request_variant") or "").strip()
    if request_variant and request_variant != BASELINE_REQUEST_VARIANT:
        header += f"  variant={request_variant}"
    lines = [
        header,
        f"query={summarize_text_preview(str(row.get('query_text') or ''), limit=132)}",
    ]
    source_prompt_excerpt = str(row.get("source_prompt_excerpt") or "").strip()
    if source_prompt_excerpt:
        lines.append(f"source={summarize_text_preview(source_prompt_excerpt, limit=132)}")
    winner_labels = _recall_eval_winner_labels(row)
    if winner_labels != "n/a":
        lines.append(f"winners={winner_labels}")
    return "\n".join(lines)


def _recall_first_file_hint(row: Mapping[str, Any] | None) -> str:
    if not isinstance(row, Mapping):
        return ""
    file_hints = [str(item).strip() for item in (row.get("file_hints") or []) if str(item).strip()]
    return file_hints[0] if file_hints else ""


def _recall_eval_winner_labels(row: Mapping[str, Any], *, limit: int = RECALL_TOP_SELECTED_COMPARE_LIMIT) -> str:
    winners = [
        item
        for item in (row.get("top_selected") or [])
        if isinstance(item, Mapping)
    ]
    if not winners:
        return "n/a"
    labels = [
        f"#{index} "
        + summarize_text_preview(
            str(item.get("prompt_excerpt") or item.get("event_id") or "n/a"),
            limit=52,
        )
        for index, item in enumerate(winners[:limit], start=1)
    ]
    more = len(winners) - len(labels)
    if more > 0:
        labels.append(f"+{more} more")
    return "; ".join(labels)


def _recall_reason_display(reason: str) -> str:
    return str(reason or "").strip().replace("_", " ").replace("-", " ")


def _recall_reason_list(value: Any, *, limit: int = 4) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for item in value or []:
        label = _recall_reason_display(str(item))
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
        if len(labels) >= limit:
            break
    return labels


@dataclass(frozen=True)
class RecallReasonCompare:
    primary_reasons: tuple[str, ...] = ()
    secondary_reasons: tuple[str, ...] = ()
    primary_only: tuple[str, ...] = ()
    shared: tuple[str, ...] = ()
    secondary_only: tuple[str, ...] = ()


def _recall_reason_compare(
    primary_reasons: tuple[str, ...],
    secondary_reasons: tuple[str, ...],
) -> RecallReasonCompare:
    primary_lookup = set(primary_reasons)
    secondary_lookup = set(secondary_reasons)
    return RecallReasonCompare(
        primary_reasons=primary_reasons,
        secondary_reasons=secondary_reasons,
        primary_only=tuple(reason for reason in primary_reasons if reason not in secondary_lookup),
        shared=tuple(reason for reason in primary_reasons if reason in secondary_lookup),
        secondary_only=tuple(reason for reason in secondary_reasons if reason not in primary_lookup),
    )


@dataclass(frozen=True)
class RecallWinnerReasonDelta:
    winner_reasons: tuple[str, ...] = ()
    source_reasons: tuple[str, ...] = ()
    winner_only: tuple[str, ...] = ()
    shared: tuple[str, ...] = ()
    source_only: tuple[str, ...] = ()


def _recall_winner_reason_delta(
    miss_row: Mapping[str, Any] | None,
    winner_row: Mapping[str, Any] | None,
) -> RecallWinnerReasonDelta:
    if not isinstance(winner_row, Mapping):
        return RecallWinnerReasonDelta()

    winner_reasons = tuple(_recall_reason_list(winner_row.get("reasons"), limit=6))
    source_reasons = tuple(_recall_reason_list((miss_row or {}).get("source_reasons"), limit=6))
    compare = _recall_reason_compare(winner_reasons, source_reasons)
    return RecallWinnerReasonDelta(
        winner_reasons=compare.primary_reasons,
        source_reasons=compare.secondary_reasons,
        winner_only=compare.primary_only,
        shared=compare.shared,
        source_only=compare.secondary_only,
    )


def _recall_top_selected_reason_list(
    row: Mapping[str, Any] | None,
    *,
    winner_limit: int = RECALL_TOP_SELECTED_COMPARE_LIMIT,
    reason_limit: int = 6,
) -> tuple[str, ...]:
    labels: list[str] = []
    seen: set[str] = set()
    for winner_row in _recall_top_selected_rows(row)[:winner_limit]:
        for label in _recall_reason_list(winner_row.get("reasons"), limit=reason_limit):
            if label in seen:
                continue
            seen.add(label)
            labels.append(label)
            if len(labels) >= reason_limit:
                return tuple(labels)
    return tuple(labels)


def _recall_source_reason_compare(
    miss_row: Mapping[str, Any] | None,
    *,
    limit: int = RECALL_TOP_SELECTED_COMPARE_LIMIT,
) -> RecallReasonCompare:
    source_reasons = tuple(_recall_reason_list((miss_row or {}).get("source_reasons"), limit=6))
    winner_reasons = _recall_top_selected_reason_list(miss_row, winner_limit=limit, reason_limit=6)
    return _recall_reason_compare(source_reasons, winner_reasons)


def _format_recall_reason_chip(prefix: str, reasons: tuple[str, ...], *, limit: int = 2) -> str:
    if not reasons:
        return ""
    return f"{prefix} {summarize_text_preview(', '.join(reasons[:limit]), limit=36)}"


def build_recall_eval_winner_chip_texts(
    miss_row: Mapping[str, Any] | None,
    winner_row: Mapping[str, Any] | None,
) -> dict[str, str]:
    chip_texts = {key: "" for key in RECALL_WINNER_CHIP_ORDER}
    delta = _recall_winner_reason_delta(miss_row, winner_row)
    if not delta.winner_reasons:
        if isinstance(winner_row, Mapping):
            chip_texts["pending"] = "Reasons pending"
        return chip_texts

    chip_texts["winner_only"] = _format_recall_reason_chip("+", delta.winner_only)
    chip_texts["shared"] = _format_recall_reason_chip("=", delta.shared)
    chip_texts["source_only"] = _format_recall_reason_chip("-", delta.source_only)
    return chip_texts


def build_recall_eval_source_chip_texts(
    miss_row: Mapping[str, Any] | None,
    *,
    limit: int = RECALL_TOP_SELECTED_COMPARE_LIMIT,
) -> dict[str, str]:
    chip_texts = {key: "" for key in RECALL_WINNER_CHIP_ORDER}
    if not isinstance(miss_row, Mapping):
        return chip_texts

    compare = _recall_source_reason_compare(miss_row, limit=limit)
    if not compare.primary_reasons:
        chip_texts["pending"] = "Reasons pending"
        return chip_texts

    chip_texts["winner_only"] = _format_recall_reason_chip("+", compare.primary_only)
    chip_texts["shared"] = _format_recall_reason_chip("=", compare.shared)
    chip_texts["source_only"] = _format_recall_reason_chip("-", compare.secondary_only)
    return chip_texts


def build_recall_eval_winner_why_text(
    miss_row: Mapping[str, Any] | None,
    winner_row: Mapping[str, Any] | None,
) -> str:
    if not isinstance(winner_row, Mapping):
        return "Why it won appears after winner details are recorded."

    delta = _recall_winner_reason_delta(miss_row, winner_row)
    if not delta.winner_reasons:
        return "Winner reasons pending in this snapshot."

    if delta.winner_only:
        return f"beat source on {summarize_text_preview(', '.join(delta.winner_only[:2]), limit=86)}"
    if delta.shared:
        return f"matched source on {summarize_text_preview(', '.join(delta.shared[:2]), limit=86)}"

    block_title = str(winner_row.get("block_title") or "").strip()
    if block_title:
        return f"winner landed in {block_title.lower()}."
    return "Winner reasons pending in this snapshot."


def build_recall_eval_source_why_text(
    miss_row: Mapping[str, Any] | None,
    *,
    limit: int = RECALL_TOP_SELECTED_COMPARE_LIMIT,
) -> str:
    if not isinstance(miss_row, Mapping):
        return "Source compare appears after you select a miss."

    compare = _recall_source_reason_compare(miss_row, limit=limit)
    if not compare.primary_reasons:
        return "Source reasons pending in this snapshot."

    winners = _recall_top_selected_rows(miss_row)[:limit]
    if winners and not compare.secondary_reasons:
        return "Source signals are ready while winner reasons are pending."
    if not winners:
        return "Source signals are ready before winners are recorded."
    if compare.secondary_only:
        return f"lost to winners on {summarize_text_preview(', '.join(compare.secondary_only[:2]), limit=86)}"
    if compare.primary_only:
        return f"kept source-only signals on {summarize_text_preview(', '.join(compare.primary_only[:2]), limit=86)}"
    if compare.shared:
        return f"matched winners on {summarize_text_preview(', '.join(compare.shared[:2]), limit=86)}"
    return "Source reasons pending in this snapshot."


def build_recall_eval_source_card_text(
    row: Mapping[str, Any] | None,
    *,
    empty: str = "Source compare appears here when you select an evaluation miss.",
) -> str:
    if not isinstance(row, Mapping):
        return empty
    return "\n".join(
        [
            build_recall_eval_source_summary_text(row, empty=empty),
            build_recall_eval_source_why_text(row),
        ]
    )


def build_recall_eval_source_summary_text(
    row: Mapping[str, Any] | None,
    *,
    empty: str = "Source compare appears here when you select an evaluation miss.",
) -> str:
    if not isinstance(row, Mapping):
        return empty

    prompt_excerpt = summarize_text_preview(
        str(row.get("source_prompt_excerpt") or row.get("query_text") or row.get("source_event_id") or "source"),
        limit=88,
    )
    metadata: list[str] = []
    source_block_title = str(row.get("source_block_title") or "").strip()
    if source_block_title:
        metadata.append(f"block={source_block_title}")
    source_rank = row.get("source_rank")
    if source_rank is not None:
        metadata.append(f"rank={source_rank}")
    miss_reason = str(row.get("miss_reason") or "").strip()
    if miss_reason:
        metadata.append(f"reason={_recall_reason_display(miss_reason)}")

    lines = [prompt_excerpt]
    if metadata:
        lines.append("  ".join(metadata[:3]))
    return "\n".join(lines)


def build_recall_eval_source_rerun_text(
    miss_row: Mapping[str, Any] | None,
    *,
    request_row: Mapping[str, Any] | None,
) -> str:
    if not isinstance(miss_row, Mapping) or not isinstance(request_row, Mapping):
        return ""

    suggestion = build_recall_miss_suggested_manual_config(
        miss_row,
        request_row=request_row,
    )
    if not suggestion.get("apply_ready"):
        return "Rerun waits until the source is back in the index."

    change_labels: list[str] = []
    base_basis = str(request_row.get("request_basis") or "").strip() or "n/a"
    suggested_basis = str(suggestion.get("request_basis") or "").strip() or "n/a"
    if suggested_basis != base_basis:
        change_labels.append(f"basis {base_basis} -> {suggested_basis}")

    base_limit = coerce_ui_int(
        request_row.get("limit"),
        default=DEFAULT_LIMIT,
        minimum=RECALL_MANUAL_LIMIT_MIN,
        maximum=RECALL_MANUAL_LIMIT_MAX,
    )
    suggested_limit = coerce_ui_int(
        suggestion.get("limit"),
        default=DEFAULT_LIMIT,
        minimum=RECALL_MANUAL_LIMIT_MIN,
        maximum=RECALL_MANUAL_LIMIT_MAX,
    )
    if suggested_limit != base_limit:
        change_labels.append(f"limit {base_limit} -> {suggested_limit}")

    base_context_budget_chars = coerce_ui_int(
        request_row.get("context_budget_chars"),
        default=DEFAULT_CONTEXT_BUDGET_CHARS,
        minimum=RECALL_MANUAL_CONTEXT_BUDGET_MIN,
        maximum=RECALL_MANUAL_CONTEXT_BUDGET_MAX,
    )
    suggested_context_budget_chars = coerce_ui_int(
        suggestion.get("context_budget_chars"),
        default=DEFAULT_CONTEXT_BUDGET_CHARS,
        minimum=RECALL_MANUAL_CONTEXT_BUDGET_MIN,
        maximum=RECALL_MANUAL_CONTEXT_BUDGET_MAX,
    )
    if suggested_context_budget_chars != base_context_budget_chars:
        change_labels.append(f"budget {base_context_budget_chars} -> {suggested_context_budget_chars}")

    if not change_labels:
        return "Rerun is ready with the suggested manual settings."
    return "Rerun suggested tweak: " + "; ".join(change_labels[:2]) + "."


def build_recall_eval_source_rerun_result_text(
    miss_row: Mapping[str, Any] | None,
    *,
    bundle: Mapping[str, Any] | None,
) -> str:
    if not isinstance(miss_row, Mapping) or not isinstance(bundle, Mapping):
        return ""

    source_evaluation = _recall_bundle_source_evaluation(bundle)
    if not source_evaluation:
        return ""

    before_reason = _recall_reason_display(str(miss_row.get("miss_reason") or "").strip())
    before_rank_raw = miss_row.get("source_rank")
    try:
        before_rank_text = str(int(before_rank_raw)) if before_rank_raw is not None else ""
    except (TypeError, ValueError):
        before_rank_text = ""

    if before_rank_text and before_reason:
        before_text = f"source was rank {before_rank_text} and {before_reason}"
    elif before_rank_text:
        before_text = f"source was rank {before_rank_text}"
    elif before_reason:
        before_text = f"source was {before_reason}"
    else:
        before_text = "source was still under review"

    after_rank_text = _recall_bundle_source_rank_text(bundle)
    after_reason_raw = _recall_bundle_miss_reason_text(bundle)
    after_reason = _recall_reason_display(after_reason_raw)
    after_selected = bool(source_evaluation.get("source_selected"))

    if after_selected:
        if after_rank_text != "n/a":
            after_text = f"source made the bundle at rank {after_rank_text}"
        else:
            after_text = "source made the bundle"
    elif after_reason == "source missing from index":
        after_text = "source is still missing from the index"
    else:
        same_rank = bool(before_rank_text and after_rank_text == before_rank_text)
        same_reason = bool(before_reason and after_reason == before_reason)
        if after_rank_text != "n/a" and after_reason and after_reason not in {"none", "selected"}:
            rank_phrase = f"{'still ' if same_rank else ''}rank {after_rank_text}"
            reason_phrase = f"{'still ' if same_reason and not same_rank else ''}{after_reason}"
            after_text = f"source is {rank_phrase} and {reason_phrase}"
        elif after_rank_text != "n/a":
            after_text = f"source is {'still ' if same_rank else 'now '}rank {after_rank_text}"
        elif after_reason and after_reason not in {"none", "selected"}:
            after_text = f"source is {'still ' if same_reason else ''}{after_reason}"
        else:
            after_text = "source state is not recorded in this rerun"

    return f"Before: {before_text}. After rerun: {after_text}."


def build_recall_eval_source_action_hint(
    miss_row: Mapping[str, Any] | None,
    *,
    request_row: Mapping[str, Any] | None,
    source_row: Mapping[str, Any] | None,
    empty: str = "Select a miss to open the source, copy it into manual recall, or rerun the suggested tweak.",
) -> str:
    if not isinstance(miss_row, Mapping):
        return empty

    messages: list[str] = []
    if source_row is None:
        messages.append("Open needs a fresh source snapshot.")
    if not isinstance(request_row, Mapping):
        messages.append("Copy and rerun need the matching prepared request. Refresh real data first.")
        return " ".join(messages) if messages else empty

    rerun_text = build_recall_eval_source_rerun_text(
        miss_row,
        request_row=request_row,
    )
    if rerun_text:
        messages.append(rerun_text)
    return " ".join(messages) if messages else empty


def build_recall_eval_winner_compare_state(
    row: Mapping[str, Any] | None,
    *,
    limit: int = RECALL_TOP_SELECTED_COMPARE_LIMIT,
) -> str:
    winners = _recall_top_selected_rows(row)
    if not winners:
        return "Top winners appear here when the selected miss records winning candidates."
    shown = min(len(winners), limit)
    message = f"Showing {shown} winner slot(s) for side-by-side compare."
    if len(winners) > shown:
        message += f" {len(winners) - shown} more recorded."
    message += " Open any card in History / Forensics."
    return message


def build_recall_eval_winner_card_text(
    miss_row: Mapping[str, Any] | None,
    row: Mapping[str, Any] | None,
    *,
    rank: int,
    root: Path,
    empty: str | None = None,
) -> str:
    if row is None:
        return empty or f"No winner recorded at #{rank}."

    prompt_excerpt = summarize_text_preview(
        str(row.get("prompt_excerpt") or row.get("event_id") or "n/a"),
        limit=88,
    )
    metadata: list[str] = []
    block_title = str(row.get("block_title") or "").strip()
    if block_title:
        metadata.append(f"block={block_title}")
    session_surface = str(row.get("session_surface") or "").strip()
    if session_surface:
        metadata.append(f"surface={session_surface}")
    score = row.get("score")
    if score is not None:
        metadata.append(f"score={score}")
    group_member_count = coerce_ui_int(
        row.get("group_member_count"),
        default=0,
        minimum=0,
    )
    if group_member_count > 1:
        metadata.append(f"group={group_member_count}")

    lines = [prompt_excerpt]
    if metadata:
        lines.append("  ".join(metadata[:3]))
    lines.append(build_recall_eval_winner_why_text(miss_row, row))
    artifact_path = str(row.get("artifact_path") or "").strip()
    if artifact_path:
        lines.append(f"artifact={summarize_workspace_path(artifact_path, root=root)}")
    return "\n".join(lines)


def build_recall_miss_suggested_manual_config(
    miss_row: Mapping[str, Any] | None,
    *,
    request_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected_row = request_row if isinstance(request_row, Mapping) else {}
    miss_payload = miss_row if isinstance(miss_row, Mapping) else {}

    task_kind = str(selected_row.get("task_kind") or miss_payload.get("task_kind") or "proposal").strip() or "proposal"
    query_text = str(selected_row.get("query_text") or miss_payload.get("query_text") or "").strip()
    request_basis = str(selected_row.get("request_basis") or "").strip()
    file_hint = _recall_first_file_hint(selected_row)
    base_limit = coerce_ui_int(
        selected_row.get("limit"),
        default=DEFAULT_LIMIT,
        minimum=RECALL_MANUAL_LIMIT_MIN,
        maximum=RECALL_MANUAL_LIMIT_MAX,
    )
    base_context_budget_chars = coerce_ui_int(
        selected_row.get("context_budget_chars"),
        default=DEFAULT_CONTEXT_BUDGET_CHARS,
        minimum=RECALL_MANUAL_CONTEXT_BUDGET_MIN,
        maximum=RECALL_MANUAL_CONTEXT_BUDGET_MAX,
    )
    source_rank_raw = miss_payload.get("source_rank")
    try:
        source_rank = int(source_rank_raw) if source_rank_raw is not None else None
    except (TypeError, ValueError):
        source_rank = None
    miss_reason = str(miss_payload.get("miss_reason") or "").strip() or "unknown"
    request_variant = str(selected_row.get("request_variant") or miss_payload.get("request_variant") or "").strip()
    source_reasons = [
        str(item).strip()
        for item in (miss_payload.get("source_reasons") or selected_row.get("source_reasons") or [])
        if str(item).strip()
    ]
    winner_labels = _recall_eval_winner_labels(miss_payload)

    limit = base_limit
    context_budget_chars = base_context_budget_chars
    apply_ready = True
    diagnosis_title = "Ranking miss"
    diagnosis = "The source came back, but stronger-looking candidates still pushed it out."
    actions: list[str] = [
        "Add or tighten the file hint if this request points at one file.",
        "Raise the limit a little and compare the new winners.",
        "Pin the source once to confirm whether the current winners really deserve the slot.",
    ]

    if miss_reason == "ranked_out_by_limit":
        diagnosis_title = "Top-N cutoff"
        diagnosis = "The source was retrieved, but it sat below the current selection limit."
        limit = min(RECALL_MANUAL_LIMIT_MAX, max(base_limit + 4, (source_rank or base_limit) + 1))
        actions = [
            f"Raise limit to {limit} so rank {source_rank or 'n/a'} can enter the bundle.",
            "Keep the query stable and check whether the new candidates are genuinely useful.",
            "Add a file hint if one path should dominate this recall.",
        ]
    elif miss_reason == "dropped_by_context_budget":
        diagnosis_title = "Context budget squeeze"
        diagnosis = "The source ranked well enough, but the assembled context ran out of room before it fit."
        context_budget_chars = min(
            RECALL_MANUAL_CONTEXT_BUDGET_MAX,
            max(base_context_budget_chars + 2000, int(base_context_budget_chars * 1.5)),
        )
        actions = [
            f"Raise context budget to {context_budget_chars} chars.",
            "Tighten the query or add a file hint to reduce competing context.",
            "Use pin compare once to see whether the source deserves the extra room.",
        ]
    elif miss_reason == "dropped_by_block_budget":
        diagnosis_title = "Block budget squeeze"
        diagnosis = "The source landed in a crowded block, and that block spent its local budget first."
        context_budget_chars = min(
            RECALL_MANUAL_CONTEXT_BUDGET_MAX,
            max(base_context_budget_chars + 1500, int(base_context_budget_chars * 1.35)),
        )
        limit = min(RECALL_MANUAL_LIMIT_MAX, max(base_limit, 10))
        actions = [
            f"Raise context budget to {context_budget_chars} chars.",
            "Make the query or file hint more exact so the source moves into a stronger block.",
            "Pin the source candidate once to compare it against the block winners.",
        ]
    elif miss_reason == "not_retrieved":
        diagnosis_title = "Retrieval miss"
        diagnosis = "The source never surfaced from retrieval, so this is a search-language or index-coverage problem first."
        limit = min(RECALL_MANUAL_LIMIT_MAX, max(base_limit + 4, 12))
        if not request_basis:
            request_basis = "prompt-or-artifact"
        actions = [
            "Refresh real data if this source is recent.",
            "Use the most literal query phrase you can, close to the source prompt or pass definition.",
            "Add the exact file hint or artifact path when one file should dominate.",
        ]
    elif miss_reason == "source_missing_from_index":
        diagnosis_title = "Index gap"
        diagnosis = "The source is missing from the current memory index, so ranking tweaks alone will not help yet."
        apply_ready = False
        actions = [
            "Refresh real data or rebuild the memory index first.",
            "Check that the source artifact still exists on disk.",
            "Rerun evaluation after the source reappears in the index.",
        ]
    elif miss_reason == "not_selected":
        diagnosis_title = "Selection loss"
        diagnosis = "The source was retrieved, but other candidates still looked stronger during selection."
        limit = min(RECALL_MANUAL_LIMIT_MAX, max(base_limit + 2, 10))

    diagnosis_lines = [diagnosis_title, diagnosis]
    if source_reasons:
        diagnosis_lines.append(f"source signals={', '.join(source_reasons[:4])}")
    if winner_labels != "n/a":
        diagnosis_lines.append(f"current winners={winner_labels}")

    suggested_manual_lines = [
        f"task={task_kind}",
        f"basis={request_basis or 'n/a'}",
        f"file_hint={file_hint or 'n/a'}",
        f"limit={limit}",
        f"budget={context_budget_chars}",
    ]
    if not apply_ready:
        suggested_manual_lines.append("manual tweak=wait until the source is back in the index")

    return {
        "task_kind": task_kind,
        "query_text": query_text,
        "request_basis": request_basis,
        "file_hint": file_hint,
        "limit": limit,
        "context_budget_chars": context_budget_chars,
        "apply_ready": apply_ready,
        "diagnosis_text": "\n".join(diagnosis_lines),
        "actions_text": "\n".join(f"{index}. {action}" for index, action in enumerate(actions[:3], start=1)),
        "suggested_manual_text": "\n".join(suggested_manual_lines),
    }


def apply_recall_miss_suggested_manual_config(
    suggestion: Mapping[str, Any],
    *,
    task_kind_var: tk.StringVar | Any,
    query_var: tk.StringVar | Any,
    request_basis_var: tk.StringVar | Any,
    file_hint_var: tk.StringVar | Any,
    limit_var: tk.Variable | Any,
    context_budget_var: tk.Variable | Any,
) -> None:
    task_kind_var.set(str(suggestion.get("task_kind") or "proposal"))
    query_var.set(str(suggestion.get("query_text") or ""))
    request_basis_var.set(str(suggestion.get("request_basis") or ""))
    file_hint_var.set(str(suggestion.get("file_hint") or ""))
    limit_var.set(
        coerce_ui_int(
            suggestion.get("limit"),
            default=DEFAULT_LIMIT,
            minimum=RECALL_MANUAL_LIMIT_MIN,
            maximum=RECALL_MANUAL_LIMIT_MAX,
        )
    )
    context_budget_var.set(
        coerce_ui_int(
            suggestion.get("context_budget_chars"),
            default=DEFAULT_CONTEXT_BUDGET_CHARS,
            minimum=RECALL_MANUAL_CONTEXT_BUDGET_MIN,
            maximum=RECALL_MANUAL_CONTEXT_BUDGET_MAX,
        )
    )


def apply_recall_diagnostic_guide_fields(
    miss_row: Mapping[str, Any] | None,
    *,
    request_row: Mapping[str, Any] | None,
    diagnosis_var: tk.StringVar | Any,
    action_var: tk.StringVar | Any,
    manual_var: tk.StringVar | Any,
) -> None:
    fields = dict(build_recall_diagnostic_guide_fields(miss_row, request_row=request_row))
    diagnosis_var.set(fields.get("Diagnosis", "n/a"))
    action_var.set(fields.get("Next", "n/a"))
    manual_var.set(fields.get("Suggested Manual", "n/a"))


def build_recall_diagnostic_guide_fields(
    miss_row: Mapping[str, Any] | None,
    *,
    request_row: Mapping[str, Any] | None,
) -> list[tuple[str, str]]:
    if not isinstance(miss_row, Mapping):
        return [
            ("Diagnosis", "Select an evaluation miss to see why it likely fell out of recall."),
            ("Next", "The next actions for retrieval, ranking, or budget tuning appear here."),
            ("Suggested Manual", "A suggested manual recall setup appears here."),
        ]

    suggestion = build_recall_miss_suggested_manual_config(miss_row, request_row=request_row)
    return [
        ("Diagnosis", str(suggestion.get("diagnosis_text") or "n/a")),
        ("Next", str(suggestion.get("actions_text") or "n/a")),
        ("Suggested Manual", str(suggestion.get("suggested_manual_text") or "n/a")),
    ]


def build_recall_request_summary(
    row: Mapping[str, Any] | None,
    *,
    empty: str = "Choose a prepared request or write a manual query.",
) -> str:
    if row is None:
        return empty

    query_text = summarize_text_preview(str(row.get("query_text") or ""), limit=132)
    file_hints = list(row.get("file_hints") or [])
    file_hint = file_hints[0] if file_hints else "n/a"
    header = (
        f"request {row.get('index')}: {row.get('task_kind') or 'n/a'}  "
        f"hit={'yes' if row.get('source_hit') else 'no'}  "
        f"status={row.get('source_status') or '-'}\n"
        f"query={query_text}\n"
        f"file_hint={file_hint}"
    )
    details: list[str] = []
    request_variant = str(row.get("request_variant") or "").strip()
    if request_variant and request_variant != BASELINE_REQUEST_VARIANT:
        details.append(f"variant={request_variant}")
    request_basis = str(row.get("request_basis") or "").strip()
    if request_basis:
        details.append(f"basis={request_basis}")
    if row.get("limit") is not None or row.get("context_budget_chars") is not None:
        details.append(
            f"limit={coerce_ui_int(row.get('limit'), default=DEFAULT_LIMIT, minimum=RECALL_MANUAL_LIMIT_MIN, maximum=RECALL_MANUAL_LIMIT_MAX)} "
            f"budget={coerce_ui_int(row.get('context_budget_chars'), default=DEFAULT_CONTEXT_BUDGET_CHARS, minimum=RECALL_MANUAL_CONTEXT_BUDGET_MIN, maximum=RECALL_MANUAL_CONTEXT_BUDGET_MAX)}"
        )
    miss_reason = str(row.get("miss_reason") or "").strip()
    if miss_reason:
        detail_line = [f"miss_reason={miss_reason}"]
        if row.get("source_rank") is not None:
            detail_line.append(f"source_rank={row.get('source_rank')}")
        source_block_title = str(row.get("source_block_title") or "").strip()
        if source_block_title:
            detail_line.append(f"source_block={source_block_title}")
        details.append(" ".join(detail_line))
    source_reasons = [
        str(item).strip()
        for item in (row.get("source_reasons") or [])
        if str(item).strip()
    ]
    if source_reasons:
        details.append(f"source_reasons={', '.join(source_reasons[:4])}")
    winner_labels = _recall_eval_winner_labels(row)
    if winner_labels != "n/a":
        details.append(f"won_by={winner_labels}")
    if not details:
        return header
    return header + "\n" + "\n".join(details)


def build_recall_candidate_summary(
    row: Mapping[str, Any] | None,
    *,
    root: Path,
    empty: str = "Select a recall candidate to open it in History / Forensics.",
) -> str:
    if row is None:
        return empty

    prompt_excerpt = summarize_text_preview(str(row.get("prompt_excerpt") or ""), limit=132)
    artifact_display = summarize_workspace_path(
        str(row.get("artifact_path") or ""),
        root=root,
    )
    reasons = [
        str(item).strip()
        for item in (row.get("reasons") or [])
        if str(item).strip()
    ]
    header = (
        f"candidate {row.get('event_id') or 'n/a'}: {row.get('block_title') or 'n/a'}  "
        f"surface={row.get('session_surface') or 'n/a'}  "
        f"status={row.get('status') or '-'}  "
        f"pinned={'yes' if row.get('pinned') else 'no'}"
    )
    group_member_count = coerce_ui_int(
        row.get("group_member_count"),
        default=0,
        minimum=0,
    )
    if group_member_count > 1:
        header += f"  group={group_member_count}"
    lines = [
        header,
        f"prompt={prompt_excerpt}\n"
        f"artifact={artifact_display}",
    ]
    group_members = _recall_group_members_text(row)
    if group_members:
        lines.append(f"group_members={group_members}")
    if reasons:
        lines.append(f"reasons={', '.join(reasons[:4])}")
    return "\n".join(lines)


def build_recall_pins_summary(
    rows: Mapping[str, Mapping[str, Any]] | list[Mapping[str, Any]] | None,
    *,
    empty: str = "Pinned candidates: none. Pin a candidate to carry it into the next recall run.",
) -> str:
    if isinstance(rows, Mapping):
        items = [dict(item) for item in rows.values() if isinstance(item, Mapping)]
    else:
        items = [dict(item) for item in (rows or []) if isinstance(item, Mapping)]
    if not items:
        return empty

    labels = [
        summarize_text_preview(
            str(item.get("prompt_excerpt") or item.get("event_id") or "n/a"),
            limit=56,
        )
        for item in items[:3]
    ]
    more = len(items) - len(labels)
    suffix = f"; +{more} more" if more > 0 else ""
    return f"Pinned for the next recall run ({len(items)}): {'; '.join(labels)}{suffix}"


def recall_bundle_has_candidate_navigation(bundle: Mapping[str, Any]) -> bool:
    selected = [item for item in (bundle.get("selected_candidates") or []) if isinstance(item, Mapping)]
    if not selected:
        return True
    for item in selected:
        artifact_path = str(item.get("artifact_path") or "").strip()
        session_surface = str(item.get("session_surface") or "").strip().lower()
        if artifact_path or session_surface in SESSION_SURFACES:
            continue
        return False
    return True


def build_evaluation_snapshot_state(
    snapshot: Mapping[str, Any] | None,
    *,
    empty: str = "Refresh the evaluation snapshot to load software-work signals.",
) -> str:
    if not snapshot:
        return empty
    counts = dict(snapshot.get("counts") or {})
    generated_at_utc = str(snapshot.get("generated_at_utc") or "").strip()
    state = (
        f"{int(snapshot.get('signal_count') or 0)} signal(s) from "
        f"{int(snapshot.get('event_count') or 0)} event(s). "
        f"accept={int(counts.get('acceptance') or 0)}, "
        f"reject={int(counts.get('rejection') or 0)}, "
        f"review={int(counts.get('review_resolved') or 0)}/{int(counts.get('review_unresolved') or 0)}, "
        f"pass={int(counts.get('test_pass') or 0)}, "
        f"fail={int(counts.get('test_fail') or 0)}."
    )
    if generated_at_utc:
        state += f" Generated {generated_at_utc}."
    return state


def build_evaluation_acceptance_text(snapshot: Mapping[str, Any] | None) -> str:
    if not snapshot:
        return "No acceptance or rejection signals loaded yet."
    counts = dict(snapshot.get("counts") or {})
    return (
        f"accepted={int(counts.get('acceptance') or 0)}; "
        f"rejected={int(counts.get('rejection') or 0)}; "
        f"review={int(counts.get('review_resolved') or 0)}/{int(counts.get('review_unresolved') or 0)}; "
        f"explicit={int(snapshot.get('explicit_signal_count') or 0)}"
    )


def build_evaluation_test_text(snapshot: Mapping[str, Any] | None) -> str:
    if not snapshot:
        return "No test pass/fail signals loaded yet."
    counts = dict(snapshot.get("counts") or {})
    return (
        f"passed={int(counts.get('test_pass') or 0)}; "
        f"failed={int(counts.get('test_fail') or 0)}; "
        f"derived={int(snapshot.get('derived_signal_count') or 0)}"
    )


def build_evaluation_repair_text(snapshot: Mapping[str, Any] | None) -> str:
    if not snapshot:
        return "No repair or follow-up links loaded yet."
    counts = dict(snapshot.get("counts") or {})
    return (
        f"repaired={int(counts.get('repaired_failures') or 0)}; "
        f"follow-up={int(counts.get('followed_up_failures') or 0)}; "
        f"pending={int(counts.get('pending_failures') or 0)}"
    )


def build_evaluation_comparison_text(snapshot: Mapping[str, Any] | None) -> str:
    if not snapshot:
        return "No comparison records loaded yet."
    counts = dict(snapshot.get("counts") or {})
    backend_comparison_count = sum(
        1
        for item in snapshot.get("comparisons") or []
        if isinstance(item, Mapping)
        if isinstance(item.get("backend_comparison"), Mapping)
        and item["backend_comparison"].get("backend_count")
    )
    return (
        f"comparisons={int(counts.get('comparisons') or 0)}; "
        f"winners={int(counts.get('comparison_winners') or 0)}; "
        f"open={int(counts.get('unresolved_comparisons') or 0)}; "
        f"backend={backend_comparison_count}"
    )


def build_evaluation_curation_text(snapshot: Mapping[str, Any] | None) -> str:
    if not snapshot:
        return "No curation candidates loaded yet."
    counts = dict(snapshot.get("counts") or {})
    return (
        f"ready={int(counts.get('curation_ready') or 0)}; "
        f"review={int(counts.get('curation_needs_review') or 0)}; "
        f"blocked={int(counts.get('curation_blocked') or 0)}"
    )


def build_evaluation_adoption_text(preview: Mapping[str, Any] | None) -> str:
    if not preview:
        return "No adoption preview loaded yet."
    counts = dict(preview.get("counts") or {})
    return (
        f"matched={int(counts.get('matched_candidate_count') or 0)}; "
        f"ready-for-policy={int(counts.get('ready_for_policy') or 0)}; "
        "export=preview-only"
    )


def build_evaluation_curation_rows(preview: Mapping[str, Any] | None, *, limit: int = 24) -> list[dict[str, str]]:
    if not preview:
        return []
    rows: list[dict[str, str]] = []
    for index, item in enumerate(preview.get("candidates") or [], start=1):
        if not isinstance(item, Mapping):
            continue
        event_id = str(item.get("event_id") or "").strip()
        steps = [
            str(step).strip()
            for step in (item.get("required_next_steps") or [])
            if str(step).strip()
        ]
        reasons = [
            str(reason).strip()
            for reason in (item.get("reasons") or [])
            if str(reason).strip()
        ]
        rows.append(
            {
                "row_id": event_id or f"curation-{index}",
                "event_id": event_id,
                "state": str(item.get("state") or ""),
                "decision": str(item.get("export_decision") or ""),
                "next": summarize_text_preview(", ".join(steps), limit=120),
                "label": summarize_text_preview(str(item.get("label") or event_id or "candidate"), limit=140),
                "reasons": summarize_text_preview(", ".join(reasons), limit=160),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _ui_clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _ui_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _ui_int(value: Any, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ui_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return False
    if isinstance(value, int):
        return value != 0
    return False


def _ui_unparseable_bool_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        return None
    return value.strip()


def _ui_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        cleaned = _ui_clean_text(value)
        return [cleaned] if cleaned is not None else []
    if not isinstance(value, list):
        return []
    return [cleaned for item in value if (cleaned := _ui_clean_text(item)) is not None]


def _ui_lifecycle_state(value: Any, *, default: str = "unknown") -> str:
    state = _ui_clean_text(value)
    if state in LEARNING_LIFECYCLE_STATES:
        return state
    return default


def _read_json_mapping(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, Mapping):
        return None, "artifact root is not a JSON object"
    return dict(payload), None


def _learning_candidate_counts_text(payload: Mapping[str, Any]) -> str:
    counts = _ui_mapping(payload.get("counts"))
    keys = [
        "source_candidate_count",
        "review_queue_count",
        "eligible_candidate_count",
        "previewed_candidate_count",
        "excluded_candidate_count",
        "requested_candidate_count",
        "matched_candidate_count",
        "selected_candidate_count",
        "inspected_candidate_count",
        "future_jsonl_candidate_if_separately_approved_count",
        "would_write_jsonl_record_count",
        "base_candidate_count",
        "target_candidate_count",
        "added_candidate_count",
        "removed_candidate_count",
        "changed_candidate_count",
    ]
    parts = [
        f"{key}={_ui_int(counts.get(key))}"
        for key in keys
        if key in counts
    ]
    return "; ".join(parts) if parts else "counts=n/a"


def _learning_candidate_pick(
    key: str,
    item: Mapping[str, Any],
    queue: Mapping[str, Any],
    *,
    default: str = "n/a",
) -> str:
    return _ui_clean_text(queue.get(key)) or _ui_clean_text(item.get(key)) or default


def _learning_candidate_policy_state(item: Mapping[str, Any], queue_summary: Mapping[str, Any]) -> str:
    for source in (
        item,
        queue_summary,
        _ui_mapping(item.get("lifecycle_summary")),
        _ui_mapping(queue_summary.get("lifecycle_summary")),
    ):
        state = _ui_clean_text(source.get("policy_state"))
        if state is not None:
            return state
    confirmation = _ui_mapping(queue_summary.get("export_policy_confirmation")) or _ui_mapping(
        item.get("export_policy_confirmation")
    )
    policy = _ui_mapping(item.get("policy"))
    if _ui_bool(confirmation.get("confirmed")) or _ui_bool(policy.get("export_policy_confirmed")):
        return "confirmed"
    if _ui_clean_text(queue_summary.get("queue_state")) == "ready":
        return "pending_confirmation"
    return "unknown"


def _learning_candidate_lifecycle_state(item: Mapping[str, Any], queue_summary: Mapping[str, Any]) -> str:
    for source in (
        _ui_mapping(queue_summary.get("lifecycle_summary")),
        _ui_mapping(item.get("lifecycle_summary")),
        queue_summary,
        item,
    ):
        state = _ui_clean_text(source.get("lifecycle_state"))
        if state is not None:
            return _ui_lifecycle_state(state)
    queue_state = _ui_clean_text(queue_summary.get("queue_state")) or _ui_clean_text(item.get("queue_state"))
    policy_state = _learning_candidate_policy_state(item, queue_summary)
    if queue_state == "ready" and policy_state == "confirmed":
        return "completed"
    if queue_state in {"blocked", "missing_source", "missing_supervised_text"}:
        return "blocked"
    if queue_state in {"ready", "needs_review"}:
        return "queued"
    return "unknown"


def _learning_candidate_comparison_role(item: Mapping[str, Any], queue_summary: Mapping[str, Any]) -> str:
    roles: list[str] = []
    for source in (item, queue_summary):
        roles.extend(_ui_string_list(source.get("comparison_roles")))
        direct_role = _ui_clean_text(source.get("comparison_role"))
        if direct_role and direct_role != "none":
            roles.extend(role.strip() for role in direct_role.split(",") if role.strip())
        roles.extend(_ui_string_list(_ui_mapping(source.get("comparison_evidence")).get("roles")))
    evidence_summary = _ui_mapping(item.get("evidence_summary"))
    roles.extend(_ui_string_list(evidence_summary.get("comparison_roles")))
    deduped: list[str] = []
    seen: set[str] = set()
    for role in roles:
        if role in seen:
            continue
        seen.add(role)
        deduped.append(role)
    return ",".join(deduped) if deduped else "n/a"


def _learning_candidate_backend_id(item: Mapping[str, Any], queue_summary: Mapping[str, Any]) -> str:
    backend_metadata = _ui_mapping(item.get("backend_metadata")) or _ui_mapping(
        queue_summary.get("backend_metadata")
    )
    return (
        _ui_clean_text(item.get("backend_id"))
        or _ui_clean_text(backend_metadata.get("backend_id"))
        or _ui_clean_text(backend_metadata.get("model_id"))
        or "n/a"
    )


def _learning_candidate_source_path(
    item: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    artifact_path: str,
) -> str:
    source_event = _ui_mapping(item.get("source_event"))
    path_sources = (
        _ui_mapping(item.get("source_paths")),
        source_event,
        _ui_mapping(artifact.get("source_paths")),
        _ui_mapping(artifact.get("paths")),
        artifact,
    )
    path_keys = (
        "source_artifact_path",
        "artifact_path",
        "source_learning_preview_path",
        "source_human_selected_candidates_path",
        "learning_preview_run_path",
        "learning_preview_latest_path",
        "human_selected_run_path",
        "human_selected_latest_path",
        "jsonl_export_dry_run_run_path",
        "jsonl_export_dry_run_latest_path",
        "base_artifact_path",
        "target_artifact_path",
        "candidate_diff_run_path",
        "candidate_diff_latest_path",
    )
    for source in path_sources:
        for key in path_keys:
            path = _ui_clean_text(source.get(key))
            if path is not None:
                return path
    return artifact_path


def _learning_candidate_row(
    item: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    artifact_key: str,
    artifact_label: str,
    artifact_path: str,
    source_collection: str,
    source_index: int,
) -> dict[str, str]:
    queue_summary = _ui_mapping(item.get("review_queue")) or dict(item)
    event_id = _ui_clean_text(item.get("event_id")) or f"{artifact_key}-{source_index}"
    blocked_reasons = _ui_string_list(item.get("blocked_reasons")) or _ui_string_list(
        queue_summary.get("blocked_reasons")
    )
    blocked_reason = (
        _ui_clean_text(queue_summary.get("blocked_reason"))
        or _ui_clean_text(item.get("blocked_reason"))
        or (blocked_reasons[0] if blocked_reasons else "n/a")
    )
    label = (
        _ui_clean_text(item.get("label"))
        or _ui_clean_text(_ui_mapping(item.get("source_event")).get("prompt_excerpt"))
        or event_id
    )
    return {
        "row_id": f"{artifact_key}:{source_collection}:{source_index}:{event_id}",
        "artifact": artifact_label,
        "artifact_key": artifact_key,
        "event_id": event_id,
        "queue_state": _learning_candidate_pick("queue_state", item, queue_summary, default="unknown"),
        "lifecycle_state": _learning_candidate_lifecycle_state(item, queue_summary),
        "blocked_reason": blocked_reason,
        "next_action": _learning_candidate_pick("next_action", item, queue_summary, default="review_candidate"),
        "policy_state": _learning_candidate_policy_state(item, queue_summary),
        "comparison_role": _learning_candidate_comparison_role(item, queue_summary),
        "backend_id": _learning_candidate_backend_id(item, queue_summary),
        "source_path": _learning_candidate_source_path(item, artifact, artifact_path=artifact_path),
        "label": summarize_text_preview(label, limit=140),
    }


def _learning_candidate_review_rows_for_artifact(
    artifact: Mapping[str, Any],
    *,
    artifact_key: str,
    artifact_label: str,
    artifact_path: str,
) -> list[dict[str, str]]:
    if artifact_key == "learning_preview":
        source_items = artifact.get("review_queue") or []
        source_collection = "review_queue"
        if not source_items:
            source_items = list(artifact.get("supervised_example_candidates") or []) + list(
                artifact.get("excluded_candidates") or []
            )
            source_collection = "learning_candidates"
    elif artifact_key == "human_selected":
        source_items = artifact.get("selected_candidates") or []
        source_collection = "selected_candidates"
    elif artifact_key == "jsonl_dry_run":
        source_items = artifact.get("candidates") or []
        source_collection = "dry_run_candidates"
    elif artifact_key == "candidate_diff":
        rows: list[dict[str, str]] = []
        for index, change in enumerate(artifact.get("changes") or [], start=1):
            if not isinstance(change, Mapping):
                continue
            change_type = _ui_clean_text(change.get("change_type")) or "changed"
            record = _ui_mapping(change.get("after")) or _ui_mapping(change.get("before"))
            if not record:
                continue
            event_id = _ui_clean_text(change.get("event_id"))
            if event_id is not None and not _ui_clean_text(record.get("event_id")):
                record = {**record, "event_id": event_id}
            row = _learning_candidate_row(
                record,
                artifact,
                artifact_key=artifact_key,
                artifact_label=artifact_label,
                artifact_path=artifact_path,
                source_collection="changes",
                source_index=index,
            )
            row["label"] = summarize_text_preview(f"{change_type}: {row['label']}", limit=140)
            rows.append(row)
        return rows
    else:
        source_items = artifact.get("candidates") or []
        source_collection = "candidates"

    rows = []
    for index, item in enumerate(source_items, start=1):
        if not isinstance(item, Mapping):
            continue
        rows.append(
            _learning_candidate_row(
                item,
                artifact,
                artifact_key=artifact_key,
                artifact_label=artifact_label,
                artifact_path=artifact_path,
                source_collection=source_collection,
                source_index=index,
            )
        )
    return rows


def build_learning_candidate_review_snapshot(
    *,
    root: Path,
    workspace_id: str,
) -> dict[str, Any]:
    resolved_root = Path(root).resolve()
    artifacts: list[dict[str, Any]] = []
    rows: list[dict[str, str]] = []
    policy_warnings: list[str] = []
    for artifact_key, artifact_label, path_builder in LEARNING_CANDIDATE_REVIEW_ARTIFACTS:
        path = path_builder(workspace_id=workspace_id, root=resolved_root)
        artifact_record: dict[str, Any] = {
            "artifact_key": artifact_key,
            "label": artifact_label,
            "path": str(path),
            "exists": path.exists(),
            "loaded": False,
            "status": "missing",
            "counts_text": "counts=n/a",
            "candidate_count": 0,
        }
        if not path.exists():
            artifacts.append(artifact_record)
            continue

        payload, error = _read_json_mapping(path)
        if payload is None:
            artifact_record.update(
                {
                    "status": "unreadable",
                    "error": error or "artifact could not be read",
                }
            )
            artifacts.append(artifact_record)
            continue

        artifact_rows = _learning_candidate_review_rows_for_artifact(
            payload,
            artifact_key=artifact_key,
            artifact_label=artifact_label,
            artifact_path=str(path),
        )
        rows.extend(artifact_rows)
        counts = _ui_mapping(payload.get("counts"))
        export_policy = _ui_mapping(payload.get("export_policy"))
        export_mode = _ui_clean_text(payload.get("export_mode")) or "unknown"
        training_export_ready = _ui_bool(payload.get("training_export_ready"))
        human_gate_required = _ui_bool(payload.get("human_gate_required"))
        jsonl_written = _ui_int(counts.get("would_write_jsonl_record_count"))
        if export_mode != "preview_only":
            policy_warnings.append(f"{artifact_label} reports export_mode={export_mode}")
        if unparseable := _ui_unparseable_bool_text(payload.get("human_gate_required")):
            policy_warnings.append(f"{artifact_label} reports human_gate_required={unparseable} (unparseable)")
        elif not human_gate_required:
            policy_warnings.append(f"{artifact_label} reports human_gate_required=false")
        if unparseable := _ui_unparseable_bool_text(export_policy.get("jsonl_file_written")):
            policy_warnings.append(f"{artifact_label} reports jsonl_file_written={unparseable} (unparseable)")
        elif _ui_bool(export_policy.get("jsonl_file_written")):
            policy_warnings.append(f"{artifact_label} reports jsonl_file_written=true")
        if jsonl_written:
            policy_warnings.append(f"{artifact_label} reports {jsonl_written} JSONL record(s) would be written")
        if unparseable := _ui_unparseable_bool_text(payload.get("training_export_ready")):
            policy_warnings.append(f"{artifact_label} reports training_export_ready={unparseable} (unparseable)")
        elif training_export_ready:
            policy_warnings.append(f"{artifact_label} reports training_export_ready=true")
        artifact_record.update(
            {
                "loaded": True,
                "status": "loaded",
                "schema_name": _ui_clean_text(payload.get("schema_name")),
                "schema_version": payload.get("schema_version"),
                "artifact_kind": artifact_key,
                "generated_at_utc": _ui_clean_text(payload.get("generated_at_utc")),
                "export_mode": export_mode,
                "training_export_ready": training_export_ready,
                "human_gate_required": human_gate_required,
                "counts": counts,
                "counts_text": _learning_candidate_counts_text(payload),
                "candidate_count": len(artifact_rows),
                "jsonl_record_write_count": jsonl_written,
            }
        )
        artifacts.append(artifact_record)

    row_state_counts = Counter(row["queue_state"] for row in rows)
    row_lifecycle_state_counts = Counter(row["lifecycle_state"] for row in rows)
    row_next_action_counts = Counter(row["next_action"] for row in rows)
    row_blocked_reason_counts = Counter(
        row["blocked_reason"] for row in rows if row["blocked_reason"] != "n/a"
    )
    loaded_artifacts = [artifact for artifact in artifacts if artifact.get("loaded")]
    counts = {
        "expected_artifact_count": len(artifacts),
        "loaded_artifact_count": len(loaded_artifacts),
        "missing_artifact_count": sum(1 for artifact in artifacts if artifact.get("status") == "missing"),
        "unreadable_artifact_count": sum(1 for artifact in artifacts if artifact.get("status") == "unreadable"),
        "candidate_row_count": len(rows),
        "preview_only_artifact_count": sum(
            1 for artifact in loaded_artifacts if artifact.get("export_mode") == "preview_only"
        ),
        "human_gate_required_artifact_count": sum(
            1 for artifact in loaded_artifacts if artifact.get("human_gate_required")
        ),
        "training_ready_artifact_count": sum(
            1 for artifact in loaded_artifacts if artifact.get("training_export_ready")
        ),
        "jsonl_record_write_count": sum(
            _ui_int(artifact.get("jsonl_record_write_count"))
            for artifact in loaded_artifacts
        ),
        "queue_states": {key: _ui_int(value) for key, value in sorted(row_state_counts.items())},
        "lifecycle_states": {key: _ui_int(value) for key, value in sorted(row_lifecycle_state_counts.items())},
        "next_actions": {key: _ui_int(value) for key, value in sorted(row_next_action_counts.items())},
        "blocked_reasons": {key: _ui_int(value) for key, value in sorted(row_blocked_reason_counts.items())},
    }
    return {
        "workspace_id": workspace_id,
        "mode": "read_only_latest_artifacts",
        "root": str(resolved_root),
        "counts": counts,
        "artifacts": artifacts,
        "rows": rows,
        "policy_warnings": policy_warnings,
    }


def build_learning_candidate_review_state(review: Mapping[str, Any] | None) -> str:
    if not review:
        return "No learning candidate artifacts loaded yet."
    counts = _ui_mapping(review.get("counts"))
    lifecycle_states = _ui_mapping(counts.get("lifecycle_states"))
    lifecycle_text = ""
    if lifecycle_states:
        ordered_states = [state for state in LEARNING_LIFECYCLE_STATES if state in lifecycle_states]
        ordered_states.extend(sorted(set(lifecycle_states) - set(ordered_states)))
        lifecycle_text = "; lifecycle=" + ",".join(
            f"{state}:{_ui_int(lifecycle_states.get(state))}"
            for state in ordered_states
        )
    return (
        f"latest artifacts loaded={_ui_int(counts.get('loaded_artifact_count'))}/"
        f"{_ui_int(counts.get('expected_artifact_count'))}; "
        f"rows={_ui_int(counts.get('candidate_row_count'))}; "
        f"preview-only={_ui_int(counts.get('preview_only_artifact_count'))}; "
        f"human-gated={_ui_int(counts.get('human_gate_required_artifact_count'))}; "
        f"training-ready={_ui_int(counts.get('training_ready_artifact_count'))}; "
        f"jsonl-would-write={_ui_int(counts.get('jsonl_record_write_count'))}; "
        f"missing={_ui_int(counts.get('missing_artifact_count'))}"
        f"{lifecycle_text}"
    )


def build_learning_candidate_review_rows(
    review: Mapping[str, Any] | None,
    *,
    limit: int = 32,
) -> list[dict[str, str]]:
    if not review:
        return []
    rows = [
        dict(row)
        for row in review.get("rows") or []
        if isinstance(row, Mapping)
    ]
    return rows[:limit]


def build_learning_candidate_review_report(review: Mapping[str, Any] | None) -> str:
    if not review:
        return "Learning candidate review: read-only\nNo latest learning artifacts were inspected."
    counts = _ui_mapping(review.get("counts"))
    lines = [
        "Learning candidate review: read-only",
        "Source: latest file-first learning artifacts",
        (
            f"Artifacts loaded: {_ui_int(counts.get('loaded_artifact_count'))}/"
            f"{_ui_int(counts.get('expected_artifact_count'))}"
        ),
        f"Candidate rows: {_ui_int(counts.get('candidate_row_count'))}",
        f"Training export ready artifacts: {_ui_int(counts.get('training_ready_artifact_count'))}",
        f"JSONL records that would be written: {_ui_int(counts.get('jsonl_record_write_count'))}",
    ]
    queue_states = _ui_mapping(counts.get("queue_states"))
    if queue_states:
        lines.append(
            "Queue states: "
            + "; ".join(f"{key}={_ui_int(value)}" for key, value in queue_states.items())
        )
    lifecycle_states = _ui_mapping(counts.get("lifecycle_states"))
    if lifecycle_states:
        ordered_states = [state for state in LEARNING_LIFECYCLE_STATES if state in lifecycle_states]
        ordered_states.extend(sorted(set(lifecycle_states) - set(ordered_states)))
        lines.append(
            "Lifecycle states: "
            + "; ".join(f"{key}={_ui_int(lifecycle_states.get(key))}" for key in ordered_states)
        )
    next_actions = _ui_mapping(counts.get("next_actions"))
    if next_actions:
        lines.append(
            "Next actions: "
            + "; ".join(f"{key}={_ui_int(value)}" for key, value in next_actions.items())
        )
    blocked_reasons = _ui_mapping(counts.get("blocked_reasons"))
    if blocked_reasons:
        lines.append(
            "Blocked reasons: "
            + "; ".join(f"{key}={_ui_int(value)}" for key, value in blocked_reasons.items())
        )
    warnings = _ui_string_list(review.get("policy_warnings"))
    if warnings:
        lines.extend(["", "Policy warnings:", *[f"- {warning}" for warning in warnings]])
    else:
        lines.append("Policy boundary: preview-only / human-gated; no JSONL output.")

    artifacts = [
        artifact
        for artifact in review.get("artifacts") or []
        if isinstance(artifact, Mapping)
    ]
    if artifacts:
        lines.extend(["", "Artifacts:"])
        for artifact in artifacts:
            status = _ui_clean_text(artifact.get("status")) or "unknown"
            label = _ui_clean_text(artifact.get("label")) or "artifact"
            counts_text = _ui_clean_text(artifact.get("counts_text")) or "counts=n/a"
            path = _ui_clean_text(artifact.get("path")) or "n/a"
            lines.append(f"- {label}: {status}; {counts_text}; source={path}")

    rows = build_learning_candidate_review_rows(review, limit=12)
    if rows:
        lines.extend(["", "Inspection rows:"])
        for row in rows:
            lines.append(
                f"- {row['artifact']} {row['event_id']} "
                f"state={row['queue_state']} lifecycle={row['lifecycle_state']} blocked={row['blocked_reason']} "
                f"next={row['next_action']} policy={row['policy_state']} "
                f"comparison={row['comparison_role']} backend={row['backend_id']} "
                f"source={row['source_path']}"
            )
    return "\n".join(lines)


def build_evaluation_signal_rows(snapshot: Mapping[str, Any] | None, *, limit: int = 12) -> list[dict[str, str]]:
    if not snapshot:
        return []
    rows: list[dict[str, str]] = []
    for index, item in enumerate(snapshot.get("recent_signals") or [], start=1):
        if not isinstance(item, Mapping):
            continue
        source_event_id = str(item.get("source_event_id") or "").strip()
        target_event_id = str(item.get("target_event_id") or "").strip()
        relation_kind = str(item.get("relation_kind") or "").strip()
        relation = f"{relation_kind} {target_event_id}" if relation_kind and target_event_id else ""
        label = (
            str(item.get("test_name") or "").strip()
            or str(item.get("prompt_excerpt") or "").strip()
            or source_event_id
            or "signal"
        )
        detail = (
            str(item.get("rationale") or "").strip()
            or str(item.get("resolution_summary") or "").strip()
            or str(item.get("review_id") or "").strip()
            or str(item.get("failure_summary") or "").strip()
            or str(item.get("quality_status") or "").strip()
            or str(item.get("status") or "").strip()
        )
        rows.append(
            {
                "row_id": str(item.get("signal_id") or f"signal-{index}"),
                "kind": str(item.get("signal_kind") or ""),
                "source": source_event_id,
                "relation": relation,
                "status": str(item.get("quality_status") or item.get("status") or ""),
                "label": summarize_text_preview(label, limit=120),
                "detail": summarize_text_preview(detail, limit=180),
            }
        )
        if len(rows) >= limit:
            break
    return rows


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

    def collect_recall_requests(
        self,
        *,
        refresh_dataset: bool = False,
        max_requests: int = RECALL_MAX_REQUESTS,
    ) -> dict[str, Any]:
        dataset, dataset_path = ensure_recall_dataset(
            root=self.workspace_store.root,
            workspace_id=self.workspace_store.workspace_id,
            refresh=refresh_dataset,
            max_requests=max_requests,
        )
        requests = request_catalog(dataset)
        request_entries = [
            dict(item)
            for item in (dataset.get("requests") or [])
            if isinstance(item, Mapping)
        ]
        return {
            "selected_model_id": self.selected_model_id,
            "dataset_path": str(dataset_path),
            "generated_at_utc": dataset.get("generated_at_utc"),
            "request_count": len(requests),
            "requests": requests,
            "request_entries": request_entries,
            "index_summary": dict(dataset.get("index_summary") or {}),
        }

    def evaluate_recall_dataset(
        self,
        *,
        refresh_dataset: bool = False,
        max_requests: int = RECALL_MAX_REQUESTS,
    ) -> dict[str, Any]:
        dataset, dataset_path = ensure_recall_dataset(
            root=self.workspace_store.root,
            workspace_id=self.workspace_store.workspace_id,
            refresh=refresh_dataset,
            max_requests=max_requests,
        )
        previous_summary, _latest_evaluation_path = load_latest_evaluation_summary(
            workspace_id=self.workspace_store.workspace_id,
            root=self.workspace_store.root,
        )
        summary = evaluate_dataset(
            dataset,
            root=self.workspace_store.root,
            workspace_id=self.workspace_store.workspace_id,
        )
        synced_dataset = sync_dataset_with_evaluation(dataset, summary)
        write_json(dataset_path, synced_dataset)
        recorded_summary, latest_path, run_path = record_evaluation_summary(
            summary,
            workspace_id=self.workspace_store.workspace_id,
            root=self.workspace_store.root,
            dataset_path=dataset_path,
            previous_summary=previous_summary,
        )
        requests = request_catalog(synced_dataset)
        request_entries = [
            dict(item)
            for item in (synced_dataset.get("requests") or [])
            if isinstance(item, Mapping)
        ]
        report_blocks = [
            format_evaluation_report(recorded_summary, dataset_path=dataset_path),
            format_miss_report(recorded_summary, dataset_path=dataset_path),
        ]
        return {
            "selected_model_id": self.selected_model_id,
            "dataset_path": str(dataset_path),
            "evaluation_latest_path": str(latest_path),
            "evaluation_run_path": str(run_path),
            "request_count": len(requests),
            "requests": requests,
            "request_entries": request_entries,
            "summary": recorded_summary,
            "previous_summary": previous_summary,
            "comparison": {
                "before_summary": previous_summary,
                "after_summary": recorded_summary,
            },
            "misses": [
                dict(item)
                for item in (recorded_summary.get("misses") or [])
                if isinstance(item, Mapping)
            ],
            "report": "\n\n".join(block for block in report_blocks if block),
        }

    def build_evaluation_snapshot(self, *, curation_filters: Mapping[str, Any] | None = None) -> dict[str, Any]:
        snapshot, latest_path, run_path = record_evaluation_snapshot(
            root=self.workspace_store.root,
            workspace_id=self.workspace_store.workspace_id,
        )
        curation_preview, preview_latest_path, preview_run_path = record_curation_export_preview(
            root=self.workspace_store.root,
            workspace_id=self.workspace_store.workspace_id,
            snapshot=snapshot,
            filters=curation_filters,
        )
        report = "\n\n".join(
            [
                format_evaluation_snapshot_report(snapshot),
                format_curation_export_preview_report(curation_preview),
            ]
        )
        return {
            "selected_model_id": self.selected_model_id,
            "snapshot": snapshot,
            "snapshot_latest_path": str(latest_path),
            "snapshot_run_path": str(run_path),
            "curation_preview": curation_preview,
            "curation_preview_latest_path": str(preview_latest_path),
            "curation_preview_run_path": str(preview_run_path),
            "report": report,
        }

    def build_learning_candidate_review(self) -> dict[str, Any]:
        review = build_learning_candidate_review_snapshot(
            root=self.workspace_store.root,
            workspace_id=self.workspace_store.workspace_id,
        )
        review["report"] = build_learning_candidate_review_report(review)
        return review

    def record_evaluation_review_resolution(
        self,
        *,
        source_event_id: str,
        resolved: bool,
        review_id: str = "",
        review_url: str = "",
        resolution_summary: str = "",
        curation_filters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        signal = record_review_resolution_signal(
            root=self.workspace_store.root,
            workspace_id=self.workspace_store.workspace_id,
            source_event_id=source_event_id,
            resolved=resolved,
            review_id=review_id,
            review_url=review_url,
            resolution_summary=resolution_summary,
            origin="local_ui",
        )
        result = self.build_evaluation_snapshot(curation_filters=curation_filters)
        result["recorded_signal"] = signal
        return result

    def _ui_recall_bundle_path(self, *, task_kind: str) -> Path:
        return (
            self.workspace_store.root
            / "artifacts"
            / "recall_data"
            / self.workspace_store.workspace_id
            / "ui"
            / f"{timestamp_slug()}-{task_kind}.json"
        )

    def build_recall_bundle(
        self,
        *,
        request_index: int | None = None,
        task_kind: str | None = None,
        query_text: str = "",
        request_basis: str | None = None,
        file_hint: str | None = None,
        pinned_event_ids: list[str] | tuple[str, ...] | None = None,
        limit: int = DEFAULT_LIMIT,
        context_budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS,
        refresh_dataset: bool = False,
        max_requests: int = RECALL_MAX_REQUESTS,
    ) -> dict[str, Any]:
        bundle_path: Path | None = None
        dataset_path: Path | None = None
        request_label: str | None = None
        pin_compare: dict[str, Any] | None = None
        request_payload: dict[str, Any]
        normalized_pinned_event_ids: list[str] = []
        seen_pins: set[str] = set()
        for event_id in pinned_event_ids or []:
            normalized = str(event_id or "").strip()
            if not normalized or normalized in seen_pins:
                continue
            seen_pins.add(normalized)
            normalized_pinned_event_ids.append(normalized)

        if request_index is not None:
            dataset, dataset_path = ensure_recall_dataset(
                root=self.workspace_store.root,
                workspace_id=self.workspace_store.workspace_id,
                refresh=refresh_dataset,
                max_requests=max_requests,
            )
            request_payload, entry = select_dataset_request(dataset, request_index=request_index)
            request_label = (
                f"dataset[{request_index}] "
                f"{str(entry.get('task_kind') or '').strip()} "
                f"hit={'yes' if entry.get('source_hit') else 'no'}"
            )
            if normalized_pinned_event_ids:
                before_bundle_path = dataset_bundle_path(entry)
                before_bundle: dict[str, Any]
                if before_bundle_path is not None and before_bundle_path.exists():
                    before_bundle = read_bundle(before_bundle_path)
                    if not recall_bundle_has_candidate_navigation(before_bundle):
                        before_bundle = build_context_bundle(
                            request_payload,
                            root=self.workspace_store.root,
                            workspace_id=self.workspace_store.workspace_id,
                        )
                        write_json(before_bundle_path, dict(before_bundle))
                else:
                    before_bundle = build_context_bundle(
                        request_payload,
                        root=self.workspace_store.root,
                        workspace_id=self.workspace_store.workspace_id,
                    )
                    if before_bundle_path is not None:
                        write_json(before_bundle_path, dict(before_bundle))
                request_payload["pinned_event_ids"] = list(normalized_pinned_event_ids)
                request_label += f" pins={len(normalized_pinned_event_ids)}"
                bundle = build_context_bundle(
                    request_payload,
                    root=self.workspace_store.root,
                    workspace_id=self.workspace_store.workspace_id,
                )
                bundle_path = self._ui_recall_bundle_path(task_kind=str(request_payload.get("task_kind") or "proposal"))
                write_json(bundle_path, dict(bundle))
                pin_compare = {
                    "before_bundle": before_bundle,
                    "after_bundle": bundle,
                }
            else:
                bundle_path = dataset_bundle_path(entry)
                if bundle_path is not None and bundle_path.exists():
                    bundle = read_bundle(bundle_path)
                    if not recall_bundle_has_candidate_navigation(bundle):
                        bundle = build_context_bundle(
                            request_payload,
                            root=self.workspace_store.root,
                            workspace_id=self.workspace_store.workspace_id,
                        )
                        write_json(bundle_path, dict(bundle))
                else:
                    bundle = build_context_bundle(
                        request_payload,
                        root=self.workspace_store.root,
                        workspace_id=self.workspace_store.workspace_id,
                    )
                    if bundle_path is not None:
                        write_json(bundle_path, dict(bundle))
        else:
            normalized_task_kind = (task_kind or "").strip().lower()
            normalized_query = query_text.strip()
            normalized_request_basis = (request_basis or "").strip() or None
            normalized_file_hint = (file_hint or "").strip()
            if normalized_task_kind not in TASK_KINDS:
                raise ValueError("Choose a valid recall task kind before running a manual query.")
            if not normalized_query:
                raise ValueError("Manual recall needs a non-empty query.")
            request_payload = {
                "task_kind": normalized_task_kind,
                "query_text": normalized_query,
                "file_hints": [normalized_file_hint] if normalized_file_hint else [],
                "limit": limit,
                "context_budget_chars": context_budget_chars,
            }
            if normalized_request_basis:
                request_payload["request_basis"] = normalized_request_basis
            if normalized_pinned_event_ids:
                before_bundle = build_context_bundle(
                    request_payload,
                    root=self.workspace_store.root,
                    workspace_id=self.workspace_store.workspace_id,
                )
                request_payload["pinned_event_ids"] = list(normalized_pinned_event_ids)
            bundle = build_context_bundle(
                request_payload,
                root=self.workspace_store.root,
                workspace_id=self.workspace_store.workspace_id,
            )
            bundle_path = self._ui_recall_bundle_path(task_kind=normalized_task_kind)
            write_json(bundle_path, dict(bundle))
            request_label = "manual"
            if normalized_pinned_event_ids:
                pin_compare = {
                    "before_bundle": before_bundle,
                    "after_bundle": bundle,
                }

        report = build_bundle_report(bundle, request_label=request_label)
        return {
            "selected_model_id": self.selected_model_id,
            "request_label": request_label,
            "request_payload": request_payload,
            "bundle": bundle,
            "pin_compare": pin_compare,
            "report": report,
            "pinned_event_ids": list(normalized_pinned_event_ids),
            "bundle_path": str(bundle_path) if bundle_path is not None else None,
            "dataset_path": str(dataset_path) if dataset_path is not None else None,
        }

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

        self.recall_task_kind = tk.StringVar(value="proposal")
        self.recall_query = tk.StringVar(value="")
        self.recall_request_basis = tk.StringVar(value="")
        self.recall_file_hint = tk.StringVar(value="")
        self.recall_limit = tk.IntVar(value=DEFAULT_LIMIT)
        self.recall_context_budget_chars = tk.IntVar(value=DEFAULT_CONTEXT_BUDGET_CHARS)
        self.recall_requests_state = tk.StringVar(
            value="Prepared recall requests are not loaded yet. Refresh real data when you want a current set."
        )
        self.recall_eval_state = tk.StringVar(
            value="Run evaluation to score source-hit recovery across the prepared requests."
        )
        self.recall_eval_misses_state = tk.StringVar(
            value="Evaluation misses appear here after the first evaluation run."
        )
        self.recall_eval_miss_selection_state = tk.StringVar(
            value="Select an evaluation miss to sync it with the prepared request list."
        )
        self.recall_eval_previous_var = tk.StringVar(
            value="Run evaluation once to capture the previous hit-quality snapshot."
        )
        self.recall_eval_current_var = tk.StringVar(
            value="Run evaluation to score source-hit recovery across the prepared requests."
        )
        self.recall_eval_change_var = tk.StringVar(
            value="Hit, miss, hit-rate, variant, and miss-reason changes appear here."
        )
        self.recall_diagnostic_diagnosis_var = tk.StringVar(
            value="Select an evaluation miss to see why it likely fell out of recall."
        )
        self.recall_diagnostic_action_var = tk.StringVar(
            value="The next actions for retrieval, ranking, or budget tuning appear here."
        )
        self.recall_diagnostic_manual_var = tk.StringVar(
            value="A suggested manual recall setup appears here."
        )
        self.recall_eval_winner_compare_state = tk.StringVar(
            value="Top winners appear here when the selected miss records winning candidates."
        )
        self.recall_eval_source_var = tk.StringVar(
            value="Source compare appears here when you select an evaluation miss."
        )
        self.recall_eval_source_why_var = tk.StringVar(
            value="Why appears here after you select an evaluation miss."
        )
        self.recall_eval_source_action_hint_var = tk.StringVar(
            value="Select a miss to open the source, copy it into manual recall, or rerun the suggested tweak."
        )
        self.recall_eval_winner_vars = [
            tk.StringVar(value=f"No winner recorded at #{index}.")
            for index in range(1, RECALL_TOP_SELECTED_COMPARE_LIMIT + 1)
        ]
        self.recall_eval_source_chip_vars = {
            key: tk.StringVar(value="")
            for key in RECALL_WINNER_CHIP_ORDER
        }
        self.recall_eval_winner_chip_vars = [
            {key: tk.StringVar(value="") for key in RECALL_WINNER_CHIP_ORDER}
            for _ in range(RECALL_TOP_SELECTED_COMPARE_LIMIT)
        ]
        self.recall_selection_state = tk.StringVar(
            value="Choose a prepared request or copy one into the manual form."
        )
        self.recall_candidates_state = tk.StringVar(
            value="Run recall to load selected candidates for jump-to-history."
        )
        self.recall_candidate_selection_state = tk.StringVar(
            value="Select a recall candidate to open it in History / Forensics."
        )
        self.recall_pins_state = tk.StringVar(
            value="Pinned candidates: none. Pin a candidate to carry it into the next recall run."
        )
        self.recall_compare_before_var = tk.StringVar(
            value="Pin a candidate to keep the unpinned bundle here as the baseline."
        )
        self.recall_compare_after_var = tk.StringVar(
            value="Run recall with one or more pins to capture the pinned bundle here."
        )
        self.recall_compare_change_var = tk.StringVar(
            value=(
                "Source rank, miss reason, selected candidates, budget, and source vs winners "
                "differences appear here."
            )
        )
        self.evaluation_state = tk.StringVar(
            value="Refresh the evaluation snapshot to load software-work signals."
        )
        self.evaluation_acceptance_var = tk.StringVar(
            value="No acceptance or rejection signals loaded yet."
        )
        self.evaluation_test_var = tk.StringVar(
            value="No test pass/fail signals loaded yet."
        )
        self.evaluation_repair_var = tk.StringVar(
            value="No repair or follow-up links loaded yet."
        )
        self.evaluation_comparison_var = tk.StringVar(
            value="No comparison records loaded yet."
        )
        self.evaluation_curation_var = tk.StringVar(
            value="No curation candidates loaded yet."
        )
        self.evaluation_adoption_var = tk.StringVar(
            value="No adoption preview loaded yet."
        )
        self.learning_candidate_review_var = tk.StringVar(
            value="No learning candidate artifacts loaded yet."
        )
        self.evaluation_curation_state_filter = tk.StringVar(value="all")
        self.evaluation_curation_decision_filter = tk.StringVar(value="all")
        self.evaluation_curation_reason_filter = tk.StringVar(value="")
        self.evaluation_review_id = tk.StringVar(value="")
        self.evaluation_review_url = tk.StringVar(value="")
        self.evaluation_review_summary = tk.StringVar(value="")
        self._recall_request_rows: dict[str, dict[str, Any]] = {}
        self._recall_selected_request_index: int | None = None
        self._recall_eval_miss_rows: dict[str, dict[str, Any]] = {}
        self._recall_selected_eval_miss_index: int | None = None
        self._recall_eval_source_row: dict[str, Any] | None = None
        self._recall_eval_winner_rows: list[dict[str, Any] | None] = [None] * RECALL_TOP_SELECTED_COMPARE_LIMIT
        self.recall_eval_source_chip_labels: dict[str, Any] = {}
        self.recall_eval_source_button: Any | None = None
        self.recall_eval_source_copy_button: Any | None = None
        self.recall_eval_source_rerun_button: Any | None = None
        self.recall_eval_winner_chip_labels: list[dict[str, Any]] = [{} for _ in range(RECALL_TOP_SELECTED_COMPARE_LIMIT)]
        self._evaluation_signal_rows: dict[str, dict[str, str]] = {}
        self._evaluation_curation_rows: dict[str, dict[str, str]] = {}
        self._learning_candidate_review_rows: dict[str, dict[str, str]] = {}
        self.recall_eval_winner_buttons: list[Any] = []
        self._recall_candidate_rows: dict[str, dict[str, Any]] = {}
        self._recall_selected_candidate_event_id: str | None = None
        self._recall_pinned_candidate_rows: dict[str, dict[str, Any]] = {}

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
        if default_dataset_path(
            workspace_id=self.controller.workspace_store.workspace_id,
            root=self.controller.workspace_store.root,
        ).exists():
            self.refresh_recall_requests(announce=False)
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

        self.notebook = ttk.Notebook(outer, style="Lab.TNotebook")
        self.notebook.grid(row=1, column=0, sticky="nsew")
        outer.grid_rowconfigure(1, weight=1)

        self.chat_tab = self._build_chat_tab(self.notebook)
        self.vision_tab = self._build_vision_tab(self.notebook)
        self.audio_tab = self._build_audio_tab(self.notebook)
        self.thinking_tab = self._build_thinking_tab(self.notebook)
        self.recall_tab = self._build_recall_tab(self.notebook)
        self.evaluation_tab = self._build_evaluation_tab(self.notebook)
        self.forensics_tab = self._build_forensics_tab(self.notebook)
        self.diagnostics_tab = self._build_diagnostics_tab(self.notebook)

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

    def _build_recall_tab(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, style="Card.TFrame", padding=18)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(25, weight=1)
        notebook.add(frame, text="Recall")

        ttk.Label(frame, text="Prepared Requests", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            frame,
            text=(
                "Prepared requests come from local workspace history plus capability artifacts. "
                "Refresh when you want the latest real-data set."
            ),
            style="Subtitle.TLabel",
            wraplength=920,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(
            frame,
            textvariable=self.recall_requests_state,
            style="Subtitle.TLabel",
            wraplength=920,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

        controls = ttk.Frame(frame, style="Card.TFrame")
        controls.grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))
        self.refresh_recall_button = ttk.Button(
            controls,
            text="Refresh Real Data",
            style="Lab.TButton",
            command=lambda: self.refresh_recall_requests(refresh_dataset=True),
        )
        self.refresh_recall_button.grid(row=0, column=0, sticky="w")
        self.run_selected_recall_button = ttk.Button(
            controls,
            text="Run Selected Request",
            style="Lab.TButton",
            command=self.run_selected_recall_request,
        )
        self.run_selected_recall_button.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.copy_recall_request_button = ttk.Button(
            controls,
            text="Copy Into Manual",
            style="Lab.TButton",
            command=self.copy_selected_recall_request,
        )
        self.copy_recall_request_button.grid(row=0, column=2, sticky="w", padx=(8, 0))

        self.recall_requests_tree = self._build_treeview(
            frame,
            row=4,
            columns=("index", "task", "hit", "status", "reason", "query"),
            headings=("Index", "Task", "Hit", "Status", "Reason", "Query"),
            columnspan=3,
            height=7,
        )
        self.recall_requests_tree.column("index", width=60, stretch=False)
        self.recall_requests_tree.column("task", width=120, stretch=False)
        self.recall_requests_tree.column("hit", width=70, stretch=False)
        self.recall_requests_tree.column("status", width=100, stretch=False)
        self.recall_requests_tree.column("reason", width=160, stretch=False)
        self.recall_requests_tree.column("query", width=480, stretch=True)
        self.recall_requests_tree.bind("<<TreeviewSelect>>", self._on_recall_request_selected)

        ttk.Label(frame, text="Evaluation Loop", style="Section.TLabel").grid(row=5, column=0, sticky="w", pady=(12, 0))
        ttk.Label(
            frame,
            textvariable=self.recall_eval_state,
            style="Subtitle.TLabel",
            wraplength=920,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(4, 0))
        eval_controls = ttk.Frame(frame, style="Card.TFrame")
        eval_controls.grid(row=7, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.evaluate_recall_button = ttk.Button(
            eval_controls,
            text="Evaluate Hits",
            style="Lab.TButton",
            command=self.run_recall_evaluation,
        )
        self.evaluate_recall_button.grid(row=0, column=0, sticky="w")
        self.refresh_and_evaluate_recall_button = ttk.Button(
            eval_controls,
            text="Refresh + Evaluate",
            style="Lab.TButton",
            command=lambda: self.run_recall_evaluation(refresh_dataset=True),
        )
        self.refresh_and_evaluate_recall_button.grid(row=0, column=1, sticky="w", padx=(8, 0))

        eval_compare = ttk.Frame(frame, style="Card.TFrame")
        eval_compare.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        for column in range(3):
            eval_compare.grid_columnconfigure(column, weight=1)
        self._build_compare_card(eval_compare, column=0, title="Previous", variable=self.recall_eval_previous_var)
        self._build_compare_card(eval_compare, column=1, title="Current", variable=self.recall_eval_current_var)
        self._build_compare_card(eval_compare, column=2, title="Diff", variable=self.recall_eval_change_var)

        ttk.Label(frame, text="Evaluation Misses", style="Section.TLabel").grid(row=9, column=0, sticky="w", pady=(12, 0))
        ttk.Label(
            frame,
            textvariable=self.recall_eval_misses_state,
            style="Subtitle.TLabel",
            wraplength=920,
        ).grid(row=10, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.recall_eval_misses_tree = self._build_treeview(
            frame,
            row=11,
            columns=("index", "task", "reason", "rank", "variant", "query"),
            headings=("Index", "Task", "Reason", "Rank", "Variant", "Query"),
            columnspan=3,
            height=5,
        )
        self.recall_eval_misses_tree.column("index", width=60, stretch=False)
        self.recall_eval_misses_tree.column("task", width=120, stretch=False)
        self.recall_eval_misses_tree.column("reason", width=180, stretch=False)
        self.recall_eval_misses_tree.column("rank", width=70, stretch=False)
        self.recall_eval_misses_tree.column("variant", width=170, stretch=False)
        self.recall_eval_misses_tree.column("query", width=360, stretch=True)
        self.recall_eval_misses_tree.bind("<<TreeviewSelect>>", self._on_recall_eval_miss_selected)

        eval_miss_controls = ttk.Frame(frame, style="Card.TFrame")
        eval_miss_controls.grid(row=12, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        eval_miss_controls.grid_columnconfigure(3, weight=1)
        self.copy_recall_miss_button = ttk.Button(
            eval_miss_controls,
            text="Copy Miss Into Manual",
            style="Lab.TButton",
            command=self.copy_selected_recall_eval_miss,
        )
        self.copy_recall_miss_button.grid(row=0, column=0, sticky="w")
        self.apply_recall_miss_tweak_button = ttk.Button(
            eval_miss_controls,
            text="Apply Suggested Tweak",
            style="Lab.TButton",
            command=self.apply_selected_recall_eval_miss_tweak,
        )
        self.apply_recall_miss_tweak_button.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.open_recall_miss_winner_button = ttk.Button(
            eval_miss_controls,
            text="Open Top Winner",
            style="Lab.TButton",
            command=self.open_selected_recall_eval_miss_winner,
        )
        self.open_recall_miss_winner_button.grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Label(
            eval_miss_controls,
            textvariable=self.recall_eval_miss_selection_state,
            style="Subtitle.TLabel",
            wraplength=660,
            justify="left",
        ).grid(row=0, column=3, sticky="w", padx=(12, 0))

        ttk.Label(frame, text="Diagnostic Guide", style="Section.TLabel").grid(row=13, column=0, sticky="w", pady=(12, 0))
        diagnostic_frame = ttk.Frame(frame, style="Card.TFrame")
        diagnostic_frame.grid(row=14, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        for column in range(3):
            diagnostic_frame.grid_columnconfigure(column, weight=1)
        self._build_compare_card(
            diagnostic_frame,
            column=0,
            title="Diagnosis",
            variable=self.recall_diagnostic_diagnosis_var,
        )
        self._build_compare_card(
            diagnostic_frame,
            column=1,
            title="Next",
            variable=self.recall_diagnostic_action_var,
        )
        self._build_compare_card(
            diagnostic_frame,
            column=2,
            title="Suggested Manual",
            variable=self.recall_diagnostic_manual_var,
        )
        ttk.Label(
            diagnostic_frame,
            textvariable=self.recall_eval_winner_compare_state,
            style="Subtitle.TLabel",
            wraplength=920,
            justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(10, 0))
        source_frame = ttk.Frame(diagnostic_frame, style="Card.TFrame")
        source_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        source_frame.grid_columnconfigure(0, weight=1)
        (
            self.recall_eval_source_button,
            self.recall_eval_source_copy_button,
            self.recall_eval_source_rerun_button,
        ) = self._build_recall_source_card(
            source_frame,
            variable=self.recall_eval_source_var,
            why_variable=self.recall_eval_source_why_var,
            action_hint_variable=self.recall_eval_source_action_hint_var,
            open_command=self.open_selected_recall_eval_source,
            copy_command=self.copy_selected_recall_eval_source_to_manual,
            rerun_command=self.rerun_selected_recall_eval_source,
        )
        winner_frame = ttk.Frame(diagnostic_frame, style="Card.TFrame")
        winner_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.recall_eval_winner_buttons = []
        for column in range(RECALL_TOP_SELECTED_COMPARE_LIMIT):
            winner_frame.grid_columnconfigure(column, weight=1)
            button = self._build_recall_winner_card(
                winner_frame,
                column=column,
                title=f"Winner {column + 1}",
                variable=self.recall_eval_winner_vars[column],
                open_command=lambda index=column: self.open_recall_eval_winner_at(index),
            )
            self.recall_eval_winner_buttons.append(button)

        ttk.Label(frame, text="Manual Recall", style="Section.TLabel").grid(row=15, column=0, sticky="w", pady=(12, 0))
        manual = ttk.Frame(frame, style="Card.TFrame")
        manual.grid(row=16, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        manual.grid_columnconfigure(1, weight=1)
        ttk.Label(manual, text="Task Kind", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            manual,
            textvariable=self.recall_task_kind,
            values=list(TASK_KINDS),
            width=18,
            state="readonly",
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(manual, text="Query", style="Section.TLabel").grid(row=1, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(manual, textvariable=self.recall_query).grid(row=1, column=1, sticky="ew", pady=(12, 0))
        ttk.Label(manual, text="Request Basis", style="Section.TLabel").grid(row=2, column=0, sticky="w", pady=(12, 0))
        ttk.Combobox(
            manual,
            textvariable=self.recall_request_basis,
            values=("", "prompt-or-artifact", "pass_definition"),
            width=22,
            state="readonly",
        ).grid(row=2, column=1, sticky="w", pady=(12, 0))
        ttk.Label(manual, text="File Hint", style="Section.TLabel").grid(row=3, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(manual, textvariable=self.recall_file_hint).grid(row=3, column=1, sticky="ew", pady=(12, 0))
        tuning = ttk.Frame(manual, style="Card.TFrame")
        tuning.grid(row=4, column=0, columnspan=2, sticky="w", pady=(12, 0))
        ttk.Label(tuning, text="Limit", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            tuning,
            from_=RECALL_MANUAL_LIMIT_MIN,
            to=RECALL_MANUAL_LIMIT_MAX,
            textvariable=self.recall_limit,
            width=6,
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(tuning, text="Context Budget", style="Section.TLabel").grid(row=0, column=2, sticky="w", padx=(18, 0))
        ttk.Spinbox(
            tuning,
            from_=RECALL_MANUAL_CONTEXT_BUDGET_MIN,
            to=RECALL_MANUAL_CONTEXT_BUDGET_MAX,
            increment=500,
            textvariable=self.recall_context_budget_chars,
            width=8,
        ).grid(row=0, column=3, sticky="w", padx=(8, 0))
        self.run_manual_recall_button = ttk.Button(
            manual,
            text="Run Manual Recall",
            style="Lab.TButton",
            command=self.run_manual_recall,
        )
        self.run_manual_recall_button.grid(row=5, column=0, sticky="w", pady=(14, 0))
        ttk.Label(
            manual,
            textvariable=self.recall_selection_state,
            style="Subtitle.TLabel",
            wraplength=760,
            justify="left",
        ).grid(row=5, column=1, sticky="w", pady=(14, 0), padx=(12, 0))

        ttk.Label(frame, text="Selected Candidates", style="Section.TLabel").grid(row=17, column=0, sticky="w", pady=(12, 0))
        ttk.Label(
            frame,
            textvariable=self.recall_candidates_state,
            style="Subtitle.TLabel",
            wraplength=920,
        ).grid(row=18, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.recall_candidates_tree = self._build_treeview(
            frame,
            row=19,
            columns=("block", "status", "surface", "recorded", "prompt"),
            headings=("Block", "Status", "Surface", "Recorded", "Prompt"),
            columnspan=3,
            height=5,
        )
        self.recall_candidates_tree.column("block", width=180, stretch=False)
        self.recall_candidates_tree.column("status", width=90, stretch=False)
        self.recall_candidates_tree.column("surface", width=90, stretch=False)
        self.recall_candidates_tree.column("recorded", width=170, stretch=False)
        self.recall_candidates_tree.column("prompt", width=520, stretch=True)
        self.recall_candidates_tree.bind("<<TreeviewSelect>>", self._on_recall_candidate_selected)

        candidate_controls = ttk.Frame(frame, style="Card.TFrame")
        candidate_controls.grid(row=20, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        candidate_controls.grid_columnconfigure(3, weight=1)
        self.open_recall_candidate_button = ttk.Button(
            candidate_controls,
            text="Open in History / Forensics",
            style="Lab.TButton",
            command=self.open_selected_recall_candidate,
        )
        self.open_recall_candidate_button.grid(row=0, column=0, sticky="w")
        self.pin_recall_candidate_button = ttk.Button(
            candidate_controls,
            text="Pin Candidate",
            style="Lab.TButton",
            command=self.pin_selected_recall_candidate,
        )
        self.pin_recall_candidate_button.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.clear_recall_pins_button = ttk.Button(
            candidate_controls,
            text="Clear Pins",
            style="Lab.TButton",
            command=self.clear_recall_pins,
        )
        self.clear_recall_pins_button.grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Label(
            candidate_controls,
            textvariable=self.recall_candidate_selection_state,
            style="Subtitle.TLabel",
            wraplength=760,
            justify="left",
        ).grid(row=0, column=3, sticky="w", padx=(12, 0))

        ttk.Label(
            frame,
            textvariable=self.recall_pins_state,
            style="Subtitle.TLabel",
            wraplength=920,
            justify="left",
        ).grid(row=21, column=0, columnspan=3, sticky="w", pady=(8, 0))

        ttk.Label(frame, text="Pin Compare", style="Section.TLabel").grid(row=22, column=0, sticky="w", pady=(12, 0))
        compare_frame = ttk.Frame(frame, style="Card.TFrame")
        compare_frame.grid(row=23, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        for column in range(3):
            compare_frame.grid_columnconfigure(column, weight=1)
        self._build_compare_card(compare_frame, column=0, title="No Pin", variable=self.recall_compare_before_var)
        self._build_compare_card(compare_frame, column=1, title="Pinned", variable=self.recall_compare_after_var)
        self._build_compare_card(compare_frame, column=2, title="Diff", variable=self.recall_compare_change_var)

        ttk.Label(frame, text="Recall Output", style="Section.TLabel").grid(row=24, column=0, sticky="w", pady=(12, 0))
        self.recall_output = self._readonly_text(frame, row=25, columnspan=3, height=12)
        return frame

    def _build_evaluation_tab(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, style="Card.TFrame", padding=18)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(6, weight=1)
        frame.grid_rowconfigure(9, weight=1)
        frame.grid_rowconfigure(11, weight=1)
        frame.grid_rowconfigure(13, weight=1)
        frame.grid_rowconfigure(15, weight=1)
        notebook.add(frame, text="Evaluation")

        header = ttk.Frame(frame, style="Card.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        ttk.Label(header, text="Snapshot", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.refresh_evaluation_button = ttk.Button(
            header,
            text="Refresh Snapshot",
            style="Lab.TButton",
            command=self.refresh_evaluation_snapshot,
        )
        self.refresh_evaluation_button.grid(row=0, column=1, sticky="e")
        ttk.Label(
            frame,
            textvariable=self.evaluation_state,
            style="Subtitle.TLabel",
            wraplength=920,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

        metric_frame = ttk.Frame(frame, style="Card.TFrame")
        metric_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        for column in range(3):
            metric_frame.grid_columnconfigure(column, weight=1)
        self._build_compare_card(metric_frame, column=0, title="Accept / Reject", variable=self.evaluation_acceptance_var)
        self._build_compare_card(metric_frame, column=1, title="Tests", variable=self.evaluation_test_var)
        self._build_compare_card(metric_frame, column=2, title="Repair", variable=self.evaluation_repair_var)

        second_metric_frame = ttk.Frame(frame, style="Card.TFrame")
        second_metric_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        for column in range(3):
            second_metric_frame.grid_columnconfigure(column, weight=1)
        self._build_compare_card(second_metric_frame, column=0, title="Comparison", variable=self.evaluation_comparison_var)
        self._build_compare_card(second_metric_frame, column=1, title="Curation", variable=self.evaluation_curation_var)
        self._build_compare_card(second_metric_frame, column=2, title="Adoption", variable=self.evaluation_adoption_var)

        filter_frame = ttk.Frame(frame, style="Card.TFrame")
        filter_frame.grid(row=4, column=0, sticky="ew", pady=(14, 0))
        filter_frame.grid_columnconfigure(3, weight=1)
        ttk.Label(filter_frame, text="Curation Filter", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        state_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.evaluation_curation_state_filter,
            values=["all", *CURATION_STATES],
            width=16,
            state="readonly",
        )
        state_combo.grid(row=0, column=1, sticky="w", padx=(10, 0))
        decision_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.evaluation_curation_decision_filter,
            values=["all", *CURATION_EXPORT_DECISIONS],
            width=24,
            state="readonly",
        )
        decision_combo.grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Entry(
            filter_frame,
            textvariable=self.evaluation_curation_reason_filter,
            width=28,
        ).grid(row=0, column=3, sticky="w", padx=(8, 0))
        self.apply_evaluation_filter_button = ttk.Button(
            filter_frame,
            text="Apply Filter",
            style="Lab.TButton",
            command=self.refresh_evaluation_snapshot,
        )
        self.apply_evaluation_filter_button.grid(row=0, column=4, sticky="w", padx=(8, 0))

        ttk.Label(frame, text="Curation Preview", style="Section.TLabel").grid(row=5, column=0, sticky="w", pady=(14, 0))
        self.evaluation_curation_tree = self._build_treeview(
            frame,
            row=6,
            columns=("state", "decision", "next", "label"),
            headings=("State", "Decision", "Next", "Label"),
            height=5,
        )
        self.evaluation_curation_tree.column("state", width=110, stretch=False)
        self.evaluation_curation_tree.column("decision", width=180, stretch=False)
        self.evaluation_curation_tree.column("next", width=220, stretch=False)
        self.evaluation_curation_tree.column("label", width=540, stretch=True)

        review_frame = ttk.Frame(frame, style="Card.TFrame")
        review_frame.grid(row=7, column=0, sticky="ew", pady=(10, 0))
        review_frame.grid_columnconfigure(2, weight=1)
        ttk.Entry(review_frame, textvariable=self.evaluation_review_id, width=18).grid(row=0, column=0, sticky="w")
        ttk.Entry(review_frame, textvariable=self.evaluation_review_url, width=28).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Entry(review_frame, textvariable=self.evaluation_review_summary, width=52).grid(row=0, column=2, sticky="ew", padx=(8, 0))
        self.mark_review_resolved_button = ttk.Button(
            review_frame,
            text="Mark Resolved",
            style="Lab.TButton",
            command=self.mark_selected_evaluation_review_resolved,
        )
        self.mark_review_resolved_button.grid(row=0, column=4, sticky="e", padx=(8, 0))
        self.mark_review_unresolved_button = ttk.Button(
            review_frame,
            text="Mark Unresolved",
            style="Lab.TButton",
            command=self.mark_selected_evaluation_review_unresolved,
        )
        self.mark_review_unresolved_button.grid(row=0, column=5, sticky="e", padx=(8, 0))

        learning_header = ttk.Frame(frame, style="Card.TFrame")
        learning_header.grid(row=8, column=0, sticky="ew", pady=(14, 0))
        learning_header.grid_columnconfigure(1, weight=1)
        ttk.Label(learning_header, text="Learning Candidate Review", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            learning_header,
            textvariable=self.learning_candidate_review_var,
            style="Subtitle.TLabel",
            wraplength=760,
        ).grid(row=0, column=1, sticky="w", padx=(12, 0))
        self.refresh_candidate_review_button = ttk.Button(
            learning_header,
            text="Refresh Review",
            style="Lab.TButton",
            command=self.refresh_learning_candidate_review,
        )
        self.refresh_candidate_review_button.grid(row=0, column=2, sticky="e")
        self.learning_candidate_review_tree = self._build_treeview(
            frame,
            row=9,
            columns=("artifact", "event", "state", "lifecycle", "blocked", "next", "policy", "role", "backend", "source"),
            headings=("Artifact", "Event", "State", "Lifecycle", "Blocked", "Next", "Policy", "Role", "Backend", "Source Path"),
            height=5,
        )
        self.learning_candidate_review_tree.column("artifact", width=130, stretch=False)
        self.learning_candidate_review_tree.column("event", width=180, stretch=False)
        self.learning_candidate_review_tree.column("state", width=120, stretch=False)
        self.learning_candidate_review_tree.column("lifecycle", width=110, stretch=False)
        self.learning_candidate_review_tree.column("blocked", width=160, stretch=False)
        self.learning_candidate_review_tree.column("next", width=210, stretch=False)
        self.learning_candidate_review_tree.column("policy", width=150, stretch=False)
        self.learning_candidate_review_tree.column("role", width=110, stretch=False)
        self.learning_candidate_review_tree.column("backend", width=150, stretch=False)
        self.learning_candidate_review_tree.column("source", width=320, stretch=True)

        ttk.Label(frame, text="Signals", style="Section.TLabel").grid(row=10, column=0, sticky="w", pady=(14, 0))
        self.evaluation_signals_tree = self._build_treeview(
            frame,
            row=11,
            columns=("kind", "source", "relation", "status", "label"),
            headings=("Kind", "Source", "Relation", "Status", "Label"),
            height=7,
        )
        self.evaluation_signals_tree.column("kind", width=110, stretch=False)
        self.evaluation_signals_tree.column("source", width=240, stretch=False)
        self.evaluation_signals_tree.column("relation", width=180, stretch=False)
        self.evaluation_signals_tree.column("status", width=110, stretch=False)
        self.evaluation_signals_tree.column("label", width=420, stretch=True)

        ttk.Label(frame, text="Details", style="Section.TLabel").grid(row=12, column=0, sticky="w", pady=(14, 0))
        self.evaluation_detail_output = self._readonly_text(frame, row=13, height=8)

        ttk.Label(frame, text="Report", style="Section.TLabel").grid(row=14, column=0, sticky="w", pady=(14, 0))
        self.evaluation_output = self._readonly_text(frame, row=15, height=10)
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
            elif column == "query":
                width = 420
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

    def _build_recall_reason_chip_labels(
        self,
        parent: tk.Widget | Any,
        *,
        chip_vars: Mapping[str, Any],
    ) -> dict[str, Any]:
        chip_labels: dict[str, Any] = {}
        for key in RECALL_WINNER_CHIP_ORDER:
            colors = RECALL_WINNER_CHIP_COLORS[key]
            label = tk.Label(
                parent,
                textvariable=chip_vars[key],
                background=colors["background"],
                foreground=colors["foreground"],
                font=("SF Pro Text", 10),
                padx=7,
                pady=3,
                bd=0,
                justify="left",
                wraplength=220,
            )
            chip_labels[key] = label
        return chip_labels

    def _apply_recall_reason_chip_texts(
        self,
        *,
        chip_vars: Mapping[str, Any],
        chip_labels: Mapping[str, Any],
        chip_texts: Mapping[str, str],
    ) -> None:
        for key in RECALL_WINNER_CHIP_ORDER:
            text = str(chip_texts.get(key) or "")
            chip_vars[key].set(text)

        for label in chip_labels.values():
            if hasattr(label, "pack_forget"):
                label.pack_forget()

        for key in RECALL_WINNER_CHIP_ORDER:
            text = str(chip_texts.get(key) or "")
            label = chip_labels.get(key)
            if not text or label is None or not hasattr(label, "pack"):
                continue
            label.pack(side="top", anchor="w", pady=(0, 4))

    def _build_recall_source_card(
        self,
        parent: ttk.Frame,
        *,
        variable: tk.StringVar,
        why_variable: tk.StringVar,
        action_hint_variable: tk.StringVar,
        open_command: Any,
        copy_command: Any,
        rerun_command: Any,
    ) -> tuple[ttk.Button, ttk.Button, ttk.Button]:
        card = ttk.Frame(parent, style="MetricCard.TFrame", padding=(12, 10))
        card.grid(row=0, column=0, sticky="ew")
        ttk.Label(card, text="Source", style="MetricTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            card,
            textvariable=variable,
            style="MetricBody.TLabel",
            wraplength=860,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        why_frame = tk.Frame(card, background="#FFFFFF", highlightthickness=0, borderwidth=0)
        why_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        why_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(why_frame, text="Why", style="MetricTitle.TLabel").grid(row=0, column=0, sticky="nw", padx=(0, 8))
        ttk.Label(
            why_frame,
            textvariable=why_variable,
            style="MetricBody.TLabel",
            wraplength=760,
            justify="left",
        ).grid(row=0, column=1, sticky="w")
        chip_frame = tk.Frame(card, background="#FFFFFF", highlightthickness=0, borderwidth=0)
        chip_frame.grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.recall_eval_source_chip_labels = self._build_recall_reason_chip_labels(
            chip_frame,
            chip_vars=self.recall_eval_source_chip_vars,
        )
        action_frame = tk.Frame(card, background="#FFFFFF", highlightthickness=0, borderwidth=0)
        action_frame.grid(row=4, column=0, sticky="w", pady=(10, 0))
        open_button = ttk.Button(action_frame, text="Open", style="Lab.TButton", command=open_command)
        open_button.grid(row=0, column=0, sticky="w")
        copy_button = ttk.Button(
            action_frame,
            text="Copy to Manual",
            style="Lab.TButton",
            command=copy_command,
        )
        copy_button.grid(row=0, column=1, sticky="w", padx=(8, 0))
        rerun_button = ttk.Button(
            action_frame,
            text="Rerun",
            style="Lab.TButton",
            command=rerun_command,
        )
        rerun_button.grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Label(
            action_frame,
            textvariable=action_hint_variable,
            style="MetricBody.TLabel",
            wraplength=780,
            justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))
        return open_button, copy_button, rerun_button

    def _build_recall_winner_card(
        self,
        parent: ttk.Frame,
        *,
        column: int,
        title: str,
        variable: tk.StringVar,
        open_command: Any,
    ) -> ttk.Button:
        card = ttk.Frame(parent, style="MetricCard.TFrame", padding=(12, 10))
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0))
        ttk.Label(card, text=title, style="MetricTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            card,
            textvariable=variable,
            style="MetricBody.TLabel",
            wraplength=250,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        chip_frame = tk.Frame(card, background="#FFFFFF", highlightthickness=0, borderwidth=0)
        chip_frame.grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.recall_eval_winner_chip_labels[column] = self._build_recall_reason_chip_labels(
            chip_frame,
            chip_vars=self.recall_eval_winner_chip_vars[column],
        )
        button = ttk.Button(card, text="Open", style="Lab.TButton", command=open_command)
        button.grid(row=3, column=0, sticky="w", pady=(10, 0))
        return button

    def _apply_recall_eval_source_chip_texts(
        self,
        chip_texts: Mapping[str, str],
    ) -> None:
        self._apply_recall_reason_chip_texts(
            chip_vars=self.recall_eval_source_chip_vars,
            chip_labels=self.recall_eval_source_chip_labels,
            chip_texts=chip_texts,
        )

    def _apply_recall_eval_winner_chip_texts(
        self,
        winner_index: int,
        chip_texts: Mapping[str, str],
    ) -> None:
        if winner_index < 0 or winner_index >= len(self.recall_eval_winner_chip_vars):
            return

        chip_vars = self.recall_eval_winner_chip_vars[winner_index]
        chip_labels = (
            self.recall_eval_winner_chip_labels[winner_index]
            if winner_index < len(self.recall_eval_winner_chip_labels)
            else {}
        )
        self._apply_recall_reason_chip_texts(
            chip_vars=chip_vars,
            chip_labels=chip_labels,
            chip_texts=chip_texts,
        )

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

    def _on_recall_request_selected(self, _event: object | None = None) -> None:
        selection = self.recall_requests_tree.selection()
        if not selection:
            return
        row = self._recall_request_rows.get(str(selection[0]))
        if row is None:
            return
        try:
            self._recall_selected_request_index = int(row.get("index"))
        except (TypeError, ValueError):
            self._recall_selected_request_index = None
        self.recall_selection_state.set(build_recall_request_summary(row))

    def _populate_recall_requests(self, snapshot: dict[str, Any]) -> None:
        requests = list(snapshot.get("requests") or [])
        request_entries = list(snapshot.get("request_entries") or [])
        dataset_path = snapshot.get("dataset_path")
        request_count = int(snapshot.get("request_count") or 0)
        generated_at = snapshot.get("generated_at_utc") or "n/a"

        children = self.recall_requests_tree.get_children()
        if children:
            self.recall_requests_tree.delete(*children)

        self._recall_request_rows = {}
        if not requests:
            self._recall_selected_request_index = None
            self.recall_requests_state.set("No prepared recall requests are available yet.")
            self.recall_selection_state.set("Refresh real data to generate the first request set.")
            return

        dataset_display = (
            summarize_workspace_path(str(dataset_path), root=self.controller.workspace_store.root)
            if dataset_path
            else "n/a"
        )
        self.recall_requests_state.set(
            f"{request_count} prepared requests from {dataset_display}. Generated {generated_at}."
        )
        selected_index = self._recall_selected_request_index
        for position, row in enumerate(requests):
            raw_row = request_entries[position] if position < len(request_entries) else {}
            merged_row = {
                **(dict(raw_row) if isinstance(raw_row, Mapping) else {}),
                **(dict(row) if isinstance(row, Mapping) else {}),
            }
            index = str(merged_row.get("index") or position + 1)
            self._recall_request_rows[index] = merged_row
            self.recall_requests_tree.insert(
                "",
                "end",
                iid=index,
                values=(
                    index,
                    merged_row.get("task_kind") or "n/a",
                    "yes" if merged_row.get("source_hit") else "no",
                    merged_row.get("source_status") or "-",
                    (
                        merged_row.get("miss_reason")
                        or (
                            merged_row.get("request_variant")
                            if merged_row.get("request_variant") not in {"", BASELINE_REQUEST_VARIANT, None}
                            else "-"
                        )
                    ),
                    summarize_text_preview(str(merged_row.get("query_text") or ""), limit=110),
                ),
            )

        if selected_index is None:
            selected_index = int(requests[0].get("index") or 1)
        selected_iid = str(selected_index)
        if self.recall_requests_tree.exists(selected_iid):
            self.recall_requests_tree.selection_set(selected_iid)
            self.recall_requests_tree.focus(selected_iid)
            self._on_recall_request_selected()

    def _reset_recall_evaluation_view(self) -> None:
        self.recall_eval_state.set(
            "Run evaluation to score source-hit recovery across the prepared requests."
        )
        self.recall_eval_misses_state.set(
            "Evaluation misses appear here after the first evaluation run."
        )
        self.recall_eval_miss_selection_state.set(
            "Select an evaluation miss to sync it with the prepared request list."
        )
        apply_recall_eval_compare_fields(
            None,
            previous_var=self.recall_eval_previous_var,
            current_var=self.recall_eval_current_var,
            change_var=self.recall_eval_change_var,
        )
        apply_recall_diagnostic_guide_fields(
            None,
            request_row=None,
            diagnosis_var=self.recall_diagnostic_diagnosis_var,
            action_var=self.recall_diagnostic_action_var,
            manual_var=self.recall_diagnostic_manual_var,
        )
        self._apply_recall_eval_winner_compare(None)
        self._recall_eval_miss_rows = {}
        self._recall_selected_eval_miss_index = None
        children = self.recall_eval_misses_tree.get_children()
        if children:
            self.recall_eval_misses_tree.delete(*children)

    def _populate_recall_evaluation(self, snapshot: Mapping[str, Any]) -> None:
        summary = snapshot.get("summary")
        dataset_path = snapshot.get("dataset_path")
        self.recall_eval_state.set(
            build_recall_eval_state(
                summary if isinstance(summary, Mapping) else None,
                dataset_path=str(dataset_path) if dataset_path else None,
                root=self.controller.workspace_store.root,
            )
        )
        apply_recall_eval_compare_fields(
            snapshot.get("comparison"),
            previous_var=self.recall_eval_previous_var,
            current_var=self.recall_eval_current_var,
            change_var=self.recall_eval_change_var,
        )

        misses = [
            dict(item)
            for item in snapshot.get("misses") or []
            if isinstance(item, Mapping)
        ]
        children = self.recall_eval_misses_tree.get_children()
        if children:
            self.recall_eval_misses_tree.delete(*children)
        self._recall_eval_miss_rows = {}
        if not misses:
            self._recall_selected_eval_miss_index = None
            self.recall_eval_misses_state.set("No source misses in the latest evaluation.")
            self.recall_eval_miss_selection_state.set(
                "Select an evaluation miss to sync it with the prepared request list."
            )
            apply_recall_diagnostic_guide_fields(
                None,
                request_row=None,
                diagnosis_var=self.recall_diagnostic_diagnosis_var,
                action_var=self.recall_diagnostic_action_var,
                manual_var=self.recall_diagnostic_manual_var,
            )
            self._apply_recall_eval_winner_compare(None)
            return

        self.recall_eval_misses_state.set(
            f"{len(misses)} source miss(es). Select one to sync it with the prepared request list, compare its winners, or copy it into manual recall."
        )
        selected_index = self._recall_selected_eval_miss_index
        for position, row in enumerate(misses):
            index = str(row.get("index") or position + 1)
            self._recall_eval_miss_rows[index] = row
            request_variant = str(row.get("request_variant") or "").strip() or BASELINE_REQUEST_VARIANT
            self.recall_eval_misses_tree.insert(
                "",
                "end",
                iid=index,
                values=(
                    index,
                    row.get("task_kind") or "n/a",
                    row.get("miss_reason") or "unknown",
                    row.get("source_rank") if row.get("source_rank") is not None else "-",
                    request_variant,
                    summarize_text_preview(str(row.get("query_text") or ""), limit=96),
                ),
            )

        if selected_index is None:
            selected_index = int(misses[0].get("index") or 1)
        selected_iid = str(selected_index)
        if self.recall_eval_misses_tree.exists(selected_iid):
            self.recall_eval_misses_tree.selection_set(selected_iid)
            self.recall_eval_misses_tree.focus(selected_iid)
            self._on_recall_eval_miss_selected()

    def _on_recall_eval_miss_selected(self, _event: object | None = None) -> None:
        selection = self.recall_eval_misses_tree.selection()
        if not selection:
            return
        selected_index = str(selection[0])
        row = self._recall_eval_miss_rows.get(selected_index)
        if row is None:
            return
        try:
            self._recall_selected_eval_miss_index = int(row.get("index"))
        except (TypeError, ValueError):
            self._recall_selected_eval_miss_index = None
        self.recall_eval_miss_selection_state.set(build_recall_eval_miss_summary(row))
        request_index = str(row.get("index") or "").strip()
        request_row = self._recall_request_rows.get(request_index)
        if request_index and self.recall_requests_tree.exists(request_index):
            self.recall_requests_tree.selection_set(request_index)
            self.recall_requests_tree.focus(request_index)
            self._on_recall_request_selected()
        apply_recall_diagnostic_guide_fields(
            row,
            request_row=request_row,
            diagnosis_var=self.recall_diagnostic_diagnosis_var,
            action_var=self.recall_diagnostic_action_var,
            manual_var=self.recall_diagnostic_manual_var,
        )
        self._apply_recall_eval_winner_compare(row)

    def _on_recall_candidate_selected(self, _event: object | None = None) -> None:
        selection = self.recall_candidates_tree.selection()
        if not selection:
            return
        event_id = str(selection[0])
        row = self._recall_candidate_rows.get(event_id)
        if row is None:
            return
        self._recall_selected_candidate_event_id = event_id
        self.recall_candidate_selection_state.set(
            build_recall_candidate_summary(
                row,
                root=self.controller.workspace_store.root,
            )
        )

    def _current_recall_pinned_event_ids(self) -> list[str]:
        return list(self._recall_pinned_candidate_rows.keys())

    def _refresh_recall_pin_state(self) -> None:
        pinned_event_ids = set(self._current_recall_pinned_event_ids())
        for event_id, row in self._recall_candidate_rows.items():
            row["pinned"] = event_id in pinned_event_ids
        self.recall_pins_state.set(
            build_recall_pins_summary(self._recall_pinned_candidate_rows)
        )
        if self._recall_candidate_rows:
            message = (
                f"{len(self._recall_candidate_rows)} selected candidates are ready for History / Forensics jump."
            )
            if pinned_event_ids:
                message += f" {len(pinned_event_ids)} pinned for the next recall run."
            self.recall_candidates_state.set(message)
        if (
            self._recall_selected_candidate_event_id is not None
            and self._recall_selected_candidate_event_id in self._recall_candidate_rows
        ):
            self.recall_candidate_selection_state.set(
                build_recall_candidate_summary(
                    self._recall_candidate_rows[self._recall_selected_candidate_event_id],
                    root=self.controller.workspace_store.root,
                )
            )

    def _populate_recall_candidates(self, bundle: Mapping[str, Any] | None) -> None:
        selected = [
            dict(item)
            for item in (bundle or {}).get("selected_candidates") or []
            if isinstance(item, Mapping)
        ]
        children = self.recall_candidates_tree.get_children()
        if children:
            self.recall_candidates_tree.delete(*children)

        self._recall_candidate_rows = {}
        if not selected:
            self._recall_selected_candidate_event_id = None
            self.recall_candidates_state.set("This recall run did not select any candidates.")
            self.recall_candidate_selection_state.set(
                "Select a recall candidate to open it in History / Forensics."
            )
            self._refresh_recall_pin_state()
            return

        pinned_event_ids = set(self._current_recall_pinned_event_ids())
        selected_event_id = self._recall_selected_candidate_event_id
        for index, row in enumerate(selected, start=1):
            event_id = str(row.get("event_id") or f"candidate-{index}")
            row["pinned"] = event_id in pinned_event_ids
            self._recall_candidate_rows[event_id] = row
            self.recall_candidates_tree.insert(
                "",
                "end",
                iid=event_id,
                values=(
                    row.get("block_title") or "n/a",
                    row.get("status") or "-",
                    row.get("session_surface") or "n/a",
                    row.get("recorded_at_utc") or "n/a",
                    summarize_text_preview(str(row.get("prompt_excerpt") or ""), limit=110),
                ),
            )

        if selected_event_id is None:
            selected_event_id = str(selected[0].get("event_id") or "candidate-1")
        if self.recall_candidates_tree.exists(selected_event_id):
            self.recall_candidates_tree.selection_set(selected_event_id)
            self.recall_candidates_tree.focus(selected_event_id)
            self._on_recall_candidate_selected()
        self._refresh_recall_pin_state()

    def pin_selected_recall_candidate(self) -> None:
        if self._recall_selected_candidate_event_id is None:
            self.status_var.set("Select a recall candidate first.")
            return

        row = self._recall_candidate_rows.get(self._recall_selected_candidate_event_id)
        if row is None:
            self.status_var.set("The selected recall candidate is no longer available.")
            return

        event_id = str(row.get("event_id") or "").strip()
        if not event_id:
            self.status_var.set("The selected recall candidate does not expose an event id.")
            return
        if event_id in self._recall_pinned_candidate_rows:
            self.status_var.set("This recall candidate is already pinned.")
            return

        self._recall_pinned_candidate_rows[event_id] = dict(row)
        self._refresh_recall_pin_state()
        self.status_var.set(f"Pinned recall candidate {event_id}.")
        self.hint_var.set("Pinned candidates are injected into the next recall run until you clear them.")

    def clear_recall_pins(self) -> None:
        if not self._recall_pinned_candidate_rows:
            self.status_var.set("There are no pinned recall candidates to clear.")
            return
        cleared_count = len(self._recall_pinned_candidate_rows)
        self._recall_pinned_candidate_rows = {}
        self._refresh_recall_pin_state()
        self.status_var.set(f"Cleared {cleared_count} pinned recall candidate(s).")
        self.hint_var.set("The next recall run will use normal ranking unless you pin another candidate.")

    def refresh_recall_requests(
        self,
        *,
        refresh_dataset: bool = False,
        announce: bool = True,
    ) -> None:
        if announce and self.job_runner.has_pending_work():
            self.status_var.set("Recall refresh is disabled while a worker job is running.")
            return

        if announce:
            self._cancel_pending_startup_prewarm()
            self._set_busy("Loading recall requests...")
        try:
            snapshot = self.controller.collect_recall_requests(refresh_dataset=refresh_dataset)
            self._populate_recall_requests(snapshot)
            self._reset_recall_evaluation_view()
            if announce:
                self.status_var.set("Recall requests refreshed." if refresh_dataset else "Recall requests loaded.")
                self.backend_var.set("Backend: local-memory-recall")
                self.device_var.set("Device: local files / SQLite")
                self.artifact_var.set(f"Recall dataset: {snapshot.get('dataset_path') or 'n/a'}")
                self.hint_var.set(
                    "Run a prepared request directly, or copy one into the manual form and tighten it."
                )
        except Exception as exc:
            self._recall_selected_request_index = None
            self._recall_request_rows = {}
            children = self.recall_requests_tree.get_children()
            if children:
                self.recall_requests_tree.delete(*children)
            self.recall_requests_state.set(f"Recall requests failed to load: {type(exc).__name__}: {exc}")
            self.recall_selection_state.set("Refresh real data after the dataset path is healthy again.")
            self._reset_recall_evaluation_view()
            if announce:
                self.status_var.set(f"Recall refresh failed: {type(exc).__name__}: {exc}")
        finally:
            if announce:
                self._clear_busy()
                self._resume_startup_prewarm_if_needed()

    def copy_selected_recall_request(self) -> None:
        if self._recall_selected_request_index is None:
            self.status_var.set("Select a prepared recall request first.")
            return
        row = self._recall_request_rows.get(str(self._recall_selected_request_index))
        if row is None:
            self.status_var.set("The selected recall request is no longer available.")
            return
        self.recall_task_kind.set(str(row.get("task_kind") or "proposal"))
        self.recall_query.set(str(row.get("query_text") or ""))
        self.recall_request_basis.set(str(row.get("request_basis") or ""))
        file_hints = list(row.get("file_hints") or [])
        self.recall_file_hint.set(str(file_hints[0]) if file_hints else "")
        self.recall_limit.set(
            coerce_ui_int(
                row.get("limit"),
                default=DEFAULT_LIMIT,
                minimum=RECALL_MANUAL_LIMIT_MIN,
                maximum=RECALL_MANUAL_LIMIT_MAX,
            )
        )
        self.recall_context_budget_chars.set(
            coerce_ui_int(
                row.get("context_budget_chars"),
                default=DEFAULT_CONTEXT_BUDGET_CHARS,
                minimum=RECALL_MANUAL_CONTEXT_BUDGET_MIN,
                maximum=RECALL_MANUAL_CONTEXT_BUDGET_MAX,
            )
        )
        self.recall_selection_state.set(
            f"Copied prepared request {self._recall_selected_request_index} into the manual form."
        )
        self.status_var.set(
            f"Prepared recall request {self._recall_selected_request_index} copied into manual recall."
        )

    def _select_recall_request_for_selected_eval_miss(self) -> bool:
        if self._recall_selected_eval_miss_index is None:
            self.status_var.set("Select an evaluation miss first.")
            return False
        request_index = str(self._recall_selected_eval_miss_index)
        request_row = self._recall_request_rows.get(request_index)
        if request_row is None:
            self.status_var.set("The matching prepared recall request is no longer available.")
            self.hint_var.set("Refresh real data to restore the request before copying it into manual recall.")
            return False
        try:
            self._recall_selected_request_index = int(request_row.get("index") or request_index)
        except (TypeError, ValueError):
            self._recall_selected_request_index = None
        if request_index and self.recall_requests_tree.exists(request_index):
            self.recall_requests_tree.selection_set(request_index)
            self.recall_requests_tree.focus(request_index)
            self._on_recall_request_selected()
        return self._recall_selected_request_index is not None

    def copy_selected_recall_eval_miss(self) -> None:
        if not self._select_recall_request_for_selected_eval_miss():
            return
        self.copy_selected_recall_request()
        if self._recall_selected_eval_miss_index is not None:
            self.hint_var.set(
                f"Evaluation miss {self._recall_selected_eval_miss_index} copied into manual recall for the next tweak."
            )

    def copy_selected_recall_eval_source_to_manual(self) -> None:
        if not self._select_recall_request_for_selected_eval_miss():
            return
        self.copy_selected_recall_request()
        if self._recall_selected_eval_miss_index is not None:
            self.recall_selection_state.set(
                f"Source candidate for evaluation miss {self._recall_selected_eval_miss_index} copied into the manual form."
            )
            self.status_var.set(
                f"Source candidate for evaluation miss {self._recall_selected_eval_miss_index} copied into manual recall."
            )
            self.hint_var.set(
                "Run manual recall now, or tweak basis / limit / budget while the source context is still in view."
            )

    def _selected_recall_eval_source_suggestion(
        self,
    ) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None, dict[str, Any] | None]:
        if self._recall_selected_eval_miss_index is None:
            return None, None, None
        request_index = str(self._recall_selected_eval_miss_index)
        miss_row = self._recall_eval_miss_rows.get(request_index)
        request_row = self._recall_request_rows.get(request_index)
        if not isinstance(miss_row, Mapping) or not isinstance(request_row, Mapping):
            return miss_row if isinstance(miss_row, Mapping) else None, request_row, None
        return (
            miss_row,
            request_row,
            build_recall_miss_suggested_manual_config(
                miss_row,
                request_row=request_row,
            ),
        )

    def _can_rerun_selected_recall_eval_source(self) -> bool:
        _miss_row, _request_row, suggestion = self._selected_recall_eval_source_suggestion()
        return bool(isinstance(suggestion, Mapping) and suggestion.get("apply_ready"))

    def rerun_selected_recall_eval_source(self) -> None:
        if self.job_runner.has_pending_work():
            self.status_var.set("Manual recall is disabled while a worker job is running.")
            return
        if not self._select_recall_request_for_selected_eval_miss():
            return

        miss_row, request_row, suggestion = self._selected_recall_eval_source_suggestion()
        if not isinstance(miss_row, Mapping):
            self.status_var.set("The selected evaluation miss is no longer available.")
            return
        if not isinstance(request_row, Mapping) or not isinstance(suggestion, Mapping):
            self.status_var.set("The matching prepared recall request is no longer available.")
            self.hint_var.set("Refresh real data to restore the request before rerunning it.")
            return

        apply_recall_miss_suggested_manual_config(
            suggestion,
            task_kind_var=self.recall_task_kind,
            query_var=self.recall_query,
            request_basis_var=self.recall_request_basis,
            file_hint_var=self.recall_file_hint,
            limit_var=self.recall_limit,
            context_budget_var=self.recall_context_budget_chars,
        )
        if not suggestion.get("apply_ready"):
            self.status_var.set(
                f"Evaluation miss {self._recall_selected_eval_miss_index} still needs the source back in the index."
            )
            self.hint_var.set("Refresh real data or rebuild the memory index, then rerun evaluation.")
            return

        self.recall_selection_state.set(
            f"Suggested rerun for evaluation miss {self._recall_selected_eval_miss_index} loaded into the manual form."
        )
        result = self.run_manual_recall()
        if result is None:
            return

        rerun_feedback = build_recall_eval_source_rerun_result_text(
            miss_row,
            bundle=result.get("bundle"),
        )
        if rerun_feedback and self._recall_selected_eval_miss_index is not None:
            self.status_var.set(f"Suggested rerun for evaluation miss {self._recall_selected_eval_miss_index} is ready.")
            self.hint_var.set(rerun_feedback)
            self.recall_eval_source_action_hint_var.set(rerun_feedback)

    def apply_selected_recall_eval_miss_tweak(self) -> None:
        if self._recall_selected_eval_miss_index is None:
            self.status_var.set("Select an evaluation miss first.")
            return
        miss_row = self._recall_eval_miss_rows.get(str(self._recall_selected_eval_miss_index))
        if miss_row is None:
            self.status_var.set("The selected evaluation miss is no longer available.")
            return
        request_row = self._recall_request_rows.get(str(self._recall_selected_eval_miss_index))
        suggestion = build_recall_miss_suggested_manual_config(
            miss_row,
            request_row=request_row,
        )
        apply_recall_miss_suggested_manual_config(
            suggestion,
            task_kind_var=self.recall_task_kind,
            query_var=self.recall_query,
            request_basis_var=self.recall_request_basis,
            file_hint_var=self.recall_file_hint,
            limit_var=self.recall_limit,
            context_budget_var=self.recall_context_budget_chars,
        )
        if suggestion.get("apply_ready"):
            self.status_var.set(
                f"Applied the suggested tweak for evaluation miss {self._recall_selected_eval_miss_index}."
            )
            self.hint_var.set("Run manual recall now to see whether the suggested tweak closes the miss.")
        else:
            self.status_var.set(
                f"Loaded evaluation miss {self._recall_selected_eval_miss_index}, but this one wants reindexing first."
            )
            self.hint_var.set("Refresh real data or rebuild the memory index, then rerun evaluation.")

    def _apply_recall_eval_winner_compare(self, miss_row: Mapping[str, Any] | None) -> None:
        self.recall_eval_winner_compare_state.set(build_recall_eval_winner_compare_state(miss_row))
        resolved_miss_row: Mapping[str, Any] | None = miss_row
        request_row: Mapping[str, Any] | None = None
        if isinstance(miss_row, Mapping):
            request_index = str(miss_row.get("index") or "").strip()
            request_row = self._recall_request_rows.get(request_index)
            if isinstance(request_row, Mapping):
                resolved_miss_row = {
                    **dict(request_row),
                    **dict(miss_row),
                }
        winners = _recall_top_selected_rows(miss_row)[:RECALL_TOP_SELECTED_COMPARE_LIMIT]
        rows: list[dict[str, Any] | None] = list(winners)
        while len(rows) < RECALL_TOP_SELECTED_COMPARE_LIMIT:
            rows.append(None)
        self._recall_eval_winner_rows = rows
        self._recall_eval_source_row = resolve_recall_forensics_row(
            build_recall_eval_source_navigation_row(resolved_miss_row),
            root=self.controller.workspace_store.root,
            current_workspace_id=self.controller.workspace_store.workspace_id,
        )
        self.recall_eval_source_var.set(build_recall_eval_source_summary_text(resolved_miss_row))
        self.recall_eval_source_why_var.set(build_recall_eval_source_why_text(resolved_miss_row))
        self.recall_eval_source_action_hint_var.set(
            build_recall_eval_source_action_hint(
                resolved_miss_row,
                request_row=request_row,
                source_row=self._recall_eval_source_row,
            )
        )
        self._apply_recall_eval_source_chip_texts(
            build_recall_eval_source_chip_texts(resolved_miss_row),
        )
        if self.recall_eval_source_button is not None:
            self.recall_eval_source_button.configure(
                state="normal" if self._recall_eval_source_row is not None else "disabled"
            )
        if self.recall_eval_source_copy_button is not None:
            can_copy_source = (
                isinstance(resolved_miss_row, Mapping)
                and str(resolved_miss_row.get("index") or "").strip() in self._recall_request_rows
            )
            self.recall_eval_source_copy_button.configure(
                state="normal" if can_copy_source else "disabled"
            )
        if self.recall_eval_source_rerun_button is not None:
            suggestion_ready = bool(
                isinstance(resolved_miss_row, Mapping)
                and isinstance(request_row, Mapping)
                and build_recall_miss_suggested_manual_config(
                    resolved_miss_row,
                    request_row=request_row,
                ).get("apply_ready")
            )
            self.recall_eval_source_rerun_button.configure(
                state="normal" if suggestion_ready else "disabled"
            )

        for index, row in enumerate(rows, start=1):
            self.recall_eval_winner_vars[index - 1].set(
                build_recall_eval_winner_card_text(
                    resolved_miss_row,
                    row,
                    rank=index,
                    root=self.controller.workspace_store.root,
                )
            )
            self._apply_recall_eval_winner_chip_texts(
                index - 1,
                build_recall_eval_winner_chip_texts(resolved_miss_row, row),
            )
            if index - 1 < len(self.recall_eval_winner_buttons):
                self.recall_eval_winner_buttons[index - 1].configure(
                    state="normal" if recall_row_has_forensics_navigation(row) else "disabled"
                )

    def _open_recall_row_in_forensics(
        self,
        row: Mapping[str, Any],
        *,
        status_message: str,
        hint_label: str,
    ) -> bool:
        resolved_row = resolve_recall_forensics_row(
            row,
            root=self.controller.workspace_store.root,
            current_workspace_id=self.controller.workspace_store.workspace_id,
        )
        if resolved_row is None:
            self.status_var.set(
                "This recall item is missing History / Forensics navigation data. Rerun recall or evaluation to refresh it."
            )
            self.hint_var.set("Older snapshots may need a fresh request sync before direct jump works.")
            return False

        artifact_path = str(resolved_row.get("artifact_path") or "").strip()
        session_id = str(resolved_row.get("session_id") or "").strip() or None
        event_id = str(resolved_row.get("event_id") or "").strip() or None
        candidate_surface = str(resolved_row.get("session_surface") or "").strip().lower()
        target_surface = candidate_surface if candidate_surface in SESSION_SURFACES else self.forensics_surface.get()
        if target_surface not in SESSION_SURFACES:
            target_surface = "chat"

        self.forensics_surface.set(target_surface)
        self._forensics_selected_session_id = session_id
        self._forensics_selected_entry_id = event_id if session_id and event_id else None
        self.forensics_artifact_path.set(artifact_path)
        self.refresh_forensics()
        if hasattr(self, "notebook") and hasattr(self, "forensics_tab"):
            self.notebook.select(self.forensics_tab)

        label = summarize_text_preview(
            str(resolved_row.get("prompt_excerpt") or resolved_row.get("event_id") or "n/a"),
            limit=96,
        )
        self.status_var.set(status_message)
        self.hint_var.set(f"{hint_label}: {label}")
        return True

    def open_selected_recall_eval_source(self) -> None:
        if self._recall_selected_eval_miss_index is None:
            self.status_var.set("Select an evaluation miss first.")
            return
        if self._recall_eval_source_row is None:
            self.status_var.set("The selected miss does not have source navigation data yet.")
            self.hint_var.set("Older snapshots may need a fresh request sync before direct source jump works.")
            return

        self._open_recall_row_in_forensics(
            self._recall_eval_source_row,
            status_message=(
                f"Opened the source candidate for evaluation miss {self._recall_selected_eval_miss_index} "
                "in History / Forensics."
            ),
            hint_label="Source",
        )

    def open_recall_eval_winner_at(self, winner_index: int) -> None:
        if self._recall_selected_eval_miss_index is None:
            self.status_var.set("Select an evaluation miss first.")
            return
        if winner_index < 0 or winner_index >= len(self._recall_eval_winner_rows):
            self.status_var.set("That winner slot is out of range.")
            return

        winner_row = self._recall_eval_winner_rows[winner_index]
        if winner_row is None:
            self.status_var.set(f"No winner is recorded at slot #{winner_index + 1}.")
            return

        self._open_recall_row_in_forensics(
            winner_row,
            status_message=f"Opened winner #{winner_index + 1} in History / Forensics.",
            hint_label=f"Winner #{winner_index + 1}",
        )

    def open_selected_recall_eval_miss_winner(self) -> None:
        self.open_recall_eval_winner_at(0)

    def _apply_recall_evaluation(self, snapshot: dict[str, Any], *, status_message: str) -> None:
        self._populate_recall_requests(snapshot)
        self._populate_recall_evaluation(snapshot)
        self._set_output(self.recall_output, str(snapshot.get("report") or ""))
        self.status_var.set(status_message)
        self.backend_var.set("Backend: local-memory-recall")
        self.device_var.set("Device: local files / SQLite")
        self.artifact_var.set(f"Recall evaluation: {snapshot.get('evaluation_run_path') or 'n/a'}")
        self.hint_var.set(
            "Select a miss, compare its winners, then open one or apply the suggested tweak before rerunning manual recall."
        )

    def _apply_recall_result(self, result: dict[str, Any], *, status_message: str) -> None:
        self._set_output(self.recall_output, str(result.get("report") or ""))
        self._populate_recall_candidates(result.get("bundle"))
        apply_recall_compare_fields(
            result.get("pin_compare"),
            before_var=self.recall_compare_before_var,
            after_var=self.recall_compare_after_var,
            change_var=self.recall_compare_change_var,
        )
        self.status_var.set(status_message)
        self.backend_var.set("Backend: local-memory-recall")
        self.device_var.set("Device: local files / SQLite")
        self.artifact_var.set(f"Recall bundle: {result.get('bundle_path') or 'n/a'}")
        if result.get("request_label"):
            pin_count = len(result.get("pinned_event_ids") or [])
            hint = f"Recall source: {result['request_label']}"
            if pin_count:
                hint += f" with {pin_count} pin(s)"
            self.hint_var.set(hint)

    def open_selected_recall_candidate(self) -> None:
        if self._recall_selected_candidate_event_id is None:
            self.status_var.set("Select a recall candidate first.")
            return

        row = self._recall_candidate_rows.get(self._recall_selected_candidate_event_id)
        if row is None:
            self.status_var.set("The selected recall candidate is no longer available.")
            return

        summary = build_recall_candidate_summary(
            row,
            root=self.controller.workspace_store.root,
        )
        self.recall_candidate_selection_state.set(summary)
        self._open_recall_row_in_forensics(
            row,
            status_message="Opened recall candidate in History / Forensics.",
            hint_label="Recall candidate",
        )

    def run_recall_evaluation(self, *, refresh_dataset: bool = False) -> None:
        if self.job_runner.has_pending_work():
            self.status_var.set("Recall evaluation is disabled while a worker job is running.")
            return

        self._cancel_pending_startup_prewarm()
        self._set_busy("Evaluating recall hit quality...")
        try:
            snapshot = self.controller.evaluate_recall_dataset(refresh_dataset=refresh_dataset)
            self._apply_recall_evaluation(
                snapshot,
                status_message=(
                    "Recall hit evaluation refreshed."
                    if refresh_dataset
                    else "Recall hit evaluation is ready."
                ),
            )
        except Exception as exc:
            self.status_var.set(f"Recall evaluation failed: {type(exc).__name__}: {exc}")
        finally:
            self._clear_busy()
            self._resume_startup_prewarm_if_needed()

    def run_selected_recall_request(self) -> None:
        if self.job_runner.has_pending_work():
            self.status_var.set("Recall is disabled while a worker job is running.")
            return
        if self._recall_selected_request_index is None:
            self.status_var.set("Select a prepared recall request first.")
            return

        pinned_event_ids = self._current_recall_pinned_event_ids()
        self._cancel_pending_startup_prewarm()
        self._set_busy("Building recall bundle...")
        try:
            result = self.controller.build_recall_bundle(
                request_index=self._recall_selected_request_index,
                pinned_event_ids=pinned_event_ids,
            )
            self._apply_recall_result(
                result,
                status_message=(
                    f"Recall request {self._recall_selected_request_index} is ready."
                    + (f" {len(pinned_event_ids)} pin(s) applied." if pinned_event_ids else "")
                ),
            )
        except Exception as exc:
            self.status_var.set(f"Recall failed: {type(exc).__name__}: {exc}")
        finally:
            self._clear_busy()
            self._resume_startup_prewarm_if_needed()

    def run_manual_recall(self) -> dict[str, Any] | None:
        if self.job_runner.has_pending_work():
            self.status_var.set("Manual recall is disabled while a worker job is running.")
            return None

        pinned_event_ids = self._current_recall_pinned_event_ids()
        self._cancel_pending_startup_prewarm()
        self._set_busy("Running manual recall...")
        try:
            limit = coerce_ui_int(
                self.recall_limit.get(),
                default=DEFAULT_LIMIT,
                minimum=RECALL_MANUAL_LIMIT_MIN,
                maximum=RECALL_MANUAL_LIMIT_MAX,
            )
            context_budget_chars = coerce_ui_int(
                self.recall_context_budget_chars.get(),
                default=DEFAULT_CONTEXT_BUDGET_CHARS,
                minimum=RECALL_MANUAL_CONTEXT_BUDGET_MIN,
                maximum=RECALL_MANUAL_CONTEXT_BUDGET_MAX,
            )
            self.recall_limit.set(limit)
            self.recall_context_budget_chars.set(context_budget_chars)
            result = self.controller.build_recall_bundle(
                task_kind=self.recall_task_kind.get(),
                query_text=self.recall_query.get(),
                request_basis=self.recall_request_basis.get(),
                file_hint=self.recall_file_hint.get(),
                pinned_event_ids=pinned_event_ids,
                limit=limit,
                context_budget_chars=context_budget_chars,
            )
            self._apply_recall_result(
                result,
                status_message=(
                    f"Manual recall for {self.recall_task_kind.get()} is ready."
                    + (f" {len(pinned_event_ids)} pin(s) applied." if pinned_event_ids else "")
                ),
            )
            return result
        except Exception as exc:
            self.status_var.set(f"Manual recall failed: {type(exc).__name__}: {exc}")
            return None
        finally:
            self._clear_busy()
            self._resume_startup_prewarm_if_needed()

    def _current_evaluation_curation_filters(self) -> dict[str, Any]:
        state = self.evaluation_curation_state_filter.get().strip()
        decision = self.evaluation_curation_decision_filter.get().strip()
        reason_text = self.evaluation_curation_reason_filter.get().strip()
        return {
            "states": [] if state in ("", "all") else [state],
            "export_decisions": [] if decision in ("", "all") else [decision],
            "reasons": [
                item.strip()
                for item in reason_text.split(",")
                if item.strip()
            ],
        }

    def _selected_evaluation_candidate_event_id(self) -> str | None:
        selection = self.evaluation_curation_tree.selection()
        if not selection:
            return None
        row = self._evaluation_curation_rows.get(str(selection[0]))
        if row is None:
            return None
        return str(row.get("event_id") or "").strip() or None

    def _populate_evaluation_curation_preview(self, preview: Mapping[str, Any] | None) -> None:
        rows = build_evaluation_curation_rows(preview)
        self._evaluation_curation_rows = {row["row_id"]: row for row in rows}
        children = self.evaluation_curation_tree.get_children()
        if children:
            self.evaluation_curation_tree.delete(*children)
        for row in rows:
            self.evaluation_curation_tree.insert(
                "",
                tk.END,
                iid=row["row_id"],
                values=(
                    row["state"],
                    row["decision"],
                    row["next"],
                    row["label"],
                ),
            )

    def _populate_learning_candidate_review(self, review: Mapping[str, Any] | None) -> None:
        rows = build_learning_candidate_review_rows(review)
        self._learning_candidate_review_rows = {row["row_id"]: row for row in rows}
        children = self.learning_candidate_review_tree.get_children()
        if children:
            self.learning_candidate_review_tree.delete(*children)
        root = self.controller.workspace_store.root
        for row in rows:
            self.learning_candidate_review_tree.insert(
                "",
                tk.END,
                iid=row["row_id"],
                values=(
                    row["artifact"],
                    summarize_text_preview(row["event_id"], limit=48),
                    row["queue_state"],
                    summarize_text_preview(row["lifecycle_state"], limit=32),
                    summarize_text_preview(row["blocked_reason"], limit=46),
                    summarize_text_preview(row["next_action"], limit=60),
                    summarize_text_preview(row["policy_state"], limit=44),
                    summarize_text_preview(row["comparison_role"], limit=36),
                    summarize_text_preview(row["backend_id"], limit=44),
                    summarize_text_preview(
                        summarize_workspace_path(row["source_path"], root=root),
                        limit=96,
                    ),
                ),
            )

    def _apply_learning_candidate_review(self, review: Mapping[str, Any], *, status_message: str) -> None:
        self.learning_candidate_review_var.set(build_learning_candidate_review_state(review))
        self._populate_learning_candidate_review(review)
        report = str(review.get("report") or build_learning_candidate_review_report(review))
        self._set_output(self.evaluation_detail_output, report)
        self._set_output(self.evaluation_output, report)
        self.status_var.set(status_message)
        self.backend_var.set("Backend: local-learning-inspection")
        self.device_var.set("Device: local files")
        artifact_paths = [
            str(artifact.get("path"))
            for artifact in review.get("artifacts") or []
            if isinstance(artifact, Mapping) and artifact.get("loaded")
        ]
        self.artifact_var.set(
            "Learning candidate artifacts: " + ("; ".join(artifact_paths) if artifact_paths else "none loaded")
        )
        self.hint_var.set("Read-only latest artifacts: preview, human-selected, JSONL dry-run, and candidate diff.")

    def refresh_learning_candidate_review(self) -> None:
        if self.job_runner.has_pending_work():
            self.status_var.set("Candidate review is disabled while a worker job is running.")
            return
        self._cancel_pending_startup_prewarm()
        self._set_busy("Refreshing candidate review...")
        try:
            review = self.controller.build_learning_candidate_review()
            self._apply_learning_candidate_review(
                review,
                status_message="Learning candidate review is ready.",
            )
        except Exception as exc:
            self.status_var.set(f"Candidate review failed: {type(exc).__name__}: {exc}")
            self.hint_var.set("")
        finally:
            self._clear_busy()
            self._resume_startup_prewarm_if_needed()

    def _record_selected_evaluation_review(self, *, resolved: bool) -> None:
        if self.job_runner.has_pending_work():
            self.status_var.set("Review signal recording is disabled while a worker job is running.")
            return
        event_id = self._selected_evaluation_candidate_event_id()
        if event_id is None:
            self.status_var.set("Select a curation candidate first.")
            return

        summary = self.evaluation_review_summary.get().strip()
        if not summary:
            summary = "Review resolved from Local UI." if resolved else "Review remains unresolved from Local UI."
            self.evaluation_review_summary.set(summary)

        self._cancel_pending_startup_prewarm()
        self._set_busy("Recording review signal...")
        try:
            result = self.controller.record_evaluation_review_resolution(
                source_event_id=event_id,
                resolved=resolved,
                review_id=self.evaluation_review_id.get(),
                review_url=self.evaluation_review_url.get(),
                resolution_summary=summary,
                curation_filters=self._current_evaluation_curation_filters(),
            )
            signal = dict(result.get("recorded_signal") or {})
            self._apply_evaluation_snapshot(
                result,
                status_message=f"Recorded {signal.get('signal_kind') or 'review signal'} for curation candidate.",
            )
        except Exception as exc:
            self.status_var.set(f"Review signal failed: {type(exc).__name__}: {exc}")
            self.hint_var.set("")
        finally:
            self._clear_busy()
            self._resume_startup_prewarm_if_needed()

    def mark_selected_evaluation_review_resolved(self) -> None:
        self._record_selected_evaluation_review(resolved=True)

    def mark_selected_evaluation_review_unresolved(self) -> None:
        self._record_selected_evaluation_review(resolved=False)

    def _populate_evaluation_signals(self, snapshot: Mapping[str, Any] | None) -> None:
        rows = build_evaluation_signal_rows(snapshot)
        self._evaluation_signal_rows = {row["row_id"]: row for row in rows}
        children = self.evaluation_signals_tree.get_children()
        if children:
            self.evaluation_signals_tree.delete(*children)
        for row in rows:
            self.evaluation_signals_tree.insert(
                "",
                tk.END,
                iid=row["row_id"],
                values=(
                    row["kind"],
                    row["source"],
                    row["relation"],
                    row["status"],
                    row["label"],
                ),
            )
        detail_lines = [
            f"{row['kind']} {row['source']}"
            + (f" -> {row['relation']}" if row["relation"] else "")
            + (f": {row['detail']}" if row["detail"] else "")
            for row in rows
        ]
        self._set_output(
            self.evaluation_detail_output,
            "\n".join(detail_lines) if detail_lines else "No evaluation signals are available yet.",
        )

    def _apply_evaluation_snapshot(self, result: Mapping[str, Any], *, status_message: str) -> None:
        snapshot = dict(result.get("snapshot") or result)
        preview = dict(result.get("curation_preview") or {})
        self.evaluation_state.set(build_evaluation_snapshot_state(snapshot))
        self.evaluation_acceptance_var.set(build_evaluation_acceptance_text(snapshot))
        self.evaluation_test_var.set(build_evaluation_test_text(snapshot))
        self.evaluation_repair_var.set(build_evaluation_repair_text(snapshot))
        self.evaluation_comparison_var.set(build_evaluation_comparison_text(snapshot))
        self.evaluation_curation_var.set(build_evaluation_curation_text(snapshot))
        self.evaluation_adoption_var.set(build_evaluation_adoption_text(preview))
        self._populate_evaluation_curation_preview(preview)
        self._populate_evaluation_signals(snapshot)
        self._set_output(
            self.evaluation_output,
            str(result.get("report") or format_evaluation_snapshot_report(snapshot)),
        )
        self.status_var.set(status_message)
        self.backend_var.set("Backend: local-evaluation-loop")
        self.device_var.set("Device: local files / SQLite")
        snapshot_path = result.get("snapshot_run_path") or "n/a"
        preview_path = result.get("curation_preview_run_path")
        if preview_path:
            self.artifact_var.set(f"Evaluation snapshot: {snapshot_path}; curation preview: {preview_path}")
        else:
            self.artifact_var.set(f"Evaluation snapshot: {snapshot_path}")
        self.hint_var.set(
            "Acceptance, rejection, review, test, repair-link, and preview signals share local files."
        )

    def refresh_evaluation_snapshot(self) -> None:
        if self.job_runner.has_pending_work():
            self.status_var.set("Evaluation snapshot is disabled while a worker job is running.")
            return
        self._cancel_pending_startup_prewarm()
        self._set_busy("Refreshing evaluation snapshot...")
        try:
            result = self.controller.build_evaluation_snapshot(
                curation_filters=self._current_evaluation_curation_filters(),
            )
            self._apply_evaluation_snapshot(
                result,
                status_message="Evaluation snapshot is ready.",
            )
        except Exception as exc:
            self.status_var.set(f"Evaluation snapshot failed: {type(exc).__name__}: {exc}")
            self.hint_var.set("")
        finally:
            self._clear_busy()
            self._resume_startup_prewarm_if_needed()

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
        self.refresh_recall_button.configure(state=run_state)
        self.evaluate_recall_button.configure(state=run_state)
        self.refresh_and_evaluate_recall_button.configure(state=run_state)
        self.run_selected_recall_button.configure(state=run_state)
        self.copy_recall_request_button.configure(state=run_state)
        self.copy_recall_miss_button.configure(state=run_state)
        self.apply_recall_miss_tweak_button.configure(state=run_state)
        self.open_recall_miss_winner_button.configure(state=run_state)
        if self.recall_eval_source_button is not None:
            self.recall_eval_source_button.configure(
                state=(
                    run_state
                    if busy
                    else ("normal" if self._recall_eval_source_row is not None else "disabled")
                )
            )
        if self.recall_eval_source_copy_button is not None:
            self.recall_eval_source_copy_button.configure(
                state=(
                    run_state
                    if busy
                    else (
                        "normal"
                        if self._recall_selected_eval_miss_index is not None
                        and str(self._recall_selected_eval_miss_index) in self._recall_request_rows
                        else "disabled"
                    )
                )
            )
        if self.recall_eval_source_rerun_button is not None:
            self.recall_eval_source_rerun_button.configure(
                state=(
                    run_state
                    if busy
                    else ("normal" if self._can_rerun_selected_recall_eval_source() else "disabled")
                )
            )
        for index, button in enumerate(self.recall_eval_winner_buttons):
            button.configure(
                state=(
                    run_state
                    if busy
                    else (
                        "normal"
                        if index < len(self._recall_eval_winner_rows)
                        and recall_row_has_forensics_navigation(self._recall_eval_winner_rows[index])
                        else "disabled"
                    )
                )
            )
        self.run_manual_recall_button.configure(state=run_state)
        self.open_recall_candidate_button.configure(state=run_state)
        self.pin_recall_candidate_button.configure(state=run_state)
        self.clear_recall_pins_button.configure(state=run_state)
        self.refresh_evaluation_button.configure(state=run_state)
        for button_name in (
            "apply_evaluation_filter_button",
            "mark_review_resolved_button",
            "mark_review_unresolved_button",
            "refresh_candidate_review_button",
        ):
            button = getattr(self, button_name, None)
            if button is not None:
                button.configure(state=run_state)
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
