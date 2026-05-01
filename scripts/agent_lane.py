#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import shlex
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from gemma_runtime import repo_root, timestamp_slug, timestamp_utc, write_json
from workspace_state import DEFAULT_WORKSPACE_ID


AGENT_TASK_SCHEMA_NAME = "software-satellite-agent-task"
AGENT_TASK_SCHEMA_VERSION = 1
AGENT_TASK_LOG_SCHEMA_NAME = "software-satellite-agent-task-log"
AGENT_TASK_LOG_SCHEMA_VERSION = 1
AGENT_RUN_SCHEMA_NAME = "software-satellite-agent-run"
AGENT_RUN_SCHEMA_VERSION = 1
AGENT_RUN_LOG_SCHEMA_NAME = "software-satellite-agent-run-log"
AGENT_RUN_LOG_SCHEMA_VERSION = 1
AGENT_LANE_SNAPSHOT_SCHEMA_NAME = "software-satellite-agent-lane-snapshot"
AGENT_LANE_SNAPSHOT_SCHEMA_VERSION = 1

TASK_KINDS = ("patch_plan_verify", "investigation", "repair_follow_up")
RUN_STATUSES = ("succeeded", "failed", "blocked")


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def agent_lane_root(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return _resolve_root(root) / "artifacts" / "agent_lane" / workspace_id


def agent_task_log_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return agent_lane_root(workspace_id=workspace_id, root=root) / "tasks.jsonl"


def agent_run_log_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return agent_lane_root(workspace_id=workspace_id, root=root) / "runs.jsonl"


def agent_run_artifact_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    run_id: str,
) -> Path:
    safe_run_id = "".join(
        char if char.isalnum() or char in ("-", "_") else "-"
        for char in run_id
    ).strip("-") or "agent-run"
    return agent_lane_root(workspace_id=workspace_id, root=root) / "runs" / f"{safe_run_id}.json"


def agent_lane_snapshot_latest_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return agent_lane_root(workspace_id=workspace_id, root=root) / "snapshots" / "latest.json"


def agent_lane_snapshot_run_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return agent_lane_root(workspace_id=workspace_id, root=root) / "snapshots" / "runs" / f"{timestamp_slug()}-agent-lane.json"


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [cleaned for item in value if (cleaned := _clean_text(item)) is not None]


def _mapping_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _normalize_task_kind(value: str | None) -> str:
    normalized = (_clean_text(value) or "patch_plan_verify").lower().replace("-", "_")
    if normalized not in TASK_KINDS:
        raise ValueError(f"Unsupported agent task kind `{value}`.")
    return normalized


def _task_log_header(*, workspace_id: str) -> dict[str, Any]:
    return {
        "schema_name": AGENT_TASK_LOG_SCHEMA_NAME,
        "schema_version": AGENT_TASK_LOG_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "created_at_utc": timestamp_utc(),
    }


def _run_log_header(*, workspace_id: str) -> dict[str, Any]:
    return {
        "schema_name": AGENT_RUN_LOG_SCHEMA_NAME,
        "schema_version": AGENT_RUN_LOG_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "created_at_utc": timestamp_utc(),
    }


def _read_log_header(path: Path, *, schema_name: str, schema_version: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        raise ValueError(f"Agent lane log `{path}` was empty.")
    header = json.loads(first_line)
    if header.get("schema_name") != schema_name:
        raise ValueError(f"Unexpected agent lane log schema name in `{path}`.")
    if header.get("schema_version") != schema_version:
        raise ValueError(f"Unsupported agent lane log schema version in `{path}`.")
    return header


def _append_jsonl(path: Path, *, header: dict[str, Any], payload: dict[str, Any], workspace_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(header, ensure_ascii=False) + "\n")
    else:
        existing_header = _read_log_header(
            path,
            schema_name=str(header["schema_name"]),
            schema_version=int(header["schema_version"]),
        )
        if existing_header.get("workspace_id") != workspace_id:
            raise ValueError(f"Agent lane log `{path}` belongs to workspace `{existing_header.get('workspace_id')}`.")

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _plan_step_records(plan_steps: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, step in enumerate(plan_steps, start=1):
        cleaned = _clean_text(step)
        if cleaned is None:
            continue
        records.append(
            {
                "step_id": f"plan-{index}",
                "description": cleaned,
                "status": "planned",
            }
        )
    return records


def _verification_command_records(commands: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, command in enumerate(commands, start=1):
        cleaned = _clean_text(command)
        if cleaned is None:
            continue
        records.append(
            {
                "command_id": f"verify-{index}",
                "command": cleaned,
                "status": "planned",
            }
        )
    return records


def _validate_plan_steps(value: Any, *, location: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Agent task is missing plan_steps{location}.")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Agent task plan step {index} is not an object{location}.")
        description = _clean_text(item.get("description"))
        if description is None:
            raise ValueError(f"Agent task plan step {index} is missing description{location}.")
        record = copy.deepcopy(dict(item))
        record["step_id"] = _clean_text(record.get("step_id")) or f"plan-{index}"
        record["description"] = description
        record["status"] = _clean_text(record.get("status")) or "planned"
        records.append(record)
    return records


def _validate_verification_commands(value: Any, *, location: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Agent task is missing verification commands{location}.")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Agent task verification command {index} is not an object{location}.")
        command = _clean_text(item.get("command"))
        if command is None:
            raise ValueError(f"Agent task verification command {index} is missing command{location}.")
        record = copy.deepcopy(dict(item))
        record["command_id"] = _clean_text(record.get("command_id")) or f"verify-{index}"
        record["command"] = command
        record["status"] = _clean_text(record.get("status")) or "planned"
        records.append(record)
    return records


def build_agent_task(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    title: str,
    goal: str,
    task_kind: str = "patch_plan_verify",
    task_id: str | None = None,
    created_at_utc: str | None = None,
    origin: str = "manual",
    scope_paths: Iterable[str] | None = None,
    plan_steps: Iterable[str] | None = None,
    verification_commands: Iterable[str] | None = None,
    acceptance_criteria: Iterable[str] | None = None,
    pass_definition: str | None = None,
    tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    cleaned_title = _clean_text(title)
    cleaned_goal = _clean_text(goal)
    if cleaned_title is None:
        raise ValueError("Agent tasks require a title.")
    if cleaned_goal is None:
        raise ValueError("Agent tasks require a goal.")

    plan = _plan_step_records(list(plan_steps or []))
    if not plan:
        raise ValueError("Agent tasks require at least one plan step.")
    verification = _verification_command_records(list(verification_commands or []))
    if not verification:
        raise ValueError("Agent tasks require at least one verification command.")

    normalized_kind = _normalize_task_kind(task_kind)
    created_at = created_at_utc or timestamp_utc()
    return {
        "schema_name": AGENT_TASK_SCHEMA_NAME,
        "schema_version": AGENT_TASK_SCHEMA_VERSION,
        "task_id": task_id or f"{workspace_id}:agent-task:{timestamp_slug()}:{uuid4().hex[:8]}",
        "workspace_id": workspace_id,
        "created_at_utc": created_at,
        "origin": _clean_text(origin) or "manual",
        "task_kind": normalized_kind,
        "title": cleaned_title,
        "goal": cleaned_goal,
        "scope": {
            "paths": _string_list(list(scope_paths or [])),
        },
        "plan_steps": plan,
        "verification": {
            "commands": verification,
            "pass_definition": _clean_text(pass_definition)
            or "All verification commands exit with status 0.",
        },
        "acceptance_criteria": _string_list(list(acceptance_criteria or [])),
        "tags": _string_list(list(tags or [])),
    }


def _validate_agent_task(task: Mapping[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    payload = copy.deepcopy(dict(task))
    location = f" in `{path}`" if path is not None else ""
    if payload.get("schema_name") != AGENT_TASK_SCHEMA_NAME:
        raise ValueError(f"Unexpected agent task schema name{location}.")
    if payload.get("schema_version") != AGENT_TASK_SCHEMA_VERSION:
        raise ValueError(f"Unsupported agent task schema version{location}.")
    task_id = _clean_text(payload.get("task_id"))
    if task_id is None:
        raise ValueError(f"Agent task is missing task_id{location}.")
    workspace_id = _clean_text(payload.get("workspace_id"))
    if workspace_id is None:
        raise ValueError(f"Agent task is missing workspace_id{location}.")
    payload["task_id"] = task_id
    payload["workspace_id"] = workspace_id
    payload["task_kind"] = _normalize_task_kind(_clean_text(payload.get("task_kind")))
    title = _clean_text(payload.get("title"))
    if title is None:
        raise ValueError(f"Agent task is missing title{location}.")
    goal = _clean_text(payload.get("goal"))
    if goal is None:
        raise ValueError(f"Agent task is missing goal{location}.")
    payload["title"] = title
    payload["goal"] = goal
    payload["plan_steps"] = _validate_plan_steps(payload.get("plan_steps"), location=location)
    verification = _mapping_dict(payload.get("verification"))
    verification["commands"] = _validate_verification_commands(verification.get("commands"), location=location)
    verification["pass_definition"] = (
        _clean_text(verification.get("pass_definition"))
        or "All verification commands exit with status 0."
    )
    payload["verification"] = verification
    return payload


def append_agent_task(path: Path, task: Mapping[str, Any], *, workspace_id: str) -> dict[str, Any]:
    payload = _validate_agent_task(task)
    if payload.get("workspace_id") != workspace_id:
        raise ValueError(f"Agent task belongs to workspace `{payload.get('workspace_id')}`.")
    if path.exists():
        for existing in read_agent_tasks(path):
            if existing.get("task_id") == payload.get("task_id"):
                raise ValueError(f"Agent task `{payload.get('task_id')}` already exists in `{path}`.")
    _append_jsonl(
        path,
        header=_task_log_header(workspace_id=workspace_id),
        payload=payload,
        workspace_id=workspace_id,
    )
    return payload


def read_agent_tasks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    _read_log_header(
        path,
        schema_name=AGENT_TASK_LOG_SCHEMA_NAME,
        schema_version=AGENT_TASK_LOG_SCHEMA_VERSION,
    )
    tasks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            cleaned = line.strip()
            if not cleaned:
                continue
            tasks.append(_validate_agent_task(json.loads(cleaned), path=path))
    return tasks


def _excerpt(value: str, *, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[truncated]"


def _run_command_trace(command: Mapping[str, Any], *, root: Path, timeout_seconds: int) -> dict[str, Any]:
    command_id = _clean_text(command.get("command_id")) or "verify"
    command_text = _clean_text(command.get("command"))
    if command_text is None:
        raise ValueError("Verification command is missing command text.")

    started_at = timestamp_utc()
    start = time.monotonic()
    trace: dict[str, Any] = {
        "trace_id": f"{command_id}:{uuid4().hex[:8]}",
        "tool_kind": "verification_command",
        "command_id": command_id,
        "command": command_text,
        "cwd": str(root),
        "started_at_utc": started_at,
    }
    try:
        command_args = shlex.split(command_text)
    except ValueError as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        trace.update(
            {
                "finished_at_utc": timestamp_utc(),
                "duration_ms": duration_ms,
                "status": "failed",
                "exit_code": None,
                "timed_out": False,
                "stdout_excerpt": "",
                "stderr_excerpt": "",
                "failure_summary": f"Invalid verification command: {exc}",
            }
        )
        return trace

    try:
        completed = subprocess.run(
            command_args,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        trace.update(
            {
                "finished_at_utc": timestamp_utc(),
                "duration_ms": duration_ms,
                "status": "failed",
                "exit_code": None,
                "timed_out": True,
                "stdout_excerpt": _excerpt(stdout),
                "stderr_excerpt": _excerpt(stderr),
                "failure_summary": f"Timed out after {timeout_seconds} seconds.",
            }
        )
        return trace
    except OSError as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        trace.update(
            {
                "finished_at_utc": timestamp_utc(),
                "duration_ms": duration_ms,
                "status": "failed",
                "exit_code": None,
                "timed_out": False,
                "stdout_excerpt": "",
                "stderr_excerpt": "",
                "failure_summary": f"{type(exc).__name__}: {exc}",
            }
        )
        return trace

    duration_ms = int((time.monotonic() - start) * 1000)
    passed = completed.returncode == 0
    trace.update(
        {
            "finished_at_utc": timestamp_utc(),
            "duration_ms": duration_ms,
            "status": "passed" if passed else "failed",
            "exit_code": completed.returncode,
            "timed_out": False,
            "stdout_excerpt": _excerpt(completed.stdout),
            "stderr_excerpt": _excerpt(completed.stderr),
        }
    )
    if not passed:
        trace["failure_summary"] = f"Command exited with status {completed.returncode}."
    return trace


def _plan_trace_records(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for step in task.get("plan_steps") or []:
        if not isinstance(step, Mapping):
            continue
        step_id = _clean_text(step.get("step_id")) or f"plan-{len(records) + 1}"
        description = _clean_text(step.get("description"))
        if description is None:
            continue
        records.append(
            {
                "trace_id": f"{step_id}:{uuid4().hex[:8]}",
                "tool_kind": "plan_step",
                "step_id": step_id,
                "description": description,
                "status": "recorded",
                "recorded_at_utc": timestamp_utc(),
            }
        )
    return records


def _verification_quality_checks(tool_traces: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for trace in tool_traces:
        if _clean_text(trace.get("tool_kind")) != "verification_command":
            continue
        command = _clean_text(trace.get("command"))
        if command is None:
            continue
        passed = _clean_text(trace.get("status")) == "passed"
        detail_parts = [
            f"exit={trace.get('exit_code')}" if trace.get("exit_code") is not None else None,
            _clean_text(trace.get("failure_summary")),
        ]
        checks.append(
            {
                "name": _clean_text(trace.get("command_id")) or command,
                "pass": passed,
                "detail": " | ".join(part for part in detail_parts if part),
            }
        )
    return checks


def run_agent_task(
    task: Mapping[str, Any],
    *,
    root: Path | None = None,
    run_id: str | None = None,
    origin: str = "cli",
    result_summary: str | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    if timeout_seconds < 1:
        raise ValueError("Agent verification timeout_seconds must be at least 1.")
    resolved_root = _resolve_root(root)
    task_payload = _validate_agent_task(task)
    workspace_id = str(task_payload.get("workspace_id") or DEFAULT_WORKSPACE_ID)
    started_at = timestamp_utc()
    traces = _plan_trace_records(task_payload)
    verification = _mapping_dict(task_payload.get("verification"))
    for command in verification.get("commands") or []:
        if not isinstance(command, Mapping):
            continue
        traces.append(_run_command_trace(command, root=resolved_root, timeout_seconds=timeout_seconds))
    completed_at = timestamp_utc()

    verification_traces = [
        trace
        for trace in traces
        if _clean_text(trace.get("tool_kind")) == "verification_command"
    ]
    if not verification_traces:
        status = "blocked"
        quality_status = "not_run"
        execution_status = "blocked"
    elif all(_clean_text(trace.get("status")) == "passed" for trace in verification_traces):
        status = "succeeded"
        quality_status = "pass"
        execution_status = "ok"
    else:
        status = "failed"
        quality_status = "fail"
        execution_status = "failed"

    failure_summaries = [
        summary
        for trace in verification_traces
        if _clean_text(trace.get("status")) != "passed"
        if (summary := _clean_text(trace.get("failure_summary"))) is not None
    ]
    summary = _clean_text(result_summary)
    if summary is None:
        if status == "succeeded":
            summary = f"Verification passed for {len(verification_traces)} command(s)."
        elif failure_summaries:
            summary = failure_summaries[0]
        else:
            summary = "Verification did not run."

    return {
        "schema_name": AGENT_RUN_SCHEMA_NAME,
        "schema_version": AGENT_RUN_SCHEMA_VERSION,
        "run_id": run_id or f"{workspace_id}:agent-run:{timestamp_slug()}:{uuid4().hex[:8]}",
        "workspace_id": workspace_id,
        "task_id": task_payload["task_id"],
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "origin": _clean_text(origin) or "cli",
        "status": status,
        "task_snapshot": task_payload,
        "tool_traces": traces,
        "outcome": {
            "status": status,
            "quality_status": quality_status,
            "execution_status": execution_status,
            "result_summary": summary,
            "verification_command_count": len(verification_traces),
            "verification_failed_count": sum(
                1 for trace in verification_traces if _clean_text(trace.get("status")) != "passed"
            ),
            "failure_summary": failure_summaries[0] if failure_summaries else None,
            "quality_checks": _verification_quality_checks(verification_traces),
        },
        "paths": {},
        "tags": _string_list(["agent_lane", task_payload.get("task_kind"), status]),
    }


def _validate_agent_run(run: Mapping[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    payload = copy.deepcopy(dict(run))
    location = f" in `{path}`" if path is not None else ""
    if payload.get("schema_name") != AGENT_RUN_SCHEMA_NAME:
        raise ValueError(f"Unexpected agent run schema name{location}.")
    if payload.get("schema_version") != AGENT_RUN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported agent run schema version{location}.")
    run_id = _clean_text(payload.get("run_id"))
    if run_id is None:
        raise ValueError(f"Agent run is missing run_id{location}.")
    workspace_id = _clean_text(payload.get("workspace_id"))
    if workspace_id is None:
        raise ValueError(f"Agent run is missing workspace_id{location}.")
    task_id = _clean_text(payload.get("task_id"))
    if task_id is None:
        raise ValueError(f"Agent run is missing task_id{location}.")
    payload["run_id"] = run_id
    payload["workspace_id"] = workspace_id
    payload["task_id"] = task_id
    status = _clean_text(payload.get("status"))
    if status not in RUN_STATUSES:
        raise ValueError(f"Unsupported agent run status `{status}`{location}.")
    payload["status"] = status
    task_snapshot = _mapping_dict(payload.get("task_snapshot"))
    payload["task_snapshot"] = _validate_agent_task(task_snapshot, path=path)
    if payload["task_snapshot"].get("task_id") != task_id:
        raise ValueError(f"Agent run task_id does not match task_snapshot.task_id{location}.")
    if payload["task_snapshot"].get("workspace_id") != workspace_id:
        raise ValueError(f"Agent run workspace_id does not match task_snapshot.workspace_id{location}.")
    if not isinstance(payload.get("tool_traces"), list):
        raise ValueError(f"Agent run is missing tool_traces{location}.")
    if not all(isinstance(item, Mapping) for item in payload["tool_traces"]):
        raise ValueError(f"Agent run tool_traces must contain objects{location}.")
    outcome = _mapping_dict(payload.get("outcome"))
    outcome_status = _clean_text(outcome.get("status"))
    if outcome_status is not None and outcome_status != status:
        raise ValueError(f"Agent run outcome.status does not match run status{location}.")
    outcome["status"] = status
    if _clean_text(outcome.get("quality_status")) is None:
        raise ValueError(f"Agent run outcome is missing quality_status{location}.")
    if _clean_text(outcome.get("execution_status")) is None:
        raise ValueError(f"Agent run outcome is missing execution_status{location}.")
    payload["outcome"] = outcome
    payload["paths"] = _mapping_dict(payload.get("paths"))
    return payload


def append_agent_run(path: Path, run: Mapping[str, Any], *, workspace_id: str) -> dict[str, Any]:
    payload = _validate_agent_run(run)
    if payload.get("workspace_id") != workspace_id:
        raise ValueError(f"Agent run belongs to workspace `{payload.get('workspace_id')}`.")
    if path.exists():
        for existing in read_agent_runs(path):
            if existing.get("run_id") == payload.get("run_id"):
                raise ValueError(f"Agent run `{payload.get('run_id')}` already exists in `{path}`.")
    _append_jsonl(
        path,
        header=_run_log_header(workspace_id=workspace_id),
        payload=payload,
        workspace_id=workspace_id,
    )
    return payload


def read_agent_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    _read_log_header(
        path,
        schema_name=AGENT_RUN_LOG_SCHEMA_NAME,
        schema_version=AGENT_RUN_LOG_SCHEMA_VERSION,
    )
    runs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            cleaned = line.strip()
            if not cleaned:
                continue
            runs.append(_validate_agent_run(json.loads(cleaned), path=path))
    return runs


def record_agent_task(
    task: Mapping[str, Any],
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
) -> dict[str, Any]:
    return append_agent_task(
        agent_task_log_path(workspace_id=workspace_id, root=root),
        task,
        workspace_id=workspace_id,
    )


def record_agent_run(
    run: Mapping[str, Any],
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
) -> tuple[dict[str, Any], Path]:
    payload = _validate_agent_run(run)
    run_path = agent_run_artifact_path(
        workspace_id=workspace_id,
        root=root,
        run_id=str(payload["run_id"]),
    )
    payload["paths"]["run_artifact_path"] = str(run_path)
    run_log_path = agent_run_log_path(workspace_id=workspace_id, root=root)
    if run_log_path.exists():
        for existing in read_agent_runs(run_log_path):
            if existing.get("run_id") == payload.get("run_id"):
                raise ValueError(f"Agent run `{payload.get('run_id')}` already exists in `{run_log_path}`.")
    write_json(run_path, payload)
    append_agent_run(
        run_log_path,
        payload,
        workspace_id=workspace_id,
    )
    return payload, run_path


def build_agent_lane_snapshot(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    task_path = agent_task_log_path(workspace_id=workspace_id, root=resolved_root)
    run_path = agent_run_log_path(workspace_id=workspace_id, root=resolved_root)
    tasks = read_agent_tasks(task_path)
    runs = read_agent_runs(run_path)
    runs.sort(
        key=lambda run: (
            str(run.get("completed_at_utc") or ""),
            str(run.get("run_id") or ""),
        ),
        reverse=True,
    )
    status_counts = Counter(str(run.get("status") or "") for run in runs)
    return {
        "schema_name": AGENT_LANE_SNAPSHOT_SCHEMA_NAME,
        "schema_version": AGENT_LANE_SNAPSHOT_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "generated_at_utc": timestamp_utc(),
        "paths": {
            "task_log_path": str(task_path),
            "run_log_path": str(run_path),
        },
        "counts": {
            "tasks": len(tasks),
            "runs": len(runs),
            "succeeded": int(status_counts.get("succeeded", 0)),
            "failed": int(status_counts.get("failed", 0)),
            "blocked": int(status_counts.get("blocked", 0)),
        },
        "latest_runs": runs[:12],
    }


def record_agent_lane_snapshot(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
) -> tuple[dict[str, Any], Path, Path]:
    resolved_root = _resolve_root(root)
    snapshot = build_agent_lane_snapshot(root=resolved_root, workspace_id=workspace_id)
    latest_path = agent_lane_snapshot_latest_path(workspace_id=workspace_id, root=resolved_root)
    run_path = agent_lane_snapshot_run_path(workspace_id=workspace_id, root=resolved_root)
    snapshot["paths"]["snapshot_latest_path"] = str(latest_path)
    snapshot["paths"]["snapshot_run_path"] = str(run_path)
    write_json(run_path, snapshot)
    write_json(latest_path, snapshot)
    return snapshot, latest_path, run_path


def format_agent_lane_snapshot_report(snapshot: Mapping[str, Any]) -> str:
    counts = _mapping_dict(snapshot.get("counts"))
    lines = [
        "Agent lane snapshot",
        f"Workspace: {_clean_text(snapshot.get('workspace_id')) or DEFAULT_WORKSPACE_ID}",
        f"Tasks: {int(counts.get('tasks') or 0)}",
        f"Runs: {int(counts.get('runs') or 0)}",
        f"Succeeded: {int(counts.get('succeeded') or 0)}",
        f"Failed: {int(counts.get('failed') or 0)}",
        f"Blocked: {int(counts.get('blocked') or 0)}",
    ]
    latest_runs = [
        item
        for item in snapshot.get("latest_runs") or []
        if isinstance(item, Mapping)
    ]
    if latest_runs:
        lines.extend(("", "Recent runs:"))
        for run in latest_runs[:4]:
            outcome = _mapping_dict(run.get("outcome"))
            task = _mapping_dict(run.get("task_snapshot"))
            title = _clean_text(task.get("title")) or _clean_text(run.get("task_id")) or "agent task"
            lines.append(
                "- "
                + f"{_clean_text(run.get('status')) or 'unknown'} "
                + f"{title}: "
                + f"{_clean_text(outcome.get('result_summary')) or 'n/a'}"
            )
    return "\n".join(lines)
