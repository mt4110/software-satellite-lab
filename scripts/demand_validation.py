#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluation_loop import evaluation_signal_log_path, read_evaluation_signals
from failure_memory_review import failure_memory_root, latest_recall_path
from gemma_runtime import repo_root, timestamp_slug, timestamp_utc, write_json
from memory_index import rebuild_memory_index
from software_work_events import (
    build_event_contract_check,
    build_event_contract_report,
    iter_workspace_events,
    read_event_log,
)
from workspace_state import DEFAULT_WORKSPACE_ID


DEMAND_VALIDATION_DOGFOOD_RUN_SCHEMA_NAME = "software-satellite-demand-validation-dogfood-run"
DEMAND_VALIDATION_DOGFOOD_RUN_SCHEMA_VERSION = 1
DEMAND_VALIDATION_DOGFOOD_RUN_LOG_SCHEMA_NAME = "software-satellite-demand-validation-dogfood-run-log"
DEMAND_VALIDATION_DOGFOOD_RUN_LOG_SCHEMA_VERSION = 1
DEMAND_VALIDATION_INTERVIEW_SCHEMA_NAME = "software-satellite-demand-validation-interview"
DEMAND_VALIDATION_INTERVIEW_SCHEMA_VERSION = 1
DEMAND_VALIDATION_INTERVIEW_LOG_SCHEMA_NAME = "software-satellite-demand-validation-interview-log"
DEMAND_VALIDATION_INTERVIEW_LOG_SCHEMA_VERSION = 1
DEMAND_VALIDATION_SETUP_SCHEMA_NAME = "software-satellite-demand-validation-demo-setup"
DEMAND_VALIDATION_SETUP_SCHEMA_VERSION = 1
DEMAND_VALIDATION_REPORT_SCHEMA_NAME = "software-satellite-demand-validation-report"
DEMAND_VALIDATION_REPORT_SCHEMA_VERSION = 1

DEMAND_VALIDATION_JUDGEMENTS = ("yes", "no", "unclear")
USEFUL_RECALL_TARGET = 0.30
SOURCE_COMPLETENESS_TARGET = 0.90
VERDICT_CAPTURE_SECONDS_TARGET = 30.0
CLONE_TO_DEMO_MINUTES_TARGET = 15.0


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def demand_validation_root(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return _resolve_root(root) / "artifacts" / "demand_validation" / workspace_id


def dogfood_run_log_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return demand_validation_root(workspace_id=workspace_id, root=root) / "dogfood-runs.jsonl"


def dogfood_run_latest_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return demand_validation_root(workspace_id=workspace_id, root=root) / "dogfood-runs" / "latest.json"


def dogfood_run_artifact_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return (
        demand_validation_root(workspace_id=workspace_id, root=root)
        / "dogfood-runs"
        / "runs"
        / f"{timestamp_slug()}-dogfood-run.json"
    )


def interview_log_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return demand_validation_root(workspace_id=workspace_id, root=root) / "interviews.jsonl"


def interview_latest_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return demand_validation_root(workspace_id=workspace_id, root=root) / "interviews" / "latest.json"


def interview_artifact_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return (
        demand_validation_root(workspace_id=workspace_id, root=root)
        / "interviews"
        / "runs"
        / f"{timestamp_slug()}-interview.json"
    )


def demo_setup_latest_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return demand_validation_root(workspace_id=workspace_id, root=root) / "demo-setup" / "latest.json"


def demo_setup_artifact_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return (
        demand_validation_root(workspace_id=workspace_id, root=root)
        / "demo-setup"
        / "runs"
        / f"{timestamp_slug()}-demo-setup.json"
    )


def demand_validation_report_latest_json_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return demand_validation_root(workspace_id=workspace_id, root=root) / "reports" / "latest.json"


def demand_validation_report_latest_md_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return demand_validation_root(workspace_id=workspace_id, root=root) / "reports" / "latest.md"


def demand_validation_report_run_json_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    run_slug: str | None = None,
) -> Path:
    slug = run_slug or timestamp_slug()
    return (
        demand_validation_root(workspace_id=workspace_id, root=root)
        / "reports"
        / "runs"
        / f"{slug}-demand-validation.json"
    )


def demand_validation_report_run_md_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    run_slug: str | None = None,
) -> Path:
    slug = run_slug or timestamp_slug()
    return (
        demand_validation_root(workspace_id=workspace_id, root=root)
        / "reports"
        / "runs"
        / f"{slug}-demand-validation.md"
    )


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _mapping_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [cleaned for item in value if (cleaned := _clean_text(item)) is not None]


def _normalize_judgement(value: str) -> str:
    normalized = (_clean_text(value) or "").lower().replace("_", "-")
    if normalized not in DEMAND_VALIDATION_JUDGEMENTS:
        raise ValueError(f"Unsupported judgement `{value}`.")
    return normalized


def _judgement_bool(value: str) -> bool | None:
    if value == "yes":
        return True
    if value == "no":
        return False
    return None


def _path_from_text(path: str | Path, *, root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _workspace_relative(path: Path, *, root: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_file_record(path: Path, *, role: str, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{role} must be a readable file: `{path}`.")
    return {
        "role": role,
        "kind": "file",
        "path": str(path),
        "workspace_relative_path": _workspace_relative(path, root=root),
        "sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _read_json_mapping(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else None


def _log_header(
    *,
    schema_name: str,
    schema_version: int,
    workspace_id: str,
) -> dict[str, Any]:
    return {
        "schema_name": schema_name,
        "schema_version": schema_version,
        "workspace_id": workspace_id,
        "created_at_utc": timestamp_utc(),
    }


def _append_log(
    path: Path,
    payload: Mapping[str, Any],
    *,
    header_schema_name: str,
    header_schema_version: int,
    workspace_id: str,
) -> dict[str, Any]:
    _validate_log_appendable(
        path,
        header_schema_name=header_schema_name,
        header_schema_version=header_schema_version,
        workspace_id=workspace_id,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    _log_header(
                        schema_name=header_schema_name,
                        schema_version=header_schema_version,
                        workspace_id=workspace_id,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
    record = copy.deepcopy(dict(payload))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _validate_log_appendable(
    path: Path,
    *,
    header_schema_name: str,
    header_schema_version: int,
    workspace_id: str,
) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    header = _read_log_header(path)
    if (
        header.get("schema_name") != header_schema_name
        or header.get("schema_version") != header_schema_version
    ):
        raise ValueError(f"Unexpected validation log schema in `{path}`.")
    if header.get("workspace_id") != workspace_id:
        raise ValueError(f"Validation log `{path}` belongs to workspace `{header.get('workspace_id')}`.")


def _read_log_header(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        raise ValueError(f"Validation log `{path}` was empty.")
    header = json.loads(first_line)
    return dict(header) if isinstance(header, Mapping) else {}


def _read_log_records(
    path: Path,
    *,
    expected_schema_name: str,
    header_schema_name: str,
    header_schema_version: int,
    workspace_id: str,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    _validate_log_appendable(
        path,
        header_schema_name=header_schema_name,
        header_schema_version=header_schema_version,
        workspace_id=workspace_id,
    )
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            cleaned = line.strip()
            if not cleaned:
                continue
            payload = json.loads(cleaned)
            if isinstance(payload, Mapping) and payload.get("schema_name") == expected_schema_name:
                records.append(dict(payload))
    return records


def read_dogfood_runs(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    return _read_log_records(
        dogfood_run_log_path(workspace_id=workspace_id, root=root),
        expected_schema_name=DEMAND_VALIDATION_DOGFOOD_RUN_SCHEMA_NAME,
        header_schema_name=DEMAND_VALIDATION_DOGFOOD_RUN_LOG_SCHEMA_NAME,
        header_schema_version=DEMAND_VALIDATION_DOGFOOD_RUN_LOG_SCHEMA_VERSION,
        workspace_id=workspace_id,
    )


def read_interviews(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    return _read_log_records(
        interview_log_path(workspace_id=workspace_id, root=root),
        expected_schema_name=DEMAND_VALIDATION_INTERVIEW_SCHEMA_NAME,
        header_schema_name=DEMAND_VALIDATION_INTERVIEW_LOG_SCHEMA_NAME,
        header_schema_version=DEMAND_VALIDATION_INTERVIEW_LOG_SCHEMA_VERSION,
        workspace_id=workspace_id,
    )


def _read_recall_artifact(path: Path) -> dict[str, Any]:
    payload = _read_json_mapping(path)
    if payload is None:
        raise ValueError(f"Recall artifact is not readable: `{path}`.")
    if payload.get("schema_name") != "software-satellite-failure-memory-recall":
        raise ValueError(f"Recall artifact has an unexpected schema: `{path}`.")
    return payload


def _events_by_id_from_index_summary(index_summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    event_log_path = Path(str(index_summary["event_log_path"]))
    event_log = read_event_log(event_log_path)
    return {
        str(event.get("event_id")): dict(event)
        for event in event_log.get("events") or []
        if isinstance(event, Mapping) and _clean_text(event.get("event_id")) is not None
    }


def record_dogfood_validation_run(
    *,
    event_id: str,
    useful_recall: str,
    critical_false_evidence_count: int = 0,
    verdict_capture_seconds: float | None = None,
    recall_path: Path | None = None,
    note: str | None = None,
    notes_file: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> tuple[dict[str, Any], Path, Path, Path]:
    resolved_root = _resolve_root(root)
    cleaned_event_id = _clean_text(event_id)
    if cleaned_event_id is None:
        raise ValueError("Dogfood validation run requires --event.")
    if critical_false_evidence_count < 0:
        raise ValueError("--critical-false-evidence-count must be zero or greater.")
    if verdict_capture_seconds is not None and verdict_capture_seconds < 0:
        raise ValueError("--verdict-capture-seconds must be zero or greater.")

    judgement = _normalize_judgement(useful_recall)
    index_summary = rebuild_memory_index(root=resolved_root, workspace_id=workspace_id)
    events_by_id = _events_by_id_from_index_summary(index_summary)
    source_event = events_by_id.get(cleaned_event_id)
    if source_event is None:
        raise ValueError(f"Unknown event id `{cleaned_event_id}`.")

    resolved_recall_path = (
        _path_from_text(recall_path, root=resolved_root)
        if recall_path is not None
        else latest_recall_path(workspace_id=workspace_id, root=resolved_root)
    )
    recall = _read_recall_artifact(resolved_recall_path)
    recall_event_id = _clean_text(_mapping_dict(recall.get("request")).get("source_event_id"))
    if recall_event_id is None:
        raise ValueError("Recall artifact must include request.source_event_id for dogfood validation.")
    if recall_event_id != cleaned_event_id:
        raise ValueError(
            "Recall artifact source_event_id does not match the dogfood event. "
            f"Expected `{cleaned_event_id}`, got `{recall_event_id}`."
        )

    source_refs: dict[str, Any] = {
        "event_contract_check": build_event_contract_check(source_event, root=resolved_root),
        "recall_artifact_ref": _source_file_record(resolved_recall_path, role="failure_memory_recall", root=resolved_root),
    }
    if notes_file is not None:
        source_refs["notes_file_ref"] = _source_file_record(
            _path_from_text(notes_file, root=resolved_root),
            role="dogfood_run_notes",
            root=resolved_root,
        )

    bundle = _mapping_dict(recall.get("bundle"))
    selected_candidates = [
        item for item in bundle.get("selected_candidates") or [] if isinstance(item, Mapping)
    ]
    run_path = dogfood_run_artifact_path(workspace_id=workspace_id, root=resolved_root)
    latest_path = dogfood_run_latest_path(workspace_id=workspace_id, root=resolved_root)
    log_path = dogfood_run_log_path(workspace_id=workspace_id, root=resolved_root)
    _validate_log_appendable(
        log_path,
        header_schema_name=DEMAND_VALIDATION_DOGFOOD_RUN_LOG_SCHEMA_NAME,
        header_schema_version=DEMAND_VALIDATION_DOGFOOD_RUN_LOG_SCHEMA_VERSION,
        workspace_id=workspace_id,
    )
    payload = {
        "schema_name": DEMAND_VALIDATION_DOGFOOD_RUN_SCHEMA_NAME,
        "schema_version": DEMAND_VALIDATION_DOGFOOD_RUN_SCHEMA_VERSION,
        "dogfood_run_id": f"{workspace_id}:demand-validation:dogfood:{timestamp_slug()}",
        "workspace_id": workspace_id,
        "recorded_at_utc": timestamp_utc(),
        "event_id": cleaned_event_id,
        "useful_recall_judgement": {
            "value": judgement,
            "useful": _judgement_bool(judgement),
        },
        "critical_false_evidence_count": critical_false_evidence_count,
        "verdict_capture_seconds": (
            round(float(verdict_capture_seconds), 3) if verdict_capture_seconds is not None else None
        ),
        "note": _clean_text(note),
        "recall_summary": {
            "recall_path": str(resolved_recall_path),
            "selected_count": int(bundle.get("selected_count") or len(selected_candidates)),
            "risk_level": _clean_text(_mapping_dict(recall.get("risk_note")).get("level")),
            "source_event_id": recall_event_id,
        },
        "source_refs": source_refs,
        "guardrails": {
            "preview_only": True,
            "writes_training_data": False,
            "starts_training_job": False,
            "uses_live_provider_integration": False,
            "runs_pack_code": False,
        },
        "paths": {
            "dogfood_run_latest_path": str(latest_path),
            "dogfood_run_path": str(run_path),
            "dogfood_run_log_path": str(log_path),
        },
    }
    write_json(run_path, payload)
    write_json(latest_path, payload)
    _append_log(
        log_path,
        payload,
        header_schema_name=DEMAND_VALIDATION_DOGFOOD_RUN_LOG_SCHEMA_NAME,
        header_schema_version=DEMAND_VALIDATION_DOGFOOD_RUN_LOG_SCHEMA_VERSION,
        workspace_id=workspace_id,
    )
    return payload, latest_path, run_path, log_path


def record_external_user_interview(
    *,
    participant_label: str,
    recognized_pain: str,
    wants_to_try: str,
    notes_file: Path,
    note: str | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> tuple[dict[str, Any], Path, Path, Path]:
    resolved_root = _resolve_root(root)
    participant = _clean_text(participant_label)
    if participant is None:
        raise ValueError("External interview requires --participant.")
    pain_judgement = _normalize_judgement(recognized_pain)
    try_judgement = _normalize_judgement(wants_to_try)
    resolved_notes = _path_from_text(notes_file, root=resolved_root)
    notes_ref = _source_file_record(resolved_notes, role="external_user_interview_notes", root=resolved_root)

    run_path = interview_artifact_path(workspace_id=workspace_id, root=resolved_root)
    latest_path = interview_latest_path(workspace_id=workspace_id, root=resolved_root)
    log_path = interview_log_path(workspace_id=workspace_id, root=resolved_root)
    _validate_log_appendable(
        log_path,
        header_schema_name=DEMAND_VALIDATION_INTERVIEW_LOG_SCHEMA_NAME,
        header_schema_version=DEMAND_VALIDATION_INTERVIEW_LOG_SCHEMA_VERSION,
        workspace_id=workspace_id,
    )
    payload = {
        "schema_name": DEMAND_VALIDATION_INTERVIEW_SCHEMA_NAME,
        "schema_version": DEMAND_VALIDATION_INTERVIEW_SCHEMA_VERSION,
        "interview_id": f"{workspace_id}:demand-validation:interview:{timestamp_slug()}",
        "workspace_id": workspace_id,
        "recorded_at_utc": timestamp_utc(),
        "participant_label": participant,
        "recognized_pain": {
            "value": pain_judgement,
            "recognized": _judgement_bool(pain_judgement),
        },
        "wants_to_try": {
            "value": try_judgement,
            "wants_try": _judgement_bool(try_judgement),
        },
        "note": _clean_text(note),
        "source_refs": {
            "notes_file_ref": notes_ref,
        },
        "prompt_questions": [
            "Have you had this exact problem?",
            "Would you run this after an AI coding session?",
            "What would make it immediately useful?",
        ],
        "guardrails": {
            "records_interview_notes_only": True,
            "does_not_contact_users": True,
            "writes_training_data": False,
        },
        "paths": {
            "interview_latest_path": str(latest_path),
            "interview_path": str(run_path),
            "interview_log_path": str(log_path),
        },
    }
    write_json(run_path, payload)
    write_json(latest_path, payload)
    _append_log(
        log_path,
        payload,
        header_schema_name=DEMAND_VALIDATION_INTERVIEW_LOG_SCHEMA_NAME,
        header_schema_version=DEMAND_VALIDATION_INTERVIEW_LOG_SCHEMA_VERSION,
        workspace_id=workspace_id,
    )
    return payload, latest_path, run_path, log_path


def record_demo_setup_metric(
    *,
    clone_to_demo_minutes: float,
    note: str | None = None,
    notes_file: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    if clone_to_demo_minutes < 0:
        raise ValueError("--clone-to-demo-minutes must be zero or greater.")
    resolved_root = _resolve_root(root)
    source_refs: dict[str, Any] = {}
    if notes_file is not None:
        source_refs["notes_file_ref"] = _source_file_record(
            _path_from_text(notes_file, root=resolved_root),
            role="demo_setup_notes",
            root=resolved_root,
        )
    latest_path = demo_setup_latest_path(workspace_id=workspace_id, root=resolved_root)
    run_path = demo_setup_artifact_path(workspace_id=workspace_id, root=resolved_root)
    payload = {
        "schema_name": DEMAND_VALIDATION_SETUP_SCHEMA_NAME,
        "schema_version": DEMAND_VALIDATION_SETUP_SCHEMA_VERSION,
        "setup_id": f"{workspace_id}:demand-validation:demo-setup:{timestamp_slug()}",
        "workspace_id": workspace_id,
        "recorded_at_utc": timestamp_utc(),
        "clone_to_demo_minutes": round(float(clone_to_demo_minutes), 3),
        "note": _clean_text(note),
        "source_refs": source_refs,
        "guardrails": {
            "measures_local_demo_only": True,
            "writes_training_data": False,
            "uses_live_provider_integration": False,
        },
        "paths": {
            "demo_setup_latest_path": str(latest_path),
            "demo_setup_path": str(run_path),
        },
    }
    write_json(run_path, payload)
    write_json(latest_path, payload)
    return payload, latest_path, run_path


def _metric(
    *,
    key: str,
    label: str,
    target: str,
    observed: Any,
    status: str,
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "target": target,
        "observed": observed,
        "status": status,
        "detail": copy.deepcopy(dict(detail or {})),
    }


def _status_min_count(value: int, target: int) -> str:
    return "pass" if value >= target else "needs_data"


def _status_max_float(value: float | None, target: float) -> str:
    if value is None:
        return "needs_data"
    return "pass" if value <= target else "fail"


def _recall_run_records(*, workspace_id: str, root: Path) -> list[dict[str, Any]]:
    runs_root = failure_memory_root(workspace_id=workspace_id, root=root) / "recall" / "runs"
    records = []
    for path in sorted(runs_root.glob("*.json")):
        payload = _read_json_mapping(path)
        if payload and payload.get("schema_name") == "software-satellite-failure-memory-recall":
            record = dict(payload)
            record["_path"] = str(path)
            records.append(record)
    return records


def _human_verdict_signals(*, workspace_id: str, root: Path) -> list[dict[str, Any]]:
    signals = read_evaluation_signals(evaluation_signal_log_path(workspace_id=workspace_id, root=root))
    return [
        signal
        for signal in signals
        if "human-verdict" in _string_list(signal.get("tags"))
    ]


def _positive_judgement_count(records: Iterable[Mapping[str, Any]], key: str, positive_key: str) -> int:
    count = 0
    for record in records:
        judgement = _mapping_dict(record.get(key))
        if judgement.get(positive_key) is True:
            count += 1
    return count


def _judged_count(records: Iterable[Mapping[str, Any]], key: str, positive_key: str) -> int:
    count = 0
    for record in records:
        judgement = _mapping_dict(record.get(key))
        if isinstance(judgement.get(positive_key), bool):
            count += 1
    return count


def build_demand_validation_report(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    index_summary = rebuild_memory_index(root=resolved_root, workspace_id=workspace_id)
    workspace_events = iter_workspace_events(root=resolved_root, workspace_id=workspace_id)
    workspace_event_count = len(workspace_events)
    event_contract = build_event_contract_report(workspace_events, root=resolved_root)
    checked_event_count = int(event_contract.get("checked_event_count") or 0)
    source_counts = _mapping_dict(event_contract.get("source_status_counts"))
    missing_source_count = int(source_counts.get("missing_source") or 0)
    source_complete_count = max(0, checked_event_count - missing_source_count)
    source_completeness = (
        round(source_complete_count / checked_event_count, 4)
        if checked_event_count
        else None
    )

    recall_records = _recall_run_records(workspace_id=workspace_id, root=resolved_root)
    verdict_signals = _human_verdict_signals(workspace_id=workspace_id, root=resolved_root)
    dogfood_runs = read_dogfood_runs(workspace_id=workspace_id, root=resolved_root)
    interviews = read_interviews(workspace_id=workspace_id, root=resolved_root)
    setup = _read_json_mapping(demo_setup_latest_path(workspace_id=workspace_id, root=resolved_root))

    useful_count = _positive_judgement_count(dogfood_runs, "useful_recall_judgement", "useful")
    useful_judged_count = _judged_count(dogfood_runs, "useful_recall_judgement", "useful")
    useful_rate = round(useful_count / useful_judged_count, 4) if useful_judged_count else None
    false_evidence_count = sum(int(run.get("critical_false_evidence_count") or 0) for run in dogfood_runs)
    verdict_seconds = [
        float(run["verdict_capture_seconds"])
        for run in dogfood_runs
        if isinstance(run.get("verdict_capture_seconds"), (int, float))
    ]
    max_verdict_seconds = max(verdict_seconds) if verdict_seconds else None
    avg_verdict_seconds = (
        round(sum(verdict_seconds) / len(verdict_seconds), 3)
        if verdict_seconds
        else None
    )
    clone_minutes = (
        float(setup["clone_to_demo_minutes"])
        if isinstance(setup, Mapping) and isinstance(setup.get("clone_to_demo_minutes"), (int, float))
        else None
    )
    recognized_pain_count = _positive_judgement_count(interviews, "recognized_pain", "recognized")
    recognized_pain_judged_count = _judged_count(interviews, "recognized_pain", "recognized")
    wants_try_count = _positive_judgement_count(interviews, "wants_to_try", "wants_try")

    metrics = [
        _metric(
            key="dogfood_event_count",
            label="Workspace source-linked events",
            target=">= 10",
            observed=workspace_event_count,
            status=_status_min_count(workspace_event_count, 10),
            detail={"count_scope": "workspace_event_log"},
        ),
        _metric(
            key="failure_memory_recall_count",
            label="Failure-memory recall runs",
            target=">= 5",
            observed=len(recall_records),
            status=_status_min_count(len(recall_records), 5),
            detail={"count_scope": "failure_memory_recall_run_artifacts"},
        ),
        _metric(
            key="human_verdict_count",
            label="Human verdicts recorded",
            target=">= 5",
            observed=len(verdict_signals),
            status=_status_min_count(len(verdict_signals), 5),
            detail={"count_scope": "evaluation_signals_with_human_verdict_tag"},
        ),
        _metric(
            key="useful_recalled_evidence_rate",
            label="Useful recalled evidence rate",
            target=f">= {USEFUL_RECALL_TARGET:.2f}",
            observed=useful_rate,
            status=(
                "needs_data"
                if useful_judged_count == 0
                else "pass"
                if useful_rate is not None and useful_rate >= USEFUL_RECALL_TARGET
                else "fail"
            ),
            detail={
                "useful_count": useful_count,
                "judged_count": useful_judged_count,
                "dogfood_run_count": len(dogfood_runs),
            },
        ),
        _metric(
            key="critical_false_evidence",
            label="Critical false evidence",
            target="0",
            observed=false_evidence_count,
            status=(
                "needs_data"
                if not dogfood_runs
                else "pass"
                if false_evidence_count == 0
                else "fail"
            ),
        ),
        _metric(
            key="source_path_completeness",
            label="Source path completeness",
            target=f">= {SOURCE_COMPLETENESS_TARGET:.2f}",
            observed=source_completeness,
            status=(
                "needs_data"
                if checked_event_count == 0
                else "pass"
                if source_completeness is not None and source_completeness >= SOURCE_COMPLETENESS_TARGET
                else "fail"
            ),
            detail={
                "checked_event_count": checked_event_count,
                "missing_source_event_count": missing_source_count,
            },
        ),
        _metric(
            key="human_verdict_capture_friction",
            label="Human verdict capture friction",
            target=f"<= {VERDICT_CAPTURE_SECONDS_TARGET:.0f} seconds",
            observed=max_verdict_seconds,
            status=_status_max_float(max_verdict_seconds, VERDICT_CAPTURE_SECONDS_TARGET),
            detail={
                "sample_count": len(verdict_seconds),
                "average_seconds": avg_verdict_seconds,
            },
        ),
        _metric(
            key="clone_to_demo_time",
            label="Clone-to-demo time",
            target=f"<= {CLONE_TO_DEMO_MINUTES_TARGET:.0f} minutes",
            observed=clone_minutes,
            status=_status_max_float(clone_minutes, CLONE_TO_DEMO_MINUTES_TARGET),
        ),
        _metric(
            key="external_user_interview_count",
            label="External technical-user interviews",
            target=">= 3",
            observed=len(interviews),
            status=_status_min_count(len(interviews), 3),
        ),
        _metric(
            key="external_user_recognized_pain_count",
            label='External "I have this problem" count',
            target=">= 3",
            observed=recognized_pain_count,
            status=_status_min_count(recognized_pain_count, 3),
            detail={"judged_count": recognized_pain_judged_count},
        ),
        _metric(
            key="external_user_wants_try_count",
            label="External users who want to try it on a repo",
            target=">= 1",
            observed=wants_try_count,
            status=_status_min_count(wants_try_count, 1),
        ),
    ]
    metric_statuses = [metric["status"] for metric in metrics]
    if any(status == "fail" for status in metric_statuses):
        overall_status = "fail"
    elif all(status == "pass" for status in metric_statuses):
        overall_status = "pass"
    else:
        overall_status = "needs_data"

    metric_status_by_key = {
        str(metric.get("key")): _clean_text(metric.get("status")) or "needs_data"
        for metric in metrics
    }
    continuation_metric_keys = {
        "dogfood_event_count",
        "failure_memory_recall_count",
        "human_verdict_count",
        "useful_recalled_evidence_rate",
        "critical_false_evidence",
        "source_path_completeness",
        "human_verdict_capture_friction",
        "clone_to_demo_time",
        "external_user_interview_count",
        "external_user_recognized_pain_count",
        "external_user_wants_try_count",
    }
    continuation_metric_statuses = [
        metric_status_by_key.get(key, "needs_data")
        for key in sorted(continuation_metric_keys)
    ]
    if any(status == "fail" for status in continuation_metric_statuses):
        continuation_status = "fail"
    elif all(status == "pass" for status in continuation_metric_statuses):
        continuation_status = "pass"
    else:
        continuation_status = "needs_data"

    next_actions = _build_next_actions(metrics)
    report_latest_json = demand_validation_report_latest_json_path(workspace_id=workspace_id, root=resolved_root)
    report_latest_md = demand_validation_report_latest_md_path(workspace_id=workspace_id, root=resolved_root)
    report_run_slug = timestamp_slug()
    report_run_json = demand_validation_report_run_json_path(
        workspace_id=workspace_id,
        root=resolved_root,
        run_slug=report_run_slug,
    )
    report_run_md = demand_validation_report_run_md_path(
        workspace_id=workspace_id,
        root=resolved_root,
        run_slug=report_run_slug,
    )
    return {
        "schema_name": DEMAND_VALIDATION_REPORT_SCHEMA_NAME,
        "schema_version": DEMAND_VALIDATION_REPORT_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "generated_at_utc": timestamp_utc(),
        "overall_status": overall_status,
        "continuation_gate": {
            "status": continuation_status,
            "requires": [
                "10 source-linked events in the validation workspace",
                "5 failure-memory recall runs",
                "5 human verdicts",
                "useful recall rate >= 30%",
                "critical false evidence = 0",
                "source path completeness >= 90%",
                "human verdict capture friction <= 30 seconds",
                "clone-to-demo time <= 15 minutes",
                "3 external technical-user interviews",
                "at least 3 external users recognize the exact pain",
                "at least 1 user wants to try it on a repo",
            ],
        },
        "metrics": metrics,
        "counts": {
            "event_count": workspace_event_count,
            "indexed_event_count": int(index_summary.get("event_count") or 0),
            "recall_run_count": len(recall_records),
            "human_verdict_count": len(verdict_signals),
            "dogfood_run_count": len(dogfood_runs),
            "interview_count": len(interviews),
        },
        "dogfood_runs": [_compact_dogfood_run(run) for run in dogfood_runs[-10:]],
        "external_interviews": [_compact_interview(interview) for interview in interviews[-10:]],
        "guardrails": {
            "preview_only": True,
            "writes_training_data": False,
            "starts_training_job": False,
            "uses_live_provider_integration": False,
            "runs_arbitrary_pack_code": False,
            "uses_vector_search": False,
        },
        "next_actions": next_actions,
        "paths": {
            "event_log_path": str(index_summary["event_log_path"]),
            "dogfood_run_log_path": str(dogfood_run_log_path(workspace_id=workspace_id, root=resolved_root)),
            "interview_log_path": str(interview_log_path(workspace_id=workspace_id, root=resolved_root)),
            "report_latest_json_path": str(report_latest_json),
            "report_latest_md_path": str(report_latest_md),
            "report_run_json_path": str(report_run_json),
            "report_run_md_path": str(report_run_md),
        },
    }


def _build_next_actions(metrics: Iterable[Mapping[str, Any]]) -> list[str]:
    by_key = {str(metric.get("key")): metric for metric in metrics}
    actions: list[str] = []

    def observed_int(key: str) -> int:
        value = by_key.get(key, {}).get("observed")
        return int(value) if isinstance(value, int) else 0

    missing_events = max(0, 10 - observed_int("dogfood_event_count"))
    if missing_events:
        actions.append(f"Record {missing_events} more source-linked dogfood events.")
    missing_recalls = max(0, 5 - observed_int("failure_memory_recall_count"))
    if missing_recalls:
        actions.append(f"Run {missing_recalls} more failure-memory recalls.")
    missing_verdicts = max(0, 5 - observed_int("human_verdict_count"))
    if missing_verdicts:
        actions.append(f"Record {missing_verdicts} more human verdicts.")

    useful_detail = _mapping_dict(by_key.get("useful_recalled_evidence_rate", {}).get("detail"))
    missing_judgements = max(0, 5 - int(useful_detail.get("judged_count") or 0))
    if missing_judgements:
        actions.append(f"Record useful-recall judgement for {missing_judgements} more dogfood runs.")

    if by_key.get("critical_false_evidence", {}).get("status") == "fail":
        actions.append("Stop and inspect critical false-evidence cases before public demo use.")
    if by_key.get("source_path_completeness", {}).get("status") == "fail":
        actions.append("Fix missing source paths before treating evidence as positive validation.")
    if by_key.get("human_verdict_capture_friction", {}).get("status") == "fail":
        actions.append("Reduce verdict capture flow to stay under 30 seconds per verdict.")
    if by_key.get("clone_to_demo_time", {}).get("status") in {"needs_data", "fail"}:
        actions.append("Record a timed clone-to-demo setup run under 15 minutes.")

    missing_interviews = max(0, 3 - observed_int("external_user_interview_count"))
    if missing_interviews:
        actions.append(f"Record {missing_interviews} more external technical-user interviews.")
    missing_problem = max(0, 3 - observed_int("external_user_recognized_pain_count"))
    if missing_problem:
        actions.append(f"Find {missing_problem} more external users who recognize the exact problem.")
    if observed_int("external_user_wants_try_count") < 1:
        actions.append("Get at least one external user who wants to try it on a repo.")
    return actions


def _compact_dogfood_run(run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dogfood_run_id": _clean_text(run.get("dogfood_run_id")),
        "recorded_at_utc": _clean_text(run.get("recorded_at_utc")),
        "event_id": _clean_text(run.get("event_id")),
        "useful_recall": _clean_text(_mapping_dict(run.get("useful_recall_judgement")).get("value")),
        "critical_false_evidence_count": int(run.get("critical_false_evidence_count") or 0),
        "verdict_capture_seconds": run.get("verdict_capture_seconds"),
        "recall_path": _clean_text(_mapping_dict(run.get("recall_summary")).get("recall_path")),
    }


def _compact_interview(interview: Mapping[str, Any]) -> dict[str, Any]:
    source_refs = _mapping_dict(interview.get("source_refs"))
    notes_ref = _mapping_dict(source_refs.get("notes_file_ref"))
    return {
        "interview_id": _clean_text(interview.get("interview_id")),
        "recorded_at_utc": _clean_text(interview.get("recorded_at_utc")),
        "participant_label": _clean_text(interview.get("participant_label")),
        "recognized_pain": _clean_text(_mapping_dict(interview.get("recognized_pain")).get("value")),
        "wants_to_try": _clean_text(_mapping_dict(interview.get("wants_to_try")).get("value")),
        "notes_path": _clean_text(notes_ref.get("path")),
    }


def record_demand_validation_report(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> tuple[dict[str, Any], str, Path, Path, Path, Path]:
    report = build_demand_validation_report(workspace_id=workspace_id, root=root)
    markdown = format_demand_validation_report(report)
    paths = _mapping_dict(report.get("paths"))
    latest_json = Path(str(paths["report_latest_json_path"]))
    latest_md = Path(str(paths["report_latest_md_path"]))
    run_json = Path(str(paths["report_run_json_path"]))
    run_md = Path(str(paths["report_run_md_path"]))
    write_json(run_json, report)
    write_json(latest_json, report)
    run_md.parent.mkdir(parents=True, exist_ok=True)
    latest_md.parent.mkdir(parents=True, exist_ok=True)
    run_md.write_text(markdown + "\n", encoding="utf-8")
    latest_md.write_text(markdown + "\n", encoding="utf-8")
    return report, markdown, latest_json, latest_md, run_json, run_md


def _format_observed(value: Any) -> str:
    if value is None:
        return "not recorded"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def format_demand_validation_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Demand Validation Report",
        "",
        f"Workspace: {_clean_text(report.get('workspace_id')) or DEFAULT_WORKSPACE_ID}",
        f"Generated: {_clean_text(report.get('generated_at_utc')) or ''}",
        f"Overall status: {_clean_text(report.get('overall_status')) or 'needs_data'}",
        f"Continuation gate: {_clean_text(_mapping_dict(report.get('continuation_gate')).get('status')) or 'needs_data'}",
        "",
        "## Metrics",
        "",
        "| Metric | Observed | Target | Status |",
        "|---|---:|---:|---|",
    ]
    for metric in report.get("metrics") or []:
        if not isinstance(metric, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    _clean_text(metric.get("label")) or "",
                    _format_observed(metric.get("observed")),
                    _clean_text(metric.get("target")) or "",
                    _clean_text(metric.get("status")) or "",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Recent Dogfood Runs",
            "",
        ]
    )
    dogfood_runs = [run for run in report.get("dogfood_runs") or [] if isinstance(run, Mapping)]
    if dogfood_runs:
        lines.extend(["| Event | Useful recall | False evidence | Verdict seconds |", "|---|---|---:|---:|"])
        for run in dogfood_runs:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _clean_text(run.get("event_id")) or "",
                        _clean_text(run.get("useful_recall")) or "unclear",
                        str(int(run.get("critical_false_evidence_count") or 0)),
                        _format_observed(run.get("verdict_capture_seconds")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No dogfood validation runs recorded yet.")

    lines.extend(["", "## External Interviews", ""])
    interviews = [item for item in report.get("external_interviews") or [] if isinstance(item, Mapping)]
    if interviews:
        lines.extend(["| Participant | Recognized pain | Wants to try | Notes |", "|---|---|---|---|"])
        for interview in interviews:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _clean_text(interview.get("participant_label")) or "",
                        _clean_text(interview.get("recognized_pain")) or "unclear",
                        _clean_text(interview.get("wants_to_try")) or "unclear",
                        _clean_text(interview.get("notes_path")) or "",
                    ]
                )
                + " |"
            )
    else:
        lines.append("No external-user interviews recorded yet.")

    lines.extend(["", "## Next Actions", ""])
    actions = _string_list(report.get("next_actions"))
    if actions:
        lines.extend(f"- {action}" for action in actions)
    else:
        lines.append("- Demand validation gates are satisfied for this local report.")

    paths = _mapping_dict(report.get("paths"))
    lines.extend(
        [
            "",
            "## Artifact Paths",
            "",
            f"- Event log: {_clean_text(paths.get('event_log_path')) or ''}",
            f"- Dogfood ledger: {_clean_text(paths.get('dogfood_run_log_path')) or ''}",
            f"- Interview ledger: {_clean_text(paths.get('interview_log_path')) or ''}",
        ]
    )
    return "\n".join(lines)


def format_dogfood_validation_run(result: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "Dogfood validation run recorded",
            f"Event: {_clean_text(result.get('event_id')) or ''}",
            f"Useful recall: {_clean_text(_mapping_dict(result.get('useful_recall_judgement')).get('value')) or 'unclear'}",
            f"Critical false evidence: {int(result.get('critical_false_evidence_count') or 0)}",
            f"Verdict capture seconds: {_format_observed(result.get('verdict_capture_seconds'))}",
            f"Ledger: {_clean_text(_mapping_dict(result.get('paths')).get('dogfood_run_log_path')) or ''}",
        ]
    )


def format_external_user_interview(result: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "External interview recorded",
            f"Participant: {_clean_text(result.get('participant_label')) or ''}",
            f"Recognized pain: {_clean_text(_mapping_dict(result.get('recognized_pain')).get('value')) or 'unclear'}",
            f"Wants to try: {_clean_text(_mapping_dict(result.get('wants_to_try')).get('value')) or 'unclear'}",
            f"Ledger: {_clean_text(_mapping_dict(result.get('paths')).get('interview_log_path')) or ''}",
        ]
    )


def format_demo_setup_metric(result: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "Demo setup metric recorded",
            f"Clone-to-demo minutes: {_format_observed(result.get('clone_to_demo_minutes'))}",
            f"Artifact: {_clean_text(_mapping_dict(result.get('paths')).get('demo_setup_path')) or ''}",
        ]
    )


def demand_validation_templates_markdown() -> str:
    return "\n".join(
        [
            "# Demand Validation Templates",
            "",
            "## Dogfood Run Notes",
            "",
            "- Event id:",
            "- Recall artifact path:",
            "- Was recalled evidence useful? yes/no/unclear",
            "- Critical false evidence count:",
            "- Verdict capture seconds:",
            "- Why it did or did not help:",
            "",
            "## External Technical-User Interview Notes",
            "",
            "- Participant label:",
            "- Have you had this exact problem? yes/no/unclear",
            "- Would you run this after an AI coding session? yes/no/unclear",
            "- What would make it immediately useful?",
            "- Repo or workflow context:",
            "",
            "## Setup Timing Notes",
            "",
            "- Start time:",
            "- End time:",
            "- Clone-to-demo minutes:",
            "- Friction observed:",
            "",
            "## CLI Wiring",
            "",
            "```bash",
            "python scripts/satlab.py validation record-run --event <event-id> --useful-recall yes --critical-false-evidence-count 0 --verdict-capture-seconds 20 --notes-file dogfood-run.md",
            "python scripts/satlab.py validation record-interview --participant user-1 --recognized-pain yes --wants-to-try yes --notes-file interview-user-1.md",
            "python scripts/satlab.py validation record-setup --clone-to-demo-minutes 12 --notes-file setup-timing.md",
            "python scripts/satlab.py validation report --format md",
            "```",
        ]
    )


def write_demand_validation_templates(output_dir: Path, *, root: Path | None = None) -> dict[str, str]:
    resolved_root = _resolve_root(root)
    resolved_output_dir = _path_from_text(output_dir, root=resolved_root)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    templates = {
        "dogfood_run_notes": "\n".join(
            [
                "# Dogfood Run Notes",
                "",
                "- Event id:",
                "- Recall artifact path:",
                "- Was recalled evidence useful? yes/no/unclear",
                "- Critical false evidence count:",
                "- Verdict capture seconds:",
                "- Why it did or did not help:",
                "",
            ]
        ),
        "external_user_interview": "\n".join(
            [
                "# External Technical-User Interview Notes",
                "",
                "- Participant label:",
                "- Have you had this exact problem? yes/no/unclear",
                "- Would you run this after an AI coding session? yes/no/unclear",
                "- What would make it immediately useful?",
                "- Repo or workflow context:",
                "",
            ]
        ),
        "setup_timing": "\n".join(
            [
                "# Setup Timing Notes",
                "",
                "- Start time:",
                "- End time:",
                "- Clone-to-demo minutes:",
                "- Friction observed:",
                "",
            ]
        ),
    }
    paths: dict[str, str] = {}
    for name, content in templates.items():
        path = resolved_output_dir / f"{name}.md"
        path.write_text(content, encoding="utf-8")
        paths[name] = str(path)
    return paths
