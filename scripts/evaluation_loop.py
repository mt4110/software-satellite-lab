#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable as IterableABC
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from gemma_runtime import repo_root, timestamp_slug, timestamp_utc, write_json
from memory_index import rebuild_memory_index
from software_work_events import read_event_log
from workspace_state import DEFAULT_WORKSPACE_ID


EVALUATION_SIGNAL_SCHEMA_NAME = "software-satellite-evaluation-signal"
EVALUATION_SIGNAL_SCHEMA_VERSION = 1
EVALUATION_SIGNAL_LOG_SCHEMA_NAME = "software-satellite-evaluation-signal-log"
EVALUATION_SIGNAL_LOG_SCHEMA_VERSION = 1
EVALUATION_COMPARISON_SCHEMA_NAME = "software-satellite-evaluation-comparison"
EVALUATION_COMPARISON_SCHEMA_VERSION = 1
EVALUATION_COMPARISON_LOG_SCHEMA_NAME = "software-satellite-evaluation-comparison-log"
EVALUATION_COMPARISON_LOG_SCHEMA_VERSION = 1
EVALUATION_SNAPSHOT_SCHEMA_NAME = "software-satellite-evaluation-snapshot"
EVALUATION_SNAPSHOT_SCHEMA_VERSION = 1
CURATION_EXPORT_PREVIEW_SCHEMA_NAME = "software-satellite-curation-export-preview"
CURATION_EXPORT_PREVIEW_SCHEMA_VERSION = 1

SIGNAL_KINDS = ("acceptance", "rejection", "test_pass", "test_fail", "review_resolved", "review_unresolved")
POSITIVE_SIGNAL_KINDS = {"acceptance", "test_pass", "review_resolved"}
NEGATIVE_SIGNAL_KINDS = {"rejection", "test_fail", "review_unresolved"}
RELATION_KINDS = ("repairs", "follow_up_for")
COMPARISON_OUTCOMES = ("winner_selected", "tie", "needs_follow_up")
CURATION_STATES = ("ready", "needs_review", "blocked")
CURATION_EXPORT_DECISIONS = ("include_when_approved", "hold_for_review", "exclude_until_repaired")

PASS_QUALITY_STATUSES = {"pass", "passed", "quality_pass"}
FAIL_QUALITY_STATUSES = {"fail", "failed", "quality_fail"}
FAIL_EXECUTION_STATUSES = {"fail", "failed", "blocked", "error", "timeout"}


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def evaluation_root(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return _resolve_root(root) / "artifacts" / "evaluation" / workspace_id


def evaluation_signal_log_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return evaluation_root(workspace_id=workspace_id, root=root) / "signals.jsonl"


def evaluation_comparison_log_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return evaluation_root(workspace_id=workspace_id, root=root) / "comparisons.jsonl"


def evaluation_snapshot_latest_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return evaluation_root(workspace_id=workspace_id, root=root) / "snapshots" / "latest.json"


def evaluation_snapshot_run_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return evaluation_root(workspace_id=workspace_id, root=root) / "snapshots" / "runs" / f"{timestamp_slug()}-evaluation.json"


def curation_export_preview_latest_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return evaluation_root(workspace_id=workspace_id, root=root) / "curation" / "preview-latest.json"


def curation_export_preview_run_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return (
        evaluation_root(workspace_id=workspace_id, root=root)
        / "curation"
        / "runs"
        / f"{timestamp_slug()}-curation-preview.json"
    )


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _timestamp_sort_key(value: Any) -> tuple[bool, float, str]:
    text = _clean_text(value)
    if text is None:
        return (False, 0.0, "")
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return (False, 0.0, text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (True, parsed.timestamp(), text)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [cleaned for item in value if (cleaned := _clean_text(item)) is not None]


def _mapping_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _compact_evidence(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        str(key): copy.deepcopy(item)
        for key, item in _mapping_dict(value).items()
        if item not in (None, "", [], {})
    }


def _signal_polarity(signal_kind: str) -> str:
    if signal_kind in POSITIVE_SIGNAL_KINDS:
        return "positive"
    if signal_kind in NEGATIVE_SIGNAL_KINDS:
        return "negative"
    return "neutral"


def _normalize_signal_kind(signal_kind: str) -> str:
    normalized = (signal_kind or "").strip().lower().replace("-", "_")
    if normalized not in SIGNAL_KINDS:
        raise ValueError(f"Unsupported evaluation signal kind `{signal_kind}`.")
    return normalized


def _normalize_relation_kind(relation_kind: str | None) -> str | None:
    normalized = _clean_text(relation_kind)
    if normalized is None:
        return None
    normalized = normalized.lower().replace("-", "_")
    if normalized not in RELATION_KINDS:
        raise ValueError(f"Unsupported evaluation relation kind `{relation_kind}`.")
    return normalized


def _normalize_comparison_outcome(outcome: str | None, *, winner_event_id: str | None) -> str:
    normalized = _clean_text(outcome)
    if normalized is None:
        return "winner_selected" if winner_event_id is not None else "needs_follow_up"
    normalized = normalized.lower().replace("-", "_")
    if normalized not in COMPARISON_OUTCOMES:
        raise ValueError(f"Unsupported evaluation comparison outcome `{outcome}`.")
    return normalized


def _event_source_record(event: Mapping[str, Any] | None, *, source_event_id: str) -> dict[str, Any]:
    event_payload = _mapping_dict(event)
    session = _mapping_dict(event_payload.get("session"))
    outcome = _mapping_dict(event_payload.get("outcome"))
    content = _mapping_dict(event_payload.get("content"))
    source_refs = _mapping_dict(event_payload.get("source_refs"))
    artifact_ref = _mapping_dict(source_refs.get("artifact_ref"))
    return {
        "source_event_id": source_event_id,
        "event_kind": _clean_text(event_payload.get("event_kind")),
        "session_id": _clean_text(session.get("session_id")),
        "session_surface": _clean_text(session.get("surface")),
        "model_id": _clean_text(session.get("selected_model_id")) or _clean_text(outcome.get("model_id")),
        "status": _clean_text(outcome.get("status")),
        "quality_status": _clean_text(outcome.get("quality_status")),
        "execution_status": _clean_text(outcome.get("execution_status")),
        "artifact_path": _clean_text(artifact_ref.get("artifact_path")),
        "prompt_excerpt": _clean_text(content.get("prompt")),
    }


def _validate_evaluation_signal(signal: Mapping[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    payload = copy.deepcopy(dict(signal))
    location = f" in `{path}`" if path is not None else ""
    if payload.get("schema_name") != EVALUATION_SIGNAL_SCHEMA_NAME:
        raise ValueError(f"Unexpected evaluation signal schema name{location}.")
    if payload.get("schema_version") != EVALUATION_SIGNAL_SCHEMA_VERSION:
        raise ValueError(f"Unsupported evaluation signal schema version{location}.")
    payload["signal_kind"] = _normalize_signal_kind(str(payload.get("signal_kind") or ""))
    payload["polarity"] = _signal_polarity(payload["signal_kind"])

    source = _mapping_dict(payload.get("source"))
    source_event_id = _clean_text(source.get("source_event_id"))
    if source_event_id is None:
        raise ValueError(f"Evaluation signal is missing source.source_event_id{location}.")
    source["source_event_id"] = source_event_id
    payload["source"] = source

    relation = _mapping_dict(payload.get("relation"))
    relation_kind = _clean_text(relation.get("relation_kind"))
    if relation_kind is not None:
        relation["relation_kind"] = _normalize_relation_kind(relation_kind)
        target_event_id = _clean_text(relation.get("target_event_id"))
        if target_event_id is None:
            raise ValueError(f"Evaluation relation signal is missing relation.target_event_id{location}.")
        relation["target_event_id"] = target_event_id
    payload["relation"] = relation
    return payload


def _comparison_log_header(*, workspace_id: str) -> dict[str, Any]:
    return {
        "schema_name": EVALUATION_COMPARISON_LOG_SCHEMA_NAME,
        "schema_version": EVALUATION_COMPARISON_LOG_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "created_at_utc": timestamp_utc(),
    }


def _read_comparison_log_header(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        raise ValueError(f"Evaluation comparison log `{path}` was empty.")
    header = json.loads(first_line)
    if header.get("schema_name") != EVALUATION_COMPARISON_LOG_SCHEMA_NAME:
        raise ValueError(f"Unexpected evaluation comparison log schema name in `{path}`.")
    if header.get("schema_version") != EVALUATION_COMPARISON_LOG_SCHEMA_VERSION:
        raise ValueError(f"Unsupported evaluation comparison log schema version in `{path}`.")
    return header


def _normalize_candidate_event_ids(candidate_event_ids: Iterable[str]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for item in candidate_event_ids:
        event_id = _clean_text(item)
        if event_id is None or event_id in seen:
            continue
        seen.add(event_id)
        candidates.append(event_id)
    if len(candidates) < 2:
        raise ValueError("Evaluation comparisons require at least two candidate event ids.")
    return candidates


def _comparison_candidate_record(
    event_id: str,
    *,
    events_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "source": _event_source_record(
            (events_by_id or {}).get(event_id),
            source_event_id=event_id,
        ),
    }


def build_evaluation_comparison(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    candidate_event_ids: Iterable[str],
    winner_event_id: str | None = None,
    outcome: str | None = None,
    comparison_id: str | None = None,
    recorded_at_utc: str | None = None,
    task_label: str | None = None,
    criteria: Iterable[str] | None = None,
    rationale: str | None = None,
    origin: str = "manual",
    events_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    candidates = _normalize_candidate_event_ids(candidate_event_ids)
    if events_by_id is not None:
        missing_event_ids = [event_id for event_id in candidates if event_id not in events_by_id]
        if missing_event_ids:
            raise ValueError(
                "Evaluation comparison candidate_event_id was not found in software-work events: "
                + ", ".join(missing_event_ids)
            )
    winner = _clean_text(winner_event_id)
    normalized_outcome = _normalize_comparison_outcome(outcome, winner_event_id=winner)
    if winner is not None and winner not in candidates:
        raise ValueError("Evaluation comparison winner_event_id must be one of the candidate_event_ids.")
    if normalized_outcome == "winner_selected" and winner is None:
        raise ValueError("Evaluation comparison winner_selected outcome requires winner_event_id.")

    return {
        "schema_name": EVALUATION_COMPARISON_SCHEMA_NAME,
        "schema_version": EVALUATION_COMPARISON_SCHEMA_VERSION,
        "comparison_id": comparison_id or f"{workspace_id}:compare:{timestamp_slug()}:{uuid4().hex[:8]}",
        "workspace_id": workspace_id,
        "recorded_at_utc": recorded_at_utc or timestamp_utc(),
        "origin": _clean_text(origin) or "manual",
        "task_label": _clean_text(task_label),
        "outcome": normalized_outcome,
        "winner_event_id": winner,
        "candidate_count": len(candidates),
        "candidates": [
            _comparison_candidate_record(event_id, events_by_id=events_by_id)
            for event_id in candidates
        ],
        "criteria": _string_list(list(criteria or [])),
        "rationale": _clean_text(rationale),
        "tags": _string_list(list(tags or [])),
    }


def _validate_evaluation_comparison(comparison: Mapping[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    payload = copy.deepcopy(dict(comparison))
    location = f" in `{path}`" if path is not None else ""
    if payload.get("schema_name") != EVALUATION_COMPARISON_SCHEMA_NAME:
        raise ValueError(f"Unexpected evaluation comparison schema name{location}.")
    if payload.get("schema_version") != EVALUATION_COMPARISON_SCHEMA_VERSION:
        raise ValueError(f"Unsupported evaluation comparison schema version{location}.")

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"Evaluation comparison is missing candidates{location}.")
    candidate_event_ids = [
        event_id
        for item in candidates
        if isinstance(item, Mapping)
        if (event_id := _clean_text(item.get("event_id"))) is not None
    ]
    normalized_candidate_event_ids = _normalize_candidate_event_ids(candidate_event_ids)
    winner = _clean_text(payload.get("winner_event_id"))
    payload["outcome"] = _normalize_comparison_outcome(payload.get("outcome"), winner_event_id=winner)
    if winner is not None and winner not in normalized_candidate_event_ids:
        raise ValueError(f"Evaluation comparison winner_event_id is not a candidate{location}.")
    if payload["outcome"] == "winner_selected" and winner is None:
        raise ValueError(f"Evaluation comparison winner_selected outcome is missing winner_event_id{location}.")
    payload["winner_event_id"] = winner
    payload["candidate_count"] = len(normalized_candidate_event_ids)
    payload["candidates"] = [
        item
        for item in candidates
        if isinstance(item, Mapping) and _clean_text(item.get("event_id")) in normalized_candidate_event_ids
    ]
    payload["criteria"] = _string_list(payload.get("criteria"))
    payload["tags"] = _string_list(payload.get("tags"))
    return payload


def append_evaluation_comparison(path: Path, comparison: Mapping[str, Any], *, workspace_id: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(_comparison_log_header(workspace_id=workspace_id), ensure_ascii=False) + "\n")
    else:
        header = _read_comparison_log_header(path)
        if header.get("workspace_id") != workspace_id:
            raise ValueError(f"Evaluation comparison log `{path}` belongs to workspace `{header.get('workspace_id')}`.")

    payload = _validate_evaluation_comparison(comparison)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def read_evaluation_comparisons(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    _read_comparison_log_header(path)
    comparisons: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            cleaned = line.strip()
            if not cleaned:
                continue
            comparisons.append(_validate_evaluation_comparison(json.loads(cleaned), path=path))
    return comparisons


def build_evaluation_signal(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    signal_kind: str,
    source_event_id: str,
    recorded_at_utc: str | None = None,
    signal_id: str | None = None,
    source_event: Mapping[str, Any] | None = None,
    target_event_id: str | None = None,
    relation_kind: str | None = None,
    rationale: str | None = None,
    evidence: Mapping[str, Any] | None = None,
    origin: str = "manual",
    tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    normalized_signal_kind = _normalize_signal_kind(signal_kind)
    normalized_relation_kind = _normalize_relation_kind(relation_kind)
    source_event_id_cleaned = _clean_text(source_event_id)
    if source_event_id_cleaned is None:
        raise ValueError("Evaluation signals require a source_event_id.")
    if normalized_relation_kind is not None and _clean_text(target_event_id) is None:
        raise ValueError("Evaluation relation signals require a target_event_id.")

    normalized_evidence = _compact_evidence(evidence)
    if rationale:
        normalized_evidence["rationale"] = rationale

    return {
        "schema_name": EVALUATION_SIGNAL_SCHEMA_NAME,
        "schema_version": EVALUATION_SIGNAL_SCHEMA_VERSION,
        "signal_id": signal_id or f"{workspace_id}:eval:{timestamp_slug()}:{uuid4().hex[:8]}",
        "workspace_id": workspace_id,
        "signal_kind": normalized_signal_kind,
        "polarity": _signal_polarity(normalized_signal_kind),
        "recorded_at_utc": recorded_at_utc or timestamp_utc(),
        "origin": _clean_text(origin) or "manual",
        "source": _event_source_record(source_event, source_event_id=source_event_id_cleaned),
        "relation": {
            "relation_kind": normalized_relation_kind,
            "target_event_id": _clean_text(target_event_id),
        },
        "evidence": copy.deepcopy(normalized_evidence),
        "tags": _string_list(list(tags or [])),
    }


def _signal_log_header(*, workspace_id: str) -> dict[str, Any]:
    return {
        "schema_name": EVALUATION_SIGNAL_LOG_SCHEMA_NAME,
        "schema_version": EVALUATION_SIGNAL_LOG_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "created_at_utc": timestamp_utc(),
    }


def append_evaluation_signal(path: Path, signal: Mapping[str, Any], *, workspace_id: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(_signal_log_header(workspace_id=workspace_id), ensure_ascii=False) + "\n")
    else:
        header = _read_signal_log_header(path)
        if header.get("workspace_id") != workspace_id:
            raise ValueError(f"Evaluation signal log `{path}` belongs to workspace `{header.get('workspace_id')}`.")

    payload = _validate_evaluation_signal(signal)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def _read_signal_log_header(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        raise ValueError(f"Evaluation signal log `{path}` was empty.")
    header = json.loads(first_line)
    if header.get("schema_name") != EVALUATION_SIGNAL_LOG_SCHEMA_NAME:
        raise ValueError(f"Unexpected evaluation signal log schema name in `{path}`.")
    if header.get("schema_version") != EVALUATION_SIGNAL_LOG_SCHEMA_VERSION:
        raise ValueError(f"Unsupported evaluation signal log schema version in `{path}`.")
    return header


def read_evaluation_signals(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    _read_signal_log_header(path)
    signals: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            cleaned = line.strip()
            if not cleaned:
                continue
            signals.append(_validate_evaluation_signal(json.loads(cleaned), path=path))
    return signals


def _quality_status(event: Mapping[str, Any]) -> str | None:
    outcome = _mapping_dict(event.get("outcome"))
    return _clean_text(outcome.get("quality_status"))


def _execution_status(event: Mapping[str, Any]) -> str | None:
    outcome = _mapping_dict(event.get("outcome"))
    return _clean_text(outcome.get("execution_status")) or _clean_text(outcome.get("status"))


def _event_options(event: Mapping[str, Any]) -> dict[str, Any]:
    content = _mapping_dict(event.get("content"))
    options = content.get("options")
    return _mapping_dict(options)


def _event_notes(event: Mapping[str, Any]) -> list[str]:
    content = _mapping_dict(event.get("content"))
    return _string_list(content.get("notes"))


def _event_has_test_shape(event: Mapping[str, Any]) -> bool:
    if _clean_text(event.get("event_kind")) == "capability_result":
        return True
    options = _event_options(event)
    return any(
        _clean_text(options.get(key))
        for key in ("validation_mode", "validation_command", "pass_definition")
    )


def derive_test_signal_from_event(event: Mapping[str, Any], *, workspace_id: str) -> dict[str, Any] | None:
    if not _event_has_test_shape(event):
        return None

    quality_status = (_quality_status(event) or "").lower()
    execution_status = (_execution_status(event) or "").lower()
    signal_kind: str | None = None
    if quality_status in PASS_QUALITY_STATUSES:
        signal_kind = "test_pass"
    elif quality_status in FAIL_QUALITY_STATUSES or execution_status in FAIL_EXECUTION_STATUSES:
        signal_kind = "test_fail"
    if signal_kind is None:
        return None

    event_id = _clean_text(event.get("event_id"))
    if event_id is None:
        return None

    options = _event_options(event)
    notes = _event_notes(event)
    evidence = _compact_evidence({
        "validation_mode": _clean_text(options.get("validation_mode")),
        "validation_command": _clean_text(options.get("validation_command")),
        "test_name": _clean_text(options.get("claim_scope")) or _clean_text(options.get("phase")),
        "pass_definition": _clean_text(options.get("pass_definition")),
        "failure_summary": notes[0] if signal_kind == "test_fail" and notes else None,
        "quality_checks": options.get("quality_checks") if isinstance(options.get("quality_checks"), list) else [],
    })
    return build_evaluation_signal(
        workspace_id=workspace_id,
        signal_id=f"{event_id}:derived:{signal_kind}",
        signal_kind=signal_kind,
        source_event_id=event_id,
        recorded_at_utc=_clean_text(event.get("recorded_at_utc")),
        source_event=event,
        evidence=evidence,
        origin="derived_from_software_work_event",
        tags=["derived", signal_kind],
    )


def _source_event_id(signal: Mapping[str, Any]) -> str | None:
    return _clean_text(_mapping_dict(signal.get("source")).get("source_event_id"))


def _target_event_id(signal: Mapping[str, Any]) -> str | None:
    relation = _mapping_dict(signal.get("relation"))
    return _clean_text(relation.get("target_event_id")) or _clean_text(signal.get("target_event_id"))


def _relation_kind(signal: Mapping[str, Any]) -> str | None:
    relation = _mapping_dict(signal.get("relation"))
    return _clean_text(relation.get("relation_kind")) or _clean_text(signal.get("relation_kind"))


def _enrich_explicit_signal(signal: Mapping[str, Any], events_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    source_event_id = _source_event_id(signal)
    source_event = events_by_id.get(source_event_id or "")
    if source_event is None:
        return copy.deepcopy(dict(signal))

    enriched = copy.deepcopy(dict(signal))
    source = _mapping_dict(enriched.get("source"))
    source.update({
        key: value
        for key, value in _event_source_record(
            source_event,
            source_event_id=source_event_id or str(source_event.get("event_id") or ""),
        ).items()
        if value not in (None, "")
    })
    enriched["source"] = source
    return enriched


def _enrich_comparison(comparison: Mapping[str, Any], events_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    enriched = copy.deepcopy(dict(comparison))
    candidates: list[dict[str, Any]] = []
    for item in enriched.get("candidates") or []:
        if not isinstance(item, Mapping):
            continue
        event_id = _clean_text(item.get("event_id"))
        if event_id is None:
            continue
        candidate = dict(item)
        candidate["source"] = _event_source_record(events_by_id.get(event_id), source_event_id=event_id)
        candidates.append(candidate)
    enriched["candidates"] = candidates
    enriched["candidate_count"] = len(candidates)
    return enriched


def _relation_targets(
    signals: Iterable[Mapping[str, Any]],
    *,
    relation_kind: str,
) -> set[str]:
    return {
        target_event_id
        for signal in signals
        if _relation_kind(signal) == relation_kind
        if (target_event_id := _target_event_id(signal)) is not None
    }


def _relation_link_count(
    signals: Iterable[Mapping[str, Any]],
    *,
    relation_kind: str,
) -> int:
    return sum(
        1
        for signal in signals
        if _relation_kind(signal) == relation_kind and _target_event_id(signal) is not None
    )


def _signal_summary(signal: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping_dict(signal.get("source"))
    evidence = _mapping_dict(signal.get("evidence"))
    return {
        "signal_id": _clean_text(signal.get("signal_id")),
        "signal_kind": _clean_text(signal.get("signal_kind")),
        "polarity": _clean_text(signal.get("polarity")),
        "origin": _clean_text(signal.get("origin")),
        "recorded_at_utc": _clean_text(signal.get("recorded_at_utc")),
        "source_event_id": _clean_text(source.get("source_event_id")),
        "target_event_id": _target_event_id(signal),
        "relation_kind": _relation_kind(signal),
        "event_kind": _clean_text(source.get("event_kind")),
        "session_surface": _clean_text(source.get("session_surface")),
        "status": _clean_text(source.get("status")),
        "quality_status": _clean_text(source.get("quality_status")),
        "execution_status": _clean_text(source.get("execution_status")),
        "artifact_path": _clean_text(source.get("artifact_path")),
        "prompt_excerpt": _clean_text(source.get("prompt_excerpt")),
        "rationale": _clean_text(evidence.get("rationale")),
        "failure_summary": _clean_text(evidence.get("failure_summary")),
        "test_name": _clean_text(evidence.get("test_name")),
        "validation_command": _clean_text(evidence.get("validation_command")),
        "review_id": _clean_text(evidence.get("review_id")),
        "review_url": _clean_text(evidence.get("review_url")),
        "resolution_summary": _clean_text(evidence.get("resolution_summary")),
    }


def software_work_events_by_id(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
) -> dict[str, dict[str, Any]]:
    resolved_root = _resolve_root(root)
    index_summary = rebuild_memory_index(root=resolved_root, workspace_id=workspace_id)
    event_log = read_event_log(Path(index_summary["event_log_path"]))
    return {
        str(event.get("event_id")): dict(event)
        for event in event_log.get("events") or []
        if isinstance(event, Mapping) and _clean_text(event.get("event_id")) is not None
    }


def _comparison_summary(comparison: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [
        item
        for item in comparison.get("candidates") or []
        if isinstance(item, Mapping)
    ]
    return {
        "comparison_id": _clean_text(comparison.get("comparison_id")),
        "origin": _clean_text(comparison.get("origin")),
        "recorded_at_utc": _clean_text(comparison.get("recorded_at_utc")),
        "task_label": _clean_text(comparison.get("task_label")),
        "outcome": _clean_text(comparison.get("outcome")),
        "winner_event_id": _clean_text(comparison.get("winner_event_id")),
        "candidate_count": len(candidates),
        "candidate_event_ids": [
            event_id
            for item in candidates
            if (event_id := _clean_text(item.get("event_id"))) is not None
        ],
        "criteria": _string_list(comparison.get("criteria")),
        "rationale": _clean_text(comparison.get("rationale")),
    }


def _curation_label(event_id: str, events_by_id: Mapping[str, Mapping[str, Any]]) -> str:
    event = events_by_id.get(event_id)
    if event is None:
        return event_id
    content = _mapping_dict(event.get("content"))
    return _clean_text(content.get("prompt")) or _clean_text(content.get("output_text")) or event_id


def _latest_review_signal_kinds(signal_summaries: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    review_signals = [
        dict(signal)
        for signal in signal_summaries
        if signal.get("signal_kind") in {"review_resolved", "review_unresolved"}
        if _clean_text(signal.get("source_event_id")) is not None
    ]
    review_signals.sort(
        key=lambda signal: (
            _timestamp_sort_key(signal.get("recorded_at_utc")),
            str(signal.get("signal_id") or ""),
        ),
        reverse=True,
    )
    latest: dict[str, str] = {}
    for signal in review_signals:
        event_id = _clean_text(signal.get("source_event_id"))
        signal_kind = _clean_text(signal.get("signal_kind"))
        if event_id is None or signal_kind is None or event_id in latest:
            continue
        latest[event_id] = signal_kind
    return latest


def build_curation_candidates(
    *,
    signal_summaries: Iterable[Mapping[str, Any]],
    comparison_summaries: Iterable[Mapping[str, Any]],
    events_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    signal_summary_list = [
        dict(signal)
        for signal in signal_summaries
        if isinstance(signal, Mapping)
    ]
    comparison_summary_list = [
        dict(comparison)
        for comparison in comparison_summaries
        if isinstance(comparison, Mapping)
    ]
    latest_review_kinds = _latest_review_signal_kinds(signal_summary_list)
    accepted = {
        event_id
        for signal in signal_summary_list
        if signal.get("signal_kind") == "acceptance"
        if (event_id := _clean_text(signal.get("source_event_id"))) is not None
    }
    review_resolved = {
        event_id
        for event_id, signal_kind in latest_review_kinds.items()
        if signal_kind == "review_resolved"
    }
    rejected = {
        event_id
        for signal in signal_summary_list
        if signal.get("signal_kind") == "rejection"
        if (event_id := _clean_text(signal.get("source_event_id"))) is not None
    }
    review_unresolved = {
        event_id
        for event_id, signal_kind in latest_review_kinds.items()
        if signal_kind == "review_unresolved"
    }
    passed = {
        event_id
        for signal in signal_summary_list
        if signal.get("signal_kind") == "test_pass"
        if (event_id := _clean_text(signal.get("source_event_id"))) is not None
    }
    failed = {
        event_id
        for signal in signal_summary_list
        if signal.get("signal_kind") == "test_fail"
        if (event_id := _clean_text(signal.get("source_event_id"))) is not None
    }
    winners = {
        event_id
        for comparison in comparison_summary_list
        if comparison.get("outcome") == "winner_selected"
        if (event_id := _clean_text(comparison.get("winner_event_id"))) is not None
    }
    candidate_ids = sorted(accepted | review_resolved | rejected | review_unresolved | passed | failed | winners)
    candidates: list[dict[str, Any]] = []
    for event_id in candidate_ids:
        reasons: list[str] = []
        if event_id in accepted:
            reasons.append("accepted")
        if event_id in review_resolved:
            reasons.append("review_resolved")
        if event_id in winners:
            reasons.append("comparison_winner")
        if event_id in passed:
            reasons.append("test_pass")
        if event_id in rejected:
            reasons.append("rejected")
        if event_id in review_unresolved:
            reasons.append("review_unresolved")
        if event_id in failed:
            reasons.append("test_fail")

        if event_id in rejected or event_id in review_unresolved or event_id in failed:
            state = "blocked"
        elif event_id in passed and (event_id in accepted or event_id in review_resolved or event_id in winners):
            state = "ready"
        else:
            state = "needs_review"
            if event_id not in passed:
                reasons.append("needs_test_signal")
            if event_id not in accepted and event_id not in review_resolved and event_id not in winners:
                reasons.append("needs_human_selection")

        candidates.append(
            {
                "event_id": event_id,
                "state": state,
                "reasons": reasons,
                "label": _curation_label(event_id, events_by_id),
            }
        )
    state_order = {state: index for index, state in enumerate(CURATION_STATES)}
    candidates.sort(
        key=lambda item: (
            state_order.get(str(item.get("state") or ""), 99),
            str(item.get("event_id") or ""),
        )
    )
    return candidates


def _sort_signals(signals: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [copy.deepcopy(dict(signal)) for signal in signals],
        key=lambda item: (
            _timestamp_sort_key(item.get("recorded_at_utc")),
            str(item.get("signal_id") or ""),
        ),
        reverse=True,
    )


def _deduplicate_signals_by_source_event(signals: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    for signal in _sort_signals(signals):
        source_event_id = _clean_text(signal.get("source_event_id"))
        if source_event_id is None:
            continue
        if source_event_id in seen_event_ids:
            continue
        seen_event_ids.add(source_event_id)
        deduplicated.append(signal)
    return deduplicated


def build_evaluation_snapshot(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    signal_log_path: Path | None = None,
    comparison_log_path: Path | None = None,
    index_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    resolved_index_summary = (
        dict(index_summary)
        if isinstance(index_summary, Mapping)
        else rebuild_memory_index(root=resolved_root, workspace_id=workspace_id)
    )
    event_log_path = Path(resolved_index_summary["event_log_path"])
    event_log = read_event_log(event_log_path)
    events = [dict(event) for event in event_log.get("events") or [] if isinstance(event, Mapping)]
    events_by_id = {
        str(event.get("event_id")): event
        for event in events
        if _clean_text(event.get("event_id")) is not None
    }

    resolved_signal_log_path = Path(signal_log_path or evaluation_signal_log_path(workspace_id=workspace_id, root=resolved_root))
    resolved_comparison_log_path = Path(comparison_log_path or evaluation_comparison_log_path(workspace_id=workspace_id, root=resolved_root))
    explicit_signals = [
        _enrich_explicit_signal(signal, events_by_id)
        for signal in read_evaluation_signals(resolved_signal_log_path)
    ]
    comparisons = [
        _enrich_comparison(comparison, events_by_id)
        for comparison in read_evaluation_comparisons(resolved_comparison_log_path)
    ]
    comparison_summaries = sorted(
        [_comparison_summary(comparison) for comparison in comparisons],
        key=lambda comparison: (
            _timestamp_sort_key(comparison.get("recorded_at_utc")),
            str(comparison.get("comparison_id") or ""),
        ),
        reverse=True,
    )
    derived_signals = [
        signal
        for event in events
        if (signal := derive_test_signal_from_event(event, workspace_id=workspace_id)) is not None
    ]
    signals = _sort_signals([*explicit_signals, *derived_signals])
    signal_summaries = [_signal_summary(signal) for signal in signals]

    counts = Counter(str(signal.get("signal_kind") or "") for signal in signals)
    review_signal_count = int(counts.get("review_resolved", 0)) + int(counts.get("review_unresolved", 0))
    failure_signals = [signal for signal in signal_summaries if signal.get("signal_kind") == "test_fail"]
    failure_event_ids = {
        str(signal.get("source_event_id"))
        for signal in failure_signals
        if signal.get("source_event_id")
    }
    repair_targets = _relation_targets(signal_summaries, relation_kind="repairs")
    follow_up_targets = _relation_targets(signal_summaries, relation_kind="follow_up_for")
    repaired_failure_event_ids = repair_targets & failure_event_ids
    follow_up_failure_event_ids = follow_up_targets & failure_event_ids
    addressed_failure_event_ids = repaired_failure_event_ids | follow_up_failure_event_ids
    unresolved_failure_signals = [
        signal
        for signal in failure_signals
        if str(signal.get("source_event_id") or "") not in addressed_failure_event_ids
    ]
    pending_failures = _deduplicate_signals_by_source_event(unresolved_failure_signals)
    curation_candidates = build_curation_candidates(
        signal_summaries=signal_summaries,
        comparison_summaries=comparison_summaries,
        events_by_id=events_by_id,
    )
    curation_state_counts = Counter(str(item.get("state") or "") for item in curation_candidates)
    comparison_counts = Counter(str(item.get("outcome") or "") for item in comparison_summaries)

    return {
        "schema_name": EVALUATION_SNAPSHOT_SCHEMA_NAME,
        "schema_version": EVALUATION_SNAPSHOT_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "generated_at_utc": timestamp_utc(),
        "event_count": len(events),
        "signal_count": len(signals),
        "derived_signal_count": len(derived_signals),
        "explicit_signal_count": len(explicit_signals),
        "paths": {
            "event_log_path": str(event_log_path),
            "index_path": str(resolved_index_summary["index_path"]),
            "signal_log_path": str(resolved_signal_log_path),
            "comparison_log_path": str(resolved_comparison_log_path),
        },
        "counts": {
            "acceptance": int(counts.get("acceptance", 0)),
            "rejection": int(counts.get("rejection", 0)),
            "review_resolved": int(counts.get("review_resolved", 0)),
            "review_unresolved": int(counts.get("review_unresolved", 0)),
            "review_resolution_rate": (
                round(int(counts.get("review_resolved", 0)) / review_signal_count, 4)
                if review_signal_count
                else None
            ),
            "test_pass": int(counts.get("test_pass", 0)),
            "test_fail": int(counts.get("test_fail", 0)),
            "repair_links": _relation_link_count(signal_summaries, relation_kind="repairs"),
            "follow_up_links": _relation_link_count(signal_summaries, relation_kind="follow_up_for"),
            "repaired_failures": len(repaired_failure_event_ids),
            "followed_up_failures": len(follow_up_failure_event_ids),
            "addressed_failures": len(addressed_failure_event_ids),
            "pending_failures": len(pending_failures),
            "comparisons": len(comparison_summaries),
            "comparison_winners": int(comparison_counts.get("winner_selected", 0)),
            "unresolved_comparisons": int(comparison_counts.get("needs_follow_up", 0)),
            "curation_ready": int(curation_state_counts.get("ready", 0)),
            "curation_needs_review": int(curation_state_counts.get("needs_review", 0)),
            "curation_blocked": int(curation_state_counts.get("blocked", 0)),
        },
        "recent_signals": signal_summaries[:12],
        "comparisons": comparison_summaries[:12],
        "failures": failure_signals[:12],
        "pending_failures": pending_failures[:12],
        "curation": {
            "candidate_count": len(curation_candidates),
            "candidate_limit": None,
            "ready_count": int(curation_state_counts.get("ready", 0)),
            "needs_review_count": int(curation_state_counts.get("needs_review", 0)),
            "blocked_count": int(curation_state_counts.get("blocked", 0)),
            "candidates": curation_candidates,
        },
        "index_summary": resolved_index_summary,
    }


def record_evaluation_snapshot(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    signal_log_path: Path | None = None,
    comparison_log_path: Path | None = None,
    index_summary: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    resolved_root = _resolve_root(root)
    snapshot = build_evaluation_snapshot(
        root=resolved_root,
        workspace_id=workspace_id,
        signal_log_path=signal_log_path,
        comparison_log_path=comparison_log_path,
        index_summary=index_summary,
    )
    latest_path = evaluation_snapshot_latest_path(workspace_id=workspace_id, root=resolved_root)
    run_path = evaluation_snapshot_run_path(workspace_id=workspace_id, root=resolved_root)
    snapshot["paths"]["snapshot_latest_path"] = str(latest_path)
    snapshot["paths"]["snapshot_run_path"] = str(run_path)
    write_json(run_path, snapshot)
    write_json(latest_path, snapshot)
    return snapshot, latest_path, run_path


def record_review_resolution_signal(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    source_event_id: str,
    resolved: bool,
    review_id: str | None = None,
    review_url: str | None = None,
    resolution_summary: str | None = None,
    origin: str = "manual",
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    event_id = _clean_text(source_event_id)
    if event_id is None:
        raise ValueError("Review-resolution signals require a source_event_id.")
    events_by_id = software_work_events_by_id(root=resolved_root, workspace_id=workspace_id)
    source_event = events_by_id.get(event_id)
    if source_event is None:
        raise ValueError(f"Unknown review-resolution source_event_id `{event_id}`.")
    signal = build_evaluation_signal(
        workspace_id=workspace_id,
        signal_kind="review_resolved" if resolved else "review_unresolved",
        source_event_id=event_id,
        source_event=source_event,
        rationale=_clean_text(resolution_summary),
        evidence={
            "review_id": _clean_text(review_id),
            "review_url": _clean_text(review_url),
            "resolution_summary": _clean_text(resolution_summary),
        },
        origin=origin,
        tags=["review-resolution", "resolved" if resolved else "unresolved"],
    )
    return append_evaluation_signal(
        evaluation_signal_log_path(workspace_id=workspace_id, root=resolved_root),
        signal,
        workspace_id=workspace_id,
    )


def _curation_export_decision(candidate: Mapping[str, Any]) -> str:
    state = _clean_text(candidate.get("state"))
    if state == "ready":
        return "include_when_approved"
    if state == "blocked":
        return "exclude_until_repaired"
    return "hold_for_review"


def _curation_candidate_adoption_checklist(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    reasons = set(_string_list(candidate.get("reasons")))
    blocked_by = {
        reason
        for reason in reasons
        if reason in {"rejected", "review_unresolved", "test_fail"}
    }
    human_selected = bool(reasons & {"accepted", "review_resolved", "comparison_winner"})
    items = [
        {
            "key": "test_pass_recorded",
            "label": "Test pass recorded",
            "status": "done" if "test_pass" in reasons else "pending",
            "required": True,
        },
        {
            "key": "human_selection_recorded",
            "label": "Human selection recorded",
            "status": "done" if human_selected else "pending",
            "required": True,
        },
        {
            "key": "no_blocking_signal",
            "label": "No blocking signal",
            "status": "blocked" if blocked_by else "done",
            "required": True,
            "blocked_by": sorted(blocked_by),
        },
        {
            "key": "export_policy_confirmed",
            "label": "Export policy confirmed",
            "status": "pending",
            "required": True,
        },
    ]
    return items


def _curation_candidate_ready_for_policy(candidate: Mapping[str, Any]) -> bool:
    checklist = _curation_candidate_adoption_checklist(candidate)
    return all(
        item.get("status") == "done"
        for item in checklist
        if item.get("required") and item.get("key") != "export_policy_confirmed"
    )


def _curation_required_next_steps(candidate: Mapping[str, Any]) -> list[str]:
    state = _clean_text(candidate.get("state"))
    reasons = set(_string_list(candidate.get("reasons")))
    if state == "ready" or _curation_candidate_ready_for_policy(candidate):
        return ["confirm_export_policy"]
    steps: list[str] = []
    if "needs_test_signal" in reasons:
        steps.append("record_test_pass")
    if "needs_human_selection" in reasons:
        steps.append("record_acceptance_or_review_resolution")
    if "test_fail" in reasons:
        steps.append("repair_or_follow_up_failure")
    if "rejected" in reasons:
        steps.append("replace_or_rework_candidate")
    if "review_unresolved" in reasons:
        steps.append("resolve_review_before_export")
    return steps or ["review_candidate"]


def _curation_preview_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    reasons = _string_list(candidate.get("reasons"))
    blocked_by = [
        reason
        for reason in reasons
        if reason in {"rejected", "review_unresolved", "test_fail"}
    ]
    checklist = _curation_candidate_adoption_checklist(candidate)
    return {
        "event_id": _clean_text(candidate.get("event_id")),
        "state": _clean_text(candidate.get("state")) or "needs_review",
        "label": _clean_text(candidate.get("label")),
        "reasons": reasons,
        "blocked_by": blocked_by,
        "export_decision": _curation_export_decision(candidate),
        "ready_for_policy": _curation_candidate_ready_for_policy(candidate),
        "adoption_checklist": checklist,
        "required_next_steps": _curation_required_next_steps(candidate),
    }


def _normalize_curation_preview_filter_values(value: Any, *, allowed: Iterable[str] | None = None) -> list[str]:
    if isinstance(value, str):
        raw_values: Iterable[str] = value.split(",")
    elif isinstance(value, IterableABC):
        raw_values = [str(item) for item in value]
    else:
        raw_values = []
    allowed_values = set(allowed or [])
    values: list[str] = []
    seen: set[str] = set()
    for item in raw_values:
        cleaned = (item or "").strip().lower().replace("-", "_")
        if not cleaned or cleaned == "all" or cleaned in seen:
            continue
        if allowed_values and cleaned not in allowed_values:
            raise ValueError(f"Unsupported curation preview filter value `{item}`.")
        seen.add(cleaned)
        values.append(cleaned)
    return values


def normalize_curation_preview_filters(filters: Mapping[str, Any] | None = None) -> dict[str, Any]:
    raw_filters = _mapping_dict(filters)
    limit_value = raw_filters.get("limit")
    limit: int | None = None
    if limit_value not in (None, ""):
        limit = int(limit_value)
        if limit <= 0:
            raise ValueError("Curation preview limit must be a positive integer.")
    return {
        "states": _normalize_curation_preview_filter_values(
            raw_filters.get("states") or raw_filters.get("state"),
            allowed=CURATION_STATES,
        ),
        "export_decisions": _normalize_curation_preview_filter_values(
            raw_filters.get("export_decisions") or raw_filters.get("export_decision"),
            allowed=CURATION_EXPORT_DECISIONS,
        ),
        "reasons": _normalize_curation_preview_filter_values(raw_filters.get("reasons") or raw_filters.get("reason")),
        "limit": limit,
    }


def _candidate_matches_curation_filters(candidate: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    state_filters = set(_string_list(filters.get("states")))
    decision_filters = set(_string_list(filters.get("export_decisions")))
    reason_filters = set(_string_list(filters.get("reasons")))
    state = _clean_text(candidate.get("state"))
    decision = _clean_text(candidate.get("export_decision"))
    reasons = set(_string_list(candidate.get("reasons")))
    if state_filters and state not in state_filters:
        return False
    if decision_filters and decision not in decision_filters:
        return False
    if reason_filters and not (reasons & reason_filters):
        return False
    return True


def _curation_adoption_checklist_counts(candidates: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for candidate in candidates:
        for item in candidate.get("adoption_checklist") or []:
            if not isinstance(item, Mapping):
                continue
            key = _clean_text(item.get("key"))
            status = _clean_text(item.get("status")) or "pending"
            if key is None:
                continue
            counts.setdefault(key, Counter())[status] += 1
    return {
        key: {
            "done": int(counter.get("done", 0)),
            "pending": int(counter.get("pending", 0)),
            "blocked": int(counter.get("blocked", 0)),
            "total": int(sum(counter.values())),
        }
        for key, counter in counts.items()
    }


def build_curation_export_preview(
    snapshot: Mapping[str, Any],
    *,
    workspace_id: str | None = None,
    filters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    curation = _mapping_dict(snapshot.get("curation"))
    all_candidates = [
        _curation_preview_candidate(candidate)
        for candidate in (curation.get("candidates") or [])
        if isinstance(candidate, Mapping)
    ]
    normalized_filters = normalize_curation_preview_filters(filters)
    matched_candidates = [
        candidate
        for candidate in all_candidates
        if _candidate_matches_curation_filters(candidate, normalized_filters)
    ]
    limit = normalized_filters.get("limit")
    candidates = matched_candidates[:limit] if isinstance(limit, int) else matched_candidates
    total_candidate_count = int(curation.get("candidate_count") or len(all_candidates))
    state_counts = Counter(str(candidate.get("state") or "needs_review") for candidate in matched_candidates)
    adoption_ready_count = sum(1 for candidate in matched_candidates if candidate.get("ready_for_policy"))
    paths = _mapping_dict(snapshot.get("paths"))
    resolved_workspace_id = _clean_text(workspace_id) or _clean_text(snapshot.get("workspace_id")) or DEFAULT_WORKSPACE_ID
    return {
        "schema_name": CURATION_EXPORT_PREVIEW_SCHEMA_NAME,
        "schema_version": CURATION_EXPORT_PREVIEW_SCHEMA_VERSION,
        "workspace_id": resolved_workspace_id,
        "generated_at_utc": timestamp_utc(),
        "export_mode": "preview_only",
        "training_export_ready": False,
        "source_snapshot_path": (
            _clean_text(paths.get("snapshot_run_path"))
            or _clean_text(paths.get("snapshot_latest_path"))
        ),
        "counts": {
            "candidate_count": total_candidate_count,
            "matched_candidate_count": len(matched_candidates),
            "previewed_candidate_count": len(candidates),
            "ready": int(state_counts.get("ready", 0)),
            "needs_review": int(state_counts.get("needs_review", 0)),
            "blocked": int(state_counts.get("blocked", 0)),
            "ready_for_policy": adoption_ready_count,
        },
        "adoption_checklist_counts": _curation_adoption_checklist_counts(matched_candidates),
        "filters": normalized_filters,
        "truncated": len(matched_candidates) > len(candidates),
        "candidates": candidates,
        "notes": [
            "Preview only; no training data export is written.",
            "Ready candidates still require an explicit export policy before downstream dataset creation.",
        ],
    }


def record_curation_export_preview(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    snapshot: Mapping[str, Any] | None = None,
    filters: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    resolved_root = _resolve_root(root)
    source_snapshot = snapshot or build_evaluation_snapshot(root=resolved_root, workspace_id=workspace_id)
    preview = build_curation_export_preview(source_snapshot, workspace_id=workspace_id, filters=filters)
    latest_path = curation_export_preview_latest_path(workspace_id=workspace_id, root=resolved_root)
    run_path = curation_export_preview_run_path(workspace_id=workspace_id, root=resolved_root)
    preview["paths"] = {
        "curation_preview_latest_path": str(latest_path),
        "curation_preview_run_path": str(run_path),
        "source_snapshot_path": _clean_text(preview.get("source_snapshot_path")),
    }
    write_json(run_path, preview)
    write_json(latest_path, preview)
    return preview, latest_path, run_path


def format_curation_export_preview_report(preview: Mapping[str, Any]) -> str:
    counts = _mapping_dict(preview.get("counts"))
    paths = _mapping_dict(preview.get("paths"))
    lines = [
        "Curation export preview: preview_only",
        f"Candidates: {int(counts.get('candidate_count') or 0)}",
        f"Matched: {int(counts.get('matched_candidate_count') or 0)}",
        f"Previewed: {int(counts.get('previewed_candidate_count') or 0)}",
        f"Ready: {int(counts.get('ready') or 0)}",
        f"Needs review: {int(counts.get('needs_review') or 0)}",
        f"Blocked: {int(counts.get('blocked') or 0)}",
        f"Ready for policy: {int(counts.get('ready_for_policy') or 0)}",
        "Training export ready: no",
    ]
    filters = _mapping_dict(preview.get("filters"))
    filter_parts: list[str] = []
    for key in ("states", "export_decisions", "reasons"):
        values = _string_list(filters.get(key))
        if values:
            filter_parts.append(f"{key}={','.join(values)}")
    if filters.get("limit"):
        filter_parts.append(f"limit={int(filters.get('limit') or 0)}")
    if filter_parts:
        lines.append(f"Filters: {'; '.join(filter_parts)}")
    if paths.get("curation_preview_latest_path"):
        lines.append(f"Preview: {paths['curation_preview_latest_path']}")
    checklist_counts = _mapping_dict(preview.get("adoption_checklist_counts"))
    if checklist_counts:
        checklist_parts: list[str] = []
        for key in ("test_pass_recorded", "human_selection_recorded", "no_blocking_signal", "export_policy_confirmed"):
            item = _mapping_dict(checklist_counts.get(key))
            total = int(item.get("total") or 0)
            if not total:
                continue
            checklist_parts.append(f"{key}={int(item.get('done') or 0)}/{total}")
        if checklist_parts:
            lines.append(f"Adoption checklist: {'; '.join(checklist_parts)}")
    candidates = [
        item
        for item in (preview.get("candidates") or [])
        if isinstance(item, Mapping)
    ]
    if candidates:
        lines.extend(("", "Preview candidates:"))
        for item in candidates[:5]:
            label = _clean_text(item.get("label")) or _clean_text(item.get("event_id")) or "candidate"
            state = _clean_text(item.get("state")) or "needs_review"
            decision = _clean_text(item.get("export_decision")) or "hold_for_review"
            steps = ", ".join(_string_list(item.get("required_next_steps"))[:2])
            suffix = f"; next={steps}" if steps else ""
            lines.append(f"- {state}: {label} ({decision}{suffix})")
    return "\n".join(lines)


def format_evaluation_snapshot_report(snapshot: Mapping[str, Any]) -> str:
    counts = dict(snapshot.get("counts") or {})
    paths = dict(snapshot.get("paths") or {})
    lines = [
        f"Workspace: {_clean_text(snapshot.get('workspace_id')) or DEFAULT_WORKSPACE_ID}",
        f"Events: {int(snapshot.get('event_count') or 0)}",
        f"Signals: {int(snapshot.get('signal_count') or 0)} "
        f"(explicit {int(snapshot.get('explicit_signal_count') or 0)}, derived {int(snapshot.get('derived_signal_count') or 0)})",
        f"Acceptance: {int(counts.get('acceptance') or 0)}",
        f"Rejection: {int(counts.get('rejection') or 0)}",
        f"Review resolved: {int(counts.get('review_resolved') or 0)}",
        f"Review unresolved: {int(counts.get('review_unresolved') or 0)}",
        f"Test pass: {int(counts.get('test_pass') or 0)}",
        f"Test fail: {int(counts.get('test_fail') or 0)}",
        f"Repair links: {int(counts.get('repair_links') or 0)}",
        f"Follow-up links: {int(counts.get('follow_up_links') or 0)}",
        f"Comparisons: {int(counts.get('comparisons') or 0)}",
        f"Curation ready: {int(counts.get('curation_ready') or 0)}",
        f"Addressed failures: {int(counts.get('addressed_failures') or 0)}",
        f"Pending failures: {int(counts.get('pending_failures') or 0)}",
    ]
    generated_at_utc = _clean_text(snapshot.get("generated_at_utc"))
    if generated_at_utc:
        lines.append(f"Generated: {generated_at_utc}")
    if paths.get("snapshot_latest_path"):
        lines.append(f"Snapshot: {paths['snapshot_latest_path']}")

    pending_failures = [
        item
        for item in (snapshot.get("pending_failures") or [])
        if isinstance(item, Mapping)
    ]
    if pending_failures:
        lines.extend(("", "Pending failures:"))
        for item in pending_failures[:5]:
            label = (
                _clean_text(item.get("test_name"))
                or _clean_text(item.get("prompt_excerpt"))
                or _clean_text(item.get("source_event_id"))
                or "failure"
            )
            detail = _clean_text(item.get("failure_summary")) or _clean_text(item.get("quality_status")) or "n/a"
            lines.append(f"- {label}: {detail}")

    recent_signals = [
        item
        for item in (snapshot.get("recent_signals") or [])
        if isinstance(item, Mapping)
    ]
    if recent_signals:
        lines.extend(("", "Recent signals:"))
        for item in recent_signals[:6]:
            source_event_id = _clean_text(item.get("source_event_id")) or "n/a"
            target_event_id = _clean_text(item.get("target_event_id"))
            relation = _clean_text(item.get("relation_kind"))
            suffix = f" -> {relation} {target_event_id}" if relation and target_event_id else ""
            lines.append(f"- {_clean_text(item.get('signal_kind')) or 'signal'} {source_event_id}{suffix}")
    comparisons = [
        item
        for item in (snapshot.get("comparisons") or [])
        if isinstance(item, Mapping)
    ]
    if comparisons:
        lines.extend(("", "Recent comparisons:"))
        for item in comparisons[:4]:
            label = _clean_text(item.get("task_label")) or _clean_text(item.get("comparison_id")) or "comparison"
            winner = _clean_text(item.get("winner_event_id")) or "n/a"
            lines.append(f"- {label}: {_clean_text(item.get('outcome')) or 'n/a'} winner={winner}")
    curation = _mapping_dict(snapshot.get("curation"))
    curation_candidates = [
        item
        for item in (curation.get("candidates") or [])
        if isinstance(item, Mapping)
    ]
    ready_candidates = [item for item in curation_candidates if item.get("state") == "ready"]
    if ready_candidates:
        lines.extend(("", "Curation ready:"))
        for item in ready_candidates[:4]:
            label = _clean_text(item.get("label")) or _clean_text(item.get("event_id")) or "candidate"
            lines.append(f"- {label}")
    return "\n".join(lines)
