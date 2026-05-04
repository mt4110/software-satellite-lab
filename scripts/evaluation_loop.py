#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable as IterableABC
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from gemma_runtime import repo_root, timestamp_slug, timestamp_utc, write_json
from memory_index import rebuild_memory_index
from software_work_events import build_event_contract_check, build_event_contract_report, read_event_log
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
EVALUATION_CONSISTENCY_REPORT_SCHEMA_NAME = "software-satellite-evaluation-consistency-report"
EVALUATION_CONSISTENCY_REPORT_SCHEMA_VERSION = 1
EVALUATION_CONSISTENCY_ITEM_SCHEMA_NAME = "software-satellite-evaluation-consistency-item"
EVALUATION_CONSISTENCY_ITEM_SCHEMA_VERSION = 1
CURATION_EXPORT_PREVIEW_SCHEMA_NAME = "software-satellite-curation-export-preview"
CURATION_EXPORT_PREVIEW_SCHEMA_VERSION = 1
LEARNING_DATASET_PREVIEW_SCHEMA_NAME = "software-satellite-learning-dataset-preview"
LEARNING_DATASET_PREVIEW_SCHEMA_VERSION = 1
SUPERVISED_EXAMPLE_CANDIDATE_SCHEMA_NAME = "software-satellite-supervised-example-candidate"
SUPERVISED_EXAMPLE_CANDIDATE_SCHEMA_VERSION = 1
LEARNING_REVIEW_QUEUE_ITEM_SCHEMA_NAME = "software-satellite-learning-review-queue-item"
LEARNING_REVIEW_QUEUE_ITEM_SCHEMA_VERSION = 1
HUMAN_SELECTED_CANDIDATE_LIST_SCHEMA_NAME = "software-satellite-human-selected-candidate-list"
HUMAN_SELECTED_CANDIDATE_LIST_SCHEMA_VERSION = 1
HUMAN_SELECTED_CANDIDATE_ITEM_SCHEMA_NAME = "software-satellite-human-selected-candidate"
HUMAN_SELECTED_CANDIDATE_ITEM_SCHEMA_VERSION = 1
JSONL_TRAINING_EXPORT_DRY_RUN_SCHEMA_NAME = "software-satellite-jsonl-training-export-dry-run"
JSONL_TRAINING_EXPORT_DRY_RUN_SCHEMA_VERSION = 1
JSONL_TRAINING_EXPORT_DRY_RUN_ITEM_SCHEMA_NAME = "software-satellite-jsonl-training-export-dry-run-item"
JSONL_TRAINING_EXPORT_DRY_RUN_ITEM_SCHEMA_VERSION = 1
EXPORT_POLICY_CONFIRMATION_SIGNAL_KIND = "export_policy_confirmed"
HUMAN_SELECTED_SUPPLIED_TRAINING_TEXT_KEYS = {
    "completion",
    "completions",
    "input",
    "input_text",
    "instruction",
    "messages",
    "output",
    "output_excerpt",
    "output_text",
    "prompt",
    "prompt_excerpt",
    "resolved_user_prompt",
    "response",
    "supervised_example",
    "system_prompt",
}

SIGNAL_KINDS = (
    "acceptance",
    "rejection",
    "test_pass",
    "test_fail",
    "review_resolved",
    "review_unresolved",
    EXPORT_POLICY_CONFIRMATION_SIGNAL_KIND,
)
POSITIVE_SIGNAL_KINDS = {"acceptance", "test_pass", "review_resolved"}
NEGATIVE_SIGNAL_KINDS = {"rejection", "test_fail", "review_unresolved"}
RELATION_KINDS = ("repairs", "follow_up_for")
COMPARISON_OUTCOMES = ("winner_selected", "tie", "needs_follow_up")
CURATION_STATES = ("ready", "needs_review", "blocked")
CURATION_EXPORT_DECISIONS = ("include_when_approved", "hold_for_review", "exclude_until_repaired")
CURATION_BLOCKING_REASONS = {"rejected", "review_unresolved", "test_fail", "failed"}
LEARNING_BLOCKING_REASONS = CURATION_BLOCKING_REASONS | {"noisy", "unresolved"}
LEARNING_SELECTION_REASONS = {"accepted", "review_resolved", "comparison_winner"}
LEARNING_REVIEW_QUEUE_STATES = ("blocked", "missing_source", "missing_supervised_text", "ready", "needs_review")
LEARNING_MISSING_SOURCE_EXCLUSIONS = {
    "missing_source_event",
    "missing_source_artifact_path",
    "missing_source_artifact",
    "unreadable_source_artifact",
    "source_artifact_not_durable",
    "comparison_winner_missing_source_event",
    "comparison_winner_missing_source_artifact",
    "comparison_winner_source_artifact_not_durable",
    "comparison_winner_unreadable_source_artifact",
}

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


def learning_dataset_preview_latest_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return evaluation_root(workspace_id=workspace_id, root=root) / "learning" / "preview-latest.json"


def learning_dataset_preview_run_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return (
        evaluation_root(workspace_id=workspace_id, root=root)
        / "learning"
        / "runs"
        / f"{timestamp_slug()}-learning-preview.json"
    )


def human_selected_candidates_latest_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return evaluation_root(workspace_id=workspace_id, root=root) / "learning" / "human-selected-latest.json"


def human_selected_candidates_run_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return (
        evaluation_root(workspace_id=workspace_id, root=root)
        / "learning"
        / "runs"
        / f"{timestamp_slug()}-human-selected-candidates.json"
    )


def jsonl_training_export_dry_run_latest_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return evaluation_root(workspace_id=workspace_id, root=root) / "learning" / "jsonl-export-dry-run-latest.json"


def jsonl_training_export_dry_run_run_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return (
        evaluation_root(workspace_id=workspace_id, root=root)
        / "learning"
        / "runs"
        / f"{timestamp_slug()}-jsonl-export-dry-run.json"
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

    if payload["signal_kind"] == EXPORT_POLICY_CONFIRMATION_SIGNAL_KIND:
        payload["evidence"] = build_export_policy_confirmation_evidence(
            _mapping_dict(payload.get("evidence"))
        )

    relation = _mapping_dict(payload.get("relation"))
    relation_kind = _clean_text(relation.get("relation_kind"))
    target_event_id = _clean_text(relation.get("target_event_id")) or _clean_text(payload.get("target_event_id"))
    if payload["signal_kind"] == EXPORT_POLICY_CONFIRMATION_SIGNAL_KIND and (
        relation_kind is not None or target_event_id is not None
    ):
        raise ValueError(f"Export-policy confirmation signals cannot define relation links{location}.")
    if relation_kind is not None:
        relation["relation_kind"] = _normalize_relation_kind(relation_kind)
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
    if normalized_signal_kind == EXPORT_POLICY_CONFIRMATION_SIGNAL_KIND:
        if normalized_relation_kind is not None or _clean_text(target_event_id) is not None:
            raise ValueError("Export-policy confirmation signals cannot define relation links.")
        normalized_evidence = build_export_policy_confirmation_evidence(normalized_evidence)
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


def build_export_policy_confirmation_evidence(evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = _compact_evidence(evidence)
    payload["confirmation_scope"] = (
        _clean_text(payload.get("confirmation_scope"))
        or "learning_dataset_preview_candidate"
    )
    payload["policy_version"] = _clean_text(payload.get("policy_version")) or "m7-preview-only-v1"
    payload["export_mode"] = "preview_only"
    payload["training_export_ready"] = False
    payload["human_gate_required"] = True
    payload["training_job_allowed"] = False
    payload["raw_log_export_allowed"] = False
    payload["downstream_export_requires_separate_approval"] = True
    return payload


def build_export_policy_confirmation_signal(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    source_event_id: str,
    recorded_at_utc: str | None = None,
    signal_id: str | None = None,
    source_event: Mapping[str, Any] | None = None,
    rationale: str | None = None,
    evidence: Mapping[str, Any] | None = None,
    origin: str = "manual",
    tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    signal_tags = [
        "learning-export-policy",
        "preview-only",
        "human-gated",
        *list(tags or []),
    ]
    return build_evaluation_signal(
        workspace_id=workspace_id,
        signal_kind=EXPORT_POLICY_CONFIRMATION_SIGNAL_KIND,
        source_event_id=source_event_id,
        recorded_at_utc=recorded_at_utc,
        signal_id=signal_id,
        source_event=source_event,
        rationale=rationale,
        evidence=build_export_policy_confirmation_evidence(evidence),
        origin=origin,
        tags=signal_tags,
    )


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
        "confirmation_scope": _clean_text(evidence.get("confirmation_scope")),
        "policy_version": _clean_text(evidence.get("policy_version")),
        "export_mode": _clean_text(evidence.get("export_mode")),
        "training_export_ready": evidence.get("training_export_ready"),
        "human_gate_required": evidence.get("human_gate_required"),
        "training_job_allowed": evidence.get("training_job_allowed"),
        "raw_log_export_allowed": evidence.get("raw_log_export_allowed"),
        "downstream_export_requires_separate_approval": evidence.get(
            "downstream_export_requires_separate_approval"
        ),
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
    policy_confirmed = {
        event_id
        for signal in signal_summary_list
        if signal.get("signal_kind") == EXPORT_POLICY_CONFIRMATION_SIGNAL_KIND
        if (event_id := _clean_text(signal.get("source_event_id"))) is not None
    }
    winners = {
        event_id
        for comparison in comparison_summary_list
        if comparison.get("outcome") == "winner_selected"
        if (event_id := _clean_text(comparison.get("winner_event_id"))) is not None
    }
    candidate_ids = sorted(
        accepted
        | review_resolved
        | rejected
        | review_unresolved
        | passed
        | failed
        | winners
        | policy_confirmed
    )
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
        if event_id in policy_confirmed:
            reasons.append(EXPORT_POLICY_CONFIRMATION_SIGNAL_KIND)

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


def _deduplicate_strings(values: Iterable[str]) -> list[str]:
    deduplicated: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_text(value)
        if cleaned is None or cleaned in seen:
            continue
        seen.add(cleaned)
        deduplicated.append(cleaned)
    return deduplicated


def _compact_signal_reference(signal: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if signal is None:
        return None
    return {
        key: value
        for key, value in {
            "signal_id": _clean_text(signal.get("signal_id")),
            "signal_kind": _clean_text(signal.get("signal_kind")),
            "polarity": _clean_text(signal.get("polarity")),
            "origin": _clean_text(signal.get("origin")),
            "recorded_at_utc": _clean_text(signal.get("recorded_at_utc")),
        }.items()
        if value is not None
    }


def _latest_signal_summary(
    signals: Iterable[Mapping[str, Any]],
    *,
    signal_kinds: set[str],
) -> dict[str, Any] | None:
    matching_signals = [
        dict(signal)
        for signal in signals
        if isinstance(signal, Mapping)
        if _clean_text(signal.get("signal_kind")) in signal_kinds
    ]
    matching_signals.sort(
        key=lambda signal: (
            _timestamp_sort_key(signal.get("recorded_at_utc")),
            str(signal.get("signal_id") or ""),
        ),
        reverse=True,
    )
    return matching_signals[0] if matching_signals else None


def _consistency_signal_bucket(
    signals: Iterable[Mapping[str, Any]],
    *,
    positive_kind: str,
    negative_kind: str,
    positive_reason: str,
    negative_reason: str,
) -> dict[str, Any]:
    signal_list = [
        dict(signal)
        for signal in signals
        if isinstance(signal, Mapping)
        if _clean_text(signal.get("signal_kind")) in {positive_kind, negative_kind}
    ]
    latest_signal = _latest_signal_summary(
        signal_list,
        signal_kinds={positive_kind, negative_kind},
    )
    positive_signals = [
        signal
        for signal in signal_list
        if _clean_text(signal.get("signal_kind")) == positive_kind
    ]
    negative_signals = [
        signal
        for signal in signal_list
        if _clean_text(signal.get("signal_kind")) == negative_kind
    ]
    latest_kind = _clean_text(latest_signal.get("signal_kind")) if latest_signal is not None else None
    negative_wins = latest_kind == negative_kind and bool(positive_signals)
    return {
        "latest_signal": _compact_signal_reference(latest_signal),
        "positive_signal_count": len(positive_signals),
        "negative_signal_count": len(negative_signals),
        "positive_reason": positive_reason,
        "negative_reason": negative_reason,
        "stale_positive_signal": negative_wins,
        "negative_signal_wins": negative_wins,
    }


def _comparison_references_by_event_id(
    comparisons: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    references_by_event_id: dict[str, list[dict[str, Any]]] = {}
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            continue
        candidate_event_ids = _string_list(comparison.get("candidate_event_ids"))
        winner_event_id = _clean_text(comparison.get("winner_event_id"))
        reference_event_ids = _deduplicate_strings(
            [
                *candidate_event_ids,
                *([winner_event_id] if winner_event_id is not None else []),
            ]
        )
        for event_id in reference_event_ids:
            references_by_event_id.setdefault(event_id, []).append(
                {
                    "comparison_id": _clean_text(comparison.get("comparison_id")),
                    "recorded_at_utc": _clean_text(comparison.get("recorded_at_utc")),
                    "outcome": _clean_text(comparison.get("outcome")),
                    "winner_event_id": winner_event_id,
                    "role": "winner" if event_id == winner_event_id else "candidate",
                }
            )
    for references in references_by_event_id.values():
        references.sort(
            key=lambda item: (
                _timestamp_sort_key(item.get("recorded_at_utc")),
                str(item.get("comparison_id") or ""),
            ),
            reverse=True,
        )
    return references_by_event_id


def _consistency_source_trace(
    *,
    event_id: str,
    event: Mapping[str, Any] | None,
    root: Path | None,
) -> dict[str, Any]:
    if event is None:
        return {
            "source_event_present": False,
            "event_contract_status": "missing_source_event",
            "source_artifact_status": None,
            "source_artifact_reasons": ["missing_source_event"],
        }
    contract = build_event_contract_check(event, root=root)
    source_artifact = _mapping_dict(contract.get("source_artifact"))
    return {
        "source_event_present": True,
        "event_contract_status": _clean_text(contract.get("contract_status")),
        "source_artifact_status": _clean_text(source_artifact.get("source_status")),
        "source_artifact_reasons": _string_list(source_artifact.get("reasons")),
        "event_contract": contract,
    }


def _comparison_winner_contract_issue_key(source_trace: Mapping[str, Any]) -> str:
    if not bool(source_trace.get("source_event_present")):
        return "comparison_winner_missing_source_event"
    contract_status = _clean_text(source_trace.get("event_contract_status"))
    source_reasons = set(_string_list(source_trace.get("source_artifact_reasons")))
    if contract_status == "missing_source":
        if "source_artifact_outside_workspace" in source_reasons:
            return "comparison_winner_source_artifact_not_durable"
        if source_reasons & {"source_artifact_not_file", "source_artifact_unreadable"}:
            return "comparison_winner_unreadable_source_artifact"
        if source_reasons & {"missing_source_artifact_path", "source_artifact_missing"}:
            return "comparison_winner_missing_source_artifact"
        return "comparison_winner_missing_source_artifact"
    return "comparison_winner_invalid_event_contract"


def build_evaluation_consistency_report(
    *,
    signal_summaries: Iterable[Mapping[str, Any]],
    comparison_summaries: Iterable[Mapping[str, Any]],
    curation_candidates: Iterable[Mapping[str, Any]],
    events_by_id: Mapping[str, Mapping[str, Any]],
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> dict[str, Any]:
    """Summarize whether positive curation evidence is still backed by latest traces."""
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
    curation_by_event_id = {
        event_id: dict(candidate)
        for candidate in curation_candidates
        if isinstance(candidate, Mapping)
        if (event_id := _clean_text(candidate.get("event_id"))) is not None
    }
    signals_by_event_id: dict[str, list[dict[str, Any]]] = {}
    for signal in signal_summary_list:
        event_id = _clean_text(signal.get("source_event_id"))
        if event_id is None:
            continue
        signals_by_event_id.setdefault(event_id, []).append(signal)

    comparison_references_by_event = _comparison_references_by_event_id(comparison_summary_list)

    event_ids = sorted(set(signals_by_event_id) | set(curation_by_event_id) | set(comparison_references_by_event))
    items: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    for event_id in event_ids:
        event_signals = signals_by_event_id.get(event_id, [])
        curation_candidate = curation_by_event_id.get(event_id, {})
        curation_reasons = set(_string_list(curation_candidate.get("reasons")))
        curation_state = _clean_text(curation_candidate.get("state"))
        buckets = {
            "test": _consistency_signal_bucket(
                event_signals,
                positive_kind="test_pass",
                negative_kind="test_fail",
                positive_reason="test_pass",
                negative_reason="test_fail",
            ),
            "review": _consistency_signal_bucket(
                event_signals,
                positive_kind="review_resolved",
                negative_kind="review_unresolved",
                positive_reason="review_resolved",
                negative_reason="review_unresolved",
            ),
            "selection": _consistency_signal_bucket(
                event_signals,
                positive_kind="acceptance",
                negative_kind="rejection",
                positive_reason="accepted",
                negative_reason="rejected",
            ),
        }
        stale_positive_signals: list[str] = []
        stale_positive_reasons: list[str] = []
        negative_signal_wins: list[str] = []
        missing_traces: list[str] = []
        inconsistencies: list[str] = []

        for bucket_name, bucket in buckets.items():
            positive_reason = _clean_text(bucket.get("positive_reason"))
            negative_reason = _clean_text(bucket.get("negative_reason"))
            latest_signal = _mapping_dict(bucket.get("latest_signal"))
            latest_kind = _clean_text(latest_signal.get("signal_kind"))
            if bucket.get("stale_positive_signal") and positive_reason is not None:
                stale_positive_signals.append(positive_reason)
                negative_signal_wins.append(negative_reason or bucket_name)
                issue_key = f"stale_{positive_reason}_signal"
                inconsistencies.append(issue_key)
                issue_counts[issue_key] += 1
                issue_counts["stale_positive_signal"] += 1
                issue_counts["negative_signal_wins"] += 1
                if positive_reason in curation_reasons:
                    stale_positive_reasons.append(positive_reason)
                    reason_issue_key = f"stale_{positive_reason}_reason"
                    inconsistencies.append(reason_issue_key)
                    issue_counts[reason_issue_key] += 1
            if positive_reason in curation_reasons and latest_kind not in {
                "acceptance" if positive_reason == "accepted" else positive_reason
            }:
                if latest_kind is None:
                    missing_key = f"missing_{positive_reason}_trace"
                    missing_traces.append(missing_key)
                    issue_counts[missing_key] += 1
                    issue_counts["missing_trace"] += 1
                elif latest_kind == negative_reason:
                    stale_reason_key = f"stale_{positive_reason}_reason"
                    if stale_reason_key not in inconsistencies:
                        inconsistencies.append(stale_reason_key)
                        issue_counts[stale_reason_key] += 1

        comparison_refs = comparison_references_by_event.get(event_id, [])
        has_winner_trace = any(
            item.get("role") == "winner" and item.get("outcome") == "winner_selected"
            for item in comparison_refs
        )
        if "comparison_winner" in curation_reasons and not has_winner_trace:
            missing_traces.append("missing_comparison_winner_trace")
            inconsistencies.append("missing_comparison_winner_trace")
            issue_counts["missing_comparison_winner_trace"] += 1
            issue_counts["missing_trace"] += 1

        source_trace = _consistency_source_trace(
            event_id=event_id,
            event=events_by_id.get(event_id),
            root=root,
        )
        winner_needs_trace = has_winner_trace or any(
            item.get("role") == "winner"
            for item in comparison_refs
        )
        source_event_present = bool(source_trace.get("source_event_present"))
        event_contract_status = _clean_text(source_trace.get("event_contract_status"))
        if winner_needs_trace and (
            not source_event_present or event_contract_status != "ok"
        ):
            issue_key = _comparison_winner_contract_issue_key(source_trace)
            missing_traces.append(issue_key)
            inconsistencies.append(issue_key)
            issue_counts[issue_key] += 1
            issue_counts["missing_trace"] += 1
            issue_counts["comparison_winner_missing_trace"] += 1

        item_issue_counts = Counter(_deduplicate_strings([*inconsistencies, *missing_traces]))
        items.append(
            {
                "schema_name": EVALUATION_CONSISTENCY_ITEM_SCHEMA_NAME,
                "schema_version": EVALUATION_CONSISTENCY_ITEM_SCHEMA_VERSION,
                "event_id": event_id,
                "curation": {
                    "present": bool(curation_candidate),
                    "state": curation_state,
                    "reasons": sorted(curation_reasons),
                },
                "latest_signals": {
                    key: copy.deepcopy(_mapping_dict(bucket.get("latest_signal")))
                    for key, bucket in buckets.items()
                    if bucket.get("latest_signal")
                },
                "signal_counts": {
                    key: {
                        "positive": int(bucket.get("positive_signal_count") or 0),
                        "negative": int(bucket.get("negative_signal_count") or 0),
                    }
                    for key, bucket in buckets.items()
                },
                "stale_positive_signals": _deduplicate_strings(stale_positive_signals),
                "stale_positive_reasons": _deduplicate_strings(stale_positive_reasons),
                "negative_signal_wins": _deduplicate_strings(negative_signal_wins),
                "missing_traces": _deduplicate_strings(missing_traces),
                "inconsistencies": _deduplicate_strings(inconsistencies),
                "issue_counts": {
                    key: int(value)
                    for key, value in sorted(item_issue_counts.items())
                },
                "comparisons": comparison_refs,
                "source_trace": source_trace,
            }
        )

    events_with_issues = [
        item
        for item in items
        if item.get("inconsistencies") or item.get("missing_traces")
    ]
    rollup_issue_keys = {
        "stale_positive_signal",
        "negative_signal_wins",
        "missing_trace",
        "comparison_winner_missing_trace",
    }
    specific_issue_count = sum(
        int(value)
        for key, value in issue_counts.items()
        if key not in rollup_issue_keys
    )
    return {
        "schema_name": EVALUATION_CONSISTENCY_REPORT_SCHEMA_NAME,
        "schema_version": EVALUATION_CONSISTENCY_REPORT_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "generated_at_utc": timestamp_utc(),
        "counts": {
            "event_count": len(items),
            "events_with_issues": len(events_with_issues),
            "stale_positive_signal": int(issue_counts.get("stale_positive_signal", 0)),
            "negative_signal_wins": int(issue_counts.get("negative_signal_wins", 0)),
            "missing_trace": int(issue_counts.get("missing_trace", 0)),
            "comparison_winner_missing_trace": int(issue_counts.get("comparison_winner_missing_trace", 0)),
            "inconsistency": specific_issue_count,
        },
        "issue_counts": {
            key: int(value)
            for key, value in sorted(issue_counts.items())
        },
        "events": items,
        "issue_events": events_with_issues[:12],
        "notes": [
            "Latest negative evidence wins over older positive evidence for consistency checks.",
            "Export-policy confirmation is reported as policy evidence only, not selection evidence.",
        ],
    }


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
    cached_event_contract = resolved_index_summary.get("event_contract")
    event_contract = (
        copy.deepcopy(dict(cached_event_contract))
        if isinstance(cached_event_contract, Mapping)
        else build_event_contract_report(events, root=resolved_root)
    )
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
    evaluation_consistency = build_evaluation_consistency_report(
        signal_summaries=signal_summaries,
        comparison_summaries=comparison_summaries,
        curation_candidates=curation_candidates,
        events_by_id=events_by_id,
        workspace_id=workspace_id,
        root=resolved_root,
    )
    curation_state_counts = Counter(str(item.get("state") or "") for item in curation_candidates)
    comparison_counts = Counter(str(item.get("outcome") or "") for item in comparison_summaries)
    consistency_counts = _mapping_dict(evaluation_consistency.get("counts"))

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
            "export_policy_confirmed": int(counts.get(EXPORT_POLICY_CONFIRMATION_SIGNAL_KIND, 0)),
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
            "stale_positive_signal": int(consistency_counts.get("stale_positive_signal") or 0),
            "negative_signal_wins": int(consistency_counts.get("negative_signal_wins") or 0),
            "missing_trace": int(consistency_counts.get("missing_trace") or 0),
            "comparison_winner_missing_trace": int(
                consistency_counts.get("comparison_winner_missing_trace") or 0
            ),
            "evaluation_inconsistency": int(consistency_counts.get("inconsistency") or 0),
            "event_contract_failed": int(event_contract.get("failed_event_count") or 0),
            "event_contract_missing_source": int(
                _mapping_dict(event_contract.get("source_status_counts")).get("missing_source") or 0
            ),
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
        "event_contract": event_contract,
        "evaluation_consistency": evaluation_consistency,
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


def record_export_policy_confirmation_signal(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    source_event_id: str,
    rationale: str | None = None,
    evidence: Mapping[str, Any] | None = None,
    origin: str = "manual",
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    event_id = _clean_text(source_event_id)
    if event_id is None:
        raise ValueError("Export-policy confirmation signals require a source_event_id.")
    events_by_id = software_work_events_by_id(root=resolved_root, workspace_id=workspace_id)
    source_event = events_by_id.get(event_id)
    if source_event is None:
        raise ValueError(f"Unknown export-policy confirmation source_event_id `{event_id}`.")
    signal = build_export_policy_confirmation_signal(
        workspace_id=workspace_id,
        source_event_id=event_id,
        source_event=source_event,
        rationale=rationale,
        evidence=evidence,
        origin=origin,
    )
    return append_evaluation_signal(
        evaluation_signal_log_path(workspace_id=workspace_id, root=resolved_root),
        signal,
        workspace_id=workspace_id,
    )


def _curation_export_decision(candidate: Mapping[str, Any]) -> str:
    state = _clean_text(candidate.get("state"))
    reasons = set(_string_list(candidate.get("reasons")))
    if reasons & CURATION_BLOCKING_REASONS:
        return "exclude_until_repaired"
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
        if reason in CURATION_BLOCKING_REASONS
    }
    human_selected = bool(reasons & {"accepted", "review_resolved", "comparison_winner"})
    export_policy_confirmed = EXPORT_POLICY_CONFIRMATION_SIGNAL_KIND in reasons
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
            "status": "done" if export_policy_confirmed else "pending",
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
    blocked_by = reasons & CURATION_BLOCKING_REASONS
    if not blocked_by and (state == "ready" or _curation_candidate_ready_for_policy(candidate)):
        if EXPORT_POLICY_CONFIRMATION_SIGNAL_KIND in reasons:
            return ["review_downstream_export_policy"]
        return ["confirm_export_policy"]
    steps: list[str] = []
    if "needs_test_signal" in reasons:
        steps.append("record_test_pass")
    if "needs_human_selection" in reasons:
        steps.append("record_acceptance_or_review_resolution")
    if "test_fail" in reasons or "failed" in reasons:
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
        if reason in CURATION_BLOCKING_REASONS
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


def _preview_excerpt(value: Any, *, limit: int = 6000) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[: max(limit - 20, 0)].rstrip() + "\n[preview truncated]"


def _stable_learning_candidate_id(*, workspace_id: str, event_id: str) -> str:
    digest = hashlib.sha256(f"{workspace_id}\n{event_id}".encode("utf-8")).hexdigest()[:12]
    return f"{workspace_id}:supervised-example-candidate:{digest}"


def _stable_learning_review_queue_item_id(
    *,
    workspace_id: str,
    event_id: str | None,
    fallback_key: str | None = None,
    source_index: int | None = None,
) -> str:
    if event_id is not None:
        key = event_id
    else:
        key = f"{fallback_key or 'missing-event-id'}\nsource_index={source_index if source_index is not None else 'unknown'}"
    digest = hashlib.sha256(f"{workspace_id}\n{key}".encode("utf-8")).hexdigest()[:12]
    return f"{workspace_id}:learning-review-queue:{digest}"


def _path_from_text(value: Any) -> Path | None:
    cleaned = _clean_text(value)
    if cleaned is None:
        return None
    return Path(cleaned).expanduser()


def _path_is_file(value: Any) -> bool:
    path = _path_from_text(value)
    if path is None:
        return False
    try:
        return path.is_file()
    except OSError:
        return False


def _read_json_object(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _learning_source_paths(
    *,
    snapshot: Mapping[str, Any],
    curation_preview: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot_paths = _mapping_dict(snapshot.get("paths"))
    curation_paths = _mapping_dict(curation_preview.get("paths"))
    return {
        "event_log_path": _clean_text(snapshot_paths.get("event_log_path")),
        "signal_log_path": _clean_text(snapshot_paths.get("signal_log_path")),
        "comparison_log_path": _clean_text(snapshot_paths.get("comparison_log_path")),
        "source_snapshot_path": (
            _clean_text(curation_preview.get("source_snapshot_path"))
            or _clean_text(curation_paths.get("source_snapshot_path"))
            or _clean_text(snapshot_paths.get("snapshot_run_path"))
            or _clean_text(snapshot_paths.get("snapshot_latest_path"))
        ),
        "source_curation_preview_path": (
            _clean_text(curation_paths.get("curation_preview_run_path"))
            or _clean_text(curation_paths.get("curation_preview_latest_path"))
        ),
    }


def _learning_contract_root(source_paths: Mapping[str, Any]) -> Path | None:
    event_log_path = _path_from_text(source_paths.get("event_log_path"))
    if (
        event_log_path is not None
        and event_log_path.parent.name == "event_logs"
        and event_log_path.parent.parent.name == "artifacts"
    ):
        return event_log_path.parent.parent.parent
    source_snapshot_path = _path_from_text(source_paths.get("source_snapshot_path"))
    if source_snapshot_path is not None:
        for parent in source_snapshot_path.parents:
            if parent.name == "artifacts":
                return parent.parent
    return None


def _artifact_root_from_path_text(path_text: Any) -> Path | None:
    cleaned_path = _clean_text(path_text)
    if cleaned_path is None:
        return None
    path = Path(cleaned_path).expanduser()
    if not path.is_absolute():
        return None
    for parent in path.parents:
        if parent.name == "artifacts":
            return parent.parent
    return None


def _learning_event_contract_root(
    event: Mapping[str, Any],
    *,
    fallback_root: Path | None,
) -> Path | None:
    if fallback_root is not None:
        return fallback_root
    source_refs = _mapping_dict(event.get("source_refs"))
    artifact_ref = _mapping_dict(source_refs.get("artifact_ref"))
    return _artifact_root_from_path_text(artifact_ref.get("artifact_path"))


def _read_events_by_id_from_snapshot(snapshot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    paths = _mapping_dict(snapshot.get("paths"))
    event_log_path = _path_from_text(paths.get("event_log_path"))
    if event_log_path is None:
        return {}
    event_log = read_event_log(event_log_path)
    return {
        str(event.get("event_id")): dict(event)
        for event in event_log.get("events") or []
        if isinstance(event, Mapping)
        if _clean_text(event.get("event_id")) is not None
    }


def _read_signals_from_snapshot(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths = _mapping_dict(snapshot.get("paths"))
    signal_log_path = _path_from_text(paths.get("signal_log_path"))
    if signal_log_path is None:
        return []
    return read_evaluation_signals(signal_log_path)


def _read_comparisons_from_snapshot(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths = _mapping_dict(snapshot.get("paths"))
    comparison_log_path = _path_from_text(paths.get("comparison_log_path"))
    if comparison_log_path is None:
        return []
    return read_evaluation_comparisons(comparison_log_path)


def _learning_source_event_record(event: Mapping[str, Any], *, event_id: str) -> dict[str, Any]:
    session = _mapping_dict(event.get("session"))
    content = _mapping_dict(event.get("content"))
    outcome = _mapping_dict(event.get("outcome"))
    source_refs = _mapping_dict(event.get("source_refs"))
    artifact_ref = _mapping_dict(source_refs.get("artifact_ref"))
    return {
        "source_event_id": event_id,
        "event_kind": _clean_text(event.get("event_kind")),
        "recorded_at_utc": _clean_text(event.get("recorded_at_utc")),
        "session_id": _clean_text(session.get("session_id")),
        "session_surface": _clean_text(session.get("surface")),
        "session_mode": _clean_text(session.get("mode")),
        "title": _clean_text(session.get("title")),
        "status": _clean_text(outcome.get("status")),
        "quality_status": _clean_text(outcome.get("quality_status")),
        "execution_status": _clean_text(outcome.get("execution_status")),
        "artifact_kind": _clean_text(artifact_ref.get("artifact_kind")),
        "artifact_path": _clean_text(artifact_ref.get("artifact_path")),
        "artifact_workspace_relative_path": _clean_text(artifact_ref.get("artifact_workspace_relative_path")),
        "prompt_excerpt": _preview_excerpt(content.get("prompt"), limit=1200),
        "output_excerpt": _preview_excerpt(content.get("output_text"), limit=1200),
    }


def _learning_review_queue_source_event_record(event: Mapping[str, Any], *, event_id: str) -> dict[str, Any]:
    record = _learning_source_event_record(event, event_id=event_id)
    record.pop("prompt_excerpt", None)
    record.pop("output_excerpt", None)
    return record


def _learning_backend_metadata(event: Mapping[str, Any]) -> dict[str, Any]:
    session = _mapping_dict(event.get("session"))
    content = _mapping_dict(event.get("content"))
    options = _mapping_dict(content.get("options"))
    outcome = _mapping_dict(event.get("outcome"))
    source_refs = _mapping_dict(event.get("source_refs"))
    backend_ref = _mapping_dict(source_refs.get("backend_ref"))
    artifact_ref = _mapping_dict(source_refs.get("artifact_ref"))
    artifact_payload: dict[str, Any] = {}
    if _clean_text(artifact_ref.get("artifact_kind")) == "agent_run":
        artifact_payload = _read_json_object(_path_from_text(artifact_ref.get("artifact_path")))
    run_backend = _mapping_dict(artifact_payload.get("backend"))
    run_compatibility = _mapping_dict(artifact_payload.get("compatibility"))

    backend_id = (
        _clean_text(run_backend.get("backend_id"))
        or _clean_text(backend_ref.get("backend_id"))
        or _clean_text(options.get("backend_id"))
        or _clean_text(outcome.get("backend_id"))
    )
    model_id = (
        _clean_text(run_backend.get("model_id"))
        or _clean_text(backend_ref.get("model_id"))
        or _clean_text(options.get("model_id"))
        or _clean_text(session.get("selected_model_id"))
        or _clean_text(outcome.get("model_id"))
    )
    adapter_kind = (
        _clean_text(run_backend.get("adapter_kind"))
        or _clean_text(backend_ref.get("adapter_kind"))
        or _clean_text(options.get("backend_adapter_kind"))
    )
    compatibility_status = (
        _clean_text(run_compatibility.get("status"))
        or _clean_text(backend_ref.get("compatibility_status"))
        or _clean_text(options.get("backend_compatibility_status"))
    )
    capabilities = (
        run_backend.get("capabilities")
        if isinstance(run_backend.get("capabilities"), Mapping)
        else backend_ref.get("capabilities")
        if isinstance(backend_ref.get("capabilities"), Mapping)
        else options.get("backend_capabilities")
        if isinstance(options.get("backend_capabilities"), Mapping)
        else {}
    )
    return {
        "backend_id": backend_id,
        "display_name": (
            _clean_text(run_backend.get("display_name"))
            or _clean_text(options.get("backend_display_name"))
        ),
        "adapter_kind": adapter_kind,
        "model_id": model_id,
        "compatibility_status": compatibility_status,
        "capabilities": copy.deepcopy(dict(capabilities)),
        "limits": copy.deepcopy(_mapping_dict(run_backend.get("limits"))),
        "metadata": copy.deepcopy(_mapping_dict(run_backend.get("metadata"))),
        "metadata_source_artifact_path": (
            _clean_text(artifact_ref.get("artifact_path"))
            if run_backend
            else None
        ),
    }


def _learning_supervised_example(event: Mapping[str, Any]) -> dict[str, Any]:
    session = _mapping_dict(event.get("session"))
    content = _mapping_dict(event.get("content"))
    options = _mapping_dict(content.get("options"))
    instruction = (
        _preview_excerpt(content.get("resolved_user_prompt"), limit=4000)
        or _preview_excerpt(content.get("prompt"), limit=4000)
    )
    response = _preview_excerpt(content.get("output_text"), limit=6000)
    return {
        "format": "instruction_response",
        "instruction": instruction,
        "response": response,
        "context": {
            "session_surface": _clean_text(session.get("surface")),
            "session_mode": _clean_text(session.get("mode")),
            "task_title": _clean_text(session.get("title")) or _clean_text(options.get("claim_scope")),
            "validation_mode": _clean_text(options.get("validation_mode")),
            "validation_command": _clean_text(options.get("validation_command")),
            "pass_definition": _clean_text(options.get("pass_definition")),
            "quality_status": _clean_text(options.get("quality_status")),
            "execution_status": _clean_text(options.get("execution_status")),
        },
    }


def _learning_signal_trace(signal: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping_dict(signal.get("source"))
    relation = _mapping_dict(signal.get("relation"))
    signal_kind = _clean_text(signal.get("signal_kind"))
    evidence = _mapping_dict(signal.get("evidence"))
    if signal_kind == EXPORT_POLICY_CONFIRMATION_SIGNAL_KIND:
        evidence = build_export_policy_confirmation_evidence(evidence)
    return {
        "signal_id": _clean_text(signal.get("signal_id")),
        "signal_kind": signal_kind,
        "polarity": _clean_text(signal.get("polarity")),
        "origin": _clean_text(signal.get("origin")),
        "recorded_at_utc": _clean_text(signal.get("recorded_at_utc")),
        "source_event_id": _clean_text(source.get("source_event_id")),
        "relation_kind": _clean_text(relation.get("relation_kind")),
        "target_event_id": _clean_text(relation.get("target_event_id")),
        "evidence": copy.deepcopy(evidence),
        "tags": _string_list(signal.get("tags")),
    }


def _learning_policy_confirmation_trace(signals: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    confirmations = [
        dict(signal)
        for signal in signals
        if isinstance(signal, Mapping)
        if _clean_text(signal.get("signal_kind")) == EXPORT_POLICY_CONFIRMATION_SIGNAL_KIND
    ]
    confirmations.sort(
        key=lambda signal: (
            _timestamp_sort_key(signal.get("recorded_at_utc")),
            str(signal.get("signal_id") or ""),
        ),
        reverse=True,
    )
    if not confirmations:
        return {
            "confirmed": False,
            "latest_signal_id": None,
            "recorded_at_utc": None,
            "origin": None,
        }
    latest = confirmations[0]
    return {
        "confirmed": True,
        "latest_signal_id": _clean_text(latest.get("signal_id")),
        "recorded_at_utc": _clean_text(latest.get("recorded_at_utc")),
        "origin": _clean_text(latest.get("origin")),
        "evidence": copy.deepcopy(_mapping_dict(latest.get("evidence"))),
    }


def _learning_signals_for_event(
    *,
    event_id: str,
    event: Mapping[str, Any],
    explicit_signals: Iterable[Mapping[str, Any]],
    workspace_id: str,
) -> list[dict[str, Any]]:
    signals = [
        dict(signal)
        for signal in explicit_signals
        if _source_event_id(signal) == event_id
    ]
    derived = derive_test_signal_from_event(event, workspace_id=workspace_id)
    if derived is not None:
        signals.append(derived)
    deduped: dict[str, dict[str, Any]] = {}
    for signal in signals:
        signal_id = _clean_text(signal.get("signal_id")) or f"{event_id}:signal:{len(deduped)}"
        deduped[signal_id] = signal
    return [
        _learning_signal_trace(signal)
        for signal in _sort_signals(deduped.values())
    ]


def _comparison_event_ids(comparison: Mapping[str, Any]) -> list[str]:
    return [
        event_id
        for candidate in comparison.get("candidates") or []
        if isinstance(candidate, Mapping)
        if (event_id := _clean_text(candidate.get("event_id"))) is not None
    ]


def _learning_comparison_trace(comparison: Mapping[str, Any], *, event_id: str) -> dict[str, Any]:
    candidate_event_ids = _comparison_event_ids(comparison)
    winner_event_id = _clean_text(comparison.get("winner_event_id"))
    if winner_event_id == event_id:
        role = "winner"
    elif event_id in candidate_event_ids:
        role = "candidate"
    else:
        role = "unrelated"
    return {
        "comparison_id": _clean_text(comparison.get("comparison_id")),
        "recorded_at_utc": _clean_text(comparison.get("recorded_at_utc")),
        "origin": _clean_text(comparison.get("origin")),
        "task_label": _clean_text(comparison.get("task_label")),
        "outcome": _clean_text(comparison.get("outcome")),
        "winner_event_id": winner_event_id,
        "role": role,
        "candidate_event_ids": candidate_event_ids,
        "criteria": _string_list(comparison.get("criteria")),
        "rationale": _clean_text(comparison.get("rationale")),
        "tags": _string_list(comparison.get("tags")),
    }


def _learning_comparisons_for_event(
    *,
    event_id: str,
    comparisons: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    traces = [
        _learning_comparison_trace(comparison, event_id=event_id)
        for comparison in comparisons
        if event_id in _comparison_event_ids(comparison)
    ]
    traces.sort(
        key=lambda comparison: (
            _timestamp_sort_key(comparison.get("recorded_at_utc")),
            str(comparison.get("comparison_id") or ""),
        ),
        reverse=True,
    )
    return traces


def _learning_candidate_blocked_reasons(
    candidate: Mapping[str, Any],
    *,
    excluded_by: Iterable[str],
) -> list[str]:
    reasons = set(_string_list(candidate.get("reasons")))
    blocked_by = set(_string_list(candidate.get("blocked_by")))
    exclusions = set(_string_list(list(excluded_by)))
    ordered_reasons: list[str] = []
    for reason in (
        "missing_source_event",
        "missing_source_artifact_path",
        "source_artifact_not_durable",
        "missing_source_artifact",
        "unreadable_source_artifact",
        "invalid_event_contract",
        "missing_supervised_text",
        "review_unresolved",
        "test_fail",
        "failed",
        "rejected",
        "noisy",
        "unresolved",
    ):
        if reason in exclusions or reason in reasons or reason in blocked_by:
            ordered_reasons.append(reason)
    trace_blockers = {
        "review_unresolved_trace": "review_unresolved",
        "test_fail_trace": "test_fail",
        "stale_review_resolved_trace": "review_unresolved",
        "stale_test_pass_trace": "test_fail",
        "rejected_trace": "rejected",
        "stale_accepted_trace": "rejected",
        "comparison_winner_missing_source_event": "missing_source_event",
        "comparison_winner_missing_source_artifact": "missing_source_artifact",
        "comparison_winner_source_artifact_not_durable": "source_artifact_not_durable",
        "comparison_winner_unreadable_source_artifact": "unreadable_source_artifact",
        "comparison_winner_invalid_event_contract": "invalid_event_contract",
    }
    for exclusion, reason in trace_blockers.items():
        if exclusion in exclusions and reason not in ordered_reasons:
            ordered_reasons.append(reason)
    if not ordered_reasons and _clean_text(candidate.get("state")) == "blocked":
        ordered_reasons.append("state_blocked")
    if not ordered_reasons and "blocking_or_noisy_signal" in exclusions:
        ordered_reasons.append("blocking_or_noisy_signal")
    return ordered_reasons


def _learning_queue_state(
    candidate: Mapping[str, Any],
    *,
    excluded_by: Iterable[str],
    blocked_reasons: Iterable[str],
) -> str:
    exclusions = set(_string_list(list(excluded_by)))
    if exclusions & LEARNING_MISSING_SOURCE_EXCLUSIONS:
        return "missing_source"
    if "missing_supervised_text" in exclusions:
        return "missing_supervised_text"
    if _string_list(list(blocked_reasons)):
        return "blocked"
    if not exclusions:
        return "ready"
    return "needs_review"


def _learning_queue_priority(queue_state: str, *, policy_confirmation: Mapping[str, Any]) -> dict[str, Any]:
    if queue_state in {"blocked", "missing_source", "missing_supervised_text"}:
        return {
            "rank": 1,
            "bucket": "blocked_first",
            "reason": "Resolve blocking or missing-source evidence before supervised review.",
        }
    if queue_state == "ready":
        if bool(_mapping_dict(policy_confirmation).get("confirmed")):
            return {
                "rank": 3,
                "bucket": "ready_policy_confirmed",
                "reason": "Policy confirmation is recorded; downstream export still requires separate approval.",
            }
        return {
            "rank": 2,
            "bucket": "ready_policy_unconfirmed",
            "reason": "Confirm export policy before any downstream dataset work.",
        }
    return {
        "rank": 4,
        "bucket": "needs_review",
        "reason": "Record missing test or human-selection evidence before supervised review.",
    }


def _learning_effective_curation_candidate(
    candidate: Mapping[str, Any],
    *,
    policy_confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    effective = copy.deepcopy(dict(candidate))
    reasons = [
        reason
        for reason in _string_list(candidate.get("reasons"))
        if reason != EXPORT_POLICY_CONFIRMATION_SIGNAL_KIND
    ]
    if bool(_mapping_dict(policy_confirmation).get("confirmed")):
        reasons.append(EXPORT_POLICY_CONFIRMATION_SIGNAL_KIND)
    effective["reasons"] = reasons
    effective["blocked_by"] = [
        reason
        for reason in reasons
        if reason in CURATION_BLOCKING_REASONS
    ]
    effective["export_decision"] = _curation_export_decision(effective)
    effective["ready_for_policy"] = _curation_candidate_ready_for_policy(effective)
    effective["adoption_checklist"] = _curation_candidate_adoption_checklist(effective)
    effective["required_next_steps"] = _curation_required_next_steps(effective)
    return effective


def _learning_next_action(
    candidate: Mapping[str, Any],
    *,
    excluded_by: Iterable[str],
    blocked_reasons: Iterable[str],
    policy_confirmation: Mapping[str, Any],
) -> str:
    reasons = set(_string_list(candidate.get("reasons")))
    exclusions = set(_string_list(list(excluded_by)))
    blocked = set(_string_list(list(blocked_reasons)))
    policy_confirmed = bool(_mapping_dict(policy_confirmation).get("confirmed"))
    if "missing_source_event" in exclusions:
        return "restore_source_event"
    if "comparison_winner_missing_source_event" in exclusions:
        return "restore_source_event"
    if "missing_source_artifact_path" in exclusions:
        return "record_source_artifact_path"
    if exclusions & {
        "missing_source_artifact",
        "source_artifact_not_durable",
        "comparison_winner_missing_source_artifact",
        "comparison_winner_source_artifact_not_durable",
    }:
        return "restore_source_artifact"
    if exclusions & {"unreadable_source_artifact", "comparison_winner_unreadable_source_artifact"}:
        return "restore_readable_source_artifact"
    if "invalid_event_contract" in exclusions:
        return "repair_event_contract"
    if "comparison_winner_invalid_event_contract" in exclusions:
        return "repair_event_contract"
    if "missing_supervised_text" in exclusions:
        return "restore_instruction_or_response_excerpt"
    if "review_unresolved" in blocked:
        return "resolve_review_before_export"
    if "test_fail" in blocked or "failed" in blocked:
        return "repair_or_follow_up_failure"
    if "rejected" in blocked:
        return "replace_or_rework_candidate"
    if blocked & {"noisy", "unresolved", "blocking_or_noisy_signal", "state_blocked"}:
        return "review_blocking_evidence"
    if not exclusions:
        if policy_confirmed:
            return "review_downstream_export_policy"
        return "confirm_export_policy"
    if "missing_test_pass" in exclusions or "missing_test_pass_trace" in exclusions:
        return "record_test_pass"
    if "missing_comparison_winner_trace" in exclusions:
        return "record_comparison_winner_trace"
    if (
        "comparison_winner_missing_source_event" in exclusions
        or "comparison_winner_missing_source_artifact" in exclusions
        or "comparison_winner_invalid_event_contract" in exclusions
        or "stale_accepted_trace" in exclusions
        or "stale_review_resolved_trace" in exclusions
        or "stale_test_pass_trace" in exclusions
        or "missing_human_or_comparison_selection" in exclusions
        or "missing_selection_trace" in exclusions
        or not (reasons & LEARNING_SELECTION_REASONS)
    ):
        return "record_acceptance_or_review_resolution"
    if "export_decision_not_include" in exclusions:
        return "review_export_decision"
    if "not_ready_for_policy" in exclusions:
        return "complete_adoption_checklist"
    required_next_steps = _string_list(candidate.get("required_next_steps"))
    return required_next_steps[0] if required_next_steps else "review_candidate"


def _learning_lifecycle_summary(
    candidate: Mapping[str, Any],
    *,
    event: Mapping[str, Any] | None,
    excluded_by: Iterable[str],
    policy_confirmation: Mapping[str, Any],
    source_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reasons = set(_string_list(candidate.get("reasons")))
    exclusions = set(_string_list(list(excluded_by)))
    missing_source = bool(exclusions & LEARNING_MISSING_SOURCE_EXCLUSIONS)
    missing_source_event = "missing_source_event" in exclusions
    policy_confirmed = bool(_mapping_dict(policy_confirmation).get("confirmed"))
    outcome = _mapping_dict(event.get("outcome")) if event is not None else {}
    source_artifact = _mapping_dict(_mapping_dict(source_contract).get("source_artifact"))
    source_artifact_state = _clean_text(source_artifact.get("source_status"))
    if missing_source_event:
        test_state = "missing_trace"
    elif "test_fail_trace" in exclusions or "stale_test_pass_trace" in exclusions:
        test_state = "failed"
    elif "missing_test_pass_trace" in exclusions:
        test_state = "missing_trace"
    elif "test_fail" in reasons or "failed" in reasons:
        test_state = "failed"
    elif "test_pass" in reasons:
        test_state = "passed"
    else:
        test_state = "missing"
    if missing_source_event:
        review_state = "unknown"
    elif "review_unresolved_trace" in exclusions or "stale_review_resolved_trace" in exclusions:
        review_state = "unresolved"
    elif "missing_selection_trace" in exclusions and "review_resolved" in reasons:
        review_state = "missing_trace"
    elif "review_unresolved" in reasons:
        review_state = "unresolved"
    elif "review_resolved" in reasons:
        review_state = "resolved"
    else:
        review_state = "not_recorded"
    if missing_source_event:
        selection_state = "missing_trace"
    elif "rejected_trace" in exclusions or "stale_accepted_trace" in exclusions:
        selection_state = "rejected"
    elif "rejected" in reasons:
        selection_state = "rejected"
    elif "missing_selection_trace" in exclusions:
        selection_state = "missing_trace"
    elif reasons & LEARNING_SELECTION_REASONS:
        selection_state = "selected"
    else:
        selection_state = "missing"
    if policy_confirmed and bool(candidate.get("ready_for_policy")):
        policy_state = "confirmed"
    elif policy_confirmed:
        policy_state = "confirmed_but_not_ready"
    elif EXPORT_POLICY_CONFIRMATION_SIGNAL_KIND in reasons:
        policy_state = "missing_trace"
    elif bool(candidate.get("ready_for_policy")):
        policy_state = "pending_confirmation"
    else:
        policy_state = "not_ready"
    if "missing_supervised_text" in exclusions:
        supervised_text_state = "missing"
    elif event is not None:
        supervised_text_state = "available"
    else:
        supervised_text_state = "unknown"
    return {
        "curation_state": _clean_text(candidate.get("state")) or "needs_review",
        "source_state": (
            "missing_source"
            if missing_source
            else "available"
            if event is not None
            else "missing"
        ),
        "source_artifact_state": source_artifact_state,
        "test_state": test_state,
        "review_state": review_state,
        "selection_state": selection_state,
        "policy_state": policy_state,
        "supervised_text_state": supervised_text_state,
        "quality_status": _clean_text(outcome.get("quality_status")),
        "execution_status": _clean_text(outcome.get("execution_status")),
    }


def _learning_review_queue_item(
    *,
    workspace_id: str,
    candidate: Mapping[str, Any],
    event: Mapping[str, Any] | None,
    excluded_by: Iterable[str],
    policy_confirmation: Mapping[str, Any],
    source_contract: Mapping[str, Any] | None = None,
    source_index: int,
) -> dict[str, Any]:
    event_id = _clean_text(candidate.get("event_id"))
    exclusions = _string_list(list(excluded_by))
    blocked_reasons = _learning_candidate_blocked_reasons(candidate, excluded_by=exclusions)
    queue_state = _learning_queue_state(
        candidate,
        excluded_by=exclusions,
        blocked_reasons=blocked_reasons,
    )
    source_event = (
        _learning_review_queue_source_event_record(event, event_id=event_id)
        if event_id is not None and event is not None
        else {"source_event_id": event_id}
    )
    fallback_key = (
        None
        if event_id is not None
        else json.dumps(dict(candidate), ensure_ascii=False, sort_keys=True, default=str)
    )
    queue_item_id = _stable_learning_review_queue_item_id(
        workspace_id=workspace_id,
        event_id=event_id,
        fallback_key=fallback_key,
        source_index=source_index,
    )
    effective_curation = _learning_effective_curation_candidate(
        candidate,
        policy_confirmation=policy_confirmation,
    )
    return {
        "schema_name": LEARNING_REVIEW_QUEUE_ITEM_SCHEMA_NAME,
        "schema_version": LEARNING_REVIEW_QUEUE_ITEM_SCHEMA_VERSION,
        "queue_item_id": queue_item_id,
        "workspace_id": workspace_id,
        "event_id": event_id,
        "source_index": source_index,
        "label": _preview_excerpt(
            _clean_text(candidate.get("label")) or event_id or queue_item_id,
            limit=240,
        ) or queue_item_id,
        "queue_state": queue_state,
        "queue_priority": _learning_queue_priority(
            queue_state,
            policy_confirmation=policy_confirmation,
        ),
        "next_action": _learning_next_action(
            candidate,
            excluded_by=exclusions,
            blocked_reasons=blocked_reasons,
            policy_confirmation=policy_confirmation,
        ),
        "blocked_reason": blocked_reasons[0] if blocked_reasons else None,
        "blocked_reasons": blocked_reasons,
        "eligible_for_supervised_candidate": not exclusions,
        "excluded_by": exclusions,
        "lifecycle_summary": _learning_lifecycle_summary(
            candidate,
            event=event,
            excluded_by=exclusions,
            policy_confirmation=policy_confirmation,
            source_contract=source_contract,
        ),
        "export_policy_confirmation": copy.deepcopy(_mapping_dict(policy_confirmation)),
        "event_contract": copy.deepcopy(_mapping_dict(source_contract)),
        "source_event": source_event,
        "curation": {
            "state": _clean_text(effective_curation.get("state")) or "needs_review",
            "reasons": _string_list(effective_curation.get("reasons")),
            "blocked_by": _string_list(effective_curation.get("blocked_by")),
            "export_decision": _clean_text(effective_curation.get("export_decision")) or "hold_for_review",
            "ready_for_policy": bool(effective_curation.get("ready_for_policy")),
            "adoption_checklist": copy.deepcopy(list(effective_curation.get("adoption_checklist") or [])),
            "required_next_steps": _string_list(effective_curation.get("required_next_steps")),
        },
    }


def _learning_review_queue_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "queue_item_id": _clean_text(item.get("queue_item_id")),
        "queue_state": _clean_text(item.get("queue_state")),
        "queue_priority": copy.deepcopy(_mapping_dict(item.get("queue_priority"))),
        "next_action": _clean_text(item.get("next_action")),
        "blocked_reason": _clean_text(item.get("blocked_reason")),
        "blocked_reasons": _string_list(item.get("blocked_reasons")),
        "eligible_for_supervised_candidate": bool(item.get("eligible_for_supervised_candidate")),
        "excluded_by": _string_list(item.get("excluded_by")),
        "lifecycle_summary": copy.deepcopy(_mapping_dict(item.get("lifecycle_summary"))),
        "export_policy_confirmation": copy.deepcopy(_mapping_dict(item.get("export_policy_confirmation"))),
        "event_contract": copy.deepcopy(_mapping_dict(item.get("event_contract"))),
    }


def _learning_candidate_exclusions(candidate: Mapping[str, Any]) -> list[str]:
    state = _clean_text(candidate.get("state"))
    export_decision = _clean_text(candidate.get("export_decision"))
    reasons = set(_string_list(candidate.get("reasons")))
    blocked_by = set(_string_list(candidate.get("blocked_by")))
    exclusions: list[str] = []
    if state != "ready":
        exclusions.append("state_not_ready")
    if export_decision != "include_when_approved":
        exclusions.append("export_decision_not_include")
    if not bool(candidate.get("ready_for_policy")):
        exclusions.append("not_ready_for_policy")
    if blocked_by or reasons & LEARNING_BLOCKING_REASONS:
        exclusions.append("blocking_or_noisy_signal")
    if "test_pass" not in reasons:
        exclusions.append("missing_test_pass")
    if not (reasons & LEARNING_SELECTION_REASONS):
        exclusions.append("missing_human_or_comparison_selection")
    return exclusions


def _evaluation_consistency_by_event_id(snapshot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    report = _mapping_dict(snapshot.get("evaluation_consistency"))
    return {
        event_id: dict(item)
        for item in report.get("events") or []
        if isinstance(item, Mapping)
        if (event_id := _clean_text(item.get("event_id"))) is not None
    }


def _fresh_learning_consistency_by_event_id(
    *,
    signal_traces: Iterable[Mapping[str, Any]],
    comparison_traces: Iterable[Mapping[str, Any]],
    curation_candidates: Iterable[Mapping[str, Any]],
    events_by_id: Mapping[str, Mapping[str, Any]],
    workspace_id: str,
    root: Path | None,
) -> dict[str, dict[str, Any]]:
    report = build_evaluation_consistency_report(
        signal_summaries=[_signal_summary(signal) for signal in signal_traces],
        comparison_summaries=[_comparison_summary(comparison) for comparison in comparison_traces],
        curation_candidates=curation_candidates,
        events_by_id=events_by_id,
        workspace_id=workspace_id,
        root=root,
    )
    return {
        event_id: dict(item)
        for item in report.get("events") or []
        if isinstance(item, Mapping)
        if (event_id := _clean_text(item.get("event_id"))) is not None
    }


def _learning_consistency_exclusions(consistency_item: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(consistency_item, Mapping):
        return []
    stale_reasons = set(_string_list(consistency_item.get("stale_positive_signals")))
    missing_traces = set(_string_list(consistency_item.get("missing_traces")))
    exclusions: list[str] = []
    if "accepted" in stale_reasons:
        exclusions.append("stale_accepted_trace")
    if "review_resolved" in stale_reasons:
        exclusions.append("stale_review_resolved_trace")
    if "test_pass" in stale_reasons:
        exclusions.append("stale_test_pass_trace")
    for missing_trace in (
        "missing_comparison_winner_trace",
        "comparison_winner_missing_source_event",
        "comparison_winner_missing_source_artifact",
        "comparison_winner_source_artifact_not_durable",
        "comparison_winner_unreadable_source_artifact",
        "comparison_winner_invalid_event_contract",
    ):
        if missing_trace in missing_traces:
            exclusions.append(missing_trace)
    return exclusions


def _learning_source_contract_exclusions(source_contract: Mapping[str, Any]) -> list[str]:
    contract_status = _clean_text(source_contract.get("contract_status"))
    source_artifact = _mapping_dict(source_contract.get("source_artifact"))
    source_reasons = set(_string_list(source_artifact.get("reasons")))
    exclusions: list[str] = []
    if "missing_source_artifact_path" in source_reasons:
        exclusions.append("missing_source_artifact_path")
    if "source_artifact_outside_workspace" in source_reasons:
        exclusions.append("source_artifact_not_durable")
    if "source_artifact_missing" in source_reasons:
        exclusions.append("missing_source_artifact")
    if source_reasons & {"source_artifact_not_file", "source_artifact_unreadable"}:
        exclusions.append("unreadable_source_artifact")
    if contract_status == "invalid_event_contract":
        exclusions.append("invalid_event_contract")
    return exclusions


def _learning_traceability_exclusions(
    *,
    signals: Iterable[Mapping[str, Any]],
    comparisons: Iterable[Mapping[str, Any]],
    expected_reasons: Iterable[str] | None = None,
) -> list[str]:
    signal_list = [
        dict(signal)
        for signal in signals
        if isinstance(signal, Mapping)
    ]
    latest_test_signal = _learning_latest_signal_kind_from_traces(
        signal_list,
        signal_kinds={"test_pass", "test_fail"},
    )
    latest_selection_signal = _learning_latest_signal_kind_from_traces(
        signal_list,
        signal_kinds={"acceptance", "rejection", "review_resolved", "review_unresolved"},
    )
    has_test_pass = latest_test_signal == "test_pass"
    has_selection_signal = latest_selection_signal in {"acceptance", "review_resolved"}
    has_comparison_winner = any(
        _clean_text(comparison.get("role")) == "winner"
        and _clean_text(comparison.get("outcome")) == "winner_selected"
        for comparison in comparisons
        if isinstance(comparison, Mapping)
    )
    reasons = set(_string_list(list(expected_reasons or [])))
    exclusions: list[str] = []
    if not has_test_pass:
        exclusions.append("missing_test_pass_trace")
    if "comparison_winner" in reasons and not has_comparison_winner:
        exclusions.append("missing_comparison_winner_trace")
    if not (has_selection_signal or has_comparison_winner):
        exclusions.append("missing_selection_trace")
    return exclusions


def _learning_latest_signal_kind_from_traces(
    signals: Iterable[Mapping[str, Any]],
    *,
    signal_kinds: set[str],
) -> str | None:
    matching_signals = [
        dict(signal)
        for signal in signals
        if isinstance(signal, Mapping)
        if _clean_text(signal.get("signal_kind")) in signal_kinds
    ]
    matching_signals.sort(
        key=lambda signal: (
            _timestamp_sort_key(signal.get("recorded_at_utc")),
            str(signal.get("signal_id") or ""),
        ),
        reverse=True,
    )
    if not matching_signals:
        return None
    return _clean_text(matching_signals[0].get("signal_kind"))


def _learning_latest_review_signal_kind_from_traces(signals: Iterable[Mapping[str, Any]]) -> str | None:
    return _learning_latest_signal_kind_from_traces(
        signals,
        signal_kinds={"review_resolved", "review_unresolved"},
    )


def _learning_blocking_trace_exclusions(*, signals: Iterable[Mapping[str, Any]]) -> list[str]:
    signal_list = [
        dict(signal)
        for signal in signals
        if isinstance(signal, Mapping)
    ]
    latest_selection_signal = _learning_latest_signal_kind_from_traces(
        signal_list,
        signal_kinds={"acceptance", "rejection"},
    )
    latest_test_signal = _learning_latest_signal_kind_from_traces(
        signal_list,
        signal_kinds={"test_pass", "test_fail"},
    )
    exclusions: list[str] = []
    if latest_selection_signal == "rejection":
        exclusions.append("rejected_trace")
        if any(_clean_text(signal.get("signal_kind")) == "acceptance" for signal in signal_list):
            exclusions.append("stale_accepted_trace")
    if _learning_latest_review_signal_kind_from_traces(signal_list) == "review_unresolved":
        exclusions.append("review_unresolved_trace")
        if any(_clean_text(signal.get("signal_kind")) == "review_resolved" for signal in signal_list):
            exclusions.append("stale_review_resolved_trace")
    if latest_test_signal == "test_fail":
        exclusions.append("test_fail_trace")
        if any(_clean_text(signal.get("signal_kind")) == "test_pass" for signal in signal_list):
            exclusions.append("stale_test_pass_trace")
    return exclusions


def _learning_excluded_candidate(
    candidate: Mapping[str, Any],
    *,
    excluded_by: Iterable[str],
    queue_item: Mapping[str, Any],
) -> dict[str, Any]:
    queue_summary = _learning_review_queue_summary(queue_item)
    curation = _mapping_dict(queue_item.get("curation"))
    return {
        "event_id": _clean_text(candidate.get("event_id")),
        "state": _clean_text(curation.get("state")) or "needs_review",
        "reasons": _string_list(curation.get("reasons")),
        "blocked_by": _string_list(curation.get("blocked_by")),
        "export_decision": _clean_text(curation.get("export_decision")) or "hold_for_review",
        "excluded_by": _string_list(excluded_by),
        "queue_state": queue_summary["queue_state"],
        "queue_priority": queue_summary["queue_priority"],
        "next_action": queue_summary["next_action"],
        "blocked_reason": queue_summary["blocked_reason"],
        "blocked_reasons": queue_summary["blocked_reasons"],
        "eligible_for_supervised_candidate": queue_summary["eligible_for_supervised_candidate"],
        "lifecycle_summary": queue_summary["lifecycle_summary"],
        "export_policy_confirmation": queue_summary["export_policy_confirmation"],
        "event_contract": queue_summary["event_contract"],
    }


def _build_supervised_example_candidate(
    *,
    workspace_id: str,
    candidate: Mapping[str, Any],
    event: Mapping[str, Any],
    signals: Iterable[Mapping[str, Any]],
    comparisons: Iterable[Mapping[str, Any]],
    source_paths: Mapping[str, Any],
    queue_item: Mapping[str, Any],
) -> dict[str, Any]:
    event_id = _clean_text(candidate.get("event_id"))
    if event_id is None:
        raise ValueError("Learning candidates require an event_id.")
    source_refs = _mapping_dict(event.get("source_refs"))
    artifact_ref = _mapping_dict(source_refs.get("artifact_ref"))
    queue_summary = _learning_review_queue_summary(queue_item)
    event_contract = _mapping_dict(queue_summary.get("event_contract"))
    contract_source_artifact = _mapping_dict(event_contract.get("source_artifact"))
    contract_source_artifact_reasons = _string_list(contract_source_artifact.get("reasons"))
    export_policy_confirmation = _mapping_dict(queue_summary.get("export_policy_confirmation"))
    effective_curation = _mapping_dict(queue_item.get("curation"))
    return {
        "schema_name": SUPERVISED_EXAMPLE_CANDIDATE_SCHEMA_NAME,
        "schema_version": SUPERVISED_EXAMPLE_CANDIDATE_SCHEMA_VERSION,
        "candidate_id": _stable_learning_candidate_id(workspace_id=workspace_id, event_id=event_id),
        "workspace_id": workspace_id,
        "event_id": event_id,
        "example_kind": "software_work_supervised_candidate",
        "source_event": _learning_source_event_record(event, event_id=event_id),
        "supervised_example": _learning_supervised_example(event),
        "curation": {
            "state": _clean_text(effective_curation.get("state")),
            "reasons": _string_list(effective_curation.get("reasons")),
            "export_decision": _clean_text(effective_curation.get("export_decision")),
            "ready_for_policy": bool(effective_curation.get("ready_for_policy")),
            "adoption_checklist": copy.deepcopy(list(effective_curation.get("adoption_checklist") or [])),
            "required_next_steps": _string_list(effective_curation.get("required_next_steps")),
        },
        "evidence": {
            "signals": list(signals),
            "comparisons": list(comparisons),
        },
        "backend_metadata": _learning_backend_metadata(event),
        "review_queue": queue_summary,
        "event_contract": queue_summary["event_contract"],
        "source_paths": {
            **copy.deepcopy(dict(source_paths)),
            "source_artifact_path": _clean_text(artifact_ref.get("artifact_path")),
            "source_artifact_workspace_relative_path": _clean_text(
                artifact_ref.get("artifact_workspace_relative_path")
            ),
            "source_artifact_state": _clean_text(contract_source_artifact.get("source_status")),
            "source_artifact_reason": (
                contract_source_artifact_reasons[0] if contract_source_artifact_reasons else None
            ),
        },
        "policy": {
            "export_mode": "preview_only",
            "human_gate_required": True,
            "training_job_allowed": False,
            "raw_log_export_allowed": False,
            "export_policy_confirmed": bool(export_policy_confirmation.get("confirmed")),
            "confirmation_signal_id": _clean_text(export_policy_confirmation.get("latest_signal_id")),
        },
    }


def build_learning_dataset_preview(
    snapshot: Mapping[str, Any],
    curation_preview: Mapping[str, Any],
    *,
    workspace_id: str | None = None,
    events_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    explicit_signals: Iterable[Mapping[str, Any]] | None = None,
    comparisons: Iterable[Mapping[str, Any]] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    if limit is not None and limit <= 0:
        raise ValueError("Learning preview limit must be a positive integer.")
    resolved_workspace_id = (
        _clean_text(workspace_id)
        or _clean_text(snapshot.get("workspace_id"))
        or DEFAULT_WORKSPACE_ID
    )
    source_paths = _learning_source_paths(snapshot=snapshot, curation_preview=curation_preview)
    resolved_events_by_id = (
        {str(key): dict(value) for key, value in events_by_id.items()}
        if events_by_id is not None
        else _read_events_by_id_from_snapshot(snapshot)
    )
    resolved_signals = (
        [dict(signal) for signal in explicit_signals]
        if explicit_signals is not None
        else _read_signals_from_snapshot(snapshot)
    )
    resolved_comparisons = (
        [dict(comparison) for comparison in comparisons]
        if comparisons is not None
        else _read_comparisons_from_snapshot(snapshot)
    )

    source_candidates = [
        dict(candidate)
        for candidate in curation_preview.get("candidates") or []
        if isinstance(candidate, Mapping)
    ]
    contract_root = _learning_contract_root(source_paths)
    has_trace_overrides = (
        explicit_signals is not None
        or comparisons is not None
        or events_by_id is not None
    )
    consistency_by_event_id = (
        _fresh_learning_consistency_by_event_id(
            signal_traces=resolved_signals,
            comparison_traces=resolved_comparisons,
            curation_candidates=source_candidates,
            events_by_id=resolved_events_by_id,
            workspace_id=resolved_workspace_id,
            root=contract_root,
        )
        if has_trace_overrides
        else _evaluation_consistency_by_event_id(snapshot)
    )
    eligible_candidates: list[dict[str, Any]] = []
    excluded_candidates: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    for source_index, candidate in enumerate(source_candidates):
        event_id = _clean_text(candidate.get("event_id"))
        exclusions = _learning_candidate_exclusions(candidate)
        event = resolved_events_by_id.get(event_id) if event_id is not None else None
        source_contract: dict[str, Any] | None = None
        signal_traces: list[dict[str, Any]] = []
        comparison_traces: list[dict[str, Any]] = []
        policy_confirmation: dict[str, Any] = _learning_policy_confirmation_trace([])
        if event_id is None or event is None:
            exclusions.append("missing_source_event")
        else:
            source_contract = build_event_contract_check(
                event,
                root=_learning_event_contract_root(event, fallback_root=contract_root),
            )
            exclusions.extend(_learning_source_contract_exclusions(source_contract))
            supervised_example = _learning_supervised_example(event)
            missing_instruction = _clean_text(supervised_example.get("instruction")) is None
            missing_response = _clean_text(supervised_example.get("response")) is None
            if missing_instruction or missing_response:
                exclusions.append("missing_supervised_text")
            signal_traces = _learning_signals_for_event(
                event_id=event_id,
                event=event,
                explicit_signals=resolved_signals,
                workspace_id=resolved_workspace_id,
            )
            comparison_traces = _learning_comparisons_for_event(
                event_id=event_id,
                comparisons=resolved_comparisons,
            )
            policy_confirmation = _learning_policy_confirmation_trace(signal_traces)
            exclusions.extend(_learning_blocking_trace_exclusions(signals=signal_traces))
            exclusions.extend(
                _learning_consistency_exclusions(consistency_by_event_id.get(event_id))
            )
            if not exclusions:
                exclusions.extend(
                    _learning_traceability_exclusions(
                        signals=signal_traces,
                        comparisons=comparison_traces,
                        expected_reasons=candidate.get("reasons"),
                    )
                )
        exclusions = _deduplicate_strings(exclusions)

        queue_item = _learning_review_queue_item(
            workspace_id=resolved_workspace_id,
            candidate=candidate,
            event=event,
            excluded_by=exclusions,
            policy_confirmation=policy_confirmation,
            source_contract=source_contract,
            source_index=source_index,
        )
        review_queue.append(queue_item)
        if exclusions:
            excluded_candidates.append(
                _learning_excluded_candidate(
                    candidate,
                    excluded_by=exclusions,
                    queue_item=queue_item,
                )
            )
            continue

        eligible_candidates.append(
            _build_supervised_example_candidate(
                workspace_id=resolved_workspace_id,
                candidate=candidate,
                event=event,
                signals=signal_traces,
                comparisons=comparison_traces,
                source_paths=source_paths,
                queue_item=queue_item,
            )
        )

    review_queue.sort(
        key=lambda item: (
            int(_mapping_dict(item.get("queue_priority")).get("rank") or 99),
            str(item.get("event_id") or ""),
        )
    )
    preview_candidates = eligible_candidates[:limit] if isinstance(limit, int) else eligible_candidates
    exclusion_counts = Counter(
        reason
        for candidate in excluded_candidates
        for reason in _string_list(candidate.get("excluded_by"))
    )
    review_queue_state_counts = Counter(
        str(item.get("queue_state") or "needs_review")
        for item in review_queue
    )
    review_queue_priority_counts = Counter(
        str(_mapping_dict(item.get("queue_priority")).get("bucket") or "unknown")
        for item in review_queue
    )
    policy_confirmed_candidate_count = sum(
        1
        for item in review_queue
        if bool(_mapping_dict(item.get("export_policy_confirmation")).get("confirmed"))
    )
    policy_pending_candidate_count = sum(
        1
        for item in review_queue
        if _mapping_dict(item.get("lifecycle_summary")).get("policy_state") == "pending_confirmation"
    )
    return {
        "schema_name": LEARNING_DATASET_PREVIEW_SCHEMA_NAME,
        "schema_version": LEARNING_DATASET_PREVIEW_SCHEMA_VERSION,
        "workspace_id": resolved_workspace_id,
        "generated_at_utc": timestamp_utc(),
        "export_mode": "preview_only",
        "training_export_ready": False,
        "human_gate_required": True,
        "source_snapshot_path": source_paths.get("source_snapshot_path"),
        "source_curation_preview_path": source_paths.get("source_curation_preview_path"),
        "source_paths": source_paths,
        "counts": {
            "source_candidate_count": len(source_candidates),
            "eligible_candidate_count": len(eligible_candidates),
            "previewed_candidate_count": len(preview_candidates),
            "excluded_candidate_count": len(excluded_candidates),
            "review_queue_count": len(review_queue),
            "exclusion_reasons": {
                key: int(value)
                for key, value in sorted(exclusion_counts.items())
            },
            "review_queue_states": {
                key: int(value)
                for key, value in sorted(review_queue_state_counts.items())
            },
            "review_queue_priorities": {
                key: int(value)
                for key, value in sorted(review_queue_priority_counts.items())
            },
            "policy_confirmed_candidate_count": policy_confirmed_candidate_count,
            "policy_pending_candidate_count": policy_pending_candidate_count,
        },
        "truncated": len(eligible_candidates) > len(preview_candidates),
        "review_queue": review_queue,
        "supervised_example_candidates": preview_candidates,
        "excluded_candidates": excluded_candidates,
        "export_policy": {
            "mode": "preview_only",
            "human_gate_required": True,
            "training_job_allowed": False,
            "requires_explicit_export_policy": True,
            "confirmation_signal_kind": EXPORT_POLICY_CONFIRMATION_SIGNAL_KIND,
            "confirmed_candidate_count": policy_confirmed_candidate_count,
            "default_exclusions": sorted(LEARNING_BLOCKING_REASONS),
            "required_positive_evidence": [
                "test_pass",
                "accepted_or_review_resolved_or_comparison_winner",
            ],
        },
        "notes": [
            "Preview only; no trainable dataset file is produced.",
            (
                "Raw logs are not exported. Candidates are built from curated "
                "software-work events and evaluation evidence."
            ),
            "A human export policy must be recorded before downstream training export or fine-tuning.",
        ],
    }


def record_learning_dataset_preview(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    snapshot: Mapping[str, Any] | None = None,
    curation_preview: Mapping[str, Any] | None = None,
    curation_filters: Mapping[str, Any] | None = None,
    limit: int | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    resolved_root = _resolve_root(root)
    source_snapshot = snapshot or build_evaluation_snapshot(root=resolved_root, workspace_id=workspace_id)
    source_curation_preview = curation_preview
    if source_curation_preview is None:
        # Preserve a file-first curation artifact so the learning preview has a durable source.
        source_curation_preview, _curation_latest_path, _curation_run_path = record_curation_export_preview(
            root=resolved_root,
            workspace_id=workspace_id,
            snapshot=source_snapshot,
            filters=curation_filters,
        )
    elif _curation_preview_artifact_path(source_curation_preview) is None:
        source_curation_preview = _record_supplied_curation_preview(
            source_curation_preview,
            root=resolved_root,
            workspace_id=workspace_id,
        )
    preview = build_learning_dataset_preview(
        source_snapshot,
        source_curation_preview,
        workspace_id=workspace_id,
        limit=limit,
    )
    latest_path = learning_dataset_preview_latest_path(workspace_id=workspace_id, root=resolved_root)
    run_path = learning_dataset_preview_run_path(workspace_id=workspace_id, root=resolved_root)
    preview["paths"] = {
        "learning_preview_latest_path": str(latest_path),
        "learning_preview_run_path": str(run_path),
        "source_snapshot_path": _clean_text(preview.get("source_snapshot_path")),
        "source_curation_preview_path": _clean_text(preview.get("source_curation_preview_path")),
    }
    write_json(run_path, preview)
    write_json(latest_path, preview)
    return preview, latest_path, run_path


def _learning_preview_artifact_path(preview: Mapping[str, Any]) -> str | None:
    paths = _mapping_dict(preview.get("paths"))
    return (
        _clean_text(paths.get("learning_preview_run_path"))
        or _clean_text(paths.get("learning_preview_latest_path"))
    )


def _learning_preview_artifact_is_readable(preview: Mapping[str, Any]) -> bool:
    artifact_path = _path_from_text(_learning_preview_artifact_path(preview))
    if artifact_path is None:
        return False
    try:
        if not artifact_path.is_file():
            return False
        with artifact_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, Mapping)
        and payload.get("schema_name") == LEARNING_DATASET_PREVIEW_SCHEMA_NAME
        and payload.get("schema_version") == LEARNING_DATASET_PREVIEW_SCHEMA_VERSION
    )


def _record_supplied_learning_preview(
    preview: Mapping[str, Any],
    *,
    root: Path,
    workspace_id: str,
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(preview))
    latest_path = learning_dataset_preview_latest_path(workspace_id=workspace_id, root=root)
    run_path = learning_dataset_preview_run_path(workspace_id=workspace_id, root=root)
    payload["paths"] = {
        "learning_preview_latest_path": str(latest_path),
        "learning_preview_run_path": str(run_path),
        "source_snapshot_path": _clean_text(payload.get("source_snapshot_path")),
        "source_curation_preview_path": _clean_text(payload.get("source_curation_preview_path")),
    }
    write_json(run_path, payload)
    write_json(latest_path, payload)
    return payload


def _normalize_human_selected_event_ids(event_ids: Iterable[str]) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for item in event_ids:
        event_id = _clean_text(item)
        if event_id is None or event_id in seen:
            continue
        seen.add(event_id)
        selected.append(event_id)
    if not selected:
        raise ValueError("Human-selected candidate lists require at least one selected event id.")
    return selected


def _stable_human_selected_candidate_id(*, workspace_id: str, event_id: str) -> str:
    digest = hashlib.sha256(f"{workspace_id}\nhuman-selected\n{event_id}".encode("utf-8")).hexdigest()[:12]
    return f"{workspace_id}:human-selected-candidate:{digest}"


def _human_selected_preview_source_path(learning_preview: Mapping[str, Any]) -> str | None:
    return _learning_preview_artifact_path(learning_preview)


def _items_by_event_id(items: Iterable[Any]) -> dict[str, dict[str, Any]]:
    by_event_id: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        event_id = _clean_text(item.get("event_id"))
        if event_id is None or event_id in by_event_id:
            continue
        by_event_id[event_id] = dict(item)
    return by_event_id


def _human_selected_events_by_id_from_learning_preview(preview: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    source_paths = _mapping_dict(preview.get("source_paths"))
    event_log_path = _path_from_text(source_paths.get("event_log_path"))
    if event_log_path is None:
        return {}
    try:
        event_log = read_event_log(event_log_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return {
        str(event.get("event_id")): dict(event)
        for event in event_log.get("events") or []
        if isinstance(event, Mapping)
        if _clean_text(event.get("event_id")) is not None
    }


def _human_selected_signals_from_learning_preview(preview: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_paths = _mapping_dict(preview.get("source_paths"))
    signal_log_path = _path_from_text(source_paths.get("signal_log_path"))
    if signal_log_path is None:
        return []
    try:
        return read_evaluation_signals(signal_log_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return []


def _human_selected_comparisons_from_learning_preview(preview: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_paths = _mapping_dict(preview.get("source_paths"))
    comparison_log_path = _path_from_text(source_paths.get("comparison_log_path"))
    if comparison_log_path is None:
        return []
    try:
        return read_evaluation_comparisons(comparison_log_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return []


def _human_selected_signal_traces_for_event(
    *,
    workspace_id: str,
    event_id: str,
    event: Mapping[str, Any] | None,
    explicit_signals: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if event is not None:
        return _learning_signals_for_event(
            event_id=event_id,
            event=event,
            explicit_signals=explicit_signals,
            workspace_id=workspace_id,
        )
    signals = [
        dict(signal)
        for signal in explicit_signals
        if isinstance(signal, Mapping)
        if _source_event_id(signal) == event_id
    ]
    deduped: dict[str, dict[str, Any]] = {}
    for signal in signals:
        signal_id = _clean_text(signal.get("signal_id")) or f"{event_id}:signal:{len(deduped)}"
        deduped[signal_id] = signal
    return [
        _learning_signal_trace(signal)
        for signal in _sort_signals(deduped.values())
    ]


def _human_selected_source_event_record(
    *,
    queue_item: Mapping[str, Any] | None,
    supervised_candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if queue_item is not None:
        source_event = _mapping_dict(queue_item.get("source_event"))
    elif supervised_candidate is not None:
        source_event = _mapping_dict(supervised_candidate.get("source_event"))
    else:
        source_event = {}
    source_record = copy.deepcopy(dict(source_event))
    source_record.pop("prompt_excerpt", None)
    source_record.pop("output_excerpt", None)
    return source_record


def _human_selected_evidence_summary(
    *,
    queue_item: Mapping[str, Any] | None,
    supervised_candidate: Mapping[str, Any] | None,
    excluded_candidate: Mapping[str, Any] | None,
    signal_traces: Iterable[Mapping[str, Any]],
    comparison_traces: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    curation = _mapping_dict(queue_item.get("curation")) if queue_item is not None else {}
    if not curation and supervised_candidate is not None:
        curation = _mapping_dict(supervised_candidate.get("curation"))
    if not curation and excluded_candidate is not None:
        curation = {
            "state": _clean_text(excluded_candidate.get("state")),
            "reasons": _string_list(excluded_candidate.get("reasons")),
            "blocked_by": _string_list(excluded_candidate.get("blocked_by")),
            "export_decision": _clean_text(excluded_candidate.get("export_decision")),
        }
    reasons = set(_string_list(curation.get("reasons")))
    evidence = _mapping_dict(supervised_candidate.get("evidence")) if supervised_candidate is not None else {}
    signal_items = [
        signal
        for signal in evidence.get("signals") or []
        if isinstance(signal, Mapping)
    ]
    signal_items.extend(
        signal
        for signal in signal_traces
        if isinstance(signal, Mapping)
    )
    signals_by_id: dict[str, dict[str, Any]] = {}
    for index, signal in enumerate(signal_items):
        signal_id = _clean_text(signal.get("signal_id")) or f"signal:{index}"
        signals_by_id[signal_id] = dict(signal)
    signals = list(signals_by_id.values())
    comparison_items = [
        comparison
        for comparison in evidence.get("comparisons") or []
        if isinstance(comparison, Mapping)
    ]
    comparison_items.extend(
        comparison
        for comparison in comparison_traces
        if isinstance(comparison, Mapping)
    )
    comparisons_by_id: dict[str, dict[str, Any]] = {}
    for index, comparison in enumerate(comparison_items):
        comparison_id = _clean_text(comparison.get("comparison_id")) or f"comparison:{index}"
        comparisons_by_id[comparison_id] = dict(comparison)
    comparisons = list(comparisons_by_id.values())
    signal_kinds = [
        signal_kind
        for signal in signals
        if (signal_kind := _clean_text(signal.get("signal_kind"))) is not None
    ]
    signal_ids = [
        signal_id
        for signal in signals
        if (signal_id := _clean_text(signal.get("signal_id"))) is not None
    ]
    comparison_roles = [
        role
        for comparison in comparisons
        if (role := _clean_text(comparison.get("role"))) is not None
    ]
    comparison_ids = [
        comparison_id
        for comparison in comparisons
        if (comparison_id := _clean_text(comparison.get("comparison_id"))) is not None
    ]
    has_winner_trace = any(
        _clean_text(comparison.get("role")) == "winner"
        and _clean_text(comparison.get("outcome")) == "winner_selected"
        for comparison in comparisons
    )
    export_policy_confirmation = (
        _mapping_dict(queue_item.get("export_policy_confirmation"))
        if queue_item is not None
        else _mapping_dict(_mapping_dict(supervised_candidate.get("review_queue")).get("export_policy_confirmation"))
        if supervised_candidate is not None
        else {}
    )
    return {
        "curation_reasons": sorted(reasons),
        "signal_kinds": signal_kinds,
        "signal_ids": signal_ids,
        "comparison_roles": comparison_roles,
        "comparison_ids": comparison_ids,
        "curation_reason_flags": {
            "test_pass": "test_pass" in reasons,
            "accepted": "accepted" in reasons,
            "review_resolved": "review_resolved" in reasons,
            "comparison_winner": "comparison_winner" in reasons,
            "export_policy_confirmed": EXPORT_POLICY_CONFIRMATION_SIGNAL_KIND in reasons,
        },
        "export_policy_confirmation_signal_id": _clean_text(export_policy_confirmation.get("latest_signal_id")),
        "traceability": {
            "test_pass": "test_pass" in signal_kinds,
            "accepted": "acceptance" in signal_kinds,
            "review_resolved": "review_resolved" in signal_kinds,
            "comparison_winner": has_winner_trace,
            "export_policy_confirmed": bool(export_policy_confirmation.get("confirmed")),
        },
    }


def _human_selected_policy(
    *,
    queue_summary: Mapping[str, Any],
    supervised_candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    supervised_policy = (
        _mapping_dict(supervised_candidate.get("policy"))
        if supervised_candidate is not None
        else {}
    )
    export_policy_confirmation = _mapping_dict(queue_summary.get("export_policy_confirmation"))
    return {
        "export_mode": "preview_only",
        "training_export_ready": False,
        "human_gate_required": True,
        "training_job_allowed": False,
        "raw_log_export_allowed": False,
        "export_policy_confirmed": bool(
            supervised_policy.get("export_policy_confirmed")
            or export_policy_confirmation.get("confirmed")
        ),
        "confirmation_signal_id": (
            _clean_text(supervised_policy.get("confirmation_signal_id"))
            or _clean_text(export_policy_confirmation.get("latest_signal_id"))
        ),
        "downstream_export_requires_separate_approval": True,
    }


def _human_selected_candidate_record(
    *,
    workspace_id: str,
    event_id: str,
    source_index: int,
    queue_item: Mapping[str, Any] | None,
    supervised_candidate: Mapping[str, Any] | None,
    excluded_candidate: Mapping[str, Any] | None,
    generated_at_utc: str,
    selection_origin: str,
    selection_rationale: str | None,
    source_learning_preview_path: str | None,
    learning_source_paths: Mapping[str, Any],
    signal_traces: Iterable[Mapping[str, Any]],
    comparison_traces: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    queue_summary = (
        _learning_review_queue_summary(queue_item)
        if queue_item is not None
        else {
            "queue_item_id": None,
            "queue_state": "missing_from_learning_preview",
            "queue_priority": {
                "rank": 99,
                "bucket": "missing_from_learning_preview",
                "reason": "Selected event id was not present in the source learning preview.",
            },
            "next_action": "refresh_learning_preview_or_check_event_id",
            "blocked_reason": "missing_learning_preview_candidate",
            "blocked_reasons": ["missing_learning_preview_candidate"],
            "eligible_for_supervised_candidate": False,
            "excluded_by": ["missing_learning_preview_candidate"],
            "lifecycle_summary": {
                "source_state": "missing_from_learning_preview",
                "test_state": "missing_trace",
                "review_state": "unknown",
                "selection_state": "missing_trace",
                "policy_state": "unknown",
                "supervised_text_state": "unknown",
            },
            "export_policy_confirmation": {
                "confirmed": False,
                "latest_signal_id": None,
                "recorded_at_utc": None,
                "origin": None,
            },
        }
    )
    supervised_candidate_id = (
        _clean_text(supervised_candidate.get("candidate_id"))
        if supervised_candidate is not None
        else None
    )
    return {
        "schema_name": HUMAN_SELECTED_CANDIDATE_ITEM_SCHEMA_NAME,
        "schema_version": HUMAN_SELECTED_CANDIDATE_ITEM_SCHEMA_VERSION,
        "selection_id": _stable_human_selected_candidate_id(workspace_id=workspace_id, event_id=event_id),
        "workspace_id": workspace_id,
        "event_id": event_id,
        "source_index": source_index,
        "selected_at_utc": generated_at_utc,
        "selection": {
            "origin": selection_origin,
            "rationale": selection_rationale,
            "selection_mode": "explicit_human_candidate_list",
        },
        "label": event_id,
        "preview_membership": {
            "in_review_queue": queue_item is not None,
            "in_supervised_example_candidates": supervised_candidate is not None,
            "in_excluded_candidates": excluded_candidate is not None,
        },
        "supervised_example_candidate_id": supervised_candidate_id,
        "eligible_for_supervised_candidate": bool(queue_summary.get("eligible_for_supervised_candidate")),
        "queue_state": _clean_text(queue_summary.get("queue_state")),
        "queue_priority": copy.deepcopy(_mapping_dict(queue_summary.get("queue_priority"))),
        "next_action": _clean_text(queue_summary.get("next_action")),
        "blocked_reason": _clean_text(queue_summary.get("blocked_reason")),
        "blocked_reasons": _string_list(queue_summary.get("blocked_reasons")),
        "excluded_by": _string_list(queue_summary.get("excluded_by")),
        "lifecycle_summary": copy.deepcopy(_mapping_dict(queue_summary.get("lifecycle_summary"))),
        "export_policy_confirmation": copy.deepcopy(_mapping_dict(queue_summary.get("export_policy_confirmation"))),
        "evidence_summary": _human_selected_evidence_summary(
            queue_item=queue_item,
            supervised_candidate=supervised_candidate,
            excluded_candidate=excluded_candidate,
            signal_traces=signal_traces,
            comparison_traces=comparison_traces,
        ),
        "source_event": _human_selected_source_event_record(
            queue_item=queue_item,
            supervised_candidate=supervised_candidate,
        ),
        "source_paths": {
            **copy.deepcopy(dict(learning_source_paths)),
            "source_learning_preview_path": source_learning_preview_path,
            **copy.deepcopy(_mapping_dict(supervised_candidate.get("source_paths")) if supervised_candidate is not None else {}),
        },
        "policy": _human_selected_policy(
            queue_summary=queue_summary,
            supervised_candidate=supervised_candidate,
        ),
    }


def build_human_selected_candidate_list(
    learning_preview: Mapping[str, Any],
    *,
    selected_event_ids: Iterable[str],
    workspace_id: str | None = None,
    rationale: str | None = None,
    origin: str = "manual",
    events_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    explicit_signals: Iterable[Mapping[str, Any]] | None = None,
    comparisons: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    selected_ids = _normalize_human_selected_event_ids(selected_event_ids)
    resolved_workspace_id = (
        _clean_text(workspace_id)
        or _clean_text(learning_preview.get("workspace_id"))
        or DEFAULT_WORKSPACE_ID
    )
    generated_at_utc = timestamp_utc()
    source_learning_preview_path = _human_selected_preview_source_path(learning_preview)
    queue_by_event = _items_by_event_id(learning_preview.get("review_queue") or [])
    supervised_by_event = _items_by_event_id(learning_preview.get("supervised_example_candidates") or [])
    excluded_by_event = _items_by_event_id(learning_preview.get("excluded_candidates") or [])
    source_paths = _mapping_dict(learning_preview.get("source_paths"))
    resolved_events_by_id = (
        {str(key): dict(value) for key, value in events_by_id.items()}
        if events_by_id is not None
        else {}
    )
    resolved_signals = (
        [dict(signal) for signal in explicit_signals if isinstance(signal, Mapping)]
        if explicit_signals is not None
        else []
    )
    resolved_comparisons = (
        [dict(comparison) for comparison in comparisons if isinstance(comparison, Mapping)]
        if comparisons is not None
        else []
    )
    selected_candidates = [
        _human_selected_candidate_record(
            workspace_id=resolved_workspace_id,
            event_id=event_id,
            source_index=index,
            queue_item=queue_by_event.get(event_id),
            supervised_candidate=supervised_by_event.get(event_id),
            excluded_candidate=excluded_by_event.get(event_id),
            generated_at_utc=generated_at_utc,
            selection_origin=_clean_text(origin) or "manual",
            selection_rationale=_clean_text(rationale),
            source_learning_preview_path=source_learning_preview_path,
            learning_source_paths=source_paths,
            signal_traces=_human_selected_signal_traces_for_event(
                workspace_id=resolved_workspace_id,
                event_id=event_id,
                event=resolved_events_by_id.get(event_id),
                explicit_signals=resolved_signals,
            ),
            comparison_traces=_learning_comparisons_for_event(
                event_id=event_id,
                comparisons=resolved_comparisons,
            ),
        )
        for index, event_id in enumerate(selected_ids)
    ]
    matched_candidates = [
        candidate
        for candidate in selected_candidates
        if _mapping_dict(candidate.get("preview_membership")).get("in_review_queue")
    ]
    supervised_selected = [
        candidate
        for candidate in selected_candidates
        if _mapping_dict(candidate.get("preview_membership")).get("in_supervised_example_candidates")
    ]
    missing_candidates = [
        candidate
        for candidate in selected_candidates
        if not _mapping_dict(candidate.get("preview_membership")).get("in_review_queue")
    ]
    policy_confirmed_selected_count = sum(
        1
        for candidate in selected_candidates
        if bool(_mapping_dict(candidate.get("policy")).get("export_policy_confirmed"))
    )
    preview_paths = _mapping_dict(learning_preview.get("paths"))
    return {
        "schema_name": HUMAN_SELECTED_CANDIDATE_LIST_SCHEMA_NAME,
        "schema_version": HUMAN_SELECTED_CANDIDATE_LIST_SCHEMA_VERSION,
        "workspace_id": resolved_workspace_id,
        "generated_at_utc": generated_at_utc,
        "export_mode": "preview_only",
        "training_export_ready": False,
        "human_gate_required": True,
        "source_learning_preview_path": source_learning_preview_path,
        "source_paths": {
            **copy.deepcopy(dict(source_paths)),
            "source_learning_preview_path": source_learning_preview_path,
            "learning_preview_latest_path": _clean_text(preview_paths.get("learning_preview_latest_path")),
            "learning_preview_run_path": _clean_text(preview_paths.get("learning_preview_run_path")),
        },
        "selection": {
            "origin": _clean_text(origin) or "manual",
            "rationale": _clean_text(rationale),
            "selected_event_ids": selected_ids,
            "selection_mode": "explicit_human_candidate_list",
        },
        "counts": {
            "requested_candidate_count": len(selected_ids),
            "selected_candidate_count": len(selected_candidates),
            "matched_candidate_count": len(matched_candidates),
            "missing_candidate_count": len(missing_candidates),
            "selected_supervised_candidate_count": len(supervised_selected),
            "selected_review_queue_count": len(matched_candidates),
            "selected_not_supervised_candidate_count": len(selected_candidates) - len(supervised_selected),
            "policy_confirmed_selected_count": policy_confirmed_selected_count,
        },
        "selected_candidates": selected_candidates,
        "missing_candidates": missing_candidates,
        "export_policy": {
            "mode": "preview_only",
            "human_gate_required": True,
            "training_export_ready": False,
            "training_job_allowed": False,
            "raw_log_export_allowed": False,
            "jsonl_training_export_allowed": False,
            "selection_does_not_promote_candidate": True,
            "downstream_export_requires_separate_approval": True,
        },
        "notes": [
            "Human selection is recorded as an inspection artifact only.",
            "This artifact does not create a trainable dataset or promote unready candidates.",
            "Raw logs and supervised example text are not copied into the selected-candidate list.",
        ],
    }


def record_human_selected_candidate_list(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    selected_event_ids: Iterable[str],
    learning_preview: Mapping[str, Any] | None = None,
    snapshot: Mapping[str, Any] | None = None,
    curation_preview: Mapping[str, Any] | None = None,
    curation_filters: Mapping[str, Any] | None = None,
    learning_limit: int | None = None,
    rationale: str | None = None,
    origin: str = "manual",
) -> tuple[dict[str, Any], Path, Path]:
    resolved_root = _resolve_root(root)
    source_learning_preview = learning_preview
    if source_learning_preview is None:
        source_learning_preview, _learning_latest_path, _learning_run_path = record_learning_dataset_preview(
            root=resolved_root,
            workspace_id=workspace_id,
            snapshot=snapshot,
            curation_preview=curation_preview,
            curation_filters=curation_filters,
            limit=learning_limit,
        )
    elif not _learning_preview_artifact_is_readable(source_learning_preview):
        source_learning_preview = _record_supplied_learning_preview(
            source_learning_preview,
            root=resolved_root,
            workspace_id=workspace_id,
        )
    payload = build_human_selected_candidate_list(
        source_learning_preview,
        workspace_id=workspace_id,
        selected_event_ids=selected_event_ids,
        rationale=rationale,
        origin=origin,
        events_by_id=_human_selected_events_by_id_from_learning_preview(source_learning_preview),
        explicit_signals=_human_selected_signals_from_learning_preview(source_learning_preview),
        comparisons=_human_selected_comparisons_from_learning_preview(source_learning_preview),
    )
    latest_path = human_selected_candidates_latest_path(workspace_id=workspace_id, root=resolved_root)
    run_path = human_selected_candidates_run_path(workspace_id=workspace_id, root=resolved_root)
    payload["paths"] = {
        "human_selected_latest_path": str(latest_path),
        "human_selected_run_path": str(run_path),
        "source_learning_preview_path": _clean_text(payload.get("source_learning_preview_path")),
    }
    write_json(run_path, payload)
    write_json(latest_path, payload)
    return payload, latest_path, run_path


def _human_selected_candidate_list_artifact_path(selection: Mapping[str, Any]) -> str | None:
    paths = _mapping_dict(selection.get("paths"))
    return (
        _clean_text(paths.get("human_selected_run_path"))
        or _clean_text(paths.get("human_selected_latest_path"))
    )


def _read_human_selected_candidate_list_artifact(selection: Mapping[str, Any]) -> dict[str, Any] | None:
    artifact_path = _path_from_text(_human_selected_candidate_list_artifact_path(selection))
    payload = _read_json_object(artifact_path)
    if (
        payload.get("schema_name") == HUMAN_SELECTED_CANDIDATE_LIST_SCHEMA_NAME
        and payload.get("schema_version") == HUMAN_SELECTED_CANDIDATE_LIST_SCHEMA_VERSION
    ):
        return payload
    return None


def _strip_supplied_training_text(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_supplied_training_text(item)
            for key, item in value.items()
            if str(key) not in HUMAN_SELECTED_SUPPLIED_TRAINING_TEXT_KEYS
        }
    if isinstance(value, list):
        return [_strip_supplied_training_text(item) for item in value]
    return copy.deepcopy(value)


def _record_supplied_human_selected_candidate_list(
    selection: Mapping[str, Any],
    *,
    root: Path,
    workspace_id: str,
) -> dict[str, Any]:
    payload = _strip_supplied_training_text(selection)
    latest_path = human_selected_candidates_latest_path(workspace_id=workspace_id, root=root)
    run_path = human_selected_candidates_run_path(workspace_id=workspace_id, root=root)
    payload["paths"] = {
        "human_selected_latest_path": str(latest_path),
        "human_selected_run_path": str(run_path),
        "source_learning_preview_path": _clean_text(payload.get("source_learning_preview_path")),
    }
    write_json(run_path, payload)
    write_json(latest_path, payload)
    return payload


def _jsonl_dry_run_policy(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    source_policy = _mapping_dict(policy)
    return {
        "export_mode": "preview_only",
        "training_export_ready": False,
        "human_gate_required": True,
        "training_job_allowed": False,
        "raw_log_export_allowed": False,
        "jsonl_training_export_allowed": False,
        "export_policy_confirmed": bool(source_policy.get("export_policy_confirmed")),
        "confirmation_signal_id": _clean_text(source_policy.get("confirmation_signal_id")),
        "downstream_export_requires_separate_approval": True,
    }


def _jsonl_dry_run_traceability(candidate: Mapping[str, Any]) -> dict[str, Any]:
    evidence_summary = _mapping_dict(candidate.get("evidence_summary"))
    if evidence_summary:
        traceability = _mapping_dict(evidence_summary.get("traceability"))
        curation_flags = _mapping_dict(evidence_summary.get("curation_reason_flags"))
        return {
            "curation_reasons": _string_list(evidence_summary.get("curation_reasons")),
            "signal_kinds": _string_list(evidence_summary.get("signal_kinds")),
            "signal_ids": _string_list(evidence_summary.get("signal_ids")),
            "comparison_roles": _string_list(evidence_summary.get("comparison_roles")),
            "comparison_ids": _string_list(evidence_summary.get("comparison_ids")),
            "curation_reason_flags": {
                "test_pass": bool(curation_flags.get("test_pass")),
                "accepted": bool(curation_flags.get("accepted")),
                "review_resolved": bool(curation_flags.get("review_resolved")),
                "comparison_winner": bool(curation_flags.get("comparison_winner")),
                "export_policy_confirmed": bool(curation_flags.get("export_policy_confirmed")),
            },
            "export_policy_confirmation_signal_id": _clean_text(
                evidence_summary.get("export_policy_confirmation_signal_id")
            ),
            "traceability": {
                "test_pass": bool(traceability.get("test_pass")),
                "accepted": bool(traceability.get("accepted")),
                "review_resolved": bool(traceability.get("review_resolved")),
                "comparison_winner": bool(traceability.get("comparison_winner")),
                "export_policy_confirmed": bool(traceability.get("export_policy_confirmed")),
            },
        }

    evidence = _mapping_dict(candidate.get("evidence"))
    signals = [
        signal
        for signal in evidence.get("signals") or []
        if isinstance(signal, Mapping)
    ]
    comparisons = [
        comparison
        for comparison in evidence.get("comparisons") or []
        if isinstance(comparison, Mapping)
    ]
    signal_kinds = [
        signal_kind
        for signal in signals
        if (signal_kind := _clean_text(signal.get("signal_kind"))) is not None
    ]
    signal_ids = [
        signal_id
        for signal in signals
        if (signal_id := _clean_text(signal.get("signal_id"))) is not None
    ]
    comparison_roles = [
        role
        for comparison in comparisons
        if (role := _clean_text(comparison.get("role"))) is not None
    ]
    comparison_ids = [
        comparison_id
        for comparison in comparisons
        if (comparison_id := _clean_text(comparison.get("comparison_id"))) is not None
    ]
    has_winner_trace = any(
        _clean_text(comparison.get("role")) == "winner"
        and _clean_text(comparison.get("outcome")) == "winner_selected"
        for comparison in comparisons
    )
    curation = _mapping_dict(candidate.get("curation"))
    curation_reasons = _string_list(curation.get("reasons"))
    policy = _jsonl_dry_run_policy(_mapping_dict(candidate.get("policy")))
    return {
        "curation_reasons": curation_reasons,
        "signal_kinds": signal_kinds,
        "signal_ids": signal_ids,
        "comparison_roles": comparison_roles,
        "comparison_ids": comparison_ids,
        "curation_reason_flags": {
            "test_pass": "test_pass" in curation_reasons,
            "accepted": "accepted" in curation_reasons,
            "review_resolved": "review_resolved" in curation_reasons,
            "comparison_winner": "comparison_winner" in curation_reasons,
            "export_policy_confirmed": EXPORT_POLICY_CONFIRMATION_SIGNAL_KIND in curation_reasons,
        },
        "export_policy_confirmation_signal_id": _clean_text(policy.get("confirmation_signal_id")),
        "traceability": {
            "test_pass": "test_pass" in signal_kinds,
            "accepted": "acceptance" in signal_kinds,
            "review_resolved": "review_resolved" in signal_kinds,
            "comparison_winner": has_winner_trace,
            "export_policy_confirmed": bool(policy.get("export_policy_confirmed")),
        },
    }


def _jsonl_dry_run_candidate_record(
    *,
    workspace_id: str,
    candidate: Mapping[str, Any],
    source_index: int,
    source_kind: str,
    source_has_durable_learning_preview: bool,
    source_has_durable_human_selection: bool,
) -> dict[str, Any]:
    queue_summary = _mapping_dict(candidate.get("review_queue"))
    if not queue_summary and "queue_state" in candidate:
        queue_summary = {
            "queue_state": _clean_text(candidate.get("queue_state")),
            "queue_priority": copy.deepcopy(_mapping_dict(candidate.get("queue_priority"))),
            "next_action": _clean_text(candidate.get("next_action")),
            "blocked_reason": _clean_text(candidate.get("blocked_reason")),
            "blocked_reasons": _string_list(candidate.get("blocked_reasons")),
            "excluded_by": _string_list(candidate.get("excluded_by")),
            "eligible_for_supervised_candidate": bool(candidate.get("eligible_for_supervised_candidate")),
            "lifecycle_summary": copy.deepcopy(_mapping_dict(candidate.get("lifecycle_summary"))),
            "export_policy_confirmation": copy.deepcopy(_mapping_dict(candidate.get("export_policy_confirmation"))),
        }
    policy = _jsonl_dry_run_policy(_mapping_dict(candidate.get("policy")))
    preview_membership = _mapping_dict(candidate.get("preview_membership"))
    if not preview_membership:
        preview_membership = {
            "in_review_queue": bool(queue_summary),
            "in_supervised_example_candidates": (
                source_kind == "learning_preview_supervised_candidates"
                and source_has_durable_learning_preview
            ),
            "in_excluded_candidates": False,
        }
    blocked_reasons = sorted(
        set(
            _string_list(candidate.get("blocked_reasons"))
            + _string_list(candidate.get("excluded_by"))
            + _string_list(queue_summary.get("blocked_reasons"))
            + _string_list(queue_summary.get("excluded_by"))
        )
    )
    candidate_claims_supervised = bool(
        candidate.get("eligible_for_supervised_candidate")
        or queue_summary.get("eligible_for_supervised_candidate")
        or preview_membership.get("in_supervised_example_candidates")
    )
    source_has_durable_artifacts = source_has_durable_learning_preview and (
        source_kind != "human_selected_candidate_list"
        or source_has_durable_human_selection
    )
    preview_membership_supports_supervised = bool(
        preview_membership.get("in_supervised_example_candidates")
        and source_has_durable_artifacts
    )
    eligible_for_supervised = candidate_claims_supervised and preview_membership_supports_supervised
    if not eligible_for_supervised and not blocked_reasons:
        blocked_reasons.append("not_eligible_for_supervised_candidate")
    if candidate_claims_supervised and not source_has_durable_artifacts:
        blocked_reasons.append("missing_durable_source_artifact")
    policy_confirmed = bool(policy.get("export_policy_confirmed"))
    if not policy_confirmed and "export_policy_not_confirmed" not in blocked_reasons:
        blocked_reasons.append("export_policy_not_confirmed")
    evidence_summary = _jsonl_dry_run_traceability(candidate)
    traceability = _mapping_dict(evidence_summary.get("traceability"))
    has_selection_trace = any(
        bool(traceability.get(key))
        for key in ("accepted", "review_resolved", "comparison_winner")
    )
    has_required_traceability = (
        bool(traceability.get("test_pass"))
        and has_selection_trace
        and bool(traceability.get("export_policy_confirmed"))
    )
    if eligible_for_supervised and policy_confirmed and not has_required_traceability:
        blocked_reasons.append("missing_required_traceability")
    blocked_reasons = sorted(set(blocked_reasons))
    future_candidate = (
        eligible_for_supervised
        and policy_confirmed
        and has_required_traceability
        and source_has_durable_artifacts
    )
    if future_candidate:
        dry_run_status = "future_jsonl_candidate_if_separately_approved"
    elif "missing_learning_preview_candidate" in blocked_reasons:
        dry_run_status = "missing_learning_preview_candidate"
    elif not eligible_for_supervised:
        dry_run_status = "not_supervised_candidate"
    elif not policy_confirmed:
        dry_run_status = "export_policy_confirmation_required"
    else:
        dry_run_status = "missing_required_traceability"
    event_id = _clean_text(candidate.get("event_id"))
    source_paths = _mapping_dict(candidate.get("source_paths"))
    return {
        "schema_name": JSONL_TRAINING_EXPORT_DRY_RUN_ITEM_SCHEMA_NAME,
        "schema_version": JSONL_TRAINING_EXPORT_DRY_RUN_ITEM_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "event_id": event_id,
        "source_index": source_index,
        "source_kind": source_kind,
        "selection_id": _clean_text(candidate.get("selection_id")),
        "supervised_example_candidate_id": (
            _clean_text(candidate.get("supervised_example_candidate_id"))
            or _clean_text(candidate.get("candidate_id"))
        ),
        "label": _clean_text(candidate.get("label")) or event_id,
        "preview_membership": copy.deepcopy(dict(preview_membership)),
        "eligible_for_supervised_candidate": eligible_for_supervised,
        "dry_run_status": dry_run_status,
        "dry_run_eligible_for_future_export_if_separately_approved": future_candidate,
        "would_write_jsonl_record": False,
        "training_export_ready": False,
        "human_gate_required": True,
        "not_trainable": True,
        "queue_state": _clean_text(queue_summary.get("queue_state")),
        "queue_priority": copy.deepcopy(_mapping_dict(queue_summary.get("queue_priority"))),
        "next_action": _clean_text(queue_summary.get("next_action")),
        "blocked_reason": _clean_text(queue_summary.get("blocked_reason")),
        "blocked_reasons": blocked_reasons,
        "required_before_training_export": [
            *([] if policy_confirmed else ["export_policy_confirmation"]),
            *([] if has_required_traceability else ["restore_required_traceability"]),
            "separate_downstream_export_approval",
            "m8_training_job_design",
        ],
        "export_policy_confirmation": copy.deepcopy(_mapping_dict(queue_summary.get("export_policy_confirmation"))),
        "evidence_summary": evidence_summary,
        "source_paths": {
            key: copy.deepcopy(value)
            for key, value in source_paths.items()
            if key not in {"prompt", "output_text", "supervised_example"}
        },
        "policy": policy,
        "jsonl_projection": {
            "record_format": "not_emitted_in_m7_dry_run",
            "jsonl_file_written": False,
            "supervised_example_text_copied": False,
            "raw_log_text_copied": False,
            "omitted_training_text_fields": ["instruction", "response", "messages"],
            "metadata_preview_fields": [
                "event_id",
                "supervised_example_candidate_id",
                "traceability",
                "policy",
                "source_paths",
            ],
        },
    }


def build_jsonl_training_export_dry_run(
    *,
    learning_preview: Mapping[str, Any] | None = None,
    human_selected_candidates: Mapping[str, Any] | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    if learning_preview is None and human_selected_candidates is None:
        raise ValueError("JSONL training export dry-run requires a learning preview or human-selected candidate list.")
    source_artifact = human_selected_candidates if human_selected_candidates is not None else learning_preview
    assert source_artifact is not None
    resolved_workspace_id = (
        _clean_text(workspace_id)
        or _clean_text(source_artifact.get("workspace_id"))
        or DEFAULT_WORKSPACE_ID
    )
    source_mode = (
        "human_selected_candidate_list"
        if human_selected_candidates is not None
        else "learning_preview_supervised_candidates"
    )
    if human_selected_candidates is not None:
        source_candidates = human_selected_candidates.get("selected_candidates") or []
    elif learning_preview is not None:
        source_candidates = learning_preview.get("supervised_example_candidates") or []
    else:
        source_candidates = []
    learning_preview_paths = _mapping_dict(learning_preview.get("paths")) if learning_preview is not None else {}
    human_selected_paths = (
        _mapping_dict(human_selected_candidates.get("paths"))
        if human_selected_candidates is not None
        else {}
    )
    source_paths = _mapping_dict(source_artifact.get("source_paths"))
    source_learning_preview_path = (
        _clean_text(source_artifact.get("source_learning_preview_path"))
        or _clean_text(source_paths.get("source_learning_preview_path"))
    )
    if source_learning_preview_path is None and learning_preview is not None:
        source_learning_preview_path = _learning_preview_artifact_path(learning_preview)
    source_human_selected_path = (
        _human_selected_candidate_list_artifact_path(human_selected_candidates)
        if human_selected_candidates is not None
        else None
    )
    source_has_durable_learning_preview = _path_is_file(source_learning_preview_path)
    source_has_durable_human_selection = (
        source_mode != "human_selected_candidate_list"
        or _path_is_file(source_human_selected_path)
    )
    candidates = [
        _jsonl_dry_run_candidate_record(
            workspace_id=resolved_workspace_id,
            candidate=candidate,
            source_index=index,
            source_kind=source_mode,
            source_has_durable_learning_preview=source_has_durable_learning_preview,
            source_has_durable_human_selection=source_has_durable_human_selection,
        )
        for index, candidate in enumerate(source_candidates)
        if isinstance(candidate, Mapping)
    ]
    future_candidate_count = sum(
        1
        for candidate in candidates
        if bool(candidate.get("dry_run_eligible_for_future_export_if_separately_approved"))
    )
    policy_confirmed_candidate_count = sum(
        1
        for candidate in candidates
        if bool(_mapping_dict(candidate.get("policy")).get("export_policy_confirmed"))
    )
    missing_candidate_count = sum(
        1
        for candidate in candidates
        if candidate.get("dry_run_status") == "missing_learning_preview_candidate"
    )
    blocked_candidate_count = len(candidates) - future_candidate_count
    return {
        "schema_name": JSONL_TRAINING_EXPORT_DRY_RUN_SCHEMA_NAME,
        "schema_version": JSONL_TRAINING_EXPORT_DRY_RUN_SCHEMA_VERSION,
        "workspace_id": resolved_workspace_id,
        "generated_at_utc": timestamp_utc(),
        "export_mode": "preview_only",
        "artifact_kind": "jsonl_training_export_dry_run_manifest",
        "training_export_ready": False,
        "human_gate_required": True,
        "not_trainable": True,
        "source_mode": source_mode,
        "source_learning_preview_path": source_learning_preview_path,
        "source_human_selected_candidates_path": source_human_selected_path,
        "source_paths": {
            **copy.deepcopy(dict(source_paths)),
            "source_learning_preview_path": source_learning_preview_path,
            "learning_preview_latest_path": _clean_text(learning_preview_paths.get("learning_preview_latest_path")),
            "learning_preview_run_path": _clean_text(learning_preview_paths.get("learning_preview_run_path")),
            "source_human_selected_candidates_path": source_human_selected_path,
            "human_selected_latest_path": _clean_text(human_selected_paths.get("human_selected_latest_path")),
            "human_selected_run_path": _clean_text(human_selected_paths.get("human_selected_run_path")),
        },
        "counts": {
            "source_candidate_count": len(source_candidates),
            "inspected_candidate_count": len(candidates),
            "future_jsonl_candidate_if_separately_approved_count": future_candidate_count,
            "blocked_candidate_count": blocked_candidate_count,
            "missing_candidate_count": missing_candidate_count,
            "policy_confirmed_candidate_count": policy_confirmed_candidate_count,
            "would_write_jsonl_record_count": 0,
            "supervised_example_text_copied_count": 0,
            "raw_log_text_copied_count": 0,
        },
        "candidates": candidates,
        "export_policy": {
            "mode": "preview_only",
            "artifact_kind": "validation_report_only",
            "training_export_ready": False,
            "human_gate_required": True,
            "not_trainable": True,
            "training_job_allowed": False,
            "raw_log_export_allowed": False,
            "jsonl_training_export_allowed": False,
            "jsonl_file_written": False,
            "selection_does_not_promote_candidate": True,
            "downstream_export_requires_separate_approval": True,
        },
        "dry_run_manifest": {
            "manifest_kind": "preview_only_validation_report",
            "file_extension": ".json",
            "jsonl_file_written": False,
            "trainable_artifact_written": False,
            "would_write_jsonl_record_count": 0,
        },
        "notes": [
            "Dry-run only; no JSONL file or trainable dataset artifact is produced.",
            "Human-selected candidates remain inspection inputs and do not promote unready candidates.",
            "Supervised example text and raw logs are not copied into this artifact.",
            "A separate downstream export approval and M8 training-job design are required before training export.",
        ],
    }


def record_jsonl_training_export_dry_run(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    learning_preview: Mapping[str, Any] | None = None,
    human_selected_candidates: Mapping[str, Any] | None = None,
    snapshot: Mapping[str, Any] | None = None,
    curation_preview: Mapping[str, Any] | None = None,
    curation_filters: Mapping[str, Any] | None = None,
    learning_limit: int | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    resolved_root = _resolve_root(root)
    source_learning_preview = learning_preview
    source_human_selected_candidates = human_selected_candidates
    if source_human_selected_candidates is not None:
        persisted_selection = _read_human_selected_candidate_list_artifact(source_human_selected_candidates)
        if persisted_selection is not None:
            source_human_selected_candidates = persisted_selection
        else:
            source_human_selected_candidates = _record_supplied_human_selected_candidate_list(
                source_human_selected_candidates,
                root=resolved_root,
                workspace_id=workspace_id,
            )
    if source_human_selected_candidates is None and source_learning_preview is None:
        source_learning_preview, _learning_latest_path, _learning_run_path = record_learning_dataset_preview(
            root=resolved_root,
            workspace_id=workspace_id,
            snapshot=snapshot,
            curation_preview=curation_preview,
            curation_filters=curation_filters,
            limit=learning_limit,
        )
    elif (
        source_human_selected_candidates is None
        and source_learning_preview is not None
        and not _learning_preview_artifact_is_readable(source_learning_preview)
    ):
        source_learning_preview = _record_supplied_learning_preview(
            source_learning_preview,
            root=resolved_root,
            workspace_id=workspace_id,
        )
    payload = build_jsonl_training_export_dry_run(
        learning_preview=source_learning_preview,
        human_selected_candidates=source_human_selected_candidates,
        workspace_id=workspace_id,
    )
    latest_path = jsonl_training_export_dry_run_latest_path(workspace_id=workspace_id, root=resolved_root)
    run_path = jsonl_training_export_dry_run_run_path(workspace_id=workspace_id, root=resolved_root)
    payload["paths"] = {
        "jsonl_export_dry_run_latest_path": str(latest_path),
        "jsonl_export_dry_run_run_path": str(run_path),
        "source_learning_preview_path": _clean_text(payload.get("source_learning_preview_path")),
        "source_human_selected_candidates_path": _clean_text(payload.get("source_human_selected_candidates_path")),
    }
    write_json(run_path, payload)
    write_json(latest_path, payload)
    return payload, latest_path, run_path


def _curation_preview_artifact_path(preview: Mapping[str, Any]) -> str | None:
    paths = _mapping_dict(preview.get("paths"))
    return (
        _clean_text(paths.get("curation_preview_run_path"))
        or _clean_text(paths.get("curation_preview_latest_path"))
    )


def _record_supplied_curation_preview(
    preview: Mapping[str, Any],
    *,
    root: Path,
    workspace_id: str,
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(preview))
    latest_path = curation_export_preview_latest_path(workspace_id=workspace_id, root=root)
    run_path = curation_export_preview_run_path(workspace_id=workspace_id, root=root)
    payload["paths"] = {
        "curation_preview_latest_path": str(latest_path),
        "curation_preview_run_path": str(run_path),
        "source_snapshot_path": _clean_text(payload.get("source_snapshot_path")),
    }
    write_json(run_path, payload)
    write_json(latest_path, payload)
    return payload


def format_human_selected_candidate_list_report(selection: Mapping[str, Any]) -> str:
    counts = _mapping_dict(selection.get("counts"))
    paths = _mapping_dict(selection.get("paths"))
    lines = [
        "Human-selected candidate list: preview_only",
        f"Requested: {int(counts.get('requested_candidate_count') or 0)}",
        f"Matched: {int(counts.get('matched_candidate_count') or 0)}",
        f"Selected supervised: {int(counts.get('selected_supervised_candidate_count') or 0)}",
        f"Missing: {int(counts.get('missing_candidate_count') or 0)}",
        f"Policy confirmed: {int(counts.get('policy_confirmed_selected_count') or 0)}",
        "Training export ready: no",
        "Human gate: required",
    ]
    if paths.get("human_selected_latest_path"):
        lines.append(f"Selection: {paths['human_selected_latest_path']}")
    candidates = [
        item
        for item in selection.get("selected_candidates") or []
        if isinstance(item, Mapping)
    ]
    if candidates:
        lines.extend(("", "Selected candidates:"))
        for item in candidates[:5]:
            label = _single_line_report_label(
                _clean_text(item.get("label"))
                or _clean_text(item.get("event_id"))
                or "candidate"
            )
            state = _clean_text(item.get("queue_state")) or "missing_from_learning_preview"
            next_action = _clean_text(item.get("next_action")) or "review_candidate"
            supervised = "yes" if _clean_text(item.get("supervised_example_candidate_id")) else "no"
            policy = _mapping_dict(item.get("policy"))
            policy_state = "confirmed" if policy.get("export_policy_confirmed") else "pending"
            lines.append(
                f"- {state}: {label} "
                f"(supervised={supervised}; policy={policy_state}; next={next_action})"
            )
    return "\n".join(lines)


def format_jsonl_training_export_dry_run_report(dry_run: Mapping[str, Any]) -> str:
    counts = _mapping_dict(dry_run.get("counts"))
    paths = _mapping_dict(dry_run.get("paths"))
    lines = [
        "JSONL training export dry-run: preview_only",
        f"Source mode: {_clean_text(dry_run.get('source_mode')) or 'n/a'}",
        f"Inspected: {int(counts.get('inspected_candidate_count') or 0)}",
        (
            "Future candidates if separately approved: "
            f"{int(counts.get('future_jsonl_candidate_if_separately_approved_count') or 0)}"
        ),
        f"Blocked: {int(counts.get('blocked_candidate_count') or 0)}",
        f"Policy confirmed: {int(counts.get('policy_confirmed_candidate_count') or 0)}",
        "JSONL file written: no",
        "Training export ready: no",
        "Trainable artifact: no",
        "Human gate: required",
    ]
    if paths.get("jsonl_export_dry_run_latest_path"):
        lines.append(f"Dry-run: {paths['jsonl_export_dry_run_latest_path']}")
    candidates = [
        item
        for item in dry_run.get("candidates") or []
        if isinstance(item, Mapping)
    ]
    if candidates:
        lines.extend(("", "Dry-run candidates:"))
        for item in candidates[:5]:
            label = _single_line_report_label(
                _clean_text(item.get("label"))
                or _clean_text(item.get("event_id"))
                or "candidate"
            )
            status = _clean_text(item.get("dry_run_status")) or "unknown"
            next_action = _clean_text(item.get("next_action")) or "review_candidate"
            policy = _mapping_dict(item.get("policy"))
            policy_state = "confirmed" if policy.get("export_policy_confirmed") else "pending"
            future = "yes" if item.get("dry_run_eligible_for_future_export_if_separately_approved") else "no"
            lines.append(
                f"- {status}: {label} "
                f"(future={future}; policy={policy_state}; next={next_action})"
            )
    return "\n".join(lines)


def format_learning_dataset_preview_report(preview: Mapping[str, Any]) -> str:
    counts = _mapping_dict(preview.get("counts"))
    paths = _mapping_dict(preview.get("paths"))
    lines = [
        "Learning dataset preview: preview_only",
        f"Source candidates: {int(counts.get('source_candidate_count') or 0)}",
        f"Eligible: {int(counts.get('eligible_candidate_count') or 0)}",
        f"Previewed: {int(counts.get('previewed_candidate_count') or 0)}",
        f"Excluded: {int(counts.get('excluded_candidate_count') or 0)}",
        f"Policy confirmed: {int(counts.get('policy_confirmed_candidate_count') or 0)}",
        "Training export ready: no",
        "Human gate: required",
    ]
    if paths.get("learning_preview_latest_path"):
        lines.append(f"Preview: {paths['learning_preview_latest_path']}")
    exclusion_reasons = _mapping_dict(counts.get("exclusion_reasons"))
    if exclusion_reasons:
        parts = [f"{key}={int(value)}" for key, value in exclusion_reasons.items()]
        lines.append(f"Exclusions: {'; '.join(parts)}")
    queue_states = _mapping_dict(counts.get("review_queue_states"))
    if queue_states:
        ordered_states = [
            state
            for state in LEARNING_REVIEW_QUEUE_STATES
            if state in queue_states
        ]
        ordered_states.extend(sorted(set(queue_states) - set(ordered_states)))
        parts = [f"{state}={int(queue_states.get(state) or 0)}" for state in ordered_states]
        lines.append(f"Review queue: {'; '.join(parts)}")
    candidates = [
        item
        for item in preview.get("supervised_example_candidates") or []
        if isinstance(item, Mapping)
    ]
    if candidates:
        lines.extend(("", "Supervised example candidates:"))
        for item in candidates[:5]:
            source = _mapping_dict(item.get("source_event"))
            backend = _mapping_dict(item.get("backend_metadata"))
            evidence = _mapping_dict(item.get("evidence"))
            signals = [
                _clean_text(signal.get("signal_kind"))
                for signal in evidence.get("signals") or []
                if isinstance(signal, Mapping)
            ]
            label = _single_line_report_label(
                _clean_text(source.get("prompt_excerpt"))
                or _clean_text(item.get("event_id"))
                or "candidate"
            )
            backend_label = _clean_text(backend.get("backend_id")) or _clean_text(backend.get("model_id")) or "n/a"
            signal_label = ",".join(item for item in signals if item) or "n/a"
            lines.append(f"- {label} (backend={backend_label}; signals={signal_label})")
    queue_items = [
        item
        for item in preview.get("review_queue") or []
        if isinstance(item, Mapping)
    ]
    if queue_items:
        lines.extend(("", "Learning review queue:"))
        for item in queue_items[:5]:
            priority = _mapping_dict(item.get("queue_priority"))
            rank = int(priority.get("rank") or 0)
            label = _single_line_report_label(
                _clean_text(item.get("label"))
                or _clean_text(item.get("event_id"))
                or "candidate"
            )
            state = _clean_text(item.get("queue_state")) or "needs_review"
            next_action = _clean_text(item.get("next_action")) or "review_candidate"
            blocked_reason = _clean_text(item.get("blocked_reason")) or "n/a"
            lines.append(
                f"- P{rank} {state} {label} "
                f"(next={next_action}; blocked={blocked_reason})"
            )
    return "\n".join(lines)


def _single_line_report_label(value: str, *, limit: int = 180) -> str:
    label = " ".join(value.split())
    if len(label) <= limit:
        return label
    return label[: max(limit - 20, 0)].rstrip() + " [truncated]"


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
        f"Export policy confirmed: {int(counts.get('export_policy_confirmed') or 0)}",
        f"Test pass: {int(counts.get('test_pass') or 0)}",
        f"Test fail: {int(counts.get('test_fail') or 0)}",
        f"Repair links: {int(counts.get('repair_links') or 0)}",
        f"Follow-up links: {int(counts.get('follow_up_links') or 0)}",
        f"Comparisons: {int(counts.get('comparisons') or 0)}",
        f"Curation ready: {int(counts.get('curation_ready') or 0)}",
        f"Consistency stale positives: {int(counts.get('stale_positive_signal') or 0)}",
        f"Consistency negative wins: {int(counts.get('negative_signal_wins') or 0)}",
        f"Consistency missing trace: {int(counts.get('missing_trace') or 0)}",
        f"Event contract failed: {int(counts.get('event_contract_failed') or 0)}",
        f"Event contract missing source: {int(counts.get('event_contract_missing_source') or 0)}",
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
    consistency = _mapping_dict(snapshot.get("evaluation_consistency"))
    issue_events = [
        item
        for item in (consistency.get("issue_events") or [])
        if isinstance(item, Mapping)
    ]
    if issue_events:
        lines.extend(("", "Evaluation consistency:"))
        for item in issue_events[:5]:
            event_id = _clean_text(item.get("event_id")) or "n/a"
            stale = ",".join(_string_list(item.get("stale_positive_signals"))) or "none"
            missing = ",".join(_string_list(item.get("missing_traces"))) or "none"
            lines.append(f"- {event_id}: stale={stale}; missing_trace={missing}")
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
