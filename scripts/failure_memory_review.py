#!/usr/bin/env python3
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from artifact_schema import build_artifact_payload, build_prompt_record, build_runtime_record, read_artifact, write_artifact
from evaluation_loop import (
    append_evaluation_comparison,
    append_evaluation_signal,
    build_evaluation_comparison,
    build_evaluation_signal,
    evaluation_comparison_log_path,
    evaluation_signal_log_path,
    record_curation_export_preview,
    record_evaluation_snapshot,
    record_learning_dataset_preview,
    software_work_events_by_id,
)
from gemma_runtime import repo_root, timestamp_slug, timestamp_utc, write_json
from memory_index import MemoryIndex, rebuild_memory_index
from recall_context import DEFAULT_CONTEXT_BUDGET_CHARS, build_context_bundle
from run_recall_demo import build_bundle_report
from satellite_pack import PackManifestError, audit_pack_path, load_pack_manifest, resolve_pack_manifest_path
from software_work_events import build_event_contract_report, read_event_log
from workspace_state import DEFAULT_WORKSPACE_ID, WorkspaceSessionStore


FAILURE_MEMORY_INPUT_SCHEMA_NAME = "software-satellite-failure-memory-input"
FAILURE_MEMORY_INPUT_SCHEMA_VERSION = 1
FAILURE_MEMORY_RECALL_SCHEMA_NAME = "software-satellite-failure-memory-recall"
FAILURE_MEMORY_RECALL_SCHEMA_VERSION = 1
FAILURE_MEMORY_VERDICT_SCHEMA_NAME = "software-satellite-human-verdict-record"
FAILURE_MEMORY_VERDICT_SCHEMA_VERSION = 1
REVIEW_RISK_REPORT_SCHEMA_NAME = "software-satellite-review-risk-report"
REVIEW_RISK_REPORT_SCHEMA_VERSION = 1
REVIEW_RISK_PACK_NAME = "review-risk-pack"

INPUT_KIND_DEFAULT_STATUS = {
    "patch": "needs_review",
    "failure": "failed",
    "proposal": "needs_review",
    "repair": "ok",
    "review_note": "needs_review",
}
INPUT_KIND_ROLE = {
    "patch": "patch_input",
    "failure": "failure_input",
    "proposal": "proposal_input",
    "repair": "repair_input",
    "review_note": "review_note_input",
}
VERDICT_SIGNAL_KIND = {
    "accept": "acceptance",
    "reject": "rejection",
    "resolve": "review_resolved",
    "unresolve": "review_unresolved",
    "needs-review": "review_unresolved",
}


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


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


def _read_text(path: Path, *, limit: int | None = 20000) -> str:
    try:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"Input file is not readable: `{path}`.") from exc
    if limit is None:
        return text
    return text[:limit]


def _truncate(text: str | None, *, limit: int = 180) -> str:
    cleaned = _clean_text(text) or ""
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 15)].rstrip() + " [truncated]"


def _coerce_utc_datetime(value: Any) -> datetime | None:
    text = _clean_text(value)
    if text is None:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _file_sha256(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        raise ValueError(f"Input file is not readable: `{path}`.") from exc


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


def failure_memory_root(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return _resolve_root(root) / "artifacts" / "failure_memory" / workspace_id


def latest_input_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return failure_memory_root(workspace_id=workspace_id, root=root) / "inputs" / "latest.json"


def input_run_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return (
        failure_memory_root(workspace_id=workspace_id, root=root)
        / "inputs"
        / "runs"
        / f"{timestamp_slug()}-input.json"
    )


def latest_recall_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return failure_memory_root(workspace_id=workspace_id, root=root) / "recall" / "latest.json"


def recall_run_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return (
        failure_memory_root(workspace_id=workspace_id, root=root)
        / "recall"
        / "runs"
        / f"{timestamp_slug()}-recall.json"
    )


def latest_verdict_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return failure_memory_root(workspace_id=workspace_id, root=root) / "verdicts" / "latest.json"


def verdict_run_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return (
        failure_memory_root(workspace_id=workspace_id, root=root)
        / "verdicts"
        / "runs"
        / f"{timestamp_slug()}-verdict.json"
    )


def latest_report_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return failure_memory_root(workspace_id=workspace_id, root=root) / "reports" / "latest.md"


def report_run_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return (
        failure_memory_root(workspace_id=workspace_id, root=root)
        / "reports"
        / "runs"
        / f"{timestamp_slug()}-review-risk-report.md"
    )


def report_metadata_latest_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return failure_memory_root(workspace_id=workspace_id, root=root) / "reports" / "latest.json"


def report_metadata_run_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return (
        failure_memory_root(workspace_id=workspace_id, root=root)
        / "reports"
        / "runs"
        / f"{timestamp_slug()}-review-risk-report.json"
    )


def _source_input_record(path: Path, *, role: str, root: Path) -> dict[str, Any]:
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"Input file is not readable: `{path}`.") from exc
    return {
        "role": role,
        "kind": "file",
        "path": str(path),
        "workspace_relative_path": _workspace_relative(path, root=root),
        "sha256": _file_sha256(path),
        "size_bytes": size_bytes,
    }


def _patch_file_hints(patch_text: str) -> list[str]:
    hints: list[str] = []
    seen: set[str] = set()
    patterns = (
        r"^diff --git a/(.*?) b/(.*?)$",
        r"^\+\+\+ b/(.*?)$",
        r"^--- a/(.*?)$",
    )
    for line in patch_text.splitlines():
        for pattern in patterns:
            match = re.match(pattern, line)
            if match is None:
                continue
            values = match.groups()
            candidate = values[-1].strip()
            if candidate == "/dev/null" or not candidate or candidate in seen:
                continue
            seen.add(candidate)
            hints.append(candidate)
    return hints


def summarize_patch(path: Path) -> dict[str, Any]:
    patch_text = _read_text(path, limit=None)
    added = sum(1 for line in patch_text.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in patch_text.splitlines() if line.startswith("-") and not line.startswith("---"))
    file_hints = _patch_file_hints(patch_text)
    return {
        "source_path": str(path),
        "sha256": _file_sha256(path),
        "changed_file_count": len(file_hints),
        "changed_files": file_hints,
        "added_lines": added,
        "removed_lines": removed,
        "excerpt": _truncate(patch_text, limit=1200),
    }


def summarize_file_input(path: Path, *, input_kind: str) -> dict[str, Any]:
    if input_kind == "patch":
        return summarize_patch(path)
    text = _read_text(path)
    return {
        "source_path": str(path),
        "sha256": _file_sha256(path),
        "line_count": len(text.splitlines()),
        "excerpt": _truncate(text, limit=1200),
    }


def _event_id_from_recorded_session(session: Mapping[str, Any], *, workspace_id: str) -> str:
    entries = [entry for entry in session.get("entries") or [] if isinstance(entry, Mapping)]
    if not entries:
        raise ValueError("Recorded session did not contain an event entry.")
    entry_id = _clean_text(entries[-1].get("entry_id"))
    session_id = _clean_text(session.get("session_id"))
    if entry_id is None or session_id is None:
        raise ValueError("Recorded session entry was missing an id.")
    return f"{workspace_id}:{session_id}:{entry_id}"


def _quality_status_for_input(input_kind: str, status: str) -> str | None:
    normalized_status = status.lower()
    if input_kind == "failure" or normalized_status in {"failed", "quality_fail", "blocked", "error"}:
        return "fail"
    if input_kind == "repair" or normalized_status in {"ok", "accepted"}:
        return "pass"
    return None


def record_file_input(
    *,
    input_kind: str,
    source_path: Path,
    note: str | None = None,
    status: str | None = None,
    backend_id: str | None = None,
    backend_display_name: str | None = None,
    backend_adapter_kind: str | None = None,
    model_id: str | None = None,
    backend_metadata: Mapping[str, Any] | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    refresh_index: bool = True,
    write_latest: bool = True,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    normalized_kind = input_kind.strip().lower().replace("-", "_")
    if normalized_kind not in INPUT_KIND_DEFAULT_STATUS:
        raise ValueError(f"Unsupported file input kind `{input_kind}`.")
    resolved_source = _path_from_text(source_path, root=resolved_root)
    try:
        source_is_file = resolved_source.is_file()
    except OSError as exc:
        raise ValueError(f"{normalized_kind} input must be a readable file: `{resolved_source}`.") from exc
    if not source_is_file:
        raise ValueError(f"{normalized_kind} input must be a readable file: `{resolved_source}`.")

    normalized_status = _clean_text(status) or INPUT_KIND_DEFAULT_STATUS[normalized_kind]
    role = INPUT_KIND_ROLE[normalized_kind]
    source_record = _source_input_record(resolved_source, role=role, root=resolved_root)
    summary = summarize_file_input(resolved_source, input_kind=normalized_kind)
    artifact_path = input_run_path(workspace_id=workspace_id, root=resolved_root)
    latest_path = latest_input_path(workspace_id=workspace_id, root=resolved_root)
    quality_status = _quality_status_for_input(normalized_kind, normalized_status)
    prompt = " | ".join(
        part
        for part in (
            f"{normalized_kind.replace('_', ' ')} input",
            _clean_text(note),
            " ".join(summary.get("changed_files") or []) if normalized_kind == "patch" else None,
        )
        if part
    )
    output_text = (
        f"Recorded {normalized_kind.replace('_', ' ')} input from {source_record['workspace_relative_path'] or source_record['path']}."
    )
    notes = [f"failure_memory_input: {normalized_kind}"]
    if note:
        notes.append(note)
    if quality_status == "fail":
        notes.append("negative_signal: failure-like input")
    backend_options = {
        key: value
        for key, value in {
            "backend_id": _clean_text(backend_id),
            "backend_display_name": _clean_text(backend_display_name),
            "backend_adapter_kind": _clean_text(backend_adapter_kind),
            "model_id": _clean_text(model_id),
            "backend_metadata": _mapping_dict(backend_metadata),
        }.items()
        if value not in (None, {})
    }
    validation = {
        "validation_mode": "file_first_ingest",
        "claim_scope": f"{normalized_kind.replace('_', ' ')} source path is durable and inspectable",
        "pass_definition": "The source input path and the recorded evidence artifact are readable local files.",
        "quality_status": quality_status,
        "execution_status": normalized_status,
        "quality_checks": [
            {
                "name": "source_input_readable",
                "pass": True,
                "detail": source_record["workspace_relative_path"] or source_record["path"],
            }
        ],
    }
    artifact = build_artifact_payload(
        artifact_kind="software_work_input",
        status=normalized_status,
        runtime=build_runtime_record(
            backend=_clean_text(backend_id) or "satlab",
            model_id=_clean_text(model_id),
            extra={"workflow": "failure_memory_review", "input_kind": normalized_kind},
        ),
        prompts=build_prompt_record(prompt=prompt, resolved_user_prompt=prompt),
        extra={
            "input_schema_name": FAILURE_MEMORY_INPUT_SCHEMA_NAME,
            "input_schema_version": FAILURE_MEMORY_INPUT_SCHEMA_VERSION,
            "workspace_id": workspace_id,
            "input_kind": normalized_kind,
            "note": _clean_text(note),
            "source_inputs": [source_record],
            "input_summary": summary,
            "validation": validation,
        },
    )
    write_artifact(artifact_path, artifact)

    store = WorkspaceSessionStore(root=resolved_root, workspace_id=workspace_id)
    session = store.record_session_run(
        surface="chat",
        model_id=None,
        mode=f"failure_memory_{normalized_kind}",
        artifact_kind="software_work_input",
        artifact_path=artifact_path,
        status=normalized_status,
        prompt=prompt,
        system_prompt=None,
        resolved_user_prompt=prompt,
        output_text=output_text,
        attachments=[{"role": role, "kind": "file", "path": resolved_source}],
        notes=notes,
        options={
            "workflow": "failure_memory_review",
            "input_kind": normalized_kind,
            "source_input_path": str(resolved_source),
            "source_input_sha256": source_record["sha256"],
            "file_hints": summary.get("changed_files") if normalized_kind == "patch" else [],
            "record_source_checksums": True,
            **backend_options,
            **validation,
        },
    )
    event_id = _event_id_from_recorded_session(session, workspace_id=workspace_id)
    artifact["event_id"] = event_id
    artifact["event_artifact_path"] = str(artifact_path)
    if write_latest:
        write_artifact(latest_path, artifact)
    index_summary = rebuild_memory_index(root=resolved_root, workspace_id=workspace_id) if refresh_index else None
    result = {
        "schema_name": FAILURE_MEMORY_INPUT_SCHEMA_NAME,
        "schema_version": FAILURE_MEMORY_INPUT_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "recorded_at_utc": timestamp_utc(),
        "event_id": event_id,
        "input_kind": normalized_kind,
        "status": normalized_status,
        "source_inputs": [source_record],
        "input_summary": summary,
        "artifact_path": str(artifact_path),
        "latest_artifact_path": str(latest_path) if write_latest else None,
        "index_summary": index_summary,
    }
    return result


def _normalize_repeated_metadata(
    values: Iterable[str] | None,
    *,
    candidate_count: int,
    field_name: str,
) -> list[str | None]:
    cleaned = [_clean_text(value) for value in (values or [])]
    cleaned = [value for value in cleaned if value is not None]
    if not cleaned:
        return [None] * candidate_count
    if len(cleaned) != candidate_count:
        raise ValueError(f"{field_name} must be repeated once per candidate when provided.")
    return cleaned


def _comparison_verdict_to_outcome(verdict: str, *, has_winner: bool) -> str:
    normalized = verdict.strip().lower().replace("-", "_")
    if normalized == "winner":
        if not has_winner:
            raise ValueError("Proposal comparison verdict `winner` requires --winner-candidate.")
        return "winner_selected"
    if normalized == "none":
        return "needs_follow_up"
    if normalized in {"tie", "needs_follow_up"}:
        return normalized
    raise ValueError(f"Unsupported proposal comparison verdict `{verdict}`.")


def _normalize_comparison_verdict(verdict: str) -> str:
    normalized = verdict.strip().lower().replace("-", "_")
    if normalized == "needs_follow_up":
        return "none"
    if normalized in {"winner", "none", "tie"}:
        return normalized
    raise ValueError(f"Unsupported proposal comparison verdict `{verdict}`.")


def _resolve_winner_candidate_index(
    candidate_paths: list[Path],
    *,
    winner_candidate: str,
    root: Path,
) -> int:
    cleaned = _clean_text(winner_candidate)
    if cleaned is None:
        raise ValueError("Proposal comparison winner requires a candidate source path or 1-based index.")
    if cleaned.isdigit():
        index = int(cleaned) - 1
        if 0 <= index < len(candidate_paths):
            return index
        raise ValueError("Proposal comparison winner index is outside the candidate range.")

    path_candidate: Path | None = None
    try:
        path_candidate = _path_from_text(cleaned, root=root)
    except (OSError, ValueError):
        path_candidate = None
    for index, candidate_path in enumerate(candidate_paths):
        if cleaned == str(candidate_path):
            return index
        try:
            relative_path = str(candidate_path.relative_to(root))
        except ValueError:
            relative_path = None
        if cleaned == relative_path:
            return index
        if path_candidate is not None and path_candidate == candidate_path:
            return index
    raise ValueError("Proposal comparison winner did not match any candidate source path.")


def record_proposal_comparison(
    *,
    candidate_paths: Iterable[Path],
    verdict: str,
    rationale: str,
    winner_candidate: str | None = None,
    task_label: str | None = None,
    criteria: Iterable[str] | None = None,
    candidate_backend_ids: Iterable[str] | None = None,
    candidate_model_ids: Iterable[str] | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    resolved_root = _resolve_root(root)
    resolved_candidates = [_path_from_text(path, root=resolved_root) for path in candidate_paths]
    if len(resolved_candidates) < 2:
        raise ValueError("Proposal comparison requires at least two candidate files.")
    seen_paths: set[str] = set()
    for path in resolved_candidates:
        path_key = str(path)
        if path_key in seen_paths:
            raise ValueError("Proposal comparison candidate files must be unique.")
        seen_paths.add(path_key)

    cleaned_rationale = _clean_text(rationale)
    if cleaned_rationale is None:
        raise ValueError("Proposal comparison requires a human rationale.")
    normalized_verdict = _normalize_comparison_verdict(verdict)
    if normalized_verdict == "winner" and _clean_text(winner_candidate) is None:
        raise ValueError("Proposal comparison verdict `winner` requires --winner-candidate.")
    if normalized_verdict != "winner" and _clean_text(winner_candidate) is not None:
        raise ValueError("Proposal comparison winner can only be supplied with verdict `winner`.")
    winner_candidate_index = (
        _resolve_winner_candidate_index(
            resolved_candidates,
            winner_candidate=winner_candidate or "",
            root=resolved_root,
        )
        if normalized_verdict == "winner"
        else None
    )

    backend_ids = _normalize_repeated_metadata(
        candidate_backend_ids,
        candidate_count=len(resolved_candidates),
        field_name="candidate backend id",
    )
    model_ids = _normalize_repeated_metadata(
        candidate_model_ids,
        candidate_count=len(resolved_candidates),
        field_name="candidate model id",
    )

    candidate_results: list[dict[str, Any]] = []
    label = _clean_text(task_label) or "file-based proposal comparison"
    for index, path in enumerate(resolved_candidates, start=1):
        candidate_results.append(
            record_file_input(
                input_kind="proposal",
                source_path=path,
                note=f"{label} candidate {index}",
                status="needs_review",
                backend_id=backend_ids[index - 1],
                model_id=model_ids[index - 1],
                backend_display_name=backend_ids[index - 1],
                backend_adapter_kind="local" if backend_ids[index - 1] or model_ids[index - 1] else None,
                backend_metadata={
                    "comparison_source": "file_based_proposal",
                    "candidate_index": index,
                    "source_path": str(path),
                },
                workspace_id=workspace_id,
                root=resolved_root,
                refresh_index=False,
                write_latest=False,
            )
        )

    events_by_id = software_work_events_by_id(root=resolved_root, workspace_id=workspace_id)
    winner_event_id = (
        str(candidate_results[winner_candidate_index]["event_id"])
        if winner_candidate_index is not None
        else None
    )
    outcome = _comparison_verdict_to_outcome(verdict, has_winner=winner_event_id is not None)

    candidate_event_ids = [str(result["event_id"]) for result in candidate_results]
    comparison = build_evaluation_comparison(
        workspace_id=workspace_id,
        candidate_event_ids=candidate_event_ids,
        winner_event_id=winner_event_id,
        outcome=outcome,
        task_label=task_label,
        criteria=criteria,
        rationale=cleaned_rationale,
        origin="satlab_file_proposal_compare",
        events_by_id=events_by_id,
        tags=["proposal_compare", "file_first", "human_rationale"],
    )
    comparison_log = evaluation_comparison_log_path(workspace_id=workspace_id, root=resolved_root)
    recorded_comparison = append_evaluation_comparison(
        comparison_log,
        comparison,
        workspace_id=workspace_id,
    )
    snapshot, snapshot_latest, snapshot_run = record_evaluation_snapshot(
        root=resolved_root,
        workspace_id=workspace_id,
    )
    result = {
        "schema_name": "software-satellite-proposal-comparison-record",
        "schema_version": 1,
        "workspace_id": workspace_id,
        "recorded_at_utc": timestamp_utc(),
        "comparison": recorded_comparison,
        "human_verdict": {
            "verdict": normalized_verdict,
            "outcome": outcome,
            "winner_event_id": winner_event_id,
            "loser_event_ids": [
                event_id
                for event_id in candidate_event_ids
                if outcome == "winner_selected" and event_id != winner_event_id
            ],
            "rationale": cleaned_rationale,
            "human_gate_required": True,
        },
        "candidates": candidate_results,
        "paths": {
            "comparison_log_path": str(comparison_log),
            "snapshot_latest_path": str(snapshot_latest),
            "snapshot_run_path": str(snapshot_run),
        },
        "evaluation_counts": _mapping_dict(snapshot.get("counts")),
        "notes": [
            "Comparison is file-first and source-path preserving.",
            "A winner is still preview-only learning evidence until source, rationale, test signal, and policy gates pass.",
        ],
    }
    return result, comparison_log


def _read_json_mapping(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _latest_input_artifact(
    *,
    root: Path,
    workspace_id: str,
) -> dict[str, Any] | None:
    path = latest_input_path(workspace_id=workspace_id, root=root)
    try:
        return read_artifact(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return _read_json_mapping(path)


def _file_hints_from_latest_input(
    *,
    root: Path,
    workspace_id: str,
) -> list[str]:
    latest = _latest_input_artifact(root=root, workspace_id=workspace_id)
    if not isinstance(latest, Mapping):
        return []
    summary = _mapping_dict(latest.get("input_summary"))
    return _string_list(summary.get("changed_files"))


def _latest_input_event_id_for_recall(
    *,
    root: Path,
    workspace_id: str,
    patch_path: Path | None,
) -> str | None:
    latest = _latest_input_artifact(root=root, workspace_id=workspace_id)
    event_id = _event_id_from_latest_input_artifact(latest)
    if event_id is None:
        return None
    if patch_path is None:
        return event_id
    if not patch_path.is_file():
        return None
    patch_sha256 = _file_sha256(patch_path)
    patch_path_text = str(patch_path.resolve())
    for source in _source_inputs_from_latest(latest):
        if (
            _clean_text(source.get("path")) == patch_path_text
            and _clean_text(source.get("sha256")) == patch_sha256
        ):
            return event_id
    return None


def build_failure_recall(
    *,
    query: str,
    patch_path: Path | None = None,
    file_hints: Iterable[str] | None = None,
    limit: int = 5,
    context_budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    source_event_id: str | None = None,
    status_filters: Iterable[str] | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    resolved_root = _resolve_root(root)
    summary = rebuild_memory_index(root=resolved_root, workspace_id=workspace_id)
    index = MemoryIndex(Path(summary["index_path"]))
    hints = list(file_hints or [])
    resolved_patch: Path | None = None
    if patch_path is not None:
        resolved_patch = _path_from_text(patch_path, root=resolved_root)
        if resolved_patch.is_file():
            hints.extend(summarize_patch(resolved_patch).get("changed_files") or [])
    if not hints:
        hints.extend(_file_hints_from_latest_input(root=resolved_root, workspace_id=workspace_id))
    deduped_hints = []
    seen_hints: set[str] = set()
    for hint in hints:
        cleaned = _clean_text(hint)
        if cleaned is None or cleaned in seen_hints:
            continue
        seen_hints.add(cleaned)
        deduped_hints.append(cleaned)
    request = {
        "task_kind": "failure_analysis",
        "query_text": query,
        "request_basis": "failure_memory_review",
        "file_hints": deduped_hints,
        "status_filters": list(status_filters or []),
        "limit": limit,
        "context_budget_chars": context_budget_chars,
    }
    resolved_source_event_id = source_event_id or _latest_input_event_id_for_recall(
        root=resolved_root,
        workspace_id=workspace_id,
        patch_path=resolved_patch,
    )
    if resolved_source_event_id:
        request["source_event_id"] = resolved_source_event_id
    bundle = build_context_bundle(
        request,
        root=resolved_root,
        workspace_id=workspace_id,
        index=index,
    )
    run_path = recall_run_path(workspace_id=workspace_id, root=resolved_root)
    latest_path = latest_recall_path(workspace_id=workspace_id, root=resolved_root)
    recall = {
        "schema_name": FAILURE_MEMORY_RECALL_SCHEMA_NAME,
        "schema_version": FAILURE_MEMORY_RECALL_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "generated_at_utc": timestamp_utc(),
        "request": request,
        "bundle": bundle,
        "risk_note": build_risk_note(bundle),
        "validation_metrics": demand_validation_metrics(summary.get("event_contract")),
        "paths": {
            "recall_latest_path": str(latest_path),
            "recall_run_path": str(run_path),
            "index_path": str(summary["index_path"]),
            "event_log_path": str(summary["event_log_path"]),
        },
    }
    write_json(run_path, recall)
    write_json(latest_path, recall)
    return recall, latest_path, run_path


def demand_validation_metrics(event_contract: Any) -> dict[str, Any]:
    contract = _mapping_dict(event_contract)
    checked_count = int(contract.get("checked_event_count") or 0)
    source_counts = _mapping_dict(contract.get("source_status_counts"))
    missing_source = int(source_counts.get("missing_source") or 0)
    complete_count = max(0, checked_count - missing_source)
    completeness = round(complete_count / checked_count, 4) if checked_count else None
    return {
        "useful_recalled_evidence_rate": {
            "target": ">= 0.30",
            "observed": None,
            "status": "needs_dogfood_verdicts",
        },
        "critical_false_evidence": {
            "target": 0,
            "observed": 0,
            "status": "pass_if_source_contract_holds",
        },
        "source_path_completeness": {
            "target": ">= 0.90",
            "observed": completeness,
            "checked_event_count": checked_count,
            "missing_source_event_count": missing_source,
        },
        "human_verdict_capture_friction": {
            "target": "<= 30 seconds",
            "observed": "single command",
        },
        "clone_to_demo_time": {
            "target": "<= 15 minutes",
            "observed": None,
        },
    }


def build_risk_note(bundle: Mapping[str, Any]) -> dict[str, Any]:
    selected = [item for item in bundle.get("selected_candidates") or [] if isinstance(item, Mapping)]
    source_evaluation = _mapping_dict(bundle.get("source_evaluation"))
    source_contract_status = _clean_text(source_evaluation.get("source_event_contract_status"))
    failure_like = [
        item
        for item in selected
        if _clean_text(item.get("status")) in {"failed", "quality_fail", "blocked", "error", "rejected"}
        or "failure-signal" in _string_list(item.get("reasons"))
        or "rejected-signal" in _string_list(item.get("reasons"))
    ]
    missing_source = [
        item
        for item in selected
        if _clean_text(item.get("event_contract_status")) in {"missing_source", "invalid_event_contract"}
    ]
    if source_contract_status in {"missing_source", "invalid_event_contract"}:
        level = "block"
        recommendation = "The source event is not recallable as positive evidence; fix the source artifact path before reuse."
    elif missing_source:
        level = "block"
        recommendation = "Do not treat missing-source evidence as a positive learning candidate."
    elif failure_like:
        level = "needs_review"
        recommendation = "Inspect the recalled source paths before accepting this patch."
    elif selected:
        level = "watch"
        recommendation = "Use the recalled evidence as review context, then record a human verdict."
    else:
        level = "unknown"
        recommendation = "No useful failure memory was recalled; review the patch without promoting it."
    return {
        "level": level,
        "selected_failure_like_count": len(failure_like),
        "selected_missing_source_count": len(missing_source),
        "recommendation": recommendation,
    }


def format_failure_recall_report(recall: Mapping[str, Any]) -> str:
    bundle = _mapping_dict(recall.get("bundle"))
    risk_note = _mapping_dict(recall.get("risk_note"))
    lines = [
        "Failure memory recall",
        f"Workspace: {_clean_text(recall.get('workspace_id')) or DEFAULT_WORKSPACE_ID}",
        f"Query: {_clean_text(_mapping_dict(recall.get('request')).get('query_text')) or ''}",
        f"Risk: {_clean_text(risk_note.get('level')) or 'unknown'}",
        f"Recommended next action: {_clean_text(risk_note.get('recommendation')) or 'Record a human verdict.'}",
        "",
        build_bundle_report(bundle),
    ]
    paths = _mapping_dict(recall.get("paths"))
    if paths.get("recall_run_path"):
        lines.extend(("", f"Recall artifact: {paths['recall_run_path']}"))
    return "\n".join(line for line in lines if line is not None)


def build_verdict_template(*, event_id: str | None = None, verdict: str = "reject", reason: str | None = None) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "event_id": event_id or "<event-id>",
        "reason": reason or "<short human reason>",
        "human_gate_required": True,
        "records_evaluation_signal": verdict in VERDICT_SIGNAL_KIND,
        "preview_only_learning": True,
    }


def format_verdict_template(template: Mapping[str, Any]) -> str:
    gate = "required" if template.get("human_gate_required") else "not required"
    signal = "yes" if template.get("records_evaluation_signal") else "no"
    learning = "preview-only" if template.get("preview_only_learning") else "not marked"
    return "\n".join(
        [
            f"Verdict template: {_clean_text(template.get('verdict')) or 'reject'}",
            f"Event: {_clean_text(template.get('event_id')) or '<event-id>'}",
            f"Reason: {_clean_text(template.get('reason')) or '<short human reason>'}",
            f"Human gate: {gate}",
            f"Records evaluation signal: {signal}",
            f"Learning mode: {learning}",
        ]
    )


def record_human_verdict(
    *,
    verdict: str,
    event_id: str,
    reason: str,
    target_event_id: str | None = None,
    relation_kind: str | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    resolved_root = _resolve_root(root)
    normalized_verdict = verdict.strip().lower()
    if normalized_verdict not in VERDICT_SIGNAL_KIND:
        raise ValueError(f"Unsupported verdict `{verdict}`.")
    index_summary = rebuild_memory_index(root=resolved_root, workspace_id=workspace_id)
    event_log = read_event_log(Path(index_summary["event_log_path"]))
    events_by_id = {
        str(event.get("event_id")): dict(event)
        for event in event_log.get("events") or []
        if isinstance(event, Mapping) and event.get("event_id")
    }
    source_event = events_by_id.get(event_id)
    if source_event is None:
        raise ValueError(f"Unknown event id `{event_id}`.")

    signal_kind = VERDICT_SIGNAL_KIND[normalized_verdict]
    signal = build_evaluation_signal(
        workspace_id=workspace_id,
        signal_kind=signal_kind,
        source_event_id=event_id,
        source_event=source_event,
        target_event_id=target_event_id,
        relation_kind=relation_kind,
        rationale=reason,
        evidence={
            "human_verdict": normalized_verdict,
            "decision_summary": reason,
        },
        origin="satlab_cli",
        tags=["human-verdict", normalized_verdict],
    )
    append_evaluation_signal(
        evaluation_signal_log_path(workspace_id=workspace_id, root=resolved_root),
        signal,
        workspace_id=workspace_id,
    )
    snapshot, snapshot_latest_path, snapshot_run_path = record_evaluation_snapshot(
        root=resolved_root,
        workspace_id=workspace_id,
        index_summary=index_summary,
    )
    curation_preview, curation_latest_path, curation_run_path = record_curation_export_preview(
        root=resolved_root,
        workspace_id=workspace_id,
        snapshot=snapshot,
    )
    learning_preview, learning_latest_path, learning_run_path = record_learning_dataset_preview(
        root=resolved_root,
        workspace_id=workspace_id,
        snapshot=snapshot,
        curation_preview=curation_preview,
    )
    latest_path = latest_verdict_path(workspace_id=workspace_id, root=resolved_root)
    run_path = verdict_run_path(workspace_id=workspace_id, root=resolved_root)
    payload = {
        "schema_name": FAILURE_MEMORY_VERDICT_SCHEMA_NAME,
        "schema_version": FAILURE_MEMORY_VERDICT_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "recorded_at_utc": timestamp_utc(),
        "event_id": event_id,
        "verdict": normalized_verdict,
        "signal": signal,
        "reason": reason,
        "paths": {
            "verdict_latest_path": str(latest_path),
            "verdict_run_path": str(run_path),
            "snapshot_latest_path": str(snapshot_latest_path),
            "snapshot_run_path": str(snapshot_run_path),
            "curation_preview_latest_path": str(curation_latest_path),
            "curation_preview_run_path": str(curation_run_path),
            "learning_preview_latest_path": str(learning_latest_path),
            "learning_preview_run_path": str(learning_run_path),
        },
        "snapshot": {
            "counts": snapshot.get("counts"),
            "curation": _event_curation_state(snapshot, event_id=event_id),
        },
        "learning_preview": {
            "counts": learning_preview.get("counts"),
        },
    }
    write_json(run_path, payload)
    write_json(latest_path, payload)
    return payload, latest_path, run_path


def _event_curation_state(snapshot: Mapping[str, Any] | None, *, event_id: str | None) -> dict[str, Any] | None:
    if snapshot is None or event_id is None:
        return None
    curation = _mapping_dict(snapshot.get("curation"))
    for item in curation.get("candidates") or []:
        if isinstance(item, Mapping) and _clean_text(item.get("event_id")) == event_id:
            return copy.deepcopy(dict(item))
    return None


def _latest_learning_preview(root: Path, *, workspace_id: str) -> dict[str, Any] | None:
    path = root / "artifacts" / "evaluation" / workspace_id / "learning" / "preview-latest.json"
    return _read_json_mapping(path)


def _latest_recall(root: Path, *, workspace_id: str) -> dict[str, Any] | None:
    return _read_json_mapping(latest_recall_path(workspace_id=workspace_id, root=root))


def _latest_verdict(root: Path, *, workspace_id: str) -> dict[str, Any] | None:
    return _read_json_mapping(latest_verdict_path(workspace_id=workspace_id, root=root))


def _verdict_for_event(
    verdict: Mapping[str, Any] | None,
    *,
    event_id: str | None,
) -> dict[str, Any] | None:
    if not isinstance(verdict, Mapping):
        return None
    if event_id is None:
        return copy.deepcopy(dict(verdict))
    if _clean_text(verdict.get("event_id")) != event_id:
        return None
    return copy.deepcopy(dict(verdict))


def _recall_source_event_id(recall: Mapping[str, Any] | None) -> str | None:
    if not isinstance(recall, Mapping):
        return None
    request = _mapping_dict(recall.get("request"))
    return _clean_text(request.get("source_event_id"))


def _recall_for_event(
    recall: Mapping[str, Any] | None,
    *,
    event_id: str | None,
    input_recorded_at_utc: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(recall, Mapping):
        return None
    recall_event_id = _recall_source_event_id(recall)
    if event_id is not None and recall_event_id is not None and recall_event_id != event_id:
        return None
    if recall_event_id is None and input_recorded_at_utc is not None:
        recall_time = _coerce_utc_datetime(recall.get("generated_at_utc"))
        input_time = _coerce_utc_datetime(input_recorded_at_utc)
        if recall_time is not None and input_time is not None and recall_time <= input_time:
            return None
    return copy.deepcopy(dict(recall))


def _source_inputs_from_latest(latest_input: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if latest_input is None:
        return []
    return [
        dict(item)
        for item in latest_input.get("source_inputs") or []
        if isinstance(item, Mapping)
    ]


def _event_id_from_latest_input_artifact(latest_input: Mapping[str, Any] | None) -> str | None:
    if latest_input is None:
        return None
    direct = _clean_text(latest_input.get("event_id"))
    if direct:
        return direct
    return None


def _top_recalled_rows(recall: Mapping[str, Any] | None, *, limit: int = 5) -> list[dict[str, Any]]:
    if recall is None:
        return []
    bundle = _mapping_dict(recall.get("bundle"))
    rows = [
        dict(item)
        for item in bundle.get("selected_candidates") or []
        if isinstance(item, Mapping)
    ]
    return rows[:limit]


def _markdown_table_cell(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("\n", " ").replace("|", "\\|")


def _learning_preview_queue_item(
    learning_preview: Mapping[str, Any] | None,
    *,
    event_id: str | None,
) -> dict[str, Any] | None:
    if event_id is None or not isinstance(learning_preview, Mapping):
        return None
    for item in learning_preview.get("review_queue") or []:
        if isinstance(item, Mapping) and _clean_text(item.get("event_id")) == event_id:
            return copy.deepcopy(dict(item))
    return None


def _learning_preview_excluded_item(
    learning_preview: Mapping[str, Any] | None,
    *,
    event_id: str | None,
) -> dict[str, Any] | None:
    if event_id is None or not isinstance(learning_preview, Mapping):
        return None
    for item in learning_preview.get("excluded_candidates") or []:
        if isinstance(item, Mapping) and _clean_text(item.get("event_id")) == event_id:
            return copy.deepcopy(dict(item))
    return None


def _learning_queue_report_fields(
    learning_preview: Mapping[str, Any] | None,
    *,
    event_id: str | None,
) -> dict[str, Any]:
    queue_item = _learning_preview_queue_item(learning_preview, event_id=event_id)
    excluded_item = _learning_preview_excluded_item(learning_preview, event_id=event_id)
    if queue_item is None and excluded_item is None:
        return {}
    source = queue_item or excluded_item or {}
    return {
        "queue_state": _clean_text(source.get("queue_state")),
        "next_action": _clean_text(source.get("next_action")),
        "blocked_reason": _clean_text(source.get("blocked_reason")),
        "blocked_reasons": _string_list(source.get("blocked_reasons")),
        "excluded_by": _string_list((excluded_item or {}).get("excluded_by")),
        "eligible_for_supervised_candidate": bool(source.get("eligible_for_supervised_candidate")),
        "comparison_evidence": _mapping_dict(source.get("comparison_evidence")),
    }


def _learning_state_for_report(
    *,
    event_id: str | None,
    latest_verdict: Mapping[str, Any] | None,
    snapshot: Mapping[str, Any] | None,
    learning_preview: Mapping[str, Any] | None,
) -> dict[str, Any]:
    queue_fields = _learning_queue_report_fields(learning_preview, event_id=event_id)
    event_contract = _mapping_dict((snapshot or {}).get("event_contract"))
    for check in event_contract.get("failed_checks") or []:
        if not isinstance(check, Mapping) or _clean_text(check.get("event_id")) != event_id:
            continue
        contract_status = _clean_text(check.get("contract_status"))
        source_artifact = _mapping_dict(check.get("source_artifact"))
        if contract_status == "missing_source":
            return {
                **queue_fields,
                "state": "missing_source",
                "blocked_reason": "missing_source",
                "source_artifact_reasons": _string_list(source_artifact.get("reasons")),
                "preview_only": True,
                "training_export_ready": False,
            }
        if contract_status == "invalid_event_contract":
            return {
                **queue_fields,
                "state": "blocked",
                "blocked_reason": "invalid_event_contract",
                "source_artifact_reasons": _string_list(source_artifact.get("reasons")),
                "preview_only": True,
                "training_export_ready": False,
            }
    curation_state = _event_curation_state(snapshot, event_id=event_id)
    if curation_state is not None:
        state = _clean_text(curation_state.get("state")) or "needs_review"
        reasons = _string_list(curation_state.get("reasons"))
        return {
            **queue_fields,
            "state": state,
            "reasons": reasons,
            "preview_only": True,
            "training_export_ready": False,
        }
    verdict = _clean_text((latest_verdict or {}).get("verdict"))
    if verdict == "reject":
        return {
            **queue_fields,
            "state": "blocked",
            "blocked_reason": "rejected",
            "preview_only": True,
            "training_export_ready": False,
        }
    counts = _mapping_dict((learning_preview or {}).get("counts"))
    return {
        **queue_fields,
        "state": "needs_review",
        "preview_only": True,
        "training_export_ready": False,
        "learning_preview_counts": counts,
    }


def build_review_risk_report(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> tuple[dict[str, Any], str, Path, Path]:
    resolved_root = _resolve_root(root)
    index_summary = rebuild_memory_index(root=resolved_root, workspace_id=workspace_id)
    snapshot, _snapshot_latest_path, _snapshot_run_path = record_evaluation_snapshot(
        root=resolved_root,
        workspace_id=workspace_id,
        index_summary=index_summary,
    )
    latest_input = _latest_input_artifact(root=resolved_root, workspace_id=workspace_id)
    latest_input_event_id = _event_id_from_latest_input_artifact(latest_input)
    raw_recall = _latest_recall(resolved_root, workspace_id=workspace_id)
    raw_verdict = _latest_verdict(resolved_root, workspace_id=workspace_id)
    learning_preview, learning_latest_path, learning_run_path = record_learning_dataset_preview(
        root=resolved_root,
        workspace_id=workspace_id,
        snapshot=snapshot,
    )
    input_summary = _mapping_dict((latest_input or {}).get("input_summary"))
    input_recorded_at_utc = _clean_text((latest_input or {}).get("timestamp_utc"))
    fallback_verdict_event_id = _clean_text((raw_verdict or {}).get("event_id"))
    event_id = latest_input_event_id or fallback_verdict_event_id
    recall = _recall_for_event(raw_recall, event_id=event_id, input_recorded_at_utc=input_recorded_at_utc)
    verdict = _verdict_for_event(raw_verdict, event_id=event_id)
    source_inputs = _source_inputs_from_latest(latest_input)
    risk_note = _mapping_dict((recall or {}).get("risk_note"))
    recalled_rows = _top_recalled_rows(recall)
    learning_state = _learning_state_for_report(
        event_id=event_id,
        latest_verdict=verdict,
        snapshot=snapshot,
        learning_preview=learning_preview,
    )
    report_latest = latest_report_path(workspace_id=workspace_id, root=resolved_root)
    report_run = report_run_path(workspace_id=workspace_id, root=resolved_root)
    metadata_latest = report_metadata_latest_path(workspace_id=workspace_id, root=resolved_root)
    metadata_run = report_metadata_run_path(workspace_id=workspace_id, root=resolved_root)

    lines = [
        "# Review Risk Report",
        "",
        "## Patch",
    ]
    if source_inputs:
        for source in source_inputs:
            lines.append(f"- Source: `{_markdown_table_cell(source.get('workspace_relative_path') or source.get('path'))}`")
            if source.get("sha256"):
                lines.append(f"- SHA-256: `{source['sha256']}`")
    else:
        lines.append("- Source: not recorded")
    if input_summary:
        changed_files = _string_list(input_summary.get("changed_files"))
        if changed_files:
            lines.append(f"- Changed files: {', '.join(f'`{item}`' for item in changed_files[:8])}")
        if input_summary.get("added_lines") is not None or input_summary.get("removed_lines") is not None:
            lines.append(
                f"- Diff size: +{int(input_summary.get('added_lines') or 0)} "
                f"/ -{int(input_summary.get('removed_lines') or 0)}"
            )
    if event_id:
        lines.append(f"- Event: `{event_id}`")

    lines.extend(("", "## Recalled Failure Memory", ""))
    if recalled_rows:
        lines.extend(
            [
                "| Rank | Prior outcome | Evidence path | Why it matters |",
                "|---:|---|---|---|",
            ]
        )
        for index, row in enumerate(recalled_rows, start=1):
            reasons = ", ".join(_string_list(row.get("reasons"))[:4]) or "similar prior evidence"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(index),
                        _markdown_table_cell(_clean_text(row.get("status")) or "-"),
                        f"`{_markdown_table_cell(row.get('artifact_path') or '-')}`",
                        _markdown_table_cell(reasons),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No similar failure memory was selected. Keep the result out of positive learning candidates until reviewed.")

    lines.extend(
        [
            "",
            "## Risk Note",
            "",
            _clean_text(risk_note.get("recommendation"))
            or "Record a human verdict before reusing this evidence.",
            "",
            "## Prior Verdicts",
            "",
        ]
    )
    recent_signals = [
        item
        for item in (snapshot or {}).get("recent_signals") or []
        if isinstance(item, Mapping)
    ]
    if recent_signals:
        for signal in recent_signals[:6]:
            lines.append(
                f"- `{_clean_text(signal.get('signal_kind')) or 'signal'}` "
                f"on `{_clean_text(signal.get('source_event_id')) or 'n/a'}`"
            )
    else:
        lines.append("- No prior human verdict signals recorded.")

    lines.extend(("", "## Backend / Proposal Comparison", ""))
    comparisons = [
        item
        for item in (snapshot or {}).get("comparisons") or []
        if isinstance(item, Mapping)
    ]
    if comparisons:
        for comparison in comparisons[:3]:
            lines.append(
                f"- `{_clean_text(comparison.get('outcome')) or 'needs_follow_up'}` "
                f"winner=`{_clean_text(comparison.get('winner_event_id')) or 'n/a'}`"
            )
            rationale = _clean_text(comparison.get("rationale"))
            if rationale:
                lines.append(f"  - Rationale: {_markdown_table_cell(rationale)}")
            criteria = _string_list(comparison.get("criteria"))
            if criteria:
                criteria_text = ", ".join("`" + _markdown_table_cell(item) + "`" for item in criteria[:6])
                lines.append(f"  - Criteria: {criteria_text}")
            candidate_sources = [
                dict(item)
                for item in comparison.get("candidate_sources") or []
                if isinstance(item, Mapping)
            ]
            for candidate in candidate_sources[:4]:
                source_path = (
                    _clean_text(candidate.get("source_input_path"))
                    or _clean_text(candidate.get("artifact_path"))
                    or "n/a"
                )
                backend_label = (
                    _clean_text(candidate.get("backend_id"))
                    or _clean_text(candidate.get("model_id"))
                    or "n/a"
                )
                lines.append(
                    "  - Candidate "
                    f"`{_clean_text(candidate.get('event_id')) or 'n/a'}` "
                    f"source=`{_markdown_table_cell(source_path)}` backend=`{_markdown_table_cell(backend_label)}`"
                )
    else:
        lines.append("- None recorded for this review.")

    lines.extend(("", "## Human Verdict", "", "```json"))
    if verdict:
        lines.append(
            json.dumps(
                {
                    "verdict": verdict.get("verdict"),
                    "event_id": verdict.get("event_id"),
                    "reason": verdict.get("reason"),
                    "human_gate_required": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        lines.append(json.dumps(build_verdict_template(event_id=event_id), ensure_ascii=False, indent=2))
    lines.extend(["```", "", "## Learning-Candidate State", "", "```json"])
    lines.append(json.dumps(learning_state, ensure_ascii=False, indent=2))
    lines.extend(["```", "", "## Validation Metrics", ""])
    metrics = demand_validation_metrics(index_summary.get("event_contract"))
    source_metric = _mapping_dict(metrics.get("source_path_completeness"))
    lines.append(
        "- Source path completeness: "
        f"{source_metric.get('observed') if source_metric.get('observed') is not None else 'n/a'} "
        f"(target {source_metric.get('target') or '>= 0.90'})"
    )
    lines.append("- Useful recalled evidence rate: needs dogfood verdicts (target >= 0.30)")
    lines.append("- Critical false evidence: target 0")
    lines.append("- Human verdict capture friction: single command target <= 30 seconds")
    lines.extend(
        [
            "",
            "## Why This Is Not Training Data",
            "",
            "This report is preview-only evidence. Missing-source, rejected, unresolved, or failed records stay blocked from learning-candidate promotion.",
        ]
    )
    markdown = "\n".join(lines) + "\n"
    report_run.parent.mkdir(parents=True, exist_ok=True)
    report_latest.parent.mkdir(parents=True, exist_ok=True)
    report_run.write_text(markdown, encoding="utf-8")
    report_latest.write_text(markdown, encoding="utf-8")

    metadata = {
        "schema_name": REVIEW_RISK_REPORT_SCHEMA_NAME,
        "schema_version": REVIEW_RISK_REPORT_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "generated_at_utc": timestamp_utc(),
        "event_id": event_id,
        "paths": {
            "report_latest_path": str(report_latest),
            "report_run_path": str(report_run),
            "report_metadata_latest_path": str(metadata_latest),
            "report_metadata_run_path": str(metadata_run),
            "learning_preview_latest_path": str(learning_latest_path),
            "learning_preview_run_path": str(learning_run_path),
        },
        "source_inputs": source_inputs,
        "risk_note": risk_note,
        "learning_state": learning_state,
        "learning_preview_counts": _mapping_dict(learning_preview.get("counts")),
        "validation_metrics": metrics,
    }
    write_json(metadata_run, metadata)
    write_json(metadata_latest, metadata)
    return metadata, markdown, report_latest, report_run


def resolve_review_risk_pack_path(pack: Path | str, *, root: Path) -> Path:
    pack_text = str(pack)
    if pack_text == REVIEW_RISK_PACK_NAME:
        return root / "templates" / "review-risk-pack.satellite.yaml"
    candidate = Path(pack)
    if not candidate.is_absolute():
        candidate = root / candidate
    return resolve_pack_manifest_path(candidate)


def _latest_patch_input_for_path(
    *,
    patch_path: Path,
    root: Path,
    workspace_id: str,
) -> dict[str, Any] | None:
    if not patch_path.is_file():
        return None
    latest_input = _latest_input_artifact(root=root, workspace_id=workspace_id)
    if not isinstance(latest_input, Mapping):
        return None
    if _clean_text(latest_input.get("input_kind")) != "patch":
        return None
    patch_sha256 = _file_sha256(patch_path)
    patch_path_text = str(patch_path.resolve())
    source_inputs = _source_inputs_from_latest(latest_input)
    if not any(
        _clean_text(source.get("sha256")) == patch_sha256
        and _clean_text(source.get("path")) == patch_path_text
        for source in source_inputs
    ):
        return None
    event_id = _event_id_from_latest_input_artifact(latest_input)
    if event_id is None:
        return None
    return {
        "event_id": event_id,
        "input_summary": _mapping_dict(latest_input.get("input_summary")),
        "source_inputs": source_inputs,
    }


def run_review_risk_pack(
    *,
    pack: Path | str,
    patch_path: Path,
    query: str | None = None,
    note: str | None = None,
    limit: int = 5,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> tuple[dict[str, Any], str, Path, Path]:
    resolved_root = _resolve_root(root)
    manifest_path = resolve_review_risk_pack_path(pack, root=resolved_root)
    manifest = load_pack_manifest(manifest_path)
    if manifest.get("name") != REVIEW_RISK_PACK_NAME:
        raise PackManifestError("Only the built-in review-risk-pack runner is available.")
    audit, _audit_latest, _audit_run = audit_pack_path(manifest_path, workspace_id=workspace_id, root=resolved_root)
    if audit.get("verdict") == "block":
        raise PackManifestError("review-risk-pack audit blocked execution.")
    resolved_patch = _path_from_text(patch_path, root=resolved_root)
    ingest = _latest_patch_input_for_path(
        patch_path=resolved_patch,
        root=resolved_root,
        workspace_id=workspace_id,
    )
    if ingest is None:
        ingest = record_file_input(
            input_kind="patch",
            source_path=resolved_patch,
            note=note or "review-risk-pack patch input",
            workspace_id=workspace_id,
            root=resolved_root,
        )
    patch_summary = _mapping_dict(ingest.get("input_summary"))
    default_query = "patch risk similar failure missing source rejected test fail repair"
    if patch_summary.get("changed_files"):
        default_query += " " + " ".join(_string_list(patch_summary.get("changed_files")))
    recall, _recall_latest, _recall_run = build_failure_recall(
        query=query or default_query,
        patch_path=resolved_patch,
        limit=limit,
        workspace_id=workspace_id,
        root=resolved_root,
        source_event_id=_clean_text(ingest.get("event_id")),
    )
    metadata, markdown, latest_path, run_path = build_review_risk_report(
        workspace_id=workspace_id,
        root=resolved_root,
    )
    metadata["pack_run"] = {
        "runner": "built_in_review_risk_pack",
        "pack_name": REVIEW_RISK_PACK_NAME,
        "manifest_path": str(manifest_path),
        "audit_verdict": audit.get("verdict"),
        "input_event_id": ingest.get("event_id"),
        "recall_selected_count": _mapping_dict(recall.get("bundle")).get("selected_count"),
    }
    paths = _mapping_dict(metadata.get("paths"))
    metadata_latest = Path(paths.get("report_metadata_latest_path") or report_metadata_latest_path(workspace_id=workspace_id, root=resolved_root))
    metadata_run = Path(paths.get("report_metadata_run_path") or report_metadata_run_path(workspace_id=workspace_id, root=resolved_root))
    write_json(metadata_run, metadata)
    write_json(metadata_latest, metadata)
    return metadata, markdown, latest_path, run_path


def format_ingest_result(result: Mapping[str, Any]) -> str:
    source = next(iter(result.get("source_inputs") or []), {})
    return "\n".join(
        [
            f"Recorded {result.get('input_kind')} input",
            f"Event: {result.get('event_id')}",
            f"Status: {result.get('status')}",
            f"Source: {source.get('workspace_relative_path') or source.get('path') or 'n/a'}",
            f"Artifact: {result.get('artifact_path')}",
        ]
    )


def format_proposal_comparison_result(result: Mapping[str, Any]) -> str:
    comparison = _mapping_dict(result.get("comparison"))
    verdict = _mapping_dict(result.get("human_verdict"))
    paths = _mapping_dict(result.get("paths"))
    lines = [
        "Proposal comparison recorded",
        f"Comparison: {comparison.get('comparison_id')}",
        f"Verdict: {verdict.get('verdict')}",
        f"Outcome: {comparison.get('outcome')}",
        f"Winner: {comparison.get('winner_event_id') or 'none'}",
        f"Losers: {', '.join(_string_list(verdict.get('loser_event_ids'))) or 'none'}",
        f"Rationale: {verdict.get('rationale')}",
        f"Ledger: {paths.get('comparison_log_path')}",
        "",
        "Candidates:",
    ]
    for index, candidate in enumerate(result.get("candidates") or [], start=1):
        if not isinstance(candidate, Mapping):
            continue
        source = next(iter(candidate.get("source_inputs") or []), {})
        source_path = source.get("workspace_relative_path") or source.get("path") if isinstance(source, Mapping) else None
        lines.append(f"- {index}. `{candidate.get('event_id')}` source=`{source_path or 'n/a'}`")
    lines.extend(
        [
            "",
            "Learning: preview-only; run `satlab learning inspect --preview-only` to inspect queue state.",
        ]
    )
    return "\n".join(lines)


def format_verdict_result(result: Mapping[str, Any]) -> str:
    paths = _mapping_dict(result.get("paths"))
    return "\n".join(
        [
            f"Recorded verdict: {result.get('verdict')}",
            f"Event: {result.get('event_id')}",
            f"Signal: {_mapping_dict(result.get('signal')).get('signal_kind')}",
            f"Snapshot: {paths.get('snapshot_run_path') or 'n/a'}",
            f"Learning preview: {paths.get('learning_preview_run_path') or 'n/a'}",
        ]
    )


def event_contract_summary(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    index_summary = rebuild_memory_index(root=resolved_root, workspace_id=workspace_id)
    event_log = read_event_log(Path(index_summary["event_log_path"]))
    contract = build_event_contract_report(event_log.get("events") or [], root=resolved_root)
    return {
        "workspace_id": workspace_id,
        "event_log_path": index_summary["event_log_path"],
        "event_contract": contract,
        "validation_metrics": demand_validation_metrics(contract),
    }
