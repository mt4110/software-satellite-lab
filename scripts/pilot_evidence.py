#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from gemma_runtime import repo_root, timestamp_slug, timestamp_utc, write_json
from workspace_state import DEFAULT_WORKSPACE_ID


PILOT_EVIDENCE_RECORD_SCHEMA_NAME = "paid_pilot_evidence_record"
PILOT_EVIDENCE_RECORD_SCHEMA_VERSION = 1
PILOT_EVIDENCE_LEDGER_SCHEMA_NAME = "software-satellite-paid-pilot-evidence-ledger"
PILOT_EVIDENCE_LEDGER_SCHEMA_VERSION = 1
PILOT_REPORT_SCHEMA_NAME = "software-satellite-paid-pilot-gate-report"
PILOT_REPORT_SCHEMA_VERSION = 1

PILOT_RECORD_TYPES = ("discovery_call", "hands_on_demo", "loi", "paid_pilot", "security_review")
PILOT_JUDGEMENTS = ("yes", "no", "unclear")
DEMO_CHECKLIST_KEYS = (
    "failure_memory_reviewed",
    "signed_evidence_reviewed",
    "no_code_upload_confirmed",
    "transcript_claim_rejection_reviewed",
)
SCORE_KEYS = (
    "exact_pain_recognized",
    "wants_to_try",
    "willing_to_install_locally",
    "willingness_to_pay",
    "security_sensitive_user",
    "no_code_upload_matters",
    "budget_path",
    "audit_trail_more_important_than_review_comments",
    "paid_pilot_commitment",
    "demo_completed",
    "demo_checklist_passed",
)
ANTI_DRIFT_GUARDRAILS = {
    "no_default_telemetry": True,
    "no_raw_code_upload": True,
    "no_remote_positive_support_without_local_validation": True,
    "no_training_export": True,
    "no_hidden_agent_execution": True,
}
PILOT_EXIT_GATES = (
    {
        "key": "discovery_calls",
        "label": "Discovery calls",
        "operator": ">=",
        "target": 20,
    },
    {
        "key": "hands_on_demos",
        "label": "Hands-on demos",
        "operator": ">=",
        "target": 5,
    },
    {
        "key": "security_sensitive_users",
        "label": "Security-sensitive users",
        "operator": ">=",
        "target": 5,
    },
    {
        "key": "exact_pain_recognition",
        "label": "Exact-pain recognition",
        "operator": ">=",
        "target": 12,
    },
    {
        "key": "wants_to_try",
        "label": "Wants to try",
        "operator": ">=",
        "target": 5,
    },
    {
        "key": "paid_pilot_commitments_or_lois",
        "label": "Paid-pilot commitments or LOIs",
        "operator": ">=",
        "target": 2,
    },
)
PILOT_TEMPLATE_FILES = {
    "pilot_interview_script": "templates/pilot_interview_script.md",
    "pilot_demo_checklist": "templates/pilot_demo_checklist.md",
    "paid_pilot_statement_of_value": "templates/paid_pilot_statement_of_value.md",
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


def _string_list(value: Iterable[Any] | None) -> list[str]:
    if value is None:
        return []
    return [cleaned for item in value if (cleaned := _clean_text(item)) is not None]


def _normalize_judgement(value: str) -> str:
    normalized = (_clean_text(value) or "").lower().replace("_", "-")
    if normalized not in PILOT_JUDGEMENTS:
        raise ValueError(f"Unsupported pilot judgement `{value}`.")
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


def pilot_evidence_root(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return _resolve_root(root) / "artifacts" / "pilot_evidence" / workspace_id


def pilot_evidence_ledger_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return pilot_evidence_root(workspace_id=workspace_id, root=root) / "pilot-evidence.jsonl"


def pilot_record_latest_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return pilot_evidence_root(workspace_id=workspace_id, root=root) / "records" / "latest.json"


def pilot_record_artifact_path(
    record_type: str,
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return (
        pilot_evidence_root(workspace_id=workspace_id, root=root)
        / "records"
        / "runs"
        / f"{timestamp_slug()}-{record_type}.json"
    )


def pilot_report_latest_json_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return pilot_evidence_root(workspace_id=workspace_id, root=root) / "reports" / "latest.json"


def pilot_report_latest_md_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return pilot_evidence_root(workspace_id=workspace_id, root=root) / "reports" / "latest.md"


def pilot_report_run_json_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    run_slug: str | None = None,
) -> Path:
    slug = run_slug or timestamp_slug()
    return pilot_evidence_root(workspace_id=workspace_id, root=root) / "reports" / "runs" / f"{slug}-pilot-gate.json"


def pilot_report_run_md_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    run_slug: str | None = None,
) -> Path:
    slug = run_slug or timestamp_slug()
    return pilot_evidence_root(workspace_id=workspace_id, root=root) / "reports" / "runs" / f"{slug}-pilot-gate.md"


def _ledger_header(*, workspace_id: str) -> dict[str, Any]:
    return {
        "schema_name": PILOT_EVIDENCE_LEDGER_SCHEMA_NAME,
        "schema_version": PILOT_EVIDENCE_LEDGER_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "created_at_utc": timestamp_utc(),
    }


def _read_log_header(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        raise ValueError(f"Pilot evidence ledger `{path}` was empty.")
    header = json.loads(first_line)
    return dict(header) if isinstance(header, Mapping) else {}


def _validate_ledger_appendable(path: Path, *, workspace_id: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    header = _read_log_header(path)
    if (
        header.get("schema_name") != PILOT_EVIDENCE_LEDGER_SCHEMA_NAME
        or header.get("schema_version") != PILOT_EVIDENCE_LEDGER_SCHEMA_VERSION
    ):
        raise ValueError(f"Unexpected pilot evidence ledger schema in `{path}`.")
    if header.get("workspace_id") != workspace_id:
        raise ValueError(f"Pilot evidence ledger `{path}` belongs to workspace `{header.get('workspace_id')}`.")


def _append_ledger(path: Path, payload: Mapping[str, Any], *, workspace_id: str) -> dict[str, Any]:
    _validate_ledger_appendable(path, workspace_id=workspace_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(_ledger_header(workspace_id=workspace_id), ensure_ascii=False) + "\n")
    record = copy.deepcopy(dict(payload))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _record_paths(record_type: str, *, workspace_id: str, root: Path) -> tuple[Path, Path, Path]:
    latest_path = pilot_record_latest_path(workspace_id=workspace_id, root=root)
    run_path = pilot_record_artifact_path(record_type, workspace_id=workspace_id, root=root)
    ledger_path = pilot_evidence_ledger_path(workspace_id=workspace_id, root=root)
    return latest_path, run_path, ledger_path


def _record_guardrails() -> dict[str, bool]:
    return dict(ANTI_DRIFT_GUARDRAILS)


def _record_base(
    *,
    record_type: str,
    participant_label: str,
    participant_segment: str,
    evidence: Mapping[str, Any],
    score: Mapping[str, Any],
    source_refs: Mapping[str, Any],
    note: str | None,
    workspace_id: str,
    root: Path,
) -> dict[str, Any]:
    if record_type not in PILOT_RECORD_TYPES:
        raise ValueError(f"Unsupported pilot record type `{record_type}`.")
    participant = _clean_text(participant_label)
    if participant is None:
        raise ValueError("Pilot evidence requires --participant.")
    segment = _clean_text(participant_segment)
    if segment is None:
        raise ValueError("Pilot evidence requires --segment.")
    latest_path, run_path, ledger_path = _record_paths(record_type, workspace_id=workspace_id, root=root)
    payload = {
        "schema_name": PILOT_EVIDENCE_RECORD_SCHEMA_NAME,
        "schema_version": PILOT_EVIDENCE_RECORD_SCHEMA_VERSION,
        "pilot_record_id": f"{workspace_id}:pilot:{record_type}:{timestamp_slug()}",
        "workspace_id": workspace_id,
        "recorded_at_utc": timestamp_utc(),
        "record_type": record_type,
        "participant_label": participant,
        "participant_segment": segment,
        "note": _clean_text(note),
        "evidence": copy.deepcopy(dict(evidence)),
        "score": copy.deepcopy(dict(score)),
        "source_refs": copy.deepcopy(dict(source_refs)),
        "guardrails": _record_guardrails(),
        "paths": {
            "pilot_record_latest_path": str(latest_path),
            "pilot_record_path": str(run_path),
            "pilot_evidence_ledger_path": str(ledger_path),
        },
    }
    validate_pilot_evidence_record(payload)
    write_json(run_path, payload)
    write_json(latest_path, payload)
    _append_ledger(ledger_path, payload, workspace_id=workspace_id)
    return payload


def record_pilot_interview(
    *,
    participant_label: str,
    participant_segment: str,
    notes_file: Path,
    exact_pain_recognized: str,
    wants_to_try: str,
    willing_to_install_locally: str,
    willingness_to_pay: str,
    security_sensitive_user: str,
    no_code_upload_matters: str,
    budget_path: str = "unclear",
    audit_trail_more_important_than_review_comments: str = "unclear",
    current_workaround: str | None = None,
    failure_example_allowed: str | None = None,
    objections: Sequence[str] | None = None,
    note: str | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    resolved_notes = _path_from_text(notes_file, root=resolved_root)
    score = {
        "exact_pain_recognized": _judgement_bool(_normalize_judgement(exact_pain_recognized)),
        "wants_to_try": _judgement_bool(_normalize_judgement(wants_to_try)),
        "willing_to_install_locally": _judgement_bool(_normalize_judgement(willing_to_install_locally)),
        "willingness_to_pay": _judgement_bool(_normalize_judgement(willingness_to_pay)),
        "security_sensitive_user": _judgement_bool(_normalize_judgement(security_sensitive_user)),
        "no_code_upload_matters": _judgement_bool(_normalize_judgement(no_code_upload_matters)),
        "budget_path": _judgement_bool(_normalize_judgement(budget_path)),
        "audit_trail_more_important_than_review_comments": _judgement_bool(
            _normalize_judgement(audit_trail_more_important_than_review_comments)
        ),
        "paid_pilot_commitment": False,
        "demo_completed": None,
        "demo_checklist_passed": None,
    }
    evidence = {
        "interview_kind": "pain_validation",
        "current_workaround": _clean_text(current_workaround),
        "failure_example_allowed": _clean_text(failure_example_allowed),
        "security_or_compliance_objections": _string_list(objections),
        "scorecard": {
            "exact_pain_recognized": _normalize_judgement(exact_pain_recognized),
            "wants_to_try": _normalize_judgement(wants_to_try),
            "willing_to_install_locally": _normalize_judgement(willing_to_install_locally),
            "willingness_to_pay": _normalize_judgement(willingness_to_pay),
            "security_sensitive_user": _normalize_judgement(security_sensitive_user),
            "no_code_upload_matters": _normalize_judgement(no_code_upload_matters),
            "budget_path": _normalize_judgement(budget_path),
            "audit_trail_priority": _normalize_judgement(audit_trail_more_important_than_review_comments),
        },
    }
    return _record_base(
        record_type="discovery_call",
        participant_label=participant_label,
        participant_segment=participant_segment,
        evidence=evidence,
        score=score,
        source_refs={
            "notes_file_ref": _source_file_record(
                resolved_notes,
                role="pilot_interview_notes",
                root=resolved_root,
            )
        },
        note=note,
        workspace_id=workspace_id,
        root=resolved_root,
    )


def record_pilot_demo(
    *,
    participant_label: str,
    participant_segment: str,
    notes_file: Path,
    failure_memory_reviewed: bool,
    signed_evidence_reviewed: bool,
    no_code_upload_confirmed: bool,
    transcript_claim_rejection_reviewed: bool,
    wants_to_try: str = "unclear",
    exact_pain_recognized: str = "unclear",
    security_sensitive_user: str = "unclear",
    note: str | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    resolved_notes = _path_from_text(notes_file, root=resolved_root)
    checklist = {
        "failure_memory_reviewed": bool(failure_memory_reviewed),
        "signed_evidence_reviewed": bool(signed_evidence_reviewed),
        "no_code_upload_confirmed": bool(no_code_upload_confirmed),
        "transcript_claim_rejection_reviewed": bool(transcript_claim_rejection_reviewed),
    }
    checklist_passed = all(checklist.values())
    score = {
        "exact_pain_recognized": _judgement_bool(_normalize_judgement(exact_pain_recognized)),
        "wants_to_try": _judgement_bool(_normalize_judgement(wants_to_try)),
        "willing_to_install_locally": None,
        "willingness_to_pay": None,
        "security_sensitive_user": _judgement_bool(_normalize_judgement(security_sensitive_user)),
        "no_code_upload_matters": True if no_code_upload_confirmed else None,
        "budget_path": None,
        "audit_trail_more_important_than_review_comments": None,
        "paid_pilot_commitment": False,
        "demo_completed": True,
        "demo_checklist_passed": checklist_passed,
    }
    evidence = {
        "demo_kind": "commercial_wedge_demo",
        "demo_checklist": checklist,
        "required_demo_topics": [
            "failure memory",
            "signed evidence",
            "no code upload",
            "transcript claim rejection",
        ],
    }
    return _record_base(
        record_type="hands_on_demo",
        participant_label=participant_label,
        participant_segment=participant_segment,
        evidence=evidence,
        score=score,
        source_refs={
            "notes_file_ref": _source_file_record(
                resolved_notes,
                role="pilot_demo_notes",
                root=resolved_root,
            )
        },
        note=note,
        workspace_id=workspace_id,
        root=resolved_root,
    )


def record_pilot_loi(
    *,
    participant_label: str,
    participant_segment: str,
    notes_file: Path,
    commitment_kind: str = "loi",
    amount: str | None = None,
    decision_maker: str | None = None,
    pilot_shape: str | None = None,
    exact_pain_recognized: str = "yes",
    wants_to_try: str = "yes",
    security_sensitive_user: str = "unclear",
    budget_path: str = "unclear",
    note: str | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> dict[str, Any]:
    normalized_kind = (_clean_text(commitment_kind) or "loi").lower().replace("-", "_")
    if normalized_kind not in {"loi", "paid_pilot"}:
        raise ValueError("--commitment-kind must be `loi` or `paid-pilot`.")
    resolved_root = _resolve_root(root)
    resolved_notes = _path_from_text(notes_file, root=resolved_root)
    score = {
        "exact_pain_recognized": _judgement_bool(_normalize_judgement(exact_pain_recognized)),
        "wants_to_try": _judgement_bool(_normalize_judgement(wants_to_try)),
        "willing_to_install_locally": True,
        "willingness_to_pay": True,
        "security_sensitive_user": _judgement_bool(_normalize_judgement(security_sensitive_user)),
        "no_code_upload_matters": None,
        "budget_path": _judgement_bool(_normalize_judgement(budget_path)),
        "audit_trail_more_important_than_review_comments": None,
        "paid_pilot_commitment": True,
        "demo_completed": None,
        "demo_checklist_passed": None,
    }
    evidence = {
        "commitment_kind": normalized_kind,
        "amount": _clean_text(amount),
        "decision_maker": _clean_text(decision_maker),
        "pilot_shape": _clean_text(pilot_shape),
        "offer": "Local-first AI coding evidence audit and failure-memory starter pack",
    }
    return _record_base(
        record_type=normalized_kind,
        participant_label=participant_label,
        participant_segment=participant_segment,
        evidence=evidence,
        score=score,
        source_refs={
            "notes_file_ref": _source_file_record(
                resolved_notes,
                role="pilot_loi_notes",
                root=resolved_root,
            )
        },
        note=note,
        workspace_id=workspace_id,
        root=resolved_root,
    )


def validate_pilot_evidence_record(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    errors: list[str] = []
    allowed_top_level = {
        "schema_name",
        "schema_version",
        "pilot_record_id",
        "workspace_id",
        "recorded_at_utc",
        "record_type",
        "participant_label",
        "participant_segment",
        "note",
        "evidence",
        "score",
        "source_refs",
        "guardrails",
        "paths",
    }
    unknown_keys = sorted(set(payload) - allowed_top_level)
    if unknown_keys:
        errors.append(f"unknown top-level keys: {', '.join(unknown_keys)}")
    if payload.get("schema_name") != PILOT_EVIDENCE_RECORD_SCHEMA_NAME:
        errors.append("schema_name must be paid_pilot_evidence_record")
    if payload.get("schema_version") != PILOT_EVIDENCE_RECORD_SCHEMA_VERSION:
        errors.append("schema_version must be 1")
    record_type = _clean_text(payload.get("record_type"))
    if record_type not in PILOT_RECORD_TYPES:
        errors.append("record_type is unsupported")
    if _clean_text(payload.get("participant_segment")) is None:
        errors.append("participant_segment is required")
    if _clean_text(payload.get("participant_label")) is None:
        errors.append("participant_label is required")
    if _clean_text(payload.get("pilot_record_id")) is None:
        errors.append("pilot_record_id is required")
    if _clean_text(payload.get("workspace_id")) is None:
        errors.append("workspace_id is required")
    if _clean_text(payload.get("recorded_at_utc")) is None:
        errors.append("recorded_at_utc is required")

    evidence = _mapping_dict(payload.get("evidence"))
    score = _mapping_dict(payload.get("score"))
    source_refs = _mapping_dict(payload.get("source_refs"))
    guardrails = _mapping_dict(payload.get("guardrails"))
    if not isinstance(payload.get("evidence"), Mapping):
        errors.append("evidence must be an object")
    if not isinstance(payload.get("score"), Mapping):
        errors.append("score must be an object")
    else:
        missing_score_keys = [key for key in SCORE_KEYS if key not in score]
        if missing_score_keys:
            errors.append(f"score missing keys: {', '.join(missing_score_keys)}")
    if not isinstance(payload.get("source_refs"), Mapping):
        errors.append("source_refs must be an object")
    notes_file_ref = source_refs.get("notes_file_ref")
    if not notes_file_ref:
        errors.append("source_refs.notes_file_ref is required")
    elif not isinstance(notes_file_ref, Mapping):
        errors.append("source_refs.notes_file_ref must be an object")
    elif _clean_text(notes_file_ref.get("path")) is None:
        errors.append("source_refs.notes_file_ref.path is required")
    if not isinstance(payload.get("guardrails"), Mapping):
        errors.append("guardrails must be an object")
    for key, expected in ANTI_DRIFT_GUARDRAILS.items():
        if guardrails.get(key) is not expected:
            errors.append(f"guardrails.{key} must be {str(expected).lower()}")

    for key, value in score.items():
        if value is not None and not isinstance(value, bool):
            errors.append(f"score.{key} must be true, false, or null")

    if record_type == "hands_on_demo":
        checklist = evidence.get("demo_checklist")
        if not isinstance(checklist, Mapping):
            errors.append("hands_on_demo evidence.demo_checklist is required")
        else:
            missing = [key for key in DEMO_CHECKLIST_KEYS if key not in checklist]
            if missing:
                errors.append(f"demo checklist missing: {', '.join(missing)}")
            non_boolean = [
                key
                for key in DEMO_CHECKLIST_KEYS
                if key in checklist and not isinstance(checklist.get(key), bool)
            ]
            if non_boolean:
                errors.append(f"demo checklist values must be booleans: {', '.join(non_boolean)}")
            expected_passed = all(bool(checklist.get(key)) for key in DEMO_CHECKLIST_KEYS)
            if score.get("demo_checklist_passed") is not expected_passed:
                errors.append("score.demo_checklist_passed must match the demo checklist")
    if record_type in {"loi", "paid_pilot"}:
        if score.get("paid_pilot_commitment") is not True:
            errors.append("LOI and paid_pilot records must set score.paid_pilot_commitment=true")
        if evidence.get("commitment_kind") not in {"loi", "paid_pilot"}:
            errors.append("LOI and paid_pilot records require evidence.commitment_kind")

    if errors:
        raise ValueError("Invalid pilot evidence record: " + "; ".join(errors))
    return payload


def _records_from_jsonl(path: Path, *, workspace_id: str | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        first_payload_seen = False
        for line_number, line in enumerate(handle, start=1):
            cleaned = line.strip()
            if not cleaned:
                continue
            payload = json.loads(cleaned)
            if not isinstance(payload, Mapping):
                raise ValueError(f"Pilot evidence ledger line {line_number} is not an object: `{path}`.")
            if not first_payload_seen:
                first_payload_seen = True
                if payload.get("schema_name") == PILOT_EVIDENCE_LEDGER_SCHEMA_NAME:
                    if payload.get("schema_version") != PILOT_EVIDENCE_LEDGER_SCHEMA_VERSION:
                        raise ValueError(f"Unexpected pilot evidence ledger schema version in `{path}`.")
                    if workspace_id is not None and payload.get("workspace_id") != workspace_id:
                        raise ValueError(
                            f"Pilot evidence ledger `{path}` belongs to workspace `{payload.get('workspace_id')}`."
                        )
                    continue
            records.append(validate_pilot_evidence_record(payload))
    return records


def _records_from_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return _validate_record_list(payload, path=path)
    if isinstance(payload, Mapping) and isinstance(payload.get("records"), list):
        return _validate_record_list(payload["records"], path=path)
    if isinstance(payload, Mapping):
        return [validate_pilot_evidence_record(payload)]
    raise ValueError(f"Pilot evidence fixture is not a record collection: `{path}`.")


def _validate_record_list(items: Sequence[Any], *, path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Pilot evidence record {index} is not an object: `{path}`.")
        records.append(validate_pilot_evidence_record(item))
    return records


def read_pilot_evidence_records(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    records_path: Path | None = None,
) -> list[dict[str, Any]]:
    resolved_root = _resolve_root(root)
    path = (
        _path_from_text(records_path, root=resolved_root)
        if records_path is not None
        else pilot_evidence_ledger_path(
            workspace_id=workspace_id,
            root=resolved_root,
        )
    )
    if not path.exists():
        return []
    if path.suffix.lower() == ".json":
        return _records_from_json(path)
    return _records_from_jsonl(path, workspace_id=workspace_id if records_path is None else None)


def validate_pilot_evidence_ledger(
    path: Path,
    *,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    resolved_root = _resolve_root(root)
    return read_pilot_evidence_records(root=resolved_root, records_path=path)


def _participant_key(record: Mapping[str, Any]) -> str:
    label = _clean_text(record.get("participant_label"))
    if label is not None:
        return label.lower()
    return f"{record.get('participant_segment')}:{record.get('pilot_record_id')}"


def _participant_count(records: Iterable[Mapping[str, Any]], predicate: Any) -> int:
    return len({_participant_key(record) for record in records if predicate(record)})


def _paid_commitment_count(records: Iterable[Mapping[str, Any]]) -> int:
    return _participant_count(
        records,
        lambda record: record.get("record_type") in {"loi", "paid_pilot"}
        or _mapping_dict(record.get("score")).get("paid_pilot_commitment") is True,
    )


def _metrics_from_records(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "discovery_calls": sum(1 for record in records if record.get("record_type") == "discovery_call"),
        "hands_on_demos": sum(
            1
            for record in records
            if record.get("record_type") == "hands_on_demo"
            and _mapping_dict(record.get("score")).get("demo_completed") is True
            and _mapping_dict(record.get("score")).get("demo_checklist_passed") is True
        ),
        "security_sensitive_users": _participant_count(
            records,
            lambda record: _mapping_dict(record.get("score")).get("security_sensitive_user") is True,
        ),
        "exact_pain_recognition": _participant_count(
            records,
            lambda record: _mapping_dict(record.get("score")).get("exact_pain_recognized") is True,
        ),
        "wants_to_try": _participant_count(
            records,
            lambda record: _mapping_dict(record.get("score")).get("wants_to_try") is True,
        ),
        "paid_pilot_commitments_or_lois": _paid_commitment_count(records),
    }


def _criterion(
    *,
    key: str,
    label: str,
    operator: str,
    target: int,
    observed: int,
) -> dict[str, Any]:
    if operator != ">=":
        raise ValueError(f"Unsupported pilot gate operator `{operator}`.")
    return {
        "key": key,
        "label": label,
        "operator": operator,
        "target": target,
        "observed": observed,
        "status": "pass" if observed >= target else "needs_data",
    }


def _criteria_from_metrics(metrics: Mapping[str, int]) -> list[dict[str, Any]]:
    return [
        _criterion(
            key=str(gate["key"]),
            label=str(gate["label"]),
            operator=str(gate["operator"]),
            target=int(gate["target"]),
            observed=int(metrics.get(str(gate["key"])) or 0),
        )
        for gate in PILOT_EXIT_GATES
    ]


def _build_next_actions(criteria: Iterable[Mapping[str, Any]]) -> list[str]:
    actions: list[str] = []
    for criterion in criteria:
        if criterion.get("status") == "pass":
            continue
        key = criterion.get("key")
        target = int(criterion.get("target") or 0)
        observed = int(criterion.get("observed") or 0)
        missing = max(0, target - observed)
        if key == "discovery_calls":
            actions.append(f"Record {missing} more discovery calls before treating the wedge as proven.")
        elif key == "hands_on_demos":
            actions.append(f"Run {missing} more hands-on demos with the four-item demo checklist.")
        elif key == "security_sensitive_users":
            actions.append(f"Find {missing} more security-sensitive users who can react to the local-first constraint.")
        elif key == "exact_pain_recognition":
            actions.append(f"Find {missing} more users who recognize the exact pain.")
        elif key == "wants_to_try":
            actions.append(f"Find {missing} more users who want to try the workflow locally.")
        elif key == "paid_pilot_commitments_or_lois":
            actions.append(
                "Do not build the team registry yet; collect at least two paid-pilot commitments or LOIs first."
            )
    return actions


def _compact_record(record: Mapping[str, Any]) -> dict[str, Any]:
    score = _mapping_dict(record.get("score"))
    return {
        "pilot_record_id": _clean_text(record.get("pilot_record_id")),
        "recorded_at_utc": _clean_text(record.get("recorded_at_utc")),
        "record_type": _clean_text(record.get("record_type")),
        "participant_label": _clean_text(record.get("participant_label")),
        "participant_segment": _clean_text(record.get("participant_segment")),
        "exact_pain_recognized": score.get("exact_pain_recognized"),
        "wants_to_try": score.get("wants_to_try"),
        "security_sensitive_user": score.get("security_sensitive_user"),
        "paid_pilot_commitment": score.get("paid_pilot_commitment"),
    }


def build_pilot_report(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    records_path: Path | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    records = read_pilot_evidence_records(
        workspace_id=workspace_id,
        root=resolved_root,
        records_path=records_path,
    )
    metrics = _metrics_from_records(records)
    criteria = _criteria_from_metrics(metrics)
    blockers = [criterion["key"] for criterion in criteria if criterion.get("status") != "pass"]
    status = "pass" if not blockers else "needs_data"
    paid_commitments = int(metrics.get("paid_pilot_commitments_or_lois") or 0)
    paid_pilot_gate_status = "pass" if paid_commitments >= 2 else "needs_data"
    run_slug = timestamp_slug()
    latest_json = pilot_report_latest_json_path(workspace_id=workspace_id, root=resolved_root)
    latest_md = pilot_report_latest_md_path(workspace_id=workspace_id, root=resolved_root)
    run_json = pilot_report_run_json_path(workspace_id=workspace_id, root=resolved_root, run_slug=run_slug)
    run_md = pilot_report_run_md_path(workspace_id=workspace_id, root=resolved_root, run_slug=run_slug)
    return {
        "schema_name": PILOT_REPORT_SCHEMA_NAME,
        "schema_version": PILOT_REPORT_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "generated_at_utc": timestamp_utc(),
        "status": status,
        "records_source": (
            str(_path_from_text(records_path, root=resolved_root))
            if records_path is not None
            else "local_ledger"
        ),
        "record_count": len(records),
        "metrics": metrics,
        "criteria": criteria,
        "blockers": blockers,
        "paid_pilot_gate": {
            "status": paid_pilot_gate_status,
            "paid_pilot_commitments_or_lois": paid_commitments,
            "target": 2,
            "do_not_build_team_registry": paid_pilot_gate_status != "pass",
            "pivot_options": [
                "static audit reports",
                "CI evidence",
                "agent transcript firewall",
            ],
        },
        "next_actions": _build_next_actions(criteria),
        "recent_records": [_compact_record(record) for record in records[-10:]],
        "templates": dict(PILOT_TEMPLATE_FILES),
        "guardrails": {
            "local_first": True,
            "file_first": True,
            "requires_api_key": False,
            "uses_network": False,
            "writes_training_data": False,
            "uses_private_docs": False,
            **ANTI_DRIFT_GUARDRAILS,
        },
        "paths": {
            "pilot_evidence_ledger_path": str(
                pilot_evidence_ledger_path(workspace_id=workspace_id, root=resolved_root)
            ),
            "report_latest_json_path": str(latest_json),
            "report_latest_md_path": str(latest_md),
            "report_run_json_path": str(run_json),
            "report_run_md_path": str(run_md),
        },
    }


def _format_observed(value: Any) -> str:
    if value is None:
        return "not recorded"
    return str(value)


def format_pilot_report_markdown(report: Mapping[str, Any]) -> str:
    gate = _mapping_dict(report.get("paid_pilot_gate"))
    lines = [
        "# Paid-Pilot Evidence Gate",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Paid-pilot gate: `{gate.get('status')}`",
        f"- Records source: `{report.get('records_source')}`",
        f"- Generated: `{report.get('generated_at_utc')}`",
        "",
        "| Criterion | Target | Observed | Status |",
        "|---|---:|---:|---|",
    ]
    for criterion in report.get("criteria") or []:
        if not isinstance(criterion, Mapping):
            continue
        target = f"{criterion.get('operator')} {criterion.get('target')}"
        lines.append(
            f"| {criterion.get('label')} | {target} | {_format_observed(criterion.get('observed'))} | "
            f"`{criterion.get('status')}` |"
        )
    actions = [str(action) for action in report.get("next_actions") or []]
    if actions:
        lines.extend(["", "## Next Actions", ""])
        lines.extend(f"- {action}" for action in actions)
    if gate.get("do_not_build_team_registry"):
        lines.extend(
            [
                "",
                "## Pivot Boundary",
                "",
                "Do not build the team registry until paid-pilot commitments or LOIs reach the gate.",
                (
                    "Refine the wedge around static audit reports, CI evidence, "
                    "or agent transcript firewall if the count stalls."
                ),
            ]
        )
    recent = [record for record in report.get("recent_records") or [] if isinstance(record, Mapping)]
    if recent:
        lines.extend(
            [
                "",
                "## Recent Records",
                "",
                "| Type | Participant | Segment | Pain | Try | Commitment |",
                "|---|---|---|---|---|---|",
            ]
        )
        for record in recent:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _clean_text(record.get("record_type")) or "",
                        _clean_text(record.get("participant_label")) or "",
                        _clean_text(record.get("participant_segment")) or "",
                        _format_observed(record.get("exact_pain_recognized")),
                        _format_observed(record.get("wants_to_try")),
                        _format_observed(record.get("paid_pilot_commitment")),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Local files are the source of truth.",
            "- No API key or network call is used by this report.",
            (
                "- No raw code upload, training export, default telemetry, hidden agent execution, "
                "or remote-only positive support is enabled."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def record_pilot_report(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    records_path: Path | None = None,
) -> tuple[dict[str, Any], str, Path, Path, Path, Path]:
    report = build_pilot_report(workspace_id=workspace_id, root=root, records_path=records_path)
    markdown = format_pilot_report_markdown(report)
    paths = _mapping_dict(report.get("paths"))
    latest_json = Path(str(paths["report_latest_json_path"]))
    latest_md = Path(str(paths["report_latest_md_path"]))
    run_json = Path(str(paths["report_run_json_path"]))
    run_md = Path(str(paths["report_run_md_path"]))
    write_json(latest_json, report)
    write_json(run_json, report)
    latest_md.parent.mkdir(parents=True, exist_ok=True)
    run_md.parent.mkdir(parents=True, exist_ok=True)
    latest_md.write_text(markdown, encoding="utf-8")
    run_md.write_text(markdown, encoding="utf-8")
    return report, markdown, latest_json, latest_md, run_json, run_md


def pilot_templates_markdown() -> str:
    return "\n".join(
        [
            "# Pilot Evidence Templates",
            "",
            "## Interview Script",
            "",
            "- Participant label:",
            "- Segment:",
            "- Exact pain recognized? yes/no/unclear",
            "- Wants to try locally? yes/no/unclear",
            "- Willing to install locally? yes/no/unclear",
            "- Willingness to pay? yes/no/unclear",
            "- Security-sensitive user? yes/no/unclear",
            "- No-code-upload matters? yes/no/unclear",
            "- Budget path? yes/no/unclear",
            "- Current workaround:",
            "- Failure example they are allowed to describe:",
            "- Objections:",
            "",
            "## Demo Checklist",
            "",
            "- Failure memory reviewed: yes/no",
            "- Signed evidence reviewed: yes/no",
            "- No-code-upload boundary confirmed: yes/no",
            "- Transcript claim rejection reviewed: yes/no",
            "- Wants to try after demo? yes/no/unclear",
            "",
            "## Statement of Value",
            "",
            "- Offer: Local-first AI coding evidence audit and failure-memory starter pack.",
            "- Scope: 2-4 weeks, one repo or workflow, no source upload.",
            "- Success: first audit report, one repeated-failure pattern, offline verification.",
            "",
            "## CLI Wiring",
            "",
            "```bash",
            "python scripts/satlab.py pilot record-interview --help",
            "python scripts/satlab.py pilot record-demo --help",
            "python scripts/satlab.py pilot record-loi --help",
            "python scripts/satlab.py pilot report --format md",
            "```",
        ]
    )


def write_pilot_templates(output_dir: Path, *, root: Path | None = None) -> dict[str, str]:
    resolved_root = _resolve_root(root)
    destination = _path_from_text(output_dir, root=resolved_root)
    paths: dict[str, str] = {}
    for key, relative in PILOT_TEMPLATE_FILES.items():
        source = resolved_root / relative
        if not source.is_file():
            raise ValueError(f"Pilot template is missing: `{relative}`.")
        target = destination / Path(relative).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        paths[key] = str(target)
    return paths
