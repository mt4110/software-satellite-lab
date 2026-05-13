#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from artifact_vault import (
    artifact_ref_object_verified,
    redact_report_excerpt,
    resolve_vault_object_path,
)
from gemma_runtime import repo_root, timestamp_utc
from software_work_events import build_event_record
from workspace_state import DEFAULT_WORKSPACE_ID


AGENT_CLAIM_SCHEMA_NAME = "software-satellite-agent-claim"
AGENT_CLAIM_SCHEMA_VERSION = 1

CLAIM_KINDS = {
    "tests_passed",
    "tests_failed",
    "file_modified",
    "command_run",
    "bug_fixed",
    "reviewer_requested_change",
    "issue_resolved",
    "agent_statement",
}
CLAIM_VERIFICATION_STATES = {"unverified_agent_claim", "verified_signal"}
VERIFICATION_EVIDENCE_KINDS = {
    "command_log",
    "ci_log",
    "test_log",
    "human_verdict",
    "source_comparison",
}

SUPPORT_POLARITY_BY_KIND = {
    "tests_passed": "positive",
    "tests_failed": "risk",
    "file_modified": "diagnostic",
    "command_run": "diagnostic",
    "bug_fixed": "positive",
    "reviewer_requested_change": "risk",
    "issue_resolved": "positive",
    "agent_statement": "none",
}
EVIDENCE_TYPE_BY_KIND = {
    "tests_passed": "test_pass",
    "tests_failed": "test_fail",
    "bug_fixed": "verification_pass",
    "reviewer_requested_change": "review_unresolved",
    "issue_resolved": "human_acceptance",
}

DEFAULT_MAX_CLAIMS = 50
DEFAULT_CLAIM_EXCERPT_CHARS = 360
DEFAULT_VERIFICATION_READ_CHARS = 256 * 1024

PATH_RE = r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+"
TEST_FAIL_RE = re.compile(
    r"(?:\btests?\s+failed\b|\bfailed\s+tests?\b|\btest\s+failure\b|"
    r"\btest\s+errors?\b|\b[1-9]\d*\s+(?:failed|failures?|errors?)\b|"
    r"\bFAILED\b|\bFAILURES?\b|\bnot\s+ok\b|\berror:)",
    re.IGNORECASE,
)
TEST_PASS_RE = re.compile(
    r"(?:\b(?:tests?\s+passed|all\s+tests\s+passed|"
    r"\d+\s+passed|PASS(?:ED)?|success(?:ful)?)\b|"
    r"\bOK\b(?:\s*\([^)]*\))?)",
    re.IGNORECASE,
)
ZERO_FAILURE_COUNT_RE = re.compile(r"\b0\s+(?:failed|failures?|errors?)\b", re.IGNORECASE)
COMMAND_PREFIX_RE = re.compile(
    r"^\s*(?:[$>]|\+\s*command:|command(?:\s+run)?:|ran:)\s*(?P<command>.+)$",
    re.IGNORECASE,
)
COMMAND_MENTION_RE = re.compile(
    r"\b(?:pytest|python3?\s+-m\s+pytest|uv\s+run|npm\s+test|"
    r"pnpm\s+test|yarn\s+test|go\s+test|cargo\s+test|make\s+test)\b",
    re.IGNORECASE,
)
FILE_MODIFIED_RE = re.compile(
    rf"\b(?:modified|changed|updated|edited|touched|wrote)\b.*?\b({PATH_RE})\b",
    re.IGNORECASE,
)
BUG_FIXED_RE = re.compile(
    r"\b(?:bug\s+fixed|fixed\s+(?:the\s+)?(?:bug|regression|issue)|"
    r"resolved\s+(?:the\s+)?(?:bug|regression))\b",
    re.IGNORECASE,
)
REVIEWER_REQUEST_RE = re.compile(
    r"\b(?:reviewer\s+requested|requested\s+changes|changes\s+requested|needs\s+changes)\b",
    re.IGNORECASE,
)
ISSUE_RESOLVED_RE = re.compile(
    r"\b(?:issue\s+resolved|resolved\s+#\d+|closes\s+#\d+|fixes\s+#\d+)\b",
    re.IGNORECASE,
)


def _text_with_zero_failure_counts_removed(text: str) -> str:
    return ZERO_FAILURE_COUNT_RE.sub("", text)


def _has_test_failure_signal(text: str) -> bool:
    return bool(TEST_FAIL_RE.search(_text_with_zero_failure_counts_removed(text)))


def _has_test_pass_signal(text: str) -> bool:
    return bool(TEST_PASS_RE.search(text)) and not _has_test_failure_signal(text)


def _normalized_command_text(text: str) -> str | None:
    line = text.strip()
    if not line:
        return None
    match = COMMAND_PREFIX_RE.search(line)
    if match:
        line = match.group("command")
    elif not COMMAND_MENTION_RE.search(line):
        return None
    normalized = " ".join(line.strip().split()).lower()
    return normalized or None


def _command_log_verifies_claim(claim_text: str, log_text: str) -> bool:
    claimed = _normalized_command_text(claim_text)
    if claimed is None:
        return False
    log_commands = {
        normalized
        for line in log_text.splitlines()
        if (normalized := _normalized_command_text(line)) is not None
    }
    return claimed in log_commands


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _mapping_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _stable_digest(value: Any, *, length: int = 16) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _claim_id(*, claim_kind: str, claim: str, source: str, source_line: int | None) -> str:
    return f"agent_claim_{_stable_digest([claim_kind, claim, source, source_line], length=20)}"


def _redacted_excerpt(text: str, *, max_chars: int = DEFAULT_CLAIM_EXCERPT_CHARS) -> tuple[str, dict[str, Any]]:
    excerpt, redaction = redact_report_excerpt(text, max_chars=max_chars)
    return excerpt, redaction


def infer_claim_kind(text: str | None) -> str:
    value = _clean_text(text) or ""
    if _has_test_failure_signal(value):
        return "tests_failed"
    if _has_test_pass_signal(value):
        return "tests_passed"
    if COMMAND_PREFIX_RE.search(value) or COMMAND_MENTION_RE.search(value):
        return "command_run"
    if FILE_MODIFIED_RE.search(value):
        return "file_modified"
    if REVIEWER_REQUEST_RE.search(value):
        return "reviewer_requested_change"
    if ISSUE_RESOLVED_RE.search(value):
        return "issue_resolved"
    if BUG_FIXED_RE.search(value):
        return "bug_fixed"
    return "agent_statement"


def _claim_from_text(
    *,
    claim_kind: str,
    text: str,
    source: str,
    source_line: int | None,
    source_artifact_id: str | None,
    max_excerpt_chars: int,
) -> dict[str, Any]:
    excerpt, redaction = _redacted_excerpt(text, max_chars=max_excerpt_chars)
    claim_id = _claim_id(
        claim_kind=claim_kind,
        claim=excerpt,
        source=source,
        source_line=source_line,
    )
    return {
        "schema_name": AGENT_CLAIM_SCHEMA_NAME,
        "schema_version": AGENT_CLAIM_SCHEMA_VERSION,
        "claim_id": claim_id,
        "claim_kind": claim_kind,
        "claim": excerpt,
        "source": source,
        "source_line": source_line,
        "source_artifact_id": source_artifact_id,
        "verification_state": "unverified_agent_claim",
        "support_polarity": SUPPORT_POLARITY_BY_KIND.get(claim_kind, "none"),
        "verification_evidence": [],
        "redaction": redaction,
    }


def _line_claim_kinds(line: str) -> list[str]:
    kinds: list[str] = []
    if _has_test_failure_signal(line):
        kinds.append("tests_failed")
    elif _has_test_pass_signal(line):
        kinds.append("tests_passed")

    if COMMAND_PREFIX_RE.search(line) or COMMAND_MENTION_RE.search(line):
        kinds.append("command_run")
    if FILE_MODIFIED_RE.search(line):
        kinds.append("file_modified")
    if REVIEWER_REQUEST_RE.search(line):
        kinds.append("reviewer_requested_change")
    if ISSUE_RESOLVED_RE.search(line):
        kinds.append("issue_resolved")
    if BUG_FIXED_RE.search(line):
        kinds.append("bug_fixed")
    return kinds


def dedupe_claims(claims: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for claim in claims:
        claim_kind = _clean_text(claim.get("claim_kind")) or "agent_statement"
        text = _clean_text(claim.get("claim")) or ""
        source = _clean_text(claim.get("source")) or "unknown"
        key = (claim_kind, source, text.lower())
        if not text or key in seen:
            continue
        seen.add(key)
        deduped.append(copy.deepcopy(dict(claim)))
    return deduped


def extract_claims_from_transcript(
    transcript_text: str,
    *,
    source: str = "transcript",
    source_artifact_id: str | None = None,
    max_claims: int = DEFAULT_MAX_CLAIMS,
    max_excerpt_chars: int = DEFAULT_CLAIM_EXCERPT_CHARS,
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(transcript_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        for claim_kind in _line_claim_kinds(line):
            claims.append(
                _claim_from_text(
                    claim_kind=claim_kind,
                    text=line,
                    source=source,
                    source_line=line_number,
                    source_artifact_id=source_artifact_id,
                    max_excerpt_chars=max_excerpt_chars,
                )
            )
            if len(claims) >= max_claims:
                return dedupe_claims(claims)
    return dedupe_claims(claims)


def normalize_declared_claims(
    declared_claims: Any,
    *,
    default_source: str = "declared_claim",
    max_excerpt_chars: int = DEFAULT_CLAIM_EXCERPT_CHARS,
) -> list[dict[str, Any]]:
    if not isinstance(declared_claims, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in declared_claims:
        if isinstance(item, str):
            raw = {"claim": item}
        elif isinstance(item, Mapping):
            raw = dict(item)
        else:
            continue

        claim_text = _clean_text(raw.get("claim"))
        if claim_text is None:
            continue
        requested_kind = (_clean_text(raw.get("claim_kind")) or "").lower().replace("-", "_")
        claim_kind = requested_kind if requested_kind in CLAIM_KINDS else infer_claim_kind(claim_text)
        source = _clean_text(raw.get("source")) or default_source
        claim = _claim_from_text(
            claim_kind=claim_kind,
            text=claim_text,
            source=source,
            source_line=None,
            source_artifact_id=_clean_text(raw.get("source_artifact_id")),
            max_excerpt_chars=max_excerpt_chars,
        )
        declared_verification = _clean_text(raw.get("verification"))
        if declared_verification is not None:
            claim["declared_verification"] = declared_verification
        normalized.append(claim)
    return dedupe_claims(normalized)


def _read_verified_artifact_text(
    ref: Mapping[str, Any],
    *,
    root: Path,
    max_chars: int = DEFAULT_VERIFICATION_READ_CHARS,
) -> str | None:
    verified, _reason = artifact_ref_object_verified(ref, root=root)
    if not verified:
        return None
    object_path = resolve_vault_object_path(ref, root=root)
    if object_path is None or not object_path.is_file():
        return None
    try:
        with object_path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(max_chars)
    except OSError:
        return None


def log_indicates_tests_failed(text: str) -> bool:
    return _has_test_failure_signal(text)


def log_indicates_tests_passed(text: str) -> bool:
    return _has_test_pass_signal(text)


def _verification_reason_for_claim(claim_kind: str, claim_text: str, text: str) -> str | None:
    if claim_kind == "tests_passed" and log_indicates_tests_passed(text):
        return "test_log_pass_signal"
    if claim_kind == "tests_failed" and log_indicates_tests_failed(text):
        return "test_log_fail_signal"
    if claim_kind == "command_run" and _command_log_verifies_claim(claim_text, text):
        return "command_log_contains_claimed_command"
    if claim_kind == "reviewer_requested_change" and REVIEWER_REQUEST_RE.search(text):
        return "human_review_request_signal"
    if claim_kind == "issue_resolved" and ISSUE_RESOLVED_RE.search(text):
        return "human_review_resolution_signal"
    return None


def verify_claims_against_artifacts(
    claims: Iterable[Mapping[str, Any]],
    artifact_refs: Iterable[Mapping[str, Any]],
    *,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    resolved_root = _resolve_root(root)
    refs = [dict(ref) for ref in artifact_refs if isinstance(ref, Mapping)]
    verification_refs: list[tuple[dict[str, Any], str]] = []
    for ref in refs:
        kind = (_clean_text(ref.get("kind")) or "unknown").lower()
        if kind not in {"test_log", "ci_log", "command_log", "review_note"}:
            continue
        text = _read_verified_artifact_text(ref, root=resolved_root)
        if text is not None:
            verification_refs.append((ref, text))

    verified_claims: list[dict[str, Any]] = []
    for claim in claims:
        updated = copy.deepcopy(dict(claim))
        claim_kind = _clean_text(updated.get("claim_kind")) or "agent_statement"
        claim_text = _clean_text(updated.get("claim")) or ""
        for ref, text in verification_refs:
            reason = _verification_reason_for_claim(claim_kind, claim_text, text)
            if reason is None:
                continue
            updated["verification_state"] = "verified_signal"
            updated.setdefault("verification_evidence", [])
            updated["verification_evidence"].append(
                {
                    "artifact_id": _clean_text(ref.get("artifact_id")),
                    "kind": _clean_text(ref.get("kind")) or "unknown",
                    "reason": reason,
                }
            )
        verified_claims.append(updated)
    return verified_claims


def claim_evidence_types(claims: Iterable[Mapping[str, Any]]) -> list[str]:
    evidence_types: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        if _clean_text(claim.get("verification_state")) != "verified_signal":
            continue
        claim_kind = _clean_text(claim.get("claim_kind")) or "agent_statement"
        evidence_type = EVIDENCE_TYPE_BY_KIND.get(claim_kind)
        if evidence_type and evidence_type not in seen:
            seen.add(evidence_type)
            evidence_types.append(evidence_type)
    return evidence_types


def claim_quality_status(claims: Iterable[Mapping[str, Any]]) -> str | None:
    verified_kinds = {
        _clean_text(claim.get("claim_kind"))
        for claim in claims
        if _clean_text(claim.get("verification_state")) == "verified_signal"
    }
    if "tests_failed" in verified_kinds or "reviewer_requested_change" in verified_kinds:
        return "fail"
    if verified_kinds & {"tests_passed", "bug_fixed", "issue_resolved"}:
        return "pass"
    return None


def claim_counts(claims: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "total": 0,
        "verified_signal": 0,
        "unverified_agent_claim": 0,
    }
    for claim in claims:
        counts["total"] += 1
        state = _clean_text(claim.get("verification_state")) or "unverified_agent_claim"
        if state == "verified_signal":
            counts["verified_signal"] += 1
        else:
            counts["unverified_agent_claim"] += 1
    return counts


def build_claim_software_work_event(
    claim: Mapping[str, Any],
    *,
    artifact_refs: Iterable[Mapping[str, Any]],
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    session_id: str = "agent-session-intake",
    recorded_at_utc: str | None = None,
    agent_label: str = "unknown",
) -> dict[str, Any]:
    claim_payload = copy.deepcopy(dict(claim))
    refs = [copy.deepcopy(dict(ref)) for ref in artifact_refs if isinstance(ref, Mapping)]
    verification_state = _clean_text(claim_payload.get("verification_state")) or "unverified_agent_claim"
    claim_kind = _clean_text(claim_payload.get("claim_kind")) or "agent_statement"
    quality_status = claim_quality_status([claim_payload])
    evidence_types = claim_evidence_types([claim_payload])
    options: dict[str, Any] = {
        "workflow": "agent_session_intake",
        "agent_label": agent_label,
        "claim_id": _clean_text(claim_payload.get("claim_id")),
        "claim_kind": claim_kind,
        "verification_state": verification_state,
        "artifact_vault_refs": refs,
    }
    if quality_status is not None:
        options["quality_status"] = quality_status
    if evidence_types:
        options["evidence_types"] = evidence_types
        options["quality_checks"] = [
            {
                "name": "agent_claim_linked_to_verification_artifact",
                "pass": True,
                "detail": verification_state,
            }
        ]

    status = "verified" if verification_state == "verified_signal" else "needs_review"
    return build_event_record(
        event_id=f"{workspace_id}:{session_id}:{_clean_text(claim_payload.get('claim_id')) or 'agent-claim'}",
        event_kind="agent_transcript_claim",
        recorded_at_utc=recorded_at_utc or timestamp_utc(),
        workspace={"workspace_id": workspace_id},
        session={
            "session_id": session_id,
            "surface": "chat",
            "mode": "agent_session_intake",
            "title": "Agent session intake",
            "selected_model_id": None,
            "session_manifest_path": None,
        },
        outcome={
            "status": status,
            "quality_status": quality_status,
            "execution_status": status,
        },
        content={
            "prompt": _clean_text(claim_payload.get("claim")),
            "system_prompt": None,
            "resolved_user_prompt": None,
            "output_text": _clean_text(claim_payload.get("claim")),
            "notes": ["agent_claim", "agent_transcript_claim", verification_state],
            "options": options,
        },
        source_refs={"artifact_vault_refs": refs},
        tags=["agent_claim", "agent_transcript_claim", claim_kind, verification_state, agent_label],
    )
