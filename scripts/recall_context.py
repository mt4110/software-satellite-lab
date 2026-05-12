#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from memory_index import MemoryIndex, default_memory_index_path, rebuild_memory_index
from software_work_events import build_event_contract_check
from workspace_state import DEFAULT_WORKSPACE_ID


TASK_KINDS = ("review", "design", "proposal", "failure_analysis")
DEFAULT_LIMIT = 12
DEFAULT_CONTEXT_BUDGET_CHARS = 6000
CONTEXT_BUNDLE_VERSION = 7
PASS_DEFINITION_CONTEXT_BUDGET_MULTIPLIER = 1.5
FAILURE_STATUSES = {"quality_fail", "failed", "blocked", "error"}
ACCEPTED_NOTE_KEYWORDS = ("accept", "accepted", "approved", "decision", "pass", "passed")
REJECTED_NOTE_KEYWORDS = ("reject", "rejected", "declined", "not accepted", "discarded")
REPAIR_NOTE_KEYWORDS = ("repair", "fixed", "fix", "follow-up", "followup", "resolved")
OPEN_RISK_KEYWORDS = ("risk", "regression", "blocker", "blocked", "fail", "failed", "error")
QUERY_TERM_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "then",
    "this",
    "to",
    "use",
    "what",
    "when",
    "with",
}
TASK_KEYWORDS = {
    "review": ("review", "regression", "patch", "comment", "risk"),
    "design": ("design", "architecture", "tradeoff", "decision", "boundary", "module"),
    "proposal": ("proposal", "plan", "implement", "implementation", "rollout", "test"),
    "failure_analysis": ("fail", "failure", "blocked", "repair", "error", "bug", "incident"),
}
TASK_EVIDENCE_PRIORITY = {
    "review": {
        "source-artifact": 2.0,
        "accepted": 3.0,
        "test_fail": 2.5,
        "repair": 2.0,
        "rejected": 0.75,
    },
    "design": {
        "source-artifact": 2.0,
        "accepted": 3.5,
        "rejected": 2.5,
        "repair": 1.0,
        "test_fail": 0.75,
    },
    "proposal": {
        "source-artifact": 2.0,
        "accepted": 3.0,
        "test_pass": 2.5,
        "repair": 2.0,
        "rejected": 1.0,
    },
    "failure_analysis": {
        "source-artifact": 1.5,
        "test_fail": 4.5,
        "repair": 3.0,
        "rejected": 2.0,
        "accepted": 0.75,
    },
}
SOURCE_CONTRACT_MISS_REASON = "source_event_contract_broken"
EVIDENCE_TYPE_MISMATCH_MISS_REASON = "evidence_type_mismatch"
BLOCK_RELEVANT = "Relevant prior prompts"
BLOCK_ACCEPTED = "Accepted outcomes"
BLOCK_FAILURE = "Failure and repair patterns"
BLOCK_FILES = "Related files and artifact paths"
BLOCK_RISKS = "Open risks"
TASK_BLOCK_ORDER = {
    "review": (BLOCK_FILES, BLOCK_ACCEPTED, BLOCK_FAILURE, BLOCK_RELEVANT, BLOCK_RISKS),
    "design": (BLOCK_ACCEPTED, BLOCK_RELEVANT, BLOCK_FAILURE, BLOCK_FILES, BLOCK_RISKS),
    "proposal": (BLOCK_RELEVANT, BLOCK_ACCEPTED, BLOCK_FAILURE, BLOCK_FILES, BLOCK_RISKS),
    "failure_analysis": (BLOCK_FAILURE, BLOCK_ACCEPTED, BLOCK_FILES, BLOCK_RELEVANT, BLOCK_RISKS),
}
BLOCK_BUDGET_RATIO = {
    "review": {
        BLOCK_FILES: 0.30,
        BLOCK_ACCEPTED: 0.24,
        BLOCK_FAILURE: 0.18,
        BLOCK_RELEVANT: 0.18,
        BLOCK_RISKS: 0.10,
    },
    "design": {
        BLOCK_ACCEPTED: 0.30,
        BLOCK_RELEVANT: 0.24,
        BLOCK_FAILURE: 0.18,
        BLOCK_FILES: 0.16,
        BLOCK_RISKS: 0.12,
    },
    "proposal": {
        BLOCK_RELEVANT: 0.28,
        BLOCK_ACCEPTED: 0.24,
        BLOCK_FAILURE: 0.18,
        BLOCK_FILES: 0.16,
        BLOCK_RISKS: 0.14,
    },
    "failure_analysis": {
        BLOCK_FAILURE: 0.40,
        BLOCK_ACCEPTED: 0.18,
        BLOCK_FILES: 0.16,
        BLOCK_RELEVANT: 0.14,
        BLOCK_RISKS: 0.12,
    },
}


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _clean_string_list(value: Any, *, lowercase: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = _clean_text(item)
        if not normalized:
            continue
        if lowercase:
            normalized = normalized.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return tuple(cleaned)


def _coerce_positive_int(value: Any, *, default: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return default
    return normalized if normalized > 0 else default


def _coerce_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _join_text(parts: Iterable[str | None]) -> str:
    cleaned = [_clean_text(part) for part in parts if _clean_text(part)]
    return "\n".join(cleaned)


def _lower_text(parts: Iterable[str | None]) -> str:
    return _join_text(parts).lower()


def _pathish_tokens(value: str) -> list[str]:
    raw_tokens = re.split(r"[^a-z0-9_]+", value.lower())
    tokens: list[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        if len(token) < 2:
            continue
        for part in (token, *[piece for piece in token.split("_") if len(piece) >= 2]):
            if part not in seen:
                seen.add(part)
                tokens.append(part)
    return tokens


def _fts_query_from_text(value: str) -> str | None:
    tokens = _pathish_tokens(value)
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens[:12])


def _query_head_anchor(value: str) -> str | None:
    if " | " not in value:
        return None
    head = _clean_text(value.split(" | ", 1)[0]).lower()
    if len(head) < 3:
        return None
    return head


def _validation_only_anchor(value: str) -> str | None:
    match = re.search(r"--only\s+([a-z0-9_-]+)", value.lower())
    if match is None:
        return None
    return match.group(1)


def _candidate_query_head_anchor(candidate: "RecallCandidate") -> str | None:
    return _query_head_anchor(candidate.prompt or "")


def _candidate_validation_only_anchor(candidate: "RecallCandidate") -> str | None:
    return _validation_only_anchor(candidate.prompt or "")


def _truncate(text: str | None, *, limit: int) -> str:
    cleaned = _clean_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "..."


def _json_char_count(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False))


@dataclass(frozen=True)
class RecallRequest:
    task_kind: str
    query_text: str
    request_basis: str | None = None
    file_hints: tuple[str, ...] = ()
    surface_filters: tuple[str, ...] = ()
    status_filters: tuple[str, ...] = ()
    pinned_event_ids: tuple[str, ...] = ()
    exclude_event_ids: tuple[str, ...] = ()
    recorded_before_utc: str | None = None
    recorded_before_event_id: str | None = None
    limit: int = DEFAULT_LIMIT
    context_budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS
    source_event_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "task_kind": self.task_kind,
            "query_text": self.query_text,
            "request_basis": self.request_basis,
            "file_hints": list(self.file_hints),
            "surface_filters": list(self.surface_filters),
            "status_filters": list(self.status_filters),
            "pinned_event_ids": list(self.pinned_event_ids),
            "exclude_event_ids": list(self.exclude_event_ids),
            "recorded_before_utc": self.recorded_before_utc,
            "recorded_before_event_id": self.recorded_before_event_id,
            "limit": self.limit,
            "context_budget_chars": self.context_budget_chars,
        }
        if self.source_event_id:
            payload["source_event_id"] = self.source_event_id
        return payload


@dataclass
class RecallCandidate:
    event_id: str
    recorded_at_utc: str | None
    session_id: str | None
    session_surface: str | None
    session_mode: str | None
    model_id: str | None
    event_kind: str | None
    status: str | None
    prompt: str | None
    output_text: str | None
    notes_text: str | None
    pass_definition: str | None
    quality_status: str | None = None
    execution_status: str | None = None
    evaluation_signal_text: str | None = None
    artifact_path: str | None = None
    payload_json: str | None = None
    raw_fts_score: float | None = None
    best_rank: int = 10_000
    query_hits: int = 0
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    block_title: str | None = None
    evidence_types: tuple[str, ...] = ()
    evidence_priority: dict[str, Any] = field(default_factory=dict)
    event_contract_status: str | None = None
    source_artifact_status: str | None = None
    source_artifact_reasons: tuple[str, ...] = ()
    source_artifact_durability_status: str | None = None
    source_artifact_readability_status: str | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any], *, rank: int) -> "RecallCandidate":
        raw_score = row.get("score")
        parsed_score: float | None = None
        if raw_score is not None:
            try:
                parsed_score = float(raw_score)
            except (TypeError, ValueError):
                parsed_score = None
        return cls(
            event_id=str(row.get("event_id") or ""),
            recorded_at_utc=row.get("recorded_at_utc"),
            session_id=row.get("session_id"),
            session_surface=row.get("session_surface"),
            session_mode=row.get("session_mode"),
            model_id=row.get("model_id"),
            event_kind=row.get("event_kind"),
            status=row.get("status"),
            prompt=row.get("prompt"),
            output_text=row.get("output_text"),
            notes_text=row.get("notes_text"),
            pass_definition=row.get("pass_definition"),
            quality_status=row.get("quality_status"),
            execution_status=row.get("execution_status"),
            evaluation_signal_text=row.get("evaluation_signal_text"),
            artifact_path=row.get("artifact_path"),
            payload_json=row.get("payload_json"),
            raw_fts_score=parsed_score,
            best_rank=rank,
            query_hits=1 if parsed_score is not None else 0,
        )

    def merged_with(self, row: Mapping[str, Any], *, rank: int) -> None:
        self.best_rank = min(self.best_rank, rank)
        raw_score = row.get("score")
        try:
            parsed_score = float(raw_score) if raw_score is not None else None
        except (TypeError, ValueError):
            parsed_score = None
        if parsed_score is not None:
            self.query_hits += 1
        if parsed_score is not None and (self.raw_fts_score is None or parsed_score < self.raw_fts_score):
            self.raw_fts_score = parsed_score
        if self.pass_definition is None and isinstance(row.get("pass_definition"), str):
            self.pass_definition = row.get("pass_definition")
        if self.payload_json is None and isinstance(row.get("payload_json"), str):
            self.payload_json = row.get("payload_json")

    def combined_text(self) -> str:
        return _lower_text(
            (
                self.prompt,
                self.output_text,
                self.notes_text,
                self.pass_definition,
                self.quality_status,
                self.execution_status,
                self.evaluation_signal_text,
                self.artifact_path,
                self.event_kind,
                self.session_mode,
                self.session_surface,
                self.status,
            )
        )

    def accepted_like(self) -> bool:
        if self.rejected_like():
            return False
        text = self.combined_text()
        if (self.status or "").lower() == "ok":
            return True
        return any(keyword in text for keyword in ACCEPTED_NOTE_KEYWORDS)

    def rejected_like(self) -> bool:
        text = self.combined_text()
        if (self.status or "").lower() == "rejected":
            return True
        return any(keyword in text for keyword in REJECTED_NOTE_KEYWORDS)

    def failure_like(self) -> bool:
        status = (self.status or "").lower()
        if status in FAILURE_STATUSES:
            return True
        text = self.combined_text()
        return any(keyword in text for keyword in ("repair", "blocked", "fail", "error", "regression", "bug"))

    def test_fail_like(self) -> bool:
        status = (self.status or "").lower()
        quality_status = (self.quality_status or "").lower()
        execution_status = (self.execution_status or "").lower()
        if status in FAILURE_STATUSES or quality_status == "fail" or execution_status in FAILURE_STATUSES:
            return True
        text = self.combined_text()
        return any(keyword in text for keyword in ("test_fail", "test fail", "failed test", "verification failed"))

    def test_pass_like(self) -> bool:
        quality_status = (self.quality_status or "").lower()
        if quality_status == "pass":
            return True
        text = self.combined_text()
        return any(keyword in text for keyword in ("test_pass", "test pass", "verification passed", "passed test"))

    def repair_like(self) -> bool:
        text = self.combined_text()
        return any(keyword in text for keyword in REPAIR_NOTE_KEYWORDS)

    def open_risk_like(self) -> bool:
        text = self.combined_text()
        if (self.status or "").lower() in FAILURE_STATUSES and not self.repair_like():
            return True
        return any(keyword in text for keyword in OPEN_RISK_KEYWORDS) and not self.repair_like()

    def prompt_excerpt(self) -> str:
        return _truncate(self.prompt or self.pass_definition, limit=120)

    def source_artifact_recallable(self) -> bool:
        if self.event_contract_status is None:
            return bool(_clean_text(self.artifact_path))
        return self.event_contract_status == "ok"

    def source_contract_broken(self) -> bool:
        return self.event_contract_status in {"missing_source", "invalid_event_contract"}

    def to_reference_dict(self) -> dict[str, Any]:
        payload = {
            "event_id": self.event_id,
            "score": round(self.score, 3),
            "reasons": list(self.reasons),
            "block_title": self.block_title,
            "status": self.status,
            "quality_status": self.quality_status,
            "execution_status": self.execution_status,
            "recorded_at_utc": self.recorded_at_utc,
            "session_id": self.session_id,
            "session_surface": self.session_surface,
            "session_mode": self.session_mode,
            "event_kind": self.event_kind,
            "artifact_path": self.artifact_path,
            "prompt_excerpt": self.prompt_excerpt(),
            "evidence_types": list(self.evidence_types),
            "evidence_priority": dict(self.evidence_priority),
        }
        if self.event_contract_status is not None:
            payload.update(
                {
                    "event_contract_status": self.event_contract_status,
                    "source_artifact_status": self.source_artifact_status,
                    "source_artifact_reasons": list(self.source_artifact_reasons),
                    "source_artifact_durability_status": self.source_artifact_durability_status,
                    "source_artifact_readability_status": self.source_artifact_readability_status,
                }
            )
        return payload


def _candidate_label(candidate: RecallCandidate) -> str:
    head = _query_head_anchor(candidate.prompt or "")
    if head:
        return head
    return _truncate(candidate.prompt or candidate.pass_definition, limit=48) or candidate.event_id


def _dedupe_reasons(reasons: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        normalized = _clean_text(reason)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _candidate_event_payload(candidate: RecallCandidate) -> dict[str, Any] | None:
    payload_json = _clean_text(candidate.payload_json)
    if not payload_json:
        return None
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _annotate_source_contract(
    candidates: Iterable[RecallCandidate],
    *,
    root: Path | None,
) -> None:
    for candidate in candidates:
        payload = _candidate_event_payload(candidate)
        if payload is None:
            continue
        check = build_event_contract_check(payload, root=root)
        source_artifact = dict(check.get("source_artifact") or {})
        candidate.event_contract_status = _clean_text(check.get("contract_status")) or None
        candidate.source_artifact_status = _clean_text(source_artifact.get("source_status")) or None
        candidate.source_artifact_reasons = tuple(_clean_string_list(source_artifact.get("reasons")))
        candidate.source_artifact_durability_status = _clean_text(source_artifact.get("durability_status")) or None
        candidate.source_artifact_readability_status = _clean_text(source_artifact.get("readability_status")) or None


def _candidate_evidence_types(candidate: RecallCandidate) -> tuple[str, ...]:
    evidence_types: list[str] = []
    if candidate.source_artifact_recallable():
        evidence_types.append("source-artifact")
    if candidate.accepted_like():
        evidence_types.append("accepted")
    if candidate.rejected_like():
        evidence_types.append("rejected")
    if candidate.test_fail_like():
        evidence_types.append("test_fail")
    if candidate.test_pass_like():
        evidence_types.append("test_pass")
    if candidate.repair_like():
        evidence_types.append("repair")
    return tuple(dict.fromkeys(evidence_types))


def _evidence_priority_for_task(
    request: RecallRequest,
    candidate: RecallCandidate,
) -> tuple[float, dict[str, Any], list[str]]:
    evidence_types = _candidate_evidence_types(candidate)
    candidate.evidence_types = evidence_types
    priority = TASK_EVIDENCE_PRIORITY[request.task_kind]
    matched = [
        {
            "evidence_type": evidence_type,
            "weight": priority[evidence_type],
        }
        for evidence_type in evidence_types
        if evidence_type in priority
    ]
    score = round(sum(float(item["weight"]) for item in matched), 3)
    candidate.evidence_priority = {
        "task_kind": request.task_kind,
        "matched_evidence_types": [str(item["evidence_type"]) for item in matched],
        "matched_task_evidence_types": [
            str(item["evidence_type"])
            for item in matched
            if item["evidence_type"] != "source-artifact"
        ],
        "score": score,
    }
    reasons = [f"priority:{request.task_kind}:{item['evidence_type']}" for item in matched]
    if evidence_types and not matched:
        reasons.append(f"priority:{request.task_kind}:mismatch")
        candidate.evidence_priority["mismatch"] = True
    return score, candidate.evidence_priority, reasons


@dataclass
class RecallSelectionUnit:
    representative: RecallCandidate
    members: tuple[RecallCandidate, ...]
    grouped_by: str | None = None

    @property
    def block_title(self) -> str:
        return self.representative.block_title or BLOCK_RELEVANT

    @property
    def event_id(self) -> str:
        return self.representative.event_id

    @property
    def score(self) -> float:
        return self.representative.score

    @property
    def status(self) -> str | None:
        statuses = {_clean_text(member.status) for member in self.members if _clean_text(member.status)}
        if not statuses:
            return self.representative.status
        if len(statuses) == 1:
            return next(iter(statuses))
        return "mixed"

    @property
    def recorded_at_utc(self) -> str | None:
        return self.representative.recorded_at_utc

    @property
    def session_surface(self) -> str | None:
        return self.representative.session_surface

    @property
    def session_id(self) -> str | None:
        return self.representative.session_id

    @property
    def event_kind(self) -> str | None:
        return self.representative.event_kind

    @property
    def artifact_path(self) -> str | None:
        return self.representative.artifact_path

    @property
    def reasons(self) -> list[str]:
        if self.grouped_by is None:
            return list(self.representative.reasons)
        return _dedupe_reasons(
            [*(reason for member in self.members[:4] for reason in member.reasons), "pass-definition-group"]
        )

    @property
    def member_event_ids(self) -> tuple[str, ...]:
        return tuple(member.event_id for member in self.members)

    def contains_event(self, event_id: str | None) -> bool:
        target = _clean_text(event_id)
        if not target:
            return False
        return any(member.event_id == target for member in self.members)

    def _group_member_labels(self, *, limit: int, label_limit: int = 32) -> list[str]:
        return [_truncate(_candidate_label(member), limit=label_limit) for member in self.members[:limit]]

    def prompt_excerpt(self) -> str:
        if self.grouped_by != "pass_definition" or len(self.members) <= 1:
            return self.representative.prompt_excerpt()
        labels = self._group_member_labels(limit=3, label_limit=24)
        more = len(self.members) - len(labels)
        suffix = f", +{more} more" if more > 0 else ""
        return _truncate(
            f"{len(self.members)} results share pass definition: {', '.join(labels)}{suffix}",
            limit=96,
        )

    def summary(self) -> str:
        if self.grouped_by != "pass_definition" or len(self.members) <= 1:
            return _build_candidate_summary(self.representative, block_title=self.block_title)
        pass_definition = _truncate(self.representative.pass_definition, limit=96)
        labels = self._group_member_labels(limit=4)
        more = len(self.members) - len(labels)
        label_text = ", ".join(labels) + (f", +{more} more" if more > 0 else "")
        return _truncate(
            f"{len(self.members)} capability results share one pass definition. "
            f"Members: {label_text}. Pass: {pass_definition}",
            limit=240,
        )

    def to_reference_dict(self) -> dict[str, Any]:
        payload = self.representative.to_reference_dict()
        payload.update(
            {
                "event_id": self.event_id,
                "score": round(self.score, 3),
                "reasons": self.reasons,
                "block_title": self.block_title,
                "status": self.status,
                "recorded_at_utc": self.recorded_at_utc,
                "session_id": self.session_id,
                "session_surface": self.session_surface,
                "event_kind": self.event_kind,
                "artifact_path": self.artifact_path,
                "prompt_excerpt": self.prompt_excerpt(),
            }
        )
        if self.grouped_by is not None and len(self.members) > 1:
            payload.update(
                {
                    "grouped_by": self.grouped_by,
                    "group_member_count": len(self.members),
                    "group_member_event_ids": list(self.member_event_ids),
                    "group_member_labels": [_candidate_label(member) for member in self.members[:6]],
                }
            )
        return payload

    def to_block_item(self) -> dict[str, Any]:
        payload = {
            "event_id": self.event_id,
            "score": round(self.score, 3),
            "reasons": self.reasons,
            "status": self.status,
            "session_surface": self.session_surface,
            "event_kind": self.event_kind,
            "recorded_at_utc": self.recorded_at_utc,
            "artifact_path": self.artifact_path,
            "evidence_types": list(self.representative.evidence_types),
            "evidence_priority": dict(self.representative.evidence_priority),
            "summary": self.summary(),
        }
        if self.representative.event_contract_status is not None:
            payload.update(
                {
                    "event_contract_status": self.representative.event_contract_status,
                    "source_artifact_status": self.representative.source_artifact_status,
                    "source_artifact_reasons": list(self.representative.source_artifact_reasons),
                }
            )
        if self.grouped_by is not None and len(self.members) > 1:
            payload.pop("artifact_path", None)
            payload.update(
                {
                    "grouped_by": self.grouped_by,
                    "group_member_count": len(self.members),
                    "group_member_labels": self._group_member_labels(limit=4),
                }
            )
        return payload


def normalize_recall_request(request: Mapping[str, Any] | RecallRequest | None = None) -> RecallRequest:
    if isinstance(request, RecallRequest):
        return request

    payload: Mapping[str, Any] = request or {}
    task_kind = _clean_text(payload.get("task_kind")).lower()
    if task_kind not in TASK_KINDS:
        raise ValueError(f"Unsupported recall task kind `{task_kind}`.")

    return RecallRequest(
        task_kind=task_kind,
        query_text=_clean_text(payload.get("query_text")),
        request_basis=_clean_text(payload.get("request_basis")) or None,
        file_hints=_clean_string_list(payload.get("file_hints")),
        surface_filters=_clean_string_list(payload.get("surface_filters"), lowercase=True),
        status_filters=_clean_string_list(payload.get("status_filters"), lowercase=True),
        pinned_event_ids=_clean_string_list(payload.get("pinned_event_ids")),
        exclude_event_ids=_clean_string_list(payload.get("exclude_event_ids")),
        recorded_before_utc=_clean_text(payload.get("recorded_before_utc")) or None,
        recorded_before_event_id=_clean_text(payload.get("recorded_before_event_id")) or None,
        limit=_coerce_positive_int(payload.get("limit"), default=DEFAULT_LIMIT),
        context_budget_chars=_coerce_positive_int(
            payload.get("context_budget_chars"),
            default=DEFAULT_CONTEXT_BUDGET_CHARS,
        ),
        source_event_id=_clean_text(payload.get("source_event_id")) or None,
    )


def _task_specific_status_queries(request: RecallRequest) -> tuple[str | None, ...]:
    if request.status_filters:
        return tuple(request.status_filters)
    if request.task_kind == "failure_analysis":
        return ("quality_fail", "failed", "blocked", None)
    return (None,)


def _event_allowed_by_request(
    request: RecallRequest,
    *,
    event_id: str,
    recorded_at_utc: str | None,
) -> bool:
    if event_id in set(request.exclude_event_ids):
        return False
    cutoff = _coerce_iso_datetime(request.recorded_before_utc)
    recorded_at = _coerce_iso_datetime(recorded_at_utc)
    if cutoff is not None and recorded_at is not None:
        if recorded_at > cutoff:
            return False
        if (
            recorded_at == cutoff
            and request.recorded_before_event_id
            and _event_order_key(event_id) >= _event_order_key(request.recorded_before_event_id)
        ):
            return False
    return True


def _event_order_key(event_id: str) -> str:
    return str(event_id).rsplit(":", 1)[-1]


def _search_queries(request: RecallRequest) -> list[str | None]:
    queries: list[str | None] = []
    seen: set[str | None] = set()

    if request.request_basis == "pass_definition":
        normalized_phrase = re.sub(r"\s+", " ", request.query_text).strip()
        if normalized_phrase:
            escaped_phrase = normalized_phrase.replace('"', '""')
            phrase_query = f'"{escaped_phrase}"'
            seen.add(phrase_query)
            queries.append(phrase_query)

    primary = _fts_query_from_text(request.query_text)
    if primary is not None and primary not in seen:
        seen.add(primary)
        queries.append(primary)

    for hint in request.file_hints[:6]:
        query = _fts_query_from_text(hint)
        if query is not None and query not in seen:
            seen.add(query)
            queries.append(query)

    if request.request_basis == "pass_definition" and queries:
        return queries

    if None not in seen:
        queries.append(None)
    return queries


def _oversample_limit(request: RecallRequest, *, broad_search: bool) -> int:
    if request.request_basis == "pass_definition" and not broad_search:
        return max(request.limit * 6, request.limit + 24)
    if broad_search:
        return max(request.limit, min(request.limit * 2, request.limit + 8))
    return max(request.limit * 3, request.limit + 8)


def _effective_context_budget_chars(request: RecallRequest) -> int:
    if request.request_basis != "pass_definition":
        return request.context_budget_chars
    return max(
        request.context_budget_chars,
        int(request.context_budget_chars * PASS_DEFINITION_CONTEXT_BUDGET_MULTIPLIER),
    )


def _load_index(
    *,
    root: Path | None,
    workspace_id: str,
    index_path: Path | None,
) -> MemoryIndex:
    target_path = index_path or default_memory_index_path(workspace_id=workspace_id, root=root)
    if not target_path.exists():
        rebuild_memory_index(root=root, workspace_id=workspace_id, index_path=target_path)
    return MemoryIndex(target_path)


def retrieve_candidates(
    request: RecallRequest,
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    index_path: Path | None = None,
    index: MemoryIndex | None = None,
) -> list[RecallCandidate]:
    target_index = index or _load_index(root=root, workspace_id=workspace_id, index_path=index_path)
    surfaces = tuple(request.surface_filters) or (None,)
    statuses = _task_specific_status_queries(request)
    candidates_by_event_id: dict[str, RecallCandidate] = {}

    for query in _search_queries(request):
        broad_search = query is None
        limit = _oversample_limit(request, broad_search=broad_search)
        for surface in surfaces:
            for status in statuses:
                rows = target_index.search(query, limit=limit, surface=surface, status=status)
                for rank, row in enumerate(rows):
                    event_id = str(row.get("event_id") or "")
                    if not event_id:
                        continue
                    if not _event_allowed_by_request(
                        request,
                        event_id=event_id,
                        recorded_at_utc=row.get("recorded_at_utc"),
                    ):
                        continue
                    existing = candidates_by_event_id.get(event_id)
                    if existing is None:
                        candidates_by_event_id[event_id] = RecallCandidate.from_row(row, rank=rank)
                        continue
                    existing.merged_with(row, rank=rank)

    get_event = getattr(target_index, "get_event", None)
    if callable(get_event):
        for event_id in request.pinned_event_ids:
            if event_id in candidates_by_event_id:
                continue
            row = get_event(event_id)
            if not row:
                continue
            if not _event_allowed_by_request(
                request,
                event_id=event_id,
                recorded_at_utc=row.get("recorded_at_utc"),
            ):
                continue
            candidates_by_event_id[event_id] = RecallCandidate.from_row(row, rank=10_000)

    return list(candidates_by_event_id.values())


def _file_match_strength(candidate: RecallCandidate, file_hints: tuple[str, ...]) -> int:
    if not file_hints:
        return 0
    haystacks = [
        _lower_text((candidate.artifact_path,)),
        _lower_text((candidate.prompt, candidate.output_text, candidate.notes_text)),
    ]
    best = 0
    for hint in file_hints:
        normalized_hint = hint.replace("\\", "/").lower()
        basename = Path(hint).name.lower()
        stem = Path(hint).stem.lower()
        if any(normalized_hint and normalized_hint in haystack for haystack in haystacks):
            best = max(best, 3)
        elif any(basename and basename in haystack for haystack in haystacks):
            best = max(best, 2)
        elif any(stem and stem in haystack for haystack in haystacks):
            best = max(best, 1)
    return best


def _keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def _normalized_match_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", _clean_text(value).lower())


def _query_terms(value: str) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in _pathish_tokens(value):
        if len(token) < 3 or token in QUERY_TERM_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return tuple(terms)


def _query_alignment(
    query_text: str,
    candidate: RecallCandidate,
) -> tuple[float, list[str]]:
    normalized_query = _normalized_match_text(query_text)
    if not normalized_query:
        return 0.0, []

    prompt_text = _normalized_match_text(candidate.prompt)
    output_text = _normalized_match_text(candidate.output_text)
    notes_text = _normalized_match_text(candidate.notes_text)
    pass_definition_text = _normalized_match_text(candidate.pass_definition)
    artifact_text = _normalized_match_text(candidate.artifact_path)
    combined = "\n".join(
        part
        for part in (prompt_text, output_text, notes_text, pass_definition_text, artifact_text)
        if part
    )

    score = 0.0
    reasons: list[str] = []
    exact_fields = tuple(part for part in (prompt_text, output_text, pass_definition_text) if part)
    if exact_fields and normalized_query in exact_fields:
        score += 18.0
        reasons.append("exact-query-match")
    elif len(normalized_query) >= 18 and any(normalized_query in part for part in exact_fields):
        score += 8.0
        reasons.append("query-phrase-match")

    terms = _query_terms(normalized_query)
    if terms and combined:
        hits = sum(1 for term in terms if term in combined)
        coverage = hits / len(terms)
        if hits >= 2 or coverage >= 0.45:
            score += round(min(5.0, 1.0 + 4.0 * coverage), 3)
            reasons.append("query-coverage")

    return score, reasons


def _recency_boost(
    candidate: RecallCandidate,
    *,
    newest_at: datetime | None,
    oldest_at: datetime | None,
) -> float:
    recorded_at = _coerce_iso_datetime(candidate.recorded_at_utc)
    if recorded_at is None or newest_at is None or oldest_at is None:
        return 0.0
    span = max((newest_at - oldest_at).total_seconds(), 1.0)
    freshness = max((recorded_at - oldest_at).total_seconds(), 0.0)
    return round(2.0 * (freshness / span), 3)


def rank_candidates(request: RecallRequest, candidates: list[RecallCandidate]) -> list[RecallCandidate]:
    timestamps = [_coerce_iso_datetime(candidate.recorded_at_utc) for candidate in candidates]
    known_timestamps = [value for value in timestamps if value is not None]
    newest_at = max(known_timestamps) if known_timestamps else None
    oldest_at = min(known_timestamps) if known_timestamps else None
    query_head_anchor = _query_head_anchor(request.query_text)
    validation_only_anchor = _validation_only_anchor(request.query_text)
    pinned_event_ids = set(request.pinned_event_ids)

    for candidate in candidates:
        score = 0.0
        reasons: list[str] = []
        if candidate.event_id in pinned_event_ids:
            reasons.append("pinned")
        if candidate.source_contract_broken():
            score -= 50.0
            reasons.append("source-contract-broken")
            for reason in candidate.source_artifact_reasons[:2]:
                reasons.append(f"source-contract:{reason}")
        if candidate.raw_fts_score is not None and candidate.best_rank < 10_000:
            score += max(1.0, 12.0 - min(candidate.best_rank, 10))
            reasons.append("fts-hit")
        if candidate.query_hits > 1:
            score += min(3.0, 0.75 * (candidate.query_hits - 1))
            reasons.append("multi-query-hit")

        file_match_strength = _file_match_strength(candidate, request.file_hints)
        if file_match_strength:
            score += {1: 2.0, 2: 4.0, 3: 7.0}[file_match_strength]
            reasons.append("file-match")

        combined = candidate.combined_text()
        query_alignment_score, query_alignment_reasons = _query_alignment(request.query_text, candidate)
        if query_alignment_score > 0:
            score += query_alignment_score
            reasons.extend(query_alignment_reasons)
        prompt_text = _clean_text(candidate.prompt).lower()
        if query_head_anchor and prompt_text.startswith(query_head_anchor):
            score += 4.5
            reasons.append("query-head-match")
        candidate_query_head_anchor = _candidate_query_head_anchor(candidate)
        if (
            query_head_anchor
            and candidate_query_head_anchor
            and candidate_query_head_anchor != query_head_anchor
        ):
            score -= 2.0
            reasons.append("query-head-mismatch")
        if validation_only_anchor and f"--only {validation_only_anchor}" in combined:
            score += 2.5
            reasons.append("validation-command-match")
        candidate_validation_only_anchor = _candidate_validation_only_anchor(candidate)
        if (
            validation_only_anchor
            and candidate_validation_only_anchor
            and candidate_validation_only_anchor != validation_only_anchor
        ):
            score -= 2.0
            reasons.append("validation-command-mismatch")

        task_hits = _keyword_hits(combined, TASK_KEYWORDS[request.task_kind])
        if task_hits:
            score += min(6.0, 2.0 + float(task_hits))
            reasons.append("task-affinity")

        priority_score, _priority_payload, priority_reasons = _evidence_priority_for_task(request, candidate)
        if priority_score:
            score += priority_score
        reasons.extend(priority_reasons)

        if request.task_kind == "failure_analysis":
            if candidate.failure_like():
                score += 5.0
                reasons.append("failure-signal")
            if candidate.repair_like():
                score += 1.5
                reasons.append("repair-signal")
            if candidate.accepted_like():
                score += 1.0
                reasons.append("accepted-signal")
        else:
            if candidate.accepted_like():
                score += 4.0
                reasons.append("accepted-signal")
            if candidate.failure_like():
                score += 2.0
                reasons.append("failure-signal")
            if candidate.open_risk_like():
                score += 1.0
                reasons.append("risk-signal")

        recency = _recency_boost(candidate, newest_at=newest_at, oldest_at=oldest_at)
        if recency > 0:
            score += recency
            reasons.append("recent")

        candidate.score = round(score, 3)
        deduped_reasons: list[str] = []
        seen: set[str] = set()
        for reason in reasons:
            if reason not in seen:
                seen.add(reason)
                deduped_reasons.append(reason)
        candidate.reasons = deduped_reasons

    return sorted(
        candidates,
        key=lambda item: (item.score, item.recorded_at_utc or "", item.event_id),
        reverse=True,
    )


def _choose_block_title(request: RecallRequest, candidate: RecallCandidate) -> str:
    file_match = _file_match_strength(candidate, request.file_hints) > 0
    accepted = candidate.accepted_like()
    failure = candidate.failure_like()
    repair = candidate.repair_like()
    open_risk = candidate.open_risk_like()

    if request.task_kind == "failure_analysis" and failure:
        return BLOCK_FAILURE
    if request.task_kind == "review" and file_match:
        return BLOCK_FILES
    if request.task_kind == "design" and accepted:
        return BLOCK_ACCEPTED
    if open_risk:
        return BLOCK_RISKS
    if failure and repair:
        return BLOCK_FAILURE
    if accepted:
        return BLOCK_ACCEPTED
    if file_match:
        return BLOCK_FILES
    return BLOCK_RELEVANT


def _build_candidate_summary(candidate: RecallCandidate, *, block_title: str) -> str:
    if block_title == BLOCK_RISKS:
        details = _join_text(
            (
                f"Status: {candidate.status}" if candidate.status else "",
                _truncate(candidate.notes_text, limit=120),
                _truncate(candidate.output_text, limit=140),
            )
        )
        prefix = _truncate(candidate.prompt, limit=100)
        if details:
            return _truncate(f"{prefix} | {details}", limit=320)
        return prefix

    pieces = []
    prompt = _truncate(candidate.prompt, limit=120)
    output = _truncate(candidate.output_text, limit=160)
    notes = _truncate(candidate.notes_text, limit=120)
    if prompt:
        pieces.append(f"Prompt: {prompt}")
    if output:
        pieces.append(f"Outcome: {output}")
    if notes:
        pieces.append(f"Notes: {notes}")
    if candidate.artifact_path:
        pieces.append(f"Artifact: {_truncate(candidate.artifact_path, limit=96)}")
    return _truncate(" | ".join(piece for piece in pieces if piece), limit=460)


def _candidate_block_item(candidate: RecallCandidate) -> dict[str, Any]:
    block_title = candidate.block_title or BLOCK_RELEVANT
    return {
        "event_id": candidate.event_id,
        "score": round(candidate.score, 3),
        "reasons": list(candidate.reasons),
        "status": candidate.status,
        "session_id": candidate.session_id,
        "session_surface": candidate.session_surface,
        "event_kind": candidate.event_kind,
        "recorded_at_utc": candidate.recorded_at_utc,
        "artifact_path": candidate.artifact_path,
        "summary": _build_candidate_summary(candidate, block_title=block_title),
    }


def _top_selected_reference_dict(unit: RecallSelectionUnit) -> dict[str, Any]:
    payload = unit.to_reference_dict()
    keys = (
        "event_id",
        "score",
        "reasons",
        "block_title",
        "prompt_excerpt",
        "status",
        "recorded_at_utc",
        "session_id",
        "session_surface",
        "artifact_path",
        "evidence_types",
        "evidence_priority",
        "event_contract_status",
        "source_artifact_status",
        "source_artifact_reasons",
        "group_member_count",
        "group_member_event_ids",
        "group_member_labels",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def _omitted_candidate_reference_dict(candidate: RecallCandidate, *, reason: str) -> dict[str, Any]:
    payload = candidate.to_reference_dict()
    keys = (
        "event_id",
        "score",
        "reasons",
        "block_title",
        "prompt_excerpt",
        "status",
        "recorded_at_utc",
        "session_id",
        "session_surface",
        "artifact_path",
        "evidence_types",
        "evidence_priority",
        "event_contract_status",
        "source_artifact_status",
        "source_artifact_reasons",
    )
    reference = {key: payload.get(key) for key in keys if key in payload}
    reference["omitted_reason"] = reason
    return reference


def _source_contract_payload(candidate: RecallCandidate | None) -> dict[str, Any]:
    if candidate is None:
        return {
            "source_event_contract_status": None,
            "source_artifact_status": None,
            "source_artifact_reasons": [],
            "source_artifact_durability_status": None,
            "source_artifact_readability_status": None,
        }
    return {
        "source_event_contract_status": candidate.event_contract_status,
        "source_artifact_status": candidate.source_artifact_status,
        "source_artifact_reasons": list(candidate.source_artifact_reasons),
        "source_artifact_durability_status": candidate.source_artifact_durability_status,
        "source_artifact_readability_status": candidate.source_artifact_readability_status,
    }


def _source_evidence_type_match(candidate: RecallCandidate | None) -> bool | None:
    if candidate is None:
        return None
    return bool(candidate.evidence_priority.get("matched_task_evidence_types"))


def _miss_reason_detail(
    miss_reason: str | None,
    *,
    pool_status: str,
    source_candidate: RecallCandidate | None,
) -> str | None:
    if miss_reason is None:
        return None
    if miss_reason == "source_missing_from_index":
        return "source event is not present in the memory index"
    if miss_reason == "excluded_current_review_subject":
        return "source event is the active review subject and is excluded from prior evidence"
    if miss_reason == "not_retrieved":
        return "source event exists in the index but was absent from the candidate pool for this query"
    if miss_reason == "ranked_out_by_limit":
        return "source event was in the candidate pool but fell below the selected candidate limit"
    if miss_reason == "dropped_by_context_budget":
        return "source event was ranked but omitted by the overall context budget"
    if miss_reason == "dropped_by_block_budget":
        return "source event was ranked but omitted by its block budget"
    if miss_reason == SOURCE_CONTRACT_MISS_REASON:
        reasons = ", ".join(source_candidate.source_artifact_reasons) if source_candidate is not None else ""
        suffix = f": {reasons}" if reasons else ""
        return f"source event is present but its event contract or source artifact is not recallable{suffix}"
    if miss_reason == EVIDENCE_TYPE_MISMATCH_MISS_REASON:
        return "source event reached ranking but its evidence types are low-priority for this task kind"
    if miss_reason == "not_selected":
        return f"source event reached {pool_status} but was not selected; inspect source and top candidate reasons"
    return miss_reason


def _miss_diagnostics(
    *,
    source_selected: bool,
    pool_status: str,
    miss_reason: str | None,
    source_candidate: RecallCandidate | None,
) -> list[str]:
    diagnostics: list[str] = []
    if source_selected:
        return diagnostics
    if miss_reason == "excluded_current_review_subject":
        diagnostics.append("excluded_current_review_subject")
    if pool_status in {"missing_from_index", "not_in_candidate_pool"}:
        diagnostics.append(pool_status)
    elif pool_status == "unknown":
        diagnostics.append("candidate_pool_unknown")
    elif pool_status:
        diagnostics.append("candidate_pool_present")
    if miss_reason in {"ranked_out_by_limit", "dropped_by_context_budget", "dropped_by_block_budget"}:
        diagnostics.append("ranking_or_budget_drop")
    if source_candidate is not None and source_candidate.source_contract_broken():
        diagnostics.append("source_event_contract_broken")
    if (
        source_candidate is not None
        and _source_evidence_type_match(source_candidate) is False
        and source_candidate.evidence_types
    ):
        diagnostics.append("evidence_type_mismatch")
    return _dedupe_reasons(diagnostics)


def _pass_definition_group_key(
    request: RecallRequest,
    candidate: RecallCandidate,
) -> tuple[str, str] | None:
    if request.request_basis != "pass_definition":
        return None
    if _clean_text(candidate.event_kind) != "capability_result":
        return None
    normalized_pass_definition = _normalized_match_text(candidate.pass_definition)
    if not normalized_pass_definition:
        return None
    return normalized_pass_definition, candidate.block_title or BLOCK_RELEVANT


def _selection_units_from_ranked(
    request: RecallRequest,
    ranked: list[RecallCandidate],
) -> list[RecallSelectionUnit]:
    ordered_keys: list[tuple[str, str] | str] = []
    grouped_members: dict[tuple[str, str], list[RecallCandidate]] = {}
    for candidate in ranked:
        if candidate.source_contract_broken():
            continue
        group_key = _pass_definition_group_key(request, candidate)
        if group_key is None:
            ordered_keys.append(candidate.event_id)
            grouped_members[(candidate.event_id, "")] = [candidate]
            continue
        if group_key not in grouped_members:
            ordered_keys.append(group_key)
            grouped_members[group_key] = []
        grouped_members[group_key].append(candidate)

    selection_units: list[RecallSelectionUnit] = []
    for key in ordered_keys:
        if isinstance(key, tuple):
            members = tuple(grouped_members[key])
            if len(members) > 1:
                selection_units.append(
                    RecallSelectionUnit(
                        representative=members[0],
                        members=members,
                        grouped_by="pass_definition",
                    )
                )
            else:
                selection_units.append(
                    RecallSelectionUnit(
                        representative=members[0],
                        members=members,
                    )
                )
            continue
        member_key = (key, "")
        members = tuple(grouped_members[member_key])
        selection_units.append(
            RecallSelectionUnit(
                representative=members[0],
                members=members,
            )
        )
    return selection_units


def _source_evaluation(
    request: RecallRequest,
    *,
    root: Path | None,
    index: MemoryIndex | None,
    ranked: list[RecallCandidate],
    selected_units: list[RecallSelectionUnit],
    omitted_reasons: Mapping[str, str],
) -> dict[str, Any] | None:
    source_event_id = _clean_text(request.source_event_id)
    if not source_event_id:
        return None
    if source_event_id in set(request.exclude_event_ids):
        get_event = getattr(index, "get_event", None)
        source_row = get_event(source_event_id) if callable(get_event) else None
        source_probe = None
        if isinstance(source_row, Mapping):
            source_probe = RecallCandidate.from_row(source_row, rank=10_000)
            _annotate_source_contract([source_probe], root=root)
            _evidence_priority_for_task(request, source_probe)
        miss_reason = "excluded_current_review_subject"
        pool_status = "excluded"
        return {
            "source_event_id": source_event_id,
            "source_selected": False,
            "source_rank": None,
            "source_score": None,
            "source_block_title": None,
            "source_prompt_excerpt": source_probe.prompt_excerpt() if source_probe is not None else None,
            "source_reasons": ["current-review-subject"],
            "source_selected_via_group": False,
            "source_group_member_count": None,
            "source_grouped_by": None,
            "source_group_event_id": None,
            "source_group_prompt_excerpt": None,
            "source_group_member_event_ids": [],
            "source_group_member_labels": [],
            "miss_reason": miss_reason,
            "miss_reason_detail": _miss_reason_detail(
                miss_reason,
                pool_status=pool_status,
                source_candidate=source_probe,
            ),
            "miss_diagnostics": _miss_diagnostics(
                source_selected=False,
                pool_status=pool_status,
                miss_reason=miss_reason,
                source_candidate=source_probe,
            ),
            "source_candidate_pool_status": pool_status,
            "source_exists_in_index": bool(source_row) if callable(get_event) else None,
            "source_evidence_types": list(source_probe.evidence_types) if source_probe is not None else [],
            "source_evidence_priority": dict(source_probe.evidence_priority) if source_probe is not None else {},
            "source_evidence_type_match": _source_evidence_type_match(source_probe),
            **_source_contract_payload(source_probe),
            "selected_count": len(selected_units),
            "top_selected": [_top_selected_reference_dict(unit) for unit in selected_units[:3]],
        }

    ranked_positions = {candidate.event_id: index for index, candidate in enumerate(ranked, start=1)}
    selected_unit = next((unit for unit in selected_units if unit.contains_event(source_event_id)), None)
    source_candidate = next((candidate for candidate in ranked if candidate.event_id == source_event_id), None)
    if source_candidate is None:
        get_event = getattr(index, "get_event", None)
        source_row = get_event(source_event_id) if callable(get_event) else None
        source_exists_in_index = bool(source_row) if callable(get_event) else None
        source_probe = None
        if isinstance(source_row, Mapping):
            source_probe = RecallCandidate.from_row(source_row, rank=10_000)
            _annotate_source_contract([source_probe], root=root)
            _evidence_priority_for_task(request, source_probe)
        if source_probe is not None and source_probe.source_contract_broken():
            miss_reason = SOURCE_CONTRACT_MISS_REASON
        else:
            miss_reason = "not_retrieved" if source_exists_in_index is not False else "source_missing_from_index"
        if source_exists_in_index is True:
            pool_status = "not_in_candidate_pool"
        elif source_exists_in_index is False:
            pool_status = "missing_from_index"
        else:
            pool_status = "unknown"
        return {
            "source_event_id": source_event_id,
            "source_selected": False,
            "source_rank": None,
            "source_score": None,
            "source_block_title": None,
            "source_prompt_excerpt": None,
            "source_reasons": [],
            "source_selected_via_group": False,
            "source_group_member_count": None,
            "source_grouped_by": None,
            "source_group_event_id": None,
            "source_group_prompt_excerpt": None,
            "source_group_member_event_ids": [],
            "source_group_member_labels": [],
            "miss_reason": miss_reason,
            "miss_reason_detail": _miss_reason_detail(
                miss_reason,
                pool_status=pool_status,
                source_candidate=source_probe,
            ),
            "miss_diagnostics": _miss_diagnostics(
                source_selected=False,
                pool_status=pool_status,
                miss_reason=miss_reason,
                source_candidate=source_probe,
            ),
            "source_candidate_pool_status": pool_status,
            "source_exists_in_index": source_exists_in_index,
            "source_evidence_types": list(source_probe.evidence_types) if source_probe is not None else [],
            "source_evidence_priority": dict(source_probe.evidence_priority) if source_probe is not None else {},
            "source_evidence_type_match": _source_evidence_type_match(source_probe),
            **_source_contract_payload(source_probe),
            "selected_count": len(selected_units),
            "top_selected": [_top_selected_reference_dict(unit) for unit in selected_units[:3]],
        }

    source_selected = selected_unit is not None and not source_candidate.source_contract_broken()
    source_group_payload = selected_unit.to_reference_dict() if selected_unit is not None else {}
    source_grouped_by = source_group_payload.get("grouped_by")
    source_group_member_count = len(selected_unit.members) if source_grouped_by and selected_unit is not None else None
    pool_status = "selected" if source_selected else "candidate_pool_present"
    if source_candidate.source_contract_broken():
        miss_reason = SOURCE_CONTRACT_MISS_REASON
    else:
        miss_reason = None if source_selected else omitted_reasons.get(source_event_id) or "not_selected"
        if (
            miss_reason == "not_selected"
            and _source_evidence_type_match(source_candidate) is False
            and source_candidate.evidence_types
        ):
            miss_reason = EVIDENCE_TYPE_MISMATCH_MISS_REASON
    return {
        "source_event_id": source_event_id,
        "source_selected": source_selected,
        "source_rank": ranked_positions.get(source_event_id),
        "source_score": round(source_candidate.score, 3),
        "source_block_title": source_candidate.block_title,
        "source_prompt_excerpt": source_candidate.prompt_excerpt(),
        "source_reasons": list(source_candidate.reasons),
        "source_selected_via_group": bool(
            source_selected
            and selected_unit is not None
            and selected_unit.grouped_by is not None
            and selected_unit.event_id != source_event_id
        ),
        "source_group_member_count": source_group_member_count,
        "source_grouped_by": source_grouped_by,
        "source_group_event_id": source_group_payload.get("event_id") if source_grouped_by else None,
        "source_group_prompt_excerpt": source_group_payload.get("prompt_excerpt") if source_grouped_by else None,
        "source_group_member_event_ids": source_group_payload.get("group_member_event_ids") or [],
        "source_group_member_labels": source_group_payload.get("group_member_labels") or [],
        "miss_reason": miss_reason,
        "miss_reason_detail": _miss_reason_detail(
            miss_reason,
            pool_status=pool_status,
            source_candidate=source_candidate,
        ),
        "miss_diagnostics": _miss_diagnostics(
            source_selected=source_selected,
            pool_status=pool_status,
            miss_reason=miss_reason,
            source_candidate=source_candidate,
        ),
        "source_candidate_pool_status": pool_status,
        "source_exists_in_index": True,
        "source_evidence_types": list(source_candidate.evidence_types),
        "source_evidence_priority": dict(source_candidate.evidence_priority),
        "source_evidence_type_match": _source_evidence_type_match(source_candidate),
        **_source_contract_payload(source_candidate),
        "selected_count": len(selected_units),
        "top_selected": [_top_selected_reference_dict(unit) for unit in selected_units[:3]],
    }


def build_context_bundle(
    request: Mapping[str, Any] | RecallRequest,
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    index_path: Path | None = None,
    index: MemoryIndex | None = None,
) -> dict[str, Any]:
    normalized_request = normalize_recall_request(request)
    target_index = index or _load_index(root=root, workspace_id=workspace_id, index_path=index_path)
    candidates = retrieve_candidates(
        normalized_request,
        root=root,
        workspace_id=workspace_id,
        index_path=index_path,
        index=target_index,
    )
    _annotate_source_contract(candidates, root=root)
    ranked = rank_candidates(normalized_request, candidates)
    for candidate in ranked:
        candidate.block_title = _choose_block_title(normalized_request, candidate)
    selection_units = _selection_units_from_ranked(normalized_request, ranked)
    effective_context_budget_chars = _effective_context_budget_chars(normalized_request)

    block_titles = TASK_BLOCK_ORDER[normalized_request.task_kind]
    block_limits = {
        title: max(120, int(effective_context_budget_chars * BLOCK_BUDGET_RATIO[normalized_request.task_kind][title]))
        for title in block_titles
    }
    block_payloads: dict[str, list[dict[str, Any]]] = {title: [] for title in block_titles}
    block_used_chars: dict[str, int] = {title: 0 for title in block_titles}
    selected_units: list[RecallSelectionUnit] = []
    omitted_reasons: dict[str, str] = {
        candidate.event_id: SOURCE_CONTRACT_MISS_REASON
        for candidate in ranked
        if candidate.source_contract_broken()
    }
    used_chars = 0
    omitted_count = len(omitted_reasons)

    for unit in selection_units:
        member_event_ids = unit.member_event_ids
        if unit.representative.source_contract_broken():
            for event_id in member_event_ids:
                if event_id not in omitted_reasons:
                    omitted_count += 1
                    omitted_reasons[event_id] = SOURCE_CONTRACT_MISS_REASON
            continue
        if len(selected_units) >= normalized_request.limit:
            omitted_count += len(member_event_ids)
            for event_id in member_event_ids:
                omitted_reasons.setdefault(event_id, "ranked_out_by_limit")
            continue

        item = unit.to_block_item()
        item_chars = _json_char_count(item)
        title = unit.block_title
        if used_chars + item_chars > effective_context_budget_chars:
            omitted_count += len(member_event_ids)
            for event_id in member_event_ids:
                omitted_reasons.setdefault(event_id, "dropped_by_context_budget")
            continue
        if block_used_chars[title] + item_chars > block_limits[title] and block_payloads[title]:
            omitted_count += len(member_event_ids)
            for event_id in member_event_ids:
                omitted_reasons.setdefault(event_id, "dropped_by_block_budget")
            continue
        block_payloads[title].append(item)
        block_used_chars[title] += item_chars
        used_chars += item_chars
        selected_units.append(unit)

    blocks = [
        {"title": title, "items": block_payloads[title]}
        for title in block_titles
        if block_payloads[title]
    ]

    bundle = {
        "bundle_version": CONTEXT_BUNDLE_VERSION,
        "task_kind": normalized_request.task_kind,
        "query_text": normalized_request.query_text,
        "request_basis": normalized_request.request_basis,
        "file_hints": list(normalized_request.file_hints),
        "surface_filters": list(normalized_request.surface_filters),
        "status_filters": list(normalized_request.status_filters),
        "pinned_event_ids": list(normalized_request.pinned_event_ids),
        "exclude_event_ids": list(normalized_request.exclude_event_ids),
        "recorded_before_utc": normalized_request.recorded_before_utc,
        "recorded_before_event_id": normalized_request.recorded_before_event_id,
        "selected_count": len(selected_units),
        "omitted_count": omitted_count,
        "omitted_candidates": [
            _omitted_candidate_reference_dict(candidate, reason=omitted_reasons[candidate.event_id])
            for candidate in ranked
            if candidate.event_id in omitted_reasons
        ][:10],
        "budget": {
            "context_budget_chars": normalized_request.context_budget_chars,
            "effective_context_budget_chars": effective_context_budget_chars,
            "used_chars": used_chars,
        },
        "selected_candidates": [unit.to_reference_dict() for unit in selected_units],
        "blocks": blocks,
    }
    source_evaluation = _source_evaluation(
        normalized_request,
        root=root,
        index=target_index,
        ranked=ranked,
        selected_units=selected_units,
        omitted_reasons=omitted_reasons,
    )
    if source_evaluation is not None:
        bundle["source_evaluation"] = source_evaluation
    return bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a grouped recall context bundle from the local SQLite memory index.",
    )
    parser.add_argument(
        "--task-kind",
        required=True,
        choices=TASK_KINDS,
        help="Recall task kind.",
    )
    parser.add_argument(
        "--query",
        default="",
        help="Primary query text.",
    )
    parser.add_argument(
        "--file-hint",
        action="append",
        default=[],
        help="Optional file hint. Repeat to add more than one.",
    )
    parser.add_argument(
        "--surface-filter",
        action="append",
        default=[],
        help="Optional session surface filter. Repeat to add more than one.",
    )
    parser.add_argument(
        "--status-filter",
        action="append",
        default=[],
        help="Optional status filter. Repeat to add more than one.",
    )
    parser.add_argument(
        "--exclude-event",
        action="append",
        default=[],
        help="Event id to exclude from selected prior evidence. Repeatable.",
    )
    parser.add_argument(
        "--recorded-before",
        default=None,
        help="Only recall events recorded at or before this UTC timestamp.",
    )
    parser.add_argument(
        "--recorded-before-event",
        default=None,
        help="When timestamps tie, only recall event ids ordered before this event id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum number of selected candidates.",
    )
    parser.add_argument(
        "--context-budget-chars",
        type=int,
        default=DEFAULT_CONTEXT_BUDGET_CHARS,
        help="Total character budget used for grouped context items.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional repo root override.",
    )
    parser.add_argument(
        "--workspace-id",
        default=DEFAULT_WORKSPACE_ID,
        help="Workspace id to read.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="Optional explicit SQLite index path.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    bundle = build_context_bundle(
        {
            "task_kind": args.task_kind,
            "query_text": args.query,
            "file_hints": args.file_hint,
            "surface_filters": args.surface_filter,
            "status_filters": args.status_filter,
            "exclude_event_ids": args.exclude_event,
            "recorded_before_utc": args.recorded_before,
            "recorded_before_event_id": args.recorded_before_event,
            "limit": args.limit,
            "context_budget_chars": args.context_budget_chars,
        },
        root=args.root,
        workspace_id=args.workspace_id,
        index_path=args.index_path,
    )
    print(json.dumps(bundle, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
