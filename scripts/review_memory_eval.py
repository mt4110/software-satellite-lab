#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from artifact_vault import capture_artifact, resolve_vault_object_path
from evidence_support import build_evidence_support_result
from gemma_runtime import repo_root, timestamp_slug, timestamp_utc, write_json
from review_memory_fixtures import (
    MISS_TAXONOMY,
    default_review_memory_suite_path,
    fixture_category_counts,
    load_review_memory_fixtures,
)
from workspace_state import DEFAULT_WORKSPACE_ID


REVIEW_MEMORY_EVAL_SCHEMA_NAME = "software-satellite-review-memory-eval"
REVIEW_MEMORY_EVAL_SCHEMA_VERSION = 1
REVIEW_MEMORY_MISS_REPORT_SCHEMA_NAME = "software-satellite-review-memory-miss-report"
REVIEW_MEMORY_MISS_REPORT_SCHEMA_VERSION = 1

DEFAULT_REPORT_EXCERPT_CHARS = 900
DEFAULT_TOP_K = 5
TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")
SECRET_MARKERS = ("sk-", "ghp_", "xoxb-", "xoxp-", "bearer ")
INVALID_SUPPORT_CLASSES = {
    "current_review_subject",
    "future_evidence",
    "missing_source",
    "modified_source",
    "weak_match",
    "contradictory",
    "unverified_agent_claim",
    "unknown",
}
EXIT_GATES = {
    "critical_false_support": ("==", 0),
    "synthetic_positive_support_precision": (">=", 0.90),
    "synthetic_useful_recall_at_5": (">=", 0.60),
    "no_evidence_honesty": (">=", 0.95),
    "miss_report_coverage": ("==", 1.0),
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
    if not isinstance(value, (list, tuple, set)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _clean_text(item)
        if text is None or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _bool(value: Any, *, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _tokens(text: str) -> set[str]:
    return {
        token.lower().strip("./:-")
        for token in TOKEN_RE.findall(text)
        if len(token.strip("./:-")) >= 3
    }


def _truncate(text: str | None, *, limit: int) -> str:
    cleaned = _clean_text(text) or ""
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 15)].rstrip() + " [truncated]"


def _safe_fixture_relative_path(value: str | None, *, fallback: str) -> str:
    text = _clean_text(value) or fallback
    candidate = Path(text)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts) or not candidate.name:
        raise ValueError(f"Fixture source_path must stay inside the fixture workspace: `{text}`.")
    return candidate.as_posix()


def review_memory_benchmark_root(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return _resolve_root(root) / "artifacts" / "review_memory_benchmark" / workspace_id


def latest_review_memory_eval_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return review_memory_benchmark_root(workspace_id=workspace_id, root=root) / "latest.json"


def latest_review_memory_eval_markdown_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return review_memory_benchmark_root(workspace_id=workspace_id, root=root) / "latest.md"


def review_memory_eval_run_path(
    *,
    run_id: str,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return review_memory_benchmark_root(workspace_id=workspace_id, root=root) / "runs" / f"{run_id}-review-memory-eval.json"


def review_memory_eval_markdown_run_path(
    *,
    run_id: str,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return review_memory_benchmark_root(workspace_id=workspace_id, root=root) / "runs" / f"{run_id}-review-memory-eval.md"


def latest_review_memory_miss_report_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return review_memory_benchmark_root(workspace_id=workspace_id, root=root) / "miss-reports" / "latest.json"


def latest_review_memory_miss_report_markdown_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return review_memory_benchmark_root(workspace_id=workspace_id, root=root) / "miss-reports" / "latest.md"


def review_memory_miss_report_run_path(
    *,
    run_id: str,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return review_memory_benchmark_root(workspace_id=workspace_id, root=root) / "miss-reports" / "runs" / f"{run_id}-miss-report.json"


def review_memory_miss_report_markdown_run_path(
    *,
    run_id: str,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return review_memory_benchmark_root(workspace_id=workspace_id, root=root) / "miss-reports" / "runs" / f"{run_id}-miss-report.md"


def _fixture_workspace_root(
    *,
    run_id: str,
    suite_id: str,
    fixture_id: str,
    workspace_id: str,
    root: Path,
) -> Path:
    return (
        review_memory_benchmark_root(workspace_id=workspace_id, root=root)
        / "fixture_workspaces"
        / run_id
        / _safe_slug(suite_id)
        / _safe_slug(fixture_id)
    )


def _candidate_text(candidate: Mapping[str, Any], event: Mapping[str, Any] | None = None) -> str:
    content = _mapping_dict((event or {}).get("content"))
    options = _mapping_dict(content.get("options"))
    pieces = [
        _clean_text(candidate.get("prompt")),
        _clean_text(candidate.get("output_text")),
        _clean_text(candidate.get("source_text")),
        " ".join(_string_list(candidate.get("notes"))),
        " ".join(_string_list(candidate.get("tags"))),
        " ".join(_string_list(candidate.get("target_paths"))),
        " ".join(_string_list(options.get("file_hints"))),
    ]
    return " ".join(piece for piece in pieces if piece)


def _default_quality_status(status: str | None, candidate: Mapping[str, Any]) -> str | None:
    if "quality_status" in candidate:
        return _clean_text(candidate.get("quality_status"))
    normalized = (status or "").strip().lower()
    if normalized in {"accepted", "accept", "ok", "passed", "pass", "resolved"}:
        return "pass"
    if normalized in {"failed", "fail", "quality_fail", "blocked", "error", "rejected", "reject", "needs_fix"}:
        return "fail"
    return None


def _write_candidate_source(
    candidate: Mapping[str, Any],
    *,
    fixture_root: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    behavior = _clean_text(candidate.get("artifact_behavior")) or "captured"
    source_rel = _safe_fixture_relative_path(
        _clean_text(candidate.get("source_path")),
        fallback=f"sources/{_safe_slug(_clean_text(candidate.get('event_id')) or 'candidate')}.txt",
    )
    source_path = fixture_root / source_rel
    source_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_kind = _clean_text(candidate.get("artifact_kind")) or "review_note"
    max_capture_bytes = int(candidate.get("max_capture_bytes") or 2 * 1024 * 1024)
    report_excerpt_chars = int(candidate.get("report_excerpt_chars") or DEFAULT_REPORT_EXCERPT_CHARS)

    if behavior == "no_ref":
        return None, {
            "source_path": str(source_path),
            "artifact_behavior": behavior,
            "source_state": "not_recorded",
            "capture_state": "not_recorded",
        }

    if behavior == "missing_source":
        ref = capture_artifact(
            source_rel,
            kind=artifact_kind,
            root=fixture_root,
            max_capture_bytes=max_capture_bytes,
            report_excerpt_chars=report_excerpt_chars,
        )
    elif behavior == "binary_refused":
        binary_hex = _clean_text(candidate.get("binary_hex")) or "000102ff"
        source_path.write_bytes(bytes.fromhex(binary_hex))
        ref = capture_artifact(
            source_rel,
            kind=artifact_kind,
            root=fixture_root,
            max_capture_bytes=max_capture_bytes,
            report_excerpt_chars=report_excerpt_chars,
        )
    else:
        source_text = _clean_text(candidate.get("source_text"))
        if source_text is None:
            source_text = _clean_text(candidate.get("output_text")) or "review memory fixture evidence\n"
        source_path.write_text(source_text, encoding="utf-8")
        ref = capture_artifact(
            source_rel,
            kind=artifact_kind,
            root=fixture_root,
            max_capture_bytes=max_capture_bytes,
            report_excerpt_chars=report_excerpt_chars,
        )
        if behavior == "modified_source":
            object_path = resolve_vault_object_path(ref, root=fixture_root)
            if object_path is not None:
                object_path.write_text("modified fixture source after capture\n", encoding="utf-8")
        elif behavior == "original_source_modified":
            source_path.write_text(source_text + "\nmodified after capture\n", encoding="utf-8")

    return ref, {
        "source_path": str(source_path),
        "artifact_behavior": behavior,
        "source_state": _clean_text(ref.get("source_state")) if ref else "not_recorded",
        "capture_state": _clean_text(ref.get("capture_state")) if ref else "not_recorded",
        "artifact_id_present": bool(ref and _clean_text(ref.get("artifact_id"))),
        "report_excerpt": _mapping_dict(ref.get("report_excerpt")).get("text") if ref else "",
        "redaction": _mapping_dict(ref.get("redaction")) if ref else {},
    }


def _materialize_candidate_event(
    fixture: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    fixture_root: Path,
    workspace_id: str,
) -> dict[str, Any]:
    current = _mapping_dict(fixture.get("current_event"))
    event_id = _clean_text(candidate.get("event_id")) or f"{_clean_text(fixture.get('fixture_id')) or 'fixture'}:candidate"
    ref, source_info = _write_candidate_source(candidate, fixture_root=fixture_root)
    status = _clean_text(candidate.get("status")) or "needs_review"
    quality_status = _default_quality_status(status, candidate)
    execution_status = _clean_text(candidate.get("execution_status")) or status
    evidence_types = _string_list(candidate.get("evidence_types"))
    notes = _string_list(candidate.get("notes"))
    tags = _string_list(candidate.get("tags"))
    options: dict[str, Any] = {
        "validation_mode": "review_memory_benchmark_fixture",
        "file_hints": _string_list(candidate.get("target_paths")),
        "source_fixture_category": _clean_text(fixture.get("category")),
    }
    if ref is not None:
        options["artifact_vault_refs"] = [ref]
    if quality_status is not None:
        options["quality_status"] = quality_status
    if evidence_types:
        options["evidence_types"] = evidence_types
    event = {
        "schema_name": "software-satellite-event",
        "schema_version": 1,
        "event_id": event_id,
        "event_kind": _clean_text(candidate.get("event_kind")) or "chat_run",
        "recorded_at_utc": _clean_text(candidate.get("recorded_at_utc"))
        or _clean_text(fixture.get("default_candidate_recorded_at_utc"))
        or "2026-05-10T00:00:00+00:00",
        "workspace": {"workspace_id": workspace_id},
        "session": {
            "session_id": _clean_text(candidate.get("session_id")) or f"fixture-{_safe_slug(event_id)}",
            "surface": _clean_text(candidate.get("session_surface")) or "chat",
            "mode": "review_memory_benchmark",
        },
        "outcome": {
            "status": status,
            "quality_status": quality_status,
            "execution_status": execution_status,
        },
        "content": {
            "prompt": _clean_text(candidate.get("prompt")) or _clean_text(fixture.get("query")) or "",
            "output_text": _clean_text(candidate.get("output_text")) or "",
            "notes": notes,
            "options": options,
        },
        "source_refs": {},
        "tags": tags,
    }
    return {
        "fixture_candidate": dict(candidate),
        "event": event,
        "source_info": source_info,
    }


def _score_candidate(fixture: Mapping[str, Any], materialized: Mapping[str, Any]) -> float:
    candidate = _mapping_dict(materialized.get("fixture_candidate"))
    event = _mapping_dict(materialized.get("event"))
    current = _mapping_dict(fixture.get("current_event"))
    query_tokens = _tokens(_clean_text(fixture.get("query")) or "")
    candidate_tokens = _tokens(_candidate_text(candidate, event))
    overlap = len(query_tokens & candidate_tokens)
    score = min(8.0, float(overlap))

    current_paths = set(_string_list(current.get("target_paths")))
    candidate_paths = set(_string_list(candidate.get("target_paths")))
    if current_paths and candidate_paths:
        if current_paths & candidate_paths:
            score += 9.0
        else:
            current_heads = {path.split("/", 1)[0] for path in current_paths}
            candidate_heads = {path.split("/", 1)[0] for path in candidate_paths}
            if current_heads & candidate_heads:
                score += 2.0

    status = (_clean_text(candidate.get("status")) or "").lower()
    if status in {"failed", "rejected", "needs_fix", "blocked"}:
        score += 3.0
    elif status in {"accepted", "resolved", "ok", "passed"}:
        score += 2.0
    if _string_list(candidate.get("evidence_types")):
        score += 1.5
    if _clean_text(candidate.get("artifact_behavior")) in {"missing_source", "modified_source", "binary_refused"}:
        score += 0.5
    return round(score, 3)


def _rank_candidates(fixture: Mapping[str, Any], materialized: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for item in materialized:
        score = _score_candidate(fixture, item)
        event = _mapping_dict(item.get("event"))
        ranked.append({**item, "score": score, "event_id": _clean_text(event.get("event_id")) or ""})
    return sorted(ranked, key=lambda item: (float(item["score"]), str(item["event_id"])), reverse=True)


def _requested_polarity(fixture: Mapping[str, Any], candidate: Mapping[str, Any]) -> str | None:
    return (
        _clean_text(candidate.get("requested_polarity"))
        or _clean_text(_mapping_dict(fixture.get("expected")).get("support_polarity"))
        or _clean_text(fixture.get("requested_polarity"))
    )


def _support_item(
    *,
    fixture: Mapping[str, Any],
    ranked_item: Mapping[str, Any],
    support: Mapping[str, Any],
    rank: int,
) -> dict[str, Any]:
    candidate = _mapping_dict(ranked_item.get("fixture_candidate"))
    source_info = _mapping_dict(ranked_item.get("source_info"))
    expected_support = _bool(_mapping_dict(candidate.get("expected")).get("support"), default=False)
    expected_useful = _bool(_mapping_dict(candidate.get("expected")).get("useful"), default=False)
    support_class = _clean_text(support.get("support_class")) or "unknown"
    can_support = bool(support.get("can_support_decision"))
    is_valid_relevant = can_support and expected_support and expected_useful and support_class not in INVALID_SUPPORT_CLASSES
    critical_false = can_support and not is_valid_relevant
    return {
        "rank": rank,
        "event_id": _clean_text(support.get("event_id")) or _clean_text(candidate.get("event_id")) or "",
        "score": float(ranked_item.get("score") or 0.0),
        "support_class": support_class,
        "support_polarity": _clean_text(support.get("support_polarity")) or "none",
        "can_support_decision": can_support,
        "blockers": _string_list(support.get("blockers")),
        "warnings": _string_list(support.get("warnings")),
        "artifact_ref_count": len(_string_list(support.get("artifact_refs"))),
        "source_state": _clean_text(source_info.get("source_state")) or "unknown",
        "capture_state": _clean_text(source_info.get("capture_state")) or "unknown",
        "expected_support": expected_support,
        "expected_useful": expected_useful,
        "valid_relevant_support": is_valid_relevant,
        "critical_false_support": critical_false,
        "report_excerpt": _truncate(_clean_text(source_info.get("report_excerpt")) or "", limit=DEFAULT_REPORT_EXCERPT_CHARS),
    }


def _secret_redaction_pass(fixture: Mapping[str, Any], support_items: list[Mapping[str, Any]], candidate_rows: list[Mapping[str, Any]]) -> bool:
    expected = _mapping_dict(fixture.get("expected"))
    if not _bool(expected.get("requires_redaction")):
        return True
    report_text = "\n".join(
        _clean_text(row.get("report_excerpt")) or ""
        for row in [*support_items, *candidate_rows]
    )
    lowered = report_text.lower()
    if any(marker in lowered for marker in SECRET_MARKERS):
        return False
    redaction_counts = [
        int(_mapping_dict(_mapping_dict(row.get("source_info")).get("redaction")).get("secret_like_tokens") or 0)
        for row in candidate_rows
    ]
    return any(count > 0 for count in redaction_counts) and "[REDACTED]" in report_text


def _bounded_report_pass(fixture: Mapping[str, Any], candidate_rows: list[Mapping[str, Any]]) -> bool:
    expected = _mapping_dict(fixture.get("expected"))
    max_chars = expected.get("max_report_excerpt_chars")
    if max_chars is None:
        return True
    limit = int(max_chars)
    excerpts = [_clean_text(row.get("report_excerpt")) or "" for row in candidate_rows]
    if any(len(excerpt) > limit for excerpt in excerpts):
        return False
    if _bool(expected.get("requires_truncation_marker")):
        return any("[truncated]" in excerpt for excerpt in excerpts)
    return True


def _binary_refusal_pass(fixture: Mapping[str, Any], candidate_rows: list[Mapping[str, Any]]) -> bool:
    if _clean_text(fixture.get("category")) != "binary_artifact":
        return True
    return any(_clean_text(_mapping_dict(row.get("source_info")).get("source_state")) == "binary_refused" for row in candidate_rows)


def _classify_failed_fixture(result: Mapping[str, Any]) -> str:
    explicit = _clean_text(_mapping_dict(result.get("expected")).get("miss_reason"))
    if explicit in MISS_TAXONOMY:
        return explicit
    category = _clean_text(result.get("category"))
    if category in {"self_recall_trap", "future_evidence_trap"}:
        return "target_identity_miss"
    if category in {"missing_source_trap", "modified_source_trap", "binary_artifact", "huge_diff"}:
        return "source_integrity_blocked"
    if category == "weak_text_match_trap":
        return "weak_match_overfiltered"
    if category == "contradictory_verdict_trap":
        return "contradiction_unresolved"
    if category == "agent_claim_trap":
        return "missing_human_signal"
    support_items = [dict(item) for item in result.get("support_items") or [] if isinstance(item, Mapping)]
    if not support_items and _bool(_mapping_dict(result.get("expected")).get("requires_support")):
        return "lexical_miss"
    return "unknown"


def _evaluate_fixture(
    fixture: Mapping[str, Any],
    *,
    workspace_id: str,
    root: Path,
    run_id: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    fixture_id = _clean_text(fixture.get("fixture_id")) or "fixture"
    suite_id = _clean_text(fixture.get("suite_id")) or "suite"
    fixture_root = _fixture_workspace_root(
        run_id=run_id,
        suite_id=suite_id,
        fixture_id=fixture_id,
        workspace_id=workspace_id,
        root=root,
    )
    fixture_root.mkdir(parents=True, exist_ok=True)
    current = _mapping_dict(fixture.get("current_event"))
    current_event_id = _clean_text(current.get("event_id")) or f"{fixture_id}:current"
    review_started_at = _clean_text(fixture.get("review_started_at_utc")) or _clean_text(current.get("review_started_at_utc"))

    materialized = [
        _materialize_candidate_event(fixture, candidate, fixture_root=fixture_root, workspace_id=workspace_id)
        for candidate in fixture.get("candidate_events") or []
        if isinstance(candidate, Mapping)
    ]
    ranked = _rank_candidates(fixture, materialized)
    top_rows = ranked[:top_k]
    support_items: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for rank, row in enumerate(top_rows, start=1):
        candidate = _mapping_dict(row.get("fixture_candidate"))
        event = _mapping_dict(row.get("event"))
        source_info = _mapping_dict(row.get("source_info"))
        support = build_evidence_support_result(
            _clean_text(event.get("event_id")) or "",
            event=event,
            review_started_at=review_started_at,
            active_subject=current_event_id,
            requested_polarity=_requested_polarity(fixture, candidate),
            root=fixture_root,
            checked_at_utc="2026-05-12T00:00:00+00:00",
        )
        item = _support_item(fixture=fixture, ranked_item=row, support=support, rank=rank)
        if item["can_support_decision"]:
            support_items.append(item)
        candidate_rows.append(
            {
                "rank": rank,
                "event_id": item["event_id"],
                "score": item["score"],
                "support_class": item["support_class"],
                "support_polarity": item["support_polarity"],
                "can_support_decision": item["can_support_decision"],
                "blockers": item["blockers"],
                "source_state": item["source_state"],
                "capture_state": item["capture_state"],
                "report_excerpt": item["report_excerpt"],
                "source_info": {
                    "source_state": _clean_text(source_info.get("source_state")),
                    "capture_state": _clean_text(source_info.get("capture_state")),
                    "redaction": _mapping_dict(source_info.get("redaction")),
                },
            }
        )

    expected = _mapping_dict(fixture.get("expected"))
    critical_false_count = sum(1 for item in support_items if item["critical_false_support"])
    requires_support = _bool(expected.get("requires_support"))
    no_strong_evidence_expected = _bool(expected.get("no_strong_evidence"))
    useful_recall_at_5 = any(item["valid_relevant_support"] for item in support_items)
    required_polarity = _clean_text(expected.get("support_polarity"))
    polarity_ok = True
    if requires_support and required_polarity:
        polarity_ok = any(
            item["valid_relevant_support"] and item["support_polarity"] == required_polarity
            for item in support_items
        )
    no_evidence_honest = not support_items if no_strong_evidence_expected else None
    redaction_ok = _secret_redaction_pass(fixture, support_items, candidate_rows)
    bounded_ok = _bounded_report_pass(fixture, candidate_rows)
    binary_ok = _binary_refusal_pass(fixture, candidate_rows)
    passed = (
        critical_false_count == 0
        and redaction_ok
        and bounded_ok
        and binary_ok
        and (not requires_support or (useful_recall_at_5 and polarity_ok))
        and (not no_strong_evidence_expected or no_evidence_honest is True)
    )
    result = {
        "fixture_id": fixture_id,
        "suite_id": suite_id,
        "suite_kind": _clean_text(fixture.get("suite_kind")) or "synthetic",
        "category": _clean_text(fixture.get("category")) or "unknown",
        "description": _clean_text(fixture.get("description")) or "",
        "passed": passed,
        "expected": {
            key: value
            for key, value in expected.items()
            if key not in {"secret_values"}
        },
        "top_k": top_k,
        "candidate_count": len(materialized),
        "top_candidate_count": len(candidate_rows),
        "support_count": len(support_items),
        "critical_false_support": critical_false_count,
        "useful_recall_at_5": useful_recall_at_5 if requires_support else None,
        "no_evidence_honest": no_evidence_honest,
        "redaction_passed": redaction_ok,
        "bounded_report_passed": bounded_ok,
        "binary_refusal_passed": binary_ok,
        "support_items": support_items,
        "top_candidates": candidate_rows,
        "fixture_workspace": str(fixture_root),
    }
    if not passed:
        result["miss_reason"] = _classify_failed_fixture(result)
    return result


def _ratio(numerator: int, denominator: int, *, empty_value: float = 1.0) -> float:
    if denominator <= 0:
        return empty_value
    return round(numerator / denominator, 4)


def _suite_metrics(results: list[Mapping[str, Any]], *, suite_kind: str) -> dict[str, Any]:
    scoped = [dict(item) for item in results if _clean_text(item.get("suite_kind")) == suite_kind]
    critical_false_support = sum(int(item.get("critical_false_support") or 0) for item in scoped)
    support_items = [
        dict(support)
        for result in scoped
        for support in result.get("support_items") or []
        if isinstance(support, Mapping)
    ]
    valid_relevant_count = sum(1 for item in support_items if item.get("valid_relevant_support"))
    artifact_complete_count = sum(
        1
        for item in support_items
        if int(item.get("artifact_ref_count") or 0) > 0 and _clean_text(item.get("source_state")) is not None
    )
    recall_required = [item for item in scoped if _bool(_mapping_dict(item.get("expected")).get("requires_support"))]
    recall_passed = sum(1 for item in recall_required if item.get("useful_recall_at_5") is True)
    no_evidence_expected = [item for item in scoped if _bool(_mapping_dict(item.get("expected")).get("no_strong_evidence"))]
    no_evidence_passed = sum(1 for item in no_evidence_expected if item.get("no_evidence_honest") is True)
    redaction_required = [item for item in scoped if _bool(_mapping_dict(item.get("expected")).get("requires_redaction"))]
    redaction_passed = sum(1 for item in redaction_required if item.get("redaction_passed") is True)
    failed = [item for item in scoped if not item.get("passed")]
    failed_with_miss = [item for item in failed if _clean_text(item.get("miss_reason")) in MISS_TAXONOMY]
    return {
        "suite_kind": suite_kind,
        "fixture_count": len(scoped),
        "passed_count": sum(1 for item in scoped if item.get("passed")),
        "failed_count": len(failed),
        "critical_false_support": critical_false_support,
        "positive_support_precision": _ratio(valid_relevant_count, len(support_items)),
        "support_item_count": len(support_items),
        "valid_relevant_support_count": valid_relevant_count,
        "useful_recall_at_5": _ratio(recall_passed, len(recall_required)),
        "useful_recall_required_count": len(recall_required),
        "no_evidence_honesty": _ratio(no_evidence_passed, len(no_evidence_expected)),
        "no_evidence_fixture_count": len(no_evidence_expected),
        "source_path_artifact_completeness": _ratio(artifact_complete_count, len(support_items)),
        "redaction_fixture_pass_rate": _ratio(redaction_passed, len(redaction_required)),
        "redaction_fixture_count": len(redaction_required),
        "miss_report_coverage": _ratio(len(failed_with_miss), len(failed)),
        "miss_count": len(failed),
    }


def _dogfood_metrics(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = _suite_metrics(results, suite_kind="dogfood")
    if metrics["fixture_count"] == 0:
        metrics["useful_recall_at_5"] = None
        metrics["dogfood_useful_recall_at_5_status"] = "needs_dogfood_runs"
        metrics["dogfood_holdout_policy"] = "keep_at_least_20_percent_holdout_until_m16"
    elif metrics["fixture_count"] < 20:
        metrics["dogfood_useful_recall_at_5_status"] = "needs_20_runs"
        metrics["dogfood_holdout_policy"] = "keep_at_least_20_percent_holdout_until_m16"
    else:
        metrics["dogfood_useful_recall_at_5_status"] = "pass" if metrics["useful_recall_at_5"] >= 0.30 else "fail"
    return metrics


def _gate_pass(value: Any, op: str, expected: float | int) -> bool:
    if value is None:
        return False
    actual = float(value)
    if op == "==":
        return math.isclose(actual, float(expected))
    if op == ">=":
        return actual >= float(expected)
    raise ValueError(f"Unsupported gate operator `{op}`.")


def _build_exit_gate(metrics: Mapping[str, Any], all_fixtures_passed: bool) -> dict[str, Any]:
    synthetic = _mapping_dict(metrics.get("synthetic"))
    values = {
        "critical_false_support": int(metrics.get("critical_false_support") or 0),
        "synthetic_positive_support_precision": float(synthetic.get("positive_support_precision") or 0.0),
        "synthetic_useful_recall_at_5": float(synthetic.get("useful_recall_at_5") or 0.0),
        "no_evidence_honesty": float(synthetic.get("no_evidence_honesty") or 0.0),
        "miss_report_coverage": float(synthetic.get("miss_report_coverage") or 0.0),
    }
    checks = []
    for name, (op, target) in EXIT_GATES.items():
        checks.append(
            {
                "metric": name,
                "observed": values[name],
                "target": f"{op} {target}",
                "passed": _gate_pass(values[name], op, target),
            }
        )
    return {
        "passed": all(check["passed"] for check in checks) and all_fixtures_passed,
        "all_fixtures_passed": all_fixtures_passed,
        "checks": checks,
    }


def build_review_memory_miss_report(eval_report: Mapping[str, Any]) -> dict[str, Any]:
    results = [dict(item) for item in eval_report.get("fixture_results") or [] if isinstance(item, Mapping)]
    misses = [
        {
            "fixture_id": _clean_text(result.get("fixture_id")) or "",
            "suite_id": _clean_text(result.get("suite_id")) or "",
            "suite_kind": _clean_text(result.get("suite_kind")) or "synthetic",
            "category": _clean_text(result.get("category")) or "unknown",
            "miss_reason": _clean_text(result.get("miss_reason")) or _classify_failed_fixture(result),
            "critical_false_support": int(result.get("critical_false_support") or 0),
            "support_count": int(result.get("support_count") or 0),
            "top_candidates": [
                {
                    "rank": row.get("rank"),
                    "event_id": row.get("event_id"),
                    "support_class": row.get("support_class"),
                    "can_support_decision": row.get("can_support_decision"),
                    "blockers": row.get("blockers"),
                }
                for row in result.get("top_candidates") or []
                if isinstance(row, Mapping)
            ],
        }
        for result in results
        if not result.get("passed")
    ]
    covered = sum(1 for miss in misses if _clean_text(miss.get("miss_reason")) in MISS_TAXONOMY)
    return {
        "schema_name": REVIEW_MEMORY_MISS_REPORT_SCHEMA_NAME,
        "schema_version": REVIEW_MEMORY_MISS_REPORT_SCHEMA_VERSION,
        "workspace_id": _clean_text(eval_report.get("workspace_id")) or DEFAULT_WORKSPACE_ID,
        "generated_at_utc": _clean_text(eval_report.get("generated_at_utc")) or timestamp_utc(),
        "source_eval_run_path": _clean_text(_mapping_dict(eval_report.get("paths")).get("eval_run_path")),
        "miss_count": len(misses),
        "covered_miss_count": covered,
        "miss_report_coverage": _ratio(covered, len(misses)),
        "miss_taxonomy": list(MISS_TAXONOMY),
        "misses": misses,
    }


def _stable_result_digest(report: Mapping[str, Any]) -> str:
    payload = {
        "metrics": report.get("metrics"),
        "exit_gate": report.get("exit_gate"),
        "fixture_results": [
            {
                "fixture_id": item.get("fixture_id"),
                "category": item.get("category"),
                "passed": item.get("passed"),
                "support_count": item.get("support_count"),
                "critical_false_support": item.get("critical_false_support"),
                "useful_recall_at_5": item.get("useful_recall_at_5"),
                "no_evidence_honest": item.get("no_evidence_honest"),
                "redaction_passed": item.get("redaction_passed"),
                "bounded_report_passed": item.get("bounded_report_passed"),
                "binary_refusal_passed": item.get("binary_refusal_passed"),
                "miss_reason": item.get("miss_reason"),
            }
            for item in report.get("fixture_results") or []
            if isinstance(item, Mapping)
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def run_review_memory_eval(
    *,
    suite: str | Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    generated_at_utc: str | None = None,
    write: bool = True,
    spartan: bool = False,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    suite_path = Path(suite).expanduser() if suite is not None else default_review_memory_suite_path(resolved_root)
    if not suite_path.is_absolute():
        suite_path = resolved_root / suite_path
    suite_path = suite_path.resolve()
    generated_at = generated_at_utc or timestamp_utc()
    run_id = _safe_slug(generated_at) if generated_at_utc else timestamp_slug()
    fixtures = load_review_memory_fixtures(suite_path, root=resolved_root)
    fixture_results = [
        _evaluate_fixture(fixture, workspace_id=workspace_id, root=resolved_root, run_id=run_id)
        for fixture in fixtures
    ]
    synthetic = _suite_metrics(fixture_results, suite_kind="synthetic")
    dogfood = _dogfood_metrics(fixture_results)
    metrics = {
        "synthetic": synthetic,
        "dogfood": dogfood,
        "critical_false_support": int(synthetic.get("critical_false_support") or 0)
        + int(dogfood.get("critical_false_support") or 0),
    }
    exit_gate = _build_exit_gate(metrics, all_fixtures_passed=all(bool(item.get("passed")) for item in fixture_results))
    category_counts = fixture_category_counts(fixture_results)
    eval_run = review_memory_eval_run_path(run_id=run_id, workspace_id=workspace_id, root=resolved_root)
    eval_latest = latest_review_memory_eval_path(workspace_id=workspace_id, root=resolved_root)
    eval_run_md = review_memory_eval_markdown_run_path(run_id=run_id, workspace_id=workspace_id, root=resolved_root)
    eval_latest_md = latest_review_memory_eval_markdown_path(workspace_id=workspace_id, root=resolved_root)
    report = {
        "schema_name": REVIEW_MEMORY_EVAL_SCHEMA_NAME,
        "schema_version": REVIEW_MEMORY_EVAL_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "generated_at_utc": generated_at,
        "run_id": run_id,
        "suite_path": str(suite_path),
        "fixture_count": len(fixture_results),
        "fixture_category_counts": category_counts,
        "metrics": metrics,
        "exit_gate": exit_gate,
        "passed": bool(exit_gate.get("passed")),
        "anti_gaming": {
            "synthetic_and_dogfood_reported_separately": True,
            "default_suite_includes_adversarial_negatives": any(
                item.get("category") not in {"true_prior_failure", "true_prior_repair"}
                for item in fixture_results
            ),
            "fixture_names_not_used_for_ranking": True,
            "dogfood_holdout_policy": "keep_at_least_20_percent_holdout_until_m16",
        },
        "local_first_policy": {
            "api_cloud_provider_vector_training_export": False,
            "fixture_sources_materialized_as_local_files": True,
            "training_export_ready": False,
        },
        "spartan": bool(spartan),
        "paths": {
            "eval_latest_path": str(eval_latest),
            "eval_run_path": str(eval_run),
            "eval_latest_markdown_path": str(eval_latest_md),
            "eval_run_markdown_path": str(eval_run_md),
        },
        "fixture_results": fixture_results,
    }
    report["result_digest"] = _stable_result_digest(report)
    miss_report = build_review_memory_miss_report(report)
    miss_run = review_memory_miss_report_run_path(run_id=run_id, workspace_id=workspace_id, root=resolved_root)
    miss_latest = latest_review_memory_miss_report_path(workspace_id=workspace_id, root=resolved_root)
    miss_run_md = review_memory_miss_report_markdown_run_path(run_id=run_id, workspace_id=workspace_id, root=resolved_root)
    miss_latest_md = latest_review_memory_miss_report_markdown_path(workspace_id=workspace_id, root=resolved_root)
    report["paths"].update(
        {
            "miss_report_latest_path": str(miss_latest),
            "miss_report_run_path": str(miss_run),
            "miss_report_latest_markdown_path": str(miss_latest_md),
            "miss_report_run_markdown_path": str(miss_run_md),
        }
    )
    if write:
        markdown = format_review_memory_eval_report(report, spartan=spartan)
        miss_markdown = format_review_memory_miss_report(miss_report)
        write_json(eval_run, report)
        write_json(eval_latest, report)
        eval_run_md.parent.mkdir(parents=True, exist_ok=True)
        eval_latest_md.parent.mkdir(parents=True, exist_ok=True)
        eval_run_md.write_text(markdown, encoding="utf-8")
        eval_latest_md.write_text(markdown, encoding="utf-8")
        write_json(miss_run, miss_report)
        write_json(miss_latest, miss_report)
        miss_run_md.parent.mkdir(parents=True, exist_ok=True)
        miss_latest_md.parent.mkdir(parents=True, exist_ok=True)
        miss_run_md.write_text(miss_markdown, encoding="utf-8")
        miss_latest_md.write_text(miss_markdown, encoding="utf-8")
    return report


def load_latest_review_memory_eval(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> dict[str, Any]:
    path = latest_review_memory_eval_path(workspace_id=workspace_id, root=root)
    if not path.is_file():
        raise ValueError("No latest review memory eval exists. Run `satlab review eval` first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_name") != REVIEW_MEMORY_EVAL_SCHEMA_NAME:
        raise ValueError(f"Unexpected review memory eval payload in `{path}`.")
    return payload


def load_or_build_review_memory_miss_report(
    *,
    latest: bool = True,
    eval_path: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    if eval_path is not None:
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Eval report `{eval_path}` must be a JSON object.")
        return build_review_memory_miss_report(payload)
    if latest:
        miss_path = latest_review_memory_miss_report_path(workspace_id=workspace_id, root=resolved_root)
        if miss_path.is_file():
            payload = json.loads(miss_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("schema_name") == REVIEW_MEMORY_MISS_REPORT_SCHEMA_NAME:
                return payload
        return build_review_memory_miss_report(load_latest_review_memory_eval(workspace_id=workspace_id, root=resolved_root))
    raise ValueError("Use --latest or --eval to select a miss report source.")


def _format_gate_table(report: Mapping[str, Any]) -> list[str]:
    lines = ["| Metric | Observed | Target | Pass |", "|---|---:|---:|---:|"]
    for check in _mapping_dict(report.get("exit_gate")).get("checks") or []:
        if not isinstance(check, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_clean_text(check.get('metric')) or ''}`",
                    str(check.get("observed")),
                    str(check.get("target")),
                    "yes" if check.get("passed") else "no",
                ]
            )
            + " |"
        )
    return lines


def format_review_memory_eval_report(report: Mapping[str, Any], *, spartan: bool = False) -> str:
    all_metrics = _mapping_dict(report.get("metrics"))
    synthetic = _mapping_dict(_mapping_dict(report.get("metrics")).get("synthetic"))
    dogfood = _mapping_dict(_mapping_dict(report.get("metrics")).get("dogfood"))
    dogfood_recall = dogfood.get("useful_recall_at_5")
    dogfood_recall_text = "n/a" if dogfood_recall is None else str(dogfood_recall)
    lines = [
        "# Adversarial Review Memory Benchmark",
        "",
        f"- Result: {'pass' if report.get('passed') else 'fail'}",
        f"- Fixtures: {int(synthetic.get('passed_count') or 0)}/{int(synthetic.get('fixture_count') or 0)} synthetic passed",
        f"- Critical false support: {int(all_metrics.get('critical_false_support') or 0)}",
        f"- Positive support precision: {synthetic.get('positive_support_precision')}",
        f"- Useful Recall@5: {synthetic.get('useful_recall_at_5')}",
        f"- No-evidence honesty: {synthetic.get('no_evidence_honesty')}",
        f"- Miss report coverage: {synthetic.get('miss_report_coverage')}",
        f"- Dogfood Recall@5: {dogfood_recall_text} ({dogfood.get('dogfood_useful_recall_at_5_status')})",
        f"- Training export ready: {str(_mapping_dict(report.get('local_first_policy')).get('training_export_ready')).lower()}",
        "",
        "## Exit Gate",
        "",
        *_format_gate_table(report),
    ]
    failed = [
        dict(item)
        for item in report.get("fixture_results") or []
        if isinstance(item, Mapping) and not item.get("passed")
    ]
    lines.extend(["", "## Failed Cases", ""])
    if failed:
        lines.extend(["| Fixture | Category | Miss reason | Critical false support |", "|---|---|---|---:|"])
        for item in failed:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{_clean_text(item.get('fixture_id')) or ''}`",
                        f"`{_clean_text(item.get('category')) or ''}`",
                        f"`{_clean_text(item.get('miss_reason')) or 'unknown'}`",
                        str(int(item.get("critical_false_support") or 0)),
                    ]
                )
                + " |"
            )
    else:
        lines.append("- None.")
    if not spartan:
        lines.extend(["", "## Cases", "", "| Fixture | Category | Passed | Support | Useful@5 | No evidence honest |", "|---|---|---:|---:|---:|---:|"])
        for item in report.get("fixture_results") or []:
            if not isinstance(item, Mapping):
                continue
            useful = item.get("useful_recall_at_5")
            honest = item.get("no_evidence_honest")
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{_clean_text(item.get('fixture_id')) or ''}`",
                        f"`{_clean_text(item.get('category')) or ''}`",
                        "yes" if item.get("passed") else "no",
                        str(int(item.get("support_count") or 0)),
                        "-" if useful is None else ("yes" if useful else "no"),
                        "-" if honest is None else ("yes" if honest else "no"),
                    ]
                )
                + " |"
            )
    paths = _mapping_dict(report.get("paths"))
    if paths.get("eval_run_path"):
        lines.extend(["", f"Eval artifact: `{paths['eval_run_path']}`"])
    if paths.get("miss_report_run_path"):
        lines.append(f"Miss report: `{paths['miss_report_run_path']}`")
    return "\n".join(lines) + "\n"


def format_review_memory_miss_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Review Memory Miss Report",
        "",
        f"- Misses: {int(report.get('covered_miss_count') or 0)}/{int(report.get('miss_count') or 0)} covered",
        f"- Coverage: {report.get('miss_report_coverage')}",
        "",
    ]
    misses = [dict(item) for item in report.get("misses") or [] if isinstance(item, Mapping)]
    if not misses:
        lines.append("- No failed fixtures.")
        return "\n".join(lines) + "\n"
    lines.extend(["| Fixture | Category | Miss reason | Support | Critical false |", "|---|---|---|---:|---:|"])
    for miss in misses:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_clean_text(miss.get('fixture_id')) or ''}`",
                    f"`{_clean_text(miss.get('category')) or ''}`",
                    f"`{_clean_text(miss.get('miss_reason')) or 'unknown'}`",
                    str(int(miss.get("support_count") or 0)),
                    str(int(miss.get("critical_false_support") or 0)),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run adversarial review memory benchmark fixtures.")
    parser.add_argument("--suite", type=Path, default=None, help="Fixture suite file or directory.")
    parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id.")
    parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")
    parser.add_argument("--spartan", action="store_true", help="Print the compact gate-focused report.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_review_memory_eval(suite=args.suite, workspace_id=args.workspace_id, spartan=args.spartan)
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_review_memory_eval_report(report, spartan=args.spartan))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
