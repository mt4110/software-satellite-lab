#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from gemma_runtime import repo_root


REVIEW_MEMORY_FIXTURE_SCHEMA_NAME = "software-satellite-review-memory-fixture-suite"
REVIEW_MEMORY_FIXTURE_SCHEMA_VERSION = 1

FIXTURE_CATEGORIES = (
    "true_prior_failure",
    "true_prior_repair",
    "no_prior_evidence",
    "self_recall_trap",
    "future_evidence_trap",
    "missing_source_trap",
    "modified_source_trap",
    "weak_text_match_trap",
    "contradictory_verdict_trap",
    "agent_claim_trap",
    "secret_redaction",
    "huge_diff",
    "binary_artifact",
)

MISS_TAXONOMY = (
    "lexical_miss",
    "structured_filter_miss",
    "target_identity_miss",
    "source_integrity_blocked",
    "weak_match_overfiltered",
    "contradiction_unresolved",
    "missing_human_signal",
    "fixture_design_error",
    "unknown",
)


def default_review_memory_suite_path(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve() / "examples" / "review_memory_benchmark"


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _mapping_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _resolve_suite_path(path: str | Path | None, *, root: Path | None = None) -> Path:
    resolved_root = Path(root or repo_root()).resolve()
    if path is None:
        return default_review_memory_suite_path(resolved_root)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    return candidate.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Fixture file `{path}` must contain a JSON object.")
    return payload


def _fixture_documents(path: Path) -> list[tuple[Path, dict[str, Any]]]:
    if path.is_file():
        return [(path, _read_json(path))]
    if not path.is_dir():
        raise ValueError(f"Fixture suite path does not exist: `{path}`.")
    docs: list[tuple[Path, dict[str, Any]]] = []
    for json_path in sorted(path.glob("*.json")):
        docs.append((json_path, _read_json(json_path)))
    if not docs:
        raise ValueError(f"Fixture suite directory has no JSON fixtures: `{path}`.")
    return docs


def _normalize_suite_document(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_name") == REVIEW_MEMORY_FIXTURE_SCHEMA_NAME:
        suite = dict(payload)
    elif "fixtures" in payload:
        suite = {
            "schema_name": REVIEW_MEMORY_FIXTURE_SCHEMA_NAME,
            "schema_version": REVIEW_MEMORY_FIXTURE_SCHEMA_VERSION,
            "suite_id": _clean_text(payload.get("suite_id")) or path.stem,
            "suite_kind": _clean_text(payload.get("suite_kind")) or "synthetic",
            "fixtures": _list_of_mappings(payload.get("fixtures")),
        }
    else:
        suite = {
            "schema_name": REVIEW_MEMORY_FIXTURE_SCHEMA_NAME,
            "schema_version": REVIEW_MEMORY_FIXTURE_SCHEMA_VERSION,
            "suite_id": path.stem,
            "suite_kind": _clean_text(payload.get("suite_kind")) or "synthetic",
            "fixtures": [dict(payload)],
        }
    suite["_source_path"] = str(path)
    return suite


def load_review_memory_fixture_suites(path: str | Path | None = None, *, root: Path | None = None) -> list[dict[str, Any]]:
    suite_path = _resolve_suite_path(path, root=root)
    suites = [_normalize_suite_document(source_path, payload) for source_path, payload in _fixture_documents(suite_path)]
    issues = []
    for suite in suites:
        issues.extend(validate_review_memory_fixture_suite(suite))
    if issues:
        raise ValueError("Invalid review memory fixture suite: " + "; ".join(issues[:8]))
    return suites


def load_review_memory_fixtures(path: str | Path | None = None, *, root: Path | None = None) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    for suite in load_review_memory_fixture_suites(path, root=root):
        suite_id = _clean_text(suite.get("suite_id")) or "suite"
        suite_kind = _clean_text(suite.get("suite_kind")) or "synthetic"
        source_path = _clean_text(suite.get("_source_path"))
        for fixture in _list_of_mappings(suite.get("fixtures")):
            fixtures.append(
                {
                    "suite_id": suite_id,
                    "suite_kind": suite_kind,
                    "_source_path": source_path,
                    **fixture,
                }
            )
    return fixtures


def validate_review_memory_fixture_suite(suite: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    if suite.get("schema_name") != REVIEW_MEMORY_FIXTURE_SCHEMA_NAME:
        issues.append("schema_name must be software-satellite-review-memory-fixture-suite")
    if suite.get("schema_version") != REVIEW_MEMORY_FIXTURE_SCHEMA_VERSION:
        issues.append("schema_version must be 1")
    if _clean_text(suite.get("suite_id")) is None:
        issues.append("suite_id is required")
    if _clean_text(suite.get("suite_kind")) not in {"synthetic", "dogfood"}:
        issues.append("suite_kind must be synthetic or dogfood")
    fixtures = _list_of_mappings(suite.get("fixtures"))
    if not fixtures:
        issues.append("fixtures must contain at least one fixture")
    seen_ids: set[str] = set()
    for index, fixture in enumerate(fixtures):
        prefix = f"fixtures[{index}]"
        fixture_id = _clean_text(fixture.get("fixture_id"))
        if fixture_id is None:
            issues.append(f"{prefix}.fixture_id is required")
        elif fixture_id in seen_ids:
            issues.append(f"{prefix}.fixture_id duplicates `{fixture_id}`")
        elif fixture_id:
            seen_ids.add(fixture_id)
        category = _clean_text(fixture.get("category"))
        if category not in FIXTURE_CATEGORIES:
            issues.append(f"{prefix}.category must be one of the M12 fixture categories")
        expected = _mapping_dict(fixture.get("expected"))
        if not expected:
            issues.append(f"{prefix}.expected is required")
        candidates = _list_of_mappings(fixture.get("candidate_events"))
        if category not in {"no_prior_evidence"} and not candidates:
            issues.append(f"{prefix}.candidate_events must not be empty")
        current = _mapping_dict(fixture.get("current_event"))
        if _clean_text(current.get("event_id")) is None:
            issues.append(f"{prefix}.current_event.event_id is required")
    return issues


def fixture_category_counts(fixtures: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for fixture in fixtures:
        category = _clean_text(fixture.get("category")) or "unknown"
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))
