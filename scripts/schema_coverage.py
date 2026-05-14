#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from gemma_runtime import repo_root, timestamp_utc


SCHEMA_COVERAGE_SCHEMA_NAME = "software-satellite-schema-coverage"
SCHEMA_COVERAGE_SCHEMA_VERSION = 1
CORE_COVERAGE_THRESHOLD = 0.90
DEFAULT_SCHEMA_NOTES_PATH = "docs/software_work_event_schema_notes.md"


@dataclass(frozen=True)
class SchemaCandidate:
    schema_id: str
    title: str
    schema_path: str | None
    runtime_paths: tuple[str, ...]
    schema_name: str
    fallback_required_fields: tuple[str, ...] = ()
    fallback_optional_fields: tuple[str, ...] = ()


CORE_SCHEMA_CANDIDATES: tuple[SchemaCandidate, ...] = (
    SchemaCandidate(
        schema_id="software_work_event",
        title="Software work event",
        schema_path=None,
        runtime_paths=("scripts/software_work_events.py",),
        schema_name="software-satellite-event",
        fallback_required_fields=(
            "schema_name",
            "schema_version",
            "event_id",
            "event_kind",
            "recorded_at_utc",
            "workspace",
            "session",
            "outcome",
            "content",
            "source_refs",
            "tags",
        ),
        fallback_optional_fields=("target_paths", "quality_status", "artifact_vault_refs", "note"),
    ),
    SchemaCandidate(
        schema_id="artifact_ref",
        title="Artifact ref",
        schema_path="schemas/artifact_ref.schema.json",
        runtime_paths=("scripts/artifact_vault.py", "scripts/artifact_schema.py"),
        schema_name="software-satellite-artifact-ref",
    ),
    SchemaCandidate(
        schema_id="human_verdict",
        title="Human verdict",
        schema_path=None,
        runtime_paths=("scripts/failure_memory_review.py",),
        schema_name="software-satellite-human-verdict-record",
        fallback_required_fields=(
            "schema_name",
            "schema_version",
            "workspace_id",
            "recorded_at_utc",
            "event_id",
            "verdict",
            "reason",
            "signal",
        ),
        fallback_optional_fields=("target_event_id", "relation_kind", "follow_up", "recall_usefulness"),
    ),
    SchemaCandidate(
        schema_id="evidence_support_result",
        title="Evidence support result",
        schema_path="schemas/evidence_support.schema.json",
        runtime_paths=("scripts/evidence_support.py",),
        schema_name="software-satellite-evidence-support-result",
    ),
    SchemaCandidate(
        schema_id="review_memory_fixture",
        title="Review memory fixture",
        schema_path="schemas/review_memory_fixture.schema.json",
        runtime_paths=("scripts/review_memory_fixtures.py", "scripts/review_memory_eval.py"),
        schema_name="software-satellite-review-memory-fixture-suite",
    ),
    SchemaCandidate(
        schema_id="agent_session_bundle",
        title="Agent session bundle",
        schema_path="schemas/agent_session_bundle.schema.json",
        runtime_paths=("scripts/agent_session_intake.py",),
        schema_name="software-satellite-agent-session-bundle",
    ),
    SchemaCandidate(
        schema_id="satellite_evidence_pack_v1",
        title="Satellite Evidence Pack v1",
        schema_path="schemas/satellite_evidence_pack_v1.schema.json",
        runtime_paths=("scripts/evidence_pack_v1.py",),
        schema_name="software-satellite-evidence-pack-v1",
    ),
)

SECTION_MARKERS = {
    "required_fields": ("required fields", "required field"),
    "optional_fields": ("optional fields", "optional field"),
    "compatibility_policy": ("compatibility policy", "compatibility"),
    "privacy_considerations": ("privacy considerations", "privacy"),
    "examples": ("examples", "example"),
    "known_gaps": ("known gaps", "gaps"),
}

CRITERION_WEIGHTS = {
    "source_available": 0.20,
    "required_fields": 0.20,
    "optional_fields": 0.15,
    "compatibility_policy": 0.15,
    "privacy_considerations": 0.15,
    "examples": 0.10,
    "known_gaps": 0.05,
}


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _schema_fields(schema: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    required = [str(item) for item in schema.get("required") or [] if isinstance(item, str)]
    properties = schema.get("properties")
    property_names = sorted(str(key) for key in properties.keys()) if isinstance(properties, Mapping) else []
    optional = [name for name in property_names if name not in set(required)]
    return sorted(required), optional


def _section_for_schema(notes_text: str, schema_id: str) -> str:
    if not notes_text.strip():
        return ""
    heading_re = re.compile(rf"^##\s+`?{re.escape(schema_id)}`?\s*$", re.MULTILINE)
    match = heading_re.search(notes_text)
    if match is None:
        return ""
    next_match = re.search(r"^##\s+", notes_text[match.end() :], re.MULTILINE)
    end = match.end() + next_match.start() if next_match else len(notes_text)
    return notes_text[match.start() : end]


def _section_has_marker(section: str, marker_id: str) -> bool:
    lowered = section.lower()
    return any(marker in lowered for marker in SECTION_MARKERS[marker_id])


def _criterion(passed: bool, evidence: str) -> dict[str, Any]:
    return {"passed": bool(passed), "evidence": evidence}


def _candidate_report(
    candidate: SchemaCandidate,
    *,
    root: Path,
    notes_text: str,
) -> dict[str, Any]:
    schema_path = root / candidate.schema_path if candidate.schema_path is not None else None
    schema = _read_json_mapping(schema_path) if schema_path is not None and schema_path.is_file() else {}
    required_fields, optional_fields = _schema_fields(schema)
    if not required_fields:
        required_fields = sorted(candidate.fallback_required_fields)
    if not optional_fields:
        optional_fields = sorted(candidate.fallback_optional_fields)

    runtime_sources = [path for path in candidate.runtime_paths if (root / path).is_file()]
    schema_exists = schema_path is not None and schema_path.is_file()
    section = _section_for_schema(notes_text, candidate.schema_id)
    criteria = {
        "source_available": _criterion(
            schema_exists or bool(runtime_sources),
            candidate.schema_path or ", ".join(runtime_sources) or "missing",
        ),
        "required_fields": _criterion(
            bool(required_fields) and _section_has_marker(section, "required_fields"),
            ", ".join(required_fields[:12]) or "missing",
        ),
        "optional_fields": _criterion(
            _section_has_marker(section, "optional_fields"),
            ", ".join(optional_fields[:12]) or "documented as strict or nested-only",
        ),
        "compatibility_policy": _criterion(
            _section_has_marker(section, "compatibility_policy"),
            DEFAULT_SCHEMA_NOTES_PATH,
        ),
        "privacy_considerations": _criterion(
            _section_has_marker(section, "privacy_considerations"),
            DEFAULT_SCHEMA_NOTES_PATH,
        ),
        "examples": _criterion(_section_has_marker(section, "examples"), DEFAULT_SCHEMA_NOTES_PATH),
        "known_gaps": _criterion(_section_has_marker(section, "known_gaps"), DEFAULT_SCHEMA_NOTES_PATH),
    }
    score = sum(CRITERION_WEIGHTS[key] for key, value in criteria.items() if value["passed"])
    missing = [key for key, value in criteria.items() if not value["passed"]]
    return {
        "schema_id": candidate.schema_id,
        "title": candidate.title,
        "schema_name": candidate.schema_name,
        "schema_path": candidate.schema_path,
        "runtime_sources": runtime_sources,
        "required_fields": required_fields,
        "optional_fields": optional_fields,
        "criteria": criteria,
        "coverage_score": round(score, 3),
        "missing_criteria": missing,
    }


def build_schema_coverage_report(
    *,
    root: Path | None = None,
    generated_at_utc: str | None = None,
    threshold: float = CORE_COVERAGE_THRESHOLD,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    notes_text = _read_text(resolved_root / DEFAULT_SCHEMA_NOTES_PATH)
    candidates = [
        _candidate_report(candidate, root=resolved_root, notes_text=notes_text)
        for candidate in CORE_SCHEMA_CANDIDATES
    ]
    coverage_ratio = round(
        sum(float(candidate["coverage_score"]) for candidate in candidates) / max(1, len(candidates)),
        3,
    )
    missing = {
        str(candidate["schema_id"]): list(candidate["missing_criteria"])
        for candidate in candidates
        if candidate["missing_criteria"]
    }
    return {
        "schema_name": SCHEMA_COVERAGE_SCHEMA_NAME,
        "schema_version": SCHEMA_COVERAGE_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc or timestamp_utc(),
        "threshold": threshold,
        "core_coverage_ratio": coverage_ratio,
        "passed": coverage_ratio >= threshold,
        "schema_count": len(candidates),
        "core_schema_ids": [candidate.schema_id for candidate in CORE_SCHEMA_CANDIDATES],
        "candidates": candidates,
        "missing": missing,
        "notes_path": DEFAULT_SCHEMA_NOTES_PATH,
    }


def format_schema_coverage_report(report: Mapping[str, Any]) -> str:
    threshold = report.get("threshold")
    if threshold is None:
        threshold = CORE_COVERAGE_THRESHOLD
    lines = [
        "# Schema Coverage Report",
        "",
        f"- Status: `{'pass' if report.get('passed') else 'fail'}`",
        f"- Core coverage: `{float(report.get('core_coverage_ratio') or 0.0):.2f}`",
        f"- Threshold: `{float(threshold):.2f}`",
        f"- Schemas: `{int(report.get('schema_count') or 0)}`",
        "",
        "| Schema | Coverage | Source | Required | Optional | Compat | Privacy | Examples | Gaps |",
        "|---|---:|---|---|---|---|---|---|---|",
    ]
    for candidate in report.get("candidates") or []:
        if not isinstance(candidate, Mapping):
            continue
        criteria = candidate.get("criteria") if isinstance(candidate.get("criteria"), Mapping) else {}
        source = candidate.get("schema_path") or ", ".join(candidate.get("runtime_sources") or []) or "missing"

        def mark(key: str) -> str:
            item = criteria.get(key) if isinstance(criteria, Mapping) else {}
            return "yes" if isinstance(item, Mapping) and item.get("passed") else "no"

        lines.append(
            "| "
            f"`{candidate.get('schema_id')}` | "
            f"{float(candidate.get('coverage_score') or 0.0):.2f} | "
            f"{source} | "
            f"{mark('required_fields')} | "
            f"{mark('optional_fields')} | "
            f"{mark('compatibility_policy')} | "
            f"{mark('privacy_considerations')} | "
            f"{mark('examples')} | "
            f"{mark('known_gaps')} |"
        )
    missing = report.get("missing") if isinstance(report.get("missing"), Mapping) else {}
    if missing:
        lines.extend(["", "## Missing Criteria", ""])
        for schema_id, criteria in sorted(missing.items()):
            joined = ", ".join(str(item) for item in criteria)
            lines.append(f"- `{schema_id}`: {joined}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report M17 core schema standardization coverage.")
    parser.add_argument("--root", type=Path, default=None, help="Optional repo root override.")
    parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = build_schema_coverage_report(root=args.root)
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_schema_coverage_report(report), end="")
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
