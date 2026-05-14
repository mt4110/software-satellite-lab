#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from demand_validation import (
    build_demand_validation_report,
    demo_setup_latest_path,
    read_dogfood_runs,
    read_interviews,
)
from gemma_runtime import repo_root, timestamp_slug, timestamp_utc, write_json
from software_work_events import iter_agent_session_intake_events
from workspace_state import DEFAULT_WORKSPACE_ID


DEMAND_GATE_SCHEMA_NAME = "software-satellite-demand-gate-report"
DEMAND_GATE_SCHEMA_VERSION = 1
DEMAND_GATE_FIXTURE_SCHEMA_NAME = "software-satellite-demand-gate-fixture-metrics"
DEMAND_GATE_FIXTURE_SCHEMA_VERSION = 1

MIN_DOGFOOD_REVIEW_SESSIONS = 20
MIN_DOGFOOD_AGENT_SESSION_INTAKES = 5
MIN_EXTERNAL_TECHNICAL_INSPECTIONS = 3
MIN_EXTERNAL_FRESH_CLONE_ATTEMPTS = 1
MIN_USEFUL_RECALL_AT_5 = 0.30
MAX_CRITICAL_FALSE_SUPPORT = 0
MAX_VERDICT_CAPTURE_MEDIAN_SECONDS = 30.0
MIN_EXTERNAL_EXACT_PAIN_RECOGNITION = 2
MIN_EXTERNAL_WANTS_TO_TRY = 1
MAX_FRESH_CLONE_DEMO_MINUTES = 15.0


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _mapping_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


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


def _median_from_values(values: Iterable[Any]) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if not numbers:
        return None
    return round(float(median(numbers)), 3)


def demand_gate_root(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return _resolve_root(root) / "artifacts" / "demand_gate" / workspace_id


def demand_gate_report_latest_json_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return demand_gate_root(workspace_id=workspace_id, root=root) / "reports" / "latest.json"


def demand_gate_report_latest_md_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return demand_gate_root(workspace_id=workspace_id, root=root) / "reports" / "latest.md"


def demand_gate_report_run_json_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    run_slug: str | None = None,
) -> Path:
    slug = run_slug or timestamp_slug()
    return demand_gate_root(workspace_id=workspace_id, root=root) / "reports" / "runs" / f"{slug}-demand-gate.json"


def demand_gate_report_run_md_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    run_slug: str | None = None,
) -> Path:
    slug = run_slug or timestamp_slug()
    return demand_gate_root(workspace_id=workspace_id, root=root) / "reports" / "runs" / f"{slug}-demand-gate.md"


def _read_json_mapping(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else None


def _load_fixture_metrics(path: Path, *, root: Path) -> dict[str, Any]:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    payload = _read_json_mapping(candidate.resolve())
    if payload is None:
        raise ValueError(f"Demand gate fixture metrics are not readable: `{candidate}`.")
    if payload.get("schema_name") != DEMAND_GATE_FIXTURE_SCHEMA_NAME:
        raise ValueError(f"Demand gate fixture metrics have an unexpected schema: `{candidate}`.")
    if payload.get("schema_version") != DEMAND_GATE_FIXTURE_SCHEMA_VERSION:
        raise ValueError(f"Demand gate fixture metrics have an unexpected schema version: `{candidate}`.")
    payload["_fixture_path"] = str(candidate.resolve())
    return payload


def _metrics_from_fixture(fixture: Mapping[str, Any]) -> dict[str, Any]:
    verdict_median = _number(fixture.get("verdict_capture_median_seconds"))
    if verdict_median is None and isinstance(fixture.get("verdict_capture_seconds"), list):
        verdict_median = _median_from_values(fixture.get("verdict_capture_seconds") or [])
    return {
        "source": "fixture_metrics",
        "fixture_path": _clean_text(fixture.get("_fixture_path")),
        "dogfood_review_sessions": _integer(fixture.get("dogfood_review_sessions")),
        "dogfood_agent_session_intakes": _integer(fixture.get("dogfood_agent_session_intakes")),
        "external_technical_user_inspections": _integer(fixture.get("external_technical_user_inspections")),
        "external_fresh_clone_demo_attempts": _integer(fixture.get("external_fresh_clone_demo_attempts")),
        "dogfood_useful_recall_at_5": _number(fixture.get("dogfood_useful_recall_at_5")),
        "critical_false_support_count": _integer(fixture.get("critical_false_support_count")),
        "verdict_capture_median_seconds": verdict_median,
        "external_exact_pain_recognition": _integer(fixture.get("external_exact_pain_recognition")),
        "external_wants_to_try": _integer(fixture.get("external_wants_to_try")),
        "fresh_clone_demo_minutes": _number(fixture.get("fresh_clone_demo_minutes")),
    }


def _metrics_from_local_artifacts(*, workspace_id: str, root: Path) -> dict[str, Any]:
    demand_report = build_demand_validation_report(workspace_id=workspace_id, root=root)
    dogfood_runs = read_dogfood_runs(workspace_id=workspace_id, root=root)
    interviews = read_interviews(workspace_id=workspace_id, root=root)
    setup = _read_json_mapping(demo_setup_latest_path(workspace_id=workspace_id, root=root))
    intakes = iter_agent_session_intake_events(workspace_id=workspace_id, root=root)

    useful_count = _positive_judgement_count(dogfood_runs, "useful_recall_judgement", "useful")
    useful_judged_count = _judged_count(dogfood_runs, "useful_recall_judgement", "useful")
    useful_rate = round(useful_count / useful_judged_count, 4) if useful_judged_count else None
    verdict_median = _median_from_values(run.get("verdict_capture_seconds") for run in dogfood_runs)
    recognized_pain_count = _positive_judgement_count(interviews, "recognized_pain", "recognized")
    wants_try_count = _positive_judgement_count(interviews, "wants_to_try", "wants_try")
    clone_minutes = (
        float(setup["clone_to_demo_minutes"])
        if isinstance(setup, Mapping) and isinstance(setup.get("clone_to_demo_minutes"), (int, float))
        else None
    )

    return {
        "source": "local_artifacts",
        "demand_validation_status": demand_report.get("overall_status"),
        "dogfood_review_sessions": len(dogfood_runs),
        "dogfood_agent_session_intakes": len(intakes),
        "external_technical_user_inspections": len(interviews),
        "external_fresh_clone_demo_attempts": 1 if setup is not None else 0,
        "dogfood_useful_recall_at_5": useful_rate,
        "critical_false_support_count": sum(int(run.get("critical_false_evidence_count") or 0) for run in dogfood_runs),
        "verdict_capture_median_seconds": verdict_median,
        "external_exact_pain_recognition": recognized_pain_count,
        "external_wants_to_try": wants_try_count,
        "fresh_clone_demo_minutes": clone_minutes,
    }


def _criterion(
    *,
    key: str,
    label: str,
    operator: str,
    target: int | float,
    observed: int | float | None,
) -> dict[str, Any]:
    if observed is None:
        status = "needs_data"
    elif operator == ">=":
        status = "pass" if observed >= target else "fail"
    elif operator == "<=":
        status = "pass" if observed <= target else "fail"
    elif operator == "==":
        status = "pass" if observed == target else "fail"
    else:
        raise ValueError(f"Unsupported demand gate operator `{operator}`.")
    return {
        "key": key,
        "label": label,
        "operator": operator,
        "target": target,
        "observed": observed,
        "status": status,
    }


def _criteria_from_metrics(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        _criterion(
            key="dogfood_review_sessions",
            label="Dogfood review sessions",
            operator=">=",
            target=MIN_DOGFOOD_REVIEW_SESSIONS,
            observed=_integer(metrics.get("dogfood_review_sessions")),
        ),
        _criterion(
            key="dogfood_agent_session_intakes",
            label="Dogfood agent-session intakes",
            operator=">=",
            target=MIN_DOGFOOD_AGENT_SESSION_INTAKES,
            observed=_integer(metrics.get("dogfood_agent_session_intakes")),
        ),
        _criterion(
            key="external_technical_user_inspections",
            label="External technical-user inspections",
            operator=">=",
            target=MIN_EXTERNAL_TECHNICAL_INSPECTIONS,
            observed=_integer(metrics.get("external_technical_user_inspections")),
        ),
        _criterion(
            key="external_fresh_clone_demo_attempts",
            label="External fresh-clone demo attempts",
            operator=">=",
            target=MIN_EXTERNAL_FRESH_CLONE_ATTEMPTS,
            observed=_integer(metrics.get("external_fresh_clone_demo_attempts")),
        ),
        _criterion(
            key="dogfood_useful_recall_at_5",
            label="Useful recall at 5",
            operator=">=",
            target=MIN_USEFUL_RECALL_AT_5,
            observed=_number(metrics.get("dogfood_useful_recall_at_5")),
        ),
        _criterion(
            key="critical_false_support_count",
            label="Critical false support",
            operator="==",
            target=MAX_CRITICAL_FALSE_SUPPORT,
            observed=_integer(metrics.get("critical_false_support_count")),
        ),
        _criterion(
            key="verdict_capture_median_seconds",
            label="Verdict capture median seconds",
            operator="<=",
            target=MAX_VERDICT_CAPTURE_MEDIAN_SECONDS,
            observed=_number(metrics.get("verdict_capture_median_seconds")),
        ),
        _criterion(
            key="external_exact_pain_recognition",
            label="External exact-pain recognition",
            operator=">=",
            target=MIN_EXTERNAL_EXACT_PAIN_RECOGNITION,
            observed=_integer(metrics.get("external_exact_pain_recognition")),
        ),
        _criterion(
            key="external_wants_to_try",
            label="External wants-to-try",
            operator=">=",
            target=MIN_EXTERNAL_WANTS_TO_TRY,
            observed=_integer(metrics.get("external_wants_to_try")),
        ),
        _criterion(
            key="fresh_clone_demo_minutes",
            label="Fresh-clone demo minutes",
            operator="<=",
            target=MAX_FRESH_CLONE_DEMO_MINUTES,
            observed=_number(metrics.get("fresh_clone_demo_minutes")),
        ),
    ]


def build_demand_gate_report(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    fixture_metrics: Path | None = None,
    metrics_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    if metrics_override is not None:
        metrics = dict(metrics_override)
        metrics.setdefault("source", "override")
    elif fixture_metrics is not None:
        metrics = _metrics_from_fixture(_load_fixture_metrics(fixture_metrics, root=resolved_root))
    else:
        metrics = _metrics_from_local_artifacts(workspace_id=workspace_id, root=resolved_root)

    criteria = _criteria_from_metrics(metrics)
    statuses = [criterion["status"] for criterion in criteria]
    if any(status == "fail" for status in statuses):
        status = "fail"
    elif any(status == "needs_data" for status in statuses):
        status = "needs_data"
    else:
        status = "pass"

    blockers = [
        criterion["key"]
        for criterion in criteria
        if criterion["status"] != "pass"
    ]
    next_actions = _next_actions(criteria)
    latest_json = demand_gate_report_latest_json_path(workspace_id=workspace_id, root=resolved_root)
    latest_md = demand_gate_report_latest_md_path(workspace_id=workspace_id, root=resolved_root)
    run_slug = timestamp_slug()
    run_json = demand_gate_report_run_json_path(workspace_id=workspace_id, root=resolved_root, run_slug=run_slug)
    run_md = demand_gate_report_run_md_path(workspace_id=workspace_id, root=resolved_root, run_slug=run_slug)

    return {
        "schema_name": DEMAND_GATE_SCHEMA_NAME,
        "schema_version": DEMAND_GATE_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "generated_at_utc": timestamp_utc(),
        "status": status,
        "metrics_source": metrics.get("source"),
        "metrics": dict(metrics),
        "criteria": criteria,
        "blockers": blockers,
        "next_actions": next_actions,
        "guardrails": {
            "local_first": True,
            "file_first": True,
            "requires_api_key": False,
            "uses_network": False,
            "writes_training_data": False,
            "uses_private_docs": False,
        },
        "paths": {
            "report_latest_json_path": str(latest_json),
            "report_latest_md_path": str(latest_md),
            "report_run_json_path": str(run_json),
            "report_run_md_path": str(run_md),
        },
    }


def _next_actions(criteria: Iterable[Mapping[str, Any]]) -> list[str]:
    actions: list[str] = []
    for criterion in criteria:
        if criterion.get("status") == "pass":
            continue
        key = criterion.get("key")
        observed = criterion.get("observed")
        target = criterion.get("target")
        if key == "dogfood_review_sessions":
            actions.append(f"Record dogfood review sessions until the count reaches {target}; current count is {observed}.")
        elif key == "dogfood_agent_session_intakes":
            actions.append(f"Record agent-session intakes until the count reaches {target}; current count is {observed}.")
        elif key == "external_technical_user_inspections":
            actions.append(f"Run external technical-user inspections until the count reaches {target}.")
        elif key == "external_fresh_clone_demo_attempts":
            actions.append("Record at least one external fresh-clone demo attempt.")
        elif key == "dogfood_useful_recall_at_5":
            actions.append("Improve or narrow recall before release messaging; useful recall is below the demand threshold.")
        elif key == "critical_false_support_count":
            actions.append("Pause release messaging and inspect every critical false-support case.")
        elif key == "verdict_capture_median_seconds":
            actions.append("Reduce verdict capture friction before asking external users to repeat the flow.")
        elif key == "external_exact_pain_recognition":
            actions.append("Keep interviewing until at least two external users recognize the exact pain.")
        elif key == "external_wants_to_try":
            actions.append("Find at least one external user who wants to try the local demo on a repo.")
        elif key == "fresh_clone_demo_minutes":
            actions.append("Trim the fresh-clone demo path to fifteen minutes or less.")
    return actions


def format_demand_gate_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Demand Validation Gate",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Metrics source: `{report.get('metrics_source')}`",
        f"- Generated: `{report.get('generated_at_utc')}`",
        "",
        "| Criterion | Target | Observed | Status |",
        "|---|---:|---:|---|",
    ]
    for criterion in report.get("criteria") or []:
        if not isinstance(criterion, Mapping):
            continue
        target = f"{criterion.get('operator')} {criterion.get('target')}"
        observed = criterion.get("observed")
        lines.append(
            f"| {criterion.get('label')} | {target} | {observed if observed is not None else 'n/a'} | "
            f"`{criterion.get('status')}` |"
        )
    actions = [str(action) for action in report.get("next_actions") or []]
    if actions:
        lines.extend(["", "## Next Actions", ""])
        lines.extend(f"- {action}" for action in actions)
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Local files are the source of truth.",
            "- No API key or network call is used by the gate.",
            "- The gate writes reports only; it does not export trainable data.",
        ]
    )
    return "\n".join(lines) + "\n"


def record_demand_gate_report(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    fixture_metrics: Path | None = None,
    metrics_override: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str, Path, Path, Path, Path]:
    report = build_demand_gate_report(
        workspace_id=workspace_id,
        root=root,
        fixture_metrics=fixture_metrics,
        metrics_override=metrics_override,
    )
    markdown = format_demand_gate_report(report)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the M16 demand validation gate from local evidence.")
    parser.add_argument("--root", type=Path, default=None, help="Optional repo root override.")
    parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id.")
    parser.add_argument("--fixture-metrics", type=Path, default=None, help="Optional public fixture metrics JSON.")
    parser.add_argument("--no-write", action="store_true", help="Print only; do not persist report artifacts.")
    parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.no_write:
            report = build_demand_gate_report(
                workspace_id=args.workspace_id,
                root=args.root,
                fixture_metrics=args.fixture_metrics,
            )
            markdown = format_demand_gate_report(report)
        else:
            report, markdown, _latest_json, _latest_md, _run_json, _run_md = record_demand_gate_report(
                workspace_id=args.workspace_id,
                root=args.root,
                fixture_metrics=args.fixture_metrics,
            )
    except ValueError as exc:
        parser.error(str(exc))
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(markdown)
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
