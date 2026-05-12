#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from failure_memory_review import record_file_input, run_evidence_gated_git_review
from workspace_state import DEFAULT_WORKSPACE_ID


REVIEW_BENCHMARK_SCHEMA_NAME = "software-satellite-review-benchmark"
REVIEW_BENCHMARK_SCHEMA_VERSION = 1


def _run_git(root: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "").strip())
    return completed.stdout


def _init_repo(root: Path) -> str:
    _run_git(root, ["init"])
    _run_git(root, ["config", "user.email", "benchmark@example.com"])
    _run_git(root, ["config", "user.name", "Benchmark User"])
    (root / "app.py").write_text("print('base')\n", encoding="utf-8")
    _run_git(root, ["add", "app.py"])
    _run_git(root, ["commit", "-m", "initial"])
    return _run_git(root, ["rev-parse", "HEAD"]).strip()


def _commit_change(root: Path, text: str) -> None:
    (root / "app.py").write_text(text, encoding="utf-8")
    _run_git(root, ["add", "app.py"])
    _run_git(root, ["commit", "-m", "change app"])


def _case_self_recall(*, workspace_id: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        base = _init_repo(root)
        failure = root / "prior-failure.log"
        failure.write_text("prior failure: app.py review memory should not self-recall\n", encoding="utf-8")
        record_file_input(
            input_kind="failure",
            source_path=failure,
            note="Prior app.py self-recall failure",
            workspace_id=workspace_id,
            root=root,
        )
        _commit_change(root, "print('base')\nprint('change')\n")
        metadata, _markdown, _latest_path, _run_path = run_evidence_gated_git_review(
            base=base,
            head="HEAD",
            workspace_id=workspace_id,
            root=root,
        )
        event_id = metadata.get("event_id")
        recalled_ids = {
            item.get("event_id")
            for item in metadata.get("evidence_gate", {}).get("classified_recalled_evidence", [])
            if isinstance(item, dict)
        }
        critical_false = int(metadata.get("evidence_gate", {}).get("critical_false_evidence_count") or 0)
        return {
            "case": "self_recall",
            "passed": event_id not in recalled_ids and critical_false == 0,
            "critical_false_evidence_count": critical_false,
            "details": {
                "active_event_id": event_id,
                "recalled_active_event": event_id in recalled_ids,
            },
        }


def _case_no_prior_evidence(*, workspace_id: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        base = _init_repo(root)
        _commit_change(root, "print('base')\nprint('no prior')\n")
        metadata, markdown, _latest_path, _run_path = run_evidence_gated_git_review(
            base=base,
            head="HEAD",
            workspace_id=workspace_id,
            root=root,
        )
        positive_count = int(metadata.get("evidence_gate", {}).get("positive_count") or 0)
        honest_no_prior = "No source-linked prior evidence" in markdown
        return {
            "case": "no_prior_evidence",
            "passed": positive_count == 0 and honest_no_prior,
            "critical_false_evidence_count": 0 if positive_count == 0 else 1,
            "details": {
                "positive_count": positive_count,
                "honest_no_prior": honest_no_prior,
            },
        }


def _case_secret_redaction(*, workspace_id: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        base = _init_repo(root)
        _commit_change(root, "print('base')\nprint('secret test')\n")
        test_log = root / "test.log"
        secret = "sk-benchmarksecret0000000000000000"
        test_log.write_text(f"FAILED test_app.py\nTOKEN={secret}\n", encoding="utf-8")
        metadata, _markdown, _latest_path, _run_path = run_evidence_gated_git_review(
            base=base,
            head="HEAD",
            test_log=test_log,
            workspace_id=workspace_id,
            root=root,
        )
        intake_path = Path(metadata["git_review"]["intake_run_path"])
        intake = json.loads(intake_path.read_text(encoding="utf-8"))
        snapshot_text = Path(intake["test_log"]["snapshot_path"]).read_text(encoding="utf-8")
        redacted = secret not in snapshot_text and "[REDACTED]" in snapshot_text
        return {
            "case": "secret_redaction",
            "passed": redacted and intake["test_log"]["status"] == "fail",
            "critical_false_evidence_count": 0 if redacted else 1,
            "details": {
                "redacted": redacted,
                "test_status": intake["test_log"]["status"],
            },
        }


def _case_missing_source_not_positive(*, workspace_id: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        base = _init_repo(root)
        failure = root / "missing-source-failure.log"
        failure.write_text("prior failure: app.py missing source must not become positive evidence\n", encoding="utf-8")
        recorded = record_file_input(
            input_kind="failure",
            source_path=failure,
            note="Prior missing-source app.py failure",
            workspace_id=workspace_id,
            root=root,
        )
        Path(recorded["artifact_path"]).unlink()
        _commit_change(root, "print('base')\nprint('missing source fixture')\n")
        metadata, _markdown, _latest_path, _run_path = run_evidence_gated_git_review(
            base=base,
            head="HEAD",
            workspace_id=workspace_id,
            root=root,
        )
        gate = metadata.get("evidence_gate", {})
        missing_rows = [
            item
            for item in gate.get("non_positive_evidence", [])
            if isinstance(item, dict) and item.get("evidence_class") == "missing_source"
        ]
        positive_ids = {
            item.get("event_id")
            for item in gate.get("top_prior_evidence", [])
            if isinstance(item, dict)
        }
        return {
            "case": "missing_source_not_positive",
            "passed": bool(missing_rows) and recorded["event_id"] not in positive_ids,
            "critical_false_evidence_count": 0 if recorded["event_id"] not in positive_ids else 1,
            "details": {
                "missing_source_visible": bool(missing_rows),
                "missing_source_positive": recorded["event_id"] in positive_ids,
            },
        }


def run_review_benchmark(*, workspace_id: str = DEFAULT_WORKSPACE_ID) -> dict[str, Any]:
    cases = [
        _case_self_recall(workspace_id=workspace_id),
        _case_no_prior_evidence(workspace_id=workspace_id),
        _case_secret_redaction(workspace_id=workspace_id),
        _case_missing_source_not_positive(workspace_id=workspace_id),
    ]
    critical_false_evidence_count = sum(int(case.get("critical_false_evidence_count") or 0) for case in cases)
    return {
        "schema_name": REVIEW_BENCHMARK_SCHEMA_NAME,
        "schema_version": REVIEW_BENCHMARK_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "case_count": len(cases),
        "passed_count": sum(1 for case in cases if case.get("passed")),
        "critical_false_evidence_count": critical_false_evidence_count,
        "passed": critical_false_evidence_count == 0 and all(case.get("passed") for case in cases),
        "cases": cases,
        "training_export_ready": False,
    }


def format_review_benchmark_report(report: dict[str, Any]) -> str:
    lines = [
        "# Evidence-Gated Review Benchmark",
        "",
        f"- Cases: {report['passed_count']}/{report['case_count']} passed",
        f"- Critical false evidence: {report['critical_false_evidence_count']}",
        f"- Training export ready: {str(report['training_export_ready']).lower()}",
        "",
        "| Case | Passed | Critical false evidence |",
        "|---|---:|---:|",
    ]
    for case in report["cases"]:
        lines.append(
            f"| `{case['case']}` | {'yes' if case['passed'] else 'no'} | {case['critical_false_evidence_count']} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic evidence-gated review benchmark fixtures.")
    parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id.")
    parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    report = run_review_benchmark(workspace_id=args.workspace_id)
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_review_benchmark_report(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
