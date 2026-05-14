from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from demand_gate import build_demand_gate_report  # noqa: E402
from release_candidate_checks import (  # noqa: E402
    build_release_candidate_report,
    build_release_demo_report,
    format_release_candidate_report_markdown,
)
from gemma_runtime import timestamp_utc  # noqa: E402
from satlab import main as satlab_main  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _good_release_doc() -> str:
    return """
# v0.1 Release Candidate

## Quickstart

```bash
git clone <repo-url>
cd software-satellite-lab
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/satlab.py release demo --no-api
```

No API key required.

## Security/Privacy Caveats

The default demo performs no network calls and uses public fixtures only.

## Known Limitations

The fixture gate demonstrates behavior; real demand still requires dogfood evidence.
"""


def _good_walkthrough_doc() -> str:
    return """
# Public Demo Walkthrough

```bash
git clone <repo-url>
cd software-satellite-lab
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/satlab.py release demo --no-api
```

No API key required.
"""


def _fixture_metrics() -> dict:
    return {
        "schema_name": "software-satellite-demand-gate-fixture-metrics",
        "schema_version": 1,
        "dogfood_review_sessions": 20,
        "dogfood_agent_session_intakes": 5,
        "external_technical_user_inspections": 3,
        "external_fresh_clone_demo_attempts": 1,
        "dogfood_useful_recall_at_5": 0.35,
        "critical_false_support_count": 0,
        "verdict_capture_seconds": [18, 21, 24, 27, 29],
        "external_exact_pain_recognition": 2,
        "external_wants_to_try": 1,
        "fresh_clone_demo_minutes": 12,
    }


def _make_static_release_root(root: Path) -> None:
    _write(root / "README.md", "# software-satellite-lab\n\nローカルファーストな AI Coding Flight Recorder です。\n")
    _write(root / "README_EN.md", "# software-satellite-lab\n\nA local-first AI Coding Flight Recorder.\n")
    _write(root / "docs" / "release_v0_1_candidate.md", _good_release_doc())
    _write(root / "docs" / "public_demo_walkthrough.md", _good_walkthrough_doc())
    for path in (
        "scripts/satlab.py",
        "scripts/release_candidate_checks.py",
        "scripts/demand_gate.py",
        "tests/test_release_candidate_checks.py",
        "examples/review_memory_benchmark/synthetic_suite.json",
        "examples/agent_session_bundles/generic.json",
        "templates/failure-memory-pack.satellite.yaml",
        "templates/agent-session-pack.satellite.yaml",
    ):
        _write(root / path, "{}\n")
    _write(
        root / "examples" / "demand_gate" / "release_candidate_fixture.json",
        json.dumps(_fixture_metrics(), ensure_ascii=False, indent=2) + "\n",
    )


def _check_status(report: dict, check_id: str) -> str:
    for check in report["checks"]:
        if check["id"] == check_id:
            return check["status"]
    raise AssertionError(f"missing check {check_id}")


class ReleaseCandidateChecksTests(unittest.TestCase):
    def test_release_check_fails_when_private_doc_path_appears(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_static_release_root(root)
            _write(
                root / "docs" / "public_demo_walkthrough.md",
                _good_walkthrough_doc() + "\nDo not depend on .private_docs/secret.md.\n",
            )

            report = build_release_candidate_report(root=root, run_runtime_checks=False)

        self.assertEqual(report["status"], "fail")
        self.assertEqual(_check_status(report, "no_private_docs_required"), "fail")
        self.assertEqual(report["metrics"]["private_doc_dependency_count"], 1)

    def test_release_check_fails_when_api_key_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_static_release_root(root)
            _write(root / "README.md", "# software-satellite-lab\n\nSet OPENAI_API_KEY before running the demo.\n")

            report = build_release_candidate_report(root=root, run_runtime_checks=False)

        self.assertEqual(report["status"], "fail")
        self.assertEqual(_check_status(report, "no_api_key_required_for_demo"), "fail")

    def test_release_check_fails_when_trainable_export_artifact_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_static_release_root(root)
            _write(root / "artifacts" / "training_exports" / "candidate.jsonl", "{\"x\": 1}\n")

            report = build_release_candidate_report(root=root, run_runtime_checks=False)

        self.assertEqual(report["status"], "fail")
        self.assertEqual(_check_status(report, "no_trainable_export_artifacts"), "fail")

    def test_release_check_fails_when_benchmark_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_static_release_root(root)
            stale_benchmark = {
                "passed": True,
                "critical_false_evidence_count": 0,
                "training_export_ready": False,
                "generated_at_utc": "2000-01-01T00:00:00+00:00",
            }

            report = build_release_candidate_report(
                root=root,
                run_runtime_checks=False,
                benchmark_report_override=stale_benchmark,
            )

        self.assertEqual(report["status"], "fail")
        self.assertEqual(_check_status(report, "review_benchmark_passes"), "fail")

    def test_release_demo_produces_markdown_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_static_release_root(root)

            report = build_release_demo_report(root=root, no_api=True, write=True, run_runtime_checks=False)
            report_path = Path(report["paths"]["report_latest_md_path"])
            report_path_exists = report_path.is_file()
            markdown = report_path.read_text(encoding="utf-8")

        self.assertTrue(report_path_exists)
        self.assertTrue(report["markdown_report_exists"])
        self.assertTrue(report["markdown_report_rendered"])
        self.assertIn("# Public Demo Walkthrough", markdown)
        self.assertFalse(report["guardrails"]["requires_api_key"])
        self.assertFalse(report["guardrails"]["uses_network"])

    def test_release_demo_no_write_reports_rendered_without_claiming_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_static_release_root(root)

            report = build_release_demo_report(root=root, no_api=True, write=False, run_runtime_checks=False)
            report_path = Path(report["paths"]["report_latest_md_path"])
            report_path_exists = report_path.exists()

        self.assertFalse(report["markdown_report_exists"])
        self.assertTrue(report["markdown_report_rendered"])
        self.assertFalse(report_path_exists)

    def test_static_release_check_does_not_claim_unwritten_reports_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_static_release_root(root)

            report = build_release_candidate_report(root=root, run_runtime_checks=False)

        demo_detail = next(check["detail"] for check in report["checks"] if check["id"] == "release_demo_markdown_report")
        demand_detail = next(check["detail"] for check in report["checks"] if check["id"] == "demand_gate_report_exists")
        self.assertFalse(demo_detail["markdown_report_exists"])
        self.assertTrue(demo_detail["markdown_report_rendered"])
        self.assertTrue(demo_detail["not_run_static_only"])
        self.assertFalse(demand_detail["report_exists"])
        self.assertTrue(demand_detail["report_rendered"])
        self.assertTrue(demand_detail["not_run_static_only"])

    def test_release_check_no_write_keeps_spartan_benchmark_from_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_static_release_root(root)
            generated_at = timestamp_utc()
            with (
                mock.patch(
                    "release_candidate_checks.run_review_benchmark",
                    return_value={
                        "passed": True,
                        "critical_false_evidence_count": 0,
                        "training_export_ready": False,
                        "generated_at_utc": generated_at,
                    },
                ),
                mock.patch(
                    "release_candidate_checks.run_review_memory_eval",
                    return_value={
                        "passed": True,
                        "metrics": {"critical_false_support": 0},
                        "generated_at_utc": generated_at,
                    },
                ) as eval_mock,
                mock.patch(
                    "release_candidate_checks.build_evidence_lint_report",
                    return_value={"verdict": "pass", "issue_count": 0},
                ),
                mock.patch(
                    "release_candidate_checks.audit_evidence_pack_v1_path",
                    return_value=({"verdict": "pass"}, None, None),
                ),
            ):
                report = build_release_candidate_report(
                    root=root,
                    run_runtime_checks=True,
                    write_subreports=False,
                )

        self.assertEqual(report["status"], "pass")
        self.assertFalse(eval_mock.call_args.kwargs["write"])

    def test_demand_gate_fails_below_thresholds(self) -> None:
        report = build_demand_gate_report(
            metrics_override={
                "dogfood_review_sessions": 19,
                "dogfood_agent_session_intakes": 4,
                "external_technical_user_inspections": 3,
                "external_fresh_clone_demo_attempts": 1,
                "dogfood_useful_recall_at_5": 0.19,
                "critical_false_support_count": 1,
                "verdict_capture_median_seconds": 31,
                "external_exact_pain_recognition": 1,
                "external_wants_to_try": 0,
                "fresh_clone_demo_minutes": 16,
            }
        )

        self.assertEqual(report["status"], "fail")
        self.assertIn("critical_false_support_count", report["blockers"])
        self.assertIn("dogfood_review_sessions", report["blockers"])

    def test_demand_gate_passes_with_fixture_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture = root / "examples" / "demand_gate" / "release_candidate_fixture.json"
            _write(fixture, json.dumps(_fixture_metrics(), ensure_ascii=False, indent=2) + "\n")

            report = build_demand_gate_report(root=root, fixture_metrics=fixture)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["metrics"]["verdict_capture_median_seconds"], 24.0)

    def test_no_raw_secrets_in_release_report_excerpt(self) -> None:
        markdown = format_release_candidate_report_markdown(
            {
                "status": "fail",
                "strict": False,
                "generated_at_utc": "2026-05-14T00:00:00+00:00",
                "checks": [
                    {
                        "id": "probe",
                        "label": "Probe",
                        "status": "fail",
                        "detail": {"excerpt": "TOKEN=sk-testsecret000000000000000"},
                    }
                ],
                "failing_check_ids": ["probe"],
            }
        )

        self.assertNotIn("sk-testsecret", markdown)
        self.assertIn("[REDACTED]", markdown)

    def test_satlab_demand_gate_cli_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_static_release_root(root)
            demand_stdout = io.StringIO()
            with redirect_stdout(demand_stdout):
                demand_exit = satlab_main(
                    [
                        "--root",
                        str(root),
                        "demand",
                        "gate",
                        "--fixture-metrics",
                        "examples/demand_gate/release_candidate_fixture.json",
                        "--no-write",
                        "--format",
                        "json",
                    ]
                )
            demand_payload = json.loads(demand_stdout.getvalue())

        self.assertEqual(demand_exit, 0)
        self.assertEqual(demand_payload["status"], "pass")


if __name__ == "__main__":
    unittest.main()
