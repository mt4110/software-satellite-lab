from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from review_memory_eval import build_review_memory_miss_report, run_review_memory_eval  # noqa: E402
from satlab import main as satlab_main  # noqa: E402


DEFAULT_SUITE = REPO_ROOT / "examples" / "review_memory_benchmark"


def _suite(fixtures: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_name": "software-satellite-review-memory-fixture-suite",
        "schema_version": 1,
        "suite_id": "test-suite",
        "suite_kind": "synthetic",
        "fixtures": fixtures,
    }


def _write_suite(root: Path, payload: dict[str, object]) -> Path:
    path = root / "suite.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


class ReviewMemoryEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.report = run_review_memory_eval(
            suite=DEFAULT_SUITE,
            workspace_id="review-memory-test",
            root=cls.root,
            generated_at_utc="2026-05-12T12:00:00+00:00",
        )
        cls.by_category = {result["category"]: result for result in cls.report["fixture_results"]}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_self_recall_trap_produces_zero_support(self) -> None:
        result = self.by_category["self_recall_trap"]

        self.assertTrue(result["passed"])
        self.assertEqual(result["support_count"], 0)
        self.assertEqual(result["top_candidates"][0]["support_class"], "current_review_subject")

    def test_no_prior_fixture_says_no_strong_evidence(self) -> None:
        result = self.by_category["no_prior_evidence"]

        self.assertTrue(result["passed"])
        self.assertEqual(result["support_count"], 0)
        self.assertTrue(result["no_evidence_honest"])

    def test_missing_source_fixture_cannot_support(self) -> None:
        result = self.by_category["missing_source_trap"]

        self.assertTrue(result["passed"])
        self.assertEqual(result["support_count"], 0)
        self.assertEqual(result["top_candidates"][0]["support_class"], "missing_source")

    def test_agent_claim_fixture_cannot_support(self) -> None:
        result = self.by_category["agent_claim_trap"]

        self.assertTrue(result["passed"])
        self.assertEqual(result["support_count"], 0)
        self.assertEqual(result["top_candidates"][0]["support_class"], "unverified_agent_claim")

    def test_true_prior_failure_is_retrieved_as_risk_evidence(self) -> None:
        result = self.by_category["true_prior_failure"]

        self.assertTrue(result["passed"])
        self.assertEqual(result["support_count"], 1)
        self.assertTrue(result["useful_recall_at_5"])
        self.assertEqual(result["support_items"][0]["support_polarity"], "risk")
        self.assertTrue(result["support_items"][0]["valid_relevant_support"])

    def test_secret_fixture_redacts_excerpts(self) -> None:
        result = self.by_category["secret_redaction"]
        report_text = json.dumps(result["top_candidates"], ensure_ascii=False)

        self.assertTrue(result["passed"])
        self.assertTrue(result["redaction_passed"])
        self.assertIn("[REDACTED]", report_text)
        self.assertNotIn("sk-testsecret0000000000000000", report_text)

    def test_huge_diff_fixture_remains_bounded(self) -> None:
        result = self.by_category["huge_diff"]
        excerpt = result["top_candidates"][0]["report_excerpt"]

        self.assertTrue(result["passed"])
        self.assertTrue(result["bounded_report_passed"])
        self.assertLessEqual(len(excerpt), 180)
        self.assertIn("[truncated]", excerpt)

    def test_miss_report_exists_for_every_failed_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            suite_path = _write_suite(
                root,
                _suite(
                    [
                        {
                            "fixture_id": "expected-support-but-empty",
                            "category": "no_prior_evidence",
                            "description": "This intentionally failing fixture requires support but has no candidates.",
                            "query": "review missing expected support",
                            "requested_polarity": "risk",
                            "review_started_at_utc": "2026-05-12T10:00:00+00:00",
                            "current_event": {
                                "event_id": "fixture:failing:current",
                                "target_paths": ["scripts/missing.py"],
                            },
                            "candidate_events": [],
                            "expected": {
                                "requires_support": True,
                                "support_polarity": "risk",
                            },
                        }
                    ]
                ),
            )

            report = run_review_memory_eval(
                suite=suite_path,
                workspace_id="failing-miss-report-test",
                root=root,
                generated_at_utc="2026-05-12T12:00:00+00:00",
            )
            miss_report = build_review_memory_miss_report(report)

        self.assertFalse(report["passed"])
        self.assertEqual(miss_report["miss_count"], 1)
        self.assertEqual(miss_report["miss_report_coverage"], 1.0)
        self.assertEqual(miss_report["misses"][0]["miss_reason"], "lexical_miss")

    def test_metrics_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            first = run_review_memory_eval(
                suite=DEFAULT_SUITE,
                workspace_id="deterministic-test",
                root=Path(left),
                generated_at_utc="2026-05-12T12:00:00+00:00",
            )
            second = run_review_memory_eval(
                suite=DEFAULT_SUITE,
                workspace_id="deterministic-test",
                root=Path(right),
                generated_at_utc="2026-05-12T12:00:00+00:00",
            )

        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["result_digest"], second["result_digest"])

    def test_benchmark_exits_nonzero_when_critical_false_support_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            suite_path = _write_suite(
                root,
                _suite(
                    [
                        {
                            "fixture_id": "positive-support-is-forbidden",
                            "category": "weak_text_match_trap",
                            "description": "This intentionally failing fixture lets invalid positive support through.",
                            "query": "review forbidden support scripts/forbidden.py",
                            "requested_polarity": "positive",
                            "review_started_at_utc": "2026-05-12T10:00:00+00:00",
                            "current_event": {
                                "event_id": "fixture:false-support:current",
                                "target_paths": ["scripts/forbidden.py"],
                            },
                            "candidate_events": [
                                {
                                    "event_id": "fixture:false-support:prior",
                                    "recorded_at_utc": "2026-05-10T10:00:00+00:00",
                                    "status": "accepted",
                                    "quality_status": "pass",
                                    "artifact_kind": "review_note",
                                    "artifact_behavior": "captured",
                                    "source_path": "sources/forbidden.md",
                                    "source_text": "Accepted note mentions scripts/forbidden.py but this fixture expects no support.",
                                    "target_paths": ["scripts/forbidden.py"],
                                    "evidence_types": ["human_acceptance"],
                                    "expected": {
                                        "support": False,
                                        "useful": False,
                                    },
                                }
                            ],
                            "expected": {
                                "no_strong_evidence": True,
                                "miss_reason": "weak_match_overfiltered",
                            },
                        }
                    ]
                ),
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = satlab_main(
                    [
                        "--root",
                        str(root),
                        "review",
                        "benchmark",
                        "--spartan",
                        "--suite",
                        str(suite_path),
                        "--format",
                        "json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["metrics"]["synthetic"]["critical_false_support"], 1)

    def test_fixture_source_paths_cannot_escape_fixture_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            suite_path = _write_suite(
                root,
                _suite(
                    [
                        {
                            "fixture_id": "path-traversal-source",
                            "category": "true_prior_failure",
                            "description": "Fixture source paths stay inside the materialized fixture workspace.",
                            "query": "review escaped source path",
                            "requested_polarity": "risk",
                            "review_started_at_utc": "2026-05-12T10:00:00+00:00",
                            "current_event": {
                                "event_id": "fixture:path-traversal:current",
                                "target_paths": ["scripts/path_guard.py"],
                            },
                            "candidate_events": [
                                {
                                    "event_id": "fixture:path-traversal:prior",
                                    "recorded_at_utc": "2026-05-10T10:00:00+00:00",
                                    "status": "failed",
                                    "quality_status": "fail",
                                    "artifact_kind": "test_log",
                                    "artifact_behavior": "captured",
                                    "source_path": "../escaped.log",
                                    "source_text": "FAILED path guard\n",
                                    "target_paths": ["scripts/path_guard.py"],
                                    "evidence_types": ["test_fail"],
                                    "expected": {
                                        "support": True,
                                        "useful": True,
                                    },
                                }
                            ],
                            "expected": {
                                "requires_support": True,
                                "support_polarity": "risk",
                            },
                        }
                    ]
                ),
            )

            with self.assertRaisesRegex(ValueError, "must stay inside the fixture workspace"):
                run_review_memory_eval(
                    suite=suite_path,
                    workspace_id="path-traversal-test",
                    root=root,
                    generated_at_utc="2026-05-12T12:00:00+00:00",
                )


if __name__ == "__main__":
    unittest.main()
