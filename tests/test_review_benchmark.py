from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from review_benchmark import run_review_benchmark  # noqa: E402


class ReviewBenchmarkTests(unittest.TestCase):
    def test_review_benchmark_has_zero_critical_false_evidence(self) -> None:
        report = run_review_benchmark(workspace_id="benchmark-test")

        self.assertTrue(report["passed"])
        self.assertEqual(report["workspace_id"], "benchmark-test")
        self.assertEqual(report["critical_false_evidence_count"], 0)
        self.assertFalse(report["training_export_ready"])
        self.assertEqual({case["case"] for case in report["cases"]}, {
            "self_recall",
            "no_prior_evidence",
            "secret_redaction",
            "missing_source_not_positive",
        })


if __name__ == "__main__":
    unittest.main()
