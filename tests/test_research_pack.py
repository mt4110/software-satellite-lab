from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_pack import (  # noqa: E402
    REQUIRED_RESEARCH_PACK_FILES,
    build_research_pack,
    reproduce_research_pack,
)
from satlab import main as satlab_main  # noqa: E402
from schema_coverage import CORE_SCHEMA_CANDIDATES, build_schema_coverage_report  # noqa: E402


def _benchmark_override() -> dict:
    return {
        "schema_name": "software-satellite-review-benchmark",
        "schema_version": 1,
        "workspace_id": "test-research-pack",
        "case_count": 2,
        "passed_count": 2,
        "critical_false_evidence_count": 0,
        "passed": True,
        "training_export_ready": False,
        "cases": [
            {"case": "no_prior_evidence", "passed": True, "critical_false_evidence_count": 0},
            {"case": "missing_source_not_positive", "passed": True, "critical_false_evidence_count": 0},
        ],
    }


class ResearchPackTests(unittest.TestCase):
    def test_schema_coverage_includes_all_core_schemas(self) -> None:
        report = build_schema_coverage_report(root=REPO_ROOT, generated_at_utc="1970-01-01T00:00:00+00:00")
        schema_ids = {candidate["schema_id"] for candidate in report["candidates"]}

        self.assertTrue(report["passed"])
        self.assertGreaterEqual(report["core_coverage_ratio"], 0.90)
        self.assertEqual(schema_ids, {candidate.schema_id for candidate in CORE_SCHEMA_CANDIDATES})

    def test_research_pack_generated_without_private_docs_and_reproduces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "research_pack"

            result = build_research_pack(
                output=output,
                root=REPO_ROOT,
                benchmark_results_override=_benchmark_override(),
            )
            latest = Path(result["latest_pack_path"])
            reproduction = reproduce_research_pack(latest, root=REPO_ROOT)

            for required in REQUIRED_RESEARCH_PACK_FILES:
                self.assertTrue((latest / required).is_file(), required)

        self.assertEqual(result["status"], "pass")
        self.assertEqual(reproduction["status"], "pass")
        self.assertTrue(reproduction["research_pack_reproduces"])
        self.assertEqual(reproduction["private_doc_dependency_count"], 0)
        self.assertGreaterEqual(reproduction["schema_coverage_core"], 0.90)
        self.assertTrue(reproduction["benchmark_results_included"])
        self.assertTrue(reproduction["limitations_included"])
        self.assertTrue(reproduction["no_trainable_export"])

    def test_research_pack_checksums_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_a = Path(tmpdir) / "pack-a"
            output_b = Path(tmpdir) / "pack-b"

            first = build_research_pack(
                output=output_a,
                root=REPO_ROOT,
                benchmark_results_override=_benchmark_override(),
            )
            second = build_research_pack(
                output=output_b,
                root=REPO_ROOT,
                benchmark_results_override=_benchmark_override(),
            )

        self.assertEqual(first["checksums"]["combined_sha256"], second["checksums"]["combined_sha256"])
        self.assertEqual(first["checksums"]["files"], second["checksums"]["files"])

    def test_reproduction_fails_when_pack_is_mutated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "research_pack"
            result = build_research_pack(
                output=output,
                root=REPO_ROOT,
                benchmark_results_override=_benchmark_override(),
            )
            latest = Path(result["latest_pack_path"])
            (latest / "limitations.md").write_text("mutated\n", encoding="utf-8")

            reproduction = reproduce_research_pack(latest, root=REPO_ROOT)

        self.assertEqual(reproduction["status"], "fail")
        self.assertFalse(reproduction["research_pack_reproduces"])
        self.assertIn("checksums_stable", reproduction["failing_check_ids"])

    def test_reproduction_fails_when_trainable_export_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "research_pack"
            result = build_research_pack(
                output=output,
                root=REPO_ROOT,
                benchmark_results_override=_benchmark_override(),
            )
            latest = Path(result["latest_pack_path"])
            export_path = latest / "training_exports" / "candidate.jsonl"
            export_path.parent.mkdir(parents=True, exist_ok=True)
            export_path.write_text("{\"prompt\": \"x\"}\n", encoding="utf-8")

            reproduction = reproduce_research_pack(latest, root=REPO_ROOT)

        self.assertEqual(reproduction["status"], "fail")
        self.assertFalse(reproduction["no_trainable_export"])
        self.assertIn("no_trainable_export", reproduction["failing_check_ids"])

    def test_satlab_research_pack_cli_surface(self) -> None:
        fake_result = {
            "status": "pass",
            "latest_pack_path": "/tmp/research-pack/latest",
            "pack_path": "/tmp/research-pack/runs/demo",
            "checksums": {"combined_sha256": "0" * 64},
            "reproduction": {
                "exit_gate": {
                    "research_pack_reproduces": True,
                    "private_doc_dependency_count": 0,
                    "schema_coverage_core": 1.0,
                    "benchmark_results_included": True,
                    "limitations_included": True,
                    "no_trainable_export": True,
                }
            },
        }
        stdout = io.StringIO()
        with mock.patch("satlab.build_research_pack", return_value=fake_result):
            with redirect_stdout(stdout):
                exit_code = satlab_main(["research", "pack", "--output", "/tmp/research-pack", "--format", "json"])
        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "pass")

    def test_satlab_research_reproduce_cli_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "research_pack"
            result = build_research_pack(
                output=output,
                root=REPO_ROOT,
                benchmark_results_override=_benchmark_override(),
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = satlab_main(
                    [
                        "research",
                        "reproduce",
                        "--pack",
                        result["latest_pack_path"],
                        "--format",
                        "json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["research_pack_reproduces"])


if __name__ == "__main__":
    unittest.main()
