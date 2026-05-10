from __future__ import annotations

import io
import hashlib
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

from failure_memory_review import (  # noqa: E402
    build_failure_recall,
    build_review_risk_report,
    record_file_input,
    record_human_verdict,
    run_review_risk_pack,
    summarize_patch,
)
from memory_index import rebuild_memory_index  # noqa: E402
from satlab import main as satlab_main  # noqa: E402
from software_work_events import build_event_contract_check, read_event_log  # noqa: E402


def _write_patch(root: Path, name: str = "changes.diff") -> Path:
    path = root / name
    path.write_text(
        "\n".join(
            [
                "diff --git a/scripts/satlab.py b/scripts/satlab.py",
                "--- a/scripts/satlab.py",
                "+++ b/scripts/satlab.py",
                "@@ -1,2 +1,3 @@",
                " import argparse",
                "+# preserve source artifact path",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_large_patch(root: Path) -> Path:
    path = root / "large.diff"
    filler = "\n".join(f"+filler line {index:05d}" for index in range(1600))
    path.write_text(
        "\n".join(
            [
                "diff --git a/scripts/first.py b/scripts/first.py",
                "--- a/scripts/first.py",
                "+++ b/scripts/first.py",
                "@@ -1 +1 @@",
                "+first change",
                filler,
                "diff --git a/scripts/late_file.py b/scripts/late_file.py",
                "--- a/scripts/late_file.py",
                "+++ b/scripts/late_file.py",
                "@@ -1 +1 @@",
                "+late change",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class FailureMemoryReviewTests(unittest.TestCase):
    def test_event_contract_verifies_optional_source_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "artifacts" / "text" / "event.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("first", encoding="utf-8")
            checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
            event = {
                "event_id": "local-default:test:event",
                "event_kind": "chat_run",
                "outcome": {"status": "ok"},
                "source_refs": {
                    "artifact_ref": {
                        "artifact_path": str(artifact),
                        "artifact_sha256": checksum,
                    }
                },
            }

            verified = build_event_contract_check(event, root=root)
            artifact.write_text("changed", encoding="utf-8")
            mismatched = build_event_contract_check(event, root=root)

        self.assertEqual(verified["contract_status"], "ok")
        self.assertEqual(verified["source_artifact"]["checksum_status"], "verified")
        self.assertEqual(mismatched["contract_status"], "invalid_event_contract")
        self.assertIn("source_artifact_checksum_mismatch", mismatched["schema_issues"])
        self.assertEqual(mismatched["source_artifact"]["checksum_status"], "mismatch")

    def test_patch_summary_uses_full_patch_for_hints_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            patch = _write_large_patch(root)

            summary = summarize_patch(patch)

            self.assertGreater(patch.stat().st_size, 20000)
            self.assertIn("scripts/first.py", summary["changed_files"])
            self.assertIn("scripts/late_file.py", summary["changed_files"])
            self.assertEqual(summary["changed_file_count"], 2)
            self.assertGreater(summary["added_lines"], 1600)

    def test_file_first_review_loop_records_recall_verdict_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            failure = root / "previous-failure.log"
            failure.write_text(
                "test failed: missing source artifact path in scripts/satlab.py review flow\n",
                encoding="utf-8",
            )
            patch = _write_patch(root)

            failure_result = record_file_input(
                input_kind="failure",
                source_path=failure,
                note="Prior missing-source bug in scripts/satlab.py",
                root=root,
            )
            patch_result = record_file_input(
                input_kind="patch",
                source_path=patch,
                note="Patch review input",
                root=root,
            )
            recall, _latest_recall, _run_recall = build_failure_recall(
                query="missing source artifact path scripts/satlab.py failure",
                patch_path=patch,
                source_event_id=patch_result["event_id"],
                root=root,
            )
            verdict, _latest_verdict, _run_verdict = record_human_verdict(
                verdict="reject",
                event_id=patch_result["event_id"],
                reason="Repeats prior missing-source bug",
                root=root,
            )
            metadata, markdown, latest_report, run_report = build_review_risk_report(root=root)
            index_summary = rebuild_memory_index(root=root)
            event_log = read_event_log(Path(index_summary["event_log_path"]))
            patch_event = next(
                event for event in event_log["events"] if event["event_id"] == patch_result["event_id"]
            )
            contract_check = build_event_contract_check(patch_event, root=root)

            self.assertTrue(failure_result["event_id"])
            self.assertGreaterEqual(recall["bundle"]["selected_count"], 1)
            self.assertEqual(verdict["signal"]["signal_kind"], "rejection")
            self.assertEqual(contract_check["source_artifact"]["checksum_status"], "verified")
            self.assertEqual(metadata["learning_state"]["state"], "blocked")
            self.assertIn("Review Risk Report", markdown)
            self.assertIn("Repeats prior missing-source bug", markdown)
            self.assertTrue(latest_report.is_file())
            self.assertTrue(run_report.is_file())

    def test_satlab_verdict_template_text_is_human_readable(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = satlab_main(
                [
                    "verdict",
                    "template",
                    "--event",
                    "local-default:chat-main:patch-0001",
                    "--verdict",
                    "reject",
                    "--reason",
                    "Needs source review",
                    "--format",
                    "text",
                ]
            )
        text = stdout.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("Verdict template: reject", text)
        self.assertIn("Event: local-default:chat-main:patch-0001", text)
        self.assertFalse(text.lstrip().startswith("{"))

    def test_satlab_cli_ingest_and_recall_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            patch = _write_patch(root)

            ingest_stdout = io.StringIO()
            with redirect_stdout(ingest_stdout):
                exit_code = satlab_main(
                    [
                        "--root",
                        str(root),
                        "event",
                        "ingest",
                        "--patch",
                        str(patch),
                        "--note",
                        "Patch input",
                        "--format",
                        "json",
                    ]
                )
            ingest = json.loads(ingest_stdout.getvalue())

            recall_stdout = io.StringIO()
            with redirect_stdout(recall_stdout):
                recall_exit = satlab_main(
                    [
                        "--root",
                        str(root),
                        "recall",
                        "failure",
                        "--query",
                        "patch risk source path scripts/satlab.py",
                        "--format",
                        "json",
                    ]
                )
            recall = json.loads(recall_stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(recall_exit, 0)
        self.assertTrue(ingest["event_id"])
        self.assertEqual(recall["schema_name"], "software-satellite-failure-memory-recall")
        self.assertEqual(recall["request"]["source_event_id"], ingest["event_id"])
        self.assertIn("source_path_completeness", recall["validation_metrics"])

    def test_review_risk_pack_runner_is_built_in_and_writes_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            (template_dir / "review-risk-pack.satellite.yaml").write_text(
                (REPO_ROOT / "templates" / "review-risk-pack.satellite.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            patch = _write_patch(root)
            ingested = record_file_input(
                input_kind="patch",
                source_path=patch,
                note="Patch review input",
                root=root,
            )

            metadata, markdown, latest_report, run_report = run_review_risk_pack(
                pack="review-risk-pack",
                patch_path=patch,
                root=root,
            )
            index_summary = rebuild_memory_index(root=root)

            self.assertEqual(metadata["pack_run"]["runner"], "built_in_review_risk_pack")
            self.assertEqual(metadata["pack_run"]["input_event_id"], ingested["event_id"])
            self.assertEqual(index_summary["event_count"], 1)
            self.assertIn("# Review Risk Report", markdown)
            self.assertTrue(latest_report.is_file())
            self.assertTrue(run_report.is_file())

    def test_report_does_not_mix_stale_verdict_or_recall_with_latest_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_patch = _write_patch(root, "first.diff")
            second_patch = _write_patch(root, "second.diff")

            first = record_file_input(
                input_kind="patch",
                source_path=first_patch,
                note="First patch",
                root=root,
            )
            build_failure_recall(
                query="missing source artifact path scripts/satlab.py failure",
                patch_path=first_patch,
                source_event_id=first["event_id"],
                root=root,
            )
            record_human_verdict(
                verdict="reject",
                event_id=first["event_id"],
                reason="First patch rejected",
                root=root,
            )
            second = record_file_input(
                input_kind="patch",
                source_path=second_patch,
                note="Second patch",
                root=root,
            )

            metadata, markdown, _latest_report, _run_report = build_review_risk_report(root=root)

            self.assertEqual(metadata["event_id"], second["event_id"])
            self.assertEqual(metadata["risk_note"], {})
            self.assertNotIn("First patch rejected", markdown)
            self.assertIn(second["event_id"], markdown)

    def test_report_ignores_source_less_recall_when_newer_input_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_patch = _write_patch(root, "first.diff")
            second_patch = _write_patch(root, "second.diff")

            record_file_input(
                input_kind="patch",
                source_path=first_patch,
                note="First patch",
                root=root,
            )
            build_failure_recall(
                query="source path scripts/satlab.py",
                patch_path=first_patch,
                root=root,
            )
            second = record_file_input(
                input_kind="patch",
                source_path=second_patch,
                note="Second patch",
                root=root,
            )

            metadata, markdown, _latest_report, _run_report = build_review_risk_report(root=root)

            self.assertEqual(metadata["event_id"], second["event_id"])
            self.assertEqual(metadata["risk_note"], {})
            self.assertIn("Record a human verdict before reusing this evidence.", markdown)

    def test_review_risk_pack_reuses_only_matching_latest_patch_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            (template_dir / "review-risk-pack.satellite.yaml").write_text(
                (REPO_ROOT / "templates" / "review-risk-pack.satellite.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            first_patch = _write_patch(root, "first.diff")
            second_patch = _write_patch(root, "second.diff")
            first = record_file_input(
                input_kind="patch",
                source_path=first_patch,
                note="First patch",
                root=root,
            )

            metadata, _markdown, _latest_report, _run_report = run_review_risk_pack(
                pack="review-risk-pack",
                patch_path=second_patch,
                root=root,
            )
            index_summary = rebuild_memory_index(root=root)

            self.assertNotEqual(metadata["pack_run"]["input_event_id"], first["event_id"])
            self.assertEqual(index_summary["event_count"], 2)


if __name__ == "__main__":
    unittest.main()
