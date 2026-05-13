from __future__ import annotations

import ast
import io
import json
import shutil
import socket
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

from agent_session_intake import (  # noqa: E402
    aggregate_intake_exit_gate,
    build_agent_session_bundle,
    ingest_agent_session_bundle,
    ingest_agent_session_bundle_path,
)
from evidence_support import build_evidence_support_result  # noqa: E402
from satlab import main as satlab_main  # noqa: E402
from software_work_events import iter_agent_session_intake_events  # noqa: E402


EXAMPLE_BUNDLES = REPO_ROOT / "examples" / "agent_session_bundles"


def _copy_examples(root: Path) -> Path:
    target = root / "examples" / "agent_session_bundles"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(EXAMPLE_BUNDLES, target)
    return target


class AgentSessionIntakeTests(unittest.TestCase):
    def test_generic_bundle_normalizes_to_software_work_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            examples = _copy_examples(root)

            result = ingest_agent_session_bundle_path(
                examples / "generic.json",
                workspace_id="m13-test",
                root=root,
                refresh_index=False,
                write_latest=False,
            )

        event = result["software_work_event"]
        self.assertEqual(event["schema_name"], "software-satellite-event")
        self.assertEqual(event["event_kind"], "agent_session_intake")
        self.assertEqual(event["content"]["options"]["workflow"], "agent_session_intake")
        self.assertGreaterEqual(len(event["source_refs"]["artifact_vault_refs"]), 4)
        self.assertEqual(result["claim_counts"]["verified_signal"], 2)

    def test_fixture_bundles_ingest_and_exit_gate_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            examples = _copy_examples(root)
            results = [
                ingest_agent_session_bundle_path(
                    path,
                    workspace_id="m13-fixtures",
                    root=root,
                    refresh_index=False,
                    write_latest=False,
                )
                for path in sorted(examples.glob("*.json"))
            ]

        gate = aggregate_intake_exit_gate(results)
        self.assertGreaterEqual(gate["fixture_bundles_normalized"], 5)
        self.assertGreaterEqual(gate["agent_labels_represented"], 3)
        self.assertEqual(gate["unverified_agent_claim_positive_support_count"], 0)
        self.assertEqual(gate["secret_redaction_fixture_failures"], 0)
        self.assertEqual(gate["network_call_count"], 0)
        self.assertTrue(gate["passed"])

    def test_missing_diff_records_diagnostic_and_cannot_support_patch_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            transcript = root / "transcript.md"
            transcript.write_text("All tests passed.\n", encoding="utf-8")
            bundle = build_agent_session_bundle(agent_label="generic", transcript=transcript)

            result = ingest_agent_session_bundle(
                bundle,
                workspace_id="m13-missing-diff",
                root=root,
                refresh_index=False,
                write_latest=False,
            )
            event = result["software_work_event"]
            support = build_evidence_support_result(
                event["event_id"],
                event=event,
                requested_polarity="positive",
                root=root,
            )

        self.assertIn("missing_diff", result["diagnostics"])
        self.assertEqual(event["outcome"]["status"], "diagnostic")
        self.assertFalse(support["can_support_decision"])
        self.assertIn(support["support_class"], {"unverified_agent_claim", "unknown"})

    def test_missing_diff_file_blocks_verified_claims_from_patch_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            transcript = root / "transcript.md"
            transcript.write_text("All tests passed.\n", encoding="utf-8")
            test_log = root / "test.txt"
            test_log.write_text("1 passed in 0.01s\n", encoding="utf-8")
            bundle = build_agent_session_bundle(
                agent_label="generic",
                diff=root / "missing.diff",
                transcript=transcript,
                test_log=test_log,
            )

            result = ingest_agent_session_bundle(
                bundle,
                workspace_id="m13-missing-diff-file",
                root=root,
                refresh_index=False,
                write_latest=False,
            )
            event = result["software_work_event"]
            support = build_evidence_support_result(
                event["event_id"],
                event=event,
                requested_polarity="positive",
                root=root,
            )

        self.assertIn("missing_diff", result["diagnostics"])
        self.assertEqual(result["claim_counts"]["verified_signal"], 1)
        self.assertIsNone(event["outcome"]["quality_status"])
        self.assertTrue(event["content"]["options"]["verified_claims_held_for_patch_evidence"])
        self.assertFalse(support["can_support_decision"])

    def test_missing_diff_preserves_verified_failing_log_as_risk_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            transcript = root / "transcript.md"
            transcript.write_text("Bug fixed: retry cache now trims whitespace.\n", encoding="utf-8")
            test_log = root / "test.txt"
            test_log.write_text("not ok 1 cache key trims whitespace\n", encoding="utf-8")
            bundle = build_agent_session_bundle(
                agent_label="generic",
                diff=root / "missing.diff",
                transcript=transcript,
                test_log=test_log,
            )

            result = ingest_agent_session_bundle(
                bundle,
                workspace_id="m13-missing-diff-risk",
                root=root,
                refresh_index=False,
                write_latest=False,
            )
            event = result["software_work_event"]
            support = build_evidence_support_result(
                event["event_id"],
                event=event,
                requested_polarity="risk",
                root=root,
            )

        self.assertIn("missing_diff", result["diagnostics"])
        self.assertEqual(event["outcome"]["status"], "failed")
        self.assertEqual(event["outcome"]["quality_status"], "fail")
        self.assertIn("test_fail", event["content"]["options"]["evidence_types"])
        self.assertTrue(event["content"]["options"]["verified_claims_held_for_patch_evidence"])
        self.assertTrue(support["can_support_decision"])
        self.assertEqual(support["support_class"], "negative_prior")

    def test_failing_test_log_records_risk_signal_without_transcript_failure_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            diff = root / "patch.diff"
            diff.write_text("+small\n", encoding="utf-8")
            transcript = root / "transcript.md"
            transcript.write_text("Bug fixed: retry cache now trims whitespace.\n", encoding="utf-8")
            test_log = root / "test.txt"
            test_log.write_text(
                "tests/test_cache.py::test_key_trims FAILED\nAssertionError: expected trimmed key\n",
                encoding="utf-8",
            )
            bundle = build_agent_session_bundle(
                agent_label="aider",
                diff=diff,
                transcript=transcript,
                test_log=test_log,
            )

            result = ingest_agent_session_bundle(
                bundle,
                workspace_id="m13-failing-log",
                root=root,
                refresh_index=False,
                write_latest=False,
            )
            event = result["software_work_event"]
            support = build_evidence_support_result(
                event["event_id"],
                event=event,
                requested_polarity="risk",
                root=root,
            )

        self.assertEqual(event["outcome"]["quality_status"], "fail")
        self.assertEqual(event["outcome"]["status"], "failed")
        self.assertIn("test_fail", event["content"]["options"]["evidence_types"])
        self.assertEqual(event["content"]["options"]["verification_log_signal"]["quality_status"], "fail")
        self.assertTrue(support["can_support_decision"])
        self.assertEqual(support["support_class"], "negative_prior")

    def test_secret_like_transcript_content_is_redacted_in_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            examples = _copy_examples(root)
            result = ingest_agent_session_bundle_path(
                examples / "generic.json",
                workspace_id="m13-secret",
                root=root,
                refresh_index=False,
                write_latest=False,
            )

        transcript_ref = next(
            item["artifact_ref"]
            for item in result["captured_artifacts"]
            if item["bundle_kind"] == "transcript"
        )
        excerpt = transcript_ref["report_excerpt"]["text"]
        self.assertNotIn("sk-testsecret0000000000000000", excerpt)
        self.assertIn("[REDACTED]", excerpt)
        self.assertEqual(result["exit_gate"]["secret_redaction_fixture_failures"], 0)

    def test_oversized_transcript_is_capped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            diff = root / "patch.diff"
            diff.write_text("+small\n", encoding="utf-8")
            transcript = root / "transcript.md"
            transcript.write_text("All tests passed.\n" + ("A" * 400), encoding="utf-8")
            bundle = build_agent_session_bundle(agent_label="generic", diff=diff, transcript=transcript)

            result = ingest_agent_session_bundle(
                bundle,
                workspace_id="m13-oversize",
                root=root,
                refresh_index=False,
                write_latest=False,
                max_capture_bytes=80,
                report_excerpt_chars=64,
                transcript_claim_read_chars=24,
            )

        transcript_ref = next(
            item["artifact_ref"]
            for item in result["captured_artifacts"]
            if item["bundle_kind"] == "transcript"
        )
        self.assertEqual(transcript_ref["source_state"], "oversize")
        self.assertIn("transcript_claim_read_capped", result["diagnostics"])
        self.assertLessEqual(len(transcript_ref["report_excerpt"]["text"]), 80)

    def test_refused_symlink_transcript_does_not_contribute_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            diff = root / "patch.diff"
            diff.write_text("+small\n", encoding="utf-8")
            target = root / "target-transcript.md"
            target.write_text("All tests passed.\n", encoding="utf-8")
            link = root / "transcript-link.md"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink setup unavailable: {exc}")
            bundle = build_agent_session_bundle(agent_label="generic", diff=diff, transcript=link)

            result = ingest_agent_session_bundle(
                bundle,
                workspace_id="m13-symlink-transcript",
                root=root,
                refresh_index=False,
                write_latest=False,
            )

        transcript_ref = next(
            item["artifact_ref"]
            for item in result["captured_artifacts"]
            if item["bundle_kind"] == "transcript"
        )
        self.assertEqual(transcript_ref["capture_state"], "refused")
        self.assertEqual(transcript_ref["source_state"], "symlink_refused")
        self.assertEqual(result["claim_counts"]["total"], 0)
        self.assertIn("transcript_claim_read_symlink_refused", result["diagnostics"])

    def test_intake_makes_no_network_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            diff = root / "patch.diff"
            diff.write_text("+small\n", encoding="utf-8")
            transcript = root / "transcript.md"
            transcript.write_text("All tests passed.\n", encoding="utf-8")
            bundle = build_agent_session_bundle(agent_label="generic", diff=diff, transcript=transcript)

            with mock.patch.object(socket, "socket", side_effect=AssertionError("network disabled")):
                result = ingest_agent_session_bundle(
                    bundle,
                    workspace_id="m13-network",
                    root=root,
                    refresh_index=False,
                    write_latest=False,
                )

        self.assertEqual(result["exit_gate"]["network_call_count"], 0)

    def test_no_tool_specific_dependency_imports_are_required(self) -> None:
        forbidden = {"openai", "anthropic", "requests", "httpx", "github", "chromadb"}
        imported: set[str] = set()
        for path in (SCRIPTS_DIR / "agent_session_intake.py", SCRIPTS_DIR / "agent_claims.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])

        self.assertFalse(forbidden & imported)

    def test_privacy_metadata_is_preserved_and_export_stays_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            diff = root / "patch.diff"
            diff.write_text("+small\n", encoding="utf-8")
            bundle = {
                "schema_name": "software-satellite-agent-session-bundle",
                "schema_version": 1,
                "agent_label": "manual",
                "task": {"title": "Privacy fixture", "goal": "Preserve metadata."},
                "artifacts": [{"kind": "diff", "path": str(diff)}],
                "privacy": {
                    "contains_private_code": False,
                    "contains_user_text": True,
                    "export_allowed": True,
                },
            }

            result = ingest_agent_session_bundle(
                bundle,
                workspace_id="m13-privacy",
                root=root,
                refresh_index=False,
                write_latest=False,
            )

        self.assertFalse(result["privacy"]["contains_private_code"])
        self.assertTrue(result["privacy"]["export_allowed"])
        self.assertFalse(result["export_policy"]["effective_export_allowed"])
        self.assertFalse(result["software_work_event"]["content"]["options"]["export_policy"]["effective_export_allowed"])
        self.assertTrue(result["software_work_event"]["content"]["options"]["export_policy"]["no_provider_hub"])
        self.assertTrue(result["software_work_event"]["content"]["options"]["export_policy"]["no_vector_export"])
        self.assertIn("declared_export_allowed_ignored", result["diagnostics"])

    def test_privacy_flags_are_required_by_runtime_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            diff = root / "patch.diff"
            diff.write_text("+small\n", encoding="utf-8")
            bundle = {
                "schema_name": "software-satellite-agent-session-bundle",
                "schema_version": 1,
                "agent_label": "manual",
                "task": {"title": "Privacy fixture", "goal": "Require explicit privacy metadata."},
                "artifacts": [{"kind": "diff", "path": str(diff)}],
                "privacy": {
                    "contains_user_text": True,
                    "export_allowed": False,
                },
            }

            with self.assertRaisesRegex(ValueError, "privacy.contains_private_code"):
                ingest_agent_session_bundle(
                    bundle,
                    workspace_id="m13-privacy-required",
                    root=root,
                    refresh_index=False,
                    write_latest=False,
                )

    def test_invalid_agent_label_in_bundle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            diff = root / "patch.diff"
            diff.write_text("+small\n", encoding="utf-8")
            bundle = {
                "schema_name": "software-satellite-agent-session-bundle",
                "schema_version": 1,
                "agent_label": "provider_plugin",
                "task": {"title": "Invalid label", "goal": "Reject provider-like labels."},
                "artifacts": [{"kind": "diff", "path": str(diff)}],
                "privacy": {
                    "contains_private_code": True,
                    "contains_user_text": True,
                    "export_allowed": False,
                },
            }

            with self.assertRaisesRegex(ValueError, "agent_label"):
                ingest_agent_session_bundle(
                    bundle,
                    workspace_id="m13-invalid-label",
                    root=root,
                    refresh_index=False,
                    write_latest=False,
                )

    def test_reingesting_same_bundle_uses_latest_intake_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            diff = root / "patch.diff"
            transcript = root / "transcript.md"
            transcript.write_text("All tests passed.\n", encoding="utf-8")
            test_log = root / "test.txt"
            test_log.write_text("1 passed in 0.01s\n", encoding="utf-8")
            bundle = build_agent_session_bundle(
                agent_label="generic",
                diff=diff,
                transcript=transcript,
                test_log=test_log,
            )

            first = ingest_agent_session_bundle(
                bundle,
                workspace_id="m13-reingest",
                root=root,
                refresh_index=False,
                write_latest=False,
            )
            diff.write_text("+small\n", encoding="utf-8")
            second = ingest_agent_session_bundle(
                bundle,
                workspace_id="m13-reingest",
                root=root,
                refresh_index=False,
                write_latest=False,
            )
            events = iter_agent_session_intake_events(root=root, workspace_id="m13-reingest")

        self.assertEqual(first["software_work_event"]["event_id"], second["software_work_event"]["event_id"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["outcome"]["status"], "verified")
        self.assertNotIn("missing_diff", events[0]["content"]["options"]["diagnostics"])

    def test_satlab_intake_cli_agent_session_and_pr_bundle_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            examples = _copy_examples(root)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                agent_exit = satlab_main(
                    [
                        "--root",
                        str(root),
                        "intake",
                        "agent-session",
                        "--bundle",
                        str(examples / "generic.json"),
                        "--workspace-id",
                        "m13-cli",
                        "--format",
                        "json",
                    ]
                )
            agent_payload = json.loads(stdout.getvalue())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                pr_exit = satlab_main(
                    [
                        "--root",
                        str(root),
                        "intake",
                        "pr-bundle",
                        "--diff",
                        str(examples / "files" / "manual-pr.diff"),
                        "--review",
                        str(examples / "files" / "manual-review.md"),
                        "--ci-log",
                        str(examples / "files" / "manual-ci.txt"),
                        "--workspace-id",
                        "m13-cli-pr",
                        "--format",
                        "json",
                    ]
                )
            pr_payload = json.loads(stdout.getvalue())

        self.assertEqual(agent_exit, 0)
        self.assertEqual(pr_exit, 0)
        self.assertEqual(agent_payload["schema_name"], "software-satellite-agent-session-intake-result")
        self.assertEqual(pr_payload["bundle"]["agent_label"], "manual")


if __name__ == "__main__":
    unittest.main()
