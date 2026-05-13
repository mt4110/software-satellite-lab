from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from agent_claims import (  # noqa: E402
    build_claim_software_work_event,
    extract_claims_from_transcript,
    log_indicates_tests_failed,
    log_indicates_tests_passed,
    normalize_declared_claims,
    verify_claims_against_artifacts,
)
from artifact_vault import capture_artifact  # noqa: E402
from evidence_support import build_evidence_support_result  # noqa: E402


class AgentClaimsTests(unittest.TestCase):
    def test_transcript_claims_start_unverified(self) -> None:
        claims = extract_claims_from_transcript(
            "\n".join(
                [
                    "$ pytest tests/test_widget.py",
                    "All tests passed.",
                    "Modified scripts/widget.py to clamp the retry count.",
                ]
            )
        )

        kinds = {claim["claim_kind"] for claim in claims}

        self.assertIn("command_run", kinds)
        self.assertIn("tests_passed", kinds)
        self.assertIn("file_modified", kinds)
        self.assertEqual({claim["verification_state"] for claim in claims}, {"unverified_agent_claim"})

    def test_claim_linked_to_test_log_becomes_verified_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            test_log = root / "test.log"
            test_log.write_text("tests/test_widget.py::test_ok PASSED\n1 passed in 0.01s\n", encoding="utf-8")
            ref = capture_artifact(test_log, kind="test_log", root=root)
            claims = normalize_declared_claims(
                [{"claim": "tests passed", "source": "transcript", "verification": "unverified"}]
            )

            verified = verify_claims_against_artifacts(claims, [ref], root=root)

        self.assertEqual(verified[0]["verification_state"], "verified_signal")
        self.assertEqual(verified[0]["verification_evidence"][0]["kind"], "test_log")

    def test_command_run_claim_can_be_verified_by_command_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            command_log = root / "commands.log"
            command_log.write_text("$ pytest tests/test_widget.py\nexit 0\n", encoding="utf-8")
            ref = capture_artifact(command_log, kind="command_log", root=root)
            claim = extract_claims_from_transcript("$ pytest tests/test_widget.py")[0]

            verified = verify_claims_against_artifacts([claim], [ref], root=root)

        self.assertEqual(verified[0]["verification_state"], "verified_signal")
        self.assertEqual(verified[0]["verification_evidence"][0]["kind"], "command_log")

    def test_unverified_claim_event_cannot_support_positive_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            transcript = root / "transcript.md"
            transcript.write_text("All tests passed.\n", encoding="utf-8")
            ref = capture_artifact(transcript, kind="transcript", root=root)
            claim = extract_claims_from_transcript("All tests passed.", source_artifact_id=ref["artifact_id"])[0]
            event = build_claim_software_work_event(
                claim,
                artifact_refs=[ref],
                workspace_id="claim-test",
                session_id="session-1",
                recorded_at_utc="2026-05-12T00:00:00+00:00",
            )

            support = build_evidence_support_result(
                event["event_id"],
                event=event,
                requested_polarity="positive",
                root=root,
            )

        self.assertFalse(support["can_support_decision"])
        self.assertEqual(support["support_class"], "unverified_agent_claim")

    def test_secret_like_claim_excerpt_is_redacted(self) -> None:
        claims = extract_claims_from_transcript(
            "Command run: OPENAI_API_KEY=sk-testsecret0000000000000000 pytest tests/test_widget.py"
        )

        self.assertEqual(claims[0]["claim_kind"], "command_run")
        self.assertNotIn("sk-testsecret", claims[0]["claim"])
        self.assertIn("[REDACTED]", claims[0]["claim"])

    def test_zero_error_summary_is_not_a_failed_test_signal(self) -> None:
        log_text = "12 passed, 0 failed, 0 errors in 0.21s\n"

        self.assertFalse(log_indicates_tests_failed(log_text))
        self.assertTrue(log_indicates_tests_passed(log_text))

    def test_zero_error_transcript_summary_extracts_pass_not_failure(self) -> None:
        claims = extract_claims_from_transcript("12 passed, 0 failed, 0 errors in 0.21s")
        declared = normalize_declared_claims(
            [{"claim": "12 passed, 0 failed, 0 errors in 0.21s"}]
        )

        self.assertEqual({claim["claim_kind"] for claim in claims}, {"tests_passed"})
        self.assertEqual(declared[0]["claim_kind"], "tests_passed")

    def test_nonzero_error_summary_is_a_failed_test_signal(self) -> None:
        log_text = "11 passed, 1 error in 0.21s\n"

        self.assertTrue(log_indicates_tests_failed(log_text))
        self.assertFalse(log_indicates_tests_passed(log_text))


if __name__ == "__main__":
    unittest.main()
