from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from artifact_vault import capture_artifact  # noqa: E402
from evidence_support import build_evidence_support_result  # noqa: E402
from failure_memory_review import record_file_input  # noqa: E402
from satlab import main as satlab_main  # noqa: E402


def _event(
    *,
    event_id: str,
    ref: dict[str, Any] | None,
    status: str = "ok",
    recorded_at_utc: str = "2026-05-11T00:00:00+00:00",
    notes: list[str] | None = None,
    tags: list[str] | None = None,
    quality_status: str | None = "pass",
    session_surface: str = "chat",
    event_kind: str = "chat_run",
    evidence_types: list[str] | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if ref is not None:
        options["artifact_vault_refs"] = [ref]
    if quality_status is not None:
        options["quality_status"] = quality_status
    if evidence_types is not None:
        options["evidence_types"] = evidence_types
    return {
        "schema_name": "software-satellite-event",
        "schema_version": 1,
        "event_id": event_id,
        "event_kind": event_kind,
        "recorded_at_utc": recorded_at_utc,
        "workspace": {"workspace_id": "local-default"},
        "session": {"session_id": "s1", "surface": session_surface, "mode": "review"},
        "outcome": {"status": status, "quality_status": quality_status, "execution_status": status},
        "content": {"notes": notes or [], "options": options},
        "source_refs": {},
        "tags": tags or [],
    }


def _captured_ref(root: Path, name: str = "evidence.txt") -> dict[str, Any]:
    source = root / name
    source.write_text("verified local evidence\n", encoding="utf-8")
    return capture_artifact(source, kind="review_note", root=root)


class EvidenceSupportTests(unittest.TestCase):
    def test_source_linked_prior_can_support_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(event_id="event-prior", ref=ref)

            result = build_evidence_support_result(
                "event-prior",
                event=event,
                requested_polarity="positive",
                review_started_at="2026-05-12T00:00:00+00:00",
                root=root,
                checked_at_utc="2026-05-12T01:00:00+00:00",
            )

        self.assertTrue(result["can_support_decision"])
        self.assertEqual(result["support_class"], "source_linked_prior")
        self.assertEqual(result["support_polarity"], "positive")
        self.assertEqual(result["blockers"], [])

    def test_ok_status_without_verification_signal_cannot_support_positive_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(event_id="event-ok-only", ref=ref, status="ok", quality_status=None)

            result = build_evidence_support_result("event-ok-only", event=event, requested_polarity="positive", root=root)

        self.assertFalse(result["can_support_decision"])
        self.assertEqual(result["support_class"], "unknown")
        self.assertIn("missing_positive_signal", result["blockers"])

    def test_missing_source_cannot_support_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            event = _event(event_id="event-missing", ref=None)

            result = build_evidence_support_result("event-missing", event=event, requested_polarity="positive", root=root)

        self.assertFalse(result["can_support_decision"])
        self.assertEqual(result["support_class"], "missing_source")
        self.assertIn("missing_source", result["blockers"])

    def test_outside_workspace_artifact_ref_cannot_support_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            root = base / "workspace"
            root.mkdir()
            outside = base / "outside.txt"
            outside.write_text("outside workspace\n", encoding="utf-8")
            ref = capture_artifact(root / ".." / "outside.txt", kind="review_note", root=root)
            event = _event(event_id="event-outside", ref=ref)

            result = build_evidence_support_result("event-outside", event=event, requested_polarity="positive", root=root)

        self.assertFalse(result["can_support_decision"])
        self.assertEqual(result["support_class"], "missing_source")
        self.assertIn("outside_workspace", result["blockers"])

    def test_current_review_subject_cannot_support_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(event_id="event-current", ref=ref)

            result = build_evidence_support_result(
                "event-current",
                event=event,
                active_subject="event-current",
                requested_polarity="positive",
                root=root,
            )

        self.assertFalse(result["can_support_decision"])
        self.assertTrue(result["active_review_excluded"])
        self.assertEqual(result["support_class"], "current_review_subject")

    def test_current_review_subject_can_match_vault_object_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(event_id="event-current-object", ref=ref)
            active_subject = str(root / ref["vault_path"])

            result = build_evidence_support_result(
                "event-current-object",
                event=event,
                active_subject=active_subject,
                requested_polarity="positive",
                root=root,
            )

        self.assertFalse(result["can_support_decision"])
        self.assertTrue(result["active_review_excluded"])
        self.assertEqual(result["support_class"], "current_review_subject")

    def test_current_review_subject_can_match_absolute_source_path_for_relative_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "relative-evidence.txt"
            source.write_text("verified local evidence\n", encoding="utf-8")
            ref = capture_artifact("relative-evidence.txt", kind="review_note", root=root)
            event = _event(event_id="event-current-relative-source", ref=ref)

            result = build_evidence_support_result(
                "event-current-relative-source",
                event=event,
                active_subject=str(source),
                requested_polarity="positive",
                root=root,
            )

        self.assertFalse(result["can_support_decision"])
        self.assertTrue(result["active_review_excluded"])
        self.assertEqual(result["support_class"], "current_review_subject")

    def test_future_evidence_cannot_support_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(event_id="event-future", ref=ref, recorded_at_utc="2026-05-13T00:00:00+00:00")

            result = build_evidence_support_result(
                "event-future",
                event=event,
                review_started_at="2026-05-12T00:00:00+00:00",
                requested_polarity="positive",
                root=root,
            )

        self.assertFalse(result["can_support_decision"])
        self.assertEqual(result["support_class"], "future_evidence")

    def test_weak_match_cannot_support_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(event_id="event-weak", ref=ref, notes=["weak_match"])

            result = build_evidence_support_result("event-weak", event=event, requested_polarity="positive", root=root)

        self.assertFalse(result["can_support_decision"])
        self.assertEqual(result["support_class"], "weak_match")

    def test_unverified_agent_claim_cannot_support_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(
                event_id="event-agent",
                ref=ref,
                quality_status=None,
                session_surface="agent_lane",
                event_kind="agent_task_run",
            )

            result = build_evidence_support_result("event-agent", event=event, requested_polarity="positive", root=root)

        self.assertFalse(result["can_support_decision"])
        self.assertEqual(result["support_class"], "unverified_agent_claim")

    def test_contradictory_evidence_cannot_support_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(
                event_id="event-contradictory",
                ref=ref,
                evidence_types=["test_pass", "test_fail"],
            )

            result = build_evidence_support_result(
                "event-contradictory",
                event=event,
                requested_polarity="positive",
                root=root,
            )

        self.assertFalse(result["can_support_decision"])
        self.assertEqual(result["support_class"], "contradictory")

    def test_negative_prior_can_support_risk_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(
                event_id="event-negative",
                ref=ref,
                status="failed",
                quality_status="fail",
                evidence_types=["test_fail"],
            )

            result = build_evidence_support_result("event-negative", event=event, requested_polarity="risk", root=root)

        self.assertTrue(result["can_support_decision"])
        self.assertEqual(result["support_class"], "negative_prior")
        self.assertEqual(result["support_polarity"], "risk")

    def test_manual_pin_cannot_override_missing_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            event = _event(event_id="event-pin", ref=None, notes=["pinned"])

            result = build_evidence_support_result("event-pin", event=event, requested_polarity="positive", root=root)

        self.assertFalse(result["can_support_decision"])
        self.assertEqual(result["support_class"], "missing_source")
        self.assertIn("missing_source", result["blockers"])
        self.assertIn("manual_pin_diagnostic", result["warnings"])

    def test_support_result_is_deterministic_when_checked_at_is_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(event_id="event-deterministic", ref=ref)

            first = build_evidence_support_result(
                "event-deterministic",
                event=event,
                requested_polarity="positive",
                root=root,
                checked_at_utc="2026-05-12T01:00:00+00:00",
            )
            second = build_evidence_support_result(
                "event-deterministic",
                event=event,
                requested_polarity="positive",
                root=root,
                checked_at_utc="2026-05-12T01:00:00+00:00",
            )

        self.assertEqual(first, second)

    def test_satlab_evidence_support_cli_uses_vault_ref_from_file_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            failure = root / "failure.log"
            failure.write_text("test failed: support kernel should carry risk note\n", encoding="utf-8")
            recorded = record_file_input(input_kind="failure", source_path=failure, root=root)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = satlab_main(
                    [
                        "--root",
                        str(root),
                        "evidence",
                        "support",
                        "--event",
                        recorded["event_id"],
                        "--polarity",
                        "risk",
                        "--format",
                        "json",
                    ]
                )
            result = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(result["can_support_decision"])
        self.assertEqual(result["support_class"], "negative_prior")
        self.assertEqual(result["artifact_refs"], [recorded["artifact_vault_refs"][0]["artifact_id"]])

    def test_file_input_symlink_ref_is_not_supporting_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "failure.log"
            target.write_text("test failed through symlink\n", encoding="utf-8")
            link = root / "failure-link.log"
            try:
                link.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlink setup unavailable: {exc}")
            recorded = record_file_input(input_kind="failure", source_path=link, root=root)

            result = build_evidence_support_result(
                recorded["event_id"],
                requested_polarity="risk",
                root=root,
            )

        self.assertEqual(recorded["artifact_vault_refs"][0]["source_state"], "symlink_refused")
        self.assertFalse(result["can_support_decision"])
        self.assertIn("symlink_refused", result["blockers"])


if __name__ == "__main__":
    unittest.main()
