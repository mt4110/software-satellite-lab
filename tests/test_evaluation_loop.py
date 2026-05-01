from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from artifact_schema import build_artifact_payload, build_prompt_record, build_runtime_record, write_artifact  # noqa: E402
from evaluation_loop import (  # noqa: E402
    append_evaluation_comparison,
    append_evaluation_signal,
    build_evaluation_comparison,
    build_evaluation_signal,
    evaluation_comparison_log_path,
    evaluation_signal_log_path,
    build_curation_export_preview,
    format_curation_export_preview_report,
    format_evaluation_snapshot_report,
    record_curation_export_preview,
    record_review_resolution_signal,
    record_evaluation_snapshot,
)
from memory_index import MemoryIndex, rebuild_memory_index  # noqa: E402
from run_evaluation_loop import main as evaluation_main  # noqa: E402
from workspace_state import WorkspaceSessionStore  # noqa: E402


def write_capability_matrix(root: Path) -> None:
    matrix_path = root / "artifacts" / "capability_matrix" / "matrix.json"
    payload = build_artifact_payload(
        artifact_kind="capability_matrix",
        status="ok",
        runtime=build_runtime_record(backend="capability-matrix", model_id="backend-a"),
        prompts=build_prompt_record(),
        extra={
            "results": [
                {
                    "capability": "chat",
                    "phase": "m4",
                    "status": "ok",
                    "artifact_kind": "text",
                    "artifact_path": str(root / "artifacts" / "text" / "pass.json"),
                    "validation_command": "python -m unittest tests.test_pass",
                    "claim_scope": "accepted patch keeps the loop green",
                    "output_preview": "All checks passed.",
                    "quality_status": "pass",
                    "quality_checks": [{"name": "unit", "pass": True, "detail": "green"}],
                    "quality_notes": [],
                    "notes": [],
                    "runtime_backend": "backend-a",
                    "execution_status": "ok",
                    "validation_mode": "unit",
                    "pass_definition": "unit tests pass for evaluation loop",
                },
                {
                    "capability": "vision",
                    "phase": "m4",
                    "status": "failed",
                    "artifact_kind": "vision",
                    "artifact_path": str(root / "artifacts" / "vision" / "fail.json"),
                    "validation_command": "python -m unittest tests.test_fail",
                    "claim_scope": "failure records a repair target",
                    "output_preview": "AssertionError: expected repair linkage.",
                    "quality_status": "fail",
                    "quality_checks": [{"name": "unit", "pass": False, "detail": "link missing"}],
                    "quality_notes": ["repair needed"],
                    "notes": ["follow-up required"],
                    "runtime_backend": "backend-a",
                    "execution_status": "failed",
                    "validation_mode": "unit",
                    "pass_definition": "failure should remain linkable",
                },
            ]
        },
    )
    write_artifact(matrix_path, payload)


class EvaluationLoopTests(unittest.TestCase):
    def test_snapshot_derives_test_signals_and_counts_repair_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)

            first_snapshot, _latest_path, _run_path = record_evaluation_snapshot(root=root)
            failure_event_id = "local-default:capability-matrix:matrix:row-2:vision"
            repair_event_id = "local-default:capability-matrix:matrix:row-1:chat"
            signal = build_evaluation_signal(
                signal_kind="acceptance",
                source_event_id=repair_event_id,
                target_event_id=failure_event_id,
                relation_kind="repairs",
                rationale="The follow-up patch closed the failing check.",
            )
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                signal,
                workspace_id="local-default",
            )
            non_failure_link = build_evaluation_signal(
                signal_kind="acceptance",
                source_event_id=repair_event_id,
                target_event_id=repair_event_id,
                relation_kind="repairs",
                rationale="This link points at accepted work, not at a failure.",
            )
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                non_failure_link,
                workspace_id="local-default",
            )
            comparison = build_evaluation_comparison(
                candidate_event_ids=[repair_event_id, failure_event_id],
                winner_event_id=repair_event_id,
                task_label="choose stable M4 foundation",
                criteria=["passing tests", "clear repair linkage"],
                rationale="The passing candidate is better evidence for curation.",
            )
            append_evaluation_comparison(
                evaluation_comparison_log_path(root=root),
                comparison,
                workspace_id="local-default",
            )

            snapshot, latest_path, run_path = record_evaluation_snapshot(root=root)
            report = format_evaluation_snapshot_report(snapshot)

            self.assertTrue(latest_path.exists())
            self.assertTrue(run_path.exists())
            self.assertEqual(first_snapshot["counts"]["test_pass"], 1)
            self.assertEqual(first_snapshot["counts"]["test_fail"], 1)
            self.assertEqual(first_snapshot["counts"]["pending_failures"], 1)
            self.assertEqual(snapshot["counts"]["acceptance"], 2)
            self.assertEqual(snapshot["counts"]["repair_links"], 2)
            self.assertEqual(snapshot["counts"]["repaired_failures"], 1)
            self.assertEqual(snapshot["counts"]["addressed_failures"], 1)
            self.assertEqual(snapshot["counts"]["pending_failures"], 0)
            self.assertEqual(snapshot["counts"]["comparisons"], 1)
            self.assertEqual(snapshot["counts"]["comparison_winners"], 1)
            self.assertEqual(snapshot["counts"]["curation_ready"], 1)
            self.assertEqual(snapshot["counts"]["curation_blocked"], 1)
            self.assertEqual(snapshot["curation"]["candidates"][0]["state"], "ready")
            self.assertIn("Test pass: 1", report)
            self.assertIn("Repair links: 2", report)
            self.assertIn("Comparisons: 1", report)
            self.assertIn("Curation ready: 1", report)
            self.assertIn("Addressed failures: 1", report)

    def test_workspace_artifact_validation_flows_into_index_and_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = WorkspaceSessionStore(root=root)
            artifact_path = root / "artifacts" / "text" / "validated.json"
            write_artifact(
                artifact_path,
                build_artifact_payload(
                    artifact_kind="text",
                    status="ok",
                    runtime=build_runtime_record(backend="backend-a", model_id="backend-a"),
                    prompts=build_prompt_record(prompt="Run a deterministic check."),
                    extra={
                        "validation": {
                            "validation_mode": "unit",
                            "claim_scope": "workspace artifact validation",
                            "pass_definition": "workspace validation is indexed",
                            "quality_status": "pass",
                            "execution_status": "ok",
                            "quality_checks": [
                                {"name": "compile", "pass": True, "detail": "ok"},
                                "malformed",
                                {"name": " ", "pass": "yes", "detail": ""},
                            ],
                        },
                        "output_text": "Validated output.",
                    },
                ),
            )
            store.record_session_run(
                surface="thinking",
                model_id="backend-a",
                mode="tool",
                artifact_kind="thinking",
                artifact_path=artifact_path,
                status="ok",
                prompt="Run a deterministic check.",
                system_prompt=None,
                resolved_user_prompt="Run a deterministic check.",
                output_text="Validated output.",
            )

            summary = rebuild_memory_index(root=root)
            index = MemoryIndex(Path(summary["index_path"]))
            matches = index.search("workspace AND validation")
            event = json.loads(index.get_event(matches[0]["event_id"])["payload_json"])
            snapshot, _latest_path, _run_path = record_evaluation_snapshot(root=root)

        self.assertEqual(matches[0]["quality_status"], "pass")
        self.assertEqual(matches[0]["execution_status"], "ok")
        self.assertEqual(
            event["content"]["options"]["quality_checks"],
            [{"name": "compile", "pass": True, "detail": "ok"}],
        )
        self.assertEqual(snapshot["counts"]["test_pass"], 1)

    def test_pending_failures_are_counted_by_source_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            failure_event_id = "local-default:capability-matrix:matrix:row-2:vision"
            explicit_failure = build_evaluation_signal(
                signal_kind="test_fail",
                source_event_id=failure_event_id,
                rationale="Manual failure signal should not double-count the same event.",
            )
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                explicit_failure,
                workspace_id="local-default",
            )

            snapshot, _latest_path, _run_path = record_evaluation_snapshot(root=root)

        self.assertEqual(snapshot["counts"]["test_fail"], 2)
        self.assertEqual(snapshot["counts"]["pending_failures"], 1)
        self.assertEqual(len(snapshot["pending_failures"]), 1)
        self.assertEqual(snapshot["pending_failures"][0]["source_event_id"], failure_event_id)

    def test_review_resolution_signal_promotes_preview_without_exporting_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            pass_event_id = "local-default:capability-matrix:matrix:row-1:chat"
            fail_event_id = "local-default:capability-matrix:matrix:row-2:vision"
            resolved_signal = build_evaluation_signal(
                signal_kind="review_resolved",
                source_event_id=pass_event_id,
                rationale="Review comments were addressed and verified.",
                evidence={
                    "review_id": "review-42",
                    "review_url": "https://example.test/reviews/42",
                    "resolution_summary": "Thread closed after the pass signal.",
                },
            )
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                resolved_signal,
                workspace_id="local-default",
            )
            unresolved_signal = build_evaluation_signal(
                signal_kind="review_unresolved",
                source_event_id=fail_event_id,
                rationale="The failure still needs follow-up.",
                evidence={
                    "review_id": "review-43",
                    "resolution_summary": "The failing check is still open.",
                },
            )
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                unresolved_signal,
                workspace_id="local-default",
            )

            snapshot, _latest_path, _run_path = record_evaluation_snapshot(root=root)
            preview, preview_latest_path, preview_run_path = record_curation_export_preview(
                root=root,
                snapshot=snapshot,
            )
            filtered_preview = build_curation_export_preview(
                snapshot,
                filters={
                    "states": ["ready"],
                    "reasons": ["review_resolved"],
                    "limit": 1,
                },
            )
            report = format_evaluation_snapshot_report(snapshot)
            preview_report = format_curation_export_preview_report(preview)
            preview_latest_exists = preview_latest_path.exists()
            preview_run_exists = preview_run_path.exists()

        self.assertTrue(preview_latest_exists)
        self.assertTrue(preview_run_exists)
        self.assertEqual(snapshot["counts"]["review_resolved"], 1)
        self.assertEqual(snapshot["counts"]["review_unresolved"], 1)
        self.assertEqual(snapshot["counts"]["review_resolution_rate"], 0.5)
        self.assertEqual(snapshot["counts"]["curation_ready"], 1)
        self.assertEqual(snapshot["counts"]["curation_blocked"], 1)
        self.assertEqual(preview["export_mode"], "preview_only")
        self.assertFalse(preview["training_export_ready"])
        self.assertEqual(preview["counts"]["ready"], 1)
        self.assertEqual(preview["counts"]["blocked"], 1)
        self.assertEqual(preview["counts"]["ready_for_policy"], 1)
        self.assertEqual(preview["adoption_checklist_counts"]["human_selection_recorded"]["done"], 1)
        self.assertEqual(preview["adoption_checklist_counts"]["no_blocking_signal"]["blocked"], 1)
        self.assertEqual(preview["candidates"][0]["export_decision"], "include_when_approved")
        self.assertTrue(preview["candidates"][0]["ready_for_policy"])
        self.assertIn("review_resolved", preview["candidates"][0]["reasons"])
        self.assertEqual(filtered_preview["counts"]["candidate_count"], 2)
        self.assertEqual(filtered_preview["counts"]["matched_candidate_count"], 1)
        self.assertEqual(filtered_preview["counts"]["previewed_candidate_count"], 1)
        self.assertEqual(filtered_preview["filters"]["states"], ["ready"])
        self.assertEqual(filtered_preview["candidates"][0]["adoption_checklist"][0]["key"], "test_pass_recorded")
        self.assertIn("Review resolved: 1", report)
        self.assertIn("Curation export preview: preview_only", preview_report)
        self.assertIn("Adoption checklist:", preview_report)

    def test_record_review_resolution_signal_helper_writes_local_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            signal = record_review_resolution_signal(
                root=root,
                source_event_id="local-default:capability-matrix:matrix:row-1:chat",
                resolved=True,
                review_id="review-99",
                resolution_summary="Reviewer accepted the passing candidate.",
                origin="test",
            )
            snapshot, _latest_path, _run_path = record_evaluation_snapshot(root=root)

        self.assertEqual(signal["signal_kind"], "review_resolved")
        self.assertEqual(signal["origin"], "test")
        self.assertEqual(snapshot["counts"]["review_resolved"], 1)
        self.assertEqual(snapshot["counts"]["curation_ready"], 1)

    def test_latest_review_resolution_signal_controls_curation_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            event_id = "local-default:capability-matrix:matrix:row-1:chat"
            unresolved_signal = build_evaluation_signal(
                signal_kind="review_unresolved",
                source_event_id=event_id,
                recorded_at_utc="2026-04-01T00:30:00+01:00",
                signal_id="local-default:eval:20260401T000000000000Z-p1:a",
                rationale="Review is still open.",
            )
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                unresolved_signal,
                workspace_id="local-default",
            )
            resolved_signal = build_evaluation_signal(
                signal_kind="review_resolved",
                source_event_id=event_id,
                recorded_at_utc="2026-04-01T00:00:00Z",
                signal_id="local-default:eval:20260401T000100000000Z-p1:b",
                rationale="Review was resolved later.",
            )
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                resolved_signal,
                workspace_id="local-default",
            )

            snapshot, _latest_path, _run_path = record_evaluation_snapshot(root=root)
            preview = build_curation_export_preview(snapshot)

        self.assertEqual(snapshot["counts"]["review_resolved"], 1)
        self.assertEqual(snapshot["counts"]["review_unresolved"], 1)
        self.assertEqual(snapshot["counts"]["curation_ready"], 1)
        self.assertEqual(snapshot["counts"]["curation_blocked"], 1)
        ready_candidates = [
            item
            for item in preview["candidates"]
            if item["event_id"] == event_id
        ]
        self.assertEqual(ready_candidates[0]["state"], "ready")
        self.assertIn("review_resolved", ready_candidates[0]["reasons"])
        self.assertNotIn("review_unresolved", ready_candidates[0]["reasons"])

    def test_latest_review_unresolved_signal_blocks_curation_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            event_id = "local-default:capability-matrix:matrix:row-1:chat"
            resolved_signal = build_evaluation_signal(
                signal_kind="review_resolved",
                source_event_id=event_id,
                recorded_at_utc="2026-04-01T00:00:00+00:00",
                signal_id="local-default:eval:20260401T000000000000Z-p1:a",
                rationale="Review was initially resolved.",
            )
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                resolved_signal,
                workspace_id="local-default",
            )
            unresolved_signal = build_evaluation_signal(
                signal_kind="review_unresolved",
                source_event_id=event_id,
                recorded_at_utc="2026-04-01T00:01:00+00:00",
                signal_id="local-default:eval:20260401T000100000000Z-p1:b",
                rationale="A later review reopened the concern.",
            )
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                unresolved_signal,
                workspace_id="local-default",
            )

            snapshot, _latest_path, _run_path = record_evaluation_snapshot(root=root)
            preview = build_curation_export_preview(snapshot)

        self.assertEqual(snapshot["counts"]["curation_ready"], 0)
        self.assertEqual(snapshot["counts"]["curation_blocked"], 2)
        blocked_candidates = [
            item
            for item in preview["candidates"]
            if item["event_id"] == event_id
        ]
        self.assertEqual(blocked_candidates[0]["state"], "blocked")
        self.assertIn("review_unresolved", blocked_candidates[0]["reasons"])
        self.assertNotIn("review_resolved", blocked_candidates[0]["reasons"])

    def test_cli_records_explicit_signal_and_prints_json_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            stdout = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "run_evaluation_loop.py",
                    "--root",
                    str(root),
                    "--record-signal",
                    "--signal-kind",
                    "rejection",
                    "--source-event-id",
                    "local-default:capability-matrix:matrix:row-2:vision",
                    "--rationale",
                    "The failure is not acceptable for promotion.",
                    "--format",
                    "json",
                ],
            ), redirect_stdout(stdout):
                exit_code = evaluation_main()
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["recorded_signal"]["signal_kind"], "rejection")
        self.assertEqual(payload["snapshot"]["counts"]["rejection"], 1)
        self.assertEqual(payload["snapshot"]["counts"]["test_fail"], 1)

    def test_cli_reuses_prebuilt_index_for_record_and_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            stdout = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "run_evaluation_loop.py",
                    "--root",
                    str(root),
                    "--record-signal",
                    "--signal-kind",
                    "acceptance",
                    "--source-event-id",
                    "local-default:capability-matrix:matrix:row-1:chat",
                    "--format",
                    "json",
                ],
            ), patch(
                "run_evaluation_loop.rebuild_memory_index",
                wraps=rebuild_memory_index,
            ) as cli_rebuild, patch(
                "evaluation_loop.rebuild_memory_index",
                wraps=rebuild_memory_index,
            ) as snapshot_rebuild, redirect_stdout(stdout):
                exit_code = evaluation_main()
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["snapshot"]["counts"]["acceptance"], 1)
        self.assertEqual(cli_rebuild.call_count, 1)
        self.assertEqual(snapshot_rebuild.call_count, 0)

    def test_cli_reports_invalid_record_arguments_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stderr = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "run_evaluation_loop.py",
                    "--root",
                    str(root),
                    "--record-comparison",
                    "--candidate-event-id",
                    "one",
                ],
            ), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    evaluation_main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("Evaluation comparisons require at least two candidate event ids.", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_rejects_unknown_signal_source_event_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            stderr = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "run_evaluation_loop.py",
                    "--root",
                    str(root),
                    "--record-signal",
                    "--signal-kind",
                    "acceptance",
                    "--source-event-id",
                    "local-default:missing-event",
                ],
            ), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    evaluation_main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("Unknown evaluation source_event_id `local-default:missing-event`.", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_comparison_rejects_unknown_candidate_when_events_are_supplied(self) -> None:
        with self.assertRaises(ValueError) as raised:
            build_evaluation_comparison(
                candidate_event_ids=["known-event", "missing-event"],
                winner_event_id="known-event",
                events_by_id={"known-event": {"event_id": "known-event"}},
            )

        self.assertIn("missing-event", str(raised.exception))

    def test_cli_records_review_resolution_and_curation_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            stdout = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "run_evaluation_loop.py",
                    "--root",
                    str(root),
                    "--record-signal",
                    "--signal-kind",
                    "review_resolved",
                    "--source-event-id",
                    "local-default:capability-matrix:matrix:row-1:chat",
                    "--review-id",
                    "review-42",
                    "--resolution-summary",
                    "Review thread closed.",
                    "--curation-preview",
                    "--curation-state",
                    "ready",
                    "--curation-reason",
                    "review_resolved",
                    "--curation-limit",
                    "1",
                    "--format",
                    "json",
                ],
            ), redirect_stdout(stdout):
                exit_code = evaluation_main()
            payload = json.loads(stdout.getvalue())
            preview_latest_exists = Path(payload["curation_preview"]["paths"]["curation_preview_latest_path"]).exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["recorded_signal"]["signal_kind"], "review_resolved")
        self.assertEqual(payload["recorded_signal"]["evidence"]["review_id"], "review-42")
        self.assertEqual(payload["snapshot"]["counts"]["review_resolved"], 1)
        self.assertEqual(payload["curation_preview"]["export_mode"], "preview_only")
        self.assertEqual(payload["curation_preview"]["filters"]["states"], ["ready"])
        self.assertEqual(payload["curation_preview"]["counts"]["previewed_candidate_count"], 1)
        self.assertTrue(preview_latest_exists)

    def test_cli_records_comparison_and_prints_json_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            stdout = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "run_evaluation_loop.py",
                    "--root",
                    str(root),
                    "--record-comparison",
                    "--candidate-event-id",
                    "local-default:capability-matrix:matrix:row-1:chat",
                    "--candidate-event-id",
                    "local-default:capability-matrix:matrix:row-2:vision",
                    "--winner-event-id",
                    "local-default:capability-matrix:matrix:row-1:chat",
                    "--comparison-label",
                    "pick resilient M4 output",
                    "--criterion",
                    "test pass",
                    "--format",
                    "json",
                ],
            ), redirect_stdout(stdout):
                exit_code = evaluation_main()
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["recorded_comparison"]["outcome"], "winner_selected")
        self.assertEqual(payload["snapshot"]["counts"]["comparisons"], 1)
        self.assertEqual(payload["snapshot"]["comparisons"][0]["winner_event_id"], "local-default:capability-matrix:matrix:row-1:chat")


if __name__ == "__main__":
    unittest.main()
