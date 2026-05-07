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
import evaluation_loop as evaluation_loop_module  # noqa: E402
from evaluation_loop import (  # noqa: E402
    EVALUATION_SIGNAL_SCHEMA_NAME,
    EVALUATION_SIGNAL_SCHEMA_VERSION,
    append_evaluation_comparison,
    append_evaluation_signal,
    build_evaluation_comparison,
    build_evaluation_consistency_report,
    build_evaluation_signal,
    build_human_selected_candidate_list,
    build_jsonl_training_export_dry_run,
    build_learning_candidate_diff_summary,
    evaluation_comparison_log_path,
    evaluation_signal_log_path,
    build_curation_export_preview,
    build_learning_dataset_preview,
    format_curation_export_preview_report,
    format_evaluation_snapshot_report,
    format_human_selected_candidate_list_report,
    format_jsonl_training_export_dry_run_report,
    format_learning_candidate_diff_summary_report,
    format_learning_dataset_preview_report,
    record_curation_export_preview,
    record_export_policy_confirmation_signal,
    record_human_selected_candidate_list,
    record_jsonl_training_export_dry_run,
    record_learning_candidate_diff_summary,
    record_review_resolution_signal,
    record_evaluation_snapshot,
    record_learning_dataset_preview,
)
from memory_index import MemoryIndex, rebuild_memory_index  # noqa: E402
from run_evaluation_loop import main as evaluation_main  # noqa: E402
from software_work_events import write_event_log  # noqa: E402
from workspace_state import WorkspaceSessionStore  # noqa: E402


def readable_test_source_refs(*, artifact_kind: str = "agent_run") -> dict[str, object]:
    return {
        "artifact_ref": {
            "artifact_kind": artifact_kind,
            "artifact_path": str(Path(__file__).resolve()),
        }
    }


def write_capability_matrix(root: Path) -> None:
    matrix_path = root / "artifacts" / "capability_matrix" / "matrix.json"
    pass_artifact_path = root / "artifacts" / "text" / "pass.json"
    fail_artifact_path = root / "artifacts" / "vision" / "fail.json"
    write_artifact(
        pass_artifact_path,
        build_artifact_payload(
            artifact_kind="text",
            status="ok",
            runtime=build_runtime_record(backend="backend-a", model_id="backend-a"),
            prompts=build_prompt_record(prompt="accepted patch keeps the loop green"),
            extra={"output_text": "All checks passed."},
        ),
    )
    write_artifact(
        fail_artifact_path,
        build_artifact_payload(
            artifact_kind="vision",
            status="failed",
            runtime=build_runtime_record(backend="backend-a", model_id="backend-a"),
            prompts=build_prompt_record(prompt="failure records a repair target"),
            extra={"output_text": "AssertionError: expected repair linkage."},
        ),
    )
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
                    "artifact_path": str(pass_artifact_path),
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
                    "artifact_path": str(fail_artifact_path),
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

    def test_learning_preview_builds_traceable_supervised_candidates_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            pass_event_id = "local-default:capability-matrix:matrix:row-1:chat"
            fail_event_id = "local-default:capability-matrix:matrix:row-2:vision"
            accepted_signal = build_evaluation_signal(
                signal_kind="acceptance",
                source_event_id=pass_event_id,
                rationale="The passing result is selected for learning preview.",
            )
            rejected_signal = build_evaluation_signal(
                signal_kind="rejection",
                source_event_id=fail_event_id,
                rationale="The failing result should not become training material.",
            )
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                accepted_signal,
                workspace_id="local-default",
            )
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                rejected_signal,
                workspace_id="local-default",
            )
            comparison = build_evaluation_comparison(
                candidate_event_ids=[pass_event_id, fail_event_id],
                winner_event_id=pass_event_id,
                task_label="choose learning candidate",
                criteria=["accepted", "passing check"],
                rationale="Only the accepted passing event should become a candidate.",
            )
            append_evaluation_comparison(
                evaluation_comparison_log_path(root=root),
                comparison,
                workspace_id="local-default",
            )

            snapshot, _latest_path, _run_path = record_evaluation_snapshot(root=root)
            curation_preview, _curation_latest_path, _curation_run_path = record_curation_export_preview(
                root=root,
                snapshot=snapshot,
            )
            with patch("evaluation_loop._read_json_object") as read_json_mock:
                learning_preview, learning_latest_path, learning_run_path = record_learning_dataset_preview(
                    root=root,
                    snapshot=snapshot,
                    curation_preview=curation_preview,
                )
                direct_preview = build_learning_dataset_preview(snapshot, curation_preview)
                read_json_called = read_json_mock.called
            learning_preview["supervised_example_candidates"][0]["source_event"]["prompt_excerpt"] = (
                "accepted patch\nkeeps the loop green"
            )
            report = format_learning_dataset_preview_report(learning_preview)
            candidate = learning_preview["supervised_example_candidates"][0]
            learning_latest_exists = learning_latest_path.exists()
            learning_run_exists = learning_run_path.exists()
            signal_kinds = {
                signal["signal_kind"]
                for signal in candidate["evidence"]["signals"]
            }
            comparison_roles = {
                item["role"]
                for item in candidate["evidence"]["comparisons"]
            }
            first_queue_item = learning_preview["review_queue"][0]
            excluded_by_event = {
                item["event_id"]: item
                for item in learning_preview["excluded_candidates"]
            }

        self.assertTrue(learning_latest_exists)
        self.assertTrue(learning_run_exists)
        self.assertEqual(learning_preview["export_mode"], "preview_only")
        self.assertFalse(learning_preview["training_export_ready"])
        self.assertTrue(learning_preview["human_gate_required"])
        self.assertEqual(learning_preview["counts"]["source_candidate_count"], 2)
        self.assertEqual(learning_preview["counts"]["eligible_candidate_count"], 1)
        self.assertEqual(learning_preview["counts"]["previewed_candidate_count"], 1)
        self.assertEqual(learning_preview["counts"]["excluded_candidate_count"], 1)
        self.assertEqual(learning_preview["counts"]["review_queue_count"], 2)
        self.assertEqual(learning_preview["counts"]["review_queue_states"]["ready"], 1)
        self.assertEqual(learning_preview["counts"]["review_queue_states"]["blocked"], 1)
        self.assertEqual(learning_preview["counts"]["review_queue_lifecycle_states"]["queued"], 1)
        self.assertEqual(learning_preview["counts"]["review_queue_lifecycle_states"]["failed"], 1)
        self.assertEqual(
            learning_preview["counts"]["review_queue_next_actions"]["confirm_export_policy"],
            1,
        )
        self.assertEqual(
            learning_preview["counts"]["review_queue_next_actions"]["repair_or_follow_up_failure"],
            1,
        )
        self.assertEqual(
            learning_preview["counts"]["review_queue_blocked_reasons"]["test_fail"],
            1,
        )
        self.assertEqual(direct_preview["counts"]["eligible_candidate_count"], 1)
        self.assertFalse(read_json_called)
        self.assertEqual(candidate["event_id"], pass_event_id)
        self.assertEqual(candidate["source_event"]["source_event_id"], pass_event_id)
        self.assertEqual(candidate["supervised_example"]["format"], "instruction_response")
        self.assertEqual(candidate["review_queue"]["queue_state"], "ready")
        self.assertEqual(candidate["review_queue"]["next_action"], "confirm_export_policy")
        self.assertEqual(candidate["review_queue"]["queue_priority"]["bucket"], "ready_policy_unconfirmed")
        self.assertEqual(candidate["review_queue"]["lifecycle_summary"]["lifecycle_state"], "queued")
        self.assertTrue(candidate["review_queue"]["eligible_for_supervised_candidate"])
        self.assertEqual(candidate["review_queue"]["excluded_by"], [])
        self.assertFalse(candidate["policy"]["raw_log_export_allowed"])
        self.assertFalse(candidate["policy"]["training_job_allowed"])
        self.assertIn("acceptance", signal_kinds)
        self.assertIn("test_pass", signal_kinds)
        self.assertIn("winner", comparison_roles)
        self.assertEqual(candidate["backend_metadata"]["model_id"], "backend-a")
        self.assertIn("event_log_path", candidate["source_paths"])
        self.assertIn("signal_log_path", candidate["source_paths"])
        self.assertIn("comparison_log_path", candidate["source_paths"])
        self.assertEqual(first_queue_item["event_id"], fail_event_id)
        self.assertEqual(first_queue_item["queue_priority"]["bucket"], "blocked_first")
        self.assertEqual(first_queue_item["lifecycle_summary"]["lifecycle_state"], "failed")
        self.assertNotIn("output_excerpt", first_queue_item["source_event"])
        self.assertEqual(excluded_by_event[fail_event_id]["queue_state"], "blocked")
        self.assertFalse(excluded_by_event[fail_event_id]["eligible_for_supervised_candidate"])
        self.assertEqual(excluded_by_event[fail_event_id]["blocked_reason"], "test_fail")
        self.assertEqual(excluded_by_event[fail_event_id]["next_action"], "repair_or_follow_up_failure")
        self.assertIn("Learning dataset preview: preview_only", report)
        self.assertIn("- accepted patch keeps the loop green", report)
        self.assertNotIn("- accepted patch\nkeeps the loop green", report)
        self.assertIn("Learning review queue:", report)
        self.assertIn("Lifecycle states: queued=1; failed=1", report)
        self.assertIn("Queue next actions: confirm_export_policy=1", report)
        self.assertIn("Queue blocked reasons:", report)
        self.assertIn("test_fail=1", report)
        self.assertIn("next=confirm_export_policy", report)
        self.assertIn("blocking_or_noisy_signal", learning_preview["counts"]["exclusion_reasons"])

    def test_export_policy_confirmation_signal_marks_preview_without_training_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            event_id = "local-default:capability-matrix:matrix:row-1:chat"
            accepted_signal = build_evaluation_signal(
                signal_kind="acceptance",
                source_event_id=event_id,
                rationale="The passing result is accepted for preview inspection.",
            )
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                accepted_signal,
                workspace_id="local-default",
            )
            policy_signal = record_export_policy_confirmation_signal(
                root=root,
                source_event_id=event_id,
                rationale="Human confirmed the preview-only export policy.",
                origin="test",
            )

            snapshot, _latest_path, _run_path = record_evaluation_snapshot(root=root)
            curation_preview, _curation_latest_path, _curation_run_path = record_curation_export_preview(
                root=root,
                snapshot=snapshot,
            )
            learning_preview, _learning_latest_path, _learning_run_path = record_learning_dataset_preview(
                root=root,
                snapshot=snapshot,
                curation_preview=curation_preview,
            )
            curation_candidate = [
                item
                for item in curation_preview["candidates"]
                if item["event_id"] == event_id
            ][0]
            checklist_by_key = {
                item["key"]: item
                for item in curation_candidate["adoption_checklist"]
            }
            candidate = learning_preview["supervised_example_candidates"][0]
            signal_kinds = {
                signal["signal_kind"]
                for signal in candidate["evidence"]["signals"]
            }

        self.assertEqual(policy_signal["signal_kind"], "export_policy_confirmed")
        self.assertEqual(policy_signal["origin"], "test")
        self.assertEqual(policy_signal["evidence"]["export_mode"], "preview_only")
        self.assertFalse(policy_signal["evidence"]["training_export_ready"])
        self.assertTrue(policy_signal["evidence"]["human_gate_required"])
        self.assertFalse(policy_signal["evidence"]["training_job_allowed"])
        self.assertFalse(policy_signal["evidence"]["raw_log_export_allowed"])
        self.assertEqual(snapshot["counts"]["export_policy_confirmed"], 1)
        self.assertIn("Export policy confirmed: 1", format_evaluation_snapshot_report(snapshot))
        self.assertIn("export_policy_confirmed", curation_candidate["reasons"])
        self.assertEqual(checklist_by_key["export_policy_confirmed"]["status"], "done")
        self.assertEqual(
            curation_preview["adoption_checklist_counts"]["export_policy_confirmed"]["done"],
            1,
        )
        self.assertEqual(curation_candidate["required_next_steps"], ["review_downstream_export_policy"])
        self.assertEqual(learning_preview["export_mode"], "preview_only")
        self.assertFalse(learning_preview["training_export_ready"])
        self.assertTrue(learning_preview["human_gate_required"])
        self.assertEqual(learning_preview["counts"]["policy_confirmed_candidate_count"], 1)
        self.assertIn("export_policy_confirmed", signal_kinds)
        self.assertEqual(candidate["review_queue"]["next_action"], "review_downstream_export_policy")
        self.assertEqual(candidate["review_queue"]["queue_priority"]["bucket"], "ready_policy_confirmed")
        self.assertEqual(candidate["review_queue"]["lifecycle_summary"]["lifecycle_state"], "completed")
        self.assertEqual(candidate["review_queue"]["lifecycle_summary"]["policy_state"], "confirmed")
        self.assertTrue(candidate["review_queue"]["export_policy_confirmation"]["confirmed"])
        self.assertTrue(candidate["policy"]["export_policy_confirmed"])
        self.assertEqual(candidate["policy"]["confirmation_signal_id"], policy_signal["signal_id"])
        self.assertFalse(candidate["policy"]["training_job_allowed"])
        self.assertFalse(candidate["policy"]["raw_log_export_allowed"])

    def test_human_selected_candidate_list_records_explicit_selection_without_training_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            pass_event_id = "local-default:capability-matrix:matrix:row-1:chat"
            fail_event_id = "local-default:capability-matrix:matrix:row-2:vision"
            accepted_signal = build_evaluation_signal(
                signal_kind="acceptance",
                source_event_id=pass_event_id,
                rationale="The passing candidate is explicitly accepted.",
            )
            rejected_signal = build_evaluation_signal(
                signal_kind="rejection",
                source_event_id=fail_event_id,
                rationale="The failing candidate should remain blocked.",
            )
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                accepted_signal,
                workspace_id="local-default",
            )
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                rejected_signal,
                workspace_id="local-default",
            )
            policy_signal = record_export_policy_confirmation_signal(
                root=root,
                source_event_id=pass_event_id,
                rationale="Human confirmed the preview-only policy before shortlist inspection.",
                origin="test",
            )

            snapshot, _latest_path, _run_path = record_evaluation_snapshot(root=root)
            curation_preview, _curation_latest_path, _curation_run_path = record_curation_export_preview(
                root=root,
                snapshot=snapshot,
            )
            learning_preview, _learning_latest_path, _learning_run_path = record_learning_dataset_preview(
                root=root,
                snapshot=snapshot,
                curation_preview=curation_preview,
            )
            selection, selection_latest_path, selection_run_path = record_human_selected_candidate_list(
                root=root,
                learning_preview=learning_preview,
                selected_event_ids=[pass_event_id, fail_event_id],
                rationale="Human-selected M7.3 shortlist.",
                origin="test",
            )
            selected_by_event = {
                item["event_id"]: item
                for item in selection["selected_candidates"]
            }
            report = format_human_selected_candidate_list_report(selection)
            selection_latest_exists = selection_latest_path.exists()
            selection_run_exists = selection_run_path.exists()

        self.assertTrue(selection_latest_exists)
        self.assertTrue(selection_run_exists)
        self.assertEqual(selection["schema_name"], "software-satellite-human-selected-candidate-list")
        self.assertEqual(selection["export_mode"], "preview_only")
        self.assertFalse(selection["training_export_ready"])
        self.assertTrue(selection["human_gate_required"])
        self.assertFalse(selection["export_policy"]["training_job_allowed"])
        self.assertFalse(selection["export_policy"]["raw_log_export_allowed"])
        self.assertFalse(selection["export_policy"]["jsonl_training_export_allowed"])
        self.assertTrue(selection["export_policy"]["selection_does_not_promote_candidate"])
        self.assertEqual(selection["counts"]["requested_candidate_count"], 2)
        self.assertEqual(selection["counts"]["matched_candidate_count"], 2)
        self.assertEqual(selection["counts"]["selected_supervised_candidate_count"], 1)
        self.assertEqual(selection["counts"]["selected_not_supervised_candidate_count"], 1)
        self.assertEqual(selection["counts"]["policy_confirmed_selected_count"], 1)
        self.assertEqual(selection["selection"]["selection_mode"], "explicit_human_candidate_list")
        self.assertEqual(selection["selection"]["rationale"], "Human-selected M7.3 shortlist.")
        self.assertNotIn("supervised_example", selected_by_event[pass_event_id])
        self.assertEqual(selected_by_event[pass_event_id]["label"], pass_event_id)
        self.assertNotIn("prompt_excerpt", selected_by_event[pass_event_id]["source_event"])
        self.assertNotIn("output_excerpt", selected_by_event[pass_event_id]["source_event"])
        self.assertTrue(selected_by_event[pass_event_id]["eligible_for_supervised_candidate"])
        self.assertTrue(selected_by_event[pass_event_id]["preview_membership"]["in_supervised_example_candidates"])
        self.assertTrue(selected_by_event[pass_event_id]["policy"]["export_policy_confirmed"])
        self.assertFalse(selected_by_event[pass_event_id]["policy"]["training_export_ready"])
        self.assertIn(accepted_signal["signal_id"], selected_by_event[pass_event_id]["evidence_summary"]["signal_ids"])
        self.assertIn(policy_signal["signal_id"], selected_by_event[pass_event_id]["evidence_summary"]["signal_ids"])
        self.assertEqual(
            selected_by_event[pass_event_id]["evidence_summary"]["export_policy_confirmation_signal_id"],
            policy_signal["signal_id"],
        )
        self.assertTrue(selected_by_event[pass_event_id]["evidence_summary"]["traceability"]["accepted"])
        self.assertTrue(selected_by_event[pass_event_id]["evidence_summary"]["traceability"]["test_pass"])
        self.assertTrue(
            selected_by_event[pass_event_id]["evidence_summary"]["traceability"]["export_policy_confirmed"]
        )
        self.assertFalse(selected_by_event[fail_event_id]["eligible_for_supervised_candidate"])
        self.assertTrue(selected_by_event[fail_event_id]["preview_membership"]["in_excluded_candidates"])
        self.assertEqual(selected_by_event[fail_event_id]["blocked_reason"], "test_fail")
        self.assertEqual(selected_by_event[fail_event_id]["next_action"], "repair_or_follow_up_failure")
        self.assertIn(rejected_signal["signal_id"], selected_by_event[fail_event_id]["evidence_summary"]["signal_ids"])
        self.assertIn("test_fail", selected_by_event[fail_event_id]["evidence_summary"]["signal_kinds"])
        self.assertFalse(selected_by_event[fail_event_id]["evidence_summary"]["traceability"]["test_pass"])
        self.assertIn("Human-selected candidate list: preview_only", report)
        self.assertIn("Training export ready: no", report)
        self.assertIn("supervised=yes", report)
        self.assertIn("supervised=no", report)

    def test_human_selected_candidate_list_keeps_missing_ids_as_inspection_items(self) -> None:
        preview = build_human_selected_candidate_list(
            {
                "workspace_id": "local-default",
                "paths": {},
                "review_queue": [],
                "supervised_example_candidates": [],
                "excluded_candidates": [],
            },
            selected_event_ids=["local-default:missing-selection"],
            rationale="Keep missing selections visible.",
            origin="test",
        )
        selected = preview["selected_candidates"][0]

        self.assertEqual(preview["counts"]["missing_candidate_count"], 1)
        self.assertEqual(selected["queue_state"], "missing_from_learning_preview")
        self.assertEqual(selected["blocked_reason"], "missing_learning_preview_candidate")
        self.assertFalse(selected["eligible_for_supervised_candidate"])
        self.assertIn("missing_learning_preview_candidate", selected["excluded_by"])
        self.assertFalse(preview["training_export_ready"])

    def test_candidate_diff_summary_compares_learning_preview_and_human_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            pass_event_id = "local-default:capability-matrix:matrix:row-1:chat"
            fail_event_id = "local-default:capability-matrix:matrix:row-2:vision"
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                build_evaluation_signal(
                    signal_kind="acceptance",
                    source_event_id=pass_event_id,
                    rationale="Accepted before candidate diff inspection.",
                ),
                workspace_id="local-default",
            )
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                build_evaluation_signal(
                    signal_kind="rejection",
                    source_event_id=fail_event_id,
                    rationale="Rejected candidates should remain visible as removed from the shortlist.",
                ),
                workspace_id="local-default",
            )
            snapshot, _latest_path, _run_path = record_evaluation_snapshot(root=root)
            curation_preview, _curation_latest_path, _curation_run_path = record_curation_export_preview(
                root=root,
                snapshot=snapshot,
            )
            learning_preview, _learning_latest_path, _learning_run_path = record_learning_dataset_preview(
                root=root,
                snapshot=snapshot,
                curation_preview=curation_preview,
            )
            selection, _selection_latest_path, _selection_run_path = record_human_selected_candidate_list(
                root=root,
                learning_preview=learning_preview,
                selected_event_ids=[pass_event_id],
                rationale="Select one candidate for diff inspection.",
                origin="test",
            )
            diff, diff_latest_path, diff_run_path = record_learning_candidate_diff_summary(
                root=root,
                base_artifact=learning_preview,
                target_artifact=selection,
                base_label="learning_preview",
                target_label="human_selected_candidate_list",
            )
            report = format_learning_candidate_diff_summary_report(diff)
            diff_latest_exists = diff_latest_path.exists()
            diff_run_exists = diff_run_path.exists()

        self.assertTrue(diff_latest_exists)
        self.assertTrue(diff_run_exists)
        self.assertEqual(diff["schema_name"], "software-satellite-learning-candidate-diff-summary")
        self.assertEqual(diff["export_mode"], "preview_only")
        self.assertFalse(diff["training_export_ready"])
        self.assertTrue(diff["human_gate_required"])
        self.assertTrue(diff["inspection_policy"]["diff_does_not_promote_candidate"])
        self.assertEqual(diff["counts"]["base_candidate_count"], 2)
        self.assertEqual(diff["counts"]["target_candidate_count"], 1)
        self.assertEqual(diff["counts"]["removed_candidate_count"], 1)
        self.assertEqual(diff["counts"]["unchanged_candidate_count"], 1)
        self.assertEqual(diff["base"]["counts"]["queue_states"]["blocked"], 1)
        self.assertEqual(diff["base"]["counts"]["queue_states"]["ready"], 1)
        self.assertEqual(sum(diff["target"]["counts"]["backend_ids"].values()), 1)
        self.assertIn("Candidate diff summary: preview_only", report)
        self.assertIn(str(diff_run_path), report)
        self.assertIn("Removed: 1", report)
        self.assertIn("Training export ready: no", report)

    def test_candidate_diff_summary_tracks_preview_field_changes(self) -> None:
        base_preview = {
            "schema_name": "software-satellite-learning-dataset-preview",
            "schema_version": 1,
            "workspace_id": "local-default",
            "export_mode": "preview_only",
            "training_export_ready": False,
            "human_gate_required": True,
            "review_queue": [
                {
                    "event_id": "candidate-a",
                    "label": "Candidate A",
                    "queue_state": "needs_review",
                    "next_action": "record_acceptance_or_review_resolution",
                    "blocked_reasons": [],
                    "lifecycle_summary": {
                        "lifecycle_state": "queued",
                        "policy_state": "pending_confirmation",
                    },
                    "backend_metadata": {"backend_id": "backend-a", "model_id": "model-a"},
                    "comparison_evidence": {"roles": ["candidate"]},
                    "eligible_for_supervised_candidate": False,
                }
            ],
            "supervised_example_candidates": [],
            "excluded_candidates": [],
        }
        target_preview = {
            **base_preview,
            "review_queue": [
                {
                    "event_id": "candidate-a",
                    "label": "Candidate A",
                    "queue_state": "ready",
                    "next_action": "review_downstream_export_policy",
                    "blocked_reasons": [],
                    "lifecycle_summary": {
                        "lifecycle_state": "completed",
                        "policy_state": "confirmed",
                    },
                    "backend_metadata": {"backend_id": "backend-b", "model_id": "model-b"},
                    "comparison_evidence": {"roles": ["winner"]},
                    "eligible_for_supervised_candidate": True,
                }
            ],
        }

        diff = build_learning_candidate_diff_summary(
            base_preview,
            target_preview,
            base_label="previous_learning_preview",
            target_label="learning_preview",
        )
        change = diff["changes"][0]

        self.assertEqual(diff["counts"]["changed_candidate_count"], 1)
        self.assertEqual(diff["counts"]["field_change_counts"]["queue_state"], 1)
        self.assertEqual(diff["counts"]["field_change_counts"]["next_action"], 1)
        self.assertEqual(diff["counts"]["field_change_counts"]["lifecycle_state"], 1)
        self.assertEqual(diff["counts"]["field_change_counts"]["policy_state"], 1)
        self.assertEqual(diff["counts"]["field_change_counts"]["comparison_role"], 1)
        self.assertEqual(diff["counts"]["field_change_counts"]["backend_id"], 1)
        self.assertEqual(change["event_id"], "candidate-a")
        self.assertIn("queue_state", change["changed_fields"])
        self.assertIn("lifecycle_state", change["changed_fields"])
        self.assertEqual(change["before"]["comparison_role"], "candidate")
        self.assertEqual(change["after"]["comparison_role"], "winner")
        self.assertEqual(change["before"]["backend_id"], "backend-a")
        self.assertEqual(change["after"]["backend_id"], "backend-b")

    def test_candidate_diff_normalizes_legacy_lifecycle_fallbacks(self) -> None:
        preview = {
            "schema_name": "software-satellite-learning-dataset-preview",
            "schema_version": 1,
            "workspace_id": "local-default",
            "export_mode": "preview_only",
            "training_export_ready": False,
            "human_gate_required": True,
            "review_queue": [
                {
                    "event_id": "paused-without-lifecycle",
                    "label": "Paused legacy candidate",
                    "queue_state": "ready",
                    "next_action": "complete_adoption_checklist",
                    "lifecycle_summary": {"policy_state": "confirmed_but_not_ready"},
                },
                {
                    "event_id": "invalid-lifecycle",
                    "label": "Invalid lifecycle candidate",
                    "queue_state": "ready",
                    "next_action": "confirm_export_policy",
                    "lifecycle_summary": {
                        "lifecycle_state": "almost_done",
                        "policy_state": "pending_confirmation",
                    },
                },
                {
                    "event_id": "blocked-policy",
                    "label": "Blocked wins over policy",
                    "queue_state": "missing_source",
                    "next_action": "restore_source_event",
                    "lifecycle_summary": {"policy_state": "confirmed_but_not_ready"},
                },
            ],
            "supervised_example_candidates": [],
            "excluded_candidates": [],
        }

        diff = build_learning_candidate_diff_summary(
            preview,
            preview,
            base_label="legacy",
            target_label="legacy",
        )

        self.assertEqual(diff["base"]["counts"]["lifecycle_states"]["paused"], 1)
        self.assertEqual(diff["base"]["counts"]["lifecycle_states"]["blocked"], 1)
        self.assertEqual(diff["base"]["counts"]["lifecycle_states"]["unknown"], 1)

    def test_jsonl_export_dry_run_from_human_selected_candidates_writes_manifest_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            pass_event_id = "local-default:capability-matrix:matrix:row-1:chat"
            fail_event_id = "local-default:capability-matrix:matrix:row-2:vision"
            accepted_signal = build_evaluation_signal(
                signal_kind="acceptance",
                source_event_id=pass_event_id,
                rationale="The passing candidate is accepted for preview inspection.",
            )
            rejected_signal = build_evaluation_signal(
                signal_kind="rejection",
                source_event_id=fail_event_id,
                rationale="The failing candidate must not become an export candidate.",
            )
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                accepted_signal,
                workspace_id="local-default",
            )
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                rejected_signal,
                workspace_id="local-default",
            )
            policy_signal = record_export_policy_confirmation_signal(
                root=root,
                source_event_id=pass_event_id,
                rationale="Human confirmed preview-only policy before the dry-run.",
                origin="test",
            )
            snapshot, _latest_path, _run_path = record_evaluation_snapshot(root=root)
            curation_preview, _curation_latest_path, _curation_run_path = record_curation_export_preview(
                root=root,
                snapshot=snapshot,
            )
            learning_preview, _learning_latest_path, _learning_run_path = record_learning_dataset_preview(
                root=root,
                snapshot=snapshot,
                curation_preview=curation_preview,
            )
            selection, _selection_latest_path, _selection_run_path = record_human_selected_candidate_list(
                root=root,
                learning_preview=learning_preview,
                selected_event_ids=[pass_event_id, fail_event_id],
                rationale="Select candidates for dry-run inspection.",
                origin="test",
            )
            dry_run, dry_run_latest_path, dry_run_run_path = record_jsonl_training_export_dry_run(
                root=root,
                learning_preview=learning_preview,
                human_selected_candidates=selection,
            )
            candidates_by_event = {
                item["event_id"]: item
                for item in dry_run["candidates"]
            }
            report = format_jsonl_training_export_dry_run_report(dry_run)
            dry_run_latest_exists = dry_run_latest_path.exists()
            dry_run_run_exists = dry_run_run_path.exists()

        self.assertTrue(dry_run_latest_exists)
        self.assertTrue(dry_run_run_exists)
        self.assertEqual(dry_run_run_path.suffix, ".json")
        self.assertEqual(dry_run["schema_name"], "software-satellite-jsonl-training-export-dry-run")
        self.assertEqual(dry_run["export_mode"], "preview_only")
        self.assertFalse(dry_run["training_export_ready"])
        self.assertTrue(dry_run["human_gate_required"])
        self.assertTrue(dry_run["not_trainable"])
        self.assertFalse(dry_run["export_policy"]["jsonl_training_export_allowed"])
        self.assertFalse(dry_run["export_policy"]["jsonl_file_written"])
        self.assertFalse(dry_run["dry_run_manifest"]["trainable_artifact_written"])
        self.assertEqual(dry_run["counts"]["source_candidate_count"], 2)
        self.assertEqual(dry_run["counts"]["inspected_candidate_count"], 2)
        self.assertEqual(dry_run["counts"]["future_jsonl_candidate_if_separately_approved_count"], 1)
        self.assertEqual(dry_run["counts"]["blocked_candidate_count"], 1)
        self.assertEqual(dry_run["counts"]["would_write_jsonl_record_count"], 0)
        self.assertEqual(dry_run["counts"]["supervised_example_text_copied_count"], 0)
        self.assertEqual(dry_run["counts"]["raw_log_text_copied_count"], 0)
        self.assertEqual(
            candidates_by_event[pass_event_id]["dry_run_status"],
            "future_jsonl_candidate_if_separately_approved",
        )
        self.assertFalse(candidates_by_event[pass_event_id]["would_write_jsonl_record"])
        self.assertTrue(
            candidates_by_event[pass_event_id]["dry_run_eligible_for_future_export_if_separately_approved"]
        )
        self.assertNotIn("supervised_example", candidates_by_event[pass_event_id])
        self.assertNotIn("supervised_example", candidates_by_event[pass_event_id]["source_paths"])
        self.assertTrue(candidates_by_event[pass_event_id]["policy"]["export_policy_confirmed"])
        self.assertEqual(
            candidates_by_event[pass_event_id]["policy"]["confirmation_signal_id"],
            policy_signal["signal_id"],
        )
        self.assertTrue(candidates_by_event[pass_event_id]["evidence_summary"]["traceability"]["accepted"])
        self.assertTrue(candidates_by_event[pass_event_id]["evidence_summary"]["traceability"]["test_pass"])
        self.assertTrue(
            candidates_by_event[pass_event_id]["evidence_summary"]["traceability"]["export_policy_confirmed"]
        )
        self.assertEqual(candidates_by_event[fail_event_id]["dry_run_status"], "not_supervised_candidate")
        self.assertFalse(
            candidates_by_event[fail_event_id]["dry_run_eligible_for_future_export_if_separately_approved"]
        )
        self.assertFalse(candidates_by_event[fail_event_id]["would_write_jsonl_record"])
        self.assertIn(rejected_signal["signal_id"], candidates_by_event[fail_event_id]["evidence_summary"]["signal_ids"])
        self.assertIn("JSONL training export dry-run: preview_only", report)
        self.assertIn("JSONL file written: no", report)
        self.assertIn("Trainable artifact: no", report)

    def test_jsonl_export_dry_run_from_learning_preview_keeps_manifest_only(self) -> None:
        event = {
            "event_id": "policy-confirmed-learning-candidate",
            "event_kind": "agent_task_run",
            "recorded_at_utc": "2026-04-01T00:00:00+00:00",
            "session": {"surface": "agent_lane", "mode": "patch_plan_verify"},
            "outcome": {"status": "ok", "quality_status": "pass", "execution_status": "ok"},
            "content": {
                "prompt": "Keep dry-runs metadata-only.",
                "output_text": "No JSONL file is emitted.",
                "options": {
                    "validation_command": "python -m unittest tests.test_dry_run",
                    "pass_definition": "Dry-run remains preview-only.",
                },
            },
            "source_refs": readable_test_source_refs(),
        }
        accepted_signal = build_evaluation_signal(
            signal_kind="acceptance",
            source_event_id="policy-confirmed-learning-candidate",
            source_event=event,
        )
        policy_signal = build_evaluation_signal(
            signal_kind="export_policy_confirmed",
            source_event_id="policy-confirmed-learning-candidate",
            source_event=event,
        )
        learning_preview = build_learning_dataset_preview(
            {"workspace_id": "local-default", "paths": {}},
            {
                "candidates": [
                    {
                        "event_id": "policy-confirmed-learning-candidate",
                        "state": "ready",
                        "label": "Policy-confirmed learning candidate",
                        "reasons": ["accepted", "test_pass"],
                        "blocked_by": [],
                        "export_decision": "include_when_approved",
                        "ready_for_policy": True,
                    }
                ]
            },
            events_by_id={"policy-confirmed-learning-candidate": event},
            explicit_signals=[accepted_signal, policy_signal],
            comparisons=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dry_run, _latest_path, _run_path = record_jsonl_training_export_dry_run(
                root=root,
                learning_preview=learning_preview,
            )
            candidate = dry_run["candidates"][0]

        self.assertEqual(dry_run["source_mode"], "learning_preview_supervised_candidates")
        self.assertFalse(dry_run["export_policy"]["jsonl_file_written"])
        self.assertEqual(dry_run["counts"]["would_write_jsonl_record_count"], 0)
        self.assertTrue(candidate["dry_run_eligible_for_future_export_if_separately_approved"])
        self.assertFalse(candidate["would_write_jsonl_record"])
        self.assertFalse(candidate["jsonl_projection"]["supervised_example_text_copied"])
        self.assertFalse(candidate["jsonl_projection"]["raw_log_text_copied"])

    def test_jsonl_export_dry_run_requires_traceability_before_future_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            learning_preview_path = root / "learning-preview.json"
            learning_preview_path.write_text("{}", encoding="utf-8")
            dry_run, _latest_path, _run_path = record_jsonl_training_export_dry_run(
                root=root,
                human_selected_candidates={
                    "workspace_id": "local-default",
                    "paths": {},
                    "source_learning_preview_path": str(learning_preview_path),
                    "source_paths": {"source_learning_preview_path": str(learning_preview_path)},
                    "selected_candidates": [
                        {
                            "event_id": "spoofed-ready-candidate",
                            "preview_membership": {
                                "in_review_queue": True,
                                "in_supervised_example_candidates": True,
                                "in_excluded_candidates": False,
                            },
                            "eligible_for_supervised_candidate": True,
                            "policy": {
                                "export_policy_confirmed": True,
                                "training_export_ready": True,
                                "training_job_allowed": True,
                                "raw_log_export_allowed": True,
                                "jsonl_training_export_allowed": True,
                            },
                            "evidence_summary": {
                                "traceability": {
                                    "test_pass": False,
                                    "accepted": False,
                                    "review_resolved": False,
                                    "comparison_winner": False,
                                    "export_policy_confirmed": False,
                                }
                            },
                        }
                    ],
                },
            )
            candidate = dry_run["candidates"][0]

        self.assertEqual(dry_run["counts"]["future_jsonl_candidate_if_separately_approved_count"], 0)
        self.assertEqual(candidate["dry_run_status"], "missing_required_traceability")
        self.assertFalse(candidate["dry_run_eligible_for_future_export_if_separately_approved"])
        self.assertFalse(candidate["would_write_jsonl_record"])
        self.assertFalse(candidate["policy"]["training_job_allowed"])
        self.assertFalse(candidate["policy"]["raw_log_export_allowed"])
        self.assertFalse(candidate["policy"]["jsonl_training_export_allowed"])
        self.assertIn("missing_required_traceability", candidate["blocked_reasons"])
        self.assertIn("restore_required_traceability", candidate["required_before_training_export"])

    def test_jsonl_export_dry_run_defaults_unknown_candidates_to_non_supervised(self) -> None:
        dry_run = build_jsonl_training_export_dry_run(
            human_selected_candidates={
                "workspace_id": "local-default",
                "paths": {},
                "source_paths": {},
                "selected_candidates": [
                    {
                        "event_id": "unknown-candidate-with-spoofed-trace",
                        "policy": {"export_policy_confirmed": True},
                        "evidence_summary": {
                            "traceability": {
                                "test_pass": True,
                                "accepted": True,
                                "review_resolved": False,
                                "comparison_winner": False,
                                "export_policy_confirmed": True,
                            }
                        },
                    }
                ],
            }
        )
        candidate = dry_run["candidates"][0]

        self.assertFalse(candidate["preview_membership"]["in_supervised_example_candidates"])
        self.assertFalse(candidate["eligible_for_supervised_candidate"])
        self.assertEqual(dry_run["counts"]["future_jsonl_candidate_if_separately_approved_count"], 0)
        self.assertEqual(candidate["dry_run_status"], "not_supervised_candidate")
        self.assertFalse(candidate["would_write_jsonl_record"])
        self.assertIn("not_eligible_for_supervised_candidate", candidate["blocked_reasons"])

    def test_jsonl_export_dry_run_persists_supplied_human_selected_candidates(self) -> None:
        supplied_selection = build_human_selected_candidate_list(
            {
                "workspace_id": "local-default",
                "paths": {},
                "review_queue": [],
                "supervised_example_candidates": [],
                "excluded_candidates": [],
            },
            selected_event_ids=["local-default:missing-selection"],
            rationale="Keep missing selections visible in dry-run review.",
            origin="test",
        )
        supplied_selection["selected_candidates"][0]["evidence_summary"]["supervised_example"] = {
            "instruction": "This malformed text must not be copied.",
            "response": "Dry-run artifacts stay metadata-only.",
        }
        supplied_selection["selected_candidates"][0]["source_event"] = {
            "event_id": "local-default:missing-selection",
            "prompt_excerpt": "Do not persist prompt previews from supplied selections.",
            "output_excerpt": "Do not persist output previews from supplied selections.",
        }
        supplied_selection["selected_candidates"][0]["source_paths"]["prompt"] = "training prompt text"
        supplied_selection["selected_candidates"][0]["source_paths"]["output_text"] = "training output text"
        supplied_selection["selected_candidates"][0]["source_paths"]["input_text"] = "training input text"
        supplied_selection["selected_candidates"][0]["source_paths"]["completion"] = "training completion text"
        supplied_selection["selected_candidates"][0]["messages"] = [
            {"role": "user", "content": "Do not persist chat messages from supplied selections."}
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dry_run, _latest_path, _run_path = record_jsonl_training_export_dry_run(
                root=root,
                human_selected_candidates=supplied_selection,
            )
            source_selection_path = Path(str(dry_run["source_human_selected_candidates_path"]))
            source_selection_exists = source_selection_path.exists()
            persisted_selection = json.loads(source_selection_path.read_text(encoding="utf-8"))
            persisted_candidate = persisted_selection["selected_candidates"][0]
            candidate = dry_run["candidates"][0]

        self.assertTrue(source_selection_exists)
        self.assertEqual(source_selection_path.suffix, ".json")
        self.assertEqual(persisted_candidate["event_id"], "local-default:missing-selection")
        self.assertIn("traceability", persisted_candidate["evidence_summary"])
        self.assertNotIn("supervised_example", persisted_candidate["evidence_summary"])
        self.assertNotIn("prompt_excerpt", persisted_candidate["source_event"])
        self.assertNotIn("output_excerpt", persisted_candidate["source_event"])
        self.assertNotIn("prompt", persisted_candidate["source_paths"])
        self.assertNotIn("output_text", persisted_candidate["source_paths"])
        self.assertNotIn("input_text", persisted_candidate["source_paths"])
        self.assertNotIn("completion", persisted_candidate["source_paths"])
        self.assertNotIn("messages", persisted_candidate)
        self.assertFalse(dry_run["training_export_ready"])
        self.assertFalse(dry_run["export_policy"]["jsonl_file_written"])
        self.assertEqual(dry_run["counts"]["missing_candidate_count"], 1)
        self.assertEqual(dry_run["counts"]["future_jsonl_candidate_if_separately_approved_count"], 0)
        self.assertEqual(candidate["dry_run_status"], "missing_learning_preview_candidate")
        self.assertFalse(candidate["would_write_jsonl_record"])
        self.assertIn("missing_learning_preview_candidate", candidate["blocked_reasons"])
        self.assertNotIn("supervised_example", candidate["evidence_summary"])

    def test_jsonl_export_dry_run_prefers_persisted_human_selected_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            persisted_selection, _latest_path, _run_path = record_human_selected_candidate_list(
                root=root,
                learning_preview={
                    "workspace_id": "local-default",
                    "paths": {},
                    "review_queue": [],
                    "supervised_example_candidates": [],
                    "excluded_candidates": [],
                },
                selected_event_ids=["local-default:persisted-selection"],
                origin="test",
            )
            spoofed_selection = json.loads(json.dumps(persisted_selection))
            spoofed_selection["selected_candidates"] = [
                {
                    "event_id": "local-default:spoofed-selection",
                    "preview_membership": {
                        "in_review_queue": True,
                        "in_supervised_example_candidates": True,
                        "in_excluded_candidates": False,
                    },
                    "eligible_for_supervised_candidate": True,
                    "policy": {"export_policy_confirmed": True},
                    "evidence_summary": {
                        "traceability": {
                            "test_pass": True,
                            "accepted": True,
                            "review_resolved": False,
                            "comparison_winner": False,
                            "export_policy_confirmed": True,
                        }
                    },
                }
            ]
            dry_run, _dry_run_latest_path, _dry_run_run_path = record_jsonl_training_export_dry_run(
                root=root,
                human_selected_candidates=spoofed_selection,
            )
            candidate = dry_run["candidates"][0]

        self.assertEqual(candidate["event_id"], "local-default:persisted-selection")
        self.assertEqual(dry_run["counts"]["missing_candidate_count"], 1)
        self.assertEqual(dry_run["counts"]["future_jsonl_candidate_if_separately_approved_count"], 0)
        self.assertFalse(candidate["would_write_jsonl_record"])

    def test_human_selected_candidate_list_persists_missing_supplied_learning_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            supplied_preview = {
                "workspace_id": "local-default",
                "paths": {
                    "learning_preview_run_path": str(root / "missing-learning-preview.json"),
                },
                "review_queue": [],
                "supervised_example_candidates": [],
                "excluded_candidates": [],
            }
            selection, _latest_path, _run_path = record_human_selected_candidate_list(
                root=root,
                learning_preview=supplied_preview,
                selected_event_ids=["local-default:missing-selection"],
                origin="test",
            )
            source_learning_preview_path = Path(str(selection["source_learning_preview_path"]))
            source_learning_preview_exists = source_learning_preview_path.exists()

        self.assertTrue(source_learning_preview_exists)
        self.assertNotEqual(source_learning_preview_path.name, "missing-learning-preview.json")
        self.assertFalse(selection["training_export_ready"])

    def test_human_selected_candidate_list_persists_unreadable_supplied_learning_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            unreadable_path = root / "unreadable-learning-preview.json"
            unreadable_path.write_text("{}", encoding="utf-8")
            supplied_preview = {
                "workspace_id": "local-default",
                "paths": {
                    "learning_preview_run_path": str(unreadable_path),
                },
                "review_queue": [],
                "supervised_example_candidates": [],
                "excluded_candidates": [],
            }
            with patch.object(Path, "is_file", side_effect=OSError("permission denied")):
                selection, _latest_path, _run_path = record_human_selected_candidate_list(
                    root=root,
                    learning_preview=supplied_preview,
                    selected_event_ids=["local-default:missing-selection"],
                    origin="test",
                )
            source_learning_preview_path = Path(str(selection["source_learning_preview_path"]))
            source_learning_preview_exists = source_learning_preview_path.exists()

        self.assertTrue(source_learning_preview_exists)
        self.assertNotEqual(source_learning_preview_path.name, "unreadable-learning-preview.json")
        self.assertFalse(selection["training_export_ready"])

    def test_human_selected_candidate_list_persists_invalid_supplied_learning_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            invalid_path = root / "invalid-learning-preview.json"
            invalid_path.write_text("{not-json", encoding="utf-8")
            supplied_preview = {
                "workspace_id": "local-default",
                "paths": {
                    "learning_preview_run_path": str(invalid_path),
                },
                "review_queue": [],
                "supervised_example_candidates": [],
                "excluded_candidates": [],
            }
            selection, _latest_path, _run_path = record_human_selected_candidate_list(
                root=root,
                learning_preview=supplied_preview,
                selected_event_ids=["local-default:missing-selection"],
                origin="test",
            )
            source_learning_preview_path = Path(str(selection["source_learning_preview_path"]))
            source_learning_preview_exists = source_learning_preview_path.exists()

        self.assertTrue(source_learning_preview_exists)
        self.assertNotEqual(source_learning_preview_path.name, "invalid-learning-preview.json")
        self.assertFalse(selection["training_export_ready"])

    def test_export_policy_confirmation_rejects_relation_links(self) -> None:
        with self.assertRaises(ValueError) as raised:
            build_evaluation_signal(
                signal_kind="export_policy_confirmed",
                source_event_id="policy-candidate",
                target_event_id="failed-candidate",
                relation_kind="repairs",
            )

        self.assertIn("cannot define relation links", str(raised.exception))

    def test_export_policy_confirmation_validation_rejects_relation_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            policy_signal = {
                "schema_name": EVALUATION_SIGNAL_SCHEMA_NAME,
                "schema_version": EVALUATION_SIGNAL_SCHEMA_VERSION,
                "signal_id": "local-default:eval:policy-relation",
                "workspace_id": "local-default",
                "signal_kind": "export_policy_confirmed",
                "polarity": "neutral",
                "recorded_at_utc": "2026-04-01T00:00:00+00:00",
                "origin": "test",
                "source": {"source_event_id": "policy-candidate"},
                "relation": {
                    "relation_kind": "follow_up_for",
                    "target_event_id": "failed-candidate",
                },
                "evidence": {},
                "tags": [],
            }

            with self.assertRaises(ValueError) as raised:
                append_evaluation_signal(
                    evaluation_signal_log_path(root=root),
                    policy_signal,
                    workspace_id="local-default",
                )

        self.assertIn("cannot define relation links", str(raised.exception))

    def test_record_export_policy_confirmation_signal_rejects_unknown_source_event_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            with self.assertRaises(ValueError) as raised:
                record_export_policy_confirmation_signal(
                    root=root,
                    source_event_id="local-default:missing-event",
                    rationale="This should not create an orphan policy confirmation.",
                    origin="test",
                )

        self.assertIn("Unknown export-policy confirmation source_event_id", str(raised.exception))

    def test_export_policy_confirmation_alone_does_not_create_supervised_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            event_id = "local-default:capability-matrix:matrix:row-1:chat"
            policy_signal = record_export_policy_confirmation_signal(
                root=root,
                source_event_id=event_id,
                rationale="Policy was confirmed before selection evidence existed.",
                origin="test",
            )

            snapshot, _latest_path, _run_path = record_evaluation_snapshot(root=root)
            curation_preview, _curation_latest_path, _curation_run_path = record_curation_export_preview(
                root=root,
                snapshot=snapshot,
            )
            learning_preview, _learning_latest_path, _learning_run_path = record_learning_dataset_preview(
                root=root,
                snapshot=snapshot,
                curation_preview=curation_preview,
            )
            curation_candidate = [
                item
                for item in curation_preview["candidates"]
                if item["event_id"] == event_id
            ][0]
            excluded = [
                item
                for item in learning_preview["excluded_candidates"]
                if item["event_id"] == event_id
            ][0]

        self.assertEqual(policy_signal["signal_kind"], "export_policy_confirmed")
        self.assertEqual(curation_candidate["state"], "needs_review")
        self.assertIn("export_policy_confirmed", curation_candidate["reasons"])
        self.assertIn("needs_human_selection", curation_candidate["reasons"])
        self.assertEqual(learning_preview["supervised_example_candidates"], [])
        self.assertIn("state_not_ready", excluded["excluded_by"])
        self.assertIn("missing_human_or_comparison_selection", excluded["excluded_by"])
        self.assertEqual(excluded["lifecycle_summary"]["policy_state"], "confirmed_but_not_ready")
        self.assertFalse(learning_preview["training_export_ready"])

    def test_export_policy_confirmation_evidence_is_canonicalized_for_direct_signals(self) -> None:
        event = {
            "event_id": "unsafe-policy-evidence-candidate",
            "event_kind": "agent_task_run",
            "recorded_at_utc": "2026-04-01T00:00:00+00:00",
            "session": {"surface": "agent_lane", "mode": "patch_plan_verify"},
            "outcome": {"status": "ok", "quality_status": "pass", "execution_status": "ok"},
            "content": {
                "prompt": "Keep policy confirmation safe.",
                "output_text": "The policy evidence remains preview-only.",
                "options": {
                    "validation_mode": "agent_lane",
                    "validation_command": "python -m unittest tests.test_policy",
                    "pass_definition": "Policy confirmation does not permit training export.",
                },
            },
            "source_refs": readable_test_source_refs(),
        }
        accepted_signal = build_evaluation_signal(
            signal_kind="acceptance",
            source_event_id="unsafe-policy-evidence-candidate",
            source_event=event,
        )
        unsafe_policy_signal = {
            "schema_name": EVALUATION_SIGNAL_SCHEMA_NAME,
            "schema_version": EVALUATION_SIGNAL_SCHEMA_VERSION,
            "signal_id": "local-default:eval:unsafe-policy",
            "workspace_id": "local-default",
            "signal_kind": "export_policy_confirmed",
            "polarity": "neutral",
            "recorded_at_utc": "2026-04-01T00:02:00+00:00",
            "origin": "test",
            "source": {"source_event_id": "unsafe-policy-evidence-candidate"},
            "relation": {},
            "evidence": {
                "export_mode": "training_jsonl",
                "training_export_ready": True,
                "human_gate_required": False,
                "training_job_allowed": True,
                "raw_log_export_allowed": True,
            },
            "tags": [],
        }

        preview = build_learning_dataset_preview(
            {"workspace_id": "local-default", "paths": {}},
            {
                "candidates": [
                    {
                        "event_id": "unsafe-policy-evidence-candidate",
                        "state": "ready",
                        "label": "Unsafe policy evidence candidate",
                        "reasons": ["accepted", "test_pass", "export_policy_confirmed"],
                        "blocked_by": [],
                        "export_decision": "include_when_approved",
                        "ready_for_policy": True,
                    }
                ]
            },
            events_by_id={"unsafe-policy-evidence-candidate": event},
            explicit_signals=[accepted_signal, unsafe_policy_signal],
            comparisons=[],
        )
        candidate = preview["supervised_example_candidates"][0]
        policy_evidence = candidate["review_queue"]["export_policy_confirmation"]["evidence"]

        self.assertEqual(policy_evidence["export_mode"], "preview_only")
        self.assertFalse(policy_evidence["training_export_ready"])
        self.assertTrue(policy_evidence["human_gate_required"])
        self.assertFalse(policy_evidence["training_job_allowed"])
        self.assertFalse(policy_evidence["raw_log_export_allowed"])
        self.assertFalse(candidate["policy"]["training_job_allowed"])

    def test_learning_preview_updates_stale_curation_metadata_from_policy_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            event_id = "local-default:capability-matrix:matrix:row-1:chat"
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                build_evaluation_signal(
                    signal_kind="acceptance",
                    source_event_id=event_id,
                    rationale="Accepted before policy confirmation.",
                ),
                workspace_id="local-default",
            )
            snapshot_before_policy, _latest_path, _run_path = record_evaluation_snapshot(root=root)
            stale_curation_preview, _curation_latest_path, _curation_run_path = record_curation_export_preview(
                root=root,
                snapshot=snapshot_before_policy,
            )
            record_export_policy_confirmation_signal(
                root=root,
                source_event_id=event_id,
                rationale="Policy confirmed after the curation preview was written.",
                origin="test",
            )
            snapshot_after_policy, _latest_path, _run_path = record_evaluation_snapshot(root=root)
            learning_preview, _learning_latest_path, _learning_run_path = record_learning_dataset_preview(
                root=root,
                snapshot=snapshot_after_policy,
                curation_preview=stale_curation_preview,
            )
            candidate = learning_preview["supervised_example_candidates"][0]
            checklist_by_key = {
                item["key"]: item
                for item in candidate["curation"]["adoption_checklist"]
            }

        self.assertTrue(candidate["review_queue"]["export_policy_confirmation"]["confirmed"])
        self.assertEqual(candidate["review_queue"]["lifecycle_summary"]["policy_state"], "confirmed")
        self.assertIn("export_policy_confirmed", candidate["curation"]["reasons"])
        self.assertEqual(checklist_by_key["export_policy_confirmed"]["status"], "done")
        self.assertEqual(candidate["curation"]["required_next_steps"], ["review_downstream_export_policy"])

    def test_learning_preview_does_not_trust_stale_policy_confirmation_reason(self) -> None:
        event = {
            "event_id": "stale-policy-reason-candidate",
            "event_kind": "agent_task_run",
            "recorded_at_utc": "2026-04-01T00:00:00+00:00",
            "session": {"surface": "agent_lane", "mode": "patch_plan_verify"},
            "outcome": {"status": "ok", "quality_status": "pass", "execution_status": "ok"},
            "content": {
                "prompt": "Keep stale policy reasons traceable.",
                "output_text": "Missing policy signal should not become a silent confirmation.",
                "options": {
                    "validation_mode": "agent_lane",
                    "validation_command": "python -m unittest tests.test_policy_trace",
                    "pass_definition": "Derived test pass remains traceable.",
                },
            },
            "source_refs": readable_test_source_refs(),
        }
        accepted_signal = build_evaluation_signal(
            signal_kind="acceptance",
            source_event_id="stale-policy-reason-candidate",
            source_event=event,
        )

        preview = build_learning_dataset_preview(
            {"workspace_id": "local-default", "paths": {}},
            {
                "candidates": [
                    {
                        "event_id": "stale-policy-reason-candidate",
                        "state": "ready",
                        "label": "Stale policy reason candidate",
                        "reasons": ["accepted", "test_pass", "export_policy_confirmed"],
                        "blocked_by": [],
                        "export_decision": "include_when_approved",
                        "ready_for_policy": True,
                    }
                ]
            },
            events_by_id={"stale-policy-reason-candidate": event},
            explicit_signals=[accepted_signal],
            comparisons=[],
        )
        candidate = preview["supervised_example_candidates"][0]

        self.assertEqual(candidate["review_queue"]["lifecycle_summary"]["policy_state"], "missing_trace")
        self.assertEqual(candidate["review_queue"]["next_action"], "confirm_export_policy")
        self.assertFalse(candidate["review_queue"]["export_policy_confirmation"]["confirmed"])
        self.assertFalse(candidate["policy"]["export_policy_confirmed"])

    def test_learning_review_queue_orders_unconfirmed_ready_before_confirmed_ready(self) -> None:
        def ready_event(event_id: str) -> dict[str, object]:
            return {
                "event_id": event_id,
                "event_kind": "agent_task_run",
                "recorded_at_utc": "2026-04-01T00:00:00+00:00",
                "session": {"surface": "agent_lane", "mode": "patch_plan_verify"},
                "outcome": {"status": "ok", "quality_status": "pass", "execution_status": "ok"},
                "content": {
                    "prompt": f"Review {event_id}.",
                    "output_text": f"{event_id} passed.",
                    "options": {
                        "validation_mode": "agent_lane",
                        "validation_command": "python -m unittest tests.test_policy_order",
                        "pass_definition": "Queue ordering remains stable.",
                    },
                },
                "source_refs": readable_test_source_refs(),
            }

        confirmed_event = ready_event("a-confirmed-policy")
        unconfirmed_event = ready_event("b-unconfirmed-policy")
        accepted_confirmed = build_evaluation_signal(
            signal_kind="acceptance",
            source_event_id="a-confirmed-policy",
            source_event=confirmed_event,
        )
        accepted_unconfirmed = build_evaluation_signal(
            signal_kind="acceptance",
            source_event_id="b-unconfirmed-policy",
            source_event=unconfirmed_event,
        )
        policy_confirmed = build_evaluation_signal(
            signal_kind="export_policy_confirmed",
            source_event_id="a-confirmed-policy",
            source_event=confirmed_event,
        )

        preview = build_learning_dataset_preview(
            {"workspace_id": "local-default", "paths": {}},
            {
                "candidates": [
                    {
                        "event_id": "a-confirmed-policy",
                        "state": "ready",
                        "label": "Confirmed policy",
                        "reasons": ["accepted", "test_pass"],
                        "blocked_by": [],
                        "export_decision": "include_when_approved",
                        "ready_for_policy": True,
                    },
                    {
                        "event_id": "b-unconfirmed-policy",
                        "state": "ready",
                        "label": "Unconfirmed policy",
                        "reasons": ["accepted", "test_pass"],
                        "blocked_by": [],
                        "export_decision": "include_when_approved",
                        "ready_for_policy": True,
                    },
                ]
            },
            events_by_id={
                "a-confirmed-policy": confirmed_event,
                "b-unconfirmed-policy": unconfirmed_event,
            },
            explicit_signals=[accepted_confirmed, accepted_unconfirmed, policy_confirmed],
            comparisons=[],
        )

        self.assertEqual(preview["review_queue"][0]["event_id"], "b-unconfirmed-policy")
        self.assertEqual(preview["review_queue"][0]["queue_priority"]["rank"], 2)
        self.assertEqual(preview["review_queue"][1]["event_id"], "a-confirmed-policy")
        self.assertEqual(preview["review_queue"][1]["queue_priority"]["rank"], 3)

    def test_learning_preview_excludes_test_pass_without_selection_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)

            snapshot, _latest_path, _run_path = record_evaluation_snapshot(root=root)
            curation_preview, _curation_latest_path, _curation_run_path = record_curation_export_preview(
                root=root,
                snapshot=snapshot,
            )
            learning_preview, _learning_latest_path, _learning_run_path = record_learning_dataset_preview(
                root=root,
                snapshot=snapshot,
                curation_preview=curation_preview,
            )
            excluded_by_event = {
                item["event_id"]: set(item["excluded_by"])
                for item in learning_preview["excluded_candidates"]
            }

        self.assertEqual(learning_preview["counts"]["source_candidate_count"], 2)
        self.assertEqual(learning_preview["counts"]["eligible_candidate_count"], 0)
        self.assertEqual(learning_preview["supervised_example_candidates"], [])
        self.assertIn(
            "missing_human_or_comparison_selection",
            excluded_by_event["local-default:capability-matrix:matrix:row-1:chat"],
        )
        self.assertIn(
            "blocking_or_noisy_signal",
            excluded_by_event["local-default:capability-matrix:matrix:row-2:vision"],
        )

    def test_learning_review_queue_marks_missing_source_and_missing_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_artifact_path = root / "artifacts" / "agent_lane" / "run.json"
            source_artifact_path.parent.mkdir(parents=True, exist_ok=True)
            source_artifact_path.write_text("{}", encoding="utf-8")
            event_without_response = {
                "event_id": "event-without-response",
                "event_kind": "agent_task_run",
                "recorded_at_utc": "2026-04-01T00:00:00+00:00",
                "session": {
                    "surface": "agent_lane",
                    "mode": "patch_plan_verify",
                    "title": "Missing response check",
                },
                "outcome": {
                    "status": "ok",
                    "quality_status": "pass",
                    "execution_status": "running",
                },
                "content": {
                    "prompt": "Keep the queue typed.",
                    "output_text": "",
                    "options": {
                        "validation_mode": "agent_lane",
                        "validation_command": "python -m unittest tests.test_queue",
                        "pass_definition": "Queue stays typed.",
                    },
                },
                "source_refs": {
                    "artifact_ref": {
                        "artifact_kind": "agent_run",
                        "artifact_path": str(source_artifact_path),
                    },
                },
            }
            source_candidates = [
                {
                    "event_id": "missing-event",
                    "state": "ready",
                    "label": "Missing source",
                    "reasons": ["accepted", "test_pass"],
                    "blocked_by": [],
                    "export_decision": "include_when_approved",
                    "ready_for_policy": True,
                    "adoption_checklist": [],
                    "required_next_steps": ["confirm_export_policy"],
                },
                {
                    "event_id": "event-without-response",
                    "state": "ready",
                    "label": "Missing supervised text",
                    "reasons": ["accepted", "test_pass"],
                    "blocked_by": [],
                    "export_decision": "include_when_approved",
                    "ready_for_policy": True,
                    "adoption_checklist": [],
                    "required_next_steps": ["confirm_export_policy"],
                },
            ]

            preview = build_learning_dataset_preview(
                {
                    "workspace_id": "local-default",
                    "paths": {
                        "event_log_path": str(root / "artifacts" / "event_logs" / "local-default.jsonl")
                    },
                },
                {"candidates": source_candidates},
                events_by_id={"event-without-response": event_without_response},
                explicit_signals=[],
                comparisons=[],
            )
            queue_by_event = {
                item["event_id"]: item
                for item in preview["review_queue"]
            }

        self.assertEqual(preview["counts"]["eligible_candidate_count"], 0)
        self.assertEqual(queue_by_event["missing-event"]["queue_state"], "missing_source")
        self.assertEqual(queue_by_event["missing-event"]["next_action"], "restore_source_event")
        self.assertEqual(queue_by_event["missing-event"]["blocked_reason"], "missing_source_event")
        self.assertEqual(
            queue_by_event["missing-event"]["lifecycle_summary"]["test_state"],
            "missing_trace",
        )
        self.assertEqual(
            queue_by_event["missing-event"]["lifecycle_summary"]["review_state"],
            "unknown",
        )
        self.assertEqual(
            queue_by_event["missing-event"]["lifecycle_summary"]["selection_state"],
            "missing_trace",
        )
        self.assertEqual(
            queue_by_event["event-without-response"]["queue_state"],
            "missing_supervised_text",
        )
        self.assertEqual(
            queue_by_event["event-without-response"]["next_action"],
            "restore_instruction_or_response_excerpt",
        )
        self.assertEqual(
            preview["counts"]["review_queue_states"],
            {"missing_source": 1, "missing_supervised_text": 1},
        )

    def test_learning_preview_blocks_ready_candidate_with_missing_source_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            missing_artifact_path = root / "artifacts" / "agent_lane" / "missing-run.json"
            event_id = "event-missing-source-artifact"
            event = {
                "event_id": event_id,
                "event_kind": "agent_task_run",
                "recorded_at_utc": "2026-04-01T00:00:00+00:00",
                "session": {
                    "surface": "agent_lane",
                    "mode": "patch_plan_verify",
                    "title": "Missing artifact check",
                },
                "outcome": {
                    "status": "ok",
                    "quality_status": "pass",
                    "execution_status": "ok",
                },
                "content": {
                    "prompt": "Keep source artifacts traceable.",
                    "output_text": "Implemented with passing verification.",
                    "options": {
                        "validation_mode": "agent_lane",
                        "validation_command": "python -m unittest tests.test_source",
                        "pass_definition": "Source artifact remains inspectable.",
                    },
                },
                "source_refs": {
                    "artifact_ref": {
                        "artifact_kind": "agent_run",
                        "artifact_path": str(missing_artifact_path),
                    },
                },
            }
            candidate = {
                "event_id": event_id,
                "state": "ready",
                "label": "Looks ready but source artifact is gone",
                "reasons": ["accepted", "test_pass"],
                "blocked_by": [],
                "export_decision": "include_when_approved",
                "ready_for_policy": True,
                "adoption_checklist": [],
                "required_next_steps": ["confirm_export_policy"],
            }
            accepted_signal = build_evaluation_signal(
                signal_kind="acceptance",
                source_event_id=event_id,
                source_event=event,
                rationale="This would otherwise be a positive learning signal.",
            )

            preview = build_learning_dataset_preview(
                {
                    "workspace_id": "local-default",
                    "paths": {
                        "event_log_path": str(root / "artifacts" / "event_logs" / "local-default.jsonl")
                    },
                },
                {"candidates": [candidate]},
                events_by_id={event_id: event},
                explicit_signals=[accepted_signal],
                comparisons=[],
            )
            queue_item = preview["review_queue"][0]
            excluded = preview["excluded_candidates"][0]

        self.assertEqual(preview["supervised_example_candidates"], [])
        self.assertEqual(preview["counts"]["eligible_candidate_count"], 0)
        self.assertEqual(queue_item["queue_state"], "missing_source")
        self.assertEqual(queue_item["next_action"], "restore_source_artifact")
        self.assertEqual(queue_item["blocked_reason"], "missing_source_artifact")
        self.assertIn("missing_source_artifact", excluded["excluded_by"])
        self.assertEqual(queue_item["lifecycle_summary"]["lifecycle_state"], "blocked")
        self.assertEqual(queue_item["lifecycle_summary"]["source_state"], "missing_source")
        self.assertEqual(queue_item["lifecycle_summary"]["test_state"], "passed")
        self.assertEqual(queue_item["lifecycle_summary"]["selection_state"], "selected")
        self.assertEqual(
            queue_item["event_contract"]["source_artifact"]["source_status"],
            "missing_source",
        )

    def test_learning_preview_infers_contract_root_from_direct_artifact_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_artifact_path = root / "artifacts" / "agent_lane" / "run.json"
            source_artifact_path.parent.mkdir(parents=True, exist_ok=True)
            source_artifact_path.write_text("{}", encoding="utf-8")
            event_id = "event-direct-source-artifact"
            event = {
                "event_id": event_id,
                "event_kind": "agent_task_run",
                "recorded_at_utc": "2026-04-01T00:00:00+00:00",
                "session": {
                    "surface": "agent_lane",
                    "mode": "patch_plan_verify",
                    "title": "Direct source artifact check",
                },
                "outcome": {
                    "status": "ok",
                    "quality_status": "pass",
                    "execution_status": "ok",
                },
                "content": {
                    "prompt": "Keep direct in-memory preview artifacts traceable.",
                    "output_text": "The source artifact remains readable.",
                    "options": {
                        "validation_mode": "agent_lane",
                        "validation_command": "python -m unittest tests.test_source",
                        "pass_definition": "Source artifact remains inspectable.",
                    },
                },
                "source_refs": {
                    "artifact_ref": {
                        "artifact_kind": "agent_run",
                        "artifact_path": str(source_artifact_path),
                    },
                },
            }
            candidate = {
                "event_id": event_id,
                "state": "ready",
                "label": "Direct source artifact candidate",
                "reasons": ["accepted", "test_pass"],
                "blocked_by": [],
                "export_decision": "include_when_approved",
                "ready_for_policy": True,
                "adoption_checklist": [],
                "required_next_steps": ["confirm_export_policy"],
            }
            accepted_signal = build_evaluation_signal(
                signal_kind="acceptance",
                source_event_id=event_id,
                source_event=event,
            )
            test_signal = build_evaluation_signal(
                signal_kind="test_pass",
                source_event_id=event_id,
                source_event=event,
            )

            preview = build_learning_dataset_preview(
                {"workspace_id": "local-default", "paths": {}},
                {"candidates": [candidate]},
                events_by_id={event_id: event},
                explicit_signals=[accepted_signal, test_signal],
                comparisons=[],
            )
            supervised_candidate = preview["supervised_example_candidates"][0]

        self.assertEqual(preview["counts"]["eligible_candidate_count"], 1)
        self.assertEqual(preview["excluded_candidates"], [])
        self.assertEqual(
            supervised_candidate["event_contract"]["source_artifact"]["source_status"],
            "readable",
        )
        self.assertEqual(
            supervised_candidate["event_contract"]["source_artifact"]["artifact_workspace_relative_path"],
            "artifacts/agent_lane/run.json",
        )

    def test_learning_review_queue_uses_stable_ids_without_event_id(self) -> None:
        long_label = "Missing source candidate " + ("x" * 320)
        source_candidates = [
            {
                "state": "ready",
                "label": long_label,
                "reasons": ["accepted", "test_pass"],
                "blocked_by": [],
                "export_decision": "include_when_approved",
                "ready_for_policy": True,
            },
            {
                "state": "ready",
                "label": "Another missing source candidate",
                "reasons": ["accepted", "test_pass"],
                "blocked_by": [],
                "export_decision": "include_when_approved",
                "ready_for_policy": True,
            },
        ]

        preview = build_learning_dataset_preview(
            {"workspace_id": "local-default", "paths": {}},
            {"candidates": source_candidates},
            events_by_id={},
            explicit_signals=[],
            comparisons=[],
        )
        queue_ids = {
            item["queue_item_id"]
            for item in preview["review_queue"]
        }

        self.assertEqual(preview["counts"]["review_queue_count"], 2)
        self.assertEqual(len(queue_ids), 2)
        self.assertTrue(all(item["event_id"] is None for item in preview["review_queue"]))
        self.assertEqual({item["source_index"] for item in preview["review_queue"]}, {0, 1})
        self.assertIn("[preview truncated]", preview["review_queue"][0]["label"])
        self.assertLess(len(preview["review_queue"][0]["label"]), len(long_label))

    def test_learning_review_queue_ids_stay_unique_for_duplicate_missing_event_rows(self) -> None:
        duplicate_candidate = {
            "state": "ready",
            "label": "Duplicate malformed candidate",
            "reasons": ["accepted", "test_pass"],
            "blocked_by": [],
            "export_decision": "include_when_approved",
            "ready_for_policy": True,
        }

        preview = build_learning_dataset_preview(
            {"workspace_id": "local-default", "paths": {}},
            {"candidates": [duplicate_candidate, dict(duplicate_candidate)]},
            events_by_id={},
            explicit_signals=[],
            comparisons=[],
        )
        queue_ids = [item["queue_item_id"] for item in preview["review_queue"]]

        self.assertEqual(len(queue_ids), 2)
        self.assertEqual(len(set(queue_ids)), 2)
        self.assertEqual([item["source_index"] for item in preview["review_queue"]], [0, 1])

    def test_learning_review_queue_label_falls_back_without_label_or_event(self) -> None:
        preview = build_learning_dataset_preview(
            {"workspace_id": "local-default", "paths": {}},
            {
                "candidates": [
                    {
                        "state": "ready",
                        "reasons": ["accepted", "test_pass"],
                        "blocked_by": [],
                        "export_decision": "include_when_approved",
                        "ready_for_policy": True,
                    }
                ]
            },
            events_by_id={},
            explicit_signals=[],
            comparisons=[],
        )
        queue_item = preview["review_queue"][0]

        self.assertIsInstance(queue_item["label"], str)
        self.assertEqual(queue_item["label"], queue_item["queue_item_id"])

    def test_learning_preview_blocks_failed_reason_without_supervised_candidate(self) -> None:
        event = {
            "event_id": "failed-candidate",
            "event_kind": "agent_task_run",
            "recorded_at_utc": "2026-04-01T00:00:00+00:00",
            "session": {"surface": "agent_lane", "mode": "patch_plan_verify"},
            "outcome": {"status": "ok", "quality_status": "pass", "execution_status": "ok"},
            "content": {
                "prompt": "Repair the failed candidate.",
                "output_text": "The candidate still has a failed lifecycle reason.",
                "options": {},
            },
            "source_refs": readable_test_source_refs(),
        }
        preview = build_learning_dataset_preview(
            {"workspace_id": "local-default", "paths": {}},
            {
                "candidates": [
                    {
                        "event_id": "failed-candidate",
                        "state": "ready",
                        "label": "Failed lifecycle candidate",
                        "reasons": ["accepted", "test_pass", "failed"],
                        "blocked_by": [],
                        "export_decision": "include_when_approved",
                        "ready_for_policy": True,
                    }
                ]
            },
            events_by_id={"failed-candidate": event},
            explicit_signals=[],
            comparisons=[],
        )

        self.assertEqual(preview["supervised_example_candidates"], [])
        self.assertEqual(preview["excluded_candidates"][0]["blocked_reason"], "failed")
        self.assertEqual(preview["excluded_candidates"][0]["next_action"], "repair_or_follow_up_failure")
        self.assertEqual(preview["review_queue"][0]["lifecycle_summary"]["test_state"], "failed")

    def test_curation_preview_treats_failed_reason_as_blocking(self) -> None:
        preview = build_curation_export_preview(
            {
                "workspace_id": "local-default",
                "paths": {},
                "curation": {
                    "candidate_count": 1,
                    "candidates": [
                        {
                            "event_id": "failed-candidate",
                            "state": "ready",
                            "label": "Failed lifecycle candidate",
                            "reasons": ["accepted", "test_pass", "failed"],
                        }
                    ],
                },
            }
        )
        candidate = preview["candidates"][0]

        self.assertEqual(candidate["blocked_by"], ["failed"])
        self.assertEqual(candidate["export_decision"], "exclude_until_repaired")
        self.assertFalse(candidate["ready_for_policy"])
        self.assertIn("repair_or_follow_up_failure", candidate["required_next_steps"])
        self.assertEqual(
            preview["adoption_checklist_counts"]["no_blocking_signal"]["blocked"],
            1,
        )

    def test_learning_lifecycle_marks_curation_rejection_as_rejected(self) -> None:
        event = {
            "event_id": "rejected-candidate",
            "event_kind": "agent_task_run",
            "recorded_at_utc": "2026-04-01T00:00:00+00:00",
            "session": {"surface": "agent_lane", "mode": "patch_plan_verify"},
            "outcome": {"status": "ok", "quality_status": "pass", "execution_status": "ok"},
            "content": {
                "prompt": "This candidate was rejected during review.",
                "output_text": "The preview should keep the rejection visible.",
                "options": {},
            },
            "source_refs": readable_test_source_refs(),
        }

        preview = build_learning_dataset_preview(
            {"workspace_id": "local-default", "paths": {}},
            {
                "candidates": [
                    {
                        "event_id": "rejected-candidate",
                        "state": "ready",
                        "label": "Rejected candidate",
                        "reasons": ["rejected", "test_pass"],
                        "blocked_by": [],
                        "export_decision": "include_when_approved",
                        "ready_for_policy": True,
                    }
                ]
            },
            events_by_id={"rejected-candidate": event},
            explicit_signals=[],
            comparisons=[],
        )

        self.assertEqual(preview["supervised_example_candidates"], [])
        self.assertEqual(preview["excluded_candidates"][0]["blocked_reason"], "rejected")
        self.assertEqual(
            preview["review_queue"][0]["lifecycle_summary"]["lifecycle_state"],
            "blocked",
        )
        self.assertEqual(
            preview["review_queue"][0]["lifecycle_summary"]["selection_state"],
            "rejected",
        )

    def test_learning_preview_requires_traceable_test_and_selection_evidence(self) -> None:
        event = {
            "event_id": "stale-ready-candidate",
            "event_kind": "agent_task_run",
            "recorded_at_utc": "2026-04-01T00:00:00+00:00",
            "session": {"surface": "agent_lane", "mode": "patch_plan_verify"},
            "outcome": {"status": "ok", "quality_status": "unknown", "execution_status": "ok"},
            "content": {
                "prompt": "This stale preview says ready.",
                "output_text": "But no traceable test or selection evidence exists.",
                "options": {},
            },
            "source_refs": readable_test_source_refs(),
        }

        preview = build_learning_dataset_preview(
            {"workspace_id": "local-default", "paths": {}},
            {
                "candidates": [
                    {
                        "event_id": "stale-ready-candidate",
                        "state": "ready",
                        "label": "Stale ready candidate",
                        "reasons": ["accepted", "test_pass"],
                        "blocked_by": [],
                        "export_decision": "include_when_approved",
                        "ready_for_policy": True,
                    }
                ]
            },
            events_by_id={"stale-ready-candidate": event},
            explicit_signals=[],
            comparisons=[],
        )

        self.assertEqual(preview["supervised_example_candidates"], [])
        self.assertEqual(preview["review_queue"][0]["queue_state"], "needs_review")
        self.assertEqual(preview["review_queue"][0]["next_action"], "record_test_pass")
        self.assertIn("missing_test_pass_trace", preview["excluded_candidates"][0]["excluded_by"])
        self.assertIn("missing_accepted_trace", preview["excluded_candidates"][0]["excluded_by"])
        self.assertIn("missing_selection_trace", preview["excluded_candidates"][0]["excluded_by"])
        self.assertEqual(
            preview["review_queue"][0]["lifecycle_summary"]["test_state"],
            "missing_trace",
        )
        self.assertEqual(
            preview["review_queue"][0]["lifecycle_summary"]["selection_state"],
            "missing_trace",
        )

    def test_learning_preview_requires_claimed_review_resolution_trace(self) -> None:
        event = {
            "event_id": "claimed-review-resolution",
            "event_kind": "agent_task_run",
            "recorded_at_utc": "2026-04-01T00:00:00+00:00",
            "session": {"surface": "agent_lane", "mode": "patch_plan_verify"},
            "outcome": {"status": "ok", "quality_status": "pass", "execution_status": "ok"},
            "content": {
                "prompt": "Keep claimed review resolution traceable.",
                "output_text": "Acceptance alone should not stand in for review resolution.",
                "options": {
                    "validation_mode": "agent_lane",
                    "validation_command": "python -m unittest tests.test_review_trace",
                    "pass_definition": "Derived test pass remains traceable.",
                },
            },
            "source_refs": readable_test_source_refs(),
        }
        accepted_signal = build_evaluation_signal(
            signal_kind="acceptance",
            source_event_id="claimed-review-resolution",
            source_event=event,
            recorded_at_utc="2026-04-01T00:01:00+00:00",
            signal_id="local-default:eval:claimed-review:accepted",
        )

        preview = build_learning_dataset_preview(
            {"workspace_id": "local-default", "paths": {}},
            {
                "candidates": [
                    {
                        "event_id": "claimed-review-resolution",
                        "state": "ready",
                        "label": "Claimed review resolution",
                        "reasons": ["review_resolved", "test_pass"],
                        "blocked_by": [],
                        "export_decision": "include_when_approved",
                        "ready_for_policy": True,
                    }
                ]
            },
            events_by_id={"claimed-review-resolution": event},
            explicit_signals=[accepted_signal],
            comparisons=[],
        )
        excluded = preview["excluded_candidates"][0]
        queue_item = preview["review_queue"][0]

        self.assertEqual(preview["supervised_example_candidates"], [])
        self.assertEqual(queue_item["queue_state"], "needs_review")
        self.assertIn("missing_review_resolved_trace", excluded["excluded_by"])
        self.assertEqual(queue_item["next_action"], "record_acceptance_or_review_resolution")
        self.assertEqual(queue_item["lifecycle_summary"]["review_state"], "missing_trace")
        self.assertEqual(queue_item["lifecycle_summary"]["selection_state"], "selected")

    def test_learning_preview_requires_claimed_comparison_winner_trace(self) -> None:
        event = {
            "event_id": "claimed-comparison-winner",
            "event_kind": "agent_task_run",
            "recorded_at_utc": "2026-04-01T00:00:00+00:00",
            "session": {"surface": "agent_lane", "mode": "patch_plan_verify"},
            "outcome": {"status": "ok", "quality_status": "pass", "execution_status": "ok"},
            "content": {
                "prompt": "Keep claimed comparison winners traceable.",
                "output_text": "Acceptance alone should not stand in for a comparison winner.",
                "options": {
                    "validation_mode": "agent_lane",
                    "validation_command": "python -m unittest tests.test_comparison_trace",
                    "pass_definition": "Derived test pass remains traceable.",
                },
            },
            "source_refs": readable_test_source_refs(),
        }
        accepted_signal = build_evaluation_signal(
            signal_kind="acceptance",
            source_event_id="claimed-comparison-winner",
            source_event=event,
            recorded_at_utc="2026-04-01T00:01:00+00:00",
            signal_id="local-default:eval:claimed-comparison:accepted",
        )

        preview = build_learning_dataset_preview(
            {"workspace_id": "local-default", "paths": {}},
            {
                "candidates": [
                    {
                        "event_id": "claimed-comparison-winner",
                        "state": "ready",
                        "label": "Claimed comparison winner",
                        "reasons": ["comparison_winner", "test_pass"],
                        "blocked_by": [],
                        "export_decision": "include_when_approved",
                        "ready_for_policy": True,
                    }
                ]
            },
            events_by_id={"claimed-comparison-winner": event},
            explicit_signals=[accepted_signal],
            comparisons=[],
        )
        excluded = preview["excluded_candidates"][0]
        queue_item = preview["review_queue"][0]

        self.assertEqual(preview["supervised_example_candidates"], [])
        self.assertEqual(queue_item["queue_state"], "needs_review")
        self.assertIn("missing_comparison_winner_trace", excluded["excluded_by"])
        self.assertEqual(queue_item["next_action"], "record_comparison_winner_trace")
        self.assertEqual(queue_item["lifecycle_summary"]["selection_state"], "missing_trace")

    def test_learning_preview_keeps_traceable_comparison_winner_candidate(self) -> None:
        event = {
            "event_id": "traceable-comparison-winner",
            "event_kind": "agent_task_run",
            "recorded_at_utc": "2026-04-01T00:00:00+00:00",
            "session": {"surface": "agent_lane", "mode": "patch_plan_verify"},
            "outcome": {"status": "ok", "quality_status": "pass", "execution_status": "ok"},
            "content": {
                "prompt": "Keep traceable comparison winners eligible.",
                "output_text": "The winner trace and test trace are both present.",
                "options": {
                    "validation_mode": "agent_lane",
                    "validation_command": "python -m unittest tests.test_comparison_trace",
                    "pass_definition": "Comparison winner trace is present.",
                },
            },
            "source_refs": readable_test_source_refs(),
        }
        test_signal = build_evaluation_signal(
            signal_kind="test_pass",
            source_event_id="traceable-comparison-winner",
            source_event=event,
            recorded_at_utc="2026-04-01T00:01:00+00:00",
            signal_id="local-default:eval:traceable-comparison:test-pass",
        )
        comparison = build_evaluation_comparison(
            candidate_event_ids=["traceable-comparison-winner", "traceable-comparison-loser"],
            winner_event_id="traceable-comparison-winner",
            recorded_at_utc="2026-04-01T00:02:00+00:00",
            comparison_id="local-default:compare:traceable-comparison",
            task_label="choose traceable learning candidate",
            criteria=["passing check", "winner trace"],
            rationale="Winner trace is present and should keep the candidate eligible.",
        )

        preview = build_learning_dataset_preview(
            {"workspace_id": "local-default", "paths": {}},
            {
                "candidates": [
                    {
                        "event_id": "traceable-comparison-winner",
                        "state": "ready",
                        "label": "Traceable comparison winner",
                        "reasons": ["comparison_winner", "test_pass"],
                        "blocked_by": [],
                        "export_decision": "include_when_approved",
                        "ready_for_policy": True,
                    }
                ]
            },
            events_by_id={"traceable-comparison-winner": event},
            explicit_signals=[test_signal],
            comparisons=[comparison],
        )
        candidate = preview["supervised_example_candidates"][0]

        self.assertEqual(preview["excluded_candidates"], [])
        self.assertEqual(candidate["event_id"], "traceable-comparison-winner")
        self.assertEqual(candidate["review_queue"]["queue_state"], "ready")
        self.assertEqual(candidate["review_queue"]["lifecycle_summary"]["selection_state"], "selected")
        self.assertEqual(candidate["review_queue"]["lifecycle_summary"]["comparison_state"], "winner")
        self.assertEqual(candidate["evidence"]["comparisons"][0]["role"], "winner")
        self.assertEqual(candidate["evidence"]["comparisons"][0]["comparison_role"], "winner")

    def test_learning_preview_enriches_legacy_comparison_backend_metadata(self) -> None:
        source_refs = readable_test_source_refs()
        source_refs["backend_ref"] = {
            "backend_id": "legacy-backend",
            "adapter_kind": "mock",
            "model_id": "legacy/model-v1",
            "compatibility_status": "compatible",
        }
        event = {
            "event_id": "legacy-comparison-winner",
            "event_kind": "agent_task_run",
            "recorded_at_utc": "2026-04-01T00:00:00+00:00",
            "session": {
                "surface": "agent_lane",
                "mode": "patch_plan_verify",
                "selected_model_id": "legacy/model-v1",
            },
            "outcome": {"status": "ok", "quality_status": "pass", "execution_status": "ok"},
            "content": {
                "prompt": "Keep legacy comparison logs explainable.",
                "output_text": "The backend metadata is recovered from the source event.",
                "options": {
                    "validation_mode": "agent_lane",
                    "validation_command": "python -m unittest tests.test_legacy_comparison",
                    "pass_definition": "Legacy comparison metadata is enriched.",
                },
            },
            "source_refs": source_refs,
        }
        test_signal = build_evaluation_signal(
            signal_kind="test_pass",
            source_event_id="legacy-comparison-winner",
            source_event=event,
            recorded_at_utc="2026-04-01T00:01:00+00:00",
            signal_id="local-default:eval:legacy-comparison:test-pass",
        )
        legacy_comparison = build_evaluation_comparison(
            candidate_event_ids=["legacy-comparison-winner", "legacy-comparison-loser"],
            winner_event_id="legacy-comparison-winner",
            recorded_at_utc="2026-04-01T00:02:00+00:00",
            comparison_id="local-default:compare:legacy-comparison",
            task_label="legacy comparison log",
            criteria=["winner trace"],
            rationale="Simulate a pre-S5 comparison record without backend metadata.",
        )

        preview = build_learning_dataset_preview(
            {"workspace_id": "local-default", "paths": {}},
            {
                "candidates": [
                    {
                        "event_id": "legacy-comparison-winner",
                        "state": "ready",
                        "label": "Legacy comparison winner",
                        "reasons": ["comparison_winner", "test_pass"],
                        "blocked_by": [],
                        "export_decision": "include_when_approved",
                        "ready_for_policy": True,
                    }
                ]
            },
            events_by_id={"legacy-comparison-winner": event},
            explicit_signals=[test_signal],
            comparisons=[legacy_comparison],
        )
        comparison_trace = preview["supervised_example_candidates"][0]["evidence"]["comparisons"][0]

        self.assertEqual(comparison_trace["backend_metadata"]["backend_id"], "legacy-backend")
        self.assertEqual(comparison_trace["backend_metadata"]["model_id"], "legacy/model-v1")
        self.assertEqual(comparison_trace["backend_comparison"]["backend_count"], 1)

    def test_learning_preview_reuses_backend_metadata_artifact_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_path = root / "artifacts" / "agent_runs" / "cached-backend.json"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(
                json.dumps(
                    {
                        "backend": {
                            "backend_id": "cached-backend",
                            "display_name": "Cached backend",
                            "adapter_kind": "mock",
                            "model_id": "cached/model-v1",
                            "capabilities": {"chat": True},
                            "limits": {"context_tokens": 8192},
                            "metadata": {"provider": "local"},
                        },
                        "compatibility": {
                            "status": "compatible",
                            "schema_name": "backend-compatibility",
                            "schema_version": 1,
                            "backend_id": "cached-backend",
                            "adapter_kind": "mock",
                            "model_id": "cached/model-v1",
                            "workflow_kind": "learning_preview",
                            "required_capabilities": ["chat"],
                            "optional_capabilities": [],
                            "missing_capabilities": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            event = {
                "event_id": "cached-backend-winner",
                "event_kind": "agent_task_run",
                "recorded_at_utc": "2026-04-01T00:00:00+00:00",
                "session": {"surface": "agent_lane", "mode": "patch_plan_verify"},
                "outcome": {"status": "ok", "quality_status": "pass", "execution_status": "ok"},
                "content": {
                    "prompt": "Keep repeated backend metadata reads cached.",
                    "output_text": "The same agent_run artifact backs comparison and review evidence.",
                    "options": {
                        "validation_mode": "agent_lane",
                        "validation_command": "python -m unittest tests.test_backend_metadata_cache",
                        "pass_definition": "Backend metadata artifact is reused in one preview build.",
                    },
                },
                "source_refs": {
                    "artifact_ref": {
                        "artifact_kind": "agent_run",
                        "artifact_path": str(artifact_path),
                    }
                },
            }
            test_signal = build_evaluation_signal(
                signal_kind="test_pass",
                source_event_id="cached-backend-winner",
                source_event=event,
                recorded_at_utc="2026-04-01T00:01:00+00:00",
                signal_id="local-default:eval:cached-backend:test-pass",
            )
            comparison = build_evaluation_comparison(
                candidate_event_ids=["cached-backend-winner", "cached-backend-loser"],
                winner_event_id="cached-backend-winner",
                recorded_at_utc="2026-04-01T00:02:00+00:00",
                comparison_id="local-default:compare:cached-backend",
                task_label="cached backend metadata",
                criteria=["winner trace", "backend metadata"],
                rationale="The winner metadata should be read once per preview build.",
            )

            with patch(
                "evaluation_loop._read_json_object",
                wraps=evaluation_loop_module._read_json_object,
            ) as read_json_mock:
                preview = build_learning_dataset_preview(
                    {
                        "workspace_id": "local-default",
                        "paths": {
                            "event_log_path": str(
                                root / "artifacts" / "event_logs" / "events.jsonl"
                            )
                        },
                    },
                    {
                        "candidates": [
                            {
                                "event_id": "cached-backend-winner",
                                "state": "ready",
                                "label": "Cached backend winner",
                                "reasons": ["comparison_winner", "test_pass"],
                                "blocked_by": [],
                                "export_decision": "include_when_approved",
                                "ready_for_policy": True,
                            }
                        ]
                    },
                    events_by_id={"cached-backend-winner": event},
                    explicit_signals=[test_signal],
                    comparisons=[comparison],
                )
                read_count = read_json_mock.call_count

        candidate = preview["supervised_example_candidates"][0]
        comparison_trace = candidate["evidence"]["comparisons"][0]

        self.assertEqual(read_count, 1)
        self.assertEqual(candidate["backend_metadata"]["backend_id"], "cached-backend")
        self.assertEqual(candidate["review_queue"]["backend_metadata"]["model_id"], "cached/model-v1")
        self.assertEqual(comparison_trace["backend_metadata"]["adapter_kind"], "mock")
        self.assertEqual(comparison_trace["backend_metadata"]["compatibility_status"], "compatible")

    def test_backend_metadata_uses_complete_event_payload_without_artifact_read(self) -> None:
        event = {
            "event_id": "complete-backend-metadata",
            "event_kind": "agent_task_run",
            "session": {"selected_model_id": "event/model-v1"},
            "outcome": {"status": "ok", "model_id": "event/model-v1"},
            "content": {"options": {}},
            "source_refs": {
                "artifact_ref": {
                    "artifact_kind": "agent_run",
                    "artifact_path": "/tmp/should-not-be-read.json",
                },
                "backend_ref": {
                    "backend_id": "event-backend",
                    "display_name": "Event backend",
                    "adapter_kind": "mock",
                    "model_id": "event/model-v1",
                    "compatibility_status": "compatible",
                    "compatibility": {
                        "status": "compatible",
                        "schema_name": "backend-compatibility",
                        "schema_version": 1,
                        "backend_id": "event-backend",
                        "adapter_kind": "mock",
                        "model_id": "event/model-v1",
                        "workflow_kind": "learning_preview",
                        "required_capabilities": ["chat"],
                        "optional_capabilities": [],
                        "missing_capabilities": [],
                    },
                    "capabilities": {"chat": True},
                    "limits": {"context_tokens": 4096},
                    "metadata": {"source": "event"},
                },
            },
        }

        with patch("evaluation_loop._read_json_object") as read_json_mock:
            metadata = evaluation_loop_module._backend_metadata_from_event(event)

        self.assertFalse(read_json_mock.called)
        self.assertEqual(metadata["backend_id"], "event-backend")
        self.assertEqual(metadata["display_name"], "Event backend")
        self.assertEqual(metadata["limits"]["context_tokens"], 4096)
        self.assertEqual(metadata["metadata"]["source"], "event")
        self.assertEqual(metadata["compatibility"]["workflow_kind"], "learning_preview")

    def test_enrich_comparison_preserves_existing_backend_metadata_without_event(self) -> None:
        comparison = {
            "comparison_id": "local-default:compare:archival-backend",
            "recorded_at_utc": "2026-04-01T00:02:00+00:00",
            "outcome": "winner_selected",
            "winner_event_id": "archival-winner",
            "candidates": [
                {
                    "event_id": "archival-winner",
                    "backend_metadata": {
                        "backend_id": "archival-backend",
                        "display_name": "Archival backend",
                        "adapter_kind": "mock",
                        "model_id": "archival/model-v1",
                        "compatibility_status": "compatible",
                        "capabilities": {"chat": True},
                    },
                }
            ],
        }

        enriched = evaluation_loop_module._enrich_comparison(comparison, {})
        metadata = enriched["candidates"][0]["backend_metadata"]

        self.assertEqual(metadata["backend_id"], "archival-backend")
        self.assertEqual(metadata["display_name"], "Archival backend")
        self.assertEqual(metadata["adapter_kind"], "mock")
        self.assertEqual(metadata["model_id"], "archival/model-v1")
        self.assertEqual(metadata["compatibility_status"], "compatible")
        self.assertEqual(metadata["capabilities"]["chat"], True)

    def test_learning_preview_blocks_stale_ready_candidate_with_negative_trace(self) -> None:
        event = {
            "event_id": "stale-negative-candidate",
            "event_kind": "agent_task_run",
            "recorded_at_utc": "2026-04-01T00:00:00+00:00",
            "session": {"surface": "agent_lane", "mode": "patch_plan_verify"},
            "outcome": {"status": "ok", "quality_status": "pass", "execution_status": "ok"},
            "content": {
                "prompt": "This stale preview still says ready.",
                "output_text": "A later review reopened the concern.",
                "options": {
                    "validation_mode": "agent_lane",
                    "validation_command": "python -m unittest tests.test_stale",
                    "pass_definition": "Derived test pass remains traceable.",
                },
            },
            "source_refs": readable_test_source_refs(),
        }
        resolved_signal = build_evaluation_signal(
            signal_kind="review_resolved",
            source_event_id="stale-negative-candidate",
            source_event=event,
            recorded_at_utc="2026-04-01T00:00:00+00:00",
            signal_id="local-default:eval:resolved",
        )
        unresolved_signal = build_evaluation_signal(
            signal_kind="review_unresolved",
            source_event_id="stale-negative-candidate",
            source_event=event,
            recorded_at_utc="2026-04-01T00:01:00+00:00",
            signal_id="local-default:eval:unresolved",
        )

        preview = build_learning_dataset_preview(
            {"workspace_id": "local-default", "paths": {}},
            {
                "candidates": [
                    {
                        "event_id": "stale-negative-candidate",
                        "state": "ready",
                        "label": "Stale negative candidate",
                        "reasons": ["review_resolved", "test_pass"],
                        "blocked_by": [],
                        "export_decision": "include_when_approved",
                        "ready_for_policy": True,
                    }
                ]
            },
            events_by_id={"stale-negative-candidate": event},
            explicit_signals=[resolved_signal, unresolved_signal],
            comparisons=[],
        )

        self.assertEqual(preview["supervised_example_candidates"], [])
        self.assertEqual(preview["excluded_candidates"][0]["blocked_reason"], "review_unresolved")
        self.assertEqual(preview["excluded_candidates"][0]["next_action"], "resolve_review_before_export")
        self.assertIn("review_unresolved_trace", preview["excluded_candidates"][0]["excluded_by"])
        self.assertEqual(
            preview["review_queue"][0]["lifecycle_summary"]["review_state"],
            "unresolved",
        )

    def test_learning_preview_uses_latest_test_trace_for_blocking(self) -> None:
        event = {
            "event_id": "repaired-candidate",
            "event_kind": "agent_task_run",
            "recorded_at_utc": "2026-04-01T00:00:00+00:00",
            "session": {"surface": "agent_lane", "mode": "patch_plan_verify"},
            "outcome": {"status": "ok", "quality_status": "pass", "execution_status": "ok"},
            "content": {
                "prompt": "Repair the candidate after a failed test.",
                "output_text": "The latest test pass should clear the old failure.",
                "options": {
                    "validation_mode": "agent_lane",
                    "validation_command": "python -m unittest tests.test_repair",
                    "pass_definition": "Latest validation passes.",
                },
            },
            "source_refs": readable_test_source_refs(),
        }
        failed_signal = build_evaluation_signal(
            signal_kind="test_fail",
            source_event_id="repaired-candidate",
            source_event=event,
            recorded_at_utc="2026-04-01T00:00:00+00:00",
            signal_id="local-default:eval:test-fail",
        )
        passed_signal = build_evaluation_signal(
            signal_kind="test_pass",
            source_event_id="repaired-candidate",
            source_event=event,
            recorded_at_utc="2026-04-01T00:02:00+00:00",
            signal_id="local-default:eval:test-pass",
        )
        accepted_signal = build_evaluation_signal(
            signal_kind="acceptance",
            source_event_id="repaired-candidate",
            source_event=event,
            recorded_at_utc="2026-04-01T00:03:00+00:00",
            signal_id="local-default:eval:accepted",
        )

        preview = build_learning_dataset_preview(
            {"workspace_id": "local-default", "paths": {}},
            {
                "candidates": [
                    {
                        "event_id": "repaired-candidate",
                        "state": "ready",
                        "label": "Repaired candidate",
                        "reasons": ["accepted", "test_pass"],
                        "blocked_by": [],
                        "export_decision": "include_when_approved",
                        "ready_for_policy": True,
                    }
                ]
            },
            events_by_id={"repaired-candidate": event},
            explicit_signals=[failed_signal, passed_signal, accepted_signal],
            comparisons=[],
        )

        self.assertEqual(preview["counts"]["eligible_candidate_count"], 1)
        self.assertEqual(preview["excluded_candidates"], [])
        self.assertEqual(preview["review_queue"][0]["queue_state"], "ready")
        self.assertEqual(
            preview["review_queue"][0]["lifecycle_summary"]["test_state"],
            "passed",
        )

    def test_learning_preview_blocks_latest_test_fail_trace(self) -> None:
        event = {
            "event_id": "regressed-candidate",
            "event_kind": "agent_task_run",
            "recorded_at_utc": "2026-04-01T00:00:00+00:00",
            "session": {"surface": "agent_lane", "mode": "patch_plan_verify"},
            "outcome": {"status": "ok", "quality_status": "pass", "execution_status": "ok"},
            "content": {
                "prompt": "This candidate regressed after an earlier pass.",
                "output_text": "The latest test failure should block export.",
                "options": {
                    "validation_mode": "agent_lane",
                    "validation_command": "python -m unittest tests.test_regression",
                    "pass_definition": "Latest validation must pass.",
                },
            },
            "source_refs": readable_test_source_refs(),
        }
        passed_signal = build_evaluation_signal(
            signal_kind="test_pass",
            source_event_id="regressed-candidate",
            source_event=event,
            recorded_at_utc="2026-04-01T00:00:00+00:00",
            signal_id="local-default:eval:test-pass",
        )
        failed_signal = build_evaluation_signal(
            signal_kind="test_fail",
            source_event_id="regressed-candidate",
            source_event=event,
            recorded_at_utc="2026-04-01T00:02:00+00:00",
            signal_id="local-default:eval:test-fail",
        )
        accepted_signal = build_evaluation_signal(
            signal_kind="acceptance",
            source_event_id="regressed-candidate",
            source_event=event,
            recorded_at_utc="2026-04-01T00:03:00+00:00",
            signal_id="local-default:eval:accepted",
        )

        preview = build_learning_dataset_preview(
            {"workspace_id": "local-default", "paths": {}},
            {
                "candidates": [
                    {
                        "event_id": "regressed-candidate",
                        "state": "ready",
                        "label": "Regressed candidate",
                        "reasons": ["accepted", "test_pass"],
                        "blocked_by": [],
                        "export_decision": "include_when_approved",
                        "ready_for_policy": True,
                    }
                ]
            },
            events_by_id={"regressed-candidate": event},
            explicit_signals=[passed_signal, failed_signal, accepted_signal],
            comparisons=[],
        )

        self.assertEqual(preview["supervised_example_candidates"], [])
        self.assertIn("test_fail_trace", preview["excluded_candidates"][0]["excluded_by"])
        self.assertEqual(preview["excluded_candidates"][0]["blocked_reason"], "test_fail")
        self.assertEqual(
            preview["review_queue"][0]["lifecycle_summary"]["test_state"],
            "failed",
        )

    def test_learning_preview_persists_supplied_in_memory_curation_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)

            snapshot, _latest_path, _run_path = record_evaluation_snapshot(root=root)
            curation_preview = build_curation_export_preview(snapshot)
            self.assertNotIn("paths", curation_preview)
            learning_preview, _learning_latest_path, _learning_run_path = record_learning_dataset_preview(
                root=root,
                snapshot=snapshot,
                curation_preview=curation_preview,
            )
            curation_preview_path = Path(str(learning_preview["source_curation_preview_path"]))
            persisted_preview = json.loads(curation_preview_path.read_text(encoding="utf-8"))
            curation_preview_exists = curation_preview_path.exists()

        self.assertTrue(curation_preview_exists)
        self.assertEqual(
            learning_preview["paths"]["source_curation_preview_path"],
            str(curation_preview_path),
        )
        self.assertEqual(
            persisted_preview["paths"]["curation_preview_run_path"],
            str(curation_preview_path),
        )

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

    def test_record_review_resolution_signal_rejects_unknown_source_event_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            with self.assertRaises(ValueError) as raised:
                record_review_resolution_signal(
                    root=root,
                    source_event_id="local-default:missing-event",
                    resolved=True,
                    resolution_summary="This should not create an orphan signal.",
                    origin="test",
                )

        self.assertIn("Unknown review-resolution source_event_id", str(raised.exception))

    def test_recent_comparisons_are_sorted_before_snapshot_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            passing_event_id = "local-default:capability-matrix:matrix:row-1:chat"
            failing_event_id = "local-default:capability-matrix:matrix:row-2:vision"
            for index in range(13):
                comparison = build_evaluation_comparison(
                    candidate_event_ids=[passing_event_id, failing_event_id],
                    winner_event_id=passing_event_id,
                    comparison_id=f"local-default:compare:test-{index:02d}",
                    recorded_at_utc=f"2026-04-01T00:{index:02d}:00Z",
                    task_label=f"comparison {index:02d}",
                )
                append_evaluation_comparison(
                    evaluation_comparison_log_path(root=root),
                    comparison,
                    workspace_id="local-default",
                )

            snapshot, _latest_path, _run_path = record_evaluation_snapshot(root=root)

        comparison_ids = [item["comparison_id"] for item in snapshot["comparisons"]]
        self.assertEqual(snapshot["counts"]["comparisons"], 13)
        self.assertEqual(len(comparison_ids), 12)
        self.assertEqual(comparison_ids[0], "local-default:compare:test-12")
        self.assertEqual(comparison_ids[-1], "local-default:compare:test-01")
        self.assertNotIn("local-default:compare:test-00", comparison_ids)

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

    def test_snapshot_consistency_report_explains_newer_negative_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_artifact_path = root / "artifacts" / "agent_lane" / "run.json"
            source_artifact_path.parent.mkdir(parents=True, exist_ok=True)
            source_artifact_path.write_text("{}", encoding="utf-8")
            event_id = "stale-positive-event"
            event = {
                "schema_name": "software-satellite-event",
                "schema_version": 1,
                "event_id": event_id,
                "event_kind": "agent_task_run",
                "recorded_at_utc": "2026-04-01T00:00:00+00:00",
                "workspace": {"workspace_id": "local-default"},
                "session": {"surface": "agent_lane", "mode": "patch_plan_verify"},
                "outcome": {"status": "ok", "quality_status": "pass", "execution_status": "ok"},
                "content": {
                    "prompt": "Keep stale positives out of promotion.",
                    "output_text": "This candidate later regressed.",
                    "options": {
                        "validation_mode": "agent_lane",
                        "validation_command": "python -m unittest tests.test_consistency",
                        "pass_definition": "Latest negative traces win.",
                    },
                },
                "source_refs": {
                    "artifact_ref": {
                        "artifact_kind": "agent_run",
                        "artifact_path": str(source_artifact_path),
                    },
                },
                "tags": [],
            }
            event_log_path = root / "artifacts" / "event_logs" / "local-default.jsonl"
            write_event_log(event_log_path, [event], workspace_id="local-default")
            for signal in (
                build_evaluation_signal(
                    signal_kind="acceptance",
                    source_event_id=event_id,
                    source_event=event,
                    recorded_at_utc="2026-04-01T00:01:00+00:00",
                    signal_id="local-default:eval:acceptance",
                ),
                build_evaluation_signal(
                    signal_kind="review_resolved",
                    source_event_id=event_id,
                    source_event=event,
                    recorded_at_utc="2026-04-01T00:02:00+00:00",
                    signal_id="local-default:eval:review-resolved",
                ),
                build_evaluation_signal(
                    signal_kind="test_fail",
                    source_event_id=event_id,
                    source_event=event,
                    recorded_at_utc="2026-04-01T00:03:00+00:00",
                    signal_id="local-default:eval:test-fail",
                ),
                build_evaluation_signal(
                    signal_kind="rejection",
                    source_event_id=event_id,
                    source_event=event,
                    recorded_at_utc="2026-04-01T00:04:00+00:00",
                    signal_id="local-default:eval:rejection",
                ),
                build_evaluation_signal(
                    signal_kind="review_unresolved",
                    source_event_id=event_id,
                    source_event=event,
                    recorded_at_utc="2026-04-01T00:05:00+00:00",
                    signal_id="local-default:eval:review-unresolved",
                ),
            ):
                append_evaluation_signal(
                    evaluation_signal_log_path(root=root),
                    signal,
                    workspace_id="local-default",
                )

            snapshot, _latest_path, _run_path = record_evaluation_snapshot(
                root=root,
                index_summary={
                    "event_log_path": str(event_log_path),
                    "index_path": str(root / "artifacts" / "indexes" / "test.sqlite3"),
                },
            )
            consistency_item = snapshot["evaluation_consistency"]["events"][0]
            report = format_evaluation_snapshot_report(snapshot)

        self.assertEqual(snapshot["counts"]["curation_blocked"], 1)
        self.assertEqual(snapshot["counts"]["stale_positive_signal"], 3)
        self.assertEqual(snapshot["counts"]["negative_signal_wins"], 3)
        self.assertIn("test_pass", consistency_item["stale_positive_signals"])
        self.assertIn("accepted", consistency_item["stale_positive_signals"])
        self.assertIn("review_resolved", consistency_item["stale_positive_signals"])
        self.assertIn("test_pass", consistency_item["stale_positive_reasons"])
        self.assertIn("accepted", consistency_item["stale_positive_reasons"])
        self.assertEqual(consistency_item["latest_signals"]["test"]["signal_kind"], "test_fail")
        self.assertEqual(consistency_item["latest_signals"]["selection"]["signal_kind"], "rejection")
        self.assertEqual(consistency_item["latest_signals"]["review"]["signal_kind"], "review_unresolved")
        self.assertIn("Consistency stale positives: 3", report)
        self.assertIn("Evaluation consistency:", report)

    def test_snapshot_consistency_report_flags_untraceable_comparison_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            event_log_path = root / "artifacts" / "event_logs" / "local-default.jsonl"
            write_event_log(event_log_path, [], workspace_id="local-default")
            comparison = build_evaluation_comparison(
                candidate_event_ids=["missing-winner", "missing-loser"],
                winner_event_id="missing-winner",
                recorded_at_utc="2026-04-01T00:00:00+00:00",
                comparison_id="local-default:compare:missing-winner",
            )
            append_evaluation_comparison(
                evaluation_comparison_log_path(root=root),
                comparison,
                workspace_id="local-default",
            )

            snapshot, _latest_path, _run_path = record_evaluation_snapshot(
                root=root,
                index_summary={
                    "event_log_path": str(event_log_path),
                    "index_path": str(root / "artifacts" / "indexes" / "test.sqlite3"),
                },
            )
            issue_event = snapshot["evaluation_consistency"]["issue_events"][0]

        self.assertEqual(snapshot["counts"]["comparison_winner_missing_trace"], 1)
        self.assertEqual(snapshot["counts"]["missing_trace"], 1)
        self.assertEqual(issue_event["event_id"], "missing-winner")
        self.assertIn("comparison_winner_missing_source_event", issue_event["missing_traces"])
        self.assertFalse(issue_event["source_trace"]["source_event_present"])

    def test_consistency_report_rolls_up_missing_comparison_winner_reason_trace(self) -> None:
        event = {
            "event_id": "stale-comparison-winner-candidate",
            "event_kind": "agent_task_run",
            "recorded_at_utc": "2026-04-01T00:00:00+00:00",
            "session": {"surface": "agent_lane", "mode": "patch_plan_verify"},
            "outcome": {"status": "ok", "quality_status": "pass", "execution_status": "ok"},
            "content": {
                "prompt": "Explain missing comparison winner traces in rollups.",
                "output_text": "The curation reason is stale without a winner_selected trace.",
            },
            "source_refs": readable_test_source_refs(),
        }

        report = build_evaluation_consistency_report(
            signal_summaries=[],
            comparison_summaries=[],
            curation_candidates=[
                {
                    "event_id": "stale-comparison-winner-candidate",
                    "state": "ready",
                    "reasons": ["comparison_winner"],
                }
            ],
            events_by_id={"stale-comparison-winner-candidate": event},
        )
        issue_event = report["issue_events"][0]

        self.assertEqual(report["counts"]["missing_trace"], 1)
        self.assertEqual(report["counts"]["comparison_winner_missing_trace"], 1)
        self.assertIn("missing_comparison_winner_trace", issue_event["missing_traces"])

    def test_consistency_report_does_not_treat_follow_up_winner_id_as_selected_winner(self) -> None:
        report = build_evaluation_consistency_report(
            signal_summaries=[],
            comparison_summaries=[
                {
                    "comparison_id": "local-default:compare:follow-up",
                    "recorded_at_utc": "2026-04-01T00:00:00+00:00",
                    "outcome": "needs_follow_up",
                    "winner_event_id": "follow-up-candidate",
                    "candidate_event_ids": ["follow-up-candidate", "other-candidate"],
                }
            ],
            curation_candidates=[],
            events_by_id={},
        )

        self.assertEqual(report["counts"]["missing_trace"], 0)
        self.assertEqual(report["counts"]["comparison_winner_missing_trace"], 0)
        self.assertEqual(report["issue_events"], [])

    def test_snapshot_consistency_report_preserves_source_artifact_contract_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            event_id = "outside-source-winner"
            event = {
                "schema_name": "software-satellite-event",
                "schema_version": 1,
                "event_id": event_id,
                "event_kind": "agent_task_run",
                "recorded_at_utc": "2026-04-01T00:00:00+00:00",
                "workspace": {"workspace_id": "local-default"},
                "session": {"surface": "agent_lane", "mode": "patch_plan_verify"},
                "outcome": {"status": "ok", "quality_status": "pass", "execution_status": "ok"},
                "content": {
                    "prompt": "Preserve source artifact contract reasons.",
                    "output_text": "The comparison winner points outside the workspace root.",
                    "options": {
                        "validation_mode": "agent_lane",
                        "validation_command": "python -m unittest tests.test_contract",
                        "pass_definition": "Winner evidence returns to a durable source artifact.",
                    },
                },
                "source_refs": {
                    "artifact_ref": {
                        "artifact_kind": "agent_run",
                        "artifact_path": str(Path(__file__).resolve()),
                    },
                },
                "tags": [],
            }
            event_log_path = root / "artifacts" / "event_logs" / "local-default.jsonl"
            write_event_log(event_log_path, [event], workspace_id="local-default")
            comparison = build_evaluation_comparison(
                candidate_event_ids=[event_id, "missing-loser"],
                winner_event_id=event_id,
                recorded_at_utc="2026-04-01T00:01:00+00:00",
                comparison_id="local-default:compare:outside-source-winner",
            )
            append_evaluation_comparison(
                evaluation_comparison_log_path(root=root),
                comparison,
                workspace_id="local-default",
            )

            snapshot, _latest_path, _run_path = record_evaluation_snapshot(
                root=root,
                index_summary={
                    "event_log_path": str(event_log_path),
                    "index_path": str(root / "artifacts" / "indexes" / "test.sqlite3"),
                },
            )
            issue_event = snapshot["evaluation_consistency"]["issue_events"][0]

        self.assertEqual(snapshot["counts"]["comparison_winner_missing_trace"], 1)
        self.assertIn("comparison_winner_source_artifact_not_durable", issue_event["missing_traces"])
        self.assertIn(
            "source_artifact_outside_workspace",
            issue_event["source_trace"]["source_artifact_reasons"],
        )

    def test_learning_preview_explains_stale_positive_trace_exclusions(self) -> None:
        event = {
            "event_id": "stale-learning-candidate",
            "event_kind": "agent_task_run",
            "recorded_at_utc": "2026-04-01T00:00:00+00:00",
            "session": {"surface": "agent_lane", "mode": "patch_plan_verify"},
            "outcome": {"status": "ok", "quality_status": "pass", "execution_status": "ok"},
            "content": {
                "prompt": "Do not promote stale positive evidence.",
                "output_text": "The later traces turned negative.",
                "options": {
                    "validation_mode": "agent_lane",
                    "validation_command": "python -m unittest tests.test_stale",
                    "pass_definition": "Latest negative traces win.",
                },
            },
            "source_refs": readable_test_source_refs(),
        }
        signals = [
            build_evaluation_signal(
                signal_kind="test_pass",
                source_event_id="stale-learning-candidate",
                source_event=event,
                recorded_at_utc="2026-04-01T00:00:00+00:00",
                signal_id="local-default:eval:test-pass",
            ),
            build_evaluation_signal(
                signal_kind="acceptance",
                source_event_id="stale-learning-candidate",
                source_event=event,
                recorded_at_utc="2026-04-01T00:01:00+00:00",
                signal_id="local-default:eval:acceptance",
            ),
            build_evaluation_signal(
                signal_kind="review_resolved",
                source_event_id="stale-learning-candidate",
                source_event=event,
                recorded_at_utc="2026-04-01T00:02:00+00:00",
                signal_id="local-default:eval:review-resolved",
            ),
            build_evaluation_signal(
                signal_kind="test_fail",
                source_event_id="stale-learning-candidate",
                source_event=event,
                recorded_at_utc="2026-04-01T00:03:00+00:00",
                signal_id="local-default:eval:test-fail",
            ),
            build_evaluation_signal(
                signal_kind="rejection",
                source_event_id="stale-learning-candidate",
                source_event=event,
                recorded_at_utc="2026-04-01T00:04:00+00:00",
                signal_id="local-default:eval:rejection",
            ),
            build_evaluation_signal(
                signal_kind="review_unresolved",
                source_event_id="stale-learning-candidate",
                source_event=event,
                recorded_at_utc="2026-04-01T00:05:00+00:00",
                signal_id="local-default:eval:review-unresolved",
            ),
        ]

        preview = build_learning_dataset_preview(
            {"workspace_id": "local-default", "paths": {}},
            {
                "candidates": [
                    {
                        "event_id": "stale-learning-candidate",
                        "state": "ready",
                        "label": "Stale learning candidate",
                        "reasons": ["accepted", "review_resolved", "test_pass"],
                        "blocked_by": [],
                        "export_decision": "include_when_approved",
                        "ready_for_policy": True,
                    }
                ]
            },
            events_by_id={"stale-learning-candidate": event},
            explicit_signals=signals,
            comparisons=[],
        )
        excluded = preview["excluded_candidates"][0]

        self.assertEqual(preview["supervised_example_candidates"], [])
        self.assertFalse(preview["training_export_ready"])
        self.assertTrue(preview["human_gate_required"])
        self.assertIn("test_fail_trace", excluded["excluded_by"])
        self.assertIn("stale_test_pass_trace", excluded["excluded_by"])
        self.assertIn("rejected_trace", excluded["excluded_by"])
        self.assertIn("stale_accepted_trace", excluded["excluded_by"])
        self.assertNotIn("missing_accepted_trace", excluded["excluded_by"])
        self.assertIn("review_unresolved_trace", excluded["excluded_by"])
        self.assertIn("stale_review_resolved_trace", excluded["excluded_by"])
        self.assertEqual(excluded["blocked_reason"], "review_unresolved")
        self.assertEqual(excluded["lifecycle_summary"]["test_state"], "failed")
        self.assertEqual(excluded["lifecycle_summary"]["selection_state"], "rejected")
        self.assertEqual(excluded["lifecycle_summary"]["review_state"], "unresolved")

    def test_learning_preview_recomputes_consistency_when_trace_overrides_are_supplied(self) -> None:
        event = {
            "event_id": "override-consistency-candidate",
            "event_kind": "agent_task_run",
            "recorded_at_utc": "2026-04-01T00:00:00+00:00",
            "session": {"surface": "agent_lane", "mode": "patch_plan_verify"},
            "outcome": {"status": "ok", "quality_status": "pass", "execution_status": "ok"},
            "content": {
                "prompt": "Current override traces should decide the preview.",
                "output_text": "The stale snapshot consistency entry should not veto this candidate.",
                "options": {
                    "validation_mode": "agent_lane",
                    "validation_command": "python -m unittest tests.test_override",
                    "pass_definition": "Override traces are authoritative for this run.",
                },
            },
            "source_refs": readable_test_source_refs(),
        }
        signals = [
            build_evaluation_signal(
                signal_kind="test_pass",
                source_event_id="override-consistency-candidate",
                source_event=event,
                recorded_at_utc="2026-04-01T00:01:00+00:00",
                signal_id="local-default:eval:override:test-pass",
            ),
            build_evaluation_signal(
                signal_kind="acceptance",
                source_event_id="override-consistency-candidate",
                source_event=event,
                recorded_at_utc="2026-04-01T00:02:00+00:00",
                signal_id="local-default:eval:override:acceptance",
            ),
        ]

        preview = build_learning_dataset_preview(
            {
                "workspace_id": "local-default",
                "paths": {},
                "evaluation_consistency": {
                    "events": [
                        {
                            "event_id": "override-consistency-candidate",
                            "stale_positive_signals": ["accepted", "test_pass"],
                            "missing_traces": [],
                        }
                    ]
                },
            },
            {
                "candidates": [
                    {
                        "event_id": "override-consistency-candidate",
                        "state": "ready",
                        "label": "Override consistency candidate",
                        "reasons": ["accepted", "test_pass"],
                        "blocked_by": [],
                        "export_decision": "include_when_approved",
                        "ready_for_policy": True,
                    }
                ]
            },
            events_by_id={"override-consistency-candidate": event},
            explicit_signals=signals,
            comparisons=[],
        )

        self.assertEqual(preview["excluded_candidates"], [])
        self.assertEqual(len(preview["supervised_example_candidates"]), 1)
        self.assertEqual(preview["review_queue"][0]["queue_state"], "ready")

    def test_learning_preview_explains_missing_comparison_winner_trace_without_snapshot_report(self) -> None:
        event = {
            "event_id": "missing-comparison-trace-candidate",
            "event_kind": "agent_task_run",
            "recorded_at_utc": "2026-04-01T00:00:00+00:00",
            "session": {"surface": "agent_lane", "mode": "patch_plan_verify"},
            "outcome": {"status": "ok", "quality_status": "pass", "execution_status": "ok"},
            "content": {
                "prompt": "Do not trust comparison_winner curation without a comparison trace.",
                "output_text": "This candidate needs the winner trace restored before learning preview promotion.",
                "options": {
                    "validation_mode": "agent_lane",
                    "validation_command": "python -m unittest tests.test_comparison_trace",
                    "pass_definition": "Comparison winner reason is traceable.",
                },
            },
            "source_refs": readable_test_source_refs(),
        }
        test_signal = build_evaluation_signal(
            signal_kind="test_pass",
            source_event_id="missing-comparison-trace-candidate",
            source_event=event,
            recorded_at_utc="2026-04-01T00:01:00+00:00",
            signal_id="local-default:eval:missing-comparison:test-pass",
        )

        preview = build_learning_dataset_preview(
            {"workspace_id": "local-default", "paths": {}},
            {
                "candidates": [
                    {
                        "event_id": "missing-comparison-trace-candidate",
                        "state": "ready",
                        "label": "Missing comparison trace candidate",
                        "reasons": ["comparison_winner", "test_pass"],
                        "blocked_by": [],
                        "export_decision": "include_when_approved",
                        "ready_for_policy": True,
                    }
                ]
            },
            events_by_id={"missing-comparison-trace-candidate": event},
            explicit_signals=[test_signal],
            comparisons=[],
        )
        excluded = preview["excluded_candidates"][0]

        self.assertEqual(preview["supervised_example_candidates"], [])
        self.assertIn("missing_comparison_winner_trace", excluded["excluded_by"])
        self.assertEqual(excluded["next_action"], "record_comparison_winner_trace")

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

    def test_cli_writes_learning_preview_without_training_export(self) -> None:
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
                    "--learning-preview",
                    "--learning-limit",
                    "1",
                    "--format",
                    "json",
                ],
            ), redirect_stdout(stdout):
                exit_code = evaluation_main()
            payload = json.loads(stdout.getvalue())
            preview_latest_exists = Path(payload["learning_preview"]["paths"]["learning_preview_latest_path"]).exists()
            source_curation_exists = Path(payload["learning_preview"]["source_curation_preview_path"]).exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["learning_preview"]["export_mode"], "preview_only")
        self.assertFalse(payload["learning_preview"]["training_export_ready"])
        self.assertEqual(payload["learning_preview"]["counts"]["previewed_candidate_count"], 1)
        self.assertFalse(
            payload["learning_preview"]["supervised_example_candidates"][0]["policy"]["training_job_allowed"]
        )
        self.assertTrue(preview_latest_exists)
        self.assertTrue(source_curation_exists)

    def test_cli_writes_human_selected_candidates_from_learning_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            event_id = "local-default:capability-matrix:matrix:row-1:chat"
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                build_evaluation_signal(
                    signal_kind="acceptance",
                    source_event_id=event_id,
                    rationale="Accepted before explicit shortlist selection.",
                ),
                workspace_id="local-default",
            )
            stdout = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "run_evaluation_loop.py",
                    "--root",
                    str(root),
                    "--human-selected-candidates",
                    "--select-candidate-event-id",
                    event_id,
                    "--rationale",
                    "Select for M7.3 preview inspection.",
                    "--format",
                    "json",
                ],
            ), redirect_stdout(stdout):
                exit_code = evaluation_main()
            payload = json.loads(stdout.getvalue())
            selection = payload["human_selected_candidates"]
            diff = payload["candidate_diff_summaries"][0]
            selection_latest_exists = Path(selection["paths"]["human_selected_latest_path"]).exists()
            diff_run_exists = Path(diff["paths"]["candidate_diff_run_path"]).exists()
            source_learning_exists = Path(selection["source_learning_preview_path"]).exists()
            selected = selection["selected_candidates"][0]

        self.assertEqual(exit_code, 0)
        self.assertEqual(selection["export_mode"], "preview_only")
        self.assertFalse(selection["training_export_ready"])
        self.assertTrue(selection["human_gate_required"])
        self.assertEqual(selection["counts"]["matched_candidate_count"], 1)
        self.assertEqual(selection["counts"]["selected_supervised_candidate_count"], 1)
        self.assertEqual(selection["selection"]["origin"], "cli")
        self.assertEqual(selection["selection"]["rationale"], "Select for M7.3 preview inspection.")
        self.assertTrue(selection_latest_exists)
        self.assertTrue(diff_run_exists)
        self.assertTrue(source_learning_exists)
        self.assertEqual(diff["target"]["artifact_kind"], "human_selected_candidate_list")
        self.assertEqual(diff["counts"]["target_candidate_count"], 1)
        self.assertFalse(diff["training_export_ready"])
        self.assertNotIn("supervised_example", selected)
        self.assertTrue(selected["evidence_summary"]["traceability"]["accepted"])
        self.assertTrue(selected["evidence_summary"]["traceability"]["test_pass"])
        self.assertFalse(selected["policy"]["training_job_allowed"])
        self.assertFalse(selected["policy"]["raw_log_export_allowed"])

    def test_cli_writes_jsonl_export_dry_run_from_human_selected_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            event_id = "local-default:capability-matrix:matrix:row-1:chat"
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                build_evaluation_signal(
                    signal_kind="acceptance",
                    source_event_id=event_id,
                    rationale="Accepted before export dry-run inspection.",
                ),
                workspace_id="local-default",
            )
            stdout = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "run_evaluation_loop.py",
                    "--root",
                    str(root),
                    "--confirm-export-policy",
                    "--source-event-id",
                    event_id,
                    "--human-selected-candidates",
                    "--select-candidate-event-id",
                    event_id,
                    "--jsonl-export-dry-run",
                    "--rationale",
                    "Inspect the selected candidate without writing JSONL.",
                    "--format",
                    "json",
                ],
            ), redirect_stdout(stdout):
                exit_code = evaluation_main()
            payload = json.loads(stdout.getvalue())
            dry_run = payload["jsonl_export_dry_run"]
            diff = payload["candidate_diff_summaries"][-1]
            dry_run_path = Path(dry_run["paths"]["jsonl_export_dry_run_run_path"])
            diff_path = Path(diff["paths"]["candidate_diff_run_path"])
            dry_run_path_exists = dry_run_path.exists()
            diff_path_exists = diff_path.exists()
            candidate = dry_run["candidates"][0]

        self.assertEqual(exit_code, 0)
        self.assertTrue(dry_run_path_exists)
        self.assertTrue(diff_path_exists)
        self.assertEqual(dry_run_path.suffix, ".json")
        self.assertEqual(dry_run["source_mode"], "human_selected_candidate_list")
        self.assertFalse(dry_run["training_export_ready"])
        self.assertTrue(dry_run["human_gate_required"])
        self.assertTrue(dry_run["not_trainable"])
        self.assertFalse(dry_run["export_policy"]["jsonl_training_export_allowed"])
        self.assertFalse(dry_run["export_policy"]["jsonl_file_written"])
        self.assertEqual(dry_run["counts"]["future_jsonl_candidate_if_separately_approved_count"], 1)
        self.assertEqual(dry_run["counts"]["would_write_jsonl_record_count"], 0)
        self.assertEqual(candidate["dry_run_status"], "future_jsonl_candidate_if_separately_approved")
        self.assertFalse(candidate["would_write_jsonl_record"])
        self.assertTrue(candidate["policy"]["export_policy_confirmed"])
        self.assertEqual(diff["base"]["artifact_kind"], "human_selected_candidate_list")
        self.assertEqual(diff["target"]["artifact_kind"], "jsonl_training_export_dry_run")
        self.assertEqual(diff["counts"]["target_candidate_count"], 1)
        self.assertFalse(diff["training_export_ready"])
        self.assertFalse(diff["inspection_policy"]["jsonl_training_export_allowed"])

    def test_cli_rejects_selected_candidate_without_human_selected_artifact(self) -> None:
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
                    "--select-candidate-event-id",
                    "local-default:capability-matrix:matrix:row-1:chat",
                ],
            ), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    evaluation_main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--select-candidate-event-id requires --human-selected-candidates.", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_confirms_export_policy_without_unlocking_training_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_capability_matrix(root)
            event_id = "local-default:capability-matrix:matrix:row-1:chat"
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                build_evaluation_signal(
                    signal_kind="acceptance",
                    source_event_id=event_id,
                    rationale="Accepted before policy confirmation.",
                ),
                workspace_id="local-default",
            )
            stdout = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "run_evaluation_loop.py",
                    "--root",
                    str(root),
                    "--confirm-export-policy",
                    "--source-event-id",
                    event_id,
                    "--rationale",
                    "Operator confirmed preview-only policy.",
                    "--learning-preview",
                    "--format",
                    "json",
                ],
            ), redirect_stdout(stdout):
                exit_code = evaluation_main()
            payload = json.loads(stdout.getvalue())
            candidate = payload["learning_preview"]["supervised_example_candidates"][0]

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["recorded_signal"]["signal_kind"], "export_policy_confirmed")
        self.assertEqual(payload["recorded_signal"]["evidence"]["export_mode"], "preview_only")
        self.assertFalse(payload["recorded_signal"]["evidence"]["training_job_allowed"])
        self.assertEqual(payload["snapshot"]["counts"]["export_policy_confirmed"], 1)
        self.assertFalse(payload["learning_preview"]["training_export_ready"])
        self.assertTrue(payload["learning_preview"]["human_gate_required"])
        self.assertEqual(candidate["review_queue"]["lifecycle_summary"]["policy_state"], "confirmed")
        self.assertEqual(candidate["review_queue"]["next_action"], "review_downstream_export_policy")
        self.assertEqual(candidate["review_queue"]["queue_priority"]["bucket"], "ready_policy_confirmed")

    def test_cli_record_signal_export_policy_uses_preview_only_evidence(self) -> None:
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
                    "export_policy_confirmed",
                    "--source-event-id",
                    "local-default:capability-matrix:matrix:row-1:chat",
                    "--format",
                    "json",
                ],
            ), redirect_stdout(stdout):
                exit_code = evaluation_main()
            payload = json.loads(stdout.getvalue())
            evidence = payload["recorded_signal"]["evidence"]

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["recorded_signal"]["signal_kind"], "export_policy_confirmed")
        self.assertEqual(evidence["export_mode"], "preview_only")
        self.assertFalse(evidence["training_export_ready"])
        self.assertTrue(evidence["human_gate_required"])
        self.assertFalse(evidence["training_job_allowed"])
        self.assertFalse(evidence["raw_log_export_allowed"])

    def test_cli_rejects_combined_record_signal_and_confirm_export_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
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
                    "--confirm-export-policy",
                    "--source-event-id",
                    "local-default:capability-matrix:matrix:row-1:chat",
                ],
            ), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    evaluation_main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--record-signal and --confirm-export-policy cannot be used together.", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_confirm_export_policy_requires_source_event_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stderr = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "run_evaluation_loop.py",
                    "--root",
                    str(root),
                    "--confirm-export-policy",
                ],
            ), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    evaluation_main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--confirm-export-policy requires --source-event-id.", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_confirm_export_policy_rejects_unknown_source_event_id(self) -> None:
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
                    "--confirm-export-policy",
                    "--source-event-id",
                    "local-default:missing-event",
                ],
            ), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    evaluation_main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn(
            "Unknown export-policy confirmation source_event_id `local-default:missing-event`.",
            stderr.getvalue(),
        )
        self.assertNotIn("Traceback", stderr.getvalue())

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
