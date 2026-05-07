from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from backend_swap import (  # noqa: E402
    BACKEND_CONFIG_SCHEMA_NAME,
    append_backend_config,
    backend_config_log_path,
    backend_harness_run_log_path,
    build_backend_config,
    check_backend_compatibility,
    ensure_default_backend_configs,
    ensure_backend_config_files,
    read_backend_configs,
    read_backend_harness_runs,
    run_backend_swap_harness,
)
from evaluation_loop import (  # noqa: E402
    append_evaluation_signal,
    build_evaluation_signal,
    evaluation_signal_log_path,
    format_learning_dataset_preview_report,
    record_curation_export_preview,
    record_evaluation_snapshot,
    record_learning_dataset_preview,
)
from memory_index import MemoryIndex  # noqa: E402
from run_backend_swap import main as backend_swap_main  # noqa: E402
from software_work_events import read_event_log  # noqa: E402


class BackendSwapTests(unittest.TestCase):
    def test_default_backends_run_side_by_side_into_index_and_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            harness_run, harness_path = run_backend_swap_harness(
                root=root,
                task_title="Backend swap smoke",
                goal="Prove two backend configs can use the same outer workflow.",
                scope_paths=["scripts/backend_swap.py"],
                plan_steps=[
                    "Load backend config.",
                    "Run the shared verification path.",
                ],
                verification_commands=[f"{sys.executable} -c \"print('backend harness ok')\""],
                pass_definition="Both backend runs complete verification.",
                timeout_seconds=10,
            )
            event_log = read_event_log(Path(harness_run["index_summary"]["event_log_path"]))
            backend_events = [
                event
                for event in event_log["events"]
                if event["event_kind"] == "agent_task_run"
            ]
            index = MemoryIndex(Path(harness_run["index_summary"]["index_path"]))
            matches = index.search("mock OR careful", limit=5)
            evaluation_snapshot, _latest_path, _run_path = record_evaluation_snapshot(root=root)
            harness_runs = read_backend_harness_runs(backend_harness_run_log_path(root=root))
            harness_path_exists = harness_path.exists()

        self.assertTrue(harness_path_exists)
        self.assertEqual(harness_runs[0]["run_id"], harness_run["run_id"])
        self.assertEqual(harness_run["status"], "completed")
        self.assertEqual(len(harness_run["backend_results"]), 2)
        self.assertEqual(harness_run["comparison"]["outcome"], "tie")
        self.assertEqual(
            {result["comparison_role"] for result in harness_run["backend_results"]},
            {"candidate"},
        )
        self.assertEqual(len(harness_run["comparison"]["backend_summary"]), 2)
        self.assertEqual(harness_run["evaluation_counts"]["test_pass"], 2)
        self.assertEqual(harness_run["evaluation_counts"]["comparisons"], 1)
        self.assertEqual(
            evaluation_snapshot["comparisons"][0]["backend_comparison"]["backend_count"],
            2,
        )
        self.assertEqual(evaluation_snapshot["counts"]["test_pass"], 2)
        self.assertEqual(evaluation_snapshot["counts"]["comparisons"], 1)
        self.assertEqual(len(backend_events), 2)
        self.assertEqual(
            {event["content"]["options"]["backend_compatibility_status"] for event in backend_events},
            {"compatible"},
        )
        self.assertEqual(
            {event["session"]["selected_model_id"] for event in backend_events},
            {"mock/fast-local-v1", "mock/careful-local-v1"},
        )
        self.assertGreaterEqual(len(matches), 2)

    def test_learning_preview_traces_backend_swap_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            harness_run, _harness_path = run_backend_swap_harness(
                root=root,
                task_title="Backend metadata learning preview",
                goal="Keep backend metadata traceable for a curated learning candidate.",
                scope_paths=["scripts/backend_swap.py"],
                plan_steps=[
                    "Load backend config.",
                    "Run shared verification.",
                ],
                verification_commands=[f"{sys.executable} -c \"print('metadata ok')\""],
                pass_definition="Backend run completes verification.",
                timeout_seconds=10,
            )
            selected_result = harness_run["backend_results"][0]
            accepted_signal = build_evaluation_signal(
                signal_kind="acceptance",
                source_event_id=selected_result["event_id"],
                rationale="The accepted backend run should remain traceable.",
            )
            append_evaluation_signal(
                evaluation_signal_log_path(root=root),
                accepted_signal,
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
            candidate = learning_preview["supervised_example_candidates"][0]
            report = format_learning_dataset_preview_report(learning_preview)

        self.assertEqual(candidate["event_id"], selected_result["event_id"])
        self.assertEqual(candidate["backend_metadata"]["backend_id"], selected_result["backend_id"])
        self.assertEqual(candidate["backend_metadata"]["model_id"], selected_result["model_id"])
        self.assertEqual(candidate["backend_metadata"]["compatibility_status"], "compatible")
        self.assertEqual(candidate["evidence"]["comparisons"][0]["comparison_role"], "candidate")
        self.assertEqual(
            candidate["evidence"]["comparisons"][0]["backend_comparison"]["backend_count"],
            2,
        )
        self.assertEqual(
            candidate["review_queue"]["comparison_evidence"]["latest_role"],
            "candidate",
        )
        self.assertEqual(candidate["review_queue"]["queue_state"], "ready")
        self.assertEqual(candidate["review_queue"]["next_action"], "confirm_export_policy")
        self.assertIn("latency_profile", candidate["backend_metadata"]["metadata"])
        self.assertEqual(
            candidate["backend_metadata"]["metadata_source_artifact_path"],
            selected_result["run_artifact_path"],
        )
        self.assertIn("Backend comparison summary:", report)
        self.assertIn("comparison=candidate", report)

    def test_compatibility_check_reports_missing_required_capability(self) -> None:
        config = build_backend_config(
            backend_id="mock-incomplete",
            display_name="Mock Incomplete",
            adapter_kind="mock",
            model_id="mock/incomplete-v1",
            capabilities={
                "text_generation": {"supported": False},
                "agent_lane": {"supported": True},
                "verification_commands": {"supported": True},
                "file_first_artifacts": {"supported": True},
            },
        )

        report = check_backend_compatibility(config)

        self.assertEqual(report["status"], "incompatible")
        self.assertEqual(report["missing_capabilities"], ["text_generation"])

    def test_default_backend_configs_are_file_first_and_not_duplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = ensure_default_backend_configs(root=root)
            second = ensure_default_backend_configs(root=root)
            config_log = backend_config_log_path(root=root)
            config_log_exists = config_log.exists()

        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertTrue(config_log_exists)
        self.assertEqual(first[0]["schema_name"], BACKEND_CONFIG_SCHEMA_NAME)

    def test_file_backend_configs_are_returned_once_per_backend_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "duplicate-backend.json"
            config = build_backend_config(
                backend_id="local-duplicate-json",
                display_name="Local Duplicate JSON",
                adapter_kind="local",
                model_id="local/duplicate-json-v1",
                capabilities={
                    "text_generation": {"supported": True},
                    "agent_lane": {"supported": True},
                    "verification_commands": {"supported": True},
                    "file_first_artifacts": {"supported": True},
                },
            )
            config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
            ensured = ensure_backend_config_files(
                [config_path, config_path],
                root=root,
                workspace_id="local-default",
            )
            logged = read_backend_configs(backend_config_log_path(root=root))

        self.assertEqual([item["backend_id"] for item in ensured], ["local-duplicate-json"])
        self.assertEqual([item["backend_id"] for item in logged], ["local-duplicate-json"])

    def test_cli_records_side_by_side_json_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stdout = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "run_backend_swap.py",
                    "--root",
                    str(root),
                    "--task-title",
                    "CLI backend swap",
                    "--goal",
                    "Capture a side-by-side backend harness run.",
                    "--scope-path",
                    "scripts/backend_swap.py",
                    "--plan-step",
                    "Load config.",
                    "--plan-step",
                    "Run verification.",
                    "--verification-command",
                    f"{sys.executable} -c \"print('cli backend swap ok')\"",
                    "--format",
                    "json",
                ],
            ), redirect_stdout(stdout):
                exit_code = backend_swap_main()
            payload = json.loads(stdout.getvalue())
            harness_path = Path(payload["harness_run_artifact_path"])
            harness_path_exists = harness_path.exists()

        self.assertEqual(exit_code, 0)
        self.assertTrue(harness_path_exists)
        self.assertEqual(payload["harness_run"]["status"], "completed")
        self.assertEqual(payload["harness_run"]["comparison"]["outcome"], "tie")
        self.assertEqual(len(payload["harness_run"]["backend_results"]), 2)

    def test_duplicate_backend_ids_are_rejected_before_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaises(ValueError) as raised:
                run_backend_swap_harness(
                    root=root,
                    task_title="Duplicate backend ids",
                    goal="Reject a side-by-side run that uses the same backend twice.",
                    plan_steps=["Load config."],
                    verification_commands=[f"{sys.executable} -c \"print('ok')\""],
                    backend_ids=["mock-fast-local", "mock-fast-local"],
                    timeout_seconds=10,
                )

        self.assertIn("distinct backend ids", str(raised.exception))

    def test_iterable_scope_and_acceptance_are_preserved_for_each_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            harness_run, _harness_path = run_backend_swap_harness(
                root=root,
                task_title="Iterable inputs",
                goal="Preserve generator inputs across backend runs.",
                scope_paths=(path for path in ["scripts/backend_swap.py"]),
                plan_steps=["Load config."],
                verification_commands=[f"{sys.executable} -c \"print('ok')\""],
                acceptance_criteria=(item for item in ["Both runs keep scoped metadata."]),
                timeout_seconds=10,
            )

        self.assertEqual(harness_run["task"]["scope_paths"], ["scripts/backend_swap.py"])
        self.assertEqual(harness_run["task"]["acceptance_criteria"], ["Both runs keep scoped metadata."])

    def test_backend_invocation_failure_blocks_backend_win(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            failing_config = build_backend_config(
                backend_id="mock-failing-local",
                display_name="Mock Failing Local",
                adapter_kind="mock",
                model_id="mock/failing-local-v1",
                capabilities={
                    "text_generation": {"supported": True},
                    "agent_lane": {"supported": True},
                    "verification_commands": {"supported": True},
                    "file_first_artifacts": {"supported": True},
                },
                adapter_options={"force_status": "failed"},
            )
            append_backend_config(
                backend_config_log_path(root=root),
                failing_config,
                workspace_id="local-default",
            )
            harness_run, _harness_path = run_backend_swap_harness(
                root=root,
                task_title="Invocation failure",
                goal="Failed backend invocation should not win on verification alone.",
                plan_steps=["Load config."],
                verification_commands=[f"{sys.executable} -c \"print('verification still passes')\""],
                backend_ids=["mock-fast-local", "mock-failing-local"],
                timeout_seconds=10,
            )
            by_backend = {
                result["backend_id"]: result
                for result in harness_run["backend_results"]
            }

        self.assertEqual(harness_run["status"], "completed_with_failures")
        self.assertEqual(harness_run["comparison"]["outcome"], "winner_selected")
        self.assertEqual(
            harness_run["comparison"]["winner_event_id"],
            by_backend["mock-fast-local"]["event_id"],
        )
        self.assertEqual(by_backend["mock-failing-local"]["run_status"], "failed")
        self.assertEqual(harness_run["evaluation_counts"]["test_pass"], 1)
        self.assertEqual(harness_run["evaluation_counts"]["test_fail"], 1)

    def test_learning_preview_traces_backend_comparison_winner_and_loser(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            failing_config = build_backend_config(
                backend_id="mock-loser-local",
                display_name="Mock Loser Local",
                adapter_kind="mock",
                model_id="mock/loser-local-v1",
                capabilities={
                    "text_generation": {"supported": True},
                    "agent_lane": {"supported": True},
                    "verification_commands": {"supported": True},
                    "file_first_artifacts": {"supported": True},
                },
                adapter_options={"force_status": "failed"},
                metadata={"comparison_role": "failure-biased local dry run"},
            )
            append_backend_config(
                backend_config_log_path(root=root),
                failing_config,
                workspace_id="local-default",
            )
            harness_run, _harness_path = run_backend_swap_harness(
                root=root,
                task_title="Backend comparison winner loser",
                goal="Keep winner and loser roles traceable from backend swap.",
                plan_steps=["Load config."],
                verification_commands=[f"{sys.executable} -c \"print('winner loser trace ok')\""],
                backend_ids=["mock-fast-local", "mock-loser-local"],
                timeout_seconds=10,
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
            by_backend = {
                result["backend_id"]: result
                for result in harness_run["backend_results"]
            }
            supervised_by_event = {
                candidate["event_id"]: candidate
                for candidate in learning_preview["supervised_example_candidates"]
            }
            queue_by_event = {
                item["event_id"]: item
                for item in learning_preview["review_queue"]
            }
            excluded_by_event = {
                item["event_id"]: item
                for item in learning_preview["excluded_candidates"]
            }

        winner_event_id = by_backend["mock-fast-local"]["event_id"]
        loser_event_id = by_backend["mock-loser-local"]["event_id"]
        winner_candidate = supervised_by_event[winner_event_id]
        loser_queue_item = queue_by_event[loser_event_id]
        loser_excluded = excluded_by_event[loser_event_id]

        self.assertEqual(by_backend["mock-fast-local"]["comparison_role"], "winner")
        self.assertEqual(by_backend["mock-loser-local"]["comparison_role"], "loser")
        self.assertEqual(winner_candidate["evidence"]["comparisons"][0]["comparison_role"], "winner")
        self.assertEqual(
            winner_candidate["review_queue"]["lifecycle_summary"]["comparison_state"],
            "winner",
        )
        self.assertEqual(loser_queue_item["comparison_evidence"]["latest_role"], "loser")
        self.assertEqual(loser_queue_item["backend_metadata"]["backend_id"], "mock-loser-local")
        self.assertEqual(loser_queue_item["lifecycle_summary"]["comparison_state"], "loser")
        self.assertEqual(loser_excluded["comparison_evidence"]["latest_role"], "loser")
        self.assertEqual(loser_excluded["blocked_reason"], "test_fail")
        self.assertFalse(learning_preview["training_export_ready"])

    def test_backend_invocation_failure_preserves_verification_failure_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            failing_config = build_backend_config(
                backend_id="mock-failing-with-test-failure",
                display_name="Mock Failing With Test Failure",
                adapter_kind="mock",
                model_id="mock/failing-with-test-failure-v1",
                capabilities={
                    "text_generation": {"supported": True},
                    "agent_lane": {"supported": True},
                    "verification_commands": {"supported": True},
                    "file_first_artifacts": {"supported": True},
                },
                adapter_options={"force_status": "failed"},
            )
            append_backend_config(
                backend_config_log_path(root=root),
                failing_config,
                workspace_id="local-default",
            )
            harness_run, _harness_path = run_backend_swap_harness(
                root=root,
                task_title="Two failure sources",
                goal="Preserve both backend invocation and verification failure evidence.",
                plan_steps=["Load config."],
                verification_commands=[f"{sys.executable} -c \"import sys; sys.exit(9)\""],
                backend_ids=["mock-fast-local", "mock-failing-with-test-failure"],
                timeout_seconds=10,
            )
            failing_result = next(
                result
                for result in harness_run["backend_results"]
                if result["backend_id"] == "mock-failing-with-test-failure"
            )
            run_payload = json.loads(Path(failing_result["run_artifact_path"]).read_text(encoding="utf-8"))
            outcome = run_payload["outcome"]

        self.assertEqual(outcome["failure_summary"], "Command exited with status 9.")
        self.assertIn("backend_invocation_failure_summary", outcome)
        self.assertIn("failed `Two failure sources`", outcome["backend_invocation_failure_summary"])

    def test_successful_invocation_does_not_hide_verification_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            harness_run, _harness_path = run_backend_swap_harness(
                root=root,
                task_title="Verification failure summary",
                goal="Keep failed verification text as the primary run summary.",
                plan_steps=["Load config."],
                verification_commands=[f"{sys.executable} -c \"import sys; sys.exit(6)\""],
                timeout_seconds=10,
            )
            first_result = harness_run["backend_results"][0]
            run_payload = json.loads(Path(first_result["run_artifact_path"]).read_text(encoding="utf-8"))
            outcome = run_payload["outcome"]
            event_log = read_event_log(Path(harness_run["index_summary"]["event_log_path"]))
            first_event = next(
                event
                for event in event_log["events"]
                if event["source_refs"]["artifact_ref"]["entry_id"] == first_result["run_id"]
            )

        self.assertEqual(outcome["failure_summary"], "Command exited with status 6.")
        self.assertEqual(outcome["result_summary"], "Command exited with status 6.")
        self.assertIn("completed `Verification failure summary`", outcome["backend_output_text"])
        self.assertEqual(first_event["content"]["output_text"], "Command exited with status 6.")

    def test_cli_lists_backends_without_task_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stdout = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "run_backend_swap.py",
                    "--root",
                    str(root),
                    "--list-backends",
                    "--format",
                    "json",
                ],
            ), redirect_stdout(stdout):
                exit_code = backend_swap_main()
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(payload["backends"]), 2)
        self.assertEqual(
            {compatibility["status"] for compatibility in payload["compatibilities"]},
            {"compatible"},
        )

    def test_cli_loads_file_first_backend_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "custom-backend.json"
            config = build_backend_config(
                backend_id="local-json-backend",
                display_name="Local JSON Backend",
                adapter_kind="local",
                model_id="local/json-backend-v1",
                capabilities={
                    "text_generation": {"supported": True},
                    "agent_lane": {"supported": True},
                    "verification_commands": {"supported": True},
                    "file_first_artifacts": {"supported": True},
                },
            )
            config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
            stdout = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "run_backend_swap.py",
                    "--root",
                    str(root),
                    "--backend-config-json",
                    str(config_path),
                    "--backend-id",
                    "mock-fast-local",
                    "--backend-id",
                    "local-json-backend",
                    "--task-title",
                    "JSON backend config",
                    "--goal",
                    "Run a backend loaded from a file-first config.",
                    "--plan-step",
                    "Load config.",
                    "--verification-command",
                    f"{sys.executable} -c \"print('json config ok')\"",
                    "--format",
                    "json",
                ],
            ), redirect_stdout(stdout):
                exit_code = backend_swap_main()
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            {result["backend_id"] for result in payload["harness_run"]["backend_results"]},
            {"mock-fast-local", "local-json-backend"},
        )


if __name__ == "__main__":
    unittest.main()
