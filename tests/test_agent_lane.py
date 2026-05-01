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

from agent_lane import (  # noqa: E402
    AGENT_TASK_SCHEMA_NAME,
    AGENT_TASK_SCHEMA_VERSION,
    agent_run_artifact_path,
    agent_run_log_path,
    agent_task_log_path,
    build_agent_lane_snapshot,
    build_agent_task,
    format_agent_lane_snapshot_report,
    record_agent_run,
    record_agent_task,
    run_agent_task,
)
from evaluation_loop import record_evaluation_snapshot  # noqa: E402
from memory_index import MemoryIndex, rebuild_memory_index  # noqa: E402
from run_agent_lane import main as agent_lane_main  # noqa: E402
from software_work_events import read_event_log  # noqa: E402


class AgentLaneTests(unittest.TestCase):
    def test_patch_plan_verify_run_flows_into_index_and_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task = build_agent_task(
                title="Patch plan verify smoke",
                goal="Exercise a bounded agent lane success path.",
                scope_paths=["scripts/agent_lane.py"],
                plan_steps=[
                    "Inspect the scoped file.",
                    "Apply the smallest patch.",
                    "Run focused verification.",
                ],
                verification_commands=[f"{sys.executable} -c \"print('agent lane happy path')\""],
                acceptance_criteria=["Verification command passes."],
            )
            recorded_task = record_agent_task(task, root=root)
            run = run_agent_task(recorded_task, root=root, timeout_seconds=10)
            recorded_run, run_path = record_agent_run(run, root=root)
            run_path_exists = run_path.exists()
            lane_snapshot = build_agent_lane_snapshot(root=root)
            report = format_agent_lane_snapshot_report(lane_snapshot)
            index_summary = rebuild_memory_index(root=root)
            event_log = read_event_log(Path(index_summary["event_log_path"]))
            index = MemoryIndex(Path(index_summary["index_path"]))
            matches = index.search("agent lane happy")
            evaluation_snapshot, _latest_path, _run_path = record_evaluation_snapshot(root=root)

        self.assertTrue(run_path_exists)
        self.assertEqual(recorded_run["status"], "succeeded")
        self.assertEqual(recorded_run["outcome"]["quality_status"], "pass")
        self.assertEqual(lane_snapshot["counts"]["runs"], 1)
        self.assertIn("Succeeded: 1", report)
        self.assertEqual(index_summary["agent_lane_event_count"], 1)
        self.assertEqual(event_log["events"][0]["event_kind"], "agent_task_run")
        self.assertEqual(event_log["events"][0]["outcome"]["status"], "ok")
        self.assertEqual(event_log["events"][0]["outcome"]["quality_status"], "pass")
        self.assertEqual(event_log["events"][0]["content"]["options"]["agent_run_status"], "succeeded")
        self.assertGreaterEqual(len(matches), 1)
        self.assertEqual(matches[0]["event_kind"], "agent_task_run")
        self.assertEqual(evaluation_snapshot["counts"]["test_pass"], 1)

    def test_failed_verification_is_first_class_test_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task = build_agent_task(
                title="Patch plan verify failure",
                goal="Record a failing verification as a first-class outcome.",
                plan_steps=["Run the failing verification."],
                verification_commands=[f"{sys.executable} -c \"import sys; sys.exit(7)\""],
                pass_definition="The command must exit with status 0.",
            )
            recorded_task = record_agent_task(task, root=root)
            run = run_agent_task(recorded_task, root=root, timeout_seconds=10)
            recorded_run, _run_path = record_agent_run(run, root=root)
            index_summary = rebuild_memory_index(root=root)
            event_log = read_event_log(Path(index_summary["event_log_path"]))
            evaluation_snapshot, _latest_path, _run_path = record_evaluation_snapshot(root=root)

        self.assertEqual(recorded_run["status"], "failed")
        self.assertEqual(recorded_run["outcome"]["quality_status"], "fail")
        self.assertEqual(recorded_run["outcome"]["verification_failed_count"], 1)
        self.assertEqual(event_log["events"][0]["outcome"]["status"], "failed")
        self.assertEqual(event_log["events"][0]["outcome"]["execution_status"], "failed")
        self.assertEqual(evaluation_snapshot["counts"]["test_fail"], 1)
        self.assertEqual(evaluation_snapshot["counts"]["pending_failures"], 1)

    def test_malformed_task_command_is_rejected_before_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            malformed_task = {
                "schema_name": AGENT_TASK_SCHEMA_NAME,
                "schema_version": AGENT_TASK_SCHEMA_VERSION,
                "task_id": "local-default:agent-task:manual",
                "workspace_id": "local-default",
                "created_at_utc": "2026-04-01T00:00:00+00:00",
                "origin": "test",
                "task_kind": "patch_plan_verify",
                "title": "Malformed command",
                "goal": "Reject malformed verification commands.",
                "scope": {"paths": []},
                "plan_steps": [{"step_id": "plan-1", "description": "Inspect."}],
                "verification": {"commands": [{"command_id": "verify-1"}]},
                "acceptance_criteria": [],
                "tags": [],
            }
            with self.assertRaises(ValueError) as raised:
                record_agent_task(malformed_task, root=root)

            task_log_exists = agent_task_log_path(root=root).exists()

        self.assertIn("verification command 1 is missing command", str(raised.exception))
        self.assertFalse(task_log_exists)

    def test_duplicate_task_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task = build_agent_task(
                task_id="local-default:agent-task:duplicate",
                title="Duplicate task",
                goal="Reject duplicate task ids.",
                plan_steps=["Record once."],
                verification_commands=[f"{sys.executable} -c \"print('ok')\""],
            )
            record_agent_task(task, root=root)
            with self.assertRaises(ValueError) as raised:
                record_agent_task(task, root=root)

        self.assertIn("already exists", str(raised.exception))

    def test_run_rejects_mismatched_task_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task = build_agent_task(
                title="Mismatched task snapshot",
                goal="Reject inconsistent run metadata.",
                plan_steps=["Run verification."],
                verification_commands=[f"{sys.executable} -c \"print('ok')\""],
            )
            recorded_task = record_agent_task(task, root=root)
            run = run_agent_task(recorded_task, root=root, timeout_seconds=10)
            run["task_snapshot"]["task_id"] = "local-default:agent-task:other"
            with self.assertRaises(ValueError) as raised:
                record_agent_run(run, root=root)

        self.assertIn("task_id does not match", str(raised.exception))

    def test_malformed_verification_command_is_recorded_as_failed_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task = build_agent_task(
                title="Malformed verification command",
                goal="Keep malformed command failures inspectable.",
                plan_steps=["Run malformed verification."],
                verification_commands=["python -c \""],
            )
            recorded_task = record_agent_task(task, root=root)
            run = run_agent_task(recorded_task, root=root, timeout_seconds=10)
            recorded_run, _run_path = record_agent_run(run, root=root)
            rebuild_memory_index(root=root)
            evaluation_snapshot, _latest_path, _run_path = record_evaluation_snapshot(root=root)

        self.assertEqual(recorded_run["status"], "failed")
        self.assertIn("Invalid verification command", recorded_run["outcome"]["failure_summary"])
        self.assertEqual(evaluation_snapshot["counts"]["test_fail"], 1)

    def test_run_rejects_mismatched_outcome_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task = build_agent_task(
                title="Mismatched outcome status",
                goal="Reject inconsistent run outcome metadata.",
                plan_steps=["Run verification."],
                verification_commands=[f"{sys.executable} -c \"print('ok')\""],
            )
            recorded_task = record_agent_task(task, root=root)
            run = run_agent_task(recorded_task, root=root, timeout_seconds=10)
            run["outcome"]["status"] = "failed"
            with self.assertRaises(ValueError) as raised:
                record_agent_run(run, root=root)

        self.assertIn("outcome.status does not match", str(raised.exception))

    def test_duplicate_run_ids_are_rejected_before_event_count_inflates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task = build_agent_task(
                title="Duplicate run",
                goal="Reject duplicate run ids.",
                plan_steps=["Run verification."],
                verification_commands=[f"{sys.executable} -c \"print('ok')\""],
            )
            recorded_task = record_agent_task(task, root=root)
            run = run_agent_task(
                recorded_task,
                root=root,
                run_id="local-default:agent-run:duplicate",
                timeout_seconds=10,
            )
            record_agent_run(run, root=root)
            with self.assertRaises(ValueError) as raised:
                record_agent_run(run, root=root)
            index_summary = rebuild_memory_index(root=root)

        self.assertIn("already exists", str(raised.exception))
        self.assertEqual(index_summary["agent_lane_event_count"], 1)

    def test_artifact_paths_do_not_collide_for_similar_run_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            colon_path = agent_run_artifact_path(root=root, run_id="local-default:agent-run:a:b")
            dash_path = agent_run_artifact_path(root=root, run_id="local-default:agent-run:a-b")

        self.assertNotEqual(colon_path.name, dash_path.name)

    def test_zero_byte_logs_are_recovered_on_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            agent_task_log_path(root=root).parent.mkdir(parents=True, exist_ok=True)
            agent_task_log_path(root=root).write_text("", encoding="utf-8")
            agent_run_log_path(root=root).parent.mkdir(parents=True, exist_ok=True)
            agent_run_log_path(root=root).write_text("", encoding="utf-8")
            task = build_agent_task(
                title="Recover empty logs",
                goal="Append should rewrite zero-byte log headers.",
                plan_steps=["Run verification."],
                verification_commands=[f"{sys.executable} -c \"print('ok')\""],
            )
            recorded_task = record_agent_task(task, root=root)
            run = run_agent_task(recorded_task, root=root, timeout_seconds=10)
            recorded_run, run_path = record_agent_run(run, root=root)
            run_path_exists = run_path.exists()
            lane_snapshot = build_agent_lane_snapshot(root=root)

        self.assertEqual(recorded_run["status"], "succeeded")
        self.assertTrue(run_path_exists)
        self.assertEqual(lane_snapshot["counts"]["tasks"], 1)
        self.assertEqual(lane_snapshot["counts"]["runs"], 1)

    def test_workspace_mismatch_does_not_write_orphan_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task = build_agent_task(
                workspace_id="other-workspace",
                title="Workspace mismatch",
                goal="Reject before writing artifacts under the wrong workspace.",
                plan_steps=["Run verification."],
                verification_commands=[f"{sys.executable} -c \"print('ok')\""],
            )
            run = run_agent_task(task, root=root, timeout_seconds=10)
            with self.assertRaises(ValueError) as raised:
                record_agent_run(run, root=root)
            artifact_files = list(root.glob("artifacts/agent_lane/**/*.json"))

        self.assertIn("other-workspace", str(raised.exception))
        self.assertEqual(artifact_files, [])

    def test_cli_records_task_run_snapshot_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stdout = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "run_agent_lane.py",
                    "--root",
                    str(root),
                    "--task-title",
                    "CLI patch plan verify",
                    "--goal",
                    "Capture a minimal CLI lane run.",
                    "--scope-path",
                    "scripts/agent_lane.py",
                    "--plan-step",
                    "Record the task.",
                    "--plan-step",
                    "Run verification.",
                    "--verification-command",
                    f"{sys.executable} -c \"print('cli verification ok')\"",
                    "--format",
                    "json",
                ],
            ), redirect_stdout(stdout):
                exit_code = agent_lane_main()
            payload = json.loads(stdout.getvalue())
            run_artifact_exists = Path(payload["run_artifact_path"]).exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["run"]["status"], "succeeded")
        self.assertEqual(payload["snapshot"]["counts"]["tasks"], 1)
        self.assertEqual(payload["snapshot"]["counts"]["runs"], 1)
        self.assertEqual(payload["index_summary"]["agent_lane_event_count"], 1)
        self.assertTrue(run_artifact_exists)

    def test_cli_invalid_timeout_does_not_leave_orphan_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stderr = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "run_agent_lane.py",
                    "--root",
                    str(root),
                    "--task-title",
                    "Invalid timeout",
                    "--goal",
                    "Avoid orphan tasks.",
                    "--plan-step",
                    "Do not persist this task.",
                    "--verification-command",
                    f"{sys.executable} -c \"print('ok')\"",
                    "--timeout-seconds",
                    "0",
                ],
            ), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    agent_lane_main()

            task_log_exists = agent_task_log_path(root=root).exists()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--timeout-seconds must be at least 1.", stderr.getvalue())
        self.assertFalse(task_log_exists)


if __name__ == "__main__":
    unittest.main()
