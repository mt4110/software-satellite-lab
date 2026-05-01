from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from software_work_events import (  # noqa: E402
    EVENT_LOG_SCHEMA_NAME,
    EVENT_SCHEMA_NAME,
    iter_capability_matrix_events,
    iter_workspace_events,
    read_event_log,
    rebuild_workspace_event_log,
)
from artifact_schema import build_artifact_payload, build_prompt_record, build_runtime_record, write_artifact  # noqa: E402
from workspace_state import WorkspaceSessionStore  # noqa: E402


class SoftwareWorkEventTests(unittest.TestCase):
    def test_workspace_entries_are_normalized_into_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = WorkspaceSessionStore(root=root)

            base_messages = store.chat_messages_for_next_turn(
                model_id="backend-a",
                system_prompt="You are concise.",
            )
            store.record_chat_turn(
                model_id="backend-a",
                status="ok",
                artifact_path=root / "artifacts" / "text" / "chat.json",
                prompt="Review this change.",
                system_prompt="You are concise.",
                resolved_user_prompt="Review this change.",
                output_text="Looks good with one note.",
                base_messages=base_messages,
                notes=["accepted with follow-up"],
            )
            store.record_session_run(
                surface="vision",
                model_id="backend-a",
                mode="pdf-summary",
                artifact_kind="vision",
                artifact_path=root / "artifacts" / "vision" / "summary.json",
                status="quality_fail",
                prompt="Summarize this file.",
                system_prompt="Summarize faithfully.",
                resolved_user_prompt="Summarize this file.",
                output_text="Partial summary",
                attachments=[{"role": "primary_input", "path": root / "assets" / "spec.pdf"}],
                notes=["needs repair"],
                options={"max_pages": 2},
            )

            events = iter_workspace_events(root=root)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["schema_name"], EVENT_SCHEMA_NAME)
        self.assertEqual(events[0]["event_kind"], "chat_turn")
        self.assertEqual(events[0]["outcome"]["status"], "ok")
        self.assertEqual(events[0]["content"]["output_text"], "Looks good with one note.")
        self.assertEqual(events[1]["session"]["surface"], "vision")
        self.assertEqual(events[1]["outcome"]["status"], "quality_fail")
        self.assertEqual(events[1]["source_refs"]["attached_assets"][0]["kind"], "pdf")

    def test_rebuild_workspace_event_log_writes_jsonl_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = WorkspaceSessionStore(root=root)
            base_messages = store.chat_messages_for_next_turn(
                model_id="backend-a",
                system_prompt="You are concise.",
            )
            store.record_chat_turn(
                model_id="backend-a",
                status="ok",
                artifact_path=root / "artifacts" / "text" / "chat.json",
                prompt="Plan this task.",
                system_prompt="You are concise.",
                resolved_user_prompt="Plan this task.",
                output_text="Three steps.",
                base_messages=base_messages,
            )

            payload = rebuild_workspace_event_log(root=root)
            loaded = read_event_log(Path(payload["path"]))

        self.assertEqual(payload["schema_name"], EVENT_LOG_SCHEMA_NAME)
        self.assertEqual(payload["event_count"], 1)
        self.assertEqual(loaded["event_count"], 1)
        self.assertEqual(loaded["events"][0]["content"]["prompt"], "Plan this task.")

    def test_capability_matrix_results_can_be_backfilled_into_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = build_artifact_payload(
                artifact_kind="capability_matrix",
                status="blocked",
                runtime=build_runtime_record(backend="capability-matrix", model_id="backend-a"),
                prompts=build_prompt_record(),
                extra={
                    "results": [
                        {
                            "capability": "long-context",
                            "phase": "phase6",
                            "status": "blocked",
                            "artifact_kind": None,
                            "artifact_path": None,
                            "validation_command": "python scripts/run_capability_matrix.py --only long-context",
                            "claim_scope": "stress long context on local hardware",
                            "output_preview": None,
                            "blocker": {
                                "kind": "hardware_limit",
                                "message": "RuntimeError: Invalid buffer size: 135.77 GiB",
                                "external": True,
                            },
                            "quality_status": "not_run",
                            "quality_checks": [],
                            "quality_notes": ["hardware-limited run"],
                            "notes": [],
                            "runtime_backend": None,
                            "execution_status": "blocked",
                            "validation_mode": None,
                            "pass_definition": None,
                            "preprocessing_lineage": [],
                        },
                        {
                            "capability": "long-context",
                            "phase": "phase6-retry",
                            "status": "ok",
                            "artifact_kind": "text",
                            "artifact_path": str(root / "artifacts" / "text" / "long-context-retry.json"),
                            "validation_command": "python scripts/run_capability_matrix.py --only long-context",
                            "claim_scope": "retry long context on local hardware",
                            "output_preview": "Retry completed.",
                            "blocker": None,
                            "quality_status": "pass",
                            "quality_checks": [],
                            "quality_notes": ["retry passed"],
                            "notes": [],
                            "runtime_backend": "local",
                            "execution_status": "ok",
                            "validation_mode": "live",
                            "pass_definition": "Retry completes without overwriting the first row.",
                            "preprocessing_lineage": [],
                        }
                    ]
                },
            )
            write_artifact(root / "artifacts" / "capability_matrix" / "matrix.json", payload)

            events = iter_capability_matrix_events(root=root)

        self.assertEqual(len(events), 2)
        self.assertEqual(len({event["event_id"] for event in events}), 2)
        self.assertTrue(events[0]["event_id"].endswith(":row-1:long-context"))
        self.assertTrue(events[1]["event_id"].endswith(":row-2:long-context"))
        self.assertEqual(
            events[0]["source_refs"]["artifact_ref"]["entry_id"],
            "matrix:row-1:long-context",
        )
        self.assertEqual(
            events[1]["source_refs"]["artifact_ref"]["entry_id"],
            "matrix:row-2:long-context",
        )
        self.assertEqual(events[0]["event_kind"], "capability_result")
        self.assertEqual(events[0]["outcome"]["status"], "blocked")
        self.assertIn("hardware-limited run", events[0]["content"]["notes"])
        self.assertEqual(events[1]["outcome"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
