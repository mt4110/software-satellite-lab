from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from workspace_state import (  # noqa: E402
    DEFAULT_WORKSPACE_ID,
    SESSION_SCHEMA_NAME,
    WORKSPACE_SCHEMA_NAME,
    WorkspaceSessionStore,
    read_session_manifest,
    read_workspace_manifest,
    session_manifest_path,
)


class WorkspaceSessionStoreTests(unittest.TestCase):
    def test_ensure_session_creates_workspace_manifest_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = WorkspaceSessionStore(root=root)

            session = store.ensure_session(
                surface="vision",
                model_id="google/gemma-4-E2B-it",
                mode="caption",
            )
            workspace = read_workspace_manifest(store.manifest_path)

        self.assertEqual(workspace["schema_name"], WORKSPACE_SCHEMA_NAME)
        self.assertEqual(workspace["workspace_id"], DEFAULT_WORKSPACE_ID)
        self.assertEqual(workspace["selected_model_id"], "google/gemma-4-E2B-it")
        self.assertEqual(workspace["active_session_ids"]["vision"], session["session_id"])
        self.assertEqual(workspace["sessions"][0]["manifest_path"], "sessions/vision-main.json")

    def test_record_chat_turn_persists_multi_turn_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = WorkspaceSessionStore(root=root)
            artifact_one = root / "artifacts" / "text" / "first.json"
            artifact_two = root / "artifacts" / "text" / "second.json"

            first_base_messages = store.chat_messages_for_next_turn(
                model_id="google/gemma-4-E2B-it",
                system_prompt="You are a concise, helpful assistant.",
            )
            store.record_chat_turn(
                model_id="google/gemma-4-E2B-it",
                status="ok",
                artifact_path=artifact_one,
                prompt="First question",
                system_prompt="You are a concise, helpful assistant.",
                resolved_user_prompt="First question",
                output_text="First answer",
                base_messages=first_base_messages,
            )

            second_base_messages = store.chat_messages_for_next_turn(
                model_id="google/gemma-4-E2B-it",
                system_prompt="You are a concise, helpful assistant.",
            )
            store.record_chat_turn(
                model_id="google/gemma-4-E2B-it",
                status="ok",
                artifact_path=artifact_two,
                prompt="Second question",
                system_prompt="You are a concise, helpful assistant.",
                resolved_user_prompt="Second question",
                output_text="Second answer",
                base_messages=second_base_messages,
            )

            session = store.active_session("chat")

        self.assertIsNotNone(session)
        self.assertEqual(session["schema_name"], SESSION_SCHEMA_NAME)
        self.assertEqual(len(session["entries"]), 2)
        self.assertEqual(
            session["history_for_next_turn"],
            [
                {"role": "system", "content": "You are a concise, helpful assistant."},
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second question"},
                {"role": "assistant", "content": "Second answer"},
            ],
        )

    def test_record_session_run_tracks_attached_assets_and_artifact_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = WorkspaceSessionStore(root=root)
            artifact_path = root / "artifacts" / "vision" / "result.json"
            primary_input = root / "assets" / "images" / "sample.png"
            secondary_input = root / "assets" / "docs" / "sample.pdf"

            store.ensure_session(
                surface="vision",
                model_id="google/gemma-4-E2B-it",
                mode="pdf-summary",
            )
            store.record_session_run(
                surface="vision",
                model_id="google/gemma-4-E2B-it",
                mode="pdf-summary",
                artifact_kind="vision",
                artifact_path=artifact_path,
                status="ok",
                prompt="Summarize this document.",
                system_prompt="Summarize faithfully.",
                resolved_user_prompt="Summarize this document.",
                output_text="Summary",
                attachments=[
                    {"role": "primary_input", "path": primary_input},
                    {"role": "secondary_input", "path": secondary_input},
                ],
                options={"max_pages": 4},
            )

            session_path = session_manifest_path(
                session_id="vision-main",
                workspace_id=DEFAULT_WORKSPACE_ID,
                root=root,
            )
            session = read_session_manifest(session_path)
            workspace = read_workspace_manifest(store.manifest_path)

        self.assertEqual(session["attached_assets"][0]["kind"], "image")
        self.assertEqual(session["attached_assets"][1]["kind"], "pdf")
        self.assertEqual(session["artifact_refs"][0]["artifact_path"], str(artifact_path.resolve()))
        self.assertEqual(
            session["artifact_refs"][0]["artifact_workspace_relative_path"],
            "artifacts/vision/result.json",
        )
        self.assertEqual(session["entries"][0]["options"]["max_pages"], 4)
        self.assertEqual(workspace["sessions"][0]["latest_artifact_path"], str(artifact_path.resolve()))
        self.assertEqual(
            workspace["sessions"][0]["latest_artifact_workspace_relative_path"],
            "artifacts/vision/result.json",
        )
        self.assertEqual(workspace["sessions"][0]["latest_artifact_status"], "ok")


if __name__ == "__main__":
    unittest.main()
