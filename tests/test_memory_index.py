from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from memory_index import MemoryIndex, rebuild_memory_index  # noqa: E402
from workspace_state import WorkspaceSessionStore  # noqa: E402


class MemoryIndexTests(unittest.TestCase):
    def test_rebuild_indexes_workspace_events_and_search_finds_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = WorkspaceSessionStore(root=root)

            first_messages = store.chat_messages_for_next_turn(
                model_id="backend-a",
                system_prompt="You are concise.",
            )
            store.record_chat_turn(
                model_id="backend-a",
                status="ok",
                artifact_path=root / "artifacts" / "text" / "first.json",
                prompt="Design a retrieval layer.",
                system_prompt="You are concise.",
                resolved_user_prompt="Design a retrieval layer.",
                output_text="Use SQLite FTS5 first.",
                base_messages=first_messages,
                notes=["design accepted"],
            )
            second_messages = store.chat_messages_for_next_turn(
                model_id="backend-a",
                system_prompt="You are concise.",
            )
            store.record_chat_turn(
                model_id="backend-a",
                status="ok",
                artifact_path=root / "artifacts" / "text" / "second.json",
                prompt="Review the memory index patch.",
                system_prompt="You are concise.",
                resolved_user_prompt="Review the memory index patch.",
                output_text="The search path looks solid.",
                base_messages=second_messages,
                notes=["review accepted"],
            )

            summary = rebuild_memory_index(root=root)
            index = MemoryIndex(Path(summary["index_path"]))
            matches = index.search("SQLite")
            latest = index.search(limit=1)

        self.assertEqual(summary["event_count"], 2)
        self.assertGreaterEqual(len(matches), 1)
        self.assertEqual(matches[0]["event_kind"], "chat_turn")
        self.assertIn("SQLite FTS5 first.", matches[0]["output_text"])
        self.assertEqual(latest[0]["prompt"], "Review the memory index patch.")

    def test_search_supports_surface_and_status_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = WorkspaceSessionStore(root=root)
            store.record_session_run(
                surface="vision",
                model_id="backend-a",
                mode="pdf-summary",
                artifact_kind="vision",
                artifact_path=root / "artifacts" / "vision" / "summary.json",
                status="quality_fail",
                prompt="Summarize this spec.",
                system_prompt="Summarize faithfully.",
                resolved_user_prompt="Summarize this spec.",
                output_text="Partial summary",
                notes=["repair needed"],
            )

            summary = rebuild_memory_index(root=root)
            index = MemoryIndex(Path(summary["index_path"]))
            matches = index.search("summary", surface="vision", status="quality_fail")

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["session_surface"], "vision")
        self.assertEqual(matches[0]["status"], "quality_fail")


if __name__ == "__main__":
    unittest.main()
