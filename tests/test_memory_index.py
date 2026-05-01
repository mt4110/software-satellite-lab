from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from artifact_schema import build_artifact_payload, build_prompt_record, build_runtime_record, write_artifact  # noqa: E402
from memory_index import INDEX_SCHEMA_VERSION, MemoryIndex, rebuild_memory_index  # noqa: E402
from software_work_events import read_event_log  # noqa: E402
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

    def test_search_indexes_pass_definition_for_capability_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            matrix_path = root / "artifacts" / "capability_matrix" / "matrix.json"
            payload = build_artifact_payload(
                artifact_kind="capability_matrix",
                status="ok",
                runtime=build_runtime_record(backend="capability-matrix", model_id="backend-a"),
                prompts=build_prompt_record(),
                extra={
                    "results": [
                        {
                            "capability": "thinking",
                            "phase": "phase5",
                            "status": "ok",
                            "artifact_kind": "thinking",
                            "artifact_path": str(root / "artifacts" / "thinking" / "thinking.json"),
                            "validation_command": "python scripts/run_capability_matrix.py --only thinking",
                            "claim_scope": "live model generation on a small local prompt",
                            "output_preview": "Use breadth-first search when you need the shortest path.",
                            "blocker": None,
                            "quality_status": "pass",
                            "quality_checks": [],
                            "quality_notes": [],
                            "notes": [],
                            "runtime_backend": "gemma-live-thinking",
                            "execution_status": "ok",
                            "validation_mode": "live",
                            "pass_definition": "Opaque scratchpad seal uncompromised verdict retention.",
                            "preprocessing_lineage": [],
                        }
                    ]
                },
            )
            write_artifact(matrix_path, payload)

            summary = rebuild_memory_index(root=root)
            index = MemoryIndex(Path(summary["index_path"]))
            matches = index.search("Opaque OR scratchpad OR retention")
            event_log = read_event_log(Path(summary["event_log_path"]))

        self.assertEqual(len(matches), 1)
        self.assertEqual(event_log["event_count"], 1)
        self.assertEqual(event_log["events"][0]["event_kind"], "capability_result")
        self.assertEqual(matches[0]["event_kind"], "capability_result")
        self.assertEqual(
            matches[0]["pass_definition"],
            "Opaque scratchpad seal uncompromised verdict retention.",
        )

    def test_rebuild_keeps_duplicate_capability_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            matrix_path = root / "artifacts" / "capability_matrix" / "matrix.json"
            payload = build_artifact_payload(
                artifact_kind="capability_matrix",
                status="ok",
                runtime=build_runtime_record(backend="capability-matrix", model_id="backend-a"),
                prompts=build_prompt_record(),
                extra={
                    "results": [
                        {
                            "capability": "thinking",
                            "phase": "phase5",
                            "status": "blocked",
                            "artifact_kind": "thinking",
                            "artifact_path": str(root / "artifacts" / "thinking" / "thinking-blocked.json"),
                            "validation_command": "python scripts/run_capability_matrix.py --only thinking",
                            "claim_scope": "first thinking attempt",
                            "output_preview": "First attempt blocked.",
                            "blocker": None,
                            "quality_status": "not_run",
                            "quality_checks": [],
                            "quality_notes": [],
                            "notes": [],
                            "runtime_backend": "backend-a",
                            "execution_status": "blocked",
                            "validation_mode": "live",
                            "pass_definition": "First row should stay indexed.",
                            "preprocessing_lineage": [],
                        },
                        {
                            "capability": "thinking",
                            "phase": "phase5-retry",
                            "status": "ok",
                            "artifact_kind": "thinking",
                            "artifact_path": str(root / "artifacts" / "thinking" / "thinking-retry.json"),
                            "validation_command": "python scripts/run_capability_matrix.py --only thinking",
                            "claim_scope": "second thinking attempt",
                            "output_preview": "Retry succeeded.",
                            "blocker": None,
                            "quality_status": "pass",
                            "quality_checks": [],
                            "quality_notes": [],
                            "notes": [],
                            "runtime_backend": "backend-a",
                            "execution_status": "ok",
                            "validation_mode": "live",
                            "pass_definition": "Second row should stay indexed.",
                            "preprocessing_lineage": [],
                        },
                    ]
                },
            )
            write_artifact(matrix_path, payload)

            summary = rebuild_memory_index(root=root)
            index = MemoryIndex(Path(summary["index_path"]))
            matches = index.search("thinking", limit=10)
            event_log = read_event_log(Path(summary["event_log_path"]))

        self.assertEqual(summary["capability_event_count"], 2)
        self.assertEqual(summary["indexed_count"], 2)
        self.assertEqual(event_log["event_count"], 2)
        self.assertEqual(len(matches), 2)
        self.assertEqual(len({match["event_id"] for match in matches}), 2)

    def test_rebuild_reports_invalid_capability_matrix_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            broken_path = root / "artifacts" / "capability_matrix" / "broken.json"
            broken_path.parent.mkdir(parents=True, exist_ok=True)
            broken_path.write_text("{not-json", encoding="utf-8")

            summary = rebuild_memory_index(root=root)

        self.assertEqual(summary["capability_matrix_error_count"], 1)
        self.assertEqual(summary["capability_matrix_errors"][0]["path"], str(broken_path.resolve()))
        self.assertIn("JSONDecodeError", summary["capability_matrix_errors"][0]["error"])

    def test_schema_version_change_rebuilds_derived_index_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.sqlite3"
            stale_schema_version = str(INDEX_SCHEMA_VERSION - 1)
            connection = sqlite3.connect(path)
            try:
                connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                    (stale_schema_version,),
                )
                connection.execute(
                    """
                    CREATE TABLE events (
                        event_id TEXT PRIMARY KEY,
                        payload_json TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO events(event_id, payload_json)
                    VALUES ('stale-event', '{}')
                    """
                )
                connection.execute(
                    "CREATE VIRTUAL TABLE events_fts USING fts5(event_id UNINDEXED, prompt)"
                )
                connection.commit()
            finally:
                connection.close()

            index = MemoryIndex(path)
            index.ensure_schema()
            with index.connection() as connection:
                version = connection.execute(
                    "SELECT value FROM meta WHERE key = 'schema_version'"
                ).fetchone()["value"]
                columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(events)").fetchall()
                }
                count_row = connection.execute("SELECT COUNT(*) AS count FROM events").fetchone()
                event_count = count_row["count"]
            indexed_count = index.index_events(
                [
                    {
                        "event_id": "local-default:test-schema-rebuild",
                        "recorded_at_utc": "2026-01-01T00:00:00Z",
                        "workspace": {"workspace_id": "local-default"},
                        "session": {
                            "session_id": "test-session",
                            "surface": "chat",
                            "mode": "unit",
                            "selected_model_id": "backend-a",
                        },
                        "outcome": {
                            "status": "ok",
                            "quality_status": "pass",
                            "execution_status": "ok",
                        },
                        "content": {
                            "prompt": "Check schema rebuild.",
                            "output_text": "Schema rebuild accepted.",
                            "notes": [],
                            "options": {
                                "validation_mode": "unit",
                                "quality_checks": [
                                    {"name": "compile", "pass": True, "detail": "ok"}
                                ],
                            },
                        },
                        "source_refs": {},
                    }
                ]
            )
            matches = index.search("compile AND pass")

        self.assertEqual(version, str(INDEX_SCHEMA_VERSION))
        self.assertEqual(event_count, 0)
        self.assertEqual(indexed_count, 1)
        self.assertIn("quality_status", columns)
        self.assertIn("execution_status", columns)
        self.assertIn("evaluation_signal_text", columns)
        self.assertEqual(matches[0]["event_id"], "local-default:test-schema-rebuild")
        self.assertEqual(matches[0]["quality_status"], "pass")


if __name__ == "__main__":
    unittest.main()
