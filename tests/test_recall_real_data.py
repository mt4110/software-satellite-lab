from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from artifact_schema import build_artifact_payload, build_prompt_record, build_runtime_record, write_artifact  # noqa: E402
from memory_index import rebuild_memory_index, MemoryIndex  # noqa: E402
from prepare_recall_real_data import (  # noqa: E402
    ADVERSARIAL_PASS_DEFINITION_VARIANT,
    BASELINE_REQUEST_VARIANT,
    build_real_recall_dataset,
)
from software_work_events import iter_capability_matrix_events  # noqa: E402
from workspace_state import WorkspaceSessionStore  # noqa: E402


class RecallRealDataTests(unittest.TestCase):
    def test_iter_capability_matrix_events_normalizes_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "artifacts" / "capability_matrix" / "matrix.json"
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
                        }
                    ]
                },
            )
            write_artifact(path, payload)

            events = iter_capability_matrix_events(root=root)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_kind"], "capability_result")
        self.assertEqual(events[0]["outcome"]["status"], "blocked")
        self.assertIn("Invalid buffer size", events[0]["content"]["output_text"])
        self.assertEqual(events[0]["session"]["surface"], "evaluation")

    def test_rebuild_memory_index_includes_capability_matrix_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "artifacts" / "capability_matrix" / "matrix.json"
            payload = build_artifact_payload(
                artifact_kind="capability_matrix",
                status="ok",
                runtime=build_runtime_record(backend="capability-matrix", model_id="backend-a"),
                prompts=build_prompt_record(),
                extra={
                    "results": [
                        {
                            "capability": "structured-json",
                            "phase": "phase1",
                            "status": "ok",
                            "artifact_kind": "text",
                            "artifact_path": str(root / "artifacts" / "text" / "structured.json"),
                            "validation_command": "python scripts/run_capability_matrix.py --only structured-json",
                            "claim_scope": "return a compact JSON object",
                            "output_preview": '{"capability":"structured-json"}',
                            "blocker": None,
                            "quality_status": "pass",
                            "quality_checks": [],
                            "quality_notes": [],
                            "notes": [],
                            "runtime_backend": "gemma-live-text",
                            "execution_status": "ok",
                            "validation_mode": "live",
                            "pass_definition": "structured output",
                            "preprocessing_lineage": [],
                        }
                    ]
                },
            )
            write_artifact(path, payload)

            summary = rebuild_memory_index(root=root)
            index = MemoryIndex(Path(summary["index_path"]))
            matches = index.search("structured")

        self.assertEqual(summary["capability_event_count"], 1)
        self.assertEqual(summary["event_count"], 1)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["event_kind"], "capability_result")
        self.assertEqual(matches[0]["session_surface"], "chat")

    def test_build_real_recall_dataset_writes_requests_and_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = WorkspaceSessionStore(root=root)
            messages = store.chat_messages_for_next_turn(
                model_id="backend-a",
                system_prompt="You are concise.",
            )
            store.record_chat_turn(
                model_id="backend-a",
                status="ok",
                artifact_path=root / "artifacts" / "text" / "chat.json",
                prompt="Review the memory index patch.",
                system_prompt="You are concise.",
                resolved_user_prompt="Review the memory index patch.",
                output_text="Looks good with one regression note.",
                base_messages=messages,
                notes=["review accepted"],
            )
            matrix_path = root / "artifacts" / "capability_matrix" / "matrix.json"
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
                        }
                    ]
                },
            )
            write_artifact(matrix_path, payload)

            dataset = build_real_recall_dataset(
                root=root,
                output_dir=root / "artifacts" / "recall_data" / "local-default",
                index_path=root / "tmp-memory-index.sqlite3",
                event_log_path=root / "tmp-event-log.jsonl",
                max_requests=4,
            )

            self.assertGreaterEqual(dataset["request_count"], 2)
            self.assertTrue(any(item["task_kind"] == "failure_analysis" for item in dataset["requests"]))
            self.assertTrue(all(Path(item["bundle_path"]).exists() for item in dataset["requests"]))
            self.assertTrue(all("source_rank" in item for item in dataset["requests"]))
            self.assertTrue(all("miss_reason" in item for item in dataset["requests"]))
            self.assertTrue(all("request_variant" in item for item in dataset["requests"]))

    def test_build_real_recall_dataset_keeps_repeated_prompt_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = WorkspaceSessionStore(root=root)
            for index in range(2):
                messages = store.chat_messages_for_next_turn(
                    model_id="backend-a",
                    system_prompt="You are concise.",
                )
                store.record_chat_turn(
                    model_id="backend-a",
                    status="ok",
                    artifact_path=root / "artifacts" / "text" / f"review-{index}.json",
                    prompt="Review the repeated release checklist.",
                    system_prompt="You are concise.",
                    resolved_user_prompt="Review the repeated release checklist.",
                    output_text=f"Repeated review outcome {index}.",
                    base_messages=messages,
                    notes=["review accepted"],
                )

            dataset = build_real_recall_dataset(
                root=root,
                output_dir=root / "artifacts" / "recall_data" / "local-default",
                index_path=root / "tmp-memory-index.sqlite3",
                event_log_path=root / "tmp-event-log.jsonl",
                max_requests=4,
                max_adversarial_requests=0,
            )

            baseline_requests = [
                item for item in dataset["requests"] if item["request_variant"] == BASELINE_REQUEST_VARIANT
            ]
            repeated_requests = [
                item
                for item in baseline_requests
                if item["query_text"] == "Review the repeated release checklist."
            ]

        self.assertEqual(len(repeated_requests), 2)
        self.assertEqual(len({item["source_event_id"] for item in repeated_requests}), 2)
        self.assertTrue(all(item["source_hit"] for item in repeated_requests))

    def test_build_real_recall_dataset_adds_adversarial_pass_definition_request_that_can_retrieve_source(self) -> None:
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

            store = WorkspaceSessionStore(root=root)
            for index in range(9):
                messages = store.chat_messages_for_next_turn(
                    model_id="backend-a",
                    system_prompt="You are concise.",
                )
                store.record_chat_turn(
                    model_id="backend-a",
                    status="ok",
                    artifact_path=root / "artifacts" / "text" / f"chat-{index}.json",
                    prompt=f"Decoy review prompt {index}",
                    system_prompt="You are concise.",
                    resolved_user_prompt=f"Decoy review prompt {index}",
                    output_text=f"Decoy output {index}",
                    base_messages=messages,
                    notes=["review accepted"],
                )

            dataset = build_real_recall_dataset(
                root=root,
                output_dir=root / "artifacts" / "recall_data" / "local-default",
                index_path=root / "tmp-memory-index.sqlite3",
                event_log_path=root / "tmp-event-log.jsonl",
                max_requests=12,
                max_adversarial_requests=1,
            )

            baseline = next(item for item in dataset["requests"] if item["request_variant"] == BASELINE_REQUEST_VARIANT and item["source_event_kind"] == "capability_result")
            adversarial = next(item for item in dataset["requests"] if item["request_variant"] == ADVERSARIAL_PASS_DEFINITION_VARIANT)

            self.assertTrue(baseline["source_hit"])
            self.assertEqual(adversarial["request_basis"], "pass_definition")
            self.assertTrue(adversarial["source_hit"])
            self.assertIsNone(adversarial["miss_reason"])

    def test_build_real_recall_dataset_marks_grouped_pass_definition_source_as_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            matrix_path = root / "artifacts" / "capability_matrix" / "matrix.json"
            shared_pass_definition = "Pass means execution completed and the output satisfied the capability-specific contract check."
            payload = build_artifact_payload(
                artifact_kind="capability_matrix",
                status="ok",
                runtime=build_runtime_record(backend="capability-matrix", model_id="backend-a"),
                prompts=build_prompt_record(),
                extra={
                    "results": [
                        {
                            "capability": "multilingual-translate",
                            "phase": "phase5",
                            "status": "ok",
                            "artifact_kind": "text",
                            "artifact_path": str(root / "artifacts" / "text" / "translate.json"),
                            "validation_command": "python scripts/run_capability_matrix.py --only multilingual-translate",
                            "claim_scope": "live model generation on a small local prompt",
                            "output_preview": "Translated output.",
                            "blocker": None,
                            "quality_status": "pass",
                            "quality_checks": [],
                            "quality_notes": [],
                            "notes": [],
                            "runtime_backend": "gemma-live-text",
                            "execution_status": "ok",
                            "validation_mode": "live",
                            "pass_definition": shared_pass_definition,
                            "preprocessing_lineage": [],
                        },
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
                            "pass_definition": shared_pass_definition,
                            "preprocessing_lineage": [],
                        },
                    ]
                },
            )
            write_artifact(matrix_path, payload)

            dataset = build_real_recall_dataset(
                root=root,
                output_dir=root / "artifacts" / "recall_data" / "local-default",
                index_path=root / "tmp-memory-index.sqlite3",
                event_log_path=root / "tmp-event-log.jsonl",
                max_requests=4,
                max_adversarial_requests=2,
            )

            adversarial_requests = [
                item
                for item in dataset["requests"]
                if item["request_variant"] == ADVERSARIAL_PASS_DEFINITION_VARIANT
            ]

            self.assertEqual(len(adversarial_requests), 2)
            self.assertTrue(all(item["source_hit"] for item in adversarial_requests))
            self.assertTrue(all(item["miss_reason"] is None for item in adversarial_requests))
            self.assertTrue(all(item["source_grouped_by"] == "pass_definition" for item in adversarial_requests))
            self.assertTrue(all(item["source_group_member_count"] == 2 for item in adversarial_requests))
            self.assertTrue(
                all(
                    {"multilingual-translate", "thinking"}.issubset(set(item["source_group_member_labels"]))
                    for item in adversarial_requests
                )
            )


if __name__ == "__main__":
    unittest.main()
