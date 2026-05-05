from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from artifact_schema import build_artifact_payload, build_prompt_record, build_runtime_record, write_artifact  # noqa: E402
from memory_index import MemoryIndex, rebuild_memory_index  # noqa: E402
from recall_context import RecallCandidate, build_context_bundle, normalize_recall_request, rank_candidates, retrieve_candidates  # noqa: E402
from workspace_state import WorkspaceSessionStore  # noqa: E402


def _materialize_workspace_artifacts(root: Path) -> None:
    sessions_root = root / "artifacts" / "workspaces" / "local-default" / "sessions"
    for session_path in sessions_root.glob("*.json"):
        payload = json.loads(session_path.read_text(encoding="utf-8"))
        for artifact_ref in payload.get("artifact_refs") or []:
            artifact_path = artifact_ref.get("artifact_path")
            if not isinstance(artifact_path, str) or not artifact_path.strip():
                continue
            target = Path(artifact_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_text("{}", encoding="utf-8")


class RecallContextTests(unittest.TestCase):
    class _StaticIndex:
        def __init__(self, *, query_rows: list[dict[str, object]], broad_rows: list[dict[str, object]]) -> None:
            self.query_rows = query_rows
            self.broad_rows = broad_rows
            self.rows_by_event_id = {
                str(row.get("event_id")): dict(row)
                for row in [*query_rows, *broad_rows]
                if row.get("event_id")
            }

        def search(
            self,
            query: str | None = None,
            *,
            limit: int = 10,
            surface: str | None = None,
            status: str | None = None,
        ) -> list[dict[str, object]]:
            rows = self.broad_rows if query is None else self.query_rows
            filtered = rows
            if surface is not None:
                filtered = [row for row in filtered if row.get("session_surface") == surface]
            if status is not None:
                filtered = [row for row in filtered if row.get("status") == status]
            return [dict(row) for row in filtered[:limit]]

        def get_event(self, event_id: str) -> dict[str, object] | None:
            row = self.rows_by_event_id.get(str(event_id))
            return dict(row) if row is not None else None

    class _QueryAwareIndex:
        def __init__(self, *, phrase_rows: list[dict[str, object]], token_rows: list[dict[str, object]], broad_rows: list[dict[str, object]]) -> None:
            self.phrase_rows = phrase_rows
            self.token_rows = token_rows
            self.broad_rows = broad_rows
            self.queries: list[str | None] = []
            self.rows_by_event_id = {
                str(row.get("event_id")): dict(row)
                for row in [*phrase_rows, *token_rows, *broad_rows]
                if row.get("event_id")
            }

        def search(
            self,
            query: str | None = None,
            *,
            limit: int = 10,
            surface: str | None = None,
            status: str | None = None,
        ) -> list[dict[str, object]]:
            self.queries.append(query)
            if query is None:
                rows = self.broad_rows
            elif query.startswith('"'):
                rows = self.phrase_rows
            else:
                rows = self.token_rows
            filtered = rows
            if surface is not None:
                filtered = [row for row in filtered if row.get("session_surface") == surface]
            if status is not None:
                filtered = [row for row in filtered if row.get("status") == status]
            return [dict(row) for row in filtered[:limit]]

        def get_event(self, event_id: str) -> dict[str, object] | None:
            row = self.rows_by_event_id.get(str(event_id))
            return dict(row) if row is not None else None

    def test_normalize_recall_request_cleans_defaults_and_filters(self) -> None:
        request = normalize_recall_request(
            {
                "task_kind": " Review ",
                "query_text": "  review memory index patch  ",
                "file_hints": [" scripts/memory_index.py ", "", "scripts/memory_index.py"],
                "surface_filters": [" Chat ", ""],
                "status_filters": [" ok ", "QUALITY_FAIL"],
                "limit": 0,
                "context_budget_chars": -10,
            }
        )

        self.assertEqual(request.task_kind, "review")
        self.assertEqual(request.query_text, "review memory index patch")
        self.assertEqual(request.file_hints, ("scripts/memory_index.py",))
        self.assertEqual(request.surface_filters, ("chat",))
        self.assertEqual(request.status_filters, ("ok", "quality_fail"))
        self.assertEqual(request.limit, 12)
        self.assertEqual(request.context_budget_chars, 6000)

    def test_rank_candidates_handles_mixed_timezone_timestamps(self) -> None:
        request = normalize_recall_request(
            {
                "task_kind": "review",
                "query_text": "memory review",
            }
        )
        older_aware = RecallCandidate(
            event_id="older-aware",
            recorded_at_utc="2026-04-11T10:00:00+00:00",
            session_id="s1",
            session_surface="chat",
            session_mode="review",
            model_id="backend-a",
            event_kind="chat_turn",
            status="ok",
            prompt="memory review",
            output_text="accepted outcome",
            notes_text="accepted",
            pass_definition=None,
            artifact_path=None,
            raw_fts_score=0.1,
            best_rank=1,
        )
        newer_naive = RecallCandidate(
            event_id="newer-naive",
            recorded_at_utc="2026-04-12T10:00:00",
            session_id="s2",
            session_surface="chat",
            session_mode="review",
            model_id="backend-a",
            event_kind="chat_turn",
            status="ok",
            prompt="memory review",
            output_text="accepted outcome",
            notes_text="accepted",
            pass_definition=None,
            artifact_path=None,
            raw_fts_score=0.1,
            best_rank=1,
        )

        ranked = rank_candidates(request, [older_aware, newer_naive])

        self.assertEqual(ranked[0].event_id, "newer-naive")
        self.assertIn("recent", ranked[0].reasons)

    def test_review_context_prioritizes_file_hits_and_accepted_outcomes(self) -> None:
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
                artifact_path=root / "artifacts" / "text" / "review.json",
                prompt="Review scripts/memory_index.py patch for regressions.",
                system_prompt="You are concise.",
                resolved_user_prompt="Review scripts/memory_index.py patch for regressions.",
                output_text="The path scoring looks steady and the regression risk is low.",
                base_messages=first_messages,
                notes=["review accepted", "fix verified"],
            )
            second_messages = store.chat_messages_for_next_turn(
                model_id="backend-a",
                system_prompt="You are concise.",
            )
            store.record_chat_turn(
                model_id="backend-a",
                status="ok",
                artifact_path=root / "artifacts" / "text" / "design.json",
                prompt="Design the retrieval layer boundary.",
                system_prompt="You are concise.",
                resolved_user_prompt="Design the retrieval layer boundary.",
                output_text="Keep SQLite FTS5 lexical-first and make ranking explainable.",
                base_messages=second_messages,
                notes=["design accepted", "decision recorded"],
            )
            store.record_session_run(
                surface="thinking",
                model_id="backend-a",
                mode="analysis",
                artifact_kind="text",
                artifact_path=root / "artifacts" / "text" / "risk.json",
                status="blocked",
                prompt="Investigate scripts/memory_index.py rollout risk.",
                system_prompt="Be precise.",
                resolved_user_prompt="Investigate scripts/memory_index.py rollout risk.",
                output_text="A stale index can hide the latest patch outcome.",
                notes=["blocked upstream dependency"],
            )

            _materialize_workspace_artifacts(root)
            summary = rebuild_memory_index(root=root)
            bundle = build_context_bundle(
                {
                    "task_kind": "review",
                    "query_text": "review memory index patch",
                    "file_hints": ["scripts/memory_index.py"],
                },
                root=root,
                index_path=Path(summary["index_path"]),
            )

        self.assertGreaterEqual(bundle["selected_count"], 2)
        self.assertIn("Review scripts/memory_index.py patch", bundle["selected_candidates"][0]["prompt_excerpt"])
        self.assertIn("file-match", bundle["selected_candidates"][0]["reasons"])
        self.assertIn("accepted-signal", bundle["selected_candidates"][0]["reasons"])
        self.assertIn("priority:review:source-artifact", bundle["selected_candidates"][0]["reasons"])
        self.assertIn("priority:review:accepted", bundle["selected_candidates"][0]["reasons"])
        self.assertEqual(bundle["blocks"][0]["title"], "Related files and artifact paths")

    def test_failure_analysis_prioritizes_failure_patterns(self) -> None:
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
                artifact_path=root / "artifacts" / "text" / "success.json",
                prompt="Review the memory index output.",
                system_prompt="You are concise.",
                resolved_user_prompt="Review the memory index output.",
                output_text="The happy path stays readable.",
                base_messages=first_messages,
                notes=["review accepted"],
            )
            store.record_session_run(
                surface="thinking",
                model_id="backend-a",
                mode="analysis",
                artifact_kind="text",
                artifact_path=root / "artifacts" / "text" / "failure.json",
                status="quality_fail",
                prompt="Investigate scripts/memory_index.py ranking regression.",
                system_prompt="Be precise.",
                resolved_user_prompt="Investigate scripts/memory_index.py ranking regression.",
                output_text="Failure analysis says the repair note was not surfaced.",
                notes=["repair needed", "failed review handoff"],
            )

            _materialize_workspace_artifacts(root)
            summary = rebuild_memory_index(root=root)
            bundle = build_context_bundle(
                {
                    "task_kind": "failure_analysis",
                    "query_text": "memory index regression",
                    "file_hints": ["scripts/memory_index.py"],
                },
                root=root,
                index_path=Path(summary["index_path"]),
            )

        self.assertGreaterEqual(bundle["selected_count"], 1)
        self.assertEqual(bundle["selected_candidates"][0]["status"], "quality_fail")
        self.assertEqual(bundle["blocks"][0]["title"], "Failure and repair patterns")
        self.assertIn("failure-signal", bundle["selected_candidates"][0]["reasons"])
        self.assertIn("priority:failure_analysis:test_fail", bundle["selected_candidates"][0]["reasons"])
        self.assertIn("priority:failure_analysis:repair", bundle["selected_candidates"][0]["reasons"])

    def test_rejected_evidence_does_not_pick_up_accepted_priority_from_not_accepted_text(self) -> None:
        request = normalize_recall_request(
            {
                "task_kind": "design",
                "query_text": "design tradeoff rejection",
            }
        )
        rejected = RecallCandidate(
            event_id="rejected",
            recorded_at_utc="2026-04-12T10:00:00+00:00",
            session_id="chat-main",
            session_surface="chat",
            session_mode="design",
            model_id="backend-a",
            event_kind="design_proposal",
            status="neutral",
            prompt="Design tradeoff rejection note",
            output_text="This option was not accepted after review.",
            notes_text="rejected tradeoff",
            pass_definition=None,
            artifact_path="artifacts/design/rejected.json",
        )

        ranked = rank_candidates(request, [rejected])

        self.assertIn("priority:design:rejected", ranked[0].reasons)
        self.assertNotIn("priority:design:accepted", ranked[0].reasons)

    def test_context_budget_trims_excess_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = WorkspaceSessionStore(root=root)
            for index in range(5):
                messages = store.chat_messages_for_next_turn(
                    model_id="backend-a",
                    system_prompt="You are concise.",
                )
                store.record_chat_turn(
                    model_id="backend-a",
                    status="ok",
                    artifact_path=root / "artifacts" / "text" / f"proposal-{index}.json",
                    prompt=f"Proposal {index} for the memory index rollout.",
                    system_prompt="You are concise.",
                    resolved_user_prompt=f"Proposal {index} for the memory index rollout.",
                    output_text=" ".join(["Use a compact rollout note."] * 18),
                    base_messages=messages,
                    notes=["proposal accepted", "test pass"],
                )

            _materialize_workspace_artifacts(root)
            summary = rebuild_memory_index(root=root)
            bundle = build_context_bundle(
                {
                    "task_kind": "proposal",
                    "query_text": "memory index rollout proposal",
                    "limit": 5,
                    "context_budget_chars": 650,
                },
                root=root,
                index_path=Path(summary["index_path"]),
            )

        self.assertLessEqual(bundle["budget"]["used_chars"], 650)
        self.assertGreater(bundle["omitted_count"], 0)
        self.assertLess(bundle["selected_count"], 5)

    def test_rank_candidates_prefers_exact_query_match_from_broad_search(self) -> None:
        request = normalize_recall_request(
            {
                "task_kind": "proposal",
                "query_text": "Source hit false request miss report",
            }
        )
        source = RecallCandidate(
            event_id="source",
            recorded_at_utc="2026-04-12T10:00:00+00:00",
            session_id="chat-main",
            session_surface="chat",
            session_mode="analysis",
            model_id="backend-a",
            event_kind="chat_turn",
            status="ok",
            prompt="Source hit false request miss report",
            output_text="Classify why the source candidate fell out of recall ranking.",
            notes_text="proposal accepted",
            pass_definition=None,
            artifact_path="artifacts/recall/source.json",
        )
        distractor = RecallCandidate(
            event_id="distractor",
            recorded_at_utc="2026-04-13T10:00:00+00:00",
            session_id="chat-main",
            session_surface="chat",
            session_mode="analysis",
            model_id="backend-a",
            event_kind="chat_turn",
            status="ok",
            prompt="Proposal for recall context ranking rollout",
            output_text="The proposal accepted the ranking patch and review follow-up.",
            notes_text="review accepted",
            pass_definition=None,
            artifact_path="artifacts/recall/rollout.json",
            raw_fts_score=0.1,
            best_rank=0,
            query_hits=1,
        )

        ranked = rank_candidates(request, [distractor, source])

        self.assertEqual(ranked[0].event_id, "source")
        self.assertIn("exact-query-match", ranked[0].reasons)
        self.assertIn("query-coverage", ranked[0].reasons)

    def test_rank_candidates_uses_pass_definition_for_query_alignment(self) -> None:
        request = normalize_recall_request(
            {
                "task_kind": "proposal",
                "query_text": "Opaque scratchpad seal uncompromised verdict retention.",
            }
        )
        source = RecallCandidate(
            event_id="source",
            recorded_at_utc="2026-04-12T10:00:00+00:00",
            session_id="thinking-main",
            session_surface="evaluation",
            session_mode="phase5",
            model_id="backend-a",
            event_kind="capability_result",
            status="ok",
            prompt="thinking | live model generation on a small local prompt | python scripts/run_capability_matrix.py --only thinking",
            output_text="Use breadth-first search when you need the shortest path.",
            notes_text="quality_status: pass",
            pass_definition="Opaque scratchpad seal uncompromised verdict retention.",
            artifact_path="artifacts/thinking/source.json",
        )
        distractor = RecallCandidate(
            event_id="distractor",
            recorded_at_utc="2026-04-13T10:00:00+00:00",
            session_id="chat-main",
            session_surface="chat",
            session_mode="analysis",
            model_id="backend-a",
            event_kind="chat_turn",
            status="ok",
            prompt="Proposal for recall context ranking rollout",
            output_text="The proposal accepted the ranking patch and review follow-up.",
            notes_text="review accepted",
            pass_definition=None,
            artifact_path="artifacts/recall/rollout.json",
            raw_fts_score=0.1,
            best_rank=0,
            query_hits=1,
        )

        ranked = rank_candidates(request, [distractor, source])

        self.assertEqual(ranked[0].event_id, "source")
        self.assertIn("exact-query-match", ranked[0].reasons)
        self.assertIn("query-coverage", ranked[0].reasons)

    def test_retrieve_candidates_uses_phrase_query_first_for_pass_definition_requests(self) -> None:
        request = normalize_recall_request(
            {
                "task_kind": "proposal",
                "query_text": "Opaque scratchpad seal uncompromised verdict retention.",
                "request_basis": "pass_definition",
                "limit": 4,
            }
        )
        source = {
            "event_id": "source",
            "recorded_at_utc": "2026-04-12T10:00:00+00:00",
            "session_surface": "evaluation",
            "session_mode": "phase5",
            "model_id": "backend-a",
            "event_kind": "capability_result",
            "status": "ok",
            "prompt": "thinking | live model generation on a small local prompt | python scripts/run_capability_matrix.py --only thinking",
            "output_text": "Use breadth-first search when you need the shortest path.",
            "notes_text": "quality_status: pass",
            "pass_definition": "Opaque scratchpad seal uncompromised verdict retention.",
            "artifact_path": "artifacts/thinking/source.json",
            "score": 0.01,
        }
        phrase_rows = [
            {
                "event_id": f"distractor-{index}",
                "recorded_at_utc": f"2026-04-13T10:{index:02d}:00+00:00",
                "session_surface": "evaluation",
                "session_mode": "phase5",
                "model_id": "backend-a",
                "event_kind": "capability_result",
                "status": "ok",
                "prompt": f"text-chat {index} | live model generation on a small local prompt | python scripts/run_capability_matrix.py --only text-chat",
                "output_text": "Generic capability contract result.",
                "notes_text": "quality_status: pass",
                "pass_definition": "Opaque scratchpad seal uncompromised verdict retention.",
                "artifact_path": f"artifacts/text/distractor-{index}.json",
                "score": 0.1 + index,
            }
            for index in range(12)
        ]
        token_rows = [
            {
                "event_id": "token-distractor",
                "recorded_at_utc": "2026-04-13T11:00:00+00:00",
                "session_surface": "evaluation",
                "session_mode": "phase5",
                "model_id": "backend-a",
                "event_kind": "capability_result",
                "status": "ok",
                "prompt": "text-chat | live model generation on a small local prompt | python scripts/run_capability_matrix.py --only text-chat",
                "output_text": "Generic capability contract result.",
                "notes_text": "quality_status: pass",
                "pass_definition": "Pass means execution completed and the output satisfied the capability-specific contract check.",
                "artifact_path": "artifacts/text/token-distractor.json",
                "score": 0.1,
            }
        ]
        index = self._QueryAwareIndex(
            phrase_rows=[*phrase_rows, source],
            token_rows=token_rows,
            broad_rows=token_rows,
        )

        candidates = retrieve_candidates(request, index=index)

        self.assertEqual(index.queries[0], '"Opaque scratchpad seal uncompromised verdict retention."')
        self.assertNotIn(None, index.queries)
        self.assertGreaterEqual(len(candidates), 13)
        self.assertIn("source", {candidate.event_id for candidate in candidates})

    def test_build_context_bundle_groups_same_pass_definition_and_counts_source_hit(self) -> None:
        shared_pass_definition = "Pass means execution completed and the output satisfied the capability-specific contract check."
        leader = {
            "event_id": "leader",
            "recorded_at_utc": "2026-04-13T10:00:00+00:00",
            "session_surface": "evaluation",
            "session_mode": "phase5",
            "model_id": "backend-a",
            "event_kind": "capability_result",
            "status": "ok",
            "prompt": "multilingual-translate | live model generation on a small local prompt | python scripts/run_capability_matrix.py --only multilingual-translate",
            "output_text": "Translated output.",
            "notes_text": "quality_status: pass",
            "pass_definition": shared_pass_definition,
            "artifact_path": "artifacts/capability/leader.json",
            "score": 0.01,
        }
        source = {
            "event_id": "source",
            "recorded_at_utc": "2026-04-12T10:00:00+00:00",
            "session_surface": "evaluation",
            "session_mode": "phase5",
            "model_id": "backend-a",
            "event_kind": "capability_result",
            "status": "ok",
            "prompt": "thinking | live model generation on a small local prompt | python scripts/run_capability_matrix.py --only thinking",
            "output_text": "Use breadth-first search when you need the shortest path.",
            "notes_text": "quality_status: pass",
            "pass_definition": shared_pass_definition,
            "artifact_path": "artifacts/capability/source.json",
            "score": 0.2,
        }
        third = {
            "event_id": "third",
            "recorded_at_utc": "2026-04-11T10:00:00+00:00",
            "session_surface": "evaluation",
            "session_mode": "phase5",
            "model_id": "backend-a",
            "event_kind": "capability_result",
            "status": "ok",
            "prompt": "structured-json | live model generation on a small local prompt | python scripts/run_capability_matrix.py --only structured-json",
            "output_text": "{\"answer\":true}",
            "notes_text": "quality_status: pass",
            "pass_definition": shared_pass_definition,
            "artifact_path": "artifacts/capability/third.json",
            "score": 0.3,
        }

        bundle = build_context_bundle(
            {
                "task_kind": "proposal",
                "query_text": shared_pass_definition,
                "request_basis": "pass_definition",
                "limit": 1,
                "context_budget_chars": 1800,
                "source_event_id": "third",
            },
            index=self._StaticIndex(query_rows=[leader, source, third], broad_rows=[leader, source, third]),
        )

        self.assertEqual(bundle["selected_count"], 1)
        self.assertEqual(bundle["budget"]["context_budget_chars"], 1800)
        self.assertGreater(bundle["budget"]["effective_context_budget_chars"], 1800)
        self.assertEqual(bundle["selected_candidates"][0]["grouped_by"], "pass_definition")
        self.assertEqual(bundle["selected_candidates"][0]["group_member_count"], 3)
        self.assertIn("source", bundle["selected_candidates"][0]["group_member_event_ids"])
        self.assertIn("third", bundle["selected_candidates"][0]["group_member_event_ids"])
        self.assertNotIn("group_member_event_ids", bundle["blocks"][0]["items"][0])
        self.assertNotIn("artifact_path", bundle["blocks"][0]["items"][0])
        self.assertTrue(bundle["source_evaluation"]["source_selected"])
        self.assertTrue(bundle["source_evaluation"]["source_selected_via_group"])
        self.assertEqual(bundle["source_evaluation"]["source_grouped_by"], "pass_definition")
        self.assertEqual(bundle["source_evaluation"]["source_group_event_id"], "source")
        self.assertEqual(bundle["source_evaluation"]["source_group_member_count"], 3)
        self.assertIn("third", bundle["source_evaluation"]["source_group_member_event_ids"])
        self.assertTrue(
            any("structured-json" in label for label in bundle["source_evaluation"]["source_group_member_labels"])
        )

    def test_pass_definition_group_excludes_member_with_broken_source_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shared_pass_definition = "Pass means execution completed and the output satisfied the capability-specific contract check."
            leader_artifact = root / "artifacts" / "capability" / "leader.json"
            third_artifact = root / "artifacts" / "capability" / "third.json"
            leader_artifact.parent.mkdir(parents=True, exist_ok=True)
            leader_artifact.write_text("{}", encoding="utf-8")
            third_artifact.write_text("{}", encoding="utf-8")

            def row(event_id: str, artifact_path: Path, *, score: float) -> dict[str, object]:
                payload = {
                    "event_id": event_id,
                    "event_kind": "capability_result",
                    "outcome": {"status": "ok"},
                    "source_refs": {
                        "artifact_ref": {
                            "artifact_path": str(artifact_path),
                        }
                    },
                }
                return {
                    "event_id": event_id,
                    "recorded_at_utc": "2026-04-12T10:00:00+00:00",
                    "session_surface": "evaluation",
                    "session_mode": "phase5",
                    "model_id": "backend-a",
                    "event_kind": "capability_result",
                    "status": "ok",
                    "prompt": f"{event_id} | python scripts/run_capability_matrix.py --only {event_id}",
                    "output_text": "Capability output.",
                    "notes_text": "quality_status: pass",
                    "pass_definition": shared_pass_definition,
                    "artifact_path": str(artifact_path),
                    "payload_json": json.dumps(payload),
                    "score": score,
                }

            bundle = build_context_bundle(
                {
                    "task_kind": "proposal",
                    "query_text": shared_pass_definition,
                    "request_basis": "pass_definition",
                    "limit": 1,
                    "context_budget_chars": 1800,
                    "source_event_id": "source",
                },
                root=root,
                index=self._StaticIndex(
                    query_rows=[
                        row("leader", leader_artifact, score=0.01),
                        row("source", root / "artifacts" / "capability" / "missing.json", score=0.02),
                        row("third", third_artifact, score=0.03),
                    ],
                    broad_rows=[],
                ),
            )

        self.assertEqual(bundle["selected_candidates"][0]["grouped_by"], "pass_definition")
        self.assertEqual(bundle["selected_candidates"][0]["group_member_count"], 2)
        self.assertNotIn("source", bundle["selected_candidates"][0]["group_member_event_ids"])
        self.assertFalse(bundle["source_evaluation"]["source_selected"])
        self.assertEqual(bundle["source_evaluation"]["miss_reason"], "source_event_contract_broken")

    def test_build_context_bundle_classifies_source_miss_when_limit_is_tight(self) -> None:
        leader = {
            "event_id": "leader",
            "recorded_at_utc": "2026-04-13T10:00:00+00:00",
            "session_id": "chat-main",
            "session_surface": "chat",
            "session_mode": "analysis",
            "model_id": "backend-a",
            "event_kind": "chat_turn",
            "status": "ok",
            "prompt": "Review recall ranking patch",
            "output_text": "Accepted review outcome for the ranking patch.",
            "notes_text": "review accepted",
            "pass_definition": None,
            "artifact_path": "artifacts/review/leader.json",
            "score": 0.1,
        }
        source = {
            "event_id": "source",
            "recorded_at_utc": "2026-04-12T10:00:00+00:00",
            "session_id": "chat-main",
            "session_surface": "chat",
            "session_mode": "analysis",
            "model_id": "backend-a",
            "event_kind": "chat_turn",
            "status": "ok",
            "prompt": "Memory index follow-up notes",
            "output_text": "Keep the miss report readable.",
            "notes_text": "follow-up queued",
            "pass_definition": None,
            "artifact_path": "artifacts/review/source.json",
            "score": None,
        }
        bundle = build_context_bundle(
            {
                "task_kind": "review",
                "query_text": "review recall ranking",
                "limit": 1,
                "context_budget_chars": 1800,
                "source_event_id": "source",
            },
            index=self._StaticIndex(query_rows=[leader], broad_rows=[leader, source]),
        )

        self.assertEqual(bundle["selected_candidates"][0]["event_id"], "leader")
        self.assertEqual(bundle["source_evaluation"]["source_selected"], False)
        self.assertEqual(bundle["source_evaluation"]["source_rank"], 2)
        self.assertEqual(bundle["source_evaluation"]["miss_reason"], "ranked_out_by_limit")
        self.assertFalse(bundle["source_evaluation"]["source_selected_via_group"])
        self.assertIsNone(bundle["source_evaluation"]["source_group_member_count"])
        self.assertIsNone(bundle["source_evaluation"]["source_grouped_by"])
        self.assertEqual(bundle["source_evaluation"]["source_group_member_labels"], [])
        self.assertEqual(bundle["selected_candidates"][0]["session_id"], "chat-main")
        self.assertEqual(bundle["source_evaluation"]["top_selected"][0]["session_id"], "chat-main")
        self.assertEqual(bundle["source_evaluation"]["top_selected"][0]["artifact_path"], "artifacts/review/leader.json")
        self.assertIn("fts-hit", bundle["source_evaluation"]["top_selected"][0]["reasons"])

    def test_source_miss_diagnostics_include_evidence_type_mismatch(self) -> None:
        leader = {
            "event_id": "leader",
            "recorded_at_utc": "2026-04-13T10:00:00+00:00",
            "session_surface": "chat",
            "session_mode": "analysis",
            "model_id": "backend-a",
            "event_kind": "chat_turn",
            "status": "ok",
            "prompt": "Review recall ranking patch",
            "output_text": "Accepted review outcome for the ranking patch.",
            "notes_text": "review accepted",
            "pass_definition": None,
            "artifact_path": "artifacts/review/leader.json",
            "score": 0.1,
        }
        source = {
            "event_id": "source",
            "recorded_at_utc": "2026-04-12T10:00:00+00:00",
            "session_surface": "chat",
            "session_mode": "analysis",
            "model_id": "backend-a",
            "event_kind": "chat_turn",
            "status": "neutral",
            "prompt": "Reference-only artifact path",
            "output_text": "Only a plain artifact pointer.",
            "notes_text": "reference only",
            "pass_definition": None,
            "artifact_path": "artifacts/review/source.json",
            "score": None,
        }

        bundle = build_context_bundle(
            {
                "task_kind": "review",
                "query_text": "review recall ranking",
                "limit": 1,
                "context_budget_chars": 1800,
                "source_event_id": "source",
            },
            index=self._StaticIndex(query_rows=[leader], broad_rows=[leader, source]),
        )

        self.assertEqual(bundle["source_evaluation"]["miss_reason"], "ranked_out_by_limit")
        self.assertFalse(bundle["source_evaluation"]["source_evidence_type_match"])
        self.assertIn("evidence_type_mismatch", bundle["source_evaluation"]["miss_diagnostics"])

    def test_build_context_bundle_injects_pinned_candidate_without_promoting_it(self) -> None:
        leader = {
            "event_id": "leader",
            "recorded_at_utc": "2026-04-13T10:00:00+00:00",
            "session_surface": "chat",
            "session_mode": "analysis",
            "model_id": "backend-a",
            "event_kind": "chat_turn",
            "status": "ok",
            "prompt": "Review recall ranking patch",
            "output_text": "Accepted review outcome for the ranking patch.",
            "notes_text": "review accepted",
            "pass_definition": None,
            "artifact_path": "artifacts/review/leader.json",
            "score": 0.1,
        }
        pinned = {
            "event_id": "pinned",
            "recorded_at_utc": "2026-04-12T10:00:00+00:00",
            "session_surface": "chat",
            "session_mode": "analysis",
            "model_id": "backend-a",
            "event_kind": "chat_turn",
            "status": "ok",
            "prompt": "Memory miss report notes",
            "output_text": "Track why source_hit=false requests fell out of the bundle.",
            "notes_text": "manual pin target",
            "pass_definition": None,
            "artifact_path": "artifacts/review/pinned.json",
            "score": None,
        }

        bundle = build_context_bundle(
            {
                "task_kind": "review",
                "query_text": "review recall ranking",
                "pinned_event_ids": ["pinned"],
                "limit": 1,
                "context_budget_chars": 1800,
                "source_event_id": "pinned",
            },
            index=self._StaticIndex(query_rows=[leader], broad_rows=[leader, pinned]),
        )

        self.assertEqual(bundle["selected_candidates"][0]["event_id"], "leader")
        self.assertFalse(bundle["source_evaluation"]["source_selected"])
        self.assertEqual(bundle["source_evaluation"]["source_rank"], 2)
        self.assertEqual(bundle["source_evaluation"]["miss_reason"], "ranked_out_by_limit")
        self.assertIn("pinned", bundle["source_evaluation"]["source_reasons"])
        self.assertEqual(bundle["pinned_event_ids"], ["pinned"])

    def test_build_context_bundle_marks_missing_source_when_event_left_the_index(self) -> None:
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
                artifact_path=root / "artifacts" / "text" / "review.json",
                prompt="Review the memory index patch.",
                system_prompt="You are concise.",
                resolved_user_prompt="Review the memory index patch.",
                output_text="Looks good with one regression note.",
                base_messages=messages,
                notes=["review accepted"],
            )
            _materialize_workspace_artifacts(root)
            summary = rebuild_memory_index(root=root)
            bundle = build_context_bundle(
                {
                    "task_kind": "review",
                    "query_text": "review the memory index patch",
                    "source_event_id": "missing-event",
                },
                root=root,
                index_path=Path(summary["index_path"]),
            )

        self.assertEqual(bundle["source_evaluation"]["source_selected"], False)
        self.assertEqual(bundle["source_evaluation"]["miss_reason"], "source_missing_from_index")
        self.assertFalse(bundle["source_evaluation"]["source_selected_via_group"])
        self.assertIsNone(bundle["source_evaluation"]["source_group_member_count"])
        self.assertEqual(bundle["source_evaluation"]["source_group_member_event_ids"], [])

    def test_build_context_bundle_does_not_select_event_with_missing_source_artifact(self) -> None:
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
                artifact_path=root / "artifacts" / "text" / "missing-review.json",
                prompt="Review the missing source artifact patch.",
                system_prompt="You are concise.",
                resolved_user_prompt="Review the missing source artifact patch.",
                output_text="The review would be useful, but the artifact is gone.",
                base_messages=messages,
                notes=["review accepted"],
            )
            summary = rebuild_memory_index(root=root)
            index = MemoryIndex(Path(summary["index_path"]))
            source_event_id = index.search("missing AND source AND artifact", limit=1)[0]["event_id"]
            bundle = build_context_bundle(
                {
                    "task_kind": "review",
                    "query_text": "Review the missing source artifact patch.",
                    "source_event_id": source_event_id,
                },
                root=root,
                index_path=Path(summary["index_path"]),
            )

        self.assertEqual(bundle["selected_count"], 0)
        self.assertFalse(bundle["source_evaluation"]["source_selected"])
        self.assertEqual(bundle["source_evaluation"]["miss_reason"], "source_event_contract_broken")
        self.assertEqual(bundle["source_evaluation"]["source_event_contract_status"], "missing_source")
        self.assertIn("source_artifact_missing", bundle["source_evaluation"]["source_artifact_reasons"])

    def test_pipe_delimited_capability_query_prefers_matching_capability_with_shared_file_hint(self) -> None:
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
                            "artifact_kind": None,
                            "artifact_path": None,
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
                            "pass_definition": "thinking output is present",
                            "preprocessing_lineage": [],
                        },
                        {
                            "capability": "function-calling",
                            "phase": "phase5",
                            "status": "ok",
                            "artifact_kind": None,
                            "artifact_path": None,
                            "validation_command": "python scripts/run_capability_matrix.py --only function-calling",
                            "claim_scope": "live model generation on a small local prompt",
                            "output_preview": "The calibration code for sensor-7 is CAL-7Q4-ALPHA.",
                            "blocker": None,
                            "quality_status": "pass",
                            "quality_checks": [],
                            "quality_notes": [],
                            "notes": [],
                            "runtime_backend": "gemma-live-text",
                            "execution_status": "ok",
                            "validation_mode": "live",
                            "pass_definition": "function call result is present",
                            "preprocessing_lineage": [],
                        },
                    ]
                },
            )
            write_artifact(matrix_path, payload)
            _materialize_workspace_artifacts(root)
            summary = rebuild_memory_index(root=root)
            request = normalize_recall_request(
                {
                    "task_kind": "proposal",
                    "query_text": "function-calling | live model generation on a small local prompt | python scripts/run_capability_matrix.py --only function-calling",
                    "file_hints": ["artifacts/capability_matrix/matrix.json"],
                    "limit": 3,
                    "context_budget_chars": 1800,
                }
            )
            ranked = rank_candidates(
                request,
                retrieve_candidates(
                    request,
                    root=root,
                    index_path=Path(summary["index_path"]),
                ),
            )

        self.assertGreaterEqual(len(ranked), 2)
        self.assertIn("function-calling", ranked[0].prompt_excerpt())
        self.assertIn("query-head-match", ranked[0].reasons)
        self.assertIn("thinking", ranked[1].prompt_excerpt())
        self.assertIn("query-head-mismatch", ranked[1].reasons)
        self.assertIn("validation-command-mismatch", ranked[1].reasons)


if __name__ == "__main__":
    unittest.main()
