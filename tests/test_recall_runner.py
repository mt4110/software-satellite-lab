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

from run_recall_demo import (  # noqa: E402
    build_pinned_event_compare,
    build_bundle_report,
    compare_evaluation_summaries,
    dataset_request_to_bundle_request,
    ensure_recall_dataset,
    format_evaluation_report,
    format_miss_report,
    format_pinned_event_compare_report,
    format_request_catalog,
    load_latest_evaluation_summary,
    main,
    record_evaluation_summary,
    select_dataset_request,
)
from memory_index import MemoryIndex, default_memory_index_path, rebuild_memory_index  # noqa: E402
from recall_context import CONTEXT_BUNDLE_VERSION  # noqa: E402
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


class RecallRunnerTests(unittest.TestCase):
    def test_dataset_request_to_bundle_request_keeps_recall_fields(self) -> None:
        payload = dataset_request_to_bundle_request(
            {
                "task_kind": "proposal",
                "query_text": "memory index rollout proposal",
                "request_basis": "pass_definition",
                "file_hints": ["scripts/memory_index.py"],
                "surface_filters": ["chat"],
                "status_filters": ["ok"],
                "limit": 4,
                "context_budget_chars": 1800,
                "source_event_id": "ignored",
            }
        )

        self.assertEqual(payload["task_kind"], "proposal")
        self.assertEqual(payload["query_text"], "memory index rollout proposal")
        self.assertEqual(payload["request_basis"], "pass_definition")
        self.assertEqual(payload["file_hints"], ["scripts/memory_index.py"])
        self.assertEqual(payload["surface_filters"], ["chat"])
        self.assertEqual(payload["status_filters"], ["ok"])
        self.assertEqual(payload["limit"], 4)
        self.assertEqual(payload["context_budget_chars"], 1800)
        self.assertEqual(payload["source_event_id"], "ignored")

    def test_format_request_catalog_lists_request_metadata(self) -> None:
        text = format_request_catalog(
            {
                "workspace_id": "local-default",
                "request_count": 2,
                "requests": [
                    {
                        "task_kind": "proposal",
                        "query_text": "memory index rollout proposal",
                        "source_hit": True,
                        "selected_count": 3,
                        "omitted_count": 4,
                        "source_status": "ok",
                    },
                    {
                        "task_kind": "failure_analysis",
                        "query_text": "ranking regression repair",
                        "source_hit": False,
                        "selected_count": 1,
                        "omitted_count": 7,
                        "source_status": "failed",
                        "miss_reason": "ranked_out_by_limit",
                        "request_variant": "adversarial-pass-definition",
                    },
                ],
            },
            dataset_path=Path("/tmp/recall.json"),
        )

        self.assertIn("Workspace: local-default", text)
        self.assertIn("Dataset: /tmp/recall.json", text)
        self.assertIn("1. proposal", text)
        self.assertIn("hit=yes", text)
        self.assertIn("2. failure_analysis", text)
        self.assertIn("hit=no", text)
        self.assertIn("variant=adversarial-pass-definition", text)
        self.assertIn("miss=ranked_out_by_limit", text)

    def test_select_dataset_request_uses_one_based_index(self) -> None:
        request, entry = select_dataset_request(
            {
                "requests": [
                    {
                        "task_kind": "review",
                        "query_text": "review the patch",
                        "file_hints": ["scripts/a.py"],
                        "limit": 2,
                        "context_budget_chars": 1200,
                    },
                    {
                        "task_kind": "proposal",
                        "query_text": "write the rollout",
                        "file_hints": ["scripts/b.py"],
                        "limit": 4,
                        "context_budget_chars": 2200,
                    },
                ]
            },
            request_index=2,
        )

        self.assertEqual(request["task_kind"], "proposal")
        self.assertEqual(request["query_text"], "write the rollout")
        self.assertEqual(entry["file_hints"], ["scripts/b.py"])

    def test_build_bundle_report_summarizes_candidates_and_blocks(self) -> None:
        report = build_bundle_report(
            {
                "task_kind": "proposal",
                "query_text": "memory index rollout proposal",
                "selected_count": 2,
                "omitted_count": 5,
                "budget": {
                    "used_chars": 900,
                    "context_budget_chars": 1800,
                },
                "source_evaluation": {
                    "source_selected": False,
                    "source_rank": 4,
                    "miss_reason": "ranked_out_by_limit",
                },
                "selected_candidates": [
                    {
                        "block_title": "Accepted outcomes",
                        "status": "ok",
                        "score": 12.5,
                        "reasons": ["fts-hit", "accepted-signal"],
                        "prompt_excerpt": "Proposal 1 for the memory index rollout.",
                    }
                ],
                "blocks": [
                    {
                        "title": "Accepted outcomes",
                        "items": [
                            {
                                "status": "ok",
                                "score": 12.5,
                                "summary": "Prompt: Proposal 1 for the memory index rollout.",
                            }
                        ],
                    }
                ],
            },
            request_label="dataset[1] proposal hit=yes",
        )

        self.assertIn("Request: dataset[1] proposal hit=yes", report)
        self.assertIn("Source hit: no", report)
        self.assertIn("Miss reason: ranked_out_by_limit", report)
        self.assertIn("Selected candidates:", report)
        self.assertIn("Accepted outcomes", report)
        self.assertIn("Blocks:", report)

    def test_build_bundle_report_explains_grouped_source_hit(self) -> None:
        report = build_bundle_report(
            {
                "task_kind": "proposal",
                "query_text": "Pass means execution completed.",
                "selected_count": 1,
                "omitted_count": 0,
                "budget": {
                    "used_chars": 720,
                    "context_budget_chars": 1800,
                },
                "source_evaluation": {
                    "source_selected": True,
                    "source_rank": 2,
                    "source_selected_via_group": True,
                    "source_grouped_by": "pass_definition",
                    "source_group_event_id": "leader",
                    "source_group_member_count": 3,
                    "source_group_member_labels": ["leader capability", "source capability", "third capability"],
                },
                "selected_candidates": [
                    {
                        "block_title": "Accepted outcomes",
                        "status": "ok",
                        "score": 21.0,
                        "group_member_count": 3,
                        "group_member_labels": ["leader capability", "source capability", "third capability"],
                        "reasons": ["query-phrase-match", "pass-definition-group"],
                        "prompt_excerpt": "3 results share pass definition.",
                    }
                ],
                "blocks": [],
            }
        )

        self.assertIn("Source hit: yes", report)
        self.assertIn("Source group: pass_definition representative=leader members=3", report)
        self.assertIn("group members: 3: leader capability; source capability; third capability", report)

    def test_pinned_event_compare_reports_normal_vs_pinned_without_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = WorkspaceSessionStore(root=root)
            for prompt in [
                "Pin-only recall diagnosis target.",
                "Decoy recall note one.",
                "Decoy recall note two.",
                "Review the leader patch.",
            ]:
                messages = store.chat_messages_for_next_turn(
                    model_id="backend-a",
                    system_prompt="You are concise.",
                )
                store.record_chat_turn(
                    model_id="backend-a",
                    status="ok",
                    artifact_path=root / "artifacts" / "text" / f"{prompt.split()[0].lower()}-{len(prompt)}.json",
                    prompt=prompt,
                    system_prompt="You are concise.",
                    resolved_user_prompt=prompt,
                    output_text=f"Accepted outcome for {prompt}",
                    base_messages=messages,
                    notes=["accepted"],
                )
            _materialize_workspace_artifacts(root)
            summary = rebuild_memory_index(root=root)
            index = MemoryIndex(Path(summary["index_path"]))
            pin_event_id = index.search("Pin", limit=1)[0]["event_id"]

            compare = build_pinned_event_compare(
                {
                    "task_kind": "review",
                    "query_text": "Review the leader patch.",
                    "limit": 1,
                    "context_budget_chars": 1800,
                },
                pinned_event_ids=[pin_event_id],
                root=root,
                workspace_id="local-default",
                index_path=Path(summary["index_path"]),
            )
            report = format_pinned_event_compare_report(compare)

        self.assertIn("without a score boost", compare["pin_policy"])
        self.assertEqual(compare["selected_added_event_ids"], [])
        self.assertEqual(compare["pinned_events"][0]["normal_source_rank"], None)
        self.assertIsNotNone(compare["pinned_events"][0]["pinned_source_rank"])
        self.assertIn("Pinned event compare", report)
        self.assertIn("pinned reasons:", report)

    def test_format_evaluation_report_summarizes_hit_rate_and_miss_reasons(self) -> None:
        report = format_evaluation_report(
            {
                "workspace_id": "local-default",
                "request_count": 4,
                "source_hits": 3,
                "source_misses": 1,
                "hit_rate": 0.75,
                "miss_reason_counts": {"ranked_out_by_limit": 1},
                "variants": {
                    "baseline": {"request_count": 3, "source_hits": 3, "hit_rate": 1.0},
                    "adversarial-pass-definition": {"request_count": 1, "source_hits": 0, "hit_rate": 0.0},
                },
            },
            dataset_path=Path("/tmp/recall.json"),
        )

        self.assertIn("Requests: 4", report)
        self.assertIn("Hit rate: 0.750", report)
        self.assertIn("- ranked_out_by_limit: 1", report)
        self.assertIn("- adversarial-pass-definition: 0 / 1", report)

    def test_compare_evaluation_summaries_reports_top_level_and_variant_delta(self) -> None:
        delta = compare_evaluation_summaries(
            {
                "request_count": 4,
                "source_hits": 2,
                "source_misses": 2,
                "hit_rate": 0.5,
                "variants": {
                    "baseline": {"request_count": 3, "source_hits": 2, "source_misses": 1, "hit_rate": 0.667},
                },
                "miss_reason_counts": {"ranked_out_by_limit": 2},
            },
            {
                "request_count": 4,
                "source_hits": 3,
                "source_misses": 1,
                "hit_rate": 0.75,
                "variants": {
                    "baseline": {"request_count": 3, "source_hits": 3, "source_misses": 0, "hit_rate": 1.0},
                    "adversarial-pass-definition": {"request_count": 1, "source_hits": 0, "source_misses": 1, "hit_rate": 0.0},
                },
                "miss_reason_counts": {"dropped_by_context_budget": 1},
            },
        )

        self.assertEqual(delta["source_hits_delta"], 1)
        self.assertEqual(delta["source_misses_delta"], -1)
        self.assertEqual(delta["hit_rate_delta"], 0.25)
        self.assertEqual(delta["variants"]["baseline"]["source_hits_delta"], 1)
        self.assertEqual(delta["miss_reason_counts"]["ranked_out_by_limit"], -2)
        self.assertEqual(delta["miss_reason_counts"]["dropped_by_context_budget"], 1)

    def test_record_evaluation_summary_writes_latest_and_includes_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload, latest_path, run_path = record_evaluation_summary(
                {
                    "workspace_id": "local-default",
                    "request_count": 2,
                    "source_hits": 1,
                    "source_misses": 1,
                    "hit_rate": 0.5,
                    "variants": {
                        "baseline": {"request_count": 2, "source_hits": 1, "source_misses": 1, "hit_rate": 0.5},
                    },
                    "miss_reason_counts": {"dropped_by_context_budget": 1},
                    "misses": [],
                },
                root=root,
                previous_summary={
                    "workspace_id": "local-default",
                    "request_count": 2,
                    "source_hits": 0,
                    "source_misses": 2,
                    "hit_rate": 0.0,
                    "variants": {
                        "baseline": {"request_count": 2, "source_hits": 0, "source_misses": 2, "hit_rate": 0.0},
                    },
                    "miss_reason_counts": {"ranked_out_by_limit": 2},
                    "evaluated_at_utc": "2026-04-14T10:00:00+00:00",
                },
                dataset_path=root / "artifacts" / "recall_data" / "local-default" / "real_recall_dataset.json",
            )

            loaded_payload, resolved_latest_path = load_latest_evaluation_summary(root=root)
            self.assertTrue(latest_path.exists())
            self.assertTrue(run_path.exists())
            self.assertEqual(resolved_latest_path, latest_path)
            self.assertEqual(payload["delta_from_previous"]["source_hits_delta"], 1)
            self.assertEqual(
                payload["dataset_path"],
                str(root / "artifacts" / "recall_data" / "local-default" / "real_recall_dataset.json"),
            )
            self.assertIsNotNone(loaded_payload)
            self.assertEqual(loaded_payload["evaluation_run_path"], str(run_path))

    def test_format_miss_report_lists_source_miss_rows(self) -> None:
        report = format_miss_report(
            {
                "workspace_id": "local-default",
                "request_count": 3,
                "misses": [
                    {
                        "index": 2,
                        "task_kind": "review",
                        "query_text": "tighten recall ranking",
                        "source_status": "ok",
                        "source_rank": 3,
                        "source_prompt_excerpt": "Tighten recall ranking around miss-report requests.",
                        "source_candidate_pool_status": "candidate_pool_present",
                        "source_event_contract_status": "ok",
                        "source_artifact_status": "readable",
                        "source_evidence_types": ["source-artifact", "accepted"],
                        "source_evidence_type_match": True,
                        "source_reasons": ["priority:review:accepted"],
                        "miss_reason_detail": "source event was in the candidate pool but fell below the selected candidate limit",
                        "top_selected": [
                            {
                                "event_id": "leader",
                                "score": 22.1,
                                "reasons": ["fts-hit", "priority:review:accepted"],
                            }
                        ],
                        "miss_reason": "ranked_out_by_limit",
                        "request_variant": "adversarial-pass-definition",
                    }
                ],
            },
            dataset_path=Path("/tmp/recall.json"),
        )

        self.assertIn("Misses: 1 / 3", report)
        self.assertIn("reason=ranked_out_by_limit", report)
        self.assertIn("pool=candidate_pool_present", report)
        self.assertIn("type_match=yes", report)
        self.assertIn("contract=ok", report)
        self.assertIn("evidence=source-artifact,accepted", report)
        self.assertIn("detail: source event was in the candidate pool", report)
        self.assertIn("source reasons: priority:review:accepted", report)
        self.assertIn("top selected: leader", report)
        self.assertIn("variant=adversarial-pass-definition", report)
        self.assertIn("source: Tighten recall ranking", report)

    def test_ensure_recall_dataset_can_refresh_into_custom_path(self) -> None:
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

            dataset_path = root / "tmp" / "custom_recall_dataset.json"
            _materialize_workspace_artifacts(root)
            dataset, resolved_path = ensure_recall_dataset(
                root=root,
                dataset_path=dataset_path,
                refresh=True,
            )

            self.assertEqual(resolved_path, dataset_path)
            self.assertTrue(dataset_path.exists())
            self.assertGreaterEqual(dataset["request_count"], 1)

    def test_main_uses_stored_bundle_for_dataset_request(self) -> None:
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

            dataset_path = root / "artifacts" / "recall_data" / "local-default" / "real_recall_dataset.json"
            _materialize_workspace_artifacts(root)
            dataset, _resolved_path = ensure_recall_dataset(
                root=root,
                dataset_path=dataset_path,
                refresh=True,
            )
            bundle_path = Path(dataset["requests"][0]["bundle_path"])
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["selected_candidates"][0]["prompt_excerpt"] = "SENTINEL STORED BUNDLE"
            bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            _materialize_workspace_artifacts(root)
            stdout = io.StringIO()
            with patch.object(
                sys,
                "argv",
                [
                    "run_recall_demo.py",
                    "--root",
                    str(root),
                    "--dataset-path",
                    str(dataset_path),
                    "--request-index",
                    "1",
                ],
            ):
                with redirect_stdout(stdout):
                    exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertIn("SENTINEL STORED BUNDLE", stdout.getvalue())

    def test_main_prints_miss_report_for_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_path = root / "artifacts" / "recall_data" / "local-default" / "bundles" / "01-review.json"
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            bundle = {
                "bundle_version": CONTEXT_BUNDLE_VERSION,
                "task_kind": "review",
                "query_text": "tighten recall ranking",
                "selected_count": 1,
                "omitted_count": 3,
                "budget": {"used_chars": 420, "context_budget_chars": 1800},
                "selected_candidates": [],
                "blocks": [],
                "source_evaluation": {
                    "source_event_id": "source-1",
                    "source_selected": False,
                    "source_rank": 2,
                    "source_score": 11.2,
                    "source_block_title": "Accepted outcomes",
                    "source_prompt_excerpt": "Tighten recall ranking around miss-report requests.",
                    "source_reasons": ["query-coverage"],
                    "miss_reason": "ranked_out_by_limit",
                    "selected_count": 1,
                    "top_selected": [],
                },
            }
            bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            dataset_path = root / "artifacts" / "recall_data" / "local-default" / "real_recall_dataset.json"
            dataset = {
                "workspace_id": "local-default",
                "request_count": 1,
                "requests": [
                    {
                        "task_kind": "review",
                        "query_text": "tighten recall ranking",
                        "file_hints": [],
                        "limit": 4,
                        "context_budget_chars": 1800,
                        "source_event_id": "source-1",
                        "source_status": "ok",
                        "bundle_path": str(bundle_path),
                        "selected_count": 1,
                        "omitted_count": 3,
                        "source_hit": False,
                        "miss_reason": "ranked_out_by_limit",
                    }
                ],
            }
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            stdout = io.StringIO()
            with patch.object(
                sys,
                "argv",
                [
                    "run_recall_demo.py",
                    "--root",
                    str(root),
                    "--dataset-path",
                    str(dataset_path),
                    "--miss-report",
                ],
            ):
                with redirect_stdout(stdout):
                    exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertIn("reason=ranked_out_by_limit", stdout.getvalue())

    def test_main_prepare_eval_does_not_rebuild_index_twice(self) -> None:
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

            stdout = io.StringIO()
            with patch("run_recall_demo.rebuild_memory_index") as rebuild_mock:
                rebuild_mock.side_effect = AssertionError("unexpected duplicate rebuild")
                with patch.object(
                    sys,
                    "argv",
                    [
                        "run_recall_demo.py",
                        "--root",
                        str(root),
                        "--prepare-real-data",
                        "--eval",
                        "--max-requests",
                        "1",
                        "--max-adversarial-requests",
                        "0",
                    ],
                ):
                    with redirect_stdout(stdout):
                        exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertIn("Source hits:", stdout.getvalue())
        rebuild_mock.assert_not_called()

    def test_main_refreshes_stale_not_retrieved_bundle_against_current_index(self) -> None:
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
            dataset_path = root / "artifacts" / "recall_data" / "local-default" / "real_recall_dataset.json"
            _materialize_workspace_artifacts(root)
            dataset, _resolved_path = ensure_recall_dataset(
                root=root,
                dataset_path=dataset_path,
                refresh=True,
            )
            entry = dict(dataset["requests"][0])
            bundle_path = Path(entry["bundle_path"])
            stale_bundle = {
                "bundle_version": CONTEXT_BUNDLE_VERSION,
                "task_kind": entry["task_kind"],
                "query_text": entry["query_text"],
                "selected_count": 0,
                "omitted_count": 0,
                "budget": {"used_chars": 0, "context_budget_chars": 1800},
                "selected_candidates": [],
                "blocks": [],
                "source_evaluation": {
                    "source_event_id": entry["source_event_id"],
                    "source_selected": False,
                    "source_rank": None,
                    "source_score": None,
                    "source_block_title": None,
                    "source_prompt_excerpt": None,
                    "source_reasons": [],
                    "miss_reason": "not_retrieved",
                    "selected_count": 0,
                    "top_selected": [],
                },
            }
            bundle_path.write_text(json.dumps(stale_bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            dataset["requests"][0]["source_hit"] = False
            dataset["requests"][0]["selected_count"] = 0
            dataset["requests"][0]["omitted_count"] = 0
            dataset["requests"][0]["miss_reason"] = "not_retrieved"
            dataset_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            stdout = io.StringIO()
            with patch.object(
                sys,
                "argv",
                [
                    "run_recall_demo.py",
                    "--root",
                    str(root),
                    "--dataset-path",
                    str(dataset_path),
                    "--miss-report",
                ],
            ):
                with redirect_stdout(stdout):
                    exit_code = main()

            refreshed_dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
            refreshed_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertIn("No source misses.", stdout.getvalue())
        self.assertTrue(refreshed_dataset["requests"][0]["source_hit"])
        self.assertIsNone(refreshed_dataset["requests"][0]["miss_reason"])
        self.assertTrue(refreshed_bundle["source_evaluation"]["source_selected"])

    def test_main_refreshes_stale_bundle_when_source_artifact_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_path = root / "artifacts" / "text" / "review.json"
            store = WorkspaceSessionStore(root=root)
            messages = store.chat_messages_for_next_turn(
                model_id="backend-a",
                system_prompt="You are concise.",
            )
            store.record_chat_turn(
                model_id="backend-a",
                status="ok",
                artifact_path=artifact_path,
                prompt="Review the memory index patch.",
                system_prompt="You are concise.",
                resolved_user_prompt="Review the memory index patch.",
                output_text="Looks good with one regression note.",
                base_messages=messages,
                notes=["review accepted"],
            )
            dataset_path = root / "artifacts" / "recall_data" / "local-default" / "real_recall_dataset.json"
            _materialize_workspace_artifacts(root)
            dataset, _resolved_path = ensure_recall_dataset(
                root=root,
                dataset_path=dataset_path,
                refresh=True,
                max_requests=1,
                max_adversarial_requests=0,
            )
            bundle_path = Path(dataset["requests"][0]["bundle_path"])
            self.assertTrue(dataset["requests"][0]["source_hit"])
            self.assertTrue(artifact_path.exists())

            artifact_path.unlink()
            stdout = io.StringIO()
            with patch.object(
                sys,
                "argv",
                [
                    "run_recall_demo.py",
                    "--root",
                    str(root),
                    "--dataset-path",
                    str(dataset_path),
                    "--miss-report",
                ],
            ):
                with redirect_stdout(stdout):
                    exit_code = main()

            refreshed_dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
            refreshed_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertIn("reason=source_event_contract_broken", stdout.getvalue())
        self.assertFalse(refreshed_dataset["requests"][0]["source_hit"])
        self.assertEqual(refreshed_dataset["requests"][0]["miss_reason"], "source_event_contract_broken")
        self.assertEqual(
            refreshed_bundle["source_evaluation"]["source_event_contract_status"],
            "missing_source",
        )
        self.assertIn(
            "source_artifact_missing",
            refreshed_bundle["source_evaluation"]["source_artifact_reasons"],
        )

    def test_main_rebuilds_stale_default_index_before_miss_report(self) -> None:
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
                prompt="Review the first patch.",
                system_prompt="You are concise.",
                resolved_user_prompt="Review the first patch.",
                output_text="First pass looks good.",
                base_messages=first_messages,
                notes=["review accepted"],
            )
            _materialize_workspace_artifacts(root)
            rebuild_memory_index(root=root)

            second_messages = store.chat_messages_for_next_turn(
                model_id="backend-a",
                system_prompt="You are concise.",
            )
            store.record_chat_turn(
                model_id="backend-a",
                status="ok",
                artifact_path=root / "artifacts" / "text" / "second.json",
                prompt="Review the second patch.",
                system_prompt="You are concise.",
                resolved_user_prompt="Review the second patch.",
                output_text="Second pass looks good.",
                base_messages=second_messages,
                notes=["review accepted"],
            )

            dataset_path = root / "artifacts" / "recall_data" / "local-default" / "real_recall_dataset.json"
            _materialize_workspace_artifacts(root)
            dataset, _resolved_path = ensure_recall_dataset(
                root=root,
                dataset_path=dataset_path,
                index_path=root / "tmp-fresh-index.sqlite3",
                refresh=True,
                max_requests=1,
            )
            source_event_id = dataset["requests"][0]["source_event_id"]
            stale_default_index = MemoryIndex(default_memory_index_path(workspace_id="local-default", root=root))
            self.assertIsNone(stale_default_index.get_event(source_event_id))

            stdout = io.StringIO()
            with patch.object(
                sys,
                "argv",
                [
                    "run_recall_demo.py",
                    "--root",
                    str(root),
                    "--dataset-path",
                    str(dataset_path),
                    "--miss-report",
                ],
            ):
                with redirect_stdout(stdout):
                    exit_code = main()

            refreshed_default_index = MemoryIndex(default_memory_index_path(workspace_id="local-default", root=root))
            source_present_after_main = refreshed_default_index.get_event(source_event_id) is not None

        self.assertEqual(exit_code, 0)
        self.assertIn("No source misses.", stdout.getvalue())
        self.assertTrue(source_present_after_main)


if __name__ == "__main__":
    unittest.main()
