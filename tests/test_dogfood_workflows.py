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

from artifact_schema import build_artifact_payload, build_prompt_record, build_runtime_record, write_artifact  # noqa: E402
from dogfood_workflows import (  # noqa: E402
    DOGFOOD_WORKFLOW_PREVIEW_SCHEMA_NAME,
    build_dogfood_workflow_preview,
    format_dogfood_workflow_preview_report,
    record_dogfood_workflow_preview,
)
from evaluation_loop import evaluation_comparison_log_path, record_review_resolution_signal  # noqa: E402
from run_dogfood_workflow import main as dogfood_main  # noqa: E402
from software_work_events import iter_workspace_events  # noqa: E402
from workspace_state import WorkspaceSessionStore  # noqa: E402


def seed_work_event(
    *,
    root: Path,
    store: WorkspaceSessionStore,
    prompt: str,
    output_text: str,
    artifact_name: str,
) -> str:
    artifact_path = root / "artifacts" / "text" / artifact_name
    write_artifact(
        artifact_path,
        build_artifact_payload(
            artifact_kind="text",
            status="ok",
            runtime=build_runtime_record(backend="backend-a", model_id="backend-a"),
            prompts=build_prompt_record(prompt=prompt, resolved_user_prompt=prompt),
            extra={
                "validation": {
                    "validation_mode": "unit",
                    "claim_scope": prompt,
                    "pass_definition": "dogfood workflow fixture passes",
                    "quality_status": "pass",
                    "execution_status": "ok",
                    "quality_checks": [{"name": "unit", "pass": True, "detail": "ok"}],
                },
                "output_text": output_text,
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
        prompt=prompt,
        system_prompt=None,
        resolved_user_prompt=prompt,
        output_text=output_text,
    )
    return str(iter_workspace_events(root=root)[-1]["event_id"])


class DogfoodWorkflowTests(unittest.TestCase):
    def test_compare_proposals_preview_does_not_record_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = WorkspaceSessionStore(root=root)
            candidate_a = seed_work_event(
                root=root,
                store=store,
                prompt="Proposal A keeps the review queue small.",
                output_text="Proposal A passes.",
                artifact_name="proposal-a.json",
            )
            candidate_b = seed_work_event(
                root=root,
                store=store,
                prompt="Proposal B adds a broader workflow layer.",
                output_text="Proposal B passes.",
                artifact_name="proposal-b.json",
            )

            preview, latest_path, run_path = record_dogfood_workflow_preview(
                root=root,
                workflow_kind="compare_proposals",
                candidate_event_ids=[candidate_a, candidate_b],
                winner_event_id=candidate_a,
                query_text="compare the two dogfood workflow proposals",
            )
            report = format_dogfood_workflow_preview_report(preview)
            latest_exists = latest_path.exists()
            run_exists = run_path.exists()
            comparison_log_exists = evaluation_comparison_log_path(root=root).exists()

        self.assertEqual(preview["schema_name"], DOGFOOD_WORKFLOW_PREVIEW_SCHEMA_NAME)
        self.assertTrue(latest_exists)
        self.assertTrue(run_exists)
        self.assertEqual(preview["comparison_preview"]["candidate_count"], 2)
        self.assertTrue(preview["comparison_preview"]["ready_to_record"])
        self.assertFalse(preview["guardrails"]["records_comparisons"])
        self.assertFalse(preview["guardrails"]["writes_training_data"])
        self.assertFalse(comparison_log_exists)
        self.assertIn("Comparison preview:", report)

    def test_cli_writes_review_patch_preview_as_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = WorkspaceSessionStore(root=root)
            source_event_id = seed_work_event(
                root=root,
                store=store,
                prompt="Review patch for local workflow launcher.",
                output_text="Patch evidence is ready.",
                artifact_name="patch.json",
            )
            stdout = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "run_dogfood_workflow.py",
                        "--root",
                        str(root),
                        "--workflow-kind",
                        "review_patch",
                        "--source-event-id",
                        source_event_id,
                        "--query",
                        "review launcher patch",
                        "--format",
                        "json",
                    ],
                ),
                redirect_stdout(stdout),
            ):
                exit_code = dogfood_main()
            payload = json.loads(stdout.getvalue())
            workflow_run_exists = Path(payload["workflow_preview_run_path"]).exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["preview"]["workflow_kind"], "review_patch")
        self.assertEqual(payload["preview"]["export_mode"], "preview_only")
        self.assertTrue(workflow_run_exists)
        self.assertFalse(payload["preview"]["training_export_ready"])

    def test_direct_resolved_work_preview_builds_in_memory_curation_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = WorkspaceSessionStore(root=root)
            source_event_id = seed_work_event(
                root=root,
                store=store,
                prompt="Resolved review work should stay preview-only.",
                output_text="Review resolved and tests pass.",
                artifact_name="resolved.json",
            )
            record_review_resolution_signal(
                root=root,
                source_event_id=source_event_id,
                resolved=True,
                resolution_summary="Review resolved before curation preview.",
            )

            preview = build_dogfood_workflow_preview(
                root=root,
                workflow_kind="resolved_work_curation_preview",
                source_event_id=source_event_id,
                curation_filters={"states": ["ready"]},
            )

        self.assertEqual(preview["curation_preview"]["export_mode"], "preview_only")
        self.assertEqual(preview["curation_preview"]["filters"]["states"], ["ready"])
        self.assertEqual(preview["curation_preview"]["filters"]["reasons"], ["review_resolved"])
        self.assertEqual(preview["curation_preview"]["counts"]["previewed_candidate_count"], 1)

    def test_compare_proposals_preview_does_not_suggest_invalid_winner_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = WorkspaceSessionStore(root=root)
            only_candidate = seed_work_event(
                root=root,
                store=store,
                prompt="Only one proposal is not enough for a valid comparison.",
                output_text="Needs another candidate.",
                artifact_name="single-proposal.json",
            )

            preview, _latest_path, _run_path = record_dogfood_workflow_preview(
                root=root,
                workflow_kind="compare_proposals",
                candidate_event_ids=[only_candidate],
                winner_event_id=only_candidate,
            )
            command = preview["comparison_preview"]["record_command_preview"]["argv"]

        self.assertFalse(preview["comparison_preview"]["ready_to_record"])
        self.assertEqual(preview["comparison_preview"]["outcome"], "needs_follow_up")
        self.assertEqual(command, [])
        self.assertTrue(preview["comparison_preview"]["record_command_preview"]["blocked"])
        self.assertIn("needs_at_least_two_candidates", preview["comparison_preview"]["blocking_reasons"])

    def test_compare_proposals_preview_blocks_missing_candidate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = WorkspaceSessionStore(root=root)
            present_candidate = seed_work_event(
                root=root,
                store=store,
                prompt="Present comparison candidate.",
                output_text="Present candidate passes.",
                artifact_name="present-proposal.json",
            )
            missing_candidate = "local-default:missing-proposal"

            preview, _latest_path, _run_path = record_dogfood_workflow_preview(
                root=root,
                workflow_kind="compare_proposals",
                candidate_event_ids=[present_candidate, missing_candidate],
                winner_event_id=present_candidate,
            )

        self.assertFalse(preview["comparison_preview"]["ready_to_record"])
        self.assertEqual(preview["comparison_preview"]["record_command_preview"]["argv"], [])
        self.assertEqual(preview["comparison_preview"]["missing_candidate_event_ids"], [missing_candidate])
        self.assertIn(
            "candidate_event_id_missing_from_index",
            preview["comparison_preview"]["blocking_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
