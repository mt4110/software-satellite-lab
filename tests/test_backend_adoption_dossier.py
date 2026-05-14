from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from artifact_vault import capture_artifact  # noqa: E402
from backend_adoption_dossier import (  # noqa: E402
    BACKEND_ADOPTION_DOSSIER_SCHEMA_NAME,
    build_backend_adoption_dossier,
    build_backend_adoption_dossier_from_comparison_id,
    build_backend_adoption_dossier_from_review,
    format_backend_adoption_dossier_markdown,
    validate_backend_adoption_dossier,
)
from evaluation_loop import (  # noqa: E402
    EVALUATION_COMPARISON_SCHEMA_NAME,
    EVALUATION_COMPARISON_SCHEMA_VERSION,
    append_evaluation_comparison,
    evaluation_comparison_log_path,
)
from failure_memory_review import record_file_input  # noqa: E402
from satlab import main as satlab_main  # noqa: E402
from software_work_events import build_event_record  # noqa: E402
from workspace_state import DEFAULT_WORKSPACE_ID  # noqa: E402


def _comparison(
    *,
    comparison_id: str,
    candidates: list[tuple[str, str, str]],
    winner_event_id: str | None,
    outcome: str = "winner_selected",
    rationale: str | None = "Human reviewer prefers the candidate for this scoped workflow.",
) -> dict:
    return {
        "schema_name": EVALUATION_COMPARISON_SCHEMA_NAME,
        "schema_version": EVALUATION_COMPARISON_SCHEMA_VERSION,
        "comparison_id": comparison_id,
        "workspace_id": DEFAULT_WORKSPACE_ID,
        "recorded_at_utc": "2026-01-01T00:00:00Z",
        "origin": "unit_test",
        "task_label": "backend adoption fixture",
        "outcome": outcome,
        "winner_event_id": winner_event_id,
        "candidate_count": len(candidates),
        "candidates": [
            {
                "event_id": event_id,
                "source": {"source_event_id": event_id},
                "backend_metadata": {
                    "backend_id": backend_id,
                    "display_name": backend_id.replace("-", " ").title(),
                    "adapter_kind": "local",
                    "model_id": model_id,
                    "compatibility_status": "compatible",
                    "metadata": {"latency_profile": "fixture-fast", "cost_profile": "local"},
                    "limits": {"max_context_chars": 4096, "max_output_chars": 1024},
                },
            }
            for event_id, backend_id, model_id in candidates
        ],
        "criteria": ["source-linked outcome", "same workflow", "human rationale"],
        "rationale": rationale,
        "tags": ["backend_adoption_fixture"],
    }


def _direct_event(
    root: Path,
    *,
    event_id: str,
    backend_id: str,
    status: str = "accepted",
    quality_status: str | None = "pass",
    execution_status: str | None = "passed",
    event_kind: str = "backend_result",
    session_surface: str = "review",
    include_quality_check: bool = True,
) -> tuple[dict, str]:
    source_path = root / f"{backend_id}.txt"
    source_path.write_text(f"{backend_id} source-linked outcome\n", encoding="utf-8")
    ref = capture_artifact(source_path, kind="candidate_output", root=root)
    quality_checks = (
        [{"name": "fixture_verification", "pass": quality_status == "pass", "detail": backend_id}]
        if include_quality_check
        else []
    )
    event = build_event_record(
        event_id=event_id,
        event_kind=event_kind,
        recorded_at_utc="2026-01-01T00:00:00Z",
        workspace={"workspace_id": DEFAULT_WORKSPACE_ID, "workspace_manifest_path": None},
        session={
            "session_id": f"session-{backend_id}",
            "surface": session_surface,
            "mode": "backend_compare",
            "title": "Backend fixture",
            "selected_model_id": f"{backend_id}/model",
            "session_manifest_path": None,
        },
        outcome={
            "status": status,
            "quality_status": quality_status,
            "execution_status": execution_status,
            "backend_id": backend_id,
            "model_id": f"{backend_id}/model",
        },
        content={
            "prompt": "Compare backend outcome.",
            "output_text": f"{backend_id} completed with status {status}.",
            "notes": ["source-linked fixture"],
            "options": {
                "backend_id": backend_id,
                "model_id": f"{backend_id}/model",
                "quality_status": quality_status,
                "execution_status": execution_status,
                "quality_checks": quality_checks,
                "artifact_vault_refs": [ref],
            },
        },
        source_refs={
            "artifact_vault_refs": [ref],
            "artifact_ref": {
                "entry_id": event_id,
                "artifact_kind": "candidate_output",
                "artifact_path": str(source_path),
                "artifact_workspace_relative_path": source_path.name,
            },
        },
        tags=["backend_compare", backend_id, status],
    )
    return event, str(ref["artifact_id"])


def _direct_adoptable_fixture(root: Path) -> tuple[dict, dict[str, dict], str]:
    candidate_event, _candidate_artifact = _direct_event(
        root,
        event_id="event-candidate",
        backend_id="candidate-backend",
    )
    baseline_event, _baseline_artifact = _direct_event(
        root,
        event_id="event-baseline",
        backend_id="baseline-backend",
    )
    comparison = _comparison(
        comparison_id="comparison-direct",
        candidates=[
            ("event-candidate", "candidate-backend", "candidate-backend/model"),
            ("event-baseline", "baseline-backend", "baseline-backend/model"),
        ],
        winner_event_id="event-candidate",
    )
    return comparison, {"event-candidate": candidate_event, "event-baseline": baseline_event}, "candidate-backend"


def _recorded_proposal_pair(root: Path) -> tuple[dict, dict]:
    candidate_path = root / "candidate.txt"
    baseline_path = root / "baseline.txt"
    candidate_path.write_text("candidate passed\n", encoding="utf-8")
    baseline_path.write_text("baseline passed\n", encoding="utf-8")
    candidate = record_file_input(
        input_kind="proposal",
        source_path=candidate_path,
        status="accepted",
        backend_id="candidate-backend",
        backend_display_name="Candidate Backend",
        backend_adapter_kind="local",
        model_id="candidate/model",
        workspace_id=DEFAULT_WORKSPACE_ID,
        root=root,
    )
    baseline = record_file_input(
        input_kind="proposal",
        source_path=baseline_path,
        status="accepted",
        backend_id="baseline-backend",
        backend_display_name="Baseline Backend",
        backend_adapter_kind="local",
        model_id="baseline/model",
        workspace_id=DEFAULT_WORKSPACE_ID,
        root=root,
    )
    return candidate, baseline


class BackendAdoptionDossierTests(unittest.TestCase):
    def test_dossier_generated_from_comparison_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate_path = root / "candidate.txt"
            baseline_path = root / "baseline.txt"
            candidate_path.write_text("candidate passed\n", encoding="utf-8")
            baseline_path.write_text("baseline passed\n", encoding="utf-8")
            candidate = record_file_input(
                input_kind="proposal",
                source_path=candidate_path,
                status="accepted",
                backend_id="candidate-backend",
                backend_display_name="Candidate Backend",
                backend_adapter_kind="local",
                model_id="candidate/model",
                workspace_id=DEFAULT_WORKSPACE_ID,
                root=root,
            )
            baseline = record_file_input(
                input_kind="proposal",
                source_path=baseline_path,
                status="accepted",
                backend_id="baseline-backend",
                backend_display_name="Baseline Backend",
                backend_adapter_kind="local",
                model_id="baseline/model",
                workspace_id=DEFAULT_WORKSPACE_ID,
                root=root,
            )
            comparison = _comparison(
                comparison_id="comparison-recorded",
                candidates=[
                    (candidate["event_id"], "candidate-backend", "candidate/model"),
                    (baseline["event_id"], "baseline-backend", "baseline/model"),
                ],
                winner_event_id=candidate["event_id"],
            )
            append_evaluation_comparison(
                evaluation_comparison_log_path(root=root, workspace_id=DEFAULT_WORKSPACE_ID),
                comparison,
                workspace_id=DEFAULT_WORKSPACE_ID,
            )

            dossier = build_backend_adoption_dossier_from_comparison_id(
                "comparison-recorded",
                candidate_backend="candidate-backend",
                baseline_backend="baseline-backend",
                root=root,
                workspace_id=DEFAULT_WORKSPACE_ID,
            )

        self.assertEqual(dossier["schema_name"], BACKEND_ADOPTION_DOSSIER_SCHEMA_NAME)
        self.assertEqual(dossier["recommendation"]["value"], "adopt")
        self.assertEqual(dossier["candidate_metadata"]["backend_id"], "candidate-backend")
        self.assertTrue(dossier["exit_gate"]["negative_evidence_visible"])
        self.assertEqual(validate_backend_adoption_dossier(dossier), [])

    def test_missing_source_yields_insufficient_evidence(self) -> None:
        comparison = _comparison(
            comparison_id="comparison-missing",
            candidates=[
                ("missing-candidate-event", "candidate-backend", "candidate/model"),
                ("missing-baseline-event", "baseline-backend", "baseline/model"),
            ],
            winner_event_id="missing-candidate-event",
        )

        dossier = build_backend_adoption_dossier(
            comparison,
            events_by_id={},
            candidate_backend="candidate-backend",
            baseline_backend="baseline-backend",
        )

        self.assertEqual(dossier["recommendation"]["value"], "insufficient_evidence")
        self.assertIn("source_evidence_missing", dossier["rule_evaluation"]["blockers"])

    def test_candidate_with_false_support_cannot_be_adopted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            comparison, events, _backend = _direct_adoptable_fixture(Path(tmpdir))
            dossier = build_backend_adoption_dossier(
                comparison,
                events_by_id=events,
                candidate_backend="candidate-backend",
                baseline_backend="baseline-backend",
                benchmark_report={
                    "schema_name": "fixture-benchmark",
                    "passed": False,
                    "critical_false_support_count": 1,
                },
                root=Path(tmpdir),
            )

        self.assertEqual(dossier["recommendation"]["value"], "reject")
        self.assertIn("critical_false_support_present", dossier["rule_evaluation"]["blockers"])

    def test_strict_missing_benchmark_is_insufficient_evidence_not_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            comparison, events, _backend = _direct_adoptable_fixture(Path(tmpdir))
            dossier = build_backend_adoption_dossier(
                comparison,
                events_by_id=events,
                candidate_backend="candidate-backend",
                baseline_backend="baseline-backend",
                strict=True,
                root=Path(tmpdir),
            )

        self.assertEqual(dossier["recommendation"]["value"], "insufficient_evidence")
        self.assertIn("benchmark_gate_missing", dossier["rule_evaluation"]["blockers"])
        self.assertNotIn("benchmark_gate_failed", dossier["rule_evaluation"]["blockers"])

    def test_no_rollback_path_blocks_adoption(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            comparison, events, _backend = _direct_adoptable_fixture(Path(tmpdir))
            dossier = build_backend_adoption_dossier(
                comparison,
                events_by_id=events,
                candidate_backend="candidate-backend",
                baseline_backend="baseline-backend",
                rollback_plan="",
                root=Path(tmpdir),
            )

        self.assertEqual(dossier["recommendation"]["value"], "insufficient_evidence")
        self.assertIn("rollback_path_absent", dossier["rule_evaluation"]["blockers"])

    def test_human_rationale_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            comparison, events, _backend = _direct_adoptable_fixture(Path(tmpdir))
            comparison["rationale"] = None
            dossier = build_backend_adoption_dossier(
                comparison,
                events_by_id=events,
                candidate_backend="candidate-backend",
                baseline_backend="baseline-backend",
                root=Path(tmpdir),
            )

        self.assertEqual(dossier["recommendation"]["value"], "insufficient_evidence")
        self.assertIn("human_rationale_absent", dossier["rule_evaluation"]["blockers"])

    def test_unverified_agent_claim_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate_event, _artifact_id = _direct_event(
                root,
                event_id="event-candidate",
                backend_id="candidate-backend",
                event_kind="agent_task_run",
                session_surface="agent_lane",
                quality_status=None,
                execution_status=None,
                include_quality_check=False,
            )
            baseline_event, _baseline_artifact = _direct_event(
                root,
                event_id="event-baseline",
                backend_id="baseline-backend",
            )
            comparison = _comparison(
                comparison_id="comparison-unverified",
                candidates=[
                    ("event-candidate", "candidate-backend", "candidate/model"),
                    ("event-baseline", "baseline-backend", "baseline/model"),
                ],
                winner_event_id="event-candidate",
            )
            dossier = build_backend_adoption_dossier(
                comparison,
                events_by_id={"event-candidate": candidate_event, "event-baseline": baseline_event},
                candidate_backend="candidate-backend",
                baseline_backend="baseline-backend",
                root=root,
            )

        self.assertEqual(dossier["recommendation"]["value"], "insufficient_evidence")
        self.assertIn("candidate_only_wins_by_agent_self_claim", dossier["rule_evaluation"]["blockers"])

    def test_negative_evidence_appears_in_dossier(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate_event, candidate_artifact = _direct_event(
                root,
                event_id="event-candidate",
                backend_id="candidate-backend",
                status="failed",
                quality_status="fail",
                execution_status="failed",
            )
            baseline_event, _baseline_artifact = _direct_event(
                root,
                event_id="event-baseline",
                backend_id="baseline-backend",
            )
            comparison = _comparison(
                comparison_id="comparison-negative",
                candidates=[
                    ("event-candidate", "candidate-backend", "candidate/model"),
                    ("event-baseline", "baseline-backend", "baseline/model"),
                ],
                winner_event_id="event-baseline",
            )
            dossier = build_backend_adoption_dossier(
                comparison,
                events_by_id={"event-candidate": candidate_event, "event-baseline": baseline_event},
                candidate_backend="candidate-backend",
                baseline_backend="baseline-backend",
                root=root,
            )
            markdown = format_backend_adoption_dossier_markdown(dossier)

        self.assertEqual(dossier["recommendation"]["value"], "reject")
        self.assertTrue(dossier["exit_gate"]["negative_evidence_visible"])
        self.assertIn(candidate_artifact, markdown)
        self.assertIn("Failures and Regressions", markdown)

    def test_recommendation_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            comparison, events, _backend = _direct_adoptable_fixture(Path(tmpdir))
            first = build_backend_adoption_dossier(
                comparison,
                events_by_id=events,
                candidate_backend="candidate-backend",
                baseline_backend="baseline-backend",
                root=Path(tmpdir),
            )
            second = build_backend_adoption_dossier(
                comparison,
                events_by_id=events,
                candidate_backend="candidate-backend",
                baseline_backend="baseline-backend",
                root=Path(tmpdir),
            )

        self.assertEqual(first["recommendation"], second["recommendation"])
        self.assertEqual(first["rule_evaluation"], second["rule_evaluation"])

    def test_json_schema_validates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            comparison, events, _backend = _direct_adoptable_fixture(Path(tmpdir))
            dossier = build_backend_adoption_dossier(
                comparison,
                events_by_id=events,
                candidate_backend="candidate-backend",
                baseline_backend="baseline-backend",
                root=Path(tmpdir),
            )

        self.assertEqual(validate_backend_adoption_dossier(dossier), [])
        self.assertTrue((REPO_ROOT / "schemas" / "backend_adoption_dossier.schema.json").is_file())
        broken = dict(dossier)
        del broken["benchmark_gate"]
        self.assertIn(
            {"path": "$.benchmark_gate", "message": "Missing required dossier field."},
            validate_backend_adoption_dossier(broken),
        )
        broken = dict(dossier)
        del broken["rule_evaluation"]
        self.assertIn(
            {"path": "$.rule_evaluation", "message": "Missing required dossier field."},
            validate_backend_adoption_dossier(broken),
        )
        broken = dict(dossier)
        broken["exit_gate"] = dict(dossier["exit_gate"])
        broken["exit_gate"]["no_live_api_required"] = False
        self.assertIn(
            {"path": "$.exit_gate.no_live_api_required", "message": "Expected true."},
            validate_backend_adoption_dossier(broken),
        )
        schema = json.loads((REPO_ROOT / "schemas" / "backend_adoption_dossier.schema.json").read_text())
        self.assertIs(
            schema["properties"]["exit_gate"]["properties"]["negative_evidence_visible"]["const"],
            True,
        )

    def test_markdown_report_includes_source_artifact_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate_event, candidate_artifact = _direct_event(
                root,
                event_id="event-candidate",
                backend_id="candidate-backend",
            )
            baseline_event, _baseline_artifact = _direct_event(
                root,
                event_id="event-baseline",
                backend_id="baseline-backend",
            )
            comparison = _comparison(
                comparison_id="comparison-markdown",
                candidates=[
                    ("event-candidate", "candidate-backend", "candidate/model"),
                    ("event-baseline", "baseline-backend", "baseline/model"),
                ],
                winner_event_id="event-candidate",
            )
            dossier = build_backend_adoption_dossier(
                comparison,
                events_by_id={"event-candidate": candidate_event, "event-baseline": baseline_event},
                candidate_backend="candidate-backend",
                baseline_backend="baseline-backend",
                root=root,
            )
            markdown = format_backend_adoption_dossier_markdown(dossier)

        self.assertIn(candidate_artifact, markdown)
        self.assertIn("Source Artifact Refs", markdown)

    def test_satlab_backend_dossier_cli_outputs_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate_path = root / "candidate.txt"
            baseline_path = root / "baseline.txt"
            candidate_path.write_text("candidate passed\n", encoding="utf-8")
            baseline_path.write_text("baseline passed\n", encoding="utf-8")
            candidate = record_file_input(
                input_kind="proposal",
                source_path=candidate_path,
                status="accepted",
                backend_id="candidate-backend",
                backend_display_name="Candidate Backend",
                backend_adapter_kind="local",
                model_id="candidate/model",
                workspace_id=DEFAULT_WORKSPACE_ID,
                root=root,
            )
            baseline = record_file_input(
                input_kind="proposal",
                source_path=baseline_path,
                status="accepted",
                backend_id="baseline-backend",
                backend_display_name="Baseline Backend",
                backend_adapter_kind="local",
                model_id="baseline/model",
                workspace_id=DEFAULT_WORKSPACE_ID,
                root=root,
            )
            comparison = _comparison(
                comparison_id="comparison-cli",
                candidates=[
                    (candidate["event_id"], "candidate-backend", "candidate/model"),
                    (baseline["event_id"], "baseline-backend", "baseline/model"),
                ],
                winner_event_id=candidate["event_id"],
            )
            append_evaluation_comparison(
                evaluation_comparison_log_path(root=root, workspace_id=DEFAULT_WORKSPACE_ID),
                comparison,
                workspace_id=DEFAULT_WORKSPACE_ID,
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = satlab_main(
                    [
                        "--root",
                        str(root),
                        "backend",
                        "dossier",
                        "--comparison",
                        "comparison-cli",
                        "--candidate-backend",
                        "candidate-backend",
                        "--baseline-backend",
                        "baseline-backend",
                        "--format",
                        "json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["recommendation"]["value"], "adopt")

    def test_satlab_backend_dossier_strict_allows_tie_with_passing_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate, baseline = _recorded_proposal_pair(root)
            comparison = _comparison(
                comparison_id="comparison-cli-tie",
                candidates=[
                    (candidate["event_id"], "candidate-backend", "candidate/model"),
                    (baseline["event_id"], "baseline-backend", "baseline/model"),
                ],
                winner_event_id=None,
                outcome="tie",
            )
            append_evaluation_comparison(
                evaluation_comparison_log_path(root=root, workspace_id=DEFAULT_WORKSPACE_ID),
                comparison,
                workspace_id=DEFAULT_WORKSPACE_ID,
            )
            benchmark_report = root / "benchmark.json"
            benchmark_report.write_text(
                json.dumps({"schema_name": "fixture-benchmark", "passed": True}),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = satlab_main(
                    [
                        "--root",
                        str(root),
                        "backend",
                        "dossier",
                        "--comparison",
                        "comparison-cli-tie",
                        "--candidate-backend",
                        "candidate-backend",
                        "--baseline-backend",
                        "baseline-backend",
                        "--benchmark-report",
                        str(benchmark_report),
                        "--strict",
                        "--format",
                        "json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["recommendation"]["value"], "experiment_only")
        self.assertTrue(payload["benchmark_gate"]["passed"])

    def test_satlab_backend_dossier_strict_fails_missing_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate, baseline = _recorded_proposal_pair(root)
            comparison = _comparison(
                comparison_id="comparison-cli-missing-benchmark",
                candidates=[
                    (candidate["event_id"], "candidate-backend", "candidate/model"),
                    (baseline["event_id"], "baseline-backend", "baseline/model"),
                ],
                winner_event_id=candidate["event_id"],
            )
            append_evaluation_comparison(
                evaluation_comparison_log_path(root=root, workspace_id=DEFAULT_WORKSPACE_ID),
                comparison,
                workspace_id=DEFAULT_WORKSPACE_ID,
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = satlab_main(
                    [
                        "--root",
                        str(root),
                        "backend",
                        "dossier",
                        "--comparison",
                        "comparison-cli-missing-benchmark",
                        "--candidate-backend",
                        "candidate-backend",
                        "--baseline-backend",
                        "baseline-backend",
                        "--strict",
                        "--format",
                        "json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 1)
        self.assertIn("benchmark_gate_missing", payload["rule_evaluation"]["blockers"])

    def test_from_review_does_not_fallback_to_unrelated_latest_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            comparison, _events, _backend = _direct_adoptable_fixture(root)
            append_evaluation_comparison(
                evaluation_comparison_log_path(root=root, workspace_id=DEFAULT_WORKSPACE_ID),
                comparison,
                workspace_id=DEFAULT_WORKSPACE_ID,
            )

            dossier = build_backend_adoption_dossier_from_review(
                "missing-review-run",
                candidate_backend="candidate-backend",
                baseline_backend="baseline-backend",
                root=root,
                workspace_id=DEFAULT_WORKSPACE_ID,
            )

        self.assertEqual(dossier["comparison_id"], "review:missing-review-run")
        self.assertFalse(dossier["review_source"]["metadata_found"])
        self.assertEqual(dossier["recommendation"]["value"], "insufficient_evidence")
        self.assertIn("candidate_not_found", dossier["rule_evaluation"]["blockers"])

    def test_from_review_scopes_backend_match_to_review_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            review_metadata = root / "review.json"
            review_metadata.write_text(
                json.dumps(
                    {
                        "schema_name": "software-satellite-review-risk-report",
                        "event_id": "review-event",
                        "review_run_id": "review-run",
                        "generated_at_utc": "2026-01-01T00:00:00Z",
                    }
                ),
                encoding="utf-8",
            )
            related = _comparison(
                comparison_id="comparison-related",
                candidates=[
                    ("review-event", "candidate-backend", "candidate/model"),
                    ("related-baseline", "baseline-backend", "baseline/model"),
                ],
                winner_event_id="review-event",
            )
            related["recorded_at_utc"] = "2026-01-02T00:00:00Z"
            unrelated = _comparison(
                comparison_id="comparison-unrelated-newer",
                candidates=[
                    ("other-candidate", "candidate-backend", "candidate/model"),
                    ("other-baseline", "baseline-backend", "baseline/model"),
                ],
                winner_event_id="other-candidate",
            )
            unrelated["recorded_at_utc"] = "2026-01-03T00:00:00Z"
            for comparison in (related, unrelated):
                append_evaluation_comparison(
                    evaluation_comparison_log_path(root=root, workspace_id=DEFAULT_WORKSPACE_ID),
                    comparison,
                    workspace_id=DEFAULT_WORKSPACE_ID,
                )

            dossier = build_backend_adoption_dossier_from_review(
                str(review_metadata),
                candidate_backend="candidate-backend",
                baseline_backend="baseline-backend",
                root=root,
                workspace_id=DEFAULT_WORKSPACE_ID,
            )

        self.assertEqual(dossier["comparison_id"], "comparison-related")

    def test_from_review_malformed_explicit_path_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bad_review = root / "bad-review.json"
            bad_review.write_text("{not json", encoding="utf-8")

            with self.assertRaises(ValueError):
                build_backend_adoption_dossier_from_review(
                    str(bad_review),
                    candidate_backend="candidate-backend",
                    baseline_backend="baseline-backend",
                    root=root,
                    workspace_id=DEFAULT_WORKSPACE_ID,
                )


if __name__ == "__main__":
    unittest.main()
