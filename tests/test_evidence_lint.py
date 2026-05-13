from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from artifact_vault import capture_artifact  # noqa: E402
from evidence_graph import build_evidence_graph  # noqa: E402
from evidence_lint import build_evidence_lint_report  # noqa: E402


def _captured_ref(root: Path, name: str = "evidence.diff") -> dict[str, Any]:
    source = root / name
    source.write_text(
        "\n".join(
            [
                "diff --git a/scripts/foo.py b/scripts/foo.py",
                "--- a/scripts/foo.py",
                "+++ b/scripts/foo.py",
                "@@ -1,1 +1,2 @@",
                "+verified local evidence",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return capture_artifact(source, kind="patch", root=root, captured_at_utc="2026-05-10T00:00:00+00:00")


def _event(
    *,
    event_id: str,
    ref: dict[str, Any] | None,
    recorded_at_utc: str = "2026-05-11T00:00:00+00:00",
    status: str = "ok",
    quality_status: str | None = "pass",
    evidence_types: list[str] | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {"file_hints": ["scripts/foo.py"]}
    if ref is not None:
        options["artifact_vault_refs"] = [ref]
    if quality_status is not None:
        options["quality_status"] = quality_status
    if evidence_types is not None:
        options["evidence_types"] = evidence_types
    return {
        "schema_name": "software-satellite-event",
        "schema_version": 1,
        "event_id": event_id,
        "event_kind": "chat_run",
        "recorded_at_utc": recorded_at_utc,
        "workspace": {"workspace_id": "local-default"},
        "session": {"session_id": "s1", "surface": "chat", "mode": "review"},
        "outcome": {"status": status, "quality_status": quality_status, "execution_status": status},
        "content": {"prompt": "Review patch", "output_text": "ok", "notes": [], "options": options},
        "source_refs": {},
        "tags": [],
    }


def _signal(signal_id: str, kind: str, event_id: str, recorded_at_utc: str) -> dict[str, Any]:
    return {
        "schema_name": "software-satellite-evaluation-signal",
        "schema_version": 1,
        "signal_id": signal_id,
        "workspace_id": "local-default",
        "signal_kind": kind,
        "polarity": "positive" if kind == "acceptance" else "negative",
        "recorded_at_utc": recorded_at_utc,
        "origin": "satlab_cli",
        "source": {"source_event_id": event_id},
        "relation": {"relation_kind": None, "target_event_id": None},
        "evidence": {"rationale": "human checked"},
        "tags": ["human-verdict"],
    }


def _graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema_name": "software-satellite-derived-evidence-graph",
        "schema_version": 1,
        "workspace_id": "local-default",
        "generated_at_utc": "2026-05-12T00:00:00+00:00",
        "derived": True,
        "rebuildable": True,
        "source_of_truth_policy": {
            "graph_is_derived": True,
            "does_not_edit_event_logs": True,
            "does_not_edit_artifact_refs": True,
            "does_not_edit_verdicts": True,
            "does_not_edit_comparison_records": True,
        },
        "target_identity_model": {
            "algorithm": "sha256(normalized changed file paths + hunk headers + optional issue/task key)",
            "version": 1,
        },
        "source_paths": {},
        "counts": {
            "node_count": len(nodes),
            "edge_count": len(edges or []),
            "support_kernel_decisions_match_graph_nodes": True,
            "learning_preview_graph_blocker_count": 0,
        },
        "index_summary": None,
        "graph_digest": "fixture",
        "nodes": nodes,
        "edges": edges or [],
    }


class EvidenceLintTests(unittest.TestCase):
    def test_strict_lint_passes_on_clean_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            graph = build_evidence_graph(events=[_event(event_id="event-clean", ref=ref)], root=root)

            report = build_evidence_lint_report(graph, strict=True, generated_at_utc="2026-05-12T00:00:00+00:00")

        self.assertEqual(report["verdict"], "pass")
        self.assertEqual(report["hard_fail_count"], 0)
        self.assertTrue(report["exit_gate"]["support_kernel_decisions_match_graph_nodes"])

    def test_missing_source_positive_support_fails_lint(self) -> None:
        graph = _graph(
            [
                {
                    "node_id": "node_event_bad",
                    "node_kind": "event",
                    "source_id": "event-bad",
                    "support_class": "missing_source",
                    "can_support_decision": True,
                    "polarity": "positive",
                    "quality_status": "unknown",
                    "target_fingerprint": "target_x",
                    "created_at_utc": "2026-05-12T00:00:00+00:00",
                    "metadata": {},
                }
            ]
        )

        report = build_evidence_lint_report(graph, strict=True)

        self.assertEqual(report["verdict"], "fail")
        self.assertIn("missing_source_positive_support", {issue["rule_id"] for issue in report["issues"]})

    def test_current_review_subject_positive_support_fails_lint(self) -> None:
        graph = _graph(
            [
                {
                    "node_id": "node_event_current",
                    "node_kind": "event",
                    "source_id": "event-current",
                    "support_class": "current_review_subject",
                    "can_support_decision": True,
                    "polarity": "positive",
                    "quality_status": "unknown",
                    "target_fingerprint": "target_x",
                    "created_at_utc": "2026-05-12T00:00:00+00:00",
                    "metadata": {},
                }
            ]
        )

        report = build_evidence_lint_report(graph, strict=True)

        self.assertEqual(report["verdict"], "fail")
        self.assertIn("current_review_subject_prior_support", {issue["rule_id"] for issue in report["issues"]})

    def test_contradiction_edge_blocks_positive_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(event_id="event-contradiction", ref=ref)
            graph = build_evidence_graph(
                events=[event],
                signals=[
                    _signal("signal-accept", "acceptance", "event-contradiction", "2026-05-11T01:00:00+00:00"),
                    _signal("signal-reject", "rejection", "event-contradiction", "2026-05-12T01:00:00+00:00"),
                ],
                root=root,
            )

            report = build_evidence_lint_report(graph, strict=True)

        self.assertEqual(report["verdict"], "fail")
        self.assertIn("contradictory_verdicts_promoted", {issue["rule_id"] for issue in report["issues"]})

    def test_later_positive_resolution_prevents_contradiction_hard_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(event_id="event-resolved", ref=ref)
            graph = build_evidence_graph(
                events=[event],
                signals=[
                    _signal("signal-accept-old", "acceptance", "event-resolved", "2026-05-11T01:00:00+00:00"),
                    _signal("signal-reject", "rejection", "event-resolved", "2026-05-12T01:00:00+00:00"),
                    _signal("signal-accept-new", "acceptance", "event-resolved", "2026-05-13T01:00:00+00:00"),
                ],
                root=root,
            )

            report = build_evidence_lint_report(graph, strict=True)

        contradiction_edges = [edge for edge in graph["edges"] if edge["relation_kind"] == "contradicts"]
        self.assertTrue(contradiction_edges)
        self.assertTrue(all(edge["causal_validity"] == "invalid" for edge in contradiction_edges))
        self.assertEqual(report["verdict"], "pass")
        self.assertNotIn("contradictory_verdicts_promoted", {issue["rule_id"] for issue in report["issues"]})

    def test_learning_candidate_with_missing_source_fails_lint(self) -> None:
        learning_preview = {
            "schema_name": "software-satellite-learning-dataset-preview",
            "schema_version": 1,
            "generated_at_utc": "2026-05-12T00:00:00+00:00",
            "review_queue": [
                {
                    "queue_item_id": "queue-missing",
                    "event_id": "event-missing",
                    "queue_state": "missing_source",
                    "blocked_reason": "missing_source_event",
                    "blocked_reasons": ["missing_source_event"],
                    "eligible_for_supervised_candidate": False,
                }
            ],
        }

        graph = build_evidence_graph(events=[], learning_previews=[learning_preview])
        report = build_evidence_lint_report(graph, strict=True)

        self.assertEqual(report["verdict"], "fail")
        self.assertIn("learning_candidate_missing_source", {issue["rule_id"] for issue in report["issues"]})
        self.assertGreater(report["counts"]["learning_preview_graph_blocker_count"], 0)

    def test_pack_output_cannot_bypass_support_kernel(self) -> None:
        graph = _graph(
            [
                {
                    "node_id": "node_review_pack",
                    "node_kind": "review",
                    "source_id": "pack-output",
                    "support_class": "source_linked_prior",
                    "can_support_decision": True,
                    "polarity": "positive",
                    "quality_status": "verified",
                    "target_fingerprint": "target_x",
                    "created_at_utc": "2026-05-12T00:00:00+00:00",
                    "metadata": {"created_by": "pack", "source_kind": "pack_output"},
                }
            ]
        )

        report = build_evidence_lint_report(graph, strict=True)

        self.assertEqual(report["verdict"], "fail")
        self.assertIn("pack_output_bypasses_support_kernel", {issue["rule_id"] for issue in report["issues"]})

    def test_stale_benchmark_report_warns_without_hard_fail(self) -> None:
        graph = _graph(
            [
                {
                    "node_id": "node_review_benchmark",
                    "node_kind": "review",
                    "source_id": "review-benchmark-latest",
                    "support_class": "source_linked_prior",
                    "can_support_decision": False,
                    "polarity": "neutral",
                    "quality_status": "verified",
                    "target_fingerprint": "target_x",
                    "created_at_utc": "2026-05-01T00:00:00+00:00",
                    "metadata": {"source_kind": "review_benchmark"},
                }
            ]
        )

        report = build_evidence_lint_report(
            graph,
            strict=True,
            generated_at_utc="2026-05-12T00:00:00+00:00",
        )

        self.assertEqual(report["verdict"], "pass")
        self.assertIn("stale_benchmark_report", {issue["rule_id"] for issue in report["issues"]})


if __name__ == "__main__":
    unittest.main()
