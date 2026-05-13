from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from artifact_vault import capture_artifact  # noqa: E402
from evidence_graph import (  # noqa: E402
    NODE_KINDS,
    RELATION_KINDS,
    build_evidence_graph,
    build_evidence_impact_report,
    build_evidence_trace,
    format_evidence_trace_markdown,
    target_identity_for_event,
    validate_evidence_graph_snapshot,
)


def _captured_ref(root: Path, name: str = "evidence.diff", text: str | None = None) -> dict[str, Any]:
    source = root / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        text
        or "\n".join(
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
    file_hints: list[str] | None = None,
    evidence_types: list[str] | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if ref is not None:
        options["artifact_vault_refs"] = [ref]
    if quality_status is not None:
        options["quality_status"] = quality_status
    if file_hints is not None:
        options["file_hints"] = file_hints
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
        "content": {"prompt": "Review SSL-123 patch", "output_text": "ok", "notes": [], "options": options},
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
        "polarity": "positive" if kind in {"acceptance", "review_resolved", "test_pass"} else "negative",
        "recorded_at_utc": recorded_at_utc,
        "origin": "satlab_cli",
        "source": {"source_event_id": event_id},
        "relation": {"relation_kind": None, "target_event_id": None},
        "evidence": {"rationale": "human checked"},
        "tags": ["human-verdict"],
    }


class EvidenceGraphTests(unittest.TestCase):
    def test_default_graph_build_does_not_write_event_log_or_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            graph = build_evidence_graph(root=root)

            event_log_path = root / "artifacts" / "event_logs" / "local-default.jsonl"
            index_path = root / "artifacts" / "indexes" / "local-default-software-memory-v1.sqlite3"

            self.assertEqual(graph["source_of_truth_policy"]["does_not_edit_event_logs"], True)
            self.assertFalse(event_log_path.exists())
            self.assertFalse(index_path.exists())

    def test_schema_enums_stay_aligned_with_graph_contract(self) -> None:
        schema = json.loads((REPO_ROOT / "schemas" / "evidence_graph.schema.json").read_text(encoding="utf-8"))
        node_kind_enum = set(schema["properties"]["nodes"]["items"]["properties"]["node_kind"]["enum"])
        relation_kind_enum = set(schema["properties"]["edges"]["items"]["properties"]["relation_kind"]["enum"])

        self.assertEqual(node_kind_enum, NODE_KINDS)
        self.assertEqual(relation_kind_enum, RELATION_KINDS)

    def test_graph_rebuild_is_deterministic_when_generated_at_is_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            events = [_event(event_id="event-prior", ref=ref)]

            first = build_evidence_graph(events=events, root=root, generated_at_utc="2026-05-12T00:00:00+00:00")
            second = build_evidence_graph(events=events, root=root, generated_at_utc="2026-05-12T00:01:00+00:00")

        self.assertEqual(first["graph_digest"], second["graph_digest"])
        self.assertEqual(first["nodes"], second["nodes"])
        self.assertEqual(first["edges"], second["edges"])
        self.assertNotEqual(first["generated_at_utc"], second["generated_at_utc"])
        self.assertEqual(validate_evidence_graph_snapshot(first), [])
        event_node = next(node for node in first["nodes"] if node["node_kind"] == "event")
        self.assertNotIn("checked_at_utc", event_node["metadata"]["support_kernel"])
        self.assertEqual(
            event_node["metadata"]["support_kernel_checked_at_policy"],
            "omitted_for_deterministic_derived_graph",
        )

    def test_artifact_event_verdict_relation_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            events = [_event(event_id="event-reviewed", ref=ref)]
            signals = [_signal("signal-accept", "acceptance", "event-reviewed", "2026-05-12T00:00:00+00:00")]

            graph = build_evidence_graph(
                events=events,
                signals=signals,
                root=root,
                generated_at_utc="2026-05-12T01:00:00+00:00",
            )

        relations = {(edge["relation_kind"], edge["from_node_id"].split("_")[1], edge["to_node_id"].split("_")[1]) for edge in graph["edges"]}
        self.assertTrue(any(edge["relation_kind"] == "uses_artifact" for edge in graph["edges"]))
        self.assertTrue(any(edge["relation_kind"] == "evaluates" for edge in graph["edges"]))
        self.assertTrue(relations)

    def test_recall_relation_stores_support_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            current_ref = _captured_ref(root, name="current.diff")
            prior_ref = _captured_ref(
                root,
                name="prior.diff",
                text="\n".join(
                    [
                        "diff --git a/scripts/bar.py b/scripts/bar.py",
                        "--- a/scripts/bar.py",
                        "+++ b/scripts/bar.py",
                        "@@ -1,1 +1,2 @@",
                        "+verified prior evidence",
                    ]
                )
                + "\n",
            )
            events = [
                _event(event_id="event-current", ref=current_ref, recorded_at_utc="2026-05-12T00:00:00+00:00"),
                _event(event_id="event-prior", ref=prior_ref, recorded_at_utc="2026-05-10T00:00:00+00:00"),
            ]
            recall = {
                "schema_name": "software-satellite-failure-memory-recall",
                "schema_version": 1,
                "generated_at_utc": "2026-05-12T00:10:00+00:00",
                "request": {"source_event_id": "event-current", "recorded_before_utc": "2026-05-12T00:00:00+00:00"},
                "bundle": {"selected_count": 1, "selected_candidates": [{"event_id": "event-prior", "reasons": ["file-match"]}]},
            }

            graph = build_evidence_graph(events=events, recalls=[recall], root=root)

        recall_edges = [edge for edge in graph["edges"] if edge["relation_kind"] == "recalls"]
        self.assertEqual(len(recall_edges), 1)
        self.assertEqual(recall_edges[0]["support_class"], "source_linked_prior")
        self.assertTrue(recall_edges[0]["can_support_decision"])
        self.assertEqual(recall_edges[0]["causal_validity"], "valid")

    def test_self_recall_edge_becomes_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            events = [_event(event_id="event-current", ref=ref, recorded_at_utc="2026-05-12T00:00:00+00:00")]
            recall = {
                "schema_name": "software-satellite-failure-memory-recall",
                "schema_version": 1,
                "generated_at_utc": "2026-05-12T00:10:00+00:00",
                "request": {"source_event_id": "event-current", "recorded_before_utc": "2026-05-12T00:00:00+00:00"},
                "bundle": {"selected_count": 1, "selected_candidates": [{"event_id": "event-current", "reasons": ["pinned"]}]},
            }

            graph = build_evidence_graph(events=events, recalls=[recall], root=root)

        edge = next(edge for edge in graph["edges"] if edge["relation_kind"] == "recalls")
        self.assertEqual(edge["causal_validity"], "invalid")
        self.assertEqual(edge["support_class"], "current_review_subject")
        self.assertFalse(edge["can_support_decision"])

    def test_future_evidence_edge_becomes_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            events = [
                _event(event_id="event-current", ref=ref, recorded_at_utc="2026-05-12T00:00:00+00:00"),
                _event(event_id="event-future", ref=ref, recorded_at_utc="2026-05-13T00:00:00+00:00"),
            ]
            recall = {
                "schema_name": "software-satellite-failure-memory-recall",
                "schema_version": 1,
                "generated_at_utc": "2026-05-12T00:10:00+00:00",
                "request": {"source_event_id": "event-current", "recorded_before_utc": "2026-05-12T00:00:00+00:00"},
                "bundle": {"selected_count": 1, "selected_candidates": [{"event_id": "event-future", "reasons": ["file-match"]}]},
            }

            graph = build_evidence_graph(events=events, recalls=[recall], root=root)

        edge = next(edge for edge in graph["edges"] if edge["relation_kind"] == "recalls")
        self.assertEqual(edge["causal_validity"], "invalid")
        self.assertEqual(edge["support_class"], "future_evidence")
        self.assertFalse(edge["can_support_decision"])

    def test_impact_report_lists_affected_events_for_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            events = [_event(event_id="event-impact", ref=ref, file_hints=["scripts/foo.py"])]
            graph = build_evidence_graph(events=events, root=root)

            report = build_evidence_impact_report("scripts/foo.py", graph=graph)

        self.assertEqual(report["affected_event_count"], 1)
        self.assertEqual(report["affected_events"][0]["event_id"], "event-impact")

    def test_trace_report_shows_why_evidence_can_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            events = [_event(event_id="event-trace", ref=ref)]
            graph = build_evidence_graph(events=events, root=root)

            trace = build_evidence_trace("event-trace", graph=graph, why_blocked=True)
            markdown = format_evidence_trace_markdown(trace)

        self.assertTrue(trace["support"]["can_support_decision"])
        self.assertIn("Support class: source_linked_prior", markdown)

    def test_target_identity_hashes_paths_hunks_and_issue_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            patch = "\n".join(
                [
                    "diff --git a/scripts/foo.py b/scripts/foo.py",
                    "--- a/scripts/foo.py",
                    "+++ b/scripts/foo.py",
                    "@@ -10,2 +10,3 @@ def run():",
                    "+return True",
                ]
            )
            ref = _captured_ref(root, text=patch)
            event = _event(event_id="event-target", ref=ref)

            identity = target_identity_for_event(event, root=root)

        self.assertIsNotNone(identity["target_fingerprint"])
        self.assertIn("scripts/foo.py", identity["changed_files"])
        self.assertEqual(identity["issue_keys"], ["SSL-123"])


if __name__ == "__main__":
    unittest.main()
