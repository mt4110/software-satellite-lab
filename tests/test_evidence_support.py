from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from artifact_vault import capture_artifact  # noqa: E402
from evidence_support import (  # noqa: E402
    FAILURE_EVIDENCE_TYPES,
    NEGATIVE_STATUSES,
    POSITIVE_EVIDENCE_TYPES,
    POSITIVE_STATUSES,
    SUPPORT_BLOCKER_REASONS,
    SUPPORT_CLASSES,
    SUPPORT_DECISION_REQUIREMENTS,
    SUPPORT_POLARITIES,
    SUPPORT_WARNING_REASONS,
    UNRESOLVED_STATUSES,
    build_evidence_support_result,
    build_support_policy_report,
    load_support_policy_registry,
    support_policy_registry_path,
    validate_support_policy_registry,
)
from failure_memory_review import record_file_input  # noqa: E402
from satlab import main as satlab_main  # noqa: E402


def _event(
    *,
    event_id: str,
    ref: dict[str, Any] | None,
    status: str = "ok",
    recorded_at_utc: str = "2026-05-11T00:00:00+00:00",
    notes: list[str] | None = None,
    tags: list[str] | None = None,
    quality_status: str | None = "pass",
    session_surface: str = "chat",
    event_kind: str = "chat_run",
    evidence_types: list[str] | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {}
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
        "event_kind": event_kind,
        "recorded_at_utc": recorded_at_utc,
        "workspace": {"workspace_id": "local-default"},
        "session": {"session_id": "s1", "surface": session_surface, "mode": "review"},
        "outcome": {"status": status, "quality_status": quality_status, "execution_status": status},
        "content": {"notes": notes or [], "options": options},
        "source_refs": {},
        "tags": tags or [],
    }


def _captured_ref(root: Path, name: str = "evidence.txt") -> dict[str, Any]:
    source = root / name
    source.write_text("verified local evidence\n", encoding="utf-8")
    return capture_artifact(source, kind="review_note", root=root)


def _class_policy(registry: dict[str, Any], support_class: str) -> dict[str, Any]:
    for item in registry["support_classes"]:
        if item["support_class"] == support_class:
            return item
    raise AssertionError(f"Missing support class policy: {support_class}")


class EvidenceSupportTests(unittest.TestCase):
    def test_support_policy_registry_mirrors_runtime_kernel(self) -> None:
        registry = load_support_policy_registry()
        issues = validate_support_policy_registry(registry)
        classes = {item["support_class"]: item for item in registry["support_classes"]}

        self.assertEqual(issues, [])
        self.assertEqual(set(classes), SUPPORT_CLASSES)
        self.assertEqual(set(registry["support_polarities"]), SUPPORT_POLARITIES)
        self.assertTrue(classes["source_linked_prior"]["can_support_decision"])
        self.assertEqual(classes["source_linked_prior"]["allowed_decision_polarities"], ["positive"])
        self.assertTrue(classes["negative_prior"]["can_support_decision"])
        self.assertEqual(classes["negative_prior"]["allowed_decision_polarities"], ["negative", "risk"])
        self.assertFalse(classes["missing_source"]["can_support_decision"])
        self.assertIn("missing_source", classes["missing_source"]["default_blockers"])

    def test_support_policy_schema_enums_match_runtime_kernel(self) -> None:
        schema_path = REPO_ROOT / "schemas" / "evidence_support_policy_registry.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        class_enum = (
            schema["properties"]["support_classes"]["items"]["properties"]["support_class"]["enum"]
        )
        polarity_enum = schema["properties"]["support_polarities"]["items"]["enum"]
        class_properties = schema["properties"]["support_classes"]["items"]["properties"]
        requirement_enum = (
            schema["properties"]["decision_requirements"]["items"]["properties"]["requirement_id"]["enum"]
        )
        signals = schema["properties"]["signals"]["properties"]

        self.assertEqual(set(class_enum), SUPPORT_CLASSES)
        self.assertEqual(set(polarity_enum), SUPPORT_POLARITIES)
        self.assertEqual(set(requirement_enum), SUPPORT_DECISION_REQUIREMENTS)
        self.assertEqual(schema["properties"]["support_polarities"]["minItems"], len(SUPPORT_POLARITIES))
        self.assertEqual(schema["properties"]["support_polarities"]["maxItems"], len(SUPPORT_POLARITIES))
        self.assertEqual(
            schema["properties"]["decision_requirements"]["minItems"],
            len(SUPPORT_DECISION_REQUIREMENTS),
        )
        self.assertEqual(
            schema["properties"]["decision_requirements"]["maxItems"],
            len(SUPPORT_DECISION_REQUIREMENTS),
        )
        self.assertEqual(schema["properties"]["support_classes"]["minItems"], len(SUPPORT_CLASSES))
        self.assertEqual(schema["properties"]["support_classes"]["maxItems"], len(SUPPORT_CLASSES))
        self.assertEqual(set(class_properties["default_blockers"]["items"]["enum"]), SUPPORT_BLOCKER_REASONS)
        self.assertEqual(set(schema["properties"]["blocker_reasons"]["items"]["enum"]), SUPPORT_BLOCKER_REASONS)
        self.assertEqual(schema["properties"]["blocker_reasons"]["minItems"], len(SUPPORT_BLOCKER_REASONS))
        self.assertEqual(schema["properties"]["blocker_reasons"]["maxItems"], len(SUPPORT_BLOCKER_REASONS))
        self.assertEqual(set(schema["properties"]["warning_reasons"]["items"]["enum"]), SUPPORT_WARNING_REASONS)
        self.assertEqual(schema["properties"]["warning_reasons"]["minItems"], len(SUPPORT_WARNING_REASONS))
        self.assertEqual(schema["properties"]["warning_reasons"]["maxItems"], len(SUPPORT_WARNING_REASONS))
        self.assertEqual(set(signals["positive_statuses"]["items"]["enum"]), POSITIVE_STATUSES)
        self.assertEqual(signals["positive_statuses"]["minItems"], len(POSITIVE_STATUSES))
        self.assertEqual(signals["positive_statuses"]["maxItems"], len(POSITIVE_STATUSES))
        self.assertEqual(set(signals["negative_statuses"]["items"]["enum"]), NEGATIVE_STATUSES)
        self.assertEqual(signals["negative_statuses"]["minItems"], len(NEGATIVE_STATUSES))
        self.assertEqual(signals["negative_statuses"]["maxItems"], len(NEGATIVE_STATUSES))
        self.assertEqual(set(signals["unresolved_statuses"]["items"]["enum"]), UNRESOLVED_STATUSES)
        self.assertEqual(signals["unresolved_statuses"]["minItems"], len(UNRESOLVED_STATUSES))
        self.assertEqual(signals["unresolved_statuses"]["maxItems"], len(UNRESOLVED_STATUSES))
        self.assertEqual(set(signals["positive_evidence_types"]["items"]["enum"]), POSITIVE_EVIDENCE_TYPES)
        self.assertEqual(signals["positive_evidence_types"]["minItems"], len(POSITIVE_EVIDENCE_TYPES))
        self.assertEqual(signals["positive_evidence_types"]["maxItems"], len(POSITIVE_EVIDENCE_TYPES))
        self.assertEqual(set(signals["failure_evidence_types"]["items"]["enum"]), FAILURE_EVIDENCE_TYPES)
        self.assertEqual(signals["failure_evidence_types"]["minItems"], len(FAILURE_EVIDENCE_TYPES))
        self.assertEqual(signals["failure_evidence_types"]["maxItems"], len(FAILURE_EVIDENCE_TYPES))

    def test_support_policy_report_maps_classes_to_decision_use(self) -> None:
        report = build_support_policy_report()
        classes = {item["support_class"]: item for item in report["support_classes"]}

        self.assertEqual(report["schema_name"], "software-satellite-evidence-support-policy-report")
        self.assertTrue(report["registry_valid"])
        self.assertEqual(classes["source_linked_prior"]["decision_use"], "positive")
        self.assertEqual(classes["negative_prior"]["decision_use"], "negative, risk")
        self.assertEqual(classes["unverified_agent_claim"]["decision_use"], "diagnostic_only")

    def test_satlab_evidence_policy_cli_json(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = satlab_main(["evidence", "policy", "--format", "json"])
        report = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["policy_id"], "evidence_support_v1")
        self.assertTrue(report["registry_valid"])
        self.assertTrue(report["registry_path"].endswith("configs/evidence_support_policies/v1.json"))

    def test_invalid_support_policy_registry_fails_validation(self) -> None:
        registry = load_support_policy_registry()
        registry["shell"] = "echo should-not-run"
        registry["support_classes"][0]["support_class"] = "made_up_support_class"
        registry["support_classes"][0]["extra"] = True
        missing_source_policy = _class_policy(registry, "missing_source")
        missing_source_index = registry["support_classes"].index(missing_source_policy)
        missing_source_policy["default_blockers"] = ["missing_source"]
        registry["decision_requirements"][0]["extra"] = True
        registry["signals"]["positive_statuses"] = ["accepted"]
        registry["signals"]["extra"] = ["ignored"]
        registry["default_paths"]["extra"] = False

        issues = validate_support_policy_registry(registry)

        issue_paths = {issue["path"] for issue in issues}
        self.assertIn("$.shell", issue_paths)
        self.assertIn("$.support_classes[0].support_class", issue_paths)
        self.assertIn("$.support_classes[0].extra", issue_paths)
        self.assertIn(f"$.support_classes[{missing_source_index}].default_blockers", issue_paths)
        self.assertIn("$.decision_requirements[0].extra", issue_paths)
        self.assertIn("$.signals.positive_statuses", issue_paths)
        self.assertIn("$.signals.extra", issue_paths)
        self.assertIn("$.default_paths.extra", issue_paths)

    def test_support_policy_registry_rejects_duplicate_values(self) -> None:
        registry = load_support_policy_registry()
        registry["support_polarities"].append("positive")
        registry["decision_requirements"].append(dict(registry["decision_requirements"][0]))
        source_policy = _class_policy(registry, "source_linked_prior")
        source_index = registry["support_classes"].index(source_policy)
        source_policy["allowed_decision_polarities"].append("positive")
        missing_source_policy = _class_policy(registry, "missing_source")
        missing_source_index = registry["support_classes"].index(missing_source_policy)
        missing_source_policy["default_blockers"].append("missing_source")
        registry["blocker_reasons"].append("missing_source")
        registry["warning_reasons"].append("manual_pin_diagnostic")
        registry["signals"]["positive_statuses"].append("accepted")

        issues = validate_support_policy_registry(registry)

        issue_paths = {issue["path"] for issue in issues}
        self.assertIn("$.support_polarities", issue_paths)
        self.assertIn("$.decision_requirements[8].requirement_id", issue_paths)
        self.assertIn(f"$.support_classes[{source_index}].allowed_decision_polarities", issue_paths)
        self.assertIn(f"$.support_classes[{missing_source_index}].default_blockers", issue_paths)
        self.assertIn("$.blocker_reasons", issue_paths)
        self.assertIn("$.warning_reasons", issue_paths)
        self.assertIn("$.signals.positive_statuses", issue_paths)

    def test_support_policy_registry_covers_missing_vault_object_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            (root / ref["vault_path"]).unlink()
            event = _event(event_id="event-missing-vault-object", ref=ref)

            result = build_evidence_support_result(
                "event-missing-vault-object",
                event=event,
                requested_polarity="positive",
                root=root,
            )
            registry = load_support_policy_registry()

        self.assertFalse(result["can_support_decision"])
        self.assertEqual(result["support_class"], "missing_source")
        self.assertIn("missing_vault_object", result["blockers"])
        self.assertTrue(set(result["blockers"]).issubset(set(registry["blocker_reasons"])))
        missing_source_policy = _class_policy(registry, "missing_source")
        self.assertIn("missing_vault_object", missing_source_policy["default_blockers"])

    def test_loading_invalid_support_policy_registry_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            policy_path = root / "bad-policy.json"
            registry = load_support_policy_registry()
            registry["support_polarities"] = ["positive"]
            policy_path.write_text(json.dumps(registry), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Invalid support policy registry"):
                load_support_policy_registry(policy_path)

    def test_satlab_evidence_policy_cli_rejects_invalid_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            policy_path = root / "bad-policy.json"
            registry = load_support_policy_registry()
            registry["default_paths"]["network_calls"] = True
            policy_path.write_text(json.dumps(registry), encoding="utf-8")

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    satlab_main(["evidence", "policy", "--policy", str(policy_path)])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("Invalid support policy registry", stderr.getvalue())

    def test_default_support_decision_does_not_require_policy_registry_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.assertFalse(support_policy_registry_path(root=root).exists())
            ref = _captured_ref(root)
            event = _event(event_id="event-policy-independent", ref=ref)

            result = build_evidence_support_result(
                "event-policy-independent",
                event=event,
                requested_polarity="positive",
                root=root,
            )

        self.assertTrue(result["can_support_decision"])
        self.assertEqual(result["support_class"], "source_linked_prior")

    def test_source_linked_prior_can_support_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(event_id="event-prior", ref=ref)

            result = build_evidence_support_result(
                "event-prior",
                event=event,
                requested_polarity="positive",
                review_started_at="2026-05-12T00:00:00+00:00",
                root=root,
                checked_at_utc="2026-05-12T01:00:00+00:00",
            )

        self.assertTrue(result["can_support_decision"])
        self.assertEqual(result["support_class"], "source_linked_prior")
        self.assertEqual(result["support_polarity"], "positive")
        self.assertEqual(result["blockers"], [])

    def test_ok_status_without_verification_signal_cannot_support_positive_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(event_id="event-ok-only", ref=ref, status="ok", quality_status=None)

            result = build_evidence_support_result("event-ok-only", event=event, requested_polarity="positive", root=root)

        self.assertFalse(result["can_support_decision"])
        self.assertEqual(result["support_class"], "unknown")
        self.assertIn("missing_positive_signal", result["blockers"])

    def test_missing_source_cannot_support_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            event = _event(event_id="event-missing", ref=None)

            result = build_evidence_support_result("event-missing", event=event, requested_polarity="positive", root=root)

        self.assertFalse(result["can_support_decision"])
        self.assertEqual(result["support_class"], "missing_source")
        self.assertIn("missing_source", result["blockers"])

    def test_outside_workspace_artifact_ref_cannot_support_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            root = base / "workspace"
            root.mkdir()
            outside = base / "outside.txt"
            outside.write_text("outside workspace\n", encoding="utf-8")
            ref = capture_artifact(root / ".." / "outside.txt", kind="review_note", root=root)
            event = _event(event_id="event-outside", ref=ref)

            result = build_evidence_support_result("event-outside", event=event, requested_polarity="positive", root=root)

        self.assertFalse(result["can_support_decision"])
        self.assertEqual(result["support_class"], "missing_source")
        self.assertIn("outside_workspace", result["blockers"])

    def test_current_review_subject_cannot_support_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(event_id="event-current", ref=ref)

            result = build_evidence_support_result(
                "event-current",
                event=event,
                active_subject="event-current",
                requested_polarity="positive",
                root=root,
            )

        self.assertFalse(result["can_support_decision"])
        self.assertTrue(result["active_review_excluded"])
        self.assertEqual(result["support_class"], "current_review_subject")

    def test_current_review_subject_can_match_vault_object_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(event_id="event-current-object", ref=ref)
            active_subject = str(root / ref["vault_path"])

            result = build_evidence_support_result(
                "event-current-object",
                event=event,
                active_subject=active_subject,
                requested_polarity="positive",
                root=root,
            )

        self.assertFalse(result["can_support_decision"])
        self.assertTrue(result["active_review_excluded"])
        self.assertEqual(result["support_class"], "current_review_subject")

    def test_current_review_subject_can_match_absolute_source_path_for_relative_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "relative-evidence.txt"
            source.write_text("verified local evidence\n", encoding="utf-8")
            ref = capture_artifact("relative-evidence.txt", kind="review_note", root=root)
            event = _event(event_id="event-current-relative-source", ref=ref)

            result = build_evidence_support_result(
                "event-current-relative-source",
                event=event,
                active_subject=str(source),
                requested_polarity="positive",
                root=root,
            )

        self.assertFalse(result["can_support_decision"])
        self.assertTrue(result["active_review_excluded"])
        self.assertEqual(result["support_class"], "current_review_subject")

    def test_future_evidence_cannot_support_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(event_id="event-future", ref=ref, recorded_at_utc="2026-05-13T00:00:00+00:00")

            result = build_evidence_support_result(
                "event-future",
                event=event,
                review_started_at="2026-05-12T00:00:00+00:00",
                requested_polarity="positive",
                root=root,
            )

        self.assertFalse(result["can_support_decision"])
        self.assertEqual(result["support_class"], "future_evidence")

    def test_weak_match_cannot_support_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(event_id="event-weak", ref=ref, notes=["weak_match"])

            result = build_evidence_support_result("event-weak", event=event, requested_polarity="positive", root=root)

        self.assertFalse(result["can_support_decision"])
        self.assertEqual(result["support_class"], "weak_match")

    def test_unverified_agent_claim_cannot_support_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(
                event_id="event-agent",
                ref=ref,
                quality_status=None,
                session_surface="agent_lane",
                event_kind="agent_task_run",
            )

            result = build_evidence_support_result("event-agent", event=event, requested_polarity="positive", root=root)

        self.assertFalse(result["can_support_decision"])
        self.assertEqual(result["support_class"], "unverified_agent_claim")

    def test_contradictory_evidence_cannot_support_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(
                event_id="event-contradictory",
                ref=ref,
                evidence_types=["test_pass", "test_fail"],
            )

            result = build_evidence_support_result(
                "event-contradictory",
                event=event,
                requested_polarity="positive",
                root=root,
            )

        self.assertFalse(result["can_support_decision"])
        self.assertEqual(result["support_class"], "contradictory")

    def test_negative_prior_can_support_risk_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(
                event_id="event-negative",
                ref=ref,
                status="failed",
                quality_status="fail",
                evidence_types=["test_fail"],
            )

            result = build_evidence_support_result("event-negative", event=event, requested_polarity="risk", root=root)

        self.assertTrue(result["can_support_decision"])
        self.assertEqual(result["support_class"], "negative_prior")
        self.assertEqual(result["support_polarity"], "risk")

    def test_manual_pin_cannot_override_missing_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            event = _event(event_id="event-pin", ref=None, notes=["pinned"])

            result = build_evidence_support_result("event-pin", event=event, requested_polarity="positive", root=root)

        self.assertFalse(result["can_support_decision"])
        self.assertEqual(result["support_class"], "missing_source")
        self.assertIn("missing_source", result["blockers"])
        self.assertIn("manual_pin_diagnostic", result["warnings"])

    def test_support_result_is_deterministic_when_checked_at_is_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = _captured_ref(root)
            event = _event(event_id="event-deterministic", ref=ref)

            first = build_evidence_support_result(
                "event-deterministic",
                event=event,
                requested_polarity="positive",
                root=root,
                checked_at_utc="2026-05-12T01:00:00+00:00",
            )
            second = build_evidence_support_result(
                "event-deterministic",
                event=event,
                requested_polarity="positive",
                root=root,
                checked_at_utc="2026-05-12T01:00:00+00:00",
            )

        self.assertEqual(first, second)

    def test_satlab_evidence_support_cli_uses_vault_ref_from_file_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            failure = root / "failure.log"
            failure.write_text("test failed: support kernel should carry risk note\n", encoding="utf-8")
            recorded = record_file_input(input_kind="failure", source_path=failure, root=root)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = satlab_main(
                    [
                        "--root",
                        str(root),
                        "evidence",
                        "support",
                        "--event",
                        recorded["event_id"],
                        "--polarity",
                        "risk",
                        "--format",
                        "json",
                    ]
                )
            result = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(result["can_support_decision"])
        self.assertEqual(result["support_class"], "negative_prior")
        self.assertEqual(result["artifact_refs"], [recorded["artifact_vault_refs"][0]["artifact_id"]])

    def test_file_input_symlink_ref_is_not_supporting_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "failure.log"
            target.write_text("test failed through symlink\n", encoding="utf-8")
            link = root / "failure-link.log"
            try:
                link.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlink setup unavailable: {exc}")
            recorded = record_file_input(input_kind="failure", source_path=link, root=root)

            result = build_evidence_support_result(
                recorded["event_id"],
                requested_polarity="risk",
                root=root,
            )

        self.assertEqual(recorded["artifact_vault_refs"][0]["source_state"], "symlink_refused")
        self.assertFalse(result["can_support_decision"])
        self.assertIn("symlink_refused", result["blockers"])


if __name__ == "__main__":
    unittest.main()
