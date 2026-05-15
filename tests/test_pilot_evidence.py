from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pilot_evidence import (  # noqa: E402
    PILOT_REPORT_SCHEMA_NAME,
    build_pilot_report,
    pilot_evidence_ledger_path,
    record_pilot_demo,
    record_pilot_interview,
    record_pilot_loi,
    record_pilot_report,
    validate_pilot_evidence_ledger,
)
from satlab import main as satlab_main  # noqa: E402


def _write_source(root: Path, name: str, text: str) -> Path:
    path = root / "fixtures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class PilotEvidenceTests(unittest.TestCase):
    def test_full_pilot_gate_passes_from_local_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            for index in range(20):
                participant = f"pilot-user-{index + 1:02d}"
                notes = _write_source(root, f"interview-{index}.md", "Interview notes.\n")
                record_pilot_interview(
                    participant_label=participant,
                    participant_segment="security-lead" if index < 5 else "technical-founder",
                    notes_file=notes,
                    exact_pain_recognized="yes" if index < 12 else "no",
                    wants_to_try="yes" if index < 5 else "no",
                    willing_to_install_locally="yes" if index < 12 else "unclear",
                    willingness_to_pay="yes" if index < 5 else "unclear",
                    security_sensitive_user="yes" if index < 5 else "no",
                    no_code_upload_matters="yes",
                    budget_path="yes" if index < 2 else "unclear",
                    root=root,
                )

            for index in range(5):
                participant = f"pilot-user-{index + 1:02d}"
                notes = _write_source(root, f"demo-{index}.md", "Demo notes.\n")
                record_pilot_demo(
                    participant_label=participant,
                    participant_segment="security-lead",
                    notes_file=notes,
                    failure_memory_reviewed=True,
                    signed_evidence_reviewed=True,
                    no_code_upload_confirmed=True,
                    transcript_claim_rejection_reviewed=True,
                    wants_to_try="yes",
                    exact_pain_recognized="yes",
                    security_sensitive_user="yes",
                    root=root,
                )

            for index, kind in enumerate(("loi", "paid-pilot")):
                participant = f"pilot-user-{index + 1:02d}"
                notes = _write_source(root, f"loi-{index}.md", "Commitment notes.\n")
                record_pilot_loi(
                    participant_label=participant,
                    participant_segment="security-lead",
                    notes_file=notes,
                    commitment_kind=kind,
                    amount="pilot budget approved",
                    budget_path="yes",
                    security_sensitive_user="yes",
                    root=root,
                )

            report, _markdown, latest_json, latest_md, run_json, run_md = record_pilot_report(root=root)
            ledger_exists = pilot_evidence_ledger_path(root=root).is_file()
            latest_json_exists = latest_json.is_file()
            latest_md_exists = latest_md.is_file()
            run_json_exists = run_json.is_file()
            run_md_exists = run_md.is_file()

        self.assertTrue(ledger_exists)
        self.assertTrue(latest_json_exists)
        self.assertTrue(latest_md_exists)
        self.assertTrue(run_json_exists)
        self.assertTrue(run_md_exists)
        self.assertEqual(report["schema_name"], PILOT_REPORT_SCHEMA_NAME)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["metrics"]["discovery_calls"], 20)
        self.assertEqual(report["metrics"]["hands_on_demos"], 5)
        self.assertEqual(report["metrics"]["security_sensitive_users"], 5)
        self.assertEqual(report["metrics"]["exact_pain_recognition"], 12)
        self.assertEqual(report["metrics"]["wants_to_try"], 5)
        self.assertEqual(report["metrics"]["paid_pilot_commitments_or_lois"], 2)
        self.assertFalse(report["paid_pilot_gate"]["do_not_build_team_registry"])

    def test_pilot_record_demo_cli_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            notes = _write_source(root, "demo.md", "Demo notes.\n")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = satlab_main(
                    [
                        "--root",
                        str(root),
                        "pilot",
                        "record-demo",
                        "--participant",
                        "pilot-user-1",
                        "--segment",
                        "platform-lead",
                        "--notes-file",
                        str(notes),
                        "--failure-memory",
                        "--signed-evidence",
                        "--no-code-upload",
                        "--transcript-claim-rejection",
                        "--wants-to-try",
                        "yes",
                        "--exact-pain",
                        "yes",
                        "--security-sensitive",
                        "yes",
                        "--format",
                        "json",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            ledger_exists = Path(payload["paths"]["pilot_evidence_ledger_path"]).is_file()

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["record_type"], "hands_on_demo")
        self.assertTrue(payload["score"]["demo_checklist_passed"])
        self.assertTrue(ledger_exists)

    def test_fixture_records_pass_paid_pilot_report_cli(self) -> None:
        fixture = Path(__file__).resolve().parents[1] / "examples" / "pilot_evidence" / "passing_gate_records.jsonl"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = satlab_main(
                [
                    "pilot",
                    "report",
                    "--fixture-records",
                    str(fixture),
                    "--format",
                    "json",
                ]
            )
        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "pass")
        self.assertEqual(payload["paid_pilot_gate"]["status"], "pass")

    def test_report_needs_data_before_commitments_and_blocks_team_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            report = build_pilot_report(root=root)

        self.assertEqual(report["status"], "needs_data")
        self.assertEqual(report["paid_pilot_gate"]["status"], "needs_data")
        self.assertTrue(report["paid_pilot_gate"]["do_not_build_team_registry"])
        self.assertIn("paid_pilot_commitments_or_lois", report["blockers"])

    def test_incomplete_demo_checklist_does_not_count_as_hands_on_demo_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            notes = _write_source(root, "demo.md", "Demo notes with one missed topic.\n")
            record_pilot_demo(
                participant_label="pilot-user-1",
                participant_segment="platform-lead",
                notes_file=notes,
                failure_memory_reviewed=True,
                signed_evidence_reviewed=True,
                no_code_upload_confirmed=True,
                transcript_claim_rejection_reviewed=False,
                wants_to_try="yes",
                exact_pain_recognized="yes",
                root=root,
            )
            report = build_pilot_report(root=root)

        self.assertEqual(report["metrics"]["hands_on_demos"], 0)
        self.assertIn("hands_on_demos", report["blockers"])

    def test_invalid_ledger_record_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            ledger = root / "bad-pilot-ledger.jsonl"
            ledger.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "schema_name": "software-satellite-paid-pilot-evidence-ledger",
                                "schema_version": 1,
                                "workspace_id": "local-default",
                            }
                        ),
                        json.dumps(
                            {
                                "schema_name": "paid_pilot_evidence_record",
                                "schema_version": 1,
                                "pilot_record_id": "bad",
                                "workspace_id": "local-default",
                                "recorded_at_utc": "2026-05-15T00:00:00+00:00",
                                "record_type": "discovery_call",
                                "participant_label": "pilot-user",
                                "participant_segment": "platform-lead",
                                "evidence": {},
                                "score": {},
                                "source_refs": {"notes_file_ref": {"path": "notes.md"}},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "guardrails"):
                validate_pilot_evidence_ledger(ledger, root=root)

    def test_record_requires_notes_file_ref_path(self) -> None:
        fixture = Path(__file__).resolve().parents[1] / "examples" / "pilot_evidence" / "passing_gate_records.jsonl"
        records = validate_pilot_evidence_ledger(fixture)
        broken = dict(records[0])
        broken["source_refs"] = {"notes_file_ref": {"role": "fixture"}}
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_fixture = Path(tmpdir) / "missing-source-path.json"
            bad_fixture.write_text(json.dumps(broken), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "notes_file_ref.path"):
                validate_pilot_evidence_ledger(bad_fixture)

    def test_json_fixture_rejects_non_object_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            fixture = root / "bad-records.json"
            fixture.write_text(json.dumps({"records": [123]}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not an object"):
                validate_pilot_evidence_ledger(fixture, root=root)

    def test_pilot_template_cli_writes_markdown_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = satlab_main(
                    [
                        "--root",
                        str(Path(__file__).resolve().parents[1]),
                        "pilot",
                        "template",
                        "--output-dir",
                        str(root / "pilot-templates"),
                        "--format",
                        "json",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            interview_template_exists = Path(payload["template_paths"]["pilot_interview_script"]).is_file()
            demo_template_exists = Path(payload["template_paths"]["pilot_demo_checklist"]).is_file()
            value_template_exists = Path(payload["template_paths"]["paid_pilot_statement_of_value"]).is_file()

        self.assertEqual(exit_code, 0)
        self.assertTrue(interview_template_exists)
        self.assertTrue(demo_template_exists)
        self.assertTrue(value_template_exists)

    def test_pilot_template_cli_prints_fillable_markdown(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = satlab_main(["pilot", "template"])
        output = stdout.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("## Interview Script", output)
        self.assertIn("## Demo Checklist", output)
        self.assertIn("python scripts/satlab.py pilot record-interview --help", output)


if __name__ == "__main__":
    unittest.main()
