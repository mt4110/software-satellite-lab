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

from demand_validation import (  # noqa: E402
    DEMAND_VALIDATION_REPORT_SCHEMA_NAME,
    interview_log_path,
    interview_latest_path,
    record_demo_setup_metric,
    record_dogfood_validation_run,
    record_external_user_interview,
)
from failure_memory_review import build_failure_recall, record_file_input, record_human_verdict  # noqa: E402
from satlab import main as satlab_main  # noqa: E402


def _write_source(root: Path, name: str, text: str) -> Path:
    path = root / "fixtures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class DemandValidationTests(unittest.TestCase):
    def test_validation_report_passes_when_dogfood_and_external_gates_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            event_ids: list[str] = []
            for index in range(10):
                source = _write_source(
                    root,
                    f"event-{index}.log",
                    f"source-linked dogfood event {index} scripts/file_{index}.py\n",
                )
                result = record_file_input(
                    input_kind="failure" if index < 5 else "patch",
                    source_path=source,
                    note=f"Dogfood validation event {index}",
                    root=root,
                    refresh_index=False,
                )
                event_ids.append(result["event_id"])

            for index, event_id in enumerate(event_ids[:5]):
                _recall, _latest_path, recall_run_path = build_failure_recall(
                    query=f"source-linked dogfood event {index} scripts/file_{index}.py",
                    source_event_id=event_id,
                    root=root,
                )
                record_human_verdict(
                    verdict="reject" if index % 2 else "needs-review",
                    event_id=event_id,
                    reason=f"Dogfood verdict {index}",
                    root=root,
                )
                notes = _write_source(root, f"dogfood-notes-{index}.md", f"Useful recall notes {index}\n")
                record_dogfood_validation_run(
                    event_id=event_id,
                    recall_path=recall_run_path,
                    useful_recall="yes" if index < 2 else "no",
                    critical_false_evidence_count=0,
                    verdict_capture_seconds=20 + index,
                    notes_file=notes,
                    root=root,
                )

            for index in range(3):
                interview_notes = _write_source(
                    root,
                    f"interview-{index}.md",
                    "The user recognized repeated review failure memory as painful.\n",
                )
                record_external_user_interview(
                    participant_label=f"user-{index + 1}",
                    recognized_pain="yes",
                    wants_to_try="yes" if index == 0 else "no",
                    notes_file=interview_notes,
                    root=root,
                )

            partial_stdout = io.StringIO()
            with redirect_stdout(partial_stdout):
                partial_exit_code = satlab_main(
                    [
                        "--root",
                        str(root),
                        "validation",
                        "report",
                        "--format",
                        "json",
                    ]
                )
            partial_report = json.loads(partial_stdout.getvalue())

            setup_notes = _write_source(root, "setup-notes.md", "Clone-to-demo completed in 12 minutes.\n")
            record_demo_setup_metric(
                clone_to_demo_minutes=12,
                notes_file=setup_notes,
                root=root,
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = satlab_main(
                    [
                        "--root",
                        str(root),
                        "validation",
                        "report",
                        "--write",
                        "--format",
                        "json",
                    ]
                )
            report = json.loads(stdout.getvalue())
            latest_report = Path(report["paths"]["report_latest_json_path"])
            latest_markdown = Path(report["paths"]["report_latest_md_path"])
            run_report = Path(report["paths"]["report_run_json_path"])
            run_markdown = Path(report["paths"]["report_run_md_path"])
            latest_report_exists = latest_report.is_file()
            latest_markdown_exists = latest_markdown.is_file()
            run_report_exists = run_report.is_file()
            run_markdown_exists = run_markdown.is_file()
            run_report_stem = run_report.stem
            run_markdown_stem = run_markdown.stem

        clone_metric = next(
            metric
            for metric in partial_report["metrics"]
            if metric["key"] == "clone_to_demo_time"
        )
        self.assertEqual(partial_exit_code, 0)
        self.assertEqual(partial_report["continuation_gate"]["status"], "needs_data")
        self.assertEqual(clone_metric["status"], "needs_data")
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["schema_name"], DEMAND_VALIDATION_REPORT_SCHEMA_NAME)
        self.assertEqual(report["overall_status"], "pass")
        self.assertEqual(report["continuation_gate"]["status"], "pass")
        self.assertEqual(report["counts"]["event_count"], 10)
        self.assertEqual(report["counts"]["recall_run_count"], 5)
        self.assertEqual(report["counts"]["human_verdict_count"], 5)
        self.assertEqual(report["counts"]["dogfood_run_count"], 5)
        self.assertEqual(report["counts"]["interview_count"], 3)
        self.assertTrue(latest_report_exists)
        self.assertTrue(latest_markdown_exists)
        self.assertTrue(run_report_exists)
        self.assertTrue(run_markdown_exists)
        self.assertEqual(run_report_stem, run_markdown_stem)

    def test_validation_template_cli_writes_markdown_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            output_dir = Path("validation-templates")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = satlab_main(
                    [
                        "--root",
                        str(root),
                        "validation",
                        "template",
                        "--output-dir",
                        str(output_dir),
                        "--format",
                        "json",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            dogfood_template_exists = Path(payload["template_paths"]["dogfood_run_notes"]).is_file()
            interview_template_exists = Path(payload["template_paths"]["external_user_interview"]).is_file()
            setup_template_exists = Path(payload["template_paths"]["setup_timing"]).is_file()

        self.assertEqual(exit_code, 0)
        self.assertTrue(dogfood_template_exists)
        self.assertTrue(interview_template_exists)
        self.assertTrue(setup_template_exists)
        self.assertTrue(str(Path(payload["template_paths"]["dogfood_run_notes"])).startswith(str(root)))

    def test_dogfood_run_rejects_recall_for_a_different_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_source = _write_source(root, "first.log", "first dogfood failure\n")
            second_source = _write_source(root, "second.log", "second dogfood failure\n")
            first = record_file_input(
                input_kind="failure",
                source_path=first_source,
                note="First event",
                root=root,
            )
            second = record_file_input(
                input_kind="failure",
                source_path=second_source,
                note="Second event",
                root=root,
            )
            _recall, _latest_path, recall_run_path = build_failure_recall(
                query="first dogfood failure",
                source_event_id=first["event_id"],
                root=root,
            )

            with self.assertRaisesRegex(ValueError, "does not match"):
                record_dogfood_validation_run(
                    event_id=second["event_id"],
                    recall_path=recall_run_path,
                    useful_recall="yes",
                    root=root,
                )

    def test_interview_log_rejects_unexpected_header_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            notes = _write_source(root, "interview-notes.md", "Interview notes.\n")
            log_path = interview_log_path(root=root)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                json.dumps(
                    {
                        "schema_name": "unexpected-log",
                        "schema_version": 1,
                        "workspace_id": "local-default",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unexpected validation log schema"):
                record_external_user_interview(
                    participant_label="user-1",
                    recognized_pain="yes",
                    wants_to_try="yes",
                    notes_file=notes,
                    root=root,
                )
            latest_path_exists = interview_latest_path(root=root).exists()

        self.assertFalse(latest_path_exists)


if __name__ == "__main__":
    unittest.main()
