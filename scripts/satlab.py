#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from failure_memory_review import (
    build_failure_recall,
    build_review_risk_report,
    build_verdict_template,
    event_contract_summary,
    format_failure_recall_report,
    format_ingest_result,
    format_proposal_comparison_result,
    format_verdict_template,
    format_verdict_result,
    record_file_input,
    record_human_verdict,
    record_proposal_comparison,
    run_review_risk_pack,
)
from demand_validation import (
    DEMAND_VALIDATION_JUDGEMENTS,
    build_demand_validation_report,
    demand_validation_templates_markdown,
    format_demand_validation_report,
    format_demo_setup_metric,
    format_dogfood_validation_run,
    format_external_user_interview,
    record_demo_setup_metric,
    record_demand_validation_report,
    record_dogfood_validation_run,
    record_external_user_interview,
    write_demand_validation_templates,
)
from evaluation_loop import format_learning_dataset_preview_report, record_learning_dataset_preview
from satellite_pack import (
    PackManifestError,
    audit_pack_path,
    format_pack_audit_report,
    format_pack_inspection_report,
    inspect_pack_path,
)
from workspace_state import DEFAULT_WORKSPACE_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="satlab",
        description="Local file-first utilities for software-satellite-lab.",
    )
    parser.add_argument("--root", type=Path, default=None, help="Optional repo root override.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    event_parser = subparsers.add_parser("event", help="Record file-first software-work events.")
    event_subparsers = event_parser.add_subparsers(dest="event_command", required=True)

    ingest_parser = event_subparsers.add_parser(
        "ingest",
        help="Record a patch, failure, repair, or review note as source-linked evidence.",
    )
    input_group = ingest_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--patch", type=Path, help="Patch or git diff file to record.")
    input_group.add_argument("--failure", type=Path, help="Failure log or failure note file to record.")
    input_group.add_argument("--proposal", type=Path, help="Proposal or candidate output file to record.")
    input_group.add_argument("--repair", type=Path, help="Repair note or repair artifact file to record.")
    input_group.add_argument("--review-note", type=Path, help="Accepted/rejected review note file to record.")
    ingest_parser.add_argument("--note", default="", help="Short note to attach to the event.")
    ingest_parser.add_argument("--status", default="", help="Optional event status override.")
    ingest_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    ingest_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    event_report_parser = event_subparsers.add_parser(
        "report",
        help="Report source-path contract status for software-work events.",
    )
    event_report_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    event_report_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    recall_parser = subparsers.add_parser("recall", help="Recall source-linked software-work memory.")
    recall_subparsers = recall_parser.add_subparsers(dest="recall_command", required=True)
    failure_recall_parser = recall_subparsers.add_parser(
        "failure",
        help="Recall similar prior failures for a patch or review query.",
    )
    failure_recall_parser.add_argument("--query", required=True, help="Failure-memory query text.")
    failure_recall_parser.add_argument("--patch", type=Path, default=None, help="Optional patch file for file hints.")
    failure_recall_parser.add_argument("--file-hint", action="append", default=[], help="Optional file hint. Repeatable.")
    failure_recall_parser.add_argument("--source-event-id", default="", help="Optional source event id to diagnose.")
    failure_recall_parser.add_argument("--status-filter", action="append", default=[], help="Optional status filter.")
    failure_recall_parser.add_argument("--limit", type=int, default=5, help="Maximum selected recall items.")
    failure_recall_parser.add_argument("--context-budget-chars", type=int, default=6000, help="Recall bundle budget.")
    failure_recall_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    failure_recall_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    compare_parser = subparsers.add_parser("compare", help="Record source-linked proposal comparisons.")
    compare_subparsers = compare_parser.add_subparsers(dest="compare_command", required=True)
    proposals_parser = compare_subparsers.add_parser(
        "proposals",
        help="Compare two or more proposal/candidate output files with a human rationale.",
    )
    proposals_parser.add_argument(
        "--candidate",
        type=Path,
        action="append",
        required=True,
        help="Candidate output file. Repeat at least twice.",
    )
    proposals_parser.add_argument(
        "--verdict",
        choices=("winner", "none", "tie", "needs-follow-up"),
        required=True,
        help="Human comparison verdict. Use `winner` with --winner-candidate or `none` when no candidate wins.",
    )
    proposals_parser.add_argument(
        "--winner-candidate",
        default="",
        help="Winning candidate path or 1-based index. Required for --verdict winner.",
    )
    proposals_parser.add_argument("--rationale", required=True, help="Human rationale for the comparison verdict.")
    proposals_parser.add_argument("--label", default="", help="Short task label for the comparison.")
    proposals_parser.add_argument("--criterion", action="append", default=[], help="Comparison criterion. Repeatable.")
    proposals_parser.add_argument(
        "--candidate-backend-id",
        action="append",
        default=[],
        help="Optional backend id for a candidate. Repeat once per candidate when provided.",
    )
    proposals_parser.add_argument(
        "--candidate-model-id",
        action="append",
        default=[],
        help="Optional model id for a candidate. Repeat once per candidate when provided.",
    )
    proposals_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    proposals_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    pack_parser = subparsers.add_parser("pack", help="Inspect and audit declarative Satellite Evidence Packs.")
    pack_subparsers = pack_parser.add_subparsers(dest="pack_command", required=True)

    inspect_parser = pack_subparsers.add_parser(
        "inspect",
        help="Load a Satellite Evidence Pack manifest and print its declarative shape.",
    )
    inspect_parser.add_argument("pack", type=Path, help="Manifest file or pack directory.")
    inspect_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    audit_parser = pack_subparsers.add_parser(
        "audit",
        help="Validate a Satellite Evidence Pack manifest and write a permission summary artifact.",
    )
    audit_parser.add_argument("pack", type=Path, help="Manifest file or pack directory.")
    audit_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    audit_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    run_parser = pack_subparsers.add_parser(
        "run",
        help="Run the built-in review-risk-pack workflow without enabling arbitrary pack execution.",
    )
    run_parser.add_argument("pack", help="Pack name/path. Only review-risk-pack is runnable in v0.")
    run_parser.add_argument("--patch", type=Path, required=True, help="Patch or git diff file to review.")
    run_parser.add_argument("--query", default="", help="Optional failure-memory query override.")
    run_parser.add_argument("--note", default="", help="Optional note for the patch input event.")
    run_parser.add_argument("--limit", type=int, default=5, help="Maximum selected recall items.")
    run_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    run_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")

    verdict_parser = subparsers.add_parser("verdict", help="Record or print a human verdict.")
    verdict_subparsers = verdict_parser.add_subparsers(dest="verdict_command", required=True)
    for name in ("accept", "reject", "resolve", "unresolve", "needs-review"):
        command_parser = verdict_subparsers.add_parser(name, help=f"Record a {name} verdict.")
        command_parser.add_argument("--event", required=True, help="Software-work event id.")
        command_parser.add_argument("--reason", required=True, help="Short human-readable reason.")
        command_parser.add_argument("--target-event", default="", help="Optional related prior event id.")
        command_parser.add_argument(
            "--relation-kind",
            choices=("repairs", "follow_up_for"),
            default=None,
            help="Optional relation to --target-event.",
        )
        command_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
        command_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    template_parser = verdict_subparsers.add_parser("template", help="Print a human verdict template.")
    template_parser.add_argument("--event", default="", help="Optional software-work event id.")
    template_parser.add_argument("--verdict", default="reject", help="Template verdict value.")
    template_parser.add_argument("--reason", default="", help="Optional template reason.")
    template_parser.add_argument("--format", choices=("text", "json"), default="json", help="Output format.")

    report_parser = subparsers.add_parser("report", help="Generate file-first review reports.")
    report_subparsers = report_parser.add_subparsers(dest="report_command", required=True)
    latest_parser = report_subparsers.add_parser(
        "latest",
        help="Generate the latest Markdown review-risk report.",
    )
    latest_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    latest_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")

    learning_parser = subparsers.add_parser("learning", help="Inspect preview-only learning candidates.")
    learning_subparsers = learning_parser.add_subparsers(dest="learning_command", required=True)
    inspect_learning_parser = learning_subparsers.add_parser(
        "inspect",
        help="Build and inspect a preview-only learning-candidate queue.",
    )
    inspect_learning_parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Required safety flag. Writes inspection artifacts only; no trainable export is produced.",
    )
    inspect_learning_parser.add_argument("--limit", type=int, default=None, help="Maximum eligible candidates to preview.")
    inspect_learning_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    inspect_learning_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    validation_parser = subparsers.add_parser(
        "validation",
        help="Record demand-validation dogfood and external-user evidence.",
    )
    validation_subparsers = validation_parser.add_subparsers(dest="validation_command", required=True)
    validation_template_parser = validation_subparsers.add_parser(
        "template",
        help="Print or write local demand-validation note templates.",
    )
    validation_template_parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory for Markdown templates.")
    validation_template_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    validation_run_parser = validation_subparsers.add_parser(
        "record-run",
        help="Record a dogfood run judgement for a recall-backed review event.",
    )
    validation_run_parser.add_argument("--event", required=True, help="Dogfood software-work event id.")
    validation_run_parser.add_argument("--recall", type=Path, default=None, help="Recall artifact path. Defaults to latest recall.")
    validation_run_parser.add_argument(
        "--useful-recall",
        choices=DEMAND_VALIDATION_JUDGEMENTS,
        required=True,
        help="Human judgement for whether recalled evidence helped.",
    )
    validation_run_parser.add_argument(
        "--critical-false-evidence-count",
        type=int,
        default=0,
        help="Count critical false-evidence cases found in this run.",
    )
    validation_run_parser.add_argument(
        "--verdict-capture-seconds",
        type=float,
        default=None,
        help="Seconds needed to capture the human verdict for this run.",
    )
    validation_run_parser.add_argument("--notes-file", type=Path, default=None, help="Optional dogfood notes file.")
    validation_run_parser.add_argument("--note", default="", help="Short note for the validation run.")
    validation_run_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    validation_run_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    validation_interview_parser = validation_subparsers.add_parser(
        "record-interview",
        help="Record external technical-user interview evidence from a local notes file.",
    )
    validation_interview_parser.add_argument("--participant", required=True, help="Participant label or anonymized id.")
    validation_interview_parser.add_argument(
        "--recognized-pain",
        choices=DEMAND_VALIDATION_JUDGEMENTS,
        required=True,
        help="Whether the user recognized the exact pain.",
    )
    validation_interview_parser.add_argument(
        "--wants-to-try",
        choices=DEMAND_VALIDATION_JUDGEMENTS,
        required=True,
        help="Whether the user wants to try it on a repo.",
    )
    validation_interview_parser.add_argument("--notes-file", type=Path, required=True, help="Interview notes file.")
    validation_interview_parser.add_argument("--note", default="", help="Short note for the interview record.")
    validation_interview_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    validation_interview_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    validation_setup_parser = validation_subparsers.add_parser(
        "record-setup",
        help="Record a timed local clone-to-demo setup measurement.",
    )
    validation_setup_parser.add_argument(
        "--clone-to-demo-minutes",
        type=float,
        required=True,
        help="Minutes from fresh clone/setup to demo report.",
    )
    validation_setup_parser.add_argument("--notes-file", type=Path, default=None, help="Optional setup timing notes file.")
    validation_setup_parser.add_argument("--note", default="", help="Short setup note.")
    validation_setup_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    validation_setup_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    validation_report_parser = validation_subparsers.add_parser(
        "report",
        help="Generate a demand-validation report from local dogfood, verdict, recall, and interview artifacts.",
    )
    validation_report_parser.add_argument("--write", action="store_true", help="Persist latest and run report artifacts.")
    validation_report_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    validation_report_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "pack" and args.pack_command == "inspect":
        try:
            inspection = inspect_pack_path(args.pack, root=args.root)
        except PackManifestError as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps({"inspection": inspection}, ensure_ascii=False, indent=2))
        else:
            print(format_pack_inspection_report(inspection))
        return 0 if inspection.get("schema_valid") else 1

    if args.command == "pack" and args.pack_command == "audit":
        try:
            audit, latest_path, run_path = audit_pack_path(
                args.pack,
                workspace_id=args.workspace_id,
                root=args.root,
            )
        except PackManifestError as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(
                json.dumps(
                    {
                        "audit": audit,
                        "audit_latest_path": str(latest_path),
                        "audit_run_path": str(run_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(format_pack_audit_report(audit))
        return 1 if audit.get("verdict") == "block" else 0

    if args.command == "event" and args.event_command == "ingest":
        input_kind = "patch"
        source_path = args.patch
        if args.failure is not None:
            input_kind = "failure"
            source_path = args.failure
        elif args.proposal is not None:
            input_kind = "proposal"
            source_path = args.proposal
        elif args.repair is not None:
            input_kind = "repair"
            source_path = args.repair
        elif args.review_note is not None:
            input_kind = "review_note"
            source_path = args.review_note
        try:
            result = record_file_input(
                input_kind=input_kind,
                source_path=source_path,
                note=args.note.strip() or None,
                status=args.status.strip() or None,
                workspace_id=args.workspace_id,
                root=args.root,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_ingest_result(result))
        return 0

    if args.command == "event" and args.event_command == "report":
        summary = event_contract_summary(workspace_id=args.workspace_id, root=args.root)
        if args.format == "json":
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            contract = summary["event_contract"]
            metrics = summary["validation_metrics"]
            source_metric = metrics["source_path_completeness"]
            print(
                "\n".join(
                    [
                        f"Workspace: {summary['workspace_id']}",
                        f"Events checked: {contract.get('checked_event_count')}",
                        f"Failed events: {contract.get('failed_event_count')}",
                        f"Source path completeness: {source_metric.get('observed')} (target {source_metric.get('target')})",
                        f"Event log: {summary['event_log_path']}",
                    ]
                )
            )
        return 0

    if args.command == "recall" and args.recall_command == "failure":
        try:
            recall, _latest_path, _run_path = build_failure_recall(
                query=args.query,
                patch_path=args.patch,
                file_hints=args.file_hint,
                limit=args.limit,
                context_budget_chars=args.context_budget_chars,
                workspace_id=args.workspace_id,
                root=args.root,
                source_event_id=args.source_event_id.strip() or None,
                status_filters=args.status_filter,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(recall, ensure_ascii=False, indent=2))
        else:
            print(format_failure_recall_report(recall))
        return 0

    if args.command == "compare" and args.compare_command == "proposals":
        try:
            result, _comparison_log = record_proposal_comparison(
                candidate_paths=args.candidate,
                verdict=args.verdict,
                winner_candidate=args.winner_candidate.strip() or None,
                rationale=args.rationale,
                task_label=args.label.strip() or None,
                criteria=args.criterion,
                candidate_backend_ids=args.candidate_backend_id,
                candidate_model_ids=args.candidate_model_id,
                workspace_id=args.workspace_id,
                root=args.root,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_proposal_comparison_result(result))
        return 0

    if args.command == "pack" and args.pack_command == "run":
        try:
            metadata, markdown, _latest_path, _run_path = run_review_risk_pack(
                pack=args.pack,
                patch_path=args.patch,
                query=args.query.strip() or None,
                note=args.note.strip() or None,
                limit=args.limit,
                workspace_id=args.workspace_id,
                root=args.root,
            )
        except (PackManifestError, ValueError) as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
        else:
            print(markdown)
        return 0

    if args.command == "learning" and args.learning_command == "inspect":
        if not args.preview_only:
            parser.error("learning inspect requires --preview-only.")
        try:
            preview, latest_path, run_path = record_learning_dataset_preview(
                workspace_id=args.workspace_id,
                root=args.root,
                limit=args.limit,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.format == "json":
            payload = {
                "learning_preview": preview,
                "learning_preview_latest_path": str(latest_path),
                "learning_preview_run_path": str(run_path),
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(format_learning_dataset_preview_report(preview))
        return 0

    if args.command == "validation":
        if args.validation_command == "template":
            if args.output_dir is not None:
                paths = write_demand_validation_templates(args.output_dir)
                if args.format == "json":
                    print(json.dumps({"template_paths": paths}, ensure_ascii=False, indent=2))
                else:
                    print("\n".join([f"{name}: {path}" for name, path in paths.items()]))
                return 0
            template = demand_validation_templates_markdown()
            if args.format == "json":
                print(json.dumps({"template_markdown": template}, ensure_ascii=False, indent=2))
            else:
                print(template)
            return 0

        if args.validation_command == "record-run":
            try:
                result, _latest_path, _run_path, _log_path = record_dogfood_validation_run(
                    event_id=args.event,
                    useful_recall=args.useful_recall,
                    critical_false_evidence_count=args.critical_false_evidence_count,
                    verdict_capture_seconds=args.verdict_capture_seconds,
                    recall_path=args.recall,
                    note=args.note.strip() or None,
                    notes_file=args.notes_file,
                    workspace_id=args.workspace_id,
                    root=args.root,
                )
            except ValueError as exc:
                parser.error(str(exc))
            if args.format == "json":
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(format_dogfood_validation_run(result))
            return 0

        if args.validation_command == "record-interview":
            try:
                result, _latest_path, _run_path, _log_path = record_external_user_interview(
                    participant_label=args.participant,
                    recognized_pain=args.recognized_pain,
                    wants_to_try=args.wants_to_try,
                    notes_file=args.notes_file,
                    note=args.note.strip() or None,
                    workspace_id=args.workspace_id,
                    root=args.root,
                )
            except ValueError as exc:
                parser.error(str(exc))
            if args.format == "json":
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(format_external_user_interview(result))
            return 0

        if args.validation_command == "record-setup":
            try:
                result, _latest_path, _run_path = record_demo_setup_metric(
                    clone_to_demo_minutes=args.clone_to_demo_minutes,
                    note=args.note.strip() or None,
                    notes_file=args.notes_file,
                    workspace_id=args.workspace_id,
                    root=args.root,
                )
            except ValueError as exc:
                parser.error(str(exc))
            if args.format == "json":
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(format_demo_setup_metric(result))
            return 0

        if args.validation_command == "report":
            if args.write:
                report, markdown, _latest_json, _latest_md, _run_json, _run_md = record_demand_validation_report(
                    workspace_id=args.workspace_id,
                    root=args.root,
                )
            else:
                report = build_demand_validation_report(workspace_id=args.workspace_id, root=args.root)
                markdown = format_demand_validation_report(report)
            if args.format == "json":
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                print(markdown)
            return 0

    if args.command == "verdict":
        if args.verdict_command == "template":
            template = build_verdict_template(
                event_id=args.event.strip() or None,
                verdict=args.verdict,
                reason=args.reason.strip() or None,
            )
            if args.format == "json":
                print(json.dumps(template, ensure_ascii=False, indent=2))
            else:
                print(format_verdict_template(template))
            return 0
        try:
            result, _latest_path, _run_path = record_human_verdict(
                verdict=args.verdict_command,
                event_id=args.event,
                reason=args.reason,
                target_event_id=args.target_event.strip() or None,
                relation_kind=args.relation_kind,
                workspace_id=args.workspace_id,
                root=args.root,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_verdict_result(result))
        return 0

    if args.command == "report" and args.report_command == "latest":
        metadata, markdown, _latest_path, _run_path = build_review_risk_report(
            workspace_id=args.workspace_id,
            root=args.root,
        )
        if args.format == "json":
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
        else:
            print(markdown)
        return 0

    parser.error("Unsupported command.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
