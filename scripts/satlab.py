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
    format_verdict_result,
    record_file_input,
    record_human_verdict,
    run_review_risk_pack,
)
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
                print(json.dumps(template, ensure_ascii=False, indent=2))
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
