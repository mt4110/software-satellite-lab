#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dogfood_workflows import (
    DOGFOOD_WORKFLOW_KINDS,
    format_dogfood_workflow_preview_report,
    record_dogfood_workflow_preview,
)
from evaluation_loop import CURATION_EXPORT_DECISIONS, CURATION_STATES
from gemma_runtime import repo_root
from recall_context import DEFAULT_CONTEXT_BUDGET_CHARS, DEFAULT_LIMIT
from workspace_state import DEFAULT_WORKSPACE_ID


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a preview-only dogfood workflow artifact from local recall and evaluation evidence.",
    )
    parser.add_argument("--root", type=Path, default=None, help="Optional repo root override.")
    parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id to inspect.")
    parser.add_argument(
        "--workflow-kind",
        choices=DOGFOOD_WORKFLOW_KINDS,
        required=True,
        help="Small dogfood workflow to preview.",
    )
    parser.add_argument("--query", default="", help="Primary recall query for the workflow.")
    parser.add_argument("--source-event-id", default="", help="Primary software-work event id.")
    parser.add_argument("--target-event-id", default="", help="Failure or follow-up event id for repair workflows.")
    parser.add_argument(
        "--candidate-event-id",
        action="append",
        default=[],
        help="Candidate event id for proposal comparison. Repeat for each proposal.",
    )
    parser.add_argument("--winner-event-id", default="", help="Optional winner candidate for comparison preview.")
    parser.add_argument("--file-hint", action="append", default=[], help="File hint for recall. Repeat as needed.")
    parser.add_argument("--criterion", action="append", default=[], help="Comparison criterion. Repeat as needed.")
    parser.add_argument(
        "--curation-state",
        action="append",
        choices=CURATION_STATES,
        default=[],
        help="Filter resolved-work curation preview candidates by state.",
    )
    parser.add_argument(
        "--curation-decision",
        action="append",
        choices=CURATION_EXPORT_DECISIONS,
        default=[],
        help="Filter resolved-work curation preview candidates by export decision.",
    )
    parser.add_argument(
        "--curation-reason",
        action="append",
        default=[],
        help="Filter resolved-work curation preview candidates by reason.",
    )
    parser.add_argument("--curation-limit", type=int, default=None, help="Maximum curation candidates to preview.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Recall candidate limit.")
    parser.add_argument(
        "--context-budget-chars",
        type=int,
        default=DEFAULT_CONTEXT_BUDGET_CHARS,
        help="Recall context budget.",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be a positive integer.")
    if args.context_budget_chars < 1:
        parser.error("--context-budget-chars must be a positive integer.")
    if args.curation_limit is not None and args.curation_limit < 1:
        parser.error("--curation-limit must be a positive integer.")

    root = _resolve_root(args.root)
    try:
        preview, latest_path, run_path = record_dogfood_workflow_preview(
            root=root,
            workspace_id=args.workspace_id,
            workflow_kind=args.workflow_kind,
            query_text=args.query.strip() or None,
            source_event_id=args.source_event_id.strip() or None,
            target_event_id=args.target_event_id.strip() or None,
            candidate_event_ids=args.candidate_event_id,
            winner_event_id=args.winner_event_id.strip() or None,
            file_hints=args.file_hint,
            criteria=args.criterion,
            curation_filters={
                "states": args.curation_state,
                "export_decisions": args.curation_decision,
                "reasons": args.curation_reason,
                "limit": args.curation_limit,
            },
            limit=args.limit,
            context_budget_chars=args.context_budget_chars,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.format == "json":
        payload = {
            "preview": preview,
            "workflow_preview_latest_path": str(latest_path),
            "workflow_preview_run_path": str(run_path),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_dogfood_workflow_preview_report(preview))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
