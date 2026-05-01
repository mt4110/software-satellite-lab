#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation_loop import (
    COMPARISON_OUTCOMES,
    CURATION_EXPORT_DECISIONS,
    CURATION_STATES,
    RELATION_KINDS,
    SIGNAL_KINDS,
    append_evaluation_comparison,
    append_evaluation_signal,
    build_evaluation_comparison,
    build_evaluation_signal,
    evaluation_comparison_log_path,
    evaluation_signal_log_path,
    format_curation_export_preview_report,
    format_evaluation_snapshot_report,
    record_curation_export_preview,
    record_evaluation_snapshot,
)
from gemma_runtime import repo_root
from memory_index import rebuild_memory_index
from software_work_events import read_event_log
from workspace_state import DEFAULT_WORKSPACE_ID


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def _events_by_id_and_index_summary(
    *,
    root: Path,
    workspace_id: str,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    summary = rebuild_memory_index(root=root, workspace_id=workspace_id)
    event_log = read_event_log(Path(summary["event_log_path"]))
    events_by_id = {
        str(event.get("event_id")): event
        for event in event_log.get("events") or []
        if isinstance(event, dict) and event.get("event_id")
    }
    return events_by_id, dict(summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record M4 evaluation signals and write a local evaluation snapshot.",
    )
    parser.add_argument("--root", type=Path, default=None, help="Optional repo root override.")
    parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id to read.")
    parser.add_argument(
        "--record-signal",
        action="store_true",
        help="Append one explicit evaluation signal before writing the snapshot.",
    )
    parser.add_argument(
        "--record-comparison",
        action="store_true",
        help="Append one comparison record before writing the snapshot.",
    )
    parser.add_argument(
        "--curation-preview",
        action="store_true",
        help="Write a preview-only curation export artifact next to the evaluation snapshot.",
    )
    parser.add_argument(
        "--curation-state",
        action="append",
        choices=CURATION_STATES,
        default=[],
        help="Filter curation preview candidates by state. Repeat to include more than one.",
    )
    parser.add_argument(
        "--curation-decision",
        action="append",
        choices=CURATION_EXPORT_DECISIONS,
        default=[],
        help="Filter curation preview candidates by export decision. Repeat to include more than one.",
    )
    parser.add_argument(
        "--curation-reason",
        action="append",
        default=[],
        help="Filter curation preview candidates by reason. Repeat to include more than one.",
    )
    parser.add_argument(
        "--curation-limit",
        type=int,
        default=None,
        help="Maximum number of filtered curation preview candidates to include.",
    )
    parser.add_argument(
        "--signal-kind",
        choices=SIGNAL_KINDS,
        default=None,
        help="Signal kind for --record-signal.",
    )
    parser.add_argument(
        "--source-event-id",
        default="",
        help="Software-work event id that the signal evaluates.",
    )
    parser.add_argument(
        "--target-event-id",
        default="",
        help="Failure event id when the signal repairs or follows up on earlier work.",
    )
    parser.add_argument(
        "--relation-kind",
        choices=RELATION_KINDS,
        default=None,
        help="Optional linkage from source event to target event.",
    )
    parser.add_argument("--rationale", default="", help="Short human-readable evidence note.")
    parser.add_argument("--test-name", default="", help="Optional test or check name.")
    parser.add_argument("--test-command", default="", help="Optional test command.")
    parser.add_argument("--failure-summary", default="", help="Optional failure summary.")
    parser.add_argument("--review-id", default="", help="Optional review or thread id for review-resolution signals.")
    parser.add_argument("--review-url", default="", help="Optional review URL for review-resolution signals.")
    parser.add_argument("--resolution-summary", default="", help="Optional review resolution summary.")
    parser.add_argument(
        "--candidate-event-id",
        action="append",
        default=[],
        help="Candidate event id for --record-comparison. Repeat at least twice.",
    )
    parser.add_argument("--winner-event-id", default="", help="Winner event id for --record-comparison.")
    parser.add_argument(
        "--comparison-outcome",
        choices=COMPARISON_OUTCOMES,
        default=None,
        help="Outcome for --record-comparison.",
    )
    parser.add_argument("--comparison-label", default="", help="Short task label for --record-comparison.")
    parser.add_argument(
        "--criterion",
        action="append",
        default=[],
        help="Comparison criterion. Repeat to add more than one.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for the written snapshot.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = _resolve_root(args.root)
    recorded_signal: dict[str, object] | None = None
    recorded_comparison: dict[str, object] | None = None
    curation_preview: dict[str, object] | None = None
    curation_preview_latest_path: Path | None = None
    curation_preview_run_path: Path | None = None
    events_by_id: dict[str, dict[str, object]] | None = None
    index_summary: dict[str, object] | None = None

    try:
        if args.record_signal:
            if args.signal_kind is None:
                parser.error("--record-signal requires --signal-kind.")
            source_event_id = args.source_event_id.strip()
            if not source_event_id:
                parser.error("--record-signal requires --source-event-id.")
            if events_by_id is None:
                events_by_id, index_summary = _events_by_id_and_index_summary(
                    root=root,
                    workspace_id=args.workspace_id,
                )
            source_event = events_by_id.get(source_event_id)
            if source_event is None:
                raise ValueError(f"Unknown evaluation source_event_id `{source_event_id}`.")
            evidence = {
                key: value
                for key, value in {
                    "test_name": args.test_name.strip() or None,
                    "validation_command": args.test_command.strip() or None,
                    "failure_summary": args.failure_summary.strip() or None,
                    "review_id": args.review_id.strip() or None,
                    "review_url": args.review_url.strip() or None,
                    "resolution_summary": args.resolution_summary.strip() or None,
                }.items()
                if value is not None
            }
            recorded_signal = build_evaluation_signal(
                workspace_id=args.workspace_id,
                signal_kind=args.signal_kind,
                source_event_id=source_event_id,
                source_event=source_event,
                target_event_id=args.target_event_id.strip() or None,
                relation_kind=args.relation_kind,
                rationale=args.rationale.strip() or None,
                evidence=evidence,
                origin="cli",
            )
            append_evaluation_signal(
                evaluation_signal_log_path(workspace_id=args.workspace_id, root=root),
                recorded_signal,
                workspace_id=args.workspace_id,
            )

        if args.record_comparison:
            if events_by_id is None:
                events_by_id, index_summary = _events_by_id_and_index_summary(
                    root=root,
                    workspace_id=args.workspace_id,
                )
            recorded_comparison = build_evaluation_comparison(
                workspace_id=args.workspace_id,
                candidate_event_ids=args.candidate_event_id,
                winner_event_id=args.winner_event_id.strip() or None,
                outcome=args.comparison_outcome,
                task_label=args.comparison_label.strip() or None,
                criteria=args.criterion,
                rationale=args.rationale.strip() or None,
                origin="cli",
                events_by_id=events_by_id,
            )
            append_evaluation_comparison(
                evaluation_comparison_log_path(workspace_id=args.workspace_id, root=root),
                recorded_comparison,
                workspace_id=args.workspace_id,
            )
    except ValueError as exc:
        parser.error(str(exc))

    snapshot, _latest_path, _run_path = record_evaluation_snapshot(
        root=root,
        workspace_id=args.workspace_id,
        index_summary=index_summary,
    )
    if args.curation_preview:
        curation_preview, curation_preview_latest_path, curation_preview_run_path = record_curation_export_preview(
            root=root,
            workspace_id=args.workspace_id,
            snapshot=snapshot,
            filters={
                "states": args.curation_state,
                "export_decisions": args.curation_decision,
                "reasons": args.curation_reason,
                "limit": args.curation_limit,
            },
        )
    if args.format == "json":
        payload: dict[str, object] = {"snapshot": snapshot}
        if recorded_signal is not None:
            payload["recorded_signal"] = recorded_signal
        if recorded_comparison is not None:
            payload["recorded_comparison"] = recorded_comparison
        if curation_preview is not None:
            payload["curation_preview"] = curation_preview
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        blocks: list[str] = []
        if recorded_signal is not None:
            blocks.append(
                "Recorded signal: "
                f"{recorded_signal.get('signal_kind')} "
                f"{((recorded_signal.get('source') or {}) if isinstance(recorded_signal.get('source'), dict) else {}).get('source_event_id')}"
            )
        if recorded_comparison is not None:
            blocks.append(
                "Recorded comparison: "
                f"{recorded_comparison.get('outcome')} "
                f"winner={recorded_comparison.get('winner_event_id') or 'n/a'}"
            )
        blocks.append(format_evaluation_snapshot_report(snapshot))
        if curation_preview is not None:
            blocks.append(format_curation_export_preview_report(curation_preview))
            blocks.append(
                "Curation preview written: "
                f"{curation_preview_run_path or curation_preview_latest_path or 'n/a'}"
            )
        print("\n\n".join(blocks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
