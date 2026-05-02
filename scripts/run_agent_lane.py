#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_lane import (
    TASK_KINDS,
    build_agent_lane_snapshot,
    build_agent_task,
    format_agent_lane_snapshot_report,
    record_agent_lane_snapshot,
    record_agent_run,
    record_agent_task,
    run_agent_task,
)
from gemma_runtime import repo_root
from memory_index import rebuild_memory_index
from workspace_state import DEFAULT_WORKSPACE_ID


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record and run a bounded M5 agent-lane patch-plan-verify task.",
    )
    parser.add_argument("--root", type=Path, default=None, help="Optional repo root override.")
    parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id to write.")
    parser.add_argument("--task-title", required=True, help="Short title for the bounded task.")
    parser.add_argument("--goal", required=True, help="Goal or intent for the task.")
    parser.add_argument(
        "--task-kind",
        choices=TASK_KINDS,
        default="patch_plan_verify",
        help="Agent-lane task kind.",
    )
    parser.add_argument(
        "--scope-path",
        action="append",
        default=[],
        help="Path considered in scope for the task. Repeat to include more than one.",
    )
    parser.add_argument(
        "--plan-step",
        action="append",
        default=[],
        help="Plan step to preserve in the task trace. Repeat to include more than one.",
    )
    parser.add_argument(
        "--verification-command",
        action="append",
        default=[],
        help="Verification command to run without a shell. Repeat to include more than one.",
    )
    parser.add_argument(
        "--acceptance-criterion",
        action="append",
        default=[],
        help="Human-readable acceptance criterion. Repeat to include more than one.",
    )
    parser.add_argument(
        "--pass-definition",
        default="",
        help="Definition used to treat the verification as passing.",
    )
    parser.add_argument("--result-summary", default="", help="Optional result summary override.")
    parser.add_argument("--timeout-seconds", type=int, default=60, help="Timeout for each verification command.")
    parser.add_argument(
        "--skip-index-rebuild",
        action="store_true",
        help="Do not rebuild the software-work event index after recording the run.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for the recorded task/run.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = _resolve_root(args.root)
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be at least 1.")

    try:
        task = build_agent_task(
            workspace_id=args.workspace_id,
            title=args.task_title,
            goal=args.goal,
            task_kind=args.task_kind,
            origin="cli",
            scope_paths=args.scope_path,
            plan_steps=args.plan_step,
            verification_commands=args.verification_command,
            acceptance_criteria=args.acceptance_criterion,
            pass_definition=args.pass_definition.strip() or None,
            tags=["m5", "agent_lane"],
        )
        recorded_task = record_agent_task(
            task,
            root=root,
            workspace_id=args.workspace_id,
        )
        run = run_agent_task(
            recorded_task,
            root=root,
            origin="cli",
            result_summary=args.result_summary.strip() or None,
            timeout_seconds=args.timeout_seconds,
        )
        recorded_run, run_path = record_agent_run(
            run,
            root=root,
            workspace_id=args.workspace_id,
        )
    except ValueError as exc:
        parser.error(str(exc))

    index_summary: dict[str, object] | None = None
    if not args.skip_index_rebuild:
        index_summary = rebuild_memory_index(root=root, workspace_id=args.workspace_id)

    snapshot, latest_path, snapshot_run_path = record_agent_lane_snapshot(
        root=root,
        workspace_id=args.workspace_id,
    )
    if args.format == "json":
        payload: dict[str, object] = {
            "task": recorded_task,
            "run": recorded_run,
            "run_artifact_path": str(run_path),
            "snapshot": snapshot,
            "snapshot_latest_path": str(latest_path),
            "snapshot_run_path": str(snapshot_run_path),
        }
        if index_summary is not None:
            payload["index_summary"] = index_summary
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        outcome = recorded_run.get("outcome") if isinstance(recorded_run.get("outcome"), dict) else {}
        print(f"Agent task: {recorded_task['task_id']}")
        print(f"Agent run: {recorded_run['run_id']}")
        print(f"Status: {recorded_run['status']}")
        print(f"Summary: {outcome.get('result_summary') or 'n/a'}")
        print(f"Run artifact: {run_path}")
        if index_summary is not None:
            print(f"Software-work events: {index_summary.get('event_count')}")
        print()
        print(format_agent_lane_snapshot_report(build_agent_lane_snapshot(root=root, workspace_id=args.workspace_id)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
