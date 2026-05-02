#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend_swap import (
    DEFAULT_WORKFLOW_KIND,
    check_backend_compatibility,
    ensure_backend_config_files,
    ensure_default_backend_configs,
    format_backend_harness_report,
    run_backend_swap_harness,
)
from gemma_runtime import repo_root
from workspace_state import DEFAULT_WORKSPACE_ID


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an M6 backend-swap side-by-side path through the same agent-lane workflow.",
    )
    parser.add_argument("--root", type=Path, default=None, help="Optional repo root override.")
    parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id to write.")
    parser.add_argument(
        "--backend-id",
        action="append",
        default=[],
        help="Backend config id to include. Repeat at least twice. Defaults to the two local mock configs.",
    )
    parser.add_argument(
        "--backend-config-json",
        action="append",
        type=Path,
        default=[],
        help="File-first backend config JSON to load into the workspace config log before listing or running.",
    )
    parser.add_argument("--task-title", default="", help="Short title for the shared workflow.")
    parser.add_argument("--goal", default="", help="Goal or intent for the shared workflow.")
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
        help="Plan step preserved in each backend run. Repeat to include more than one.",
    )
    parser.add_argument(
        "--verification-command",
        action="append",
        default=[],
        help="Verification command to run without a shell for each backend. Repeat to include more than one.",
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
        help="Definition used to treat verification as passing.",
    )
    parser.add_argument(
        "--comparison-label",
        default="",
        help="Optional task label for the M4 comparison record.",
    )
    parser.add_argument(
        "--workflow-kind",
        default=DEFAULT_WORKFLOW_KIND,
        help="Backend compatibility workflow contract.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=60, help="Timeout for each verification command.")
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="Ensure default backend configs, print compatibility metadata, and exit.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = _resolve_root(args.root)
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be at least 1.")

    try:
        if args.backend_config_json:
            ensure_backend_config_files(
                args.backend_config_json,
                root=root,
                workspace_id=args.workspace_id,
            )

        if args.list_backends:
            configs = ensure_default_backend_configs(root=root, workspace_id=args.workspace_id)
            compatibilities = [
                check_backend_compatibility(config, workflow_kind=args.workflow_kind)
                for config in configs
            ]
            if args.format == "json":
                print(json.dumps({"backends": configs, "compatibilities": compatibilities}, ensure_ascii=False, indent=2))
            else:
                print("Backend configs")
                for config, compatibility in zip(configs, compatibilities, strict=True):
                    print(
                        "- "
                        + f"{config['backend_id']} "
                        + f"model={config['model_id']} "
                        + f"adapter={config['adapter_kind']} "
                        + f"compatibility={compatibility['status']}"
                    )
            return 0

        if len(args.backend_id) == 1:
            parser.error("--backend-id must be repeated at least twice for side-by-side runs.")
        if not args.task_title.strip():
            parser.error("--task-title is required unless --list-backends is used.")
        if not args.goal.strip():
            parser.error("--goal is required unless --list-backends is used.")

        harness_run, harness_run_path = run_backend_swap_harness(
            root=root,
            workspace_id=args.workspace_id,
            task_title=args.task_title.strip(),
            goal=args.goal.strip(),
            scope_paths=args.scope_path,
            plan_steps=args.plan_step,
            verification_commands=args.verification_command,
            acceptance_criteria=args.acceptance_criterion,
            pass_definition=args.pass_definition.strip() or None,
            backend_ids=args.backend_id,
            workflow_kind=args.workflow_kind,
            comparison_label=args.comparison_label.strip() or None,
            timeout_seconds=args.timeout_seconds,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.format == "json":
        print(
            json.dumps(
                {
                    "harness_run": harness_run,
                    "harness_run_artifact_path": str(harness_run_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(format_backend_harness_report(harness_run))
        print(f"Harness artifact: {harness_run_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
