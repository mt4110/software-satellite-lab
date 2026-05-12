#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from failure_memory_review import build_review_risk_report, run_evidence_gated_git_review
from workspace_state import DEFAULT_WORKSPACE_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate evidence-gated review reports without treating weak recall as support.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    latest_parser = subparsers.add_parser("latest", help="Regenerate the latest evidence-gated report.")
    latest_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id.")
    latest_parser.add_argument("--root", type=Path, default=None, help="Optional repo root.")
    latest_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")

    git_parser = subparsers.add_parser("git", help="Capture a git diff and generate an evidence-gated report.")
    git_parser.add_argument("--base", required=True, help="Base git ref.")
    git_parser.add_argument("--head", default="HEAD", help="Head git ref.")
    git_parser.add_argument("--test-log", type=Path, default=None, help="Optional local test log file.")
    git_parser.add_argument("--query", default="", help="Optional recall query override.")
    git_parser.add_argument("--note", default="", help="Optional review note.")
    git_parser.add_argument("--limit", type=int, default=5, help="Maximum prior evidence rows.")
    git_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id.")
    git_parser.add_argument("--root", type=Path, default=None, help="Optional repo root.")
    git_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "latest":
            metadata, markdown, _latest_path, _run_path = build_review_risk_report(
                workspace_id=args.workspace_id,
                root=args.root,
            )
        else:
            metadata, markdown, _latest_path, _run_path = run_evidence_gated_git_review(
                base=args.base,
                head=args.head,
                test_log=args.test_log,
                query=args.query.strip() or None,
                note=args.note.strip() or None,
                limit=args.limit,
                workspace_id=args.workspace_id,
                root=args.root,
            )
    except ValueError as exc:
        parser.error(str(exc))

    if args.format == "json":
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
