#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_index import rebuild_memory_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild the local software-work event log and SQLite memory index.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional repo root override.",
    )
    parser.add_argument(
        "--workspace-id",
        default="local-default",
        help="Workspace id to ingest.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="Optional explicit SQLite output path.",
    )
    parser.add_argument(
        "--event-log-path",
        type=Path,
        default=None,
        help="Optional explicit JSONL event log output path.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    result = rebuild_memory_index(
        root=args.root,
        workspace_id=args.workspace_id,
        index_path=args.index_path,
        event_log_path=args.event_log_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
