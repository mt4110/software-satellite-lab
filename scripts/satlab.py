#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

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

    parser.error("Unsupported command.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
