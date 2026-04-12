#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from artifact_schema import (
    build_artifact_payload,
    build_prompt_record,
    build_runtime_record,
    collect_asset_lineage,
    write_artifact,
)
from gemma_runtime import UserFacingError, print_runtime_header
from vision_service import (
    GENERATION_SETTINGS,
    MODES,
    build_messages,
    build_user_prompt,
    default_output_path,
    resolve_inputs,
    resolve_prompt,
    resolve_system_prompt,
    run_vision_mode,
    serialize_input_records,
    validate_mode_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal local Gemma 4 vision/document demo with Transformers."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=MODES,
        help="Vision task to run.",
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="One or more local image paths, or a PDF path for pdf-summary.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional task instruction or VQA question override.",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Override the default system prompt for the selected mode.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output path. Defaults to artifacts/vision/<timestamp>-<mode>.json",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=4,
        help="Maximum PDF pages to render locally (default: 4).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.out or default_output_path(args.mode)
    prompt: str | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    records: list[dict[str, Any]] = []
    mode_result: dict[str, object] | None = None

    def artifact_payload(status: str, blocker_message: str | None = None) -> dict[str, Any]:
        runtime = build_runtime_record(
            backend="gemma-live-vision",
            model_id=mode_result["model_id"] if mode_result is not None else None,
            device_info=mode_result["device_info"] if mode_result is not None else "unresolved",
            elapsed_seconds=mode_result["elapsed_seconds"] if mode_result is not None else None,
        )
        prompts = build_prompt_record(
            system_prompt=mode_result["system_prompt"] if mode_result is not None else system_prompt,
            prompt=mode_result["prompt"] if mode_result is not None else prompt,
            resolved_user_prompt=mode_result["resolved_user_prompt"] if mode_result is not None else user_prompt,
        )
        generation_settings = mode_result["generation_settings"] if mode_result is not None else dict(GENERATION_SETTINGS[args.mode])
        return build_artifact_payload(
            artifact_kind="vision",
            status=status,
            runtime=runtime,
            prompts=prompts,
            asset_lineage=collect_asset_lineage(records),
            blocker_message=blocker_message,
            extra={
                "mode": args.mode,
                "model_id": mode_result["model_id"] if mode_result is not None else None,
                "device": runtime["device"]["label"],
                "dtype": runtime["device"]["dtype"],
                "generation_settings": generation_settings,
                "system_prompt": mode_result["system_prompt"] if mode_result is not None else system_prompt,
                "prompt": mode_result["prompt"] if mode_result is not None else prompt,
                "resolved_user_prompt": mode_result["resolved_user_prompt"] if mode_result is not None else user_prompt,
                "inputs": serialize_input_records(records),
                "elapsed_seconds": round(float(mode_result["elapsed_seconds"]), 3) if mode_result is not None else None,
                "output_text": mode_result["output_text"] if mode_result is not None else None,
            },
        )

    try:
        prompt = resolve_prompt(args.mode, args.prompt)
        system_prompt = resolve_system_prompt(args.mode, args.system)
        mode_result = run_vision_mode(
            mode=args.mode,
            inputs=args.inputs,
            prompt=args.prompt,
            system_prompt=args.system,
            max_pages=args.max_pages,
        )
        records = list(mode_result["records"])
        prompt = str(mode_result["prompt"])
        system_prompt = str(mode_result["system_prompt"])
        user_prompt = str(mode_result["resolved_user_prompt"])
        print_runtime_header(
            str(mode_result["model_id"]),
            mode_result["device_info"],
            mode_result["generation_settings"],
        )
    except UserFacingError as exc:
        write_artifact(output_path, artifact_payload("blocked", str(exc)))
        print(f"artifact_path: {output_path}", file=sys.stderr)
        raise
    except Exception as exc:
        write_artifact(output_path, artifact_payload("failed", f"{type(exc).__name__}: {exc}"))
        print(f"artifact_path: {output_path}", file=sys.stderr)
        raise

    write_artifact(output_path, artifact_payload("ok"))

    print(f"elapsed_seconds: {float(mode_result['elapsed_seconds']):.3f}")
    print(f"artifact_path: {output_path}")
    print("\n[output]")
    print(mode_result["output_text"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserFacingError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
