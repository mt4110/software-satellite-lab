#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from artifact_schema import (
    build_artifact_payload,
    build_prompt_record,
    build_runtime_record,
    write_artifact,
)
from gemma_runtime import UserFacingError, print_runtime_header
from text_service import (
    DEFAULT_PROMPTS,
    GENERATION_SETTINGS,
    TASKS,
    build_user_prompt,
    default_output_path,
    resolve_prompt,
    resolve_system_prompt,
    run_text_task,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal local Gemma 4 text smoke test with Transformers."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=TASKS,
        help="Demo task to run.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Task input. If omitted, a small built-in sample is used.",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Override the default system prompt for the selected task.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output path. Defaults to artifacts/text/<timestamp>-<task>.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt = resolve_prompt(args.task, args.prompt)
    system_prompt = resolve_system_prompt(args.task, args.system)
    user_prompt = build_user_prompt(args.task, prompt)
    output_path = args.out or default_output_path(args.task)
    task_result: dict[str, object] | None = None

    def artifact_payload(status: str, blocker_message: str | None = None) -> dict[str, object]:
        runtime = build_runtime_record(
            backend="gemma-live-text",
            model_id=task_result["model_id"] if task_result is not None else None,
            device_info=task_result["device_info"] if task_result is not None else "unresolved",
            elapsed_seconds=task_result["elapsed_seconds"] if task_result is not None else None,
        )
        prompts = build_prompt_record(
            system_prompt=task_result["system_prompt"] if task_result is not None else system_prompt,
            prompt=task_result["prompt"] if task_result is not None else prompt,
            resolved_user_prompt=task_result["resolved_user_prompt"] if task_result is not None else user_prompt,
        )
        generation_settings = task_result["generation_settings"] if task_result is not None else dict(GENERATION_SETTINGS[args.task])
        return build_artifact_payload(
            artifact_kind="text",
            status=status,
            runtime=runtime,
            prompts=prompts,
            blocker_message=blocker_message,
            extra={
                "task": args.task,
                "model_id": task_result["model_id"] if task_result is not None else None,
                "device": runtime["device"]["label"],
                "dtype": runtime["device"]["dtype"],
                "generation_settings": generation_settings,
                "system_prompt": task_result["system_prompt"] if task_result is not None else system_prompt,
                "prompt": task_result["prompt"] if task_result is not None else prompt,
                "resolved_user_prompt": task_result["resolved_user_prompt"] if task_result is not None else user_prompt,
                "elapsed_seconds": round(float(task_result["elapsed_seconds"]), 3) if task_result is not None else None,
                "output_text": task_result["output_text"] if task_result is not None else None,
            },
        )

    try:
        task_result = run_text_task(
            task=args.task,
            prompt=args.prompt,
            system_prompt=args.system,
        )
        print_runtime_header(
            str(task_result["model_id"]),
            task_result["device_info"],
            task_result["generation_settings"],
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

    print(f"elapsed_seconds: {float(task_result['elapsed_seconds']):.3f}")
    print(f"artifact_path: {output_path}")
    print("\n[output]")
    print(task_result["output_text"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserFacingError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
