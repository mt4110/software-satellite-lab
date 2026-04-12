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
from gemma_runtime import UserFacingError, default_thinking_artifact_path
from thinking_service import (
    MODES,
    TEXT_FOLLOW_UP,
    TEXT_PROMPT,
    TEXT_SYSTEM_PROMPT,
    TOOL_PROMPT,
    TOOL_SYSTEM_PROMPT,
    run_text_mode,
    run_text_mode_simulated,
    run_thinking_session,
    run_tool_mode,
    run_tool_mode_simulated,
    total_elapsed_seconds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Gemma 4 thinking-mode demo and save raw thinking to artifacts."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=MODES,
        help="`text` runs a multi-turn thinking chat, `tool` runs a tool-assisted thinking loop.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional first-turn prompt override.",
    )
    parser.add_argument(
        "--follow-up",
        default=None,
        help="Optional follow-up prompt for text mode.",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Optional system prompt override.",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Print parsed thinking to stderr. Stdout still prints only the final answer.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use canned Gemma-style responses to validate thinking behavior without loading a model.",
    )
    parser.add_argument(
        "--max-tool-iterations",
        type=int,
        default=3,
        help="Maximum assistant/tool loop iterations for tool mode (default: 3).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional artifact path. Defaults to artifacts/thinking/<timestamp>-<mode>.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.out or default_thinking_artifact_path(args.mode)

    system_prompt = args.system or (TEXT_SYSTEM_PROMPT if args.mode == "text" else TOOL_SYSTEM_PROMPT)
    first_prompt = args.prompt or (TEXT_PROMPT if args.mode == "text" else TOOL_PROMPT)
    follow_up_prompt = args.follow_up or TEXT_FOLLOW_UP
    result: dict[str, object] | None = None

    def artifact_payload(status: str, blocker_message: str | None = None) -> dict[str, object]:
        runtime = build_runtime_record(
            backend="simulated-thinking" if args.simulate else "gemma-live-thinking",
            model_id=result["model_id"] if result is not None else None,
            device_info=result["device_info"] if result is not None else ("simulated" if args.simulate else "unresolved"),
            elapsed_seconds=result["elapsed_seconds"] if result is not None else 0.0,
        )
        prompts = build_prompt_record(
            system_prompt=result["system_prompt"] if result is not None else system_prompt,
            prompt=result["first_prompt"] if result is not None else first_prompt,
            resolved_user_prompt=result["first_prompt"] if result is not None else first_prompt,
            extra={
                "follow_up_prompt": result["follow_up_prompt"] if result is not None else follow_up_prompt if args.mode == "text" else None,
            },
        )
        payload_extra = {
            "mode": args.mode,
            "model_id": result["model_id"] if result is not None else None,
            "device": runtime["device"]["label"],
            "dtype": runtime["device"]["dtype"],
            "stdout_behavior": "final_answer_only",
            "raw_thinking_saved_to_artifact": True,
            "simulation": args.simulate,
        }
        if result is not None:
            payload_extra.update(result)
        return build_artifact_payload(
            artifact_kind="thinking",
            status=status,
            runtime=runtime,
            prompts=prompts,
            blocker_message=blocker_message,
            extra=payload_extra,
        )

    try:
        result = run_thinking_session(
            mode=args.mode,
            system_prompt=args.system,
            prompt=args.prompt,
            follow_up=args.follow_up,
            show_thinking=args.show_thinking,
            simulate=args.simulate,
            max_tool_iterations=args.max_tool_iterations,
        )
    except UserFacingError as exc:
        write_artifact(output_path, artifact_payload("blocked", str(exc)))
        print(f"artifact_path: {output_path}", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        write_artifact(output_path, artifact_payload("failed", f"{type(exc).__name__}: {exc}"))
        print(f"artifact_path: {output_path}", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    payload = artifact_payload("ok")
    write_artifact(output_path, payload)

    print(payload["final_answer"])
    print(f"artifact_path: {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
