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
from audio_service import (
    DEFAULT_TARGET_LANGUAGE,
    MODES,
    assess_audio_output,
    default_output_path,
    resolve_prompt,
    resolve_system_prompt,
    run_audio_mode,
    serialize_audio_record,
)
from gemma_runtime import UserFacingError, print_runtime_header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal local Gemma 4 audio demo with Transformers."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=MODES,
        help="Audio task to run.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Local audio path. Keep clips at or under 30 seconds.",
    )
    parser.add_argument(
        "--target-language",
        default=DEFAULT_TARGET_LANGUAGE,
        help="Target language for translate mode (default: Japanese).",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional task instruction override.",
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
        help="Optional output path. Defaults to artifacts/audio/<timestamp>-<mode>.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.out or default_output_path(args.mode)
    mode_result: dict[str, Any] | None = None

    def artifact_payload(status: str, blocker_message: str | None = None) -> dict[str, Any]:
        runtime = build_runtime_record(
            backend="gemma-live-audio",
            model_id=mode_result["model_id"] if mode_result is not None else None,
            device_info=mode_result["device_info"] if mode_result is not None else "unresolved",
            elapsed_seconds=mode_result["elapsed_seconds"] if mode_result is not None else None,
            extra={
                "base_model_id": mode_result["base_model_id"] if mode_result is not None else None,
                "model_id_source": mode_result["model_id_source"] if mode_result is not None else None,
            },
        )
        prompts = build_prompt_record(
            system_prompt=mode_result["system_prompt"] if mode_result is not None else None,
            prompt=mode_result["prompt"] if mode_result is not None else None,
            resolved_user_prompt=mode_result["prompt"] if mode_result is not None else None,
        )
        return build_artifact_payload(
            artifact_kind="audio",
            status=status,
            runtime=runtime,
            prompts=prompts,
            asset_lineage=collect_asset_lineage(mode_result["record"] if mode_result is not None else None),
            blocker_message=blocker_message,
            extra={
                "mode": args.mode,
                "base_model_id": mode_result["base_model_id"] if mode_result is not None else None,
                "model_id": mode_result["model_id"] if mode_result is not None else None,
                "model_id_source": mode_result["model_id_source"] if mode_result is not None else None,
                "device": runtime["device"]["label"],
                "dtype": runtime["device"]["dtype"],
                "generation_settings": mode_result["generation_settings"] if mode_result is not None else None,
                "target_language": mode_result["target_language"] if mode_result is not None else None,
                "system_prompt": mode_result["system_prompt"] if mode_result is not None else None,
                "prompt": mode_result["prompt"] if mode_result is not None else None,
                "resolved_user_prompt": mode_result["prompt"] if mode_result is not None else None,
                "input": serialize_audio_record(mode_result["record"]) if mode_result is not None else None,
                "elapsed_seconds": round(mode_result["elapsed_seconds"], 3) if mode_result is not None else None,
                "output_text": mode_result["output_text"] if mode_result is not None else None,
                "pipeline": mode_result["pipeline"] if mode_result is not None else None,
                "validation": mode_result["validation"] if mode_result is not None else None,
            },
        )

    try:
        mode_result = run_audio_mode(
            mode=args.mode,
            input_path=args.input,
            target_language=args.target_language,
            prompt=args.prompt,
            system_prompt=args.system,
        )
        print_runtime_header(
            mode_result["model_id"],
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

    print(f"elapsed_seconds: {mode_result['elapsed_seconds']:.3f}")
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
