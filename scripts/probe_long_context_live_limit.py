#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from artifact_schema import build_artifact_payload, build_prompt_record, build_runtime_record, write_artifact
from blocker_taxonomy import classify_blocker
from gemma_core import SessionManager
from gemma_runtime import UserFacingError, repo_root, resolve_model_id, timestamp_slug
from run_long_context_demo import detect_context_profile, run_long_context_report


DEFAULT_TARGET_STEP = 8_192


def artifacts_root() -> Path:
    return repo_root() / "artifacts" / "long_context"


def default_output_path() -> Path:
    return artifacts_root() / f"{timestamp_slug()}-live-limit-probe.json"


def build_default_targets(max_target: int, *, step: int = DEFAULT_TARGET_STEP) -> list[int]:
    if max_target < 1:
        raise ValueError("`max_target` must be positive.")
    if step < 1:
        raise ValueError("`step` must be positive.")

    targets = list(range(step, max_target + 1, step))
    if not targets or targets[-1] != max_target:
        targets.append(max_target)
    return targets


def parse_targets(raw_targets: str) -> list[int]:
    parsed: list[int] = []
    for item in raw_targets.split(","):
        cleaned = item.strip()
        if not cleaned:
            continue
        value = int(cleaned)
        if value < 1:
            raise ValueError("Probe targets must be positive integers.")
        parsed.append(value)
    if not parsed:
        raise ValueError("At least one probe target is required.")
    return sorted(set(parsed))


def minimum_case_prompt_tokens(case_reports: list[dict[str, Any]]) -> int | None:
    prompt_tokens = [
        int(report["prompt_token_estimate"])
        for report in case_reports
        if report.get("prompt_token_estimate") is not None
    ]
    if not prompt_tokens:
        return None
    return min(prompt_tokens)


def minimum_case_prompt_fit_ratio(case_reports: list[dict[str, Any]]) -> float | None:
    fit_ratios = [
        float(report["prompt_fit_ratio"])
        for report in case_reports
        if report.get("prompt_fit_ratio") is not None
    ]
    if not fit_ratios:
        return None
    return min(fit_ratios)


def summarize_probe_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    highest_validated: dict[str, Any] | None = None
    first_limit: dict[str, Any] | None = None

    for result in results:
        if result["status"] == "ok":
            highest_validated = result
            continue
        if first_limit is None:
            first_limit = result

    return {
        "targets_attempted": [result["target_prompt_tokens"] for result in results],
        "attempt_count": len(results),
        "highest_live_validated_target_prompt_tokens": (
            highest_validated["target_prompt_tokens"] if highest_validated is not None else None
        ),
        "highest_live_validated_realized_prompt_tokens": (
            highest_validated["minimum_case_prompt_tokens"] if highest_validated is not None else None
        ),
        "highest_live_validated_artifact_path": (
            highest_validated["artifact_path"] if highest_validated is not None else None
        ),
        "first_limit_target_prompt_tokens": (
            first_limit["target_prompt_tokens"] if first_limit is not None else None
        ),
        "first_limit_status": first_limit["status"] if first_limit is not None else None,
        "first_limit_message": first_limit["message"] if first_limit is not None else None,
        "recommended_live_target_prompt_tokens": (
            highest_validated["target_prompt_tokens"] if highest_validated is not None else None
        ),
        "recommendation_basis": (
            "Highest probed target that still achieved live validated status on every executed case."
            if highest_validated is not None
            else "No probed target reached live validated status."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe the practical live long-context limit on the current machine."
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional model id override. Defaults to GEMMA_MODEL_ID or the repo default.",
    )
    parser.add_argument(
        "--case",
        choices=("synthetic", "repo", "both"),
        default="both",
        help="Which long-context case set to probe (default: both).",
    )
    parser.add_argument(
        "--targets",
        default=None,
        help="Optional comma-separated target prompt token series such as `8192,16384,24576`.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_TARGET_STEP,
        help=f"Step size for the default target series (default: {DEFAULT_TARGET_STEP}).",
    )
    parser.add_argument(
        "--max-target",
        type=int,
        default=None,
        help="Optional maximum target prompt tokens for the default target series.",
    )
    parser.add_argument(
        "--seed",
        default="gemma-live-limit-v1",
        help="Deterministic seed namespace used for the probe.",
    )
    parser.add_argument(
        "--continue-after-limit",
        action="store_true",
        help="Keep probing after the first blocked or not-validated result.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional summary artifact path. Defaults to artifacts/long_context/<timestamp>-live-limit-probe.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()
    model_id = (args.model_id or resolve_model_id()).strip()
    profile = detect_context_profile(model_id)

    if args.targets is not None:
        target_series = parse_targets(args.targets)
    else:
        target_series = build_default_targets(
            args.max_target or profile.safe_prompt_tokens,
            step=args.step,
        )

    summary_output_path = args.out or default_output_path()
    manager = SessionManager()
    results: list[dict[str, Any]] = []

    def summary_payload(status: str) -> dict[str, Any]:
        summary = summarize_probe_results(results)
        runtime = build_runtime_record(
            backend="long-context-live-limit-probe",
            model_id=model_id,
            device_info="probe-multi-run",
            elapsed_seconds=time.perf_counter() - started_at,
        )
        prompts = build_prompt_record(
            system_prompt=None,
            prompt=None,
            resolved_user_prompt=None,
            extra={
                "probe_case": args.case,
                "target_series": target_series,
            },
        )
        return build_artifact_payload(
            artifact_kind="long_context_probe",
            status=status,
            runtime=runtime,
            prompts=prompts,
            extra={
                "model_id": model_id,
                "probe_case": args.case,
                "target_series": target_series,
                "context_profile": {
                    "label": profile.label,
                    "max_context_tokens": profile.max_context_tokens,
                    "safe_prompt_tokens": profile.safe_prompt_tokens,
                    "rationale": profile.rationale,
                },
                "results": results,
                "summary": summary,
            },
        )

    print(f"selected_model_id: {model_id}")
    print(f"probe_case: {args.case}")
    print(f"context_profile: {profile.label}")
    print(f"target_series: {target_series}")

    try:
        for target_prompt_tokens in target_series:
            step_output_path = artifacts_root() / f"{timestamp_slug()}-live-probe-{target_prompt_tokens}.json"
            print(f"probe_start target_prompt_tokens={target_prompt_tokens}", flush=True)
            step_started_at = time.perf_counter()
            step_record: dict[str, Any] = {
                "target_prompt_tokens": target_prompt_tokens,
                "artifact_path": str(step_output_path),
            }

            try:
                payload, _ = run_long_context_report(
                    requested_backend="live",
                    case=args.case,
                    model_id=model_id,
                    target_prompt_tokens=target_prompt_tokens,
                    seed=f"{args.seed}:{target_prompt_tokens}",
                    output_path=step_output_path,
                    session_manager=manager,
                    print_runtime=False,
                )
                case_reports = list(payload.get("cases") or [])
                step_record.update(
                    {
                        "status": "ok" if payload.get("long_context_validated", False) else "not_validated",
                        "message": None if payload.get("long_context_validated", False) else "Live execution completed but did not satisfy the repo validation contract.",
                        "resolved_backend": payload.get("resolved_backend"),
                        "long_context_validated": bool(payload.get("long_context_validated", False)),
                        "overall_pass": bool(payload.get("overall_pass", False)),
                        "minimum_case_prompt_tokens": minimum_case_prompt_tokens(case_reports),
                        "minimum_case_prompt_fit_ratio": minimum_case_prompt_fit_ratio(case_reports),
                        "case_reports": case_reports,
                        "validation": payload.get("validation"),
                        "elapsed_seconds": round(time.perf_counter() - step_started_at, 3),
                    }
                )
            except UserFacingError as exc:
                blocker_message = str(exc)
                blocker = classify_blocker(blocker_message)
                step_record.update(
                    {
                        "status": "blocked",
                        "message": blocker_message,
                        "blocker": {
                            "kind": blocker.kind,
                            "external": blocker.external,
                        },
                        "resolved_backend": "live",
                        "long_context_validated": False,
                        "overall_pass": False,
                        "minimum_case_prompt_tokens": None,
                        "minimum_case_prompt_fit_ratio": None,
                        "case_reports": [],
                        "validation": None,
                        "elapsed_seconds": round(time.perf_counter() - step_started_at, 3),
                    }
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                step_record.update(
                    {
                        "status": "failed",
                        "message": message,
                        "resolved_backend": "live",
                        "long_context_validated": False,
                        "overall_pass": False,
                        "minimum_case_prompt_tokens": None,
                        "minimum_case_prompt_fit_ratio": None,
                        "case_reports": [],
                        "validation": None,
                        "elapsed_seconds": round(time.perf_counter() - step_started_at, 3),
                    }
                )

            results.append(step_record)
            print(
                "probe_result "
                f"target_prompt_tokens={target_prompt_tokens} "
                f"status={step_record['status']} "
                f"minimum_case_prompt_tokens={step_record['minimum_case_prompt_tokens']} "
                f"minimum_case_prompt_fit_ratio={step_record['minimum_case_prompt_fit_ratio']} "
                f"artifact_path={step_record['artifact_path']}",
                flush=True,
            )

            if step_record["status"] != "ok" and not args.continue_after_limit:
                break
    finally:
        manager.close_all()

    summary = summarize_probe_results(results)
    payload = summary_payload("ok")
    write_artifact(summary_output_path, payload)

    print(f"highest_live_validated_target_prompt_tokens: {summary['highest_live_validated_target_prompt_tokens']}")
    print(f"first_limit_target_prompt_tokens: {summary['first_limit_target_prompt_tokens']}")
    print(f"recommended_live_target_prompt_tokens: {summary['recommended_live_target_prompt_tokens']}")
    print(f"summary_artifact_path: {summary_output_path}")

    if summary["recommended_live_target_prompt_tokens"] is None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
