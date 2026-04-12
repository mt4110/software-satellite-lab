#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from artifact_schema import (
    build_artifact_payload,
    build_prompt_record,
    build_runtime_record,
    write_artifact,
)
from blocker_taxonomy import classify_blocker
from gemma_core import RuntimeSession, SessionManager, generate_text_from_messages
from gemma_runtime import (
    UserFacingError,
    apply_chat_template,
    assert_model_fetch_is_possible,
    build_text_messages,
    import_text_runtime,
    print_runtime_header,
    repo_root,
    resolve_model_id,
    select_device,
    timestamp_slug,
)
from long_context_corpus import (
    MARKER_LABELS,
    build_repo_corpus,
    build_synthetic_corpus,
)


BACKENDS = ("auto", "live", "simulate", "lexical")
CASES = ("synthetic", "repo", "both")
AUTO_LIVE_PROMPT_TOKEN_LIMIT = 32_768
TARGET_FIT_LOWER_BOUND_RATIO = 0.85
DEFAULT_GENERATION_SETTINGS = {"max_new_tokens": 80, "do_sample": False}
SYSTEM_PROMPT = (
    "You retrieve exact marker values from long context. "
    "Do not summarize, do not paraphrase, and do not invent missing values."
)
PROMPT_TEMPLATE = """Read the corpus and extract the exact value for each retrieval marker.

Return exactly these three lines and nothing else:
beginning=<value>
middle=<value>
end=<value>

If a value is missing, use NONE.

[corpus]
{corpus}
"""
RESPONSE_LINE_RE = re.compile(r"^\s*(beginning|middle|end)\s*[:=]\s*(?P<value>\S+)\s*$", re.MULTILINE)
CODE_BLOCK_RE = re.compile(r"^```(?:json|text)?\s*|\s*```$", re.MULTILINE)


@dataclass(frozen=True)
class ContextProfile:
    label: str
    max_context_tokens: int
    safe_prompt_tokens: int
    rationale: str


def artifacts_root() -> Path:
    return repo_root() / "artifacts" / "long_context"


def default_output_path() -> Path:
    return artifacts_root() / f"{timestamp_slug()}-report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic long-context needle-in-a-haystack demo for Gemma 4."
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="auto",
        help="`auto` prefers live then falls back honestly, `simulate` uses tokenizer-only sizing, `live` forces Gemma inference, and `lexical` stays deterministic.",
    )
    parser.add_argument(
        "--case",
        choices=CASES,
        default="both",
        help="Which long-context corpus case to run.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional model id override. Defaults to GEMMA_MODEL_ID or the lab default.",
    )
    parser.add_argument(
        "--target-prompt-tokens",
        type=int,
        default=None,
        help="Override the context-class token target used for corpus sizing.",
    )
    parser.add_argument(
        "--seed",
        default="gemma-long-context-v1",
        help="Deterministic seed namespace used for corpus generation.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output path. Defaults to artifacts/long_context/<timestamp>-report.json",
    )
    return parser.parse_args()


def detect_context_profile(model_id: str) -> ContextProfile:
    normalized = model_id.strip().lower()
    if "gemma-4-26b" in normalized or "26b-a4b" in normalized or "gemma-4-31b" in normalized:
        return ContextProfile(
            label="256k-class",
            max_context_tokens=256_000,
            safe_prompt_tokens=196_608,
            rationale="26B A4B and 31B models expose the 256K context class, so the harness uses a conservative 192K prompt target.",
        )

    if "gemma-4-e2b" in normalized or "gemma-4-e4b" in normalized:
        return ContextProfile(
            label="128k-class",
            max_context_tokens=128_000,
            safe_prompt_tokens=98_304,
            rationale="E2B and E4B models expose the 128K context class, so the harness uses a conservative 96K prompt target.",
        )

    return ContextProfile(
        label="128k-class",
        max_context_tokens=128_000,
        safe_prompt_tokens=98_304,
        rationale="Unknown Gemma 4 variant; defaulting to the 128K-class profile unless the model id indicates 26B A4B or 31B.",
    )


def build_user_prompt(corpus_text: str) -> str:
    return PROMPT_TEMPLATE.format(corpus=corpus_text.rstrip())


def choose_backend(
    requested_backend: str,
    target_prompt_tokens: int,
    device_info: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    notes: list[str] = []
    if requested_backend != "auto":
        return requested_backend, notes

    if target_prompt_tokens > AUTO_LIVE_PROMPT_TOKEN_LIMIT:
        notes.append(
            "Auto backend avoided live inference because the context-class workload is larger than the safe automatic live threshold."
        )
        return "simulate", notes

    if device_info is None:
        notes.append(
            "Auto backend avoided live inference because no live device/runtime information was available."
        )
        return "simulate", notes

    notes.append("Auto backend selected live inference because the prompt size stayed within the automatic live threshold.")
    return "live", notes


def processor_auth_kwargs() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"token": token} if token else {}


def maybe_load_processor(model_id: str) -> tuple[Any | None, str | None]:
    try:
        assert_model_fetch_is_possible(model_id)
        _, auto_processor_cls, _ = import_text_runtime()
        processor = auto_processor_cls.from_pretrained(model_id, **processor_auth_kwargs())
        return processor, None
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"{type(exc).__name__}: {exc}"


def build_prompt_token_estimate(
    processor: Any | None,
    system_prompt: str,
    user_prompt: str,
) -> tuple[int, str, str | None]:
    if processor is None:
        return len((system_prompt + "\n" + user_prompt).split()), "word-count-estimate", None

    rendered_prompt = apply_chat_template(
        processor,
        build_text_messages(system_prompt=system_prompt, user_prompt=user_prompt),
    )

    try:
        batch = processor(text=rendered_prompt, return_tensors="pt")
        return int(batch["input_ids"].shape[-1]), "processor-token-count", rendered_prompt
    except Exception:
        return len(rendered_prompt.split()), "rendered-word-count-estimate", rendered_prompt


def parse_marker_payload(text: str) -> dict[str, str | None]:
    cleaned = CODE_BLOCK_RE.sub("", (text or "").strip()).strip()

    parsed_values: dict[str, str | None] = {}
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        for label in MARKER_LABELS:
            value = payload.get(label)
            parsed_values[label] = str(value).strip() if value not in (None, "") else None
        return parsed_values

    for match in RESPONSE_LINE_RE.finditer(cleaned):
        parsed_values[match.group(1)] = match.group("value").strip()

    return {label: parsed_values.get(label) for label in MARKER_LABELS}


def response_text_from_markers(markers: dict[str, str | None]) -> str:
    return "\n".join(f"{label}={markers.get(label) or 'NONE'}" for label in MARKER_LABELS)


def extract_markers_from_corpus(corpus_text: str) -> dict[str, str | None]:
    extracted: dict[str, str | None] = {}
    for label in MARKER_LABELS:
        pattern = rf"retrieval_marker_{label}\s*=\s*(?P<value>\S+)"
        match = re.search(pattern, corpus_text)
        extracted[label] = match.group("value").strip() if match else None
    return extracted


def summarize_marker_results(
    expected: dict[str, str],
    actual: dict[str, str | None],
) -> tuple[dict[str, dict[str, Any]], bool]:
    results: dict[str, dict[str, Any]] = {}
    all_passed = True
    for label in MARKER_LABELS:
        observed = actual.get(label)
        passed = observed == expected[label]
        if not passed:
            all_passed = False
        results[label] = {
            "expected": expected[label],
            "actual": observed,
            "pass": passed,
        }
    return results, all_passed


def run_lexical_case(case_payload: dict[str, Any]) -> tuple[str, dict[str, str | None], float]:
    started_at = time.perf_counter()
    markers = extract_markers_from_corpus(case_payload["corpus_text"])
    elapsed_seconds = time.perf_counter() - started_at
    return response_text_from_markers(markers), markers, elapsed_seconds


def run_live_case(
    session: RuntimeSession,
    user_prompt: str,
) -> tuple[str, dict[str, str | None], float, int]:
    generation = generate_text_from_messages(
        session=session,
        messages=build_text_messages(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt),
        generation_settings=DEFAULT_GENERATION_SETTINGS,
    )
    response_text = generation.output_text
    parsed_markers = parse_marker_payload(response_text)
    return response_text, parsed_markers, generation.elapsed_seconds, generation.input_token_count


def execute_case(
    case_payload: dict[str, Any],
    backend: str,
    processor: Any | None,
    session: RuntimeSession | None,
    target_prompt_tokens: int,
) -> dict[str, Any]:
    user_prompt = build_user_prompt(case_payload["corpus_text"])
    prompt_tokens, prompt_token_source, rendered_prompt = build_prompt_token_estimate(
        processor=processor,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    prompt_character_count = len(rendered_prompt) if rendered_prompt is not None else len(user_prompt)

    if backend == "live":
        if session is None:
            raise UserFacingError("Live backend requested but the model runtime was not initialized.")
        response_text, parsed_markers, elapsed_seconds, live_prompt_tokens = run_live_case(
            session=session,
            user_prompt=user_prompt,
        )
        prompt_tokens = live_prompt_tokens
        prompt_token_source = "live-input-token-count"
        backend_notes: list[str] = []
    else:
        response_text, parsed_markers, elapsed_seconds = run_lexical_case(case_payload)
        backend_notes = [
            "Lexical retrieval validates corpus construction and exact marker placement without live Gemma generation."
        ]
        if backend == "simulate":
            backend_notes = [
                "Simulate mode used the tokenizer/processor for prompt-fit estimation, then validated retrieval deterministically without live Gemma generation."
            ]

    marker_results, passed = summarize_marker_results(
        expected=case_payload["markers"],
        actual=parsed_markers,
    )
    prompt_fit_ratio = round(prompt_tokens / float(max(target_prompt_tokens, 1)), 4)
    meets_prompt_floor = prompt_tokens >= int(target_prompt_tokens * TARGET_FIT_LOWER_BOUND_RATIO)

    return {
        "case_id": case_payload["case_id"],
        "case_label": case_payload["case_label"],
        "backend": backend,
        "validation_mode": backend,
        "pass": passed,
        "target_prompt_tokens": target_prompt_tokens,
        "prompt_token_estimate": prompt_tokens,
        "prompt_token_estimate_source": prompt_token_source,
        "prompt_fit_ratio": prompt_fit_ratio,
        "meets_prompt_floor": meets_prompt_floor,
        "prompt_character_count": prompt_character_count,
        "latency_seconds": round(elapsed_seconds, 3),
        "response_text": response_text,
        "marker_results": marker_results,
        "expected_markers": case_payload["markers"],
        "retrieved_markers": parsed_markers,
        "corpus_metadata": case_payload["metadata"],
        "backend_notes": backend_notes,
    }


def requested_case_ids(case_name: str) -> list[str]:
    if case_name == "both":
        return ["synthetic", "repo"]
    return [case_name]


def build_case_payload(
    case_name: str,
    target_word_budget: int,
    seed: str,
) -> dict[str, Any] | None:
    if case_name == "synthetic":
        return build_synthetic_corpus(target_word_budget=target_word_budget, seed=seed)
    if case_name == "repo":
        return build_repo_corpus(
            root=repo_root(),
            target_word_budget=target_word_budget,
            seed=seed,
        )
    raise UserFacingError(f"Unsupported long-context case `{case_name}`.")


def fit_case_payload_to_prompt_target(
    case_name: str,
    processor: Any | None,
    target_prompt_tokens: int,
    seed: str,
) -> dict[str, Any] | None:
    target_word_budget = max(4096, target_prompt_tokens - 1_024)
    best_payload: dict[str, Any] | None = None
    best_estimate: int | None = None

    for _ in range(6):
        payload = build_case_payload(
            case_name=case_name,
            target_word_budget=target_word_budget,
            seed=seed,
        )
        if payload is None:
            return None

        estimate, estimate_source, _ = build_prompt_token_estimate(
            processor=processor,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(payload["corpus_text"]),
        )
        payload["metadata"]["fitted_word_budget"] = target_word_budget
        payload["metadata"]["preflight_prompt_token_estimate"] = estimate
        payload["metadata"]["preflight_prompt_token_source"] = estimate_source

        if best_payload is None or best_estimate is None or abs(estimate - target_prompt_tokens) < abs(
            best_estimate - target_prompt_tokens
        ):
            best_payload = payload
            best_estimate = estimate

        if processor is None:
            return payload

        lower_bound = int(target_prompt_tokens * TARGET_FIT_LOWER_BOUND_RATIO)
        if lower_bound <= estimate <= target_prompt_tokens:
            return payload

        scale = target_prompt_tokens / max(estimate, 1)
        if estimate > target_prompt_tokens:
            target_word_budget = max(1024, int(target_word_budget * scale * 0.97))
        else:
            target_word_budget = int(target_word_budget * min(scale * 1.03, 1.35))

    return best_payload


def maybe_detect_text_device() -> dict[str, Any] | None:
    device_info: dict[str, Any] | None = None
    try:
        torch_runtime, _, _ = import_text_runtime()
        device_info = select_device(torch_runtime)
    except Exception:
        device_info = None
    return device_info


def build_case_lineage(case_payloads: list[dict[str, Any]], backend: str) -> list[dict[str, Any]]:
    lineage: list[dict[str, Any]] = []
    for case_payload in case_payloads:
        lineage.append(
            {
                "source_path": None,
                "resolved_path": None,
                "cache_path": None,
                "asset_kind": "long_context_corpus",
                "transform": str(case_payload["case_id"]),
                "cache_key": str(case_payload["case_id"]),
                "cache_hit": None,
                "metadata": {
                    "case_label": case_payload["case_label"],
                    "validation_mode": backend,
                    "corpus_metadata": case_payload["metadata"],
                },
            }
        )
    return lineage


def validation_claim_scope(validation_mode: str) -> str:
    if validation_mode == "live":
        return "live Gemma retrieval on a long prompt"
    if validation_mode == "simulate":
        return "tokenizer-backed prompt-fit simulation plus deterministic lexical retrieval"
    return "deterministic lexical retrieval harness without tokenizer or live generation"


def validation_pass_definition(validation_mode: str) -> str:
    if validation_mode == "live":
        return (
            "Pass means the live backend executed, every retrieval marker matched exactly, and each case stayed near the "
            "target prompt size for the resolved context class."
        )
    if validation_mode == "simulate":
        return (
            "Pass means the tokenizer/processor was available for prompt-fit estimation and the deterministic retrieval harness "
            "still recovered every marker exactly, but no live Gemma generation was attempted."
        )
    return (
        "Pass means the deterministic lexical harness recovered every marker exactly, but no tokenizer-backed sizing or live Gemma generation was attempted."
    )


def build_quality_check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "name": name,
        "pass": bool(passed),
        "detail": detail,
    }


def build_long_context_validation(
    *,
    validation_mode: str,
    case_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    markers_exact = bool(case_reports) and all(report["pass"] for report in case_reports)
    prompt_floor_met = bool(case_reports) and all(report["meets_prompt_floor"] for report in case_reports)
    long_context_validated = validation_mode == "live" and markers_exact and prompt_floor_met
    quality_checks = [
        build_quality_check(
            "cases_executed",
            bool(case_reports),
            f"Observed {len(case_reports)} long-context case(s).",
        ),
        build_quality_check(
            "markers_exact",
            markers_exact,
            "Each retrieval marker must match its exact expected value.",
        ),
        build_quality_check(
            "prompt_floor_met",
            prompt_floor_met,
            f"Each case should reach at least {TARGET_FIT_LOWER_BOUND_RATIO:.0%} of the target prompt size.",
        ),
        build_quality_check(
            "live_backend_executed",
            validation_mode == "live",
            "Only live mode counts as a true long-context validation claim in this repo.",
        ),
    ]
    quality_notes: list[str] = []
    if validation_mode != "live":
        quality_notes.append("This artifact is useful for harness inspection, but it does not count as long-context validated.")
    if validation_mode == "live" and case_reports and not prompt_floor_met:
        quality_notes.append("Live execution completed, but the realized prompt size stayed below the configured long-context floor.")
    return {
        "validation_mode": validation_mode,
        "claim_scope": validation_claim_scope(validation_mode),
        "pass_definition": validation_pass_definition(validation_mode),
        "execution_status": "ok",
        "quality_status": "pass" if markers_exact and (validation_mode != "live" or prompt_floor_met) else "fail",
        "quality_checks": quality_checks,
        "quality_notes": quality_notes,
        "long_context_validated": long_context_validated,
    }


def run_long_context_report(
    *,
    requested_backend: str = "auto",
    case: str = "both",
    model_id: str | None = None,
    target_prompt_tokens: int | None = None,
    seed: str = "gemma-long-context-v1",
    output_path: Path | None = None,
    session_manager: SessionManager | None = None,
    print_runtime: bool = False,
) -> tuple[dict[str, Any], Path]:
    started_at = time.perf_counter()
    resolved_model_id = (model_id or resolve_model_id()).strip()
    profile = detect_context_profile(resolved_model_id)
    resolved_target_prompt_tokens = target_prompt_tokens or profile.safe_prompt_tokens
    resolved_output_path = output_path or default_output_path()
    device_info = maybe_detect_text_device()

    resolved_backend, backend_notes = choose_backend(
        requested_backend=requested_backend,
        target_prompt_tokens=resolved_target_prompt_tokens,
        device_info=device_info,
    )

    owns_session_manager = session_manager is None
    manager = session_manager or SessionManager()
    session: RuntimeSession | None = None
    processor: Any | None = None
    processor_error: str | None = None
    case_payloads: list[dict[str, Any]] = []
    case_reports: list[dict[str, Any]] = []

    def artifact_payload(status: str, overall_pass: bool | None, blocker_message: str | None = None) -> dict[str, Any]:
        validation = build_long_context_validation(
            validation_mode=resolved_backend,
            case_reports=case_reports,
        )
        validation["execution_status"] = status
        runtime = build_runtime_record(
            backend=f"long-context-{resolved_backend}",
            model_id=resolved_model_id,
            device_info=device_info if device_info is not None else "unresolved",
            elapsed_seconds=time.perf_counter() - started_at,
            extra={
                "requested_backend": requested_backend,
                "resolved_backend": resolved_backend,
            },
        )
        prompts = build_prompt_record(
            system_prompt=SYSTEM_PROMPT,
            prompt="Deterministic marker extraction prompt template.",
            resolved_user_prompt=None,
            extra={
                "prompt_template": PROMPT_TEMPLATE.strip(),
            },
        )
        return build_artifact_payload(
            artifact_kind="long_context",
            status=status,
            runtime=runtime,
            prompts=prompts,
            asset_lineage=build_case_lineage(case_payloads, resolved_backend),
            blocker_message=blocker_message,
            extra={
                "model_id": resolved_model_id,
                "requested_backend": requested_backend,
                "resolved_backend": resolved_backend,
                "backend_notes": backend_notes,
                "processor_tokenizer_available": processor is not None,
                "processor_tokenizer_error": processor_error,
                "device": runtime["device"]["label"],
                "context_profile": {
                    "label": profile.label,
                    "max_context_tokens": profile.max_context_tokens,
                    "safe_prompt_tokens": resolved_target_prompt_tokens,
                    "rationale": profile.rationale,
                },
                "generation_settings": DEFAULT_GENERATION_SETTINGS,
                "overall_pass": overall_pass,
                "long_context_validated": validation["long_context_validated"],
                "cases": case_reports,
                "validation": validation,
            },
        )

    try:
        if resolved_backend == "live":
            if device_info is None:
                live_error = "Torch runtime or device detection was unavailable for the live backend."
                if requested_backend == "live":
                    raise UserFacingError(live_error)
                backend_notes.append(live_error)
                resolved_backend = "simulate"
            else:
                try:
                    session = manager.get_session("long-context", resolved_model_id)
                    processor = session.processor
                    device_info = session.device_info
                    if print_runtime:
                        print_runtime_header(resolved_model_id, device_info, DEFAULT_GENERATION_SETTINGS)
                except UserFacingError as exc:
                    if requested_backend == "live":
                        raise
                    backend_notes.append(str(exc))
                    resolved_backend = "simulate"
                    session = None

        if resolved_backend == "simulate":
            processor, processor_error = maybe_load_processor(resolved_model_id)
            if processor is None:
                if requested_backend == "simulate":
                    raise UserFacingError(
                        processor_error
                        or "Tokenizer/processor setup was unavailable for the requested simulate backend."
                    )
                backend_notes.append(
                    "Tokenizer/processor setup was unavailable for simulate mode, so the run fell back to lexical retrieval only."
                )
                resolved_backend = "lexical"

        if resolved_backend not in {"live", "simulate", "lexical"}:
            raise UserFacingError(f"Unsupported resolved backend `{resolved_backend}`.")

        if resolved_backend == "lexical":
            processor = None
            if requested_backend == "auto":
                backend_notes.append("Auto backend fell back to lexical mode because simulate requirements were unavailable.")

        for case_name in requested_case_ids(case):
            payload = fit_case_payload_to_prompt_target(
                case_name=case_name,
                processor=processor,
                target_prompt_tokens=resolved_target_prompt_tokens,
                seed=seed,
            )
            if payload is not None:
                case_payloads.append(payload)

        if not case_payloads:
            raise UserFacingError("No long-context cases were available to run.")

        overall_pass = True
        for case_payload in case_payloads:
            report = execute_case(
                case_payload=case_payload,
                backend=resolved_backend,
                processor=processor,
                session=session,
                target_prompt_tokens=resolved_target_prompt_tokens,
            )
            case_reports.append(report)
            if not report["pass"]:
                overall_pass = False

        payload = artifact_payload("ok", overall_pass)
        write_artifact(resolved_output_path, payload)
        return payload, resolved_output_path
    except UserFacingError as exc:
        payload = artifact_payload("blocked", False, str(exc))
        write_artifact(resolved_output_path, payload)
        raise
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if classify_blocker(message).external:
            payload = artifact_payload("blocked", False, message)
            write_artifact(resolved_output_path, payload)
            raise UserFacingError(message) from exc
        payload = artifact_payload("failed", False, message)
        write_artifact(resolved_output_path, payload)
        raise
    finally:
        if owns_session_manager:
            manager.close_all()


def main() -> int:
    args = parse_args()
    payload, output_path = run_long_context_report(
        requested_backend=args.backend,
        case=args.case,
        model_id=args.model_id,
        target_prompt_tokens=args.target_prompt_tokens,
        seed=args.seed,
        output_path=args.out,
        print_runtime=True,
    )

    print(f"context_profile: {payload['context_profile']['label']}")
    print(f"requested_backend: {payload['requested_backend']}")
    print(f"resolved_backend: {payload['resolved_backend']}")
    print(f"long_context_validated: {payload['long_context_validated']}")
    for report in payload["cases"]:
        print(
            f"case={report['case_label']} pass={report['pass']} "
            f"prompt_tokens={report['prompt_token_estimate']} "
            f"prompt_fit_ratio={report['prompt_fit_ratio']:.3f} "
            f"latency_seconds={report['latency_seconds']:.3f}"
        )
    print(f"artifact_path: {output_path}")

    if not payload["overall_pass"]:
        return 1
    if payload["requested_backend"] == "live" and not payload["long_context_validated"]:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserFacingError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
