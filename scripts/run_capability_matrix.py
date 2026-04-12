#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from artifact_schema import (
    build_artifact_payload,
    build_prompt_record,
    build_runtime_record,
    collect_asset_lineage,
    normalize_device_info,
    read_artifact,
    write_artifact,
)
from audio_service import run_audio_mode, serialize_audio_record
from asset_preprocessing import sample_video_frames
from blocker_taxonomy import REPO_BUG, build_blocker_record, classify_blocker
from fetch_demo_assets import assets_root, ensure_asset_dirs, process_samples, repo_root as assets_repo_root
from gemma_core import RuntimeSession, SessionManager, generate_text_from_messages
from gemma_runtime import (
    UserFacingError,
    build_text_messages,
    import_image_module,
    repo_root,
    resolve_audio_model_selection,
    resolve_model_id,
    timestamp_slug,
)
from run_long_context_demo import detect_context_profile, run_long_context_report
from text_service import (
    DEFAULT_PROMPTS as TEXT_DEFAULT_PROMPTS,
    GENERATION_SETTINGS as TEXT_GENERATION_SETTINGS,
    build_user_prompt as build_text_user_prompt,
    resolve_system_prompt as resolve_text_system_prompt,
)
from thinking_service import (
    TEXT_FOLLOW_UP,
    TEXT_PROMPT,
    TEXT_SYSTEM_PROMPT,
    TOOL_PROMPT,
    TOOL_SYSTEM_PROMPT,
    run_text_mode,
    run_tool_mode,
)
from vision_service import (
    GENERATION_SETTINGS as VISION_GENERATION_SETTINGS,
    build_messages as build_vision_messages,
    build_user_prompt as build_vision_user_prompt,
    resolve_inputs,
    resolve_prompt as resolve_vision_prompt,
    resolve_system_prompt as resolve_vision_system_prompt,
    serialize_input_records,
    validate_mode_inputs,
)


ALL_CAPABILITIES = [
    "text-chat",
    "summarization",
    "multilingual-translate",
    "code-generation",
    "image-caption",
    "ocr",
    "image-compare",
    "pdf-doc-summary",
    "audio-asr",
    "audio-translation",
    "function-calling",
    "structured-json",
    "thinking",
    "long-context",
    "video-understanding",
]

SMOKE_CAPABILITIES = [
    "text-chat",
    "structured-json",
    "image-caption",
    "audio-asr",
    "thinking",
]

PHASE_CAPABILITIES = {
    "phase1": [
        "text-chat",
        "summarization",
        "multilingual-translate",
        "code-generation",
        "structured-json",
    ],
    "phase2": [
        "image-caption",
        "ocr",
        "image-compare",
        "pdf-doc-summary",
        "video-understanding",
    ],
    "phase3": [
        "audio-asr",
        "audio-translation",
    ],
    "phase5": [
        "thinking",
        "function-calling",
    ],
    "phase6": [
        "long-context",
    ],
    "phase7": ALL_CAPABILITIES,
}

CAPABILITY_PHASE = {
    "text-chat": "phase1",
    "summarization": "phase1",
    "multilingual-translate": "phase1",
    "code-generation": "phase1",
    "structured-json": "phase1",
    "image-caption": "phase2",
    "ocr": "phase2",
    "image-compare": "phase2",
    "pdf-doc-summary": "phase2",
    "video-understanding": "phase2",
    "audio-asr": "phase3",
    "audio-translation": "phase3",
    "thinking": "phase5",
    "function-calling": "phase5",
    "long-context": "phase6",
}

JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
JAPANESE_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
TOKEN_RE = re.compile(r"[a-z0-9]+")
NO_DIFFERENCE_PHRASES = (
    "no difference",
    "no meaningful difference",
    "images are identical",
    "same image",
    "appear identical",
    "look identical",
)


class ExternalBlocker(RuntimeError):
    pass


class ValidationFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        execution_status: str = "ok",
        quality_status: str = "fail",
        quality_checks: list[dict[str, Any]] | None = None,
        quality_notes: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.execution_status = execution_status
        self.quality_status = quality_status
        self.quality_checks = list(quality_checks or [])
        self.quality_notes = list(quality_notes or [])


@dataclass
class CapabilityResult:
    capability: str
    phase: str
    status: str
    model_used: str | None
    asset_used: str | None
    validation_command: str
    artifact_path: str | None
    elapsed_seconds: float | None
    result: str
    blocker: dict[str, Any] | None
    execution_status: str
    quality_status: str
    validation_mode: str | None
    claim_scope: str | None
    pass_definition: str | None
    quality_checks: list[dict[str, Any]]
    quality_notes: list[str]
    output_preview: str | None
    notes: list[str]
    artifact_kind: str | None = None
    runtime_backend: str | None = None
    preprocessing_lineage: list[dict[str, Any]] = field(default_factory=list)


def artifacts_root() -> Path:
    return repo_root() / "artifacts" / "capability_matrix"


def default_output_path() -> Path:
    return artifacts_root() / f"{timestamp_slug()}-matrix.json"


def repo_python_path() -> Path:
    return Path(sys.executable).resolve()


def validation_command(capability: str, smoke: bool = False) -> str:
    base = "python scripts/run_capability_matrix.py"
    if smoke:
        return f"{base} --smoke --only {capability}"
    return f"{base} --only {capability}"


def capability_artifact_kind(capability: str) -> str:
    if capability in {
        "text-chat",
        "summarization",
        "multilingual-translate",
        "code-generation",
        "structured-json",
    }:
        return "text"
    if capability in {
        "image-caption",
        "ocr",
        "image-compare",
        "pdf-doc-summary",
        "video-understanding",
    }:
        return "vision"
    if capability in {"audio-asr", "audio-translation"}:
        return "audio"
    if capability in {"thinking", "function-calling"}:
        return "thinking"
    if capability == "long-context":
        return "long_context"
    raise ValueError(f"Unsupported capability artifact kind for `{capability}`.")


def artifact_root_for_kind(artifact_kind: str) -> Path:
    roots = {
        "text": repo_root() / "artifacts" / "text",
        "vision": repo_root() / "artifacts" / "vision",
        "audio": repo_root() / "artifacts" / "audio",
        "thinking": repo_root() / "artifacts" / "thinking",
        "long_context": repo_root() / "artifacts" / "capability_matrix",
    }
    try:
        return roots[artifact_kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported artifact kind `{artifact_kind}`.") from exc


def default_capability_artifact_path(capability: str) -> Path:
    artifact_kind = capability_artifact_kind(capability)
    return artifact_root_for_kind(artifact_kind) / f"{timestamp_slug()}-{capability}.json"


def default_artifact_index_path(matrix_output_path: Path) -> Path:
    name = matrix_output_path.name
    if name.endswith("-matrix.json"):
        return matrix_output_path.with_name(name[: -len("-matrix.json")] + "-artifact-index.json")
    return matrix_output_path.with_name(matrix_output_path.stem + "-artifact-index.json")


def workspace_relative_path(path: str | Path | None) -> str | None:
    if path in (None, ""):
        return None
    resolved = Path(path).expanduser().resolve()
    root = Path(repo_root()).resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return None


def build_validation_record(
    *,
    validation_mode: str | None,
    claim_scope: str | None,
    pass_definition: str | None,
    execution_status: str,
    quality_status: str,
    quality_checks: list[dict[str, Any]] | None = None,
    quality_notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "validation_mode": validation_mode,
        "claim_scope": claim_scope,
        "pass_definition": pass_definition,
        "execution_status": execution_status,
        "quality_status": quality_status,
        "quality_checks": list(quality_checks or []),
        "quality_notes": list(quality_notes or []),
    }


def write_capability_artifact(
    *,
    capability: str,
    runtime_backend: str,
    model_id: str | None,
    device_info: dict[str, Any] | str | None,
    elapsed_seconds: float | None,
    prompts: dict[str, Any],
    validation: dict[str, Any],
    extra: dict[str, Any] | None = None,
    asset_lineage: list[dict[str, Any]] | None = None,
    blocker_message: str | None = None,
    output_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    artifact_kind = capability_artifact_kind(capability)
    resolved_output_path = (output_path or default_capability_artifact_path(capability)).resolve()
    status = derive_result_status(
        execution_status=str(validation.get("execution_status") or "ok"),
        quality_status=str(validation.get("quality_status") or "pass"),
    )
    payload = build_artifact_payload(
        artifact_kind=artifact_kind,
        status=status,
        runtime=build_runtime_record(
            backend=runtime_backend,
            model_id=model_id,
            device_info=device_info,
            elapsed_seconds=elapsed_seconds,
        ),
        prompts=prompts,
        asset_lineage=asset_lineage,
        blocker_message=blocker_message,
        extra={
            "capability": capability,
            "phase": CAPABILITY_PHASE[capability],
            "entrypoint": "capability_matrix",
            "validation": validation,
            **(extra or {}),
        },
    )
    write_artifact(resolved_output_path, payload)
    return resolved_output_path, payload


def build_artifact_index_entry(result: CapabilityResult) -> dict[str, Any]:
    artifact_path = result.artifact_path
    stored_payload: dict[str, Any] | None = None
    if artifact_path:
        resolved_path = Path(artifact_path).expanduser().resolve()
        if resolved_path.exists():
            stored_payload = read_artifact(resolved_path)
            artifact_path = str(resolved_path)

    validation = (stored_payload or {}).get("validation") or build_validation_record(
        validation_mode=result.validation_mode,
        claim_scope=result.claim_scope,
        pass_definition=result.pass_definition,
        execution_status=result.execution_status,
        quality_status=result.quality_status,
        quality_checks=result.quality_checks,
        quality_notes=result.quality_notes,
    )
    lineage = (
        ((stored_payload or {}).get("assets") or {}).get("lineage")
        or list(result.preprocessing_lineage)
    )
    runtime = (stored_payload or {}).get("runtime") or {}
    source_paths = [
        item.get("source_path")
        for item in lineage
        if isinstance(item, dict) and item.get("source_path")
    ]
    return {
        "capability": result.capability,
        "phase": result.phase,
        "status": result.status,
        "artifact_kind": result.artifact_kind or (stored_payload or {}).get("artifact_kind"),
        "artifact_path": artifact_path,
        "artifact_workspace_relative_path": workspace_relative_path(artifact_path),
        "artifact_status": (stored_payload or {}).get("status"),
        "artifact_timestamp_utc": (stored_payload or {}).get("timestamp_utc"),
        "runtime_backend": result.runtime_backend or runtime.get("backend"),
        "validation_mode": validation.get("validation_mode"),
        "claim_scope": validation.get("claim_scope"),
        "execution_status": validation.get("execution_status"),
        "quality_status": validation.get("quality_status"),
        "quality_checks": list(validation.get("quality_checks") or []),
        "quality_notes": list(validation.get("quality_notes") or []),
        "lineage_count": len(lineage),
        "lineage_source_paths": source_paths,
    }


def write_artifact_index(
    *,
    matrix_output_path: Path,
    overall_status: str,
    selected: list[str],
    results: list[CapabilityResult],
    model_id: str | None,
    elapsed_seconds: float,
) -> Path:
    output_path = default_artifact_index_path(matrix_output_path)
    entries = [build_artifact_index_entry(result) for result in results]
    payload = build_artifact_payload(
        artifact_kind="artifact_index",
        status=overall_status,
        runtime=build_runtime_record(
            backend="capability-matrix-index",
            model_id=model_id,
            device_info="matrix-artifact-index",
            elapsed_seconds=elapsed_seconds,
        ),
        prompts=build_prompt_record(),
        extra={
            "matrix_artifact_path": str(matrix_output_path.resolve()),
            "matrix_artifact_workspace_relative_path": workspace_relative_path(matrix_output_path),
            "selection": list(selected),
            "entry_count": len(entries),
            "entries": entries,
        },
    )
    write_artifact(output_path, payload)
    return output_path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Gemma 4 capability matrix with graceful skips for external blockers."
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated capability ids to run.",
    )
    parser.add_argument(
        "--phase",
        choices=["phase1", "phase2", "phase3", "phase5", "phase6", "phase7"],
        default=None,
        help="Run only the capabilities mapped to a single phase.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the reduced smoke subset instead of the full matrix.",
    )
    parser.add_argument(
        "--skip-prepare-assets",
        action="store_true",
        help="Skip asset preparation and use the current assets directory as-is.",
    )
    parser.add_argument(
        "--asset-timeout",
        type=float,
        default=3.0,
        help="Timeout in seconds for asset generation network probes (default: 3.0).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=4,
        help="Maximum PDF pages to rasterize for pdf/doc summary (default: 4).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output path. Defaults to artifacts/capability_matrix/<timestamp>-matrix.json",
    )
    return parser.parse_args()


def normalize_output_preview(text: str | None, limit: int = 220) -> str | None:
    if not text:
        return None
    compact = " ".join(text.strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def build_quality_check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "name": name,
        "pass": bool(passed),
        "detail": detail,
    }


def normalize_tokens(text: str | None) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def normalized_token_set(text: str | None) -> set[str]:
    return set(normalize_tokens(text))


def has_all_tokens(tokens: set[str], required: tuple[str, ...]) -> bool:
    return all(token in tokens for token in required)


def has_any_token_group(tokens: set[str], groups: tuple[tuple[str, ...], ...]) -> bool:
    return any(has_all_tokens(tokens, group) for group in groups)


def derive_result_status(*, execution_status: str, quality_status: str) -> str:
    if execution_status == "blocked":
        return "blocked"
    if execution_status != "ok":
        return "failed"
    if quality_status == "pass":
        return "ok"
    if quality_status in {"warning", "not_validated", "inconclusive", "not_run"}:
        return "not_validated"
    if quality_status == "fail":
        return "quality_fail"
    return "failed"


def summarize_failed_checks(checks: list[dict[str, Any]]) -> str:
    failed_checks = [check["detail"] for check in checks if not check["pass"]]
    return "; ".join(failed_checks) or "Deterministic quality checks did not pass."


def text_validation_record(
    *,
    quality_status: str = "pass",
    quality_checks: list[dict[str, Any]] | None = None,
    quality_notes: list[str] | None = None,
) -> dict[str, Any]:
    return build_validation_record(
        validation_mode="live",
        claim_scope="live model generation on a small local prompt",
        pass_definition="Pass means execution completed and the output satisfied the capability-specific contract check.",
        execution_status="ok",
        quality_status=quality_status,
        quality_checks=quality_checks,
        quality_notes=quality_notes,
    )


def assess_text_translation_output(output_text: str) -> dict[str, Any]:
    checks = [
        build_quality_check(
            "translated_out_of_source_script",
            not bool(JAPANESE_PATTERN.search(output_text)),
            "The translated output should not still contain Japanese characters for this English target check.",
        )
    ]
    quality_status = "pass" if all(check["pass"] for check in checks) else "fail"
    return text_validation_record(quality_status=quality_status, quality_checks=checks)


def assess_code_generation_output(output_text: str) -> dict[str, Any]:
    checks = [
        build_quality_check(
            "contains_python_function_shape",
            "def " in output_text or "```python" in output_text,
            "The code-generation sample should contain an obvious Python function definition or fenced Python block.",
        )
    ]
    quality_status = "pass" if all(check["pass"] for check in checks) else "fail"
    return text_validation_record(quality_status=quality_status, quality_checks=checks)


def assess_structured_json_output(output_text: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        payload = extract_json_object(output_text)
    except ValidationFailure as exc:
        checks = [
            build_quality_check(
                "valid_json_object",
                False,
                str(exc),
            )
        ]
        return text_validation_record(
            quality_status="fail",
            quality_checks=checks,
            quality_notes=[str(exc)],
        ), None

    checks = [
        build_quality_check(
            "capability_field_present",
            payload.get("capability") == "structured-json",
            "The JSON object should contain capability='structured-json'.",
        ),
        build_quality_check(
            "language_field_present",
            payload.get("language") == "English",
            "The JSON object should contain language='English'.",
        ),
        build_quality_check(
            "checks_field_shape",
            isinstance(payload.get("checks"), list) and len(payload["checks"]) == 2,
            "The JSON object should contain exactly 2 checks.",
        ),
    ]
    quality_status = "pass" if all(check["pass"] for check in checks) else "fail"
    return text_validation_record(
        quality_status=quality_status,
        quality_checks=checks,
    ), payload


def assess_thinking_output(result: dict[str, Any]) -> dict[str, Any]:
    final_answer = str(result.get("final_answer") or "").strip()
    checks = [
        build_quality_check(
            "thinking_removed_before_follow_up",
            bool(result.get("thinking_stripped_for_follow_up")),
            "The follow-up turn should strip prior-turn private thinking from the conversation history.",
        ),
        build_quality_check(
            "final_answer_present",
            bool(final_answer),
            "The thinking run should produce a visible final answer.",
        ),
    ]
    quality_status = "pass" if all(check["pass"] for check in checks) else "fail"
    return text_validation_record(
        quality_status=quality_status,
        quality_checks=checks,
    )


def assess_function_calling_output(result: dict[str, Any]) -> dict[str, Any]:
    iterations = result.get("iterations") or []
    saw_tool_results = any(item.get("tool_results") for item in iterations if isinstance(item, dict))
    final_answer = str(result.get("final_answer") or "").strip()
    checks = [
        build_quality_check(
            "tool_call_executed",
            saw_tool_results,
            "The tool-assisted run should execute at least one tool call before answering.",
        ),
        build_quality_check(
            "final_answer_present",
            bool(final_answer),
            "The tool-assisted run should produce a visible final answer.",
        ),
    ]
    quality_status = "pass" if all(check["pass"] for check in checks) else "fail"
    return text_validation_record(
        quality_status=quality_status,
        quality_checks=checks,
    )


def quality_assessment_result(
    *,
    capability: str,
    model_used: str | None,
    message: str,
    execution_status: str = "ok",
    quality_status: str = "fail",
    quality_checks: list[dict[str, Any]] | None = None,
    quality_notes: list[str] | None = None,
    notes: list[str] | None = None,
    artifact_path: str | None = None,
    artifact_kind: str | None = None,
    runtime_backend: str | None = None,
    preprocessing_lineage: list[dict[str, Any]] | None = None,
) -> CapabilityResult:
    assessment_notes = list(quality_notes or [])
    normalized_message = " ".join((message or "").strip().split())
    if normalized_message and normalized_message not in assessment_notes:
        assessment_notes.append(normalized_message)

    status = derive_result_status(execution_status=execution_status, quality_status=quality_status)
    return CapabilityResult(
        capability=capability,
        phase=CAPABILITY_PHASE[capability],
        status=status,
        model_used=model_used,
        asset_used=None,
        validation_command=validation_command(capability),
        artifact_path=artifact_path,
        elapsed_seconds=None,
        result=status,
        blocker=None,
        execution_status=execution_status,
        quality_status=quality_status,
        validation_mode=None,
        claim_scope=None,
        pass_definition=None,
        quality_checks=list(quality_checks or []),
        quality_notes=assessment_notes,
        output_preview=None,
        notes=notes or [],
        artifact_kind=artifact_kind,
        runtime_backend=runtime_backend,
        preprocessing_lineage=list(preprocessing_lineage or []),
    )


def assess_caption_output(output_text: str) -> dict[str, Any]:
    tokens = normalized_token_set(output_text)
    checks = [
        build_quality_check(
            "mentions_cup_or_mug",
            bool({"cup", "mug"} & tokens),
            "The synthetic caption fixture should mention the visible cup or mug.",
        ),
        build_quality_check(
            "mentions_notebook",
            bool({"notebook", "notepad"} & tokens),
            "The synthetic caption fixture should mention the notebook beside the cup.",
        ),
    ]
    quality_status = "pass" if all(check["pass"] for check in checks) else "fail"
    return {
        "validation_mode": "live",
        "claim_scope": "live caption generation on the deterministic local desk-scene fixture",
        "pass_definition": (
            "Pass means the caption named the two primary fixture objects instead of only producing a generic scene description."
        ),
        "quality_status": quality_status,
        "quality_checks": checks,
        "quality_notes": [
            "Validated against the deterministic local image fixture generated by scripts/fetch_demo_assets.py."
        ],
    }


def assess_ocr_output(output_text: str) -> dict[str, Any]:
    tokens = normalized_token_set(output_text)
    checks = [
        build_quality_check(
            "invoice_title_present",
            has_all_tokens(tokens, ("gemma", "lab", "invoice")),
            "The OCR output should preserve the 'Gemma Lab Invoice' title.",
        ),
        build_quality_check(
            "invoice_id_present",
            has_all_tokens(tokens, ("invoice", "id", "g4", "001")),
            "The OCR output should preserve the local invoice id `G4-001`.",
        ),
        build_quality_check(
            "total_items_present",
            has_all_tokens(tokens, ("total", "items", "3")),
            "The OCR output should preserve the `Total Items: 3` line.",
        ),
        build_quality_check(
            "status_line_present",
            has_all_tokens(tokens, ("status", "ready", "for", "ocr")),
            "The OCR output should preserve the `STATUS: READY FOR OCR` line.",
        ),
    ]
    quality_status = "pass" if all(check["pass"] for check in checks) else "fail"
    return {
        "validation_mode": "live",
        "claim_scope": "live OCR on the deterministic local invoice fixture",
        "pass_definition": (
            "Pass means the OCR output preserved the anchor text fields from the deterministic local invoice image."
        ),
        "quality_status": quality_status,
        "quality_checks": checks,
        "quality_notes": [
            "Validated against anchor lines from the locally generated OCR fixture image."
        ],
    }


def assess_compare_output(output_text: str) -> dict[str, Any]:
    lowered = (output_text or "").lower()
    tokens = normalized_token_set(output_text)
    cup_tokens = {"cup", "mug"}
    color_tokens = {"teal", "blue", "green", "turquoise"}
    moved_right = (
        ("right" in tokens and bool({"move", "moved", "shift", "shifted"} & tokens))
        or "moved to the right" in lowered
        or "shifted to the right" in lowered
    )
    text_change = (
        has_any_token_group(
            tokens,
            (
                ("updated", "layout"),
                ("annotation", "changed"),
                ("text", "changed"),
                ("notebook", "annotation"),
                ("notebook", "text"),
            ),
        )
        or "updated layout" in lowered
    )
    denies_difference = any(phrase in lowered for phrase in NO_DIFFERENCE_PHRASES)

    checks = [
        build_quality_check(
            "does_not_claim_identical_images",
            not denies_difference,
            "The comparison fixture has visible differences and should not be described as identical.",
        ),
        build_quality_check(
            "mentions_cup_shift",
            bool(cup_tokens & tokens) and moved_right,
            "The output should note that the cup or mug moved to the right in the second image.",
        ),
        build_quality_check(
            "mentions_cup_color_change",
            bool(cup_tokens & tokens) and bool(color_tokens & tokens),
            "The output should note that the cup or mug color changed toward teal/blue-green.",
        ),
        build_quality_check(
            "mentions_annotation_change",
            text_change,
            "The output should note that the notebook annotation or text changed to the updated layout wording.",
        ),
    ]

    matched_differences = sum(
        1
        for check in checks
        if check["name"] in {"mentions_cup_shift", "mentions_cup_color_change", "mentions_annotation_change"}
        and check["pass"]
    )
    quality_status = "pass" if checks[0]["pass"] and matched_differences >= 2 else "fail"
    return {
        "validation_mode": "live",
        "claim_scope": "live image comparison on the deterministic local fixture pair",
        "pass_definition": (
            "Pass means the output identified at least two of the three fixture differences and did not deny that the images changed."
        ),
        "quality_status": quality_status,
        "quality_checks": checks,
        "quality_notes": [
            "Validated against the deterministic sample and sample_compare fixture pair generated locally."
        ],
    }


def assess_pdf_summary_output(output_text: str) -> dict[str, Any]:
    lowered = (output_text or "").lower()
    tokens = normalized_token_set(output_text)
    goal_terms = {"caption", "ocr", "compare", "document", "summary"}

    checks = [
        build_quality_check(
            "includes_key_facts_heading",
            "key facts" in lowered,
            "The pdf-summary prompt requires the heading `Key facts:` in the output.",
        ),
        build_quality_check(
            "mentions_owner",
            has_all_tokens(tokens, ("gemma", "lab")),
            "The summary should preserve that the sample document is owned by Gemma Lab.",
        ),
        build_quality_check(
            "mentions_phase",
            has_all_tokens(tokens, ("phase", "2")),
            "The summary should preserve that the sample document is for Phase 2.",
        ),
        build_quality_check(
            "mentions_launch_date",
            "2026-04-06" in lowered or has_all_tokens(tokens, ("2026", "04", "06")),
            "The summary should preserve the launch date `2026-04-06`.",
        ),
        build_quality_check(
            "mentions_goal_capabilities",
            len(goal_terms & tokens) >= 3,
            "The summary should preserve that the goal covers caption, OCR, compare, and document summary flows.",
        ),
    ]

    matched_facts = sum(check["pass"] for check in checks[1:])
    quality_status = "pass" if checks[0]["pass"] and matched_facts >= 3 else "fail"
    return {
        "validation_mode": "live",
        "claim_scope": "live PDF summary on the deterministic local single-page fixture",
        "pass_definition": (
            "Pass means the summary preserved at least three fixture facts and kept the requested `Key facts:` section."
        ),
        "quality_status": quality_status,
        "quality_checks": checks,
        "quality_notes": [
            "Validated against fixture facts from the locally generated sample PDF."
        ],
    }


def is_external_blocker_message(message: str) -> bool:
    return classify_blocker(message).external


def asset_path(relative_path: str) -> Path:
    return repo_root() / relative_path


def ensure_existing_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ExternalBlocker(f"Input file does not exist: `{resolved}`.")
    return resolved


def prepare_assets(timeout: float) -> dict[str, Any]:
    if assets_repo_root() != repo_root():
        raise RuntimeError("Asset preparation resolved an unexpected repository root.")

    created_dirs = ensure_asset_dirs(dry_run=False)
    manifest = process_samples(dry_run=False, timeout=timeout)
    manifest["assets_root"] = str(assets_root().relative_to(repo_root()))
    manifest["created_directories"] = [
        str(path.relative_to(repo_root()))
        for path in created_dirs
    ]
    manifest_path = assets_root() / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_text_generation(
    session: RuntimeSession,
    system_prompt: str,
    user_prompt: str,
    generation_settings: dict[str, Any],
) -> tuple[str, float]:
    try:
        generation = generate_text_from_messages(
            session=session,
            messages=build_text_messages(system_prompt=system_prompt, user_prompt=user_prompt),
            generation_settings=generation_settings,
        )
        output_text = generation.output_text.strip()
        if not output_text:
            raise ValidationFailure("Model returned an empty text response.")
        return output_text, generation.elapsed_seconds
    except UserFacingError as exc:
        message = str(exc)
        if is_external_blocker_message(message):
            raise ExternalBlocker(message) from exc
        raise ValidationFailure(message) from exc


def run_vision_generation(
    session: RuntimeSession,
    mode: str,
    input_paths: list[Path],
    max_pages: int,
    prompt: str | None = None,
) -> tuple[str, float]:
    try:
        image_module = import_image_module()
        records, pdf_inputs = resolve_inputs(
            input_paths=input_paths,
            image_module=image_module,
            max_pages=max_pages,
        )
        validate_mode_inputs(mode, records, pdf_inputs)
        return run_vision_generation_from_records(session, mode, records, prompt=prompt)
    except UserFacingError as exc:
        message = str(exc)
        if is_external_blocker_message(message):
            raise ExternalBlocker(message) from exc
        raise ValidationFailure(message) from exc


def assess_video_proxy_output(
    records: list[dict[str, Any]],
    output_text: str,
) -> dict[str, Any]:
    deltas: list[float] = []
    for left, right in zip(records, records[1:]):
        left_pixels = np.asarray(left["image"], dtype=np.float32)
        right_pixels = np.asarray(right["image"], dtype=np.float32)
        deltas.append(float(np.abs(left_pixels - right_pixels).mean()))

    max_delta = max(deltas, default=0.0)
    lower_text = (output_text or "").lower()
    mentions_change = any(
        phrase in lower_text
        for phrase in (
            "change",
            "changes",
            "changing",
            "move",
            "moves",
            "moving",
            "shift",
            "different",
            "over time",
            "progress",
        )
    )
    denies_change = any(
        phrase in lower_text
        for phrase in (
            "no meaningful change",
            "identical",
            "same image",
            "all three images appear to be identical",
            "not possible to describe changes",
        )
    )

    checks = [
        build_quality_check(
            "sampled_multiple_frames",
            len(records) >= 3,
            f"Observed {len(records)} sampled frame(s).",
        ),
        build_quality_check(
            "frame_differences_detected",
            max_delta >= 4.0,
            f"Maximum mean pixel delta across sampled frames was {max_delta:.3f}.",
        ),
        build_quality_check(
            "output_acknowledges_change",
            mentions_change and not denies_change,
            "The output should describe visible change over time for the proxy clip.",
        ),
    ]
    quality_status = "pass" if all(item["pass"] for item in checks) else "fail"
    quality_notes = [
        "This remains a frame-sampled proxy validated through ffmpeg extraction plus the vision model.",
        "It does not claim native video-token support.",
    ]
    return {
        "validation_mode": "proxy",
        "claim_scope": "frame-sampled video proxy over cached still frames",
        "pass_definition": (
            "Pass means ffmpeg extracted changing frames from the sample clip and the vision model explicitly described that change."
        ),
        "quality_status": quality_status,
        "quality_checks": checks,
        "quality_notes": quality_notes,
    }


def run_vision_generation_from_records(
    session: RuntimeSession,
    mode: str,
    records: list[dict[str, Any]],
    prompt: str | None = None,
) -> tuple[str, float]:
    base_prompt = resolve_vision_prompt(mode, prompt)
    system_prompt = resolve_vision_system_prompt(mode, None)
    user_prompt = build_vision_user_prompt(mode, base_prompt, image_count=len(records))
    generation = generate_text_from_messages(
        session=session,
        messages=build_vision_messages(system_prompt=system_prompt, user_prompt=user_prompt, records=records),
        generation_settings=VISION_GENERATION_SETTINGS[mode],
        template_kwargs={
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        },
    )
    output_text = generation.output_text.strip()
    if not output_text:
        raise ValidationFailure(f"Model returned an empty `{mode}` response.")
    return output_text, generation.elapsed_seconds


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = JSON_FENCE_RE.sub("", text.strip()).strip()
    candidate = cleaned
    if not (candidate.startswith("{") and candidate.endswith("}")):
        match = JSON_OBJECT_RE.search(cleaned)
        if not match:
            raise ValidationFailure("Structured JSON response did not contain a JSON object.")
        candidate = match.group(0)

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValidationFailure(f"Structured JSON response was not valid JSON ({exc}).") from exc

    if not isinstance(payload, dict):
        raise ValidationFailure("Structured JSON response was not a JSON object.")
    return payload


def run_long_context(output_path: Path, session_manager: SessionManager) -> tuple[dict[str, Any], float]:
    started_at = time.perf_counter()
    try:
        payload, _ = run_long_context_report(
            output_path=output_path,
            session_manager=session_manager,
            requested_backend="live",
            print_runtime=False,
        )
    except UserFacingError as exc:
        message = str(exc)
        if is_external_blocker_message(message):
            raise ExternalBlocker(message) from exc
        raise ValidationFailure(message) from exc

    elapsed_seconds = time.perf_counter() - started_at
    if not output_path.exists():
        raise RuntimeError("Long-context report did not write an artifact.")
    stored_payload = read_artifact(output_path)
    if stored_payload.get("status") != "ok":
        raise ValidationFailure(
            "Long-context validation did not finish cleanly.",
            quality_status="not_validated",
        )
    if not stored_payload.get("overall_pass", False):
        raise ValidationFailure("Long-context validation reported a failed marker retrieval.")
    if not stored_payload.get("long_context_validated", False):
        raise ValidationFailure(
            "Long-context validation did not reach a live validated result.",
            quality_status="not_validated",
        )
    return stored_payload, elapsed_seconds


def build_text_result(
    capability: str,
    model_used: str,
    output_text: str,
    elapsed_seconds: float,
    *,
    execution_status: str = "ok",
    quality_status: str = "pass",
    quality_checks: list[dict[str, Any]] | None = None,
    quality_notes: list[str] | None = None,
    notes: list[str] | None = None,
    artifact_path: str | None = None,
    artifact_kind: str | None = None,
    runtime_backend: str | None = None,
    preprocessing_lineage: list[dict[str, Any]] | None = None,
) -> CapabilityResult:
    status = derive_result_status(execution_status=execution_status, quality_status=quality_status)
    return CapabilityResult(
        capability=capability,
        phase=CAPABILITY_PHASE[capability],
        status=status,
        model_used=model_used,
        asset_used=None,
        validation_command=validation_command(capability),
        artifact_path=artifact_path,
        elapsed_seconds=round(elapsed_seconds, 3),
        result=status,
        blocker=None,
        execution_status=execution_status,
        quality_status=quality_status,
        validation_mode="live",
        claim_scope="live model generation on a small local prompt",
        pass_definition="Pass means execution completed and the output satisfied the capability-specific contract check.",
        quality_checks=quality_checks or [],
        quality_notes=quality_notes or [],
        output_preview=normalize_output_preview(output_text),
        notes=notes or [],
        artifact_kind=artifact_kind,
        runtime_backend=runtime_backend,
        preprocessing_lineage=list(preprocessing_lineage or []),
    )


def run_selected_text_capabilities(
    session: RuntimeSession,
    selected: set[str],
) -> list[CapabilityResult]:
    results: list[CapabilityResult] = []

    def attempt(capability: str, runner: Any) -> None:
        try:
            results.append(runner())
        except ExternalBlocker as exc:
            results.append(blocked_result(capability, session.model_id, str(exc)))
        except ValidationFailure as exc:
            results.append(
                quality_assessment_result(
                    capability=capability,
                    model_used=session.model_id,
                    message=str(exc),
                    execution_status=exc.execution_status,
                    quality_status=exc.quality_status,
                    quality_checks=exc.quality_checks,
                    quality_notes=exc.quality_notes,
                )
            )
        except Exception as exc:
            results.append(failed_result(capability, session.model_id, str(exc)))

    if "text-chat" in selected:
        def runner() -> CapabilityResult:
            prompt = TEXT_DEFAULT_PROMPTS["chat"]
            system_prompt = resolve_text_system_prompt("chat", None)
            user_prompt = build_text_user_prompt("chat", prompt)
            output_text, elapsed = run_text_generation(
                session=session,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                generation_settings=TEXT_GENERATION_SETTINGS["chat"],
            )
            validation = text_validation_record()
            artifact_path, artifact_payload = write_capability_artifact(
                capability="text-chat",
                runtime_backend="gemma-live-text",
                model_id=session.model_id,
                device_info=session.device_info,
                elapsed_seconds=elapsed,
                prompts=build_prompt_record(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    resolved_user_prompt=user_prompt,
                ),
                validation=validation,
                extra={
                    "task": "chat",
                    "model_id": session.model_id,
                    "device": normalize_device_info(session.device_info)["label"],
                    "dtype": normalize_device_info(session.device_info)["dtype"],
                    "generation_settings": dict(TEXT_GENERATION_SETTINGS["chat"]),
                    "output_text": output_text,
                    "elapsed_seconds": round(elapsed, 3),
                },
            )
            artifact_payload = read_artifact(artifact_path)
            return build_text_result(
                "text-chat",
                session.model_id,
                output_text,
                elapsed,
                execution_status=str(validation["execution_status"]),
                quality_status=str(validation["quality_status"]),
                quality_checks=list(validation["quality_checks"]),
                quality_notes=list(validation["quality_notes"]),
                artifact_path=str(artifact_path),
                artifact_kind=artifact_payload.get("artifact_kind"),
                runtime_backend=(artifact_payload.get("runtime") or {}).get("backend"),
                preprocessing_lineage=((artifact_payload.get("assets") or {}).get("lineage") or []),
            )

        attempt("text-chat", runner)

    if "summarization" in selected:
        def runner() -> CapabilityResult:
            prompt = TEXT_DEFAULT_PROMPTS["summarize"]
            system_prompt = resolve_text_system_prompt("summarize", None)
            user_prompt = build_text_user_prompt("summarize", prompt)
            output_text, elapsed = run_text_generation(
                session=session,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                generation_settings=TEXT_GENERATION_SETTINGS["summarize"],
            )
            validation = text_validation_record()
            artifact_path, artifact_payload = write_capability_artifact(
                capability="summarization",
                runtime_backend="gemma-live-text",
                model_id=session.model_id,
                device_info=session.device_info,
                elapsed_seconds=elapsed,
                prompts=build_prompt_record(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    resolved_user_prompt=user_prompt,
                ),
                validation=validation,
                extra={
                    "task": "summarize",
                    "model_id": session.model_id,
                    "device": normalize_device_info(session.device_info)["label"],
                    "dtype": normalize_device_info(session.device_info)["dtype"],
                    "generation_settings": dict(TEXT_GENERATION_SETTINGS["summarize"]),
                    "output_text": output_text,
                    "elapsed_seconds": round(elapsed, 3),
                },
            )
            artifact_payload = read_artifact(artifact_path)
            return build_text_result(
                "summarization",
                session.model_id,
                output_text,
                elapsed,
                execution_status=str(validation["execution_status"]),
                quality_status=str(validation["quality_status"]),
                quality_checks=list(validation["quality_checks"]),
                quality_notes=list(validation["quality_notes"]),
                artifact_path=str(artifact_path),
                artifact_kind=artifact_payload.get("artifact_kind"),
                runtime_backend=(artifact_payload.get("runtime") or {}).get("backend"),
                preprocessing_lineage=((artifact_payload.get("assets") or {}).get("lineage") or []),
            )

        attempt("summarization", runner)

    if "multilingual-translate" in selected:
        def runner() -> CapabilityResult:
            prompt = TEXT_DEFAULT_PROMPTS["translate"]
            system_prompt = resolve_text_system_prompt("translate", None)
            user_prompt = build_text_user_prompt("translate", prompt)
            output_text, elapsed = run_text_generation(
                session=session,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                generation_settings=TEXT_GENERATION_SETTINGS["translate"],
            )
            validation = assess_text_translation_output(output_text)
            artifact_path, artifact_payload = write_capability_artifact(
                capability="multilingual-translate",
                runtime_backend="gemma-live-text",
                model_id=session.model_id,
                device_info=session.device_info,
                elapsed_seconds=elapsed,
                prompts=build_prompt_record(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    resolved_user_prompt=user_prompt,
                ),
                validation=validation,
                extra={
                    "task": "translate",
                    "model_id": session.model_id,
                    "device": normalize_device_info(session.device_info)["label"],
                    "dtype": normalize_device_info(session.device_info)["dtype"],
                    "generation_settings": dict(TEXT_GENERATION_SETTINGS["translate"]),
                    "output_text": output_text,
                    "elapsed_seconds": round(elapsed, 3),
                },
            )
            artifact_payload = read_artifact(artifact_path)
            return build_text_result(
                "multilingual-translate",
                session.model_id,
                output_text,
                elapsed,
                execution_status=str(validation["execution_status"]),
                quality_status=str(validation["quality_status"]),
                quality_checks=list(validation["quality_checks"]),
                quality_notes=list(validation["quality_notes"]),
                artifact_path=str(artifact_path),
                artifact_kind=artifact_payload.get("artifact_kind"),
                runtime_backend=(artifact_payload.get("runtime") or {}).get("backend"),
                preprocessing_lineage=((artifact_payload.get("assets") or {}).get("lineage") or []),
            )

        attempt("multilingual-translate", runner)

    if "code-generation" in selected:
        def runner() -> CapabilityResult:
            prompt = TEXT_DEFAULT_PROMPTS["code"]
            system_prompt = resolve_text_system_prompt("code", None)
            user_prompt = build_text_user_prompt("code", prompt)
            output_text, elapsed = run_text_generation(
                session=session,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                generation_settings=TEXT_GENERATION_SETTINGS["code"],
            )
            validation = assess_code_generation_output(output_text)
            artifact_path, artifact_payload = write_capability_artifact(
                capability="code-generation",
                runtime_backend="gemma-live-text",
                model_id=session.model_id,
                device_info=session.device_info,
                elapsed_seconds=elapsed,
                prompts=build_prompt_record(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    resolved_user_prompt=user_prompt,
                ),
                validation=validation,
                extra={
                    "task": "code",
                    "model_id": session.model_id,
                    "device": normalize_device_info(session.device_info)["label"],
                    "dtype": normalize_device_info(session.device_info)["dtype"],
                    "generation_settings": dict(TEXT_GENERATION_SETTINGS["code"]),
                    "output_text": output_text,
                    "elapsed_seconds": round(elapsed, 3),
                },
            )
            artifact_payload = read_artifact(artifact_path)
            return build_text_result(
                "code-generation",
                session.model_id,
                output_text,
                elapsed,
                execution_status=str(validation["execution_status"]),
                quality_status=str(validation["quality_status"]),
                quality_checks=list(validation["quality_checks"]),
                quality_notes=list(validation["quality_notes"]),
                artifact_path=str(artifact_path),
                artifact_kind=artifact_payload.get("artifact_kind"),
                runtime_backend=(artifact_payload.get("runtime") or {}).get("backend"),
                preprocessing_lineage=((artifact_payload.get("assets") or {}).get("lineage") or []),
            )

        attempt("code-generation", runner)

    if "structured-json" in selected:
        def runner() -> CapabilityResult:
            system_prompt = "You return strictly valid JSON and nothing else."
            prompt = (
                "Return a JSON object with exactly these keys: capability, language, checks. "
                "Set capability to 'structured-json', language to 'English', and checks to an array "
                "with exactly 2 short strings."
            )
            output_text, elapsed = run_text_generation(
                session=session,
                system_prompt=system_prompt,
                user_prompt=prompt,
                generation_settings={"max_new_tokens": 128, "do_sample": False},
            )
            validation, payload = assess_structured_json_output(output_text)
            preview_text = json.dumps(payload, ensure_ascii=False) if payload is not None else output_text
            artifact_path, artifact_payload = write_capability_artifact(
                capability="structured-json",
                runtime_backend="gemma-live-text",
                model_id=session.model_id,
                device_info=session.device_info,
                elapsed_seconds=elapsed,
                prompts=build_prompt_record(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    resolved_user_prompt=prompt,
                ),
                validation=validation,
                extra={
                    "task": "structured-json",
                    "model_id": session.model_id,
                    "device": normalize_device_info(session.device_info)["label"],
                    "dtype": normalize_device_info(session.device_info)["dtype"],
                    "generation_settings": {"max_new_tokens": 128, "do_sample": False},
                    "output_text": output_text,
                    "parsed_output": payload,
                    "elapsed_seconds": round(elapsed, 3),
                },
            )
            artifact_payload = read_artifact(artifact_path)
            return build_text_result(
                "structured-json",
                session.model_id,
                preview_text,
                elapsed,
                execution_status=str(validation["execution_status"]),
                quality_status=str(validation["quality_status"]),
                quality_checks=list(validation["quality_checks"]),
                quality_notes=list(validation["quality_notes"]),
                artifact_path=str(artifact_path),
                artifact_kind=artifact_payload.get("artifact_kind"),
                runtime_backend=(artifact_payload.get("runtime") or {}).get("backend"),
                preprocessing_lineage=((artifact_payload.get("assets") or {}).get("lineage") or []),
            )

        attempt("structured-json", runner)

    if "thinking" in selected:
        def runner() -> CapabilityResult:
            started_at = time.perf_counter()
            thinking_result = run_text_mode(
                session=session,
                system_prompt=TEXT_SYSTEM_PROMPT,
                first_prompt=TEXT_PROMPT,
                follow_up_prompt=TEXT_FOLLOW_UP,
                show_thinking=False,
            )
            elapsed = time.perf_counter() - started_at
            final_answer = str(thinking_result.get("final_answer", "")).strip()
            validation = assess_thinking_output(thinking_result)
            artifact_path, artifact_payload = write_capability_artifact(
                capability="thinking",
                runtime_backend="gemma-live-thinking",
                model_id=session.model_id,
                device_info=session.device_info,
                elapsed_seconds=elapsed,
                prompts=build_prompt_record(
                    system_prompt=TEXT_SYSTEM_PROMPT,
                    prompt=TEXT_PROMPT,
                    resolved_user_prompt=TEXT_PROMPT,
                    extra={"follow_up_prompt": TEXT_FOLLOW_UP},
                ),
                validation=validation,
                extra={
                    "mode": "text",
                    "model_id": session.model_id,
                    "device": normalize_device_info(session.device_info)["label"],
                    "dtype": normalize_device_info(session.device_info)["dtype"],
                    "stdout_behavior": "final_answer_only",
                    "raw_thinking_saved_to_artifact": True,
                    "simulation": False,
                    **thinking_result,
                    "elapsed_seconds": round(elapsed, 3),
                },
            )
            artifact_payload = read_artifact(artifact_path)
            return build_text_result(
                "thinking",
                session.model_id,
                final_answer,
                elapsed,
                execution_status=str(validation["execution_status"]),
                quality_status=str(validation["quality_status"]),
                quality_checks=list(validation["quality_checks"]),
                quality_notes=list(validation["quality_notes"]),
                artifact_path=str(artifact_path),
                artifact_kind=artifact_payload.get("artifact_kind"),
                runtime_backend=(artifact_payload.get("runtime") or {}).get("backend"),
                preprocessing_lineage=((artifact_payload.get("assets") or {}).get("lineage") or []),
            )

        attempt("thinking", runner)

    if "function-calling" in selected:
        def runner() -> CapabilityResult:
            started_at = time.perf_counter()
            tool_result = run_tool_mode(
                session=session,
                system_prompt=TOOL_SYSTEM_PROMPT,
                prompt=TOOL_PROMPT,
                show_thinking=False,
                max_tool_iterations=3,
            )
            elapsed = time.perf_counter() - started_at
            validation = assess_function_calling_output(tool_result)
            final_answer = str(tool_result.get("final_answer", "")).strip()
            artifact_path, artifact_payload = write_capability_artifact(
                capability="function-calling",
                runtime_backend="gemma-live-tool-use",
                model_id=session.model_id,
                device_info=session.device_info,
                elapsed_seconds=elapsed,
                prompts=build_prompt_record(
                    system_prompt=TOOL_SYSTEM_PROMPT,
                    prompt=TOOL_PROMPT,
                    resolved_user_prompt=TOOL_PROMPT,
                ),
                validation=validation,
                extra={
                    "mode": "tool",
                    "model_id": session.model_id,
                    "device": normalize_device_info(session.device_info)["label"],
                    "dtype": normalize_device_info(session.device_info)["dtype"],
                    "stdout_behavior": "final_answer_only",
                    "raw_thinking_saved_to_artifact": True,
                    "simulation": False,
                    **tool_result,
                    "elapsed_seconds": round(elapsed, 3),
                },
            )
            artifact_payload = read_artifact(artifact_path)
            return build_text_result(
                "function-calling",
                session.model_id,
                final_answer,
                elapsed,
                execution_status=str(validation["execution_status"]),
                quality_status=str(validation["quality_status"]),
                quality_checks=list(validation["quality_checks"]),
                quality_notes=list(validation["quality_notes"]),
                artifact_path=str(artifact_path),
                artifact_kind=artifact_payload.get("artifact_kind"),
                runtime_backend=(artifact_payload.get("runtime") or {}).get("backend"),
                preprocessing_lineage=((artifact_payload.get("assets") or {}).get("lineage") or []),
            )

        attempt("function-calling", runner)

    return results


def build_multimodal_result(
    capability: str,
    model_used: str,
    asset_used: str,
    output_text: str,
    elapsed_seconds: float,
    *,
    execution_status: str = "ok",
    validation_mode: str = "live",
    claim_scope: str = "live model generation on a local asset",
    pass_definition: str = "Pass means the capability-specific deterministic quality checks passed for the recorded local asset.",
    quality_status: str = "not_validated",
    quality_checks: list[dict[str, Any]] | None = None,
    quality_notes: list[str] | None = None,
    notes: list[str] | None = None,
    artifact_path: str | None = None,
    artifact_kind: str | None = None,
    runtime_backend: str | None = None,
    preprocessing_lineage: list[dict[str, Any]] | None = None,
) -> CapabilityResult:
    status = derive_result_status(execution_status=execution_status, quality_status=quality_status)
    return CapabilityResult(
        capability=capability,
        phase=CAPABILITY_PHASE[capability],
        status=status,
        model_used=model_used,
        asset_used=asset_used,
        validation_command=validation_command(capability),
        artifact_path=artifact_path,
        elapsed_seconds=round(elapsed_seconds, 3),
        result=status,
        blocker=None,
        execution_status=execution_status,
        quality_status=quality_status,
        validation_mode=validation_mode,
        claim_scope=claim_scope,
        pass_definition=pass_definition,
        quality_checks=quality_checks or [],
        quality_notes=quality_notes or [],
        output_preview=normalize_output_preview(output_text),
        notes=notes or [],
        artifact_kind=artifact_kind,
        runtime_backend=runtime_backend,
        preprocessing_lineage=list(preprocessing_lineage or []),
    )


def run_selected_vision_capabilities(
    bundle: RuntimeSession,
    selected: set[str],
    max_pages: int,
) -> list[CapabilityResult]:
    results: list[CapabilityResult] = []

    def attempt(capability: str, runner: Any) -> None:
        try:
            results.append(runner())
        except ExternalBlocker as exc:
            results.append(blocked_result(capability, bundle.model_id, str(exc)))
        except ValidationFailure as exc:
            results.append(
                quality_assessment_result(
                    capability=capability,
                    model_used=bundle.model_id,
                    message=str(exc),
                    execution_status=exc.execution_status,
                    quality_status=exc.quality_status,
                    quality_checks=exc.quality_checks,
                    quality_notes=exc.quality_notes,
                )
            )
        except Exception as exc:
            results.append(failed_result(capability, bundle.model_id, str(exc)))

    if "image-caption" in selected:
        def runner() -> CapabilityResult:
            image_path = ensure_existing_path(asset_path("assets/images/sample.png"))
            image_module = import_image_module()
            records, pdf_inputs = resolve_inputs(
                input_paths=[image_path],
                image_module=image_module,
                max_pages=max_pages,
            )
            validate_mode_inputs("caption", records, pdf_inputs)
            prompt = resolve_vision_prompt("caption", None)
            system_prompt = resolve_vision_system_prompt("caption", None)
            user_prompt = build_vision_user_prompt("caption", prompt, image_count=len(records))
            output_text, elapsed = run_vision_generation_from_records(bundle, "caption", records)
            validation = assess_caption_output(output_text)
            artifact_path, artifact_payload = write_capability_artifact(
                capability="image-caption",
                runtime_backend="gemma-live-vision",
                model_id=bundle.model_id,
                device_info=bundle.device_info,
                elapsed_seconds=elapsed,
                prompts=build_prompt_record(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    resolved_user_prompt=user_prompt,
                ),
                validation=validation,
                asset_lineage=collect_asset_lineage(records),
                extra={
                    "mode": "caption",
                    "model_id": bundle.model_id,
                    "device": normalize_device_info(bundle.device_info)["label"],
                    "dtype": normalize_device_info(bundle.device_info)["dtype"],
                    "generation_settings": dict(VISION_GENERATION_SETTINGS["caption"]),
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "resolved_user_prompt": user_prompt,
                    "inputs": serialize_input_records(records),
                    "output_text": output_text,
                    "elapsed_seconds": round(elapsed, 3),
                },
            )
            return build_multimodal_result(
                "image-caption",
                bundle.model_id,
                str(image_path),
                output_text,
                elapsed,
                validation_mode=validation["validation_mode"],
                claim_scope=validation["claim_scope"],
                pass_definition=validation["pass_definition"],
                quality_status=validation["quality_status"],
                quality_checks=validation["quality_checks"],
                quality_notes=validation["quality_notes"],
                artifact_path=str(artifact_path),
                artifact_kind=artifact_payload.get("artifact_kind"),
                runtime_backend=(artifact_payload.get("runtime") or {}).get("backend"),
                preprocessing_lineage=((artifact_payload.get("assets") or {}).get("lineage") or []),
            )

        attempt("image-caption", runner)

    if "ocr" in selected:
        def runner() -> CapabilityResult:
            image_path = ensure_existing_path(asset_path("assets/images/sample_text.png"))
            image_module = import_image_module()
            records, pdf_inputs = resolve_inputs(
                input_paths=[image_path],
                image_module=image_module,
                max_pages=max_pages,
            )
            validate_mode_inputs("ocr", records, pdf_inputs)
            prompt = resolve_vision_prompt("ocr", None)
            system_prompt = resolve_vision_system_prompt("ocr", None)
            user_prompt = build_vision_user_prompt("ocr", prompt, image_count=len(records))
            output_text, elapsed = run_vision_generation_from_records(bundle, "ocr", records)
            validation = assess_ocr_output(output_text)
            artifact_path, artifact_payload = write_capability_artifact(
                capability="ocr",
                runtime_backend="gemma-live-vision",
                model_id=bundle.model_id,
                device_info=bundle.device_info,
                elapsed_seconds=elapsed,
                prompts=build_prompt_record(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    resolved_user_prompt=user_prompt,
                ),
                validation=validation,
                asset_lineage=collect_asset_lineage(records),
                extra={
                    "mode": "ocr",
                    "model_id": bundle.model_id,
                    "device": normalize_device_info(bundle.device_info)["label"],
                    "dtype": normalize_device_info(bundle.device_info)["dtype"],
                    "generation_settings": dict(VISION_GENERATION_SETTINGS["ocr"]),
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "resolved_user_prompt": user_prompt,
                    "inputs": serialize_input_records(records),
                    "output_text": output_text,
                    "elapsed_seconds": round(elapsed, 3),
                },
            )
            return build_multimodal_result(
                "ocr",
                bundle.model_id,
                str(image_path),
                output_text,
                elapsed,
                validation_mode=validation["validation_mode"],
                claim_scope=validation["claim_scope"],
                pass_definition=validation["pass_definition"],
                quality_status=validation["quality_status"],
                quality_checks=validation["quality_checks"],
                quality_notes=validation["quality_notes"],
                artifact_path=str(artifact_path),
                artifact_kind=artifact_payload.get("artifact_kind"),
                runtime_backend=(artifact_payload.get("runtime") or {}).get("backend"),
                preprocessing_lineage=((artifact_payload.get("assets") or {}).get("lineage") or []),
            )

        attempt("ocr", runner)

    if "image-compare" in selected:
        def runner() -> CapabilityResult:
            left = ensure_existing_path(asset_path("assets/images/sample.png"))
            right = ensure_existing_path(asset_path("assets/images/sample_compare.png"))
            image_module = import_image_module()
            records, pdf_inputs = resolve_inputs(
                input_paths=[left, right],
                image_module=image_module,
                max_pages=max_pages,
            )
            validate_mode_inputs("compare", records, pdf_inputs)
            prompt = resolve_vision_prompt("compare", None)
            system_prompt = resolve_vision_system_prompt("compare", None)
            user_prompt = build_vision_user_prompt("compare", prompt, image_count=len(records))
            output_text, elapsed = run_vision_generation_from_records(bundle, "compare", records)
            validation = assess_compare_output(output_text)
            artifact_path, artifact_payload = write_capability_artifact(
                capability="image-compare",
                runtime_backend="gemma-live-vision",
                model_id=bundle.model_id,
                device_info=bundle.device_info,
                elapsed_seconds=elapsed,
                prompts=build_prompt_record(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    resolved_user_prompt=user_prompt,
                ),
                validation=validation,
                asset_lineage=collect_asset_lineage(records),
                extra={
                    "mode": "compare",
                    "model_id": bundle.model_id,
                    "device": normalize_device_info(bundle.device_info)["label"],
                    "dtype": normalize_device_info(bundle.device_info)["dtype"],
                    "generation_settings": dict(VISION_GENERATION_SETTINGS["compare"]),
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "resolved_user_prompt": user_prompt,
                    "inputs": serialize_input_records(records),
                    "output_text": output_text,
                    "elapsed_seconds": round(elapsed, 3),
                },
            )
            return build_multimodal_result(
                "image-compare",
                bundle.model_id,
                f"{left}, {right}",
                output_text,
                elapsed,
                validation_mode=validation["validation_mode"],
                claim_scope=validation["claim_scope"],
                pass_definition=validation["pass_definition"],
                quality_status=validation["quality_status"],
                quality_checks=validation["quality_checks"],
                quality_notes=validation["quality_notes"],
                artifact_path=str(artifact_path),
                artifact_kind=artifact_payload.get("artifact_kind"),
                runtime_backend=(artifact_payload.get("runtime") or {}).get("backend"),
                preprocessing_lineage=((artifact_payload.get("assets") or {}).get("lineage") or []),
            )

        attempt("image-compare", runner)

    if "pdf-doc-summary" in selected:
        def runner() -> CapabilityResult:
            pdf_path = ensure_existing_path(asset_path("assets/docs/sample.pdf"))
            image_module = import_image_module()
            records, pdf_inputs = resolve_inputs(
                input_paths=[pdf_path],
                image_module=image_module,
                max_pages=max_pages,
            )
            validate_mode_inputs("pdf-summary", records, pdf_inputs)
            prompt = resolve_vision_prompt("pdf-summary", None)
            system_prompt = resolve_vision_system_prompt("pdf-summary", None)
            user_prompt = build_vision_user_prompt("pdf-summary", prompt, image_count=len(records))
            output_text, elapsed = run_vision_generation_from_records(bundle, "pdf-summary", records)
            validation = assess_pdf_summary_output(output_text)
            artifact_path, artifact_payload = write_capability_artifact(
                capability="pdf-doc-summary",
                runtime_backend="gemma-live-vision",
                model_id=bundle.model_id,
                device_info=bundle.device_info,
                elapsed_seconds=elapsed,
                prompts=build_prompt_record(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    resolved_user_prompt=user_prompt,
                ),
                validation=validation,
                asset_lineage=collect_asset_lineage(records),
                extra={
                    "mode": "pdf-summary",
                    "model_id": bundle.model_id,
                    "device": normalize_device_info(bundle.device_info)["label"],
                    "dtype": normalize_device_info(bundle.device_info)["dtype"],
                    "generation_settings": dict(VISION_GENERATION_SETTINGS["pdf-summary"]),
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "resolved_user_prompt": user_prompt,
                    "inputs": serialize_input_records(records),
                    "output_text": output_text,
                    "elapsed_seconds": round(elapsed, 3),
                },
            )
            return build_multimodal_result(
                "pdf-doc-summary",
                bundle.model_id,
                str(pdf_path),
                output_text,
                elapsed,
                validation_mode=validation["validation_mode"],
                claim_scope=validation["claim_scope"],
                pass_definition=validation["pass_definition"],
                quality_status=validation["quality_status"],
                quality_checks=validation["quality_checks"],
                quality_notes=validation["quality_notes"],
                artifact_path=str(artifact_path),
                artifact_kind=artifact_payload.get("artifact_kind"),
                runtime_backend=(artifact_payload.get("runtime") or {}).get("backend"),
                preprocessing_lineage=((artifact_payload.get("assets") or {}).get("lineage") or []),
            )

        attempt("pdf-doc-summary", runner)

    if "video-understanding" in selected:
        def runner() -> CapabilityResult:
            video_path = ensure_existing_path(asset_path("assets/video/sample_video.mp4"))
            image_module = import_image_module()
            frames = sample_video_frames(
                video_path,
                image_module=image_module,
            )
            output_text, elapsed = run_vision_generation_from_records(
                bundle,
                "vqa",
                frames,
                prompt=(
                    "These frames come from the same short video in chronological order. "
                    "Describe the scene and say what changes over time. If no meaningful change is visible, say so plainly."
                ),
            )
            validation = assess_video_proxy_output(frames, output_text)
            prompt = (
                "These frames come from the same short video in chronological order. "
                "Describe the scene and say what changes over time. If no meaningful change is visible, say so plainly."
            )
            system_prompt = resolve_vision_system_prompt("vqa", None)
            user_prompt = build_vision_user_prompt("vqa", prompt, image_count=len(frames))
            artifact_path, artifact_payload = write_capability_artifact(
                capability="video-understanding",
                runtime_backend="gemma-live-vision",
                model_id=bundle.model_id,
                device_info=bundle.device_info,
                elapsed_seconds=elapsed,
                prompts=build_prompt_record(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    resolved_user_prompt=user_prompt,
                ),
                validation=validation,
                asset_lineage=collect_asset_lineage(frames),
                extra={
                    "mode": "video-proxy-vqa",
                    "proxy_mode": "frame-sampled-video",
                    "model_id": bundle.model_id,
                    "device": normalize_device_info(bundle.device_info)["label"],
                    "dtype": normalize_device_info(bundle.device_info)["dtype"],
                    "generation_settings": dict(VISION_GENERATION_SETTINGS["vqa"]),
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "resolved_user_prompt": user_prompt,
                    "inputs": serialize_input_records(frames),
                    "output_text": output_text,
                    "elapsed_seconds": round(elapsed, 3),
                },
            )
            return build_multimodal_result(
                "video-understanding",
                bundle.model_id,
                str(video_path),
                output_text,
                elapsed,
                validation_mode=validation["validation_mode"],
                claim_scope=validation["claim_scope"],
                pass_definition=validation["pass_definition"],
                quality_status=validation["quality_status"],
                quality_checks=validation["quality_checks"],
                quality_notes=validation["quality_notes"],
                notes=["Validated as a frame-sampled proxy via ffmpeg, deterministic frame sampling, and the vision runner."],
                artifact_path=str(artifact_path),
                artifact_kind=artifact_payload.get("artifact_kind"),
                runtime_backend=(artifact_payload.get("runtime") or {}).get("backend"),
                preprocessing_lineage=((artifact_payload.get("assets") or {}).get("lineage") or []),
            )

        attempt("video-understanding", runner)

    return results


def run_selected_audio_capabilities(
    session_manager: SessionManager,
    audio_model_id: str,
    audio_model_source: str,
    selected: set[str],
) -> list[CapabilityResult]:
    results: list[CapabilityResult] = []

    def attempt(capability: str, runner: Any) -> None:
        try:
            results.append(runner())
        except ValidationFailure as exc:
            results.append(
                quality_assessment_result(
                    capability=capability,
                    model_used=audio_model_id,
                    message=str(exc),
                    execution_status=exc.execution_status,
                    quality_status=exc.quality_status,
                    quality_checks=exc.quality_checks,
                    quality_notes=exc.quality_notes,
                    notes=[f"audio model selection source: {audio_model_source}"],
                )
            )
        except UserFacingError as exc:
            message = str(exc)
            if is_external_blocker_message(message):
                results.append(
                    blocked_result(
                        capability,
                        audio_model_id,
                        message,
                        notes=[f"audio model selection source: {audio_model_source}"],
                    )
                )
            else:
                results.append(
                    failed_result(
                        capability,
                        audio_model_id,
                        message,
                        notes=[f"audio model selection source: {audio_model_source}"],
                    )
                )
        except ExternalBlocker as exc:
            results.append(
                blocked_result(
                    capability,
                    audio_model_id,
                    str(exc),
                    notes=[f"audio model selection source: {audio_model_source}"],
                )
            )
        except Exception as exc:
            results.append(
                failed_result(
                    capability,
                    audio_model_id,
                    str(exc),
                    notes=[f"audio model selection source: {audio_model_source}"],
                )
            )

    if "audio-asr" in selected:
        def runner() -> CapabilityResult:
            audio_path = ensure_existing_path(asset_path("assets/audio/sample_audio.wav"))
            mode_result = run_audio_mode(
                mode="transcribe",
                input_path=audio_path,
                session_manager=session_manager,
            )
            validation = mode_result["validation"]
            artifact_path, artifact_payload = write_capability_artifact(
                capability="audio-asr",
                runtime_backend="gemma-live-audio",
                model_id=mode_result["model_id"],
                device_info=mode_result["device_info"],
                elapsed_seconds=mode_result["elapsed_seconds"],
                prompts=build_prompt_record(
                    system_prompt=mode_result["system_prompt"],
                    prompt=mode_result["prompt"],
                    resolved_user_prompt=mode_result["prompt"],
                ),
                validation=validation,
                asset_lineage=collect_asset_lineage(mode_result["record"]),
                extra={
                    "mode": "transcribe",
                    "base_model_id": mode_result["base_model_id"],
                    "model_id": mode_result["model_id"],
                    "model_id_source": mode_result["model_id_source"],
                    "device": normalize_device_info(mode_result["device_info"])["label"],
                    "dtype": normalize_device_info(mode_result["device_info"])["dtype"],
                    "generation_settings": mode_result["generation_settings"],
                    "target_language": mode_result["target_language"],
                    "system_prompt": mode_result["system_prompt"],
                    "prompt": mode_result["prompt"],
                    "resolved_user_prompt": mode_result["prompt"],
                    "input": serialize_audio_record(mode_result["record"]),
                    "elapsed_seconds": round(mode_result["elapsed_seconds"], 3),
                    "output_text": mode_result["output_text"],
                    "pipeline": mode_result["pipeline"],
                },
            )
            return build_multimodal_result(
                "audio-asr",
                mode_result["model_id"],
                str(audio_path),
                mode_result["output_text"],
                mode_result["elapsed_seconds"],
                validation_mode=validation["validation_mode"],
                claim_scope=validation["claim_scope"],
                pass_definition=validation["pass_definition"],
                quality_status=validation["quality_status"],
                quality_checks=validation["quality_checks"],
                quality_notes=validation["quality_notes"],
                artifact_path=str(artifact_path),
                artifact_kind=artifact_payload.get("artifact_kind"),
                runtime_backend=(artifact_payload.get("runtime") or {}).get("backend"),
                preprocessing_lineage=((artifact_payload.get("assets") or {}).get("lineage") or []),
            )

        attempt("audio-asr", runner)

    if "audio-translation" in selected:
        def runner() -> CapabilityResult:
            audio_path = ensure_existing_path(asset_path("assets/audio/sample_audio.wav"))
            mode_result = run_audio_mode(
                mode="translate",
                input_path=audio_path,
                target_language="Japanese",
                session_manager=session_manager,
            )
            validation = mode_result["validation"]
            model_used = mode_result["model_id"]
            if mode_result.get("pipeline") and mode_result["pipeline"].get("translation_model_id"):
                model_used = f"{model_used} + text:{mode_result['pipeline']['translation_model_id']}"
            artifact_path, artifact_payload = write_capability_artifact(
                capability="audio-translation",
                runtime_backend="gemma-live-audio-translation-pipeline",
                model_id=model_used,
                device_info=mode_result["device_info"],
                elapsed_seconds=mode_result["elapsed_seconds"],
                prompts=build_prompt_record(
                    system_prompt=mode_result["system_prompt"],
                    prompt=mode_result["prompt"],
                    resolved_user_prompt=mode_result["prompt"],
                ),
                validation=validation,
                asset_lineage=collect_asset_lineage(mode_result["record"]),
                extra={
                    "mode": "translate",
                    "base_model_id": mode_result["base_model_id"],
                    "model_id": mode_result["model_id"],
                    "model_id_source": mode_result["model_id_source"],
                    "device": normalize_device_info(mode_result["device_info"])["label"],
                    "dtype": normalize_device_info(mode_result["device_info"])["dtype"],
                    "generation_settings": mode_result["generation_settings"],
                    "target_language": mode_result["target_language"],
                    "system_prompt": mode_result["system_prompt"],
                    "prompt": mode_result["prompt"],
                    "resolved_user_prompt": mode_result["prompt"],
                    "input": serialize_audio_record(mode_result["record"]),
                    "elapsed_seconds": round(mode_result["elapsed_seconds"], 3),
                    "output_text": mode_result["output_text"],
                    "pipeline": mode_result["pipeline"],
                },
            )
            return build_multimodal_result(
                "audio-translation",
                model_used,
                str(audio_path),
                mode_result["output_text"],
                mode_result["elapsed_seconds"],
                validation_mode=validation["validation_mode"],
                claim_scope=validation["claim_scope"],
                pass_definition=validation["pass_definition"],
                quality_status=validation["quality_status"],
                quality_checks=validation["quality_checks"],
                quality_notes=validation["quality_notes"],
                notes=["Translation used the conservative transcript-first pipeline."],
                artifact_path=str(artifact_path),
                artifact_kind=artifact_payload.get("artifact_kind"),
                runtime_backend=(artifact_payload.get("runtime") or {}).get("backend"),
                preprocessing_lineage=((artifact_payload.get("assets") or {}).get("lineage") or []),
            )

        attempt("audio-translation", runner)

    return results


def blocked_result(
    capability: str,
    model_used: str | None,
    blocker: str,
    notes: list[str] | None = None,
    artifact_path: str | None = None,
    artifact_kind: str | None = None,
    runtime_backend: str | None = None,
    preprocessing_lineage: list[dict[str, Any]] | None = None,
) -> CapabilityResult:
    return CapabilityResult(
        capability=capability,
        phase=CAPABILITY_PHASE[capability],
        status="blocked",
        model_used=model_used,
        asset_used=None,
        validation_command=validation_command(capability),
        artifact_path=artifact_path,
        elapsed_seconds=None,
        result="blocked",
        blocker=build_blocker_record(blocker),
        execution_status="blocked",
        quality_status="not_run",
        validation_mode=None,
        claim_scope=None,
        pass_definition=None,
        quality_checks=[],
        quality_notes=[],
        output_preview=None,
        notes=notes or [],
        artifact_kind=artifact_kind,
        runtime_backend=runtime_backend,
        preprocessing_lineage=list(preprocessing_lineage or []),
    )


def failed_result(
    capability: str,
    model_used: str | None,
    message: str,
    notes: list[str] | None = None,
    artifact_path: str | None = None,
    artifact_kind: str | None = None,
    runtime_backend: str | None = None,
    preprocessing_lineage: list[dict[str, Any]] | None = None,
) -> CapabilityResult:
    return CapabilityResult(
        capability=capability,
        phase=CAPABILITY_PHASE[capability],
        status="failed",
        model_used=model_used,
        asset_used=None,
        validation_command=validation_command(capability),
        artifact_path=artifact_path,
        elapsed_seconds=None,
        result="failed",
        blocker=build_blocker_record(message, kind=REPO_BUG),
        execution_status="failed",
        quality_status="fail",
        validation_mode=None,
        claim_scope=None,
        pass_definition=None,
        quality_checks=[],
        quality_notes=[],
        output_preview=None,
        notes=notes or [],
        artifact_kind=artifact_kind,
        runtime_backend=runtime_backend,
        preprocessing_lineage=list(preprocessing_lineage or []),
    )


def selected_capabilities(args: argparse.Namespace) -> list[str]:
    if args.only:
        requested = [item.strip() for item in args.only.split(",") if item.strip()]
    elif args.phase:
        requested = PHASE_CAPABILITIES[args.phase]
    elif args.smoke:
        requested = SMOKE_CAPABILITIES
    else:
        requested = ALL_CAPABILITIES

    invalid = [item for item in requested if item not in ALL_CAPABILITIES]
    if invalid:
        raise SystemExit(f"Unknown capability ids: {', '.join(invalid)}")
    return requested


def summarize_results(results: list[CapabilityResult]) -> dict[str, int]:
    counts = {
        "ok": 0,
        "blocked": 0,
        "quality_fail": 0,
        "not_validated": 0,
        "failed": 0,
    }
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    selected = selected_capabilities(args)
    selected_set = set(selected)
    output_path = args.out or default_output_path()
    started_at = time.perf_counter()

    manifest = None
    if not args.skip_prepare_assets:
        manifest = prepare_assets(timeout=args.asset_timeout)

    results: list[CapabilityResult] = []
    notes: list[str] = []
    session_manager = SessionManager()

    try:
        text_capabilities = {
            "text-chat",
            "summarization",
            "multilingual-translate",
            "code-generation",
            "structured-json",
            "thinking",
            "function-calling",
        } & selected_set

        text_model_id = resolve_model_id()
        if text_capabilities:
            try:
                text_session = session_manager.get_session("text", text_model_id)
                results.extend(run_selected_text_capabilities(text_session, text_capabilities))
            except ValidationFailure as exc:
                notes.append(f"text validation did not close cleanly: {exc}")
                for capability in selected:
                    if capability in text_capabilities:
                        results.append(
                            quality_assessment_result(
                                capability=capability,
                                model_used=text_model_id,
                                message=str(exc),
                                execution_status=exc.execution_status,
                                quality_status=exc.quality_status,
                                quality_checks=exc.quality_checks,
                                quality_notes=exc.quality_notes,
                            )
                        )
            except UserFacingError as exc:
                blocker = str(exc)
                notes.append(f"text runtime blocked: {blocker}")
                for capability in selected:
                    if capability in text_capabilities:
                        results.append(blocked_result(capability, text_model_id, blocker))
            except Exception as exc:
                message = str(exc)
                notes.append(f"text runtime failed: {message}")
                for capability in selected:
                    if capability in text_capabilities:
                        results.append(failed_result(capability, text_model_id, message))

        vision_like = {"image-caption", "ocr", "image-compare", "pdf-doc-summary", "video-understanding"} & selected_set
        audio_capabilities = {"audio-asr", "audio-translation"} & selected_set
        vision_model_id = resolve_model_id()
        audio_model_id, audio_model_source = resolve_audio_model_selection()

        if vision_like:
            try:
                bundle = session_manager.get_session("vision", vision_model_id)
                results.extend(
                    run_selected_vision_capabilities(
                        bundle=bundle,
                        selected=vision_like,
                        max_pages=args.max_pages,
                    )
                )
            except ValidationFailure as exc:
                notes.append(f"vision validation did not close cleanly: {exc}")
                for capability in selected:
                    if capability in vision_like:
                        results.append(
                            quality_assessment_result(
                                capability=capability,
                                model_used=vision_model_id,
                                message=str(exc),
                                execution_status=exc.execution_status,
                                quality_status=exc.quality_status,
                                quality_checks=exc.quality_checks,
                                quality_notes=exc.quality_notes,
                            )
                        )
            except UserFacingError as exc:
                blocker = str(exc)
                notes.append(f"vision runtime blocked: {blocker}")
                for capability in selected:
                    if capability in vision_like:
                        results.append(blocked_result(capability, vision_model_id, blocker))
            except Exception as exc:
                message = str(exc)
                notes.append(f"vision runtime failed: {message}")
                for capability in selected:
                    if capability in vision_like:
                        results.append(failed_result(capability, vision_model_id, message))

        if audio_capabilities:
            try:
                results.extend(
                    run_selected_audio_capabilities(
                        session_manager=session_manager,
                        audio_model_id=audio_model_id,
                        audio_model_source=audio_model_source,
                        selected=audio_capabilities,
                    )
                )
            except ValidationFailure as exc:
                notes.append(f"audio validation did not close cleanly: {exc}")
                for capability in selected:
                    if capability in audio_capabilities:
                        results.append(
                            quality_assessment_result(
                                capability=capability,
                                model_used=audio_model_id,
                                message=str(exc),
                                execution_status=exc.execution_status,
                                quality_status=exc.quality_status,
                                quality_checks=exc.quality_checks,
                                quality_notes=exc.quality_notes,
                                notes=[f"audio model selection source: {audio_model_source}"],
                            )
                        )
            except UserFacingError as exc:
                blocker = str(exc)
                notes.append(f"audio runtime blocked: {blocker}")
                for capability in selected:
                    if capability in audio_capabilities:
                        results.append(
                            blocked_result(
                                capability,
                                audio_model_id,
                                blocker,
                                notes=[f"audio model selection source: {audio_model_source}"],
                            )
                        )
            except Exception as exc:
                message = str(exc)
                notes.append(f"audio runtime failed: {message}")
                for capability in selected:
                    if capability in audio_capabilities:
                        results.append(
                            failed_result(
                                capability,
                                audio_model_id,
                                message,
                                notes=[f"audio model selection source: {audio_model_source}"],
                            )
                        )

        if "long-context" in selected_set:
            long_context_output = artifacts_root() / f"{timestamp_slug()}-long-context.json"
            try:
                payload, elapsed = run_long_context(long_context_output, session_manager)
                profile = detect_context_profile(payload["model_id"])
                validation = payload.get("validation") or {}
                results.append(
                    CapabilityResult(
                        capability="long-context",
                        phase=CAPABILITY_PHASE["long-context"],
                        status="ok",
                        model_used=payload["model_id"],
                        asset_used="synthetic corpus + repository snippets",
                        validation_command=validation_command("long-context"),
                        artifact_path=str(long_context_output),
                        elapsed_seconds=round(elapsed, 3),
                        result=f"ok ({payload['resolved_backend']}, {profile.label})",
                        blocker=None,
                        execution_status=str(validation.get("execution_status") or "ok"),
                        quality_status=str(validation.get("quality_status") or "pass"),
                        validation_mode=validation.get("validation_mode"),
                        claim_scope=validation.get("claim_scope"),
                        pass_definition=validation.get("pass_definition"),
                        quality_checks=list(validation.get("quality_checks") or []),
                        quality_notes=list(validation.get("quality_notes") or []),
                        output_preview=normalize_output_preview(
                            "; ".join(
                                f"{case['case_label']}={case['pass']} fit={case['prompt_fit_ratio']}"
                                for case in payload.get("cases", [])
                            )
                        ),
                        notes=[f"context profile: {profile.label}"],
                        artifact_kind=payload.get("artifact_kind"),
                        runtime_backend=(payload.get("runtime") or {}).get("backend"),
                        preprocessing_lineage=(((payload.get("assets") or {}).get("lineage")) or []),
                    )
                )
            except ExternalBlocker as exc:
                results.append(blocked_result("long-context", resolve_model_id(), str(exc)))
            except ValidationFailure as exc:
                quality_notes = list(exc.quality_notes)
                if str(exc) not in quality_notes:
                    quality_notes.append(str(exc))
                stored_long_context = read_artifact(long_context_output) if long_context_output.exists() else None
                results.append(
                    CapabilityResult(
                        capability="long-context",
                        phase=CAPABILITY_PHASE["long-context"],
                        status=derive_result_status(
                            execution_status=exc.execution_status,
                            quality_status=exc.quality_status,
                        ),
                        model_used=resolve_model_id(),
                        asset_used="synthetic corpus + repository snippets",
                        validation_command=validation_command("long-context"),
                        artifact_path=str(long_context_output) if long_context_output.exists() else None,
                        elapsed_seconds=None,
                        result=derive_result_status(
                            execution_status=exc.execution_status,
                            quality_status=exc.quality_status,
                        ),
                        blocker=None,
                        execution_status=exc.execution_status,
                        quality_status=exc.quality_status,
                        validation_mode="live",
                        claim_scope="synthetic long-context live validation",
                        pass_definition=(
                            "Pass means the long-context live validation reached the requested live claim and preserved exact marker retrieval."
                        ),
                        quality_checks=list(exc.quality_checks),
                        quality_notes=quality_notes,
                        output_preview=None,
                        notes=[],
                        artifact_kind=stored_long_context.get("artifact_kind") if stored_long_context else None,
                        runtime_backend=((stored_long_context.get("runtime") or {}).get("backend") if stored_long_context else None),
                        preprocessing_lineage=((((stored_long_context.get("assets") or {}).get("lineage")) or []) if stored_long_context else []),
                    )
                )
            except Exception as exc:
                results.append(failed_result("long-context", resolve_model_id(), str(exc)))

        result_by_capability = {result.capability: result for result in results}
        ordered_results = [result_by_capability[name] for name in selected if name in result_by_capability]
        counts = summarize_results(ordered_results)
        overall_status = (
            "failed"
            if counts["failed"] or counts["quality_fail"] or counts["not_validated"]
            else ("blocked" if counts["blocked"] else "ok")
        )
        artifact_index_path = default_artifact_index_path(output_path)
        matrix_blocker_message = None
        if counts["failed"] or (overall_status == "blocked" and counts["blocked"]):
            if notes:
                matrix_blocker_message = notes[0]
            else:
                matrix_blocker_message = next(
                    (
                        result.blocker["message"]
                        for result in ordered_results
                        if result.blocker is not None
                    ),
                    None,
                )

        payload = build_artifact_payload(
            artifact_kind="capability_matrix",
            status=overall_status,
            runtime=build_runtime_record(
                backend="capability-matrix",
                model_id=resolve_model_id(),
                device_info="matrix-multi-run",
                elapsed_seconds=time.perf_counter() - started_at,
                extra={
                    "audio_model_id": audio_model_id,
                    "audio_model_source": audio_model_source,
                    "python": str(repo_python_path()),
                    "workspace": str(repo_root()),
                },
            ),
            blocker_message=matrix_blocker_message,
            extra={
                "workspace": str(repo_root()),
                "python": str(repo_python_path()),
                "selection": selected,
                "smoke": args.smoke,
                "prepared_assets": not args.skip_prepare_assets,
                "asset_manifest": manifest,
                "summary": counts,
                "overall_status": overall_status,
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                "artifact_index_path": str(artifact_index_path.resolve()),
                "results": [asdict(result) for result in ordered_results],
                "notes": notes,
            },
        )
        write_artifact(output_path, payload)
        index_output_path = write_artifact_index(
            matrix_output_path=output_path,
            overall_status=overall_status,
            selected=selected,
            results=ordered_results,
            model_id=resolve_model_id(),
            elapsed_seconds=time.perf_counter() - started_at,
        )
        payload["artifact_path"] = str(output_path)
        payload["artifact_index_path"] = str(index_output_path)
        return payload
    finally:
        session_manager.close_all()


def print_payload(payload: dict[str, Any]) -> None:
    print(f"overall_status: {payload['overall_status']}")
    summary = payload["summary"]
    print(
        "summary: "
        f"ok={summary['ok']} "
        f"blocked={summary['blocked']} "
        f"quality_fail={summary['quality_fail']} "
        f"not_validated={summary['not_validated']} "
        f"failed={summary['failed']}"
    )
    for result in payload["results"]:
        line = [
            f"capability={result['capability']}",
            f"status={result['status']}",
        ]
        if result.get("validation_mode"):
            line.append(f"mode={result['validation_mode']}")
        if result.get("quality_status"):
            line.append(f"quality={result['quality_status']}")
        if result.get("model_used"):
            line.append(f"model={result['model_used']}")
        if result.get("elapsed_seconds") is not None:
            line.append(f"elapsed={result['elapsed_seconds']:.3f}s")
        if result.get("blocker"):
            blocker = result["blocker"]
            line.append(f"blocker={blocker['kind']}: {blocker['message']}")
        print(" ".join(line))
    print(f"artifact_path: {payload['artifact_path']}")
    if payload.get("artifact_index_path"):
        print(f"artifact_index_path: {payload['artifact_index_path']}")


def main() -> int:
    args = parse_args()
    payload = run_matrix(args)
    print_payload(payload)
    if payload["overall_status"] == "failed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
