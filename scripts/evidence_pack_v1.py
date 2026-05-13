#!/usr/bin/env python3
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from artifact_vault import capture_artifact
from evidence_support import build_evidence_support_result
from gemma_runtime import repo_root, timestamp_slug, timestamp_utc, write_json
from satellite_pack import PackManifestError, load_pack_manifest, resolve_pack_manifest_path
from workspace_state import DEFAULT_WORKSPACE_ID


PACK_V1_SCHEMA_NAME = "software-satellite-evidence-pack-v1"
PACK_V1_SCHEMA_VERSION = 1
PACK_V1_AUDIT_SCHEMA_NAME = "software-satellite-evidence-pack-v1-audit"
PACK_V1_TEST_SCHEMA_NAME = "software-satellite-evidence-pack-v1-test-result"
PACK_V1_LOCK_SCHEMA_NAME = "software-satellite-evidence-pack-v1-lock"
PACK_V1_LOCK_SCHEMA_VERSION = 1

ALLOWED_CORE_TRANSFORMS = {
    "agent_session_intake",
    "artifact_capture",
    "evidence_lint",
    "markdown_report",
    "recall_failure",
    "redaction_report",
    "support_gate",
}
ALLOWED_INPUT_KINDS = {
    "agent_session_bundle",
    "ci_log",
    "diff",
    "failure",
    "patch",
    "review_note",
    "software_work_event",
    "test_log",
    "transcript",
}
ALLOWED_OUTPUT_SCHEMA_REFS = {
    "agent_session_bundle.schema.json",
    "evidence_graph.schema.json",
    "evidence_support.schema.json",
    "review_memory_fixture.schema.json",
    "satellite_evidence_pack_v1.schema.json",
}
ALLOWED_TOP_LEVEL_KEYS = {
    "schema_name",
    "schema_version",
    "metadata",
    "input_kinds",
    "artifact_policy",
    "recall_policy",
    "support_policy",
    "report_sections",
    "benchmark_fixtures",
    "redaction_policy",
    "output_schema_refs",
    "core_transform_refs",
}
FIXTURE_ARTIFACT_KINDS = {
    "agent_session_bundle": "transcript",
    "ci_log": "ci_log",
    "diff": "patch",
    "failure": "test_log",
    "patch": "patch",
    "review_note": "review_note",
    "software_work_event": "review_note",
    "test_log": "test_log",
    "transcript": "transcript",
}
REQUIRED_TOP_LEVEL_KEYS = tuple(sorted(ALLOWED_TOP_LEVEL_KEYS))
METADATA_KEYS = {"pack_id", "version", "title", "summary", "license"}
ARTIFACT_POLICY_KEYS = {"selected_roots", "path_mode", "link_policy", "missing_source_policy"}
RECALL_POLICY_KEYS = {"mode", "max_items", "require_source_refs"}
SUPPORT_POLICY_KEYS = {
    "required_kernel",
    "requested_polarity",
    "require_source_linked_evidence",
    "require_support_kernel",
}
REPORT_SECTION_KEYS = {"id", "title", "source"}
BENCHMARK_FIXTURE_KEYS = {
    "id",
    "input_kind",
    "artifact_name",
    "fixture_text",
    "requested_polarity",
    "expected_support_class",
    "expected_can_support_decision",
}
REDACTION_POLICY_KEYS = {"mode", "token_handling"}

PACK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
FIELD_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
REMOTE_URL_RE = re.compile(r"\b(?:https?|ftp)://", re.IGNORECASE)
PATH_TRAVERSAL_RE = re.compile(r"(^|[\\/])\.\.([\\/]|$)")
GLOB_TOKEN_RE = re.compile(r"[*?\[]")

DENIED_KEY_PATTERNS: dict[str, re.Pattern[str]] = {
    "python_field": re.compile(r"(^|[_-])python([_-]|$)|\.py$", re.IGNORECASE),
    "javascript_field": re.compile(r"(^|[_-])(?:javascript|js|node|npm|npx)([_-]|$)|\.js$", re.IGNORECASE),
    "shell_field": re.compile(
        r"(^|[_-])(?:shell|sh|bash|zsh|command|commands|script|run_command|install_script)([_-]|$)|"
        r"\.(?:sh|bash|zsh)$",
        re.IGNORECASE,
    ),
    "network_field": re.compile(r"(^|[_-])(?:network|http|https|url|webhook|remote_url)([_-]|$)", re.IGNORECASE),
    "api_field": re.compile(r"(^|[_-])(?:api|api_call|api_endpoint)([_-]|$)", re.IGNORECASE),
    "secret_access_field": re.compile(
        r"(^|[_-])(?:secret|secrets|api_key|apikey|credential|credentials|password)([_-]|$)",
        re.IGNORECASE,
    ),
    "environment_field": re.compile(r"(^|[_-])(?:env|environment|environment_variables)([_-]|$)", re.IGNORECASE),
    "repo_write_field": re.compile(
        r"(^|[_-])(?:repo_write|write_repo|repo_mutation|write_file|write_files)([_-]|$)",
        re.IGNORECASE,
    ),
    "file_glob_field": re.compile(r"(^|[_-])(?:glob|globs|file_glob|file_globs)([_-]|$)", re.IGNORECASE),
    "symlink_traversal_field": re.compile(
        r"(^|[_-])(?:symlink|symlinks|follow_links|follow_symlinks)([_-]|$)",
        re.IGNORECASE,
    ),
    "install_or_update_field": re.compile(
        r"(^|[_-])(?:install|install_script|postinstall|auto_update|autoupdate)([_-]|$)",
        re.IGNORECASE,
    ),
    "model_call_field": re.compile(
        r"(^|[_-])(?:model_call|call_model|llm_call|use_backend|backend_call)([_-]|$)",
        re.IGNORECASE,
    ),
    "training_export_field": re.compile(
        r"(^|[_-])(?:training_export|trainable|jsonl_export|fine_tune|finetune)([_-]|$)",
        re.IGNORECASE,
    ),
}

DENIED_VALUE_PATTERNS: dict[str, re.Pattern[str]] = {
    "python_content": re.compile(r"(^#!.*python|\bpython\b|```python|\.py\b)", re.IGNORECASE),
    "javascript_content": re.compile(r"(\bjavascript\b|\bnode\b|\bnpm\b|\bnpx\b|<script\b|\.js\b)", re.IGNORECASE),
    "shell_content": re.compile(
        r"(^#!|\b(?:bash|zsh|shell|chmod|curl|wget)\b|&&|\|\||\$\(|`[^`]+`)",
        re.IGNORECASE,
    ),
    "network_content": re.compile(r"(\bnetwork\b|\bwebhook\b|\bhttp\b|https?://|ftp://)", re.IGNORECASE),
    "api_content": re.compile(r"\b(?:api call|api_call|api endpoint|api_endpoint)\b", re.IGNORECASE),
    "secret_access_content": re.compile(
        r"\b(?:secret|secrets|api[_-]?key|credential|credentials|password)\b",
        re.IGNORECASE,
    ),
    "environment_content": re.compile(r"\b(?:environment variable|environment variables|os\.environ|process\.env)\b", re.IGNORECASE),
    "repo_write_content": re.compile(r"\b(?:repo write|repo_write|write_repo|repo mutation|write file|write files)\b", re.IGNORECASE),
    "path_traversal_content": PATH_TRAVERSAL_RE,
    "symlink_traversal_content": re.compile(r"\b(?:symlink|follow links|follow_links|allow_links)\b", re.IGNORECASE),
    "remote_url_content": REMOTE_URL_RE,
    "install_or_update_content": re.compile(r"\b(?:install script|postinstall|auto-update|auto_update)\b", re.IGNORECASE),
    "model_call_content": re.compile(r"\b(?:model call|call_model|llm call|use_backend|backend invocation)\b", re.IGNORECASE),
    "training_export_content": re.compile(
        r"\b(?:training export|training_export|trainable|fine[-_ ]?tune|finetune|jsonl export|jsonl_export)\b",
        re.IGNORECASE,
    ),
}


class EvidencePackV1Error(ValueError):
    pass


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def _timestamp_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _issue(path: str, message: str, *, check_id: str = "schema", actual: Any = None) -> dict[str, Any]:
    payload = {
        "path": path,
        "message": message,
        "severity": "block",
        "check_id": check_id,
    }
    if actual is not None:
        payload["actual"] = actual if isinstance(actual, str) else type(actual).__name__
    return payload


def _iter_manifest_keys(value: Any, *, path: str = "$") -> list[tuple[str, str]]:
    if isinstance(value, Mapping):
        pairs: list[tuple[str, str]] = []
        for key, item in value.items():
            key_text = str(key)
            item_path = f"{path}.{key_text}"
            pairs.append((item_path, key_text))
            pairs.extend(_iter_manifest_keys(item, path=item_path))
        return pairs
    if isinstance(value, list):
        pairs = []
        for index, item in enumerate(value):
            pairs.extend(_iter_manifest_keys(item, path=f"{path}[{index}]"))
        return pairs
    return []


def _iter_manifest_strings(value: Any, *, path: str = "$") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, Mapping):
        strings: list[tuple[str, str]] = []
        for key, item in value.items():
            strings.extend(_iter_manifest_strings(item, path=f"{path}.{key}"))
        return strings
    if isinstance(value, list):
        strings = []
        for index, item in enumerate(value):
            strings.extend(_iter_manifest_strings(item, path=f"{path}[{index}]"))
        return strings
    return []


def _append_unknown_key_issues(
    issues: list[dict[str, Any]],
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    path: str,
) -> None:
    for key in sorted(str(key) for key in value.keys() if str(key) not in allowed):
        issues.append(_issue(f"{path}.{key}", "Unknown fields are not allowed in Evidence Pack v1.", check_id="unknown_field"))


def _validate_string(
    issues: list[dict[str, Any]],
    value: Any,
    path: str,
    *,
    pattern: re.Pattern[str] | None = None,
    allowed: set[str] | None = None,
) -> str | None:
    if not isinstance(value, str):
        issues.append(_issue(path, "Expected a string.", actual=value))
        return None
    if pattern is not None and not pattern.match(value):
        issues.append(_issue(path, "String does not match the required pattern.", actual=value))
    if allowed is not None and value not in allowed:
        issues.append(_issue(path, "String is outside the allowlist.", check_id="allowlist", actual=value))
    return value


def _validate_bool(issues: list[dict[str, Any]], value: Any, path: str) -> bool | None:
    if not isinstance(value, bool):
        issues.append(_issue(path, "Expected a boolean.", actual=value))
        return None
    return value


def _validate_int(issues: list[dict[str, Any]], value: Any, path: str, *, minimum: int = 0) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        issues.append(_issue(path, "Expected an integer.", actual=value))
        return None
    if value < minimum:
        issues.append(_issue(path, f"Expected an integer >= {minimum}.", actual=value))
    return value


def _validate_string_list(
    issues: list[dict[str, Any]],
    value: Any,
    path: str,
    *,
    min_items: int = 0,
    allowed: set[str] | None = None,
) -> list[str]:
    if not isinstance(value, list):
        issues.append(_issue(path, "Expected an array.", actual=value))
        return []
    if len(value) < min_items:
        issues.append(_issue(path, f"Expected at least {min_items} item(s).", actual=len(value)))
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        text = _validate_string(issues, item, item_path, allowed=allowed)
        if text is None:
            continue
        if text in seen:
            issues.append(_issue(item_path, "Duplicate values are not allowed.", actual=text))
            continue
        seen.add(text)
        result.append(text)
    return result


def _validate_metadata(value: Any, issues: list[dict[str, Any]]) -> None:
    if not isinstance(value, Mapping):
        issues.append(_issue("$.metadata", "Expected an object.", actual=value))
        return
    _append_unknown_key_issues(issues, value, allowed=METADATA_KEYS, path="$.metadata")
    for key in ("pack_id", "version", "title", "summary"):
        if key not in value:
            issues.append(_issue(f"$.metadata.{key}", "Missing required metadata field."))
    if "pack_id" in value:
        _validate_string(issues, value.get("pack_id"), "$.metadata.pack_id", pattern=PACK_ID_RE)
    for key in ("version", "title", "summary", "license"):
        if key in value:
            _validate_string(issues, value.get(key), f"$.metadata.{key}")


def _validate_artifact_policy(value: Any, issues: list[dict[str, Any]], *, root: Path) -> None:
    if not isinstance(value, Mapping):
        issues.append(_issue("$.artifact_policy", "Expected an object.", actual=value))
        return
    _append_unknown_key_issues(issues, value, allowed=ARTIFACT_POLICY_KEYS, path="$.artifact_policy")
    for key in ARTIFACT_POLICY_KEYS:
        if key not in value:
            issues.append(_issue(f"$.artifact_policy.{key}", "Missing required artifact policy field."))
    roots = _validate_string_list(issues, value.get("selected_roots"), "$.artifact_policy.selected_roots", min_items=1)
    if "path_mode" in value:
        _validate_string(issues, value.get("path_mode"), "$.artifact_policy.path_mode", allowed={"selected_roots_only"})
    if "link_policy" in value:
        _validate_string(issues, value.get("link_policy"), "$.artifact_policy.link_policy", allowed={"refuse_links"})
    if "missing_source_policy" in value:
        _validate_string(issues, value.get("missing_source_policy"), "$.artifact_policy.missing_source_policy", allowed={"block"})
    for index, root_text in enumerate(roots):
        _validate_local_path_ref(
            issues,
            root_text,
            f"$.artifact_policy.selected_roots[{index}]",
            root=root,
            allow_missing=True,
            allow_schema_ref=False,
        )


def _validate_recall_policy(value: Any, issues: list[dict[str, Any]]) -> None:
    if not isinstance(value, Mapping):
        issues.append(_issue("$.recall_policy", "Expected an object.", actual=value))
        return
    _append_unknown_key_issues(issues, value, allowed=RECALL_POLICY_KEYS, path="$.recall_policy")
    for key in RECALL_POLICY_KEYS:
        if key not in value:
            issues.append(_issue(f"$.recall_policy.{key}", "Missing required recall policy field."))
    if "mode" in value:
        _validate_string(issues, value.get("mode"), "$.recall_policy.mode", allowed={"failure_memory", "agent_session", "none"})
    if "max_items" in value:
        _validate_int(issues, value.get("max_items"), "$.recall_policy.max_items", minimum=0)
    if "require_source_refs" in value:
        require_source_refs = _validate_bool(issues, value.get("require_source_refs"), "$.recall_policy.require_source_refs")
        if require_source_refs is False:
            issues.append(_issue("$.recall_policy.require_source_refs", "Recall policy must preserve source refs."))


def _validate_support_policy(value: Any, issues: list[dict[str, Any]]) -> None:
    if not isinstance(value, Mapping):
        issues.append(_issue("$.support_policy", "Expected an object.", actual=value))
        return
    _append_unknown_key_issues(issues, value, allowed=SUPPORT_POLICY_KEYS, path="$.support_policy")
    for key in SUPPORT_POLICY_KEYS:
        if key not in value:
            issues.append(_issue(f"$.support_policy.{key}", "Missing required support policy field."))
    if "required_kernel" in value:
        _validate_string(issues, value.get("required_kernel"), "$.support_policy.required_kernel", allowed={"evidence_support_v1"})
    if "requested_polarity" in value:
        _validate_string(
            issues,
            value.get("requested_polarity"),
            "$.support_policy.requested_polarity",
            allowed={"positive", "negative", "risk", "diagnostic", "none"},
        )
    if "require_source_linked_evidence" in value:
        require_source_linked = _validate_bool(
            issues,
            value.get("require_source_linked_evidence"),
            "$.support_policy.require_source_linked_evidence",
        )
        if require_source_linked is False:
            issues.append(
                _issue(
                    "$.support_policy.require_source_linked_evidence",
                    "Support policy must require source-linked evidence.",
                )
            )
    if "require_support_kernel" in value:
        require_support = _validate_bool(issues, value.get("require_support_kernel"), "$.support_policy.require_support_kernel")
        if require_support is False:
            issues.append(_issue("$.support_policy.require_support_kernel", "Pack output must pass through the support kernel."))


def _validate_report_sections(value: Any, issues: list[dict[str, Any]]) -> None:
    if not isinstance(value, list):
        issues.append(_issue("$.report_sections", "Expected an array.", actual=value))
        return
    if not value:
        issues.append(_issue("$.report_sections", "Expected at least one report section.", actual=0))
    for index, item in enumerate(value):
        path = f"$.report_sections[{index}]"
        if not isinstance(item, Mapping):
            issues.append(_issue(path, "Expected an object.", actual=item))
            continue
        _append_unknown_key_issues(issues, item, allowed=REPORT_SECTION_KEYS, path=path)
        for key in ("id", "title", "source"):
            if key not in item:
                issues.append(_issue(f"{path}.{key}", "Missing required report section field."))
        if "id" in item:
            _validate_string(issues, item.get("id"), f"{path}.id", pattern=FIELD_ID_RE)
        if "title" in item:
            _validate_string(issues, item.get("title"), f"{path}.title")
        if "source" in item:
            _validate_string(
                issues,
                item.get("source"),
                f"{path}.source",
                allowed={"benchmark_summary", "input_summary", "recall_summary", "support_kernel"},
            )


def _validate_benchmark_fixtures(value: Any, issues: list[dict[str, Any]]) -> None:
    if not isinstance(value, list):
        issues.append(_issue("$.benchmark_fixtures", "Expected an array.", actual=value))
        return
    if not value:
        issues.append(_issue("$.benchmark_fixtures", "Expected at least one benchmark fixture.", actual=0))
    for index, item in enumerate(value):
        path = f"$.benchmark_fixtures[{index}]"
        if not isinstance(item, Mapping):
            issues.append(_issue(path, "Expected an object.", actual=item))
            continue
        _append_unknown_key_issues(issues, item, allowed=BENCHMARK_FIXTURE_KEYS, path=path)
        for key in BENCHMARK_FIXTURE_KEYS:
            if key not in item:
                issues.append(_issue(f"{path}.{key}", "Missing required benchmark fixture field."))
        if "id" in item:
            _validate_string(issues, item.get("id"), f"{path}.id", pattern=FIELD_ID_RE)
        if "input_kind" in item:
            _validate_string(issues, item.get("input_kind"), f"{path}.input_kind", allowed=ALLOWED_INPUT_KINDS)
        if "artifact_name" in item:
            artifact_name = _validate_string(issues, item.get("artifact_name"), f"{path}.artifact_name", pattern=FIELD_ID_RE)
            if artifact_name and any(separator in artifact_name for separator in ("/", "\\")):
                issues.append(_issue(f"{path}.artifact_name", "Fixture artifact names must be a local filename, not a path."))
        if "fixture_text" in item:
            _validate_string(issues, item.get("fixture_text"), f"{path}.fixture_text")
        if "requested_polarity" in item:
            _validate_string(
                issues,
                item.get("requested_polarity"),
                f"{path}.requested_polarity",
                allowed={"positive", "negative", "risk", "diagnostic", "none"},
            )
        if "expected_support_class" in item:
            _validate_string(
                issues,
                item.get("expected_support_class"),
                f"{path}.expected_support_class",
                allowed={"source_linked_prior", "negative_prior", "manual_pin_diagnostic", "unknown"},
            )
        if "expected_can_support_decision" in item:
            _validate_bool(issues, item.get("expected_can_support_decision"), f"{path}.expected_can_support_decision")


def _validate_redaction_policy(value: Any, issues: list[dict[str, Any]]) -> None:
    if not isinstance(value, Mapping):
        issues.append(_issue("$.redaction_policy", "Expected an object.", actual=value))
        return
    _append_unknown_key_issues(issues, value, allowed=REDACTION_POLICY_KEYS, path="$.redaction_policy")
    for key in REDACTION_POLICY_KEYS:
        if key not in value:
            issues.append(_issue(f"$.redaction_policy.{key}", "Missing required redaction policy field."))
    if "mode" in value:
        _validate_string(issues, value.get("mode"), "$.redaction_policy.mode", allowed={"best_effort_local"})
    if "token_handling" in value:
        _validate_string(issues, value.get("token_handling"), "$.redaction_policy.token_handling", allowed={"redact_known_tokens"})


def _validate_local_path_ref(
    issues: list[dict[str, Any]],
    value: str,
    path: str,
    *,
    root: Path,
    allow_missing: bool,
    allow_schema_ref: bool,
) -> None:
    if REMOTE_URL_RE.search(value):
        issues.append(_issue(path, "Remote URLs are not allowed.", check_id="remote_url", actual=value))
    if PATH_TRAVERSAL_RE.search(value):
        issues.append(_issue(path, "Path traversal is not allowed.", check_id="path_traversal", actual=value))
    if GLOB_TOKEN_RE.search(value):
        issues.append(_issue(path, "File globs are not allowed in Evidence Pack v1.", check_id="file_glob", actual=value))
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        issues.append(_issue(path, "Absolute paths are not allowed.", check_id="path_boundary", actual=value))
        return
    if str(candidate).startswith("~"):
        issues.append(_issue(path, "Home-directory shortcuts are not allowed.", check_id="path_boundary", actual=value))
        return
    if allow_schema_ref:
        candidate = Path("schemas") / candidate
    candidate_path = root / candidate
    try:
        candidate_is_symlink = candidate_path.is_symlink() or any(
            parent.is_symlink()
            for parent in candidate_path.parents
            if str(parent).startswith(str(root))
        )
    except OSError:
        candidate_is_symlink = False
    resolved = candidate_path.resolve()
    if not str(resolved).startswith(str(root) + "/") and resolved != root:
        issues.append(_issue(path, "Path must remain inside the repository root.", check_id="path_boundary", actual=value))
        return
    try:
        exists = resolved.exists()
        resolved_is_symlink = resolved.is_symlink() or any(
            parent.is_symlink()
            for parent in resolved.parents
            if str(parent).startswith(str(root))
        )
    except OSError:
        exists = False
        resolved_is_symlink = False
    if candidate_is_symlink or resolved_is_symlink:
        issues.append(_issue(path, "Symlink traversal is not allowed.", check_id="symlink_traversal", actual=value))
    if not allow_missing and not exists:
        issues.append(_issue(path, "Referenced local file does not exist.", check_id="path_boundary", actual=value))


def _validate_output_schema_refs(manifest: Mapping[str, Any], issues: list[dict[str, Any]], *, root: Path) -> None:
    refs = _validate_string_list(
        issues,
        manifest.get("output_schema_refs"),
        "$.output_schema_refs",
        min_items=1,
        allowed=ALLOWED_OUTPUT_SCHEMA_REFS,
    )
    for index, ref in enumerate(refs):
        _validate_local_path_ref(
            issues,
            ref,
            f"$.output_schema_refs[{index}]",
            root=root,
            allow_missing=False,
            allow_schema_ref=True,
        )


def _validate_core_transform_refs(manifest: Mapping[str, Any], issues: list[dict[str, Any]]) -> None:
    _validate_string_list(
        issues,
        manifest.get("core_transform_refs"),
        "$.core_transform_refs",
        min_items=1,
        allowed=ALLOWED_CORE_TRANSFORMS,
    )


def validate_evidence_pack_v1_manifest(manifest: Mapping[str, Any] | Any, *, root: Path | None = None) -> list[dict[str, Any]]:
    resolved_root = _resolve_root(root)
    issues: list[dict[str, Any]] = []
    if not isinstance(manifest, Mapping):
        return [_issue("$", "Manifest root must be an object.", actual=manifest)]

    _append_unknown_key_issues(issues, manifest, allowed=ALLOWED_TOP_LEVEL_KEYS, path="$")
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in manifest:
            issues.append(_issue(f"$.{key}", "Missing required Evidence Pack v1 field."))

    if manifest.get("schema_name") != PACK_V1_SCHEMA_NAME:
        issues.append(
            _issue(
                "$.schema_name",
                f"Expected schema_name {PACK_V1_SCHEMA_NAME}.",
                actual=manifest.get("schema_name"),
            )
        )
    schema_version = manifest.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version != PACK_V1_SCHEMA_VERSION:
        issues.append(_issue("$.schema_version", "Unsupported Evidence Pack v1 schema version.", actual=schema_version))

    if "metadata" in manifest:
        _validate_metadata(manifest.get("metadata"), issues)
    if "input_kinds" in manifest:
        _validate_string_list(issues, manifest.get("input_kinds"), "$.input_kinds", min_items=1, allowed=ALLOWED_INPUT_KINDS)
    if "artifact_policy" in manifest:
        _validate_artifact_policy(manifest.get("artifact_policy"), issues, root=resolved_root)
    if "recall_policy" in manifest:
        _validate_recall_policy(manifest.get("recall_policy"), issues)
    if "support_policy" in manifest:
        _validate_support_policy(manifest.get("support_policy"), issues)
    if "report_sections" in manifest:
        _validate_report_sections(manifest.get("report_sections"), issues)
    if "benchmark_fixtures" in manifest:
        _validate_benchmark_fixtures(manifest.get("benchmark_fixtures"), issues)
    if "redaction_policy" in manifest:
        _validate_redaction_policy(manifest.get("redaction_policy"), issues)
    if "output_schema_refs" in manifest:
        _validate_output_schema_refs(manifest, issues, root=resolved_root)
    if "core_transform_refs" in manifest:
        _validate_core_transform_refs(manifest, issues)

    return issues


def build_policy_security_checks(
    manifest: Mapping[str, Any] | Any,
    *,
    validation_issues: list[dict[str, Any]],
    manifest_path: Path,
    root: Path,
    check_lock: bool = True,
    lock_status: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    key_hits: dict[str, list[str]] = {name: [] for name in DENIED_KEY_PATTERNS}
    value_hits: dict[str, list[str]] = {name: [] for name in DENIED_VALUE_PATTERNS}
    if isinstance(manifest, Mapping):
        for path, key in _iter_manifest_keys(manifest):
            for name, pattern in DENIED_KEY_PATTERNS.items():
                if pattern.search(key):
                    key_hits[name].append(f"{path}={key}")
        for path, value in _iter_manifest_strings(manifest):
            for name, pattern in DENIED_VALUE_PATTERNS.items():
                if pattern.search(value):
                    value_hits[name].append(f"{path}={value}")

    denied_runtime = [
        *key_hits["python_field"],
        *key_hits["javascript_field"],
        *key_hits["shell_field"],
        *value_hits["python_content"],
        *value_hits["javascript_content"],
        *value_hits["shell_content"],
    ]
    denied_permissions = [
        *key_hits["network_field"],
        *key_hits["api_field"],
        *key_hits["secret_access_field"],
        *key_hits["environment_field"],
        *key_hits["repo_write_field"],
        *key_hits["model_call_field"],
        *key_hits["training_export_field"],
        *value_hits["network_content"],
        *value_hits["api_content"],
        *value_hits["secret_access_content"],
        *value_hits["environment_content"],
        *value_hits["repo_write_content"],
        *value_hits["model_call_content"],
        *value_hits["training_export_content"],
    ]
    denied_install_update = [
        *key_hits["install_or_update_field"],
        *value_hits["install_or_update_content"],
    ]
    unknown_fields = sorted(issue["path"] for issue in validation_issues if issue.get("check_id") == "unknown_field")
    path_validation_issues = sorted(
        issue["path"]
        for issue in validation_issues
        if issue.get("check_id") in {"file_glob", "path_boundary", "path_traversal", "remote_url", "symlink_traversal"}
    )
    denied_paths = [
        *key_hits["file_glob_field"],
        *key_hits["symlink_traversal_field"],
        *path_validation_issues,
        *value_hits["path_traversal_content"],
        *value_hits["symlink_traversal_content"],
        *value_hits["remote_url_content"],
    ]
    invalid_core_transforms = sorted(
        issue["actual"]
        for issue in validation_issues
        if issue.get("path", "").startswith("$.core_transform_refs[") and issue.get("check_id") == "allowlist"
    )
    support_policy = manifest.get("support_policy") if isinstance(manifest, Mapping) else {}
    support_kernel_ok = (
        isinstance(support_policy, Mapping)
        and support_policy.get("required_kernel") == "evidence_support_v1"
        and support_policy.get("require_support_kernel") is True
    )
    if lock_status is None:
        lock_status = inspect_pack_lock_status(manifest_path, manifest=manifest, root=root) if check_lock else {
            "status": "skipped",
            "message": "Lock integrity check skipped while writing a new lockfile.",
            "evidence": [],
        }

    return [
        _security_check(
            "schema_validity",
            "pass" if not validation_issues else "block",
            "Evidence Pack v1 schema is valid." if not validation_issues else "Evidence Pack v1 schema has blocking issues.",
            [issue["path"] for issue in validation_issues],
        ),
        _security_check(
            "unknown_fields",
            "pass" if not unknown_fields else "block",
            "No unknown fields are present." if not unknown_fields else "Unknown fields are present.",
            unknown_fields,
        ),
        _security_check(
            "no_executable_runtime",
            "pass" if not denied_runtime else "block",
            "No Python, JavaScript, or shell indicators are present."
            if not denied_runtime
            else "Executable runtime indicators are present.",
            denied_runtime,
        ),
        _security_check(
            "no_external_or_privileged_access",
            "pass" if not denied_permissions else "block",
            "No network, credential, environment, repo mutation, model-call, or training-export requests are present."
            if not denied_permissions
            else "Privileged access indicators are present.",
            denied_permissions,
        ),
        _security_check(
            "path_boundary",
            "pass" if not denied_paths else "block",
            "No remote URL, traversal, glob, or symlink traversal indicators are present."
            if not denied_paths
            else "Path boundary violations are present.",
            denied_paths,
        ),
        _security_check(
            "no_install_or_auto_update",
            "pass" if not denied_install_update else "block",
            "No install or auto-update hooks are present."
            if not denied_install_update
            else "Install or auto-update indicators are present.",
            denied_install_update,
        ),
        _security_check(
            "core_transform_allowlist",
            "pass" if not invalid_core_transforms else "block",
            "Core transforms are from the allowlist."
            if not invalid_core_transforms
            else "Core transforms include values outside the allowlist.",
            invalid_core_transforms,
        ),
        _security_check(
            "support_kernel_required",
            "pass" if support_kernel_ok else "block",
            "Pack output is required to pass through the Evidence Support Kernel."
            if support_kernel_ok
            else "Pack output is not required to pass through the Evidence Support Kernel.",
            [],
        ),
        _security_check(
            "lock_manifest_integrity",
            "pass" if lock_status["status"] in {"not_found", "match", "skipped"} else "block",
            lock_status["message"],
            lock_status.get("evidence", []),
        ),
    ]


def _security_check(check_id: str, status: str, message: str, evidence: list[str]) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "status": status,
        "message": message,
        "evidence": evidence,
    }


def evidence_pack_v1_manifest_schema_path(root: Path | None = None) -> Path:
    return _resolve_root(root) / "schemas" / "satellite_evidence_pack_v1.schema.json"


def evidence_pack_v1_artifact_root(*, workspace_id: str = DEFAULT_WORKSPACE_ID, root: Path | None = None) -> Path:
    return _resolve_root(root) / "artifacts" / "satellite_evidence_pack_v1" / workspace_id


def evidence_pack_v1_audit_latest_path(*, workspace_id: str = DEFAULT_WORKSPACE_ID, root: Path | None = None) -> Path:
    return evidence_pack_v1_artifact_root(workspace_id=workspace_id, root=root) / "audits" / "latest.json"


def evidence_pack_v1_audit_run_path(pack_id: str, *, workspace_id: str = DEFAULT_WORKSPACE_ID, root: Path | None = None) -> Path:
    return (
        evidence_pack_v1_artifact_root(workspace_id=workspace_id, root=root)
        / "audits"
        / "runs"
        / f"{timestamp_slug()}-{pack_id}-audit.json"
    )


def evidence_pack_v1_test_latest_path(*, workspace_id: str = DEFAULT_WORKSPACE_ID, root: Path | None = None) -> Path:
    return evidence_pack_v1_artifact_root(workspace_id=workspace_id, root=root) / "tests" / "latest.json"


def evidence_pack_v1_test_run_path(pack_id: str, *, workspace_id: str = DEFAULT_WORKSPACE_ID, root: Path | None = None) -> Path:
    return (
        evidence_pack_v1_artifact_root(workspace_id=workspace_id, root=root)
        / "tests"
        / "runs"
        / f"{timestamp_slug()}-{pack_id}-test.json"
    )


def pack_id_from_manifest(manifest: Mapping[str, Any] | Any) -> str:
    metadata = manifest.get("metadata") if isinstance(manifest, Mapping) else None
    pack_id = metadata.get("pack_id") if isinstance(metadata, Mapping) else None
    return _clean_text(pack_id) or "unknown-pack"


def manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_pack_v1_lock_path(manifest_path: Path) -> Path:
    name = manifest_path.name
    for suffix in (".satellite.yaml", ".satellite.yml", ".satellite.json"):
        if name.endswith(suffix):
            return manifest_path.with_name(name[: -len(suffix)] + ".satellite.lock.json")
    return manifest_path.with_suffix(manifest_path.suffix + ".lock.json")


def load_evidence_pack_v1_manifest(pack_path: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = resolve_pack_manifest_path(pack_path)
    manifest = load_pack_manifest(manifest_path)
    if not isinstance(manifest, dict):
        raise EvidencePackV1Error(f"{manifest_path}: manifest root must be an object.")
    return manifest, manifest_path


def is_evidence_pack_v1_path(pack_path: Path) -> bool:
    try:
        manifest, _manifest_path = load_evidence_pack_v1_manifest(pack_path)
    except (OSError, PackManifestError, EvidencePackV1Error, json.JSONDecodeError):
        return False
    if manifest.get("schema_name") == PACK_V1_SCHEMA_NAME:
        return True
    return any(key in manifest for key in ("metadata", "input_kinds", "artifact_policy", "core_transform_refs"))


def _manifest_core_transforms(manifest: Mapping[str, Any] | Any) -> list[str]:
    transforms = manifest.get("core_transform_refs") if isinstance(manifest, Mapping) else []
    return [value for value in transforms if isinstance(value, str)] if isinstance(transforms, list) else []


def inspect_pack_lock_status(
    manifest_path: Path,
    *,
    manifest: Mapping[str, Any] | Any | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    lock_path = evidence_pack_v1_lock_path(manifest_path)
    if not lock_path.exists():
        return {
            "status": "not_found",
            "lock_path": str(lock_path),
            "message": "No lockfile is present; run `pack lock` when you want mutation detection.",
            "evidence": [],
        }
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "lock_path": str(lock_path),
            "message": f"Lockfile could not be read: {exc}",
            "evidence": [str(lock_path)],
        }
    expected = _clean_text(lock.get("manifest_sha256"))
    actual = manifest_sha256(manifest_path)
    issues: list[str] = []
    if lock.get("schema_name") != PACK_V1_LOCK_SCHEMA_NAME:
        issues.append(f"schema_name={lock.get('schema_name')}")
    if lock.get("schema_version") != PACK_V1_LOCK_SCHEMA_VERSION:
        issues.append(f"schema_version={lock.get('schema_version')}")
    if expected != actual:
        issues.extend([f"expected={expected}", f"actual={actual}"])
    if manifest is not None:
        expected_pack_id = pack_id_from_manifest(manifest)
        if lock.get("pack_id") != expected_pack_id:
            issues.append(f"pack_id={lock.get('pack_id')} expected_pack_id={expected_pack_id}")
        expected_transforms = _manifest_core_transforms(manifest)
        locked_transforms = lock.get("allowed_core_transforms")
        if locked_transforms != expected_transforms:
            issues.append("allowed_core_transforms_mismatch")
    if issues:
        return {
            "status": "mismatch",
            "lock_path": str(lock_path),
            "message": "Lockfile does not match the current Evidence Pack v1 manifest.",
            "evidence": issues,
        }
    return {
        "status": "match",
        "lock_path": str(lock_path),
        "message": "Lockfile manifest hash matches the current manifest.",
        "evidence": [str(lock_path)],
    }


def build_evidence_pack_v1_audit(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    root: Path | None = None,
    strict: bool = False,
    check_lock: bool = True,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    schema_path = evidence_pack_v1_manifest_schema_path(resolved_root)
    validation_issues = validate_evidence_pack_v1_manifest(manifest, root=resolved_root)
    lock_status = inspect_pack_lock_status(manifest_path, manifest=manifest, root=resolved_root) if check_lock else {
        "status": "skipped",
        "lock_path": str(evidence_pack_v1_lock_path(manifest_path)),
        "message": "Lock integrity check skipped while writing a new lockfile.",
        "evidence": [],
    }
    security_checks = build_policy_security_checks(
        manifest,
        validation_issues=validation_issues,
        manifest_path=manifest_path,
        root=resolved_root,
        check_lock=check_lock,
        lock_status=lock_status,
    )
    blocked_reasons = [
        f"{issue['path']}: {issue['message']}"
        for issue in validation_issues
    ]
    blocked_reasons.extend(
        f"{check['check_id']}: {check['message']}"
        for check in security_checks
        if check.get("status") == "block"
        and check.get("check_id") != "schema_validity"
    )
    verdict = "block" if blocked_reasons else "pass"
    return {
        "schema_name": PACK_V1_AUDIT_SCHEMA_NAME,
        "schema_version": PACK_V1_SCHEMA_VERSION,
        "audit_id": f"evidence-pack-v1-audit:{timestamp_slug()}",
        "audited_at_utc": timestamp_utc(),
        "strict": bool(strict),
        "pack_id": pack_id_from_manifest(manifest),
        "pack_version": (manifest.get("metadata") or {}).get("version") if isinstance(manifest.get("metadata"), Mapping) else None,
        "verdict": verdict,
        "manifest_path": str(Path(manifest_path).resolve()),
        "schema_path": str(schema_path.resolve()),
        "manifest_sha256": manifest_sha256(manifest_path),
        "allowed_fields": sorted(ALLOWED_TOP_LEVEL_KEYS),
        "allowed_core_transforms": sorted(ALLOWED_CORE_TRANSFORMS),
        "validation": {
            "schema_valid": not validation_issues,
            "issues": copy.deepcopy(validation_issues),
        },
        "security_checks": security_checks,
        "blocked_reasons": blocked_reasons,
        "lock_status": lock_status,
        "policy_boundaries": {
            "executable_plugin_runtime": "blocked",
            "api": "blocked",
            "network": "blocked",
            "secrets": "blocked",
            "repo_write": "blocked",
            "model_call": "blocked",
            "training_export": "blocked",
        },
        "source_paths": [str(Path(manifest_path).resolve()), str(schema_path.resolve())],
    }


def audit_evidence_pack_v1_path(
    pack_path: Path,
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    strict: bool = False,
    write_artifact: bool = True,
) -> tuple[dict[str, Any], Path | None, Path | None]:
    resolved_root = _resolve_root(root)
    manifest, manifest_path = load_evidence_pack_v1_manifest(pack_path)
    audit = build_evidence_pack_v1_audit(
        manifest,
        manifest_path=manifest_path,
        root=resolved_root,
        strict=strict,
    )
    latest_path: Path | None = None
    run_path: Path | None = None
    if write_artifact:
        latest_path = evidence_pack_v1_audit_latest_path(workspace_id=workspace_id, root=resolved_root)
        run_path = evidence_pack_v1_audit_run_path(audit["pack_id"], workspace_id=workspace_id, root=resolved_root)
        audit["paths"] = {
            "audit_latest_path": str(latest_path),
            "audit_run_path": str(run_path),
        }
        write_json(run_path, audit)
        write_json(latest_path, audit)
    return audit, latest_path, run_path


def _status_for_fixture(input_kind: str, requested_polarity: str) -> tuple[str, str | None, list[str]]:
    if input_kind == "failure" or requested_polarity in {"risk", "negative"}:
        return "failed", "fail", ["test_fail"]
    if requested_polarity == "positive":
        return "passed", "pass", ["test_pass"]
    return "diagnostic", None, []


def _event_for_fixture(fixture: Mapping[str, Any], artifact_ref: Mapping[str, Any], *, pack_id: str) -> dict[str, Any]:
    input_kind = str(fixture.get("input_kind") or "failure")
    requested_polarity = str(fixture.get("requested_polarity") or "risk")
    status, quality_status, evidence_types = _status_for_fixture(input_kind, requested_polarity)
    return {
        "schema_name": "software-satellite-event",
        "schema_version": 1,
        "event_id": f"pack-fixture:{pack_id}:{fixture.get('id')}",
        "event_kind": "evidence_pack_v1_fixture",
        "recorded_at_utc": "2026-05-12T00:00:00+00:00",
        "workspace": {"workspace_id": "pack-v1-test"},
        "session": {"session_id": f"pack-fixture:{pack_id}", "surface": "pack_test", "mode": "policy_kernel"},
        "outcome": {"status": status, "quality_status": quality_status, "execution_status": status},
        "content": {
            "notes": [f"evidence_pack_v1_fixture: {fixture.get('id')}"],
            "options": {
                "workflow": "evidence_pack_v1_test",
                "evidence_types": evidence_types,
                "artifact_vault_refs": [dict(artifact_ref)],
            },
        },
        "source_refs": {},
        "tags": ["evidence_pack_v1_fixture"],
    }


def _run_fixture(fixture: Mapping[str, Any], *, pack_id: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        fixture_root = Path(tmpdir)
        artifact_name = _clean_text(fixture.get("artifact_name")) or "fixture.txt"
        fixture_path = fixture_root / artifact_name
        fixture_path.write_text(str(fixture.get("fixture_text") or ""), encoding="utf-8")
        input_kind = _clean_text(fixture.get("input_kind")) or "review_note"
        artifact_ref = capture_artifact(
            fixture_path,
            kind=FIXTURE_ARTIFACT_KINDS.get(input_kind, "unknown"),
            root=fixture_root,
        )
        event = _event_for_fixture(fixture, artifact_ref, pack_id=pack_id)
        support = build_evidence_support_result(
            event["event_id"],
            event=event,
            requested_polarity=_clean_text(fixture.get("requested_polarity")),
            root=fixture_root,
        )
    expected_class = _clean_text(fixture.get("expected_support_class"))
    expected_can_support = fixture.get("expected_can_support_decision")
    class_ok = expected_class is None or support.get("support_class") == expected_class
    support_ok = not isinstance(expected_can_support, bool) or support.get("can_support_decision") is expected_can_support
    return {
        "fixture_id": fixture.get("id"),
        "passed": bool(class_ok and support_ok),
        "support_kernel_used": support.get("schema_name") == "software-satellite-evidence-support-result",
        "expected_support_class": expected_class,
        "actual_support_class": support.get("support_class"),
        "expected_can_support_decision": expected_can_support,
        "actual_can_support_decision": support.get("can_support_decision"),
        "support_result": support,
    }


def test_evidence_pack_v1_path(
    pack_path: Path,
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    strict: bool = False,
    write_artifact: bool = True,
) -> tuple[dict[str, Any], Path | None, Path | None]:
    resolved_root = _resolve_root(root)
    manifest, manifest_path = load_evidence_pack_v1_manifest(pack_path)
    audit = build_evidence_pack_v1_audit(
        manifest,
        manifest_path=manifest_path,
        root=resolved_root,
        strict=strict,
        check_lock=False,
    )
    pack_id = pack_id_from_manifest(manifest)
    fixture_results: list[dict[str, Any]] = []
    if audit["verdict"] != "block":
        fixtures = manifest.get("benchmark_fixtures") if isinstance(manifest.get("benchmark_fixtures"), list) else []
        for fixture in fixtures:
            if isinstance(fixture, Mapping):
                fixture_results.append(_run_fixture(fixture, pack_id=pack_id))

    all_fixtures_passed = bool(fixture_results) and all(item.get("passed") for item in fixture_results)
    support_kernel_used = bool(fixture_results) and all(item.get("support_kernel_used") for item in fixture_results)
    pack_output_bypasses_support_kernel = not support_kernel_used
    passed = audit["verdict"] != "block" and all_fixtures_passed and not pack_output_bypasses_support_kernel
    result = {
        "schema_name": PACK_V1_TEST_SCHEMA_NAME,
        "schema_version": PACK_V1_SCHEMA_VERSION,
        "test_id": f"evidence-pack-v1-test:{timestamp_slug()}",
        "tested_at_utc": timestamp_utc(),
        "strict": bool(strict),
        "pack_id": pack_id,
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": manifest_sha256(manifest_path),
        "passed": passed,
        "audit_verdict": audit["verdict"],
        "api_key_required": False,
        "fixture_count": len(fixture_results),
        "fixture_results": fixture_results,
        "support_kernel_result_count": sum(1 for item in fixture_results if item.get("support_kernel_used")),
        "pack_output_bypasses_support_kernel": pack_output_bypasses_support_kernel,
        "blocked_reasons": list(audit.get("blocked_reasons") or []),
        "source_paths": [str(Path(manifest_path).resolve())],
    }
    latest_path: Path | None = None
    run_path: Path | None = None
    if write_artifact:
        latest_path = evidence_pack_v1_test_latest_path(workspace_id=workspace_id, root=resolved_root)
        run_path = evidence_pack_v1_test_run_path(pack_id, workspace_id=workspace_id, root=resolved_root)
        result["paths"] = {
            "test_latest_path": str(latest_path),
            "test_run_path": str(run_path),
        }
        write_json(run_path, result)
        write_json(latest_path, result)
    return result, latest_path, run_path


def lock_evidence_pack_v1_path(pack_path: Path, *, root: Path | None = None, strict: bool = True) -> tuple[dict[str, Any], Path]:
    resolved_root = _resolve_root(root)
    manifest, manifest_path = load_evidence_pack_v1_manifest(pack_path)
    audit = build_evidence_pack_v1_audit(
        manifest,
        manifest_path=manifest_path,
        root=resolved_root,
        strict=strict,
        check_lock=False,
    )
    if audit["verdict"] == "block":
        raise EvidencePackV1Error("Cannot lock an invalid Evidence Pack v1 manifest.")
    lock_path = evidence_pack_v1_lock_path(manifest_path)
    lock = {
        "schema_name": PACK_V1_LOCK_SCHEMA_NAME,
        "schema_version": PACK_V1_LOCK_SCHEMA_VERSION,
        "pack_id": pack_id_from_manifest(manifest),
        "manifest_sha256": manifest_sha256(manifest_path),
        "allowed_core_transforms": [
            value
            for value in manifest.get("core_transform_refs", [])
            if isinstance(value, str)
        ],
        "locked_at_utc": _timestamp_z(),
        "manifest_path": str(Path(manifest_path).resolve()),
        "source_paths": [str(Path(manifest_path).resolve())],
    }
    write_json(lock_path, lock)
    return lock, lock_path


BUILTIN_PACK_TEMPLATE_FILENAMES = {
    "failure-memory": "failure-memory-pack.satellite.yaml",
    "agent-session": "agent-session-pack.satellite.yaml",
}


def scaffold_evidence_pack_v1(kind: str, output: Path, *, overwrite: bool = False) -> dict[str, Any]:
    normalized_kind = kind.strip().lower().replace("_", "-")
    template_filename = BUILTIN_PACK_TEMPLATE_FILENAMES.get(normalized_kind)
    if template_filename is None:
        raise EvidencePackV1Error(f"Unsupported built-in Evidence Pack v1 kind `{kind}`.")
    template_path = repo_root() / "templates" / template_filename
    try:
        template_text = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvidencePackV1Error(f"Built-in template is missing: {template_path}") from exc
    output_path = Path(output).expanduser()
    status = "written"
    if output_path.exists():
        try:
            existing_text = output_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise EvidencePackV1Error(f"Could not read existing output: {output_path}") from exc
        if existing_text == template_text:
            status = "unchanged"
        elif not overwrite:
            raise EvidencePackV1Error(f"Output already exists with different content: {output_path}. Use --force to overwrite.")
        else:
            status = "overwritten"
    if status != "unchanged":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(template_text, encoding="utf-8")
    return {
        "kind": normalized_kind,
        "output_path": str(output_path.resolve()),
        "schema_name": PACK_V1_SCHEMA_NAME,
        "status": status,
        "template_path": str(template_path.resolve()),
    }


def builtin_pack_list() -> list[dict[str, Any]]:
    return [
        {
            "kind": kind,
            "schema_name": PACK_V1_SCHEMA_NAME,
            "template_filename": BUILTIN_PACK_TEMPLATE_FILENAMES[kind],
        }
        for kind in sorted(BUILTIN_PACK_TEMPLATE_FILENAMES)
    ]


def format_evidence_pack_v1_audit_report(audit: Mapping[str, Any]) -> str:
    lines = [
        f"Evidence Pack v1 Audit: {audit.get('verdict')}",
        f"Pack: {audit.get('pack_id') or 'unknown'} {audit.get('pack_version') or ''}".rstrip(),
        f"Strict: {str(bool(audit.get('strict'))).lower()}",
        f"Manifest: {audit.get('manifest_path')}",
    ]
    paths = audit.get("paths") if isinstance(audit.get("paths"), Mapping) else {}
    if paths:
        lines.append(f"Latest: {paths.get('audit_latest_path')}")
        lines.append(f"Run: {paths.get('audit_run_path')}")
    blocked_reasons = audit.get("blocked_reasons") or []
    if blocked_reasons:
        lines.extend(["", "Blocked Reasons:"])
        for reason in blocked_reasons:
            lines.append(f"- {reason}")
    security_checks = audit.get("security_checks") or []
    if security_checks:
        lines.extend(["", "Security Checks:"])
        for check in security_checks:
            if isinstance(check, Mapping):
                lines.append(f"- {check.get('check_id')}: {check.get('status')} - {check.get('message')}")
    return "\n".join(lines)


def format_evidence_pack_v1_test_report(result: Mapping[str, Any]) -> str:
    lines = [
        f"Evidence Pack v1 Test: {'pass' if result.get('passed') else 'fail'}",
        f"Pack: {result.get('pack_id') or 'unknown'}",
        f"Fixtures: {result.get('fixture_count', 0)}",
        f"Support kernel results: {result.get('support_kernel_result_count', 0)}",
        f"API key required: {str(bool(result.get('api_key_required'))).lower()}",
        f"Support bypass: {str(bool(result.get('pack_output_bypasses_support_kernel'))).lower()}",
    ]
    paths = result.get("paths") if isinstance(result.get("paths"), Mapping) else {}
    if paths:
        lines.append(f"Latest: {paths.get('test_latest_path')}")
        lines.append(f"Run: {paths.get('test_run_path')}")
    blocked_reasons = result.get("blocked_reasons") or []
    if blocked_reasons:
        lines.extend(["", "Blocked Reasons:"])
        for reason in blocked_reasons:
            lines.append(f"- {reason}")
    fixture_results = result.get("fixture_results") or []
    if fixture_results:
        lines.extend(["", "Fixtures:"])
        for fixture in fixture_results:
            if isinstance(fixture, Mapping):
                state = "pass" if fixture.get("passed") else "fail"
                lines.append(
                    f"- {fixture.get('fixture_id')}: {state} "
                    f"({fixture.get('actual_support_class')}, can_support={fixture.get('actual_can_support_decision')})"
                )
    return "\n".join(lines)
