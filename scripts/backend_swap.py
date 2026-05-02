#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol
from uuid import uuid4

from agent_lane import (
    TASK_KINDS,
    build_agent_task,
    record_agent_lane_snapshot,
    record_agent_run,
    record_agent_task,
    run_agent_task,
)
from evaluation_loop import (
    append_evaluation_comparison,
    build_evaluation_comparison,
    evaluation_comparison_log_path,
    record_evaluation_snapshot,
)
from gemma_runtime import repo_root, timestamp_slug, timestamp_utc, write_json
from memory_index import rebuild_memory_index
from software_work_events import build_event_from_agent_run, read_event_log
from workspace_state import DEFAULT_WORKSPACE_ID


BACKEND_CONFIG_SCHEMA_NAME = "software-satellite-backend-config"
BACKEND_CONFIG_SCHEMA_VERSION = 1
BACKEND_CONFIG_LOG_SCHEMA_NAME = "software-satellite-backend-config-log"
BACKEND_CONFIG_LOG_SCHEMA_VERSION = 1
BACKEND_COMPATIBILITY_SCHEMA_NAME = "software-satellite-backend-compatibility-report"
BACKEND_COMPATIBILITY_SCHEMA_VERSION = 1
BACKEND_INVOCATION_SCHEMA_NAME = "software-satellite-backend-invocation"
BACKEND_INVOCATION_SCHEMA_VERSION = 1
BACKEND_HARNESS_RUN_SCHEMA_NAME = "software-satellite-backend-swap-run"
BACKEND_HARNESS_RUN_SCHEMA_VERSION = 1
BACKEND_HARNESS_RUN_LOG_SCHEMA_NAME = "software-satellite-backend-swap-run-log"
BACKEND_HARNESS_RUN_LOG_SCHEMA_VERSION = 1

ADAPTER_KINDS = ("mock", "local")
DEFAULT_WORKFLOW_KIND = "agent_lane_patch_plan_verify"
DEFAULT_WORKFLOW_REQUIRED_CAPABILITIES = (
    "text_generation",
    "agent_lane",
    "verification_commands",
    "file_first_artifacts",
)
DEFAULT_WORKFLOW_OPTIONAL_CAPABILITIES = (
    "tool_call_formatting",
    "embeddings",
)


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def backend_swap_root(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return _resolve_root(root) / "artifacts" / "backend_swap" / workspace_id


def backend_config_log_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return backend_swap_root(workspace_id=workspace_id, root=root) / "backend_configs.jsonl"


def backend_harness_run_log_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return backend_swap_root(workspace_id=workspace_id, root=root) / "harness_runs.jsonl"


def _safe_path_slug(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    slug = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in value).strip("-")
    return f"{(slug or 'backend-swap-run')[:72]}-{digest}"


def backend_harness_run_artifact_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    run_id: str,
) -> Path:
    return backend_swap_root(workspace_id=workspace_id, root=root) / "runs" / f"{_safe_path_slug(run_id)}.json"


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _string_list(value: Any) -> list[str]:
    if value is None or isinstance(value, (str, bytes, dict)):
        return []
    cleaned: list[str] = []
    try:
        iterator = iter(value)
    except TypeError:
        return []
    for item in iterator:
        text = _clean_text(item)
        if text is not None:
            cleaned.append(text)
    return cleaned


def _mapping_dict(value: Any) -> dict[str, Any]:
    return copy.deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _capability_supported(value: Any) -> bool:
    if isinstance(value, Mapping):
        return value.get("supported") is True
    return value is True


def _normalize_capabilities(value: Any, *, location: str = "") -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Backend config capabilities must be an object{location}.")

    capabilities: dict[str, dict[str, Any]] = {}
    for key, raw_capability in value.items():
        name = _clean_text(key)
        if name is None:
            raise ValueError(f"Backend config capability names must be non-empty strings{location}.")
        if isinstance(raw_capability, bool):
            capabilities[name] = {"supported": raw_capability}
            continue
        if not isinstance(raw_capability, Mapping):
            raise ValueError(f"Backend capability `{name}` must be an object or boolean{location}.")
        capability = copy.deepcopy(dict(raw_capability))
        if not isinstance(capability.get("supported"), bool):
            raise ValueError(f"Backend capability `{name}` requires boolean supported{location}.")
        capabilities[name] = capability
    return capabilities


def _config_log_header(*, workspace_id: str) -> dict[str, Any]:
    return {
        "schema_name": BACKEND_CONFIG_LOG_SCHEMA_NAME,
        "schema_version": BACKEND_CONFIG_LOG_SCHEMA_VERSION,
        "workspace_id": workspace_id,
    }


def _harness_run_log_header(*, workspace_id: str) -> dict[str, Any]:
    return {
        "schema_name": BACKEND_HARNESS_RUN_LOG_SCHEMA_NAME,
        "schema_version": BACKEND_HARNESS_RUN_LOG_SCHEMA_VERSION,
        "workspace_id": workspace_id,
    }


def _read_log_header(path: Path, *, schema_name: str, schema_version: int) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Backend swap log `{path}` was empty.")
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
    header = json.loads(first_line)
    if header.get("schema_name") != schema_name:
        raise ValueError(f"Unexpected backend swap log schema name in `{path}`.")
    if header.get("schema_version") != schema_version:
        raise ValueError(f"Unsupported backend swap log schema version in `{path}`.")
    return header


def _append_jsonl(
    path: Path,
    *,
    header: Mapping[str, Any],
    payload: Mapping[str, Any],
    workspace_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(header), ensure_ascii=False) + "\n")
    else:
        existing_header = _read_log_header(
            path,
            schema_name=str(header["schema_name"]),
            schema_version=int(header["schema_version"]),
        )
        if existing_header.get("workspace_id") != workspace_id:
            raise ValueError(f"Backend swap log `{path}` belongs to workspace `{existing_header.get('workspace_id')}`.")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")


def build_backend_config(
    *,
    backend_id: str,
    display_name: str,
    adapter_kind: str,
    model_id: str,
    capabilities: Mapping[str, Any],
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    created_at_utc: str | None = None,
    adapter_options: Mapping[str, Any] | None = None,
    limits: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    cleaned_backend_id = _clean_text(backend_id)
    cleaned_display_name = _clean_text(display_name)
    cleaned_adapter_kind = _clean_text(adapter_kind)
    cleaned_model_id = _clean_text(model_id)
    if cleaned_backend_id is None:
        raise ValueError("Backend config requires backend_id.")
    if cleaned_display_name is None:
        raise ValueError("Backend config requires display_name.")
    if cleaned_adapter_kind not in ADAPTER_KINDS:
        raise ValueError(f"Unsupported backend adapter_kind `{adapter_kind}`.")
    if cleaned_model_id is None:
        raise ValueError("Backend config requires model_id.")
    return {
        "schema_name": BACKEND_CONFIG_SCHEMA_NAME,
        "schema_version": BACKEND_CONFIG_SCHEMA_VERSION,
        "backend_id": cleaned_backend_id,
        "workspace_id": workspace_id,
        "created_at_utc": created_at_utc or timestamp_utc(),
        "display_name": cleaned_display_name,
        "adapter_kind": cleaned_adapter_kind,
        "model_id": cleaned_model_id,
        "capabilities": _normalize_capabilities(capabilities),
        "adapter_options": _mapping_dict(adapter_options),
        "limits": _mapping_dict(limits),
        "metadata": _mapping_dict(metadata),
        "tags": _string_list(tags),
    }


def validate_backend_config(config: Mapping[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    payload = copy.deepcopy(dict(config))
    location = f" in `{path}`" if path is not None else ""
    if payload.get("schema_name") != BACKEND_CONFIG_SCHEMA_NAME:
        raise ValueError(f"Unexpected backend config schema name{location}.")
    if payload.get("schema_version") != BACKEND_CONFIG_SCHEMA_VERSION:
        raise ValueError(f"Unsupported backend config schema version{location}.")
    backend_id = _clean_text(payload.get("backend_id"))
    if backend_id is None:
        raise ValueError(f"Backend config is missing backend_id{location}.")
    workspace_id = _clean_text(payload.get("workspace_id"))
    if workspace_id is None:
        raise ValueError(f"Backend config is missing workspace_id{location}.")
    adapter_kind = _clean_text(payload.get("adapter_kind"))
    if adapter_kind not in ADAPTER_KINDS:
        raise ValueError(f"Unsupported backend adapter_kind `{adapter_kind}`{location}.")
    model_id = _clean_text(payload.get("model_id"))
    if model_id is None:
        raise ValueError(f"Backend config is missing model_id{location}.")
    display_name = _clean_text(payload.get("display_name"))
    if display_name is None:
        raise ValueError(f"Backend config is missing display_name{location}.")
    payload["backend_id"] = backend_id
    payload["workspace_id"] = workspace_id
    payload["adapter_kind"] = adapter_kind
    payload["model_id"] = model_id
    payload["display_name"] = display_name
    payload["capabilities"] = _normalize_capabilities(payload.get("capabilities"), location=location)
    payload["adapter_options"] = _mapping_dict(payload.get("adapter_options"))
    payload["limits"] = _mapping_dict(payload.get("limits"))
    payload["metadata"] = _mapping_dict(payload.get("metadata"))
    payload["tags"] = _string_list(payload.get("tags"))
    return payload


def default_backend_configs(*, workspace_id: str = DEFAULT_WORKSPACE_ID) -> list[dict[str, Any]]:
    common_capabilities = {
        "text_generation": {
            "supported": True,
            "interface": "generate_text",
            "streaming": False,
            "mode": "deterministic_mock",
        },
        "agent_lane": {
            "supported": True,
            "task_kinds": ["patch_plan_verify"],
        },
        "verification_commands": {
            "supported": True,
            "runner": "local_subprocess",
        },
        "file_first_artifacts": {
            "supported": True,
            "paths": ["artifacts/backend_swap", "artifacts/agent_lane", "artifacts/evaluation"],
        },
        "tool_call_formatting": {
            "supported": False,
            "reason": "M6 keeps tool-call formatting out of the first compatibility contract.",
        },
        "embeddings": {
            "supported": False,
            "reason": "M6 keeps semantic recall optional and downstream.",
        },
    }
    return [
        build_backend_config(
            workspace_id=workspace_id,
            backend_id="mock-fast-local",
            display_name="Mock Fast Local",
            adapter_kind="mock",
            model_id="mock/fast-local-v1",
            capabilities=common_capabilities,
            adapter_options={
                "response_style": "fast",
                "summary_prefix": "Fast local mock",
            },
            limits={
                "max_context_chars": 4000,
                "max_output_chars": 800,
            },
            metadata={
                "latency_profile": "fast",
                "comparison_role": "speed-biased local dry run",
            },
            tags=["m6", "backend_swap", "default"],
        ),
        build_backend_config(
            workspace_id=workspace_id,
            backend_id="mock-careful-local",
            display_name="Mock Careful Local",
            adapter_kind="mock",
            model_id="mock/careful-local-v1",
            capabilities=common_capabilities,
            adapter_options={
                "response_style": "careful",
                "summary_prefix": "Careful local mock",
            },
            limits={
                "max_context_chars": 8000,
                "max_output_chars": 1200,
            },
            metadata={
                "latency_profile": "careful",
                "comparison_role": "quality-biased local dry run",
            },
            tags=["m6", "backend_swap", "default"],
        ),
    ]


def append_backend_config(path: Path, config: Mapping[str, Any], *, workspace_id: str) -> dict[str, Any]:
    payload = validate_backend_config(config)
    if payload.get("workspace_id") != workspace_id:
        raise ValueError(f"Backend config belongs to workspace `{payload.get('workspace_id')}`.")
    if path.exists() and path.stat().st_size > 0:
        for existing in read_backend_configs(path):
            if existing.get("backend_id") == payload.get("backend_id"):
                raise ValueError(f"Backend config `{payload.get('backend_id')}` already exists in `{path}`.")
    _append_jsonl(
        path,
        header=_config_log_header(workspace_id=workspace_id),
        payload=payload,
        workspace_id=workspace_id,
    )
    return payload


def _backend_config_equivalence_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_backend_config(config)
    return {
        key: copy.deepcopy(payload.get(key))
        for key in (
            "workspace_id",
            "display_name",
            "adapter_kind",
            "model_id",
            "capabilities",
            "adapter_options",
            "limits",
            "metadata",
            "tags",
        )
    }


def read_backend_configs(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    header = _read_log_header(
        path,
        schema_name=BACKEND_CONFIG_LOG_SCHEMA_NAME,
        schema_version=BACKEND_CONFIG_LOG_SCHEMA_VERSION,
    )
    workspace_id = _clean_text(header.get("workspace_id"))
    configs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            cleaned = line.strip()
            if not cleaned:
                continue
            config = validate_backend_config(json.loads(cleaned), path=path)
            if workspace_id is not None and config.get("workspace_id") != workspace_id:
                raise ValueError(f"Backend config log `{path}` contains workspace `{config.get('workspace_id')}`.")
            configs.append(config)
    return configs


def read_backend_config_file(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_backend_config(payload, path=Path(path))


def ensure_backend_config_files(
    paths: Iterable[Path],
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
) -> list[dict[str, Any]]:
    config_log = backend_config_log_path(workspace_id=workspace_id, root=root)
    existing = read_backend_configs(config_log)
    by_id = {str(config["backend_id"]): config for config in existing}
    ensured: list[dict[str, Any]] = []
    ensured_by_id: dict[str, dict[str, Any]] = {}

    for path in paths:
        config = read_backend_config_file(Path(path))
        if config.get("workspace_id") != workspace_id:
            raise ValueError(f"Backend config `{path}` belongs to workspace `{config.get('workspace_id')}`.")
        backend_id = str(config["backend_id"])
        ensured_config = ensured_by_id.get(backend_id)
        if ensured_config is not None:
            if _backend_config_equivalence_payload(ensured_config) != _backend_config_equivalence_payload(config):
                raise ValueError(f"Backend config `{backend_id}` was provided more than once with different metadata.")
            continue
        existing_config = by_id.get(backend_id)
        if existing_config is not None:
            if _backend_config_equivalence_payload(existing_config) != _backend_config_equivalence_payload(config):
                raise ValueError(f"Backend config `{backend_id}` already exists with different metadata.")
            ensured_by_id[backend_id] = existing_config
            ensured.append(existing_config)
            continue
        appended = append_backend_config(config_log, config, workspace_id=workspace_id)
        by_id[backend_id] = appended
        ensured_by_id[backend_id] = appended
        ensured.append(appended)
    return ensured


def ensure_default_backend_configs(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
) -> list[dict[str, Any]]:
    path = backend_config_log_path(workspace_id=workspace_id, root=root)
    existing = read_backend_configs(path)
    existing_ids = {str(config.get("backend_id")) for config in existing}
    configs = list(existing)
    for config in default_backend_configs(workspace_id=workspace_id):
        if str(config.get("backend_id")) in existing_ids:
            continue
        configs.append(append_backend_config(path, config, workspace_id=workspace_id))
    return configs


def workflow_capability_requirements(
    workflow_kind: str = DEFAULT_WORKFLOW_KIND,
) -> dict[str, list[str]]:
    if workflow_kind != DEFAULT_WORKFLOW_KIND:
        raise ValueError(f"Unsupported backend swap workflow_kind `{workflow_kind}`.")
    return {
        "required": list(DEFAULT_WORKFLOW_REQUIRED_CAPABILITIES),
        "optional": list(DEFAULT_WORKFLOW_OPTIONAL_CAPABILITIES),
    }


def check_backend_compatibility(
    config: Mapping[str, Any],
    *,
    workflow_kind: str = DEFAULT_WORKFLOW_KIND,
    required_capabilities: Iterable[str] | None = None,
    optional_capabilities: Iterable[str] | None = None,
) -> dict[str, Any]:
    payload = validate_backend_config(config)
    requirements = workflow_capability_requirements(workflow_kind)
    required = (
        requirements["required"]
        if required_capabilities is None
        else _string_list(required_capabilities)
    )
    optional = (
        requirements["optional"]
        if optional_capabilities is None
        else _string_list(optional_capabilities)
    )
    capabilities = _mapping_dict(payload.get("capabilities"))
    required_records: list[dict[str, Any]] = []
    optional_records: list[dict[str, Any]] = []
    missing: list[str] = []

    for name in required:
        capability = _mapping_dict(capabilities.get(name))
        supported = _capability_supported(capability)
        required_records.append(
            {
                "name": name,
                "supported": supported,
                "metadata": capability,
            }
        )
        if not supported:
            missing.append(name)

    for name in optional:
        capability = _mapping_dict(capabilities.get(name))
        optional_records.append(
            {
                "name": name,
                "supported": _capability_supported(capability),
                "metadata": capability,
            }
        )

    return {
        "schema_name": BACKEND_COMPATIBILITY_SCHEMA_NAME,
        "schema_version": BACKEND_COMPATIBILITY_SCHEMA_VERSION,
        "backend_id": payload["backend_id"],
        "adapter_kind": payload["adapter_kind"],
        "model_id": payload["model_id"],
        "workflow_kind": workflow_kind,
        "checked_at_utc": timestamp_utc(),
        "status": "compatible" if not missing else "incompatible",
        "required_capabilities": required_records,
        "optional_capabilities": optional_records,
        "missing_capabilities": missing,
    }


class BackendAdapter(Protocol):
    def generate_text(
        self,
        prompt: str,
        *,
        config: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...


class MockBackendAdapter:
    def generate_text(
        self,
        prompt: str,
        *,
        config: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = validate_backend_config(config)
        options = _mapping_dict(payload.get("adapter_options"))
        context_payload = _mapping_dict(context)
        prompt_text = _clean_text(prompt) or ""
        prompt_digest = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]
        response_style = _clean_text(options.get("response_style")) or "mock"
        summary_prefix = _clean_text(options.get("summary_prefix")) or payload["display_name"]
        forced_status = (_clean_text(options.get("force_status")) or "ok").lower()
        status = "failed" if forced_status in {"failed", "fail", "error"} else "ok"
        task_title = _clean_text(context_payload.get("task_title")) or "backend swap task"
        started_at = timestamp_utc()
        if status == "ok":
            output_text = (
                f"{summary_prefix} completed `{task_title}` with the shared outer workflow. "
                f"Style={response_style}; prompt_digest={prompt_digest}."
            )
        else:
            output_text = (
                f"{summary_prefix} failed `{task_title}` before the shared workflow completed. "
                f"Style={response_style}; prompt_digest={prompt_digest}."
            )
        completed_at = timestamp_utc()
        return {
            "schema_name": BACKEND_INVOCATION_SCHEMA_NAME,
            "schema_version": BACKEND_INVOCATION_SCHEMA_VERSION,
            "invocation_id": f"{payload['backend_id']}:invoke:{timestamp_slug()}:{uuid4().hex[:8]}",
            "backend_id": payload["backend_id"],
            "adapter_kind": payload["adapter_kind"],
            "model_id": payload["model_id"],
            "status": status,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "prompt_digest": prompt_digest,
            "prompt_excerpt": prompt_text[:500],
            "output_text": output_text,
            "metadata": {
                "response_style": response_style,
                "mock": True,
            },
        }


class LocalBackendAdapter(MockBackendAdapter):
    def generate_text(
        self,
        prompt: str,
        *,
        config: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        invocation = super().generate_text(prompt, config=config, context=context)
        metadata = _mapping_dict(invocation.get("metadata"))
        metadata["mock"] = False
        metadata["local_dry_run"] = True
        invocation["metadata"] = metadata
        return invocation


def adapter_for_config(config: Mapping[str, Any]) -> BackendAdapter:
    payload = validate_backend_config(config)
    if payload["adapter_kind"] == "mock":
        return MockBackendAdapter()
    if payload["adapter_kind"] == "local":
        return LocalBackendAdapter()
    raise ValueError(f"Unsupported backend adapter_kind `{payload['adapter_kind']}`.")


def _backend_reference(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_backend_config(config)
    return {
        "backend_id": payload["backend_id"],
        "display_name": payload["display_name"],
        "adapter_kind": payload["adapter_kind"],
        "model_id": payload["model_id"],
        "capabilities": copy.deepcopy(payload.get("capabilities") or {}),
        "limits": copy.deepcopy(payload.get("limits") or {}),
        "metadata": copy.deepcopy(payload.get("metadata") or {}),
        "tags": list(payload.get("tags") or []),
    }


def _backend_invocation_trace(invocation: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_backend_config(config)
    status = _clean_text(invocation.get("status")) or "unknown"
    return {
        "trace_id": _clean_text(invocation.get("invocation_id")) or f"{payload['backend_id']}:invoke",
        "tool_kind": "backend_invocation",
        "status": "passed" if status == "ok" else "failed",
        "backend_id": payload["backend_id"],
        "adapter_kind": payload["adapter_kind"],
        "model_id": payload["model_id"],
        "started_at_utc": _clean_text(invocation.get("started_at_utc")),
        "completed_at_utc": _clean_text(invocation.get("completed_at_utc")),
        "prompt_digest": _clean_text(invocation.get("prompt_digest")),
        "output_excerpt": (_clean_text(invocation.get("output_text")) or "")[:500],
    }


def _backend_invocation_failure_summary(invocation: Mapping[str, Any]) -> str | None:
    status = (_clean_text(invocation.get("status")) or "").lower()
    if status == "ok":
        return None
    output_text = _clean_text(invocation.get("output_text"))
    if output_text is not None:
        return output_text
    return f"Backend invocation failed with status `{status or 'unknown'}`."


def _apply_backend_invocation_outcome(run: dict[str, Any], invocation: Mapping[str, Any]) -> None:
    failure_summary = _backend_invocation_failure_summary(invocation)
    if failure_summary is None:
        return

    outcome = _mapping_dict(run.get("outcome"))
    previous_run_status = _clean_text(run.get("status"))
    previous_outcome_status = _clean_text(outcome.get("status"))
    previous_failure_summary = _clean_text(outcome.get("failure_summary"))
    previous_result_summary = _clean_text(outcome.get("result_summary"))
    run["status"] = "failed"
    quality_checks = [
        dict(item)
        for item in outcome.get("quality_checks") or []
        if isinstance(item, Mapping)
    ]
    quality_checks.append(
        {
            "name": "backend_invocation",
            "pass": False,
            "detail": failure_summary,
        }
    )
    outcome.update(
        {
            "status": "failed",
            "quality_status": "fail",
            "execution_status": "failed",
            "backend_invocation_failure_summary": failure_summary,
            "quality_checks": quality_checks,
        }
    )
    if previous_outcome_status in {"succeeded", "passed", "pass", "ok"} or previous_run_status == "succeeded":
        outcome["failure_summary"] = failure_summary
        outcome["result_summary"] = failure_summary
    else:
        outcome["failure_summary"] = previous_failure_summary or failure_summary
        outcome["result_summary"] = previous_result_summary or failure_summary
    run["outcome"] = outcome
    existing_tags = [
        tag
        for tag in _string_list(run.get("tags"))
        if tag not in {"succeeded", "failed", "blocked"}
    ]
    run["tags"] = [*existing_tags, "failed"]


def _comparison_decision(results: list[dict[str, Any]]) -> tuple[str, str | None, str]:
    successful = [
        result
        for result in results
        if result.get("run_status") == "succeeded"
        and result.get("compatibility", {}).get("status") == "compatible"
    ]
    if len(successful) == 1:
        winner = _clean_text(successful[0].get("event_id"))
        return "winner_selected", winner, "Only one backend completed the workflow successfully."
    if len(successful) == len(results) and len(successful) >= 2:
        return "tie", None, "All selected backends completed the same workflow successfully."
    return "needs_follow_up", None, "One or more selected backends needs follow-up before adoption."


def _load_selected_backend_configs(
    *,
    root: Path,
    workspace_id: str,
    backend_ids: Iterable[str] | None,
) -> list[dict[str, Any]]:
    configs = ensure_default_backend_configs(root=root, workspace_id=workspace_id)
    by_id = {str(config["backend_id"]): config for config in configs}
    selected_ids = _string_list(backend_ids)
    if not selected_ids:
        selected_ids = ["mock-fast-local", "mock-careful-local"]
    duplicate_ids = sorted({backend_id for backend_id in selected_ids if selected_ids.count(backend_id) > 1})
    if duplicate_ids:
        raise ValueError("Backend swap side-by-side runs require distinct backend ids: " + ", ".join(duplicate_ids))
    missing = [backend_id for backend_id in selected_ids if backend_id not in by_id]
    if missing:
        raise ValueError("Unknown backend_id for side-by-side run: " + ", ".join(missing))
    selected = [by_id[backend_id] for backend_id in selected_ids]
    if len(selected) < 2:
        raise ValueError("Backend swap side-by-side runs require at least two backend configs.")
    return selected


def _validate_backend_harness_run(run: Mapping[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    payload = copy.deepcopy(dict(run))
    location = f" in `{path}`" if path is not None else ""
    if payload.get("schema_name") != BACKEND_HARNESS_RUN_SCHEMA_NAME:
        raise ValueError(f"Unexpected backend harness run schema name{location}.")
    if payload.get("schema_version") != BACKEND_HARNESS_RUN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported backend harness run schema version{location}.")
    run_id = _clean_text(payload.get("run_id"))
    if run_id is None:
        raise ValueError(f"Backend harness run is missing run_id{location}.")
    workspace_id = _clean_text(payload.get("workspace_id"))
    if workspace_id is None:
        raise ValueError(f"Backend harness run is missing workspace_id{location}.")
    workflow_kind = _clean_text(payload.get("workflow_kind"))
    if workflow_kind is None:
        raise ValueError(f"Backend harness run is missing workflow_kind{location}.")
    if not isinstance(payload.get("backend_results"), list):
        raise ValueError(f"Backend harness run is missing backend_results{location}.")
    if not all(isinstance(item, Mapping) for item in payload["backend_results"]):
        raise ValueError(f"Backend harness run backend_results must contain objects{location}.")
    comparison = _mapping_dict(payload.get("comparison"))
    candidate_event_ids = _string_list(comparison.get("candidate_event_ids"))
    if len(candidate_event_ids) < 2:
        raise ValueError(f"Backend harness run comparison requires at least two candidate_event_ids{location}.")
    payload["run_id"] = run_id
    payload["workspace_id"] = workspace_id
    payload["workflow_kind"] = workflow_kind
    payload["comparison"] = comparison
    payload["paths"] = _mapping_dict(payload.get("paths"))
    return payload


def record_backend_harness_run(
    run: Mapping[str, Any],
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
) -> tuple[dict[str, Any], Path]:
    payload = _validate_backend_harness_run(run)
    run_id = str(payload["run_id"])
    if payload.get("workspace_id") != workspace_id:
        raise ValueError(f"Backend harness run belongs to workspace `{payload.get('workspace_id')}`.")
    run_path = backend_harness_run_artifact_path(workspace_id=workspace_id, root=root, run_id=run_id)
    if run_path.exists():
        raise ValueError(f"Backend harness run artifact `{run_path}` already exists.")
    payload["paths"]["harness_run_artifact_path"] = str(run_path)
    write_json(run_path, payload)
    _append_jsonl(
        backend_harness_run_log_path(workspace_id=workspace_id, root=root),
        header=_harness_run_log_header(workspace_id=workspace_id),
        payload=payload,
        workspace_id=workspace_id,
    )
    return payload, run_path


def read_backend_harness_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    header = _read_log_header(
        path,
        schema_name=BACKEND_HARNESS_RUN_LOG_SCHEMA_NAME,
        schema_version=BACKEND_HARNESS_RUN_LOG_SCHEMA_VERSION,
    )
    workspace_id = _clean_text(header.get("workspace_id"))
    runs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            cleaned = line.strip()
            if not cleaned:
                continue
            run = _validate_backend_harness_run(json.loads(cleaned), path=path)
            if workspace_id is not None and run.get("workspace_id") != workspace_id:
                raise ValueError(f"Backend harness run log `{path}` contains workspace `{run.get('workspace_id')}`.")
            runs.append(run)
    return runs


def run_backend_swap_harness(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    task_title: str,
    goal: str,
    task_kind: str = "patch_plan_verify",
    scope_paths: Iterable[str] | None = None,
    plan_steps: Iterable[str] | None = None,
    verification_commands: Iterable[str] | None = None,
    acceptance_criteria: Iterable[str] | None = None,
    pass_definition: str | None = None,
    backend_ids: Iterable[str] | None = None,
    workflow_kind: str = DEFAULT_WORKFLOW_KIND,
    comparison_label: str | None = None,
    timeout_seconds: int = 60,
) -> tuple[dict[str, Any], Path]:
    if task_kind not in TASK_KINDS:
        raise ValueError(f"Unsupported agent-lane task_kind `{task_kind}`.")
    if timeout_seconds < 1:
        raise ValueError("Backend swap timeout_seconds must be at least 1.")
    clean_task_title = _clean_text(task_title)
    clean_goal = _clean_text(goal)
    clean_scope_paths = _string_list(scope_paths)
    clean_plan_steps = _string_list(plan_steps)
    clean_verification_commands = _string_list(verification_commands)
    clean_acceptance_criteria = _string_list(acceptance_criteria)
    if clean_task_title is None:
        raise ValueError("Backend swap runs require a task_title.")
    if clean_goal is None:
        raise ValueError("Backend swap runs require a goal.")
    if not clean_plan_steps:
        raise ValueError("Backend swap runs require at least one plan step.")
    if not clean_verification_commands:
        raise ValueError("Backend swap runs require at least one verification command.")

    resolved_root = _resolve_root(root)
    selected_configs = _load_selected_backend_configs(
        root=resolved_root,
        workspace_id=workspace_id,
        backend_ids=backend_ids,
    )
    compatibilities = [
        check_backend_compatibility(config, workflow_kind=workflow_kind)
        for config in selected_configs
    ]
    incompatible = [
        compatibility
        for compatibility in compatibilities
        if compatibility.get("status") != "compatible"
    ]
    if incompatible:
        details = [
            f"{item.get('backend_id')}: {', '.join(item.get('missing_capabilities') or [])}"
            for item in incompatible
        ]
        raise ValueError("Backend compatibility check failed: " + "; ".join(details))

    run_id = f"{workspace_id}:backend-swap:{timestamp_slug()}:{uuid4().hex[:8]}"
    backend_results: list[dict[str, Any]] = []

    for config, compatibility in zip(selected_configs, compatibilities, strict=True):
        backend_ref = _backend_reference(config)
        adapter = adapter_for_config(config)
        prompt = "\n".join(
            item
            for item in [
                f"Task: {clean_task_title}",
                f"Goal: {clean_goal}",
                "Plan:",
                *[f"- {step}" for step in clean_plan_steps],
            ]
            if item
        )
        invocation = adapter.generate_text(
            prompt,
            config=config,
            context={
                "task_title": clean_task_title,
                "goal": clean_goal,
                "workflow_kind": workflow_kind,
                "harness_run_id": run_id,
            },
        )
        backend_id = str(config["backend_id"])
        task = build_agent_task(
            workspace_id=workspace_id,
            title=f"{clean_task_title} [{backend_id}]",
            goal=clean_goal,
            task_kind=task_kind,
            origin="backend_swap_harness",
            scope_paths=clean_scope_paths,
            plan_steps=clean_plan_steps,
            verification_commands=clean_verification_commands,
            acceptance_criteria=clean_acceptance_criteria,
            pass_definition=pass_definition,
            tags=["m6", "backend_swap", backend_id, str(config["adapter_kind"])],
        )
        task["backend"] = backend_ref
        task["compatibility"] = copy.deepcopy(compatibility)
        task["workflow"] = {
            "workflow_kind": workflow_kind,
            "harness_run_id": run_id,
        }
        recorded_task = record_agent_task(task, root=resolved_root, workspace_id=workspace_id)
        run = run_agent_task(
            recorded_task,
            root=resolved_root,
            origin="backend_swap_harness",
            result_summary=None,
            timeout_seconds=timeout_seconds,
        )
        if run.get("status") == "succeeded" and _clean_text(invocation.get("status")) == "ok":
            run["outcome"]["result_summary"] = _clean_text(invocation.get("output_text"))
        run["backend"] = backend_ref
        run["compatibility"] = copy.deepcopy(compatibility)
        run["workflow"] = {
            "workflow_kind": workflow_kind,
            "harness_run_id": run_id,
        }
        run["tool_traces"] = [
            _backend_invocation_trace(invocation, config),
            *list(run.get("tool_traces") or []),
        ]
        _apply_backend_invocation_outcome(run, invocation)
        run["outcome"]["backend_id"] = backend_id
        run["outcome"]["model_id"] = str(config["model_id"])
        run["outcome"]["backend_output_text"] = _clean_text(invocation.get("output_text"))
        run["tags"] = _string_list([*list(run.get("tags") or []), "m6", "backend_swap", backend_id])
        recorded_run, run_path = record_agent_run(run, root=resolved_root, workspace_id=workspace_id)
        event = build_event_from_agent_run(
            root=resolved_root,
            workspace_id=workspace_id,
            run=recorded_run,
        )
        backend_results.append(
            {
                "backend_id": backend_id,
                "adapter_kind": str(config["adapter_kind"]),
                "model_id": str(config["model_id"]),
                "compatibility": compatibility,
                "invocation_id": _clean_text(invocation.get("invocation_id")),
                "task_id": recorded_task["task_id"],
                "run_id": recorded_run["run_id"],
                "event_id": event["event_id"],
                "run_status": recorded_run["status"],
                "quality_status": recorded_run["outcome"]["quality_status"],
                "execution_status": recorded_run["outcome"]["execution_status"],
                "run_artifact_path": str(run_path),
                "output_excerpt": (_clean_text(invocation.get("output_text")) or "")[:500],
            }
        )

    index_summary = rebuild_memory_index(root=resolved_root, workspace_id=workspace_id)
    event_log = read_event_log(Path(index_summary["event_log_path"]))
    events_by_id = {
        str(event.get("event_id")): event
        for event in event_log.get("events") or []
        if isinstance(event, Mapping) and event.get("event_id")
    }
    comparison_outcome, winner_event_id, comparison_rationale = _comparison_decision(backend_results)
    comparison = build_evaluation_comparison(
        workspace_id=workspace_id,
        candidate_event_ids=[str(result["event_id"]) for result in backend_results],
        winner_event_id=winner_event_id,
        outcome=comparison_outcome,
        task_label=_clean_text(comparison_label) or clean_task_title,
        criteria=[
            "same outer agent-lane workflow",
            "backend compatibility passed",
            "verification outcome",
        ],
        rationale=comparison_rationale,
        origin="backend_swap_harness",
        events_by_id=events_by_id,
        tags=["m6", "backend_swap"],
    )
    append_evaluation_comparison(
        evaluation_comparison_log_path(workspace_id=workspace_id, root=resolved_root),
        comparison,
        workspace_id=workspace_id,
    )
    evaluation_snapshot, evaluation_latest_path, evaluation_run_path = record_evaluation_snapshot(
        root=resolved_root,
        workspace_id=workspace_id,
        index_summary=index_summary,
    )
    agent_lane_snapshot, agent_lane_latest_path, agent_lane_run_path = record_agent_lane_snapshot(
        root=resolved_root,
        workspace_id=workspace_id,
    )

    if all(result.get("run_status") == "succeeded" for result in backend_results):
        status = "completed"
    else:
        status = "completed_with_failures"
    harness_run = {
        "schema_name": BACKEND_HARNESS_RUN_SCHEMA_NAME,
        "schema_version": BACKEND_HARNESS_RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "workspace_id": workspace_id,
        "workflow_kind": workflow_kind,
        "recorded_at_utc": timestamp_utc(),
        "status": status,
        "task": {
            "title": clean_task_title,
            "goal": clean_goal,
            "task_kind": task_kind,
            "scope_paths": clean_scope_paths,
            "plan_steps": clean_plan_steps,
            "verification_commands": clean_verification_commands,
            "acceptance_criteria": clean_acceptance_criteria,
            "pass_definition": _clean_text(pass_definition),
        },
        "backend_results": backend_results,
        "comparison": {
            "comparison_id": comparison["comparison_id"],
            "outcome": comparison["outcome"],
            "winner_event_id": comparison["winner_event_id"],
            "candidate_event_ids": [str(result["event_id"]) for result in backend_results],
        },
        "index_summary": index_summary,
        "evaluation_counts": copy.deepcopy(evaluation_snapshot.get("counts") or {}),
        "agent_lane_counts": copy.deepcopy(agent_lane_snapshot.get("counts") or {}),
        "paths": {
            "backend_config_log_path": str(backend_config_log_path(workspace_id=workspace_id, root=resolved_root)),
            "comparison_log_path": str(evaluation_comparison_log_path(workspace_id=workspace_id, root=resolved_root)),
            "evaluation_snapshot_latest_path": str(evaluation_latest_path),
            "evaluation_snapshot_run_path": str(evaluation_run_path),
            "agent_lane_snapshot_latest_path": str(agent_lane_latest_path),
            "agent_lane_snapshot_run_path": str(agent_lane_run_path),
        },
        "tags": ["m6", "backend_swap"],
    }
    return record_backend_harness_run(harness_run, root=resolved_root, workspace_id=workspace_id)


def format_backend_harness_report(run: Mapping[str, Any]) -> str:
    comparison = _mapping_dict(run.get("comparison"))
    lines = [
        "Backend swap harness",
        f"Workspace: {_clean_text(run.get('workspace_id')) or DEFAULT_WORKSPACE_ID}",
        f"Run: {_clean_text(run.get('run_id')) or 'n/a'}",
        f"Status: {_clean_text(run.get('status')) or 'n/a'}",
        f"Workflow: {_clean_text(run.get('workflow_kind')) or DEFAULT_WORKFLOW_KIND}",
        f"Comparison: {_clean_text(comparison.get('outcome')) or 'n/a'}",
    ]
    winner = _clean_text(comparison.get("winner_event_id"))
    if winner:
        lines.append(f"Winner event: {winner}")
    results = [item for item in run.get("backend_results") or [] if isinstance(item, Mapping)]
    if results:
        lines.extend(("", "Backends:"))
        for result in results:
            lines.append(
                "- "
                + f"{_clean_text(result.get('backend_id')) or 'backend'} "
                + f"model={_clean_text(result.get('model_id')) or 'n/a'} "
                + f"run={_clean_text(result.get('run_status')) or 'n/a'} "
                + f"event={_clean_text(result.get('event_id')) or 'n/a'}"
            )
    return "\n".join(lines)
