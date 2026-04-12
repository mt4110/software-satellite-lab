#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from blocker_taxonomy import build_blocker_record
from gemma_runtime import timestamp_utc, write_json


ARTIFACT_SCHEMA_NAME = "gemma-lab-artifact"
ARTIFACT_SCHEMA_VERSION = 1


def normalize_device_info(
    device_info: dict[str, Any] | str | None,
    *,
    dtype_name: str | None = None,
) -> dict[str, Any]:
    if isinstance(device_info, dict):
        name = device_info.get("name")
        label = device_info.get("label") or name
        dtype_value = device_info.get("dtype_name") or device_info.get("dtype") or dtype_name
        return {
            "name": str(name) if name is not None else None,
            "label": str(label) if label is not None else None,
            "dtype": str(dtype_value) if dtype_value is not None else None,
        }

    label = str(device_info).strip() if device_info not in (None, "") else None
    return {
        "name": label,
        "label": label,
        "dtype": dtype_name,
    }


def build_runtime_record(
    *,
    backend: str,
    model_id: str | None = None,
    device_info: dict[str, Any] | str | None = None,
    elapsed_seconds: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = {
        "backend": backend,
        "model_id": model_id,
        "device": normalize_device_info(device_info),
        "elapsed_seconds": round(float(elapsed_seconds), 3) if elapsed_seconds is not None else None,
    }
    if extra:
        runtime.update(copy.deepcopy(extra))
    return runtime


def build_prompt_record(
    *,
    system_prompt: str | None = None,
    prompt: str | None = None,
    resolved_user_prompt: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompts = {
        "system_prompt": system_prompt,
        "prompt": prompt,
        "resolved_user_prompt": resolved_user_prompt,
    }
    if extra:
        prompts.update(copy.deepcopy(extra))
    return prompts


def collect_asset_lineage(*records: Any) -> list[dict[str, Any]]:
    lineage: list[dict[str, Any]] = []
    for record in records:
        if record is None:
            continue
        if isinstance(record, list):
            lineage.extend(collect_asset_lineage(*record))
            continue
        if not isinstance(record, dict):
            continue
        record_lineage = record.get("lineage")
        if isinstance(record_lineage, list):
            lineage.extend(copy.deepcopy(record_lineage))
        elif isinstance(record_lineage, dict):
            lineage.append(copy.deepcopy(record_lineage))
    return lineage


def build_artifact_payload(
    *,
    artifact_kind: str,
    status: str,
    runtime: dict[str, Any],
    prompts: dict[str, Any] | None = None,
    asset_lineage: list[dict[str, Any]] | None = None,
    blocker_message: str | None = None,
    timestamp: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_name": ARTIFACT_SCHEMA_NAME,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "timestamp_utc": timestamp or timestamp_utc(),
        "status": status,
        "runtime": copy.deepcopy(runtime),
        "prompts": copy.deepcopy(prompts or build_prompt_record()),
        "assets": {
            "lineage": copy.deepcopy(asset_lineage or []),
        },
        "blocker": build_blocker_record(blocker_message) if blocker_message else None,
    }
    if extra:
        payload.update(copy.deepcopy(extra))
    return payload


def write_artifact(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def read_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_name") != ARTIFACT_SCHEMA_NAME:
        raise ValueError(f"Unexpected artifact schema name in `{path}`.")
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported artifact schema version in `{path}`.")
    return payload
