#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

from gemma_runtime import repo_root, timestamp_slug, timestamp_utc, write_json
from memory_index import MemoryIndex, default_memory_index_path, rebuild_memory_index
from prepare_recall_real_data import (
    BASELINE_REQUEST_VARIANT,
    MAX_ADVERSARIAL_REQUESTS,
    MAX_REQUESTS,
    build_real_recall_dataset,
    default_output_dir,
)
from recall_context import (
    CONTEXT_BUNDLE_VERSION,
    DEFAULT_CONTEXT_BUDGET_CHARS,
    DEFAULT_LIMIT,
    TASK_KINDS,
    build_context_bundle,
)
from workspace_state import DEFAULT_WORKSPACE_ID


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _truncate(text: str, *, limit: int) -> str:
    cleaned = _clean_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "..."


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _format_group_members(
    payload: Mapping[str, Any],
    *,
    count_key: str = "group_member_count",
    labels_key: str = "group_member_labels",
    limit: int = 3,
) -> str | None:
    count = _nonnegative_int(payload.get(count_key))
    if count <= 1:
        return None
    labels = [
        _truncate(str(item), limit=42)
        for item in (payload.get(labels_key) or [])
        if _clean_text(item)
    ][: min(limit, count)]
    if not labels:
        return str(count)
    more = count - len(labels)
    suffix = f"; +{more} more" if more > 0 else ""
    return f"{count}: {'; '.join(labels)}{suffix}"


def _load_index(
    *,
    root: Path,
    workspace_id: str,
    index_path: Path | None = None,
) -> tuple[MemoryIndex, Path]:
    target_path = Path(index_path or default_memory_index_path(workspace_id=workspace_id, root=root))
    if not target_path.exists():
        rebuild_memory_index(root=root, workspace_id=workspace_id, index_path=target_path)
    return MemoryIndex(target_path), target_path


def default_dataset_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return default_output_dir(workspace_id=workspace_id, root=_resolve_root(root)) / "real_recall_dataset.json"


def default_evaluation_dir(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return default_output_dir(workspace_id=workspace_id, root=_resolve_root(root)) / "evaluation"


def default_evaluation_latest_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return default_evaluation_dir(workspace_id=workspace_id, root=_resolve_root(root)) / "latest.json"


def read_recall_dataset(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected recall dataset payload in `{path}`.")
    return payload


def read_bundle(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected recall bundle payload in `{path}`.")
    return payload


def read_evaluation_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected recall evaluation payload in `{path}`.")
    return payload


def load_latest_evaluation_summary(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> tuple[dict[str, Any] | None, Path]:
    target_path = default_evaluation_latest_path(workspace_id=workspace_id, root=_resolve_root(root))
    if not target_path.exists():
        return None, target_path
    return read_evaluation_summary(target_path), target_path


def ensure_recall_dataset(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    dataset_path: Path | None = None,
    index_path: Path | None = None,
    refresh: bool = False,
    max_requests: int = MAX_REQUESTS,
    max_adversarial_requests: int = MAX_ADVERSARIAL_REQUESTS,
) -> tuple[dict[str, Any], Path]:
    resolved_root = _resolve_root(root)
    target_path = Path(dataset_path or default_dataset_path(workspace_id=workspace_id, root=resolved_root))
    if refresh or not target_path.exists():
        dataset = build_real_recall_dataset(
            root=resolved_root,
            workspace_id=workspace_id,
            output_dir=target_path.parent,
            index_path=index_path,
            max_requests=max_requests,
            max_adversarial_requests=max_adversarial_requests,
        )
        if target_path.name != "real_recall_dataset.json":
            write_json(target_path, dataset)
        return dataset, target_path
    return read_recall_dataset(target_path), target_path


def dataset_request_to_bundle_request(entry: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "task_kind": _clean_text(entry.get("task_kind")),
        "query_text": _clean_text(entry.get("query_text")),
        "request_basis": _clean_text(entry.get("request_basis")) or None,
        "file_hints": list(entry.get("file_hints") or []),
        "surface_filters": list(entry.get("surface_filters") or []),
        "status_filters": list(entry.get("status_filters") or []),
        "limit": int(entry.get("limit") or DEFAULT_LIMIT),
        "context_budget_chars": int(entry.get("context_budget_chars") or DEFAULT_CONTEXT_BUDGET_CHARS),
    }
    source_event_id = _clean_text(entry.get("source_event_id"))
    if source_event_id:
        payload["source_event_id"] = source_event_id
    return payload


def request_catalog(dataset: Mapping[str, Any]) -> list[dict[str, Any]]:
    requests = dataset.get("requests") or []
    catalog: list[dict[str, Any]] = []
    for index, entry in enumerate(requests, start=1):
        if not isinstance(entry, Mapping):
            continue
        catalog.append(
            {
                "index": index,
                "task_kind": _clean_text(entry.get("task_kind")),
                "query_text": _clean_text(entry.get("query_text")),
                "source_hit": bool(entry.get("source_hit")),
                "selected_count": int(entry.get("selected_count") or 0),
                "omitted_count": int(entry.get("omitted_count") or 0),
                "source_status": _clean_text(entry.get("source_status")) or "-",
                "source_rank": entry.get("source_rank"),
                "miss_reason": _clean_text(entry.get("miss_reason")) or None,
                "request_variant": _clean_text(entry.get("request_variant")) or BASELINE_REQUEST_VARIANT,
            }
        )
    return catalog


def format_request_catalog(dataset: Mapping[str, Any], *, dataset_path: Path | None = None) -> str:
    lines = [
        f"Workspace: {_clean_text(dataset.get('workspace_id')) or DEFAULT_WORKSPACE_ID}",
        f"Request count: {int(dataset.get('request_count') or 0)}",
    ]
    if dataset_path is not None:
        lines.append(f"Dataset: {dataset_path}")
    requests = request_catalog(dataset)
    if not requests:
        lines.append("")
        lines.append("No recall requests are available yet.")
        return "\n".join(lines)

    lines.extend(("", "Requests:"))
    for entry in requests:
        lines.append(
            f"{entry['index']:>2}. {entry['task_kind']:<16} hit={'yes' if entry['source_hit'] else 'no '} "
            f"selected={entry['selected_count']} omitted={entry['omitted_count']} status={entry['source_status']}"
        )
        if entry["request_variant"] != BASELINE_REQUEST_VARIANT:
            lines[-1] += f" variant={entry['request_variant']}"
        if not entry["source_hit"] and entry["miss_reason"]:
            lines[-1] += f" miss={entry['miss_reason']}"
        lines.append(f"    {_truncate(entry['query_text'], limit=120)}")
    return "\n".join(lines)


def select_dataset_request(
    dataset: Mapping[str, Any],
    *,
    request_index: int,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    requests = dataset.get("requests") or []
    if request_index < 1 or request_index > len(requests):
        raise ValueError(f"`--request-index` must be between 1 and {len(requests)}.")
    entry = requests[request_index - 1]
    if not isinstance(entry, Mapping):
        raise ValueError(f"Request {request_index} is not a valid mapping.")
    return dataset_request_to_bundle_request(entry), entry


def dataset_bundle_path(entry: Mapping[str, Any]) -> Path | None:
    bundle_path = _clean_text(entry.get("bundle_path"))
    if not bundle_path:
        return None
    return Path(bundle_path)


def bundle_source_evaluation(bundle: Mapping[str, Any]) -> dict[str, Any]:
    payload = bundle.get("source_evaluation")
    if not isinstance(payload, Mapping):
        return {}
    return dict(payload)


def bundle_has_source_evaluation(
    bundle: Mapping[str, Any],
    *,
    source_event_id: str | None = None,
) -> bool:
    payload = bundle_source_evaluation(bundle)
    if not payload:
        return False
    if source_event_id is None:
        return True
    return _clean_text(payload.get("source_event_id")) == _clean_text(source_event_id)


def bundle_needs_source_refresh(
    bundle: Mapping[str, Any],
    *,
    source_event_id: str | None,
    index: MemoryIndex,
) -> bool:
    try:
        bundle_version = int(bundle.get("bundle_version") or 0)
    except (TypeError, ValueError):
        bundle_version = 0
    if bundle_version < CONTEXT_BUNDLE_VERSION:
        return True
    if not bundle_has_source_evaluation(bundle, source_event_id=source_event_id):
        return True
    if not source_event_id:
        return False

    source_evaluation = bundle_source_evaluation(bundle)
    source_exists_now = index.get_event(source_event_id) is not None
    recorded_exists = source_evaluation.get("source_exists_in_index")
    if recorded_exists is not None and bool(recorded_exists) != source_exists_now:
        return True

    source_selected = bool(source_evaluation.get("source_selected"))
    miss_reason = _clean_text(source_evaluation.get("miss_reason"))
    if source_exists_now and not source_selected and miss_reason in {"not_retrieved", "source_missing_from_index"}:
        return True
    if not source_exists_now and source_selected:
        return True
    return False


def build_bundle_for_dataset_entry(
    entry: Mapping[str, Any],
    *,
    root: Path,
    workspace_id: str,
    index_path: Path | None = None,
) -> dict[str, Any]:
    request_payload = dataset_request_to_bundle_request(entry)
    source_event_id = _clean_text(request_payload.get("source_event_id")) or None
    index_handle, _resolved_index_path = _load_index(
        root=root,
        workspace_id=workspace_id,
        index_path=index_path,
    )
    bundle_path = dataset_bundle_path(entry)
    if bundle_path is not None and bundle_path.exists():
        bundle = read_bundle(bundle_path)
        if not bundle_needs_source_refresh(bundle, source_event_id=source_event_id, index=index_handle):
            return bundle
    bundle = build_context_bundle(
        request_payload,
        root=root,
        workspace_id=workspace_id,
        index=index_handle,
    )
    if bundle_path is not None:
        write_json(bundle_path, dict(bundle))
    return bundle


def evaluate_dataset(
    dataset: Mapping[str, Any],
    *,
    root: Path,
    workspace_id: str,
    index_path: Path | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    miss_reason_counts: Counter[str] = Counter()
    requests = dataset.get("requests") or []
    for index, entry in enumerate(requests, start=1):
        if not isinstance(entry, Mapping):
            continue
        bundle = build_bundle_for_dataset_entry(
            entry,
            root=root,
            workspace_id=workspace_id,
            index_path=index_path,
        )
        source_evaluation = bundle_source_evaluation(bundle)
        source_hit = bool(source_evaluation.get("source_selected"))
        miss_reason = _clean_text(source_evaluation.get("miss_reason")) or None
        if not source_hit and miss_reason:
            miss_reason_counts[miss_reason] += 1
        rows.append(
            {
                "index": index,
                "task_kind": _clean_text(entry.get("task_kind")),
                "query_text": _clean_text(entry.get("query_text")),
                "source_status": _clean_text(entry.get("source_status")) or "-",
                "source_event_id": _clean_text(entry.get("source_event_id")) or None,
                "request_variant": _clean_text(entry.get("request_variant")) or BASELINE_REQUEST_VARIANT,
                "source_hit": source_hit,
                "selected_count": int(bundle.get("selected_count") or 0),
                "omitted_count": int(bundle.get("omitted_count") or 0),
                "source_rank": source_evaluation.get("source_rank"),
                "source_score": source_evaluation.get("source_score"),
                "source_block_title": _clean_text(source_evaluation.get("source_block_title")) or None,
                "source_prompt_excerpt": _clean_text(source_evaluation.get("source_prompt_excerpt")) or None,
                "source_reasons": list(source_evaluation.get("source_reasons") or []),
                "source_selected_via_group": bool(source_evaluation.get("source_selected_via_group")),
                "source_grouped_by": _clean_text(source_evaluation.get("source_grouped_by")) or None,
                "source_group_event_id": _clean_text(source_evaluation.get("source_group_event_id")) or None,
                "source_group_member_count": source_evaluation.get("source_group_member_count"),
                "source_group_member_labels": list(source_evaluation.get("source_group_member_labels") or []),
                "miss_reason": miss_reason,
                "top_selected": list(source_evaluation.get("top_selected") or []),
            }
        )

    hit_count = sum(1 for row in rows if row["source_hit"])
    misses = [row for row in rows if not row["source_hit"]]
    variants: dict[str, dict[str, Any]] = {}
    for row in rows:
        variant = _clean_text(row.get("request_variant")) or BASELINE_REQUEST_VARIANT
        bucket = variants.setdefault(
            variant,
            {
                "request_count": 0,
                "source_hits": 0,
            },
        )
        bucket["request_count"] += 1
        if row["source_hit"]:
            bucket["source_hits"] += 1
    variant_summary = {
        variant: {
            "request_count": int(bucket["request_count"]),
            "source_hits": int(bucket["source_hits"]),
            "source_misses": int(bucket["request_count"] - bucket["source_hits"]),
            "hit_rate": round(bucket["source_hits"] / bucket["request_count"], 3) if bucket["request_count"] else 0.0,
        }
        for variant, bucket in sorted(variants.items())
    }
    return {
        "workspace_id": _clean_text(dataset.get("workspace_id")) or workspace_id,
        "request_count": len(rows),
        "source_hits": hit_count,
        "source_misses": len(misses),
        "hit_rate": round(hit_count / len(rows), 3) if rows else 0.0,
        "miss_reason_counts": dict(sorted(miss_reason_counts.items())),
        "variants": variant_summary,
        "requests": rows,
        "misses": misses,
    }


def sync_dataset_with_evaluation(
    dataset: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    requests = dataset.get("requests") or []
    rows_by_index = {
        int(row.get("index") or 0): row
        for row in (summary.get("requests") or [])
        if isinstance(row, Mapping)
    }
    synced_requests: list[dict[str, Any]] = []
    for index, entry in enumerate(requests, start=1):
        if not isinstance(entry, Mapping):
            continue
        row = rows_by_index.get(index)
        payload = dict(entry)
        if row is not None:
            payload.update(
                {
                    "source_hit": bool(row.get("source_hit")),
                    "selected_count": int(row.get("selected_count") or 0),
                    "omitted_count": int(row.get("omitted_count") or 0),
                    "source_rank": row.get("source_rank"),
                    "miss_reason": _clean_text(row.get("miss_reason")) or None,
                    "source_block_title": _clean_text(row.get("source_block_title")) or None,
                    "source_prompt_excerpt": _clean_text(row.get("source_prompt_excerpt")) or None,
                    "source_reasons": list(row.get("source_reasons") or []),
                    "source_selected_via_group": bool(row.get("source_selected_via_group")),
                    "source_grouped_by": _clean_text(row.get("source_grouped_by")) or None,
                    "source_group_event_id": _clean_text(row.get("source_group_event_id")) or None,
                    "source_group_member_count": row.get("source_group_member_count"),
                    "source_group_member_labels": list(row.get("source_group_member_labels") or []),
                    "top_selected": list(row.get("top_selected") or []),
                    "request_variant": _clean_text(row.get("request_variant")) or BASELINE_REQUEST_VARIANT,
                }
            )
        synced_requests.append(payload)
    synced_dataset = dict(dataset)
    synced_dataset["requests"] = synced_requests
    synced_dataset["request_count"] = len(synced_requests)
    synced_dataset["evaluated_at_utc"] = timestamp_utc()
    return synced_dataset


def compare_evaluation_summaries(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> dict[str, Any]:
    previous_summary = previous or {}
    current_variants = dict(current.get("variants") or {})
    previous_variants = dict(previous_summary.get("variants") or {})
    variant_names = sorted(set(current_variants) | set(previous_variants))
    variant_deltas: dict[str, dict[str, Any]] = {}
    for variant in variant_names:
        current_variant = dict(current_variants.get(variant) or {})
        previous_variant = dict(previous_variants.get(variant) or {})
        request_count_delta = int(current_variant.get("request_count") or 0) - int(previous_variant.get("request_count") or 0)
        source_hits_delta = int(current_variant.get("source_hits") or 0) - int(previous_variant.get("source_hits") or 0)
        source_misses_delta = int(current_variant.get("source_misses") or 0) - int(previous_variant.get("source_misses") or 0)
        hit_rate_delta = round(
            float(current_variant.get("hit_rate") or 0.0) - float(previous_variant.get("hit_rate") or 0.0),
            3,
        )
        if request_count_delta or source_hits_delta or source_misses_delta or hit_rate_delta:
            variant_deltas[variant] = {
                "request_count_delta": request_count_delta,
                "source_hits_delta": source_hits_delta,
                "source_misses_delta": source_misses_delta,
                "hit_rate_delta": hit_rate_delta,
            }

    current_miss_reason_counts = dict(current.get("miss_reason_counts") or {})
    previous_miss_reason_counts = dict(previous_summary.get("miss_reason_counts") or {})
    miss_reason_delta: dict[str, int] = {}
    for miss_reason in sorted(set(current_miss_reason_counts) | set(previous_miss_reason_counts)):
        count_delta = int(current_miss_reason_counts.get(miss_reason) or 0) - int(previous_miss_reason_counts.get(miss_reason) or 0)
        if count_delta:
            miss_reason_delta[miss_reason] = count_delta

    return {
        "request_count_delta": int(current.get("request_count") or 0) - int(previous_summary.get("request_count") or 0),
        "source_hits_delta": int(current.get("source_hits") or 0) - int(previous_summary.get("source_hits") or 0),
        "source_misses_delta": int(current.get("source_misses") or 0) - int(previous_summary.get("source_misses") or 0),
        "hit_rate_delta": round(
            float(current.get("hit_rate") or 0.0) - float(previous_summary.get("hit_rate") or 0.0),
            3,
        ),
        "variants": variant_deltas,
        "miss_reason_counts": miss_reason_delta,
    }


def record_evaluation_summary(
    summary: Mapping[str, Any],
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    dataset_path: Path | None = None,
    previous_summary: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    evaluation_dir = default_evaluation_dir(workspace_id=workspace_id, root=_resolve_root(root))
    latest_path = evaluation_dir / "latest.json"
    run_path = evaluation_dir / "runs" / f"{timestamp_slug()}-recall-eval.json"
    payload = dict(summary)
    payload["evaluated_at_utc"] = timestamp_utc()
    if dataset_path is not None:
        payload["dataset_path"] = str(dataset_path)
    if previous_summary:
        previous_evaluated_at_utc = _clean_text(previous_summary.get("evaluated_at_utc"))
        if previous_evaluated_at_utc:
            payload["previous_evaluated_at_utc"] = previous_evaluated_at_utc
        payload["delta_from_previous"] = compare_evaluation_summaries(previous_summary, payload)
    payload["evaluation_latest_path"] = str(latest_path)
    payload["evaluation_run_path"] = str(run_path)
    write_json(run_path, payload)
    write_json(latest_path, payload)
    return payload, latest_path, run_path


def format_evaluation_report(
    summary: Mapping[str, Any],
    *,
    dataset_path: Path | None = None,
) -> str:
    request_count = int(summary.get("request_count") or 0)
    source_hits = int(summary.get("source_hits") or 0)
    source_misses = int(summary.get("source_misses") or 0)
    lines = [
        f"Workspace: {_clean_text(summary.get('workspace_id')) or DEFAULT_WORKSPACE_ID}",
        f"Requests: {request_count}",
        f"Source hits: {source_hits}",
        f"Source misses: {source_misses}",
        f"Hit rate: {float(summary.get('hit_rate') or 0.0):.3f}",
    ]
    if dataset_path is not None:
        lines.append(f"Dataset: {dataset_path}")
    evaluated_at_utc = _clean_text(summary.get("evaluated_at_utc"))
    if evaluated_at_utc:
        lines.append(f"Evaluated: {evaluated_at_utc}")

    miss_reason_counts = dict(summary.get("miss_reason_counts") or {})
    if miss_reason_counts:
        lines.extend(("", "Miss reasons:"))
        for miss_reason, count in miss_reason_counts.items():
            lines.append(f"- {miss_reason}: {count}")
    variants = dict(summary.get("variants") or {})
    if variants:
        lines.extend(("", "By variant:"))
        for variant, payload in variants.items():
            if not isinstance(payload, Mapping):
                continue
            lines.append(
                f"- {variant}: "
                f"{int(payload.get('source_hits') or 0)} / {int(payload.get('request_count') or 0)} "
                f"(hit rate {float(payload.get('hit_rate') or 0.0):.3f})"
            )
    delta = dict(summary.get("delta_from_previous") or {})
    if delta:
        lines.extend(("", "Change vs previous:"))
        lines.append(f"- source_hits: {int(delta.get('source_hits_delta') or 0):+d}")
        lines.append(f"- source_misses: {int(delta.get('source_misses_delta') or 0):+d}")
        lines.append(f"- hit_rate: {float(delta.get('hit_rate_delta') or 0.0):+.3f}")
        variant_delta = dict(delta.get("variants") or {})
        for variant, payload in sorted(variant_delta.items()):
            if not isinstance(payload, Mapping):
                continue
            lines.append(
                f"- {variant} delta: "
                f"hits {int(payload.get('source_hits_delta') or 0):+d}, "
                f"misses {int(payload.get('source_misses_delta') or 0):+d}, "
                f"rate {float(payload.get('hit_rate_delta') or 0.0):+.3f}"
            )
    return "\n".join(lines)


def format_miss_report(
    summary: Mapping[str, Any],
    *,
    dataset_path: Path | None = None,
) -> str:
    misses = [item for item in (summary.get("misses") or []) if isinstance(item, Mapping)]
    lines = [
        f"Workspace: {_clean_text(summary.get('workspace_id')) or DEFAULT_WORKSPACE_ID}",
        f"Misses: {len(misses)} / {int(summary.get('request_count') or 0)}",
    ]
    if dataset_path is not None:
        lines.append(f"Dataset: {dataset_path}")
    if not misses:
        lines.extend(("", "No source misses."))
        return "\n".join(lines)

    lines.extend(("", "Source misses:"))
    for row in misses:
        lines.append(
            f"{int(row.get('index') or 0):>2}. {_clean_text(row.get('task_kind')):<16} "
            f"reason={_clean_text(row.get('miss_reason')) or 'unknown'} "
            f"rank={row.get('source_rank') if row.get('source_rank') is not None else '-'} "
            f"status={_clean_text(row.get('source_status')) or '-'}"
        )
        variant = _clean_text(row.get("request_variant")) or BASELINE_REQUEST_VARIANT
        if variant != BASELINE_REQUEST_VARIANT:
            lines[-1] += f" variant={variant}"
        lines.append(f"    {_truncate(_clean_text(row.get('query_text')), limit=120)}")
        source_prompt_excerpt = _clean_text(row.get("source_prompt_excerpt"))
        if source_prompt_excerpt:
            lines.append(f"    source: {_truncate(source_prompt_excerpt, limit=120)}")
    return "\n".join(lines)


def build_bundle_report(
    bundle: Mapping[str, Any],
    *,
    request_label: str | None = None,
) -> str:
    budget = dict(bundle.get("budget") or {})
    source_evaluation = dict(bundle.get("source_evaluation") or {})
    pinned_event_ids = list(bundle.get("pinned_event_ids") or [])
    lines = []
    if request_label:
        lines.append(f"Request: {request_label}")
    lines.extend(
        [
            f"Task: {_clean_text(bundle.get('task_kind'))}",
            f"Query: {_clean_text(bundle.get('query_text'))}",
            (
                "Selected: "
                f"{int(bundle.get('selected_count') or 0)} "
                f"(omitted {int(bundle.get('omitted_count') or 0)})"
            ),
            (
                "Budget: "
                f"{int(budget.get('used_chars') or 0)} / "
                f"{int(budget.get('context_budget_chars') or 0)} chars"
            ),
        ]
    )
    if source_evaluation:
        lines.append(
            "Source hit: "
            f"{'yes' if source_evaluation.get('source_selected') else 'no'}"
        )
        if source_evaluation.get("source_rank") is not None:
            lines.append(f"Source rank: {int(source_evaluation.get('source_rank') or 0)}")
        if source_evaluation.get("miss_reason"):
            lines.append(f"Miss reason: {_clean_text(source_evaluation.get('miss_reason'))}")
        source_group = _format_group_members(
            source_evaluation,
            count_key="source_group_member_count",
            labels_key="source_group_member_labels",
        )
        if source_group:
            grouped_by = _clean_text(source_evaluation.get("source_grouped_by")) or "group"
            group_event_id = _clean_text(source_evaluation.get("source_group_event_id"))
            prefix = f"Source group: {grouped_by}"
            if group_event_id:
                prefix += f" representative={group_event_id}"
            lines.append(f"{prefix} members={source_group}")
    if pinned_event_ids:
        lines.append(f"Pins: {len(pinned_event_ids)}")

    selected = bundle.get("selected_candidates") or []
    if selected:
        lines.extend(("", "Selected candidates:"))
        for index, item in enumerate(selected, start=1):
            if not isinstance(item, Mapping):
                continue
            lines.append(
                f"{index}. [{_clean_text(item.get('block_title')) or '-'}] "
                f"status={_clean_text(item.get('status')) or '-'} score={item.get('score')}"
            )
            group_member_count = _nonnegative_int(item.get("group_member_count"))
            if group_member_count > 1:
                lines[-1] += f" group={group_member_count}"
                group_members = _format_group_members(item)
                if group_members:
                    lines.append(f"   group members: {group_members}")
            reasons = ", ".join(str(reason) for reason in (item.get("reasons") or []))
            if reasons:
                lines.append(f"   reasons: {reasons}")
            prompt_excerpt = _clean_text(item.get("prompt_excerpt"))
            if prompt_excerpt:
                lines.append(f"   prompt: {prompt_excerpt}")

    blocks = bundle.get("blocks") or []
    if blocks:
        lines.extend(("", "Blocks:"))
        for block in blocks:
            if not isinstance(block, Mapping):
                continue
            title = _clean_text(block.get("title")) or "Untitled"
            items = block.get("items") or []
            lines.append(f"- {title} ({len(items)})")
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                summary = _truncate(_clean_text(item.get("summary")), limit=180)
                status = _clean_text(item.get("status")) or "-"
                score = item.get("score")
                lines.append(f"  * status={status} score={score}: {summary}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a thin recall workflow over the local memory index and prepared real-data requests.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional repo root override.",
    )
    parser.add_argument(
        "--workspace-id",
        default=DEFAULT_WORKSPACE_ID,
        help="Workspace id to read.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional recall dataset JSON path. Defaults to artifacts/recall_data/<workspace>/real_recall_dataset.json.",
    )
    parser.add_argument(
        "--prepare-real-data",
        action="store_true",
        help="Refresh the real-data recall dataset before listing or selecting a request.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=MAX_REQUESTS,
        help="Maximum request count when refreshing the real-data dataset.",
    )
    parser.add_argument(
        "--max-adversarial-requests",
        type=int,
        default=MAX_ADVERSARIAL_REQUESTS,
        help="Maximum adversarial request count appended when refreshing the real-data dataset.",
    )
    parser.add_argument(
        "--list-requests",
        action="store_true",
        help="List the prepared real-data recall requests and exit.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate source-hit coverage across the prepared real-data dataset.",
    )
    parser.add_argument(
        "--miss-report",
        action="store_true",
        help="List prepared requests where the source event was not recovered.",
    )
    parser.add_argument(
        "--request-index",
        type=int,
        default=None,
        help="1-based request index from the prepared real-data dataset.",
    )
    parser.add_argument(
        "--task-kind",
        choices=TASK_KINDS,
        default=None,
        help="Manual recall task kind.",
    )
    parser.add_argument(
        "--query",
        default="",
        help="Manual recall query text.",
    )
    parser.add_argument(
        "--file-hint",
        action="append",
        default=[],
        help="Optional manual file hint. Repeat to add more than one.",
    )
    parser.add_argument(
        "--surface-filter",
        action="append",
        default=[],
        help="Optional manual surface filter. Repeat to add more than one.",
    )
    parser.add_argument(
        "--status-filter",
        action="append",
        default=[],
        help="Optional manual status filter. Repeat to add more than one.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Manual maximum number of selected candidates.",
    )
    parser.add_argument(
        "--context-budget-chars",
        type=int,
        default=DEFAULT_CONTEXT_BUDGET_CHARS,
        help="Manual character budget for grouped context.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="Optional explicit SQLite index path.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for the built bundle.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output path for the built bundle JSON.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    resolved_root = _resolve_root(args.root)
    effective_index_path = args.index_path
    if effective_index_path is None and (args.request_index is not None or args.eval or args.miss_report):
        index_summary = rebuild_memory_index(
            root=resolved_root,
            workspace_id=args.workspace_id,
        )
        effective_index_path = Path(index_summary["index_path"])
    dataset: dict[str, Any] | None = None
    dataset_path = Path(args.dataset_path) if args.dataset_path is not None else default_dataset_path(
        workspace_id=args.workspace_id,
        root=resolved_root,
    )

    need_dataset = args.list_requests or args.request_index is not None or args.eval or args.miss_report
    if args.prepare_real_data or need_dataset:
        dataset, dataset_path = ensure_recall_dataset(
            root=resolved_root,
            workspace_id=args.workspace_id,
            dataset_path=dataset_path,
            index_path=effective_index_path,
            refresh=args.prepare_real_data,
            max_requests=args.max_requests,
            max_adversarial_requests=args.max_adversarial_requests,
        )

    if args.list_requests:
        print(format_request_catalog(dataset or {}, dataset_path=dataset_path))
        return 0

    if args.eval or args.miss_report:
        previous_summary, _latest_evaluation_path = load_latest_evaluation_summary(
            workspace_id=args.workspace_id,
            root=resolved_root,
        )
        summary = evaluate_dataset(
            dataset or {},
            root=resolved_root,
            workspace_id=args.workspace_id,
            index_path=effective_index_path,
        )
        synced_dataset = sync_dataset_with_evaluation(dataset or {}, summary)
        if dataset_path is not None:
            write_json(dataset_path, synced_dataset)
        summary, _latest_evaluation_path, _run_evaluation_path = record_evaluation_summary(
            summary,
            workspace_id=args.workspace_id,
            root=resolved_root,
            dataset_path=dataset_path,
            previous_summary=previous_summary,
        )
        if args.format == "json":
            payload: dict[str, Any]
            if args.eval and not args.miss_report:
                payload = dict(summary)
            elif args.miss_report and not args.eval:
                payload = {
                    "workspace_id": summary.get("workspace_id"),
                    "request_count": summary.get("request_count"),
                    "source_misses": summary.get("source_misses"),
                    "miss_reason_counts": summary.get("miss_reason_counts"),
                    "misses": summary.get("misses"),
                    "evaluated_at_utc": summary.get("evaluated_at_utc"),
                    "evaluation_latest_path": summary.get("evaluation_latest_path"),
                    "evaluation_run_path": summary.get("evaluation_run_path"),
                }
            else:
                payload = dict(summary)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            blocks: list[str] = []
            if args.eval:
                blocks.append(format_evaluation_report(summary, dataset_path=dataset_path))
            if args.miss_report:
                blocks.append(format_miss_report(summary, dataset_path=dataset_path))
            print("\n\n".join(block for block in blocks if block))
        return 0

    request_label: str | None = None
    if args.request_index is not None:
        if dataset is None:
            dataset, dataset_path = ensure_recall_dataset(
                root=resolved_root,
                workspace_id=args.workspace_id,
                dataset_path=dataset_path,
                index_path=effective_index_path,
                refresh=False,
                max_requests=args.max_requests,
                max_adversarial_requests=args.max_adversarial_requests,
            )
        try:
            request_payload, entry = select_dataset_request(dataset, request_index=args.request_index)
        except ValueError as exc:
            parser.error(str(exc))
        request_label = (
            f"dataset[{args.request_index}] "
            f"{_clean_text(entry.get('task_kind'))} "
            f"hit={'yes' if entry.get('source_hit') else 'no'}"
        )
        request_variant = _clean_text(entry.get("request_variant")) or BASELINE_REQUEST_VARIANT
        if request_variant != BASELINE_REQUEST_VARIANT:
            request_label += f" variant={request_variant}"
    else:
        if args.task_kind is None:
            parser.error("manual mode requires `--task-kind` when `--request-index` is omitted.")
        if not _clean_text(args.query):
            parser.error("manual mode requires a non-empty `--query`.")
        request_payload = {
            "task_kind": args.task_kind,
            "query_text": args.query,
            "file_hints": args.file_hint,
            "surface_filters": args.surface_filter,
            "status_filters": args.status_filter,
            "limit": args.limit,
            "context_budget_chars": args.context_budget_chars,
        }

    bundle_path = dataset_bundle_path(entry) if args.request_index is not None else None
    if args.request_index is not None:
        bundle = build_bundle_for_dataset_entry(
            entry,
            root=resolved_root,
            workspace_id=args.workspace_id,
            index_path=effective_index_path,
        )
    elif bundle_path is not None and effective_index_path is None and bundle_path.exists():
        bundle = read_bundle(bundle_path)
    else:
        bundle = build_context_bundle(
            request_payload,
            root=resolved_root,
            workspace_id=args.workspace_id,
            index_path=effective_index_path,
        )
    if args.out is not None:
        write_json(args.out, dict(bundle))

    if args.format == "json":
        print(json.dumps(bundle, ensure_ascii=False, indent=2))
    else:
        print(build_bundle_report(bundle, request_label=request_label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
