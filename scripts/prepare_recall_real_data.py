#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from gemma_runtime import repo_root, timestamp_utc, write_json
from memory_index import rebuild_memory_index
from recall_context import build_context_bundle
from software_work_events import iter_capability_matrix_events, iter_workspace_events
from workspace_state import DEFAULT_WORKSPACE_ID


FAILURE_STATUSES = {"quality_fail", "failed", "blocked", "error"}
MAX_REQUESTS = 16
MAX_ADVERSARIAL_REQUESTS = 8
BASELINE_REQUEST_VARIANT = "baseline"
ADVERSARIAL_PASS_DEFINITION_VARIANT = "adversarial-pass-definition"
ADVERSARIAL_LIMIT = 4
ADVERSARIAL_CONTEXT_BUDGET_CHARS = 3000


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def recall_data_root(root: Path | None = None) -> Path:
    return _resolve_root(root) / "artifacts" / "recall_data"


def default_output_dir(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return recall_data_root(root) / workspace_id


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _request_task_kind(event: dict[str, Any]) -> str:
    status = _clean_text(dict(event.get("outcome") or {}).get("status")).lower()
    prompt = _clean_text(dict(event.get("content") or {}).get("prompt")).lower()
    notes_text = "\n".join(
        item
        for item in (dict(event.get("content") or {}).get("notes") or [])
        if isinstance(item, str)
    ).lower()
    joined = "\n".join(part for part in (prompt, notes_text) if part)
    if status in FAILURE_STATUSES:
        return "failure_analysis"
    if any(keyword in joined for keyword in ("review", "regression", "risk", "comment")):
        return "review"
    if any(keyword in joined for keyword in ("design", "architecture", "tradeoff", "boundary", "decision")):
        return "design"
    return "proposal"


def _artifact_hint(event: dict[str, Any]) -> str | None:
    artifact_ref = dict(dict(event.get("source_refs") or {}).get("artifact_ref") or {})
    return _clean_text(artifact_ref.get("artifact_workspace_relative_path")) or _clean_text(
        artifact_ref.get("artifact_path")
    ) or None


def _query_text(event: dict[str, Any]) -> str | None:
    content = dict(event.get("content") or {})
    prompt = _clean_text(content.get("prompt"))
    if prompt:
        return prompt
    output_text = _clean_text(content.get("output_text"))
    if output_text:
        return output_text
    artifact_hint = _artifact_hint(event)
    if artifact_hint:
        return Path(artifact_hint).stem.replace("-", " ")
    return None


def _event_options(event: dict[str, Any]) -> dict[str, Any]:
    return dict(dict(event.get("content") or {}).get("options") or {})


def _request_payload(event: dict[str, Any], *, query_text: str, file_hints: list[str], limit: int, context_budget_chars: int) -> dict[str, Any]:
    return {
        "task_kind": _request_task_kind(event),
        "query_text": query_text,
        "file_hints": list(file_hints),
        "limit": limit,
        "context_budget_chars": context_budget_chars,
        "source_event_id": event.get("event_id"),
        "source_status": dict(event.get("outcome") or {}).get("status"),
        "source_event_kind": event.get("event_kind"),
    }


def _baseline_request(event: dict[str, Any]) -> dict[str, Any] | None:
    query_text = _query_text(event)
    if not query_text:
        return None
    artifact_hint = _artifact_hint(event)
    request = _request_payload(
        event,
        query_text=query_text,
        file_hints=[artifact_hint] if artifact_hint else [],
        limit=8,
        context_budget_chars=5000,
    )
    request["request_variant"] = BASELINE_REQUEST_VARIANT
    request["request_basis"] = "prompt-or-artifact"
    return request


def _adversarial_pass_definition_request(event: dict[str, Any]) -> dict[str, Any] | None:
    if _clean_text(event.get("event_kind")) != "capability_result":
        return None
    pass_definition = _clean_text(_event_options(event).get("pass_definition"))
    if len(pass_definition) < 24:
        return None
    request = _request_payload(
        event,
        query_text=pass_definition,
        file_hints=[],
        limit=ADVERSARIAL_LIMIT,
        context_budget_chars=ADVERSARIAL_CONTEXT_BUDGET_CHARS,
    )
    request["request_variant"] = ADVERSARIAL_PASS_DEFINITION_VARIANT
    request["request_basis"] = "pass_definition"
    request["adversarial"] = True
    return request


def _bundle_slug(request: dict[str, Any]) -> str:
    parts = [_clean_text(request.get("task_kind")) or "request"]
    variant = _clean_text(request.get("request_variant"))
    if variant and variant != BASELINE_REQUEST_VARIANT:
        parts.append(variant)
    slug = "-".join(part.lower() for part in parts if part)
    return re.sub(r"[^a-z0-9-]+", "-", slug).strip("-") or "request"


def build_real_recall_dataset(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    output_dir: Path | None = None,
    index_path: Path | None = None,
    event_log_path: Path | None = None,
    max_requests: int = MAX_REQUESTS,
    max_adversarial_requests: int = MAX_ADVERSARIAL_REQUESTS,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    target_output_dir = Path(output_dir or default_output_dir(workspace_id=workspace_id, root=resolved_root))
    bundle_dir = target_output_dir / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    index_summary = rebuild_memory_index(
        root=resolved_root,
        workspace_id=workspace_id,
        index_path=index_path,
        event_log_path=event_log_path,
    )
    events = sorted(
        [
            *iter_workspace_events(root=resolved_root, workspace_id=workspace_id),
            *iter_capability_matrix_events(root=resolved_root, workspace_id=workspace_id),
        ],
        key=lambda item: (
            str(item.get("recorded_at_utc") or ""),
            str(item.get("event_id") or ""),
        ),
        reverse=True,
    )

    requests: list[dict[str, Any]] = []
    request_events: dict[str, dict[str, Any]] = {}
    seen: set[tuple[str, str]] = set()
    for event in events:
        request = _baseline_request(event)
        if request is None:
            continue
        task_kind = str(request["task_kind"])
        query_text = str(request["query_text"])
        dedupe_key = (task_kind, query_text)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        requests.append(request)
        source_event_id = _clean_text(request.get("source_event_id"))
        if source_event_id:
            request_events[source_event_id] = event
        if len(requests) >= max_requests:
            break

    adversarial_requests: list[dict[str, Any]] = []
    for request in requests:
        if len(adversarial_requests) >= max_adversarial_requests:
            break
        source_event_id = _clean_text(request.get("source_event_id"))
        if not source_event_id:
            continue
        event = request_events.get(source_event_id)
        if event is None:
            continue
        adversarial_request = _adversarial_pass_definition_request(event)
        if adversarial_request is None:
            continue
        adversarial_requests.append(adversarial_request)

    all_requests = [*requests, *adversarial_requests]
    prepared_requests: list[dict[str, Any]] = []
    for index, request in enumerate(all_requests, start=1):
        bundle = build_context_bundle(
            {
                "task_kind": request["task_kind"],
                "query_text": request["query_text"],
                "request_basis": request.get("request_basis"),
                "file_hints": request["file_hints"],
                "limit": request["limit"],
                "context_budget_chars": request["context_budget_chars"],
                "source_event_id": request["source_event_id"],
            },
            root=resolved_root,
            workspace_id=workspace_id,
            index_path=Path(index_summary["index_path"]),
        )
        bundle_path = bundle_dir / f"{index:02d}-{_bundle_slug(request)}.json"
        write_json(bundle_path, bundle)
        source_evaluation = dict(bundle.get("source_evaluation") or {})
        prepared_requests.append(
            {
                **request,
                "auto_generated": True,
                "bundle_path": str(bundle_path),
                "selected_count": bundle.get("selected_count"),
                "omitted_count": bundle.get("omitted_count"),
                "source_hit": bool(source_evaluation.get("source_selected")),
                "source_rank": source_evaluation.get("source_rank"),
                "miss_reason": source_evaluation.get("miss_reason"),
                "source_block_title": source_evaluation.get("source_block_title"),
                "source_selected_via_group": bool(source_evaluation.get("source_selected_via_group")),
                "source_grouped_by": source_evaluation.get("source_grouped_by"),
                "source_group_event_id": source_evaluation.get("source_group_event_id"),
                "source_group_member_count": source_evaluation.get("source_group_member_count"),
                "source_group_member_labels": list(source_evaluation.get("source_group_member_labels") or []),
            }
        )

    dataset = {
        "generated_at_utc": timestamp_utc(),
        "workspace_id": workspace_id,
        "index_summary": index_summary,
        "request_count": len(prepared_requests),
        "baseline_request_count": len(requests),
        "adversarial_request_count": len(adversarial_requests),
        "requests": prepared_requests,
    }
    write_json(target_output_dir / "real_recall_dataset.json", dataset)
    return dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a first real-data recall dataset from workspace and capability artifacts.",
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
        help="Workspace id to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="Optional explicit SQLite index path.",
    )
    parser.add_argument(
        "--event-log-path",
        type=Path,
        default=None,
        help="Optional explicit event log output path.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=MAX_REQUESTS,
        help="Maximum number of auto-generated requests.",
    )
    parser.add_argument(
        "--max-adversarial-requests",
        type=int,
        default=MAX_ADVERSARIAL_REQUESTS,
        help="Maximum number of adversarial recall requests appended after the baseline set.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    dataset = build_real_recall_dataset(
        root=args.root,
        workspace_id=args.workspace_id,
        output_dir=args.output_dir,
        index_path=args.index_path,
        event_log_path=args.event_log_path,
        max_requests=args.max_requests,
        max_adversarial_requests=args.max_adversarial_requests,
    )
    write_json((Path(args.output_dir) if args.output_dir else default_output_dir(workspace_id=args.workspace_id, root=args.root)) / "last_run_summary.json", dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
