#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from gemma_runtime import repo_root, timestamp_slug, timestamp_utc, write_json
from workspace_state import DEFAULT_WORKSPACE_ID


GIT_WORK_INTAKE_SCHEMA_NAME = "software-satellite-git-work-intake"
GIT_WORK_INTAKE_SCHEMA_VERSION = 1
DEFAULT_MAX_DIFF_CHARS = 200_000
DEFAULT_MAX_TEST_LOG_CHARS = 80_000
REDACTION_OVERLAP_CHARS = 4096

SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"hf_[A-Za-z0-9]{16,}"),
    re.compile(r"ghp_[A-Za-z0-9]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(
        r"(?i)\b([A-Za-z0-9_-]*(?:api[_-]?key|token|secret|password)[A-Za-z0-9_-]*)\s*[:=]\s*['\"]?[^'\"\s]+"
    ),
)


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_relative(path: Path, *, root: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return None


def git_review_root(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return _resolve_root(root) / "artifacts" / "git_review" / workspace_id


def latest_git_intake_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return git_review_root(workspace_id=workspace_id, root=root) / "intake" / "latest.json"


def git_intake_run_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return (
        git_review_root(workspace_id=workspace_id, root=root)
        / "intake"
        / "runs"
        / f"{timestamp_slug()}-git-intake.json"
    )


def git_patch_snapshot_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return (
        git_review_root(workspace_id=workspace_id, root=root)
        / "patches"
        / "runs"
        / f"{timestamp_slug()}-diff.patch"
    )


def git_test_log_snapshot_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return (
        git_review_root(workspace_id=workspace_id, root=root)
        / "test_logs"
        / "runs"
        / f"{timestamp_slug()}-test-log.txt"
    )


def _run_git(root: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        raise ValueError(f"git {' '.join(args)} failed: {message}")
    return completed.stdout


def _run_git_text_window(root: Path, args: list[str], *, max_chars: int) -> tuple[str, dict[str, Any]]:
    limit = max_chars + REDACTION_OVERLAP_CHARS
    process = subprocess.Popen(
        ["git", *args],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    raw_output = process.stdout.read(limit + 1)
    truncated = len(raw_output) > limit
    if truncated:
        raw_output = raw_output[:limit]
        process.kill()
    _stdout_tail, stderr = process.communicate()
    if process.returncode not in (0, -9) and not truncated:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"git {' '.join(args)} failed: {message}")
    return raw_output.decode("utf-8", errors="replace"), {
        "source_truncated": truncated,
        "source_capture_chars": len(raw_output.decode("utf-8", errors="replace")),
        "redaction_overlap_chars": REDACTION_OVERLAP_CHARS,
        "raw_sha256": _sha256_bytes(raw_output),
        "raw_sha256_scope": "captured_window" if truncated else "full_output",
    }


def _git_ref(root: Path, ref: str) -> str:
    return _run_git(root, ["rev-parse", "--verify", ref]).strip()


def _git_dirty_summary(root: Path) -> dict[str, Any]:
    status = _run_git(root, ["status", "--porcelain=v1"])
    lines = [line for line in status.splitlines() if line.strip()]
    return {
        "dirty": bool(lines),
        "entry_count": len(lines),
        "entries": lines[:30],
        "truncated": len(lines) > 30,
    }


def redact_text(text: str) -> tuple[str, dict[str, Any]]:
    redacted = text
    counts: dict[str, int] = {}
    for index, pattern in enumerate(SECRET_PATTERNS, start=1):
        label = f"pattern_{index}"

        def replacement(match: re.Match[str]) -> str:
            counts[label] = counts.get(label, 0) + 1
            if match.lastindex:
                return f"{match.group(1)}=[REDACTED]"
            return "[REDACTED]"

        redacted = pattern.sub(replacement, redacted)
    return redacted, {
        "redacted": bool(counts),
        "pattern_counts": dict(sorted(counts.items())),
    }


def _bounded_text(text: str, *, max_chars: int) -> tuple[str, dict[str, Any]]:
    if len(text) <= max_chars:
        return text, {
            "truncated": False,
            "original_chars": len(text),
            "captured_chars": len(text),
            "max_chars": max_chars,
        }
    return text[:max_chars], {
        "truncated": True,
        "original_chars": len(text),
        "captured_chars": max_chars,
        "max_chars": max_chars,
    }


def _redact_then_bound(text: str, *, max_chars: int) -> tuple[str, dict[str, Any], dict[str, Any]]:
    redacted, redaction = redact_text(text)
    bounded, bounds = _bounded_text(redacted, max_chars=max_chars)
    return bounded, bounds, redaction


def _parse_name_status(text: str) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = line.split("\t")
        if not parts:
            continue
        status = parts[0]
        if status.startswith("R") and len(parts) >= 3:
            files.append({"status": "R", "old_path": parts[1], "path": parts[2]})
        elif len(parts) >= 2:
            files.append({"status": status, "path": parts[1]})
    return files


def _parse_numstat(text: str) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    stats: dict[str, dict[str, Any]] = {}
    unsupported: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added_text, removed_text, path = parts[0], parts[1], parts[2]
        if added_text == "-" or removed_text == "-":
            stats[path] = {"added": None, "removed": None, "binary": True}
            unsupported.append({"kind": "binary", "path": path})
            continue
        stats[path] = {
            "added": int(added_text),
            "removed": int(removed_text),
            "binary": False,
        }
    return stats, unsupported


def _summarize_changed_files(name_status: str, numstat: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files = _parse_name_status(name_status)
    stats_by_path, unsupported = _parse_numstat(numstat)
    for item in files:
        stats = stats_by_path.get(str(item.get("path") or ""))
        if stats is not None:
            item.update(stats)
        if item.get("status") == "D":
            unsupported.append({"kind": "deleted_file", "path": item.get("path")})
    return files, unsupported


def _test_status_from_log(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\b(failed|failure|error|traceback)\b", lowered) and not re.search(r"\b0 failed\b", lowered):
        return "fail"
    if re.search(r"\b(passed|success|successful|ok)\b", lowered):
        return "pass"
    return "unknown"


def _read_optional_text(
    path: Path | None,
    *,
    root: Path,
    workspace_id: str,
    max_chars: int,
) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    if not candidate.is_file():
        raise ValueError(f"test log must be a readable file: `{candidate}`.")
    read_limit = max_chars + REDACTION_OVERLAP_CHARS
    with candidate.open("r", encoding="utf-8", errors="replace") as handle:
        raw_text = handle.read(read_limit + 1)
    source_truncated = len(raw_text) > read_limit
    if source_truncated:
        raw_text = raw_text[:read_limit]
    redacted, bounds, redaction = _redact_then_bound(raw_text, max_chars=max_chars)
    bounds.update(
        {
            "source_truncated": source_truncated,
            "source_capture_chars": len(raw_text),
            "redaction_overlap_chars": REDACTION_OVERLAP_CHARS,
        }
    )
    snapshot_path = git_test_log_snapshot_path(workspace_id=workspace_id, root=root)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(redacted, encoding="utf-8")
    return {
        "source_path": str(candidate.resolve()),
        "source_workspace_relative_path": _workspace_relative(candidate, root=root),
        "source_sha256": _file_sha256(candidate),
        "captured_sha256": _sha256_text(redacted),
        "snapshot_path": str(snapshot_path),
        "snapshot_workspace_relative_path": _workspace_relative(snapshot_path, root=root),
        "status": _test_status_from_log(redacted),
        "excerpt": redacted[:1200],
        "bounds": bounds,
        "redaction": redaction,
    }


def capture_git_work_intake(
    *,
    base: str,
    head: str = "HEAD",
    test_log: Path | None = None,
    note: str | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    max_diff_chars: int = DEFAULT_MAX_DIFF_CHARS,
    max_test_log_chars: int = DEFAULT_MAX_TEST_LOG_CHARS,
) -> tuple[dict[str, Any], Path, Path]:
    resolved_root = _resolve_root(root)
    _run_git(resolved_root, ["rev-parse", "--show-toplevel"])
    base_commit = _git_ref(resolved_root, base)
    head_commit = _git_ref(resolved_root, head)
    dirty = _git_dirty_summary(resolved_root)
    diff_text, diff_source_bounds = _run_git_text_window(
        resolved_root,
        ["diff", "--no-ext-diff", "--find-renames", base, head],
        max_chars=max_diff_chars,
    )
    name_status = _run_git(resolved_root, ["diff", "--name-status", "--find-renames", base, head])
    numstat = _run_git(resolved_root, ["diff", "--numstat", "--find-renames", base, head])
    changed_files, unsupported_components = _summarize_changed_files(name_status, numstat)
    if "GIT binary patch" in diff_text or "Binary files " in diff_text:
        unsupported_components.append({"kind": "binary_diff_marker", "path": None})

    redacted_diff, diff_bounds, diff_redaction = _redact_then_bound(diff_text, max_chars=max_diff_chars)
    diff_bounds.update(diff_source_bounds)
    patch_path = git_patch_snapshot_path(workspace_id=workspace_id, root=resolved_root)
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(redacted_diff, encoding="utf-8")

    test_log_record = _read_optional_text(
        test_log,
        root=resolved_root,
        workspace_id=workspace_id,
        max_chars=max_test_log_chars,
    )
    generated_at = timestamp_utc()
    hunk_count = sum(1 for line in redacted_diff.splitlines() if line.startswith("@@"))
    intake = {
        "schema_name": GIT_WORK_INTAKE_SCHEMA_NAME,
        "schema_version": GIT_WORK_INTAKE_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "generated_at_utc": generated_at,
        "base": base,
        "head": head,
        "base_commit": base_commit,
        "head_commit": head_commit,
        "note": _clean_text(note),
        "dirty_tree": dirty,
        "diff": {
            "source": "git diff --no-ext-diff --find-renames",
            "raw_sha256": diff_source_bounds["raw_sha256"],
            "raw_sha256_scope": diff_source_bounds["raw_sha256_scope"],
            "captured_sha256": _sha256_text(redacted_diff),
            "snapshot_path": str(patch_path),
            "snapshot_workspace_relative_path": _workspace_relative(patch_path, root=resolved_root),
            "bounds": diff_bounds,
            "redaction": diff_redaction,
            "hunk_count": hunk_count,
        },
        "changed_files": changed_files,
        "changed_file_count": len(changed_files),
        "unsupported_components": unsupported_components,
        "test_log": test_log_record,
        "training_export_ready": False,
    }
    run_path = git_intake_run_path(workspace_id=workspace_id, root=resolved_root)
    latest_path = latest_git_intake_path(workspace_id=workspace_id, root=resolved_root)
    write_json(run_path, intake)
    write_json(latest_path, intake)
    return intake, latest_path, run_path


def format_git_intake_summary(intake: Mapping[str, Any]) -> str:
    diff = intake.get("diff") if isinstance(intake.get("diff"), Mapping) else {}
    dirty = intake.get("dirty_tree") if isinstance(intake.get("dirty_tree"), Mapping) else {}
    test_log = intake.get("test_log") if isinstance(intake.get("test_log"), Mapping) else None
    lines = [
        "Git work intake",
        f"Base: {intake.get('base')} ({str(intake.get('base_commit') or '')[:12]})",
        f"Head: {intake.get('head')} ({str(intake.get('head_commit') or '')[:12]})",
        f"Changed files: {intake.get('changed_file_count')}",
        f"Hunks: {diff.get('hunk_count')}",
        f"Patch snapshot: {diff.get('snapshot_path')}",
        f"Dirty tree: {'yes' if dirty.get('dirty') else 'no'}",
    ]
    if test_log is not None:
        lines.append(f"Test log status: {test_log.get('status')}")
    if intake.get("unsupported_components"):
        lines.append(f"Unsupported components: {len(intake.get('unsupported_components') or [])}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture a bounded, redacted git diff intake artifact.")
    parser.add_argument("--base", required=True, help="Base git ref.")
    parser.add_argument("--head", default="HEAD", help="Head git ref.")
    parser.add_argument("--test-log", type=Path, default=None, help="Optional local test log file.")
    parser.add_argument("--note", default="", help="Optional note.")
    parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id.")
    parser.add_argument("--root", type=Path, default=None, help="Optional repo root.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        intake, _latest_path, _run_path = capture_git_work_intake(
            base=args.base,
            head=args.head,
            test_log=args.test_log,
            note=args.note,
            workspace_id=args.workspace_id,
            root=args.root,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.format == "json":
        print(json.dumps(intake, ensure_ascii=False, indent=2))
    else:
        print(format_git_intake_summary(intake))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
