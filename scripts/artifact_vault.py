#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

from gemma_runtime import repo_root, timestamp_slug, timestamp_utc, write_json


ARTIFACT_REF_SCHEMA_NAME = "software-satellite-artifact-ref"
ARTIFACT_REF_SCHEMA_VERSION = 1

ARTIFACT_KINDS = {
    "patch",
    "test_log",
    "command_log",
    "transcript",
    "review_note",
    "candidate_output",
    "source_file",
    "ci_log",
    "unknown",
}
SOURCE_STATES = {
    "present",
    "missing",
    "redacted",
    "oversize",
    "binary_refused",
    "outside_workspace",
    "symlink_refused",
}
CAPTURE_STATES = {"captured", "ref_only", "refused"}

DEFAULT_MAX_CAPTURE_BYTES = 2 * 1024 * 1024
DEFAULT_REPORT_EXCERPT_CHARS = 1200
LONG_LINE_LIMIT = 240
REDACTION_RULES_VERSION = 1
ARTIFACT_ID_RE = re.compile(r"^artifact_[A-Za-z0-9_-]+$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

SECRET_PATTERNS = (
    re.compile(r"\b(sk-[A-Za-z0-9_-]{16,})\b"),
    re.compile(r"\b(ghp_[A-Za-z0-9_]{16,})\b"),
    re.compile(r"\b(xox[baprs]-[A-Za-z0-9-]{16,})\b"),
    re.compile(
        r"\b([A-Z0-9_]*(?:API|TOKEN|SECRET|KEY|PASSWORD)[A-Z0-9_]*\s*=\s*)([^\s]+)",
        re.IGNORECASE,
    ),
    re.compile(r"\b(Bearer\s+)([A-Za-z0-9._~+/=-]{16,})\b", re.IGNORECASE),
)


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def artifact_vault_root(root: Path | None = None) -> Path:
    return _resolve_root(root) / "artifacts" / "vault"


def artifact_objects_root(root: Path | None = None) -> Path:
    return artifact_vault_root(root) / "objects"


def artifact_refs_root(root: Path | None = None) -> Path:
    return artifact_vault_root(root) / "refs"


def artifact_index_path(root: Path | None = None) -> Path:
    return artifact_vault_root(root) / "index.jsonl"


def artifact_ref_path(artifact_id: str, *, root: Path | None = None) -> Path:
    if not ARTIFACT_ID_RE.match(artifact_id):
        raise ValueError(f"Invalid artifact id `{artifact_id}`.")
    return artifact_refs_root(root) / f"{artifact_id}.json"


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalize_kind(kind: str | None) -> str:
    normalized = (kind or "unknown").strip().lower().replace("-", "_")
    return normalized if normalized in ARTIFACT_KINDS else "unknown"


def _resolve_source_path(path: str | Path, *, root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        return candidate.resolve()
    except (OSError, RuntimeError):
        return candidate.absolute()


def _source_candidate_path(path: str | Path, *, root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate


def _path_is_inside_root(path: Path, *, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _repo_relative_path(path: Path, *, root: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(root))
    except (OSError, RuntimeError, ValueError):
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_binary(sample: bytes) -> bool:
    if b"\x00" in sample:
        return True
    if not sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def _mime_hint(path: Path, *, binary: bool) -> str:
    suffix = path.suffix.lower()
    if suffix in {".diff", ".patch"}:
        return "text/x-diff"
    if suffix in {".log", ".txt", ".md"}:
        return "text/plain"
    guessed, _encoding = mimetypes.guess_type(str(path))
    if guessed:
        return guessed
    return "application/octet-stream" if binary else "text/plain"


def _redact_secret_like_tokens(text: str) -> tuple[str, int]:
    count = 0
    redacted = text
    for pattern in SECRET_PATTERNS:
        def replace(match: re.Match[str]) -> str:
            nonlocal count
            count += 1
            if match.lastindex and match.lastindex >= 2:
                return f"{match.group(1)}[REDACTED]"
            return "[REDACTED]"

        redacted = pattern.sub(replace, redacted)
    return redacted, count


def redact_report_excerpt(
    text: str,
    *,
    max_chars: int = DEFAULT_REPORT_EXCERPT_CHARS,
    long_line_limit: int = LONG_LINE_LIMIT,
) -> tuple[str, dict[str, Any]]:
    redacted, secret_count = _redact_secret_like_tokens(text)
    lines: list[str] = []
    long_lines_truncated = 0
    for line in redacted.splitlines():
        if len(line) > long_line_limit:
            lines.append(line[:long_line_limit].rstrip() + " [truncated]")
            long_lines_truncated += 1
        else:
            lines.append(line)
    excerpt = "\n".join(lines)
    if len(excerpt) > max_chars:
        excerpt = excerpt[: max(0, max_chars - 15)].rstrip() + " [truncated]"
        long_lines_truncated += 1
    report = {
        "applied": bool(secret_count or long_lines_truncated),
        "secret_like_tokens": secret_count,
        "long_lines_truncated": long_lines_truncated,
        "binary_bytes_refused": 0,
        "rules_version": REDACTION_RULES_VERSION,
    }
    return excerpt, report


def _run_git(root: Path, args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_metadata(path: Path, *, root: Path) -> dict[str, Any]:
    repo_relative = _repo_relative_path(path, root=root)
    if repo_relative is None:
        return {"commit": None, "blob_sha": None, "dirty_tree": False}
    if _run_git(root, ["rev-parse", "--is-inside-work-tree"]) != "true":
        return {"commit": None, "blob_sha": None, "dirty_tree": False}
    commit = _run_git(root, ["rev-parse", "HEAD"])
    blob_sha = _run_git(root, ["hash-object", "--", repo_relative])
    status = _run_git(root, ["status", "--porcelain", "--", repo_relative])
    return {
        "commit": commit,
        "blob_sha": blob_sha,
        "dirty_tree": bool(status),
    }


def _artifact_id(*, sha256: str | None, captured_at_utc: str) -> str:
    slug = timestamp_slug()
    suffix = sha256[:12] if sha256 else hashlib.sha256(captured_at_utc.encode("utf-8")).hexdigest()[:12]
    return f"artifact_{slug}_{suffix}"


def _vault_path_for_sha(sha256: str, *, root: Path) -> Path:
    return artifact_objects_root(root) / sha256[:2] / sha256


def _workspace_path_text(path: Path, *, root: Path) -> str:
    relative = _repo_relative_path(path, root=root)
    return relative or str(path)


def _workspace_path_text_lexical(path: Path, *, root: Path) -> str:
    try:
        return str(path.absolute().relative_to(root))
    except ValueError:
        return str(path)


def _write_ref(ref: dict[str, Any], *, root: Path) -> Path:
    ref_path = artifact_ref_path(str(ref["artifact_id"]), root=root)
    write_json(ref_path, ref)
    index = artifact_index_path(root)
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(ref, ensure_ascii=False, sort_keys=True) + "\n")
    return ref_path


def capture_artifact(
    path: str | Path,
    *,
    kind: str = "unknown",
    root: Path | None = None,
    max_capture_bytes: int = DEFAULT_MAX_CAPTURE_BYTES,
    report_excerpt_chars: int = DEFAULT_REPORT_EXCERPT_CHARS,
    captured_at_utc: str | None = None,
    write_ref: bool = True,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    candidate_path = _source_candidate_path(path, root=resolved_root)
    resolved_path = _resolve_source_path(path, root=resolved_root)
    original_path = str(path)
    normalized_kind = _normalize_kind(kind)
    captured_at = captured_at_utc or timestamp_utc()

    base_ref: dict[str, Any] = {
        "schema_name": ARTIFACT_REF_SCHEMA_NAME,
        "schema_version": ARTIFACT_REF_SCHEMA_VERSION,
        "artifact_id": None,
        "kind": normalized_kind,
        "original_path": original_path,
        "vault_path": None,
        "repo_relative_path": _repo_relative_path(resolved_path, root=resolved_root),
        "sha256": None,
        "size_bytes": None,
        "mime_hint": None,
        "source_state": "missing",
        "capture_state": "refused",
        "git": {"commit": None, "blob_sha": None, "dirty_tree": False},
        "redaction": {
            "applied": False,
            "secret_like_tokens": 0,
            "long_lines_truncated": 0,
            "binary_bytes_refused": 0,
            "rules_version": REDACTION_RULES_VERSION,
        },
        "report_excerpt": {
            "text": "",
            "best_effort_redaction": True,
            "never_upload_notice": "Redaction is best-effort; do not treat captured logs as safe to upload.",
        },
        "captured_at_utc": captured_at,
    }

    if not _path_is_inside_root(resolved_path, root=resolved_root):
        base_ref["artifact_id"] = _artifact_id(sha256=None, captured_at_utc=captured_at)
        base_ref["source_state"] = "outside_workspace"
        base_ref["capture_state"] = "refused"
        base_ref["report_excerpt"]["text"] = "Capture refused: source path is outside the workspace."
        if write_ref:
            _write_ref(base_ref, root=resolved_root)
        return base_ref

    if candidate_path.is_symlink():
        base_ref["artifact_id"] = _artifact_id(sha256=None, captured_at_utc=captured_at)
        base_ref["source_state"] = "symlink_refused"
        base_ref["capture_state"] = "refused"
        base_ref["report_excerpt"]["text"] = "Capture refused: symlink source paths are not copied into the vault."
        if write_ref:
            _write_ref(base_ref, root=resolved_root)
        return base_ref

    if not resolved_path.is_file():
        base_ref["artifact_id"] = _artifact_id(sha256=None, captured_at_utc=captured_at)
        if write_ref:
            _write_ref(base_ref, root=resolved_root)
        return base_ref

    size_bytes = resolved_path.stat().st_size
    sha256 = _file_sha256(resolved_path)
    with resolved_path.open("rb") as handle:
        sample = handle.read(min(size_bytes, 8192))
    binary = _looks_binary(sample)
    base_ref.update(
        {
            "artifact_id": _artifact_id(sha256=sha256, captured_at_utc=captured_at),
            "sha256": sha256,
            "size_bytes": size_bytes,
            "mime_hint": _mime_hint(resolved_path, binary=binary),
            "git": _git_metadata(resolved_path, root=resolved_root),
        }
    )

    if binary:
        base_ref["source_state"] = "binary_refused"
        base_ref["capture_state"] = "refused"
        base_ref["redaction"]["binary_bytes_refused"] = size_bytes
        if write_ref:
            _write_ref(base_ref, root=resolved_root)
        return base_ref

    if size_bytes > max_capture_bytes:
        preview_bytes = max(report_excerpt_chars * 4, LONG_LINE_LIMIT * 4, 16 * 1024)
        with resolved_path.open("rb") as handle:
            text = handle.read(preview_bytes).decode("utf-8", errors="replace")
        excerpt, redaction = redact_report_excerpt(text, max_chars=report_excerpt_chars)
        redaction["long_lines_truncated"] += 1
        redaction["applied"] = True
        base_ref["redaction"] = redaction
        base_ref["report_excerpt"]["text"] = excerpt
        base_ref["source_state"] = "oversize"
        base_ref["capture_state"] = "ref_only"
        if write_ref:
            _write_ref(base_ref, root=resolved_root)
        return base_ref

    text = resolved_path.read_text(encoding="utf-8", errors="replace")
    excerpt, redaction = redact_report_excerpt(text, max_chars=report_excerpt_chars)
    base_ref["redaction"] = redaction
    base_ref["report_excerpt"]["text"] = excerpt

    vault_path = _vault_path_for_sha(sha256, root=resolved_root)
    vault_path.parent.mkdir(parents=True, exist_ok=True)
    if not vault_path.exists():
        shutil.copyfile(resolved_path, vault_path)
    base_ref["vault_path"] = _workspace_path_text(vault_path, root=resolved_root)
    base_ref["source_state"] = "redacted" if redaction.get("applied") else "present"
    base_ref["capture_state"] = "captured"
    if write_ref:
        _write_ref(base_ref, root=resolved_root)
    return base_ref


def load_artifact_ref(artifact_id: str, *, root: Path | None = None) -> dict[str, Any]:
    path = artifact_ref_path(artifact_id, root=root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_name") != ARTIFACT_REF_SCHEMA_NAME:
        raise ValueError(f"Unexpected artifact ref schema in `{path}`.")
    if payload.get("schema_version") != ARTIFACT_REF_SCHEMA_VERSION:
        raise ValueError(f"Unsupported artifact ref schema version in `{path}`.")
    return payload


def resolve_vault_object_path(ref: Mapping[str, Any], *, root: Path | None = None) -> Path | None:
    vault_path = _clean_text(ref.get("vault_path"))
    if vault_path is None:
        return None
    resolved_root = _resolve_root(root)
    candidate = Path(vault_path).expanduser()
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    try:
        return candidate.resolve()
    except (OSError, RuntimeError):
        return candidate.absolute()


def artifact_ref_object_verified(ref: Mapping[str, Any], *, root: Path | None = None) -> tuple[bool, str | None]:
    resolved_root = _resolve_root(root)
    if ref.get("schema_name") != ARTIFACT_REF_SCHEMA_NAME:
        return False, "not_artifact_ref"
    if _clean_text(ref.get("capture_state")) != "captured":
        return False, _clean_text(ref.get("source_state")) or "not_captured"
    sha256 = _clean_text(ref.get("sha256"))
    object_path = resolve_vault_object_path(ref, root=resolved_root)
    if sha256 is None or object_path is None:
        return False, "missing_vault_object"
    try:
        object_path.relative_to(artifact_objects_root(resolved_root).resolve())
    except ValueError:
        return False, "vault_object_outside_vault"
    if object_path != _vault_path_for_sha(sha256, root=resolved_root).resolve():
        return False, "noncanonical_vault_object_path"
    if not object_path.is_file():
        return False, "missing_vault_object"
    actual = _file_sha256(object_path)
    if actual.lower() != sha256.lower():
        return False, "vault_checksum_mismatch"
    return True, None


def inspect_artifact(artifact_id: str, *, root: Path | None = None) -> dict[str, Any]:
    ref = load_artifact_ref(artifact_id, root=root)
    verified, reason = artifact_ref_object_verified(ref, root=root)
    return {
        "schema_name": "software-satellite-artifact-inspection",
        "schema_version": 1,
        "artifact_id": artifact_id,
        "ref": ref,
        "object_verified": verified,
        "verification_reason": reason,
        "redaction_notice": "Redaction is best-effort; never upload captured logs or vault objects without a separate human review.",
    }


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _object_inventory_entry(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "vault_path": _workspace_path_text(path, root=root),
        "sha256": path.name,
        "size_bytes": _safe_size(path),
    }


def _malformed_ref_entry(path: Path, *, root: Path, reason: str) -> dict[str, str]:
    return {
        "ref_path": _workspace_path_text(path, root=root),
        "reason": reason,
    }


def _malformed_ref_entry_lexical(path: Path, *, root: Path, reason: str) -> dict[str, str]:
    return {
        "ref_path": _workspace_path_text_lexical(path, root=root),
        "reason": reason,
    }


def _skipped_object_entry(path: Path, *, root: Path, reason: str) -> dict[str, str]:
    return {
        "vault_path": _workspace_path_text_lexical(path, root=root),
        "reason": reason,
    }


def _resolved_path_is_inside(path: Path, *, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _walk_vault_object_files(objects_root: Path, *, root: Path) -> tuple[list[Path], list[dict[str, str]]]:
    object_paths: list[Path] = []
    skipped_objects: list[dict[str, str]] = []
    if objects_root.is_symlink():
        skipped_objects.append(_skipped_object_entry(objects_root, root=root, reason="symlink_refused"))
        return object_paths, skipped_objects
    if not objects_root.exists():
        return object_paths, skipped_objects

    objects_root_resolved = objects_root.resolve()
    for current_root_text, dirnames, filenames in os.walk(objects_root, followlinks=False):
        current_root = Path(current_root_text)
        safe_dirnames: list[str] = []
        for dirname in sorted(dirnames):
            directory = current_root / dirname
            if directory.is_symlink():
                skipped_objects.append(_skipped_object_entry(directory, root=root, reason="symlink_refused"))
                continue
            safe_dirnames.append(dirname)
        dirnames[:] = safe_dirnames

        for filename in sorted(filenames):
            candidate = current_root / filename
            if candidate.is_symlink():
                skipped_objects.append(_skipped_object_entry(candidate, root=root, reason="symlink_refused"))
                continue
            if not candidate.is_file():
                continue
            if not _resolved_path_is_inside(candidate, root=objects_root_resolved):
                skipped_objects.append(_skipped_object_entry(candidate, root=root, reason="outside_vault"))
                continue
            object_paths.append(candidate.resolve())

    return object_paths, skipped_objects


def _load_gc_ref(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return None, "invalid_utf8"
    except json.JSONDecodeError:
        return None, "invalid_json"
    except OSError:
        return None, "unreadable_ref"
    if not isinstance(payload, dict):
        return None, "ref_is_not_object"
    if payload.get("schema_name") != ARTIFACT_REF_SCHEMA_NAME:
        return None, "unexpected_schema_name"
    if payload.get("schema_version") != ARTIFACT_REF_SCHEMA_VERSION:
        return None, "unsupported_schema_version"
    artifact_id = _clean_text(payload.get("artifact_id"))
    if artifact_id is None or not ARTIFACT_ID_RE.match(artifact_id):
        return None, "invalid_artifact_id"
    capture_state = _clean_text(payload.get("capture_state"))
    if capture_state not in CAPTURE_STATES:
        return None, "invalid_capture_state"
    sha256 = _clean_text(payload.get("sha256"))
    if sha256 is not None and not SHA256_RE.match(sha256):
        return None, "invalid_sha256"
    return payload, None


def artifact_gc_dry_run(*, root: Path | None = None) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    objects_root = artifact_objects_root(resolved_root)
    refs_root = artifact_refs_root(resolved_root)

    object_paths, skipped_objects = _walk_vault_object_files(objects_root, root=resolved_root)
    object_by_key = {str(path): _object_inventory_entry(path, root=resolved_root) for path in object_paths}
    referenced: dict[str, dict[str, Any]] = {}
    missing_objects: list[dict[str, Any]] = []
    malformed_refs: list[dict[str, str]] = []
    valid_ref_count = 0
    captured_ref_count = 0
    verification_cache: dict[tuple[str | None, str | None], tuple[bool, str | None]] = {}

    if refs_root.is_symlink():
        malformed_refs.append(
            _malformed_ref_entry_lexical(
                refs_root,
                root=resolved_root,
                reason="refs_root_symlink_refused",
            )
        )
        ref_paths = []
    elif refs_root.exists() and not refs_root.is_dir():
        malformed_refs.append(_malformed_ref_entry(refs_root, root=resolved_root, reason="refs_root_not_directory"))
        ref_paths = []
    else:
        ref_paths = sorted(refs_root.glob("*.json")) if refs_root.exists() else []
    for ref_path in ref_paths:
        ref, malformed_reason = _load_gc_ref(ref_path)
        if ref is None:
            malformed_refs.append(
                _malformed_ref_entry(
                    ref_path,
                    root=resolved_root,
                    reason=malformed_reason or "malformed_ref",
                )
            )
            continue
        valid_ref_count += 1
        if _clean_text(ref.get("capture_state")) != "captured":
            continue
        captured_ref_count += 1

        artifact_id = _clean_text(ref.get("artifact_id")) or "unknown"
        sha256 = _clean_text(ref.get("sha256"))
        object_path = resolve_vault_object_path(ref, root=resolved_root)
        expected_path = _vault_path_for_sha(sha256, root=resolved_root).resolve() if sha256 else object_path
        object_key = str(object_path.resolve()) if object_path else None
        verification_key = (sha256, object_key)
        if verification_key not in verification_cache:
            verification_cache[verification_key] = artifact_ref_object_verified(ref, root=resolved_root)
        verified, reason = verification_cache[verification_key]
        if not verified or object_path is None:
            missing_objects.append(
                {
                    "artifact_id": artifact_id,
                    "ref_path": _workspace_path_text(ref_path, root=resolved_root),
                    "expected_vault_path": (
                        _workspace_path_text(expected_path, root=resolved_root)
                        if expected_path
                        else None
                    ),
                    "sha256": sha256,
                    "reason": reason or "missing_vault_object",
                }
            )
            continue

        key = object_key or str(object_path.resolve())
        entry = referenced.setdefault(
            key,
            {
                **_object_inventory_entry(object_path, root=resolved_root),
                "artifact_ids": [],
                "ref_paths": [],
            },
        )
        entry["artifact_ids"].append(artifact_id)
        entry["ref_paths"].append(_workspace_path_text(ref_path, root=resolved_root))

    referenced_objects: list[dict[str, Any]] = []
    for entry in referenced.values():
        entry["artifact_ids"] = sorted(set(entry["artifact_ids"]))
        entry["ref_paths"] = sorted(set(entry["ref_paths"]))
        entry["ref_count"] = len(entry["ref_paths"])
        referenced_objects.append(entry)
    referenced_objects.sort(key=lambda item: str(item.get("vault_path") or ""))

    unreferenced_objects = [
        object_by_key[key]
        for key in sorted(object_by_key)
        if key not in referenced
    ]
    reclaimable_bytes = sum(int(item.get("size_bytes") or 0) for item in unreferenced_objects)
    missing_objects.sort(
        key=lambda item: (
            str(item.get("expected_vault_path") or ""),
            str(item.get("artifact_id") or ""),
        )
    )
    malformed_refs.sort(key=lambda item: str(item.get("ref_path") or ""))

    counts = {
        "ref_count": valid_ref_count,
        "captured_ref_count": captured_ref_count,
        "malformed_ref_count": len(malformed_refs),
        "object_count": len(object_paths),
        "referenced_object_count": len(referenced_objects),
        "unreferenced_object_count": len(unreferenced_objects),
        "missing_object_count": len(missing_objects),
        "skipped_object_count": len(skipped_objects),
        "reclaimable_bytes": reclaimable_bytes,
    }
    return {
        "schema_name": "software-satellite-artifact-gc-dry-run",
        "schema_version": 1,
        "dry_run": True,
        "delete_performed": False,
        "vault_root": _workspace_path_text(artifact_vault_root(resolved_root), root=resolved_root),
        "counts": counts,
        "referenced_objects": referenced_objects,
        "unreferenced_objects": unreferenced_objects,
        "missing_objects": missing_objects,
        "malformed_refs": malformed_refs,
        "skipped_objects": skipped_objects,
        "reclaimable_bytes": reclaimable_bytes,
        "safety_notice": "Dry run only; no vault refs or objects were removed.",
    }


def format_artifact_gc_dry_run_report(report: Mapping[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "Artifact GC dry run",
        f"Vault: {_clean_text(report.get('vault_root')) or 'artifacts/vault'}",
        "Mode: dry-run; no files removed.",
        f"Referenced objects: {int(counts.get('referenced_object_count') or 0)}",
        f"Unreferenced objects: {int(counts.get('unreferenced_object_count') or 0)}",
        f"Missing objects: {int(counts.get('missing_object_count') or 0)}",
        f"Malformed refs: {int(counts.get('malformed_ref_count') or 0)}",
        f"Skipped objects: {int(counts.get('skipped_object_count') or 0)}",
        f"Reclaimable bytes: {int(report.get('reclaimable_bytes') or counts.get('reclaimable_bytes') or 0)}",
    ]

    unreferenced = list(report.get("unreferenced_objects") or [])
    if unreferenced:
        lines.extend(["", "Unreferenced objects:"])
        for item in unreferenced:
            lines.append(
                f"- {_clean_text(item.get('vault_path')) or 'unknown'} "
                f"({int(item.get('size_bytes') or 0)} bytes)"
            )

    missing = list(report.get("missing_objects") or [])
    if missing:
        lines.extend(["", "Missing objects:"])
        for item in missing:
            lines.append(
                "- "
                f"{_clean_text(item.get('artifact_id')) or 'unknown'} -> "
                f"{_clean_text(item.get('expected_vault_path')) or 'unknown'} "
                f"({_clean_text(item.get('reason')) or 'missing_vault_object'})"
            )

    malformed = list(report.get("malformed_refs") or [])
    if malformed:
        lines.extend(["", "Malformed refs:"])
        for item in malformed:
            lines.append(
                f"- {_clean_text(item.get('ref_path')) or 'unknown'} "
                f"({_clean_text(item.get('reason')) or 'malformed_ref'})"
            )

    skipped = list(report.get("skipped_objects") or [])
    if skipped:
        lines.extend(["", "Skipped objects:"])
        for item in skipped:
            lines.append(
                f"- {_clean_text(item.get('vault_path')) or 'unknown'} "
                f"({_clean_text(item.get('reason')) or 'skipped_object'})"
            )

    return "\n".join(lines)


def format_artifact_inspection(inspection: Mapping[str, Any]) -> str:
    ref = dict(inspection.get("ref") or {})
    redaction = dict(ref.get("redaction") or {})
    excerpt = dict(ref.get("report_excerpt") or {}).get("text") or ""
    lines = [
        "Artifact inspection",
        f"Artifact: {_clean_text(ref.get('artifact_id')) or _clean_text(inspection.get('artifact_id')) or 'unknown'}",
        f"Kind: {_clean_text(ref.get('kind')) or 'unknown'}",
        f"Source state: {_clean_text(ref.get('source_state')) or 'unknown'}",
        f"Capture state: {_clean_text(ref.get('capture_state')) or 'unknown'}",
        f"SHA-256: {_clean_text(ref.get('sha256')) or 'n/a'}",
        f"Vault object verified: {'yes' if inspection.get('object_verified') else 'no'}",
        "Redaction: best-effort only; do not treat captured logs as safe to upload.",
    ]
    if redaction:
        lines.append(
            "Redaction counts: "
            f"secret_like_tokens={int(redaction.get('secret_like_tokens') or 0)}, "
            f"long_lines_truncated={int(redaction.get('long_lines_truncated') or 0)}, "
            f"binary_bytes_refused={int(redaction.get('binary_bytes_refused') or 0)}"
        )
    if excerpt:
        lines.extend(["", "Report excerpt:", excerpt])
    return "\n".join(lines)
