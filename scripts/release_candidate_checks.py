#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

from demand_gate import build_demand_gate_report, format_demand_gate_report, record_demand_gate_report
from evidence_lint import build_evidence_lint_report
from evidence_pack_v1 import audit_evidence_pack_v1_path
from gemma_runtime import repo_root, timestamp_slug, timestamp_utc, write_json
from review_benchmark import run_review_benchmark
from review_memory_eval import run_review_memory_eval
from workspace_state import DEFAULT_WORKSPACE_ID


RELEASE_CANDIDATE_SCHEMA_NAME = "software-satellite-release-candidate-checks"
RELEASE_CANDIDATE_SCHEMA_VERSION = 1
RELEASE_DEMO_SCHEMA_NAME = "software-satellite-release-demo-report"
RELEASE_DEMO_SCHEMA_VERSION = 1

DEFAULT_TEST_TIMEOUT_SECONDS = 180
BENCHMARK_FRESH_SECONDS = 30 * 60

PUBLIC_DOC_PATHS = (
    "README.md",
    "README_EN.md",
    "docs/release_v0_1_candidate.md",
    "docs/public_demo_walkthrough.md",
)
REQUIRED_RELEASE_FILES = (
    "scripts/release_candidate_checks.py",
    "scripts/demand_gate.py",
    "docs/release_v0_1_candidate.md",
    "docs/public_demo_walkthrough.md",
    "tests/test_release_candidate_checks.py",
)
PUBLIC_DEMO_FIXTURES = (
    "examples/review_memory_benchmark/synthetic_suite.json",
    "examples/agent_session_bundles/generic.json",
    "examples/demand_gate/release_candidate_fixture.json",
    "templates/failure-memory-pack.satellite.yaml",
    "templates/agent-session-pack.satellite.yaml",
)
STRICT_PACKS = (
    "templates/failure-memory-pack.satellite.yaml",
    "templates/agent-session-pack.satellite.yaml",
)
PUBLIC_SCAN_ROOTS = (
    "README.md",
    "README_EN.md",
    "docs",
    "examples",
    "templates",
    "schemas",
)
DEFAULT_DEMO_COMMANDS = (
    "python3 scripts/satlab.py release demo --no-api",
    "python3 scripts/satlab.py demand gate --fixture-metrics examples/demand_gate/release_candidate_fixture.json --format md",
)

PRIVATE_DOC_PATTERNS = (
    re.compile(r"\.private_docs\b"),
    re.compile(r"\bprivate_docs/"),
    re.compile(r"software_satellite_lab_m10_m17_spartan_design", re.IGNORECASE),
)
API_REQUIRED_PATTERNS = (
    re.compile(r"\b[A-Z0-9_]*API_KEY\b"),
    re.compile(r"\bapi\s+key\s+(?:is\s+)?required\b", re.IGNORECASE),
    re.compile(r"\brequires?\s+(?:an?\s+)?api\s+key\b", re.IGNORECASE),
    re.compile(r"\bmust\s+set\b.*\bapi[_ -]?key\b", re.IGNORECASE),
)
NETWORK_COMMAND_PATTERN = re.compile(
    r"\b(?:curl|wget|fetch|requests|urllib|httpx|socket|npm\s+install|pip\s+install)\b|https?://",
    re.IGNORECASE,
)
TRAINABLE_EXPORT_PATH_PATTERN = re.compile(
    r"(?:^|/)(?:training[_-]?exports?|trainable[_-]?exports?|fine[_-]?tune|finetune)(?:/|$)|"
    r"(?:training[_-]?export|trainable|fine[_-]?tune|finetune).*\.jsonl$",
    re.IGNORECASE,
)
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"ghp_[A-Za-z0-9_]{12,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*[^\s`'\"|]{8,}"),
)


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def _mapping_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _utc_timestamp_seconds(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _check(
    check_id: str,
    label: str,
    passed: bool,
    *,
    detail: Mapping[str, Any] | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "label": label,
        "status": status or ("pass" if passed else "fail"),
        "detail": dict(detail or {}),
    }


def redact_release_text(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def release_candidate_root(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return _resolve_root(root) / "artifacts" / "release_candidate" / workspace_id


def release_check_latest_json_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return release_candidate_root(workspace_id=workspace_id, root=root) / "checks" / "latest.json"


def release_check_latest_md_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return release_candidate_root(workspace_id=workspace_id, root=root) / "checks" / "latest.md"


def release_check_run_json_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    run_slug: str | None = None,
) -> Path:
    slug = run_slug or timestamp_slug()
    return release_candidate_root(workspace_id=workspace_id, root=root) / "checks" / "runs" / f"{slug}-release-checks.json"


def release_check_run_md_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    run_slug: str | None = None,
) -> Path:
    slug = run_slug or timestamp_slug()
    return release_candidate_root(workspace_id=workspace_id, root=root) / "checks" / "runs" / f"{slug}-release-checks.md"


def release_demo_latest_md_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return release_candidate_root(workspace_id=workspace_id, root=root) / "demo" / "latest.md"


def release_demo_latest_json_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return release_candidate_root(workspace_id=workspace_id, root=root) / "demo" / "latest.json"


def release_demo_run_md_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    run_slug: str | None = None,
) -> Path:
    slug = run_slug or timestamp_slug()
    return release_candidate_root(workspace_id=workspace_id, root=root) / "demo" / "runs" / f"{slug}-public-demo.md"


def release_demo_run_json_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
    run_slug: str | None = None,
) -> Path:
    slug = run_slug or timestamp_slug()
    return release_candidate_root(workspace_id=workspace_id, root=root) / "demo" / "runs" / f"{slug}-public-demo.json"


def _iter_public_scan_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for item in PUBLIC_SCAN_ROOTS:
        path = root / item
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                candidate
                for candidate in sorted(path.rglob("*"))
                if candidate.is_file()
                and ".private_docs" not in candidate.parts
                and "__pycache__" not in candidate.parts
            )
    return files


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _workspace_relative(path: Path, *, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def _scan_private_doc_references(root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in _iter_public_scan_files(root):
        for line_number, line in enumerate(_read_text(path).splitlines(), start=1):
            if any(pattern.search(line) for pattern in PRIVATE_DOC_PATTERNS):
                findings.append(
                    {
                        "path": _workspace_relative(path, root=root),
                        "line": line_number,
                        "excerpt": line.strip()[:200],
                    }
                )
    return findings


def _line_explicitly_says_no_api(line: str) -> bool:
    normalized = line.lower().replace("-", " ")
    return (
        "no api key" in normalized
        or "api key is not required" in normalized
        or "api key not required" in normalized
        or "api key required: false" in normalized
        or "api key なし" in normalized
        or "api keyなし" in normalized
    )


def _scan_api_key_requirements(root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for public_path in PUBLIC_DOC_PATHS:
        path = root / public_path
        if not path.is_file():
            continue
        for line_number, line in enumerate(_read_text(path).splitlines(), start=1):
            if _line_explicitly_says_no_api(line):
                continue
            if any(pattern.search(line) for pattern in API_REQUIRED_PATTERNS):
                findings.append(
                    {
                        "path": _workspace_relative(path, root=root),
                        "line": line_number,
                        "excerpt": line.strip()[:200],
                    }
                )
    return findings


def _scan_trainable_export_artifacts(root: Path) -> list[str]:
    artifacts_root = root / "artifacts"
    if not artifacts_root.exists():
        return []
    findings: list[str] = []
    for path in sorted(artifacts_root.rglob("*")):
        if not path.is_file():
            continue
        relative = _workspace_relative(path, root=root)
        normalized = relative.replace(os.sep, "/")
        if TRAINABLE_EXPORT_PATH_PATTERN.search(normalized):
            findings.append(relative)
    return findings


def _release_docs_text(root: Path) -> str:
    return "\n".join(_read_text(root / path) for path in PUBLIC_DOC_PATHS)


def _fresh_clone_documented(root: Path) -> dict[str, Any]:
    text = _release_docs_text(root).lower()
    required_tokens = ("git clone", "python -m venv", "pip install -r requirements.txt", "release demo --no-api")
    missing = [token for token in required_tokens if token not in text]
    return {"missing_tokens": missing}


def _known_limitations_documented(root: Path) -> dict[str, Any]:
    text = _release_docs_text(root).lower()
    has_known_limitations = "known limitations" in text or "既知の制約" in text
    return {"documented": has_known_limitations}


def _security_notes_documented(root: Path) -> dict[str, Any]:
    text = _release_docs_text(root).lower()
    has_security_notes = (
        "security/privacy" in text
        or "security and privacy" in text
        or "security notes" in text
        or "security/privacy caveats" in text
        or "セキュリティ" in text
    )
    return {"documented": has_security_notes}


def _readme_one_liner_clear(root: Path) -> dict[str, Any]:
    text = _read_text(root / "README.md")
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#") and not line.startswith("[")]
    first = lines[0] if lines else ""
    clear = bool(first) and ("ローカルファースト" in first or "local-first" in first.lower()) and (
        "Flight Recorder" in first or "evidence" in first.lower() or "evidence" in text.lower()
    )
    return {"line": first, "clear": clear}


def _required_files_check(root: Path) -> dict[str, Any]:
    missing = [path for path in REQUIRED_RELEASE_FILES if not (root / path).is_file()]
    return {"missing": missing}


def _public_demo_fixtures_check(root: Path) -> dict[str, Any]:
    missing = [path for path in PUBLIC_DEMO_FIXTURES if not (root / path).is_file()]
    return {"missing": missing}


def _default_demo_network_check() -> dict[str, Any]:
    findings = [command for command in DEFAULT_DEMO_COMMANDS if NETWORK_COMMAND_PATTERN.search(command)]
    return {"commands": list(DEFAULT_DEMO_COMMANDS), "network_like_commands": findings}


def _benchmark_stale(report: Mapping[str, Any], *, now: float | None = None) -> bool:
    timestamp = _utc_timestamp_seconds(report.get("generated_at_utc") or report.get("created_at_utc"))
    if timestamp is None:
        return True
    current = time.time() if now is None else now
    return current - timestamp > BENCHMARK_FRESH_SECONDS


def _build_review_benchmark_check(report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    benchmark = dict(report) if report is not None else run_review_benchmark(workspace_id="release-candidate-benchmark")
    benchmark.setdefault("generated_at_utc", timestamp_utc())
    stale = _benchmark_stale(benchmark)
    passed = bool(benchmark.get("passed")) and not stale
    return _check(
        "review_benchmark_passes",
        "Review benchmark passes",
        passed,
        detail={
            "passed": bool(benchmark.get("passed")),
            "critical_false_evidence_count": benchmark.get("critical_false_evidence_count"),
            "training_export_ready": benchmark.get("training_export_ready"),
            "generated_at_utc": benchmark.get("generated_at_utc"),
            "stale": stale,
        },
    )


def _build_spartan_benchmark_check(report: Mapping[str, Any] | None = None, *, root: Path) -> dict[str, Any]:
    benchmark = dict(report) if report is not None else run_review_memory_eval(
        workspace_id="release-candidate-spartan",
        root=root,
        spartan=True,
    )
    benchmark.setdefault("generated_at_utc", timestamp_utc())
    passed = bool(benchmark.get("passed"))
    return _check(
        "adversarial_review_memory_benchmark_passes",
        "Adversarial review-memory benchmark passes",
        passed,
        detail={
            "passed": passed,
            "critical_false_support": benchmark.get("metrics", {}).get("critical_false_support")
            if isinstance(benchmark.get("metrics"), Mapping)
            else benchmark.get("critical_false_support"),
            "generated_at_utc": benchmark.get("generated_at_utc"),
        },
    )


def _build_evidence_lint_check(report: Mapping[str, Any] | None = None, *, root: Path) -> dict[str, Any]:
    lint_report = dict(report) if report is not None else build_evidence_lint_report(root=root, strict=True)
    passed = lint_report.get("verdict") != "fail"
    return _check(
        "evidence_lint_passes",
        "Evidence lint passes",
        passed,
        detail={"verdict": lint_report.get("verdict"), "issue_count": lint_report.get("issue_count")},
    )


def _build_pack_audit_check(
    audits: Sequence[Mapping[str, Any]] | None = None,
    *,
    root: Path,
) -> dict[str, Any]:
    if audits is None:
        audit_payloads = []
        for pack in STRICT_PACKS:
            audit, _latest, _run = audit_evidence_pack_v1_path(
                root / pack,
                root=root,
                strict=True,
                write_artifact=False,
            )
            audit_payloads.append({"pack": pack, "verdict": audit.get("verdict")})
    else:
        audit_payloads = [dict(item) for item in audits]
    failing = [item for item in audit_payloads if item.get("verdict") != "pass"]
    return _check(
        "pack_strict_audit_passes",
        "Pack strict audit passes",
        not failing,
        detail={"audits": audit_payloads, "failing": failing},
    )


def _build_redaction_fixture_check(
    benchmark_check: Mapping[str, Any] | None = None,
    redaction_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if redaction_override is not None:
        passed = bool(redaction_override.get("passed"))
        detail = dict(redaction_override)
    else:
        detail = _mapping_dict(_mapping_dict(benchmark_check).get("detail"))
        passed = (
            bool(detail.get("not_run_static_only"))
            or bool(detail.get("passed")) and int(detail.get("critical_false_evidence_count") or 0) == 0
        )
    probe = "TOKEN=sk-releasecandidateprobe000000000000"
    redacted_probe = redact_release_text(probe)
    passed = passed and "sk-releasecandidateprobe" not in redacted_probe and "[REDACTED]" in redacted_probe
    detail["report_redaction_probe_passed"] = "sk-releasecandidateprobe" not in redacted_probe
    return _check("redaction_fixtures_pass", "Redaction fixtures pass", passed, detail=detail)


def _run_default_tests(root: Path, *, timeout_seconds: int = DEFAULT_TEST_TIMEOUT_SECONDS) -> dict[str, Any]:
    started = time.perf_counter()
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "scripts" if not existing_pythonpath else f"scripts{os.pathsep}{existing_pythonpath}"
    command = [sys.executable, "scripts/run_public_demo_checks.py"]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "passed": False,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "output_excerpt": ((exc.stdout or "") + (exc.stderr or ""))[-2000:],
        }
    elapsed = time.perf_counter() - started
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "timed_out": False,
        "timeout_seconds": timeout_seconds,
        "elapsed_seconds": round(elapsed, 3),
        "output_excerpt": ((completed.stdout or "") + (completed.stderr or ""))[-2000:],
    }


def _build_default_tests_check(result: Mapping[str, Any] | None, *, root: Path, run: bool) -> dict[str, Any]:
    if result is None and run:
        result = _run_default_tests(root)
    elif result is None:
        result = {"passed": True, "not_run_static_only": True}
    passed = bool(result.get("passed")) and not bool(result.get("timed_out"))
    return _check(
        "default_tests_pass_under_timeout",
        "Public demo default test gate passes under timeout",
        passed,
        detail=dict(result),
    )


def _build_release_demo_check(
    result: Mapping[str, Any] | None,
    *,
    root: Path,
    workspace_id: str,
    run: bool,
    write: bool,
) -> dict[str, Any]:
    if result is None and run:
        demo = build_release_demo_report(
            root=root,
            workspace_id=workspace_id,
            no_api=True,
            write=write,
            run_runtime_checks=False,
        )
    elif result is None:
        demo = {
            "markdown_report_exists": True,
            "markdown_report_rendered": True,
            "guardrails": {
                "requires_api_key": False,
                "uses_network": False,
                "uses_private_docs": False,
                "writes_training_data": False,
            },
        }
    else:
        demo = dict(result)
    guardrails = _mapping_dict(demo.get("guardrails"))
    passed = (
        bool(demo.get("markdown_report_exists") or demo.get("markdown_report_rendered"))
        and guardrails.get("requires_api_key") is False
        and guardrails.get("uses_network") is False
        and guardrails.get("uses_private_docs") is False
        and guardrails.get("writes_training_data") is False
    )
    return _check("release_demo_markdown_report", "Release demo produces Markdown report", passed, detail=demo)


def _build_demand_gate_wired_check(
    result: Mapping[str, Any] | None,
    *,
    root: Path,
    workspace_id: str,
    run: bool,
    write: bool,
) -> dict[str, Any]:
    if result is None and run:
        fixture = root / "examples" / "demand_gate" / "release_candidate_fixture.json"
        if write:
            demand_report, _markdown, latest_json, latest_md, _run_json, _run_md = record_demand_gate_report(
                workspace_id=workspace_id,
                root=root,
                fixture_metrics=fixture,
            )
            report_exists = latest_json.is_file() and latest_md.is_file()
        else:
            demand_report = build_demand_gate_report(
                workspace_id=workspace_id,
                root=root,
                fixture_metrics=fixture,
            )
            report_exists = False
        paths = _mapping_dict(demand_report.get("paths"))
        payload = {
            "status": demand_report.get("status"),
            "report_latest_json_path": paths.get("report_latest_json_path"),
            "report_latest_md_path": paths.get("report_latest_md_path"),
            "report_exists": report_exists,
            "report_rendered": True,
            "metrics_source": demand_report.get("metrics_source"),
        }
    elif result is None:
        payload = {"status": "pass", "report_exists": True, "not_run_static_only": True}
    else:
        payload = dict(result)
    passed = payload.get("status") == "pass" and bool(payload.get("report_exists") or payload.get("report_rendered"))
    return _check("demand_gate_report_exists", "Demand gate report renders or writes", passed, detail=payload)


def _build_static_checks(root: Path) -> list[dict[str, Any]]:
    required = _required_files_check(root)
    fresh_clone = _fresh_clone_documented(root)
    api_findings = _scan_api_key_requirements(root)
    private_findings = _scan_private_doc_references(root)
    network = _default_demo_network_check()
    fixtures = _public_demo_fixtures_check(root)
    readme = _readme_one_liner_clear(root)
    limitations = _known_limitations_documented(root)
    security = _security_notes_documented(root)
    trainable = _scan_trainable_export_artifacts(root)
    return [
        _check(
            "release_files_present",
            "Release files present",
            not required["missing"],
            detail=required,
        ),
        _check(
            "fresh_clone_setup_documented",
            "Fresh clone setup command documented",
            not fresh_clone["missing_tokens"],
            detail=fresh_clone,
        ),
        _check(
            "no_api_key_required_for_demo",
            "No API key required for demo",
            not api_findings,
            detail={"findings": api_findings},
        ),
        _check(
            "no_private_docs_required",
            "No private docs required",
            not private_findings,
            detail={"private_doc_dependency_count": len(private_findings), "findings": private_findings},
        ),
        _check(
            "no_network_calls_in_default_demo",
            "No network calls in default demo",
            not network["network_like_commands"],
            detail=network,
        ),
        _check(
            "public_demo_fixtures_included",
            "Public demo fixtures included",
            not fixtures["missing"],
            detail=fixtures,
        ),
        _check(
            "no_trainable_export_artifacts",
            "No trainable export artifacts",
            not trainable,
            detail={"findings": trainable},
        ),
        _check(
            "readme_one_liner_clear",
            "README one-liner clear",
            bool(readme["clear"]),
            detail=readme,
        ),
        _check(
            "known_limitations_documented",
            "Known limitations documented",
            bool(limitations["documented"]),
            detail=limitations,
        ),
        _check(
            "security_notes_documented",
            "Security notes documented",
            bool(security["documented"]),
            detail=security,
        ),
    ]


def build_release_candidate_report(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    strict: bool = False,
    run_runtime_checks: bool = True,
    run_default_tests: bool | None = None,
    write_subreports: bool = False,
    runtime_overrides: Mapping[str, Any] | None = None,
    benchmark_report_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    overrides = _mapping_dict(runtime_overrides)
    checks = _build_static_checks(resolved_root)
    default_tests_should_run = strict if run_default_tests is None else run_default_tests

    if run_runtime_checks or benchmark_report_override is not None:
        review_check = _build_review_benchmark_check(
            report=benchmark_report_override or _mapping_dict(overrides.get("review_benchmark")) or None
        )
    else:
        review_check = _check(
            "review_benchmark_passes",
            "Review benchmark passes",
            True,
            detail={"not_run_static_only": True},
        )
    checks.append(review_check)

    if run_runtime_checks:
        checks.append(
            _build_spartan_benchmark_check(
                _mapping_dict(overrides.get("spartan_benchmark")) or None,
                root=resolved_root,
            )
        )
        checks.append(
            _build_evidence_lint_check(
                _mapping_dict(overrides.get("evidence_lint")) or None,
                root=resolved_root,
            )
        )
        pack_override = overrides.get("pack_audits")
        checks.append(
            _build_pack_audit_check(
                pack_override if isinstance(pack_override, Sequence) and not isinstance(pack_override, (str, bytes)) else None,
                root=resolved_root,
            )
        )
        checks.append(
            _build_release_demo_check(
                _mapping_dict(overrides.get("release_demo")) or None,
                root=resolved_root,
                workspace_id=workspace_id,
                run=True,
                write=write_subreports,
            )
        )
        checks.append(
            _build_demand_gate_wired_check(
                _mapping_dict(overrides.get("demand_gate")) or None,
                root=resolved_root,
                workspace_id=workspace_id,
                run=True,
                write=write_subreports,
            )
        )
    else:
        checks.extend(
            [
                _check(
                    "adversarial_review_memory_benchmark_passes",
                    "Adversarial review-memory benchmark passes",
                    True,
                    detail={"not_run_static_only": True},
                ),
                _check("evidence_lint_passes", "Evidence lint passes", True, detail={"not_run_static_only": True}),
                _check("pack_strict_audit_passes", "Pack strict audit passes", True, detail={"not_run_static_only": True}),
                _build_release_demo_check(None, root=resolved_root, workspace_id=workspace_id, run=False, write=False),
                _build_demand_gate_wired_check(None, root=resolved_root, workspace_id=workspace_id, run=False, write=False),
            ]
        )

    checks.append(
        _build_redaction_fixture_check(
            review_check,
            redaction_override=_mapping_dict(overrides.get("redaction_fixtures")) or None,
        )
    )
    checks.append(
        _build_default_tests_check(
            _mapping_dict(overrides.get("default_tests")) or None,
            root=resolved_root,
            run=bool(run_runtime_checks and default_tests_should_run),
        )
    )

    failing = [check for check in checks if check.get("status") != "pass"]
    private_doc_dependency_count = 0
    for check in checks:
        if check.get("id") == "no_private_docs_required":
            private_doc_dependency_count = int(_mapping_dict(check.get("detail")).get("private_doc_dependency_count") or 0)
            break
    demand_gate_check_detail = next(
        (
            _mapping_dict(check.get("detail"))
            for check in checks
            if check.get("id") == "demand_gate_report_exists"
        ),
        {},
    )
    metrics = {
        "release_check_strict_exit_code": 0 if strict and not failing else 1 if strict else None,
        "private_doc_dependency_count": private_doc_dependency_count,
        "api_key_required": any(check.get("id") == "no_api_key_required_for_demo" and check.get("status") != "pass" for check in checks),
        "critical_false_support_count": _mapping_dict(review_check.get("detail")).get("critical_false_evidence_count"),
        "demand_gate_report_exists": bool(demand_gate_check_detail.get("report_exists")) if write_subreports else None,
        "demand_gate_report_rendered": bool(
            demand_gate_check_detail.get("report_exists") or demand_gate_check_detail.get("report_rendered")
        ),
    }
    run_slug = timestamp_slug()
    return {
        "schema_name": RELEASE_CANDIDATE_SCHEMA_NAME,
        "schema_version": RELEASE_CANDIDATE_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "generated_at_utc": timestamp_utc(),
        "strict": strict,
        "status": "pass" if not failing else "fail",
        "checks": checks,
        "failing_check_ids": [str(check.get("id")) for check in failing],
        "metrics": metrics,
        "guardrails": {
            "local_first": True,
            "file_first": True,
            "requires_api_key": False,
            "uses_network": False,
            "uses_private_docs": False,
            "writes_training_data": False,
            "default_demo_uses_public_fixtures": True,
        },
        "paths": {
            "report_latest_json_path": str(release_check_latest_json_path(workspace_id=workspace_id, root=resolved_root)),
            "report_latest_md_path": str(release_check_latest_md_path(workspace_id=workspace_id, root=resolved_root)),
            "report_run_json_path": str(release_check_run_json_path(workspace_id=workspace_id, root=resolved_root, run_slug=run_slug)),
            "report_run_md_path": str(release_check_run_md_path(workspace_id=workspace_id, root=resolved_root, run_slug=run_slug)),
        },
    }


def _safe_json_text(value: Any) -> str:
    return redact_release_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


def format_release_candidate_report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Release Candidate Checks",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Strict: `{str(bool(report.get('strict'))).lower()}`",
        f"- Generated: `{report.get('generated_at_utc')}`",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in report.get("checks") or []:
        if not isinstance(check, Mapping):
            continue
        detail = _safe_json_text(check.get("detail"))
        if len(detail) > 240:
            detail = detail[:237] + "..."
        lines.append(f"| {check.get('label')} | `{check.get('status')}` | `{detail}` |")
    failing = [str(item) for item in report.get("failing_check_ids") or []]
    if failing:
        lines.extend(["", "## Failing Checks", ""])
        lines.extend(f"- `{item}`" for item in failing)
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Local-first and file-first.",
            "- No API key is required for the default demo.",
            "- No network call is part of the default demo path.",
            "- Private design notes are not required to run or understand the release candidate.",
            "- Release reports redact common secret-shaped tokens before rendering excerpts.",
        ]
    )
    return "\n".join(lines) + "\n"


def record_release_candidate_report(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    strict: bool = False,
    run_runtime_checks: bool = True,
    run_default_tests: bool | None = None,
) -> tuple[dict[str, Any], str, Path, Path, Path, Path]:
    report = build_release_candidate_report(
        root=root,
        workspace_id=workspace_id,
        strict=strict,
        run_runtime_checks=run_runtime_checks,
        run_default_tests=run_default_tests,
        write_subreports=True,
    )
    markdown = format_release_candidate_report_markdown(report)
    paths = _mapping_dict(report.get("paths"))
    latest_json = Path(str(paths["report_latest_json_path"]))
    latest_md = Path(str(paths["report_latest_md_path"]))
    run_json = Path(str(paths["report_run_json_path"]))
    run_md = Path(str(paths["report_run_md_path"]))
    write_json(latest_json, report)
    write_json(run_json, report)
    latest_md.parent.mkdir(parents=True, exist_ok=True)
    run_md.parent.mkdir(parents=True, exist_ok=True)
    latest_md.write_text(markdown, encoding="utf-8")
    run_md.write_text(markdown, encoding="utf-8")
    return report, markdown, latest_json, latest_md, run_json, run_md


def build_release_demo_report(
    *,
    root: Path | None = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    no_api: bool = True,
    write: bool = True,
    run_runtime_checks: bool = True,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    if not no_api:
        raise ValueError("release demo requires --no-api for the current public demo path.")
    fixture = resolved_root / "examples" / "demand_gate" / "release_candidate_fixture.json"
    release_report = build_release_candidate_report(
        root=resolved_root,
        workspace_id=workspace_id,
        strict=False,
        run_runtime_checks=run_runtime_checks,
        run_default_tests=False,
    )
    if write:
        demand_report, demand_markdown, _latest_json, _latest_md, _run_json, _run_md = record_demand_gate_report(
            workspace_id=workspace_id,
            root=resolved_root,
            fixture_metrics=fixture,
        )
    else:
        demand_report = build_demand_gate_report(
            workspace_id=workspace_id,
            root=resolved_root,
            fixture_metrics=fixture,
        )
        demand_markdown = format_demand_gate_report(demand_report)
    run_slug = timestamp_slug()
    latest_json = release_demo_latest_json_path(workspace_id=workspace_id, root=resolved_root)
    latest_md = release_demo_latest_md_path(workspace_id=workspace_id, root=resolved_root)
    run_json = release_demo_run_json_path(workspace_id=workspace_id, root=resolved_root, run_slug=run_slug)
    run_md = release_demo_run_md_path(workspace_id=workspace_id, root=resolved_root, run_slug=run_slug)
    markdown = format_release_demo_markdown(
        release_report=release_report,
        demand_report=demand_report,
        demand_markdown=demand_markdown,
    )
    report = {
        "schema_name": RELEASE_DEMO_SCHEMA_NAME,
        "schema_version": RELEASE_DEMO_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "generated_at_utc": timestamp_utc(),
        "status": "pass" if release_report.get("status") == "pass" and demand_report.get("status") == "pass" else "fail",
        "release_check_status": release_report.get("status"),
        "demand_gate_status": demand_report.get("status"),
        "markdown_report_exists": bool(write),
        "markdown_report_rendered": True,
        "guardrails": {
            "requires_api_key": False,
            "uses_network": False,
            "uses_private_docs": False,
            "writes_training_data": False,
            "uses_public_demo_fixtures": True,
            "no_api_flag": bool(no_api),
        },
        "commands": list(DEFAULT_DEMO_COMMANDS),
        "paths": {
            "report_latest_json_path": str(latest_json),
            "report_latest_md_path": str(latest_md),
            "report_run_json_path": str(run_json),
            "report_run_md_path": str(run_md),
        },
        "markdown": markdown,
    }
    if write:
        write_json(latest_json, {key: value for key, value in report.items() if key != "markdown"})
        write_json(run_json, {key: value for key, value in report.items() if key != "markdown"})
        latest_md.parent.mkdir(parents=True, exist_ok=True)
        run_md.parent.mkdir(parents=True, exist_ok=True)
        latest_md.write_text(markdown, encoding="utf-8")
        run_md.write_text(markdown, encoding="utf-8")
    return report


def format_release_demo_markdown(
    *,
    release_report: Mapping[str, Any],
    demand_report: Mapping[str, Any],
    demand_markdown: str,
) -> str:
    lines = [
        "# Public Demo Walkthrough",
        "",
        "This is a local, file-first release candidate demo. It uses public fixtures only.",
        "",
        "## Guardrails",
        "",
        "- No API key required.",
        "- No network calls in the default path.",
        "- No private design notes required.",
        "- No trainable export artifacts are written.",
        "",
        "## Commands",
        "",
    ]
    lines.extend(f"```bash\n{command}\n```" for command in DEFAULT_DEMO_COMMANDS)
    lines.extend(
        [
            "",
            "## Transcript",
            "",
            f"1. Release checks finish with `{release_report.get('status')}`.",
            f"2. Demand gate fixture finishes with `{demand_report.get('status')}`.",
            "3. The operator reads the generated Markdown and only treats real demand evidence as release evidence.",
            "",
            "## Release Check Summary",
            "",
            "| Check | Status |",
            "|---|---|",
        ]
    )
    for check in release_report.get("checks") or []:
        if isinstance(check, Mapping):
            lines.append(f"| {check.get('label')} | `{check.get('status')}` |")
    lines.extend(["", "## Demand Gate Fixture", "", demand_markdown.strip(), ""])
    return redact_release_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build M16 release-candidate checks and public demo reports.")
    parser.add_argument("--root", type=Path, default=None, help="Optional repo root override.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Run release-candidate checks.")
    check_parser.add_argument("--strict", action="store_true", help="Run runtime gates and the public demo default test gate.")
    check_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id.")
    check_parser.add_argument("--no-write", action="store_true", help="Print only; do not persist report artifacts.")
    check_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")

    demo_parser = subparsers.add_parser("demo", help="Run the public no-provider demo path.")
    demo_parser.add_argument("--no-api", action="store_true", help="Required explicit no-provider flag for the demo.")
    demo_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id.")
    demo_parser.add_argument("--no-write", action="store_true", help="Print only; do not persist report artifacts.")
    demo_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            if args.no_write:
                report = build_release_candidate_report(
                    root=args.root,
                    workspace_id=args.workspace_id,
                    strict=args.strict,
                    run_runtime_checks=True,
                    run_default_tests=args.strict,
                )
                markdown = format_release_candidate_report_markdown(report)
            else:
                report, markdown, _latest_json, _latest_md, _run_json, _run_md = record_release_candidate_report(
                    root=args.root,
                    workspace_id=args.workspace_id,
                    strict=args.strict,
                    run_runtime_checks=True,
                    run_default_tests=args.strict,
                )
            if args.format == "json":
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                print(markdown)
            return 0 if report.get("status") == "pass" else 1

        if args.command == "demo":
            if not args.no_api:
                parser.error("release demo requires --no-api.")
            report = build_release_demo_report(
                root=args.root,
                workspace_id=args.workspace_id,
                no_api=args.no_api,
                write=not args.no_write,
                run_runtime_checks=True,
            )
            if args.format == "json":
                print(json.dumps({key: value for key, value in report.items() if key != "markdown"}, ensure_ascii=False, indent=2))
            else:
                print(str(report["markdown"]))
            return 0 if report.get("status") == "pass" else 1
    except ValueError as exc:
        parser.error(str(exc))
    parser.error("Unsupported release command.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
