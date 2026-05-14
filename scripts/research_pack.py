#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

from evidence_graph import build_evidence_graph
from evidence_lint import build_evidence_lint_report, format_evidence_lint_report
from evidence_pack_v1 import audit_evidence_pack_v1_path
from gemma_runtime import repo_root, timestamp_slug
from release_candidate_checks import build_release_candidate_report, format_release_candidate_report_markdown
from review_benchmark import run_review_benchmark
from schema_coverage import build_schema_coverage_report, format_schema_coverage_report


RESEARCH_PACK_SCHEMA_NAME = "software-satellite-research-pack"
RESEARCH_PACK_SCHEMA_VERSION = 1
RESEARCH_REPRODUCTION_SCHEMA_NAME = "software-satellite-research-pack-reproduction"
RESEARCH_REPRODUCTION_SCHEMA_VERSION = 1
RESEARCH_PACK_CHECKSUM_SCHEMA_NAME = "software-satellite-research-pack-checksums"
REPRODUCIBLE_GENERATED_AT_UTC = "1970-01-01T00:00:00+00:00"
DEFAULT_RESEARCH_WORKSPACE_ID = "research-pack-public-fixtures"
DEFAULT_RESEARCH_PACK_OUTPUT = Path("artifacts/research_pack")
SCHEMA_COVERAGE_EXIT_THRESHOLD = 0.90
PROJECT_SUMMARY_SENTENCE = (
    "software-satellite-lab is a local-first, file-first recorder for "
    "software-work evidence and review reproducibility."
)

PUBLIC_DEMO_SOURCES = (
    "README.md",
    "README_EN.md",
    "docs/public_demo_walkthrough.md",
    "docs/release_v0_1_candidate.md",
    "examples/demand_gate/release_candidate_fixture.json",
)
BENCHMARK_FIXTURE_SOURCES = (
    "examples/review_memory_benchmark/synthetic_suite.json",
    "templates/failure-memory-pack.satellite.yaml",
    "templates/agent-session-pack.satellite.yaml",
)
STRICT_PACKS = (
    "templates/failure-memory-pack.satellite.yaml",
    "templates/agent-session-pack.satellite.yaml",
)
REQUIRED_RESEARCH_PACK_FILES = (
    "README.md",
    "demo_artifacts/README.md",
    "demo_artifacts/README_EN.md",
    "demo_artifacts/docs/public_demo_walkthrough.md",
    "demo_artifacts/docs/release_v0_1_candidate.md",
    "demo_artifacts/examples/demand_gate/release_candidate_fixture.json",
    "benchmark_fixtures/examples/review_memory_benchmark/synthetic_suite.json",
    "benchmark_fixtures/templates/agent-session-pack.satellite.yaml",
    "benchmark_fixtures/templates/failure-memory-pack.satellite.yaml",
    "benchmark_results.json",
    "evidence_graph_snapshot.json",
    "evidence_lint_report.md",
    "pack_audit_report.md",
    "release_check_report.md",
    "schema_coverage_report.md",
    "schema_coverage_report.json",
    "limitations.md",
    "manifest.json",
    "checksums.json",
)

PRIVATE_DOC_PATTERNS = (
    re.compile(r"\.private_docs\b"),
    re.compile(r"\bprivate_docs/"),
    re.compile(r"software_satellite_lab_m10_m17_spartan_design", re.IGNORECASE),
)
TRAINABLE_EXPORT_PATH_PATTERN = re.compile(
    r"(?:^|/)(?:training[_-]?exports?|trainable[_-]?exports?|fine[_-]?tune|finetune)(?:/|$)|"
    r"(?:training[_-]?export|trainable|fine[_-]?tune|finetune).*\.jsonl$",
    re.IGNORECASE,
)


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def _resolve_output_path(output: Path, *, root: Path) -> Path:
    path = Path(output).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pack_relative(path: Path, *, pack_dir: Path) -> str:
    return path.relative_to(pack_dir).as_posix()


def _iter_pack_files(pack_dir: Path, *, include_checksums: bool = False) -> list[Path]:
    files = []
    if not pack_dir.exists():
        return files
    for path in sorted(pack_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = _pack_relative(path, pack_dir=pack_dir)
        if not include_checksums and rel == "checksums.json":
            continue
        files.append(path)
    return files


def _copy_source_file(root: Path, source: str, destination_root: Path) -> str:
    source_path = root / source
    if not source_path.is_file():
        raise ValueError(f"Required public research source is missing: `{source}`.")
    destination = destination_root / source
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return destination.relative_to(destination_root).as_posix()


def _copy_public_sources(root: Path, pack_dir: Path) -> dict[str, list[str]]:
    demo_files = [
        _copy_source_file(root, source, pack_dir / "demo_artifacts")
        for source in PUBLIC_DEMO_SOURCES
    ]
    benchmark_files = [
        _copy_source_file(root, source, pack_dir / "benchmark_fixtures")
        for source in BENCHMARK_FIXTURE_SOURCES
    ]
    return {"demo_artifacts": demo_files, "benchmark_fixtures": benchmark_files}


def _stable_benchmark_results(report: Mapping[str, Any]) -> dict[str, Any]:
    cases = []
    for item in report.get("cases") or []:
        if not isinstance(item, Mapping):
            continue
        cases.append(
            {
                "case": item.get("case"),
                "passed": bool(item.get("passed")),
                "critical_false_evidence_count": int(item.get("critical_false_evidence_count") or 0),
            }
        )
    cases.sort(key=lambda item: str(item.get("case") or ""))
    return {
        "schema_name": report.get("schema_name") or "software-satellite-review-benchmark",
        "schema_version": report.get("schema_version") or 1,
        "generated_at_utc": REPRODUCIBLE_GENERATED_AT_UTC,
        "workspace_id": DEFAULT_RESEARCH_WORKSPACE_ID,
        "case_count": int(report.get("case_count") or len(cases)),
        "passed_count": int(report.get("passed_count") or sum(1 for item in cases if item.get("passed"))),
        "critical_false_evidence_count": int(report.get("critical_false_evidence_count") or 0),
        "passed": bool(report.get("passed")),
        "training_export_ready": bool(report.get("training_export_ready")),
        "cases": cases,
    }


def _build_pack_audit_summary(root: Path) -> dict[str, Any]:
    audits: list[dict[str, Any]] = []
    for pack in STRICT_PACKS:
        audit, _latest_path, _run_path = audit_evidence_pack_v1_path(
            root / pack,
            root=root,
            strict=True,
            write_artifact=False,
        )
        security_checks = [
            {
                "check_id": check.get("check_id"),
                "status": check.get("status"),
            }
            for check in audit.get("security_checks") or []
            if isinstance(check, Mapping)
        ]
        audits.append(
            {
                "pack_path": pack,
                "pack_id": audit.get("pack_id"),
                "verdict": audit.get("verdict"),
                "schema_valid": bool((audit.get("validation") or {}).get("schema_valid"))
                if isinstance(audit.get("validation"), Mapping)
                else False,
                "blocked_reasons": [str(item) for item in audit.get("blocked_reasons") or []],
                "security_checks": security_checks,
            }
        )
    return {
        "schema_name": "software-satellite-research-pack-audit-summary",
        "schema_version": 1,
        "generated_at_utc": REPRODUCIBLE_GENERATED_AT_UTC,
        "audits": audits,
        "passed": all(item.get("verdict") == "pass" for item in audits),
    }


def _format_pack_audit_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Evidence Pack Audit Summary",
        "",
        f"- Status: `{'pass' if summary.get('passed') else 'fail'}`",
        f"- Generated: `{summary.get('generated_at_utc')}`",
        "",
        "| Pack | Verdict | Schema | Blockers |",
        "|---|---|---|---:|",
    ]
    for audit in summary.get("audits") or []:
        if not isinstance(audit, Mapping):
            continue
        blockers = len(audit.get("blocked_reasons") or [])
        schema_status = "valid" if audit.get("schema_valid") else "invalid"
        lines.append(
            f"| `{audit.get('pack_path')}` | `{audit.get('verdict')}` | `{schema_status}` | {blockers} |"
        )
    lines.extend(
        [
            "",
            "The audit is strict, local, and limited to built-in public Evidence Pack v1 templates.",
            "",
        ]
    )
    return "\n".join(lines)


def _research_pack_readme() -> str:
    return "\n".join(
        [
            "# Research Pack",
            "",
            PROJECT_SUMMARY_SENTENCE,
            "",
            "This pack contains public fixtures, deterministic benchmark summaries, "
            "schema coverage, and gate reports for external inspection.",
            "",
            "Default reproduction uses local files only: no private design notes, "
            "no API key, no network calls, and no trainable export artifacts.",
            "",
        ]
    )


def _limitations_markdown() -> str:
    return "\n".join(
        [
            "# Research Reproducibility Limitations",
            "",
            "- The pack is a reproducibility artifact, not a paper or benchmark leaderboard.",
            "- Benchmark results are summarized from public deterministic fixtures and omit volatile local event ids.",
            "- Evidence graph snapshots in this pack are derived from public fixture scope only.",
            "- Evidence Pack v1 execution remains limited to core-owned transforms.",
            "- Learning-candidate inspection is preview-only; no trainable export artifacts are included.",
            "- Backend/model adoption still requires source-linked local comparison evidence and rollback notes.",
            "- Live provider integrations, network access, and private design notes are outside the default path.",
            "",
            "Research questions preserved:",
            "",
            "1. What makes a software-work record valid evidence?",
            "2. How should negative evidence be represented without poisoning useful recall?",
            "3. What is the right benchmark for failure-memory review?",
            "4. How do human verdicts change recall quality over time?",
            "5. Can cross-agent session artifacts be normalized without live integrations?",
            "6. What evidence is sufficient for backend/model adoption?",
            "7. When, if ever, can learning-candidate inspection safely become exportable?",
            "",
            "Release artifact preparation note: a future GitHub or Zenodo release should "
            "attach this pack as a static artifact after the same reproduction gate passes locally.",
            "",
        ]
    )


def build_research_pack_checksums(pack_dir: Path) -> dict[str, Any]:
    files = []
    for path in _iter_pack_files(pack_dir, include_checksums=False):
        rel = _pack_relative(path, pack_dir=pack_dir)
        files.append(
            {
                "path": rel,
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    files.sort(key=lambda item: str(item["path"]))
    combined_input = "\n".join(
        f"{item['path']} {item['sha256']} {item['size_bytes']}" for item in files
    )
    return {
        "schema_name": RESEARCH_PACK_CHECKSUM_SCHEMA_NAME,
        "schema_version": RESEARCH_PACK_SCHEMA_VERSION,
        "generated_at_utc": REPRODUCIBLE_GENERATED_AT_UTC,
        "algorithm": "sha256",
        "file_count": len(files),
        "combined_sha256": hashlib.sha256(combined_input.encode("utf-8")).hexdigest(),
        "files": files,
    }


def _scan_private_doc_references(pack_dir: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in _iter_pack_files(pack_dir, include_checksums=True):
        rel = _pack_relative(path, pack_dir=pack_dir)
        text = _read_text(path)
        if not text:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in PRIVATE_DOC_PATTERNS):
                findings.append({"path": rel, "line": line_number, "excerpt": line.strip()[:200]})
    return findings


def _scan_trainable_export_artifacts(pack_dir: Path) -> list[str]:
    findings = []
    for path in _iter_pack_files(pack_dir, include_checksums=True):
        rel = _pack_relative(path, pack_dir=pack_dir)
        if TRAINABLE_EXPORT_PATH_PATTERN.search(rel):
            findings.append(rel)
    return findings


def _readme_one_sentence_ok(pack_dir: Path) -> dict[str, Any]:
    text = _read_text(pack_dir / "README.md")
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    first = lines[0] if lines else ""
    sentence_count = len([part for part in re.split(r"[.!?]+", first) if part.strip()])
    clear = (
        bool(first)
        and sentence_count == 1
        and ("local-first" in first.lower() or "file-first" in first.lower())
        and "evidence" in first.lower()
    )
    return {"line": first, "clear": clear, "sentence_count": sentence_count}


def _check(check_id: str, label: str, passed: bool, detail: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": check_id,
        "label": label,
        "status": "pass" if passed else "fail",
        "detail": dict(detail or {}),
    }


def reproduce_research_pack(pack: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    pack_path = Path(pack).expanduser()
    if not pack_path.is_absolute():
        pack_path = resolved_root / pack_path
    pack_dir = pack_path.resolve()
    required_missing = [path for path in REQUIRED_RESEARCH_PACK_FILES if not (pack_dir / path).is_file()]
    private_findings = _scan_private_doc_references(pack_dir)
    trainable_findings = _scan_trainable_export_artifacts(pack_dir)
    readme = _readme_one_sentence_ok(pack_dir)
    benchmark = _read_json_mapping(pack_dir / "benchmark_results.json")
    schema_coverage = _read_json_mapping(pack_dir / "schema_coverage_report.json")
    checksums = _read_json_mapping(pack_dir / "checksums.json")
    actual_checksums = build_research_pack_checksums(pack_dir)
    checksum_match = (
        bool(checksums)
        and checksums.get("combined_sha256") == actual_checksums.get("combined_sha256")
        and checksums.get("files") == actual_checksums.get("files")
    )
    schema_coverage_core = float(schema_coverage.get("core_coverage_ratio") or 0.0)
    benchmark_results_included = (
        benchmark.get("passed") is not None
        and int(benchmark.get("case_count") or 0) > 0
        and benchmark.get("training_export_ready") is False
    )
    checks = [
        _check(
            "required_files_present",
            "Required research pack files are present",
            not required_missing,
            {"missing": required_missing},
        ),
        _check(
            "checksums_stable",
            "Pack checksums match generated contents",
            checksum_match,
            {
                "expected": checksums.get("combined_sha256"),
                "actual": actual_checksums.get("combined_sha256"),
            },
        ),
        _check(
            "readme_one_sentence",
            "Research pack README has a one-sentence project summary",
            bool(readme.get("clear")),
            readme,
        ),
        _check(
            "no_private_docs",
            "No private docs are required or referenced",
            not private_findings,
            {
                "private_doc_dependency_count": len(private_findings),
                "findings": private_findings,
            },
        ),
        _check(
            "benchmark_results_included",
            "Benchmark results are included",
            benchmark_results_included,
            {"case_count": benchmark.get("case_count"), "passed": benchmark.get("passed")},
        ),
        _check(
            "evidence_lint_report_included",
            "Evidence lint report is included",
            (pack_dir / "evidence_lint_report.md").is_file(),
        ),
        _check(
            "pack_audit_report_included",
            "Pack audit report is included",
            (pack_dir / "pack_audit_report.md").is_file(),
        ),
        _check(
            "release_check_report_included",
            "Release check report is included",
            (pack_dir / "release_check_report.md").is_file(),
        ),
        _check("limitations_included", "Limitations file is included", (pack_dir / "limitations.md").is_file()),
        _check(
            "schema_coverage_core",
            "Core schema coverage meets threshold",
            schema_coverage_core >= SCHEMA_COVERAGE_EXIT_THRESHOLD,
            {
                "core_coverage_ratio": schema_coverage_core,
                "threshold": SCHEMA_COVERAGE_EXIT_THRESHOLD,
            },
        ),
        _check(
            "no_trainable_export",
            "No trainable export artifact is included",
            not trainable_findings,
            {"findings": trainable_findings},
        ),
    ]
    failing = [check for check in checks if check.get("status") != "pass"]
    exit_gate = {
        "research_pack_reproduces": not failing,
        "private_doc_dependency_count": len(private_findings),
        "schema_coverage_core": schema_coverage_core,
        "benchmark_results_included": benchmark_results_included,
        "limitations_included": (pack_dir / "limitations.md").is_file(),
        "no_trainable_export": not trainable_findings,
    }
    return {
        "schema_name": RESEARCH_REPRODUCTION_SCHEMA_NAME,
        "schema_version": RESEARCH_REPRODUCTION_SCHEMA_VERSION,
        "generated_at_utc": REPRODUCIBLE_GENERATED_AT_UTC,
        "repo_root": str(resolved_root),
        "pack_path": str(pack_dir),
        "status": "pass" if not failing else "fail",
        "checks": checks,
        "failing_check_ids": [str(check.get("id")) for check in failing],
        "exit_gate": exit_gate,
        **exit_gate,
        "checksums": {
            "expected_combined_sha256": checksums.get("combined_sha256"),
            "actual_combined_sha256": actual_checksums.get("combined_sha256"),
            "match": checksum_match,
        },
    }


def format_reproduction_report(report: Mapping[str, Any]) -> str:
    gate = report.get("exit_gate") if isinstance(report.get("exit_gate"), Mapping) else {}
    lines = [
        "# Research Pack Reproduction",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Pack: `{report.get('pack_path')}`",
        f"- research_pack_reproduces: `{str(bool(gate.get('research_pack_reproduces'))).lower()}`",
        f"- private_doc_dependency_count: `{int(gate.get('private_doc_dependency_count') or 0)}`",
        f"- schema_coverage_core: `{float(gate.get('schema_coverage_core') or 0.0):.2f}`",
        f"- benchmark_results_included: `{str(bool(gate.get('benchmark_results_included'))).lower()}`",
        f"- limitations_included: `{str(bool(gate.get('limitations_included'))).lower()}`",
        f"- no_trainable_export: `{str(bool(gate.get('no_trainable_export'))).lower()}`",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    for check in report.get("checks") or []:
        if isinstance(check, Mapping):
            lines.append(f"| {check.get('label')} | `{check.get('status')}` |")
    failing = [str(item) for item in report.get("failing_check_ids") or []]
    if failing:
        lines.extend(["", "## Failing Checks", ""])
        lines.extend(f"- `{item}`" for item in failing)
    return "\n".join(lines) + "\n"


def build_research_pack(
    *,
    output: Path = DEFAULT_RESEARCH_PACK_OUTPUT,
    root: Path | None = None,
    benchmark_results_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_root = _resolve_root(root)
    output_root = _resolve_output_path(output, root=resolved_root)
    run_id = timestamp_slug()
    run_dir = output_root / "runs" / run_id
    latest_dir = output_root / "latest"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    copied_sources = _copy_public_sources(resolved_root, run_dir)
    benchmark_source = (
        dict(benchmark_results_override)
        if benchmark_results_override is not None
        else run_review_benchmark(workspace_id=DEFAULT_RESEARCH_WORKSPACE_ID)
    )
    benchmark_results = _stable_benchmark_results(benchmark_source)
    graph = build_evidence_graph(
        root=resolved_root,
        workspace_id=DEFAULT_RESEARCH_WORKSPACE_ID,
        events=[],
        signals=[],
        comparisons=[],
        recalls=[],
        learning_previews=[],
        pack_audits=[],
        review_reports=[],
        generated_at_utc=REPRODUCIBLE_GENERATED_AT_UTC,
    )
    lint_report = build_evidence_lint_report(
        graph,
        root=resolved_root,
        workspace_id=DEFAULT_RESEARCH_WORKSPACE_ID,
        strict=True,
        generated_at_utc=REPRODUCIBLE_GENERATED_AT_UTC,
    )
    pack_audit_summary = _build_pack_audit_summary(resolved_root)
    release_report = build_release_candidate_report(
        root=resolved_root,
        workspace_id=DEFAULT_RESEARCH_WORKSPACE_ID,
        strict=False,
        run_runtime_checks=False,
        run_default_tests=False,
    )
    release_report["generated_at_utc"] = REPRODUCIBLE_GENERATED_AT_UTC
    schema_coverage = build_schema_coverage_report(
        root=resolved_root,
        generated_at_utc=REPRODUCIBLE_GENERATED_AT_UTC,
    )
    manifest = {
        "schema_name": RESEARCH_PACK_SCHEMA_NAME,
        "schema_version": RESEARCH_PACK_SCHEMA_VERSION,
        "generated_at_utc": REPRODUCIBLE_GENERATED_AT_UTC,
        "pack_id": "research-reproducibility-and-schema-standardization",
        "local_first": True,
        "file_first": True,
        "requires_api_key": False,
        "uses_network": False,
        "uses_private_docs": False,
        "public_fixtures_only": True,
        "no_trainable_export": True,
        "schema_coverage_core": schema_coverage.get("core_coverage_ratio"),
        "benchmark_results_included": True,
        "limitations_included": True,
        "contents": list(REQUIRED_RESEARCH_PACK_FILES),
        "copied_sources": copied_sources,
    }

    _write_text(run_dir / "README.md", _research_pack_readme())
    _write_json(run_dir / "benchmark_results.json", benchmark_results)
    _write_json(run_dir / "evidence_graph_snapshot.json", graph)
    _write_text(run_dir / "evidence_lint_report.md", format_evidence_lint_report(lint_report) + "\n")
    _write_text(run_dir / "pack_audit_report.md", _format_pack_audit_summary(pack_audit_summary))
    _write_text(run_dir / "release_check_report.md", format_release_candidate_report_markdown(release_report))
    _write_json(run_dir / "schema_coverage_report.json", schema_coverage)
    _write_text(run_dir / "schema_coverage_report.md", format_schema_coverage_report(schema_coverage))
    _write_text(run_dir / "limitations.md", _limitations_markdown())
    _write_json(run_dir / "manifest.json", manifest)
    checksums = build_research_pack_checksums(run_dir)
    _write_json(run_dir / "checksums.json", checksums)

    reproduction = reproduce_research_pack(run_dir, root=resolved_root)
    if latest_dir.exists():
        if not latest_dir.is_dir():
            raise ValueError(f"Latest research pack path exists and is not a directory: `{latest_dir}`.")
        shutil.rmtree(latest_dir)
    latest_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, latest_dir)
    return {
        "schema_name": RESEARCH_PACK_SCHEMA_NAME,
        "schema_version": RESEARCH_PACK_SCHEMA_VERSION,
        "generated_at_utc": REPRODUCIBLE_GENERATED_AT_UTC,
        "status": "pass" if reproduction.get("status") == "pass" else "fail",
        "pack_path": str(run_dir),
        "latest_pack_path": str(latest_dir),
        "checksums": checksums,
        "reproduction": reproduction,
        "exit_gate": reproduction.get("exit_gate"),
    }


def format_research_pack_result(result: Mapping[str, Any]) -> str:
    reproduction = result.get("reproduction") if isinstance(result.get("reproduction"), Mapping) else {}
    gate = reproduction.get("exit_gate") if isinstance(reproduction.get("exit_gate"), Mapping) else {}
    checksums = result.get("checksums") if isinstance(result.get("checksums"), Mapping) else {}
    combined_sha256 = checksums.get("combined_sha256") or "unknown"
    lines = [
        "# Research Pack",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Latest: `{result.get('latest_pack_path')}`",
        f"- Run: `{result.get('pack_path')}`",
        f"- Combined SHA-256: `{combined_sha256}`",
        "",
        "## Exit Gate",
        "",
        f"- research_pack_reproduces: `{str(bool(gate.get('research_pack_reproduces'))).lower()}`",
        f"- private_doc_dependency_count: `{int(gate.get('private_doc_dependency_count') or 0)}`",
        f"- schema_coverage_core: `{float(gate.get('schema_coverage_core') or 0.0):.2f}`",
        f"- benchmark_results_included: `{str(bool(gate.get('benchmark_results_included'))).lower()}`",
        f"- limitations_included: `{str(bool(gate.get('limitations_included'))).lower()}`",
        f"- no_trainable_export: `{str(bool(gate.get('no_trainable_export'))).lower()}`",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and reproduce the M17 research-quality OSS artifact pack.")
    parser.add_argument("--root", type=Path, default=None, help="Optional repo root override.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_parser = subparsers.add_parser("pack", help="Generate a reproducible research pack from public fixtures.")
    pack_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESEARCH_PACK_OUTPUT,
        help="Research pack output root.",
    )
    pack_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")

    reproduce_parser = subparsers.add_parser("reproduce", help="Reproduce and validate an existing research pack.")
    reproduce_parser.add_argument(
        "--pack",
        type=Path,
        required=True,
        help="Research pack directory, usually artifacts/research_pack/latest.",
    )
    reproduce_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            result = build_research_pack(output=args.output, root=args.root)
            if args.format == "json":
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print(format_research_pack_result(result))
            return 0 if result.get("status") == "pass" else 1
        if args.command == "reproduce":
            report = reproduce_research_pack(args.pack, root=args.root)
            if args.format == "json":
                print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print(format_reproduction_report(report), end="")
            return 0 if report.get("status") == "pass" else 1
    except ValueError as exc:
        parser.error(str(exc))
    parser.error("Unsupported research pack command.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
