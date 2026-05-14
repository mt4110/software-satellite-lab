#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from artifact_vault import ARTIFACT_KINDS, capture_artifact, format_artifact_inspection, inspect_artifact
from agent_session_intake import (
    SUPPORTED_AGENT_LABELS,
    build_agent_session_bundle,
    build_pr_bundle,
    format_agent_session_intake_result,
    ingest_agent_session_bundle,
    ingest_agent_session_bundle_path,
)
from backend_adoption_dossier import (
    build_backend_adoption_dossier_from_comparison_id,
    build_backend_adoption_dossier_from_review,
    format_backend_adoption_dossier_markdown,
)
from evidence_graph import (
    build_evidence_graph,
    build_evidence_impact_report,
    build_evidence_trace,
    format_evidence_graph_markdown,
    format_evidence_impact_markdown,
    format_evidence_trace_markdown,
)
from evidence_lint import build_evidence_lint_report, format_evidence_lint_report
from evidence_pack_v1 import (
    EvidencePackV1Error,
    PACK_V1_SCHEMA_NAME,
    audit_evidence_pack_v1_path,
    builtin_pack_list,
    format_evidence_pack_v1_audit_report,
    format_evidence_pack_v1_test_report,
    is_evidence_pack_v1_path,
    load_evidence_pack_v1_manifest,
    lock_evidence_pack_v1_path,
    scaffold_evidence_pack_v1,
    test_evidence_pack_v1_path,
)
from evidence_support import build_evidence_support_result, format_evidence_support_result
from failure_memory_review import (
    build_failure_recall,
    build_review_risk_report,
    build_verdict_template,
    event_contract_summary,
    format_failure_recall_report,
    format_ingest_result,
    format_proposal_comparison_result,
    format_verdict_template,
    format_verdict_result,
    record_latest_review_verdict,
    record_file_input,
    record_human_verdict,
    record_proposal_comparison,
    run_evidence_gated_git_review,
    run_review_risk_pack,
)
from demand_validation import (
    DEMAND_VALIDATION_JUDGEMENTS,
    build_demand_validation_report,
    demand_validation_templates_markdown,
    format_demand_validation_report,
    format_demo_setup_metric,
    format_dogfood_validation_run,
    format_external_user_interview,
    record_demo_setup_metric,
    record_demand_validation_report,
    record_dogfood_validation_run,
    record_external_user_interview,
    write_demand_validation_templates,
)
from demand_gate import (
    build_demand_gate_report,
    format_demand_gate_report,
    record_demand_gate_report,
)
from evaluation_loop import format_learning_dataset_preview_report, record_learning_dataset_preview
from release_candidate_checks import (
    build_release_demo_report,
    build_release_candidate_report,
    format_release_candidate_report_markdown,
    record_release_candidate_report,
)
from review_benchmark import format_review_benchmark_report, run_review_benchmark
from review_memory_eval import (
    format_review_memory_eval_report,
    format_review_memory_miss_report,
    load_or_build_review_memory_miss_report,
    run_review_memory_eval,
)
from satellite_pack import (
    PackManifestError,
    audit_pack_path,
    format_pack_audit_report,
    format_pack_inspection_report,
    inspect_pack_path,
)
from workspace_state import DEFAULT_WORKSPACE_ID


def _backend_dossier_strict_gate_failed(dossier: Mapping[str, Any]) -> bool:
    benchmark_gate = dossier.get("benchmark_gate")
    if not isinstance(benchmark_gate, Mapping):
        return False
    if not benchmark_gate.get("strict_required"):
        return False
    return benchmark_gate.get("available") is False or benchmark_gate.get("passed") is False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="satlab",
        description="Local file-first utilities for software-satellite-lab.",
    )
    parser.add_argument("--root", type=Path, default=None, help="Optional repo root override.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    event_parser = subparsers.add_parser("event", help="Record file-first software-work events.")
    event_subparsers = event_parser.add_subparsers(dest="event_command", required=True)

    ingest_parser = event_subparsers.add_parser(
        "ingest",
        help="Record a patch, failure, repair, or review note as source-linked evidence.",
    )
    input_group = ingest_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--patch", type=Path, help="Patch or git diff file to record.")
    input_group.add_argument("--failure", type=Path, help="Failure log or failure note file to record.")
    input_group.add_argument("--proposal", type=Path, help="Proposal or candidate output file to record.")
    input_group.add_argument("--repair", type=Path, help="Repair note or repair artifact file to record.")
    input_group.add_argument("--review-note", type=Path, help="Accepted/rejected review note file to record.")
    ingest_parser.add_argument("--note", default="", help="Short note to attach to the event.")
    ingest_parser.add_argument("--status", default="", help="Optional event status override.")
    ingest_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    ingest_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    event_report_parser = event_subparsers.add_parser(
        "report",
        help="Report source-path contract status for software-work events.",
    )
    event_report_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    event_report_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    artifact_parser = subparsers.add_parser("artifact", help="Capture and inspect local immutable artifact refs.")
    artifact_subparsers = artifact_parser.add_subparsers(dest="artifact_command", required=True)
    artifact_capture_parser = artifact_subparsers.add_parser(
        "capture",
        help="Capture a local source artifact into the file-first artifact vault.",
    )
    artifact_capture_parser.add_argument("--path", type=Path, required=True, help="Local source artifact path.")
    artifact_capture_parser.add_argument(
        "--kind",
        choices=tuple(sorted(ARTIFACT_KINDS)),
        default="unknown",
        help="Artifact kind.",
    )
    artifact_capture_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    artifact_inspect_parser = artifact_subparsers.add_parser(
        "inspect",
        help="Inspect a captured artifact ref and verify the local vault object when present.",
    )
    artifact_inspect_parser.add_argument("--artifact", required=True, help="Artifact id.")
    artifact_inspect_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    intake_parser = subparsers.add_parser("intake", help="Ingest local file-first artifacts from external work sessions.")
    intake_subparsers = intake_parser.add_subparsers(dest="intake_command", required=True)
    agent_session_parser = intake_subparsers.add_parser(
        "agent-session",
        help="Normalize a local agent session bundle or loose files into software-work evidence.",
    )
    agent_session_parser.add_argument("--bundle", type=Path, default=None, help="Agent session bundle JSON.")
    agent_session_parser.add_argument(
        "--agent",
        choices=SUPPORTED_AGENT_LABELS,
        default="generic",
        help="Agent label metadata only; no provider integration is enabled.",
    )
    agent_session_parser.add_argument("--diff", type=Path, default=None, help="Local diff or patch file.")
    agent_session_parser.add_argument("--transcript", type=Path, default=None, help="Local transcript file.")
    agent_session_parser.add_argument("--test-log", type=Path, default=None, help="Local test log file.")
    agent_session_parser.add_argument("--ci-log", type=Path, default=None, help="Local CI log file.")
    agent_session_parser.add_argument("--note", default="", help="Short note for the intake.")
    agent_session_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    agent_session_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    pr_bundle_parser = intake_subparsers.add_parser(
        "pr-bundle",
        help="Normalize local PR diff, review, and CI log files as a file-first bundle.",
    )
    pr_bundle_parser.add_argument("--diff", type=Path, required=True, help="Local PR diff file.")
    pr_bundle_parser.add_argument("--review", type=Path, default=None, help="Local review note file.")
    pr_bundle_parser.add_argument("--ci-log", type=Path, default=None, help="Local CI log file.")
    pr_bundle_parser.add_argument("--note", default="", help="Short note for the intake.")
    pr_bundle_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    pr_bundle_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    evidence_parser = subparsers.add_parser("evidence", help="Classify whether evidence can support a decision.")
    evidence_subparsers = evidence_parser.add_subparsers(dest="evidence_command", required=True)
    evidence_support_parser = evidence_subparsers.add_parser(
        "support",
        help="Run the Evidence Support Kernel for one software-work event.",
    )
    evidence_support_parser.add_argument("--event", required=True, help="Software-work event id.")
    evidence_support_parser.add_argument(
        "--review-started-at",
        default=None,
        help="Optional ISO-8601 review start time. Evidence at or after this time cannot support the review.",
    )
    evidence_support_parser.add_argument(
        "--active-subject",
        default=None,
        help="Optional active review event id, artifact id, sha256, path, or blob fingerprint to exclude.",
    )
    evidence_support_parser.add_argument(
        "--polarity",
        choices=("positive", "negative", "risk", "diagnostic", "none"),
        default=None,
        help="Optional requested support polarity. Defaults to event-derived polarity.",
    )
    evidence_support_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    evidence_support_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    evidence_graph_parser = evidence_subparsers.add_parser(
        "graph",
        help="Build a derived, rebuildable graph of local evidence relations.",
    )
    evidence_graph_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    evidence_graph_parser.add_argument("--format", choices=("md", "json"), default="json", help="Output format.")

    evidence_lint_parser = evidence_subparsers.add_parser(
        "lint",
        help="Lint the derived evidence graph for decision-support blockers.",
    )
    evidence_lint_parser.add_argument("--strict", action="store_true", help="Return non-zero when hard-fail rules trigger.")
    evidence_lint_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    evidence_lint_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    evidence_trace_parser = evidence_subparsers.add_parser(
        "trace",
        help="Trace why one event can or cannot support a decision.",
    )
    evidence_trace_parser.add_argument("--event", required=True, help="Software-work event id.")
    evidence_trace_parser.add_argument("--why-blocked", action="store_true", help="Show blocker reasons explicitly.")
    evidence_trace_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    evidence_trace_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")

    evidence_impact_parser = evidence_subparsers.add_parser(
        "impact",
        help="List graph evidence affected by a changed path.",
    )
    evidence_impact_parser.add_argument("--path", required=True, help="Repo path to inspect.")
    evidence_impact_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    evidence_impact_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")

    recall_parser = subparsers.add_parser("recall", help="Recall source-linked software-work memory.")
    recall_subparsers = recall_parser.add_subparsers(dest="recall_command", required=True)
    failure_recall_parser = recall_subparsers.add_parser(
        "failure",
        help="Recall similar prior failures for a patch or review query.",
    )
    failure_recall_parser.add_argument("--query", required=True, help="Failure-memory query text.")
    failure_recall_parser.add_argument("--patch", type=Path, default=None, help="Optional patch file for file hints.")
    failure_recall_parser.add_argument("--file-hint", action="append", default=[], help="Optional file hint. Repeatable.")
    failure_recall_parser.add_argument("--source-event-id", default="", help="Optional source event id to diagnose.")
    failure_recall_parser.add_argument("--status-filter", action="append", default=[], help="Optional status filter.")
    failure_recall_parser.add_argument("--limit", type=int, default=5, help="Maximum selected recall items.")
    failure_recall_parser.add_argument("--context-budget-chars", type=int, default=6000, help="Recall bundle budget.")
    failure_recall_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    failure_recall_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    compare_parser = subparsers.add_parser("compare", help="Record source-linked proposal comparisons.")
    compare_subparsers = compare_parser.add_subparsers(dest="compare_command", required=True)
    proposals_parser = compare_subparsers.add_parser(
        "proposals",
        help="Compare two or more proposal/candidate output files with a human rationale.",
    )
    proposals_parser.add_argument(
        "--candidate",
        type=Path,
        action="append",
        required=True,
        help="Candidate output file. Repeat at least twice.",
    )
    proposals_parser.add_argument(
        "--verdict",
        choices=("winner", "none", "tie", "needs-follow-up"),
        required=True,
        help="Human comparison verdict. Use `winner` with --winner-candidate or `none` when no candidate wins.",
    )
    proposals_parser.add_argument(
        "--winner-candidate",
        default="",
        help="Winning candidate path or 1-based index. Required for --verdict winner.",
    )
    proposals_parser.add_argument("--rationale", required=True, help="Human rationale for the comparison verdict.")
    proposals_parser.add_argument("--label", default="", help="Short task label for the comparison.")
    proposals_parser.add_argument("--criterion", action="append", default=[], help="Comparison criterion. Repeatable.")
    proposals_parser.add_argument(
        "--candidate-backend-id",
        action="append",
        default=[],
        help="Optional backend id for a candidate. Repeat once per candidate when provided.",
    )
    proposals_parser.add_argument(
        "--candidate-model-id",
        action="append",
        default=[],
        help="Optional model id for a candidate. Repeat once per candidate when provided.",
    )
    proposals_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    proposals_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    backend_parser = subparsers.add_parser("backend", help="Inspect backend/model adoption evidence without provider hubs.")
    backend_subparsers = backend_parser.add_subparsers(dest="backend_command", required=True)
    backend_dossier_parser = backend_subparsers.add_parser(
        "dossier",
        help="Build a source-linked backend/model adoption dossier from local comparison evidence.",
    )
    dossier_source = backend_dossier_parser.add_mutually_exclusive_group(required=True)
    dossier_source.add_argument("--comparison", default=None, help="Evaluation comparison id to inspect.")
    dossier_source.add_argument("--from-review", default=None, help="Review metadata path or review event/run id to inspect.")
    backend_dossier_parser.add_argument("--candidate-backend", default=None, help="Candidate backend id, model id, or event id.")
    backend_dossier_parser.add_argument("--baseline-backend", default=None, help="Baseline backend id, model id, or event id.")
    backend_dossier_parser.add_argument(
        "--workflow-kind",
        choices=("review_git", "agent_session_intake", "pack_report", "backend_compare", "learning_inspection"),
        default=None,
        help="Scoped workflow kind for the adoption question.",
    )
    backend_dossier_parser.add_argument(
        "--repo-scope",
        choices=("current_repo", "fixture_only", "dogfood_only"),
        default="current_repo",
        help="Repository scope for the adoption question.",
    )
    backend_dossier_parser.add_argument(
        "--risk-scope",
        choices=("experimental", "default_candidate", "default_backend"),
        default="experimental",
        help="Risk scope; default-backend adoption remains separately gated.",
    )
    backend_dossier_parser.add_argument("--rollback-plan", default=None, help="Explicit rollback plan. Empty means absent.")
    backend_dossier_parser.add_argument("--benchmark-report", type=Path, default=None, help="Optional local JSON benchmark report.")
    backend_dossier_parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat missing or failing benchmark evidence as a strict adoption gate.",
    )
    backend_dossier_parser.add_argument(
        "--lint",
        action="store_true",
        help="Also run the local Evidence Lint gate while building the dossier.",
    )
    backend_dossier_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    backend_dossier_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")

    pack_parser = subparsers.add_parser("pack", help="Inspect and audit declarative Satellite Evidence Packs.")
    pack_subparsers = pack_parser.add_subparsers(dest="pack_command", required=True)

    inspect_parser = pack_subparsers.add_parser(
        "inspect",
        help="Load a Satellite Evidence Pack manifest and print its declarative shape.",
    )
    inspect_parser.add_argument("pack", type=Path, help="Manifest file or pack directory.")
    inspect_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    audit_parser = pack_subparsers.add_parser(
        "audit",
        help="Validate a Satellite Evidence Pack manifest and write a permission summary artifact.",
    )
    audit_parser.add_argument("pack", type=Path, help="Manifest file or pack directory.")
    audit_parser.add_argument("--strict", action="store_true", help="Use the strict Evidence Pack v1 policy gate when applicable.")
    audit_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    audit_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    scaffold_parser = pack_subparsers.add_parser(
        "scaffold",
        help="Write a built-in Evidence Pack v1 manifest template.",
    )
    scaffold_parser.add_argument("--kind", choices=("failure-memory", "agent-session"), required=True, help="Built-in pack kind.")
    scaffold_parser.add_argument("--output", type=Path, required=True, help="Output manifest path.")
    scaffold_parser.add_argument("--force", action="store_true", help="Overwrite an existing output with different content.")
    scaffold_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    pack_test_parser = pack_subparsers.add_parser(
        "test",
        help="Run local Evidence Pack v1 benchmark fixtures through the support kernel.",
    )
    pack_test_parser.add_argument("pack", type=Path, help="Manifest file or pack directory.")
    pack_test_parser.add_argument("--strict", action="store_true", help="Block on every policy issue before running fixtures.")
    pack_test_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    pack_test_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    lock_parser = pack_subparsers.add_parser(
        "lock",
        help="Write an Evidence Pack v1 lockfile that detects manifest mutation.",
    )
    lock_parser.add_argument("pack", type=Path, help="Manifest file or pack directory.")
    lock_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    list_parser = pack_subparsers.add_parser(
        "list",
        help="List built-in Evidence Pack v1 templates.",
    )
    list_parser.add_argument("--builtin", action="store_true", required=True, help="List built-in pack templates.")
    list_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    run_parser = pack_subparsers.add_parser(
        "run",
        help="Run the built-in review-risk-pack workflow without enabling arbitrary pack execution.",
    )
    run_parser.add_argument("pack", help="Pack name/path. Only review-risk-pack is runnable in v0.")
    run_parser.add_argument("--patch", type=Path, required=True, help="Patch or git diff file to review.")
    run_parser.add_argument("--query", default="", help="Optional failure-memory query override.")
    run_parser.add_argument("--note", default="", help="Optional note for the patch input event.")
    run_parser.add_argument("--limit", type=int, default=5, help="Maximum selected recall items.")
    run_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    run_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")

    review_parser = subparsers.add_parser("review", help="Run evidence-gated review memory workflows.")
    review_subparsers = review_parser.add_subparsers(dest="review_command", required=True)
    review_git_parser = review_subparsers.add_parser(
        "git",
        help="Capture a git diff and generate an evidence-gated review report.",
    )
    review_git_parser.add_argument("--base", required=True, help="Base git ref.")
    review_git_parser.add_argument("--head", default="HEAD", help="Head git ref.")
    review_git_parser.add_argument("--test-log", type=Path, default=None, help="Optional local test log file.")
    review_git_parser.add_argument("--query", default="", help="Optional recall query override.")
    review_git_parser.add_argument("--note", default="", help="Optional review note.")
    review_git_parser.add_argument("--limit", type=int, default=5, help="Maximum prior evidence rows.")
    review_git_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    review_git_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")

    review_verdict_parser = review_subparsers.add_parser(
        "verdict",
        help="Record a human verdict for the latest evidence-gated review.",
    )
    review_verdict_parser.add_argument(
        "--from-latest",
        action="store_true",
        required=True,
        help="Use the latest review report event without manual event id lookup.",
    )
    review_verdict_parser.add_argument(
        "--decision",
        choices=("accept", "reject", "needs_fix", "needs_more_evidence"),
        required=True,
        help="Human decision for the latest review.",
    )
    review_verdict_parser.add_argument("--rationale", required=True, help="Human rationale for the decision.")
    review_verdict_parser.add_argument("--follow-up", default="", help="Optional follow-up note.")
    review_verdict_parser.add_argument(
        "--recall-usefulness",
        choices=("useful", "irrelevant", "misleading", "not_checked"),
        default="not_checked",
        help="Whether recalled evidence helped the decision.",
    )
    review_verdict_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    review_verdict_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    review_eval_parser = review_subparsers.add_parser(
        "eval",
        help="Run adversarial review-memory fixtures against the local evidence support gates.",
    )
    review_eval_parser.add_argument("--suite", type=Path, default=None, help="Fixture suite file or directory.")
    review_eval_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    review_eval_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")

    review_miss_parser = review_subparsers.add_parser(
        "miss-report",
        help="Print the latest review-memory benchmark miss report.",
    )
    review_miss_source = review_miss_parser.add_mutually_exclusive_group()
    review_miss_source.add_argument("--latest", action="store_true", help="Use the latest review-memory eval artifact.")
    review_miss_source.add_argument("--eval", type=Path, default=None, help="Eval JSON artifact to summarize.")
    review_miss_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    review_miss_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")

    review_benchmark_parser = review_subparsers.add_parser(
        "benchmark",
        help="Run deterministic evidence-gate benchmark fixtures.",
    )
    review_benchmark_parser.add_argument("--suite", type=Path, default=None, help="M12 fixture suite for --spartan.")
    review_benchmark_parser.add_argument(
        "--spartan",
        action="store_true",
        help="Run the M12 adversarial review-memory benchmark.",
    )
    review_benchmark_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    review_benchmark_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")

    verdict_parser = subparsers.add_parser("verdict", help="Record or print a human verdict.")
    verdict_subparsers = verdict_parser.add_subparsers(dest="verdict_command", required=True)
    for name in ("accept", "reject", "resolve", "unresolve", "needs-review"):
        command_parser = verdict_subparsers.add_parser(name, help=f"Record a {name} verdict.")
        command_parser.add_argument("--event", required=True, help="Software-work event id.")
        command_parser.add_argument("--reason", required=True, help="Short human-readable reason.")
        command_parser.add_argument("--target-event", default="", help="Optional related prior event id.")
        command_parser.add_argument(
            "--relation-kind",
            choices=("repairs", "follow_up_for"),
            default=None,
            help="Optional relation to --target-event.",
        )
        command_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
        command_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    template_parser = verdict_subparsers.add_parser("template", help="Print a human verdict template.")
    template_parser.add_argument("--event", default="", help="Optional software-work event id.")
    template_parser.add_argument("--verdict", default="reject", help="Template verdict value.")
    template_parser.add_argument("--reason", default="", help="Optional template reason.")
    template_parser.add_argument("--format", choices=("text", "json"), default="json", help="Output format.")

    report_parser = subparsers.add_parser("report", help="Generate file-first review reports.")
    report_subparsers = report_parser.add_subparsers(dest="report_command", required=True)
    latest_parser = report_subparsers.add_parser(
        "latest",
        help="Generate the latest Markdown review-risk report.",
    )
    latest_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    latest_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")

    learning_parser = subparsers.add_parser("learning", help="Inspect preview-only learning candidates.")
    learning_subparsers = learning_parser.add_subparsers(dest="learning_command", required=True)
    inspect_learning_parser = learning_subparsers.add_parser(
        "inspect",
        help="Build and inspect a preview-only learning-candidate queue.",
    )
    inspect_learning_parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Required safety flag. Writes inspection artifacts only; no trainable export is produced.",
    )
    inspect_learning_parser.add_argument("--limit", type=int, default=None, help="Maximum eligible candidates to preview.")
    inspect_learning_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    inspect_learning_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    validation_parser = subparsers.add_parser(
        "validation",
        help="Record demand-validation dogfood and external-user evidence.",
    )
    validation_subparsers = validation_parser.add_subparsers(dest="validation_command", required=True)
    validation_template_parser = validation_subparsers.add_parser(
        "template",
        help="Print or write local demand-validation note templates.",
    )
    validation_template_parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory for Markdown templates.")
    validation_template_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    validation_run_parser = validation_subparsers.add_parser(
        "record-run",
        help="Record a dogfood run judgement for a recall-backed review event.",
    )
    validation_run_parser.add_argument("--event", required=True, help="Dogfood software-work event id.")
    validation_run_parser.add_argument("--recall", type=Path, default=None, help="Recall artifact path. Defaults to latest recall.")
    validation_run_parser.add_argument(
        "--useful-recall",
        choices=DEMAND_VALIDATION_JUDGEMENTS,
        required=True,
        help="Human judgement for whether recalled evidence helped.",
    )
    validation_run_parser.add_argument(
        "--critical-false-evidence-count",
        type=int,
        default=0,
        help="Count critical false-evidence cases found in this run.",
    )
    validation_run_parser.add_argument(
        "--verdict-capture-seconds",
        type=float,
        default=None,
        help="Seconds needed to capture the human verdict for this run.",
    )
    validation_run_parser.add_argument("--notes-file", type=Path, default=None, help="Optional dogfood notes file.")
    validation_run_parser.add_argument("--note", default="", help="Short note for the validation run.")
    validation_run_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    validation_run_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    validation_interview_parser = validation_subparsers.add_parser(
        "record-interview",
        help="Record external technical-user interview evidence from a local notes file.",
    )
    validation_interview_parser.add_argument("--participant", required=True, help="Participant label or anonymized id.")
    validation_interview_parser.add_argument(
        "--recognized-pain",
        choices=DEMAND_VALIDATION_JUDGEMENTS,
        required=True,
        help="Whether the user recognized the exact pain.",
    )
    validation_interview_parser.add_argument(
        "--wants-to-try",
        choices=DEMAND_VALIDATION_JUDGEMENTS,
        required=True,
        help="Whether the user wants to try it on a repo.",
    )
    validation_interview_parser.add_argument("--notes-file", type=Path, required=True, help="Interview notes file.")
    validation_interview_parser.add_argument("--note", default="", help="Short note for the interview record.")
    validation_interview_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    validation_interview_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    validation_setup_parser = validation_subparsers.add_parser(
        "record-setup",
        help="Record a timed local clone-to-demo setup measurement.",
    )
    validation_setup_parser.add_argument(
        "--clone-to-demo-minutes",
        type=float,
        required=True,
        help="Minutes from fresh clone/setup to demo report.",
    )
    validation_setup_parser.add_argument("--notes-file", type=Path, default=None, help="Optional setup timing notes file.")
    validation_setup_parser.add_argument("--note", default="", help="Short setup note.")
    validation_setup_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    validation_setup_parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    validation_report_parser = validation_subparsers.add_parser(
        "report",
        help="Generate a demand-validation report from local dogfood, verdict, recall, and interview artifacts.",
    )
    validation_report_parser.add_argument("--write", action="store_true", help="Persist latest and run report artifacts.")
    validation_report_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    validation_report_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")

    release_parser = subparsers.add_parser(
        "release",
        help="Run OSS release-candidate checks and the no-provider public demo.",
    )
    release_subparsers = release_parser.add_subparsers(dest="release_command", required=True)
    release_check_parser = release_subparsers.add_parser(
        "check",
        help="Run release-candidate checks.",
    )
    release_check_parser.add_argument("--strict", action="store_true", help="Run runtime gates and the public demo default test gate.")
    release_check_parser.add_argument("--no-write", action="store_true", help="Print only; do not persist report artifacts.")
    release_check_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    release_check_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")

    release_demo_parser = release_subparsers.add_parser(
        "demo",
        help="Run the public no-provider demo path.",
    )
    release_demo_parser.add_argument("--no-api", action="store_true", help="Required explicit no-provider demo flag.")
    release_demo_parser.add_argument("--no-write", action="store_true", help="Print only; do not persist report artifacts.")
    release_demo_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    release_demo_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")

    demand_parser = subparsers.add_parser(
        "demand",
        help="Evaluate demand validation gates.",
    )
    demand_subparsers = demand_parser.add_subparsers(dest="demand_command", required=True)
    demand_gate_parser = demand_subparsers.add_parser(
        "gate",
        help="Evaluate the M16 dogfood and external-user demand gate.",
    )
    demand_gate_parser.add_argument("--fixture-metrics", type=Path, default=None, help="Optional public fixture metrics JSON.")
    demand_gate_parser.add_argument("--no-write", action="store_true", help="Print only; do not persist report artifacts.")
    demand_gate_parser.add_argument("--workspace-id", default=DEFAULT_WORKSPACE_ID, help="Workspace id for artifacts.")
    demand_gate_parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "artifact" and args.artifact_command == "capture":
        result = capture_artifact(args.path, kind=args.kind, root=args.root)
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(
                "\n".join(
                    [
                        f"Artifact: {result['artifact_id']}",
                        f"Kind: {result['kind']}",
                        f"Source state: {result['source_state']}",
                        f"Capture state: {result['capture_state']}",
                        f"SHA-256: {result.get('sha256') or 'n/a'}",
                        "Redaction: best-effort only; do not treat captured logs as safe to upload.",
                    ]
                )
            )
        return 0 if result.get("capture_state") in {"captured", "ref_only"} else 1

    if args.command == "artifact" and args.artifact_command == "inspect":
        try:
            inspection = inspect_artifact(args.artifact, root=args.root)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(inspection, ensure_ascii=False, indent=2))
        else:
            print(format_artifact_inspection(inspection))
        return 0 if inspection.get("object_verified") or inspection.get("ref", {}).get("capture_state") != "captured" else 1

    if args.command == "intake" and args.intake_command == "agent-session":
        has_loose_files = any(
            value is not None
            for value in (args.diff, args.transcript, args.test_log, args.ci_log)
        )
        if args.bundle is not None and has_loose_files:
            parser.error("intake agent-session accepts either --bundle or loose file arguments, not both.")
        if args.bundle is None and not has_loose_files:
            parser.error("intake agent-session requires --bundle or at least one local artifact file.")
        try:
            if args.bundle is not None:
                result = ingest_agent_session_bundle_path(
                    args.bundle,
                    workspace_id=args.workspace_id,
                    root=args.root,
                )
            else:
                bundle = build_agent_session_bundle(
                    agent_label=args.agent,
                    diff=args.diff,
                    transcript=args.transcript,
                    test_log=args.test_log,
                    ci_log=args.ci_log,
                    note=args.note.strip() or None,
                )
                result = ingest_agent_session_bundle(
                    bundle,
                    workspace_id=args.workspace_id,
                    root=args.root,
                )
        except ValueError as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_agent_session_intake_result(result))
        return 0

    if args.command == "intake" and args.intake_command == "pr-bundle":
        try:
            bundle = build_pr_bundle(
                diff=args.diff,
                review=args.review,
                ci_log=args.ci_log,
                note=args.note.strip() or None,
            )
            result = ingest_agent_session_bundle(
                bundle,
                workspace_id=args.workspace_id,
                root=args.root,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_agent_session_intake_result(result))
        return 0

    if args.command == "evidence" and args.evidence_command == "support":
        result = build_evidence_support_result(
            args.event,
            review_started_at=args.review_started_at,
            active_subject=args.active_subject,
            requested_polarity=args.polarity,
            workspace_id=args.workspace_id,
            root=args.root,
        )
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_evidence_support_result(result))
        return 0 if result.get("can_support_decision") else 1

    if args.command == "evidence" and args.evidence_command == "graph":
        graph = build_evidence_graph(
            workspace_id=args.workspace_id,
            root=args.root,
        )
        if args.format == "json":
            print(json.dumps(graph, ensure_ascii=False, indent=2))
        else:
            print(format_evidence_graph_markdown(graph), end="")
        return 0

    if args.command == "evidence" and args.evidence_command == "lint":
        report = build_evidence_lint_report(
            workspace_id=args.workspace_id,
            root=args.root,
            strict=args.strict,
        )
        if args.format == "json":
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(format_evidence_lint_report(report))
        return 1 if args.strict and report.get("verdict") == "fail" else 0

    if args.command == "evidence" and args.evidence_command == "trace":
        trace = build_evidence_trace(
            args.event,
            workspace_id=args.workspace_id,
            root=args.root,
            why_blocked=args.why_blocked,
        )
        if args.format == "json":
            print(json.dumps(trace, ensure_ascii=False, indent=2))
        else:
            print(format_evidence_trace_markdown(trace), end="")
        return 0 if trace.get("found") else 1

    if args.command == "evidence" and args.evidence_command == "impact":
        report = build_evidence_impact_report(
            args.path,
            workspace_id=args.workspace_id,
            root=args.root,
        )
        if args.format == "json":
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(format_evidence_impact_markdown(report), end="")
        return 0

    if args.command == "pack" and args.pack_command == "inspect":
        try:
            inspection = inspect_pack_path(args.pack, root=args.root)
        except PackManifestError as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps({"inspection": inspection}, ensure_ascii=False, indent=2))
        else:
            print(format_pack_inspection_report(inspection))
        return 0 if inspection.get("schema_valid") else 1

    if args.command == "pack" and args.pack_command == "audit":
        try:
            if is_evidence_pack_v1_path(args.pack):
                audit, latest_path, run_path = audit_evidence_pack_v1_path(
                    args.pack,
                    workspace_id=args.workspace_id,
                    root=args.root,
                    strict=args.strict,
                )
                if args.format == "json":
                    print(
                        json.dumps(
                            {
                                "audit": audit,
                                "audit_latest_path": str(latest_path) if latest_path is not None else None,
                                "audit_run_path": str(run_path) if run_path is not None else None,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                else:
                    print(format_evidence_pack_v1_audit_report(audit))
                return 1 if audit.get("verdict") == "block" else 0

            audit, latest_path, run_path = audit_pack_path(
                args.pack,
                workspace_id=args.workspace_id,
                root=args.root,
            )
        except (PackManifestError, EvidencePackV1Error) as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(
                json.dumps(
                    {
                        "audit": audit,
                        "audit_latest_path": str(latest_path),
                        "audit_run_path": str(run_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(format_pack_audit_report(audit))
        return 1 if audit.get("verdict") == "block" else 0

    if args.command == "pack" and args.pack_command == "scaffold":
        try:
            result = scaffold_evidence_pack_v1(args.kind, args.output, overwrite=args.force)
        except EvidencePackV1Error as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            status = result.get("status")
            verb = "Up to date" if status == "unchanged" else "Wrote" if status == "written" else "Overwrote"
            print(f"{verb} {result['kind']} Evidence Pack v1 template: {result['output_path']}")
        return 0

    if args.command == "pack" and args.pack_command == "test":
        try:
            manifest, _manifest_path = load_evidence_pack_v1_manifest(args.pack)
            if manifest.get("schema_name") != PACK_V1_SCHEMA_NAME:
                parser.error(
                    "pack test supports only explicit Evidence Pack v1 manifests "
                    f"(schema_name: {PACK_V1_SCHEMA_NAME})."
                )
            result, latest_path, run_path = test_evidence_pack_v1_path(
                args.pack,
                workspace_id=args.workspace_id,
                root=args.root,
                strict=args.strict,
            )
        except (PackManifestError, EvidencePackV1Error) as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(
                json.dumps(
                    {
                        "test": result,
                        "test_latest_path": str(latest_path) if latest_path is not None else None,
                        "test_run_path": str(run_path) if run_path is not None else None,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(format_evidence_pack_v1_test_report(result))
        return 0 if result.get("passed") else 1

    if args.command == "pack" and args.pack_command == "lock":
        try:
            lock, lock_path = lock_evidence_pack_v1_path(args.pack, root=args.root)
        except (PackManifestError, EvidencePackV1Error) as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps({"lock": lock, "lock_path": str(lock_path)}, ensure_ascii=False, indent=2))
        else:
            print(f"Evidence Pack v1 lock: {lock_path}")
            print(f"Manifest SHA-256: {lock['manifest_sha256']}")
        return 0

    if args.command == "pack" and args.pack_command == "list":
        packs = builtin_pack_list()
        if args.format == "json":
            print(json.dumps({"builtin_packs": packs}, ensure_ascii=False, indent=2))
        else:
            print("\n".join(f"{item['kind']}: {item['template_filename']}" for item in packs))
        return 0

    if args.command == "event" and args.event_command == "ingest":
        input_kind = "patch"
        source_path = args.patch
        if args.failure is not None:
            input_kind = "failure"
            source_path = args.failure
        elif args.proposal is not None:
            input_kind = "proposal"
            source_path = args.proposal
        elif args.repair is not None:
            input_kind = "repair"
            source_path = args.repair
        elif args.review_note is not None:
            input_kind = "review_note"
            source_path = args.review_note
        try:
            result = record_file_input(
                input_kind=input_kind,
                source_path=source_path,
                note=args.note.strip() or None,
                status=args.status.strip() or None,
                workspace_id=args.workspace_id,
                root=args.root,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_ingest_result(result))
        return 0

    if args.command == "event" and args.event_command == "report":
        summary = event_contract_summary(workspace_id=args.workspace_id, root=args.root)
        if args.format == "json":
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            contract = summary["event_contract"]
            metrics = summary["validation_metrics"]
            source_metric = metrics["source_path_completeness"]
            print(
                "\n".join(
                    [
                        f"Workspace: {summary['workspace_id']}",
                        f"Events checked: {contract.get('checked_event_count')}",
                        f"Failed events: {contract.get('failed_event_count')}",
                        f"Source path completeness: {source_metric.get('observed')} (target {source_metric.get('target')})",
                        f"Event log: {summary['event_log_path']}",
                    ]
                )
            )
        return 0

    if args.command == "recall" and args.recall_command == "failure":
        try:
            recall, _latest_path, _run_path = build_failure_recall(
                query=args.query,
                patch_path=args.patch,
                file_hints=args.file_hint,
                limit=args.limit,
                context_budget_chars=args.context_budget_chars,
                workspace_id=args.workspace_id,
                root=args.root,
                source_event_id=args.source_event_id.strip() or None,
                status_filters=args.status_filter,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(recall, ensure_ascii=False, indent=2))
        else:
            print(format_failure_recall_report(recall))
        return 0

    if args.command == "compare" and args.compare_command == "proposals":
        try:
            result, _comparison_log = record_proposal_comparison(
                candidate_paths=args.candidate,
                verdict=args.verdict,
                winner_candidate=args.winner_candidate.strip() or None,
                rationale=args.rationale,
                task_label=args.label.strip() or None,
                criteria=args.criterion,
                candidate_backend_ids=args.candidate_backend_id,
                candidate_model_ids=args.candidate_model_id,
                workspace_id=args.workspace_id,
                root=args.root,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_proposal_comparison_result(result))
        return 0

    if args.command == "backend" and args.backend_command == "dossier":
        try:
            if args.comparison is not None:
                dossier = build_backend_adoption_dossier_from_comparison_id(
                    args.comparison,
                    candidate_backend=args.candidate_backend,
                    baseline_backend=args.baseline_backend,
                    workflow_kind=args.workflow_kind or "backend_compare",
                    repo_scope=args.repo_scope,
                    risk_scope=args.risk_scope,
                    rollback_plan=args.rollback_plan,
                    benchmark_report_path=args.benchmark_report,
                    strict=args.strict,
                    run_evidence_lint=args.lint,
                    workspace_id=args.workspace_id,
                    root=args.root,
                )
            else:
                dossier = build_backend_adoption_dossier_from_review(
                    args.from_review,
                    candidate_backend=args.candidate_backend,
                    baseline_backend=args.baseline_backend,
                    workflow_kind=args.workflow_kind or "review_git",
                    repo_scope=args.repo_scope,
                    risk_scope=args.risk_scope,
                    rollback_plan=args.rollback_plan,
                    benchmark_report_path=args.benchmark_report,
                    strict=args.strict,
                    run_evidence_lint=args.lint,
                    workspace_id=args.workspace_id,
                    root=args.root,
                )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(dossier, ensure_ascii=False, indent=2))
        else:
            print(format_backend_adoption_dossier_markdown(dossier))
        if _backend_dossier_strict_gate_failed(dossier):
            return 1
        return 0

    if args.command == "pack" and args.pack_command == "run":
        try:
            metadata, markdown, _latest_path, _run_path = run_review_risk_pack(
                pack=args.pack,
                patch_path=args.patch,
                query=args.query.strip() or None,
                note=args.note.strip() or None,
                limit=args.limit,
                workspace_id=args.workspace_id,
                root=args.root,
            )
        except (PackManifestError, ValueError) as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
        else:
            print(markdown)
        return 0

    if args.command == "review" and args.review_command == "git":
        try:
            metadata, markdown, _latest_path, _run_path = run_evidence_gated_git_review(
                base=args.base,
                head=args.head,
                test_log=args.test_log,
                query=args.query.strip() or None,
                note=args.note.strip() or None,
                limit=args.limit,
                workspace_id=args.workspace_id,
                root=args.root,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
        else:
            print(markdown)
        return 0

    if args.command == "review" and args.review_command == "verdict":
        try:
            result, _latest_path, _run_path = record_latest_review_verdict(
                decision=args.decision,
                rationale=args.rationale,
                follow_up=args.follow_up.strip() or None,
                recall_usefulness=args.recall_usefulness,
                workspace_id=args.workspace_id,
                root=args.root,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_verdict_result(result))
        return 0

    if args.command == "review" and args.review_command == "eval":
        try:
            report = run_review_memory_eval(
                suite=args.suite,
                workspace_id=args.workspace_id,
                root=args.root,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(format_review_memory_eval_report(report))
        return 0 if report.get("passed") else 1

    if args.command == "review" and args.review_command == "miss-report":
        latest = args.latest or args.eval is None
        try:
            report = load_or_build_review_memory_miss_report(
                latest=latest,
                eval_path=args.eval,
                workspace_id=args.workspace_id,
                root=args.root,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(format_review_memory_miss_report(report))
        return 0 if float(report.get("miss_report_coverage") or 0.0) >= 1.0 else 1

    if args.command == "review" and args.review_command == "benchmark":
        if args.spartan:
            try:
                report = run_review_memory_eval(
                    suite=args.suite,
                    workspace_id=args.workspace_id,
                    root=args.root,
                    spartan=True,
                )
            except ValueError as exc:
                parser.error(str(exc))
            if args.format == "json":
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                print(format_review_memory_eval_report(report, spartan=True))
            return 0 if report.get("passed") else 1
        report = run_review_benchmark(workspace_id=args.workspace_id)
        if args.format == "json":
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(format_review_benchmark_report(report))
        return 0 if report.get("passed") else 1

    if args.command == "learning" and args.learning_command == "inspect":
        if not args.preview_only:
            parser.error("learning inspect requires --preview-only.")
        try:
            preview, latest_path, run_path = record_learning_dataset_preview(
                workspace_id=args.workspace_id,
                root=args.root,
                limit=args.limit,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.format == "json":
            payload = {
                "learning_preview": preview,
                "learning_preview_latest_path": str(latest_path),
                "learning_preview_run_path": str(run_path),
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(format_learning_dataset_preview_report(preview))
        return 0

    if args.command == "validation":
        if args.validation_command == "template":
            if args.output_dir is not None:
                paths = write_demand_validation_templates(args.output_dir, root=args.root)
                if args.format == "json":
                    print(json.dumps({"template_paths": paths}, ensure_ascii=False, indent=2))
                else:
                    print("\n".join([f"{name}: {path}" for name, path in paths.items()]))
                return 0
            template = demand_validation_templates_markdown()
            if args.format == "json":
                print(json.dumps({"template_markdown": template}, ensure_ascii=False, indent=2))
            else:
                print(template)
            return 0

        if args.validation_command == "record-run":
            try:
                result, _latest_path, _run_path, _log_path = record_dogfood_validation_run(
                    event_id=args.event,
                    useful_recall=args.useful_recall,
                    critical_false_evidence_count=args.critical_false_evidence_count,
                    verdict_capture_seconds=args.verdict_capture_seconds,
                    recall_path=args.recall,
                    note=args.note.strip() or None,
                    notes_file=args.notes_file,
                    workspace_id=args.workspace_id,
                    root=args.root,
                )
            except ValueError as exc:
                parser.error(str(exc))
            if args.format == "json":
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(format_dogfood_validation_run(result))
            return 0

        if args.validation_command == "record-interview":
            try:
                result, _latest_path, _run_path, _log_path = record_external_user_interview(
                    participant_label=args.participant,
                    recognized_pain=args.recognized_pain,
                    wants_to_try=args.wants_to_try,
                    notes_file=args.notes_file,
                    note=args.note.strip() or None,
                    workspace_id=args.workspace_id,
                    root=args.root,
                )
            except ValueError as exc:
                parser.error(str(exc))
            if args.format == "json":
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(format_external_user_interview(result))
            return 0

        if args.validation_command == "record-setup":
            try:
                result, _latest_path, _run_path = record_demo_setup_metric(
                    clone_to_demo_minutes=args.clone_to_demo_minutes,
                    note=args.note.strip() or None,
                    notes_file=args.notes_file,
                    workspace_id=args.workspace_id,
                    root=args.root,
                )
            except ValueError as exc:
                parser.error(str(exc))
            if args.format == "json":
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(format_demo_setup_metric(result))
            return 0

        if args.validation_command == "report":
            if args.write:
                report, markdown, _latest_json, _latest_md, _run_json, _run_md = record_demand_validation_report(
                    workspace_id=args.workspace_id,
                    root=args.root,
                )
            else:
                report = build_demand_validation_report(workspace_id=args.workspace_id, root=args.root)
                markdown = format_demand_validation_report(report)
            if args.format == "json":
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                print(markdown)
            return 0

    if args.command == "release":
        if args.release_command == "check":
            if args.no_write:
                report = build_release_candidate_report(
                    workspace_id=args.workspace_id,
                    root=args.root,
                    strict=args.strict,
                    run_runtime_checks=True,
                    run_default_tests=args.strict,
                )
                markdown = format_release_candidate_report_markdown(report)
            else:
                report, markdown, _latest_json, _latest_md, _run_json, _run_md = record_release_candidate_report(
                    workspace_id=args.workspace_id,
                    root=args.root,
                    strict=args.strict,
                    run_runtime_checks=True,
                    run_default_tests=args.strict,
                )
            if args.format == "json":
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                print(markdown)
            return 0 if report.get("status") == "pass" else 1

        if args.release_command == "demo":
            if not args.no_api:
                parser.error("release demo requires --no-api.")
            try:
                report = build_release_demo_report(
                    workspace_id=args.workspace_id,
                    root=args.root,
                    no_api=args.no_api,
                    write=not args.no_write,
                    run_runtime_checks=True,
                )
            except ValueError as exc:
                parser.error(str(exc))
            if args.format == "json":
                print(
                    json.dumps(
                        {key: value for key, value in report.items() if key != "markdown"},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                print(str(report["markdown"]))
            return 0 if report.get("status") == "pass" else 1

    if args.command == "demand":
        if args.demand_command == "gate":
            try:
                if args.no_write:
                    report = build_demand_gate_report(
                        workspace_id=args.workspace_id,
                        root=args.root,
                        fixture_metrics=args.fixture_metrics,
                    )
                    markdown = format_demand_gate_report(report)
                else:
                    report, markdown, _latest_json, _latest_md, _run_json, _run_md = record_demand_gate_report(
                        workspace_id=args.workspace_id,
                        root=args.root,
                        fixture_metrics=args.fixture_metrics,
                    )
            except ValueError as exc:
                parser.error(str(exc))
            if args.format == "json":
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                print(markdown)
            return 0 if report.get("status") == "pass" else 1

    if args.command == "verdict":
        if args.verdict_command == "template":
            template = build_verdict_template(
                event_id=args.event.strip() or None,
                verdict=args.verdict,
                reason=args.reason.strip() or None,
            )
            if args.format == "json":
                print(json.dumps(template, ensure_ascii=False, indent=2))
            else:
                print(format_verdict_template(template))
            return 0
        try:
            result, _latest_path, _run_path = record_human_verdict(
                verdict=args.verdict_command,
                event_id=args.event,
                reason=args.reason,
                target_event_id=args.target_event.strip() or None,
                relation_kind=args.relation_kind,
                workspace_id=args.workspace_id,
                root=args.root,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_verdict_result(result))
        return 0

    if args.command == "report" and args.report_command == "latest":
        metadata, markdown, _latest_path, _run_path = build_review_risk_report(
            workspace_id=args.workspace_id,
            root=args.root,
        )
        if args.format == "json":
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
        else:
            print(markdown)
        return 0

    parser.error("Unsupported command.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
