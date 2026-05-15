#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TestGate:
    path: str
    timeout_seconds: int
    budget_seconds: float


PUBLIC_DEMO_GATES = (
    TestGate("tests/test_satellite_pack.py", timeout_seconds=45, budget_seconds=15.0),
    TestGate("tests/test_failure_memory_review.py", timeout_seconds=60, budget_seconds=20.0),
    TestGate("tests/test_review_benchmark.py", timeout_seconds=60, budget_seconds=20.0),
    TestGate("tests/test_evaluation_loop.py", timeout_seconds=90, budget_seconds=35.0),
    TestGate("tests/test_backend_swap.py", timeout_seconds=120, budget_seconds=60.0),
    TestGate("tests/test_demand_validation.py", timeout_seconds=90, budget_seconds=30.0),
    TestGate("tests/test_pilot_evidence.py", timeout_seconds=60, budget_seconds=20.0),
    TestGate("tests/test_release_candidate_checks.py", timeout_seconds=60, budget_seconds=20.0),
)


def run_gate(gate: TestGate) -> tuple[bool, str]:
    started = time.perf_counter()
    command = [sys.executable, "-m", "unittest", gate.path]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=gate.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return False, (
            f"{gate.path}: timed out after {gate.timeout_seconds}s\n"
            f"{exc.stdout or ''}{exc.stderr or ''}"
        )

    elapsed = time.perf_counter() - started
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        return False, f"{gate.path}: failed in {elapsed:.2f}s\n{output}"
    if elapsed > gate.budget_seconds:
        return False, (
            f"{gate.path}: exceeded performance budget "
            f"{elapsed:.2f}s > {gate.budget_seconds:.2f}s\n{output}"
        )
    return True, f"{gate.path}: ok {elapsed:.2f}s <= {gate.budget_seconds:.2f}s"


def main() -> int:
    failures: list[str] = []
    for gate in PUBLIC_DEMO_GATES:
        ok, message = run_gate(gate)
        print(message)
        if not ok:
            failures.append(message)

    if failures:
        print("\nPublic demo checks failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure.splitlines()[0]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
