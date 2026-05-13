from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evidence_pack_v1 import (  # noqa: E402
    ALLOWED_CORE_TRANSFORMS,
    ALLOWED_INPUT_KINDS,
    PACK_V1_SCHEMA_NAME,
    audit_evidence_pack_v1_path,
    build_evidence_pack_v1_audit,
    evidence_pack_v1_lock_path,
    is_evidence_pack_v1_path,
    lock_evidence_pack_v1_path,
    scaffold_evidence_pack_v1,
    test_evidence_pack_v1_path as run_evidence_pack_v1_test,
)
from satellite_pack import load_pack_manifest  # noqa: E402


FAILURE_PACK = REPO_ROOT / "templates" / "failure-memory-pack.satellite.yaml"
AGENT_PACK = REPO_ROOT / "templates" / "agent-session-pack.satellite.yaml"
SCHEMA = REPO_ROOT / "schemas" / "satellite_evidence_pack_v1.schema.json"


def _load_failure_pack() -> dict:
    return load_pack_manifest(FAILURE_PACK)


def _write_manifest(root: Path, manifest: dict, filename: str = "pack.satellite.json") -> Path:
    path = root / filename
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _copy_schema_refs(root: Path) -> None:
    schemas = root / "schemas"
    schemas.mkdir(parents=True, exist_ok=True)
    for name in (
        "agent_session_bundle.schema.json",
        "evidence_graph.schema.json",
        "evidence_support.schema.json",
        "review_memory_fixture.schema.json",
        "satellite_evidence_pack_v1.schema.json",
    ):
        shutil.copy2(REPO_ROOT / "schemas" / name, schemas / name)


def _security_statuses(audit: dict) -> dict[str, str]:
    return {
        item["check_id"]: item["status"]
        for item in audit["security_checks"]
    }


def _blocked_audit(manifest: dict, *, root: Path | None = None, manifest_path: Path | None = None) -> dict:
    path = manifest_path or FAILURE_PACK
    return build_evidence_pack_v1_audit(manifest, manifest_path=path, root=root or REPO_ROOT, strict=True)


class EvidencePackV1PolicyKernelTests(unittest.TestCase):
    def test_builtin_packs_pass_strict_audit(self) -> None:
        audits = [
            audit_evidence_pack_v1_path(path, root=REPO_ROOT, strict=True, write_artifact=False)[0]
            for path in (FAILURE_PACK, AGENT_PACK)
        ]

        self.assertGreaterEqual(len(audits), 2)
        self.assertTrue(all(audit["verdict"] == "pass" for audit in audits))

    def test_v1_detection_requires_explicit_schema_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = _load_failure_pack()
            manifest["schema_name"] = "software-satellite-evidence-pack"
            manifest_path = _write_manifest(root, manifest)

            detected = is_evidence_pack_v1_path(manifest_path)

        self.assertFalse(detected)
        self.assertTrue(is_evidence_pack_v1_path(FAILURE_PACK))

    def test_unknown_field_fails_strict_audit(self) -> None:
        manifest = _load_failure_pack()
        manifest["plugin_runtime"] = {"enabled": True}

        audit = _blocked_audit(manifest)

        self.assertEqual(audit["verdict"], "block")
        self.assertEqual(_security_statuses(audit)["unknown_fields"], "block")

    def test_unknown_field_is_needs_review_without_strict(self) -> None:
        manifest = _load_failure_pack()
        manifest["plugin_runtime"] = {"enabled": True}

        audit = build_evidence_pack_v1_audit(manifest, manifest_path=FAILURE_PACK, root=REPO_ROOT, strict=False)

        self.assertEqual(audit["verdict"], "needs_review")
        self.assertFalse(audit["strict_blocking_enforced"])
        self.assertEqual(_security_statuses(audit)["unknown_fields"], "block")

    def test_python_field_fails(self) -> None:
        manifest = _load_failure_pack()
        manifest["metadata"]["python"] = "print local result"

        audit = _blocked_audit(manifest)

        self.assertEqual(audit["verdict"], "block")
        self.assertEqual(_security_statuses(audit)["no_executable_runtime"], "block")

    def test_javascript_field_fails(self) -> None:
        manifest = _load_failure_pack()
        manifest["metadata"]["javascript"] = "node local result"

        audit = _blocked_audit(manifest)

        self.assertEqual(audit["verdict"], "block")
        self.assertEqual(_security_statuses(audit)["no_executable_runtime"], "block")

    def test_shell_field_fails(self) -> None:
        manifest = _load_failure_pack()
        manifest["support_policy"]["shell"] = "echo local result"

        audit = _blocked_audit(manifest)

        self.assertEqual(audit["verdict"], "block")
        self.assertEqual(_security_statuses(audit)["no_executable_runtime"], "block")

    def test_network_permission_fails(self) -> None:
        manifest = _load_failure_pack()
        manifest["artifact_policy"]["network"] = True

        audit = _blocked_audit(manifest)

        self.assertEqual(audit["verdict"], "block")
        self.assertEqual(_security_statuses(audit)["no_external_or_privileged_access"], "block")

    def test_api_access_fails(self) -> None:
        manifest = _load_failure_pack()
        manifest["support_policy"]["api"] = "api call"

        audit = _blocked_audit(manifest)

        self.assertEqual(audit["verdict"], "block")
        self.assertEqual(_security_statuses(audit)["no_external_or_privileged_access"], "block")

    def test_environment_variable_access_fails(self) -> None:
        manifest = _load_failure_pack()
        manifest["recall_policy"]["environment_variables"] = ["LOCAL_TOKEN"]

        audit = _blocked_audit(manifest)

        self.assertEqual(audit["verdict"], "block")
        self.assertEqual(_security_statuses(audit)["no_external_or_privileged_access"], "block")

    def test_secret_access_fails(self) -> None:
        manifest = _load_failure_pack()
        manifest["redaction_policy"]["secrets"] = "read"

        audit = _blocked_audit(manifest)

        self.assertEqual(audit["verdict"], "block")
        self.assertEqual(_security_statuses(audit)["no_external_or_privileged_access"], "block")

    def test_repo_write_fails(self) -> None:
        manifest = _load_failure_pack()
        manifest["artifact_policy"]["write_repo"] = True

        audit = _blocked_audit(manifest)

        self.assertEqual(audit["verdict"], "block")
        self.assertEqual(_security_statuses(audit)["no_external_or_privileged_access"], "block")

    def test_path_traversal_fails(self) -> None:
        manifest = _load_failure_pack()
        manifest["artifact_policy"]["selected_roots"] = ["../outside"]

        audit = _blocked_audit(manifest)

        self.assertEqual(audit["verdict"], "block")
        self.assertEqual(_security_statuses(audit)["path_boundary"], "block")

    def test_windows_absolute_paths_fail_on_posix(self) -> None:
        for selected_root in ("C:\\Windows", "\\\\server\\share"):
            with self.subTest(selected_root=selected_root):
                manifest = _load_failure_pack()
                manifest["artifact_policy"]["selected_roots"] = [selected_root]

                audit = _blocked_audit(manifest)

                self.assertEqual(audit["verdict"], "block")
                self.assertEqual(_security_statuses(audit)["path_boundary"], "block")

    def test_file_glob_fails(self) -> None:
        manifest = _load_failure_pack()
        manifest["artifact_policy"]["selected_roots"] = ["artifacts/*.json"]

        audit = _blocked_audit(manifest)

        self.assertEqual(audit["verdict"], "block")
        self.assertEqual(_security_statuses(audit)["path_boundary"], "block")

    def test_symlink_traversal_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _copy_schema_refs(root)
            real_dir = root / "real"
            real_dir.mkdir()
            (root / "linked").symlink_to(real_dir, target_is_directory=True)
            manifest = _load_failure_pack()
            manifest["artifact_policy"]["selected_roots"] = ["linked"]
            manifest_path = _write_manifest(root, manifest)

            audit = _blocked_audit(manifest, root=root, manifest_path=manifest_path)

        self.assertEqual(audit["verdict"], "block")
        self.assertEqual(_security_statuses(audit)["path_boundary"], "block")

    def test_remote_url_fails(self) -> None:
        manifest = _load_failure_pack()
        manifest["output_schema_refs"] = ["https://example.test/schema.json"]

        audit = _blocked_audit(manifest)

        self.assertEqual(audit["verdict"], "block")
        self.assertEqual(_security_statuses(audit)["path_boundary"], "block")

    def test_model_call_fails(self) -> None:
        manifest = _load_failure_pack()
        manifest["support_policy"]["model_call"] = "llm call"

        audit = _blocked_audit(manifest)

        self.assertEqual(audit["verdict"], "block")
        self.assertEqual(_security_statuses(audit)["no_external_or_privileged_access"], "block")

    def test_training_export_fails(self) -> None:
        manifest = _load_failure_pack()
        manifest["report_sections"][0]["training_export"] = True

        audit = _blocked_audit(manifest)

        self.assertEqual(audit["verdict"], "block")
        self.assertEqual(_security_statuses(audit)["no_external_or_privileged_access"], "block")

    def test_install_script_and_auto_update_fail(self) -> None:
        manifest = _load_failure_pack()
        manifest["metadata"]["install_script"] = "postinstall hook"
        manifest["metadata"]["auto_update"] = True

        audit = _blocked_audit(manifest)

        self.assertEqual(audit["verdict"], "block")
        self.assertEqual(_security_statuses(audit)["no_install_or_auto_update"], "block")

    def test_lock_detects_manifest_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _copy_schema_refs(root)
            manifest = _load_failure_pack()
            manifest_path = _write_manifest(root, manifest)
            lock, lock_path = lock_evidence_pack_v1_path(manifest_path, root=root)
            mutated = copy.deepcopy(manifest)
            mutated["metadata"]["summary"] = "Mutated local summary."
            manifest_path.write_text(json.dumps(mutated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            audit = audit_evidence_pack_v1_path(
                manifest_path,
                root=root,
                strict=True,
                write_artifact=False,
            )[0]

        self.assertEqual(lock["pack_id"], "failure-memory-pack")
        self.assertEqual(lock_path.resolve(), evidence_pack_v1_lock_path(manifest_path).resolve())
        self.assertEqual(audit["verdict"], "block")
        self.assertEqual(audit["lock_status"]["status"], "mismatch")
        self.assertEqual(_security_statuses(audit)["lock_manifest_integrity"], "block")

    def test_lock_can_be_refreshed_after_manifest_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _copy_schema_refs(root)
            manifest = _load_failure_pack()
            manifest_path = _write_manifest(root, manifest)
            lock_evidence_pack_v1_path(manifest_path, root=root)
            mutated = copy.deepcopy(manifest)
            mutated["metadata"]["summary"] = "Updated local summary."
            manifest_path.write_text(json.dumps(mutated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            refreshed_lock, _lock_path = lock_evidence_pack_v1_path(manifest_path, root=root)
            audit = audit_evidence_pack_v1_path(
                manifest_path,
                root=root,
                strict=True,
                write_artifact=False,
            )[0]

        self.assertEqual(refreshed_lock["manifest_sha256"], audit["manifest_sha256"])
        self.assertEqual(audit["verdict"], "pass")
        self.assertEqual(audit["lock_status"]["status"], "match")

    def test_lock_transform_mismatch_blocks_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _copy_schema_refs(root)
            manifest_path = _write_manifest(root, _load_failure_pack())
            lock, lock_path = lock_evidence_pack_v1_path(manifest_path, root=root)
            lock["allowed_core_transforms"] = ["artifact_capture"]
            lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            audit = audit_evidence_pack_v1_path(
                manifest_path,
                root=root,
                strict=True,
                write_artifact=False,
            )[0]

        self.assertEqual(audit["verdict"], "block")
        self.assertIn("allowed_core_transforms_mismatch", audit["lock_status"]["evidence"])

    def test_pack_test_runs_without_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _copy_schema_refs(root)
            manifest_path = _write_manifest(root, _load_failure_pack())
            env = dict(os.environ)
            for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN", "GEMMA_MODEL_ID"):
                env.pop(key, None)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "satlab.py"),
                    "--root",
                    str(root),
                    "pack",
                    "test",
                    str(manifest_path),
                    "--strict",
                    "--format",
                    "json",
                ],
                capture_output=True,
                check=False,
                text=True,
                env=env,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertFalse(payload["test"]["api_key_required"])
        self.assertTrue(payload["test"]["passed"])

    def test_cli_pack_test_rejects_legacy_manifest_without_v1_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "satlab.py"),
                    "--root",
                    str(root),
                    "pack",
                    "test",
                    str(REPO_ROOT / "templates" / "review-risk-pack.satellite.yaml"),
                    "--format",
                    "json",
                ],
                capture_output=True,
                check=False,
                text=True,
            )
            v1_artifact_root = root / "artifacts" / "satellite_evidence_pack_v1"

        self.assertEqual(completed.returncode, 2)
        self.assertIn("pack test supports only explicit Evidence Pack v1 manifests", completed.stderr)
        self.assertFalse(v1_artifact_root.exists())

    def test_pack_output_goes_through_support_kernel(self) -> None:
        result = run_evidence_pack_v1_test(
            FAILURE_PACK,
            root=REPO_ROOT,
            strict=True,
            write_artifact=False,
        )[0]

        self.assertTrue(result["passed"])
        self.assertFalse(result["pack_output_bypasses_support_kernel"])
        self.assertEqual(result["support_kernel_result_count"], result["fixture_count"])
        self.assertEqual(result["fixture_results"][0]["support_result"]["schema_name"], "software-satellite-evidence-support-result")
        self.assertEqual(result["fixture_results"][0]["support_result"]["support_polarity"], "risk")
        recorded_at = result["fixture_results"][0]["event_recorded_at_utc"]
        self.assertNotEqual(recorded_at, "2026-05-12T00:00:00+00:00")
        datetime.fromisoformat(recorded_at)

    def test_non_strict_pack_test_runs_fixtures_for_draft_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _copy_schema_refs(root)
            manifest = _load_failure_pack()
            manifest["plugin_runtime"] = {"enabled": True}
            manifest_path = _write_manifest(root, manifest)

            result = run_evidence_pack_v1_test(
                manifest_path,
                root=root,
                strict=False,
                write_artifact=False,
            )[0]

        self.assertTrue(result["passed"])
        self.assertEqual(result["audit_verdict"], "needs_review")
        self.assertEqual(result["fixture_count"], 1)

    def test_strict_pack_test_blocks_draft_manifest_before_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _copy_schema_refs(root)
            manifest = _load_failure_pack()
            manifest["plugin_runtime"] = {"enabled": True}
            manifest_path = _write_manifest(root, manifest)

            result = run_evidence_pack_v1_test(
                manifest_path,
                root=root,
                strict=True,
                write_artifact=False,
            )[0]

        self.assertFalse(result["passed"])
        self.assertEqual(result["audit_verdict"], "block")
        self.assertEqual(result["fixture_count"], 0)

    def test_scaffold_writes_builtin_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "failure-memory-pack.satellite.yaml"
            result = scaffold_evidence_pack_v1("failure-memory", output)
            manifest = load_pack_manifest(output)

        self.assertEqual(result["status"], "written")
        self.assertEqual(result["schema_name"], PACK_V1_SCHEMA_NAME)
        self.assertEqual(manifest["metadata"]["pack_id"], "failure-memory-pack")

    def test_scaffold_refuses_to_overwrite_different_existing_output_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "failure-memory-pack.satellite.yaml"
            output.write_text("schema_name: local-draft\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Use --force"):
                scaffold_evidence_pack_v1("failure-memory", output)

            result = scaffold_evidence_pack_v1("failure-memory", output, overwrite=True)
            manifest = load_pack_manifest(output)

        self.assertEqual(result["status"], "overwritten")
        self.assertEqual(manifest["schema_name"], PACK_V1_SCHEMA_NAME)

    def test_scaffold_is_idempotent_for_matching_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "failure-memory-pack.satellite.yaml"
            first = scaffold_evidence_pack_v1("failure-memory", output)
            second = scaffold_evidence_pack_v1("failure-memory", output)

        self.assertEqual(first["status"], "written")
        self.assertEqual(second["status"], "unchanged")

    def test_schema_allowlist_stays_aligned_with_policy_kernel(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["schema_name"]["const"], PACK_V1_SCHEMA_NAME)
        self.assertEqual(set(schema["properties"]["core_transform_refs"]["items"]["enum"]), ALLOWED_CORE_TRANSFORMS)
        fixture_schema = schema["properties"]["benchmark_fixtures"]["items"]["properties"]
        self.assertEqual(set(fixture_schema["input_kind"]["enum"]), ALLOWED_INPUT_KINDS)


if __name__ == "__main__":
    unittest.main()
