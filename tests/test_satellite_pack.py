from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from satellite_pack import (  # noqa: E402
    PACK_AUDIT_SCHEMA_NAME,
    PACK_KINDS,
    PACK_MANIFEST_SCHEMA_NAME,
    PackManifestError,
    REQUIRED_PERMISSION_KEYS,
    V0_DENIED_TRUE_PERMISSIONS,
    audit_pack_path,
    build_pack_audit,
    inspect_pack_path,
    load_pack_manifest,
    parse_yaml_manifest_subset,
    resolve_pack_manifest_path,
    validate_manifest_schema,
)


REVIEW_RISK_TEMPLATE = REPO_ROOT / "templates" / "review-risk-pack.satellite.yaml"
AUDIT_SCHEMA = REPO_ROOT / "schemas" / "satellite_pack_audit.schema.json"
MANIFEST_SCHEMA = REPO_ROOT / "schemas" / "satellite_evidence_pack.schema.json"


class SatellitePackManifestTests(unittest.TestCase):
    def test_review_risk_template_loads_and_validates(self) -> None:
        manifest = load_pack_manifest(REVIEW_RISK_TEMPLATE)
        issues = validate_manifest_schema(manifest)

        self.assertEqual(issues, [])
        self.assertEqual(manifest["schema_name"], PACK_MANIFEST_SCHEMA_NAME)
        self.assertEqual(manifest["name"], "review-risk-pack")
        self.assertEqual(manifest["kind"], "workflow_pack")
        self.assertTrue(manifest["permissions"]["read_repo"])
        self.assertFalse(manifest["permissions"]["network"])

    def test_manifest_loader_accepts_pack_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pack_dir = root / "packs" / "review-risk-pack"
            pack_dir.mkdir(parents=True)
            manifest_path = pack_dir / "pack.satellite.yaml"
            manifest_path.write_text(REVIEW_RISK_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")

            resolved = resolve_pack_manifest_path(pack_dir)
            manifest = load_pack_manifest(pack_dir)

        self.assertEqual(resolved, manifest_path.resolve())
        self.assertEqual(manifest["name"], "review-risk-pack")

    def test_manifest_loader_accepts_json_manifest_in_pack_directory(self) -> None:
        manifest = load_pack_manifest(REVIEW_RISK_TEMPLATE)
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_dir = Path(tmpdir) / "packs" / "json-pack"
            pack_dir.mkdir(parents=True)
            manifest_path = pack_dir / "pack.satellite.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            resolved = resolve_pack_manifest_path(pack_dir)
            loaded = load_pack_manifest(pack_dir)

        self.assertEqual(resolved, manifest_path.resolve())
        self.assertEqual(loaded["name"], "review-risk-pack")

    def test_manifest_loader_rejects_ambiguous_pack_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_dir = Path(tmpdir)
            (pack_dir / "one.satellite.yaml").write_text(
                f"schema_name: {PACK_MANIFEST_SCHEMA_NAME}\n",
                encoding="utf-8",
            )
            (pack_dir / "two.satellite.yaml").write_text(
                f"schema_name: {PACK_MANIFEST_SCHEMA_NAME}\n",
                encoding="utf-8",
            )

            with self.assertRaises(PackManifestError):
                resolve_pack_manifest_path(pack_dir)

    def test_manifest_yaml_subset_rejects_tabs_in_indentation(self) -> None:
        with self.assertRaisesRegex(PackManifestError, "tabs are not supported"):
            parse_yaml_manifest_subset(
                f"schema_name: {PACK_MANIFEST_SCHEMA_NAME}\npermissions:\n\tread_repo: false\n",
                Path("pack.satellite.yaml"),
            )

    def test_manifest_yaml_subset_keeps_colon_scalars_in_lists(self) -> None:
        manifest = parse_yaml_manifest_subset(
            f"schema_name: {PACK_MANIFEST_SCHEMA_NAME}\ninputs:\n  - https://example.test/event-log.json\n",
            Path("pack.satellite.yaml"),
        )

        self.assertEqual(manifest["inputs"], ["https://example.test/event-log.json"])

    def test_manifest_yaml_subset_parses_flow_style_string_arrays(self) -> None:
        manifest = parse_yaml_manifest_subset(
            "\n".join(
                [
                    f"schema_name: {PACK_MANIFEST_SCHEMA_NAME}",
                    "inputs: []",
                    'outputs: ["review_note", "evidence_bundle"]',
                    "widgets: [evidence_path_card, human_verdict_card]",
                ]
            ),
            Path("pack.satellite.yaml"),
        )

        self.assertEqual(manifest["inputs"], [])
        self.assertEqual(manifest["outputs"], ["review_note", "evidence_bundle"])
        self.assertEqual(manifest["widgets"], ["evidence_path_card", "human_verdict_card"])

    def test_manifest_yaml_subset_parses_flow_style_mapping(self) -> None:
        manifest = parse_yaml_manifest_subset(
            "\n".join(
                [
                    f"schema_name: {PACK_MANIFEST_SCHEMA_NAME}",
                    "permissions: {read_repo: false, write_artifacts: true}",
                ]
            ),
            Path("pack.satellite.yaml"),
        )

        self.assertEqual(
            manifest["permissions"],
            {
                "read_repo": False,
                "write_artifacts": True,
            },
        )

    def test_manifest_yaml_subset_preserves_nested_flow_collections(self) -> None:
        manifest = parse_yaml_manifest_subset(
            "recipes: [{id: patch, steps: [lint, test]}]",
            Path("pack.satellite.yaml"),
        )

        self.assertEqual(
            manifest["recipes"],
            [
                {
                    "id": "patch",
                    "steps": ["lint", "test"],
                }
            ],
        )

    def test_schema_validation_blocks_denied_v0_permission(self) -> None:
        manifest = load_pack_manifest(REVIEW_RISK_TEMPLATE)
        harmful = copy.deepcopy(manifest)
        harmful["permissions"]["network"] = True

        issues = validate_manifest_schema(harmful)
        audit = build_pack_audit(harmful, manifest_path=REVIEW_RISK_TEMPLATE, root=REPO_ROOT)

        self.assertTrue(any(issue.path == "$.permissions.network" for issue in issues))
        self.assertEqual(audit["verdict"], "block")
        self.assertTrue(any("$.permissions.network" in reason for reason in audit["blocked_reasons"]))

    def test_schema_validation_accepts_legacy_pack_schema_name(self) -> None:
        manifest = load_pack_manifest(REVIEW_RISK_TEMPLATE)
        manifest["schema_name"] = "software-satellite-pack"

        issues = validate_manifest_schema(manifest)

        self.assertEqual(issues, [])

    def test_schema_validation_rejects_boolean_schema_version(self) -> None:
        manifest = load_pack_manifest(REVIEW_RISK_TEMPLATE)
        invalid = copy.deepcopy(manifest)
        invalid["schema_version"] = True

        issues = validate_manifest_schema(invalid)
        version_issue = next(issue for issue in issues if issue.path == "$.schema_version")

        self.assertEqual(version_issue.actual, "true")

    def test_schema_validation_rejects_null_recipes_when_present(self) -> None:
        manifest = load_pack_manifest(REVIEW_RISK_TEMPLATE)
        invalid = copy.deepcopy(manifest)
        invalid["kind"] = "recall_pack"
        invalid["recipes"] = None

        issues = validate_manifest_schema(invalid)
        recipes_issue = next(issue for issue in issues if issue.path == "$.recipes")

        self.assertEqual(recipes_issue.expected, "array")

    def test_schema_validation_preserves_primitive_actual_values(self) -> None:
        manifest = load_pack_manifest(REVIEW_RISK_TEMPLATE)
        invalid = copy.deepcopy(manifest)
        invalid["recipes"][0]["steps"] = []

        issues = validate_manifest_schema(invalid)
        steps_issue = next(issue for issue in issues if issue.path == "$.recipes[0].steps")

        self.assertEqual(steps_issue.actual, "0")

    def test_manifest_validator_stays_aligned_with_manifest_schema_contract(self) -> None:
        schema = json.loads(MANIFEST_SCHEMA.read_text(encoding="utf-8"))
        permission_schema = schema["properties"]["permissions"]
        const_false_permissions = {
            key
            for key, value in permission_schema["properties"].items()
            if value.get("const") is False
        }

        self.assertEqual(tuple(permission_schema["required"]), REQUIRED_PERMISSION_KEYS)
        self.assertEqual(schema["properties"]["schema_name"]["const"], PACK_MANIFEST_SCHEMA_NAME)
        self.assertEqual(set(schema["properties"]["kind"]["enum"]), set(PACK_KINDS))
        self.assertEqual(const_false_permissions, V0_DENIED_TRUE_PERMISSIONS)

    def test_inspect_marks_template_as_valid_but_review_needed_for_repo_read(self) -> None:
        inspection = inspect_pack_path(REVIEW_RISK_TEMPLATE, root=REPO_ROOT)
        permission_status = {
            item["permission"]: item["status"]
            for item in inspection["permission_summary"]
        }

        self.assertTrue(inspection["schema_valid"])
        self.assertEqual(permission_status["read_repo"], "needs_review")
        self.assertEqual(permission_status["run_command"], "not_requested")
        self.assertEqual(inspection["recipes"][0]["id"], "patch_risk_review")

    def test_audit_pack_path_writes_permission_summary_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pack_dir = root / "packs" / "review-risk-pack"
            pack_dir.mkdir(parents=True)
            manifest_path = pack_dir / "pack.satellite.yaml"
            manifest_path.write_text(REVIEW_RISK_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")

            audit, latest_path, run_path = audit_pack_path(pack_dir, root=root)
            loaded = json.loads(latest_path.read_text(encoding="utf-8"))

            self.assertEqual(audit["schema_name"], PACK_AUDIT_SCHEMA_NAME)
            self.assertEqual(audit["verdict"], "needs_review")
            self.assertTrue(latest_path.is_file())
            self.assertTrue(run_path.is_file())
            self.assertEqual(loaded["pack_name"], "review-risk-pack")
            self.assertIn(str(manifest_path.resolve()), loaded["source_paths"])
            self.assertTrue(any(item["permission"] == "network" for item in loaded["permission_summary"]))

    def test_audit_schema_covers_generated_artifact_shape(self) -> None:
        manifest = load_pack_manifest(REVIEW_RISK_TEMPLATE)
        audit = build_pack_audit(manifest, manifest_path=REVIEW_RISK_TEMPLATE, root=REPO_ROOT)
        audit["paths"] = {
            "audit_latest_path": "/tmp/latest.json",
            "audit_run_path": "/tmp/run.json",
        }
        schema = json.loads(AUDIT_SCHEMA.read_text(encoding="utf-8"))
        permission_item_schema = schema["properties"]["permission_summary"]["items"]

        self.assertFalse(schema["additionalProperties"])
        self.assertLessEqual(set(audit), set(schema["properties"]))
        self.assertLessEqual(set(schema["required"]), set(audit))
        for item in audit["permission_summary"]:
            self.assertLessEqual(set(item), set(permission_item_schema["properties"]))

    def test_satlab_pack_inspect_cli_outputs_json(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "satlab.py"),
                "pack",
                "inspect",
                str(REVIEW_RISK_TEMPLATE),
                "--format",
                "json",
            ],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["inspection"]["schema_valid"])
        self.assertEqual(payload["inspection"]["manifest_summary"]["name"], "review-risk-pack")

    def test_satlab_pack_audit_cli_writes_json_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pack_dir = root / "packs" / "review-risk-pack"
            pack_dir.mkdir(parents=True)
            (pack_dir / "pack.satellite.yaml").write_text(
                REVIEW_RISK_TEMPLATE.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "satlab.py"),
                    "--root",
                    str(root),
                    "pack",
                    "audit",
                    str(pack_dir),
                    "--format",
                    "json",
                ],
                capture_output=True,
                check=False,
                text=True,
            )
            payload = json.loads(completed.stdout)
            latest_path = Path(payload["audit_latest_path"])
            run_path = Path(payload["audit_run_path"])

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(payload["audit"]["verdict"], "needs_review")
            self.assertTrue(latest_path.is_file())
            self.assertTrue(run_path.is_file())


if __name__ == "__main__":
    unittest.main()
