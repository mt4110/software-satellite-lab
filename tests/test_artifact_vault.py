from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import artifact_vault as artifact_vault_module  # noqa: E402
from artifact_vault import (  # noqa: E402
    artifact_gc_dry_run,
    artifact_ref_object_verified,
    capture_artifact,
    format_artifact_gc_dry_run_report,
    inspect_artifact,
    load_artifact_ref,
)
from artifact_vault import ARTIFACT_KINDS  # noqa: E402
from satlab import build_parser, main as satlab_main  # noqa: E402


class ArtifactVaultTests(unittest.TestCase):
    def test_satlab_artifact_kind_choices_stay_in_sync_with_vault(self) -> None:
        parser = build_parser()
        artifact_parser = parser._subparsers._group_actions[0].choices["artifact"]
        capture_parser = artifact_parser._subparsers._group_actions[0].choices["capture"]
        kind_action = next(action for action in capture_parser._actions if "--kind" in action.option_strings)

        self.assertEqual(tuple(kind_action.choices), tuple(sorted(ARTIFACT_KINDS)))

    def test_captures_text_artifact_and_computes_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            patch = root / "changes.diff"
            patch.write_text("diff --git a/app.py b/app.py\n+print('hi')\n", encoding="utf-8")

            ref = capture_artifact(patch, kind="patch", root=root, captured_at_utc="2026-05-12T00:00:00+00:00")
            loaded = load_artifact_ref(ref["artifact_id"], root=root)
            vault_path = root / ref["vault_path"]

            self.assertEqual(ref["schema_name"], "software-satellite-artifact-ref")
            self.assertEqual(ref["capture_state"], "captured")
            self.assertEqual(ref["source_state"], "present")
            self.assertEqual(ref["sha256"], hashlib.sha256("diff --git a/app.py b/app.py\n+print('hi')\n".encode()).hexdigest())
            self.assertTrue(vault_path.is_file())
            self.assertEqual(loaded["artifact_id"], ref["artifact_id"])

    def test_refuses_binary_artifact_body_and_records_binary_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            binary = root / "screen.bin"
            binary.write_bytes(b"\x00\x01\x02not-text")
            size_bytes = binary.stat().st_size

            ref = capture_artifact(binary, kind="unknown", root=root)

        self.assertEqual(ref["source_state"], "binary_refused")
        self.assertEqual(ref["capture_state"], "refused")
        self.assertIsNone(ref["vault_path"])
        self.assertEqual(ref["redaction"]["binary_bytes_refused"], size_bytes)

    def test_caps_oversized_artifact_and_records_oversize(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            large = root / "large.log"
            large.write_text("A" * 200, encoding="utf-8")

            ref = capture_artifact(large, kind="test_log", root=root, max_capture_bytes=20, report_excerpt_chars=64)

        self.assertEqual(ref["source_state"], "oversize")
        self.assertEqual(ref["capture_state"], "ref_only")
        self.assertIsNone(ref["vault_path"])
        self.assertIn("[truncated]", ref["report_excerpt"]["text"])

    def test_redacts_secret_like_tokens_in_report_excerpt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log = root / "test.log"
            log.write_text("OPENAI_API_KEY=sk-secretsecretsecretsecret\n", encoding="utf-8")

            ref = capture_artifact(log, kind="test_log", root=root)

        excerpt = ref["report_excerpt"]["text"]
        self.assertNotIn("sk-secret", excerpt)
        self.assertIn("[REDACTED]", excerpt)
        self.assertTrue(ref["redaction"]["applied"])
        self.assertGreater(ref["redaction"]["secret_like_tokens"], 0)

    def test_rejects_forged_vault_object_path_outside_vault(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.txt"
            source.write_text("real vault evidence\n", encoding="utf-8")
            outside = root / "outside-copy.txt"
            outside.write_text("real vault evidence\n", encoding="utf-8")
            ref = capture_artifact(source, kind="review_note", root=root)
            forged = {**ref, "vault_path": str(outside)}

            verified, reason = artifact_ref_object_verified(forged, root=root)

        self.assertFalse(verified)
        self.assertEqual(reason, "vault_object_outside_vault")

    def test_rejects_noncanonical_vault_object_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.txt"
            source.write_text("real vault evidence\n", encoding="utf-8")
            ref = capture_artifact(source, kind="review_note", root=root)
            duplicate = root / "artifacts" / "vault" / "objects" / "manual-copy"
            duplicate.parent.mkdir(parents=True, exist_ok=True)
            duplicate.write_text("real vault evidence\n", encoding="utf-8")
            forged = {**ref, "vault_path": str(duplicate)}

            verified, reason = artifact_ref_object_verified(forged, root=root)

        self.assertFalse(verified)
        self.assertEqual(reason, "noncanonical_vault_object_path")

    def test_invalid_artifact_id_cannot_traverse_ref_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            with self.assertRaisesRegex(ValueError, "Invalid artifact id"):
                load_artifact_ref("../outside", root=root)

    def test_refuses_path_traversal_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            root = base / "workspace"
            root.mkdir()
            outside = base / "outside.txt"
            outside.write_text("outside workspace\n", encoding="utf-8")

            ref = capture_artifact(root / ".." / "outside.txt", kind="review_note", root=root)

        self.assertEqual(ref["source_state"], "outside_workspace")
        self.assertEqual(ref["capture_state"], "refused")
        self.assertIsNone(ref["vault_path"])

    def test_refuses_symlink_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "target.txt"
            target.write_text("target evidence\n", encoding="utf-8")
            link = root / "linked.txt"
            try:
                link.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlink setup unavailable: {exc}")

            ref = capture_artifact(link, kind="review_note", root=root)

        self.assertEqual(ref["source_state"], "symlink_refused")
        self.assertEqual(ref["capture_state"], "refused")
        self.assertIsNone(ref["vault_path"])

    def test_satlab_artifact_capture_and_inspect_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            patch = root / "changes.diff"
            patch.write_text("+hello\n", encoding="utf-8")

            capture_stdout = io.StringIO()
            with redirect_stdout(capture_stdout):
                capture_exit = satlab_main(
                    [
                        "--root",
                        str(root),
                        "artifact",
                        "capture",
                        "--path",
                        str(patch),
                        "--kind",
                        "patch",
                        "--format",
                        "json",
                    ]
                )
            captured = json.loads(capture_stdout.getvalue())

            inspect_stdout = io.StringIO()
            with redirect_stdout(inspect_stdout):
                inspect_exit = satlab_main(
                    [
                        "--root",
                        str(root),
                        "artifact",
                        "inspect",
                        "--artifact",
                        captured["artifact_id"],
                        "--format",
                        "json",
                    ]
                )
            inspection = json.loads(inspect_stdout.getvalue())

        self.assertEqual(capture_exit, 0)
        self.assertEqual(inspect_exit, 0)
        self.assertTrue(inspection["object_verified"])
        self.assertEqual(inspection["ref"]["artifact_id"], captured["artifact_id"])
        self.assertIn("best-effort", inspection["redaction_notice"])

    def test_artifact_gc_dry_run_reports_inventory_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            kept_source = root / "kept.txt"
            kept_source.write_text("kept vault object\n", encoding="utf-8")
            kept_ref = capture_artifact(
                kept_source,
                kind="review_note",
                root=root,
                captured_at_utc="2026-05-12T00:00:00+00:00",
            )
            missing_source = root / "missing.txt"
            missing_source.write_text("missing vault object\n", encoding="utf-8")
            missing_ref = capture_artifact(
                missing_source,
                kind="review_note",
                root=root,
                captured_at_utc="2026-05-12T00:00:01+00:00",
            )
            missing_object = root / missing_ref["vault_path"]
            missing_object.unlink()
            stale_object = root / "artifacts" / "vault" / "objects" / "zz" / "stale-object"
            stale_object.parent.mkdir(parents=True, exist_ok=True)
            stale_object.write_text("stale object\n", encoding="utf-8")
            stale_size = stale_object.stat().st_size
            malformed_ref = root / "artifacts" / "vault" / "refs" / "artifact_bad.json"
            malformed_ref.write_text("{not-json", encoding="utf-8")

            report = artifact_gc_dry_run(root=root)
            formatted = format_artifact_gc_dry_run_report(report)

            self.assertTrue(stale_object.exists())
            self.assertEqual(report["schema_name"], "software-satellite-artifact-gc-dry-run")
            self.assertTrue(report["dry_run"])
            self.assertFalse(report["delete_performed"])
            self.assertEqual(report["counts"]["referenced_object_count"], 1)
            self.assertEqual(report["counts"]["unreferenced_object_count"], 1)
            self.assertEqual(report["counts"]["missing_object_count"], 1)
            self.assertEqual(report["counts"]["malformed_ref_count"], 1)
            self.assertEqual(report["reclaimable_bytes"], stale_size)
            self.assertEqual(report["referenced_objects"][0]["artifact_ids"], [kept_ref["artifact_id"]])
            self.assertEqual(report["unreferenced_objects"][0]["vault_path"], "artifacts/vault/objects/zz/stale-object")
            self.assertEqual(report["missing_objects"][0]["artifact_id"], missing_ref["artifact_id"])
            self.assertEqual(report["missing_objects"][0]["reason"], "missing_vault_object")
            self.assertEqual(report["malformed_refs"][0]["reason"], "invalid_json")
            self.assertIn("Mode: dry-run; no files removed.", formatted)
            self.assertIn("artifact_bad.json", formatted)

    def test_satlab_artifact_gc_dry_run_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stale_object = root / "artifacts" / "vault" / "objects" / "aa" / "stale-object"
            stale_object.parent.mkdir(parents=True, exist_ok=True)
            stale_object.write_text("stale object\n", encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = satlab_main(
                    [
                        "--root",
                        str(root),
                        "artifact",
                        "gc",
                        "--dry-run",
                        "--format",
                        "json",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            stale_object_exists = stale_object.exists()

        self.assertEqual(exit_code, 0)
        self.assertTrue(stale_object_exists)
        self.assertTrue(payload["dry_run"])
        self.assertFalse(payload["delete_performed"])
        self.assertEqual(payload["counts"]["unreferenced_object_count"], 1)
        self.assertEqual(payload["reclaimable_bytes"], len("stale object\n".encode()))

    def test_satlab_artifact_gc_requires_dry_run_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    satlab_main(["--root", tmpdir, "artifact", "gc"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--dry-run", stderr.getvalue())

    def test_artifact_gc_dry_run_reports_symlink_objects_as_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            root = base / "workspace"
            root.mkdir()
            outside = base / "outside-vault-object.txt"
            outside.write_text("outside vault\n", encoding="utf-8")
            symlink_object = root / "artifacts" / "vault" / "objects" / "aa" / "linked-object"
            symlink_object.parent.mkdir(parents=True, exist_ok=True)
            try:
                symlink_object.symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"symlink setup unavailable: {exc}")

            report = artifact_gc_dry_run(root=root)
            formatted = format_artifact_gc_dry_run_report(report)

            self.assertTrue(symlink_object.exists() or symlink_object.is_symlink())
            self.assertEqual(report["counts"]["object_count"], 0)
            self.assertEqual(report["counts"]["unreferenced_object_count"], 0)
            self.assertEqual(report["counts"]["skipped_object_count"], 1)
            self.assertEqual(report["skipped_objects"][0]["vault_path"], "artifacts/vault/objects/aa/linked-object")
            self.assertEqual(report["skipped_objects"][0]["reason"], "symlink_refused")
            self.assertIn("Skipped objects: 1", formatted)

    def test_artifact_gc_dry_run_reports_symlink_object_directories_as_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            root = base / "workspace"
            root.mkdir()
            outside_dir = base / "outside-object-dir"
            outside_dir.mkdir()
            (outside_dir / "escaped-object").write_text("outside vault\n", encoding="utf-8")
            symlink_dir = root / "artifacts" / "vault" / "objects" / "aa"
            symlink_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                symlink_dir.symlink_to(outside_dir, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlink setup unavailable: {exc}")

            report = artifact_gc_dry_run(root=root)

            self.assertEqual(report["counts"]["object_count"], 0)
            self.assertEqual(report["counts"]["unreferenced_object_count"], 0)
            self.assertEqual(report["counts"]["skipped_object_count"], 1)
            self.assertEqual(report["skipped_objects"][0]["vault_path"], "artifacts/vault/objects/aa")
            self.assertEqual(report["skipped_objects"][0]["reason"], "symlink_refused")

    def test_artifact_gc_dry_run_reports_symlink_objects_root_as_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            root = base / "workspace"
            root.mkdir()
            outside_objects = base / "outside-objects"
            outside_objects.mkdir()
            (outside_objects / "escaped-object").write_text("outside vault\n", encoding="utf-8")
            objects_root = root / "artifacts" / "vault" / "objects"
            objects_root.parent.mkdir(parents=True, exist_ok=True)
            try:
                objects_root.symlink_to(outside_objects, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlink setup unavailable: {exc}")

            report = artifact_gc_dry_run(root=root)

            self.assertEqual(report["counts"]["object_count"], 0)
            self.assertEqual(report["counts"]["unreferenced_object_count"], 0)
            self.assertEqual(report["counts"]["skipped_object_count"], 1)
            self.assertEqual(report["reclaimable_bytes"], 0)
            self.assertEqual(report["skipped_objects"][0]["vault_path"], "artifacts/vault/objects")
            self.assertEqual(report["skipped_objects"][0]["reason"], "symlink_refused")

    def test_artifact_gc_dry_run_reports_broken_symlink_objects_root_as_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            objects_root = root / "artifacts" / "vault" / "objects"
            objects_root.parent.mkdir(parents=True, exist_ok=True)
            try:
                objects_root.symlink_to(root / "missing-objects", target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlink setup unavailable: {exc}")

            report = artifact_gc_dry_run(root=root)

            self.assertEqual(report["counts"]["object_count"], 0)
            self.assertEqual(report["counts"]["unreferenced_object_count"], 0)
            self.assertEqual(report["counts"]["skipped_object_count"], 1)
            self.assertEqual(report["skipped_objects"][0]["vault_path"], "artifacts/vault/objects")
            self.assertEqual(report["skipped_objects"][0]["reason"], "symlink_refused")

    def test_artifact_gc_dry_run_reports_non_utf8_refs_as_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            refs_root = root / "artifacts" / "vault" / "refs"
            refs_root.mkdir(parents=True, exist_ok=True)
            bad_ref = refs_root / "artifact_nonutf8.json"
            bad_ref.write_bytes(b"\xff\xfe\x00")

            report = artifact_gc_dry_run(root=root)

            self.assertEqual(report["counts"]["ref_count"], 0)
            self.assertEqual(report["counts"]["malformed_ref_count"], 1)
            self.assertEqual(report["malformed_refs"][0]["ref_path"], "artifacts/vault/refs/artifact_nonutf8.json")
            self.assertEqual(report["malformed_refs"][0]["reason"], "invalid_utf8")

    def test_artifact_gc_dry_run_reports_invalid_sha256_refs_as_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            refs_root = root / "artifacts" / "vault" / "refs"
            refs_root.mkdir(parents=True, exist_ok=True)
            bad_ref = refs_root / "artifact_badsha.json"
            bad_ref.write_text(
                json.dumps(
                    {
                        "schema_name": "software-satellite-artifact-ref",
                        "schema_version": 1,
                        "artifact_id": "artifact_badsha",
                        "capture_state": "captured",
                        "sha256": "../not-a-sha",
                        "vault_path": "artifacts/vault/objects/..",
                    }
                ),
                encoding="utf-8",
            )

            report = artifact_gc_dry_run(root=root)

            self.assertEqual(report["counts"]["ref_count"], 0)
            self.assertEqual(report["counts"]["malformed_ref_count"], 1)
            self.assertEqual(report["counts"]["missing_object_count"], 0)
            self.assertEqual(report["malformed_refs"][0]["ref_path"], "artifacts/vault/refs/artifact_badsha.json")
            self.assertEqual(report["malformed_refs"][0]["reason"], "invalid_sha256")

    def test_artifact_gc_dry_run_reports_invalid_capture_state_refs_as_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            refs_root = root / "artifacts" / "vault" / "refs"
            refs_root.mkdir(parents=True, exist_ok=True)
            bad_ref = refs_root / "artifact_badstate.json"
            bad_ref.write_text(
                json.dumps(
                    {
                        "schema_name": "software-satellite-artifact-ref",
                        "schema_version": 1,
                        "artifact_id": "artifact_badstate",
                        "capture_state": "capturd",
                    }
                ),
                encoding="utf-8",
            )

            report = artifact_gc_dry_run(root=root)

            self.assertEqual(report["counts"]["ref_count"], 0)
            self.assertEqual(report["counts"]["malformed_ref_count"], 1)
            self.assertEqual(report["malformed_refs"][0]["ref_path"], "artifacts/vault/refs/artifact_badstate.json")
            self.assertEqual(report["malformed_refs"][0]["reason"], "invalid_capture_state")

    def test_artifact_gc_dry_run_reports_refs_root_file_as_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            refs_root = root / "artifacts" / "vault" / "refs"
            refs_root.parent.mkdir(parents=True, exist_ok=True)
            refs_root.write_text("not a directory\n", encoding="utf-8")

            report = artifact_gc_dry_run(root=root)

            self.assertEqual(report["counts"]["ref_count"], 0)
            self.assertEqual(report["counts"]["malformed_ref_count"], 1)
            self.assertEqual(report["malformed_refs"][0]["ref_path"], "artifacts/vault/refs")
            self.assertEqual(report["malformed_refs"][0]["reason"], "refs_root_not_directory")

    def test_artifact_gc_dry_run_reports_symlink_refs_root_as_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            root = base / "workspace"
            root.mkdir()
            outside_refs = base / "outside-refs"
            outside_refs.mkdir()
            refs_root = root / "artifacts" / "vault" / "refs"
            refs_root.parent.mkdir(parents=True, exist_ok=True)
            try:
                refs_root.symlink_to(outside_refs, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlink setup unavailable: {exc}")

            report = artifact_gc_dry_run(root=root)

            self.assertEqual(report["counts"]["ref_count"], 0)
            self.assertEqual(report["counts"]["malformed_ref_count"], 1)
            self.assertEqual(report["malformed_refs"][0]["ref_path"], "artifacts/vault/refs")
            self.assertEqual(report["malformed_refs"][0]["reason"], "refs_root_symlink_refused")

    def test_artifact_gc_dry_run_reports_symlink_ref_files_as_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            root = base / "workspace"
            root.mkdir()
            outside_ref = base / "outside-ref.json"
            outside_ref.write_text(
                json.dumps(
                    {
                        "schema_name": "software-satellite-artifact-ref",
                        "schema_version": 1,
                        "artifact_id": "artifact_outside",
                        "capture_state": "captured",
                    }
                ),
                encoding="utf-8",
            )
            refs_root = root / "artifacts" / "vault" / "refs"
            refs_root.mkdir(parents=True, exist_ok=True)
            linked_ref = refs_root / "artifact_linked.json"
            try:
                linked_ref.symlink_to(outside_ref)
            except OSError as exc:
                self.skipTest(f"symlink setup unavailable: {exc}")

            report = artifact_gc_dry_run(root=root)

            self.assertEqual(report["counts"]["ref_count"], 0)
            self.assertEqual(report["counts"]["malformed_ref_count"], 1)
            self.assertEqual(report["malformed_refs"][0]["ref_path"], "artifacts/vault/refs/artifact_linked.json")
            self.assertEqual(report["malformed_refs"][0]["reason"], "ref_symlink_refused")

    def test_artifact_gc_dry_run_reuses_verification_for_duplicate_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "duplicate.txt"
            source.write_text("shared object\n", encoding="utf-8")
            ref = capture_artifact(
                source,
                kind="review_note",
                root=root,
                captured_at_utc="2026-05-12T00:00:00+00:00",
            )
            duplicate_ref = {
                **ref,
                "artifact_id": "artifact_duplicate",
                "captured_at_utc": "2026-05-12T00:00:01+00:00",
            }
            duplicate_ref_path = root / "artifacts" / "vault" / "refs" / "artifact_duplicate.json"
            duplicate_ref_path.write_text(json.dumps(duplicate_ref, sort_keys=True), encoding="utf-8")

            original = artifact_vault_module.artifact_ref_object_verified
            calls = []

            def spy(ref_payload: dict[str, object], *, root: Path | None = None) -> tuple[bool, str | None]:
                calls.append(ref_payload.get("artifact_id"))
                return original(ref_payload, root=root)

            artifact_vault_module.artifact_ref_object_verified = spy
            try:
                report = artifact_gc_dry_run(root=root)
            finally:
                artifact_vault_module.artifact_ref_object_verified = original

            self.assertEqual(calls, [ref["artifact_id"]])
            self.assertEqual(report["counts"]["referenced_object_count"], 1)
            self.assertEqual(report["referenced_objects"][0]["ref_count"], 2)


if __name__ == "__main__":
    unittest.main()
