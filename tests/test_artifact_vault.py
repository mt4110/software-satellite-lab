from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from artifact_vault import artifact_ref_object_verified, capture_artifact, inspect_artifact, load_artifact_ref  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
