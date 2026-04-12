from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from artifact_schema import (  # noqa: E402
    build_artifact_payload,
    build_prompt_record,
    build_runtime_record,
    read_artifact,
    write_artifact,
)
from run_capability_matrix import (  # noqa: E402
    CapabilityResult,
    run_matrix,
    run_selected_text_capabilities,
)


class MatrixForensicsTests(unittest.TestCase):
    def test_run_selected_text_capabilities_writes_traceable_artifact(self) -> None:
        fake_session = SimpleNamespace(
            model_id="google/gemma-4-E2B-it",
            device_info={"name": "cpu", "label": "cpu", "dtype_name": "float32"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "text-chat.json"
            with (
                patch(
                    "run_capability_matrix.run_text_generation",
                    return_value=("Binary search halves the search space.", 0.25),
                ),
                patch(
                    "run_capability_matrix.default_capability_artifact_path",
                    return_value=artifact_path,
                ),
            ):
                results = run_selected_text_capabilities(fake_session, {"text-chat"})

            self.assertEqual(len(results), 1)
            result = results[0]
            stored = read_artifact(artifact_path)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.artifact_path, str(artifact_path.resolve()))
        self.assertEqual(result.artifact_kind, "text")
        self.assertEqual(result.runtime_backend, "gemma-live-text")
        self.assertEqual(stored["entrypoint"], "capability_matrix")
        self.assertEqual(stored["validation"]["validation_mode"], "live")
        self.assertEqual(stored["validation"]["quality_status"], "pass")
        self.assertEqual(stored["output_text"], "Binary search halves the search space.")

    def test_run_matrix_writes_artifact_index_aligned_with_results(self) -> None:
        class FakeSessionManager:
            def get_session(self, session_kind: str, model_id: str):
                return SimpleNamespace(model_id=model_id, device_info={"name": "cpu", "label": "cpu"})

            def close_all(self) -> int:
                return 0

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_artifact_path = root / "artifacts" / "text" / "raw-text-chat.json"
            write_artifact(
                raw_artifact_path,
                build_artifact_payload(
                    artifact_kind="text",
                    status="ok",
                    runtime=build_runtime_record(
                        backend="gemma-live-text",
                        model_id="google/gemma-4-E2B-it",
                        device_info="cpu",
                        elapsed_seconds=0.25,
                    ),
                    prompts=build_prompt_record(
                        system_prompt="You are a concise, helpful assistant.",
                        prompt="Explain binary search.",
                        resolved_user_prompt="Explain binary search.",
                    ),
                    extra={
                        "entrypoint": "capability_matrix",
                        "validation": {
                            "validation_mode": "live",
                            "claim_scope": "live model generation on a small local prompt",
                            "pass_definition": "Pass means execution completed.",
                            "execution_status": "ok",
                            "quality_status": "pass",
                            "quality_checks": [],
                            "quality_notes": [],
                        },
                        "output_text": "Binary search halves the search space.",
                    },
                ),
            )
            expected_result = CapabilityResult(
                capability="text-chat",
                phase="phase1",
                status="ok",
                model_used="google/gemma-4-E2B-it",
                asset_used=None,
                validation_command="python scripts/run_capability_matrix.py --only text-chat",
                artifact_path=str(raw_artifact_path.resolve()),
                elapsed_seconds=0.25,
                result="ok",
                blocker=None,
                execution_status="ok",
                quality_status="pass",
                validation_mode="live",
                claim_scope="live model generation on a small local prompt",
                pass_definition="Pass means execution completed.",
                quality_checks=[],
                quality_notes=[],
                output_preview="Binary search halves the search space.",
                notes=[],
                artifact_kind="text",
                runtime_backend="gemma-live-text",
                preprocessing_lineage=[],
            )
            args = Namespace(
                only="text-chat",
                phase=None,
                smoke=False,
                skip_prepare_assets=True,
                asset_timeout=3.0,
                max_pages=4,
                out=root / "artifacts" / "capability_matrix" / "matrix-run-matrix.json",
            )

            with (
                patch("run_capability_matrix.repo_root", return_value=root),
                patch("run_capability_matrix.resolve_model_id", return_value="google/gemma-4-E2B-it"),
                patch(
                    "run_capability_matrix.resolve_audio_model_selection",
                    return_value=("google/gemma-4-E2B-it", "GEMMA_MODEL_ID"),
                ),
                patch("run_capability_matrix.SessionManager", return_value=FakeSessionManager()),
                patch("run_capability_matrix.run_selected_text_capabilities", return_value=[expected_result]),
            ):
                payload = run_matrix(args)

            index_path = Path(payload["artifact_index_path"])
            index_payload = read_artifact(index_path)
            self.assertTrue(index_path.exists())
            self.assertEqual(payload["results"][0]["artifact_path"], str(raw_artifact_path.resolve()))
            self.assertEqual(index_payload["matrix_artifact_path"], str(args.out.resolve()))
            self.assertEqual(index_payload["entries"][0]["capability"], "text-chat")
            self.assertEqual(index_payload["entries"][0]["artifact_path"], str(raw_artifact_path.resolve()))
            self.assertEqual(index_payload["entries"][0]["runtime_backend"], "gemma-live-text")
            self.assertEqual(index_payload["entries"][0]["artifact_workspace_relative_path"], "artifacts/text/raw-text-chat.json")


if __name__ == "__main__":
    unittest.main()
