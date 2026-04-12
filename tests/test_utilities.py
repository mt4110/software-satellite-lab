from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from doctor import preferred_runtime_device_label  # noqa: E402
from gemma_runtime import (  # noqa: E402
    build_repo_runtime_hint,
    build_requirements_install_command,
    parse_tool_arguments,
    split_gemma_response,
    timestamp_slug,
)
from long_context_corpus import (  # noqa: E402
    MARKER_LABELS,
    build_markers,
    build_synthetic_corpus,
    collect_repo_snippets,
)
from audio_service import assess_audio_output  # noqa: E402
from run_capability_matrix import (  # noqa: E402
    ValidationFailure,
    assess_compare_output,
    assess_ocr_output,
    assess_pdf_summary_output,
    assess_video_proxy_output,
    extract_json_object,
    is_external_blocker_message,
    run_selected_text_capabilities,
)
from run_long_context_demo import choose_backend  # noqa: E402


class GemmaRuntimeUtilityTests(unittest.TestCase):
    def test_timestamp_slug_uses_microsecond_precision(self) -> None:
        self.assertRegex(timestamp_slug(), r"^\d{8}T\d{12}Z-p\d+$")

    def test_build_repo_runtime_hint_flags_wrong_interpreter_and_old_python(self) -> None:
        hint = build_repo_runtime_hint(
            root=Path("/tmp/gemma-lab"),
            current_executable="/usr/bin/python3",
            current_version=(3, 9, 6),
        )
        self.assertIn("Python 3.10+", hint)
        self.assertIn("/tmp/gemma-lab/.venv/bin/python", hint)

    def test_build_repo_runtime_hint_is_empty_for_expected_repo_venv(self) -> None:
        root = Path("/tmp/gemma-lab")
        hint = build_repo_runtime_hint(
            root=root,
            current_executable=str(root / ".venv" / "bin" / "python"),
            current_version=(3, 13, 12),
        )
        self.assertEqual(hint, "")

    def test_build_requirements_install_command_targets_repo_venv(self) -> None:
        command = build_requirements_install_command(Path("/tmp/gemma-lab"))
        self.assertEqual(
            command,
            "/tmp/gemma-lab/.venv/bin/python -m pip install -r requirements.txt",
        )

    def test_parse_tool_arguments_handles_placeholder_quotes(self) -> None:
        payload = parse_tool_arguments('{asset_id:<|"|>sensor-7<|"|>, retries:2, enabled:true}')
        self.assertEqual(
            payload,
            {
                "asset_id": "sensor-7",
                "retries": 2,
                "enabled": True,
            },
        )

    def test_split_gemma_response_extracts_thinking_and_tool_call(self) -> None:
        raw = (
            "<|channel>thought\nNeed the tool first.<channel|>"
            "<|tool_call>call:lookup_lab_record{asset_id:<|\"|>sensor-7<|\"|>}<tool_call|>"
            "The calibration code is CAL-7Q4-ALPHA."
        )
        parsed = split_gemma_response(raw)
        self.assertEqual(parsed["thinking"], "Need the tool first.")
        self.assertEqual(parsed["content"], "The calibration code is CAL-7Q4-ALPHA.")
        self.assertEqual(parsed["tool_calls"][0]["function"]["name"], "lookup_lab_record")
        self.assertEqual(parsed["tool_calls"][0]["function"]["arguments"]["asset_id"], "sensor-7")


class LongContextUtilityTests(unittest.TestCase):
    def test_build_markers_is_deterministic(self) -> None:
        first = build_markers("seed-1")
        second = build_markers("seed-1")
        third = build_markers("seed-2")
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_build_synthetic_corpus_contains_all_markers(self) -> None:
        payload = build_synthetic_corpus(target_word_budget=512, seed="matrix-test")
        self.assertEqual(set(payload["markers"].keys()), set(MARKER_LABELS))
        corpus_text = payload["corpus_text"]
        for label, value in payload["markers"].items():
            self.assertIn(f"retrieval_marker_{label} = {value}", corpus_text)

    def test_collect_repo_snippets_skips_assets_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "README.md").write_text("hello repo\n", encoding="utf-8")
            (root / "scripts.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "assets").mkdir()
            (root / "assets" / "ignored.txt").write_text("ignore me\n", encoding="utf-8")
            (root / "artifacts").mkdir()
            (root / "artifacts" / "ignored.md").write_text("ignore me too\n", encoding="utf-8")

            snippets = collect_repo_snippets(root)
            paths = {item["path"] for item in snippets}
            self.assertIn("README.md", paths)
            self.assertNotIn("assets/ignored.txt", paths)
            self.assertNotIn("artifacts/ignored.md", paths)


class CapabilityMatrixUtilityTests(unittest.TestCase):
    def test_extract_json_object_accepts_fenced_json(self) -> None:
        payload = extract_json_object(
            """```json
            {"capability":"structured-json","language":"English","checks":["a","b"]}
            ```"""
        )
        self.assertEqual(payload["capability"], "structured-json")
        self.assertEqual(payload["checks"], ["a", "b"])

    def test_is_external_blocker_message_detects_dependency_issue(self) -> None:
        self.assertTrue(is_external_blocker_message("ffmpeg is not available for this run"))
        self.assertFalse(is_external_blocker_message("Model returned an empty text response."))

    def test_assess_video_proxy_output_requires_visible_change(self) -> None:
        records = [
            {"image": Image.new("RGB", (8, 8), (20, 40, 60))},
            {"image": Image.new("RGB", (8, 8), (100, 120, 140))},
            {"image": Image.new("RGB", (8, 8), (180, 200, 220))},
        ]
        validation = assess_video_proxy_output(records, "The object changes over time and moves to the right.")
        self.assertEqual(validation["quality_status"], "pass")
        self.assertTrue(all(check["pass"] for check in validation["quality_checks"]))

    def test_assess_ocr_output_requires_anchor_lines(self) -> None:
        validation = assess_ocr_output(
            "\n".join(
                [
                    "Gemma Lab Invoice",
                    "Invoice ID: G4-001",
                    "Ship To: Gemma Lab",
                    "Total Items: 3",
                    "STATUS: READY FOR OCR",
                ]
            )
        )
        self.assertEqual(validation["quality_status"], "pass")

        failed = assess_ocr_output("Gemma Lab Invoice\nInvoice ID: G4-001")
        self.assertEqual(failed["quality_status"], "fail")

    def test_assess_compare_output_requires_multiple_fixture_differences(self) -> None:
        validation = assess_compare_output(
            "The mug moved to the right, changed to a teal color, and the notebook annotation now says Updated layout."
        )
        self.assertEqual(validation["quality_status"], "pass")

        failed = assess_compare_output("The two images look identical with no meaningful difference.")
        self.assertEqual(failed["quality_status"], "fail")

    def test_assess_pdf_summary_requires_key_facts_and_fixture_facts(self) -> None:
        validation = assess_pdf_summary_output(
            "\n".join(
                [
                    "- Gemma Lab owns the sample document.",
                    "- It is a Phase 2 vision sample launched on 2026-04-06.",
                    "- The goal covers caption, OCR, compare, and document summary flows.",
                    "Key facts:",
                    "- Owner: Gemma Lab",
                    "- Phase 2",
                    "- Launch Date: 2026-04-06",
                ]
            )
        )
        self.assertEqual(validation["quality_status"], "pass")

        failed = assess_pdf_summary_output("Short summary only. No heading and no concrete facts.")
        self.assertEqual(failed["quality_status"], "fail")

    def test_validation_failure_maps_to_quality_fail_in_capability_runner(self) -> None:
        fake_session = SimpleNamespace(model_id="google/gemma-4-E2B-it")

        with patch(
            "run_capability_matrix.run_text_generation",
            side_effect=ValidationFailure("Model returned an empty text response."),
        ):
            results = run_selected_text_capabilities(fake_session, {"text-chat"})

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "quality_fail")
        self.assertEqual(results[0].quality_status, "fail")
        self.assertIsNone(results[0].blocker)


class AudioUtilityTests(unittest.TestCase):
    def test_assess_audio_output_accepts_clear_sample_transcript(self) -> None:
        record = {
            "source_path": str((Path(__file__).resolve().parents[1] / "assets" / "audio" / "sample_audio.wav").resolve()),
            "signal_summary": {
                "active_seconds": 4.0,
                "active_frame_ratio": 0.7,
            },
        }
        validation = assess_audio_output(
            mode="transcribe",
            record=record,
            output_text="Jema audio demo please transcribe this short validation clip",
            target_language="Japanese",
        )
        self.assertEqual(validation["quality_status"], "pass")

    def test_assess_audio_output_rejects_invented_transcript_for_quiet_clip(self) -> None:
        record = {
            "source_path": "/tmp/quiet.wav",
            "signal_summary": {
                "active_seconds": 0.1,
                "active_frame_ratio": 0.02,
            },
        }
        validation = assess_audio_output(
            mode="transcribe",
            record=record,
            output_text="invented transcript",
            target_language="Japanese",
        )
        self.assertEqual(validation["quality_status"], "fail")

    def test_assess_audio_output_rejects_translation_with_source_language_leakage(self) -> None:
        record = {
            "source_path": str((Path(__file__).resolve().parents[1] / "assets" / "audio" / "sample_audio.wav").resolve()),
            "signal_summary": {
                "active_seconds": 4.0,
                "active_frame_ratio": 0.7,
            },
        }
        validation = assess_audio_output(
            mode="translate",
            record=record,
            output_text="Gemma audio demo をお願いします please transcribe this validation clip",
            target_language="Japanese",
            pipeline={
                "intermediate_transcript": "Gemma audio demo please transcribe this short validation clip",
            },
        )
        self.assertEqual(validation["quality_status"], "fail")
        leakage_checks = [check for check in validation["quality_checks"] if check["name"] == "source_language_tokens_removed"]
        self.assertEqual(len(leakage_checks), 1)
        self.assertFalse(leakage_checks[0]["pass"])


class LongContextModeTests(unittest.TestCase):
    def test_choose_backend_prefers_simulate_for_large_auto_target(self) -> None:
        backend, notes = choose_backend("auto", target_prompt_tokens=98_304, device_info={"name": "mps"})
        self.assertEqual(backend, "simulate")
        self.assertTrue(notes)


class DoctorUtilityTests(unittest.TestCase):
    def test_preferred_runtime_device_label_prefers_mps_on_apple_silicon(self) -> None:
        label = preferred_runtime_device_label(
            {
                "cuda_available": False,
                "mps_available": True,
                "gpu_devices": [],
            }
        )
        self.assertEqual(label, "mps (Apple Metal)")


if __name__ == "__main__":
    unittest.main()
