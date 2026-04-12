from __future__ import annotations

import base64
import sys
import tempfile
import threading
import time
import unittest
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
    write_artifact,
)
from gemma_core import SessionKey  # noqa: E402
from gemma_runtime import (  # noqa: E402
    WarmupProgress,
    WARMUP_PHASE_LOAD_MODEL,
    WARMUP_PHASE_PRIME_TOKEN,
)
from run_local_ui import (  # noqa: E402
    AUDIO_BACKEND,
    CHAT_BACKEND,
    JOB_STATE_CANCELLED,
    JOB_STATE_COMPLETED,
    JOB_STATE_FAILED,
    JOB_STATE_RUNNING,
    LocalUiApp,
    LocalUiController,
    LocalUiJobRunner,
    PREWARM_ACTION,
    PREWARM_BACKEND,
    STARTUP_PREWARM_GRACE_MS,
    UiActionResult,
    UiJobSnapshot,
    build_artifact_overview_fields,
    build_entry_compare_fields,
    build_thinking_debug_report,
)
from workspace_state import WorkspaceSessionStore, session_manifest_path  # noqa: E402


class FakeRuntimeSession:
    def __init__(self, device_info: dict[str, object]) -> None:
        self.device_info = device_info


class FakeDType:
    def __str__(self) -> str:
        return "float16"


class FakeSessionManager:
    def __init__(self) -> None:
        self.closed = 0
        self.requests: list[tuple[str, str]] = []
        self._cached_keys: list[SessionKey] = []

    def get_session(
        self,
        session_kind: str,
        model_id: str,
        *,
        progress_callback=None,
    ) -> FakeRuntimeSession:
        self.requests.append((session_kind, model_id))
        key = SessionKey(session_kind=session_kind, model_id=model_id, device_class="cpu")
        self._cached_keys = [key]
        if progress_callback is not None:
            progress_callback("Using cached shared session")
        return FakeRuntimeSession(
            {
                "name": "cpu",
                "label": "cpu",
                "dtype_name": "float32",
                "dtype": "float32",
            }
        )

    def close_all(self) -> int:
        self.closed += 1
        self._cached_keys = []
        return 1

    def cached_keys(self) -> list[SessionKey]:
        return list(self._cached_keys)


class FakeVar:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


class FakeRoot:
    def __init__(self) -> None:
        self.after_calls: list[SimpleNamespace] = []
        self.cancelled_after_ids: list[object] = []
        self._next_after_id = 1

    def after(self, delay_ms: int, callback) -> str:
        after_id = f"after-{self._next_after_id}"
        self._next_after_id += 1
        self.after_calls.append(
            SimpleNamespace(
                after_id=after_id,
                delay_ms=delay_ms,
                callback=callback,
            )
        )
        return after_id

    def after_cancel(self, after_id: object) -> None:
        self.cancelled_after_ids.append(after_id)


class FakeAppController:
    def __init__(self, *, warm: bool = False) -> None:
        self.selected_model_id = "google/gemma-4-E2B-it"
        self.warm = warm

    def has_cached_selected_text_session(self) -> bool:
        return self.warm

    def collect_diagnostics(self) -> dict[str, object]:
        return {"selected_model_id": self.selected_model_id, "warm": self.warm}


class FakeAppJobRunner:
    def __init__(self) -> None:
        self.pending = False
        self.active_snapshot = None
        self.submissions: list[dict[str, object]] = []

    def has_pending_work(self) -> bool:
        return self.pending

    def active_job(self):
        return self.active_snapshot

    def submit(self, *, action: str, work, timeout_seconds: float | None, message: str):
        self.submissions.append(
            {
                "action": action,
                "work": work,
                "timeout_seconds": timeout_seconds,
                "message": message,
            }
        )
        return SimpleNamespace(job_id=len(self.submissions))


class LocalUiControllerTests(unittest.TestCase):
    def make_controller(
        self,
        manager: FakeSessionManager | None = None,
    ) -> tuple[LocalUiController, WorkspaceSessionStore, Path]:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name)
        store = WorkspaceSessionStore(root=root)
        controller = LocalUiController(
            session_manager=manager or FakeSessionManager(),
            workspace_store=store,
        )
        return controller, store, root

    def test_set_model_id_clears_sessions_only_when_model_changes(self) -> None:
        manager = FakeSessionManager()
        controller, _store, _root = self.make_controller(manager)
        initial = controller.selected_model_id

        self.assertEqual(controller.set_model_id(initial), initial)
        self.assertEqual(manager.closed, 0)

        updated = controller.set_model_id("google/gemma-4-E4B-it")
        self.assertEqual(updated, "google/gemma-4-E4B-it")
        self.assertEqual(manager.closed, 1)

    def test_build_artifact_overview_fields_formats_primary_metadata(self) -> None:
        fields = dict(
            build_artifact_overview_fields(
                {
                    "artifact_path": "/tmp/example.json",
                    "artifact_summary": {
                        "artifact_kind": "vision",
                        "status": "ok",
                        "validation_mode": "live",
                        "execution_status": "ok",
                        "quality_status": "not_validated",
                        "runtime_backend": "gemma-live-vision",
                        "model_id": "google/gemma-4-E2B-it",
                        "timestamp_utc": "2026-04-10T01:02:03Z",
                    },
                }
            )
        )

        self.assertEqual(fields["Status"], "vision / ok")
        self.assertEqual(fields["Validation"], "live / ok / not_validated")
        self.assertEqual(fields["Backend"], "gemma-live-vision")
        self.assertEqual(fields["Model"], "google/gemma-4-E2B-it")
        self.assertEqual(fields["Timestamp"], "2026-04-10T01:02:03Z")

    def test_build_artifact_overview_fields_handles_missing_or_unreadable_artifact(self) -> None:
        no_selection = dict(build_artifact_overview_fields({}))
        unreadable = dict(
            build_artifact_overview_fields(
                {
                    "artifact_path": "/tmp/example.json",
                    "artifact_error": "FileNotFoundError: missing",
                }
            )
        )

        self.assertEqual(no_selection["Status"], "No artifact selected")
        self.assertEqual(no_selection["Validation"], "Choose an artifact")
        self.assertEqual(unreadable["Status"], "Artifact unreadable")
        self.assertEqual(unreadable["Validation"], "Unavailable")

    def test_build_entry_compare_fields_reports_previous_selected_and_change(self) -> None:
        fields = dict(
            build_entry_compare_fields(
                {
                    "selected_entry_summary": {
                        "recorded_at_utc": "2026-04-12T01:10:00Z",
                        "mode": "caption",
                        "status": "ok",
                        "validation_summary": "live / ok / not_validated",
                        "output_preview": "Selected output",
                    },
                    "previous_entry_summary": {
                        "recorded_at_utc": "2026-04-12T01:05:00Z",
                        "mode": "ocr",
                        "status": "blocked",
                        "validation_summary": "live / blocked / not_run",
                        "output_preview": "Previous output",
                    },
                }
            )
        )

        self.assertIn("2026-04-12T01:05:00Z", fields["Previous"])
        self.assertIn("2026-04-12T01:10:00Z", fields["Selected"])
        self.assertIn("status: blocked -> ok", fields["Change"])
        self.assertIn("mode: ocr -> caption", fields["Change"])
        self.assertIn("output: preview changed", fields["Change"])

    def test_run_chat_uses_shared_session_manager_and_writes_artifact(self) -> None:
        manager = FakeSessionManager()
        controller, store, _root = self.make_controller(manager)
        controller.set_model_id("google/gemma-4-E4B-it")

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "chat-ui.json"
            with (
                patch("run_local_ui.default_text_output_path", return_value=artifact_path),
                patch("run_local_ui.run_text_task") as mocked_run_text_task,
                patch("run_local_ui.write_artifact") as mocked_write_artifact,
            ):
                mocked_run_text_task.return_value = {
                    "task": "chat",
                    "model_id": "google/gemma-4-E4B-it",
                    "prompt": "Explain binary search.",
                    "system_prompt": "You are a concise, helpful assistant.",
                    "resolved_user_prompt": "Explain binary search.",
                    "generation_settings": {"max_new_tokens": 192, "do_sample": False},
                    "device_info": {
                        "name": "cpu",
                        "label": "cpu",
                        "dtype_name": "float32",
                        "dtype": "float32",
                    },
                    "elapsed_seconds": 0.25,
                    "output_text": "Binary search halves the search space.",
                }

                result = controller.run_chat(prompt="Explain binary search.")

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.backend, CHAT_BACKEND)
        self.assertEqual(result.artifact_path, artifact_path)
        self.assertIs(mocked_run_text_task.call_args.kwargs["session_manager"], manager)
        self.assertEqual(mocked_run_text_task.call_args.kwargs["model_id"], "google/gemma-4-E4B-it")
        self.assertEqual(
            mocked_run_text_task.call_args.kwargs["messages"],
            [
                {"role": "system", "content": "You are a concise, helpful assistant."},
                {"role": "user", "content": "Explain binary search."},
            ],
        )
        self.assertIs(controller.last_result, result)
        self.assertEqual(mocked_write_artifact.call_args[0][0], artifact_path)
        self.assertEqual(mocked_write_artifact.call_args[0][1]["status"], "ok")
        self.assertEqual(mocked_write_artifact.call_args[0][1]["validation"]["validation_mode"], "live")
        self.assertEqual(mocked_write_artifact.call_args[0][1]["validation"]["quality_status"], "not_validated")
        active_session = store.active_session("chat")
        self.assertIsNotNone(active_session)
        self.assertEqual(active_session["artifact_refs"][0]["artifact_path"], str(artifact_path.resolve()))
        self.assertEqual(active_session["history_for_next_turn"][-1]["content"], "Binary search halves the search space.")

    def test_run_chat_reuses_saved_history_on_next_turn(self) -> None:
        manager = FakeSessionManager()
        controller, _store, _root = self.make_controller(manager)

        first_result = {
            "task": "chat",
            "model_id": controller.selected_model_id,
            "prompt": "First question",
            "system_prompt": "You are a concise, helpful assistant.",
            "resolved_user_prompt": "First question",
            "generation_settings": {"max_new_tokens": 192, "do_sample": False},
            "device_info": {
                "name": "cpu",
                "label": "cpu",
                "dtype_name": "float32",
                "dtype": "float32",
            },
            "elapsed_seconds": 0.2,
            "output_text": "First answer",
        }
        second_result = dict(first_result)
        second_result.update(
            {
                "prompt": "Second question",
                "resolved_user_prompt": "Second question",
                "output_text": "Second answer",
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_one = Path(tmpdir) / "chat-one.json"
            artifact_two = Path(tmpdir) / "chat-two.json"
            with (
                patch("run_local_ui.default_text_output_path", side_effect=[artifact_one, artifact_two]),
                patch("run_local_ui.run_text_task", side_effect=[first_result, second_result]) as mocked_run_text_task,
                patch("run_local_ui.write_artifact"),
            ):
                controller.run_chat(prompt="First question")
                controller.run_chat(prompt="Second question")

        first_messages = mocked_run_text_task.call_args_list[0].kwargs["messages"]
        second_messages = mocked_run_text_task.call_args_list[1].kwargs["messages"]
        self.assertEqual(
            first_messages,
            [
                {"role": "system", "content": "You are a concise, helpful assistant."},
                {"role": "user", "content": "First question"},
            ],
        )
        self.assertEqual(
            second_messages,
            [
                {"role": "system", "content": "You are a concise, helpful assistant."},
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second question"},
            ],
        )

    def test_run_audio_passes_selected_model_into_audio_runner(self) -> None:
        manager = FakeSessionManager()
        controller, _store, _root = self.make_controller(manager)
        controller.set_model_id("google/gemma-4-E4B-it")

        captured: dict[str, object] = {}

        def fake_run_audio_mode(**kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return {
                "base_model_id": "google/gemma-4-E4B-it",
                "model_id": "google/gemma-4-E4B-it",
                "model_id_source": "selected_base_model",
                "device_info": {
                    "name": "cpu",
                    "label": "cpu",
                    "dtype_name": "float32",
                    "dtype": "float32",
                },
                "mode": "transcribe",
                "prompt": "prompt",
                "system_prompt": "system",
                "generation_settings": {"max_new_tokens": 32, "do_sample": False},
                "target_language": None,
                "record": {
                    "source_path": "/tmp/sample.wav",
                    "resolved_path": "/tmp/sample.wav",
                    "format": "wav",
                    "normalized_format": "wav",
                    "loader": "soundfile",
                    "source_sample_rate_hz": 16000,
                    "sample_rate_hz": 16000,
                    "source_channels": 1,
                    "channels": 1,
                    "duration_seconds": 1.0,
                    "frame_count": 16000,
                    "signal_summary": {"active_seconds": 1.0, "active_frame_ratio": 0.8},
                    "lineage": [],
                },
                "elapsed_seconds": 0.5,
                "output_text": "transcript",
                "validation": {
                    "quality_notes": [],
                },
                "pipeline": None,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "audio-ui.json"
            with (
                patch("run_local_ui.default_audio_output_path", return_value=artifact_path),
                patch("run_local_ui.run_audio_mode", side_effect=fake_run_audio_mode),
                patch("run_local_ui.write_artifact"),
            ):
                result = controller.run_audio(mode="transcribe", input_path="/tmp/sample.wav")

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.backend, AUDIO_BACKEND)
        self.assertEqual(captured["base_model_id"], "google/gemma-4-E4B-it")
        self.assertIs(captured["session_manager"], manager)

    def test_run_vision_writes_validation_stub_for_forensics(self) -> None:
        manager = FakeSessionManager()
        controller, _store, _root = self.make_controller(manager)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "vision-ui.json"
            with (
                patch("run_local_ui.default_vision_output_path", return_value=artifact_path),
                patch("run_local_ui.run_vision_mode") as mocked_run_vision_mode,
                patch("run_local_ui.write_artifact") as mocked_write_artifact,
            ):
                mocked_run_vision_mode.return_value = {
                    "mode": "pdf-summary",
                    "model_id": controller.selected_model_id,
                    "prompt": "Summarize this document.",
                    "system_prompt": "Summarize faithfully.",
                    "resolved_user_prompt": "Summarize this document.",
                    "generation_settings": {"max_new_tokens": 256, "do_sample": False},
                    "records": [
                        {
                            "source_path": "/tmp/sample.pdf",
                            "resolved_path": "/tmp/sample-page-1.png",
                            "kind": "image",
                            "page_number": 1,
                            "frame_index": None,
                            "timestamp_seconds": None,
                            "width": 512,
                            "height": 512,
                            "lineage": [],
                        }
                    ],
                    "device_info": {
                        "name": "cpu",
                        "label": "cpu",
                        "dtype_name": "float32",
                        "dtype": "float32",
                    },
                    "elapsed_seconds": 0.4,
                    "output_text": "Summary",
                }

                result = controller.run_vision(mode="pdf-summary", inputs=["/tmp/sample.pdf"])

        payload = mocked_write_artifact.call_args[0][1]
        self.assertEqual(result.status, "ok")
        self.assertEqual(payload["validation"]["validation_mode"], "live")
        self.assertEqual(payload["validation"]["quality_status"], "not_validated")

    def test_collect_diagnostics_reports_cached_sessions(self) -> None:
        manager = FakeSessionManager()
        controller, _store, _root = self.make_controller(manager)

        with (
            patch("run_local_ui.probe_torch", return_value={"installed": True}),
            patch("run_local_ui.probe_transformers", return_value={"installed": True}),
            patch("run_local_ui.probe_optional_modules", return_value={"pillow": True}),
            patch("run_local_ui.assets_summary", return_value={"assets_root_exists": True}),
            patch("run_local_ui.resolve_audio_model_selection", return_value=("google/gemma-4-E2B-it", "selected_base_model")),
        ):
            manager.get_session("text", controller.selected_model_id)
            diagnostics = controller.collect_diagnostics()

        self.assertEqual(diagnostics["selected_model_id"], controller.selected_model_id)
        self.assertEqual(diagnostics["resolved_audio_model_source"], "selected_base_model")
        self.assertEqual(
            diagnostics["cached_sessions"][0]["session_kind"],
            "text",
        )
        self.assertEqual(diagnostics["workspace_state"]["selected_model_id"], controller.selected_model_id)

    def test_collect_forensics_returns_session_trail_and_artifact_summary(self) -> None:
        controller, store, root = self.make_controller()
        artifact_path = root / "artifacts" / "audio" / "trace-audio.json"
        write_artifact(
            artifact_path,
            build_artifact_payload(
                artifact_kind="audio",
                status="ok",
                runtime=build_runtime_record(
                    backend="gemma-live-audio-translation-pipeline",
                    model_id=controller.selected_model_id,
                    device_info="cpu",
                    elapsed_seconds=0.75,
                ),
                prompts=build_prompt_record(
                    system_prompt="Translate carefully.",
                    prompt="Translate the spoken content.",
                    resolved_user_prompt="Translate the spoken content.",
                ),
                asset_lineage=[
                    {
                        "source_path": str(root / "assets" / "audio" / "sample_audio.wav"),
                        "resolved_path": str(root / "artifacts" / "cache" / "sample.wav"),
                        "cache_path": str(root / "artifacts" / "cache" / "sample.wav"),
                        "asset_kind": "audio",
                        "transform": "audio_normalization",
                        "cache_key": "sample-audio",
                        "cache_hit": True,
                        "metadata": {"duration_seconds": 1.0},
                    }
                ],
                extra={
                    "output_text": "これは検証用の翻訳です。",
                    "pipeline": {"strategy": "audio_transcript_then_text_translate"},
                    "validation": {
                        "validation_mode": "pipeline",
                        "claim_scope": "conservative transcript-first translation pipeline",
                        "pass_definition": "Pass means the transcript and final translation both meet the deterministic checks.",
                        "execution_status": "ok",
                        "quality_status": "pass",
                        "quality_checks": [
                            {"name": "target_script_present", "pass": True, "detail": "Japanese characters detected."}
                        ],
                        "quality_notes": ["Translation used the conservative transcript-first pipeline."],
                    },
                },
            ),
        )
        store.record_session_run(
            surface="audio",
            model_id=controller.selected_model_id,
            mode="translate",
            artifact_kind="audio",
            artifact_path=artifact_path,
            status="ok",
            prompt="Translate the spoken content.",
            system_prompt="Translate carefully.",
            resolved_user_prompt="Translate the spoken content.",
            output_text="これは検証用の翻訳です。",
            attachments=[{"role": "audio_input", "path": root / "assets" / "audio" / "sample_audio.wav"}],
            notes=["Translation used the conservative transcript-first pipeline."],
            options={"target_language": "Japanese"},
        )

        snapshot = controller.collect_forensics(surface="audio")

        self.assertEqual(snapshot["surface"], "audio")
        self.assertIsNotNone(snapshot["active_session_summary"])
        self.assertEqual(snapshot["artifact_source"], "selected_entry")
        self.assertEqual(snapshot["artifact_summary"]["validation_mode"], "pipeline")
        self.assertEqual(snapshot["artifact_summary"]["runtime_backend"], "gemma-live-audio-translation-pipeline")
        self.assertEqual(snapshot["artifact_summary"]["lineage_count"], 1)
        self.assertEqual(snapshot["recent_entries"][0]["mode"], "translate")
        self.assertEqual(snapshot["recent_sessions"][0]["surface"], "audio")
        self.assertEqual(snapshot["recent_sessions"][0]["latest_validation_summary"], "pipeline / ok / pass")
        self.assertIn("Long-context local 16k / external 96k", snapshot["capability_badges"])

    def test_collect_forensics_accepts_artifact_override_without_session_history(self) -> None:
        controller, _store, root = self.make_controller()
        artifact_path = root / "artifacts" / "thinking" / "manual-thinking.json"
        write_artifact(
            artifact_path,
            build_artifact_payload(
                artifact_kind="thinking",
                status="ok",
                runtime=build_runtime_record(
                    backend="gemma-live-thinking",
                    model_id=controller.selected_model_id,
                    device_info="cpu",
                    elapsed_seconds=0.42,
                ),
                prompts=build_prompt_record(
                    system_prompt="Think privately.",
                    prompt="Question",
                    resolved_user_prompt="Question",
                ),
                extra={
                    "final_answer": "Answer",
                    "raw_thinking_saved_to_artifact": True,
                    "debug_default_hidden": True,
                    "validation": {
                        "validation_mode": "live",
                        "claim_scope": "live model generation on a small local prompt",
                        "pass_definition": "Pass means the run completed.",
                        "execution_status": "ok",
                        "quality_status": "pass",
                        "quality_checks": [],
                        "quality_notes": [],
                    },
                },
            ),
        )

        snapshot = controller.collect_forensics(surface="thinking", artifact_path=str(artifact_path))

        self.assertEqual(snapshot["artifact_source"], "override")
        self.assertEqual(snapshot["artifact_summary"]["artifact_kind"], "thinking")
        self.assertTrue(snapshot["artifact_summary"]["raw_thinking_saved_to_artifact"])

    def test_collect_forensics_exposes_quality_checks_lineage_and_preview_image(self) -> None:
        controller, store, root = self.make_controller()
        preview_path = root / "artifacts" / "cache" / "preview.png"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WnSUs8AAAAASUVORK5CYII="
            )
        )
        artifact_path = root / "artifacts" / "vision" / "trace-vision.json"
        write_artifact(
            artifact_path,
            build_artifact_payload(
                artifact_kind="vision",
                status="ok",
                runtime=build_runtime_record(
                    backend="gemma-live-vision",
                    model_id=controller.selected_model_id,
                    device_info="cpu",
                    elapsed_seconds=0.4,
                ),
                prompts=build_prompt_record(
                    system_prompt="Describe only what is visible.",
                    prompt="Describe this image.",
                    resolved_user_prompt="Describe this image.",
                ),
                asset_lineage=[
                    {
                        "source_path": str(root / "assets" / "images" / "sample.png"),
                        "resolved_path": str(preview_path),
                        "cache_path": str(preview_path),
                        "asset_kind": "image",
                        "transform": "image_normalization",
                        "cache_key": "sample-preview",
                    }
                ],
                extra={
                    "output_text": "A concise caption.",
                    "inputs": [
                        {
                            "source_path": str(root / "assets" / "images" / "sample.png"),
                            "resolved_path": str(preview_path),
                            "kind": "image",
                            "page_number": None,
                            "frame_index": None,
                            "timestamp_seconds": None,
                            "width": 1,
                            "height": 1,
                        }
                    ],
                    "validation": {
                        "validation_mode": "live",
                        "claim_scope": "user-driven local UI image/pdf execution",
                        "pass_definition": "Pass means the run completed.",
                        "execution_status": "ok",
                        "quality_status": "not_validated",
                        "quality_checks": [
                            {"name": "asset_loaded", "pass": True, "detail": "Image input resolved."},
                            {"name": "fixture_match", "pass": False, "detail": "User input is not a deterministic fixture."},
                        ],
                        "quality_notes": ["Preview should come from the normalized PNG cache path."],
                    },
                },
            ),
        )
        store.record_session_run(
            surface="vision",
            model_id=controller.selected_model_id,
            mode="caption",
            artifact_kind="vision",
            artifact_path=artifact_path,
            status="ok",
            prompt="Describe this image.",
            system_prompt="Describe only what is visible.",
            resolved_user_prompt="Describe this image.",
            output_text="A concise caption.",
            attachments=[{"role": "primary_input", "path": root / "assets" / "images" / "sample.png"}],
            notes=[],
            options={},
        )

        snapshot = controller.collect_forensics(surface="vision")

        self.assertEqual(snapshot["artifact_source"], "selected_entry")
        self.assertEqual(snapshot["artifact_summary"]["preview_image_path"], str(preview_path.resolve()))
        self.assertEqual(len(snapshot["artifact_summary"]["quality_checks"]), 2)
        self.assertEqual(snapshot["artifact_summary"]["lineage_preview"][0]["transform"], "image_normalization")

    def test_collect_forensics_can_select_prior_entry_without_manual_artifact_override(self) -> None:
        controller, store, root = self.make_controller()
        artifact_one = root / "artifacts" / "text" / "chat-one.json"
        artifact_two = root / "artifacts" / "text" / "chat-two.json"

        for artifact_path, prompt_text, output_text in (
            (artifact_one, "First question", "First answer"),
            (artifact_two, "Second question", "Second answer"),
        ):
            write_artifact(
                artifact_path,
                build_artifact_payload(
                    artifact_kind="text",
                    status="ok",
                    runtime=build_runtime_record(
                        backend="gemma-live-text",
                        model_id=controller.selected_model_id,
                        device_info="cpu",
                        elapsed_seconds=0.2,
                    ),
                    prompts=build_prompt_record(
                        system_prompt="You are a concise, helpful assistant.",
                        prompt=prompt_text,
                        resolved_user_prompt=prompt_text,
                    ),
                    extra={
                        "output_text": output_text,
                        "validation": {
                            "validation_mode": "live",
                            "claim_scope": "user-driven local UI text/chat execution",
                            "pass_definition": "Pass means the run completed.",
                            "execution_status": "ok",
                            "quality_status": "not_validated",
                            "quality_checks": [],
                            "quality_notes": [],
                        },
                    },
                ),
            )

        first_messages = store.chat_messages_for_next_turn(
            model_id=controller.selected_model_id,
            system_prompt="You are a concise, helpful assistant.",
        )
        store.record_chat_turn(
            model_id=controller.selected_model_id,
            status="ok",
            artifact_path=artifact_one,
            prompt="First question",
            system_prompt="You are a concise, helpful assistant.",
            resolved_user_prompt="First question",
            output_text="First answer",
            base_messages=first_messages,
        )
        second_messages = store.chat_messages_for_next_turn(
            model_id=controller.selected_model_id,
            system_prompt="You are a concise, helpful assistant.",
        )
        store.record_chat_turn(
            model_id=controller.selected_model_id,
            status="ok",
            artifact_path=artifact_two,
            prompt="Second question",
            system_prompt="You are a concise, helpful assistant.",
            resolved_user_prompt="Second question",
            output_text="Second answer",
            base_messages=second_messages,
        )

        session = store.active_session("chat")
        self.assertIsNotNone(session)
        prior_entry_id = session["entries"][0]["entry_id"]
        latest_entry_id = session["entries"][1]["entry_id"]

        snapshot = controller.collect_forensics(surface="chat", entry_id=prior_entry_id)

        self.assertEqual(snapshot["selected_entry_id"], prior_entry_id)
        self.assertEqual(snapshot["artifact_source"], "selected_entry")
        self.assertEqual(snapshot["artifact_path"], str(artifact_one.resolve()))
        self.assertEqual(snapshot["selected_entry_summary"]["entry_id"], prior_entry_id)
        self.assertIsNone(snapshot["previous_entry_summary"])
        self.assertEqual(snapshot["recent_entries"][0]["entry_id"], latest_entry_id)
        self.assertEqual(snapshot["recent_entries"][1]["entry_id"], prior_entry_id)
        self.assertEqual(snapshot["recent_entries"][1]["validation_summary"], "live / ok / not_validated")

        latest_snapshot = controller.collect_forensics(surface="chat")
        self.assertEqual(latest_snapshot["selected_entry_summary"]["entry_id"], latest_entry_id)
        self.assertEqual(latest_snapshot["previous_entry_summary"]["entry_id"], prior_entry_id)

    def test_collect_forensics_handles_empty_workspace_and_stale_session_manifest(self) -> None:
        controller, store, root = self.make_controller()

        empty_snapshot = controller.collect_forensics(surface="chat")
        self.assertEqual(empty_snapshot["recent_sessions"], [])
        self.assertIsNone(empty_snapshot["selected_session_summary"])
        self.assertIsNone(empty_snapshot["artifact_path"])

        artifact_path = root / "artifacts" / "audio" / "stale-audio.json"
        write_artifact(
            artifact_path,
            build_artifact_payload(
                artifact_kind="audio",
                status="ok",
                runtime=build_runtime_record(
                    backend="gemma-live-audio",
                    model_id=controller.selected_model_id,
                    device_info="cpu",
                    elapsed_seconds=0.3,
                ),
                prompts=build_prompt_record(
                    system_prompt="Transcribe carefully.",
                    prompt="Transcribe this clip.",
                    resolved_user_prompt="Transcribe this clip.",
                ),
                extra={
                    "output_text": "transcript",
                    "validation": {
                        "validation_mode": "live",
                        "claim_scope": "user-driven local UI audio execution",
                        "pass_definition": "Pass means the run completed.",
                        "execution_status": "ok",
                        "quality_status": "not_validated",
                        "quality_checks": [],
                        "quality_notes": [],
                    },
                },
            ),
        )
        store.record_session_run(
            surface="audio",
            model_id=controller.selected_model_id,
            mode="transcribe",
            artifact_kind="audio",
            artifact_path=artifact_path,
            status="ok",
            prompt="Transcribe this clip.",
            system_prompt="Transcribe carefully.",
            resolved_user_prompt="Transcribe this clip.",
            output_text="transcript",
            attachments=[{"role": "audio_input", "path": root / "assets" / "audio" / "sample_audio.wav"}],
            notes=[],
            options={},
        )
        stale_manifest = session_manifest_path(
            session_id="audio-main",
            workspace_id=store.workspace_id,
            root=root,
        )
        stale_manifest.unlink()

        stale_snapshot = controller.collect_forensics(surface="audio")

        self.assertEqual(stale_snapshot["recent_sessions"][0]["session_state"], "stale_manifest")
        self.assertIn("Session manifest is missing", stale_snapshot["selected_session_error"])
        self.assertEqual(stale_snapshot["artifact_source"], "selected_session_latest")
        self.assertEqual(stale_snapshot["artifact_summary"]["artifact_kind"], "audio")

    def test_prewarm_selected_model_uses_shared_manager_and_records_summary(self) -> None:
        manager = FakeSessionManager()
        controller, _store, _root = self.make_controller(manager)

        with patch("run_local_ui.warm_thinking_session") as mocked_warm:
            mocked_warm.return_value = {
                "model_id": controller.selected_model_id,
                "device_info": {
                    "name": "cpu",
                    "label": "cpu",
                    "dtype_name": "float32",
                    "dtype": "float32",
                },
                "elapsed_seconds": 1.25,
                "primed_text": "READY",
            }

            result = controller.prewarm_selected_model()

        self.assertEqual(result.action, PREWARM_ACTION)
        self.assertEqual(result.backend, PREWARM_BACKEND)
        self.assertEqual(result.status, "ok")
        self.assertEqual(controller.last_prewarm["elapsed_seconds"], 1.25)
        self.assertEqual(controller.last_prewarm["primed_text"], "READY")
        self.assertIs(mocked_warm.call_args.kwargs["session_manager"], manager)
        self.assertEqual(mocked_warm.call_args.kwargs["model_id"], controller.selected_model_id)

    def test_run_thinking_normalizes_device_info_before_writing_artifact(self) -> None:
        manager = FakeSessionManager()
        controller, _store, _root = self.make_controller(manager)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "thinking-ui.json"
            with (
                patch("run_local_ui.default_thinking_artifact_path", return_value=artifact_path),
                patch("run_local_ui.run_thinking_session") as mocked_run_thinking_session,
                patch("run_local_ui.write_artifact") as mocked_write_artifact,
            ):
                mocked_run_thinking_session.return_value = {
                    "mode": "text",
                    "model_id": controller.selected_model_id,
                    "device_info": {
                        "name": "mps",
                        "label": "mps (Apple Metal)",
                        "dtype_name": "float16",
                        "dtype": FakeDType(),
                    },
                    "system_prompt": "system",
                    "first_prompt": "prompt",
                    "follow_up_prompt": "follow-up",
                    "elapsed_seconds": 0.42,
                    "final_answer": "answer",
                    "turns": [],
                }

                result = controller.run_thinking(mode="text")

        payload = mocked_write_artifact.call_args[0][1]
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.output_text, "answer")
        self.assertEqual(payload["device_info"]["dtype"], "float16")
        self.assertEqual(payload["runtime"]["device"]["dtype"], "float16")


class LocalUiWorkerTests(unittest.TestCase):
    def _wait_for_state(self, runner: LocalUiJobRunner, *, job_id: int, state: str, timeout: float = 2.0):
        deadline = time.monotonic() + timeout
        snapshots = []
        while time.monotonic() < deadline:
            runner.expire_timeouts()
            events = runner.pop_events()
            snapshots.extend(events)
            for snapshot in events:
                if snapshot.job_id == job_id and snapshot.state == state:
                    return snapshot, snapshots
            time.sleep(0.01)
        self.fail(f"Timed out waiting for job {job_id} to reach state {state}.")

    def test_job_runner_executes_work_on_background_thread(self) -> None:
        runner = LocalUiJobRunner()
        main_thread_id = threading.get_ident()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_path = Path(tmpdir) / "worker-chat.json"

                def work(_cancel_signal, _report_progress) -> UiActionResult:
                    return UiActionResult(
                        action="chat",
                        status=f"thread:{threading.get_ident()}",
                        output_text="ok",
                        artifact_path=artifact_path,
                        model_id="google/gemma-4-E2B-it",
                        backend=CHAT_BACKEND,
                        device_label="cpu",
                        dtype_name="float32",
                    )

                submitted = runner.submit(
                    action="chat",
                    work=work,
                    timeout_seconds=1.0,
                    message="Running chat...",
                )
                completed, snapshots = self._wait_for_state(runner, job_id=submitted.job_id, state=JOB_STATE_COMPLETED)
        finally:
            runner.close()

        seen_states = {snapshot.state for snapshot in snapshots if snapshot.job_id == submitted.job_id}
        self.assertIn(JOB_STATE_RUNNING, seen_states)
        self.assertIn(JOB_STATE_COMPLETED, seen_states)
        self.assertIsNotNone(completed.result)
        self.assertNotEqual(completed.result.status, f"thread:{main_thread_id}")

    def test_job_runner_emits_progress_updates_while_running(self) -> None:
        runner = LocalUiJobRunner()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_path = Path(tmpdir) / "worker-progress.json"

                def work(_cancel_signal, report_progress) -> UiActionResult:
                    report_progress("Loading model weights")
                    time.sleep(0.02)
                    report_progress("Priming first thinking token")
                    return UiActionResult(
                        action="thinking",
                        status="ok",
                        output_text="warm",
                        artifact_path=artifact_path,
                        model_id="google/gemma-4-E2B-it",
                        backend=CHAT_BACKEND,
                        device_label="cpu",
                        dtype_name="float32",
                    )

                submitted = runner.submit(
                    action="thinking",
                    work=work,
                    timeout_seconds=1.0,
                    message="warming",
                )
                completed, snapshots = self._wait_for_state(runner, job_id=submitted.job_id, state=JOB_STATE_COMPLETED)
        finally:
            runner.close()

        running_messages = [
            snapshot.message
            for snapshot in snapshots
            if snapshot.job_id == submitted.job_id and snapshot.state == JOB_STATE_RUNNING
        ]
        self.assertIsNotNone(completed.result)
        self.assertIn("Loading model weights", running_messages)
        self.assertIn("Priming first thinking token", running_messages)

    def test_job_runner_preserves_structured_prewarm_phase_updates(self) -> None:
        runner = LocalUiJobRunner()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_path = Path(tmpdir) / "worker-prewarm.json"

                def work(_cancel_signal, report_progress) -> UiActionResult:
                    report_progress(
                        WarmupProgress(
                            phase=WARMUP_PHASE_LOAD_MODEL,
                            message="Loading model weights",
                        )
                    )
                    return UiActionResult(
                        action=PREWARM_ACTION,
                        status="ok",
                        output_text="warm",
                        artifact_path=artifact_path,
                        model_id="google/gemma-4-E2B-it",
                        backend=PREWARM_BACKEND,
                        device_label="cpu",
                        dtype_name="float32",
                    )

                submitted = runner.submit(
                    action=PREWARM_ACTION,
                    work=work,
                    timeout_seconds=1.0,
                    message="warming",
                )
                completed, snapshots = self._wait_for_state(runner, job_id=submitted.job_id, state=JOB_STATE_COMPLETED)
        finally:
            runner.close()

        running_phases = [
            snapshot.progress_phase
            for snapshot in snapshots
            if snapshot.job_id == submitted.job_id and snapshot.state == JOB_STATE_RUNNING
        ]
        self.assertIsNotNone(completed.result)
        self.assertIn(WARMUP_PHASE_LOAD_MODEL, running_phases)

    def test_job_runner_can_cancel_queued_job(self) -> None:
        runner = LocalUiJobRunner()
        release_first = threading.Event()
        executed: list[str] = []

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_root = Path(tmpdir)

                def first_work(_cancel_signal, _report_progress) -> UiActionResult:
                    release_first.wait(timeout=1.0)
                    executed.append("first")
                    return UiActionResult(
                        action="chat",
                        status="ok",
                        output_text="first",
                        artifact_path=artifact_root / "first.json",
                        model_id="google/gemma-4-E2B-it",
                        backend=CHAT_BACKEND,
                        device_label="cpu",
                        dtype_name="float32",
                    )

                def second_work(_cancel_signal, _report_progress) -> UiActionResult:
                    executed.append("second")
                    return UiActionResult(
                        action="audio",
                        status="ok",
                        output_text="second",
                        artifact_path=artifact_root / "second.json",
                        model_id="google/gemma-4-E2B-it",
                        backend=AUDIO_BACKEND,
                        device_label="cpu",
                        dtype_name="float32",
                    )

                first = runner.submit(action="chat", work=first_work, timeout_seconds=1.0, message="first")
                self._wait_for_state(runner, job_id=first.job_id, state=JOB_STATE_RUNNING)
                second = runner.submit(action="audio", work=second_work, timeout_seconds=1.0, message="second")
                cancelled = runner.cancel(second.job_id)
                self.assertIsNotNone(cancelled)
                self.assertEqual(cancelled.state, JOB_STATE_CANCELLED)
                release_first.set()
                self._wait_for_state(runner, job_id=first.job_id, state=JOB_STATE_COMPLETED)
        finally:
            runner.close()

        self.assertIn("first", executed)
        self.assertNotIn("second", executed)

    def test_job_runner_marks_timeout_and_discards_late_result(self) -> None:
        runner = LocalUiJobRunner()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_path = Path(tmpdir) / "thinking.json"

                def slow_work(_cancel_signal, _report_progress) -> UiActionResult:
                    time.sleep(0.08)
                    return UiActionResult(
                        action="thinking",
                        status="ok",
                        output_text="late",
                        artifact_path=artifact_path,
                        model_id="google/gemma-4-E2B-it",
                        backend="thinking",
                        device_label="cpu",
                        dtype_name="float32",
                    )

                submitted = runner.submit(
                    action="thinking",
                    work=slow_work,
                    timeout_seconds=0.02,
                    message="slow",
                )
                self._wait_for_state(runner, job_id=submitted.job_id, state=JOB_STATE_RUNNING)
                time.sleep(0.04)
                runner.expire_timeouts()
                failed, _ = self._wait_for_state(runner, job_id=submitted.job_id, state=JOB_STATE_FAILED)
        finally:
            runner.close()

        self.assertTrue(failed.deadline_exceeded)
        self.assertIsNone(failed.result)
        self.assertIn("timed out", failed.message)


class LocalUiAppStartupPrewarmTests(unittest.TestCase):
    def _build_app(self, *, warm: bool = False) -> LocalUiApp:
        app = LocalUiApp.__new__(LocalUiApp)
        app.root = FakeRoot()
        app.controller = FakeAppController(warm=warm)
        app.job_runner = FakeAppJobRunner()
        app._job_bindings = {}
        app.hint_var = FakeVar("")
        app.status_var = FakeVar("")
        app.backend_var = FakeVar("Backend: idle")
        app.device_var = FakeVar("")
        app.artifact_var = FakeVar("")
        app._animated_status_job_id = None
        app._animated_status_base = ""
        app._animated_status_frame = 0
        app._prewarm_hint_job_id = None
        app._startup_prewarm_after_id = None
        app._startup_prewarm_pending = False
        app._startup_prewarm_reason = ""
        app._startup_prewarm_needs_idle_retry = False
        app._set_busy = lambda _message: None
        app._clear_busy = lambda: None
        app._clear_status_animation = lambda **_kwargs: None
        app._begin_status_animation = lambda **_kwargs: None
        app._begin_prewarm_hints = lambda **_kwargs: None
        app._clear_prewarm_hints = lambda **_kwargs: None
        app.refresh_diagnostics = lambda: None
        return app

    def test_schedule_startup_prewarm_uses_grace_window(self) -> None:
        app = self._build_app()

        scheduled = app._schedule_startup_prewarm(reason="launch")

        self.assertTrue(scheduled)
        self.assertTrue(app._startup_prewarm_pending)
        self.assertEqual(app._startup_prewarm_reason, "launch")
        self.assertEqual(app.root.after_calls[0].delay_ms, STARTUP_PREWARM_GRACE_MS)
        self.assertIn("launch settles", app.hint_var.get())

    def test_submit_job_prioritizes_real_action_over_pending_startup_prewarm(self) -> None:
        app = self._build_app()
        app._schedule_startup_prewarm(reason="launch")

        app._submit_job(
            action="chat",
            running_message="Running chat on the shared core...",
            output_widget=None,
            work=lambda _cancel_signal, _report_progress: UiActionResult(
                action="chat",
                status="ok",
                output_text="ok",
                artifact_path=Path("/tmp/chat.json"),
                model_id="google/gemma-4-E2B-it",
                backend=CHAT_BACKEND,
                device_label="cpu",
                dtype_name="float32",
            ),
        )

        self.assertEqual(app.root.cancelled_after_ids, ["after-1"])
        self.assertFalse(app._startup_prewarm_pending)
        self.assertTrue(app._startup_prewarm_needs_idle_retry)
        self.assertEqual(app.job_runner.submissions[0]["action"], "chat")
        self.assertIn("deferred", app.hint_var.get())

    def test_completed_real_action_reschedules_startup_prewarm_when_still_cold(self) -> None:
        app = self._build_app(warm=False)
        app._startup_prewarm_needs_idle_retry = True

        app._handle_job_snapshot(
            UiJobSnapshot(
                job_id=7,
                action="vision",
                state=JOB_STATE_COMPLETED,
                submitted_at=0.0,
                started_at=0.1,
                finished_at=0.2,
                timeout_seconds=30.0,
                cancel_requested=False,
                deadline_exceeded=False,
                progress_phase=None,
                message="vision completed.",
                result=None,
            )
        )

        self.assertTrue(app._startup_prewarm_pending)
        self.assertEqual(app._startup_prewarm_reason, "idle")
        self.assertEqual(app.root.after_calls[0].delay_ms, STARTUP_PREWARM_GRACE_MS)
        self.assertIn("retry once the lane stays idle", app.hint_var.get())

    def test_completed_real_action_clears_idle_retry_when_thinking_session_is_warm(self) -> None:
        app = self._build_app(warm=True)
        app._startup_prewarm_needs_idle_retry = True
        app.hint_var.set("Startup warmup deferred so your action can start first.")

        app._handle_job_snapshot(
            UiJobSnapshot(
                job_id=8,
                action="chat",
                state=JOB_STATE_COMPLETED,
                submitted_at=0.0,
                started_at=0.1,
                finished_at=0.2,
                timeout_seconds=30.0,
                cancel_requested=False,
                deadline_exceeded=False,
                progress_phase=None,
                message="chat completed.",
                result=None,
            )
        )

        self.assertFalse(app._startup_prewarm_needs_idle_retry)
        self.assertFalse(app._startup_prewarm_pending)
        self.assertEqual(app.root.after_calls, [])
        self.assertEqual(app.hint_var.get(), "")

    def test_running_prewarm_load_phase_cancel_is_accepted_but_not_immediate(self) -> None:
        app = self._build_app()

        app._handle_job_snapshot(
            UiJobSnapshot(
                job_id=9,
                action=PREWARM_ACTION,
                state=JOB_STATE_RUNNING,
                submitted_at=0.0,
                started_at=0.1,
                finished_at=None,
                timeout_seconds=30.0,
                cancel_requested=True,
                deadline_exceeded=False,
                progress_phase=WARMUP_PHASE_LOAD_MODEL,
                message="Loading model weights",
                result=None,
            )
        )

        self.assertIn("Cancel accepted", app.status_var.get())
        self.assertIn("lane release waits", app.status_var.get())
        self.assertIn("Immediate effect is not expected", app.hint_var.get())

    def test_running_prewarm_prime_phase_cancel_should_release_quickly(self) -> None:
        app = self._build_app()

        app._handle_job_snapshot(
            UiJobSnapshot(
                job_id=10,
                action=PREWARM_ACTION,
                state=JOB_STATE_RUNNING,
                submitted_at=0.0,
                started_at=0.1,
                finished_at=None,
                timeout_seconds=30.0,
                cancel_requested=True,
                deadline_exceeded=False,
                progress_phase=WARMUP_PHASE_PRIME_TOKEN,
                message="Priming first thinking token",
                result=None,
            )
        )

        self.assertIn("Cancel accepted", app.status_var.get())
        self.assertIn("take effect quickly", app.status_var.get())
        self.assertIn("Immediate effect is expected", app.hint_var.get())


class ThinkingDebugTests(unittest.TestCase):
    def test_debug_report_is_hidden_by_default_and_expands_when_enabled(self) -> None:
        result = {
            "turns": [
                {
                    "assistant": {
                        "thinking": "Plan privately.",
                    }
                }
            ],
            "iterations": [
                {
                    "assistant": {
                        "thinking": "Use the lookup tool.",
                    },
                    "tool_results": [
                        {
                            "tool_result": {"asset_id": "sensor-7", "found": True},
                        }
                    ],
                }
            ],
        }

        self.assertEqual(build_thinking_debug_report(result, enabled=False), "")

        expanded = build_thinking_debug_report(result, enabled=True)
        self.assertIn("Plan privately.", expanded)
        self.assertIn("Use the lookup tool.", expanded)
        self.assertIn("sensor-7", expanded)


if __name__ == "__main__":
    unittest.main()
