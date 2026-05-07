from __future__ import annotations

import base64
import json
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
    DEFAULT_CONTEXT_BUDGET_CHARS,
    DEFAULT_LIMIT,
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
    build_recall_diagnostic_guide_fields,
    build_recall_eval_compare_fields,
    build_recall_eval_source_card_text,
    build_recall_eval_source_chip_texts,
    build_recall_eval_source_why_text,
    build_recall_eval_winner_chip_texts,
    build_recall_eval_winner_why_text,
    build_entry_compare_fields,
    build_recall_compare_fields,
    build_recall_candidate_summary,
    build_recall_eval_miss_summary,
    build_recall_miss_suggested_manual_config,
    build_recall_pins_summary,
    build_recall_request_summary,
    build_evaluation_acceptance_text,
    build_evaluation_adoption_text,
    build_evaluation_comparison_text,
    build_evaluation_curation_rows,
    build_evaluation_curation_text,
    build_evaluation_repair_text,
    build_evaluation_snapshot_state,
    build_evaluation_test_text,
    build_learning_candidate_review_report,
    build_learning_candidate_review_rows,
    build_learning_candidate_review_state,
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


class FakeTextWidget:
    def __init__(self) -> None:
        self.state = "disabled"
        self.content = ""

    def configure(self, **kwargs: object) -> None:
        if "state" in kwargs:
            self.state = str(kwargs["state"])

    def delete(self, *_args: object) -> None:
        self.content = ""

    def insert(self, _index: str, text: str) -> None:
        self.content = text


class FakeTreeview:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[object, ...]] = {}
        self.focused: str | None = None
        self._selection: tuple[str, ...] = ()

    def get_children(self) -> tuple[str, ...]:
        return tuple(self.rows.keys())

    def delete(self, *children: str) -> None:
        for child in children:
            self.rows.pop(str(child), None)
            if self.focused == str(child):
                self.focused = None
        self._selection = tuple(item for item in self._selection if item in self.rows)

    def insert(self, _parent: str, _index: str, *, iid: str, values: tuple[object, ...]) -> None:
        self.rows[str(iid)] = values

    def selection(self) -> tuple[str, ...]:
        return self._selection

    def selection_set(self, iid: str) -> None:
        self._selection = (str(iid),)

    def focus(self, iid: str) -> None:
        self.focused = str(iid)

    def exists(self, iid: str) -> bool:
        return str(iid) in self.rows


class FakeNotebook:
    def __init__(self) -> None:
        self.selected_tabs: list[object] = []

    def select(self, tab: object) -> None:
        self.selected_tabs.append(tab)


class FakeRecallController:
    def __init__(
        self,
        *,
        root: Path,
        snapshot: dict[str, object],
        evaluation_snapshot: dict[str, object],
        result: dict[str, object],
        pinned_result: dict[str, object] | None = None,
    ) -> None:
        self.workspace_store = SimpleNamespace(root=root, workspace_id="local-default")
        self.snapshot = snapshot
        self.evaluation_snapshot = evaluation_snapshot
        self.result = result
        self.pinned_result = pinned_result or result
        self.collect_calls: list[bool] = []
        self.evaluate_calls: list[bool] = []
        self.bundle_calls: list[dict[str, object]] = []

    def collect_recall_requests(self, *, refresh_dataset: bool = False) -> dict[str, object]:
        self.collect_calls.append(refresh_dataset)
        return self.snapshot

    def evaluate_recall_dataset(self, *, refresh_dataset: bool = False) -> dict[str, object]:
        self.evaluate_calls.append(refresh_dataset)
        return self.evaluation_snapshot

    def build_recall_bundle(self, **kwargs: object) -> dict[str, object]:
        self.bundle_calls.append(dict(kwargs))
        pinned_event_ids = list(kwargs.get("pinned_event_ids") or [])
        if pinned_event_ids:
            return self.pinned_result
        return self.result


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

    def seed_recall_history(
        self,
        *,
        store: WorkspaceSessionStore,
        root: Path,
        model_id: str = "backend-a",
        prompt: str = "Review the memory index patch.",
        output_text: str = "Looks good with one regression note.",
        notes: list[str] | None = None,
    ) -> Path:
        artifact_path = root / "artifacts" / "text" / "chat.json"
        write_artifact(
            artifact_path,
            build_artifact_payload(
                artifact_kind="text",
                status="ok",
                runtime=build_runtime_record(
                    backend=CHAT_BACKEND,
                    model_id=model_id,
                    device_info="cpu",
                    elapsed_seconds=0.2,
                ),
                prompts=build_prompt_record(
                    system_prompt="You are concise.",
                    prompt=prompt,
                    resolved_user_prompt=prompt,
                ),
                extra={"output_text": output_text},
            ),
        )
        messages = store.chat_messages_for_next_turn(
            model_id=model_id,
            system_prompt="You are concise.",
        )
        store.record_chat_turn(
            model_id=model_id,
            status="ok",
            artifact_path=artifact_path,
            prompt=prompt,
            system_prompt="You are concise.",
            resolved_user_prompt=prompt,
            output_text=output_text,
            base_messages=messages,
            notes=notes or ["review accepted"],
        )
        return artifact_path

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

    def test_collect_recall_requests_returns_prepared_catalog(self) -> None:
        controller, store, _root = self.make_controller()
        self.seed_recall_history(store=store, root=controller.workspace_store.root)

        snapshot = controller.collect_recall_requests(refresh_dataset=True, max_requests=4)

        self.assertGreaterEqual(snapshot["request_count"], 1)
        self.assertTrue(Path(snapshot["dataset_path"]).exists())
        self.assertEqual(snapshot["requests"][0]["index"], 1)
        self.assertEqual(snapshot["request_entries"][0]["task_kind"], snapshot["requests"][0]["task_kind"])

    def test_evaluate_recall_dataset_writes_summary_and_syncs_catalog(self) -> None:
        controller, store, _root = self.make_controller()
        self.seed_recall_history(store=store, root=controller.workspace_store.root)

        snapshot = controller.evaluate_recall_dataset(refresh_dataset=True, max_requests=4)

        self.assertEqual(snapshot["request_count"], 1)
        self.assertEqual(snapshot["summary"]["source_hits"], 1)
        self.assertTrue(Path(snapshot["dataset_path"]).exists())
        self.assertTrue(Path(snapshot["evaluation_latest_path"]).exists())
        self.assertTrue(Path(snapshot["evaluation_run_path"]).exists())
        self.assertIn("Hit rate:", snapshot["report"])
        self.assertEqual(snapshot["requests"][0]["index"], 1)
        self.assertTrue(snapshot["requests"][0]["source_hit"])

    def test_build_evaluation_snapshot_exposes_m4_signal_metrics(self) -> None:
        controller, store, root = self.make_controller()
        artifact_path = root / "artifacts" / "text" / "validated.json"
        write_artifact(
            artifact_path,
            build_artifact_payload(
                artifact_kind="text",
                status="ok",
                runtime=build_runtime_record(
                    backend=CHAT_BACKEND,
                    model_id="backend-a",
                    device_info="cpu",
                ),
                prompts=build_prompt_record(
                    prompt="Run evaluation loop check.",
                    resolved_user_prompt="Run evaluation loop check.",
                ),
                extra={
                    "validation": {
                        "validation_mode": "unit",
                        "claim_scope": "local UI evaluation snapshot",
                        "pass_definition": "M4 snapshot sees workspace validation",
                        "quality_status": "pass",
                        "execution_status": "ok",
                        "quality_checks": [{"name": "snapshot", "pass": True, "detail": "ready"}],
                    },
                    "output_text": "Snapshot ready.",
                },
            ),
        )
        store.record_session_run(
            surface="thinking",
            model_id="backend-a",
            mode="tool",
            artifact_kind="thinking",
            artifact_path=artifact_path,
            status="ok",
            prompt="Run evaluation loop check.",
            system_prompt=None,
            resolved_user_prompt="Run evaluation loop check.",
            output_text="Snapshot ready.",
        )

        result = controller.build_evaluation_snapshot()
        snapshot = result["snapshot"]

        self.assertTrue(Path(result["snapshot_latest_path"]).exists())
        self.assertTrue(Path(result["snapshot_run_path"]).exists())
        self.assertTrue(Path(result["curation_preview_latest_path"]).exists())
        self.assertTrue(Path(result["curation_preview_run_path"]).exists())
        self.assertIn("Test pass: 1", result["report"])
        self.assertIn("Curation export preview: preview_only", result["report"])
        self.assertIn("pass=1", build_evaluation_snapshot_state(snapshot))
        self.assertIn("accepted=0", build_evaluation_acceptance_text(snapshot))
        self.assertIn("review=0/0", build_evaluation_acceptance_text(snapshot))
        self.assertIn("passed=1", build_evaluation_test_text(snapshot))
        self.assertIn("pending=0", build_evaluation_repair_text(snapshot))
        self.assertIn("comparisons=0", build_evaluation_comparison_text(snapshot))
        self.assertIn("review=1", build_evaluation_curation_text(snapshot))
        self.assertIn("matched=1", build_evaluation_adoption_text(result["curation_preview"]))
        self.assertIn("ready-for-policy=0", build_evaluation_adoption_text(result["curation_preview"]))
        curation_rows = build_evaluation_curation_rows(result["curation_preview"])
        self.assertEqual(curation_rows[0]["state"], "needs_review")

        resolved = controller.record_evaluation_review_resolution(
            source_event_id=curation_rows[0]["event_id"],
            resolved=True,
            review_id="ui-review-1",
            resolution_summary="UI review resolved.",
            curation_filters={"states": ["ready"]},
        )

        self.assertEqual(resolved["recorded_signal"]["signal_kind"], "review_resolved")
        self.assertEqual(resolved["snapshot"]["counts"]["review_resolved"], 1)
        self.assertIn("ready=1", build_evaluation_curation_text(resolved["snapshot"]))
        self.assertIn("ready-for-policy=1", build_evaluation_adoption_text(resolved["curation_preview"]))
        self.assertEqual(resolved["curation_preview"]["filters"]["states"], ["ready"])

    def test_learning_candidate_review_reads_latest_artifacts_only(self) -> None:
        controller, _store, root = self.make_controller()
        learning_root = root / "artifacts" / "evaluation" / "local-default" / "learning"
        learning_root.mkdir(parents=True)
        source_artifact_path = root / "artifacts" / "agent_lane" / "local-default" / "run.json"
        preview_path = learning_root / "preview-latest.json"
        human_selected_path = learning_root / "human-selected-latest.json"
        dry_run_path = learning_root / "jsonl-export-dry-run-latest.json"
        diff_path = learning_root / "candidate-diff-latest.json"

        preview = {
            "schema_name": "software-satellite-learning-dataset-preview",
            "schema_version": 1,
            "workspace_id": "local-default",
            "export_mode": "preview_only",
            "training_export_ready": False,
            "human_gate_required": True,
            "counts": {
                "source_candidate_count": 1,
                "review_queue_count": 1,
                "eligible_candidate_count": 1,
                "previewed_candidate_count": 1,
                "excluded_candidate_count": 0,
            },
            "paths": {"learning_preview_latest_path": str(preview_path)},
            "review_queue": [
                {
                    "event_id": "event-ready",
                    "label": "Ready candidate",
                    "queue_state": "ready",
                    "next_action": "confirm_export_policy",
                    "blocked_reason": None,
                    "lifecycle_summary": {"policy_state": "pending_confirmation"},
                    "comparison_evidence": {"roles": ["winner"]},
                    "backend_metadata": {"backend_id": "mock-careful-local"},
                    "source_paths": {"source_artifact_path": str(source_artifact_path)},
                }
            ],
        }
        human_selected = {
            "schema_name": "software-satellite-human-selected-candidate-list",
            "schema_version": 1,
            "workspace_id": "local-default",
            "export_mode": "preview_only",
            "training_export_ready": False,
            "human_gate_required": True,
            "source_learning_preview_path": str(preview_path),
            "counts": {
                "requested_candidate_count": 1,
                "matched_candidate_count": 1,
                "selected_candidate_count": 1,
            },
            "selected_candidates": [
                {
                    "event_id": "event-ready",
                    "queue_state": "ready",
                    "next_action": "confirm_export_policy",
                    "blocked_reason": None,
                    "lifecycle_summary": {"policy_state": "pending_confirmation"},
                    "comparison_evidence": {"roles": ["winner"]},
                    "backend_metadata": {"backend_id": "mock-careful-local"},
                    "source_paths": {
                        "source_learning_preview_path": str(preview_path),
                        "source_artifact_path": str(source_artifact_path),
                    },
                    "policy": {"export_policy_confirmed": False},
                }
            ],
        }
        dry_run = {
            "schema_name": "software-satellite-jsonl-training-export-dry-run",
            "schema_version": 1,
            "workspace_id": "local-default",
            "export_mode": "preview_only",
            "training_export_ready": "false",
            "human_gate_required": "true",
            "not_trainable": True,
            "source_learning_preview_path": str(preview_path),
            "counts": {
                "inspected_candidate_count": 1,
                "future_jsonl_candidate_if_separately_approved_count": 0,
                "would_write_jsonl_record_count": "0",
            },
            "export_policy": {
                "jsonl_file_written": "false",
                "jsonl_training_export_allowed": False,
                "training_job_allowed": False,
            },
            "candidates": [
                {
                    "event_id": "event-ready",
                    "dry_run_status": "export_policy_confirmation_required",
                    "queue_state": "ready",
                    "next_action": "confirm_export_policy",
                    "blocked_reasons": ["export_policy_not_confirmed"],
                    "lifecycle_summary": {"policy_state": "pending_confirmation"},
                    "comparison_evidence": {"roles": ["winner"]},
                    "backend_metadata": {"backend_id": "mock-careful-local"},
                    "source_paths": {"source_artifact_path": str(source_artifact_path)},
                    "policy": {"export_policy_confirmed": False},
                }
            ],
        }
        diff = {
            "schema_name": "software-satellite-learning-candidate-diff-summary",
            "schema_version": 1,
            "workspace_id": "local-default",
            "export_mode": "preview_only",
            "training_export_ready": False,
            "human_gate_required": True,
            "counts": {
                "base_candidate_count": 1,
                "target_candidate_count": 1,
                "changed_candidate_count": 1,
                "field_change_counts": {"policy_state": 1},
            },
            "paths": {
                "candidate_diff_latest_path": str(diff_path),
                "base_artifact_path": str(preview_path),
                "target_artifact_path": str(human_selected_path),
            },
            "changes": [
                {
                    "event_id": "event-ready",
                    "change_type": "changed",
                    "changed_fields": ["policy_state"],
                    "before": {
                        "event_id": "event-ready",
                        "queue_state": "ready",
                        "next_action": "confirm_export_policy",
                        "policy_state": "pending_confirmation",
                        "comparison_role": "winner",
                        "backend_id": "mock-careful-local",
                    },
                    "after": {
                        "event_id": "event-ready",
                        "queue_state": "ready",
                        "next_action": "confirm_export_policy",
                        "policy_state": "pending_confirmation",
                        "comparison_role": "winner",
                        "backend_id": "mock-careful-local",
                    },
                }
            ],
        }
        for path, payload in (
            (preview_path, preview),
            (human_selected_path, human_selected),
            (dry_run_path, dry_run),
            (diff_path, diff),
        ):
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        review = controller.build_learning_candidate_review()
        rows = build_learning_candidate_review_rows(review)
        report = build_learning_candidate_review_report(review)

        self.assertEqual(review["mode"], "read_only_latest_artifacts")
        self.assertEqual(review["counts"]["loaded_artifact_count"], 4)
        self.assertEqual(review["counts"]["training_ready_artifact_count"], 0)
        self.assertEqual(review["counts"]["jsonl_record_write_count"], 0)
        self.assertEqual(review["counts"]["candidate_row_count"], 4)
        self.assertIn("training-ready=0", build_learning_candidate_review_state(review))
        self.assertIn("jsonl-written=0", build_learning_candidate_review_state(review))
        self.assertIn("Learning candidate review: read-only", report)
        self.assertIn("JSONL records written: 0", report)
        self.assertEqual(rows[0]["queue_state"], "ready")
        self.assertEqual(rows[0]["policy_state"], "pending_confirmation")
        self.assertEqual(rows[0]["comparison_role"], "winner")
        self.assertEqual(rows[0]["backend_id"], "mock-careful-local")
        self.assertEqual(rows[0]["source_path"], str(source_artifact_path))

    def test_learning_candidate_review_reports_missing_latest_artifacts(self) -> None:
        controller, _store, _root = self.make_controller()

        review = controller.build_learning_candidate_review()

        self.assertEqual(review["counts"]["loaded_artifact_count"], 0)
        self.assertEqual(review["counts"]["missing_artifact_count"], 4)
        self.assertEqual(build_learning_candidate_review_rows(review), [])
        self.assertIn("missing=4", build_learning_candidate_review_state(review))
        self.assertIn("source=", build_learning_candidate_review_report(review))

    def test_learning_candidate_review_flags_unsafe_policy_strings(self) -> None:
        controller, _store, root = self.make_controller()
        learning_root = root / "artifacts" / "evaluation" / "local-default" / "learning"
        learning_root.mkdir(parents=True)
        dry_run_path = learning_root / "jsonl-export-dry-run-latest.json"
        dry_run_path.write_text(
            json.dumps(
                {
                    "schema_name": "software-satellite-jsonl-training-export-dry-run",
                    "schema_version": 1,
                    "workspace_id": "local-default",
                    "export_mode": "dry_run",
                    "training_export_ready": "unknown",
                    "human_gate_required": "false",
                    "counts": {"would_write_jsonl_record_count": "2"},
                    "export_policy": {"jsonl_file_written": "unknown"},
                    "candidates": [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        review = controller.build_learning_candidate_review()

        self.assertEqual(review["counts"]["loaded_artifact_count"], 1)
        self.assertEqual(review["counts"]["training_ready_artifact_count"], 1)
        self.assertEqual(review["counts"]["jsonl_record_write_count"], 2)
        self.assertIn("JSONL dry-run reports export_mode=dry_run", review["policy_warnings"])
        self.assertIn("JSONL dry-run reports human_gate_required=false", review["policy_warnings"])
        self.assertIn("JSONL dry-run reports jsonl_file_written=true", review["policy_warnings"])
        self.assertIn("JSONL dry-run reports training_export_ready=true", review["policy_warnings"])

    def test_build_recall_bundle_prefers_stored_dataset_bundle(self) -> None:
        controller, store, _root = self.make_controller()
        self.seed_recall_history(store=store, root=controller.workspace_store.root)
        snapshot = controller.collect_recall_requests(refresh_dataset=True, max_requests=4)

        bundle_path = Path(snapshot["request_entries"][0]["bundle_path"])
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        bundle["query_text"] = "SENTINEL UI RECALL BUNDLE"
        bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

        result = controller.build_recall_bundle(request_index=1, max_requests=4)

        self.assertIn("SENTINEL UI RECALL BUNDLE", result["report"])
        self.assertEqual(result["bundle_path"], str(bundle_path))

    def test_build_recall_bundle_manual_query_writes_ui_bundle(self) -> None:
        controller, store, _root = self.make_controller()
        self.seed_recall_history(store=store, root=controller.workspace_store.root)

        result = controller.build_recall_bundle(
            task_kind="review",
            query_text="Review the memory index patch.",
            request_basis="pass_definition",
            file_hint="scripts/memory_index.py",
            pinned_event_ids=["local-default:chat-main:pinned"],
        )

        self.assertIn("Task: review", result["report"])
        self.assertTrue(Path(result["bundle_path"]).exists())
        self.assertIn("/artifacts/recall_data/local-default/ui/", result["bundle_path"])
        self.assertEqual(
            result["request_payload"]["pinned_event_ids"],
            ["local-default:chat-main:pinned"],
        )
        self.assertEqual(result["request_payload"]["request_basis"], "pass_definition")
        self.assertEqual(result["pin_compare"]["before_bundle"]["pinned_event_ids"], [])
        self.assertEqual(
            result["pin_compare"]["after_bundle"]["pinned_event_ids"],
            ["local-default:chat-main:pinned"],
        )

    def test_build_recall_bundle_with_pins_uses_ui_bundle_for_dataset_request(self) -> None:
        controller, store, _root = self.make_controller()
        self.seed_recall_history(store=store, root=controller.workspace_store.root)
        controller.collect_recall_requests(refresh_dataset=True, max_requests=4)

        result = controller.build_recall_bundle(
            request_index=1,
            pinned_event_ids=["local-default:chat-main:pinned"],
            max_requests=4,
        )

        self.assertIn("/artifacts/recall_data/local-default/ui/", result["bundle_path"])
        self.assertEqual(
            result["request_payload"]["pinned_event_ids"],
            ["local-default:chat-main:pinned"],
        )
        self.assertEqual(result["pin_compare"]["before_bundle"]["pinned_event_ids"], [])
        self.assertEqual(
            result["pin_compare"]["after_bundle"]["pinned_event_ids"],
            ["local-default:chat-main:pinned"],
        )

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


class LocalUiAppRecallTests(unittest.TestCase):
    def _build_app(self) -> tuple[LocalUiApp, FakeRecallController]:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name)
        snapshot = {
            "dataset_path": str(root / "artifacts" / "recall_data" / "local-default" / "real_recall_dataset.json"),
            "generated_at_utc": "2026-04-13T12:00:00Z",
            "request_count": 1,
            "requests": [
                {
                    "index": 1,
                    "task_kind": "review",
                    "query_text": "Review the memory index patch.",
                    "source_hit": False,
                    "source_status": "ok",
                }
            ],
            "request_entries": [
                {
                    "index": 1,
                    "task_kind": "review",
                    "query_text": "Review the memory index patch.",
                    "source_hit": False,
                    "source_status": "ok",
                    "file_hints": ["scripts/memory_index.py"],
                    "request_basis": "pass_definition",
                    "limit": 8,
                    "context_budget_chars": 5000,
                    "source_event_id": "local-default:chat-main:entry-1",
                    "miss_reason": "dropped_by_context_budget",
                    "source_rank": 6,
                    "source_block_title": "Accepted outcomes",
                    "source_reasons": ["query-coverage", "accepted-signal"],
                    "top_selected": [
                        {
                            "event_id": "top-1",
                            "session_id": "chat-main",
                            "session_surface": "chat",
                            "recorded_at_utc": "2026-04-13T11:40:00Z",
                            "artifact_path": str(root / "artifacts" / "text" / "other.json"),
                            "block_title": "Open risks",
                            "prompt_excerpt": "Code generation notes outranked the source bundle.",
                            "reasons": ["fts-hit", "risk-signal"],
                        },
                        {
                            "event_id": "top-2",
                            "session_id": "chat-main",
                            "session_surface": "chat",
                            "recorded_at_utc": "2026-04-13T11:32:00Z",
                            "artifact_path": str(root / "artifacts" / "text" / "other-two.json"),
                            "block_title": "Accepted outcomes",
                            "prompt_excerpt": "Accepted outcome summary from the prior review.",
                            "reasons": ["accepted-signal", "recent"],
                        },
                        {
                            "event_id": "top-3",
                            "session_id": "chat-main",
                            "session_surface": "chat",
                            "recorded_at_utc": "2026-04-13T11:25:00Z",
                            "artifact_path": str(root / "artifacts" / "text" / "other-three.json"),
                            "block_title": "Relevant context",
                            "prompt_excerpt": "Relevant context bundle from the earlier recall run.",
                            "reasons": ["multi-query-hit", "recent"],
                        }
                    ],
                }
            ],
        }
        result = {
            "report": "Task: review\nQuery: Review the memory index patch.",
            "bundle_path": str(root / "artifacts" / "recall_data" / "local-default" / "ui" / "bundle.json"),
            "request_label": "dataset[1] review hit=yes",
            "bundle": {
                "budget": {
                    "used_chars": 880,
                    "context_budget_chars": 7500,
                },
                "selected_candidates": [
                    {
                        "event_id": "local-default:chat-main:entry-1",
                        "session_id": "chat-main",
                        "block_title": "Relevant context",
                        "status": "ok",
                        "session_surface": "chat",
                        "recorded_at_utc": "2026-04-13T11:55:00Z",
                        "artifact_path": str(root / "artifacts" / "text" / "chat.json"),
                        "prompt_excerpt": "Review the memory index patch.",
                        "reasons": ["pinned", "query-coverage"],
                    }
                ],
                "source_evaluation": {
                    "source_event_id": "local-default:chat-main:entry-1",
                    "source_selected": True,
                    "source_rank": 1,
                    "source_prompt_excerpt": "Review the memory index patch.",
                    "top_selected": [
                        {
                            "event_id": "local-default:chat-main:entry-1",
                            "prompt_excerpt": "Review the memory index patch.",
                        }
                    ],
                },
            },
        }
        evaluation_snapshot = {
            **snapshot,
            "summary": {
                "workspace_id": "local-default",
                "request_count": 1,
                "source_hits": 0,
                "source_misses": 1,
                "hit_rate": 0.0,
                "evaluated_at_utc": "2026-04-14T10:00:00+00:00",
                "variants": {
                    "baseline": {
                        "request_count": 1,
                        "source_hits": 0,
                        "source_misses": 1,
                        "hit_rate": 0.0,
                    }
                },
                "miss_reason_counts": {"dropped_by_context_budget": 1},
                "misses": [
                    {
                        "index": 1,
                        "task_kind": "review",
                        "query_text": "Review the memory index patch.",
                        "source_status": "ok",
                        "source_event_id": "local-default:chat-main:entry-1",
                        "source_rank": 6,
                        "source_prompt_excerpt": "Review the memory index patch.",
                        "miss_reason": "dropped_by_context_budget",
                        "request_variant": "baseline",
                        "top_selected": [
                            {
                                "event_id": "top-1",
                                "session_id": "chat-main",
                                "session_surface": "chat",
                                "recorded_at_utc": "2026-04-13T11:40:00Z",
                                "artifact_path": str(root / "artifacts" / "text" / "other.json"),
                                "block_title": "Open risks",
                                "prompt_excerpt": "Code generation notes outranked the source bundle.",
                                "reasons": ["fts-hit", "risk-signal"],
                            },
                            {
                                "event_id": "top-2",
                                "session_id": "chat-main",
                                "session_surface": "chat",
                                "recorded_at_utc": "2026-04-13T11:32:00Z",
                                "artifact_path": str(root / "artifacts" / "text" / "other-two.json"),
                                "block_title": "Accepted outcomes",
                                "prompt_excerpt": "Accepted outcome summary from the prior review.",
                                "reasons": ["accepted-signal", "recent"],
                            },
                            {
                                "event_id": "top-3",
                                "session_id": "chat-main",
                                "session_surface": "chat",
                                "recorded_at_utc": "2026-04-13T11:25:00Z",
                                "artifact_path": str(root / "artifacts" / "text" / "other-three.json"),
                                "block_title": "Relevant context",
                                "prompt_excerpt": "Relevant context bundle from the earlier recall run.",
                                "reasons": ["multi-query-hit", "recent"],
                            }
                        ],
                    }
                ],
            },
            "comparison": {
                "before_summary": {
                    "workspace_id": "local-default",
                    "request_count": 1,
                    "source_hits": 0,
                    "source_misses": 1,
                    "hit_rate": 0.0,
                    "variants": {
                        "baseline": {
                            "request_count": 1,
                            "source_hits": 0,
                            "source_misses": 1,
                            "hit_rate": 0.0,
                        }
                    },
                    "miss_reason_counts": {"ranked_out_by_limit": 1},
                },
                "after_summary": {
                    "workspace_id": "local-default",
                    "request_count": 1,
                    "source_hits": 0,
                    "source_misses": 1,
                    "hit_rate": 0.0,
                    "variants": {
                        "baseline": {
                            "request_count": 1,
                            "source_hits": 0,
                            "source_misses": 1,
                            "hit_rate": 0.0,
                        }
                    },
                    "miss_reason_counts": {"dropped_by_context_budget": 1},
                },
            },
            "misses": [
                {
                    "index": 1,
                    "task_kind": "review",
                    "query_text": "Review the memory index patch.",
                    "source_status": "ok",
                    "source_event_id": "local-default:chat-main:entry-1",
                    "source_rank": 6,
                    "source_prompt_excerpt": "Review the memory index patch.",
                    "miss_reason": "dropped_by_context_budget",
                    "request_variant": "baseline",
                    "top_selected": [
                        {
                            "event_id": "top-1",
                            "session_id": "chat-main",
                            "session_surface": "chat",
                            "recorded_at_utc": "2026-04-13T11:40:00Z",
                            "artifact_path": str(root / "artifacts" / "text" / "other.json"),
                            "block_title": "Open risks",
                            "prompt_excerpt": "Code generation notes outranked the source bundle.",
                            "reasons": ["fts-hit", "risk-signal"],
                        },
                        {
                            "event_id": "top-2",
                            "session_id": "chat-main",
                            "session_surface": "chat",
                            "recorded_at_utc": "2026-04-13T11:32:00Z",
                            "artifact_path": str(root / "artifacts" / "text" / "other-two.json"),
                            "block_title": "Accepted outcomes",
                            "prompt_excerpt": "Accepted outcome summary from the prior review.",
                            "reasons": ["accepted-signal", "recent"],
                        },
                        {
                            "event_id": "top-3",
                            "session_id": "chat-main",
                            "session_surface": "chat",
                            "recorded_at_utc": "2026-04-13T11:25:00Z",
                            "artifact_path": str(root / "artifacts" / "text" / "other-three.json"),
                            "block_title": "Relevant context",
                            "prompt_excerpt": "Relevant context bundle from the earlier recall run.",
                            "reasons": ["multi-query-hit", "recent"],
                        }
                    ],
                }
            ],
            "evaluation_run_path": str(
                root / "artifacts" / "recall_data" / "local-default" / "evaluation" / "runs" / "eval.json"
            ),
            "report": (
                "Workspace: local-default\nRequests: 1\nSource hits: 0\nSource misses: 1\nHit rate: 0.000"
            ),
        }
        pinned_result = {
            "report": "Task: review\nQuery: Review the memory index patch.\nPins: 1",
            "bundle_path": str(root / "artifacts" / "recall_data" / "local-default" / "ui" / "bundle-with-pin.json"),
            "request_label": "dataset[1] review hit=yes pins=1",
            "bundle": {
                "budget": {
                    "used_chars": 520,
                    "context_budget_chars": 1800,
                },
                "selected_candidates": [
                    {
                        "event_id": "local-default:chat-main:entry-1",
                        "session_id": "chat-main",
                        "block_title": "Relevant context",
                        "status": "ok",
                        "session_surface": "chat",
                        "recorded_at_utc": "2026-04-13T11:55:00Z",
                        "artifact_path": str(root / "artifacts" / "text" / "chat.json"),
                        "prompt_excerpt": "Review the memory index patch.",
                        "reasons": ["pinned", "query-coverage"],
                    },
                    {
                        "event_id": "top-1",
                        "session_id": "chat-main",
                        "block_title": "Open risks",
                        "status": "ok",
                        "session_surface": "chat",
                        "recorded_at_utc": "2026-04-13T11:40:00Z",
                        "artifact_path": str(root / "artifacts" / "text" / "other.json"),
                        "prompt_excerpt": "Code generation notes outranked the source bundle.",
                        "reasons": ["query-coverage"],
                    },
                ],
                "source_evaluation": {
                    "source_event_id": "local-default:chat-main:entry-1",
                    "source_selected": True,
                    "source_rank": 1,
                    "source_prompt_excerpt": "Review the memory index patch.",
                    "top_selected": [
                        {
                            "event_id": "local-default:chat-main:entry-1",
                            "prompt_excerpt": "Review the memory index patch.",
                        },
                        {
                            "event_id": "top-1",
                            "prompt_excerpt": "Code generation notes outranked the source bundle.",
                        },
                    ],
                },
            },
            "pin_compare": {
                "before_bundle": {
                    "budget": {
                        "used_chars": 420,
                        "context_budget_chars": 1800,
                    },
                    "selected_candidates": [
                        {
                            "event_id": "top-1",
                            "prompt_excerpt": "Code generation notes outranked the source bundle.",
                        },
                        {
                            "event_id": "top-2",
                            "prompt_excerpt": "Accepted outcomes from the prior review.",
                        },
                    ],
                    "source_evaluation": {
                        "source_event_id": "local-default:chat-main:entry-1",
                        "source_selected": False,
                        "source_rank": 6,
                        "miss_reason": "dropped_by_context_budget",
                        "source_prompt_excerpt": "Review the memory index patch.",
                        "top_selected": [
                            {
                                "event_id": "top-1",
                                "session_id": "chat-main",
                                "session_surface": "chat",
                                "recorded_at_utc": "2026-04-13T11:40:00Z",
                                "artifact_path": str(root / "artifacts" / "text" / "other.json"),
                                "prompt_excerpt": "Code generation notes outranked the source bundle.",
                            },
                            {
                                "event_id": "top-2",
                                "prompt_excerpt": "Accepted outcomes from the prior review.",
                            },
                        ],
                    },
                },
                "after_bundle": {
                    "budget": {
                        "used_chars": 520,
                        "context_budget_chars": 1800,
                    },
                    "selected_candidates": [
                        {
                            "event_id": "local-default:chat-main:entry-1",
                            "prompt_excerpt": "Review the memory index patch.",
                        },
                        {
                            "event_id": "top-1",
                            "prompt_excerpt": "Code generation notes outranked the source bundle.",
                        },
                    ],
                    "source_evaluation": {
                        "source_event_id": "local-default:chat-main:entry-1",
                        "source_selected": True,
                        "source_rank": 1,
                        "source_prompt_excerpt": "Review the memory index patch.",
                        "top_selected": [
                            {
                                "event_id": "local-default:chat-main:entry-1",
                                "session_id": "chat-main",
                                "session_surface": "chat",
                                "recorded_at_utc": "2026-04-13T11:55:00Z",
                                "artifact_path": str(root / "artifacts" / "text" / "chat.json"),
                                "prompt_excerpt": "Review the memory index patch.",
                            },
                            {
                                "event_id": "top-1",
                                "session_id": "chat-main",
                                "session_surface": "chat",
                                "recorded_at_utc": "2026-04-13T11:40:00Z",
                                "artifact_path": str(root / "artifacts" / "text" / "other.json"),
                                "prompt_excerpt": "Code generation notes outranked the source bundle.",
                            },
                        ],
                    },
                },
            },
        }
        controller = FakeRecallController(
            root=root,
            snapshot=snapshot,
            evaluation_snapshot=evaluation_snapshot,
            result=result,
            pinned_result=pinned_result,
        )

        app = LocalUiApp.__new__(LocalUiApp)
        app.controller = controller
        app.job_runner = FakeAppJobRunner()
        app.status_var = FakeVar("")
        app.backend_var = FakeVar("")
        app.device_var = FakeVar("")
        app.artifact_var = FakeVar("")
        app.hint_var = FakeVar("")
        app.recall_requests_state = FakeVar("")
        app.recall_eval_state = FakeVar("")
        app.recall_eval_misses_state = FakeVar("")
        app.recall_eval_miss_selection_state = FakeVar("")
        app.recall_eval_previous_var = FakeVar("")
        app.recall_eval_current_var = FakeVar("")
        app.recall_eval_change_var = FakeVar("")
        app.recall_eval_winner_compare_state = FakeVar("")
        app.recall_eval_source_var = FakeVar("")
        app.recall_eval_source_why_var = FakeVar("")
        app.recall_eval_source_action_hint_var = FakeVar("")
        app.recall_eval_winner_vars = [FakeVar("") for _ in range(3)]
        app.recall_eval_source_chip_vars = {
            "winner_only": FakeVar(""),
            "shared": FakeVar(""),
            "source_only": FakeVar(""),
            "pending": FakeVar(""),
        }
        app.recall_eval_winner_chip_vars = [
            {
                "winner_only": FakeVar(""),
                "shared": FakeVar(""),
                "source_only": FakeVar(""),
                "pending": FakeVar(""),
            }
            for _ in range(3)
        ]
        app.recall_selection_state = FakeVar("")
        app.recall_task_kind = FakeVar("proposal")
        app.recall_query = FakeVar("")
        app.recall_request_basis = FakeVar("")
        app.recall_file_hint = FakeVar("")
        app.recall_limit = FakeVar(DEFAULT_LIMIT)
        app.recall_context_budget_chars = FakeVar(DEFAULT_CONTEXT_BUDGET_CHARS)
        app.recall_candidates_state = FakeVar("")
        app.recall_candidate_selection_state = FakeVar("")
        app.recall_pins_state = FakeVar("")
        app.recall_compare_before_var = FakeVar("")
        app.recall_compare_after_var = FakeVar("")
        app.recall_compare_change_var = FakeVar("")
        app.recall_diagnostic_diagnosis_var = FakeVar("")
        app.recall_diagnostic_action_var = FakeVar("")
        app.recall_diagnostic_manual_var = FakeVar("")
        app.recall_requests_tree = FakeTreeview()
        app.recall_eval_misses_tree = FakeTreeview()
        app.recall_candidates_tree = FakeTreeview()
        app.recall_output = FakeTextWidget()
        app._recall_request_rows = {}
        app._recall_selected_request_index = None
        app._recall_eval_miss_rows = {}
        app._recall_selected_eval_miss_index = None
        app._recall_eval_source_row = None
        app._recall_eval_winner_rows = [None, None, None]
        app.recall_eval_source_chip_labels = {}
        app.recall_eval_winner_chip_labels = [{}, {}, {}]
        app._recall_candidate_rows = {}
        app._recall_selected_candidate_event_id = None
        app._recall_pinned_candidate_rows = {}
        app._set_output = lambda widget, text: setattr(widget, "content", text)
        app._cancel_pending_startup_prewarm = lambda: None
        app._resume_startup_prewarm_if_needed = lambda: None
        app._set_busy = lambda _message: None
        app._clear_busy = lambda: None
        app.apply_recall_miss_tweak_button = SimpleNamespace(configure=lambda **_kwargs: None)
        app.open_recall_miss_winner_button = SimpleNamespace(configure=lambda **_kwargs: None)
        app.recall_eval_source_button = SimpleNamespace(configure=lambda **_kwargs: None)
        app.recall_eval_source_copy_button = SimpleNamespace(configure=lambda **_kwargs: None)
        app.recall_eval_source_rerun_button = SimpleNamespace(configure=lambda **_kwargs: None)
        app.recall_eval_winner_buttons = [SimpleNamespace(configure=lambda **_kwargs: None) for _ in range(3)]
        app.forensics_surface = FakeVar("chat")
        app.forensics_artifact_path = FakeVar("")
        app._forensics_selected_session_id = None
        app._forensics_selected_entry_id = None
        app.refresh_forensics_calls = 0
        app.refresh_forensics = lambda: setattr(app, "refresh_forensics_calls", app.refresh_forensics_calls + 1)
        app.notebook = FakeNotebook()
        app.forensics_tab = "forensics-tab"
        return app, controller

    def test_refresh_recall_requests_populates_selection_and_copy(self) -> None:
        app, controller = self._build_app()

        app.refresh_recall_requests()
        self.assertEqual(controller.collect_calls, [False])
        self.assertEqual(app._recall_selected_request_index, 1)
        self.assertIn("prepared requests", app.recall_requests_state.get())
        self.assertIn("request 1", app.recall_selection_state.get())
        self.assertIn("miss_reason=dropped_by_context_budget", app.recall_selection_state.get())

        app.copy_selected_recall_request()

        self.assertEqual(app.recall_task_kind.get(), "review")
        self.assertEqual(app.recall_query.get(), "Review the memory index patch.")
        self.assertEqual(app.recall_request_basis.get(), "pass_definition")
        self.assertEqual(app.recall_file_hint.get(), "scripts/memory_index.py")
        self.assertEqual(app.recall_limit.get(), 8)
        self.assertEqual(app.recall_context_budget_chars.get(), 5000)

    def test_run_recall_evaluation_populates_compare_and_misses(self) -> None:
        app, controller = self._build_app()
        app.refresh_recall_requests()

        app.run_recall_evaluation()

        self.assertEqual(controller.evaluate_calls, [False])
        self.assertIn("hit rate 0.000", app.recall_eval_state.get())
        self.assertIn("requests=1", app.recall_eval_current_var.get())
        self.assertIn("miss reasons:", app.recall_eval_change_var.get())
        self.assertEqual(app._recall_selected_eval_miss_index, 1)
        self.assertIn("source miss(es)", app.recall_eval_misses_state.get())
        self.assertIn("Recall evaluation:", app.artifact_var.get())
        self.assertIn("Context budget squeeze", app.recall_diagnostic_diagnosis_var.get())
        self.assertIn("Raise context budget", app.recall_diagnostic_action_var.get())
        self.assertIn("budget=7500", app.recall_diagnostic_manual_var.get())
        self.assertIn("Showing 3 winner slot", app.recall_eval_winner_compare_state.get())
        self.assertIn("Review the memory index patch.", app.recall_eval_source_var.get())
        self.assertIn("reason=dropped by context budget", app.recall_eval_source_var.get())
        self.assertEqual(app.recall_eval_source_why_var.get(), "lost to winners on fts hit, risk signal")
        self.assertEqual(app.recall_eval_source_chip_vars["winner_only"].get(), "+ query coverage")
        self.assertEqual(app.recall_eval_source_chip_vars["shared"].get(), "= accepted signal")
        self.assertEqual(app.recall_eval_source_chip_vars["source_only"].get(), "- fts hit, risk signal")
        self.assertIn("Rerun suggested tweak: budget 5000 -> 7500.", app.recall_eval_source_action_hint_var.get())
        self.assertIn("Code generation notes", app.recall_eval_winner_vars[0].get())
        self.assertIn("beat source on fts hit, risk signal", app.recall_eval_winner_vars[0].get())
        self.assertEqual(app.recall_eval_winner_chip_vars[0]["winner_only"].get(), "+ fts hit, risk signal")
        self.assertEqual(app.recall_eval_winner_chip_vars[0]["shared"].get(), "")
        self.assertEqual(app.recall_eval_winner_chip_vars[0]["source_only"].get(), "- query coverage, accepted signal")
        self.assertIn("Accepted outcome summary", app.recall_eval_winner_vars[1].get())
        self.assertEqual(app.recall_eval_winner_chip_vars[1]["winner_only"].get(), "+ recent")
        self.assertEqual(app.recall_eval_winner_chip_vars[1]["shared"].get(), "= accepted signal")

    def test_copy_selected_recall_eval_miss_syncs_request_and_manual_form(self) -> None:
        app, _controller = self._build_app()
        app.refresh_recall_requests()
        app.run_recall_evaluation()

        app.copy_selected_recall_eval_miss()

        self.assertEqual(app._recall_selected_request_index, 1)
        self.assertEqual(app.recall_task_kind.get(), "review")
        self.assertEqual(app.recall_query.get(), "Review the memory index patch.")
        self.assertEqual(app.recall_request_basis.get(), "pass_definition")
        self.assertEqual(app.recall_file_hint.get(), "scripts/memory_index.py")
        self.assertEqual(app.recall_limit.get(), 8)
        self.assertEqual(app.recall_context_budget_chars.get(), 5000)
        self.assertIn("manual recall", app.status_var.get())

    def test_apply_selected_recall_eval_miss_tweak_updates_manual_controls(self) -> None:
        app, _controller = self._build_app()
        app.refresh_recall_requests()
        app.run_recall_evaluation()

        app.apply_selected_recall_eval_miss_tweak()

        self.assertEqual(app.recall_task_kind.get(), "review")
        self.assertEqual(app.recall_query.get(), "Review the memory index patch.")
        self.assertEqual(app.recall_request_basis.get(), "pass_definition")
        self.assertEqual(app.recall_file_hint.get(), "scripts/memory_index.py")
        self.assertEqual(app.recall_limit.get(), 8)
        self.assertEqual(app.recall_context_budget_chars.get(), 7500)
        self.assertIn("suggested tweak", app.status_var.get())

    def test_open_selected_recall_eval_miss_winner_routes_to_forensics(self) -> None:
        app, _controller = self._build_app()
        app.refresh_recall_requests()
        app.run_recall_evaluation()

        app.open_selected_recall_eval_miss_winner()

        self.assertEqual(app.forensics_surface.get(), "chat")
        self.assertEqual(app._forensics_selected_session_id, "chat-main")
        self.assertEqual(app._forensics_selected_entry_id, "top-1")
        self.assertTrue(app.forensics_artifact_path.get().endswith("/artifacts/text/other.json"))
        self.assertEqual(app.refresh_forensics_calls, 1)
        self.assertEqual(app.notebook.selected_tabs, ["forensics-tab"])
        self.assertIn("winner #1", app.status_var.get().lower())

    def test_open_second_recall_eval_winner_routes_to_forensics(self) -> None:
        app, _controller = self._build_app()
        app.refresh_recall_requests()
        app.run_recall_evaluation()

        app.open_recall_eval_winner_at(1)

        self.assertEqual(app.forensics_surface.get(), "chat")
        self.assertEqual(app._forensics_selected_session_id, "chat-main")
        self.assertEqual(app._forensics_selected_entry_id, "top-2")
        self.assertTrue(app.forensics_artifact_path.get().endswith("/artifacts/text/other-two.json"))
        self.assertEqual(app.refresh_forensics_calls, 1)
        self.assertIn("winner #2", app.status_var.get().lower())

    def test_open_selected_recall_eval_source_routes_to_forensics(self) -> None:
        app, _controller = self._build_app()
        app.refresh_recall_requests()
        app.run_recall_evaluation()

        app.open_selected_recall_eval_source()

        self.assertEqual(app.forensics_surface.get(), "chat")
        self.assertEqual(app._forensics_selected_session_id, "chat-main")
        self.assertEqual(app._forensics_selected_entry_id, "entry-1")
        self.assertEqual(app.forensics_artifact_path.get(), "")
        self.assertEqual(app.refresh_forensics_calls, 1)
        self.assertEqual(app.notebook.selected_tabs, ["forensics-tab"])
        self.assertIn("source candidate", app.status_var.get().lower())

    def test_copy_selected_recall_eval_source_to_manual_updates_manual_form(self) -> None:
        app, _controller = self._build_app()
        app.refresh_recall_requests()
        app.run_recall_evaluation()

        app.copy_selected_recall_eval_source_to_manual()

        self.assertEqual(app._recall_selected_request_index, 1)
        self.assertEqual(app.recall_task_kind.get(), "review")
        self.assertEqual(app.recall_query.get(), "Review the memory index patch.")
        self.assertEqual(app.recall_request_basis.get(), "pass_definition")
        self.assertEqual(app.recall_file_hint.get(), "scripts/memory_index.py")
        self.assertEqual(app.recall_limit.get(), 8)
        self.assertEqual(app.recall_context_budget_chars.get(), 5000)
        self.assertIn("source candidate", app.recall_selection_state.get().lower())
        self.assertIn("source candidate", app.status_var.get().lower())
        self.assertIn("run manual recall now", app.hint_var.get().lower())

    def test_rerun_selected_recall_eval_source_applies_suggestion_and_runs_manual_recall(self) -> None:
        app, controller = self._build_app()
        app.refresh_recall_requests()
        app.run_recall_evaluation()

        app.rerun_selected_recall_eval_source()

        self.assertEqual(len(controller.bundle_calls), 1)
        self.assertEqual(controller.bundle_calls[0]["task_kind"], "review")
        self.assertEqual(controller.bundle_calls[0]["query_text"], "Review the memory index patch.")
        self.assertEqual(controller.bundle_calls[0]["request_basis"], "pass_definition")
        self.assertEqual(controller.bundle_calls[0]["file_hint"], "scripts/memory_index.py")
        self.assertEqual(controller.bundle_calls[0]["limit"], 8)
        self.assertEqual(controller.bundle_calls[0]["context_budget_chars"], 7500)
        self.assertEqual(app.recall_context_budget_chars.get(), 7500)
        self.assertIn("Suggested rerun for evaluation miss 1", app.recall_selection_state.get())
        self.assertEqual(app.status_var.get(), "Suggested rerun for evaluation miss 1 is ready.")
        self.assertEqual(
            app.hint_var.get(),
            "Before: source was rank 6 and dropped by context budget. After rerun: source made the bundle at rank 1.",
        )
        self.assertEqual(
            app.recall_eval_source_action_hint_var.get(),
            "Before: source was rank 6 and dropped by context budget. After rerun: source made the bundle at rank 1.",
        )
        self.assertIn("Recall bundle:", app.artifact_var.get())

    def test_source_action_hint_explains_missing_source_navigation(self) -> None:
        app, _controller = self._build_app()
        app.refresh_recall_requests()
        app.run_recall_evaluation()

        miss_row = dict(app._recall_eval_miss_rows["1"])
        request_row = dict(app._recall_request_rows["1"])
        miss_row.pop("source_event_id", None)
        miss_row.pop("source_entry_id", None)
        miss_row.pop("source_session_id", None)
        miss_row.pop("source_artifact_path", None)
        request_row.pop("source_event_id", None)
        request_row.pop("source_entry_id", None)
        request_row.pop("source_session_id", None)
        request_row.pop("source_artifact_path", None)
        app._recall_request_rows["1"] = request_row

        app._apply_recall_eval_winner_compare(miss_row)

        self.assertIsNone(app._recall_eval_source_row)
        self.assertIn("Open needs a fresh source snapshot.", app.recall_eval_source_action_hint_var.get())
        self.assertIn("Rerun suggested tweak: budget 5000 -> 7500.", app.recall_eval_source_action_hint_var.get())

    def test_run_selected_and_manual_recall_update_output(self) -> None:
        app, controller = self._build_app()
        app.refresh_recall_requests()

        app.run_selected_recall_request()
        self.assertEqual(controller.bundle_calls[0]["request_index"], 1)
        self.assertIn("Task: review", app.recall_output.content)
        self.assertIn("Recall bundle:", app.artifact_var.get())
        self.assertEqual(app._recall_selected_candidate_event_id, "local-default:chat-main:entry-1")
        self.assertIn("selected candidates", app.recall_candidates_state.get())
        self.assertIn("unpin", app.recall_compare_before_var.get().lower())
        self.assertIn("differences appear here", app.recall_compare_change_var.get())

        app.recall_task_kind.set("review")
        app.recall_query.set("Tighten ranking weights for source_hit=false requests.")
        app.recall_request_basis.set("pass_definition")
        app.recall_file_hint.set("scripts/recall_context.py")
        app.run_manual_recall()

        self.assertEqual(controller.bundle_calls[1]["task_kind"], "review")
        self.assertEqual(
            controller.bundle_calls[1]["query_text"],
            "Tighten ranking weights for source_hit=false requests.",
        )
        self.assertEqual(controller.bundle_calls[1]["request_basis"], "pass_definition")
        self.assertEqual(controller.bundle_calls[1]["file_hint"], "scripts/recall_context.py")
        self.assertEqual(controller.bundle_calls[1]["limit"], DEFAULT_LIMIT)
        self.assertEqual(controller.bundle_calls[1]["context_budget_chars"], DEFAULT_CONTEXT_BUDGET_CHARS)

    def test_pin_selected_recall_candidate_applies_to_next_runs_and_can_clear(self) -> None:
        app, controller = self._build_app()
        app.refresh_recall_requests()
        app.run_selected_recall_request()

        app.pin_selected_recall_candidate()
        self.assertIn("Pinned for the next recall run (1)", app.recall_pins_state.get())
        self.assertTrue(app._recall_candidate_rows["local-default:chat-main:entry-1"]["pinned"])
        self.assertIn("pinned=yes", app.recall_candidate_selection_state.get())

        app.run_manual_recall()
        self.assertEqual(
            controller.bundle_calls[1]["pinned_event_ids"],
            ["local-default:chat-main:entry-1"],
        )
        self.assertIn("source rank=6", app.recall_compare_before_var.get())
        self.assertIn("source rank=1", app.recall_compare_after_var.get())
        self.assertIn("source rank: 6 -> 1", app.recall_compare_change_var.get())
        self.assertIn("miss reason: dropped_by_context_budget -> selected", app.recall_compare_change_var.get())
        self.assertIn("selected candidates:", app.recall_compare_change_var.get())
        self.assertIn("budget:", app.recall_compare_change_var.get())
        self.assertIn("source vs winners:", app.recall_compare_change_var.get())

        app.clear_recall_pins()
        self.assertIn("Pinned candidates: none", app.recall_pins_state.get())
        self.assertFalse(app._recall_candidate_rows["local-default:chat-main:entry-1"]["pinned"])

    def test_open_selected_recall_candidate_routes_to_forensics(self) -> None:
        app, _controller = self._build_app()
        app.refresh_recall_requests()
        app.run_selected_recall_request()

        app.open_selected_recall_candidate()

        self.assertEqual(app.forensics_surface.get(), "chat")
        self.assertEqual(app._forensics_selected_session_id, "chat-main")
        self.assertEqual(app._forensics_selected_entry_id, "entry-1")
        self.assertTrue(app.forensics_artifact_path.get().endswith("/artifacts/text/chat.json"))
        self.assertEqual(app.refresh_forensics_calls, 1)
        self.assertEqual(app.notebook.selected_tabs, ["forensics-tab"])
        self.assertIn("Opened recall candidate", app.status_var.get())

    def test_recall_summary_helpers_include_miss_and_pin_state(self) -> None:
        request_summary = build_recall_request_summary(
            {
                "index": 2,
                "task_kind": "proposal",
                "query_text": "Opaque scratchpad seal uncompromised verdict retention.",
                "file_hints": ["scripts/recall_context.py"],
                "source_hit": False,
                "source_status": "ok",
                "request_basis": "pass_definition",
                "request_variant": "adversarial-pass-definition",
                "miss_reason": "dropped_by_context_budget",
                "source_rank": 11,
                "source_block_title": "Accepted outcomes",
            }
        )
        self.assertIn("basis=pass_definition", request_summary)
        self.assertIn("miss_reason=dropped_by_context_budget", request_summary)

        pins_summary = build_recall_pins_summary(
            [
                {"event_id": "event-1", "prompt_excerpt": "Review the memory index patch."},
                {"event_id": "event-2", "prompt_excerpt": "Pin the accepted candidate for the next run."},
            ]
        )
        self.assertIn("Pinned for the next recall run (2)", pins_summary)

        candidate_summary = build_recall_candidate_summary(
            {
                "event_id": "leader",
                "block_title": "Accepted outcomes",
                "session_surface": "evaluation",
                "status": "ok",
                "prompt_excerpt": "3 results share pass definition.",
                "artifact_path": "/tmp/source.json",
                "group_member_count": 3,
                "group_member_labels": ["leader capability", "source capability", "third capability"],
            },
            root=Path("/tmp"),
        )
        self.assertIn("group=3", candidate_summary)
        self.assertIn("group_members=3: leader capability; source capability; third capability", candidate_summary)

    def test_recall_compare_fields_summarize_pin_diff(self) -> None:
        fields = dict(
            build_recall_compare_fields(
                {
                    "before_bundle": {
                        "budget": {"used_chars": 420, "context_budget_chars": 1800},
                        "selected_candidates": [
                            {"event_id": "winner-1", "prompt_excerpt": "Code generation notes outranked the source bundle."},
                        ],
                        "source_evaluation": {
                            "source_event_id": "source-1",
                            "source_selected": False,
                            "source_rank": 6,
                            "miss_reason": "dropped_by_context_budget",
                            "source_prompt_excerpt": "Review the memory index patch.",
                            "top_selected": [
                                {"event_id": "winner-1", "prompt_excerpt": "Code generation notes outranked the source bundle."},
                            ],
                        },
                    },
                    "after_bundle": {
                        "budget": {"used_chars": 520, "context_budget_chars": 1800},
                        "selected_candidates": [
                            {"event_id": "source-1", "prompt_excerpt": "Review the memory index patch."},
                            {"event_id": "winner-1", "prompt_excerpt": "Code generation notes outranked the source bundle."},
                        ],
                        "source_evaluation": {
                            "source_event_id": "source-1",
                            "source_selected": True,
                            "source_rank": 1,
                            "source_prompt_excerpt": "Review the memory index patch.",
                            "source_selected_via_group": True,
                            "source_grouped_by": "pass_definition",
                            "source_group_member_count": 2,
                            "source_group_member_labels": ["Review the memory index patch.", "Code generation notes"],
                            "top_selected": [
                                {"event_id": "source-1", "prompt_excerpt": "Review the memory index patch."},
                                {"event_id": "winner-1", "prompt_excerpt": "Code generation notes outranked the source bundle."},
                            ],
                        },
                    },
                }
            )
        )

        self.assertIn("source rank=6", fields["Before"])
        self.assertIn("source rank=1", fields["After"])
        self.assertIn("source rank: 6 -> 1", fields["Change"])
        self.assertIn("miss reason: dropped_by_context_budget -> selected", fields["Change"])
        self.assertIn("selected candidates:", fields["Change"])
        self.assertIn("budget:", fields["Change"])
        self.assertIn("source vs winners:", fields["Change"])
        self.assertIn("joined pass_definition", fields["After"])

    def test_recall_eval_helpers_summarize_delta_and_selected_miss(self) -> None:
        fields = dict(
            build_recall_eval_compare_fields(
                {
                    "before_summary": {
                        "request_count": 2,
                        "source_hits": 0,
                        "source_misses": 2,
                        "hit_rate": 0.0,
                        "variants": {
                            "baseline": {
                                "request_count": 2,
                                "source_hits": 0,
                                "source_misses": 2,
                                "hit_rate": 0.0,
                            }
                        },
                        "miss_reason_counts": {"ranked_out_by_limit": 2},
                    },
                    "after_summary": {
                        "request_count": 2,
                        "source_hits": 1,
                        "source_misses": 1,
                        "hit_rate": 0.5,
                        "variants": {
                            "baseline": {
                                "request_count": 2,
                                "source_hits": 1,
                                "source_misses": 1,
                                "hit_rate": 0.5,
                            }
                        },
                        "miss_reason_counts": {"dropped_by_context_budget": 1},
                    },
                }
            )
        )

        self.assertIn("source hits=0", fields["Previous"])
        self.assertIn("source hits=1", fields["Current"])
        self.assertIn("source hits: 0 -> 1 (+1)", fields["Change"])
        self.assertIn("miss reasons:", fields["Change"])

        miss_summary = build_recall_eval_miss_summary(
            {
                "index": 3,
                "task_kind": "review",
                "query_text": "Review the memory index patch.",
                "source_rank": 6,
                "source_prompt_excerpt": "Review the memory index patch.",
                "miss_reason": "dropped_by_context_budget",
                "request_variant": "adversarial-pass-definition",
            }
        )
        self.assertIn("request 3", miss_summary)
        self.assertIn("variant=adversarial-pass-definition", miss_summary)
        self.assertIn("source=Review the memory index patch.", miss_summary)

        suggestion = build_recall_miss_suggested_manual_config(
            {
                "index": 3,
                "task_kind": "review",
                "query_text": "Review the memory index patch.",
                "source_rank": 6,
                "source_prompt_excerpt": "Review the memory index patch.",
                "source_reasons": ["query-coverage", "accepted-signal"],
                "miss_reason": "dropped_by_context_budget",
                "top_selected": [
                    {"event_id": "winner-1", "prompt_excerpt": "Code generation notes outranked the source bundle."},
                ],
            },
            request_row={
                "task_kind": "review",
                "query_text": "Review the memory index patch.",
                "request_basis": "pass_definition",
                "file_hints": ["scripts/memory_index.py"],
                "limit": 8,
                "context_budget_chars": 5000,
            },
        )
        self.assertEqual(suggestion["context_budget_chars"], 7500)
        self.assertTrue(suggestion["apply_ready"])
        self.assertIn("Context budget squeeze", suggestion["diagnosis_text"])
        self.assertIn("Raise context budget", suggestion["actions_text"])

        guide_fields = dict(
            build_recall_diagnostic_guide_fields(
                {
                    "index": 3,
                    "task_kind": "review",
                    "query_text": "Review the memory index patch.",
                    "source_rank": 6,
                    "source_reasons": ["query-coverage"],
                    "miss_reason": "dropped_by_context_budget",
                    "top_selected": [
                        {"event_id": "winner-1", "prompt_excerpt": "Code generation notes outranked the source bundle."},
                    ],
                },
                request_row={
                    "task_kind": "review",
                    "query_text": "Review the memory index patch.",
                    "request_basis": "pass_definition",
                    "file_hints": ["scripts/memory_index.py"],
                    "limit": 8,
                    "context_budget_chars": 5000,
                },
            )
        )
        self.assertIn("Context budget squeeze", guide_fields["Diagnosis"])
        self.assertIn("Raise context budget", guide_fields["Next"])
        self.assertIn("budget=7500", guide_fields["Suggested Manual"])

        chip_texts = build_recall_eval_winner_chip_texts(
            {"source_reasons": ["query-coverage", "accepted-signal"]},
            {"reasons": ["fts-hit", "accepted-signal", "risk-signal"]},
        )
        self.assertEqual(chip_texts["winner_only"], "+ fts hit, risk signal")
        self.assertEqual(chip_texts["shared"], "= accepted signal")
        self.assertEqual(chip_texts["source_only"], "- query coverage")
        self.assertEqual(chip_texts["pending"], "")

        self.assertEqual(
            build_recall_eval_winner_why_text(
                {"source_reasons": ["query-coverage", "accepted-signal"]},
                {"reasons": []},
            ),
            "Winner reasons pending in this snapshot.",
        )
        pending_chip_texts = build_recall_eval_winner_chip_texts(
            {"source_reasons": ["query-coverage"]},
            {"event_id": "winner-1"},
        )
        self.assertEqual(pending_chip_texts["pending"], "Reasons pending")
        self.assertEqual(pending_chip_texts["winner_only"], "")

        source_chip_texts = build_recall_eval_source_chip_texts(
            {
                "source_reasons": ["query-coverage", "accepted-signal"],
                "top_selected": [
                    {"reasons": ["fts-hit", "accepted-signal", "risk-signal"]},
                    {"reasons": ["accepted-signal", "recent"]},
                ],
            }
        )
        self.assertEqual(source_chip_texts["winner_only"], "+ query coverage")
        self.assertEqual(source_chip_texts["shared"], "= accepted signal")
        self.assertEqual(source_chip_texts["source_only"], "- fts hit, risk signal")
        self.assertEqual(source_chip_texts["pending"], "")
        self.assertEqual(
            build_recall_eval_source_why_text(
                {
                    "source_reasons": ["query-coverage", "accepted-signal"],
                    "top_selected": [
                        {"reasons": ["fts-hit", "accepted-signal", "risk-signal"]},
                    ],
                }
            ),
            "lost to winners on fts hit, risk signal",
        )
        self.assertIn(
            "reason=dropped by context budget",
            build_recall_eval_source_card_text(
                {
                    "query_text": "Review the memory index patch.",
                    "source_prompt_excerpt": "Review the memory index patch.",
                    "source_block_title": "Accepted outcomes",
                    "source_rank": 6,
                    "source_reasons": ["query-coverage", "accepted-signal"],
                    "miss_reason": "dropped_by_context_budget",
                    "top_selected": [
                        {"reasons": ["fts-hit", "accepted-signal", "risk-signal"]},
                    ],
                }
            ),
        )
        pending_source_chip_texts = build_recall_eval_source_chip_texts(
            {"top_selected": [{"reasons": ["fts-hit"]}]}
        )
        self.assertEqual(pending_source_chip_texts["pending"], "Reasons pending")
        self.assertEqual(pending_source_chip_texts["winner_only"], "")


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
