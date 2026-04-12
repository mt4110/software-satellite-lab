from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gemma_core import (  # noqa: E402
    CancellationSignal,
    GenerationResult,
    GenerationCancelled,
    RuntimeSession,
    SessionKey,
    SessionManager,
    generate_text_from_messages,
)


class FakeIds(list):
    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self))


class FakeBatch(dict):
    def to(self, device_name: str) -> "FakeBatch":
        self["moved_to"] = device_name
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.template_calls: list[tuple[list[dict[str, object]], dict[str, object]]] = []

    def apply_chat_template(self, messages: list[dict[str, object]], **kwargs: object) -> str:
        self.template_calls.append((messages, kwargs))
        return "rendered prompt"

    def __call__(self, **kwargs: object) -> FakeBatch:
        self.calls.append(kwargs)
        return FakeBatch({"input_ids": FakeIds([10, 20, 30])})

    def decode(self, tokens: object, skip_special_tokens: bool = False) -> str:
        return "assistant: hello world"


class FakeModel:
    device = "cpu"

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> list[FakeIds]:
        self.generate_calls.append(kwargs)
        return [FakeIds([10, 20, 30, 40, 50])]


class FakeInferenceMode:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class FakeTorch:
    def inference_mode(self) -> FakeInferenceMode:
        return FakeInferenceMode()


def fake_device_info() -> dict[str, object]:
    return {
        "name": "cpu",
        "label": "cpu",
        "dtype": "float32",
        "dtype_name": "float32",
    }


class SessionManagerTests(unittest.TestCase):
    def test_reuses_text_session_for_same_model_and_device(self) -> None:
        manager = SessionManager()
        load_calls: list[str] = []

        with (
            patch("gemma_core.import_text_runtime", return_value=(object(), object(), object())),
            patch("gemma_core.select_device", return_value=fake_device_info()),
            patch("gemma_core.assert_model_fetch_is_possible"),
            patch("gemma_core.load_model_and_processor") as mocked_load,
        ):
            mocked_load.side_effect = lambda **_: (load_calls.append("loaded"), object(), object())[1:]

            first = manager.get_session("text", "google/gemma-4-E2B-it")
            second = manager.get_session("text", "google/gemma-4-E2B-it")

        self.assertIs(first, second)
        self.assertEqual(load_calls, ["loaded"])
        self.assertEqual(
            manager.cached_keys(),
            [SessionKey(session_kind="text", model_id="google/gemma-4-E2B-it", device_class="cpu")],
        )

    def test_multimodal_aliases_share_cache_and_clear_forces_reload(self) -> None:
        manager = SessionManager()
        load_count = 0

        def fake_load(**_: object) -> tuple[object, object]:
            nonlocal load_count
            load_count += 1
            return object(), object()

        with (
            patch("gemma_core.import_multimodal_runtime", return_value=(object(), object(), object())),
            patch("gemma_core.select_device", return_value=fake_device_info()),
            patch("gemma_core.assert_model_fetch_is_possible"),
            patch("gemma_core.load_model_and_processor", side_effect=fake_load),
        ):
            vision_session = manager.get_session("vision", "google/gemma-4-E2B-it")
            audio_session = manager.get_session("audio", "google/gemma-4-E2B-it")
            cleared = manager.clear(session_kind="audio", model_id="google/gemma-4-E2B-it")
            reloaded_session = manager.get_session("vision", "google/gemma-4-E2B-it")

        self.assertIs(vision_session, audio_session)
        self.assertIsNot(vision_session, reloaded_session)
        self.assertEqual(cleared, 1)
        self.assertEqual(load_count, 2)


class SharedGenerationTests(unittest.TestCase):
    def test_generate_text_from_messages_uses_shared_text_path(self) -> None:
        processor = FakeProcessor()
        model = FakeModel()
        session = RuntimeSession(
            key=SessionKey(session_kind="text", model_id="google/gemma-4-E2B-it", device_class="cpu"),
            processor=processor,
            model=model,
            torch=FakeTorch(),
            device_info=fake_device_info(),
        )

        result = generate_text_from_messages(
            session=session,
            messages=[
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Say hello."},
            ],
            generation_settings={"max_new_tokens": 8, "do_sample": False},
        )

        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(result.output_text, "hello world")
        self.assertEqual(result.input_token_count, 3)
        self.assertEqual(processor.calls[0]["text"], "rendered prompt")
        self.assertEqual(model.generate_calls[0]["max_new_tokens"], 8)

    def test_generate_text_from_messages_passes_stopping_criteria_when_cancellable(self) -> None:
        processor = FakeProcessor()
        model = FakeModel()
        session = RuntimeSession(
            key=SessionKey(session_kind="text", model_id="google/gemma-4-E2B-it", device_class="cpu"),
            processor=processor,
            model=model,
            torch=FakeTorch(),
            device_info=fake_device_info(),
        )

        generate_text_from_messages(
            session=session,
            messages=[
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Say hello."},
            ],
            generation_settings={"max_new_tokens": 8, "do_sample": False},
            cancellation_signal=CancellationSignal(),
        )

        self.assertIn("stopping_criteria", model.generate_calls[0])

    def test_generate_text_from_messages_aborts_before_generate_when_already_cancelled(self) -> None:
        processor = FakeProcessor()
        model = FakeModel()
        session = RuntimeSession(
            key=SessionKey(session_kind="text", model_id="google/gemma-4-E2B-it", device_class="cpu"),
            processor=processor,
            model=model,
            torch=FakeTorch(),
            device_info=fake_device_info(),
        )
        cancellation_signal = CancellationSignal()
        cancellation_signal.request_cancel()

        with self.assertRaises(GenerationCancelled):
            generate_text_from_messages(
                session=session,
                messages=[
                    {"role": "system", "content": "You are concise."},
                    {"role": "user", "content": "Say hello."},
                ],
                generation_settings={"max_new_tokens": 8, "do_sample": False},
                cancellation_signal=cancellation_signal,
            )

        self.assertEqual(model.generate_calls, [])


if __name__ == "__main__":
    unittest.main()
