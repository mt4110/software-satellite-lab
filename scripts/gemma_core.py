#!/usr/bin/env python3
from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from gemma_runtime import (
    UserFacingError,
    apply_chat_template,
    assert_model_fetch_is_possible,
    emit_warmup_progress,
    format_load_error,
    import_multimodal_runtime,
    import_text_runtime,
    load_model_and_processor,
    move_batch_to_device,
    parse_generated_response,
    parse_generated_text,
    select_device,
    WARMUP_PHASE_CACHED_SESSION,
    WarmupProgressCallback,
)


TEXT_SESSION_KIND = "text"
MULTIMODAL_SESSION_KIND = "multimodal"


@dataclass(frozen=True)
class SessionKey:
    session_kind: str
    model_id: str
    device_class: str


@dataclass
class RuntimeSession:
    key: SessionKey
    processor: Any
    model: Any
    torch: Any
    device_info: dict[str, Any]

    @property
    def model_id(self) -> str:
        return self.key.model_id


@dataclass
class GenerationResult:
    response: dict[str, Any]
    output_text: str
    elapsed_seconds: float
    input_token_count: int


class GenerationCancelled(RuntimeError):
    pass


class CancellationSignal:
    def __init__(self) -> None:
        self._event = threading.Event()

    def request_cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


def canonical_session_kind(session_kind: str) -> str:
    normalized = (session_kind or "").strip().lower()
    if normalized in {TEXT_SESSION_KIND, "thinking", "tools", "long-context"}:
        return TEXT_SESSION_KIND
    if normalized in {MULTIMODAL_SESSION_KIND, "vision", "audio"}:
        return MULTIMODAL_SESSION_KIND
    raise ValueError(f"Unsupported session kind `{session_kind}`.")


def _import_runtime(session_kind: str) -> tuple[Any, Any, Any]:
    if session_kind == TEXT_SESSION_KIND:
        return import_text_runtime()
    if session_kind == MULTIMODAL_SESSION_KIND:
        return import_multimodal_runtime()
    raise ValueError(f"Unsupported session kind `{session_kind}`.")


def release_session(session: RuntimeSession | None) -> None:
    if session is None:
        return

    torch_module = session.torch
    session.processor = None
    session.model = None
    gc.collect()

    try:
        if session.device_info["name"] == "cuda" and hasattr(torch_module.cuda, "empty_cache"):
            torch_module.cuda.empty_cache()
    except Exception:
        pass

    try:
        if session.device_info["name"] == "mps" and hasattr(torch_module, "mps"):
            empty_cache = getattr(torch_module.mps, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
    except Exception:
        pass


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[SessionKey, RuntimeSession] = {}

    def get_session(
        self,
        session_kind: str,
        model_id: str,
        *,
        progress_callback: WarmupProgressCallback | None = None,
    ) -> RuntimeSession:
        normalized_kind = canonical_session_kind(session_kind)
        device_info: dict[str, Any] | None = None

        try:
            torch, auto_processor_cls, auto_model_cls = _import_runtime(normalized_kind)
            device_info = select_device(torch)
            key = SessionKey(
                session_kind=normalized_kind,
                model_id=model_id,
                device_class=str(device_info["name"]),
            )

            cached = self._sessions.get(key)
            if cached is not None:
                emit_warmup_progress(
                    progress_callback,
                    phase=WARMUP_PHASE_CACHED_SESSION,
                    message="Using cached shared session",
                )
                return cached

            assert_model_fetch_is_possible(model_id)
            processor, model = load_model_and_processor(
                model_id=model_id,
                device_info=device_info,
                auto_model_cls=auto_model_cls,
                auto_processor_cls=auto_processor_cls,
                progress_callback=progress_callback,
            )
            session = RuntimeSession(
                key=key,
                processor=processor,
                model=model,
                torch=torch,
                device_info=device_info,
            )
            self._sessions[key] = session
            return session
        except UserFacingError:
            raise
        except Exception as exc:
            raise UserFacingError(
                format_load_error(model_id, device_info or {"label": "unresolved"}, exc)
            ) from exc

    def clear(
        self,
        *,
        session_kind: str | None = None,
        model_id: str | None = None,
        device_class: str | None = None,
    ) -> int:
        normalized_kind = canonical_session_kind(session_kind) if session_kind is not None else None
        keys_to_close = [
            key
            for key in self._sessions
            if (normalized_kind is None or key.session_kind == normalized_kind)
            and (model_id is None or key.model_id == model_id)
            and (device_class is None or key.device_class == device_class)
        ]
        for key in keys_to_close:
            release_session(self._sessions.pop(key))
        return len(keys_to_close)

    def close_all(self) -> int:
        return self.clear()

    def cached_keys(self) -> list[SessionKey]:
        return list(self._sessions.keys())


def build_generation_inputs(
    session: RuntimeSession,
    messages: list[dict[str, Any]],
    *,
    template_kwargs: dict[str, Any] | None = None,
    processor_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, int]:
    resolved_template_kwargs = dict(template_kwargs or {})
    if resolved_template_kwargs.get("tokenize"):
        inputs = apply_chat_template(session.processor, messages, **resolved_template_kwargs)
    else:
        rendered_prompt = apply_chat_template(session.processor, messages, **resolved_template_kwargs)
        resolved_processor_kwargs = {"text": rendered_prompt, "return_tensors": "pt"}
        resolved_processor_kwargs.update(processor_kwargs or {})
        inputs = session.processor(**resolved_processor_kwargs)

    inputs = move_batch_to_device(inputs, getattr(session.model, "device", session.device_info["name"]))
    input_length = int(inputs["input_ids"].shape[-1])
    return inputs, input_length


def _raise_if_cancelled(cancellation_signal: CancellationSignal | None) -> None:
    if cancellation_signal is not None and cancellation_signal.is_cancelled():
        raise GenerationCancelled("Generation was cancelled before the next safe boundary completed.")


def _build_stopping_criteria(
    session: RuntimeSession,
    cancellation_signal: CancellationSignal | None,
) -> Any | None:
    if cancellation_signal is None:
        return None

    from transformers import StoppingCriteria, StoppingCriteriaList  # type: ignore

    torch_module = session.torch

    class _CancellationStoppingCriteria(StoppingCriteria):
        def __init__(self, signal: CancellationSignal) -> None:
            super().__init__()
            self._signal = signal

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            should_stop = self._signal.is_cancelled()
            return torch_module.full(
                (input_ids.shape[0],),
                should_stop,
                device=input_ids.device,
                dtype=torch_module.bool,
            )

    return StoppingCriteriaList([_CancellationStoppingCriteria(cancellation_signal)])


def generate_text_from_messages(
    session: RuntimeSession,
    messages: list[dict[str, Any]],
    generation_settings: dict[str, Any],
    *,
    template_kwargs: dict[str, Any] | None = None,
    processor_kwargs: dict[str, Any] | None = None,
    cancellation_signal: CancellationSignal | None = None,
) -> GenerationResult:
    _raise_if_cancelled(cancellation_signal)
    inputs, input_length = build_generation_inputs(
        session=session,
        messages=messages,
        template_kwargs=template_kwargs,
        processor_kwargs=processor_kwargs,
    )
    generation_kwargs = dict(generation_settings)
    stopping_criteria = _build_stopping_criteria(session, cancellation_signal)
    if stopping_criteria is not None and "stopping_criteria" not in generation_kwargs:
        generation_kwargs["stopping_criteria"] = stopping_criteria

    started_at = time.perf_counter()
    with session.torch.inference_mode():
        outputs = session.model.generate(**inputs, **generation_kwargs)
    elapsed_seconds = time.perf_counter() - started_at
    _raise_if_cancelled(cancellation_signal)

    generated_tokens = outputs[0][input_length:]
    response = parse_generated_response(session.processor, generated_tokens)
    output_text = parse_generated_text(session.processor, generated_tokens).strip()

    return GenerationResult(
        response=response,
        output_text=output_text,
        elapsed_seconds=elapsed_seconds,
        input_token_count=input_length,
    )
