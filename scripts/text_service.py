from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from gemma_core import CancellationSignal, SessionManager, generate_text_from_messages
from gemma_runtime import UserFacingError, build_text_messages, repo_root, resolve_model_id, timestamp_slug


TASKS = ("chat", "summarize", "translate", "code")
JAPANESE_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")

DEFAULT_PROMPTS = {
    "chat": "Explain binary search in 3 bullets.",
    "summarize": (
        "Gemma Lab keeps the stack intentionally small: plain Python, a local venv, "
        "Hugging Face Transformers, and file-based artifacts. The goal is to validate "
        "one capability at a time, keep failure modes obvious, and avoid framework "
        "layers until they are truly necessary."
    ),
    "translate": "今日はいい天気です。",
    "code": "Write a Python function that returns the factorial of n recursively.",
}

DEFAULT_SYSTEM_PROMPTS = {
    "chat": "You are a concise, helpful assistant.",
    "summarize": "You summarize clearly and preserve the core facts.",
    "translate": (
        "You are a precise translator between Japanese and English. "
        "Return only the translation."
    ),
    "code": (
        "You are a careful coding assistant. Prefer plain Python and produce minimal, "
        "correct output."
    ),
}

GENERATION_SETTINGS = {
    "chat": {"max_new_tokens": 192, "do_sample": False},
    "summarize": {"max_new_tokens": 160, "do_sample": False},
    "translate": {"max_new_tokens": 128, "do_sample": False},
    "code": {"max_new_tokens": 256, "do_sample": False},
}


def artifacts_root() -> Path:
    return repo_root() / "artifacts" / "text"


def detect_translation_target(text: str) -> str:
    return "English" if JAPANESE_PATTERN.search(text) else "Japanese"


def default_output_path(task: str) -> Path:
    return artifacts_root() / f"{timestamp_slug()}-{task}.json"


def resolve_prompt(task: str, prompt: str | None) -> str:
    return (prompt or DEFAULT_PROMPTS[task]).strip()


def resolve_system_prompt(task: str, system_prompt: str | None) -> str:
    return (system_prompt or DEFAULT_SYSTEM_PROMPTS[task]).strip()


def build_user_prompt(task: str, prompt: str) -> str:
    if task == "summarize":
        return "Summarize the following text in 3 concise bullet points.\n\n" + prompt
    if task == "translate":
        target_language = detect_translation_target(prompt)
        return (
            f"Translate the following text into {target_language}. "
            "Return only the translation.\n\n"
            + prompt
        )
    return prompt


def run_text_task(
    *,
    task: str,
    prompt: str | None = None,
    system_prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    model_id: str | None = None,
    session_manager: SessionManager | None = None,
    cancellation_signal: CancellationSignal | None = None,
) -> dict[str, Any]:
    if task not in TASKS:
        raise UserFacingError(f"Unsupported text task `{task}`.")

    resolved_model_id = (model_id or resolve_model_id()).strip() or resolve_model_id()
    resolved_prompt = resolve_prompt(task, prompt)
    resolved_system_prompt = resolve_system_prompt(task, system_prompt)
    user_prompt = build_user_prompt(task, resolved_prompt)
    generation_settings = dict(GENERATION_SETTINGS[task])

    owns_session_manager = session_manager is None
    manager = session_manager or SessionManager()

    try:
        session = manager.get_session("text", resolved_model_id)
        device_info = dict(session.device_info)
        messages_to_run = copy.deepcopy(messages) if messages is not None else build_text_messages(
            system_prompt=resolved_system_prompt,
            user_prompt=user_prompt,
        )
        generation = generate_text_from_messages(
            session=session,
            messages=messages_to_run,
            generation_settings=generation_settings,
            cancellation_signal=cancellation_signal,
        )
        output_text = generation.output_text.strip()
        if not output_text:
            raise UserFacingError("Model returned an empty text response.")
    finally:
        if owns_session_manager:
            manager.close_all()

    return {
        "task": task,
        "model_id": resolved_model_id,
        "prompt": resolved_prompt,
        "system_prompt": resolved_system_prompt,
        "resolved_user_prompt": user_prompt,
        "messages": messages_to_run,
        "generation_settings": generation_settings,
        "device_info": device_info,
        "elapsed_seconds": generation.elapsed_seconds,
        "output_text": output_text,
    }
