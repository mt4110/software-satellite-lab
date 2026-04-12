from __future__ import annotations

import json
import sys
import time
from typing import Any, Callable

from gemma_core import CancellationSignal, RuntimeSession, SessionManager, generate_text_from_messages
from gemma_runtime import (
    UserFacingError,
    assistant_message_from_response,
    emit_warmup_progress,
    resolve_model_id,
    split_gemma_response,
    strip_thinking_from_messages,
    WARMUP_PHASE_PRIME_TOKEN,
    WarmupProgressCallback,
)


MODES = ("text", "tool")

TEXT_SYSTEM_PROMPT = (
    "You are a concise, careful assistant. Think privately before answering, but keep the "
    "visible answer direct and polished."
)
TEXT_PROMPT = "Explain when breadth-first search is a better fit than depth-first search."
TEXT_FOLLOW_UP = "Now compress that advice into one sentence I can remember."

TOOL_SYSTEM_PROMPT = (
    "You are a careful lab assistant. Calibration codes are not part of your world knowledge. "
    "When a user asks for a calibration code, use the available tool instead of guessing. "
    "Keep private reasoning private in the final answer."
)
TOOL_PROMPT = (
    "What is the calibration code for sensor-7? Use the tool first, then answer in one short sentence."
)

TEXT_GENERATION_SETTINGS = {"max_new_tokens": 256, "do_sample": False}
TOOL_GENERATION_SETTINGS = {"max_new_tokens": 192, "do_sample": False}
PREWARM_SYSTEM_PROMPT = (
    "You are a concise assistant. Think privately if needed, then reply with READY."
)
PREWARM_PROMPT = "Reply with READY."
PREWARM_GENERATION_SETTINGS = {"max_new_tokens": 1, "do_sample": False}

LAB_RECORD_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_lab_record",
        "description": "Looks up the latest calibration record for a lab device.",
        "parameters": {
            "type": "object",
            "properties": {
                "asset_id": {
                    "type": "string",
                    "description": "The device identifier such as sensor-7.",
                }
            },
            "required": ["asset_id"],
        },
    },
}

LAB_RECORDS = {
    "sensor-7": {
        "asset_id": "sensor-7",
        "calibration_code": "CAL-7Q4-ALPHA",
        "status": "ready",
        "checked_at_utc": "2026-04-07T08:30:00Z",
    }
}


def generate_assistant_response(
    session: RuntimeSession,
    messages: list[dict[str, Any]],
    generation_settings: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
    cancellation_signal: CancellationSignal | None = None,
) -> tuple[dict[str, Any], float]:
    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "enable_thinking": True,
    }
    if tools:
        template_kwargs["tools"] = tools

    current_settings = dict(generation_settings)
    total_elapsed = 0.0

    for _ in range(3):
        generation = generate_text_from_messages(
            session=session,
            messages=messages,
            generation_settings=current_settings,
            template_kwargs=template_kwargs,
            cancellation_signal=cancellation_signal,
        )
        response = generation.response
        total_elapsed += generation.elapsed_seconds

        tool_calls = response.get("tool_calls") or []
        content = response.get("content")
        has_final_content = isinstance(content, str) and content.strip() and not content.lstrip().startswith(
            "<|channel>thought"
        )

        if tool_calls or has_final_content:
            return response, total_elapsed

        max_new_tokens = int(current_settings.get("max_new_tokens", 0) or 0)
        if max_new_tokens >= 1024:
            return response, total_elapsed

        current_settings["max_new_tokens"] = min(max(max_new_tokens * 2, 256), 1024)

    return response, total_elapsed


def print_optional_thinking(response: dict[str, Any], enabled: bool) -> None:
    thinking = response.get("thinking")
    if enabled and isinstance(thinking, str) and thinking.strip():
        print("[thinking]", file=sys.stderr)
        print(thinking.strip(), file=sys.stderr)


def serialize_response(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": response.get("content"),
        "thinking": response.get("thinking"),
        "tool_calls": response.get("tool_calls"),
        "raw_text": response.get("raw_text"),
    }


def execute_lab_tool(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function", {})
    tool_name = function.get("name")
    arguments = function.get("arguments", {})
    if tool_name != "lookup_lab_record":
        raise UserFacingError(f"Unsupported tool requested during demo: `{tool_name}`.")

    if not isinstance(arguments, dict):
        raise UserFacingError("Tool call arguments were not parsed into an object.")

    asset_id = str(arguments.get("asset_id", "")).strip()
    if not asset_id:
        raise UserFacingError("`lookup_lab_record` requires a non-empty `asset_id`.")

    record = LAB_RECORDS.get(asset_id)
    if record is None:
        return {
            "asset_id": asset_id,
            "found": False,
            "message": "No record was found for that asset_id.",
        }

    return {
        "asset_id": asset_id,
        "found": True,
        "record": record,
    }


def tool_result_message(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "name": tool_name,
        "content": json.dumps(payload, ensure_ascii=False),
    }


def run_text_mode(
    session: RuntimeSession,
    system_prompt: str,
    first_prompt: str,
    follow_up_prompt: str,
    show_thinking: bool,
    cancellation_signal: CancellationSignal | None = None,
) -> dict[str, Any]:
    generation_settings = dict(TEXT_GENERATION_SETTINGS)

    first_turn_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": first_prompt},
    ]
    first_response, first_elapsed = generate_assistant_response(
        session=session,
        messages=first_turn_messages,
        generation_settings=generation_settings,
        cancellation_signal=cancellation_signal,
    )
    print_optional_thinking(first_response, show_thinking)

    first_assistant_raw = assistant_message_from_response(first_response, include_thinking=True)
    history_with_raw_thinking = first_turn_messages + [first_assistant_raw]
    stripped_history = strip_thinking_from_messages(history_with_raw_thinking)
    second_turn_messages = stripped_history + [{"role": "user", "content": follow_up_prompt}]

    second_response, second_elapsed = generate_assistant_response(
        session=session,
        messages=second_turn_messages,
        generation_settings=generation_settings,
        cancellation_signal=cancellation_signal,
    )
    print_optional_thinking(second_response, show_thinking)

    final_answer = second_response.get("content")
    if not isinstance(final_answer, str) or not final_answer.strip():
        raise UserFacingError("Thinking demo finished without a final answer.")

    return {
        "mode": "text",
        "system_prompt": system_prompt,
        "generation_settings": generation_settings,
        "simulation": False,
        "turns": [
            {
                "user_prompt": first_prompt,
                "assistant": serialize_response(first_response),
                "elapsed_seconds": round(first_elapsed, 3),
            },
            {
                "user_prompt": follow_up_prompt,
                "assistant": serialize_response(second_response),
                "elapsed_seconds": round(second_elapsed, 3),
            },
        ],
        "history_before_follow_up": history_with_raw_thinking,
        "history_used_for_follow_up": second_turn_messages[:-1],
        "thinking_stripped_for_follow_up": history_with_raw_thinking != second_turn_messages[:-1],
        "final_answer": final_answer.strip(),
    }


def run_tool_mode(
    session: RuntimeSession,
    system_prompt: str,
    prompt: str,
    show_thinking: bool,
    max_tool_iterations: int,
    cancellation_signal: CancellationSignal | None = None,
) -> dict[str, Any]:
    if max_tool_iterations < 1:
        raise UserFacingError("`--max-tool-iterations` must be at least 1.")

    generation_settings = dict(TOOL_GENERATION_SETTINGS)
    tools = [LAB_RECORD_TOOL]
    active_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    iterations = []
    saw_tool_call = False

    for _ in range(max_tool_iterations):
        response, elapsed_seconds = generate_assistant_response(
            session=session,
            messages=active_messages,
            generation_settings=generation_settings,
            tools=tools,
            cancellation_signal=cancellation_signal,
        )
        print_optional_thinking(response, show_thinking)

        assistant_raw = assistant_message_from_response(response, include_thinking=True)
        active_messages.append(assistant_raw)

        iteration_record: dict[str, Any] = {
            "assistant": serialize_response(response),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "messages_after_assistant": strip_thinking_from_messages(active_messages),
        }

        tool_calls = response.get("tool_calls") or []
        if tool_calls:
            saw_tool_call = True
            tool_results = []
            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                tool_name = function.get("name")
                tool_payload = execute_lab_tool(tool_call)
                active_messages.append(tool_result_message(str(tool_name), tool_payload))
                tool_results.append(
                    {
                        "tool_call": tool_call,
                        "tool_result": tool_payload,
                    }
                )
            iteration_record["tool_results"] = tool_results
            iterations.append(iteration_record)
            continue

        final_answer = response.get("content")
        if not isinstance(final_answer, str) or not final_answer.strip():
            raise UserFacingError("Tool-assisted thinking demo finished without a final answer.")
        if not saw_tool_call:
            raise UserFacingError(
                "Tool mode returned a final answer without calling the tool, so the tool-assisted demo is incomplete."
            )

        iterations.append(iteration_record)
        return {
            "mode": "tool",
            "system_prompt": system_prompt,
            "generation_settings": generation_settings,
            "simulation": False,
            "tools": tools,
            "prompt": prompt,
            "iterations": iterations,
            "active_turn_messages": active_messages,
            "history_for_next_user_turn": strip_thinking_from_messages(active_messages),
            "thought_continuity_preserved_within_tool_loop": True,
            "final_answer": final_answer.strip(),
        }

    raise UserFacingError(
        "Tool-assisted thinking demo did not finish within the configured tool loop limit."
    )


def simulated_response(raw_text: str) -> dict[str, Any]:
    response = split_gemma_response(raw_text)
    response["role"] = "assistant"
    response["raw_text"] = raw_text
    return response


def run_text_mode_simulated(
    system_prompt: str,
    first_prompt: str,
    follow_up_prompt: str,
    show_thinking: bool,
) -> dict[str, Any]:
    first_response = simulated_response(
        "<|channel>thought\n"
        "I should explain that BFS is better when the shallowest solution matters and when level order matters."
        "<channel|>"
        "Breadth-first search is better when you need the shortest path in an unweighted graph, when the likely answer is close to the start, and when level-by-level traversal matters."
    )
    print_optional_thinking(first_response, show_thinking)

    first_turn_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": first_prompt},
    ]
    first_assistant_raw = assistant_message_from_response(first_response, include_thinking=True)
    history_with_raw_thinking = first_turn_messages + [first_assistant_raw]
    stripped_history = strip_thinking_from_messages(history_with_raw_thinking)
    second_turn_messages = stripped_history + [{"role": "user", "content": follow_up_prompt}]

    second_response = simulated_response(
        "<|channel>thought\n"
        "The answer should be short and memorable without exposing the chain of thought."
        "<channel|>"
        "Use breadth-first search when you care most about the nearest valid answer or shortest unweighted path."
    )
    print_optional_thinking(second_response, show_thinking)

    return {
        "mode": "text",
        "system_prompt": system_prompt,
        "generation_settings": dict(TEXT_GENERATION_SETTINGS),
        "simulation": True,
        "turns": [
            {
                "user_prompt": first_prompt,
                "assistant": serialize_response(first_response),
                "elapsed_seconds": 0.0,
            },
            {
                "user_prompt": follow_up_prompt,
                "assistant": serialize_response(second_response),
                "elapsed_seconds": 0.0,
            },
        ],
        "history_before_follow_up": history_with_raw_thinking,
        "history_used_for_follow_up": second_turn_messages[:-1],
        "thinking_stripped_for_follow_up": history_with_raw_thinking != second_turn_messages[:-1],
        "final_answer": str(second_response["content"]).strip(),
    }


def run_tool_mode_simulated(
    system_prompt: str,
    prompt: str,
    show_thinking: bool,
) -> dict[str, Any]:
    tools = [LAB_RECORD_TOOL]
    active_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    first_response = simulated_response(
        "<|channel>thought\n"
        "I do not know calibration codes from memory, so I need to use the lookup tool for sensor-7."
        "<channel|>"
        "<|tool_call>call:lookup_lab_record{asset_id:<|\"|>sensor-7<|\"|>}<tool_call|>"
    )
    print_optional_thinking(first_response, show_thinking)
    active_messages.append(assistant_message_from_response(first_response, include_thinking=True))

    tool_call = first_response["tool_calls"][0]
    tool_payload = execute_lab_tool(tool_call)
    active_messages.append(tool_result_message("lookup_lab_record", tool_payload))

    second_response = simulated_response(
        "<|channel>thought\n"
        "The tool returned the code CAL-7Q4-ALPHA, so I can answer directly."
        "<channel|>"
        "The calibration code for sensor-7 is CAL-7Q4-ALPHA."
    )
    print_optional_thinking(second_response, show_thinking)
    active_messages.append(assistant_message_from_response(second_response, include_thinking=True))

    return {
        "mode": "tool",
        "system_prompt": system_prompt,
        "generation_settings": dict(TOOL_GENERATION_SETTINGS),
        "simulation": True,
        "tools": tools,
        "prompt": prompt,
        "iterations": [
            {
                "assistant": serialize_response(first_response),
                "elapsed_seconds": 0.0,
                "messages_after_assistant": strip_thinking_from_messages(active_messages[:3]),
                "tool_results": [
                    {
                        "tool_call": tool_call,
                        "tool_result": tool_payload,
                    }
                ],
            },
            {
                "assistant": serialize_response(second_response),
                "elapsed_seconds": 0.0,
                "messages_after_assistant": strip_thinking_from_messages(active_messages),
            },
        ],
        "active_turn_messages": active_messages,
        "history_for_next_user_turn": strip_thinking_from_messages(active_messages),
        "thought_continuity_preserved_within_tool_loop": True,
        "final_answer": str(second_response["content"]).strip(),
    }


def total_elapsed_seconds(result: dict[str, Any]) -> float:
    if "turns" in result:
        return round(sum(float(item.get("elapsed_seconds", 0.0)) for item in result["turns"]), 3)
    if "iterations" in result:
        return round(sum(float(item.get("elapsed_seconds", 0.0)) for item in result["iterations"]), 3)
    return 0.0


def warm_thinking_session(
    *,
    model_id: str | None = None,
    session_manager: SessionManager | None = None,
    cancellation_signal: CancellationSignal | None = None,
    progress_callback: WarmupProgressCallback | None = None,
) -> dict[str, Any]:
    resolved_model_id = (model_id or resolve_model_id()).strip() or resolve_model_id()
    owns_session_manager = session_manager is None
    manager = session_manager or SessionManager()
    started_at = time.perf_counter()

    try:
        session = manager.get_session(
            "thinking",
            resolved_model_id,
            progress_callback=progress_callback,
        )
        device_info = dict(session.device_info)

        emit_warmup_progress(
            progress_callback,
            phase=WARMUP_PHASE_PRIME_TOKEN,
            message="Priming first thinking token",
        )

        generation = generate_text_from_messages(
            session=session,
            messages=[
                {"role": "system", "content": PREWARM_SYSTEM_PROMPT},
                {"role": "user", "content": PREWARM_PROMPT},
            ],
            generation_settings=dict(PREWARM_GENERATION_SETTINGS),
            template_kwargs={
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
                "enable_thinking": True,
            },
            cancellation_signal=cancellation_signal,
        )
    finally:
        if owns_session_manager:
            manager.close_all()

    return {
        "model_id": resolved_model_id,
        "device_info": device_info,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "primed_text": generation.output_text.strip(),
    }


def run_thinking_session(
    *,
    mode: str,
    system_prompt: str | None = None,
    prompt: str | None = None,
    follow_up: str | None = None,
    show_thinking: bool = False,
    simulate: bool = False,
    max_tool_iterations: int = 3,
    model_id: str | None = None,
    session_manager: SessionManager | None = None,
    cancellation_signal: CancellationSignal | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise UserFacingError(f"Unsupported thinking mode `{mode}`.")

    resolved_model_id = (model_id or resolve_model_id()).strip() or resolve_model_id()
    resolved_system_prompt = system_prompt or (TEXT_SYSTEM_PROMPT if mode == "text" else TOOL_SYSTEM_PROMPT)
    first_prompt = (prompt or (TEXT_PROMPT if mode == "text" else TOOL_PROMPT)).strip()
    follow_up_prompt = (follow_up or TEXT_FOLLOW_UP).strip()

    if simulate:
        result = (
            run_text_mode_simulated(
                system_prompt=resolved_system_prompt,
                first_prompt=first_prompt,
                follow_up_prompt=follow_up_prompt,
                show_thinking=show_thinking,
            )
            if mode == "text"
            else run_tool_mode_simulated(
                system_prompt=resolved_system_prompt,
                prompt=first_prompt,
                show_thinking=show_thinking,
            )
        )
        return {
            **result,
            "model_id": resolved_model_id,
            "device_info": "simulated",
            "first_prompt": first_prompt,
            "follow_up_prompt": follow_up_prompt if mode == "text" else None,
            "elapsed_seconds": total_elapsed_seconds(result),
        }

    owns_session_manager = session_manager is None
    manager = session_manager or SessionManager()

    try:
        session = manager.get_session("thinking", resolved_model_id)
        device_info = dict(session.device_info)
        result = (
            run_text_mode(
                session=session,
                system_prompt=resolved_system_prompt,
                first_prompt=first_prompt,
                follow_up_prompt=follow_up_prompt,
                show_thinking=show_thinking,
                cancellation_signal=cancellation_signal,
            )
            if mode == "text"
            else run_tool_mode(
                session=session,
                system_prompt=resolved_system_prompt,
                prompt=first_prompt,
                show_thinking=show_thinking,
                max_tool_iterations=max_tool_iterations,
                cancellation_signal=cancellation_signal,
            )
        )
    finally:
        if owns_session_manager:
            manager.close_all()

    return {
        **result,
        "model_id": resolved_model_id,
        "device_info": device_info,
        "first_prompt": first_prompt,
        "follow_up_prompt": follow_up_prompt if mode == "text" else None,
        "elapsed_seconds": total_elapsed_seconds(result),
    }
