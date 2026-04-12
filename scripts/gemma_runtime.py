#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
import re
import socket
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_AUDIO_MODEL_ID = "google/gemma-4-E2B-it"
MIN_RUNTIME_PYTHON = (3, 10)
AUDIO_CAPABLE_MODEL_MARKERS = ("gemma-4-e2b", "gemma-4-e4b")
THINKING_TRIGGER_TOKEN = "<|think|>"
THINKING_BLOCK_RE = re.compile(r"<\|channel\>thought\n(?P<thinking>.*?)<channel\|>", re.DOTALL)
TOOL_CALL_BLOCK_RE = re.compile(r"<\|tool_call\>(?P<tool_call>.*?)<tool_call\|>", re.DOTALL)
TURN_END_RE = re.compile(r"<\|turn\|>|<\|end_of_text\|>|<eos>|</s>")

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+",
)


class UserFacingError(RuntimeError):
    pass


WARMUP_PHASE_LOAD_PROCESSOR = "load_processor_assets"
WARMUP_PHASE_LOAD_MODEL = "load_model_weights"
WARMUP_PHASE_ATTACH_MODEL = "attach_model_to_device"
WARMUP_PHASE_SESSION_READY = "shared_session_ready"
WARMUP_PHASE_PRIME_TOKEN = "prime_first_thinking_token"
WARMUP_PHASE_CACHED_SESSION = "cached_shared_session"


@dataclass(frozen=True)
class WarmupProgress:
    phase: str
    message: str


WarmupProgressCallback = Callable[[WarmupProgress | str], None]


def emit_warmup_progress(
    progress_callback: WarmupProgressCallback | None,
    *,
    phase: str,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(WarmupProgress(phase=phase, message=message))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def expected_repo_venv_python(root: Path | None = None) -> Path:
    return (root or repo_root()) / ".venv" / "bin" / "python"


def build_requirements_install_command(root: Path | None = None) -> str:
    return f"{expected_repo_venv_python(root)} -m pip install -r requirements.txt"


def current_python_version_text(version_info: tuple[int, int, int] | None = None) -> str:
    if version_info is None:
        return sys.version.split()[0]
    major, minor, patch = version_info[:3]
    return f"{major}.{minor}.{patch}"


def using_repo_venv_python(
    *,
    root: Path | None = None,
    current_executable: str | None = None,
) -> bool:
    expected_python = expected_repo_venv_python(root)
    current_python = Path(current_executable or sys.executable).expanduser()
    return str(current_python) == str(expected_python)


def build_repo_runtime_hint(
    *,
    root: Path | None = None,
    current_executable: str | None = None,
    current_version: tuple[int, int, int] | None = None,
) -> str:
    expected_python = expected_repo_venv_python(root)
    current_python = str(Path(current_executable or sys.executable).expanduser())
    version_info = current_version or tuple(sys.version_info[:3])
    hints: list[str] = []

    if version_info < MIN_RUNTIME_PYTHON:
        hints.append(
            "Current interpreter is Python "
            f"{current_python_version_text(version_info)}, but this repo expects Python 3.10+."
        )

    if not using_repo_venv_python(root=root, current_executable=current_python):
        hints.append(f"Current interpreter: `{current_python}`. Use `{expected_python}` for repo commands.")

    return " ".join(hints).strip()


def missing_runtime_dependency_message(subject: str, *, root: Path | None = None) -> str:
    message = (
        f"Missing {subject}. Install the repo requirements with "
        f"`{build_requirements_install_command(root)}`."
    )
    runtime_hint = build_repo_runtime_hint(root=root)
    if runtime_hint:
        return f"{message} {runtime_hint}"
    return message


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def timestamp_slug() -> str:
    current = datetime.now(timezone.utc)
    return current.strftime("%Y%m%dT%H%M%S%fZ") + f"-p{os.getpid()}"


def resolve_model_id() -> str:
    return os.environ.get("GEMMA_MODEL_ID", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID


def model_supports_audio(model_id: str) -> bool:
    normalized = (model_id or "").strip().lower()
    return any(marker in normalized for marker in AUDIO_CAPABLE_MODEL_MARKERS)


def resolve_audio_model_selection(
    *,
    base_model_id: str | None = None,
    audio_model_id: str | None = None,
) -> tuple[str, str]:
    explicit_audio_model = (audio_model_id or "").strip()
    if explicit_audio_model:
        return explicit_audio_model, "explicit_audio_model"

    audio_override = os.environ.get("GEMMA_AUDIO_MODEL_ID", "").strip()
    if audio_override:
        return audio_override, "GEMMA_AUDIO_MODEL_ID"

    model_id = (base_model_id or resolve_model_id()).strip() or DEFAULT_MODEL_ID
    if model_supports_audio(model_id):
        return model_id, "selected_base_model" if base_model_id else "GEMMA_MODEL_ID"

    return DEFAULT_AUDIO_MODEL_ID, "default"


def resolve_audio_model_id() -> str:
    return resolve_audio_model_selection()[0]


def import_text_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency dependent
        raise UserFacingError(missing_runtime_dependency_message("runtime dependency")) from exc

    return torch, AutoProcessor, AutoModelForCausalLM


def import_multimodal_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch  # type: ignore
        import transformers  # type: ignore
        from transformers import AutoProcessor  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency dependent
        raise UserFacingError(missing_runtime_dependency_message("runtime dependency")) from exc

    auto_model_cls = getattr(transformers, "AutoModelForMultimodalLM", None)
    if auto_model_cls is None:
        version = getattr(transformers, "__version__", "unknown")
        message = (
            "The installed Transformers runtime does not expose `AutoModelForMultimodalLM`, "
            "which Gemma 4 vision/audio requires. "
            f"Current transformers version: {version}. "
            "Recreate or upgrade the local venv to Python 3.10+ and reinstall the repo "
            f"requirements with `{build_requirements_install_command()}`."
        )
        runtime_hint = build_repo_runtime_hint()
        if runtime_hint:
            message = f"{message} {runtime_hint}"
        raise UserFacingError(message)

    return torch, AutoProcessor, auto_model_cls


def import_image_module() -> Any:
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency dependent
        raise UserFacingError(missing_runtime_dependency_message("runtime dependency")) from exc

    return Image


def import_vision_runtime() -> tuple[Any, Any, Any, Any]:
    Image = import_image_module()
    torch, auto_processor_cls, auto_model_cls = import_multimodal_runtime()
    return torch, Image, auto_processor_cls, auto_model_cls


def import_audio_runtime() -> tuple[Any, Any, Any]:
    return import_multimodal_runtime()


def import_pdf_renderer() -> Any:
    try:
        import pypdfium2 as pdfium  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency dependent
        raise UserFacingError(missing_runtime_dependency_message("PDF rendering dependency")) from exc

    return pdfium


def select_device(torch: Any) -> dict[str, Any]:
    if bool(torch.cuda.is_available()):
        dtype = (
            torch.bfloat16
            if bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            else torch.float16
        )
        device_name = torch.cuda.get_device_name(0)
        return {
            "name": "cuda",
            "label": f"cuda ({device_name})",
            "dtype": dtype,
            "dtype_name": str(dtype).replace("torch.", ""),
        }

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is not None and bool(mps_backend.is_available()):
        return {
            "name": "mps",
            "label": "mps (Apple Metal)",
            "dtype": torch.float16,
            "dtype_name": "float16",
        }

    return {
        "name": "cpu",
        "label": "cpu",
        "dtype": torch.float32,
        "dtype_name": "float32",
    }


def huggingface_cache_root() -> Path:
    cache_override = os.environ.get("HF_HUB_CACHE") or os.environ.get("TRANSFORMERS_CACHE")
    if cache_override:
        return Path(cache_override).expanduser()

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def model_cache_dir(model_id: str) -> Path:
    return huggingface_cache_root() / f"models--{model_id.replace('/', '--')}"


def huggingface_dns_available() -> bool:
    try:
        socket.getaddrinfo("huggingface.co", 443, type=socket.SOCK_STREAM)
    except OSError:
        return False
    return True


def assert_model_fetch_is_possible(model_id: str) -> None:
    cache_dir = model_cache_dir(model_id)
    if cache_dir.exists():
        return

    if huggingface_dns_available():
        return

    raise UserFacingError(
        f"Failed to reach Hugging Face while loading `{model_id}`. DNS resolution for `huggingface.co` is unavailable "
        f"and no local model cache was found at `{cache_dir}`."
    )


def apply_chat_template(processor: Any, messages: list[dict[str, Any]], **overrides: Any) -> Any:
    enable_thinking = bool(overrides.get("enable_thinking", False))
    rendered_messages = (
        enable_gemma_thinking(messages) if enable_thinking else copy.deepcopy(messages)
    )
    rendered_messages = normalize_chat_messages(rendered_messages)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    kwargs.update(overrides)
    try:
        return processor.apply_chat_template(rendered_messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return processor.apply_chat_template(rendered_messages, **kwargs)


def move_batch_to_device(batch: Any, device_name: str) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device_name)

    if isinstance(batch, dict):
        return {
            key: value.to(device_name) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

    return batch


def add_system_prefix(messages: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    updated = copy.deepcopy(messages)
    for message in updated:
        if message.get("role") != "system":
            continue

        content = message.get("content")
        if isinstance(content, str):
            text = content.lstrip()
            if text.startswith(prefix):
                return updated
            message["content"] = f"{prefix}\n{text}" if text else prefix
            return updated

        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    continue
                text = str(part.get("text", "")).lstrip()
                if text.startswith(prefix):
                    return updated
                part["text"] = f"{prefix}\n{text}" if text else prefix
                return updated

            content.insert(0, {"type": "text", "text": prefix})
            return updated

    updated.insert(0, {"role": "system", "content": prefix})
    return updated


def normalize_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(messages)
    for message in normalized:
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}]
            continue

        if isinstance(content, list):
            rebuilt = []
            for part in content:
                if isinstance(part, str):
                    rebuilt.append({"type": "text", "text": part})
                    continue
                rebuilt.append(part)
            message["content"] = rebuilt
    return normalized


def enable_gemma_thinking(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return add_system_prefix(messages, THINKING_TRIGGER_TOKEN)


def normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []

    normalized = []
    for item in tool_calls:
        if not isinstance(item, dict):
            continue

        if item.get("type") == "function" and isinstance(item.get("function"), dict):
            normalized.append(item)
            continue

        if isinstance(item.get("name"), str):
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "arguments": item.get("arguments", {}),
                    },
                }
            )
    return normalized


def parse_scalar_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def parse_tool_arguments(arguments_text: str) -> dict[str, Any] | None:
    cleaned = arguments_text.strip()
    if not cleaned:
        return {}

    cleaned = cleaned.replace('<|"|>', '"')
    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            loaded = json.loads(cleaned)
        except json.JSONDecodeError:
            inner = cleaned[1:-1].strip()
        else:
            return loaded if isinstance(loaded, dict) else None
    else:
        inner = cleaned

    if not inner:
        return {}

    parsed: dict[str, Any] = {}
    for part in inner.split(","):
        key, separator, value = part.partition(":")
        if not separator:
            return None
        normalized_key = key.strip().strip('"').strip("'")
        if not normalized_key:
            return None
        parsed[normalized_key] = parse_scalar_value(value.strip())
    return parsed


def parse_tool_call_block(block: str) -> dict[str, Any] | None:
    cleaned = block.strip()
    if not cleaned.startswith("call:"):
        return None

    remainder = cleaned[len("call:") :].strip()
    name, brace, arguments = remainder.partition("{")
    tool_name = name.strip()
    if not tool_name:
        return None

    arguments_payload: Any = {}
    if brace:
        parsed_arguments = parse_tool_arguments("{" + arguments)
        arguments_payload = parsed_arguments if parsed_arguments is not None else {}

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": arguments_payload,
        },
    }


def cleanup_response_text(text: str) -> str:
    cleaned = TURN_END_RE.sub("", text or "")
    cleaned = cleaned.replace("<bos>", "").replace("</bos>", "")
    cleaned = cleaned.strip()
    cleaned = re.sub(r"^\s*(assistant|model)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def split_gemma_response(raw_text: str) -> dict[str, Any]:
    working = raw_text or ""
    thinking = None

    thinking_match = THINKING_BLOCK_RE.search(working)
    if thinking_match:
        thinking = cleanup_response_text(thinking_match.group("thinking"))
        working = THINKING_BLOCK_RE.sub("", working, count=1)

    tool_calls = []
    for match in TOOL_CALL_BLOCK_RE.finditer(working):
        tool_call = parse_tool_call_block(match.group("tool_call"))
        if tool_call is not None:
            tool_calls.append(tool_call)
    working = TOOL_CALL_BLOCK_RE.sub("", working)

    return {
        "thinking": thinking or None,
        "tool_calls": tool_calls,
        "content": cleanup_response_text(working) or None,
    }


def normalize_response_text(parsed: dict[str, Any] | str | None, fallback: dict[str, Any]) -> dict[str, Any]:
    if isinstance(parsed, str):
        message: dict[str, Any] = {"content": parsed}
    elif isinstance(parsed, dict):
        message = parsed
    else:
        message = {}
    tool_calls = normalize_tool_calls(message.get("tool_calls"))
    content = message.get("content")
    thinking = message.get("thinking")

    if not isinstance(content, str) or not content.strip():
        for key in ("text", "response", "answer"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                content = value
                break

    if (not isinstance(content, str) or not content.strip()) and fallback.get("content"):
        content = fallback["content"]

    if (not isinstance(thinking, str) or not thinking.strip()) and fallback.get("thinking"):
        thinking = fallback["thinking"]

    if not tool_calls and fallback.get("tool_calls"):
        tool_calls = fallback["tool_calls"]

    normalized_content = cleanup_response_text(content) if isinstance(content, str) else ""
    normalized_thinking = cleanup_response_text(thinking) if isinstance(thinking, str) else ""

    return {
        "role": "assistant",
        "content": normalized_content or None,
        "thinking": normalized_thinking or None,
        "tool_calls": tool_calls,
    }


def parse_generated_response(processor: Any, generated_tokens: Any) -> dict[str, Any]:
    raw_text = processor.decode(generated_tokens, skip_special_tokens=False)
    parser = getattr(processor, "parse_response", None)
    parsed = None

    if callable(parser):
        try:
            parsed = parser(raw_text)
        except Exception:
            cleaned = processor.decode(generated_tokens, skip_special_tokens=True)
            try:
                parsed = parser(cleaned)
            except Exception:
                parsed = None

    fallback = split_gemma_response(raw_text)
    normalized = normalize_response_text(parsed, fallback)
    normalized["raw_text"] = raw_text
    return normalized


def parse_generated_text(processor: Any, generated_tokens: Any) -> str:
    parsed = parse_generated_response(processor, generated_tokens)
    if isinstance(parsed.get("content"), str) and parsed["content"].strip():
        return parsed["content"].strip()

    cleaned = processor.decode(generated_tokens, skip_special_tokens=True).strip()
    cleaned = re.sub(r"^\s*(assistant|model)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def assistant_message_from_response(
    response: dict[str, Any],
    include_thinking: bool = True,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant"}
    content = response.get("content")
    thinking = response.get("thinking")
    tool_calls = response.get("tool_calls")

    if isinstance(content, str) and content.strip():
        message["content"] = content.strip()
    if include_thinking and isinstance(thinking, str) and thinking.strip():
        message["thinking"] = thinking.strip()
    if isinstance(tool_calls, list) and tool_calls:
        message["tool_calls"] = copy.deepcopy(tool_calls)

    return message


def strip_assistant_thinking(message: dict[str, Any]) -> dict[str, Any]:
    sanitized = copy.deepcopy(message)
    if sanitized.get("role") == "assistant":
        sanitized.pop("thinking", None)
        sanitized.pop("raw_text", None)
    return sanitized


def strip_thinking_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [strip_assistant_thinking(message) for message in messages]


def thinking_artifacts_root() -> Path:
    return repo_root() / "artifacts" / "thinking"


def default_thinking_artifact_path(mode: str) -> Path:
    return thinking_artifacts_root() / f"{timestamp_slug()}-{mode}.json"


def build_text_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def load_model_and_processor(
    model_id: str,
    device_info: dict[str, Any],
    auto_model_cls: Any,
    auto_processor_cls: Any,
    progress_callback: WarmupProgressCallback | None = None,
) -> tuple[Any, Any]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    common_kwargs: dict[str, Any] = {}
    if token:
        common_kwargs["token"] = token

    emit_warmup_progress(
        progress_callback,
        phase=WARMUP_PHASE_LOAD_PROCESSOR,
        message="Loading processor assets",
    )
    processor = auto_processor_cls.from_pretrained(model_id, **common_kwargs)

    model_kwargs: dict[str, Any] = {"dtype": device_info["dtype"]}
    model_kwargs.update(common_kwargs)

    emit_warmup_progress(
        progress_callback,
        phase=WARMUP_PHASE_LOAD_MODEL,
        message="Loading model weights",
    )
    if device_info["name"] == "mps":
        model = auto_model_cls.from_pretrained(model_id, **model_kwargs)
        emit_warmup_progress(
            progress_callback,
            phase=WARMUP_PHASE_ATTACH_MODEL,
            message="Attaching model to Apple Metal",
        )
        model.to("mps")
    else:
        emit_warmup_progress(
            progress_callback,
            phase=WARMUP_PHASE_ATTACH_MODEL,
            message=f"Placing model on {device_info['label']}",
        )
        model_kwargs["device_map"] = "auto"
        model = auto_model_cls.from_pretrained(model_id, **model_kwargs)

    model.eval()
    emit_warmup_progress(
        progress_callback,
        phase=WARMUP_PHASE_SESSION_READY,
        message="Shared session ready",
    )
    return processor, model


def is_memory_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "out of memory",
        "not enough memory",
        "insufficient memory",
        "mps backend out of memory",
        "cuda out of memory",
    )
    return any(marker in text for marker in markers)


def format_load_error(model_id: str, device_info: dict[str, Any], exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {exc}"
    lowered = message.lower()

    if is_memory_error(exc):
        return (
            f"Insufficient memory while loading or running `{model_id}` on {device_info['label']}. "
            "Try a smaller Gemma 4 model, reduce concurrent apps, or fall back to CPU if a GPU backend is exhausted.\n"
            f"Original error: {message}"
        )

    if any(token in lowered for token in ("401", "403", "gated", "forbidden", "unauthorized", "access")):
        return (
            f"Model access failed for `{model_id}`. Accept the model terms on Hugging Face and provide a valid "
            "`HF_TOKEN` or login if the weights require authorization.\n"
            f"Original error: {message}"
        )

    if any(token in lowered for token in ("404", "not found", "repositorynotfounderror")):
        return (
            f"Model `{model_id}` was not found. Check `GEMMA_MODEL_ID` and confirm the model id is correct.\n"
            f"Original error: {message}"
        )

    if any(token in lowered for token in ("name or service not known", "connection", "timed out", "dns", "offline")):
        return (
            f"Failed to reach Hugging Face while loading `{model_id}`. Network access or DNS resolution appears unavailable.\n"
            f"Original error: {message}"
        )

    return f"Failed to run `{model_id}` on {device_info['label']}.\nOriginal error: {message}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_runtime_header(
    model_id: str,
    device_info: dict[str, Any],
    generation_settings: dict[str, Any],
) -> None:
    print(f"selected_model_id: {model_id}", flush=True)
    print(f"selected_device: {device_info['label']}", flush=True)
    print(f"selected_dtype: {device_info['dtype_name']}", flush=True)
    print("generation_settings:", flush=True)
    print(json.dumps(generation_settings, ensure_ascii=False, indent=2), flush=True)
