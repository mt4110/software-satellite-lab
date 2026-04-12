#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


REPO_BUG = "repo_bug"
MISSING_DEPENDENCY = "missing_dependency"
MISSING_AUTH = "missing_auth"
NETWORK_ISSUE = "network_issue"
HARDWARE_LIMIT = "hardware_limit"
UNSUPPORTED_MODE = "unsupported_mode"

BLOCKER_KINDS = (
    REPO_BUG,
    MISSING_DEPENDENCY,
    MISSING_AUTH,
    NETWORK_ISSUE,
    HARDWARE_LIMIT,
    UNSUPPORTED_MODE,
)

EXTERNAL_BLOCKER_KINDS = {
    MISSING_DEPENDENCY,
    MISSING_AUTH,
    NETWORK_ISSUE,
    HARDWARE_LIMIT,
    UNSUPPORTED_MODE,
}


@dataclass(frozen=True)
class BlockerInfo:
    kind: str
    message: str
    external: bool


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def is_external_blocker_kind(kind: str) -> bool:
    return kind in EXTERNAL_BLOCKER_KINDS


def classify_blocker(message: str) -> BlockerInfo:
    normalized_message = " ".join((message or "").strip().split()) or "Unknown issue."
    lowered = normalized_message.lower()

    if _contains_any(
        lowered,
        (
            "accept the model terms",
            "accept the model",
            "hf_token",
            "hugging_face_hub_token",
            "unauthorized",
            "forbidden",
            "401",
            "403",
            "model access failed",
            "login if the weights require authorization",
        ),
    ):
        kind = MISSING_AUTH
    elif _contains_any(
        lowered,
        (
            "failed to reach hugging face",
            "dns resolution",
            "network access",
            "timed out",
            "timeout",
            "connection error",
            "temporary failure in name resolution",
            "name or service not known",
            "offline",
            "no local model cache",
            "urlopen error",
            "could not resolve host",
        ),
    ):
        kind = NETWORK_ISSUE
    elif _contains_any(
        lowered,
        (
            "insufficient memory",
            "out of memory",
            "backend out of memory",
            "invalid buffer size",
            "buffer size:",
            "reduce concurrent apps",
            "fall back to cpu",
            "gpu backend is exhausted",
        ),
    ):
        kind = HARDWARE_LIMIT
    elif _contains_any(
        lowered,
        (
            "missing runtime dependency",
            "missing pdf rendering dependency",
            "install the local venv requirements",
            "ffmpeg is not available",
            "without `ffmpeg`",
            "use a local wav file or make sure `ffmpeg` is available",
            "input file does not exist",
            "input audio file does not exist",
            "sample audio",
            "python 3.10+",
            "pillow is required",
            "torch runtime or device detection was unavailable",
            "does not expose `automodelformultimodallm`",
        ),
    ):
        kind = MISSING_DEPENDENCY
    elif _contains_any(
        lowered,
        (
            "unsupported input",
            "unsupported audio input",
            "svg input is not accepted directly",
            "requires exactly",
            "must be at least",
            "no usable images were resolved",
            "tool-assisted thinking demo did not finish within the configured tool loop limit",
        ),
    ):
        kind = UNSUPPORTED_MODE
    else:
        kind = REPO_BUG

    return BlockerInfo(
        kind=kind,
        message=normalized_message,
        external=is_external_blocker_kind(kind),
    )


def build_blocker_record(message: str, kind: str | None = None) -> dict[str, Any]:
    if kind is None:
        info = classify_blocker(message)
    else:
        normalized_message = " ".join((message or "").strip().split()) or "Unknown issue."
        info = BlockerInfo(
            kind=kind,
            message=normalized_message,
            external=is_external_blocker_kind(kind),
        )
    return asdict(info)
