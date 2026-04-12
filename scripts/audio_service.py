from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from asset_preprocessing import normalize_audio_input
from fetch_demo_assets import SAMPLE_AUDIO_TEXT
from gemma_core import CancellationSignal, SessionManager, generate_text_from_messages
from gemma_runtime import (
    UserFacingError,
    build_text_messages,
    repo_root,
    resolve_audio_model_selection,
    resolve_model_id,
    timestamp_slug,
)


MODES = ("transcribe", "translate", "summarize")
MAX_AUDIO_SECONDS = 30.0
TARGET_SAMPLE_RATE = 16_000
UNCLEAR_AUDIO_RESPONSE = "[unclear audio]"
SAMPLE_AUDIO_RELATIVE_PATH = Path("assets/audio/sample_audio.wav")
MIN_CLEAR_AUDIO_ACTIVE_SECONDS = 0.75
MIN_CLEAR_AUDIO_ACTIVE_RATIO = 0.12
TOKEN_RE = re.compile(r"[a-z0-9]+")
JAPANESE_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")

DEFAULT_TARGET_LANGUAGE = "Japanese"
MIN_JAPANESE_TRANSLATION_CHAR_COUNT = 6
ALLOWED_TRANSLATION_SOURCE_TOKENS = {"gemma"}

DEFAULT_SYSTEM_PROMPTS = {
    "transcribe": (
        "You are a careful speech transcription assistant. "
        "Do not invent words. If the audio is unclear, say so plainly."
    ),
    "translate": (
        "You are a careful speech translation assistant. "
        "Translate only what the audio supports and do not guess."
    ),
    "summarize": (
        "You summarize spoken content conservatively. "
        "If the speech is unclear or absent, say so clearly instead of filling gaps."
    ),
}

GENERATION_SETTINGS = {
    "transcribe": {"max_new_tokens": 192, "do_sample": False},
    "translate": {"max_new_tokens": 192, "do_sample": False},
    "summarize": {"max_new_tokens": 128, "do_sample": False},
}


def artifacts_root() -> Path:
    return repo_root() / "artifacts" / "audio"


def default_output_path(mode: str) -> Path:
    return artifacts_root() / f"{timestamp_slug()}-{mode}.json"


def resolve_prompt(mode: str, prompt: str | None, target_language: str) -> str:
    if prompt:
        return prompt.strip()

    if mode == "transcribe":
        return (
            "Transcribe the speech in its original spoken language. "
            "Return only the transcript text on one line. "
            "Use ASCII when the speech is in English. "
            f"If no clearly intelligible words are present, return exactly {UNCLEAR_AUDIO_RESPONSE}."
        )

    if mode == "translate":
        return (
            f"Translate the spoken content into {target_language}. "
            f"Return only the {target_language} translation. "
            f"If the speech is unclear, return exactly {UNCLEAR_AUDIO_RESPONSE}."
        )

    return (
        "Summarize the spoken content in 2 concise bullet points. "
        f"If the speech is unclear or absent, return exactly {UNCLEAR_AUDIO_RESPONSE}."
    )


def resolve_system_prompt(mode: str, system_prompt: str | None) -> str:
    return (system_prompt or DEFAULT_SYSTEM_PROMPTS[mode]).strip()


def resolve_audio_path(raw_path: Path) -> Path:
    path = raw_path.expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    return path.resolve()


def resolve_audio_input(raw_path: Path) -> dict[str, Any]:
    path = resolve_audio_path(raw_path)

    if not path.exists():
        sample_path = repo_root() / "assets" / "audio" / "sample_audio.wav"
        if path == sample_path.resolve():
            raise UserFacingError(
                f"Input audio file does not exist: `{path}`. "
                "Run `python scripts/fetch_demo_assets.py` to create the demo asset, "
                "or place a readable clip at that exact path."
            )
        raise UserFacingError(f"Input audio file does not exist: `{path}`.")

    return normalize_audio_input(
        path,
        target_sample_rate=TARGET_SAMPLE_RATE,
        max_audio_seconds=MAX_AUDIO_SECONDS,
    )


def serialize_audio_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_path": record["source_path"],
        "resolved_path": record["resolved_path"],
        "format": record["format"],
        "normalized_format": record["normalized_format"],
        "loader": record["loader"],
        "source_sample_rate_hz": record["source_sample_rate_hz"],
        "sample_rate_hz": record["sample_rate_hz"],
        "source_channels": record["source_channels"],
        "channels": record["channels"],
        "duration_seconds": record["duration_seconds"],
        "frame_count": record["frame_count"],
        "signal_summary": dict(record.get("signal_summary") or {}),
    }


def build_messages(system_prompt: str, user_prompt: str, audio: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def sample_audio_path() -> Path:
    return (repo_root() / SAMPLE_AUDIO_RELATIVE_PATH).resolve()


def is_sample_audio_record(record: dict[str, Any]) -> bool:
    return Path(str(record["source_path"])).resolve() == sample_audio_path()


def normalize_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def token_overlap_ratio(actual_text: str, expected_text: str) -> float:
    actual_tokens = normalize_tokens(actual_text)
    expected_tokens = normalize_tokens(expected_text)
    if not expected_tokens:
        return 0.0

    remaining = list(actual_tokens)
    matched = 0
    for token in expected_tokens:
        if token in remaining:
            matched += 1
            remaining.remove(token)
    return matched / float(len(expected_tokens))


def is_unclear_audio_response(text: str | None) -> bool:
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized:
        return True
    return normalized in {
        UNCLEAR_AUDIO_RESPONSE,
        "[unclear]",
        "unclear audio",
        "audio unclear",
    } or "unclear" in normalized or "聞き取" in normalized


def build_quality_check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "name": name,
        "pass": bool(passed),
        "detail": detail,
    }


def target_language_has_expected_script(target_language: str, text: str) -> bool | None:
    normalized = target_language.strip().lower()
    if normalized == "japanese":
        return bool(JAPANESE_PATTERN.search(text or ""))
    return None


def target_language_script_character_count(target_language: str, text: str) -> int | None:
    normalized = target_language.strip().lower()
    if normalized == "japanese":
        return len(JAPANESE_PATTERN.findall(text or ""))
    return None


def translation_source_token_leakage_count(
    translated_text: str,
    transcript_text: str,
    *,
    allowed_tokens: set[str] | None = None,
) -> int:
    allowed = set(allowed_tokens or set())
    source_tokens = {
        token
        for token in normalize_tokens(transcript_text)
        if token not in allowed
    }
    translated_tokens = set(normalize_tokens(translated_text))
    return len(source_tokens & translated_tokens)


def assess_audio_output(
    *,
    mode: str,
    record: dict[str, Any],
    output_text: str,
    target_language: str,
    pipeline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    signal_summary = dict(record.get("signal_summary") or {})
    active_seconds = float(signal_summary.get("active_seconds", 0.0) or 0.0)
    active_ratio = float(signal_summary.get("active_frame_ratio", 0.0) or 0.0)
    speech_like = active_seconds >= MIN_CLEAR_AUDIO_ACTIVE_SECONDS or active_ratio >= MIN_CLEAR_AUDIO_ACTIVE_RATIO

    checks = [
        build_quality_check(
            "nonempty_output",
            bool((output_text or "").strip()),
            "The user-facing output must not be empty.",
        ),
        build_quality_check(
            "speech_activity_detected",
            speech_like,
            (
                f"Signal summary reported active_seconds={active_seconds:.3f} "
                f"and active_frame_ratio={active_ratio:.3f}."
            ),
        ),
    ]
    notes: list[str] = []
    quality_status = "pass"

    unclear_output = is_unclear_audio_response(output_text)
    if not speech_like:
        checks.append(
            build_quality_check(
                "unclear_fallback_used",
                unclear_output,
                "When the clip does not show enough speech activity, the output should be an explicit unclear fallback.",
            )
        )
        if not unclear_output:
            quality_status = "fail"
            notes.append("The clip looked too quiet for a trustworthy transcript, but the output was not an unclear fallback.")

    if is_sample_audio_record(record):
        transcript_candidate = output_text
        if pipeline and pipeline.get("intermediate_transcript"):
            transcript_candidate = str(pipeline["intermediate_transcript"])
        transcript_overlap = token_overlap_ratio(transcript_candidate, SAMPLE_AUDIO_TEXT)
        checks.append(
            build_quality_check(
                "sample_transcript_alignment",
                transcript_overlap >= 0.85,
                (
                    "The local sample clip has a known spoken transcript and should align at >= 0.85 token overlap. "
                    f"Observed overlap={transcript_overlap:.3f}."
                ),
            )
        )
        if transcript_overlap < 0.85:
            quality_status = "fail"
            notes.append("The sample clip transcript did not align closely enough with the known local fixture text.")

        if mode == "translate":
            translation_has_expected_script = target_language_has_expected_script(target_language, output_text)
            if translation_has_expected_script is not None:
                script_char_count = target_language_script_character_count(target_language, output_text)
                checks.append(
                    build_quality_check(
                        "target_language_script_present",
                        translation_has_expected_script,
                        f"The translation output should visibly look like {target_language}.",
                    )
                )
                checks.append(
                    build_quality_check(
                        "target_language_script_substantial",
                        (script_char_count or 0) >= MIN_JAPANESE_TRANSLATION_CHAR_COUNT,
                        (
                            f"The translation output should contain at least {MIN_JAPANESE_TRANSLATION_CHAR_COUNT} "
                            f"{target_language} characters for this local sample fixture."
                        ),
                    )
                )
                if not translation_has_expected_script or (script_char_count or 0) < MIN_JAPANESE_TRANSLATION_CHAR_COUNT:
                    quality_status = "fail"
                    notes.append(f"The translation output did not contain enough visible {target_language} script.")

            source_token_leakage = translation_source_token_leakage_count(
                output_text,
                transcript_candidate,
                allowed_tokens=ALLOWED_TRANSLATION_SOURCE_TOKENS,
            )
            checks.append(
                build_quality_check(
                    "source_language_tokens_removed",
                    source_token_leakage <= 1,
                    (
                        "The translated sample output should not keep more than one untranslated source-language token "
                        f"other than allowed proper nouns. Observed leakage={source_token_leakage}."
                    ),
                )
            )
            if source_token_leakage > 1:
                quality_status = "fail"
                notes.append("The translation output still leaked too many source-language tokens from the sample transcript.")

    if speech_like and unclear_output and mode in {"transcribe", "translate"} and not is_sample_audio_record(record):
        if quality_status == "pass":
            quality_status = "warning"
        notes.append("The clip had speech-like activity, but the model still returned an unclear fallback.")

    claim_scope = (
        "native audio transcription on a local clip"
        if mode == "transcribe"
        else "audio transcription followed by text translation"
        if mode == "translate"
        else "native audio summarization on a local clip"
    )
    pass_definition = (
        "Pass means execution completed, the signal looked speech-bearing or produced an explicit unclear fallback, "
        "and the known sample clip transcript stayed aligned with the local fixture when that sample was used."
        if mode in {"transcribe", "translate"}
        else "Pass means execution completed and the model returned a conservative summary or an explicit unclear fallback."
    )
    validation_mode = "pipeline" if mode == "translate" else "live"
    if quality_status == "warning":
        notes.append("This result stayed conservative, but it is not strong enough to count as a clean validation pass.")

    return {
        "validation_mode": validation_mode,
        "claim_scope": claim_scope,
        "pass_definition": pass_definition,
        "execution_status": "ok",
        "quality_status": quality_status,
        "quality_checks": checks,
        "quality_notes": notes,
    }


def run_audio_generation(
    *,
    session: Any,
    record: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    generation_settings: dict[str, Any],
    cancellation_signal: CancellationSignal | None = None,
) -> tuple[str, float]:
    generation = generate_text_from_messages(
        session=session,
        messages=build_messages(system_prompt=system_prompt, user_prompt=user_prompt, audio=record["audio"]),
        generation_settings=generation_settings,
        template_kwargs={
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "processor_kwargs": {"sampling_rate": record["sample_rate_hz"]},
        },
        cancellation_signal=cancellation_signal,
    )
    output_text = generation.output_text.strip()
    if not output_text:
        raise UserFacingError("Model returned an empty audio response.")
    return output_text, generation.elapsed_seconds


def translate_transcript(
    *,
    session: Any,
    transcript_text: str,
    target_language: str,
    cancellation_signal: CancellationSignal | None = None,
) -> tuple[str, float]:
    generation = generate_text_from_messages(
        session=session,
        messages=build_text_messages(
            system_prompt=(
                "You are a careful translator. Translate only the provided transcript. "
                f"Return only the {target_language} translation. "
                f"If the transcript is unclear or marked {UNCLEAR_AUDIO_RESPONSE}, return exactly {UNCLEAR_AUDIO_RESPONSE}."
            ),
            user_prompt=transcript_text,
        ),
        generation_settings={"max_new_tokens": 192, "do_sample": False},
        cancellation_signal=cancellation_signal,
    )
    output_text = generation.output_text.strip()
    if not output_text:
        raise UserFacingError("Model returned an empty text translation.")
    return output_text, generation.elapsed_seconds


def run_audio_mode(
    *,
    mode: str,
    input_path: Path,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    prompt: str | None = None,
    system_prompt: str | None = None,
    base_model_id: str | None = None,
    audio_model_id: str | None = None,
    session_manager: SessionManager | None = None,
    cancellation_signal: CancellationSignal | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise UserFacingError(f"Unsupported audio mode `{mode}`.")

    resolved_base_model_id = (base_model_id or resolve_model_id()).strip()
    if not resolved_base_model_id:
        resolved_base_model_id = resolve_model_id()
    model_id, model_source = resolve_audio_model_selection(
        base_model_id=resolved_base_model_id,
        audio_model_id=audio_model_id,
    )
    resolved_prompt = resolve_prompt(mode, prompt, target_language)
    resolved_system_prompt = resolve_system_prompt(mode, system_prompt)
    generation_settings = dict(GENERATION_SETTINGS[mode])
    record = resolve_audio_input(input_path)

    owns_session_manager = session_manager is None
    manager = session_manager or SessionManager()
    device_info: dict[str, str] | str | None = "unresolved"
    output_text: str
    elapsed_seconds: float
    pipeline: dict[str, Any] | None = None

    try:
        audio_session = manager.get_session("audio", model_id)
        device_info = dict(audio_session.device_info)

        if mode == "translate":
            transcript_text, transcript_elapsed = run_audio_generation(
                session=audio_session,
                record=record,
                system_prompt=DEFAULT_SYSTEM_PROMPTS["transcribe"],
                user_prompt=resolve_prompt("transcribe", None, target_language),
                generation_settings=GENERATION_SETTINGS["transcribe"],
                cancellation_signal=cancellation_signal,
            )
            if is_unclear_audio_response(transcript_text):
                output_text = UNCLEAR_AUDIO_RESPONSE
                translation_elapsed = 0.0
            else:
                text_session = manager.get_session("text", resolved_base_model_id)
                translated_text, translation_elapsed = translate_transcript(
                    session=text_session,
                    transcript_text=transcript_text,
                    target_language=target_language,
                    cancellation_signal=cancellation_signal,
                )
                output_text = translated_text

            elapsed_seconds = transcript_elapsed + translation_elapsed
            pipeline = {
                "strategy": "audio_transcript_then_text_translate",
                "intermediate_transcript": transcript_text,
                "transcript_generation_settings": dict(GENERATION_SETTINGS["transcribe"]),
                "translation_model_id": resolved_base_model_id,
                "translation_generation_settings": {"max_new_tokens": 192, "do_sample": False},
            }
        else:
            output_text, elapsed_seconds = run_audio_generation(
                session=audio_session,
                record=record,
                system_prompt=resolved_system_prompt,
                user_prompt=resolved_prompt,
                generation_settings=generation_settings,
                cancellation_signal=cancellation_signal,
            )
    finally:
        if owns_session_manager:
            manager.close_all()

    validation = assess_audio_output(
        mode=mode,
        record=record,
        output_text=output_text,
        target_language=target_language,
        pipeline=pipeline,
    )

    return {
        "base_model_id": resolved_base_model_id,
        "model_id": model_id,
        "model_id_source": model_source,
        "device_info": device_info,
        "mode": mode,
        "prompt": resolved_prompt,
        "system_prompt": resolved_system_prompt,
        "generation_settings": generation_settings,
        "target_language": target_language if mode == "translate" else None,
        "record": record,
        "elapsed_seconds": elapsed_seconds,
        "output_text": output_text,
        "validation": validation,
        "pipeline": pipeline,
    }
