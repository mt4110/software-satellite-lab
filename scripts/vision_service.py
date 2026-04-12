from __future__ import annotations

from pathlib import Path
from typing import Any

from artifact_schema import collect_asset_lineage
from asset_preprocessing import normalize_image_input, rasterize_pdf_input
from gemma_core import CancellationSignal, SessionManager, generate_text_from_messages
from gemma_runtime import UserFacingError, import_image_module, import_pdf_renderer, repo_root, resolve_model_id, timestamp_slug


MODES = ("caption", "vqa", "ocr", "compare", "pdf-summary")
IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
}

DEFAULT_PROMPTS = {
    "caption": "Describe the image in 3 concise bullet points. Mention visible text if it matters.",
    "vqa": "What is the most important information visible in this image?",
    "ocr": "Extract the visible text from this image. Return only the extracted text.",
    "compare": "Compare these two images and explain the most important differences in 3 concise bullet points.",
    "pdf-summary": (
        "Summarize this document in 3 concise bullet points. Then list exactly 3 key facts under the heading "
        "'Key facts:'."
    ),
}

DEFAULT_SYSTEM_PROMPTS = {
    "caption": "You are a careful visual assistant. Describe only what is visible.",
    "vqa": "You answer questions about images directly and clearly. If something is uncertain, say so.",
    "ocr": "You extract visible text from images carefully and preserve the original wording.",
    "compare": "You compare images precisely and focus on meaningful differences.",
    "pdf-summary": "You summarize short documents clearly and keep the important facts explicit.",
}

GENERATION_SETTINGS = {
    "caption": {"max_new_tokens": 192, "do_sample": False},
    "vqa": {"max_new_tokens": 160, "do_sample": False},
    "ocr": {"max_new_tokens": 256, "do_sample": False},
    "compare": {"max_new_tokens": 220, "do_sample": False},
    "pdf-summary": {"max_new_tokens": 256, "do_sample": False},
}


def artifacts_root() -> Path:
    return repo_root() / "artifacts" / "vision"


def default_output_path(mode: str) -> Path:
    return artifacts_root() / f"{timestamp_slug()}-{mode}.json"


def resolve_prompt(mode: str, prompt: str | None) -> str:
    return (prompt or DEFAULT_PROMPTS[mode]).strip()


def resolve_system_prompt(mode: str, system_prompt: str | None) -> str:
    return (system_prompt or DEFAULT_SYSTEM_PROMPTS[mode]).strip()


def serialize_input_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized = []
    for record in records:
        serialized.append(
            {
                "source_path": record["source_path"],
                "resolved_path": record["resolved_path"],
                "kind": record["kind"],
                "page_number": record["page_number"],
                "frame_index": record.get("frame_index"),
                "timestamp_seconds": record.get("timestamp_seconds"),
                "width": record["width"],
                "height": record["height"],
            }
        )
    return serialized


def resolve_inputs(
    input_paths: list[Path],
    image_module: Any,
    max_pages: int,
) -> tuple[list[dict[str, Any]], int]:
    pdfium = None
    records: list[dict[str, Any]] = []
    pdf_inputs = 0

    for raw_path in input_paths:
        path = raw_path.expanduser()
        if not path.is_absolute():
            path = repo_root() / path
        path = path.resolve()

        if not path.exists():
            raise UserFacingError(f"Input file does not exist: `{path}`.")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            if pdfium is None:
                pdfium = import_pdf_renderer()
            pdf_inputs += 1
            records.extend(
                rasterize_pdf_input(
                    path,
                    image_module=image_module,
                    pdfium=pdfium,
                    max_pages=max_pages,
                )
            )
            continue

        if suffix == ".svg":
            raise UserFacingError(
                f"SVG input is not accepted directly by this Phase 2 runner: `{path}`. "
                "Use a raster image such as PNG, or regenerate the demo assets with `scripts/fetch_demo_assets.py`."
            )

        if suffix not in IMAGE_SUFFIXES:
            raise UserFacingError(
                f"Unsupported input type `{suffix or 'no extension'}` for `{path}`. "
                "Use a local image file or PDF."
            )

        records.append(normalize_image_input(path, image_module=image_module))

    if not records:
        raise UserFacingError("No usable images were resolved from the provided inputs.")

    return records, pdf_inputs


def validate_mode_inputs(mode: str, records: list[dict[str, Any]], pdf_inputs: int) -> None:
    if mode == "compare" and len(records) != 2:
        raise UserFacingError("`compare` mode requires exactly 2 resolved images.")

    if mode == "pdf-summary" and pdf_inputs != 1:
        raise UserFacingError("`pdf-summary` mode requires exactly 1 PDF input.")


def build_user_prompt(mode: str, prompt: str, image_count: int) -> str:
    if mode == "caption":
        if image_count == 1:
            return prompt
        return f"You will receive {image_count} images in order. {prompt}"

    if mode == "vqa":
        if image_count == 1:
            return f"Answer this question about the image:\n{prompt}"
        return f"Answer this question using all {image_count} images:\n{prompt}"

    if mode == "ocr":
        if image_count == 1:
            return prompt
        return (
            f"You will receive {image_count} images. Extract the visible text from each image in order. "
            "Label each section with the image number.\n\n"
            + prompt
        )

    if mode == "compare":
        return prompt

    return (
        f"You will receive {image_count} rasterized PDF page images in reading order. "
        "Use only the document content that is visible on these pages.\n\n"
        + prompt
    )


def build_messages(system_prompt: str, user_prompt: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    user_content = [{"type": "image", "image": record["image"]} for record in records]
    user_content.append({"type": "text", "text": user_prompt})
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def run_vision_mode(
    *,
    mode: str,
    inputs: list[Path | str],
    prompt: str | None = None,
    system_prompt: str | None = None,
    max_pages: int = 4,
    model_id: str | None = None,
    session_manager: SessionManager | None = None,
    cancellation_signal: CancellationSignal | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise UserFacingError(f"Unsupported vision mode `{mode}`.")

    resolved_model_id = (model_id or resolve_model_id()).strip() or resolve_model_id()
    resolved_prompt = resolve_prompt(mode, prompt)
    resolved_system_prompt = resolve_system_prompt(mode, system_prompt)
    generation_settings = dict(GENERATION_SETTINGS[mode])
    image_module = import_image_module()
    input_paths = [item if isinstance(item, Path) else Path(item) for item in inputs]
    records, pdf_inputs = resolve_inputs(
        input_paths=input_paths,
        image_module=image_module,
        max_pages=max_pages,
    )
    validate_mode_inputs(mode, records, pdf_inputs)
    user_prompt = build_user_prompt(mode, resolved_prompt, image_count=len(records))

    owns_session_manager = session_manager is None
    manager = session_manager or SessionManager()

    try:
        session = manager.get_session("vision", resolved_model_id)
        device_info = dict(session.device_info)
        generation = generate_text_from_messages(
            session=session,
            messages=build_messages(system_prompt=resolved_system_prompt, user_prompt=user_prompt, records=records),
            generation_settings=generation_settings,
            template_kwargs={
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
            },
            cancellation_signal=cancellation_signal,
        )
        output_text = generation.output_text.strip()
        if not output_text:
            raise UserFacingError("Model returned an empty vision response.")
    finally:
        if owns_session_manager:
            manager.close_all()

    return {
        "mode": mode,
        "model_id": resolved_model_id,
        "prompt": resolved_prompt,
        "system_prompt": resolved_system_prompt,
        "resolved_user_prompt": user_prompt,
        "generation_settings": generation_settings,
        "records": records,
        "asset_lineage": collect_asset_lineage(records),
        "device_info": device_info,
        "elapsed_seconds": generation.elapsed_seconds,
        "output_text": output_text,
    }
