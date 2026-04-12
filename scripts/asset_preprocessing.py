#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gemma_runtime import UserFacingError, repo_root


PREPROCESSING_CACHE_VERSION = 1
DEFAULT_PDF_RENDER_SCALE = 2.0
DEFAULT_VIDEO_FRAME_OFFSETS = (0.0, 0.4, 0.8)
AUDIO_ACTIVITY_FRAME_SECONDS = 0.02
MIN_AUDIO_ACTIVITY_ABS_THRESHOLD = 0.015
SUPPORTED_AUDIO_SUFFIXES = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".wav",
    ".webm",
}


def preprocessing_cache_root(root: Path | None = None) -> Path:
    base_root = root or repo_root()
    return base_root / "artifacts" / "cache" / f"preprocessed-v{PREPROCESSING_CACHE_VERSION}"


def resolve_local_path(raw_path: Path, root: Path | None = None) -> Path:
    path = raw_path.expanduser()
    if not path.is_absolute():
        path = (root or repo_root()) / path
    return path.resolve()


def _stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_preprocessing_cache_key(
    source_path: Path,
    *,
    transform: str,
    config: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"v{PREPROCESSING_CACHE_VERSION}:{transform}:".encode("utf-8"))
    digest.update(str(source_path).encode("utf-8"))
    digest.update(_hash_file(source_path).encode("utf-8"))
    digest.update(_stable_json(config).encode("utf-8"))
    return digest.hexdigest()[:24]


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _build_lineage(
    *,
    source_path: Path,
    resolved_path: Path,
    asset_kind: str,
    transform: str,
    cache_key: str,
    cache_hit: bool,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source_path": str(source_path),
        "resolved_path": str(resolved_path),
        "cache_path": str(resolved_path),
        "asset_kind": asset_kind,
        "transform": transform,
        "cache_key": cache_key,
        "cache_hit": cache_hit,
        "metadata": metadata or {},
    }


def _import_image_ops() -> Any:
    try:
        from PIL import ImageOps  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency dependent
        raise UserFacingError(
            "Missing runtime dependency. Install the local venv requirements first with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return ImageOps


def _open_rgb_copy(image_module: Any, path: Path) -> Any:
    try:
        with image_module.open(path) as image:
            return image.convert("RGB").copy()
    except Exception as exc:
        raise UserFacingError(f"Failed to open image `{path}` ({type(exc).__name__}: {exc}).") from exc


def normalize_image_input(
    raw_path: Path,
    *,
    image_module: Any,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    source_path = resolve_local_path(raw_path)
    if not source_path.exists():
        raise UserFacingError(f"Input file does not exist: `{source_path}`.")

    ImageOps = _import_image_ops()
    config = {
        "mode": "RGB",
        "operation": "image_normalization",
    }
    cache_key = build_preprocessing_cache_key(
        source_path,
        transform="image_normalization",
        config=config,
    )
    cache_dir = (cache_root or preprocessing_cache_root()) / "images" / cache_key
    cache_path = cache_dir / "normalized.png"
    metadata_path = cache_dir / "metadata.json"
    cache_hit = cache_path.exists() and metadata_path.exists()

    if not cache_hit:
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with image_module.open(source_path) as image:
                normalized = ImageOps.exif_transpose(image).convert("RGB")
                width = normalized.width
                height = normalized.height
                normalized.save(cache_path, format="PNG", optimize=False, compress_level=9)
        except Exception as exc:
            raise UserFacingError(
                f"Failed to normalize image `{source_path}` ({type(exc).__name__}: {exc})."
            ) from exc
        _write_metadata(
            metadata_path,
            {
                "source_path": str(source_path),
                "resolved_path": str(cache_path),
                "width": width,
                "height": height,
                "config": config,
            },
        )

    metadata = _read_metadata(metadata_path) or {}
    image = _open_rgb_copy(image_module, cache_path)
    width = int(metadata.get("width", image.width))
    height = int(metadata.get("height", image.height))
    return {
        "source_path": str(source_path),
        "resolved_path": str(cache_path),
        "kind": "image",
        "page_number": None,
        "frame_index": None,
        "timestamp_seconds": None,
        "width": width,
        "height": height,
        "image": image,
        "lineage": _build_lineage(
            source_path=source_path,
            resolved_path=cache_path,
            asset_kind="image",
            transform="image_normalization",
            cache_key=cache_key,
            cache_hit=cache_hit,
            metadata={
                "width": width,
                "height": height,
            },
        ),
    }


def rasterize_pdf_input(
    raw_path: Path,
    *,
    image_module: Any,
    pdfium: Any,
    max_pages: int,
    cache_root: Path | None = None,
    scale: float = DEFAULT_PDF_RENDER_SCALE,
) -> list[dict[str, Any]]:
    if max_pages < 1:
        raise UserFacingError("`--max-pages` must be at least 1.")

    source_path = resolve_local_path(raw_path)
    if not source_path.exists():
        raise UserFacingError(f"Input file does not exist: `{source_path}`.")

    config = {
        "max_pages": max_pages,
        "scale": scale,
    }
    cache_key = build_preprocessing_cache_key(
        source_path,
        transform="pdf_rasterization",
        config=config,
    )
    cache_dir = (cache_root or preprocessing_cache_root()) / "pdf" / cache_key
    metadata_path = cache_dir / "metadata.json"

    metadata = _read_metadata(metadata_path)
    page_entries: list[dict[str, Any]] = []
    cache_hit = False
    if metadata:
        candidate_pages = metadata.get("pages")
        if isinstance(candidate_pages, list) and all(
            isinstance(item, dict) and Path(str(item.get("resolved_path", ""))).exists()
            for item in candidate_pages
        ):
            page_entries = candidate_pages
            cache_hit = True

    if not cache_hit:
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            document = pdfium.PdfDocument(str(source_path))
        except Exception as exc:
            raise UserFacingError(f"Failed to open PDF `{source_path}` ({type(exc).__name__}: {exc}).") from exc

        try:
            page_count = len(document)
            if page_count == 0:
                raise UserFacingError(f"PDF `{source_path}` contains no pages.")

            page_entries = []
            for index in range(min(page_count, max_pages)):
                page = document[index]
                bitmap = None
                try:
                    bitmap = page.render(scale=scale)
                    rendered = bitmap.to_pil().convert("RGB")
                finally:
                    if bitmap is not None:
                        bitmap.close()
                    page.close()

                rendered_path = cache_dir / f"page-{index + 1:02d}.png"
                rendered.save(rendered_path, format="PNG", optimize=False, compress_level=9)
                page_entries.append(
                    {
                        "page_number": index + 1,
                        "resolved_path": str(rendered_path),
                        "width": rendered.width,
                        "height": rendered.height,
                    }
                )
        finally:
            document.close()

        _write_metadata(
            metadata_path,
            {
                "source_path": str(source_path),
                "config": config,
                "pages": page_entries,
            },
        )

    records: list[dict[str, Any]] = []
    for page_entry in page_entries:
        rendered_path = Path(str(page_entry["resolved_path"]))
        image = _open_rgb_copy(image_module, rendered_path)
        records.append(
            {
                "source_path": str(source_path),
                "resolved_path": str(rendered_path),
                "kind": "pdf-page",
                "page_number": int(page_entry["page_number"]),
                "frame_index": None,
                "timestamp_seconds": None,
                "width": int(page_entry["width"]),
                "height": int(page_entry["height"]),
                "image": image,
                "lineage": _build_lineage(
                    source_path=source_path,
                    resolved_path=rendered_path,
                    asset_kind="pdf_page",
                    transform="pdf_rasterization",
                    cache_key=cache_key,
                    cache_hit=cache_hit,
                    metadata={
                        "page_number": int(page_entry["page_number"]),
                        "width": int(page_entry["width"]),
                        "height": int(page_entry["height"]),
                        "scale": scale,
                    },
                ),
            }
        )
    return records


def decode_pcm_samples(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (samples - 128.0) / 128.0

    if sample_width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        return samples / 32768.0

    if sample_width == 3:
        payload = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        extended = np.zeros((payload.shape[0], 4), dtype=np.uint8)
        extended[:, :3] = payload
        extended[:, 3] = np.where(payload[:, 2] & 0x80, 0xFF, 0x00)
        samples = extended.view("<i4").reshape(-1).astype(np.float32)
        return samples / 8388608.0

    if sample_width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32)
        return samples / 2147483648.0

    raise UserFacingError(f"Unsupported WAV sample width {sample_width} bytes.")


def convert_to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    if channels == 1:
        return samples.astype(np.float32, copy=False)

    try:
        matrix = samples.reshape(-1, channels)
    except ValueError as exc:
        raise UserFacingError("Decoded WAV audio had an unexpected channel layout.") from exc

    return matrix.mean(axis=1, dtype=np.float32)


def resample_audio(samples: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if source_sample_rate == target_sample_rate or len(samples) == 0:
        return samples.astype(np.float32, copy=False)

    target_length = max(
        1,
        int(round(len(samples) * float(target_sample_rate) / float(source_sample_rate))),
    )
    source_positions = np.linspace(0.0, len(samples) - 1, num=len(samples), dtype=np.float32)
    target_positions = np.linspace(0.0, len(samples) - 1, num=target_length, dtype=np.float32)
    resampled = np.interp(target_positions, source_positions, samples.astype(np.float32))
    return resampled.astype(np.float32, copy=False)


def summarize_audio_signal(samples: np.ndarray, sample_rate_hz: int) -> dict[str, Any]:
    normalized = samples.astype(np.float32, copy=False)
    if len(normalized) == 0 or sample_rate_hz <= 0:
        return {
            "peak_abs": 0.0,
            "rms_abs": 0.0,
            "activity_threshold_abs": MIN_AUDIO_ACTIVITY_ABS_THRESHOLD,
            "active_frame_ratio": 0.0,
            "active_seconds": 0.0,
            "leading_silence_seconds": 0.0,
            "trailing_silence_seconds": 0.0,
            "clipped_sample_ratio": 0.0,
        }

    frame_length = max(1, int(round(sample_rate_hz * AUDIO_ACTIVITY_FRAME_SECONDS)))
    frame_count = int(np.ceil(len(normalized) / float(frame_length)))
    padded = np.zeros(frame_count * frame_length, dtype=np.float32)
    padded[: len(normalized)] = normalized
    frames = padded.reshape(frame_count, frame_length)

    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1, dtype=np.float32)).astype(np.float32)
    peak_abs = float(np.max(np.abs(normalized)))
    rms_abs = float(np.sqrt(np.mean(np.square(normalized), dtype=np.float32)))
    activity_threshold = max(MIN_AUDIO_ACTIVITY_ABS_THRESHOLD, rms_abs * 0.45, peak_abs * 0.08)
    active_mask = frame_rms >= activity_threshold

    leading_silent_frames = 0
    for is_active in active_mask:
        if is_active:
            break
        leading_silent_frames += 1

    trailing_silent_frames = 0
    for is_active in active_mask[::-1]:
        if is_active:
            break
        trailing_silent_frames += 1

    active_frame_ratio = float(active_mask.mean(dtype=np.float32))
    active_seconds = active_frame_ratio * (frame_count * frame_length) / float(sample_rate_hz)
    return {
        "peak_abs": round(peak_abs, 6),
        "rms_abs": round(rms_abs, 6),
        "activity_threshold_abs": round(float(activity_threshold), 6),
        "active_frame_ratio": round(active_frame_ratio, 6),
        "active_seconds": round(active_seconds, 3),
        "leading_silence_seconds": round(leading_silent_frames * frame_length / float(sample_rate_hz), 3),
        "trailing_silence_seconds": round(trailing_silent_frames * frame_length / float(sample_rate_hz), 3),
        "clipped_sample_ratio": round(float((np.abs(normalized) >= 0.999).mean(dtype=np.float32)), 6),
    }


def _read_wav_source(path: Path, target_sample_rate: int) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            source_sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            source_channels = wav_file.getnchannels()
            frame_count = wav_file.getnframes()
            raw = wav_file.readframes(frame_count)
    except (wave.Error, OSError) as exc:
        raise UserFacingError(
            f"Failed to read WAV audio `{path}` ({type(exc).__name__}: {exc})."
        ) from exc

    if sample_width not in (1, 2, 3, 4):
        raise UserFacingError(f"Unsupported WAV sample width {sample_width} bytes in `{path}`.")

    samples = decode_pcm_samples(raw, sample_width)
    samples = convert_to_mono(samples, source_channels)
    samples = resample_audio(samples, source_sample_rate, target_sample_rate)
    return samples, {
        "loader": "wave",
        "source_sample_rate_hz": source_sample_rate,
        "source_channels": source_channels,
    }


def _read_audio_with_ffmpeg(path: Path, target_sample_rate: int) -> tuple[np.ndarray, dict[str, Any]]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise UserFacingError(
            f"Unsupported audio input `{path.suffix.lower() or 'no extension'}` for `{path}` without `ffmpeg`. "
            "Use a local WAV file or make sure `ffmpeg` is available."
        )

    completed = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(target_sample_rate),
            "pipe:1",
        ],
        capture_output=True,
        check=False,
        timeout=45,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip() or "unknown ffmpeg error"
        raise UserFacingError(f"Failed to decode audio `{path}` ({detail}).")

    samples = np.frombuffer(completed.stdout, dtype="<i2").astype(np.float32) / 32768.0
    return samples, {
        "loader": "ffmpeg",
        "source_sample_rate_hz": None,
        "source_channels": None,
    }


def _write_normalized_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
    pcm = (clipped * 32767.0).round().astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _read_cached_normalized_wav(path: Path, expected_sample_rate: int) -> np.ndarray:
    try:
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            frame_count = wav_file.getnframes()
            raw = wav_file.readframes(frame_count)
    except (wave.Error, OSError) as exc:
        raise UserFacingError(
            f"Failed to read normalized audio cache `{path}` ({type(exc).__name__}: {exc})."
        ) from exc

    if sample_rate != expected_sample_rate:
        raise UserFacingError(
            f"Normalized audio cache `{path}` used sample rate {sample_rate}, expected {expected_sample_rate}."
        )
    if channels != 1 or sample_width != 2:
        raise UserFacingError(f"Normalized audio cache `{path}` is not mono 16-bit PCM.")
    return decode_pcm_samples(raw, sample_width)


def normalize_audio_input(
    raw_path: Path,
    *,
    target_sample_rate: int,
    max_audio_seconds: float,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    source_path = resolve_local_path(raw_path)
    if not source_path.exists():
        raise UserFacingError(f"Input audio file does not exist: `{source_path}`.")

    suffix = source_path.suffix.lower()
    if suffix not in SUPPORTED_AUDIO_SUFFIXES:
        raise UserFacingError(
            f"Unsupported audio input `{suffix or 'no extension'}` for `{source_path}`. "
            "Use a local WAV file or another format that `ffmpeg` can decode."
        )

    config = {
        "target_sample_rate_hz": target_sample_rate,
        "normalized_channels": 1,
        "normalized_format": ".wav",
    }
    cache_key = build_preprocessing_cache_key(
        source_path,
        transform="audio_normalization",
        config=config,
    )
    cache_dir = (cache_root or preprocessing_cache_root()) / "audio" / cache_key
    cache_path = cache_dir / "normalized.wav"
    metadata_path = cache_dir / "metadata.json"

    metadata = _read_metadata(metadata_path)
    cache_hit = bool(metadata and cache_path.exists())

    if not cache_hit:
        if suffix == ".wav":
            samples, source_metadata = _read_wav_source(source_path, target_sample_rate)
        else:
            samples, source_metadata = _read_audio_with_ffmpeg(source_path, target_sample_rate)

        if len(samples) == 0:
            raise UserFacingError(f"Decoded audio was empty: `{source_path}`.")

        duration_seconds = len(samples) / float(target_sample_rate)
        if duration_seconds > max_audio_seconds:
            raise UserFacingError(
                f"Audio input `{source_path}` is {duration_seconds:.3f} seconds long. "
                f"Gemma 4 audio input for this lab is capped at {max_audio_seconds:.0f} seconds."
            )
        signal_summary = summarize_audio_signal(samples, target_sample_rate)

        cache_dir.mkdir(parents=True, exist_ok=True)
        _write_normalized_wav(cache_path, samples, target_sample_rate)
        metadata = {
            "source_path": str(source_path),
            "resolved_path": str(cache_path),
            "format": suffix or ".wav",
            "normalized_format": ".wav",
            "sample_rate_hz": target_sample_rate,
            "channels": 1,
            "frame_count": int(len(samples)),
            "duration_seconds": round(duration_seconds, 3),
            "signal_summary": signal_summary,
        }
        metadata.update(source_metadata)
        _write_metadata(metadata_path, metadata)
    else:
        samples = _read_cached_normalized_wav(cache_path, target_sample_rate)

    if metadata is None:
        raise UserFacingError(f"Normalized audio metadata was missing for `{source_path}`.")

    return {
        "source_path": str(source_path),
        "resolved_path": str(cache_path),
        "format": metadata["format"],
        "normalized_format": metadata["normalized_format"],
        "loader": metadata["loader"],
        "source_sample_rate_hz": metadata["source_sample_rate_hz"],
        "sample_rate_hz": metadata["sample_rate_hz"],
        "source_channels": metadata["source_channels"],
        "channels": metadata["channels"],
        "duration_seconds": metadata["duration_seconds"],
        "frame_count": metadata["frame_count"],
        "signal_summary": dict(metadata.get("signal_summary") or summarize_audio_signal(samples, target_sample_rate)),
        "audio": samples,
        "lineage": _build_lineage(
            source_path=source_path,
            resolved_path=cache_path,
            asset_kind="audio",
            transform="audio_normalization",
            cache_key=cache_key,
            cache_hit=cache_hit,
            metadata={
                "loader": metadata["loader"],
                "sample_rate_hz": metadata["sample_rate_hz"],
                "channels": metadata["channels"],
                "duration_seconds": metadata["duration_seconds"],
                "signal_summary": dict(
                    metadata.get("signal_summary") or summarize_audio_signal(samples, target_sample_rate)
                ),
            },
        ),
    }


def sample_video_frames(
    raw_path: Path,
    *,
    image_module: Any,
    cache_root: Path | None = None,
    offsets_seconds: Sequence[float] = DEFAULT_VIDEO_FRAME_OFFSETS,
) -> list[dict[str, Any]]:
    source_path = resolve_local_path(raw_path)
    if not source_path.exists():
        raise UserFacingError(f"Input file does not exist: `{source_path}`.")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise UserFacingError(
            "ffmpeg is not available, so the frame-sampled video understanding proxy cannot extract frames."
        )

    normalized_offsets = tuple(float(offset) for offset in offsets_seconds)
    config = {
        "offsets_seconds": [round(offset, 3) for offset in normalized_offsets],
        "format": "png",
    }
    cache_key = build_preprocessing_cache_key(
        source_path,
        transform="video_frame_sampling",
        config=config,
    )
    cache_dir = (cache_root or preprocessing_cache_root()) / "video" / cache_key
    metadata_path = cache_dir / "metadata.json"

    metadata = _read_metadata(metadata_path)
    frame_entries: list[dict[str, Any]] = []
    cache_hit = False
    if metadata:
        candidate_frames = metadata.get("frames")
        if isinstance(candidate_frames, list) and all(
            isinstance(item, dict) and Path(str(item.get("resolved_path", ""))).exists()
            for item in candidate_frames
        ):
            frame_entries = candidate_frames
            cache_hit = True

    if not cache_hit:
        cache_dir.mkdir(parents=True, exist_ok=True)
        frame_entries = []
        for index, offset in enumerate(normalized_offsets, start=1):
            frame_path = cache_dir / f"frame-{index:02d}.png"
            completed = subprocess.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-nostdin",
                    "-y",
                    "-ss",
                    f"{offset:.3f}",
                    "-i",
                    str(source_path),
                    "-frames:v",
                    "1",
                    str(frame_path),
                ],
                capture_output=True,
                check=False,
                timeout=20,
            )
            if completed.returncode != 0 or not frame_path.exists() or frame_path.stat().st_size <= 0:
                continue

            try:
                with image_module.open(frame_path) as image:
                    normalized = image.convert("RGB")
                    width = normalized.width
                    height = normalized.height
                    normalized.save(frame_path, format="PNG", optimize=False, compress_level=9)
            except Exception as exc:
                raise UserFacingError(
                    f"Failed to normalize extracted frame `{frame_path}` ({type(exc).__name__}: {exc})."
                ) from exc

            frame_entries.append(
                {
                    "frame_index": index,
                    "timestamp_seconds": round(offset, 3),
                    "resolved_path": str(frame_path),
                    "width": width,
                    "height": height,
                }
            )

        if not frame_entries:
            raise UserFacingError(
                f"Failed to extract any frames from `{source_path}` with ffmpeg for the video proxy check."
            )

        _write_metadata(
            metadata_path,
            {
                "source_path": str(source_path),
                "config": config,
                "frames": frame_entries,
            },
        )

    records: list[dict[str, Any]] = []
    for frame_entry in frame_entries:
        frame_path = Path(str(frame_entry["resolved_path"]))
        image = _open_rgb_copy(image_module, frame_path)
        records.append(
            {
                "source_path": str(source_path),
                "resolved_path": str(frame_path),
                "kind": "video-frame",
                "page_number": None,
                "frame_index": int(frame_entry["frame_index"]),
                "timestamp_seconds": float(frame_entry["timestamp_seconds"]),
                "width": int(frame_entry["width"]),
                "height": int(frame_entry["height"]),
                "image": image,
                "lineage": _build_lineage(
                    source_path=source_path,
                    resolved_path=frame_path,
                    asset_kind="video_frame",
                    transform="video_frame_sampling",
                    cache_key=cache_key,
                    cache_hit=cache_hit,
                    metadata={
                        "frame_index": int(frame_entry["frame_index"]),
                        "timestamp_seconds": float(frame_entry["timestamp_seconds"]),
                        "width": int(frame_entry["width"]),
                        "height": int(frame_entry["height"]),
                    },
                ),
            }
        )
    return records
