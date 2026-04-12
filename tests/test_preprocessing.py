from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from asset_preprocessing import (  # noqa: E402
    normalize_audio_input,
    normalize_image_input,
    rasterize_pdf_input,
    sample_video_frames,
)
from gemma_runtime import import_pdf_renderer, repo_root  # noqa: E402


def write_stereo_wav(path: Path, sample_rate: int = 8_000, frame_count: int = 4_000) -> None:
    left = np.linspace(-0.5, 0.5, frame_count, dtype=np.float32)
    right = np.linspace(0.5, -0.5, frame_count, dtype=np.float32)
    stereo = np.stack([left, right], axis=1)
    pcm = (np.clip(stereo, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


class PreprocessingTests(unittest.TestCase):
    def test_normalize_image_input_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source.png"
            cache_root = root / "cache"
            Image.new("RGBA", (18, 12), (120, 80, 40, 255)).save(source_path, format="PNG")

            first = normalize_image_input(source_path, image_module=Image, cache_root=cache_root)
            second = normalize_image_input(source_path, image_module=Image, cache_root=cache_root)

            self.assertEqual(first["resolved_path"], second["resolved_path"])
            self.assertFalse(first["lineage"]["cache_hit"])
            self.assertTrue(second["lineage"]["cache_hit"])
            self.assertEqual(first["width"], 18)
            self.assertEqual(first["height"], 12)

            first["image"].close()
            second["image"].close()

    def test_normalize_audio_input_reuses_cache_and_normalizes_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source.wav"
            cache_root = root / "cache"
            write_stereo_wav(source_path)

            first = normalize_audio_input(
                source_path,
                target_sample_rate=16_000,
                max_audio_seconds=30.0,
                cache_root=cache_root,
            )
            second = normalize_audio_input(
                source_path,
                target_sample_rate=16_000,
                max_audio_seconds=30.0,
                cache_root=cache_root,
            )

            self.assertEqual(first["resolved_path"], second["resolved_path"])
            self.assertEqual(first["sample_rate_hz"], 16_000)
            self.assertEqual(first["channels"], 1)
            self.assertEqual(first["source_channels"], 2)
            self.assertGreater(first["frame_count"], 0)
            self.assertIn("signal_summary", first)
            self.assertIn("active_seconds", first["signal_summary"])
            self.assertFalse(first["lineage"]["cache_hit"])
            self.assertTrue(second["lineage"]["cache_hit"])

    def test_rasterize_pdf_input_reuses_cache(self) -> None:
        pdf_path = repo_root() / "assets" / "docs" / "sample.pdf"
        pdfium = import_pdf_renderer()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "cache"
            first = rasterize_pdf_input(
                pdf_path,
                image_module=Image,
                pdfium=pdfium,
                max_pages=1,
                cache_root=cache_root,
            )
            second = rasterize_pdf_input(
                pdf_path,
                image_module=Image,
                pdfium=pdfium,
                max_pages=1,
                cache_root=cache_root,
            )

            self.assertEqual(len(first), 1)
            self.assertEqual(first[0]["resolved_path"], second[0]["resolved_path"])
            self.assertFalse(first[0]["lineage"]["cache_hit"])
            self.assertTrue(second[0]["lineage"]["cache_hit"])

            first[0]["image"].close()
            second[0]["image"].close()

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for video frame sampling")
    def test_sample_video_frames_reuses_cache(self) -> None:
        video_path = repo_root() / "assets" / "video" / "sample_video.mp4"

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "cache"
            first = sample_video_frames(video_path, image_module=Image, cache_root=cache_root)
            second = sample_video_frames(video_path, image_module=Image, cache_root=cache_root)

            self.assertGreaterEqual(len(first), 1)
            self.assertEqual(first[0]["resolved_path"], second[0]["resolved_path"])
            self.assertFalse(first[0]["lineage"]["cache_hit"])
            self.assertTrue(second[0]["lineage"]["cache_hit"])

            for record in first + second:
                record["image"].close()


if __name__ == "__main__":
    unittest.main()
