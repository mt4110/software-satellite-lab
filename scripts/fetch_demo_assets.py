#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import wave
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_DIRS = {
    "image": "image",
    "text_image": "text_image",
    "chart_ui_screenshot": "chart_ui_screenshot",
    "audio": "audio",
    "video": "video",
    "pdf": "pdf",
    "images": "images",
    "docs": "docs",
}
SAMPLE_AUDIO_TEXT = "Gemma audio demo. Please transcribe this short validation clip."


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def assets_root() -> Path:
    return repo_root() / "assets"


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_asset_dirs(dry_run: bool) -> list[Path]:
    created = []
    root = assets_root()
    targets = [root] + [root / directory for directory in EXPECTED_DIRS.values()]
    for path in targets:
        if not path.exists():
            created.append(path)
            if not dry_run:
                path.mkdir(parents=True, exist_ok=True)
    return created


def network_available(timeout: float) -> tuple[bool, str]:
    request = urllib.request.Request(
        "https://example.com/",
        headers={"User-Agent": "gemma-lab-bootstrap/2.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            return True, f"reachable (status={status})"
    except Exception as exc:
        return False, f"unavailable ({type(exc).__name__}: {exc})"


def write_bytes(path: Path, content: bytes, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def import_pillow() -> tuple[object, object, object]:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency dependent
        raise RuntimeError(
            "Pillow is required to generate the Phase 2 raster image assets. "
            "Install the local venv requirements first with `python -m pip install -r requirements.txt`."
        ) from exc

    return Image, ImageDraw, ImageFont


def load_font(image_font: object, size: int) -> object:
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in font_candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return image_font.truetype(str(path), size=size)
            except Exception:
                continue
    return image_font.load_default()


def generate_sample_image(path: Path, dry_run: bool) -> str:
    if dry_run:
        return "would generate local PNG sample image"

    Image, ImageDraw, ImageFont = import_pillow()
    image = Image.new("RGB", (1280, 768), "#eef3f8")
    draw = ImageDraw.Draw(image)

    for y in range(image.height):
        mix = y / max(1, image.height - 1)
        r = int(238 - 26 * mix)
        g = int(243 - 14 * mix)
        b = int(248 - 6 * mix)
        draw.line((0, y, image.width, y), fill=(r, g, b))

    draw.rounded_rectangle((80, 92, 1200, 676), radius=40, fill="#fffdf8", outline="#d9d2c6", width=3)
    draw.ellipse((132, 148, 420, 436), fill="#d7b98f")
    draw.ellipse((198, 232, 334, 368), fill="#fff5e8")
    draw.rounded_rectangle((520, 162, 1044, 226), radius=18, fill="#25364a")
    draw.rounded_rectangle((520, 270, 962, 324), radius=16, fill="#506680")
    draw.rounded_rectangle((520, 360, 1126, 434), radius=22, fill="#ffffff", outline="#d8dde5", width=3)
    draw.rounded_rectangle((520, 462, 1108, 590), radius=26, fill="#dbe7f1")
    draw.ellipse((904, 494, 1038, 628), fill="#6e93b2")

    title_font = load_font(ImageFont, 48)
    body_font = load_font(ImageFont, 30)
    draw.text((566, 174), "Sample desk scene", fill="#fbfcff", font=title_font)
    draw.text((566, 276), "Ceramic cup beside a clean notebook.", fill="#f3f7fb", font=body_font)
    draw.text((562, 384), "Warm morning light enters from the left.", fill="#25364a", font=body_font)
    draw.text((562, 508), "Synthetic asset for caption and localization prompts.", fill="#2f445b", font=body_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return "generated locally as PNG sample image"


def generate_compare_image(path: Path, dry_run: bool) -> str:
    if dry_run:
        return "would generate local PNG comparison image"

    Image, ImageDraw, ImageFont = import_pillow()
    image = Image.new("RGB", (1280, 768), "#edf5f3")
    draw = ImageDraw.Draw(image)

    for y in range(image.height):
        mix = y / max(1, image.height - 1)
        r = int(237 - 18 * mix)
        g = int(245 - 10 * mix)
        b = int(243 - 14 * mix)
        draw.line((0, y, image.width, y), fill=(r, g, b))

    draw.rounded_rectangle((80, 92, 1200, 676), radius=40, fill="#fffdf8", outline="#d0d9d4", width=3)
    draw.ellipse((160, 176, 448, 464), fill="#7ea7b0")
    draw.ellipse((226, 260, 362, 396), fill="#f5fbff")
    draw.rounded_rectangle((520, 162, 1044, 226), radius=18, fill="#204d52")
    draw.rounded_rectangle((520, 270, 990, 324), radius=16, fill="#4e7f79")
    draw.rounded_rectangle((520, 360, 1126, 434), radius=22, fill="#ffffff", outline="#d5dfdb", width=3)
    draw.rounded_rectangle((520, 462, 1108, 590), radius=26, fill="#d9ede8")
    draw.rounded_rectangle((884, 486, 1092, 610), radius=18, fill="#78b9a7")

    title_font = load_font(ImageFont, 48)
    body_font = load_font(ImageFont, 30)
    draw.text((566, 174), "Sample desk scene", fill="#f8fffd", font=title_font)
    draw.text((566, 276), "Cup moved right and turned teal.", fill="#eef9f5", font=body_font)
    draw.text((562, 384), "Notebook annotation now says Updated layout.", fill="#22454a", font=body_font)
    draw.text((562, 508), "Synthetic asset for image comparison prompts.", fill="#2c5656", font=body_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return "generated locally as PNG comparison image"


def generate_text_image(path: Path, dry_run: bool) -> str:
    if dry_run:
        return "would generate local PNG OCR sample image"

    Image, ImageDraw, ImageFont = import_pillow()
    image = Image.new("RGB", (1100, 1400), "#f3efe7")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((90, 90, 1010, 1310), radius=28, fill="#fffdf8", outline="#d9d0c2", width=4)

    title_font = load_font(ImageFont, 48)
    heading_font = load_font(ImageFont, 36)
    body_font = load_font(ImageFont, 28)
    mono_font = load_font(ImageFont, 30)

    draw.text((150, 154), "Gemma Lab Invoice", fill="#27231d", font=title_font)
    draw.text((150, 248), "Invoice ID: G4-001", fill="#3e382f", font=heading_font)
    draw.text((150, 308), "Ship To: Gemma Lab", fill="#3e382f", font=heading_font)
    draw.text((150, 398), "Line Item A ............ 2", fill="#4b4439", font=body_font)
    draw.text((150, 452), "Line Item B ............ 1", fill="#4b4439", font=body_font)
    draw.text((150, 548), "Total Items: 3", fill="#2a5c4d", font=heading_font)
    draw.text((150, 654), "Notes", fill="#27231d", font=heading_font)
    draw.text(
        (150, 714),
        "Use Gemma native vision first for OCR.",
        fill="#5d5548",
        font=body_font,
    )
    draw.text(
        (150, 766),
        "Keep the pipeline small and reproducible.",
        fill="#5d5548",
        font=body_font,
    )
    draw.text((150, 930), "STATUS: READY FOR OCR", fill="#8b4b2d", font=mono_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return "generated locally as PNG OCR sample image"


def generate_chart_ui_image(path: Path, dry_run: bool) -> str:
    if dry_run:
        return "would generate local PNG chart/UI screenshot"

    Image, ImageDraw, ImageFont = import_pillow()
    image = Image.new("RGB", (1440, 900), "#edf2f5")
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle((44, 36, 1396, 864), radius=34, fill="#ffffff")
    draw.rounded_rectangle((74, 72, 320, 828), radius=28, fill="#10263d")
    draw.rounded_rectangle((362, 74, 1348, 152), radius=22, fill="#edf3f8")
    draw.rounded_rectangle((364, 190, 840, 464), radius=24, fill="#f8fbfd", outline="#d8e1e8", width=3)
    draw.rounded_rectangle((872, 190, 1348, 464), radius=24, fill="#f8fbfd", outline="#d8e1e8", width=3)
    draw.rounded_rectangle((364, 500, 1348, 794), radius=24, fill="#f8fbfd", outline="#d8e1e8", width=3)

    draw.line((418, 416, 506, 378, 594, 392, 682, 316, 770, 266), fill="#2a9d8f", width=10, joint="curve")
    draw.ellipse((408, 406, 428, 426), fill="#2a9d8f")
    draw.ellipse((496, 368, 516, 388), fill="#2a9d8f")
    draw.ellipse((584, 382, 604, 402), fill="#2a9d8f")
    draw.ellipse((672, 306, 692, 326), fill="#2a9d8f")
    draw.ellipse((760, 256, 780, 276), fill="#2a9d8f")

    bars = [
        (944, 312, 1016, 404, "#577590"),
        (1044, 266, 1116, 404, "#43aa8b"),
        (1144, 224, 1216, 404, "#f4a261"),
        (1244, 290, 1316, 404, "#e76f51"),
    ]
    for left, top, right, bottom, color in bars:
        draw.rounded_rectangle((left, top, right, bottom), radius=12, fill=color)

    title_font = load_font(ImageFont, 42)
    heading_font = load_font(ImageFont, 30)
    body_font = load_font(ImageFont, 24)

    draw.text((114, 110), "Gemma Lab", fill="#f9fbfd", font=title_font)
    draw.text((114, 196), "Overview", fill="#d6e5f2", font=heading_font)
    draw.text((114, 244), "Revenue", fill="#d6e5f2", font=heading_font)
    draw.text((114, 292), "Experiments", fill="#d6e5f2", font=heading_font)

    draw.text((410, 94), "Capability Dashboard", fill="#21384d", font=title_font)
    draw.text((404, 216), "Weekly trend", fill="#21384d", font=heading_font)
    draw.text((912, 216), "Category totals", fill="#21384d", font=heading_font)
    draw.text((404, 526), "Notes", fill="#21384d", font=heading_font)
    draw.text((408, 588), "Top metric: OCR confidence rose from 74 to 91.", fill="#46627a", font=body_font)
    draw.text((408, 634), "UI sample includes charts, labels, and dashboard chrome.", fill="#46627a", font=body_font)
    draw.text((408, 680), "Use this asset for chart or screenshot VQA.", fill="#46627a", font=body_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return "generated locally as PNG chart/UI screenshot"


def generate_audio(path: Path, dry_run: bool) -> str:
    if dry_run:
        return f"would generate local spoken WAV clip saying: {SAMPLE_AUDIO_TEXT}"

    if sys.platform != "darwin":
        raise RuntimeError(
            "Phase 3 sample audio generation expects macOS `say`. "
            "Place a short spoken clip at `assets/audio/sample_audio.wav` if this workspace is not on macOS."
        )

    say = shutil.which("say")
    if say is None:
        raise RuntimeError(
            "macOS `say` is not available to synthesize the Phase 3 validation clip."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            say,
            "-o",
            str(path),
            "--file-format=WAVE",
            "--data-format=LEI16@16000",
            SAMPLE_AUDIO_TEXT,
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown `say` error"
        path.unlink(missing_ok=True)
        raise RuntimeError(f"speech synthesis failed ({detail})")

    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
    except wave.Error as exc:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"speech synthesis produced an unreadable WAV file ({exc})") from exc

    if frame_count <= 0:
        path.unlink(missing_ok=True)
        raise RuntimeError("macOS `say` produced an empty WAV file in this environment")

    return (
        "generated locally as spoken mono 16 kHz WAV via say; "
        f"spoken text: {SAMPLE_AUDIO_TEXT}"
    )


def generate_pdf(path: Path, dry_run: bool) -> str:
    lines = [
        ("Gemma Lab Vision Sample", 18),
        ("This short PDF validates local page rasterization for pdf-summary mode.", 12),
        ("Owner: Gemma Lab", 12),
        ("Phase: 2", 12),
        ("Launch Date: 2026-04-06", 12),
        ("Goal: Validate caption, OCR, compare, and document summary flows.", 12),
    ]

    content_parts = ["BT\n"]
    current_y = 150
    for text, font_size in lines:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content_parts.append(f"/F1 {font_size} Tf\n50 {current_y} Td\n({escaped}) Tj\n".encode("ascii").decode("ascii"))
        current_y -= 22
    content_parts.append("ET\n")
    content_stream = "".join(content_parts).encode("ascii")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        b"<< /Length %d >>\nstream\n%bendstream" % (len(content_stream), content_stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    buffer = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, payload in enumerate(objects, start=1):
        offsets.append(len(buffer))
        buffer.extend(f"{index} 0 obj\n".encode("ascii"))
        buffer.extend(payload)
        buffer.extend(b"\nendobj\n")

    xref_start = len(buffer)
    buffer.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    buffer.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        buffer.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    buffer.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("ascii")
    )

    write_bytes(path, bytes(buffer), dry_run)
    return "generated locally as sample PDF"


def generate_video_with_ffmpeg(path: Path, dry_run: bool) -> tuple[bool, str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False, "ffmpeg not available for local video generation"

    if dry_run:
        return True, "would generate local MP4 via ffmpeg with visible motion"

    try:
        Image, ImageDraw, ImageFont = import_pillow()
    except Exception as exc:
        return False, f"video sample generation could not import Pillow ({type(exc).__name__}: {exc})"

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        frames_root = Path(tmpdir)
        title_font = load_font(ImageFont, 18)
        body_font = load_font(ImageFont, 14)

        for index in range(12):
            image = Image.new("RGB", (320, 180), "#e8f0f8")
            draw = ImageDraw.Draw(image)

            for y in range(image.height):
                mix = y / max(1, image.height - 1)
                r = int(232 - 34 * mix)
                g = int(240 - 18 * mix)
                b = int(248 - 6 * mix)
                draw.line((0, y, image.width, y), fill=(r, g, b))

            draw.rounded_rectangle((18, 18, 302, 162), radius=18, fill="#fefcf8", outline="#cbd7e5", width=2)

            progress = index / 11.0
            panel_left = 28 + int(150 * progress)
            panel_top = 64
            panel_color = (
                int(48 + 84 * progress),
                int(92 + 48 * progress),
                int(124 - 24 * progress),
            )
            draw.rounded_rectangle(
                (panel_left, panel_top, panel_left + 74, panel_top + 54),
                radius=12,
                fill=panel_color,
            )
            accent_left = 224 - int(92 * progress)
            draw.ellipse((accent_left, 92, accent_left + 34, 126), fill="#f2a65a")

            phase_label = "START" if index < 4 else "MIDDLE" if index < 8 else "END"
            draw.text((34, 30), "Gemma Video Proxy", fill="#23415a", font=title_font)
            draw.text((34, 132), f"Frame phase: {phase_label}", fill="#425d73", font=body_font)
            draw.text((182, 30), "Object shifts right", fill="#425d73", font=body_font)

            frame_path = frames_root / f"frame-{index + 1:02d}.png"
            image.save(frame_path, format="PNG")

        command = [
            ffmpeg,
            "-y",
            "-framerate",
            "10",
            "-i",
            str(frames_root / "frame-%02d.png"),
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
                timeout=20,
            )
        except Exception as exc:
            return False, f"ffmpeg execution failed ({type(exc).__name__}: {exc})"

    if completed.returncode != 0:
        stderr = completed.stderr.strip().splitlines()
        detail = stderr[-1] if stderr else "unknown ffmpeg error"
        return False, f"ffmpeg failed ({detail})"

    return True, "generated locally as MP4 via ffmpeg with visible motion across sampled frames"


def download_file(url: str, destination: Path, timeout: float, dry_run: bool) -> tuple[bool, str]:
    if dry_run:
        return True, f"would download from {url}"

    request = urllib.request.Request(url, headers={"User-Agent": "gemma-lab-bootstrap/2.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read()
    except urllib.error.URLError as exc:
        return False, f"download failed ({type(exc).__name__}: {exc})"
    except Exception as exc:
        return False, f"download failed ({type(exc).__name__}: {exc})"

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return True, f"downloaded from {url}"


def sample_specs() -> list[dict[str, object]]:
    root = assets_root()
    return [
        {
            "id": "image-sample",
            "kind": "image",
            "path": root / EXPECTED_DIRS["images"] / "sample.png",
            "generator": generate_sample_image,
            "download_url": None,
        },
        {
            "id": "image-compare-sample",
            "kind": "image",
            "path": root / EXPECTED_DIRS["images"] / "sample_compare.png",
            "generator": generate_compare_image,
            "download_url": None,
        },
        {
            "id": "text-image-sample",
            "kind": "text-image",
            "path": root / EXPECTED_DIRS["images"] / "sample_text.png",
            "generator": generate_text_image,
            "download_url": None,
        },
        {
            "id": "chart-ui-screenshot-sample",
            "kind": "chart-or-ui-screenshot",
            "path": root / EXPECTED_DIRS["images"] / "sample_chart_ui.png",
            "generator": generate_chart_ui_image,
            "download_url": None,
        },
        {
            "id": "audio-sample",
            "kind": "audio",
            "path": root / EXPECTED_DIRS["audio"] / "sample_audio.wav",
            "generator": generate_audio,
            "download_url": (
                "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/"
                "Demos/sample-data/journal1.wav"
            ),
        },
        {
            "id": "video-sample",
            "kind": "video",
            "path": root / EXPECTED_DIRS["video"] / "sample_video.mp4",
            "generator": None,
            "download_url": "https://samplelib.com/lib/preview/mp4/sample-5s.mp4",
        },
        {
            "id": "pdf-sample",
            "kind": "pdf",
            "path": root / EXPECTED_DIRS["docs"] / "sample.pdf",
            "generator": generate_pdf,
            "download_url": None,
        },
    ]


def process_samples(dry_run: bool, timeout: float) -> dict[str, object]:
    online, network_note = network_available(timeout=timeout)
    items = []

    for spec in sample_specs():
        destination = Path(spec["path"])
        generator = spec["generator"]
        download_url = spec["download_url"]

        action = "skipped"
        status = "unavailable"
        notes = []

        if spec["kind"] == "video":
            success, note = generate_video_with_ffmpeg(destination, dry_run=dry_run)
            if success:
                action = "generated"
                status = "ok"
                notes.append(note)
            elif online and download_url:
                success, note = download_file(str(download_url), destination, timeout, dry_run)
                action = "downloaded" if success else "skipped"
                status = "ok" if success else "unavailable"
                notes.append(note)
            else:
                notes.append(note)
                if download_url:
                    notes.append(f"download skipped because network is {network_note}")
        else:
            if callable(generator):
                try:
                    note = generator(destination, dry_run=dry_run)
                except Exception as exc:
                    destination.unlink(missing_ok=True)
                    notes.append(f"generation failed ({type(exc).__name__}: {exc})")
                    if online and download_url:
                        success, note = download_file(str(download_url), destination, timeout, dry_run)
                        action = "downloaded" if success else "skipped"
                        status = "ok" if success else "unavailable"
                        notes.append(note)
                    else:
                        action = "skipped"
                        status = "unavailable"
                        if download_url:
                            notes.append(f"download skipped because network is {network_note}")
                else:
                    action = "generated"
                    status = "ok"
                    notes.append(note)
            elif online and download_url:
                success, note = download_file(str(download_url), destination, timeout, dry_run)
                action = "downloaded" if success else "skipped"
                status = "ok" if success else "unavailable"
                notes.append(note)
            elif download_url:
                notes.append(f"download skipped because network is {network_note}")

        if not dry_run and status != "ok":
            destination.unlink(missing_ok=True)

        items.append(
            {
                "id": spec["id"],
                "kind": spec["kind"],
                "relative_path": str(destination.relative_to(repo_root())),
                "action": action,
                "status": status,
                "notes": notes,
            }
        )

    return {
        "generated_at_utc": timestamp_utc(),
        "dry_run": dry_run,
        "network": {
            "available": online,
            "note": network_note,
        },
        "items": items,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare local demo assets for the Gemma capability lab and write a manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without writing files.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Network timeout in seconds for probes/downloads (default: 3.0).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    created_dirs = ensure_asset_dirs(dry_run=args.dry_run)
    manifest = process_samples(dry_run=args.dry_run, timeout=args.timeout)
    manifest["assets_root"] = str(assets_root().relative_to(repo_root()))
    manifest["created_directories"] = [
        str(path.relative_to(repo_root())) for path in created_dirs
    ]

    manifest_path = assets_root() / "manifest.json"
    if not args.dry_run:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
