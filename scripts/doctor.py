#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from gemma_runtime import build_repo_runtime_hint, expected_repo_venv_python, using_repo_venv_python


EXPECTED_ASSET_DIRS = [
    "image",
    "text_image",
    "chart_ui_screenshot",
    "audio",
    "video",
    "pdf",
    "images",
    "docs",
]


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def python_summary() -> str:
    return sys.version.replace("\n", " ")


def probe_torch() -> dict[str, object]:
    result: dict[str, object] = {
        "installed": False,
        "version": None,
        "cuda_available": False,
        "mps_available": False,
        "cpu_available": True,
        "gpu_devices": [],
        "error": None,
    }

    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["installed"] = True
    result["version"] = getattr(torch, "__version__", "unknown")

    try:
        cuda_available = bool(torch.cuda.is_available())
        result["cuda_available"] = cuda_available
        if cuda_available:
            devices = []
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "memory": format_bytes(int(props.total_memory)),
                    }
                )
            result["gpu_devices"] = devices
    except Exception as exc:  # pragma: no cover - environment dependent
        result["error"] = f"CUDA probe failed: {type(exc).__name__}: {exc}"

    try:
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is not None:
            result["mps_available"] = bool(mps_backend.is_available())
    except Exception as exc:  # pragma: no cover - environment dependent
        if result["error"] is None:
            result["error"] = f"MPS probe failed: {type(exc).__name__}: {exc}"

    return result


def probe_transformers() -> dict[str, object]:
    result: dict[str, object] = {
        "installed": False,
        "version": None,
        "auto_model_for_multimodal_lm": False,
        "error": None,
    }

    try:
        import transformers  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["installed"] = True
    result["version"] = getattr(transformers, "__version__", "unknown")
    result["auto_model_for_multimodal_lm"] = hasattr(transformers, "AutoModelForMultimodalLM")
    return result


def probe_optional_modules() -> dict[str, bool]:
    modules = {
        "pillow": "PIL",
        "pypdfium2": "pypdfium2",
        "torchvision": "torchvision",
    }
    results = {}
    for label, module_name in modules.items():
        try:
            __import__(module_name)
        except Exception:
            results[label] = False
        else:
            results[label] = True
    return results


def probe_tkinter() -> dict[str, object]:
    result: dict[str, object] = {
        "available": False,
        "tk_version": None,
        "error": None,
    }

    try:
        import tkinter as tk  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["available"] = True
    result["tk_version"] = getattr(tk, "TkVersion", None)
    return result


def preferred_runtime_device_label(torch_info: dict[str, object]) -> str:
    if bool(torch_info.get("cuda_available")):
        devices = torch_info.get("gpu_devices") or []
        if devices:
            first = devices[0]
            name = first.get("name", "unknown")
            return f"cuda ({name})"
        return "cuda"

    if bool(torch_info.get("mps_available")):
        return "mps (Apple Metal)"

    return "cpu"


def command_output(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except Exception:
        return None

    if completed.returncode != 0:
        return None

    return completed.stdout.strip()


def fallback_gpu_summary() -> list[dict[str, str]]:
    nvidia = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader",
        ]
    )
    if nvidia:
        devices = []
        for index, line in enumerate(nvidia.splitlines()):
            parts = [item.strip() for item in line.split(",")]
            if len(parts) >= 2:
                devices.append(
                    {
                        "index": str(index),
                        "name": parts[0],
                        "memory": parts[1],
                    }
                )
        if devices:
            return devices

    if platform.system() == "Darwin":
        profiler = command_output(["system_profiler", "SPDisplaysDataType", "-json"])
        if profiler:
            try:
                payload = json.loads(profiler)
            except json.JSONDecodeError:
                return []

            devices = []
            for index, item in enumerate(payload.get("SPDisplaysDataType", [])):
                name = item.get("sppci_model") or item.get("_name")
                memory = (
                    item.get("spdisplays_vram")
                    or item.get("spdisplays_vram_shared")
                    or "shared/unreported"
                )
                if name:
                    devices.append(
                        {
                            "index": str(index),
                            "name": str(name),
                            "memory": str(memory),
                        }
                    )
            return devices

    return []


def assets_summary(root: Path) -> dict[str, object]:
    assets_root = root / "assets"
    present = {}
    for name in EXPECTED_ASSET_DIRS:
        present[name] = (assets_root / name).is_dir()
    return {
        "assets_root_exists": assets_root.is_dir(),
        "expected_directories": present,
        "manifest_exists": (assets_root / "manifest.json").is_file(),
    }


def print_section(title: str) -> None:
    print(f"\n[{title}]")


def main() -> int:
    root = repo_root()
    torch_info = probe_torch()
    transformers_info = probe_transformers()
    optional_modules = probe_optional_modules()
    tkinter_info = probe_tkinter()
    gpu_devices = torch_info.get("gpu_devices") or fallback_gpu_summary()
    ffmpeg_path = shutil.which("ffmpeg")
    say_path = shutil.which("say")
    assets_info = assets_summary(root)
    phase3_python_ready = sys.version_info >= (3, 10)
    phase3_audio_ready = phase3_python_ready and bool(transformers_info["auto_model_for_multimodal_lm"])
    preferred_device = preferred_runtime_device_label(torch_info)
    expected_python = expected_repo_venv_python(root)
    using_repo_python = using_repo_venv_python(root=root)
    runtime_hint = build_repo_runtime_hint(root=root)
    sample_audio_path = root / "assets" / "audio" / "sample_audio.wav"

    print_section("environment")
    print(f"os: {platform.system()} {platform.release()}")
    print(f"machine: {platform.machine()}")
    print(f"python: {python_summary()}")
    print(f"executable: {Path(sys.executable).expanduser()}")
    print(f"repo_root: {root}")

    print_section("interpreter")
    print(f"repo_venv_python: {expected_python}")
    print(f"using_repo_venv: {using_repo_python}")
    print(f"python_3_10_or_newer: {phase3_python_ready}")
    print(f"tkinter_available: {tkinter_info['available']}")
    print(f"tk_version: {tkinter_info['tk_version']}")
    if runtime_hint:
        print(f"note: {runtime_hint}")
    if tkinter_info.get("error"):
        print(f"note: {tkinter_info['error']}")

    print_section("torch")
    print(f"installed: {torch_info['installed']}")
    print(f"version: {torch_info['version']}")
    print(f"cuda_available: {torch_info['cuda_available']}")
    print(f"mps_available: {torch_info['mps_available']}")
    print(f"cpu_available: {torch_info['cpu_available']}")
    if torch_info.get("error"):
        print(f"note: {torch_info['error']}")

    print_section("transformers")
    print(f"installed: {transformers_info['installed']}")
    print(f"version: {transformers_info['version']}")
    print(f"auto_model_for_multimodal_lm: {transformers_info['auto_model_for_multimodal_lm']}")
    if transformers_info.get("error"):
        print(f"note: {transformers_info['error']}")

    print_section("gpu")
    if gpu_devices:
        for device in gpu_devices:
            print(
                "- index={index} name={name} memory={memory}".format(
                    index=device.get("index", "?"),
                    name=device.get("name", "unknown"),
                    memory=device.get("memory", "unknown"),
                )
            )
    else:
        print("- unavailable or not detected")

    print_section("runtime_device")
    print(f"preferred_runtime_device: {preferred_device}")
    print(f"max_local_acceleration: {preferred_device != 'cpu'}")

    print_section("tooling")
    print(f"ffmpeg: {'found at ' + ffmpeg_path if ffmpeg_path else 'not found'}")
    print(f"say: {'found at ' + say_path if say_path else 'not found'}")
    print(f"pillow: {optional_modules['pillow']}")
    print(f"pypdfium2: {optional_modules['pypdfium2']}")
    print(f"torchvision: {optional_modules['torchvision']}")

    print_section("phase3_audio")
    print(f"python_3_10_or_newer: {phase3_python_ready}")
    print(f"transformers_multimodal_audio_ready: {transformers_info['auto_model_for_multimodal_lm']}")
    print(f"phase3_audio_ready: {phase3_audio_ready}")
    print(f"sample_audio_exists: {sample_audio_path.is_file()}")

    print_section("assets")
    print(f"assets_root_exists: {assets_info['assets_root_exists']}")
    print(f"manifest_exists: {assets_info['manifest_exists']}")
    expected_directories = assets_info["expected_directories"]
    for name in EXPECTED_ASSET_DIRS:
        print(f"- {name}: {expected_directories[name]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
