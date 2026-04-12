from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from artifact_schema import (  # noqa: E402
    ARTIFACT_SCHEMA_NAME,
    ARTIFACT_SCHEMA_VERSION,
    build_artifact_payload,
    build_prompt_record,
    build_runtime_record,
    read_artifact,
    write_artifact,
)
from blocker_taxonomy import (  # noqa: E402
    HARDWARE_LIMIT,
    MISSING_AUTH,
    MISSING_DEPENDENCY,
    NETWORK_ISSUE,
    REPO_BUG,
    UNSUPPORTED_MODE,
    classify_blocker,
)


class ArtifactSchemaTests(unittest.TestCase):
    def test_write_and_read_artifact_round_trip(self) -> None:
        runtime = build_runtime_record(
            backend="gemma-live-text",
            model_id="google/gemma-4-E2B-it",
            device_info={"name": "mps", "label": "mps (Apple Metal)", "dtype_name": "float16"},
            elapsed_seconds=1.2349,
        )
        prompts = build_prompt_record(
            system_prompt="Be careful.",
            prompt="Say hello.",
            resolved_user_prompt="Say hello.",
        )
        payload = build_artifact_payload(
            artifact_kind="text",
            status="ok",
            runtime=runtime,
            prompts=prompts,
            asset_lineage=[
                {
                    "source_path": "/tmp/source.png",
                    "resolved_path": "/tmp/cache/normalized.png",
                    "cache_path": "/tmp/cache/normalized.png",
                    "asset_kind": "image",
                    "transform": "image_normalization",
                    "cache_key": "abc123",
                    "cache_hit": False,
                    "metadata": {"width": 8, "height": 6},
                }
            ],
            extra={"output_text": "hello"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.json"
            write_artifact(path, payload)
            loaded = read_artifact(path)

        self.assertEqual(loaded["schema_name"], ARTIFACT_SCHEMA_NAME)
        self.assertEqual(loaded["schema_version"], ARTIFACT_SCHEMA_VERSION)
        self.assertEqual(loaded["runtime"]["backend"], "gemma-live-text")
        self.assertEqual(loaded["runtime"]["device"]["label"], "mps (Apple Metal)")
        self.assertEqual(loaded["runtime"]["elapsed_seconds"], 1.235)
        self.assertEqual(loaded["prompts"]["system_prompt"], "Be careful.")
        self.assertEqual(loaded["assets"]["lineage"][0]["transform"], "image_normalization")
        self.assertEqual(loaded["output_text"], "hello")


class BlockerTaxonomyTests(unittest.TestCase):
    def test_classify_blocker_distinguishes_taxonomy(self) -> None:
        cases = {
            "ffmpeg is not available for this run": MISSING_DEPENDENCY,
            "Model access failed for `google/gemma-4-E2B-it`. Accept the model terms and provide HF_TOKEN.": MISSING_AUTH,
            "Failed to reach Hugging Face while loading `google/gemma-4-E2B-it`. DNS resolution appears unavailable.": NETWORK_ISSUE,
            "Insufficient memory while loading or running `google/gemma-4-E2B-it` on mps (Apple Metal).": HARDWARE_LIMIT,
            "ValueError: Invalid buffer size: 135.77 GiB": HARDWARE_LIMIT,
            "Unsupported audio input `.ogg` for `/tmp/demo.ogg`.": UNSUPPORTED_MODE,
            "Model returned an empty text response.": REPO_BUG,
        }

        for message, expected_kind in cases.items():
            with self.subTest(message=message):
                info = classify_blocker(message)
                self.assertEqual(info.kind, expected_kind)
                self.assertEqual(info.external, expected_kind != REPO_BUG)


if __name__ == "__main__":
    unittest.main()
