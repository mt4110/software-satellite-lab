from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from probe_long_context_live_limit import (  # noqa: E402
    build_default_targets,
    parse_targets,
    summarize_probe_results,
)


class LongContextProbeUtilityTests(unittest.TestCase):
    def test_build_default_targets_includes_max_target(self) -> None:
        self.assertEqual(
            build_default_targets(24_576, step=8_192),
            [8_192, 16_384, 24_576],
        )
        self.assertEqual(
            build_default_targets(20_000, step=8_192),
            [8_192, 16_384, 20_000],
        )

    def test_parse_targets_sorts_and_deduplicates(self) -> None:
        self.assertEqual(
            parse_targets("24576,8192,16384,16384"),
            [8_192, 16_384, 24_576],
        )

    def test_summarize_probe_results_picks_last_validated_before_limit(self) -> None:
        summary = summarize_probe_results(
            [
                {
                    "target_prompt_tokens": 8_192,
                    "status": "ok",
                    "minimum_case_prompt_tokens": 7_900,
                    "artifact_path": "/tmp/8k.json",
                    "message": None,
                },
                {
                    "target_prompt_tokens": 16_384,
                    "status": "ok",
                    "minimum_case_prompt_tokens": 15_900,
                    "artifact_path": "/tmp/16k.json",
                    "message": None,
                },
                {
                    "target_prompt_tokens": 24_576,
                    "status": "blocked",
                    "minimum_case_prompt_tokens": None,
                    "artifact_path": "/tmp/24k.json",
                    "message": "RuntimeError: Invalid buffer size",
                },
            ]
        )

        self.assertEqual(summary["highest_live_validated_target_prompt_tokens"], 16_384)
        self.assertEqual(summary["highest_live_validated_realized_prompt_tokens"], 15_900)
        self.assertEqual(summary["first_limit_target_prompt_tokens"], 24_576)
        self.assertEqual(summary["first_limit_status"], "blocked")
        self.assertEqual(summary["recommended_live_target_prompt_tokens"], 16_384)


if __name__ == "__main__":
    unittest.main()
