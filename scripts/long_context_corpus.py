#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path


MARKER_LABELS = ("beginning", "middle", "end")
TOPICS = (
    "calibration",
    "retrieval",
    "alignment",
    "safety",
    "latency",
    "evaluation",
    "coverage",
    "logging",
    "handoff",
    "observability",
)
TONES = (
    "steady",
    "compact",
    "repeatable",
    "measured",
    "careful",
    "deterministic",
    "deliberate",
    "traceable",
)
WORD_BANK = (
    "amber",
    "anchor",
    "arc",
    "atlas",
    "birch",
    "brisk",
    "cable",
    "calm",
    "cedar",
    "clear",
    "cobalt",
    "copper",
    "coral",
    "craft",
    "delta",
    "drift",
    "ember",
    "field",
    "flare",
    "frame",
    "glint",
    "grain",
    "graph",
    "harbor",
    "horizon",
    "index",
    "keel",
    "kind",
    "ledger",
    "lilac",
    "linen",
    "marker",
    "matrix",
    "merit",
    "metric",
    "mint",
    "model",
    "mosaic",
    "north",
    "orbit",
    "packet",
    "plain",
    "prism",
    "proof",
    "query",
    "quiet",
    "raster",
    "record",
    "ridge",
    "river",
    "signal",
    "silver",
    "slate",
    "spark",
    "stable",
    "stone",
    "table",
    "tidy",
    "trace",
    "vector",
    "violet",
    "warm",
    "weave",
)
ALLOWED_TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".txt",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
}
ALLOWED_TEXT_FILENAMES = {
    ".gitignore",
    "README",
    "README.md",
    "PLAN.md",
    "DOCUMENTATION.md",
    "requirements.txt",
}
SKIP_TOP_LEVEL_PARTS = {
    ".venv",
    "artifacts",
    "assets",
    "__pycache__",
}


def _advance_state(state: int) -> int:
    return (6364136223846793005 * state + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF


def deterministic_words(namespace: str, count: int) -> list[str]:
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    state = int.from_bytes(digest[:8], "big") or 1
    words: list[str] = []
    for _ in range(count):
        state = _advance_state(state)
        words.append(WORD_BANK[state % len(WORD_BANK)])
    return words


def build_markers(seed: str) -> dict[str, str]:
    markers: dict[str, str] = {}
    for label in MARKER_LABELS:
        digest = hashlib.sha256(f"{seed}:marker:{label}".encode("utf-8")).hexdigest().upper()
        markers[label] = (
            f"GEMMA-LC-{label.upper()}-{digest[:6]}-{digest[6:10]}-{digest[10:14]}"
        )
    return markers


def build_synthetic_segment(seed: str, index: int) -> str:
    topic = TOPICS[index % len(TOPICS)]
    tone = TONES[index % len(TONES)]
    words = deterministic_words(f"{seed}:synthetic:{index}", 48)
    return (
        f"[segment {index:05d}] topic={topic} tone={tone} "
        f"lab note {' '.join(words[:16])} "
        f"checkpoint {' '.join(words[16:32])} "
        f"summary {' '.join(words[32:48])}"
    )


def build_needle_segment(label: str, value: str) -> str:
    return (
        f"[needle {label}] retrieval_marker_{label} = {value}\n"
        f"[needle {label}] marker_location = {label}\n"
        f"[needle {label}] instruction = return the value exactly if asked"
    )


def marker_insert_indices(segment_count: int) -> dict[str, int]:
    if segment_count < 6:
        return {
            "beginning": 1,
            "middle": max(2, segment_count // 2),
            "end": max(3, segment_count - 2),
        }
    return {
        "beginning": max(1, segment_count // 20),
        "middle": segment_count // 2,
        "end": max(segment_count - max(2, segment_count // 20), 0),
    }


def _with_needles(segments: list[str], markers: dict[str, str]) -> tuple[str, dict[str, object]]:
    working = list(segments)
    insert_indices = marker_insert_indices(len(working))
    offset = 0
    applied_indices: dict[str, int] = {}
    for label in MARKER_LABELS:
        insertion_index = min(insert_indices[label] + offset, len(working))
        working.insert(insertion_index, build_needle_segment(label, markers[label]))
        applied_indices[label] = insertion_index
        offset += 1

    corpus_text = "\n\n".join(working).strip() + "\n"
    char_offsets = {}
    for label in MARKER_LABELS:
        target = f"retrieval_marker_{label} = {markers[label]}"
        char_offsets[label] = corpus_text.find(target)

    return corpus_text, {
        "segment_count": len(segments),
        "marker_insert_indices": applied_indices,
        "marker_char_offsets": char_offsets,
        "word_count_estimate": len(corpus_text.split()),
        "character_count": len(corpus_text),
    }


def build_synthetic_corpus(target_word_budget: int, seed: str) -> dict[str, object]:
    markers = build_markers(seed)
    segment_count = max(12, (max(target_word_budget, 1) + 53) // 54)
    segments = [build_synthetic_segment(seed, index) for index in range(segment_count)]
    corpus_text, metadata = _with_needles(segments, markers)
    return {
        "case_id": "synthetic-corpus",
        "case_label": "synthetic",
        "corpus_text": corpus_text,
        "markers": markers,
        "metadata": metadata,
    }


def _should_skip_path(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    parts = relative.parts
    if not parts:
        return True
    first = parts[0]
    if first in SKIP_TOP_LEVEL_PARTS:
        return True
    if first.startswith(".venv-py"):
        return True
    return False


def collect_repo_snippets(
    root: Path,
    max_files: int = 24,
    max_chars_per_file: int = 4000,
) -> list[dict[str, object]]:
    snippets: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if _should_skip_path(root, path):
            continue

        if path.suffix.lower() not in ALLOWED_TEXT_SUFFIXES and path.name not in ALLOWED_TEXT_FILENAMES:
            continue

        try:
            if path.stat().st_size > 250_000:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            continue

        if not text:
            continue

        relative = path.relative_to(root)
        snippet_text = text[:max_chars_per_file].strip()
        if not snippet_text:
            continue

        body = f"[file {relative}]\n{snippet_text}"
        snippets.append(
            {
                "path": str(relative),
                "text": body,
                "word_count": len(body.split()),
                "character_count": len(body),
            }
        )
        if len(snippets) >= max_files:
            break
    return snippets


def build_repo_corpus(root: Path, target_word_budget: int, seed: str) -> dict[str, object] | None:
    base_snippets = collect_repo_snippets(root)
    if not base_snippets:
        return None

    markers = build_markers(f"{seed}:repo")
    segments: list[str] = []
    total_words = 0
    cycle = 0

    while total_words < max(target_word_budget, 1):
        for index, record in enumerate(base_snippets):
            header = f"[repo-cycle {cycle:03d} item {index:03d}]"
            segment = f"{header}\n{record['text']}"
            segments.append(segment)
            total_words += int(record["word_count"]) + 3
            if total_words >= target_word_budget:
                break
        cycle += 1

    corpus_text, metadata = _with_needles(segments, markers)
    metadata["source_files"] = [record["path"] for record in base_snippets]
    metadata["source_file_count"] = len(base_snippets)
    metadata["repo_cycles"] = cycle

    return {
        "case_id": "repo-file-corpus",
        "case_label": "repo",
        "corpus_text": corpus_text,
        "markers": markers,
        "metadata": metadata,
    }
