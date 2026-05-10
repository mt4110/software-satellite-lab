#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from gemma_runtime import repo_root, timestamp_slug, timestamp_utc, write_json
from workspace_state import DEFAULT_WORKSPACE_ID


PACK_MANIFEST_SCHEMA_NAME = "software-satellite-pack"
PACK_MANIFEST_SCHEMA_VERSION = 1
PACK_AUDIT_SCHEMA_NAME = "software-satellite-pack-audit"
PACK_AUDIT_SCHEMA_VERSION = 1
PACK_INSPECTION_SCHEMA_NAME = "software-satellite-pack-inspection"
PACK_INSPECTION_SCHEMA_VERSION = 1

PACK_KINDS = ("workflow_pack", "recall_pack", "evaluation_pack", "widget_pack")
MANIFEST_FILENAMES = (
    "satellite.json",
    "satellite.yaml",
    "satellite.yml",
    "manifest.json",
    "manifest.yaml",
    "manifest.yml",
    "pack.satellite.json",
    "pack.satellite.yaml",
    "pack.satellite.yml",
)
MANIFEST_TOP_LEVEL_KEYS = {
    "schema_name",
    "schema_version",
    "name",
    "version",
    "kind",
    "summary",
    "inputs",
    "outputs",
    "permissions",
    "recipes",
    "widgets",
}
REQUIRED_TOP_LEVEL_KEYS = (
    "schema_name",
    "schema_version",
    "name",
    "version",
    "kind",
    "permissions",
    "inputs",
    "outputs",
)
REQUIRED_PERMISSION_KEYS = (
    "read_repo",
    "write_repo",
    "read_artifacts",
    "write_artifacts",
    "read_memory_index",
    "write_evaluation_signal",
    "request_human_verdict",
    "run_command",
    "network",
    "secrets",
    "use_backend",
)
V0_DENIED_TRUE_PERMISSIONS = {
    "write_repo",
    "write_evaluation_signal",
    "run_command",
    "network",
    "secrets",
    "use_backend",
}
PACK_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
RECIPE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

PERMISSION_POLICIES: dict[str, dict[str, str]] = {
    "read_repo": {
        "default": "explicit",
        "v0_policy": "allowed only when the manifest asks for it and the operator can inspect it",
    },
    "write_repo": {
        "default": "deny",
        "v0_policy": "blocked in v0; pack audit must not permit repo mutation",
    },
    "read_artifacts": {
        "default": "allow",
        "v0_policy": "allowed for local file-first evidence inspection",
    },
    "write_artifacts": {
        "default": "allow",
        "v0_policy": "allowed only through core-owned artifact writers",
    },
    "read_memory_index": {
        "default": "allow",
        "v0_policy": "allowed locally; no external transfer",
    },
    "write_evaluation_signal": {
        "default": "deny",
        "v0_policy": "blocked in v0; human-gated evaluation writes come later",
    },
    "request_human_verdict": {
        "default": "allow",
        "v0_policy": "allowed and expected for decision-bearing workflows",
    },
    "run_command": {
        "default": "deny",
        "v0_policy": "blocked in v0; no arbitrary shell or verification runner",
    },
    "network": {
        "default": "deny",
        "v0_policy": "blocked in v0; no marketplace, exfiltration, or remote calls",
    },
    "secrets": {
        "default": "deny",
        "v0_policy": "blocked in v0 with no exception",
    },
    "use_backend": {
        "default": "deny",
        "v0_policy": "blocked in v0; audit foundation does not call live backends",
    },
}


class PackManifestError(ValueError):
    pass


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str
    severity: str = "block"
    expected: str | None = None
    actual: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "path": self.path,
            "message": self.message,
            "severity": self.severity,
        }
        if self.expected is not None:
            payload["expected"] = self.expected
        if self.actual is not None:
            payload["actual"] = self.actual
        return payload


@dataclass(frozen=True)
class _YamlLine:
    lineno: int
    indent: int
    text: str


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root or repo_root()).resolve()


def satellite_pack_root(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return _resolve_root(root) / "artifacts" / "satellite_packs" / workspace_id


def pack_audit_latest_path(
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return satellite_pack_root(workspace_id=workspace_id, root=root) / "audits" / "latest.json"


def pack_audit_run_path(
    *,
    pack_name: str,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> Path:
    return (
        satellite_pack_root(workspace_id=workspace_id, root=root)
        / "audits"
        / "runs"
        / f"{timestamp_slug()}-{_slugify_pack_name(pack_name)}-audit.json"
    )


def satellite_pack_manifest_schema_path(root: Path | None = None) -> Path:
    candidate = _resolve_root(root) / "schemas" / "satellite_pack_manifest.schema.json"
    if candidate.is_file():
        return candidate
    return repo_root() / "schemas" / "satellite_pack_manifest.schema.json"


def _slugify_pack_name(pack_name: str | None) -> str:
    cleaned = (pack_name or "unknown-pack").strip().lower()
    cleaned = re.sub(r"[^a-z0-9-]+", "-", cleaned).strip("-")
    return cleaned or "unknown-pack"


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


def _strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            if index == 0 or line[index - 1].isspace():
                return line[:index].rstrip()
    return line.rstrip()


def _prepare_yaml_lines(text: str, path: Path) -> list[_YamlLine]:
    lines: list[_YamlLine] = []
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        leading_whitespace = raw_line[: len(raw_line) - len(raw_line.lstrip())]
        if "\t" in leading_whitespace:
            raise PackManifestError(f"{path}:{lineno}: tabs are not supported in manifest indentation.")
        stripped_comment = _strip_yaml_comment(raw_line)
        if not stripped_comment.strip():
            continue
        indent = len(stripped_comment) - len(stripped_comment.lstrip(" "))
        if indent % 2 != 0:
            raise PackManifestError(f"{path}:{lineno}: indentation must use two-space steps.")
        lines.append(_YamlLine(lineno=lineno, indent=indent, text=stripped_comment.lstrip(" ")))
    return lines


def _split_key_value(text: str, *, path: Path, lineno: int) -> tuple[str, str]:
    if ":" not in text:
        raise PackManifestError(f"{path}:{lineno}: expected a `key: value` entry.")
    key, value = text.split(":", 1)
    key = key.strip()
    if not key:
        raise PackManifestError(f"{path}:{lineno}: manifest keys must not be empty.")
    return key, value.strip()


def _looks_like_key_value(text: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_-]*:(?:\s|$)", text))


def _parse_yaml_scalar(value: str) -> Any:
    if value == "":
        return ""
    if value.startswith("[") and value.endswith("]"):
        return _parse_yaml_flow_sequence(value)
    if value.startswith("{") and value.endswith("}"):
        return _parse_yaml_flow_mapping(value)
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "~"}:
        return None
    if re.match(r"^-?[0-9]+$", value):
        return int(value)
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    if len(value) >= 2 and value[0] == value[-1] == '"':
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]
    return value


def _split_flow_items(value: str, *, path: Path | None = None) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    escape = False

    for char in value:
        if escape:
            current.append(char)
            escape = False
            continue
        if char == "\\" and in_double:
            current.append(char)
            escape = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            current.append(char)
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            current.append(char)
            continue
        if char == "," and not in_single and not in_double:
            items.append("".join(current).strip())
            current = []
            continue
        current.append(char)

    if in_single or in_double:
        location = f"{path}: " if path is not None else ""
        raise PackManifestError(f"{location}unterminated quoted string in flow sequence.")

    trailing = "".join(current).strip()
    if trailing or value.rstrip().endswith(","):
        items.append(trailing)
    return items


def _parse_yaml_flow_sequence(value: str, *, path: Path | None = None) -> list[Any]:
    inner = value[1:-1].strip()
    if not inner:
        return []
    items = _split_flow_items(inner, path=path)
    return [_parse_yaml_scalar(item) for item in items]


def _parse_yaml_flow_key(value: str) -> str:
    parsed = _parse_yaml_scalar(value)
    if isinstance(parsed, str):
        return parsed
    return str(parsed)


def _parse_yaml_flow_mapping(value: str, *, path: Path | None = None) -> dict[str, Any]:
    inner = value[1:-1].strip()
    if not inner:
        return {}
    mapping: dict[str, Any] = {}
    for item in _split_flow_items(inner, path=path):
        if ":" not in item:
            location = f"{path}: " if path is not None else ""
            raise PackManifestError(f"{location}flow mapping item must use `key: value` syntax.")
        key_text, value_text = item.split(":", 1)
        key = _parse_yaml_flow_key(key_text.strip())
        if key in mapping:
            location = f"{path}: " if path is not None else ""
            raise PackManifestError(f"{location}duplicate flow mapping key `{key}`.")
        mapping[key] = _parse_yaml_scalar(value_text.strip())
    return mapping


def _parse_yaml_block(
    lines: list[_YamlLine],
    index: int,
    indent: int,
    path: Path,
) -> tuple[Any, int]:
    if index >= len(lines):
        return None, index
    if lines[index].indent != indent:
        raise PackManifestError(
            f"{path}:{lines[index].lineno}: unexpected indentation level {lines[index].indent}; "
            f"expected {indent}."
        )
    if lines[index].text.startswith("- "):
        return _parse_yaml_list(lines, index, indent, path)
    return _parse_yaml_mapping(lines, index, indent, path)


def _parse_yaml_mapping(
    lines: list[_YamlLine],
    index: int,
    indent: int,
    path: Path,
) -> tuple[dict[str, Any], int]:
    payload: dict[str, Any] = {}
    while index < len(lines):
        line = lines[index]
        if line.indent < indent:
            break
        if line.indent > indent:
            raise PackManifestError(f"{path}:{line.lineno}: unexpected nested mapping entry.")
        if line.text.startswith("- "):
            break

        key, value_text = _split_key_value(line.text, path=path, lineno=line.lineno)
        if key in payload:
            raise PackManifestError(f"{path}:{line.lineno}: duplicate key `{key}`.")

        index += 1
        if value_text:
            payload[key] = _parse_yaml_scalar(value_text)
            continue

        if index >= len(lines) or lines[index].indent <= indent:
            payload[key] = None
            continue

        payload[key], index = _parse_yaml_block(lines, index, lines[index].indent, path)
    return payload, index


def _parse_yaml_list(
    lines: list[_YamlLine],
    index: int,
    indent: int,
    path: Path,
) -> tuple[list[Any], int]:
    items: list[Any] = []
    while index < len(lines):
        line = lines[index]
        if line.indent < indent:
            break
        if line.indent > indent:
            raise PackManifestError(f"{path}:{line.lineno}: unexpected nested list entry.")
        if not line.text.startswith("- "):
            break

        item_text = line.text[2:].strip()
        index += 1
        if not item_text:
            if index >= len(lines) or lines[index].indent <= indent:
                items.append(None)
                continue
            child, index = _parse_yaml_block(lines, index, lines[index].indent, path)
            items.append(child)
            continue

        if _looks_like_key_value(item_text):
            key, value_text = _split_key_value(item_text, path=path, lineno=line.lineno)
            item: dict[str, Any] = {}
            if value_text:
                item[key] = _parse_yaml_scalar(value_text)
            elif index < len(lines) and lines[index].indent > indent:
                item[key], index = _parse_yaml_block(lines, index, lines[index].indent, path)
            else:
                item[key] = None

            while index < len(lines):
                nested = lines[index]
                if nested.indent < indent + 2:
                    break
                if nested.indent > indent + 2:
                    raise PackManifestError(f"{path}:{nested.lineno}: unexpected nested mapping entry.")
                if nested.text.startswith("- "):
                    break
                nested_key, nested_value_text = _split_key_value(
                    nested.text,
                    path=path,
                    lineno=nested.lineno,
                )
                if nested_key in item:
                    raise PackManifestError(f"{path}:{nested.lineno}: duplicate key `{nested_key}`.")
                index += 1
                if nested_value_text:
                    item[nested_key] = _parse_yaml_scalar(nested_value_text)
                elif index < len(lines) and lines[index].indent > nested.indent:
                    item[nested_key], index = _parse_yaml_block(lines, index, lines[index].indent, path)
                else:
                    item[nested_key] = None

            items.append(item)
            continue

        items.append(_parse_yaml_scalar(item_text))
        if index < len(lines) and lines[index].indent > indent:
            raise PackManifestError(f"{path}:{lines[index].lineno}: scalar list item cannot have a nested block.")
    return items, index


def parse_yaml_manifest_subset(text: str, path: Path) -> dict[str, Any]:
    """Parse the small YAML subset allowed for declarative pack manifests."""
    lines = _prepare_yaml_lines(text, path)
    if not lines:
        raise PackManifestError(f"{path}: manifest is empty.")
    if lines[0].indent != 0:
        raise PackManifestError(f"{path}:{lines[0].lineno}: manifest must start at indentation 0.")
    payload, index = _parse_yaml_block(lines, 0, 0, path)
    if index != len(lines):
        raise PackManifestError(f"{path}:{lines[index].lineno}: could not parse manifest line.")
    if not isinstance(payload, dict):
        raise PackManifestError(f"{path}: manifest root must be an object.")
    return payload


def resolve_pack_manifest_path(pack_path: Path) -> Path:
    path = Path(pack_path).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.exists():
        raise PackManifestError(f"Pack path does not exist: {path}")
    if not path.is_dir():
        raise PackManifestError(f"Pack path must be a manifest file or directory: {path}")

    exact_candidates = [path / name for name in MANIFEST_FILENAMES if (path / name).is_file()]
    glob_candidates = (
        sorted(path.glob("*.satellite.json"))
        + sorted(path.glob("*.satellite.yaml"))
        + sorted(path.glob("*.satellite.yml"))
    )
    candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in exact_candidates + glob_candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(resolved)

    if not candidates:
        expected = ", ".join(MANIFEST_FILENAMES) + ", or one *.satellite.{json,yaml,yml} file"
        raise PackManifestError(f"No Satellite Pack manifest found in {path}. Expected {expected}.")
    if len(candidates) > 1:
        choices = ", ".join(str(candidate) for candidate in candidates)
        raise PackManifestError(f"Multiple Satellite Pack manifests found in {path}: {choices}")
    return candidates[0]


def load_pack_manifest(pack_path: Path) -> dict[str, Any]:
    manifest_path = resolve_pack_manifest_path(pack_path)
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PackManifestError(f"Could not read Satellite Pack manifest `{manifest_path}`: {exc}") from exc

    suffix = manifest_path.suffix.lower()
    if suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PackManifestError(f"{manifest_path}: invalid JSON manifest: {exc}") from exc
    elif suffix in {".yaml", ".yml"}:
        payload = parse_yaml_manifest_subset(text, manifest_path)
    else:
        raise PackManifestError(f"{manifest_path}: manifest must be .json, .yaml, or .yml.")

    if not isinstance(payload, dict):
        raise PackManifestError(f"{manifest_path}: manifest root must be an object.")
    return payload


def _issue(
    issues: list[ValidationIssue],
    path: str,
    message: str,
    *,
    expected: str | None = None,
    actual: Any = None,
) -> None:
    if actual is None:
        actual_text = None
    elif isinstance(actual, str):
        actual_text = actual
    elif isinstance(actual, (bool, int, float)):
        actual_text = json.dumps(actual)
    else:
        actual_text = _type_name(actual)
    issues.append(
        ValidationIssue(
            path=path,
            message=message,
            expected=expected,
            actual=actual_text,
        )
    )


def _validate_string_field(
    manifest: Mapping[str, Any],
    key: str,
    issues: list[ValidationIssue],
    *,
    pattern: re.Pattern[str] | None = None,
    allowed_values: tuple[str, ...] | None = None,
) -> None:
    if key not in manifest:
        return
    value = manifest.get(key)
    path = f"$.{key}"
    if not isinstance(value, str):
        _issue(issues, path, "Expected a string.", expected="string", actual=value)
        return
    if pattern is not None and not pattern.match(value):
        _issue(issues, path, "Value does not match the required pattern.", expected=pattern.pattern, actual=value)
    if allowed_values is not None and value not in allowed_values:
        _issue(issues, path, "Unsupported value.", expected=" | ".join(allowed_values), actual=value)


def _validate_string_list(
    value: Any,
    path: str,
    issues: list[ValidationIssue],
    *,
    min_items: int = 0,
    item_pattern: re.Pattern[str] | None = None,
) -> None:
    if not isinstance(value, list):
        _issue(issues, path, "Expected an array.", expected="array", actual=value)
        return
    if len(value) < min_items:
        _issue(
            issues,
            path,
            f"Expected at least {min_items} item(s).",
            expected=f"minItems {min_items}",
            actual=len(value),
        )
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, str):
            _issue(issues, item_path, "Expected a string.", expected="string", actual=item)
            continue
        if item_pattern is not None and not item_pattern.match(item):
            _issue(
                issues,
                item_path,
                "Value does not match the required pattern.",
                expected=item_pattern.pattern,
                actual=item,
            )


def _validate_permissions(value: Any, issues: list[ValidationIssue]) -> None:
    if not isinstance(value, Mapping):
        _issue(issues, "$.permissions", "Expected an object.", expected="object", actual=value)
        return

    for key in REQUIRED_PERMISSION_KEYS:
        if key not in value:
            _issue(issues, f"$.permissions.{key}", "Missing required permission flag.", expected="boolean")

    additional = sorted(str(key) for key in value.keys() if key not in REQUIRED_PERMISSION_KEYS)
    for key in additional:
        _issue(issues, f"$.permissions.{key}", "Additional permission flags are not allowed in v0.")

    for key in REQUIRED_PERMISSION_KEYS:
        if key not in value:
            continue
        permission_value = value.get(key)
        if not isinstance(permission_value, bool):
            _issue(
                issues,
                f"$.permissions.{key}",
                "Permission flag must be a boolean.",
                expected="boolean",
                actual=permission_value,
            )
            continue
        if key in V0_DENIED_TRUE_PERMISSIONS and permission_value is not False:
            _issue(
                issues,
                f"$.permissions.{key}",
                "Permission must be false in Satellite Pack v0.",
                expected="false",
                actual=str(permission_value).lower(),
            )


def _validate_recipes(value: Any, issues: list[ValidationIssue], *, required: bool) -> None:
    if value is None:
        if required:
            _issue(issues, "$.recipes", "workflow_pack manifests must include recipes.", expected="array")
        return
    if not isinstance(value, list):
        _issue(issues, "$.recipes", "Expected an array.", expected="array", actual=value)
        return
    if not value:
        _issue(issues, "$.recipes", "Expected at least one recipe.", expected="minItems 1", actual=0)
    for index, recipe in enumerate(value):
        recipe_path = f"$.recipes[{index}]"
        if not isinstance(recipe, Mapping):
            _issue(issues, recipe_path, "Expected an object.", expected="object", actual=recipe)
            continue
        for key in ("id", "steps"):
            if key not in recipe:
                _issue(issues, f"{recipe_path}.{key}", "Missing required recipe field.")
        additional = sorted(str(key) for key in recipe.keys() if key not in {"id", "steps"})
        for key in additional:
            _issue(issues, f"{recipe_path}.{key}", "Additional recipe fields are not allowed in v0.")
        if "id" in recipe:
            value_id = recipe.get("id")
            if not isinstance(value_id, str):
                _issue(issues, f"{recipe_path}.id", "Expected a string.", expected="string", actual=value_id)
            elif not RECIPE_ID_RE.match(value_id):
                _issue(
                    issues,
                    f"{recipe_path}.id",
                    "Recipe id does not match the required pattern.",
                    expected=RECIPE_ID_RE.pattern,
                    actual=value_id,
                )
        if "steps" in recipe:
            _validate_string_list(
                recipe.get("steps"),
                f"{recipe_path}.steps",
                issues,
                min_items=1,
                item_pattern=RECIPE_ID_RE,
            )


def validate_manifest_schema(manifest: Mapping[str, Any] | Any) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not isinstance(manifest, Mapping):
        _issue(issues, "$", "Manifest root must be an object.", expected="object", actual=manifest)
        return issues

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in manifest:
            _issue(issues, f"$.{key}", "Missing required manifest field.")

    for key in sorted(str(key) for key in manifest.keys() if key not in MANIFEST_TOP_LEVEL_KEYS):
        _issue(issues, f"$.{key}", "Additional manifest fields are not allowed in v0.")

    if "schema_name" in manifest and manifest.get("schema_name") != PACK_MANIFEST_SCHEMA_NAME:
        _issue(
            issues,
            "$.schema_name",
            "Unsupported manifest schema name.",
            expected=PACK_MANIFEST_SCHEMA_NAME,
            actual=manifest.get("schema_name"),
        )
    schema_version = manifest.get("schema_version")
    if "schema_version" in manifest and (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != PACK_MANIFEST_SCHEMA_VERSION
    ):
        _issue(
            issues,
            "$.schema_version",
            "Unsupported manifest schema version.",
            expected=str(PACK_MANIFEST_SCHEMA_VERSION),
            actual=schema_version,
        )

    _validate_string_field(manifest, "name", issues, pattern=PACK_NAME_RE)
    _validate_string_field(manifest, "version", issues)
    _validate_string_field(manifest, "kind", issues, allowed_values=PACK_KINDS)
    _validate_string_field(manifest, "summary", issues)

    if "inputs" in manifest:
        _validate_string_list(manifest.get("inputs"), "$.inputs", issues)
    if "outputs" in manifest:
        _validate_string_list(manifest.get("outputs"), "$.outputs", issues)
    if "widgets" in manifest:
        _validate_string_list(manifest.get("widgets"), "$.widgets", issues)
    if "permissions" in manifest:
        _validate_permissions(manifest.get("permissions"), issues)

    _validate_recipes(
        manifest.get("recipes"),
        issues,
        required=manifest.get("kind") == "workflow_pack",
    )
    return issues


def build_permission_summary(manifest: Mapping[str, Any] | Any) -> list[dict[str, Any]]:
    permissions = manifest.get("permissions") if isinstance(manifest, Mapping) else None
    kind = manifest.get("kind") if isinstance(manifest, Mapping) else None
    permission_map = permissions if isinstance(permissions, Mapping) else {}
    summary: list[dict[str, Any]] = []

    for permission in REQUIRED_PERMISSION_KEYS:
        policy = PERMISSION_POLICIES[permission]
        present = permission in permission_map
        value = permission_map.get(permission)
        requested = value if isinstance(value, bool) else None
        status = "not_requested"
        reason = "Permission is not requested."

        if not present:
            status = "invalid"
            reason = "Required permission flag is missing."
        elif not isinstance(value, bool):
            status = "invalid"
            reason = "Permission flag must be boolean."
        elif permission in V0_DENIED_TRUE_PERMISSIONS:
            if value:
                status = "blocked"
                reason = policy["v0_policy"]
            else:
                status = "not_requested"
                reason = "Denied-by-default permission remains false."
        elif permission == "read_repo":
            if value:
                status = "needs_review"
                reason = "Pack asks to read repository content; operator should confirm the scope."
            else:
                status = "not_requested"
                reason = "Repository read access is not requested."
        elif permission == "request_human_verdict":
            if value:
                status = "allowed"
                reason = "Human verdict request is compatible with v0 human-gated workflows."
            elif kind == "workflow_pack":
                status = "needs_review"
                reason = "workflow_pack does not request a human verdict; confirm it cannot finalize decisions."
            else:
                status = "not_requested"
                reason = "Human verdict is not requested by this non-workflow pack."
        elif value:
            status = "allowed"
            reason = policy["v0_policy"]

        summary.append(
            {
                "permission": permission,
                "requested": requested,
                "default": policy["default"],
                "v0_policy": policy["v0_policy"],
                "status": status,
                "reason": reason,
            }
        )
    return summary


def _compact_manifest_summary(manifest: Mapping[str, Any] | Any) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        return {
            "name": None,
            "version": None,
            "kind": None,
            "summary": None,
            "input_count": 0,
            "output_count": 0,
            "recipe_count": 0,
            "widget_count": 0,
        }
    recipes = manifest.get("recipes") if isinstance(manifest.get("recipes"), list) else []
    widgets = manifest.get("widgets") if isinstance(manifest.get("widgets"), list) else []
    inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), list) else []
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), list) else []
    return {
        "name": manifest.get("name"),
        "version": manifest.get("version"),
        "kind": manifest.get("kind"),
        "summary": manifest.get("summary"),
        "input_count": len(inputs),
        "output_count": len(outputs),
        "recipe_count": len(recipes),
        "widget_count": len(widgets),
    }


def _compact_recipes(manifest: Mapping[str, Any] | Any) -> list[dict[str, Any]]:
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("recipes"), list):
        return []
    recipes: list[dict[str, Any]] = []
    for recipe in manifest.get("recipes") or []:
        if not isinstance(recipe, Mapping):
            continue
        steps = recipe.get("steps") if isinstance(recipe.get("steps"), list) else []
        recipes.append(
            {
                "id": recipe.get("id"),
                "step_count": len(steps),
                "steps": copy.deepcopy(steps),
            }
        )
    return recipes


def _validation_issue_dicts(issues: list[ValidationIssue]) -> list[dict[str, Any]]:
    return [issue.to_dict() for issue in issues]


def inspect_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    root: Path | None = None,
) -> dict[str, Any]:
    schema_path = satellite_pack_manifest_schema_path(root)
    validation_issues = validate_manifest_schema(manifest)
    return {
        "schema_name": PACK_INSPECTION_SCHEMA_NAME,
        "schema_version": PACK_INSPECTION_SCHEMA_VERSION,
        "inspected_at_utc": timestamp_utc(),
        "manifest_path": str(Path(manifest_path).resolve()),
        "schema_path": str(schema_path.resolve()),
        "schema_valid": not validation_issues,
        "validation_issues": _validation_issue_dicts(validation_issues),
        "manifest": copy.deepcopy(dict(manifest)),
        "manifest_summary": _compact_manifest_summary(manifest),
        "permission_summary": build_permission_summary(manifest),
        "recipes": _compact_recipes(manifest),
        "widgets": copy.deepcopy(manifest.get("widgets") if isinstance(manifest.get("widgets"), list) else []),
        "source_paths": [str(Path(manifest_path).resolve()), str(schema_path.resolve())],
    }


def inspect_pack_path(pack_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    manifest_path = resolve_pack_manifest_path(pack_path)
    manifest = load_pack_manifest(manifest_path)
    return inspect_manifest(manifest, manifest_path=manifest_path, root=root)


def build_pack_audit(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    root: Path | None = None,
) -> dict[str, Any]:
    schema_path = satellite_pack_manifest_schema_path(root)
    validation_issues = validate_manifest_schema(manifest)
    permission_summary = build_permission_summary(manifest)
    blocked_reasons = [
        f"{issue.path}: {issue.message}"
        for issue in validation_issues
        if issue.severity == "block"
    ]
    blocked_reasons.extend(
        f"{item['permission']}: {item['reason']}"
        for item in permission_summary
        if item.get("status") in {"blocked", "invalid"}
        and f"$.permissions.{item['permission']}" not in {issue.path for issue in validation_issues}
    )
    review_reasons = [
        f"{item['permission']}: {item['reason']}"
        for item in permission_summary
        if item.get("status") == "needs_review"
    ]

    if blocked_reasons:
        verdict = "block"
    elif review_reasons:
        verdict = "needs_review"
    else:
        verdict = "pass"

    return {
        "schema_name": PACK_AUDIT_SCHEMA_NAME,
        "schema_version": PACK_AUDIT_SCHEMA_VERSION,
        "audit_id": f"pack-audit:{timestamp_slug()}",
        "audited_at_utc": timestamp_utc(),
        "pack_name": manifest.get("name") if isinstance(manifest, Mapping) else None,
        "pack_version": manifest.get("version") if isinstance(manifest, Mapping) else None,
        "pack_kind": manifest.get("kind") if isinstance(manifest, Mapping) else None,
        "verdict": verdict,
        "permission_summary": permission_summary,
        "blocked_reasons": blocked_reasons,
        "review_reasons": review_reasons,
        "human_review_required": True,
        "source_paths": [str(Path(manifest_path).resolve()), str(schema_path.resolve())],
        "manifest_path": str(Path(manifest_path).resolve()),
        "schema_path": str(schema_path.resolve()),
        "manifest_summary": _compact_manifest_summary(manifest),
        "validation": {
            "schema_valid": not validation_issues,
            "issues": _validation_issue_dicts(validation_issues),
        },
        "v0_restrictions": {
            "arbitrary_runtime": "blocked",
            "marketplace": "blocked",
            "network": "blocked",
            "secrets": "blocked",
            "repo_write": "blocked",
            "live_backend": "blocked",
        },
        "notes": [
            "Audit is declarative only; it does not run recipes or call backends.",
            "Any future pack run must keep source artifact paths and human verdict gates intact.",
        ],
    }


def audit_pack_path(
    pack_path: Path,
    *,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    root: Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    resolved_root = _resolve_root(root)
    manifest_path = resolve_pack_manifest_path(pack_path)
    manifest = load_pack_manifest(manifest_path)
    audit = build_pack_audit(manifest, manifest_path=manifest_path, root=resolved_root)
    latest_path = pack_audit_latest_path(workspace_id=workspace_id, root=resolved_root)
    run_path = pack_audit_run_path(
        pack_name=str(audit.get("pack_name") or "unknown-pack"),
        workspace_id=workspace_id,
        root=resolved_root,
    )
    audit["paths"] = {
        "audit_latest_path": str(latest_path),
        "audit_run_path": str(run_path),
    }
    write_json(run_path, audit)
    write_json(latest_path, audit)
    return audit, latest_path, run_path


def _format_permission_line(item: Mapping[str, Any]) -> str:
    requested = item.get("requested")
    if requested is True:
        requested_text = "true"
    elif requested is False:
        requested_text = "false"
    else:
        requested_text = "missing/invalid"
    return f"- {item.get('permission')}: {requested_text} [{item.get('status')}] {item.get('reason')}"


def format_pack_inspection_report(inspection: Mapping[str, Any]) -> str:
    summary = inspection.get("manifest_summary") if isinstance(inspection.get("manifest_summary"), Mapping) else {}
    lines = [
        f"Satellite Pack: {summary.get('name') or 'unknown'} {summary.get('version') or ''}".rstrip(),
        f"Kind: {summary.get('kind') or 'unknown'}",
        f"Manifest: {inspection.get('manifest_path')}",
        f"Schema: {'valid' if inspection.get('schema_valid') else 'invalid'}",
        f"Inputs: {summary.get('input_count', 0)}",
        f"Outputs: {summary.get('output_count', 0)}",
        f"Recipes: {summary.get('recipe_count', 0)}",
        f"Widgets: {summary.get('widget_count', 0)}",
        "",
        "Permissions:",
    ]
    for item in inspection.get("permission_summary") or []:
        if isinstance(item, Mapping):
            lines.append(_format_permission_line(item))

    issues = inspection.get("validation_issues") or []
    if issues:
        lines.extend(["", "Validation Issues:"])
        for issue in issues:
            if isinstance(issue, Mapping):
                lines.append(f"- {issue.get('path')}: {issue.get('message')}")

    recipes = inspection.get("recipes") or []
    if recipes:
        lines.extend(["", "Recipes:"])
        for recipe in recipes:
            if isinstance(recipe, Mapping):
                lines.append(f"- {recipe.get('id')}: {recipe.get('step_count', 0)} step(s)")
    return "\n".join(lines)


def format_pack_audit_report(audit: Mapping[str, Any]) -> str:
    lines = [
        f"Pack Audit: {audit.get('verdict')}",
        f"Pack: {audit.get('pack_name') or 'unknown'} {audit.get('pack_version') or ''}".rstrip(),
        f"Kind: {audit.get('pack_kind') or 'unknown'}",
    ]
    paths = audit.get("paths") if isinstance(audit.get("paths"), Mapping) else {}
    if paths:
        lines.append(f"Latest: {paths.get('audit_latest_path')}")
        lines.append(f"Run: {paths.get('audit_run_path')}")

    blocked_reasons = audit.get("blocked_reasons") or []
    if blocked_reasons:
        lines.extend(["", "Blocked Reasons:"])
        for reason in blocked_reasons:
            lines.append(f"- {reason}")

    review_reasons = audit.get("review_reasons") or []
    if review_reasons:
        lines.extend(["", "Review Reasons:"])
        for reason in review_reasons:
            lines.append(f"- {reason}")

    lines.extend(["", "Permissions:"])
    for item in audit.get("permission_summary") or []:
        if isinstance(item, Mapping):
            lines.append(_format_permission_line(item))
    return "\n".join(lines)
