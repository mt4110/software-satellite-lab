# 04. Security and Permission Model

## 1. Security Principle

`software-satellite-lab` は、agent platform ではなく evidence OS である。

したがって v0 の安全原則は次。

```text
observe more than execute
record more than mutate
denied by default
human gate before irreversible action
```

## 2. Threat Model

想定する危険:

| Threat | 例 | v0 対策 |
|---|---|---|
| Malicious pack | repo を書き換える | write_repo deny |
| Prompt injection | pack が hidden instruction を実行する | declarative-only, no arbitrary runtime |
| Secret exfiltration | API key を読む | secrets deny |
| Destructive command | rm / deploy / DB 操作 | run_command deny |
| Network leak | evidence を外へ送る | network deny |
| Training contamination | raw output が学習候補になる | learning preview gate |
| False positive evidence | source path がない winner | missing_source block |

## 3. Permission Classes

```yaml
permissions:
  read_repo: false
  write_repo: false
  read_artifacts: true
  write_artifacts: true
  run_command: false
  network: false
  secrets: false
  use_backend: false
  read_memory_index: true
  write_evaluation_signal: false
  request_human_verdict: true
```

## 4. Manifest Permission Defaults

| Permission | Default | v0 note |
|---|---:|---|
| read_artifacts | allow | core value |
| write_artifacts | allow | sandboxed path only |
| read_memory_index | allow | no external transfer |
| read_repo | prompt / manifest explicit | patch review requires it |
| write_repo | deny | not in v0 |
| write_evaluation_signal | deny | write suggestion artifacts, not final signals |
| request_human_verdict | allow | required for adoption / curation decisions |
| run_command | deny | only core-owned verification runner later |
| network | deny | no marketplace / exfiltration |
| secrets | deny | no exception in v0 |
| use_backend | deny | no live backend invocation in v0 packs |

These are the only permission keys declared by the v0 manifest schema.
The denied v0 permissions are schema-enforced as `false`, not merely documented as defaults.

## 5. Runtime Restrictions

The following are not manifest permissions in v0.
They are runtime capabilities that remain unavailable regardless of manifest content.

| Capability | v0 policy | Note |
|---|---:|---|
| custom_js_widget | deny | use schema-rendered widgets |
| arbitrary_python | deny | no executable pack in v0 |
| arbitrary_javascript | deny | no executable pack in v0 |
| arbitrary_shell | deny | no executable pack in v0 |

## 6. Pack Audit Output

`pack audit` は、人間が読むための artifact を出す。

```json
{
  "schema_name": "software-satellite-pack-audit",
  "schema_version": 1,
  "pack_name": "review-risk-pack",
  "verdict": "pass|needs_review|block",
  "permission_summary": [],
  "blocked_reasons": [],
  "human_review_required": true,
  "source_paths": []
}
```

## 7. Human Gate Policy

次は必ず human gate を要求する。

```text
- accepted / rejected signal の確定
- comparison winner の確定
- learning preview candidate の昇格
- export policy confirmation
- repo write / patch apply
- shell command execution
```

## 8. Widget Safety

v0 widget は custom JS を禁止する。

Widget は JSON artifact を読み、core UI が定義済みコンポーネントで描画する。

```text
widget definition
  ↓
artifact query
  ↓
core renderer
```

## 9. Future Sandbox Levels

v0 では実行コードを避ける。

将来必要になった場合のみ、段階的に許可する。

| Level | 内容 | 解禁条件 |
|---|---|---|
| L0 | declarative-only | v0 |
| L1 | core-owned transform only | tests / audit ready |
| L2 | allowlisted command | local sandbox + explicit human approval |
| L3 | executable pack | signing / sandbox / rollback / logs required |

## 10. Security Non-Negotiables

```text
- secrets は v0 で触らない
- network は v0 で触らない
- marketplace は v0 で作らない
- custom JS は v0 で許可しない
- pack は repo を直接書き換えない
- source artifact path を持たない evidence は learning candidate にしない
```
