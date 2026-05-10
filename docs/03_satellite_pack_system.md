# 03. Satellite Pack System

## 1. Why Satellite Pack?

`plugin` と呼ぶと、自由な実行コードや marketplace を連想し、設計が危険に広がる。

`Satellite Pack` は違う。

```text
Satellite Pack = software-work evidence OS に安全に接続できる、宣言的な workflow / evidence / recall / widget / evaluation bundle
```

目的は「何でもできる拡張」ではなく、**evidence を壊さず、判断を増やす拡張**。

## 2. Pack Philosophy

```text
declarative first
read-only first
local-first
artifact-first
permission-denied by default
human-gated
no arbitrary code in v0
```

## 3. Pack Types

| Pack kind | 目的 | v0 |
|---|---|---|
| workflow_pack | review / compare / failure-analysis などの手順 | Yes |
| recall_pack | task kind ごとの recall policy | Yes |
| evaluation_pack | check criteria / comparison criteria | Yes |
| widget_pack | read-only artifact view | Yes |
| evidence_adapter_pack | 外部ログを software-work event へ正規化 | Later, not valid in v0 schema |
| backend_adapter_pack | live model / agent adapter | Later, not valid in v0 schema |
| executable_pack | shell / Python / JS 実行を含む | No in v0, not valid in v0 schema |

## 4. Pack Manifest

v0 manifest schema で有効な `kind` は、`workflow_pack`、`recall_pack`、
`evaluation_pack`、`widget_pack` のみ。
adapter / executable 系の Pack kind は将来検討対象だが、v0 では有効な `kind` 値ではない。

最小 manifest:

```yaml
schema_name: software-satellite-pack
schema_version: 1
name: review-risk-pack
version: 0.1.0
kind: workflow_pack
summary: Recall similar failures and produce patch risk evidence.

inputs:
  - git_diff
  - software_work_events
  - memory_index

outputs:
  - review_note
  - similar_failure_bundle
  - risk_evidence_bundle
  - human_verdict_request
  - learning_preview_candidate_or_blocked

permissions:
  read_repo: true
  write_repo: false
  read_artifacts: true
  write_artifacts: true
  read_memory_index: true
  write_evaluation_signal: false
  request_human_verdict: true
  run_command: false
  network: false
  secrets: false
  use_backend: false

recipes:
  - id: patch_risk_review
    steps:
      - resolve_patch_input
      - recall_similar_failures
      - build_risk_note
      - write_evidence_bundle
      - request_human_verdict
      - update_curation_preview

widgets:
  - evidence_path_card
  - similar_failures_card
  - blocked_reason_card
  - human_verdict_card
```

## 5. Pack Runtime Pipeline

```text
satlab pack run <pack>
  ↓
load manifest
  ↓
audit permissions
  ↓
resolve inputs
  ↓
execute declarative recipe
  ↓
write pack run artifact
  ↓
append software-work events
  ↓
append evaluation / comparison suggestions
  ↓
render read-only widgets
  ↓
wait for human verdict
```

## 6. v0 Pack Runtime Restrictions

v0 では、以下を禁止する。

```text
- arbitrary Python
- arbitrary JavaScript
- arbitrary shell
- network access
- secrets access
- repo write
- background daemon
- remote marketplace install
```

許可するもの:

```text
- YAML manifest
- recipe definition
- prompt / instruction template
- recall policy
- evaluation criteria
- read-only widget definition
- artifact transform using core-owned code
```

## 7. First Three Packs

### 7.1 review-risk-pack

```text
patch を読む
過去の類似失敗を recall する
risk note を作る
human verdict を要求する
curation / learning preview に接続する
```

### 7.2 backend-compare-pack

```text
同じ task を backend A / B に流す
output / verification / failure を比較する
winner / loser / none を human verdict 付きで保存する
```

### 7.3 failure-memory-pack

```text
test failure / failed run から過去の修正 pattern を recall する
repair candidate を出す
unresolved / resolved を保存する
```

## 8. Pack Quality Checklist

Pack は merge 前に次を満たす。

```text
[ ] manifest が schema valid
[ ] permission が最小
[ ] source artifact path を失わない
[ ] outputs が core schema に準拠
[ ] human verdict が必要な箇所で bypass されない
[ ] learning preview に raw output を入れない
[ ] negative evidence / exclusion reason を保存する
[ ] CLI で実行できる
[ ] read-only widget で inspect できる
[ ] tests / golden artifact がある
```

## 9. Pack Install Policy

v0:

```text
satlab pack inspect ./packs/review-risk-pack
satlab pack audit ./packs/review-risk-pack
satlab pack run ./packs/review-risk-pack --patch changes.diff
```

No marketplace.
No remote install.
No auto-update.

## 10. Why This Is Unique

競合の拡張は、多くの場合「agent に何かをさせる」方向へ進む。

Satellite Pack は、「agent の作業を evidence OS に残す」方向へ進む。

この違いを守る。
