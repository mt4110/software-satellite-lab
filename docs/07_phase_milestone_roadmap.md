# 07. Phase and Milestone Roadmap

## Overview

このロードマップは、派手さではなく **ブレなさ** を優先する。

最初の 12 週間は、AI coding agent を増やす期間ではない。

```text
Evidence ledger kernel を固める
Satellite Evidence Pack v0 を固める
review-risk-pack demo を通す
human-gated learning-candidate inspection を守る
```

## Phase 0 — Doctrine Freeze / Rename Closeout

期間目安: 1 week

目的:

```text
maakie-brainlab から software-satellite-lab への思想移行を固定する。
```

Deliverables:

```text
- docs/00_core_anchor.md
- docs/01_product_doctrine.md
- README 冒頭の新定義
- no-go list
- old direction の closeout note
```

Done when:

```text
[ ] README が AI coding agent と誤読されない
[ ] product doctrine に What this is / is not がある
[ ] no-go list が merge されている
```

## Phase 1 — Evidence Ledger Kernel Hardening

期間目安: 2 weeks

目的:

```text
software-work event / evaluation signal / comparison / source path の contract を固める。
```

Deliverables:

```text
- schemas/software_work_event.schema.json
- schemas/evaluation_signal.schema.json
- schemas/comparison_record.schema.json
- event contract check
- missing_source handling
- evaluation snapshot consistency check
```

Done when:

```text
[ ] source path missing candidate が positive evidence にならない
[ ] index が file artifact から rebuild できる
[ ] stale positive signal より新しい negative signal が説明される
```

## Phase 2 — Satellite Evidence Pack v0 Contract

期間目安: 2 weeks

目的:

```text
Pack を arbitrary plugin ではなく declarative evidence workflow として固定する。
```

Deliverables:

```text
- docs/03_satellite_pack_system.md
- schemas/satellite_evidence_pack.schema.json
- satlab pack inspect
- satlab pack audit
- v0 permission model
```

Done when:

```text
[ ] pack manifest が validate できる
[ ] permission audit artifact が出る
[ ] run_command/network/secrets/write_repo が default deny
[ ] pack output が core schema に乗る
```

## Phase 3 — review-risk-pack Public Demo

期間目安: 2 weeks

目的:

```text
最初の公開デモを、agent execution ではなく AI Coding Flight Recorder / failure-memory review として見せる。
```

Deliverables:

```text
- review-risk-pack manifest
- patch input resolver
- recall similar failures
- risk note artifact
- human verdict request
- curation / learning-candidate inspection connection
```

Done when:

```text
[ ] demo patch で risk note が出る
[ ] similar failure or miss reason が出る
[ ] human verdict なしで learning ready にならない
[ ] CLI で完走する
```

## Phase 4 — backend-compare-pack

期間目安: 2 weeks

目的:

```text
model/provider 対応数ではなく、比較 evidence の質を見せる。
```

Deliverables:

```text
- backend-compare-pack manifest
- comparison criteria
- backend role metadata
- winner / loser / none
- human rationale required
```

Done when:

```text
[ ] 2 backend の出力比較が comparison_record に残る
[ ] winner だけでなく loser / none / excluded reason が残る
[ ] comparison winner が source path なしで learning candidate にならない
```

## Phase 5 — Thin Satellite Console

期間目安: 2 weeks

目的:

```text
大きな dashboard ではなく、判断に必要な4画面を作る。
```

Deliverables:

```text
- Work Queue
- Evidence View
- Recall View
- Compare View
- read-only widgets
```

Done when:

```text
[ ] human review queue が見える
[ ] source path へ戻れる
[ ] blocked reason / next action が見える
[ ] learning-candidate inspection queue が preview-only と分かる
```

## Phase 6 — Public Dogfood / OSS Onboarding

期間目安: 3 weeks

目的:

```text
他人が試せる最低限の物語、issue、pack contribution path を作る。
```

Deliverables:

```text
- public demo script
- example artifacts
- contributor guide for Pack
- sponsor narrative
- issue templates
```

Done when:

```text
[ ] clone して demo を再現できる
[ ] contributor が Pack を追加する入口を理解できる
[ ] sponsor 向け README section がある
```

## Milestone Gates

| Milestone | Gate | Stop condition |
|---|---|---|
| M0 Doctrine | README が agent 競争に見えない | “best coding agent” と説明されるなら未完 |
| M1 Evidence | source path に戻れる | missing source が positive 扱いなら停止 |
| M2 Pack | permission audit がある | arbitrary runtime が入るなら停止 |
| M3 Demo | review-risk flow が通る | backend/provider 数が主役なら停止 |
| M4 Compare | human verdict が残る | model output を正解扱いするなら停止 |
| M5 UI | 4画面で判断できる | dashboard が肥大化するなら停止 |
| M6 OSS | Pack contribution が可能 | marketplace を始めるなら停止 |

## Success Metrics

```text
- recall_source_hit_rate
- source_missing_rate
- human_verdict_rate
- comparison_record_count
- learning_candidate_blocked_reason_count
- repeated_failure_rate
- review_risk_pack_dogfood_count
- curation_ready_vs_blocked_ratio
```

## 12-week Outcome

12週間後に目指す状態:

```text
software-satellite-lab は、AI coding agent ではなく、AI-assisted software work の AI Coding Flight Recorder / evidence ledger であると説明できる。
review-risk-pack demo が動き、過去失敗 recall、backend comparison、human verdict、learning-candidate inspection が一連で見える。
Pack は安全な declarative workflow として追加できる。
```
