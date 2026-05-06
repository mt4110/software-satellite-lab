# Learning and Fine-Tune Prep Design

## 目的

M7 の目的は、raw logs をそのまま学習に流さないための境界を作ることです。

この段階で作るのは fine-tuning 実行基盤ではありません。
M4 evaluation loop、M5 agent lane、M6 backend swap から残った evidence を使い、
あとから人間が確認できる supervised example 候補を preview-only で作る土台です。

良い方向です。
学習の前に「何を学ばせてよいか」を小さく、検査可能にする判断は、
この repo の local-first / file-first 方針とよく合っています。

## 設計姿勢

M7 は、汎用 agent platform や大きな cockpit を作る milestone ではありません。
software-work satellite として、日々の設計、実装、レビュー、評価から出た
小さな evidence を積み上げ、あとで人間が検査できる learning-prep queue を作る milestone です。

中心に置く判断:

- orchestration の見た目より、replay できる evidence path を優先する
- hidden autonomy より、typed state と next action が読める artifact を優先する
- raw trace の量より、accept / reject / resolve / compare の outcome-linked signal を優先する
- heavy dashboard より、file-first artifact と薄い CLI / Local UI report を優先する
- vector-heavy retrieval や training integration より、lexical / structured / SQLite-rebuildable な検査可能性を優先する

ここを間違えると、M7 は「学習準備」ではなく「ログを集める装置」になります。
それはこの repo の強みではありません。
M7 の良さは、学習に値するかどうかを急がず、証拠と状態をそろえてから前に進める点です。

## 非目的

M7 では次をしない。

- raw logs の一括 training export
- 外部 training job の起動
- LoRA / SFT の実行
- 自動採用された dataset による silent feedback loop
- rejected / noisy / unresolved evidence の既定採用

ここを曖昧にすると、評価ループがそのまま学習汚染の入口になります。
それははっきり間違いです。
M7 の責務は、学習候補を増やすことではなく、
候補にしてよい evidence だけを説明可能に残すことです。

## 入力

M7 の入力は、既存の file-first artifact に限定する。

- software-work event log
  - `artifacts/event_logs/<workspace>/software_work_events.json`
- evaluation signal log
  - `artifacts/evaluation/<workspace>/signals.jsonl`
- evaluation comparison log
  - `artifacts/evaluation/<workspace>/comparisons.jsonl`
- evaluation snapshot
  - `artifacts/evaluation/<workspace>/snapshots/latest.json`
  - `artifacts/evaluation/<workspace>/snapshots/runs/*.json`
- curation preview
  - `artifacts/evaluation/<workspace>/curation/preview-latest.json`
  - `artifacts/evaluation/<workspace>/curation/runs/*.json`
- agent-lane / backend-swap source artifacts
  - event の `source_refs.artifact_ref.artifact_path` から辿る

SQLite memory index は source event を作るための rebuildable cache として使う。
training candidate の真実の層は、引き続きローカルファイルです。

## 出力

M7 の主な出力は learning dataset preview です。

保存先:

- `artifacts/evaluation/<workspace>/learning/preview-latest.json`
- `artifacts/evaluation/<workspace>/learning/runs/*-learning-preview.json`

M7.3 では、人間が明示的に選んだ候補リストも preview-only artifact として残せる。

保存先:

- `artifacts/evaluation/<workspace>/learning/human-selected-latest.json`
- `artifacts/evaluation/<workspace>/learning/runs/*-human-selected-candidates.json`

M7.4 では、JSONL training export の dry-run も preview-only artifact として残せる。
これは manifest / validation report だけで、本物の `.jsonl` は作らない。

保存先:

- `artifacts/evaluation/<workspace>/learning/jsonl-export-dry-run-latest.json`
- `artifacts/evaluation/<workspace>/learning/runs/*-jsonl-export-dry-run.json`

この artifact は preview-only で、trainable dataset ではない。
`training_export_ready` は常に `false` で、`human_gate_required` は常に `true` です。

## 候補の採用条件

supervised example candidate に進めるには、curation preview 上で次をすべて満たす必要がある。

- `state = ready`
- `export_decision = include_when_approved`
- `ready_for_policy = true`
- `test_pass` がある
- `accepted`、`review_resolved`、`comparison_winner` のいずれかがある
- blocking reason がない

既定で除外する reason:

- `rejected`
- `review_unresolved`
- `test_fail`
- `failed`
- `noisy`
- `unresolved`

`test_pass` だけでは足りません。
緑のチェックは必要条件ですが、学習に値するとは限らないからです。
人間の採用、レビュー解決、または比較での勝ちがないものは `needs_review` に留める。

## 状態と review queue の境界

M7 では、候補を単に include / exclude に潰さない。
source event、curation candidate、evaluation signal から読める状態を保ったまま、
人間が次に何を判断すべきかが分かる queue にする。

最低限、次の状態は区別して扱う。

- `ready`: test pass と positive selection signal があり、policy 確認待ち
- `needs_review`: test pass はあるが、人間の採用、review resolution、comparison winner が足りない
- `blocked`: rejected、review unresolved、test fail、failed、noisy、unresolved などが残っている
- `missing_source`: source event や durable curation preview に戻れない
- `missing_supervised_text`: instruction / response excerpt が作れない

M7 の実装では、これらを新しい巨大な state machine として増やす必要はない。
まずは既存の `source_event.status`、`quality_status`、`execution_status`、
`curation.state`、`blocked_by`、`required_next_steps`、`excluded_by` から読める形にする。

失敗、blocked、unresolved は捨てる対象ではなく、学習候補から除外された理由として残す対象です。
ただし supervised example candidate には入れない。
これは、失敗から学ぶ余地を残しながら、training data を汚さないための線引きです。

M7.1 の最小実装では、learning preview に `review_queue` を持たせる。
これは supervised candidate 一覧とは別に、source curation candidate 全体を検査するための queue です。

各 queue item は最低限、次を持つ。

- `queue_state`
  - `ready`
  - `needs_review`
  - `blocked`
  - `missing_source`
  - `missing_supervised_text`
- `queue_priority`
  - blocked / missing source / missing supervised text を先に見る
  - ready だが policy 未確認のものを次に見る
  - test signal や human selection が足りないものをその次に見る
- `next_action`
- `blocked_reason`
- `lifecycle_summary`
  - `test_state` / `review_state` / `selection_state` は curation reason だけでなく trace evidence を優先する
  - reason 上は `test_pass` / `review_resolved` / selected でも、対応する trace が辿れない場合は `missing_trace` として扱う
  - latest trace が `test_fail` / `review_unresolved` / `rejection` の場合は、その negative trace を lifecycle に反映する
- `eligible_for_supervised_candidate`
- `excluded_by`

これにより、learning preview は「採用候補だけの一覧」ではなく、
人間が判断すべき状態と次の行動を読める review queue になる。

supervised example candidate に進む場合は、curation preview の reason だけでなく、
実際の signal / comparison trace 上でも `test_pass` と positive selection evidence が辿れることを確認する。
stale な preview に後続の rejection、latest `review_unresolved`、`test_fail` が残っている場合は、
supervised candidate から外し、queue 上の blocked reason として残す。
特に `accepted` / `review_resolved` / `comparison_winner` として分類された候補は、
その分類に対応する trace が辿れない場合、`needs_review` の診断として残す。

## 最小 supervised example schema

候補 1 件は次の shape を持つ。

```json
{
  "schema_name": "software-satellite-supervised-example-candidate",
  "schema_version": 1,
  "candidate_id": "local-default:supervised-example-candidate:<digest>",
  "workspace_id": "local-default",
  "event_id": "<source-event-id>",
  "example_kind": "software_work_supervised_candidate",
  "source_event": {
    "source_event_id": "<source-event-id>",
    "event_kind": "agent_task_run",
    "recorded_at_utc": "2026-01-01T00:00:00+00:00",
    "session_surface": "agent_lane",
    "session_mode": "patch_plan_verify",
    "quality_status": "pass",
    "execution_status": "ok",
    "artifact_path": "artifacts/agent_lane/..."
  },
  "supervised_example": {
    "format": "instruction_response",
    "instruction": "bounded task goal or resolved prompt",
    "response": "accepted output summary",
    "context": {
      "validation_mode": "agent_lane",
      "validation_command": "python -m unittest ...",
      "pass_definition": "All verification commands exit with status 0."
    }
  },
  "curation": {
    "state": "ready",
    "reasons": ["accepted", "test_pass"],
    "export_decision": "include_when_approved",
    "ready_for_policy": true
  },
  "evidence": {
    "signals": [],
    "comparisons": []
  },
  "backend_metadata": {
    "backend_id": "mock-careful-local",
    "adapter_kind": "mock",
    "model_id": "mock/careful-local-v1",
    "metadata": {}
  },
  "review_queue": {
    "queue_state": "ready",
    "queue_priority": {
      "rank": 2,
      "bucket": "ready_policy_unconfirmed"
    },
    "next_action": "confirm_export_policy",
    "blocked_reason": null,
    "eligible_for_supervised_candidate": true,
    "excluded_by": [],
    "lifecycle_summary": {
      "test_state": "passed",
      "selection_state": "selected",
      "policy_state": "pending_confirmation"
    }
  },
  "source_paths": {
    "event_log_path": "...",
    "signal_log_path": "...",
    "comparison_log_path": "...",
    "source_artifact_path": "..."
  },
  "policy": {
    "export_mode": "preview_only",
    "human_gate_required": true,
    "training_job_allowed": false,
    "raw_log_export_allowed": false
  }
}
```

`instruction` と `response` は preview 用に excerpt される。
raw log dump を学習候補として保存する設計にはしない。

## Traceability

各候補は次へ戻れる必要がある。

- source event
  - event id、event kind、session surface、quality status
- signal
  - acceptance / review_resolved / test_pass などの signal id、origin、evidence
- comparison
  - comparison id、winner、criteria、rationale
- backend metadata
  - backend id、adapter kind、model id、compatibility、limits、metadata
- file path
  - event log、signal log、comparison log、source artifact

M6 backend swap 由来の agent-lane event は、source artifact の run JSON から backend metadata を補完する。
software-work event だけに閉じると metadata が薄くなるため、ここは artifact path を辿る。

Traceability の目的は、あとから「なぜこの候補が学習候補になったのか」を説明できることです。
そのため、候補 artifact は answer だけではなく、採用理由、除外理由、比較上の役割、
backend の由来、source artifact path を一緒に持つ。

逆に、辿れない候補は品質が高そうに見えても採用しない。
durable source に戻れない learning data は、local-first な改善ループの外に落ちるからです。

## Export Policy

M7 の export policy は次で固定する。

- preview artifact は作ってよい
- trainable dataset は作らない
- training job は起動しない
- export policy confirmation は明示 signal が記録されるまで checklist 上の pending に残す
- downstream export は人間が候補を確認し、別途 policy を明示してから

この repo では、学習は評価済み evidence の downstream consumer です。
評価ループから学習へ直結する経路は作らない。

## M7.2 Export Policy Confirmation

M7.2 では、export policy confirmation を silent state ではなく明示 signal として残す。

保存先は既存の evaluation signal log です。

- `artifacts/evaluation/<workspace>/signals.jsonl`

signal kind:

- `export_policy_confirmed`

この signal は、candidate の採用や選抜を意味しない。
`accepted` / `review_resolved` / `comparison_winner` / `test_pass` の traceability を置き換えず、
ready candidate に対して人間が preview-only policy を確認した evidence だけを足す。

最低限の evidence:

```json
{
  "confirmation_scope": "learning_dataset_preview_candidate",
  "policy_version": "m7-preview-only-v1",
  "export_mode": "preview_only",
  "training_export_ready": false,
  "human_gate_required": true,
  "training_job_allowed": false,
  "raw_log_export_allowed": false,
  "downstream_export_requires_separate_approval": true
}
```

この signal があっても、learning preview は引き続き次を維持する。

- `training_export_ready = false`
- `human_gate_required = true`
- raw log export は不可
- training job 起動は不可
- JSONL training export は不可

review queue では `policy_state = confirmed` として読めるようにし、
priority bucket は `ready_policy_confirmed` に移す。
ただし inspection queue では、未確認の ready candidate を先に見るため、
`ready_policy_unconfirmed` は rank 2、`ready_policy_confirmed` は rank 3 とする。
次 action は downstream export の別判断に留める。
policy confirmation は「学習に進める許可」ではなく、
「preview-only 境界を人間が確認した証跡」です。

## M7.3 Human-Selected Candidate List

M7.3 では、learning preview / review queue を見た人間が、
候補 event id を明示的に選んだ事実だけを file-first artifact として残す。

これは supervised candidate の採用条件を置き換えない。
`accepted` / `review_resolved` / `comparison_winner` / `test_pass` /
`export_policy_confirmed` の traceability は、引き続き learning preview 側の evidence から読む。
human-selected list は、それらを silent に補完したり、未 ready 候補を昇格したりしない。

最低限の shape:

```json
{
  "schema_name": "software-satellite-human-selected-candidate-list",
  "schema_version": 1,
  "workspace_id": "local-default",
  "export_mode": "preview_only",
  "training_export_ready": false,
  "human_gate_required": true,
  "source_learning_preview_path": "artifacts/evaluation/local-default/learning/runs/...",
  "selection": {
    "origin": "cli",
    "selection_mode": "explicit_human_candidate_list",
    "selected_event_ids": ["<source-event-id>"],
    "rationale": "Human-selected shortlist."
  },
  "selected_candidates": [
    {
      "event_id": "<source-event-id>",
      "preview_membership": {
        "in_review_queue": true,
        "in_supervised_example_candidates": true,
        "in_excluded_candidates": false
      },
      "eligible_for_supervised_candidate": true,
      "queue_state": "ready",
      "next_action": "review_downstream_export_policy",
      "evidence_summary": {
        "signal_ids": ["local-default:eval:..."],
        "comparison_ids": [],
        "export_policy_confirmation_signal_id": "local-default:eval:...",
        "traceability": {
          "test_pass": true,
          "accepted": true,
          "review_resolved": false,
          "comparison_winner": false,
          "export_policy_confirmed": true
        }
      },
      "policy": {
        "export_mode": "preview_only",
        "training_export_ready": false,
        "human_gate_required": true,
        "training_job_allowed": false,
        "raw_log_export_allowed": false,
        "downstream_export_requires_separate_approval": true
      }
    }
  ]
}
```

選ばれた event id が learning preview に存在しない場合も、捨てずに
`missing_learning_preview_candidate` として artifact に残す。
存在しているが `blocked` / `needs_review` の候補も、
`eligible_for_supervised_candidate=false` のまま残す。

大事なのは、人間の shortlist が「次に見る対象」を示すだけで、
training export の許可や JSONL 生成を意味しない点です。
ここを混ぜると silent feedback loop になります。
それは M7 の境界として間違いです。

## M7.4 JSONL Training Export Dry-Run

M7.4 では、learning preview または human-selected candidate list から、
「もし将来 JSONL training export を別 approval で作るなら、どの候補が対象になりうるか」
だけを file-first artifact として検証する。

これは export ではない。
本物の `.jsonl`、trainable dataset、外部 training job は作らない。
dry-run artifact では次を明示する。

```json
{
  "schema_name": "software-satellite-jsonl-training-export-dry-run",
  "schema_version": 1,
  "export_mode": "preview_only",
  "artifact_kind": "jsonl_training_export_dry_run_manifest",
  "training_export_ready": false,
  "human_gate_required": true,
  "not_trainable": true,
  "source_mode": "human_selected_candidate_list",
  "counts": {
    "future_jsonl_candidate_if_separately_approved_count": 1,
    "would_write_jsonl_record_count": 0,
    "supervised_example_text_copied_count": 0,
    "raw_log_text_copied_count": 0
  }
}
```

candidate ごとの record でも、`would_write_jsonl_record=false` を維持する。
対象になりうる候補は
`dry_run_eligible_for_future_export_if_separately_approved=true`
として見えるが、これは training export approval ではない。
別の downstream approval と M8 の training job design が必要です。

human-selected candidate list を入力にした場合も、選択は昇格条件ではない。
`accepted` / `review_resolved` / `comparison_winner` / `test_pass` /
`export_policy_confirmed` の traceability は、元の learning preview evidence から読む。
missing / blocked / needs_review の候補は、dry-run でも blocked のまま残す。

## CLI

評価スナップショット、curation preview、learning preview は同じ CLI から作れる。

```bash
.venv/bin/python scripts/run_evaluation_loop.py --curation-preview --learning-preview
```

候補数だけを絞る場合:

```bash
.venv/bin/python scripts/run_evaluation_loop.py --learning-preview --learning-limit 10
```

export policy confirmation を記録する場合:

```bash
.venv/bin/python scripts/run_evaluation_loop.py \
  --confirm-export-policy \
  --source-event-id <source-event-id> \
  --rationale "Human confirmed the preview-only export policy." \
  --learning-preview
```

これは `signals.jsonl` に `export_policy_confirmed` を追記し、learning preview を更新するだけです。
trainable dataset、JSONL training export、外部 training job は作らない。

human-selected candidate list を記録する場合:

```bash
.venv/bin/python scripts/run_evaluation_loop.py \
  --human-selected-candidates \
  --select-candidate-event-id <source-event-id> \
  --rationale "Human selected this candidate for preview inspection."
```

これは learning preview を source artifact として参照し、選択された event id の shortlist を書くだけです。
supervised example の本文はコピーせず、trainable dataset、JSONL training export、外部 training job は作らない。

JSONL training export dry-run を記録する場合:

```bash
.venv/bin/python scripts/run_evaluation_loop.py \
  --human-selected-candidates \
  --select-candidate-event-id <source-event-id> \
  --jsonl-export-dry-run \
  --rationale "Inspect selected candidates without writing JSONL."
```

これは human-selected candidate list または learning preview を source artifact として参照し、
dry-run manifest / validation report を書くだけです。
`.jsonl`、trainable dataset、外部 training job は作らない。

`--curation-state` や `--curation-reason` は source curation preview のフィルタとして使えるが、
learning preview 側はさらに安全な採用条件をかける。
たとえば `blocked` を明示しても、supervised example candidate には進めない。

## Operator UX

M7 の report / Local UI は、派手な dashboard ではなく inspection queue として振る舞う。
人間が短時間で見るべきものは次です。

- candidate が何件あり、何件が supervised example candidate になったか
- 除外理由の内訳
- candidate ごとの source event、positive signal、comparison role、backend
- 次に必要な action
  - export policy confirmation
  - human selection
  - unresolved review の解決
  - failed verification の修正
  - source artifact の復旧
- queue state / next action / blocked reason の内訳

UI が価値を持つのは、データの真実を隠すときではなく、
file-first artifact の inspection を速くするときです。
Local UI を足す場合も、同じ JSON artifact を読む薄い表示に留める。

## 次に足してよいもの

M7 の次に進むなら、次を小さく足す。

- export policy confirmation の明示 signal
  - M7.2 で `export_policy_confirmed` として追加済み
- human-selected candidate list
  - M7.3 で preview-only artifact として追加済み
- JSONL training export の dry-run
  - M7.4 で dry-run manifest / validation report として追加済み
- candidate diff
- candidate review UI
- typed lifecycle summary
  - queued / running / blocked / paused / completed / failed / cancelled
- blocked reason と next action の明示 field
- review queue priority
  - blocked first
  - ready but policy-unconfirmed second
  - ready policy-confirmed third
  - needs human selection fourth

ただし、外部 training job 統合は M8 以降に置く。
