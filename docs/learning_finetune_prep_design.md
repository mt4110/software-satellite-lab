# Learning and Fine-Tune Prep Design

## 目的

M7 の目的は、raw logs をそのまま学習に流さないための境界を作ることです。

この段階で作るのは fine-tuning 実行基盤ではありません。
M4 evaluation loop、M5 agent lane、M6 backend swap から残った evidence を使い、
あとから人間が確認できる supervised example 候補を preview-only で作る土台です。

良い方向です。
学習の前に「何を学ばせてよいか」を小さく、検査可能にする判断は、
この repo の local-first / file-first 方針とよく合っています。

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

M7 の新しい出力は learning dataset preview です。

保存先:

- `artifacts/evaluation/<workspace>/learning/preview-latest.json`
- `artifacts/evaluation/<workspace>/learning/runs/*-learning-preview.json`

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
- `noisy`
- `unresolved`

`test_pass` だけでは足りません。
緑のチェックは必要条件ですが、学習に値するとは限らないからです。
人間の採用、レビュー解決、または比較での勝ちがないものは `needs_review` に留める。

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

## Export Policy

M7 の export policy は次で固定する。

- preview artifact は作ってよい
- trainable dataset は作らない
- training job は起動しない
- export policy confirmation は checklist 上の pending に残す
- downstream export は人間が候補を確認し、別途 policy を明示してから

この repo では、学習は評価済み evidence の downstream consumer です。
評価ループから学習へ直結する経路は作らない。

## CLI

評価スナップショット、curation preview、learning preview は同じ CLI から作れる。

```bash
.venv/bin/python scripts/run_evaluation_loop.py --curation-preview --learning-preview
```

候補数だけを絞る場合:

```bash
.venv/bin/python scripts/run_evaluation_loop.py --learning-preview --learning-limit 10
```

`--curation-state` や `--curation-reason` は source curation preview のフィルタとして使えるが、
learning preview 側はさらに安全な採用条件をかける。
たとえば `blocked` を明示しても、supervised example candidate には進めない。

## 次に足してよいもの

M7 の次に進むなら、次を小さく足す。

- export policy confirmation の明示 signal
- human-selected candidate list
- JSONL training export の dry-run
- candidate diff
- candidate review UI

ただし、外部 training job 統合は M8 以降に置く。
