# software-satellite-lab

[English README](./README_EN.md)

ソフトウェア開発特化のサテライトシステム実験室です。

このリポジトリは、新しい巨大モデルをいきなり作るための場所ではありません。
まずは、ソフトウェア制作を実務で前に進めるための外付け知能層を育てることを目的にしています。

## 目的

- レビューを強くする
- 設計と提案の質を上げる
- エージェント実行を扱いやすくする
- 評価ループを残す
- セッションをまたいで記憶を使えるようにする
- 将来的なモデル差し替えや学習基盤化につなげる

## 何を作るのか

中核にあるのは単体のモデルではなく、ソフトウェア開発のためのサテライトシステムです。

主な機能は次の5つです。

1. イベントと成果物の記録
2. 記憶と検索
3. レビューと提案
4. エージェント実行
5. 評価と学習準備

## 設計方針

- ローカルファースト
- file-first
- まずはシンプルな構成
- 証拠を残す
- モデルバックエンドは差し替え可能にする
- 学習は後段、評価付きデータが育ってから行う

## ストレージ方針

初期方針は次の通りです。

- 真実の層はローカルファイル
- 記憶索引は `SQLite` を第一候補にする
- 検索は lexical / structured retrieval を先に育てる
- vector search や重い基盤は必要が出てから足す

## 現在のベース資産

この repo には、再利用できる土台がすでにあります。

- shared runtime / session 管理
- file-based workspace / session state
- schema-versioned artifacts
- capability services
- thin local UI
- evaluation harness
- software-work event 正規化の土台
- SQLite memory index の土台

これらは捨てる前提ではなく、次の設計に引き継ぐ前提です。

## ドキュメント

- `README.md`: 日本語の概要
- `README_EN.md`: English overview
- `PLAN.md`: 再設計の設計図とマイルストーン
- `docs/recall_context_builder_design.md`: 最初に取り掛かる Recall / Context Builder 設計
- `docs/recall_hit_quality_loop.md`: Recall ヒット品質の可視化と軽量評価ループ
- `docs/learning_finetune_prep_design.md`: M7 Learning and Fine-Tune Prep の preview-only dataset 候補設計

## Git ルール

- feature branch は `feat/<topic>` で統一する
- branch 名、PR title、commit message にツール名 prefix を持ち込まない

## 新しく入った土台

- `scripts/software_work_events.py`
  - 既存の workspace/session 記録を software-work event に正規化
- `scripts/memory_index.py`
  - `SQLite FTS5` ベースのローカル memory index
- `scripts/rebuild_memory_index.py`
  - event log と memory index の再構築コマンド
- `scripts/evaluation_loop.py`
  - evaluation signal、comparison、curation filter、typed learning review queue、learning dataset preview、evaluation snapshot の土台
- `scripts/run_evaluation_loop.py`
  - CLI から evaluation snapshot と preview-only artifact を作成するコマンド
- `scripts/agent_lane.py`
  - bounded software task の task/run schema、tool trace、verification outcome を file-first に保存する M5 の土台
- `scripts/run_agent_lane.py`
  - CLI から patch-plan-verify task を記録し、verification trace を software-work event / evaluation loop に接続するコマンド
- `scripts/backend_swap.py`
  - backend adapter config、capability metadata、compatibility check、side-by-side harness を file-first に保存する M6 の土台
- `scripts/run_backend_swap.py`
  - 2つ以上の backend config を同じ agent-lane workflow に通し、memory index / evaluation comparison に接続するコマンド
- `scripts/dogfood_workflows.py`
  - recall / evaluation / curation preview を束ね、小さい software-work workflow を preview-only で反復する土台
- `scripts/run_dogfood_workflow.py`
  - review patch、proposal comparison、prior failure recall、decision explanation、resolved-work curation preview を CLI から起動するコマンド
- `scripts/satellite_pack.py`
  - Satellite Pack manifest の読み込み、v0 schema validation、permission audit artifact の土台
- `scripts/satlab.py`
  - `pack inspect` / `pack audit` を提供する薄い CLI

## セットアップ

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## よく使うコマンド

```bash
.venv/bin/python scripts/doctor.py
.venv/bin/python scripts/fetch_demo_assets.py
PYTHONPATH=scripts .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
PYTHONPATH=scripts .venv/bin/python -m py_compile scripts/*.py tests/*.py
.venv/bin/python scripts/rebuild_memory_index.py
.venv/bin/python scripts/run_recall_demo.py --list-requests
.venv/bin/python scripts/run_recall_demo.py --eval --miss-report
.venv/bin/python scripts/run_recall_demo.py --request-index 1
.venv/bin/python scripts/run_evaluation_loop.py
.venv/bin/python scripts/run_evaluation_loop.py --accept-candidate --source-event-id <event-id> --rationale "Accepted after review." --curation-preview
.venv/bin/python scripts/run_evaluation_loop.py --mark-review-resolved --source-event-id <event-id> --review-id <review-id> --resolution-summary "Review thread closed." --curation-preview
.venv/bin/python scripts/run_evaluation_loop.py --curation-preview --curation-state ready --curation-reason review_resolved --curation-limit 10
.venv/bin/python scripts/run_evaluation_loop.py --curation-preview --learning-preview --learning-limit 10
.venv/bin/python scripts/run_dogfood_workflow.py --workflow-kind review_patch --source-event-id <event-id> --query "review patch risk and verification"
.venv/bin/python scripts/run_dogfood_workflow.py --workflow-kind compare_proposals --candidate-event-id <event-a> --candidate-event-id <event-b> --winner-event-id <event-a>
.venv/bin/python scripts/run_dogfood_workflow.py --workflow-kind resolved_work_curation_preview --curation-reason review_resolved --curation-limit 10
.venv/bin/python scripts/run_agent_lane.py --task-title "Patch smoke" --goal "Run a bounded patch-plan-verify loop." --plan-step "Inspect scope." --plan-step "Run verification." --verification-command ".venv/bin/python -m unittest tests.test_agent_lane"
.venv/bin/python scripts/run_backend_swap.py --list-backends
.venv/bin/python scripts/run_backend_swap.py --task-title "Backend swap smoke" --goal "Run the same workflow across local backend configs." --plan-step "Load backend config." --plan-step "Run verification." --verification-command ".venv/bin/python -m unittest tests.test_backend_swap"
.venv/bin/python scripts/satlab.py pack inspect templates/review-risk-pack.satellite.yaml
.venv/bin/python scripts/satlab.py pack audit templates/review-risk-pack.satellite.yaml
.venv/bin/python scripts/run_local_ui.py
.venv/bin/python scripts/run_capability_matrix.py --smoke
```
