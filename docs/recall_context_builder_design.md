# Recall / Context Builder Design

## 目的

この設計は、`software-satellite-lab` が最初に本格実装へ進むべき `M3: Recall / Context Builder` を定義する。

狙いは単純です。

- 記憶をただ保存するだけで終わらせない
- その場の仕事に効く形で思い出せるようにする
- レビュー、設計、提案、失敗分析に使える文脈を安定して組み立てる

いまの repo にはすでに次の土台がある。

- `scripts/software_work_events.py`
- `scripts/memory_index.py`
- `scripts/rebuild_memory_index.py`

なので、次の一手は「保存方法」ではなく「取り出し方」を設計するのが自然。

## なぜ最初にこれをやるのか

今のロードマップ順でいえば、M1 と M2 は土台が入り始めている。
その次に必要なのは、実務で効く検索と文脈化の筋道です。

これがないと:

- 記憶があっても使われない
- エージェントが毎回ゼロから考える
- 評価データが蓄積されても再利用されない
- 後段の LoRA / SFT 候補抽出も弱くなる

つまり、Recall / Context Builder は「使い物になるサテライトシステム」の最初の関門です。

## 非目的

この設計では次をやらない。

- vector search を最初から必須にする
- 複雑な re-ranker モデルを先に入れる
- 自動学習や自動プロンプト最適化まで踏み込む
- 巨大な graph DB や外部検索基盤を入れる
- UI を先に広げる

まずはローカルで速く、見通しがよく、直せる設計を優先する。

## 対象ユースケース

最初の対象は4つに絞る。

1. Review Recall
   - 類似レビュー
   - 過去の修正パターン
   - 失敗しやすい論点

2. Design Recall
   - 過去の設計判断
   - 似たモジュール構成
   - 採用された方針と却下された方針

3. Proposal Recall
   - 似た実装提案
   - 受け入れられた提案文
   - テストや検証付きの提案

4. Failure Analysis Recall
   - quality_fail
   - blocked
   - failed
   - repair needed

## 入出力

### Input

Recall 層は次の情報を受け取る。

- task kind
  - `review`
  - `design`
  - `proposal`
  - `failure_analysis`
- user prompt
- optional file paths
- optional surface
- optional model/backend hint
- optional status filters

### Output

出力は検索結果の羅列ではなく、文脈束にする。

最低限ほしい出力:

- selected candidates
- why each candidate was chosen
- grouped context blocks
- omitted-but-relevant count
- token/character budget metadata

## 論理構成

### 1. Query Intake

入力を `RecallRequest` に正規化する。

最低限のフィールド:

- `task_kind`
- `query_text`
- `file_hints`
- `surface_filters`
- `status_filters`
- `limit`
- `context_budget_chars`

### 2. Candidate Retrieval

第一段は `SQLite FTS5` から候補を引く。

検索対象:

- `prompt`
- `output_text`
- `notes_text`
- `artifact_path`
- `event_kind`
- `session_surface`
- `session_mode`
- `model_id`
- `status`

初期方針:

- lexical first
- status と surface で先に絞る
- ファイルヒントがあるときは path hit を強く見る

### 3. Heuristic Ranking

最初は learned ranking ではなく、明示的ヒューリスティクスでよい。

候補スコアの要素:

- FTS score
- file path exact / partial match
- task kind と event kind の相性
- accepted / ok の優先
- quality_fail / failed を failure analysis では優先
- recency
- notes に `accepted`, `repair`, `review` などがあるか

### 4. Context Assembly

検索結果をそのまま並べるのではなく、用途別に束ねる。

初期の context block:

- `Relevant prior prompts`
- `Accepted outcomes`
- `Failure and repair patterns`
- `Related files and artifact paths`
- `Open risks`

### 5. Budget Control

長くしすぎると逆効果なので、文脈サイズを制御する。

初期方針:

- token ではなく char budget から始める
- 各 block に上限を持つ
- 上限超過時は:
  - 重複候補を落とす
  - 同趣旨の note を圧縮する
  - status の弱い候補から落とす

## データ契約

### RecallRequest

```python
{
  "task_kind": "review",
  "query_text": "review the memory index patch",
  "file_hints": ["scripts/memory_index.py"],
  "surface_filters": ["chat", "thinking"],
  "status_filters": ["ok", "quality_fail"],
  "limit": 12,
  "context_budget_chars": 6000,
}
```

### RecallCandidate

```python
{
  "event_id": "...",
  "score": -3.42,
  "reasons": ["fts-hit", "path-match", "accepted-note"],
  "session_surface": "chat",
  "event_kind": "chat_turn",
  "status": "ok",
  "prompt": "...",
  "output_text": "...",
  "artifact_path": "...",
}
```

### ContextBundle

```python
{
  "task_kind": "review",
  "query_text": "...",
  "selected_count": 6,
  "omitted_count": 11,
  "budget": {
    "context_budget_chars": 6000,
    "used_chars": 4820,
  },
  "blocks": [
    {"title": "Accepted outcomes", "items": [...]},
    {"title": "Failure and repair patterns", "items": [...]},
  ],
}
```

## 予定モジュール

最初の実装は次の分割が扱いやすい。

- `scripts/recall_context.py`
  - request normalization
  - retrieval orchestration
  - context assembly
- `tests/test_recall_context.py`
  - recall request normalization
  - ranking
  - budget trimming
  - grouped output

既存モジュールとの関係:

- `software_work_events.py`
  - source event contract
- `memory_index.py`
  - candidate retrieval source

## ルール

### Task Kind ごとの優先度

#### Review

- accepted review-like items を優先
- file path hit を強く優先
- 過去の quality_fail も残す

#### Design

- accepted design-like proposals を優先
- notes に decision や tradeoff があるものを優先

#### Proposal

- 実装提案と最終 outcome の両方を見せる
- test pass に繋がったものを優先

#### Failure Analysis

- `quality_fail`, `failed`, `blocked` を優先
- repair note や follow-up があるものを優先

## 実装フェーズ

### Phase A

- `RecallRequest`
- `RecallCandidate`
- `ContextBundle`
- `MemoryIndex.search(...)` を使った最小 retrieval
- 単純 ranking

### Phase B

- file hint matching
- status-aware grouping
- budget trim
- explainable reason tags

### Phase C

- task-kind specific block assembly
- related file clustering
- future vector fallback hook

## 検証

最初の検証は派手でなくてよい。

必要なもの:

- hand-labeled query set を 20-30 件
- 各 query ごとに「欲しい候補」数件を人手で定義
- `Hit@5`
- `Hit@10`
- noise rate
- context usefulness の手動評価

最低ライン:

- review/design/proposal/failure_analysis の各 task kind で、少なくとも 1 件は「これは明らかに使える」と言える recall が返る

## リスク

1. Event schema がまだ薄い
   - notes の意味づけが弱い
2. FTS only だと似た概念の言い換えに弱い
3. 現在の session entries は review/design/proposal のラベルがまだ粗い
4. char budget は token budget より雑

ただし、これは止まる理由ではない。
最初は雑でも、使われる回路を先に通す方が価値が高い。

## 完了条件

この設計の Done は次。

- Recall request から context bundle まで一通り通る
- review/design/proposal/failure_analysis の4種類で挙動が分かれる
- 結果に理由タグが付き、なぜ拾ったかが分かる
- 長すぎる context を落とせる
- テストで ranking と budget 制御が確認できる

## 結論

最初に取り掛かる設計としては、Recall / Context Builder が最適です。

理由は明快で:

- 今ある M1/M2 の土台をそのまま活かせる
- 実務で「使える感」が最初に出る
- 後続の評価ループと agent lane を強くできる
- 将来の学習基盤にも直結する

ここが通ると、この repo は単なる保存庫から、思い出して役に立つ開発エンジンに一段進みます。
