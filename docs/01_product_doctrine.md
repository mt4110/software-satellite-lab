# 01. Product Doctrine — 何を作り、何を作らないか

## 1. Product Category

`software-satellite-lab` のカテゴリは、AI coding agent ではない。

新しいカテゴリ名:

```text
software-work evidence OS
AI-assisted software work operating memory
agent flight recorder for software engineering
```

このプロジェクトは、AI agent / LLM / local model / API backend / IDE assistant を置き換えるのではなく、それらが行った作業を観測・記録・比較・評価・再利用する。

## 2. First User

最初のユーザーは「AI coding tool を複数使い始めて、作業の判断履歴が崩壊している開発者」。

典型的な痛み:

```text
- cloud agent / IDE agent / terminal agent / local LLM の出力が散らばる
- どの修正がなぜ採用されたか分からない
- 同じ失敗を繰り返す
- test failure と修正履歴が記憶に残らない
- review の採用/却下理由が次回に使えない
- モデル比較をしても結果が消える
- learning / fine-tune 候補にして良いデータか判断できない
```

## 3. Jobs To Be Done

| Job | ユーザーの言葉 | 提供価値 |
|---|---|---|
| Recall | 前も似た問題あったよね？ | 類似失敗・修正・判断を source 付きで出す |
| Compare | どの backend / proposal が良かった？ | 比較記録と human verdict を残す |
| Review | この patch の危険点は？ | 過去 evidence に基づく risk note を作る |
| Verify | 本当に通った？ | test / verification result を evidence 化する |
| Curate | 学習候補にしていい？ | preview-only で採用/除外理由を出す |
| Audit | 何を根拠に決めた？ | source artifact path へ戻れる |

## 4. What This Is

```text
- local-first evidence system
- file-first artifact store
- software-work event schema
- recall / context builder
- evaluation signal ledger
- backend comparison harness
- human-gated learning preview
- Satellite Pack による declarative workflow 拡張
- thin UI + strong CLI
```

## 5. What This Is Not

```text
- 最強 coding agent
- IDE assistant
- autonomous dev team
- model provider hub
- cloud agent platform
- marketplace-first plugin system
- vector DB product
- dashboard SaaS
- automatic fine-tuning system
```

## 6. Strategic Positioning

競合は「作業する agent」を作る。

`software-satellite-lab` は「作業を忘れない OS」を作る。

```text
cloud agent が作業する。
IDE agent が作業する。
terminal agent が作業する。
local model が作業する。

software-satellite-lab は、それらの作業 evidence を保存し、比較し、次の判断に戻す。
```

## 7. First Public Promise

README / LP / GitHub description では、最初は大きく言い過ぎない。

推奨コピー:

```text
A local-first evidence OS for AI-assisted software work.
Record agent runs, reviews, failures, backend comparisons, and human verdicts — then reuse them as memory.
```

日本語:

```text
AI 開発作業の evidence OS。
agent 実行、レビュー、失敗、backend 比較、人間判断を記録し、次の作業の記憶として再利用する。
```

## 8. Product Rules

1. UI より evidence schema を優先する。
2. Agent execution より evaluation / recall を優先する。
3. Model provider 数より comparison quality を優先する。
4. Training より curation preview を優先する。
5. Plugin 自由度より permission safety を優先する。
6. Cloud 展開より local-first reproducibility を優先する。
7. Star 数より dogfood usefulness を優先する。
