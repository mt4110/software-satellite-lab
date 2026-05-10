# 00. Core Anchor — ブレない芯

## 1. 絶対定義

`software-satellite-lab` は、AI coding agent ではない。

`software-satellite-lab` は、AI-assisted software work のための **local-first operating memory** である。

目的は、AI にコードを書かせること自体ではなく、AI と人間のソフトウェア作業から生まれる evidence を、次の判断に再利用できる状態へ変換すること。

```text
agent が作業する。
satellite が観測する。
evidence に正規化する。
記憶する。
比較する。
人間が判断する。
次の作業に再利用する。
```

## 2. 一文のプロダクト定義

```text
software-satellite-lab は、AI 開発作業を、忘れない・比べられる・検査できる・再利用できる状態にする、ローカルファーストな software-work evidence OS である。
```

英語:

```text
software-satellite-lab is a local-first operating memory for AI-assisted software work.
It does not compete to be the best coding agent.
It makes every agent run, review, failure, backend comparison, and human verdict inspectable, reusable, and learning-prep ready.
```

## 3. 研究レベルのインテリジェンスとは何か

このプロジェクトで言う「研究レベルのインテリジェンス」は、賢そうなチャット UI ではない。

次の性質を満たす system design を指す。

| 性質 | 意味 |
|---|---|
| Observability | 何が起きたかを source artifact へ戻れる |
| Provenance | 出力、判断、検証、採用/却下の由来が追跡できる |
| Falsifiability | 良かった/悪かったを後から検査できる |
| Reusability | 過去の判断・失敗・修正を次回の context に戻せる |
| Neutrality | 特定 model / agent / IDE / provider に依存しない |
| Human-gated learning | training-ready と呼ぶ前に人間の gate を通す |
| Ergonomics | 人間が次に何を見るべきかを迷わない |

つまり、知能は「答えを出すこと」だけではない。

```text
知能 = evidence を観測する能力
     + outcome を評価する能力
     + 過去を recall する能力
     + 不確実性を除外する能力
     + 人間判断を蓄積する能力
```

## 4. 絶対に中心にしないもの

以下を中心に置いた瞬間、既存の AI coding agent 競争に飲み込まれる。

```text
- agent execution の賢さ競争
- provider 対応数競争
- IDE extension 競争
- marketplace 競争
- cloud orchestration 競争
- dashboard の豪華さ競争
- benchmark score 競争
- autonomous self-improvement 競争
```

中心に置くのは、常に次。

```text
source-linked evidence
human verdict
comparison
recall
curation
learning preview
```

## 5. Anti-Drift Rule

新機能を入れる前に、必ずこの質問に答える。

```text
この機能は、software-work evidence を
忘れない・比べられる・検査できる・再利用できる状態にするか？
```

答えが No なら、作らない。

答えが Yes でも、次の質問に答える。

```text
それは agent 本体の競争に戻っていないか？
それは local-first / file-first を壊していないか？
それは human gate を迂回していないか？
それは dashboard / marketplace / cloud へ早すぎる拡張をしていないか？
```

## 6. 公式マントラ

```text
Do not build another coding agent.
Build the evidence OS around all coding agents.
```

日本語:

```text
もう一つの coding agent を作らない。
すべての coding agent の外側にある evidence OS を作る。
```
