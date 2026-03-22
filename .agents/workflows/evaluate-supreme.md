---
description: akagi_supreme ボットの戦略コードを雀魂トッププレイヤーの思考基準で厳しく評価し、問題があれば修正する
---

# akagi_supreme 評価ワークフロー

このワークフローは `akagi_supreme` ボットの戦略コードを、雀魂トッププレイヤー（魂天・雀聖上位）の思考基準で厳しく評価し、問題があれば即座に修正するものです。

## 実行手順

### 1. 評価基準の確認

`REVIEW.md` を読み、評価基準を理解する。

### 2. 全戦略ファイルの精読

以下のファイルを **すべて** 精読する（省略禁止）：

// turbo-all

```
cat mjai_bot/akagi_supreme/strategy_engine.py
cat mjai_bot/akagi_supreme/push_fold.py
cat mjai_bot/akagi_supreme/placement_strategy.py
cat mjai_bot/akagi_supreme/game_state.py
cat mjai_bot/akagi_supreme/bot.py
cat mjai_bot/akagi_supreme/supreme_engine.py
```

### 3. テストの確認

既存テストも精読する：

```
cat tests/test_akagi_supreme.py
cat tests/test_akagi_eval8.py
```

### 4. REVIEW.md の基準に照らした評価

`REVIEW.md` に記載された全13項目の基準でコードを評価する：

1. 攻守判断（PUSH/MAWASHI/FOLD の三値判断）
2. ベタオリ（現物優先順位、赤牌正規化）
3. 回し打ち（MAWASHI の実装）
4. 副露判断（Q値差分、役牌/ドラポン、門前維持ペナルティ）
5. アガリ判断（Mortal のアガリ見逃しの尊重）
6. リーチ/ダマテン判断（待ち形、山残り、Q値符号バグ）
7. 順位戦略・オーラス（1位〜4位の各分岐）
8. 順位戦略・南場/東場
9. 脅威度推定（リーチ巡目、ダマテンパイシグナル）
10. 点数計算（符計算、三倍満親36000）
11. 受入枚数（面子抽出順序、3P萬子除外）
12. 3人麻雀対応
13. コード品質（テスト、副露打点計算の `my_melds` 参照）

### 5. 問題の修正

- **致命的問題**（Mortal より悪くなるバグ）: 即座に修正
- **重大な問題**（トッププレイヤーとの乖離が大きい）: 修正
- **中程度の問題**: 可能であれば修正
- **軽微な問題**: コメントのみ

### 6. テスト実行

修正後、テストを実行して全テストが通ることを確認：

// turbo
```bash
cd /Users/maruno/source/Akagi && .venv/bin/python -m pytest tests/test_akagi_supreme.py tests/test_akagi_eval8.py -v
```

修正に応じて新しいテストケースも追加する。

### 7. 評価レポートの追記

`mjai_bot/akagi_supreme/EVALUATION.md` に評価結果を追記する。
フォーマットは `REVIEW.md` の「評価レポートのフォーマット」セクションに従う。

### 8. 完了報告

評価結果のサマリーをユーザーに報告する。内容：
- 発見した問題の数と深刻度
- 修正した内容
- テスト結果（追加テスト含む）
- 前回からのランク変化
