# akagi_supreme 評価レポート（最新）

## 第17回評価（2026-03-22）: トッププレイヤー思考の反映度 — A

---

### 1. 発見された問題

#### 1-1. MAWASHI時にpost_riichi_safe牌が安全牌として認識されない（strategy_engine.py:_score_discards_by_safety）
**問題**: `_score_discards_by_safety` が `aggregate_danger` のみで危険度を算出しており、`post_riichi_safe`（リーチ宣言後に他家が通した牌＝リーチ者にとって100%安全な牌）を考慮していなかった。FOLD時は `_find_genbutsu_discard` で `post_riichi_safe` を参照するため問題ないが、MAWASHI時には `_score_discards_by_safety` のみを使用するため、確認済み安全牌が「中程度の危険」と分類され、より危険な牌が選択される可能性があった。
**修正**: `_score_discards_by_safety` で各牌について `post_riichi_safe` を照合し、マッチした場合は `danger` を `DANGER_SAFE * 0.5` 以下にキャップ。`tile_base()` で赤牌正規化も適用。
**深刻度**: 中程度（回し打ち中にリーチ者に対して確認済み安全牌を選ばず、わざわざ危険な牌を切るケースが発生し得る）

#### 1-2. _check_riichi_override が should_damaten に hand_value/acceptance_count を渡していない（strategy_engine.py:_check_riichi_override）
**問題**: `_check_riichi_override`（Mortalが打牌を選んだがリーチに上書きすべきか判定する関数）が `should_damaten(self.gs, p_adj)` をデフォルト引数（`hand_value=0.0`, `acceptance_count=0`）で呼んでいた。これにより `should_damaten` 内の満貫以上のダマテン判定（`hand_value >= 8000`）等が正しく評価されない。
**実影響**: 現行のコードでは `riichi_multiplier >= 1.05` と `prefer_damaten = True` が同時に成立するケースが存在しないため、実質的な影響はゼロ。しかし将来の変更で条件が変わった場合にバグとなるため、予防的に修正。
**修正**: `_check_riichi_override` のシグネチャに `hand_value`, `acceptance_count` パラメータを追加し、`adjust_action` からの呼び出し時に渡すように変更。
**深刻度**: 軽微（現時点では到達不能パスだが、正確性とメンテナンス性のため修正）

---

### 2. 修正内容

- `strategy_engine.py`: `_score_discards_by_safety` に `post_riichi_safe` のオーバーライドロジック追加。リーチ者に対して確認済み安全な牌は `DANGER_SAFE * 0.5` 以下にキャップ
- `strategy_engine.py`: `_check_riichi_override` のシグネチャに `hand_value`, `acceptance_count` を追加。`adjust_action` からの呼び出しで正しく値を渡すよう修正
- `tests/test_akagi_eval8.py`: 3件のテスト追加
  - `TestPostRiichiSafeInMawashi`: post_riichi_safe牌の低危険度確認（2ケース：通常牌・赤牌正規化）
  - `TestCheckRiichiOverrideHandValue`: hand_value/acceptance引数の正常受理確認（1ケース）

---

### 3. テスト結果

- 既存テスト: 138件パス（1件は jsonschema 未インストールによる controller インポート失敗、akagi_supreme 無関係）
- 追加テスト: 3件
- 合計: 141件パス

---

### 4. 総合評価

| カテゴリ | 前回(第16回) | 今回(第17回) | 変化 |
|---------|-------------|-------------|------|
| 1. 攻守判断（押し引き） | A+ | A+ | → |
| 2. ベタオリ | A+ | A+ | → |
| 3. 回し打ち | A | A+ | ↑ post_riichi_safe反映で安全牌選択精度向上 |
| 4. 副露判断 | A | A | → |
| 5. アガリ判断 | A+ | A+ | → |
| 6. リーチ/ダマテン判断 | A+ | A+ | → |
| 7. 順位戦略（オーラス） | A+ | A+ | → |
| 8. 順位戦略（南場・東場） | A | A | → |
| 9. 脅威度推定 | A+ | A+ | → |
| 10. 点数計算 | A+ | A+ | → |
| 11. 受入枚数 | A | A | → |
| 12. 3人麻雀 | A | A | → |
| 13. コード品質 | A+ | A+ | ↑ 防衛的パラメータ修正 |

---

### 5. 詳細レビュー所見

#### 正しく実装されている主要項目
- **攻守判断**: PUSH/MAWASHI/FOLDの三値判断が明確。safety_weightによる二重カウントなし。リーチ宣言済みは無条件PUSH。`my_turn`（per-player巡目）を一貫使用。
- **ベタオリ**: FOLD時は現物→後付け現物→スジ→壁の優先順位。tile_base()による赤牌正規化一貫。孤立牌優先の正しいタイブレーク。共通安全牌（複数リーチ者に対して安全な牌）を最優先。
- **回し打ち**: 安全牌をSAFE/MODERATE/DANGEROUSの3段階に分類し、最安全カテゴリ内でQ値最高を選択。post_riichi_safe牌も安全として正しく認識（今回修正）。
- **副露判断**: FOLD時は鳴き拒否（オーラス4位例外）。門前維持価値の評価（テンパイ>イーシャンテン>リャンシャンテン）。七対子ルート検出で対子5組以上ならポン拒否。高打点時（8000点以上）は門前ペナルティ無視。
- **アガリ判断**: Mortalのhora見逃しを完全に尊重。無条件強制なし（REVIEW.md基準5に完全準拠）。
- **リーチ/ダマテン**: Q値負のリーチ乗算は除算で正しく処理。追っかけリーチはデフォルト強制だがラス回避薄差は例外化。終盤（14巡目以降）は1000点リスク回避でダマテン。1種1枚の極薄待ちはダマテン推奨。山残り分析（wait_tile_details）で精密判断。
- **順位戦略（オーラス）**: 1位大差リード→FOLD、中差→MAWASHI、薄差→PUSH維持。2位は直撃条件も計算。3位はラス回避と上位逆転のバランス。4位は原則PUSH（降りても4位のまま）だがmin_push_value制限あり。
- **脅威度**: riichi_turnはper-player巡目で正しく格納。早リーチ+0.5、ツモ切り連続→手出しのダマテンシグナル、副露手打点推定（白ポンvs白ポン+中ポン+ホンイツを区別）、門前ホンイツ河検出、小三元検出すべて実装済み。
- **点数計算**: 三倍満親36000（正しい）。符計算は20/25/30/40/50に対応。`_calculate_points`と`estimate_hand_value`の整合性あり。
- **受入枚数**: 面子抽出で順子優先・刻子優先の両方を試行。3人麻雀で2m-8mを除外。テンパイ時の14枚完成手判定正常。
- **my_melds**: count_dora_in_hand, estimate_hand_value ともに副露牌を含めて計算（致命的バグ歴の再発なし）。

#### 許容可能な設計上の判断
- 東場の非親は原則Mortal委譲: 東場は順位変動が少なく介入リスクの方が大きい
- min_han_for_points のデフォルト30符: 戦略判断としては十分な精度（20/25/50符の差は順位条件判断に大きく影響しない）
- noten_penalty_effect 固定-3000: 保守的推定（最悪ケース想定）で安全側に倒している
- ippatsu回避がshanten>=2でのみFOLD: トッププレイヤーもイーシャンテン以上なら一発巡目でも手牌状況で判断する

---

### 6. 総合ランク: A

**理由**: 致命的バグなし。主要な13カテゴリすべてが正しく実装されており、大半がA+評価。今回発見されたMAWASHI時のpost_riichi_safe未反映は中程度の問題で、回し打ち時に確認済み安全牌を見落とすケースがあったが修正済み。Mortalとの統合は精密で、介入は「Mortalが明確に苦手とするパターン」に限定されている。S評価には、noten_penalty_effectの動的計算（テンパイ者数に応じた罰符変動）、実戦データによるA/Bテスト検証、および3人麻雀固有の戦略（北抜きドラ等）のさらなる最適化が必要。
