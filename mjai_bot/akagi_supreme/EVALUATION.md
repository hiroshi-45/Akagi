# akagi_supreme 評価レポート（最新）

## 第16回評価（2026-03-22）: トッププレイヤー思考の反映度 — A

---

### 1. 発見された問題

#### 1-1. オーラス2位/3位の追っかけリーチが薄差を無視（placement_strategy.py:354-360）
**問題**: `should_damaten` の追っかけリーチ（oi-riichi）ロジックが、オーラス1位のみを例外としていた。
オーラス2位（3位と1000点差）や3位（4位と1000点差）で、対リーチでもリーチ供託1000点が順位を直接フリップさせるケースを考慮していなかった。
例: オーラス3位(25000点) vs 4位(24000点)で追っかけリーチ → 供託1000点で24000点に → 4位と同点で実質ラス転落リスク。
**修正**: `gs.is_all_last and gs.my_placement in (2, 3) and gs.diff_to_below <= 1000` の場合は追っかけリーチを強制せず、ダマテンロジックに委ねるように修正。
**深刻度**: 中程度（ラス回避は競技麻雀の最優先事項であり、この条件を見落とすと順位を直接失う）

#### 1-2. オーラス3位、4位僅差で prefer_damaten 未設定（placement_strategy.py:166-177）
**問題**: `_all_last_strategy` のオーラス3位・4位僅差（diff_below < 4000）の分岐で、diff_below ≤ 1000 のケースが未分離。
`prefer_damaten=False`（デフォルト）のため、`should_damaten` が最初の `if not adj.prefer_damaten: return False` で即座にFalseを返し、ダマテン判断が一切行われなかった。
**修正**: diff_below ≤ 1000 の場合に独立した `PlacementAdjustment` を返すよう分岐を追加。`prefer_damaten=True`, `riichi_multiplier=0.7` に設定。
**深刻度**: 中程度（上記 1-1 と連動。ラス回避のための基本的な順位保護が欠如していた）

#### 1-3. strategy_engine.py のモジュールdocstring が実装と矛盾（strategy_engine.py:19）
**問題**: docstring に「Force hora in all-last」と記載されていたが、実際のコードは Mortal の hora 判断を完全に信頼しており、強制的なアガリは行っていない。REVIEW.md 基準5（アガリ判断）に明記された「Mortal が hora を選ばなかった場合は信頼する」に準拠した正しい実装だが、docstring が誤解を招く。
**修正**: docstring を「Hora: trust Mortal's decision (Mortal already encodes 着順 conditions).」に修正。
**深刻度**: 軽微（ドキュメントのみ、ロジックへの影響なし）

---

### 2. 修正内容

- `placement_strategy.py`: `_all_last_strategy` 3位分岐に diff_below ≤ 1000 のケースを追加（prefer_damaten=True, riichi_multiplier=0.7）
- `placement_strategy.py`: `should_damaten` の追っかけリーチセクションにオーラス2位/3位 × diff_to_below ≤ 1000 の例外を追加
- `strategy_engine.py`: モジュールdocstring の「Force hora in all-last」を正しい記述に修正
- `tests/test_akagi_eval8.py`: 5件のテスト追加
  - `TestAllLast3rdThinLeadDamaten`: 3位薄差ダマテン判断（3ケース）
  - `TestAllLast2ndThinLeadChaseRiichi`: 2位薄差ダマテン判断（2ケース）

---

### 3. テスト結果

- 既存テスト: 133件パス（1件は jsonschema 未インストールによる controller インポート失敗、akagi_supreme 無関係）
- 追加テスト: 5件
- 合計: 138件パス

---

### 4. 総合評価

| カテゴリ | 前回(第15回) | 今回(第16回) | 変化 |
|---------|-------------|-------------|------|
| 1. 攻守判断（押し引き） | A+ | A+ | → |
| 2. ベタオリ | A+ | A+ | → |
| 3. 回し打ち | A | A | → |
| 4. 副露判断 | A | A | → |
| 5. アガリ判断 | A+ | A+ | → |
| 6. リーチ/ダマテン判断 | A | A+ | ↑ 薄差ラス回避ダマテン追加 |
| 7. 順位戦略（オーラス） | A+ | A+ | ↑ 3位薄差保護追加（既にA+だが改善） |
| 8. 順位戦略（南場・東場） | A | A | → |
| 9. 脅威度推定 | A+ | A+ | → |
| 10. 点数計算 | A+ | A+ | → |
| 11. 受入枚数 | A | A | → |
| 12. 3人麻雀 | A | A | → |
| 13. コード品質 | A+ | A+ | ↑ docstring 整合性改善 |

---

### 5. 詳細レビュー所見

#### 正しく実装されている主要項目
- **攻守判断**: PUSH/MAWASHI/FOLDの三値判断が明確。safety_weightによる二重カウントなし。リーチ宣言済みは無条件PUSH。
- **ベタオリ**: FOLD時は現物→後付け現物→スジ→壁の優先順位。tile_base()による赤牌正規化一貫。孤立牌優先の正しいタイブレーク。
- **アガリ判断**: Mortalのhora見逃しを完全に尊重。無条件強制なし。
- **脅威度**: riichi_turnはper-player巡目（turn // num_players）で正しく格納。早リーチ+0.5、ツモ切り連続→手出しのダマテンシグナル、副露手打点推定、門前ホンイツ河検出すべて実装済み。
- **点数計算**: 三倍満親36000（正しい）。符計算は20/25/30/40/50に対応。
- **受入枚数**: 面子抽出で順子優先・刻子優先の両方を試行。3人麻雀で2m-8mを除外。テンパイ時の14枚完成手判定正常。
- **Q値負のリーチ乗算**: _check_riichi_override で正しく除算処理。
- **my_melds**: count_dora_in_hand, estimate_hand_value ともに副露牌を含めて計算。

#### 許容可能な設計上の判断
- 東場の非親マンガン+ダマテン非介入: Mortal委譲で妥当
- min_han_for_points のデフォルト30符: 戦略判断としては十分な精度
- 追っかけリーチのデフォルト強制: diff_to_below > 1000 では正しい判断（今回の修正で薄差のみ例外化）

---

### 6. 総合ランク: A

**理由**: 致命的バグなし。主要な13カテゴリすべてが正しく実装されている。今回発見された薄差ラス回避ダマテンの問題は中程度であり、特定のエッジケース（オーラス2位/3位 × 下位と1000点以内 × 対リーチ）でのみ発生する。Mortalとの統合は精密で、介入は「Mortalが明確に苦手とするパターン」に限定されている。S評価には、より複雑な順位条件計算（例: ツモ/ロン/直撃別の逆転条件自動計算）や、実戦データによる検証が必要。
