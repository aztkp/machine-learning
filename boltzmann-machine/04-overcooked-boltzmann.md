# ボルツマンマシン：Overcooked実装編

**対象読者**: 理系大学院生（ボルツマン有理性の基礎を学んだ後に読むことを想定）

---

## 1. この資料の目的

[sotassss/Overcooked](https://github.com/sotassss/Overcooked.git) の実装を題材に、ボルツマン有理性が実際のAIエージェントでどのように使われているかを解説する。

特に、この実装に存在する**2つの「温度」に相当するパラメータ**の意味と解釈を明らかにする。

---

## 2. Overcookedゲームの概要

### 2.1 ゲーム環境

```
┌─────────────────────────────────────┐
│  🍅   🔪   🔪   🥬                  │  🍅 = トマトクレート
│  ─────────────────                  │  🥬 = レタスクレート
│       [AI]     [Human]              │  🔪 = 包丁台
│  ─────────────────                  │  🍽️ = 皿スタック
│  🗑️        🍽️   📤                  │  📤 = 提供口
└─────────────────────────────────────┘
```

- **目標**: AIと人間が協力してサラダを作り、注文を処理する
- **課題**: 人間の行動を予測し、協調的に行動する必要がある

### 2.2 AIエージェントの構造

```
┌─────────────────────────────────────────────────────┐
│                    AgentAI                          │
│  ┌─────────────────┐    ┌─────────────────┐        │
│  │   TaskPlanner   │    │  ActionExecutor │        │
│  │  （タスク選択）  │ →  │  （行動実行）   │        │
│  │                 │    │                 │        │
│  │  optimal_rate   │    │    accuracy     │        │
│  │  （温度①）      │    │   （温度②）    │        │
│  └─────────────────┘    └─────────────────┘        │
└─────────────────────────────────────────────────────┘
```

---

## 3. 2つの「温度」パラメータ

### 3.1 温度①：`optimal_rate`（タスク選択レベル）

**場所**: `planner.py` のタスク選択ロジック

```python
# config.json での設定
"optimal_rate": {
    "value": 0.7,
    "range": [0.6, 0.85],
    "description": "最適行動を取る確率（残りは次善策を選ぶ）"
}
```

**動作**:

```python
# planner.py より抜粋
if deterministic or random.random() < p.optimal_rate or is_urgent:
    # 最適タスク（最高優先度）を選択
    _, task_type, target = candidates[0]
else:
    # ボルツマン分布に類似した確率的選択
    weights = [c[0] for c in candidates]  # c[0] = 優先度スコア
    total = sum(weights)
    weights = [w / total for w in weights]
    idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
    _, task_type, target = candidates[idx]
```

#### ボルツマン有理性との対応

| ボルツマン有理性 | この実装 |
|-----------------|---------|
| $Q(s, a)$（Q値） | `priority`（タスク優先度、0〜100） |
| $\beta$（逆温度） | `optimal_rate` で**暗黙的に**制御 |
| $P(a) \propto \exp(\beta Q)$ | `weights = [priority / sum(priorities)]` |

**解釈**:

$$P(\text{タスク } t) = \begin{cases}
1 & \text{if } t = \arg\max_t \text{priority}(t) \text{ （確率 } \texttt{optimal\_rate}\text{）} \\
\frac{\text{priority}(t)}{\sum_{t'} \text{priority}(t')} & \text{（確率 } 1 - \texttt{optimal\_rate}\text{）}
\end{cases}$$

これは**混合ボルツマン有理性**と呼べる形式で：
- 確率 `optimal_rate` で完全合理的（$\beta \to \infty$）
- 確率 `1 - optimal_rate` で有限温度のボルツマン分布

### 3.2 温度②：`accuracy`（行動実行レベル）

**場所**: `executor.py` のターゲット解決、`agent_ai.py` のミス判定

```python
# config.json での設定
"accuracy": {
    "value": 0.85,
    "range": [0.75, 0.95],
    "description": "判断力（0.0=ミスだらけ, 1.0=完璧）"
}
```

**動作**:

```python
# executor.py より抜粋
def _resolve_varied(self, state, task_type, target, reachable):
    # accuracy ミス: 同種タイルの中からランダムなものを選ぶ
    use_random = random.random() > self.personality.accuracy

    if task_type == "serve":
        if use_random:
            return self._pick_random_tile(state, Tile.SERVING, reachable)
        return self._pick_nearest_tile_varied(state, Tile.SERVING, reachable)
```

```python
# agent_ai.py より抜粋
effective_accuracy = p.accuracy
if is_urgent:
    effective_accuracy -= p.panic_error_bonus  # 焦るとミスが増える

if random.random() > effective_accuracy:
    # ミス発動: 多様なミスパターン
    miss_roll = random.random()
    if miss_roll < 0.3:
        # 一時停止（混乱）
        return {"action": "wait"}
    elif miss_roll < 0.6:
        # タスク忘れ
        self._committed_task = None
        return {"action": "wait"}
    # else: 方向ミス
```

#### ボルツマン有理性との対応

| ボルツマン有理性 | この実装 |
|-----------------|---------|
| 低温（$\beta$ 大） | `accuracy` 高 → 最適ターゲット選択 |
| 高温（$\beta$ 小） | `accuracy` 低 → ランダム選択 |

---

## 4. 2段階モデルの理論的解釈

### 4.1 階層的ボルツマン有理性

この実装は**階層的意思決定モデル**として解釈できる：

```
レベル1: タスク選択（What to do）
    ↓ optimal_rate で制御
レベル2: 行動実行（How to do）
    ↓ accuracy で制御
実際の行動
```

これは人間の意思決定における**計画段階と実行段階の分離**に対応する。

### 4.2 数式による表現

状態 $s$ における最終的な行動 $a$ の確率：

$$P(a | s) = \sum_{\tau \in \text{Tasks}} P(\tau | s) \cdot P(a | \tau, s)$$

ここで：
- $P(\tau | s)$：タスク選択確率（`optimal_rate` で制御）
- $P(a | \tau, s)$：タスクが決まった後の行動確率（`accuracy` で制御）

### 4.3 2つの温度が必要な理由

| 温度 | 制御対象 | 人間行動の模倣 |
|------|---------|---------------|
| `optimal_rate` | **何を**するか | 「最適ではないタスクを選ぶことがある」|
| `accuracy` | **どう**するか | 「タスクは正しいが実行でミスする」|

**例**:
- `optimal_rate` が低い：皿を取るべき時に材料を取りに行く（タスク選択ミス）
- `accuracy` が低い：皿を取りに行くが、近くの皿ではなく遠くの皿を取る（実行ミス）

---

## 5. 関連パラメータの詳細

### 5.1 Personalityクラスの全パラメータ

```python
@dataclass
class Personality:
    # ===== 反応速度 =====
    think_time_base: float = 0.3      # 基本思考時間（秒）
    think_time_variance: float = 0.4  # 思考時間の揺らぎ

    # ===== 温度①相当 =====
    optimal_rate: float = 0.7         # 最適タスク選択確率

    # ===== 温度②相当 =====
    accuracy: float = 0.85            # 行動精度

    # ===== その他の人間らしさ =====
    hesitation_rate: float = 0.15     # タスク切替時の迷い確率
    detour_rate: float = 0.1          # 寄り道確率
    side_preference: float = 0.5      # キッチン左右の好み

    # ===== 緊急時の変化 =====
    urgency_threshold: int = 12       # 焦り始める残り秒数
    panic_error_bonus: float = 0.15   # 焦り時のミス増加率
```

### 5.2 各パラメータのボルツマン的解釈

| パラメータ | ボルツマン的解釈 |
|-----------|-----------------|
| `optimal_rate` | タスク選択の逆温度（高い = 低温 = 合理的）|
| `accuracy` | 行動実行の逆温度（高い = 低温 = 正確）|
| `hesitation_rate` | 状態遷移時のエントロピー |
| `detour_rate` | パス探索の温度 |
| `panic_error_bonus` | 文脈依存の温度上昇（焦ると温度が上がる）|

---

## 6. 実装の詳細分析

### 6.1 タスク優先度システム

```python
# planner.py のタスク優先度（高い = 重要）

# 完成サラダ提供: 最高優先度
candidates.append((100, "serve", None))

# 包丁台クリア（緊急時）
candidates.append((95, "clear_board", occupied))

# サラダに材料追加
candidates.append((90, "pickup_to_salad", pos))

# 皿取得
candidates.append((85, "get_plate", None))

# 材料を切る
candidates.append((70, "cut", board))

# 材料取得
candidates.append((65, "get_ingredient", Tile.TOMATO_CRATE))
```

この優先度がボルツマン分布における「負のエネルギー」（= Q値）に相当する。

### 6.2 パス探索のノイズ

```python
# executor.py の A* 探索

def _a_star(self, state, start, goal, ignore_human=False):
    # ...
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        # ランダムノイズを追加（人間らしい経路選択）
        noise = random.uniform(0, 0.4)
        repeat_pen = 0.5 if nb in self._recent_cells else 0.0
        tg = g_score[current] + 1.0 + noise + repeat_pen
```

これもボルツマン分布的な確率的選択の一種で、**パス選択の温度**と解釈できる。

### 6.3 文脈依存の温度変化

```python
# agent_ai.py

# 緊急時は optimal_rate を事実上 1.0 に
if deterministic or random.random() < p.optimal_rate or is_urgent:
    _, task_type, target = candidates[0]

# 緊急時は accuracy を下げる（焦るとミスが増える）
effective_accuracy = p.accuracy
if is_urgent:
    effective_accuracy -= p.panic_error_bonus
```

**人間行動の模倣**: 時間的プレッシャーがあると、タスク選択は合理的になる（温度↓）が、実行精度は下がる（温度↑）

---

## 7. 従来のボルツマン有理性との比較

### 7.1 標準的なボルツマン有理性

$$P(a | s) = \frac{\exp(\beta \cdot Q(s, a))}{\sum_{a'} \exp(\beta \cdot Q(s, a'))}$$

- 単一の温度パラメータ $\beta$
- 行動空間上で直接確率分布を定義

### 7.2 この実装のモデル

$$P(a | s) = \sum_{\tau} P(\tau | s; \beta_1) \cdot P(a | \tau, s; \beta_2)$$

- **2つの温度**: $\beta_1$（タスク選択）、$\beta_2$（行動実行）
- **階層的構造**: タスク → 行動の2段階

### 7.3 利点

| 観点 | 単一温度モデル | 2段階温度モデル |
|------|---------------|----------------|
| 表現力 | タスクレベルと行動レベルを区別できない | 両レベルで異なる確率的挙動を表現可能 |
| 解釈性 | 「なぜその行動をしたか」が不明確 | 「タスク選択ミス」と「実行ミス」を区別可能 |
| 調整 | 1つのパラメータで全体の挙動が変化 | 計画と実行を独立に調整可能 |

---

## 8. 実験・調整のポイント

### 8.1 パラメータ調整の指針

```
人間らしさを増すには:
├── optimal_rate を下げる（0.6〜0.7）→ タスク選択に揺らぎ
├── accuracy を下げる（0.75〜0.85）→ 実行にミス
└── hesitation_rate を上げる（0.15〜0.25）→ 迷いを追加

協調性を高めるには:
├── optimal_rate を上げる（0.8〜0.9）→ 合理的なタスク選択
├── accuracy を上げる（0.9〜0.95）→ 正確な実行
└── urgency_threshold を上げる（15〜20）→ 焦りにくく
```

### 8.2 温度の組み合わせパターン

| optimal_rate | accuracy | 特徴 |
|--------------|----------|------|
| 高 | 高 | 完璧なAI（人間離れ）|
| 高 | 低 | 「何をすべきかは分かっているが不器用」|
| 低 | 高 | 「器用だが判断が甘い」|
| 低 | 低 | 初心者的な挙動 |

---

## 9. まとめ

### 9.1 この実装における「2つの温度」

1. **`optimal_rate`（温度①）**: タスク選択レベルの合理性
   - 高い値 = 低温 = 常に最適タスクを選択
   - 低い値 = 高温 = 次善策も確率的に選択

2. **`accuracy`（温度②）**: 行動実行レベルの精度
   - 高い値 = 低温 = 正確なターゲット選択・実行
   - 低い値 = 高温 = ミスが発生しやすい

### 9.2 理論的意義

- **階層的ボルツマン有理性**: 計画と実行を分離したモデル
- **文脈依存の温度**: 緊急時に温度が変化（焦り = 実行の温度上昇）
- **人間らしさのモデル化**: 単一温度では表現できない多様なミスパターン

### 9.3 実用上の価値

このアプローチにより、AIエージェントは：
- 人間と協調しやすい「人間らしい」挙動を示す
- パラメータ調整で難易度やプレイスタイルを制御できる
- タスク選択ミスと実行ミスを独立に分析・調整できる

---

## 参考文献

- [sotassss/Overcooked](https://github.com/sotassss/Overcooked.git) - 分析対象の実装
- [HumanCompatibleAI/overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai) - オリジナルのOvercooked-AI環境
- Carroll, M., et al. (2019). *On the Utility of Learning about Humans for Human-AI Coordination*
- Laidlaw, C., et al. (2022). *The Boltzmann Policy Distribution* (ICLR 2022)

---

[← 前へ: 発展と応用](./03-advanced-and-applications.md) | [概要へ戻る →](./boltzmann-machine-overview.md)
