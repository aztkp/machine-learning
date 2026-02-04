# 5. 制限ボルツマンマシン (RBM)

## RBMとは

**制限ボルツマンマシン（Restricted Boltzmann Machine, RBM）** は、ボルツマンマシンに「制限」を加えたモデルです。

### 制限の内容

**同じ層内のユニット間に接続がない**

```
完全ボルツマンマシン:        制限ボルツマンマシン (RBM):

  h₁ ── h₂ ── h₃              h₁    h₂    h₃
  │╲  ╱│╲  ╱│                 │╲  ╱│╲  ╱│
  │ ╲╱ │ ╲╱ │                 │ ╲╱ │ ╲╱ │
  │ ╱╲ │ ╱╲ │                 │ ╱╲ │ ╱╲ │
  │╱  ╲│╱  ╲│                 │╱  ╲│╱  ╲│
  v₁ ── v₂ ── v₃              v₁    v₂    v₃

  層内接続あり                 層内接続なし
  → 学習が困難                 → 効率的な学習が可能
```

## RBMの構造

### 二部グラフ

RBMは**二部グラフ**構造を持ちます：
- 可視層（visible layer）: $\mathbf{v} = (v_1, ..., v_m)$
- 隠れ層（hidden layer）: $\mathbf{h} = (h_1, ..., h_n)$
- 可視-隠れ間のみ接続

### パラメータ

- $\mathbf{W} \in \mathbb{R}^{m \times n}$: 重み行列
- $\mathbf{b} \in \mathbb{R}^{m}$: 可視バイアス
- $\mathbf{c} \in \mathbb{R}^{n}$: 隠れバイアス

## エネルギー関数

$$E(\mathbf{v}, \mathbf{h}) = -\mathbf{b}^T\mathbf{v} - \mathbf{c}^T\mathbf{h} - \mathbf{v}^T\mathbf{W}\mathbf{h}$$

展開すると：

$$E(\mathbf{v}, \mathbf{h}) = -\sum_i b_i v_i - \sum_j c_j h_j - \sum_{i,j} v_i w_{ij} h_j$$

## RBMの最大の利点：条件付き独立性

層内接続がないため、**条件付き分布が分解**できます。

### 可視が与えられた時の隠れ

$$P(\mathbf{h}|\mathbf{v}) = \prod_j P(h_j|\mathbf{v})$$

各隠れユニットは**独立に**サンプリング可能：

$$P(h_j = 1|\mathbf{v}) = \sigma\left(c_j + \sum_i v_i w_{ij}\right)$$

### 隠れが与えられた時の可視

$$P(\mathbf{v}|\mathbf{h}) = \prod_i P(v_i|\mathbf{h})$$

$$P(v_i = 1|\mathbf{h}) = \sigma\left(b_i + \sum_j w_{ij} h_j\right)$$

### なぜこれが重要か

```
完全BM: ユニットを1つずつ順番に更新
        → O(n) ステップ/層

RBM:    層全体を一度に並列更新
        → O(1) ステップ/層
```

## ブロックギブスサンプリング

RBMでは効率的な**ブロックギブスサンプリング**が可能：

```
┌─────────────────────────────────────┐
│ ブロックギブスサンプリング           │
├─────────────────────────────────────┤
│ 1. v を固定 → h を全て並列サンプル   │
│ 2. h を固定 → v を全て並列サンプル   │
│ 3. 1-2 を繰り返す                   │
└─────────────────────────────────────┘
```

GPU での並列計算と相性が良い。

## 自由エネルギー

隠れユニットを周辺化した自由エネルギー：

$$F(\mathbf{v}) = -\mathbf{b}^T\mathbf{v} - \sum_j \log(1 + e^{c_j + \mathbf{W}_j^T\mathbf{v}})$$

RBMでは自由エネルギーが**閉形式**で計算可能（完全BMでは不可能）。

## CD-1 によるRBMの学習

```python
def cd1_update(v_data, W, b, c, lr):
    """CD-1による1回の更新"""

    # 正相: データから隠れ層をサンプル
    h_prob = sigmoid(c + v_data @ W)
    h_sample = (np.random.rand(*h_prob.shape) < h_prob).astype(float)
    positive = v_data.T @ h_prob

    # 負相: 1ステップの再構成
    v_recon = sigmoid(b + h_sample @ W.T)
    h_recon = sigmoid(c + v_recon @ W)
    negative = v_recon.T @ h_recon

    # パラメータ更新
    W += lr * (positive - negative) / len(v_data)
    b += lr * (v_data - v_recon).mean(axis=0)
    c += lr * (h_prob - h_recon).mean(axis=0)

    return W, b, c
```

## RBMの種類

### Binary-Binary RBM

最も基本的な形式。両層とも二値。

### Gaussian-Binary RBM

可視層が連続値（実数値データ用）：

$$E(\mathbf{v}, \mathbf{h}) = \sum_i \frac{(v_i - b_i)^2}{2\sigma_i^2} - \sum_j c_j h_j - \sum_{i,j} \frac{v_i}{\sigma_i} w_{ij} h_j$$

画像などの連続値データに使用。

### Softmax RBM

可視層がソフトマックス（カテゴリカルデータ用）：

- 単語の埋め込み
- 多クラス分類

### Conditional RBM (CRBM)

時系列データ用：

$$E(\mathbf{v}^{(t)}, \mathbf{h}^{(t)}|\mathbf{v}^{(t-1)}) = ...$$

過去の状態を条件として現在を生成。

## RBMの応用

### 1. 次元削減

```
入力データ → [RBM] → 隠れ層の活性
   784次元      →      100次元
  (MNIST)            (特徴表現)
```

### 2. 特徴学習

隠れユニットが自動的に有用な特徴を学習：

```
顔画像の場合:
h₁: 目を検出
h₂: 鼻を検出
h₃: 輪郭を検出
...
```

### 3. 協調フィルタリング

Netflix Prize で使用された推薦システム：

```
ユーザーの映画評価 → [RBM] → 未評価映画の予測
```

### 4. 深層学習の事前学習

RBMを積み上げて深いネットワークを構築（次章で詳述）。

## 可視化：RBMが学習するフィルタ

MNISTで学習したRBMの重みは、数字の部分パターンを表現：

```
重み W[:,j] を画像として可視化:

┌───┬───┬───┬───┐
│ ╱ │ ○ │ ─ │ │ │  ← 各隠れユニットが
├───┼───┼───┼───┤     学習した特徴
│ ╲ │ ─ │ ╱ │ ○ │
├───┼───┼───┼───┤
│ │ │ ╲ │ ○ │ ─ │
└───┴───┴───┴───┘
```

## 実装のヒント

### 重みの初期化

```python
# Xavier/Glorot 初期化
W = np.random.randn(n_visible, n_hidden) * np.sqrt(2.0 / (n_visible + n_hidden))
b = np.zeros(n_visible)
c = np.zeros(n_hidden)
```

### 学習のモニタリング

```python
# 再構成誤差（学習の進捗確認用、厳密ではない）
recon_error = np.mean((v_data - v_recon)**2)

# 自由エネルギー
free_energy = -np.dot(v, b) - np.sum(np.log(1 + np.exp(c + v @ W)), axis=1)
```

### ミニバッチ学習

```python
batch_size = 100
for epoch in range(100):
    np.random.shuffle(data)
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        W, b, c = cd1_update(batch, W, b, c, lr=0.01)
```

---

[← 前へ: 学習アルゴリズム](./04-learning.md) | [次へ: 深層ボルツマンマシン →](./06-deep-boltzmann.md)
