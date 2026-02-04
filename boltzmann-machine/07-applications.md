# 7. 応用例

## 画像処理

### 画像生成

RBMを使って手書き数字（MNIST）を生成：

```
学習データ:        生成されたサンプル:
┌───┬───┬───┐      ┌───┬───┬───┐
│ 0 │ 1 │ 2 │  →   │ 0 │ 7 │ 3 │
├───┼───┼───┤      ├───┼───┼───┤
│ 3 │ 4 │ 5 │      │ 8 │ 1 │ 9 │
└───┴───┴───┘      └───┴───┴───┘
```

### 画像ノイズ除去

ノイズのある画像を入力として、RBMの再構成機能で復元：

```
ノイズ入力 → [RBM] → h → [RBM] → クリーンな出力
    ↓                              ↓
  ██░░██             →           ██████
  ░░██░░                         ██████
```

### 画像補完

欠損部分がある画像の補完：

```python
# 欠損部分をマスク
masked_image = image.copy()
masked_image[missing_pixels] = 0

# ギブスサンプリングで補完
for _ in range(1000):
    h = rbm.sample_hidden(masked_image)
    v = rbm.sample_visible(h)
    # 欠損部分のみ更新
    masked_image[missing_pixels] = v[missing_pixels]
```

## 推薦システム

### Netflix Prize での活用

2006年のNetflix Prizeで、RBMベースのモデルが上位入賞：

```
ユーザー×映画の評価行列:

          映画1  映画2  映画3  映画4
ユーザーA   5     ?      3     ?
ユーザーB   ?     4      ?     2
ユーザーC   4     ?      5     ?

  ↓ RBM で学習

「?」の部分を予測
```

### 実装のポイント

- 各映画を softmax ユニットとして表現（1-5の評価）
- 未評価の映画はマスクして学習
- 条件付きRBMで時間発展も考慮可能

## 自然言語処理

### 単語埋め込み

RBMで単語の分散表現を学習：

```
One-hot 表現     →  [RBM]  →  低次元埋め込み
(10000次元)                    (100次元)

king - man + woman ≈ queen のような関係を学習
```

### トピックモデリング

文書のトピック分布をRBMで学習：

```
文書 = 単語の集合 → [Replicated Softmax RBM] → トピック分布

隠れユニット: トピックを表現
可視ユニット: 単語の出現
```

## 音声・音楽

### 音声認識の特徴学習

```
音声波形 → スペクトログラム → [RBM] → 音響特徴
                                        ↓
                                   分類器（音素認識）
```

### 音楽生成

CRBM（Conditional RBM）を使った音楽生成：

```
時刻 t-1 の音符 → [CRBM] → 時刻 t の音符予測

過去の文脈を考慮した生成が可能
```

## 異常検知

### 原理

正常データでRBMを学習し、異常データは高い再構成誤差を示す：

```python
def detect_anomaly(rbm, data, threshold):
    # 再構成
    h = rbm.encode(data)
    reconstruction = rbm.decode(h)

    # 再構成誤差
    error = np.mean((data - reconstruction)**2, axis=1)

    # 閾値判定
    is_anomaly = error > threshold
    return is_anomaly
```

### 応用分野

- 製造業の品質管理
- ネットワーク侵入検知
- 金融詐欺検出

## 分類タスク

### RBMを特徴抽出器として使用

```
┌─────────────────┐
│    Softmax      │ ← 分類層
├─────────────────┤
│   RBM 隠れ層    │ ← 特徴抽出
├─────────────────┤
│   入力データ    │
└─────────────────┘

1. RBMを教師なしで事前学習
2. 分類層を追加
3. 全体をファインチューニング
```

### 半教師あり学習

ラベル付きデータが少ない場合に特に有効：

```
大量の unlabeled データ → [RBM 事前学習]
                              ↓
少量の labeled データ → [ファインチューニング]
```

## 量子コンピューティング

### 量子アニーリング

D-Wave などの量子コンピュータは、本質的にボルツマン分布からのサンプリングを行う：

```
量子アニーリングマシン ≈ 物理的なボルツマンマシン

・より高速なサンプリング
・より大規模な問題に対応可能
・学習も量子で行う研究が進行中
```

### 量子機械学習

```
古典データ → [量子BM] → 量子状態 → 測定 → 予測

量子重ね合わせを利用した表現力の向上
```

## 医療・バイオインフォマティクス

### 遺伝子発現解析

```
遺伝子発現データ → [DBN] → 疾患予測

・高次元（数万遺伝子）
・サンプル数が少ない
・事前学習が有効
```

### 薬物発見

分子構造から薬効を予測：

```
分子の特徴量 → [RBM] → 潜在表現 → 薬効予測
```

## 実装例：MNIST分類

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# データ準備
mnist = fetch_openml('mnist_784')
X = mnist.data.astype('float32') / 255.0
y = mnist.target.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# RBMで特徴学習
rbm = RBM(n_visible=784, n_hidden=256)
rbm.train(X_train, epochs=10, lr=0.01, batch_size=100)

# 特徴抽出
X_train_features = rbm.transform(X_train)
X_test_features = rbm.transform(X_test)

# 分類器
clf = LogisticRegression()
clf.fit(X_train_features, y_train)
accuracy = clf.score(X_test_features, y_test)
print(f"Test accuracy: {accuracy:.4f}")
```

## 現代の代替手法との比較

| タスク | ボルツマンマシン系 | 現代の主流 |
|--------|-------------------|-----------|
| 画像生成 | DBN/DBM | GAN, Diffusion Models |
| 特徴学習 | RBM | 自己教師あり学習 |
| 推薦 | RBM | Matrix Factorization, DNN |
| 異常検知 | RBM | VAE, Autoencoder |
| 事前学習 | DBN | Transformer (BERT等) |

ボルツマンマシンは多くの分野で他の手法に置き換わりましたが、確率モデルとしての理論的基盤は今でも重要です。

---

[← 前へ: 深層ボルツマンマシン](./06-deep-boltzmann.md) | [トップへ戻る →](./README.md)
