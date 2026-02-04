# ボルツマンマシン学習ガイド

ボルツマンマシン（Boltzmann Machine）について体系的に学ぶための資料集です。

**対象読者**: 理系大学院生（線形代数・確率論の基礎はあるが機械学習は初学者）

---

## 資料構成

### 1ページ概要版

- **[ボルツマンマシン概説](./boltzmann-machine-overview.md)** - 全体像を1ページで把握

### 詳細版（4部構成）

| パート | 内容 |
|--------|------|
| **[1. 基礎編](./01-foundations.md)** | 導入・歴史、基礎理論、エネルギー関数 |
| **[2. 学習とRBM編](./02-learning-and-rbm.md)** | 学習アルゴリズム、制限ボルツマンマシン、CD法 |
| **[3. 発展と応用編](./03-advanced-and-applications.md)** | DBN/DBM、応用例、ボルツマン有理性 |
| **[4. Overcooked実装編](./04-overcooked-boltzmann.md)** | 実装における2つの温度パラメータの解釈 |

---

## インタラクティブ教材

- [ボルツマンマシン可視化](./interactive/visualization.html) - 動作原理を視覚的に理解
- [RBMシミュレーター](./interactive/rbm-simulator.html) - 制限ボルツマンマシンの学習過程を体験

---

## 前提知識

- **必須**: 線形代数の基礎（行列、ベクトル）、確率・統計の基礎（確率分布、期待値）
- **あると良い**: 統計力学の基本概念、最適化の基礎

---

## 参考文献

### 基礎
- Hinton, G. E., & Sejnowski, T. J. (1986). *Learning and Relearning in Boltzmann Machines*
- Hinton, G. E. (2002). *Training Products of Experts by Minimizing Contrastive Divergence*

### 深層化
- Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). *A Fast Learning Algorithm for Deep Belief Nets*
- Salakhutdinov, R., & Hinton, G. (2009). *Deep Boltzmann Machines*

### 応用
- Laidlaw, C., et al. (2022). *The Boltzmann Policy Distribution* (ICLR 2022)
- Carroll, M., et al. (2019). *On the Utility of Learning about Humans for Human-AI Coordination*
