# ボルツマンマシン学習ガイド

ボルツマンマシン（Boltzmann Machine）について基礎から体系的に学ぶための資料集です。

## 目次

1. [導入と歴史](./01-introduction.md) - ボルツマンマシンとは何か、その誕生の背景
2. [基礎理論](./02-basics.md) - ニューロン、状態、確率分布の基本
3. [エネルギー関数](./03-energy-function.md) - エネルギーベースモデルの核心
4. [学習アルゴリズム](./04-learning.md) - コントラスティブダイバージェンス等
5. [制限ボルツマンマシン (RBM)](./05-rbm.md) - 実用的な変種
6. [深層ボルツマンマシン](./06-deep-boltzmann.md) - DBMとDBN
7. [応用例](./07-applications.md) - 実世界での活用
8. [ボルツマン有理性と人間-AI協調](./08-boltzmann-rationality.md) - Overcooked-AIでの活用

## インタラクティブ教材

- [ボルツマンマシン可視化](./interactive/visualization.html) - 動作原理を視覚的に理解
- [RBMシミュレーター](./interactive/rbm-simulator.html) - 制限ボルツマンマシンの学習過程を体験

## 前提知識

- 線形代数の基礎（行列、ベクトル）
- 確率・統計の基礎（確率分布、期待値）
- 機械学習の基本概念

## 参考文献

- Hinton, G. E., & Sejnowski, T. J. (1986). Learning and Relearning in Boltzmann Machines
- Hinton, G. E. (2002). Training Products of Experts by Minimizing Contrastive Divergence
- Salakhutdinov, R., & Hinton, G. (2009). Deep Boltzmann Machines
- Laidlaw, C., et al. (2022). The Boltzmann Policy Distribution (ICLR 2022)
- Carroll, M., et al. (2019). On the Utility of Learning about Humans for Human-AI Coordination
