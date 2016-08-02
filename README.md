# BayesianOptimizer
ベイズ最適化の練習用コード

## 内容について
機械学習のハイパーパラメータチューニングが面倒で、かと言ってグリッドサーチするのも時間がかかるしな、と思ったので
最近（2016/08/02あたり）流行りの（？）ベイズ最適化の練習をしました。その際のコードです。

* アルゴリズム
 - アルゴリズムは参考文献１，２を参考にしています。
 - アルゴリズムに関連したパラメータはBayesianOptimizerParameter.pyに記述してあります
 
* カーネル
 - ガウシアンカーネルと、Maternカーネルが使えます。パラメータは、カーネルオブジェクトを呼び出す際に渡してください（kenerl.py参照）


## References
1. "Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design" http://arxiv.org/abs/0912.3995
 - GP-UCBアルゴリズムを用いた更新方法について。
2. "Gaussian Process Optimization with Mutual Information" http://arxiv.org/abs/1311.4825
 - 相互情報量を用いた更新方法について
3. "1020：ベイズ的最適化の入門と応用 機械学習による機械学習の実験計画" https://www.youtube.com/watch?v=pQHWew4YYao
 - 佐藤一誠さんのGTC Japan2015での講演。機械学習の人にとっては、詳細は省いて、イメージを掴みたい場合に有用では
