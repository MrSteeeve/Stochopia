# P2-M2 · 合成环境中数据属性 Z 与模型算子条件效应研究

**完成时间（UTC）**：2026-11-27T08:24:59.859377+00:00。**实验代码 SHA**：`ef9389f3e1c20b675d1af38eb1ddb28c303d6a1d`。

## 实验口径

- 78 次独立固定预算拟合：E1 36、E2 24、E3 18；各元组固定三个训练初值 (103, 137, 211)。同架构跨实验独立拟合；未增加 Tier 0 训练。
- 五个合成逐笔世界：各 16 训练日、12 评估日 × 2 会话窗口、每会话 256 事件；历史上下文 128 步。固定四头似然；训练 96 步、subsequence=128、burn-in=32、batch=8。
- 模型 K=4、已知生成器 K=16：可见母单−无母单响应使用共同随机数，已知真实生成机制给出有限 Monte Carlo 均值估计，而非无抽样误差的数学期望；单位为 tick。`counterfactual_impact_mae`（CF）是该反事实目标误差，独立于 stylized impact MAE 或 C2 稳定性。
- 每日、每种子先聚合物理窗口和副本；配对对比先在日×训练种子内作差，后对三种子取均值；按独立评估日索引重抽 2000 次。CI 是 95% percentile 区间，要求至少 6 个配对日与 1800 个有效重抽，不以训练种子或 Monte Carlo 副本扩大独立样本。
- 模型绝对读数是三个训练种子的日均指标平均，允许的缺失值用 — 或明确标出种子支持；配对 Δ=候选−父模型（负误差差值为改善）；G_mem(L)=NLL_reset(L)−NLL_stateful，计算使用完全一致的监督掩码，G_mem 描述对历史的依赖，非越大越好。
- 数据、真值、副本、bootstrap 种子分别为 20261127、76193、9127、41397；评估日配对与各世界生成内容哈希见 manipulation_check.json。
- 母单子单时刻：[100, 300, 500, 700] ms，每笔 100 股；时间网格 [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600] ms。总执行墙钟 121.994 秒；各训练累计 164.313 秒、评估累计 154.544 秒。不使用私有数据，不调用 LLM API。

**模型训练前的生成器校准与真值边界**：

- 初步生成试验：Depth 2000 and 8 x 200-share children produced degenerate thick truth and low W6 support.
- 最终训练期锚点校准：Depth 300/100 and 4 x 100-share children chosen before any model fit using training anchors.
- 校准时是否看过正式模型结果：否。
- 真值条件：Standard ten-level aggregated snapshot with true latent sign/clock state; common random marks. 从标准十档聚合盘口快照和生成器真实的隐符号/时钟状态启动反事实，保留共同随机标记；未建模原始订单队列的逐单账龄（cancel age）。这是特定合成生成机制的条件真值，不将其外推为真实交易所订单级反事实真值。

## 表 1 · do(λ) → Z(λ) 干预核验

| 配置或统计量（median / IQR） | W1_thick_short | W2_thick_long | W3_thin_short | W4_thin_long | W6_thin_hawkes | Tier 0 short | Tier 0 long |
| --- | --- | --- | --- | --- | --- | --- | --- |
| λ · q_scale | 300 | 300 | 100 | 100 | 100 | — | — |
| λ · cancel_probability | 0.12 | 0.12 | 0.18 | 0.18 | 0.18 | — | — |
| λ · slow_persistence | 0.5 | 0.97 | 0.5 | 0.97 | 0.5 | — | — |
| λ · fast_persistence | 0.5 | 0.8 | 0.5 | 0.8 | 0.5 | — | — |
| λ · alpha_hawk | 0 | 0 | 0 | 0 | 0.7 | — | — |
| λ · mean_dt_ms | 40 | 40 | 40 | 40 | 40 | — | — |
| λ · beta_hawk | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | — | — |
| Tier 0 λ · n_slots | — | — | — | — | — | 1 | 1 |
| Tier 0 λ · alpha | — | — | — | — | — | 1.5 | 1.1 |
| Tier 0 λ · max_len | — | — | — | — | — | 1 | 256 |
| z_mem.sign_acf_sum_1_32 | -0.350509 / 0.298682 | 2.11534 / 4.24529 | -0.308047 / 0.425813 | 3.0407 / 4.21875 | -0.308047 / 0.425813 | 0.0476113 / 0.268046 | 8.2132 / 6.11322 |
| z_mem.sign_acf_log_slope | -0.175772 / 0.431401 | -0.789193 / 0.293611 | -0.0976355 / 0.397056 | -0.810538 / 0.188577 | -0.0976355 / 0.397056 | — | — |
| z_mem.flow_vr_32_4 | 0.529779 / 0.421663 | 1.72374 / 1.91233 | 0.431969 / 0.296537 | 2.18137 / 1.27313 | 0.431969 / 0.296537 | — | — |
| z_thin.trade_to_top1_depth | 0.333333 / 0 | 0.333333 / 0 | 1 / 0 | 1 / 0 | 1 / 0 | — | — |
| z_thin.midprice_shift_rate | 0 / 0 | 0 / 0 | 0.639216 / 0.0637255 | 0.637255 / 0.0441176 | 0.639216 / 0.0637255 | — | — |
| z_thin.cancel_to_limit_ratio | 0.100715 / 0.0251377 | 0.0980083 / 0.0175222 | 0.282276 / 0.0365854 | 0.286798 / 0.0581913 | 0.282276 / 0.0365854 | — | — |
| z_burst.zero_wait_prob | 0.0117188 / 0.00878906 | 0.0117188 / 0.00878906 | 0.0117188 / 0.00878906 | 0.0117188 / 0.00878906 | 0.162109 / 0.0400391 | — | — |
| z_burst.interarrival_cv2 | 0.942237 / 0.127498 | 0.942237 / 0.127498 | 0.942237 / 0.127498 | 0.942237 / 0.127498 | 5.02834 / 0.911469 | — | — |
| z_asym.book_imbalance_bias | 0.0885155 / 0.0638267 | 0.0829113 / 0.0973924 | 0.0331036 / 0.0318962 | 0.0530678 / 0.0847459 | 0.0331036 / 0.0318962 | — | — |
| z_asym.flow_sign_bias | 0.0606122 / 0.0648204 | 0.341834 / 0.271865 | 0.0760825 / 0.0906433 | 0.163043 / 0.305542 | 0.0760825 / 0.0906433 | — | — |
| 训练事件无效价差率（物理质量，不属于 Z） | 0 | 0 | 0 | 0 | 0 | — | — |

IQR 为训练期各会话窗口 q75−q25；W5 为 W3 的别名，非第六个独立数据集。“—” 表示不适用或不可测。重构质量函数独立于 Z；合成盘口没有外部重构真值，此处只报告撮合路径实际的无效价差率。

- `burst` 干预/物理核验：**通过**；预注册规则：same thin/short world with alpha=.7 increases median zero waits and interarrival CV²。
- `memory` 干预/物理核验：**通过**；预注册规则：long memory ACF sum exceeds short by >2 and VR is higher, on both thicknesses。
- `thin` 干预/物理核验：**通过**；预注册规则：both thick worlds median shift < .05; both thin worlds > .20。
- `valid_book` 干预/物理核验：**通过**；预注册规则：no empty/crossed pre-event touch in the training panel。

**伴随属性漂移**（中位数差 / max(参照 IQR, 10⁻⁶)；RMS 不作因果效应）：

生成器旋钮是受控的，实测 Z 各维并不保证统计正交。例如盘口、拆单持久性可同时影响方向偏度，不能将交互无条件归因于某一个 Z 标量。参照 IQR 为零时，标准化漂移使用 10⁻⁶ 下限，很大的漂移数值不代表同等量级的经济效应。
- W1_thick_short->W2_thick_long：RMS=3.11511；`z_asym.book_imbalance_bias`=-0.0878036，`z_asym.flow_sign_bias`=4.33847，`z_burst.interarrival_cv2`=0，`z_burst.zero_wait_prob`=0，`z_mem.flow_vr_32_4`=2.83156，`z_mem.sign_acf_log_slope`=-1.42193，`z_mem.sign_acf_sum_1_32`=8.25576，`z_thin.cancel_to_limit_ratio`=-0.107666，`z_thin.midprice_shift_rate`=0，`z_thin.trade_to_top1_depth`=0
- W1_thick_short->W3_thin_short：RMS=292069；`z_asym.book_imbalance_bias`=-0.868161，`z_asym.flow_sign_bias`=0.238663，`z_burst.interarrival_cv2`=0，`z_burst.zero_wait_prob`=0，`z_mem.flow_vr_32_4`=-0.231962，`z_mem.sign_acf_log_slope`=0.181123，`z_mem.sign_acf_sum_1_32`=0.142162，`z_thin.cancel_to_limit_ratio`=7.22268，`z_thin.midprice_shift_rate`=639216，`z_thin.trade_to_top1_depth`=666667
- W2_thick_long->W4_thin_long：RMS=291640；`z_asym.book_imbalance_bias`=-0.306425，`z_asym.flow_sign_bias`=-0.657645，`z_burst.interarrival_cv2`=0，`z_burst.zero_wait_prob`=0，`z_mem.flow_vr_32_4`=0.239302，`z_mem.sign_acf_log_slope`=-0.0726982，`z_mem.sign_acf_sum_1_32`=0.217972，`z_thin.cancel_to_limit_ratio`=10.7744，`z_thin.midprice_shift_rate`=637255，`z_thin.trade_to_top1_depth`=666667
- W3_thin_short->W4_thin_long：RMS=3.18125；`z_asym.book_imbalance_bias`=0.625911，`z_asym.flow_sign_bias`=0.959376，`z_burst.interarrival_cv2`=0，`z_burst.zero_wait_prob`=0，`z_mem.flow_vr_32_4`=5.89943，`z_mem.sign_acf_log_slope`=-1.79547，`z_mem.sign_acf_sum_1_32`=7.86435，`z_thin.cancel_to_limit_ratio`=0.123599，`z_thin.midprice_shift_rate`=-0.0307692，`z_thin.trade_to_top1_depth`=0
- W3_thin_short->W6_thin_hawkes：RMS=11.4886；`z_asym.book_imbalance_bias`=0，`z_asym.flow_sign_bias`=0，`z_burst.interarrival_cv2`=32.0483，`z_burst.zero_wait_prob`=17.1111，`z_mem.flow_vr_32_4`=0，`z_mem.sign_acf_log_slope`=0，`z_mem.sign_acf_sum_1_32`=0，`z_thin.cancel_to_limit_ratio`=0，`z_thin.midprice_shift_rate`=0，`z_thin.trade_to_top1_depth`=0

**Tier 0 传播子交叉对照（仅机制与真值检查，不计额外训练）**：
- long：固定传播核响应峰值/末值 3.10387/1.38794；独立改变核 τ_max=1000 后，峰值/末值 3.10387/1.38794。
- short：固定传播核响应峰值/末值 3.10387/1.38794；独立改变核 τ_max=16 后，峰值/末值 1.59763/0.0252538。

固定传播核时短/长记忆真值曲线最大绝对差为 8.88178e-16。该对照将背景流记忆与价格核响应分开检查，不能替代 E1 的跨模型训练结论。

**已知生成器真值的报价支持**（逐窗口 × 时间点）：
- W1_thick_short：K=16，最少有效真值副本数 16，平均有效比例 1，平均响应绝对峰值 0.198568 tick。
- W2_thick_long：K=16，最少有效真值副本数 16，平均有效比例 1，平均响应绝对峰值 0.158203 tick。
- W3_thin_short：K=16，最少有效真值副本数 16，平均有效比例 1，平均响应绝对峰值 2.22852 tick。
- W4_thin_long：K=16，最少有效真值副本数 16，平均有效比例 1，平均响应绝对峰值 2.29753 tick。
- W6_thin_hawkes：K=16，最少有效真值副本数 16，平均有效比例 1，平均响应绝对峰值 2.14779 tick。

## 表 2 · M2-E1 记忆强度条件效应

| 世界 | 模型 | 总 NLL | G(1) | G(4) | G(16) | G(64) | 方向 ACF MAE | CF MAE (tick) | 配对父模型 | CF Δ [95% CI] / 支持 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1_thick_short | gru-base | 4.86228 | 0.582163 | 0.192389 | 0.0521848 | 0.00952095 | 0.163839 | 0.202757 | 基线 | — |
| W1_thick_short | gru-multiscale | 5.02997 | 2.43243 | 1.86182 | 0.913597 | 0.228726 | 0.159108 | 0.212976 | gru-base | 0.0102194 [-0.0197828, 0.0365636] (12/12 日; 2000/2000 抽样) |
| W1_thick_short | gru-surprise | 4.87547 | 1.02638 | 0.393731 | 0.102867 | 0.0199804 | 0.16124 | 0.205032 | gru-base | 0.00227536 [-0.0119818, 0.0162435] (12/12 日; 2000/2000 抽样) |
| W1_thick_short | lru-base | 5.00387 | 2.03868 | 1.22871 | 0.414985 | 0.0843939 | 0.158216 | 0.221078 | 基线 | — |
| W1_thick_short | lru-multiscale | 5.0183 | 2.2945 | 1.64346 | 0.764317 | 0.18778 | 0.159616 | 0.220394 | lru-base | -0.000683923 [-0.0130245, 0.0105357] (12/12 日; 2000/2000 抽样) |
| W1_thick_short | lru-surprise | 5.00476 | 2.33479 | 1.58919 | 0.622726 | 0.134706 | 0.1591 | 0.229048 | lru-base | 0.00797033 [0.00147306, 0.0144285] (12/12 日; 2000/2000 抽样) |
| W2_thick_long | gru-base | 4.83804 | 0.629472 | 0.210593 | 0.0555136 | 0.0109079 | 0.139846 | 0.223623 | 基线 | — |
| W2_thick_long | gru-multiscale | 5.05624 | 2.42282 | 1.85067 | 0.905309 | 0.23421 | 0.13989 | 0.22482 | gru-base | 0.00119686 [-0.0129949, 0.0148885] (12/12 日; 2000/2000 抽样) |
| W2_thick_long | gru-surprise | 4.86706 | 1.07266 | 0.417328 | 0.108964 | 0.021789 | 0.138986 | 0.226175 | gru-base | 0.00255156 [-0.0168225, 0.0230308] (12/12 日; 2000/2000 抽样) |
| W2_thick_long | lru-base | 5.02275 | 2.04458 | 1.23029 | 0.414231 | 0.0869072 | 0.143394 | 0.228121 | 基线 | — |
| W2_thick_long | lru-multiscale | 5.0422 | 2.29608 | 1.64785 | 0.769293 | 0.198584 | 0.139935 | 0.223084 | lru-base | -0.00503735 [-0.0324617, 0.0208435] (12/12 日; 2000/2000 抽样) |
| W2_thick_long | lru-surprise | 5.02618 | 2.34362 | 1.59937 | 0.630168 | 0.140273 | 0.143528 | 0.230804 | lru-base | 0.00268308 [-0.00520866, 0.0113258] (12/12 日; 2000/2000 抽样) |

**条件交互（高属性世界的候选−父模型差，再减去低属性世界的同一差）：**
- gru-multiscale vs gru-base · counterfactual_impact_mae：-0.00902252 [-0.0404175, 0.0268775] (12/12 日; 2000/2000 抽样)。
- gru-multiscale vs gru-base · direction_acf_mae：0.00477433 [-0.000920635, 0.0102875] (12/12 日; 2000/2000 抽样)。
  - W1_thick_short 单世界配对效应：-0.00473069 [-0.0120304, 0.00228632] (12/12 日; 2000/2000 抽样)。
  - W2_thick_long 单世界配对效应：4.36436e-05 [-0.00674941, 0.00596246] (12/12 日; 2000/2000 抽样)。
- gru-multiscale vs gru-base · memory_gap_64：0.00409664 [-0.00429516, 0.0122725] (12/12 日; 2000/2000 抽样)。
  - W1_thick_short 单世界配对效应：0.219205 [0.203442, 0.235359] (12/12 日; 2000/2000 抽样)。
  - W2_thick_long 单世界配对效应：0.223302 [0.208203, 0.236863] (12/12 日; 2000/2000 抽样)。
- gru-surprise vs gru-base · counterfactual_impact_mae：0.000276199 [-0.0229117, 0.0232139] (12/12 日; 2000/2000 抽样)。
- gru-surprise vs gru-base · direction_acf_mae：0.0017385 [-0.00237138, 0.00566902] (12/12 日; 2000/2000 抽样)。
  - W1_thick_short 单世界配对效应：-0.00259857 [-0.0058709, 0.000736906] (12/12 日; 2000/2000 抽样)。
  - W2_thick_long 单世界配对效应：-0.000860073 [-0.00458546, 0.00307957] (12/12 日; 2000/2000 抽样)。
- gru-surprise vs gru-base · memory_gap_64：0.000421598 [-0.000403589, 0.00131464] (12/12 日; 2000/2000 抽样)。
  - W1_thick_short 单世界配对效应：0.0104595 [0.00919588, 0.0117628] (12/12 日; 2000/2000 抽样)。
  - W2_thick_long 单世界配对效应：0.0108811 [0.00939296, 0.0123535] (12/12 日; 2000/2000 抽样)。
- lru-multiscale vs lru-base · counterfactual_impact_mae：-0.00435343 [-0.0318567, 0.0217823] (12/12 日; 2000/2000 抽样)。
- lru-multiscale vs lru-base · direction_acf_mae：-0.00485959 [-0.0104946, 0.000668993] (12/12 日; 2000/2000 抽样)。
  - W1_thick_short 单世界配对效应：0.00140054 [-0.00189632, 0.00491986] (12/12 日; 2000/2000 抽样)。
  - W2_thick_long 单世界配对效应：-0.00345905 [-0.00795354, 0.000121458] (12/12 日; 2000/2000 抽样)。
- lru-multiscale vs lru-base · memory_gap_64：0.00829118 [0.000993292, 0.0149894] (12/12 日; 2000/2000 抽样)。
  - W1_thick_short 单世界配对效应：0.103386 [0.0912773, 0.115182] (12/12 日; 2000/2000 抽样)。
  - W2_thick_long 单世界配对效应：0.111677 [0.0989297, 0.124119] (12/12 日; 2000/2000 抽样)。
- lru-surprise vs lru-base · counterfactual_impact_mae：-0.00528725 [-0.0160732, 0.00606554] (12/12 日; 2000/2000 抽样)。
- lru-surprise vs lru-base · direction_acf_mae：-0.000750646 [-0.00386703, 0.00189277] (12/12 日; 2000/2000 抽样)。
  - W1_thick_short 单世界配对效应：0.000884764 [-0.000329579, 0.00253396] (12/12 日; 2000/2000 抽样)。
  - W2_thick_long 单世界配对效应：0.000134118 [-0.00272764, 0.00245432] (12/12 日; 2000/2000 抽样)。
- lru-surprise vs lru-base · memory_gap_64：0.00305315 [0.00137985, 0.00485065] (12/12 日; 2000/2000 抽样)。
  - W1_thick_short 单世界配对效应：0.0503125 [0.0465192, 0.0543708] (12/12 日; 2000/2000 抽样)。
  - W2_thick_long 单世界配对效应：0.0533657 [0.0496525, 0.0571643] (12/12 日; 2000/2000 抽样)。

## 表 3 · M2-E2 同状态容量的档位矩阵与坐标平移

| 世界 | 模型 | 参数数 | 状态标量 | 价格档位 NLL | 耗尽副本率 | CF MAE (tick) | 配对父模型 | CF Δ [95% CI] / 支持 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1_thick_short | flat-gru | 24244 | 60 | 0.862037 | 0 | 0.202757 | 基线 | — |
| W1_thick_short | matrix-noshift | 25722 | 60 | 0.926083 | 0 | 0.22188 | flat-gru | 0.0191235 [-0.00689216, 0.0432223] (12/12 日; 2000/2000 抽样) |
| W1_thick_short | matrix-advect | 25722 | 60 | 0.926083 | 0 | 0.222012 | matrix-noshift | 0.000131524 [-0.00160459, 0.00215699] (12/12 日; 2000/2000 抽样) |
| W1_thick_short | flat-readoutgate | 26227 | 60 | 0.844459 | 0 | 0.191051 | flat-gru | -0.0117056 [-0.032434, 0.00869897] (12/12 日; 2000/2000 抽样) |
| W3_thin_short | flat-gru | 24244 | 60 | 1.10996 | 0.84375 | 35.0277 | 基线 | — |
| W3_thin_short | matrix-noshift | 25722 | 60 | 1.19684 | 0.888889 | 31.3639 | flat-gru | -30.1883 — (3/12 日; 1931/2000 抽样) |
| W3_thin_short | matrix-advect | 25722 | 60 | 1.19658 | 0.885417 | 28.0241 | matrix-noshift | -0.035101 — (5/12 日; 1997/2000 抽样) |
| W3_thin_short | flat-readoutgate | 26227 | 60 | 1.09902 | 0.777778 | 65.7346 | flat-gru | 5.16452 — (4/12 日; 1985/2000 抽样) |

矩阵态预算 10×6 共 60 个实标量，其中两个槽保存因果已知的参考 bid/ask tick，58 个为可学习状态；NoShift 和 Advect 参数完全相同，仅改变因果平移机制。相同总状态容量不意味着各骨干可训练参数完全相等，表中记录精确参数量。耗尽副本率是报价缺失/盘口耗尽的窗口×副本轨迹比例，不把缺失报价编成价格。

**条件交互（高属性世界的候选−父模型差，再减去低属性世界的同一差）：**
- flat-readoutgate vs flat-gru · counterfactual_impact_mae：5.17723 — (4/12 日; 1985/2000 抽样)。
- flat-readoutgate vs flat-gru · rel_ticks_nll：0.00663755 [0.00397104, 0.00941244] (12/12 日; 2000/2000 抽样)。
  - W1_thick_short 单世界配对效应：-0.0175784 [-0.0199843, -0.0153311] (12/12 日; 2000/2000 抽样)。
  - W3_thin_short 单世界配对效应：-0.0109409 [-0.0119013, -0.0099578] (12/12 日; 2000/2000 抽样)。
- matrix-advect vs matrix-noshift · counterfactual_impact_mae：-0.0368687 — (5/12 日; 1997/2000 抽样)。
- matrix-advect vs matrix-noshift · rel_ticks_nll：-0.000262861 [-0.000299946, -0.000226259] (12/12 日; 2000/2000 抽样)。
  - W1_thick_short 单世界配对效应：0 [0, 0] (12/12 日; 2000/2000 抽样)。
  - W3_thin_short 单世界配对效应：-0.000262861 [-0.000299946, -0.000226259] (12/12 日; 2000/2000 抽样)。
- matrix-noshift vs flat-gru · counterfactual_impact_mae：-30.1948 — (3/12 日; 1931/2000 抽样)。
- matrix-noshift vs flat-gru · rel_ticks_nll：0.0228391 [0.0143614, 0.0309621] (12/12 日; 2000/2000 抽样)。
  - W1_thick_short 单世界配对效应：0.0640458 [0.060078, 0.0680909] (12/12 日; 2000/2000 抽样)。
  - W3_thin_short 单世界配对效应：0.0868849 [0.0816458, 0.0918359] (12/12 日; 2000/2000 抽样)。

## 表 4 · M2-E3 时间簇聚度与 Hawkes 强度态

| 世界 | 模型 | 总 NLL | dt NLL | 零等待率绝对残差 | CV² 绝对残差 | 耗尽副本率 | CF MAE (tick) | 配对父模型 | CF Δ [95% CI] / 支持 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W3_thin_short | gru-base | 4.62777 | 1.92119 | 0.0162811 | 0.29831 | 0.84375 | 35.0277 | 基线 | — |
| W3_thin_short | selective-ssm | 4.50052 | 1.92415 | 0.0164436 | 0.307565 | 0.701389 | 38.6374 | gru-base | -20.1423 — (4/12 日; 1985/2000 抽样) |
| W3_thin_short | gru-hawkes | 4.5401 | 1.92519 | 0.0166859 | 0.319961 | 0.829861 | 19.6874 | gru-base | -6.60943 — (4/12 日; 1985/2000 抽样) |
| W6_thin_hawkes | gru-base | 4.30627 | 1.59644 | 0.0738825 | 2.80433 | 0.871528 | 49.7326 | 基线 | — |
| W6_thin_hawkes | selective-ssm | 4.07083 | 1.59806 | 0.0746898 | 2.7371 | 0.694444 | 2.31806 | gru-base | -4.5864 [-9.65956, -0.72105] (6/12 日; 2000/2000 抽样) |
| W6_thin_hawkes | gru-hawkes | 4.21339 | 1.57766 | 0.0745208 | 2.86901 | 0.805556 | 2.90327 | gru-base | -3.14056 — (5/12 日; 1997/2000 抽样) |

**条件交互（高属性世界的候选−父模型差，再减去低属性世界的同一差）：**
- gru-hawkes vs gru-base · counterfactual_impact_mae：1.52701 — (3/12 日; 1931/2000 抽样)。
- gru-hawkes vs gru-base · dt_nll：-0.0227897 [-0.0260534, -0.0198284] (12/12 日; 2000/2000 抽样)。
  - W3_thin_short 单世界配对效应：0.00400276 [0.00303104, 0.00501053] (12/12 日; 2000/2000 抽样)。
  - W6_thin_hawkes 单世界配对效应：-0.0187869 [-0.0216258, -0.0161991] (12/12 日; 2000/2000 抽样)。
- gru-hawkes vs gru-base · interarrival_cv2_error：0.0430301 [-0.0914731, 0.191618] (12/12 日; 2000/2000 抽样)。
  - W3_thin_short 单世界配对效应：0.0216518 [-0.0299033, 0.0819074] (12/12 日; 2000/2000 抽样)。
  - W6_thin_hawkes 单世界配对效应：0.0646819 [-0.0836055, 0.217763] (12/12 日; 2000/2000 抽样)。
- gru-hawkes vs gru-base · zero_wait_rate_error：0.000233403 [-0.00614488, 0.00836671] (12/12 日; 2000/2000 抽样)。
  - W3_thin_short 单世界配对效应：0.000404868 [-0.000795674, 0.00129604] (12/12 日; 2000/2000 抽样)。
  - W6_thin_hawkes 单世界配对效应：0.000638271 [-0.00537905, 0.00854492] (12/12 日; 2000/2000 抽样)。
- selective-ssm vs gru-base · counterfactual_impact_mae：17.1868 — (4/12 日; 1985/2000 抽样)。
- selective-ssm vs gru-base · dt_nll：-0.00134591 [-0.00278478, 0.000234465] (12/12 日; 2000/2000 抽样)。
  - W3_thin_short 单世界配对效应：0.0029628 [0.00167405, 0.00426658] (12/12 日; 2000/2000 抽样)。
  - W6_thin_hawkes 单世界配对效应：0.00161688 [0.000237416, 0.00284742] (12/12 日; 2000/2000 抽样)。
- selective-ssm vs gru-base · interarrival_cv2_error：-0.0764846 [-0.168014, 0.0149388] (12/12 日; 2000/2000 抽样)。
  - W3_thin_short 单世界配对效应：0.00925553 [-0.0255356, 0.0443065] (12/12 日; 2000/2000 抽样)。
  - W6_thin_hawkes 单世界配对效应：-0.067229 [-0.15964, 0.0193252] (12/12 日; 2000/2000 抽样)。
- selective-ssm vs gru-base · zero_wait_rate_error：0.000644682 [-0.00169384, 0.00326217] (12/12 日; 2000/2000 抽样)。
  - W3_thin_short 单世界配对效应：0.000162555 [-0.000478101, 0.000776899] (12/12 日; 2000/2000 抽样)。
  - W6_thin_hawkes 单世界配对效应：0.000807237 [-0.00141581, 0.00322754] (12/12 日; 2000/2000 抽样)。

**反事实误差可用性**：110/936 个拟合×评估日读数不可用（E1: 0/432, E2: 54/288, E3: 56/216）。这些是逐拟合日计数，不是独立日数。若有一个窗口缺少完整报价响应，该拟合日的 CF 误差不可用，不填零，不以次级指标替代。配对分析进一步要求各训练种子和对照分支在同一日期共同有值；低于预注册门槛则不产生区间。可用样本的均值不能代表报价耗尽条件下的无条件性能；这种非随机缺失本身是模型局限。

**生成轨迹的有效性和尺度限制**：
- E2 W1_thick_short：各模型耗尽副本率范围 0%–0%。
- E2 W3_thin_short：各模型耗尽副本率范围 77.7778%–88.8889%。
- E3 W3_thin_short：各模型耗尽副本率范围 70.1389%–84.375%。
- E3 W6_thin_hawkes：各模型耗尽副本率范围 69.4444%–87.1528%。
- 绝对 CF MAE 的最大三种子描述性均值：E2 W3_thin_short / flat-readoutgate，65.7346 tick；该世界已知生成器平均响应绝对峰值 2.22852 tick。两者仅作量级检查，不是同一个估计量。
高耗尽率与缺失 CF 提示局限：即使某个似然分项改善，也不表示生成模型已具备完整可用反事实能力。

## 表 5 · 可迁移条件规律与证伪反例冻结

| 假设 | 预注册主估计量 | 冻结判定 | 操作检查 | 主效应 [95% CI] / 支持 | 适用边界 / 反例 |
| --- | --- | --- | --- | --- | --- |
| H1 | equal-weight impact MAE interaction over four backbone/operator comparisons | INCONCLUSIVE | 通过 | -0.00459675 [-0.02039, 0.0141195] (12/12 日; 2000/2000 抽样) | individual operator and backbone results reported; not universal |
| H2 | advect vs no-shift impact MAE interaction; identical parameter count and state size | INCONCLUSIVE | 通过 | -0.0368687 — (5/12 日; 1997/2000 抽样) | matrix vs flat and readout gate are secondary controls |
| H3 | Hawkes vs GRU dt NLL interaction | SUPPORTED_WITH_BOUNDARY | 通过 | -0.0227897 [-0.0260534, -0.0198284] (12/12 日; 2000/2000 抽样) | impact and burst residuals reported separately; no-gain equivalence requires the Poisson interval within ±0.02 nats；泊松等效（±0.02 nats）：支持；0.00400276 [0.00303104, 0.00501053] (12/12 日; 2000/2000 抽样) |

判定只读取预注册后的 `frozen_synthetic_laws.json`：干预核验通过且主效应置信区间完全低于零，支持所述条件规律；完全高于零则证伪；跨零、缺少区间或干预未通过均不可判定。H3 的泊松零增益仅在整个区间位于等效带内时成立。支持的结论仍限定于合成旋钮、所选骨干/算子、状态容量及母单协议，不代表已验证真实市场因果规律。次级指标和单一算子结果不自动继承主假设的判定。

**三条规律的实证解释**：

- H1 **INCONCLUSIVE**：-0.00459675 [-0.02039, 0.0141195] (12/12 日; 2000/2000 抽样)。区间跨零，不能把负方向的点估计解释为已证实收益，也不能据此宣称无效。

- H2 **INCONCLUSIVE**：-0.0368687 — (5/12 日; 1997/2000 抽样)。只有 5 个共同配对日；无预注册有效区间，不作收益或无收益结论。

- H3 **SUPPORTED_WITH_BOUNDARY**：-0.0227897 [-0.0260534, -0.0198284] (12/12 日; 2000/2000 抽样)。只在预注册主效应与适用边界内解释。

- H1 的四个骨干×算子 CF 交互区间中，4/4 个包括零；各算子的方向 ACF 与 G_mem(64) 检查是次级证据，不自动升级为因果收益。

- H3 分解：强簇聚 W6 的 dt NLL 配对差 -0.0187869 [-0.0216258, -0.0161991] (12/12 日; 2000/2000 抽样)；泊松 W3 为 0.00400276 [0.00303104, 0.00501053] (12/12 日; 2000/2000 抽样)。统计上的非零差异可以同时满足事前定义的实际等效带，若泊松差值为正则表现小幅下降；等效不是逐次运行完全相同。时间头的条件收益不能替代冲击预测收益。Hawkes 对 CF 误差的交互为 1.52701 — (3/12 日; 1931/2000 抽样)。

## 外部复现（无私有数据、无 API）

从包含本 M2 报告及其生成器的 Stochopia 版本仓库根目录，使用 Python 3.12 环境。以下命令用独立 worktree 检出锁定的训练代码，在新的临时目录写入证据，避免覆盖已封存的原实验：

```bash
study_repo="$(git rev-parse --show-toplevel)"
study_tmp="$(mktemp -d "${TMPDIR:-/tmp}/stochopia-p2-m2.XXXXXX")"
git -C "$study_repo" worktree add --detach "$study_tmp/locked-src" ef9389f3e1c20b675d1af38eb1ddb28c303d6a1d
(
  cd "$study_tmp/locked-src"
  uv sync --locked --python 3.12 --all-extras --dev
  .venv/bin/python -m experiments.run_p2_m2_synthetic_study --artifact-root "$study_tmp/evidence" --work-root "$study_tmp/cache" --jobs 3
)
cd "$study_repo"
uv sync --locked --python 3.12 --all-extras --dev
.venv/bin/python -m experiments.p2_m2_report "$study_tmp/evidence" "$study_tmp/report.md"
```

机器可读证据位于 `OpenHyra/artifacts/p2-m2-synthetic/`：`registration.json`、`manipulation_check.json`、`experiment_results.csv`、`daily_results.json`、`paired_effects.json`、`frozen_synthetic_laws.json` 与 `execution_summary.json`。逐日结果和全部置信区间的完整精度以机器可读记录为准；本报告为只读摘要，不再做训练或改变冻结结论。
