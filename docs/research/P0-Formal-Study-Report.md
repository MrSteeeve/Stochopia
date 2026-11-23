# Tier 0 正式五组对照研究

GPT‑6.2 Sol 的证据组和指标组分别有 6/6、6/6 次首选达标。DeepSeek V4.1 Flash 的对应次数为 3/6、5/6，无 LLM 基线为 3/6。三块共享相同训练权重与模拟日。每组的六次任务比较菜单位置和搜索种子条件下的选择表现。表中比例与命中数用于描述本次 Tier 0 实验。

## 实验登记

- 四个 LLM 组形成两个供应商与两种输入的交叉设计。两个证据组获得开发诊断和区间，两个指标组获得相同的基础标量快照。两家请求均省略 `temperature`。OpenAI 两组使用 `reasoning_effort=low`，DeepSeek 两组关闭思考模式，两家共用 3,000 输出 token 上限。各块另有一个不调用 LLM 的伪随机顺序基线。
- 模型用 64 个训练日和 32 个开发日，训练 200 步、学习率为 0.001，初始化种子为 103、137、211、293、367。三个块分别使用 `p0-formal-s17`、`p0-formal-s29`、`p0-formal-s41`，每组每题的主动搜索预算为两次试验。冻结测量使用 64 个独立测试日，采用事前锁定的主项门槛、`ΔNLL ≤ +0.06` 护栏及 2,000 次成组日重抽。
- 各块开发计划封存后，在独立的复验位置执行一次 `freeze()` 和一次 `run_verifier()`，再读取已保存的逐日结果重建报告。三份重建报告均与原报告完全一致。公开清单、轮次登记、用量和评估报告可用于重算组级汇总。复核日级重抽依赖本地逐日观测，模型权重也保存在本地。 [正式证据](https://github.com/MrSteeeve/OpenHyra/tree/world-model/artifacts/p0-formal-20261123) · [校准参数](Tier0-Calibration-Summary.md)
- `s17` 第一轮登记后，开发评估因读数的本地类型转换错误在首次测量前暂停。恢复沿用原登记的选择、预测和已记录的请求用量。

## 局限

- **推理配置。** 正式运行中，`gpt-6.2-sol` 请求携带 `reasoning_effort="low"`，`deepseek-flash` 请求携带 `thinking={"type": "disabled"}`。两个供应商各自使用低推理开销的 JSON 模式。共同上限约束 API 输出，内部思维链预算由各厂商的配置决定。
- **固定菜单的区分度。** 每题菜单包含父模和三种改动。`separate_flow_own_price` 的两个联合候选共用 `ΔNLL ≤ +0.06` 护栏，结构复杂度在计分之外。A、B 两题的联合候选冻结 NLL 分别增加 0.0272 和 0.0119 nats，且各块均达标。因此仅凭固定菜单，每题都选联合候选即可取得 6/6。两组 GPT 的实际首选均为 A 题的联合改动和 B 题的单通道价格改动，同获 6/6。跨机制的结构权衡与开放式算子设计留待 P1 检验。见表 1、表 3。

## 表 1 · 结构选择（Z → X）

率以每组六个任务为分母。`abstain` 计作弃权，`keep_parent` 按零差值判定。括号内是均值实际采用的有效任务数。首次开发达标成本按有达标记录的任务计算，单位为累计搜索试验。主项增益取冻结 ΔRMSE 的相反数，A 用自由路径，B 用钳制路径。共同菜单效应缺口是同一冻结支持集上菜单内最优增益与最终候选增益的差值。

| 组 | 任务数 | 首选冻结达标率 | 首次开发达标累计成本均值（有效数） | 首选弃权率 | 最终增益均值（有效数） | 共同菜单效应缺口均值（有效数） |
| --- | --- | --- | --- | --- | --- | --- |
| `evidence_gpt` | 6 | 6/6 (100.0%) | 1.0000 (6) | 0/6 (0.0%) | 5.6048 (6) | 0.0000 (6) |
| `metrics_gpt` | 6 | 6/6 (100.0%) | 1.0000 (6) | 0/6 (0.0%) | 5.6048 (6) | 0.0000 (6) |
| `evidence_deepseek` | 6 | 3/6 (50.0%) | 1.0000 (5) | 0/6 (0.0%) | 4.7163 (6) | 0.8885 (6) |
| `metrics_deepseek` | 6 | 5/6 (83.3%) | 1.0000 (6) | 0/6 (0.0%) | 5.6048 (6) | 0.0000 (6) |
| `no_llm` | 6 | 3/6 (50.0%) | 1.5000 (6) | 0/6 (0.0%) | 5.6048 (6) | 0.0000 (6) |

### 分块逐题

| 块 | 组 | 题 | 首选 | 首选冻结状态 | 首次开发达标成本 | 总搜索成本 | 最终候选 | 最终冻结状态 | 最终增益 | 共同菜单效应缺口 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p0-formal-s17 | `evidence_gpt` | case-a | `c_a3` | supported | 1.0000 | 1.0000 | `c_a3` | supported | 8.4853 | 0.0000 |
| p0-formal-s17 | `evidence_gpt` | case-b | `c_b2` | supported | 1.0000 | 1.0000 | `c_b2` | supported | 2.7243 | 0.0000 |
| p0-formal-s17 | `metrics_gpt` | case-a | `c_a3` | supported | 1.0000 | 1.0000 | `c_a3` | supported | 8.4853 | 0.0000 |
| p0-formal-s17 | `metrics_gpt` | case-b | `c_b2` | supported | 1.0000 | 1.0000 | `c_b2` | supported | 2.7243 | 0.0000 |
| p0-formal-s17 | `evidence_deepseek` | case-a | `c_a3` | supported | 1.0000 | 1.0000 | `c_a3` | supported | 8.4853 | 0.0000 |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b0` | refuted | — | 0.0000 | `c_b0` | refuted | 0.0000 | 2.7243 |
| p0-formal-s17 | `metrics_deepseek` | case-a | `c_a3` | supported | 1.0000 | 1.0000 | `c_a3` | supported | 8.4853 | 0.0000 |
| p0-formal-s17 | `metrics_deepseek` | case-b | `c_b2` | supported | 1.0000 | 1.0000 | `c_b2` | supported | 2.7243 | 0.0000 |
| p0-formal-s17 | `no_llm` | case-a | `c_a3` | supported | 1.0000 | 1.0000 | `c_a3` | supported | 8.4853 | 0.0000 |
| p0-formal-s17 | `no_llm` | case-b | `c_b3` | refuted | 2.0000 | 2.0000 | `c_b1` | supported | 2.7243 | 0.0000 |
| p0-formal-s29 | `evidence_gpt` | case-a | `c_a1` | supported | 1.0000 | 1.0000 | `c_a1` | supported | 8.4853 | 0.0000 |
| p0-formal-s29 | `evidence_gpt` | case-b | `c_b3` | supported | 1.0000 | 1.0000 | `c_b3` | supported | 2.7243 | 0.0000 |
| p0-formal-s29 | `metrics_gpt` | case-a | `c_a1` | supported | 1.0000 | 1.0000 | `c_a1` | supported | 8.4853 | 0.0000 |
| p0-formal-s29 | `metrics_gpt` | case-b | `c_b3` | supported | 1.0000 | 1.0000 | `c_b3` | supported | 2.7243 | 0.0000 |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a0` | refuted | 1.0000 | 1.0000 | `c_a2` | supported | 5.8784 | 2.6069 |
| p0-formal-s29 | `evidence_deepseek` | case-b | `c_b3` | supported | 1.0000 | 1.0000 | `c_b3` | supported | 2.7243 | 0.0000 |
| p0-formal-s29 | `metrics_deepseek` | case-a | `c_a1` | supported | 1.0000 | 1.0000 | `c_a1` | supported | 8.4853 | 0.0000 |
| p0-formal-s29 | `metrics_deepseek` | case-b | `c_b3` | supported | 1.0000 | 1.0000 | `c_b3` | supported | 2.7243 | 0.0000 |
| p0-formal-s29 | `no_llm` | case-a | `c_a3` | refuted | 2.0000 | 2.0000 | `c_a1` | supported | 8.4853 | 0.0000 |
| p0-formal-s29 | `no_llm` | case-b | `c_b1` | refuted | 2.0000 | 2.0000 | `c_b3` | supported | 2.7243 | 0.0000 |
| p0-formal-s41 | `evidence_gpt` | case-a | `c_a2` | supported | 1.0000 | 1.0000 | `c_a2` | supported | 8.4853 | 0.0000 |
| p0-formal-s41 | `evidence_gpt` | case-b | `c_b1` | supported | 1.0000 | 1.0000 | `c_b1` | supported | 2.7243 | 0.0000 |
| p0-formal-s41 | `metrics_gpt` | case-a | `c_a2` | supported | 1.0000 | 1.0000 | `c_a2` | supported | 8.4853 | 0.0000 |
| p0-formal-s41 | `metrics_gpt` | case-b | `c_b1` | supported | 1.0000 | 1.0000 | `c_b1` | supported | 2.7243 | 0.0000 |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a2` | supported | 1.0000 | 1.0000 | `c_a2` | supported | 8.4853 | 0.0000 |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b0` | refuted | 1.0000 | 1.0000 | `c_b1` | supported | 2.7243 | 0.0000 |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a0` | refuted | 1.0000 | 1.0000 | `c_a2` | supported | 8.4853 | 0.0000 |
| p0-formal-s41 | `metrics_deepseek` | case-b | `c_b3` | supported | 1.0000 | 1.0000 | `c_b3` | supported | 2.7243 | 0.0000 |
| p0-formal-s41 | `no_llm` | case-a | `c_a2` | supported | 1.0000 | 1.0000 | `c_a2` | supported | 8.4853 | 0.0000 |
| p0-formal-s41 | `no_llm` | case-b | `c_b3` | supported | 1.0000 | 1.0000 | `c_b3` | supported | 2.7243 | 0.0000 |

## 表 2 · 事前定量预测（X → ΔY）

GPT‑6.2 Sol 的证据组和指标组分别有 49、35 条整条预测获得支持。DeepSeek V4.1 Flash 的对应条数为 9、20。共同预测集取第一轮四组共同登记的候选及读数。六个候选各有七个读数，每块每组登记 42 条，合计每组 126 条。覆盖率为已提交除以共同预测数，可判定率为 `SUPPORTED` 与 `REFUTED` 合计除以已提交数。三个分项命中率均以该分项有明确支持或反驳的预测为分母。`UNAVAILABLE` 对应缺失测量。

| 块 | 组 | 共同预测数 | 覆盖率 | 可判定率 | 方向命中率 | 幅度命中率 | 等效命中率 | SUPPORTED | REFUTED | INCONCLUSIVE | ABSTAINED | UNAVAILABLE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 合计 | `evidence_gpt` | 126 | 85/126 (67.5%) | 76/85 (89.4%) | 45/53 (84.9%) | 26/47 (55.3%) | 26/29 (89.7%) | 49 | 27 | 9 | 41 | 0 |
| 合计 | `metrics_gpt` | 126 | 78/126 (61.9%) | 69/78 (88.5%) | 43/49 (87.8%) | 14/43 (32.6%) | 23/26 (88.5%) | 35 | 34 | 9 | 48 | 0 |
| 合计 | `evidence_deepseek` | 126 | 105/126 (83.3%) | 103/105 (98.1%) | 33/88 (37.5%) | 3/86 (3.5%) | 9/16 (56.2%) | 9 | 94 | 2 | 21 | 0 |
| 合计 | `metrics_deepseek` | 126 | 126/126 (100.0%) | 120/126 (95.2%) | 39/96 (40.6%) | 6/92 (6.5%) | 17/27 (63.0%) | 20 | 100 | 6 | 0 | 0 |
| p0-formal-s17 | `evidence_gpt` | 42 | 31/42 (73.8%) | 30/31 (96.8%) | 17/21 (81.0%) | 14/21 (66.7%) | 9/9 (100.0%) | 22 | 8 | 1 | 11 | 0 |
| p0-formal-s17 | `metrics_gpt` | 42 | 29/42 (69.0%) | 24/29 (82.8%) | 16/19 (84.2%) | 5/15 (33.3%) | 8/9 (88.9%) | 12 | 12 | 5 | 13 | 0 |
| p0-formal-s17 | `evidence_deepseek` | 42 | 42/42 (100.0%) | 40/42 (95.2%) | 17/33 (51.5%) | 1/32 (3.1%) | 4/8 (50.0%) | 4 | 36 | 2 | 0 | 0 |
| p0-formal-s17 | `metrics_deepseek` | 42 | 42/42 (100.0%) | 39/42 (92.9%) | 12/38 (31.6%) | 2/36 (5.6%) | 3/3 (100.0%) | 3 | 36 | 3 | 0 | 0 |
| p0-formal-s29 | `evidence_gpt` | 42 | 29/42 (69.0%) | 25/29 (86.2%) | 14/17 (82.4%) | 6/14 (42.9%) | 9/11 (81.8%) | 14 | 11 | 4 | 13 | 0 |
| p0-formal-s29 | `metrics_gpt` | 42 | 22/42 (52.4%) | 19/22 (86.4%) | 13/13 (100.0%) | 3/11 (27.3%) | 7/8 (87.5%) | 10 | 9 | 3 | 20 | 0 |
| p0-formal-s29 | `evidence_deepseek` | 42 | 21/42 (50.0%) | 21/21 (100.0%) | 8/19 (42.1%) | 0/19 (0.0%) | 2/2 (100.0%) | 2 | 19 | 0 | 21 | 0 |
| p0-formal-s29 | `metrics_deepseek` | 42 | 42/42 (100.0%) | 41/42 (97.6%) | 15/29 (51.7%) | 1/29 (3.4%) | 10/12 (83.3%) | 10 | 31 | 1 | 0 | 0 |
| p0-formal-s41 | `evidence_gpt` | 42 | 25/42 (59.5%) | 21/25 (84.0%) | 14/15 (93.3%) | 6/12 (50.0%) | 8/9 (88.9%) | 13 | 8 | 4 | 17 | 0 |
| p0-formal-s41 | `metrics_gpt` | 42 | 27/42 (64.3%) | 26/27 (96.3%) | 14/17 (82.4%) | 6/17 (35.3%) | 8/9 (88.9%) | 13 | 13 | 1 | 15 | 0 |
| p0-formal-s41 | `evidence_deepseek` | 42 | 42/42 (100.0%) | 42/42 (100.0%) | 8/36 (22.2%) | 2/35 (5.7%) | 3/6 (50.0%) | 3 | 39 | 0 | 0 | 0 |
| p0-formal-s41 | `metrics_deepseek` | 42 | 42/42 (100.0%) | 40/42 (95.2%) | 12/29 (41.4%) | 3/27 (11.1%) | 4/12 (33.3%) | 7 | 33 | 2 | 0 | 0 |

## 表 3 · 冻结测试集上的配对模型表现

按块统计，18/18 个候选的冻结门控和主校准一致。绝对值和点估计采用相同的对齐日与训练种子。各候选相对同题父模计算有符号 Δ，冻结区间来自一次性复验。开发列来自锁定参数的主校准。门控状态按同一候选的全部门控判定，因而在各读数行重复。

| 块 | 题 | 候选 / 父模 | 校准对应 | 读数 | 冻结父模值 | 冻结候选值 | 冻结 Δ [95% CI] | 主校准 Δ [95% CI] | 配对日 | 训练种子 | 冻结 / 开发门控 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p0-formal-s17 | case-a | `c_a1` / `c_a0` | `x01` | background_bias/execution (signed_trade) | 0.3946 | 0.0000 | -0.3946 [-0.4105, -0.3795] | -0.3837 [-0.4005, -0.3672] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-a | `c_a1` / `c_a0` | `x01` | clamped_rmse/full (tick) | 3.3259 | 3.3259 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-a | `c_a1` / `c_a0` | `x01` | free_rmse/full (tick) | 9.4667 | 3.5882 | -5.8784 [-6.0950, -5.6670] | -5.8565 [-6.1471, -5.5859] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-a | `c_a1` / `c_a0` | `x01` | end_response/end (tick) | 5.6829 | 5.6829 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-a | `c_a1` / `c_a0` | `x01` | nll/full (nats) | 0.1435 | 0.1435 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-a | `c_a1` / `c_a0` | `x01` | background_bias/recovery (signed_trade) | 0.2398 | 0.0000 | -0.2398 [-0.2563, -0.2239] | -0.2696 [-0.2990, -0.2401] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-a | `c_a1` / `c_a0` | `x01` | free_end_response/end (tick) | 14.9702 | 6.1584 | -8.8117 [-9.2243, -8.4075] | -9.0565 [-9.6334, -8.4978] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-a | `c_a2` / `c_a0` | `a_self` | background_bias/execution (signed_trade) | 0.3946 | 0.3681 | -0.0265 [-0.0311, -0.0219] | -0.0228 [-0.0290, -0.0170] | 64 | 5 | refuted / refuted |
| p0-formal-s17 | case-a | `c_a2` / `c_a0` | `a_self` | clamped_rmse/full (tick) | 3.3259 | 0.9854 | -2.3406 [-2.5169, -2.1527] | -2.3379 [-2.5277, -2.1369] | 64 | 5 | refuted / refuted |
| p0-formal-s17 | case-a | `c_a2` / `c_a0` | `a_self` | free_rmse/full (tick) | 9.4667 | 7.6188 | -1.8478 [-1.9701, -1.7274] | -1.8730 [-2.0135, -1.7387] | 64 | 5 | refuted / refuted |
| p0-formal-s17 | case-a | `c_a2` / `c_a0` | `a_self` | end_response/end (tick) | 5.6829 | 1.4988 | -4.1842 [-4.3227, -4.0459] | -4.1811 [-4.3875, -3.9804] | 64 | 5 | refuted / refuted |
| p0-formal-s17 | case-a | `c_a2` / `c_a0` | `a_self` | nll/full (nats) | 0.1435 | 0.1692 | 0.0257 [0.0244, 0.0272] | 0.0267 [0.0245, 0.0289] | 64 | 5 | refuted / refuted |
| p0-formal-s17 | case-a | `c_a2` / `c_a0` | `a_self` | background_bias/recovery (signed_trade) | 0.2398 | 0.2266 | -0.0132 [-0.0193, -0.0071] | -0.0163 [-0.0249, -0.0090] | 64 | 5 | refuted / refuted |
| p0-formal-s17 | case-a | `c_a2` / `c_a0` | `a_self` | free_end_response/end (tick) | 14.9702 | 12.9023 | -2.0679 [-2.2899, -1.8551] | -2.1514 [-2.3530, -1.9557] | 64 | 5 | refuted / refuted |
| p0-formal-s17 | case-a | `c_a3` / `c_a0` | `a_joint` | background_bias/execution (signed_trade) | 0.3946 | 0.0000 | -0.3946 [-0.4105, -0.3795] | -0.3837 [-0.4005, -0.3672] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-a | `c_a3` / `c_a0` | `a_joint` | clamped_rmse/full (tick) | 3.3259 | 0.9854 | -2.3406 [-2.5169, -2.1527] | -2.3379 [-2.5277, -2.1369] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-a | `c_a3` / `c_a0` | `a_joint` | free_rmse/full (tick) | 9.4667 | 0.9814 | -8.4853 [-8.7119, -8.2605] | -8.5907 [-8.8369, -8.3430] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-a | `c_a3` / `c_a0` | `a_joint` | end_response/end (tick) | 5.6829 | 1.4988 | -4.1842 [-4.3227, -4.0459] | -4.1811 [-4.3875, -3.9804] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-a | `c_a3` / `c_a0` | `a_joint` | nll/full (nats) | 0.1435 | 0.1707 | 0.0272 [0.0262, 0.0282] | 0.0274 [0.0257, 0.0291] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-a | `c_a3` / `c_a0` | `a_joint` | background_bias/recovery (signed_trade) | 0.2398 | 0.0000 | -0.2398 [-0.2563, -0.2239] | -0.2696 [-0.2990, -0.2401] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-a | `c_a3` / `c_a0` | `a_joint` | free_end_response/end (tick) | 14.9702 | 1.8672 | -13.1030 [-13.4862, -12.7010] | -13.5899 [-14.1185, -13.0930] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-b | `c_b1` / `c_b0` | `b_joint` | background_bias/execution (signed_trade) | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-b | `c_b1` / `c_b0` | `b_joint` | clamped_rmse/full (tick) | 3.6983 | 0.9740 | -2.7243 [-2.8866, -2.5586] | -2.8750 [-3.1229, -2.6271] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-b | `c_b1` / `c_b0` | `b_joint` | free_rmse/full (tick) | 4.1058 | 1.0104 | -3.0954 [-3.1928, -2.9821] | -3.0825 [-3.2263, -2.9330] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-b | `c_b1` / `c_b0` | `b_joint` | end_response/end (tick) | 6.1630 | 1.5304 | -4.6327 [-4.7882, -4.4863] | -4.8219 [-5.0762, -4.5836] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-b | `c_b1` / `c_b0` | `b_joint` | nll/full (nats) | 0.3945 | 0.4064 | 0.0119 [0.0099, 0.0138] | 0.0172 [0.0134, 0.0216] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-b | `c_b1` / `c_b0` | `b_joint` | background_bias/recovery (signed_trade) | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-b | `c_b1` / `c_b0` | `b_joint` | free_end_response/end (tick) | 6.8182 | 1.9703 | -4.8479 [-4.9742, -4.7176] | -4.9863 [-5.1376, -4.8269] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-b | `c_b2` / `c_b0` | `x12` | background_bias/execution (signed_trade) | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-b | `c_b2` / `c_b0` | `x12` | clamped_rmse/full (tick) | 3.6983 | 0.9740 | -2.7243 [-2.8866, -2.5586] | -2.8750 [-3.1229, -2.6271] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-b | `c_b2` / `c_b0` | `x12` | free_rmse/full (tick) | 4.1058 | 0.9933 | -3.1125 [-3.2079, -2.9999] | -3.1008 [-3.2433, -2.9542] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-b | `c_b2` / `c_b0` | `x12` | end_response/end (tick) | 6.1630 | 1.5304 | -4.6327 [-4.7882, -4.4863] | -4.8219 [-5.0762, -4.5836] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-b | `c_b2` / `c_b0` | `x12` | nll/full (nats) | 0.3945 | 0.4056 | 0.0111 [0.0094, 0.0128] | 0.0146 [0.0116, 0.0179] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-b | `c_b2` / `c_b0` | `x12` | background_bias/recovery (signed_trade) | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-b | `c_b2` / `c_b0` | `x12` | free_end_response/end (tick) | 6.8182 | 2.1010 | -4.7172 [-4.8391, -4.5971] | -4.9072 [-5.0638, -4.7410] | 64 | 5 | supported / supported |
| p0-formal-s17 | case-b | `c_b3` / `c_b0` | `x13` | background_bias/execution (signed_trade) | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | refuted / refuted |
| p0-formal-s17 | case-b | `c_b3` / `c_b0` | `x13` | clamped_rmse/full (tick) | 3.6983 | 3.6983 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | refuted / refuted |
| p0-formal-s17 | case-b | `c_b3` / `c_b0` | `x13` | free_rmse/full (tick) | 4.1058 | 3.9764 | -0.1294 [-0.1656, -0.0926] | -0.0957 [-0.1469, -0.0457] | 64 | 5 | refuted / refuted |
| p0-formal-s17 | case-b | `c_b3` / `c_b0` | `x13` | end_response/end (tick) | 6.1630 | 6.1630 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | refuted / refuted |
| p0-formal-s17 | case-b | `c_b3` / `c_b0` | `x13` | nll/full (nats) | 0.3945 | 0.3953 | 0.0008 [-0.0004, 0.0018] | 0.0026 [0.0011, 0.0042] | 64 | 5 | refuted / refuted |
| p0-formal-s17 | case-b | `c_b3` / `c_b0` | `x13` | background_bias/recovery (signed_trade) | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | refuted / refuted |
| p0-formal-s17 | case-b | `c_b3` / `c_b0` | `x13` | free_end_response/end (tick) | 6.8182 | 6.5898 | -0.2284 [-0.2935, -0.1660] | -0.1744 [-0.2634, -0.0893] | 64 | 5 | refuted / refuted |
| p0-formal-s29 | case-a | `c_a1` / `c_a0` | `a_joint` | background_bias/execution (signed_trade) | 0.3946 | 0.0000 | -0.3946 [-0.4105, -0.3795] | -0.3837 [-0.4005, -0.3672] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-a | `c_a1` / `c_a0` | `a_joint` | clamped_rmse/full (tick) | 3.3259 | 0.9854 | -2.3406 [-2.5169, -2.1527] | -2.3379 [-2.5277, -2.1369] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-a | `c_a1` / `c_a0` | `a_joint` | free_rmse/full (tick) | 9.4667 | 0.9814 | -8.4853 [-8.7119, -8.2605] | -8.5907 [-8.8369, -8.3430] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-a | `c_a1` / `c_a0` | `a_joint` | end_response/end (tick) | 5.6829 | 1.4988 | -4.1842 [-4.3227, -4.0459] | -4.1811 [-4.3875, -3.9804] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-a | `c_a1` / `c_a0` | `a_joint` | nll/full (nats) | 0.1435 | 0.1707 | 0.0272 [0.0262, 0.0282] | 0.0274 [0.0257, 0.0291] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-a | `c_a1` / `c_a0` | `a_joint` | background_bias/recovery (signed_trade) | 0.2398 | 0.0000 | -0.2398 [-0.2563, -0.2239] | -0.2696 [-0.2990, -0.2401] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-a | `c_a1` / `c_a0` | `a_joint` | free_end_response/end (tick) | 14.9702 | 1.8672 | -13.1030 [-13.4862, -12.7010] | -13.5899 [-14.1185, -13.0930] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-a | `c_a2` / `c_a0` | `x01` | background_bias/execution (signed_trade) | 0.3946 | 0.0000 | -0.3946 [-0.4105, -0.3795] | -0.3837 [-0.4005, -0.3672] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-a | `c_a2` / `c_a0` | `x01` | clamped_rmse/full (tick) | 3.3259 | 3.3259 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-a | `c_a2` / `c_a0` | `x01` | free_rmse/full (tick) | 9.4667 | 3.5882 | -5.8784 [-6.0950, -5.6670] | -5.8565 [-6.1471, -5.5859] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-a | `c_a2` / `c_a0` | `x01` | end_response/end (tick) | 5.6829 | 5.6829 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-a | `c_a2` / `c_a0` | `x01` | nll/full (nats) | 0.1435 | 0.1435 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-a | `c_a2` / `c_a0` | `x01` | background_bias/recovery (signed_trade) | 0.2398 | 0.0000 | -0.2398 [-0.2563, -0.2239] | -0.2696 [-0.2990, -0.2401] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-a | `c_a2` / `c_a0` | `x01` | free_end_response/end (tick) | 14.9702 | 6.1584 | -8.8117 [-9.2243, -8.4075] | -9.0565 [-9.6334, -8.4978] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-a | `c_a3` / `c_a0` | `a_self` | background_bias/execution (signed_trade) | 0.3946 | 0.3681 | -0.0265 [-0.0311, -0.0219] | -0.0228 [-0.0290, -0.0170] | 64 | 5 | refuted / refuted |
| p0-formal-s29 | case-a | `c_a3` / `c_a0` | `a_self` | clamped_rmse/full (tick) | 3.3259 | 0.9854 | -2.3406 [-2.5169, -2.1527] | -2.3379 [-2.5277, -2.1369] | 64 | 5 | refuted / refuted |
| p0-formal-s29 | case-a | `c_a3` / `c_a0` | `a_self` | free_rmse/full (tick) | 9.4667 | 7.6188 | -1.8478 [-1.9701, -1.7274] | -1.8730 [-2.0135, -1.7387] | 64 | 5 | refuted / refuted |
| p0-formal-s29 | case-a | `c_a3` / `c_a0` | `a_self` | end_response/end (tick) | 5.6829 | 1.4988 | -4.1842 [-4.3227, -4.0459] | -4.1811 [-4.3875, -3.9804] | 64 | 5 | refuted / refuted |
| p0-formal-s29 | case-a | `c_a3` / `c_a0` | `a_self` | nll/full (nats) | 0.1435 | 0.1692 | 0.0257 [0.0244, 0.0272] | 0.0267 [0.0245, 0.0289] | 64 | 5 | refuted / refuted |
| p0-formal-s29 | case-a | `c_a3` / `c_a0` | `a_self` | background_bias/recovery (signed_trade) | 0.2398 | 0.2266 | -0.0132 [-0.0193, -0.0071] | -0.0163 [-0.0249, -0.0090] | 64 | 5 | refuted / refuted |
| p0-formal-s29 | case-a | `c_a3` / `c_a0` | `a_self` | free_end_response/end (tick) | 14.9702 | 12.9023 | -2.0679 [-2.2899, -1.8551] | -2.1514 [-2.3530, -1.9557] | 64 | 5 | refuted / refuted |
| p0-formal-s29 | case-b | `c_b1` / `c_b0` | `x13` | background_bias/execution (signed_trade) | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | refuted / refuted |
| p0-formal-s29 | case-b | `c_b1` / `c_b0` | `x13` | clamped_rmse/full (tick) | 3.6983 | 3.6983 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | refuted / refuted |
| p0-formal-s29 | case-b | `c_b1` / `c_b0` | `x13` | free_rmse/full (tick) | 4.1058 | 3.9764 | -0.1294 [-0.1656, -0.0926] | -0.0957 [-0.1469, -0.0457] | 64 | 5 | refuted / refuted |
| p0-formal-s29 | case-b | `c_b1` / `c_b0` | `x13` | end_response/end (tick) | 6.1630 | 6.1630 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | refuted / refuted |
| p0-formal-s29 | case-b | `c_b1` / `c_b0` | `x13` | nll/full (nats) | 0.3945 | 0.3953 | 0.0008 [-0.0004, 0.0018] | 0.0026 [0.0011, 0.0042] | 64 | 5 | refuted / refuted |
| p0-formal-s29 | case-b | `c_b1` / `c_b0` | `x13` | background_bias/recovery (signed_trade) | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | refuted / refuted |
| p0-formal-s29 | case-b | `c_b1` / `c_b0` | `x13` | free_end_response/end (tick) | 6.8182 | 6.5898 | -0.2284 [-0.2935, -0.1660] | -0.1744 [-0.2634, -0.0893] | 64 | 5 | refuted / refuted |
| p0-formal-s29 | case-b | `c_b2` / `c_b0` | `b_joint` | background_bias/execution (signed_trade) | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-b | `c_b2` / `c_b0` | `b_joint` | clamped_rmse/full (tick) | 3.6983 | 0.9740 | -2.7243 [-2.8866, -2.5586] | -2.8750 [-3.1229, -2.6271] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-b | `c_b2` / `c_b0` | `b_joint` | free_rmse/full (tick) | 4.1058 | 1.0104 | -3.0954 [-3.1928, -2.9821] | -3.0825 [-3.2263, -2.9330] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-b | `c_b2` / `c_b0` | `b_joint` | end_response/end (tick) | 6.1630 | 1.5304 | -4.6327 [-4.7882, -4.4863] | -4.8219 [-5.0762, -4.5836] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-b | `c_b2` / `c_b0` | `b_joint` | nll/full (nats) | 0.3945 | 0.4064 | 0.0119 [0.0099, 0.0138] | 0.0172 [0.0134, 0.0216] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-b | `c_b2` / `c_b0` | `b_joint` | background_bias/recovery (signed_trade) | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-b | `c_b2` / `c_b0` | `b_joint` | free_end_response/end (tick) | 6.8182 | 1.9703 | -4.8479 [-4.9742, -4.7176] | -4.9863 [-5.1376, -4.8269] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-b | `c_b3` / `c_b0` | `x12` | background_bias/execution (signed_trade) | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-b | `c_b3` / `c_b0` | `x12` | clamped_rmse/full (tick) | 3.6983 | 0.9740 | -2.7243 [-2.8866, -2.5586] | -2.8750 [-3.1229, -2.6271] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-b | `c_b3` / `c_b0` | `x12` | free_rmse/full (tick) | 4.1058 | 0.9933 | -3.1125 [-3.2079, -2.9999] | -3.1008 [-3.2433, -2.9542] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-b | `c_b3` / `c_b0` | `x12` | end_response/end (tick) | 6.1630 | 1.5304 | -4.6327 [-4.7882, -4.4863] | -4.8219 [-5.0762, -4.5836] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-b | `c_b3` / `c_b0` | `x12` | nll/full (nats) | 0.3945 | 0.4056 | 0.0111 [0.0094, 0.0128] | 0.0146 [0.0116, 0.0179] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-b | `c_b3` / `c_b0` | `x12` | background_bias/recovery (signed_trade) | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s29 | case-b | `c_b3` / `c_b0` | `x12` | free_end_response/end (tick) | 6.8182 | 2.1010 | -4.7172 [-4.8391, -4.5971] | -4.9072 [-5.0638, -4.7410] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-a | `c_a1` / `c_a0` | `a_self` | background_bias/execution (signed_trade) | 0.3946 | 0.3681 | -0.0265 [-0.0311, -0.0219] | -0.0228 [-0.0290, -0.0170] | 64 | 5 | refuted / refuted |
| p0-formal-s41 | case-a | `c_a1` / `c_a0` | `a_self` | clamped_rmse/full (tick) | 3.3259 | 0.9854 | -2.3406 [-2.5169, -2.1527] | -2.3379 [-2.5277, -2.1369] | 64 | 5 | refuted / refuted |
| p0-formal-s41 | case-a | `c_a1` / `c_a0` | `a_self` | free_rmse/full (tick) | 9.4667 | 7.6188 | -1.8478 [-1.9701, -1.7274] | -1.8730 [-2.0135, -1.7387] | 64 | 5 | refuted / refuted |
| p0-formal-s41 | case-a | `c_a1` / `c_a0` | `a_self` | end_response/end (tick) | 5.6829 | 1.4988 | -4.1842 [-4.3227, -4.0459] | -4.1811 [-4.3875, -3.9804] | 64 | 5 | refuted / refuted |
| p0-formal-s41 | case-a | `c_a1` / `c_a0` | `a_self` | nll/full (nats) | 0.1435 | 0.1692 | 0.0257 [0.0244, 0.0272] | 0.0267 [0.0245, 0.0289] | 64 | 5 | refuted / refuted |
| p0-formal-s41 | case-a | `c_a1` / `c_a0` | `a_self` | background_bias/recovery (signed_trade) | 0.2398 | 0.2266 | -0.0132 [-0.0193, -0.0071] | -0.0163 [-0.0249, -0.0090] | 64 | 5 | refuted / refuted |
| p0-formal-s41 | case-a | `c_a1` / `c_a0` | `a_self` | free_end_response/end (tick) | 14.9702 | 12.9023 | -2.0679 [-2.2899, -1.8551] | -2.1514 [-2.3530, -1.9557] | 64 | 5 | refuted / refuted |
| p0-formal-s41 | case-a | `c_a2` / `c_a0` | `a_joint` | background_bias/execution (signed_trade) | 0.3946 | 0.0000 | -0.3946 [-0.4105, -0.3795] | -0.3837 [-0.4005, -0.3672] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-a | `c_a2` / `c_a0` | `a_joint` | clamped_rmse/full (tick) | 3.3259 | 0.9854 | -2.3406 [-2.5169, -2.1527] | -2.3379 [-2.5277, -2.1369] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-a | `c_a2` / `c_a0` | `a_joint` | free_rmse/full (tick) | 9.4667 | 0.9814 | -8.4853 [-8.7119, -8.2605] | -8.5907 [-8.8369, -8.3430] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-a | `c_a2` / `c_a0` | `a_joint` | end_response/end (tick) | 5.6829 | 1.4988 | -4.1842 [-4.3227, -4.0459] | -4.1811 [-4.3875, -3.9804] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-a | `c_a2` / `c_a0` | `a_joint` | nll/full (nats) | 0.1435 | 0.1707 | 0.0272 [0.0262, 0.0282] | 0.0274 [0.0257, 0.0291] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-a | `c_a2` / `c_a0` | `a_joint` | background_bias/recovery (signed_trade) | 0.2398 | 0.0000 | -0.2398 [-0.2563, -0.2239] | -0.2696 [-0.2990, -0.2401] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-a | `c_a2` / `c_a0` | `a_joint` | free_end_response/end (tick) | 14.9702 | 1.8672 | -13.1030 [-13.4862, -12.7010] | -13.5899 [-14.1185, -13.0930] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-a | `c_a3` / `c_a0` | `x01` | background_bias/execution (signed_trade) | 0.3946 | 0.0000 | -0.3946 [-0.4105, -0.3795] | -0.3837 [-0.4005, -0.3672] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-a | `c_a3` / `c_a0` | `x01` | clamped_rmse/full (tick) | 3.3259 | 3.3259 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-a | `c_a3` / `c_a0` | `x01` | free_rmse/full (tick) | 9.4667 | 3.5882 | -5.8784 [-6.0950, -5.6670] | -5.8565 [-6.1471, -5.5859] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-a | `c_a3` / `c_a0` | `x01` | end_response/end (tick) | 5.6829 | 5.6829 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-a | `c_a3` / `c_a0` | `x01` | nll/full (nats) | 0.1435 | 0.1435 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-a | `c_a3` / `c_a0` | `x01` | background_bias/recovery (signed_trade) | 0.2398 | 0.0000 | -0.2398 [-0.2563, -0.2239] | -0.2696 [-0.2990, -0.2401] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-a | `c_a3` / `c_a0` | `x01` | free_end_response/end (tick) | 14.9702 | 6.1584 | -8.8117 [-9.2243, -8.4075] | -9.0565 [-9.6334, -8.4978] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-b | `c_b1` / `c_b0` | `x12` | background_bias/execution (signed_trade) | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-b | `c_b1` / `c_b0` | `x12` | clamped_rmse/full (tick) | 3.6983 | 0.9740 | -2.7243 [-2.8866, -2.5586] | -2.8750 [-3.1229, -2.6271] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-b | `c_b1` / `c_b0` | `x12` | free_rmse/full (tick) | 4.1058 | 0.9933 | -3.1125 [-3.2079, -2.9999] | -3.1008 [-3.2433, -2.9542] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-b | `c_b1` / `c_b0` | `x12` | end_response/end (tick) | 6.1630 | 1.5304 | -4.6327 [-4.7882, -4.4863] | -4.8219 [-5.0762, -4.5836] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-b | `c_b1` / `c_b0` | `x12` | nll/full (nats) | 0.3945 | 0.4056 | 0.0111 [0.0094, 0.0128] | 0.0146 [0.0116, 0.0179] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-b | `c_b1` / `c_b0` | `x12` | background_bias/recovery (signed_trade) | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-b | `c_b1` / `c_b0` | `x12` | free_end_response/end (tick) | 6.8182 | 2.1010 | -4.7172 [-4.8391, -4.5971] | -4.9072 [-5.0638, -4.7410] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-b | `c_b2` / `c_b0` | `x13` | background_bias/execution (signed_trade) | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | refuted / refuted |
| p0-formal-s41 | case-b | `c_b2` / `c_b0` | `x13` | clamped_rmse/full (tick) | 3.6983 | 3.6983 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | refuted / refuted |
| p0-formal-s41 | case-b | `c_b2` / `c_b0` | `x13` | free_rmse/full (tick) | 4.1058 | 3.9764 | -0.1294 [-0.1656, -0.0926] | -0.0957 [-0.1469, -0.0457] | 64 | 5 | refuted / refuted |
| p0-formal-s41 | case-b | `c_b2` / `c_b0` | `x13` | end_response/end (tick) | 6.1630 | 6.1630 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | refuted / refuted |
| p0-formal-s41 | case-b | `c_b2` / `c_b0` | `x13` | nll/full (nats) | 0.3945 | 0.3953 | 0.0008 [-0.0004, 0.0018] | 0.0026 [0.0011, 0.0042] | 64 | 5 | refuted / refuted |
| p0-formal-s41 | case-b | `c_b2` / `c_b0` | `x13` | background_bias/recovery (signed_trade) | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | refuted / refuted |
| p0-formal-s41 | case-b | `c_b2` / `c_b0` | `x13` | free_end_response/end (tick) | 6.8182 | 6.5898 | -0.2284 [-0.2935, -0.1660] | -0.1744 [-0.2634, -0.0893] | 64 | 5 | refuted / refuted |
| p0-formal-s41 | case-b | `c_b3` / `c_b0` | `b_joint` | background_bias/execution (signed_trade) | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-b | `c_b3` / `c_b0` | `b_joint` | clamped_rmse/full (tick) | 3.6983 | 0.9740 | -2.7243 [-2.8866, -2.5586] | -2.8750 [-3.1229, -2.6271] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-b | `c_b3` / `c_b0` | `b_joint` | free_rmse/full (tick) | 4.1058 | 1.0104 | -3.0954 [-3.1928, -2.9821] | -3.0825 [-3.2263, -2.9330] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-b | `c_b3` / `c_b0` | `b_joint` | end_response/end (tick) | 6.1630 | 1.5304 | -4.6327 [-4.7882, -4.4863] | -4.8219 [-5.0762, -4.5836] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-b | `c_b3` / `c_b0` | `b_joint` | nll/full (nats) | 0.3945 | 0.4064 | 0.0119 [0.0099, 0.0138] | 0.0172 [0.0134, 0.0216] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-b | `c_b3` / `c_b0` | `b_joint` | background_bias/recovery (signed_trade) | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 64 | 5 | supported / supported |
| p0-formal-s41 | case-b | `c_b3` / `c_b0` | `b_joint` | free_end_response/end (tick) | 6.8182 | 1.9703 | -4.8479 [-4.9742, -4.7176] | -4.9863 [-5.1376, -4.8269] | 64 | 5 | supported / supported |

## 表 4 · 实际 API 成本与反例审计

仅统计正式轮次请求，包含格式纠正产生的额外请求。耗时是各请求用时之和。输入和输出 token 按接口返回的用量累加，使用响应中通过核对的模型 ID。请求缺失用量时，实验预算按预留量计入。

| 块 | 组 | 响应模型 ID | 实际请求次数 | 有用量记录的调用 | 输入 token | 输出 token | 请求累计耗时（秒） | 预算预留 token | 计入预算 token |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 合计 | — | `gpt-6.2-sol` | 13 | 12 | 16872 | 7025 | 114.9717 | 7861 | 31758 |
| 合计 | — | `deepseek-flash` | 18 | 18 | 27533 | 10933 | 61.1720 | 0 | 38466 |
| p0-formal-s17 | `evidence_gpt` | `gpt-6.2-sol` | 3 | 2 | 2928 | 1201 | 29.4517 | 7861 | 11990 |
| p0-formal-s17 | `metrics_gpt` | `gpt-6.2-sol` | 2 | 2 | 2696 | 1181 | 17.6183 | 0 | 3877 |
| p0-formal-s17 | `evidence_deepseek` | `deepseek-flash` | 4 | 4 | 6363 | 2457 | 14.2577 | 0 | 8820 |
| p0-formal-s17 | `metrics_deepseek` | `deepseek-flash` | 2 | 2 | 2910 | 1245 | 6.1314 | 0 | 4155 |
| p0-formal-s29 | `evidence_gpt` | `gpt-6.2-sol` | 2 | 2 | 2928 | 1168 | 16.6623 | 0 | 4096 |
| p0-formal-s29 | `metrics_gpt` | `gpt-6.2-sol` | 2 | 2 | 2696 | 1146 | 17.0383 | 0 | 3842 |
| p0-formal-s29 | `evidence_deepseek` | `deepseek-flash` | 3 | 3 | 4756 | 1759 | 10.1345 | 0 | 6515 |
| p0-formal-s29 | `metrics_deepseek` | `deepseek-flash` | 2 | 2 | 2910 | 1197 | 7.1956 | 0 | 4107 |
| p0-formal-s41 | `evidence_gpt` | `gpt-6.2-sol` | 2 | 2 | 2928 | 1161 | 16.8956 | 0 | 4089 |
| p0-formal-s41 | `metrics_gpt` | `gpt-6.2-sol` | 2 | 2 | 2696 | 1168 | 17.3054 | 0 | 3864 |
| p0-formal-s41 | `evidence_deepseek` | `deepseek-flash` | 3 | 3 | 4760 | 1838 | 9.7665 | 0 | 6598 |
| p0-formal-s41 | `metrics_deepseek` | `deepseek-flash` | 4 | 4 | 5834 | 2437 | 13.6863 | 0 | 8271 |

正式轮次前的 JSON 连通测试另计。`gpt-6.2-sol` 1 次，17 / 11 输入 / 输出 token。`deepseek-flash` 2 次，121 / 14 输入 / 输出 token。DeepSeek 首次未通过精确内容校验，获准追加的一次复测通过。

### 模型改善但预测被反驳的记录

冻结报告按单条预测计数，共登记 284 条反例，其中 111 条涉及该组同题的最终候选。纳入条件为主项增益大于 0 且预测判为 `REFUTED`。表中最终选择 ID 标记与最终候选吻合的记录。

`p0-formal-s17` 的 A 题中，两个 GPT 组均首选 `c_a3`。其自由路径 RMSE 的冻结 Δ 为 -8.4853 [-8.7119, -8.2605] tick。证据组事前给出 [-9.2000, -5.4000]，整条预测获支持。指标组事前给出 [-7.0000, -2.0000]，方向正确，幅度被反驳。对应的两条预测为 `r1:evidence_gpt:case-a:C3:T3` 与 `r1:metrics_gpt:case-a:C3:T3`。

| 块 | 组 | 题 | 候选 | 轮次 | 读数 | 事前方向 | 事前范围 | 冻结 Δ [95% CI] | 主项增益 | 被反驳分项 | 预测 ID | 测量 ID | 同候选最终选择 ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p0-formal-s17 | `evidence_deepseek` | case-a | `c_a1` | 1 | background_bias/execution | decrease | [-0.2000, -0.0500] | -0.3946 [-0.4105, -0.3795] | 5.8784 | 幅度 | `r1:evidence_deepseek:case-a:C1:T1` | `aggregate:case-a:c_a1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-a | `c_a1` | 1 | clamped_rmse/full | decrease | [-1.0000, -0.3000] | 0.0000 [0.0000, 0.0000] | 5.8784 | 方向/幅度 | `r1:evidence_deepseek:case-a:C1:T2` | `aggregate:case-a:c_a1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-a | `c_a1` | 1 | free_rmse/full | decrease | [-2.5000, -0.6000] | -5.8784 [-6.0950, -5.6670] | 5.8784 | 幅度 | `r1:evidence_deepseek:case-a:C1:T3` | `aggregate:case-a:c_a1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-a | `c_a1` | 1 | end_response/end | decrease | [-1.2000, -0.3000] | 0.0000 [0.0000, 0.0000] | 5.8784 | 方向/幅度 | `r1:evidence_deepseek:case-a:C1:T4` | `aggregate:case-a:c_a1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-a | `c_a1` | 1 | nll/full | decrease | [-0.0500, -0.0100] | 0.0000 [0.0000, 0.0000] | 5.8784 | 方向/幅度 | `r1:evidence_deepseek:case-a:C1:T5` | `aggregate:case-a:c_a1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-a | `c_a1` | 1 | background_bias/recovery | decrease | [-0.2000, -0.0500] | -0.2398 [-0.2563, -0.2239] | 5.8784 | 幅度 | `r1:evidence_deepseek:case-a:C1:T6` | `aggregate:case-a:c_a1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-a | `c_a1` | 1 | free_end_response/end | decrease | [-1.2000, -0.3000] | -8.8117 [-9.2243, -8.4075] | 5.8784 | 幅度 | `r1:evidence_deepseek:case-a:C1:T7` | `aggregate:case-a:c_a1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-a | `c_a2` | 1 | clamped_rmse/full | decrease | [-0.9000, -0.2000] | -2.3406 [-2.5169, -2.1527] | 1.8478 | 幅度 | `r1:evidence_deepseek:case-a:C2:T2` | `aggregate:case-a:c_a2` | — |
| p0-formal-s17 | `evidence_deepseek` | case-a | `c_a2` | 1 | free_rmse/full | decrease | [-2.2000, -0.4000] | -1.8478 [-1.9701, -1.7274] | 1.8478 | 方向 | `r1:evidence_deepseek:case-a:C2:T3` | `aggregate:case-a:c_a2` | — |
| p0-formal-s17 | `evidence_deepseek` | case-a | `c_a2` | 1 | end_response/end | decrease | [-1.0000, -0.2000] | -4.1842 [-4.3227, -4.0459] | 1.8478 | 幅度 | `r1:evidence_deepseek:case-a:C2:T4` | `aggregate:case-a:c_a2` | — |
| p0-formal-s17 | `evidence_deepseek` | case-a | `c_a2` | 1 | nll/full | decrease | [-0.0500, -0.0100] | 0.0257 [0.0244, 0.0272] | 1.8478 | 方向/幅度 | `r1:evidence_deepseek:case-a:C2:T5` | `aggregate:case-a:c_a2` | — |
| p0-formal-s17 | `evidence_deepseek` | case-a | `c_a2` | 1 | free_end_response/end | decrease | [-1.0000, -0.2000] | -2.0679 [-2.2899, -1.8551] | 1.8478 | 幅度 | `r1:evidence_deepseek:case-a:C2:T7` | `aggregate:case-a:c_a2` | — |
| p0-formal-s17 | `evidence_deepseek` | case-a | `c_a3` | 1 | background_bias/execution | decrease | [-0.2500, -0.0500] | -0.3946 [-0.4105, -0.3795] | 8.4853 | 幅度 | `r1:evidence_deepseek:case-a:C3:T1` | `aggregate:case-a:c_a3` | `r1:evidence_deepseek:case-a` |
| p0-formal-s17 | `evidence_deepseek` | case-a | `c_a3` | 1 | clamped_rmse/full | decrease | [-1.3000, -0.3500] | -2.3406 [-2.5169, -2.1527] | 8.4853 | 幅度 | `r1:evidence_deepseek:case-a:C3:T2` | `aggregate:case-a:c_a3` | `r1:evidence_deepseek:case-a` |
| p0-formal-s17 | `evidence_deepseek` | case-a | `c_a3` | 1 | free_rmse/full | decrease | [-3.0000, -0.7000] | -8.4853 [-8.7119, -8.2605] | 8.4853 | 幅度 | `r1:evidence_deepseek:case-a:C3:T3` | `aggregate:case-a:c_a3` | `r1:evidence_deepseek:case-a` |
| p0-formal-s17 | `evidence_deepseek` | case-a | `c_a3` | 1 | end_response/end | decrease | [-1.4000, -0.3500] | -4.1842 [-4.3227, -4.0459] | 8.4853 | 幅度 | `r1:evidence_deepseek:case-a:C3:T4` | `aggregate:case-a:c_a3` | `r1:evidence_deepseek:case-a` |
| p0-formal-s17 | `evidence_deepseek` | case-a | `c_a3` | 1 | nll/full | decrease | [-0.0600, -0.0150] | 0.0272 [0.0262, 0.0282] | 8.4853 | 方向/幅度 | `r1:evidence_deepseek:case-a:C3:T5` | `aggregate:case-a:c_a3` | `r1:evidence_deepseek:case-a` |
| p0-formal-s17 | `evidence_deepseek` | case-a | `c_a3` | 1 | free_end_response/end | decrease | [-1.4000, -0.3500] | -13.1030 [-13.4862, -12.7010] | 8.4853 | 幅度 | `r1:evidence_deepseek:case-a:C3:T7` | `aggregate:case-a:c_a3` | `r1:evidence_deepseek:case-a` |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b1` | 1 | background_bias/execution | decrease | [-0.1200, -0.0200] | 0.0000 [0.0000, 0.0000] | 2.7243 | 方向/幅度 | `r1:evidence_deepseek:case-b:C1:T1` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b1` | 1 | clamped_rmse/full | decrease | [-0.6000, -0.1000] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:evidence_deepseek:case-b:C1:T2` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b1` | 1 | free_rmse/full | decrease | [-0.5000, -0.0500] | -3.0954 [-3.1928, -2.9821] | 2.7243 | 幅度 | `r1:evidence_deepseek:case-b:C1:T3` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b1` | 1 | end_response/end | decrease | [-1.5000, -0.2000] | -4.6327 [-4.7882, -4.4863] | 2.7243 | 幅度 | `r1:evidence_deepseek:case-b:C1:T4` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b1` | 1 | nll/full | decrease | [-0.0400, -0.0050] | 0.0119 [0.0099, 0.0138] | 2.7243 | 方向/幅度 | `r1:evidence_deepseek:case-b:C1:T5` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b1` | 1 | background_bias/recovery | decrease | [-0.1000, -0.0100] | 0.0000 [0.0000, 0.0000] | 2.7243 | 方向/幅度 | `r1:evidence_deepseek:case-b:C1:T6` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b1` | 1 | free_end_response/end | decrease | [-1.2000, -0.1500] | -4.8479 [-4.9742, -4.7176] | 2.7243 | 幅度 | `r1:evidence_deepseek:case-b:C1:T7` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b2` | 1 | clamped_rmse/full | unchanged | [-0.1000, 0.1000] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 等效 | `r1:evidence_deepseek:case-b:C2:T2` | `aggregate:case-b:c_b2` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b2` | 1 | free_rmse/full | unchanged | [-0.2500, 0.2500] | -3.1125 [-3.2079, -2.9999] | 2.7243 | 等效 | `r1:evidence_deepseek:case-b:C2:T3` | `aggregate:case-b:c_b2` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b2` | 1 | end_response/end | unchanged | [-0.2500, 0.2500] | -4.6327 [-4.7882, -4.4863] | 2.7243 | 等效 | `r1:evidence_deepseek:case-b:C2:T4` | `aggregate:case-b:c_b2` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b2` | 1 | free_end_response/end | unchanged | [-0.2500, 0.2500] | -4.7172 [-4.8391, -4.5971] | 2.7243 | 等效 | `r1:evidence_deepseek:case-b:C2:T7` | `aggregate:case-b:c_b2` | — |
| p0-formal-s17 | `evidence_gpt` | case-a | `c_a2` | 1 | free_rmse/full | decrease | [-4.0000, -0.7000] | -1.8478 [-1.9701, -1.7274] | 1.8478 | 方向 | `r1:evidence_gpt:case-a:C2:T3` | `aggregate:case-a:c_a2` | — |
| p0-formal-s17 | `evidence_gpt` | case-a | `c_a2` | 1 | nll/full | decrease | [-0.0800, -0.0200] | 0.0257 [0.0244, 0.0272] | 1.8478 | 方向/幅度 | `r1:evidence_gpt:case-a:C2:T5` | `aggregate:case-a:c_a2` | — |
| p0-formal-s17 | `evidence_gpt` | case-a | `c_a3` | 1 | nll/full | decrease | [-0.0800, -0.0200] | 0.0272 [0.0262, 0.0282] | 8.4853 | 方向/幅度 | `r1:evidence_gpt:case-a:C3:T5` | `aggregate:case-a:c_a3` | `r1:evidence_gpt:case-a` |
| p0-formal-s17 | `evidence_gpt` | case-b | `c_b1` | 1 | clamped_rmse/full | decrease | [-1.2000, -0.1000] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:evidence_gpt:case-b:C1:T2` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `evidence_gpt` | case-b | `c_b1` | 1 | free_rmse/full | decrease | [-1.3000, -0.1000] | -3.0954 [-3.1928, -2.9821] | 2.7243 | 幅度 | `r1:evidence_gpt:case-b:C1:T3` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `evidence_gpt` | case-b | `c_b2` | 1 | clamped_rmse/full | decrease | [-1.2000, -0.1000] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:evidence_gpt:case-b:C2:T2` | `aggregate:case-b:c_b2` | `r1:evidence_gpt:case-b` |
| p0-formal-s17 | `evidence_gpt` | case-b | `c_b2` | 1 | free_rmse/full | decrease | [-1.2000, -0.1000] | -3.1125 [-3.2079, -2.9999] | 2.7243 | 幅度 | `r1:evidence_gpt:case-b:C2:T3` | `aggregate:case-b:c_b2` | `r1:evidence_gpt:case-b` |
| p0-formal-s17 | `evidence_gpt` | case-b | `c_b2` | 1 | nll/full | decrease | [-0.0800, -0.0050] | 0.0111 [0.0094, 0.0128] | 2.7243 | 方向/幅度 | `r1:evidence_gpt:case-b:C2:T5` | `aggregate:case-b:c_b2` | `r1:evidence_gpt:case-b` |
| p0-formal-s17 | `metrics_deepseek` | case-a | `c_a1` | 1 | background_bias/execution | decrease | [-0.2500, -0.1200] | -0.3946 [-0.4105, -0.3795] | 5.8784 | 幅度 | `r1:metrics_deepseek:case-a:C1:T1` | `aggregate:case-a:c_a1` | — |
| p0-formal-s17 | `metrics_deepseek` | case-a | `c_a1` | 1 | clamped_rmse/full | decrease | [-1.5000, -0.5000] | 0.0000 [0.0000, 0.0000] | 5.8784 | 方向/幅度 | `r1:metrics_deepseek:case-a:C1:T2` | `aggregate:case-a:c_a1` | — |
| p0-formal-s17 | `metrics_deepseek` | case-a | `c_a1` | 1 | free_rmse/full | decrease | [-3.0000, -1.0000] | -5.8784 [-6.0950, -5.6670] | 5.8784 | 幅度 | `r1:metrics_deepseek:case-a:C1:T3` | `aggregate:case-a:c_a1` | — |
| p0-formal-s17 | `metrics_deepseek` | case-a | `c_a1` | 1 | end_response/end | increase | [0.7500, 2.0000] | 0.0000 [0.0000, 0.0000] | 5.8784 | 方向/幅度 | `r1:metrics_deepseek:case-a:C1:T4` | `aggregate:case-a:c_a1` | — |
| p0-formal-s17 | `metrics_deepseek` | case-a | `c_a1` | 1 | nll/full | decrease | [-0.0500, -0.0100] | 0.0000 [0.0000, 0.0000] | 5.8784 | 方向/幅度 | `r1:metrics_deepseek:case-a:C1:T5` | `aggregate:case-a:c_a1` | — |
| p0-formal-s17 | `metrics_deepseek` | case-a | `c_a1` | 1 | free_end_response/end | increase | [0.7500, 2.0000] | -8.8117 [-9.2243, -8.4075] | 5.8784 | 方向/幅度 | `r1:metrics_deepseek:case-a:C1:T7` | `aggregate:case-a:c_a1` | — |
| p0-formal-s17 | `metrics_deepseek` | case-a | `c_a2` | 1 | clamped_rmse/full | decrease | [-1.0000, -0.3000] | -2.3406 [-2.5169, -2.1527] | 1.8478 | 幅度 | `r1:metrics_deepseek:case-a:C2:T2` | `aggregate:case-a:c_a2` | — |
| p0-formal-s17 | `metrics_deepseek` | case-a | `c_a2` | 1 | free_rmse/full | decrease | [-2.5000, -0.8000] | -1.8478 [-1.9701, -1.7274] | 1.8478 | 方向 | `r1:metrics_deepseek:case-a:C2:T3` | `aggregate:case-a:c_a2` | — |
| p0-formal-s17 | `metrics_deepseek` | case-a | `c_a2` | 1 | end_response/end | increase | [0.5000, 1.5000] | -4.1842 [-4.3227, -4.0459] | 1.8478 | 方向/幅度 | `r1:metrics_deepseek:case-a:C2:T4` | `aggregate:case-a:c_a2` | — |
| p0-formal-s17 | `metrics_deepseek` | case-a | `c_a2` | 1 | nll/full | decrease | [-0.0400, -0.0050] | 0.0257 [0.0244, 0.0272] | 1.8478 | 方向/幅度 | `r1:metrics_deepseek:case-a:C2:T5` | `aggregate:case-a:c_a2` | — |
| p0-formal-s17 | `metrics_deepseek` | case-a | `c_a2` | 1 | free_end_response/end | increase | [0.5000, 1.5000] | -2.0679 [-2.2899, -1.8551] | 1.8478 | 方向/幅度 | `r1:metrics_deepseek:case-a:C2:T7` | `aggregate:case-a:c_a2` | — |
| p0-formal-s17 | `metrics_deepseek` | case-a | `c_a3` | 1 | background_bias/execution | decrease | [-0.2500, -0.1200] | -0.3946 [-0.4105, -0.3795] | 8.4853 | 幅度 | `r1:metrics_deepseek:case-a:C3:T1` | `aggregate:case-a:c_a3` | `r1:metrics_deepseek:case-a` |
| p0-formal-s17 | `metrics_deepseek` | case-a | `c_a3` | 1 | clamped_rmse/full | decrease | [-1.5000, -0.5000] | -2.3406 [-2.5169, -2.1527] | 8.4853 | 幅度 | `r1:metrics_deepseek:case-a:C3:T2` | `aggregate:case-a:c_a3` | `r1:metrics_deepseek:case-a` |
| p0-formal-s17 | `metrics_deepseek` | case-a | `c_a3` | 1 | free_rmse/full | decrease | [-3.0000, -1.0000] | -8.4853 [-8.7119, -8.2605] | 8.4853 | 幅度 | `r1:metrics_deepseek:case-a:C3:T3` | `aggregate:case-a:c_a3` | `r1:metrics_deepseek:case-a` |
| p0-formal-s17 | `metrics_deepseek` | case-a | `c_a3` | 1 | end_response/end | increase | [0.7500, 2.0000] | -4.1842 [-4.3227, -4.0459] | 8.4853 | 方向/幅度 | `r1:metrics_deepseek:case-a:C3:T4` | `aggregate:case-a:c_a3` | `r1:metrics_deepseek:case-a` |
| p0-formal-s17 | `metrics_deepseek` | case-a | `c_a3` | 1 | nll/full | decrease | [-0.0500, -0.0100] | 0.0272 [0.0262, 0.0282] | 8.4853 | 方向/幅度 | `r1:metrics_deepseek:case-a:C3:T5` | `aggregate:case-a:c_a3` | `r1:metrics_deepseek:case-a` |
| p0-formal-s17 | `metrics_deepseek` | case-a | `c_a3` | 1 | free_end_response/end | increase | [0.7500, 2.0000] | -13.1030 [-13.4862, -12.7010] | 8.4853 | 方向/幅度 | `r1:metrics_deepseek:case-a:C3:T7` | `aggregate:case-a:c_a3` | `r1:metrics_deepseek:case-a` |
| p0-formal-s17 | `metrics_deepseek` | case-b | `c_b1` | 1 | background_bias/execution | decrease | [-0.1500, -0.0300] | 0.0000 [0.0000, 0.0000] | 2.7243 | 方向/幅度 | `r1:metrics_deepseek:case-b:C1:T1` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `metrics_deepseek` | case-b | `c_b1` | 1 | clamped_rmse/full | decrease | [-0.8000, -0.3000] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:metrics_deepseek:case-b:C1:T2` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `metrics_deepseek` | case-b | `c_b1` | 1 | free_rmse/full | decrease | [-0.7000, -0.2000] | -3.0954 [-3.1928, -2.9821] | 2.7243 | 幅度 | `r1:metrics_deepseek:case-b:C1:T3` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `metrics_deepseek` | case-b | `c_b1` | 1 | end_response/end | increase | [0.2000, 0.8000] | -4.6327 [-4.7882, -4.4863] | 2.7243 | 方向/幅度 | `r1:metrics_deepseek:case-b:C1:T4` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `metrics_deepseek` | case-b | `c_b1` | 1 | nll/full | decrease | [-0.0500, -0.0100] | 0.0119 [0.0099, 0.0138] | 2.7243 | 方向/幅度 | `r1:metrics_deepseek:case-b:C1:T5` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `metrics_deepseek` | case-b | `c_b1` | 1 | background_bias/recovery | decrease | [-0.1500, -0.0300] | 0.0000 [0.0000, 0.0000] | 2.7243 | 方向/幅度 | `r1:metrics_deepseek:case-b:C1:T6` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `metrics_deepseek` | case-b | `c_b1` | 1 | free_end_response/end | increase | [0.2000, 0.8000] | -4.8479 [-4.9742, -4.7176] | 2.7243 | 方向/幅度 | `r1:metrics_deepseek:case-b:C1:T7` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `metrics_deepseek` | case-b | `c_b2` | 1 | background_bias/execution | decrease | [-0.1200, -0.0300] | 0.0000 [0.0000, 0.0000] | 2.7243 | 方向/幅度 | `r1:metrics_deepseek:case-b:C2:T1` | `aggregate:case-b:c_b2` | `r1:metrics_deepseek:case-b` |
| p0-formal-s17 | `metrics_deepseek` | case-b | `c_b2` | 1 | clamped_rmse/full | decrease | [-0.6000, -0.2000] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:metrics_deepseek:case-b:C2:T2` | `aggregate:case-b:c_b2` | `r1:metrics_deepseek:case-b` |
| p0-formal-s17 | `metrics_deepseek` | case-b | `c_b2` | 1 | free_rmse/full | decrease | [-0.5000, -0.1500] | -3.1125 [-3.2079, -2.9999] | 2.7243 | 幅度 | `r1:metrics_deepseek:case-b:C2:T3` | `aggregate:case-b:c_b2` | `r1:metrics_deepseek:case-b` |
| p0-formal-s17 | `metrics_deepseek` | case-b | `c_b2` | 1 | end_response/end | increase | [0.1500, 0.6000] | -4.6327 [-4.7882, -4.4863] | 2.7243 | 方向/幅度 | `r1:metrics_deepseek:case-b:C2:T4` | `aggregate:case-b:c_b2` | `r1:metrics_deepseek:case-b` |
| p0-formal-s17 | `metrics_deepseek` | case-b | `c_b2` | 1 | nll/full | decrease | [-0.0400, -0.0100] | 0.0111 [0.0094, 0.0128] | 2.7243 | 方向/幅度 | `r1:metrics_deepseek:case-b:C2:T5` | `aggregate:case-b:c_b2` | `r1:metrics_deepseek:case-b` |
| p0-formal-s17 | `metrics_deepseek` | case-b | `c_b2` | 1 | background_bias/recovery | decrease | [-0.1200, -0.0300] | 0.0000 [0.0000, 0.0000] | 2.7243 | 方向/幅度 | `r1:metrics_deepseek:case-b:C2:T6` | `aggregate:case-b:c_b2` | `r1:metrics_deepseek:case-b` |
| p0-formal-s17 | `metrics_deepseek` | case-b | `c_b2` | 1 | free_end_response/end | increase | [0.1500, 0.6000] | -4.7172 [-4.8391, -4.5971] | 2.7243 | 方向/幅度 | `r1:metrics_deepseek:case-b:C2:T7` | `aggregate:case-b:c_b2` | `r1:metrics_deepseek:case-b` |
| p0-formal-s17 | `metrics_gpt` | case-a | `c_a1` | 1 | free_rmse/full | decrease | [-5.5000, -1.0000] | -5.8784 [-6.0950, -5.6670] | 5.8784 | 幅度 | `r1:metrics_gpt:case-a:C1:T3` | `aggregate:case-a:c_a1` | — |
| p0-formal-s17 | `metrics_gpt` | case-a | `c_a1` | 1 | nll/full | increase | [0.0300, 0.2500] | 0.0000 [0.0000, 0.0000] | 5.8784 | 方向/幅度 | `r1:metrics_gpt:case-a:C1:T5` | `aggregate:case-a:c_a1` | — |
| p0-formal-s17 | `metrics_gpt` | case-a | `c_a2` | 1 | free_rmse/full | decrease | [-3.0000, -0.3000] | -1.8478 [-1.9701, -1.7274] | 1.8478 | 方向 | `r1:metrics_gpt:case-a:C2:T3` | `aggregate:case-a:c_a2` | — |
| p0-formal-s17 | `metrics_gpt` | case-a | `c_a2` | 1 | end_response/end | decrease | [-4.0000, -0.3000] | -4.1842 [-4.3227, -4.0459] | 1.8478 | 幅度 | `r1:metrics_gpt:case-a:C2:T4` | `aggregate:case-a:c_a2` | — |
| p0-formal-s17 | `metrics_gpt` | case-a | `c_a2` | 1 | nll/full | unchanged | [-0.0200, 0.0200] | 0.0257 [0.0244, 0.0272] | 1.8478 | 等效 | `r1:metrics_gpt:case-a:C2:T5` | `aggregate:case-a:c_a2` | — |
| p0-formal-s17 | `metrics_gpt` | case-a | `c_a3` | 1 | free_rmse/full | decrease | [-7.0000, -2.0000] | -8.4853 [-8.7119, -8.2605] | 8.4853 | 幅度 | `r1:metrics_gpt:case-a:C3:T3` | `aggregate:case-a:c_a3` | `r1:metrics_gpt:case-a` |
| p0-formal-s17 | `metrics_gpt` | case-a | `c_a3` | 1 | end_response/end | decrease | [-4.0000, -0.3000] | -4.1842 [-4.3227, -4.0459] | 8.4853 | 幅度 | `r1:metrics_gpt:case-a:C3:T4` | `aggregate:case-a:c_a3` | `r1:metrics_gpt:case-a` |
| p0-formal-s17 | `metrics_gpt` | case-a | `c_a3` | 1 | nll/full | increase | [0.0300, 0.2500] | 0.0272 [0.0262, 0.0282] | 8.4853 | 方向/幅度 | `r1:metrics_gpt:case-a:C3:T5` | `aggregate:case-a:c_a3` | `r1:metrics_gpt:case-a` |
| p0-formal-s17 | `metrics_gpt` | case-a | `c_a3` | 1 | free_end_response/end | decrease | [-12.0000, -3.0000] | -13.1030 [-13.4862, -12.7010] | 8.4853 | 幅度 | `r1:metrics_gpt:case-a:C3:T7` | `aggregate:case-a:c_a3` | `r1:metrics_gpt:case-a` |
| p0-formal-s17 | `metrics_gpt` | case-b | `c_b1` | 1 | clamped_rmse/full | decrease | [-0.9000, -0.1200] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:metrics_gpt:case-b:C1:T2` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `metrics_gpt` | case-b | `c_b2` | 1 | clamped_rmse/full | decrease | [-0.9000, -0.1200] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:metrics_gpt:case-b:C2:T2` | `aggregate:case-b:c_b2` | `r1:metrics_gpt:case-b` |
| p0-formal-s17 | `metrics_gpt` | case-b | `c_b2` | 1 | free_rmse/full | decrease | [-0.7500, -0.1500] | -3.1125 [-3.2079, -2.9999] | 2.7243 | 幅度 | `r1:metrics_gpt:case-b:C2:T3` | `aggregate:case-b:c_b2` | `r1:metrics_gpt:case-b` |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b1` | 2 | background_bias/execution | decrease | [-0.3000, -0.0500] | 0.0000 [0.0000, 0.0000] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C1:T1` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b1` | 2 | clamped_rmse/full | increase | [0.2000, 1.2000] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C1:T2` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b1` | 2 | free_rmse/full | increase | [0.3000, 1.5000] | -3.0954 [-3.1928, -2.9821] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C1:T3` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b1` | 2 | end_response/end | increase | [0.2000, 1.4000] | -4.6327 [-4.7882, -4.4863] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C1:T4` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b1` | 2 | nll/full | increase | [0.0200, 0.0900] | 0.0119 [0.0099, 0.0138] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C1:T5` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b1` | 2 | background_bias/recovery | decrease | [-0.3000, -0.0500] | 0.0000 [0.0000, 0.0000] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C1:T6` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b1` | 2 | free_end_response/end | increase | [0.3000, 1.6000] | -4.8479 [-4.9742, -4.7176] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C1:T7` | `aggregate:case-b:c_b1` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b2` | 2 | clamped_rmse/full | increase | [0.1000, 0.9000] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C2:T2` | `aggregate:case-b:c_b2` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b2` | 2 | free_rmse/full | increase | [0.1500, 1.1000] | -3.1125 [-3.2079, -2.9999] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C2:T3` | `aggregate:case-b:c_b2` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b2` | 2 | end_response/end | increase | [0.1000, 1.0000] | -4.6327 [-4.7882, -4.4863] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C2:T4` | `aggregate:case-b:c_b2` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b2` | 2 | nll/full | increase | [0.0100, 0.0600] | 0.0111 [0.0094, 0.0128] | 2.7243 | 方向 | `r2:evidence_deepseek:case-b:C2:T5` | `aggregate:case-b:c_b2` | — |
| p0-formal-s17 | `evidence_deepseek` | case-b | `c_b2` | 2 | free_end_response/end | increase | [0.1500, 1.2000] | -4.7172 [-4.8391, -4.5971] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C2:T7` | `aggregate:case-b:c_b2` | — |
| p0-formal-s29 | `evidence_deepseek` | case-b | `c_b2` | 1 | background_bias/execution | increase | [0.0200, 0.2000] | 0.0000 [0.0000, 0.0000] | 2.7243 | 方向/幅度 | `r1:evidence_deepseek:case-b:C2:T1` | `aggregate:case-b:c_b2` | — |
| p0-formal-s29 | `evidence_deepseek` | case-b | `c_b2` | 1 | clamped_rmse/full | decrease | [-1.5000, -0.2000] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:evidence_deepseek:case-b:C2:T2` | `aggregate:case-b:c_b2` | — |
| p0-formal-s29 | `evidence_deepseek` | case-b | `c_b2` | 1 | free_rmse/full | decrease | [-1.5000, -0.2000] | -3.0954 [-3.1928, -2.9821] | 2.7243 | 幅度 | `r1:evidence_deepseek:case-b:C2:T3` | `aggregate:case-b:c_b2` | — |
| p0-formal-s29 | `evidence_deepseek` | case-b | `c_b2` | 1 | end_response/end | decrease | [-2.0000, -0.3000] | -4.6327 [-4.7882, -4.4863] | 2.7243 | 幅度 | `r1:evidence_deepseek:case-b:C2:T4` | `aggregate:case-b:c_b2` | — |
| p0-formal-s29 | `evidence_deepseek` | case-b | `c_b2` | 1 | nll/full | decrease | [-0.0800, -0.0100] | 0.0119 [0.0099, 0.0138] | 2.7243 | 方向/幅度 | `r1:evidence_deepseek:case-b:C2:T5` | `aggregate:case-b:c_b2` | — |
| p0-formal-s29 | `evidence_deepseek` | case-b | `c_b2` | 1 | background_bias/recovery | increase | [0.0200, 0.2000] | 0.0000 [0.0000, 0.0000] | 2.7243 | 方向/幅度 | `r1:evidence_deepseek:case-b:C2:T6` | `aggregate:case-b:c_b2` | — |
| p0-formal-s29 | `evidence_deepseek` | case-b | `c_b2` | 1 | free_end_response/end | decrease | [-2.0000, -0.3000] | -4.8479 [-4.9742, -4.7176] | 2.7243 | 幅度 | `r1:evidence_deepseek:case-b:C2:T7` | `aggregate:case-b:c_b2` | — |
| p0-formal-s29 | `evidence_deepseek` | case-b | `c_b3` | 1 | clamped_rmse/full | decrease | [-1.2000, -0.3000] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:evidence_deepseek:case-b:C3:T2` | `aggregate:case-b:c_b3` | `r1:evidence_deepseek:case-b` |
| p0-formal-s29 | `evidence_deepseek` | case-b | `c_b3` | 1 | free_rmse/full | decrease | [-1.2000, -0.3000] | -3.1125 [-3.2079, -2.9999] | 2.7243 | 幅度 | `r1:evidence_deepseek:case-b:C3:T3` | `aggregate:case-b:c_b3` | `r1:evidence_deepseek:case-b` |
| p0-formal-s29 | `evidence_deepseek` | case-b | `c_b3` | 1 | end_response/end | decrease | [-1.5000, -0.3000] | -4.6327 [-4.7882, -4.4863] | 2.7243 | 幅度 | `r1:evidence_deepseek:case-b:C3:T4` | `aggregate:case-b:c_b3` | `r1:evidence_deepseek:case-b` |
| p0-formal-s29 | `evidence_deepseek` | case-b | `c_b3` | 1 | nll/full | decrease | [-0.0600, -0.0100] | 0.0111 [0.0094, 0.0128] | 2.7243 | 方向/幅度 | `r1:evidence_deepseek:case-b:C3:T5` | `aggregate:case-b:c_b3` | `r1:evidence_deepseek:case-b` |
| p0-formal-s29 | `evidence_deepseek` | case-b | `c_b3` | 1 | free_end_response/end | decrease | [-1.5000, -0.3000] | -4.7172 [-4.8391, -4.5971] | 2.7243 | 幅度 | `r1:evidence_deepseek:case-b:C3:T7` | `aggregate:case-b:c_b3` | `r1:evidence_deepseek:case-b` |
| p0-formal-s29 | `evidence_gpt` | case-a | `c_a1` | 1 | free_rmse/full | decrease | [-7.5000, -3.0000] | -8.4853 [-8.7119, -8.2605] | 8.4853 | 幅度 | `r1:evidence_gpt:case-a:C1:T3` | `aggregate:case-a:c_a1` | `r1:evidence_gpt:case-a` |
| p0-formal-s29 | `evidence_gpt` | case-a | `c_a1` | 1 | nll/full | unchanged | [-0.0200, 0.0200] | 0.0272 [0.0262, 0.0282] | 8.4853 | 等效 | `r1:evidence_gpt:case-a:C1:T5` | `aggregate:case-a:c_a1` | `r1:evidence_gpt:case-a` |
| p0-formal-s29 | `evidence_gpt` | case-a | `c_a1` | 1 | free_end_response/end | decrease | [-12.0000, -3.0000] | -13.1030 [-13.4862, -12.7010] | 8.4853 | 幅度 | `r1:evidence_gpt:case-a:C1:T7` | `aggregate:case-a:c_a1` | `r1:evidence_gpt:case-a` |
| p0-formal-s29 | `evidence_gpt` | case-a | `c_a3` | 1 | free_rmse/full | decrease | [-3.0000, -0.3000] | -1.8478 [-1.9701, -1.7274] | 1.8478 | 方向 | `r1:evidence_gpt:case-a:C3:T3` | `aggregate:case-a:c_a3` | — |
| p0-formal-s29 | `evidence_gpt` | case-a | `c_a3` | 1 | nll/full | unchanged | [-0.0200, 0.0200] | 0.0257 [0.0244, 0.0272] | 1.8478 | 等效 | `r1:evidence_gpt:case-a:C3:T5` | `aggregate:case-a:c_a3` | — |
| p0-formal-s29 | `evidence_gpt` | case-b | `c_b2` | 1 | clamped_rmse/full | decrease | [-1.3000, -0.1500] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:evidence_gpt:case-b:C2:T2` | `aggregate:case-b:c_b2` | — |
| p0-formal-s29 | `evidence_gpt` | case-b | `c_b2` | 1 | free_rmse/full | decrease | [-1.3000, -0.1000] | -3.0954 [-3.1928, -2.9821] | 2.7243 | 幅度 | `r1:evidence_gpt:case-b:C2:T3` | `aggregate:case-b:c_b2` | — |
| p0-formal-s29 | `evidence_gpt` | case-b | `c_b2` | 1 | nll/full | decrease | [-0.0900, -0.0050] | 0.0119 [0.0099, 0.0138] | 2.7243 | 方向/幅度 | `r1:evidence_gpt:case-b:C2:T5` | `aggregate:case-b:c_b2` | — |
| p0-formal-s29 | `evidence_gpt` | case-b | `c_b3` | 1 | clamped_rmse/full | decrease | [-1.3000, -0.1500] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:evidence_gpt:case-b:C3:T2` | `aggregate:case-b:c_b3` | `r1:evidence_gpt:case-b` |
| p0-formal-s29 | `evidence_gpt` | case-b | `c_b3` | 1 | free_rmse/full | decrease | [-1.3000, -0.1500] | -3.1125 [-3.2079, -2.9999] | 2.7243 | 幅度 | `r1:evidence_gpt:case-b:C3:T3` | `aggregate:case-b:c_b3` | `r1:evidence_gpt:case-b` |
| p0-formal-s29 | `evidence_gpt` | case-b | `c_b3` | 1 | nll/full | decrease | [-0.0800, -0.0050] | 0.0111 [0.0094, 0.0128] | 2.7243 | 方向/幅度 | `r1:evidence_gpt:case-b:C3:T5` | `aggregate:case-b:c_b3` | `r1:evidence_gpt:case-b` |
| p0-formal-s29 | `metrics_deepseek` | case-a | `c_a1` | 1 | background_bias/execution | decrease | [-0.1800, -0.0500] | -0.3946 [-0.4105, -0.3795] | 8.4853 | 幅度 | `r1:metrics_deepseek:case-a:C1:T1` | `aggregate:case-a:c_a1` | `r1:metrics_deepseek:case-a` |
| p0-formal-s29 | `metrics_deepseek` | case-a | `c_a1` | 1 | clamped_rmse/full | decrease | [-0.6000, -0.0500] | -2.3406 [-2.5169, -2.1527] | 8.4853 | 幅度 | `r1:metrics_deepseek:case-a:C1:T2` | `aggregate:case-a:c_a1` | `r1:metrics_deepseek:case-a` |
| p0-formal-s29 | `metrics_deepseek` | case-a | `c_a1` | 1 | free_rmse/full | decrease | [-2.5000, -0.2000] | -8.4853 [-8.7119, -8.2605] | 8.4853 | 幅度 | `r1:metrics_deepseek:case-a:C1:T3` | `aggregate:case-a:c_a1` | `r1:metrics_deepseek:case-a` |
| p0-formal-s29 | `metrics_deepseek` | case-a | `c_a1` | 1 | end_response/end | increase | [0.0500, 1.2000] | -4.1842 [-4.3227, -4.0459] | 8.4853 | 方向/幅度 | `r1:metrics_deepseek:case-a:C1:T4` | `aggregate:case-a:c_a1` | `r1:metrics_deepseek:case-a` |
| p0-formal-s29 | `metrics_deepseek` | case-a | `c_a1` | 1 | nll/full | decrease | [-0.0500, -0.0050] | 0.0272 [0.0262, 0.0282] | 8.4853 | 方向/幅度 | `r1:metrics_deepseek:case-a:C1:T5` | `aggregate:case-a:c_a1` | `r1:metrics_deepseek:case-a` |
| p0-formal-s29 | `metrics_deepseek` | case-a | `c_a1` | 1 | background_bias/recovery | decrease | [-0.1800, -0.0500] | -0.2398 [-0.2563, -0.2239] | 8.4853 | 幅度 | `r1:metrics_deepseek:case-a:C1:T6` | `aggregate:case-a:c_a1` | `r1:metrics_deepseek:case-a` |
| p0-formal-s29 | `metrics_deepseek` | case-a | `c_a1` | 1 | free_end_response/end | increase | [0.0500, 1.2000] | -13.1030 [-13.4862, -12.7010] | 8.4853 | 方向/幅度 | `r1:metrics_deepseek:case-a:C1:T7` | `aggregate:case-a:c_a1` | `r1:metrics_deepseek:case-a` |
| p0-formal-s29 | `metrics_deepseek` | case-a | `c_a2` | 1 | background_bias/execution | decrease | [-0.1500, -0.0300] | -0.3946 [-0.4105, -0.3795] | 5.8784 | 幅度 | `r1:metrics_deepseek:case-a:C2:T1` | `aggregate:case-a:c_a2` | — |
| p0-formal-s29 | `metrics_deepseek` | case-a | `c_a2` | 1 | free_rmse/full | unchanged | [-0.2500, 0.2500] | -5.8784 [-6.0950, -5.6670] | 5.8784 | 等效 | `r1:metrics_deepseek:case-a:C2:T3` | `aggregate:case-a:c_a2` | — |
| p0-formal-s29 | `metrics_deepseek` | case-a | `c_a2` | 1 | background_bias/recovery | decrease | [-0.1500, -0.0300] | -0.2398 [-0.2563, -0.2239] | 5.8784 | 幅度 | `r1:metrics_deepseek:case-a:C2:T6` | `aggregate:case-a:c_a2` | — |
| p0-formal-s29 | `metrics_deepseek` | case-a | `c_a2` | 1 | free_end_response/end | unchanged | [-0.2500, 0.2500] | -8.8117 [-9.2243, -8.4075] | 5.8784 | 等效 | `r1:metrics_deepseek:case-a:C2:T7` | `aggregate:case-a:c_a2` | — |
| p0-formal-s29 | `metrics_deepseek` | case-a | `c_a3` | 1 | clamped_rmse/full | decrease | [-0.5000, -0.0200] | -2.3406 [-2.5169, -2.1527] | 1.8478 | 幅度 | `r1:metrics_deepseek:case-a:C3:T2` | `aggregate:case-a:c_a3` | — |
| p0-formal-s29 | `metrics_deepseek` | case-a | `c_a3` | 1 | free_rmse/full | decrease | [-2.2000, -0.1500] | -1.8478 [-1.9701, -1.7274] | 1.8478 | 方向 | `r1:metrics_deepseek:case-a:C3:T3` | `aggregate:case-a:c_a3` | — |
| p0-formal-s29 | `metrics_deepseek` | case-a | `c_a3` | 1 | end_response/end | increase | [0.0500, 1.0000] | -4.1842 [-4.3227, -4.0459] | 1.8478 | 方向/幅度 | `r1:metrics_deepseek:case-a:C3:T4` | `aggregate:case-a:c_a3` | — |
| p0-formal-s29 | `metrics_deepseek` | case-a | `c_a3` | 1 | nll/full | decrease | [-0.0400, -0.0040] | 0.0257 [0.0244, 0.0272] | 1.8478 | 方向/幅度 | `r1:metrics_deepseek:case-a:C3:T5` | `aggregate:case-a:c_a3` | — |
| p0-formal-s29 | `metrics_deepseek` | case-a | `c_a3` | 1 | free_end_response/end | increase | [0.0500, 1.0000] | -2.0679 [-2.2899, -1.8551] | 1.8478 | 方向/幅度 | `r1:metrics_deepseek:case-a:C3:T7` | `aggregate:case-a:c_a3` | — |
| p0-formal-s29 | `metrics_deepseek` | case-b | `c_b2` | 1 | clamped_rmse/full | decrease | [-2.0000, -0.5000] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:metrics_deepseek:case-b:C2:T2` | `aggregate:case-b:c_b2` | — |
| p0-formal-s29 | `metrics_deepseek` | case-b | `c_b2` | 1 | free_rmse/full | decrease | [-1.5000, -0.3000] | -3.0954 [-3.1928, -2.9821] | 2.7243 | 幅度 | `r1:metrics_deepseek:case-b:C2:T3` | `aggregate:case-b:c_b2` | — |
| p0-formal-s29 | `metrics_deepseek` | case-b | `c_b2` | 1 | end_response/end | decrease | [-2.0000, -0.5000] | -4.6327 [-4.7882, -4.4863] | 2.7243 | 幅度 | `r1:metrics_deepseek:case-b:C2:T4` | `aggregate:case-b:c_b2` | — |
| p0-formal-s29 | `metrics_deepseek` | case-b | `c_b2` | 1 | nll/full | decrease | [-0.0800, -0.0200] | 0.0119 [0.0099, 0.0138] | 2.7243 | 方向/幅度 | `r1:metrics_deepseek:case-b:C2:T5` | `aggregate:case-b:c_b2` | — |
| p0-formal-s29 | `metrics_deepseek` | case-b | `c_b2` | 1 | free_end_response/end | decrease | [-2.0000, -0.5000] | -4.8479 [-4.9742, -4.7176] | 2.7243 | 幅度 | `r1:metrics_deepseek:case-b:C2:T7` | `aggregate:case-b:c_b2` | — |
| p0-formal-s29 | `metrics_deepseek` | case-b | `c_b3` | 1 | clamped_rmse/full | decrease | [-1.5000, -0.3000] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:metrics_deepseek:case-b:C3:T2` | `aggregate:case-b:c_b3` | `r1:metrics_deepseek:case-b` |
| p0-formal-s29 | `metrics_deepseek` | case-b | `c_b3` | 1 | free_rmse/full | decrease | [-1.0000, -0.2000] | -3.1125 [-3.2079, -2.9999] | 2.7243 | 幅度 | `r1:metrics_deepseek:case-b:C3:T3` | `aggregate:case-b:c_b3` | `r1:metrics_deepseek:case-b` |
| p0-formal-s29 | `metrics_deepseek` | case-b | `c_b3` | 1 | end_response/end | decrease | [-1.5000, -0.3000] | -4.6327 [-4.7882, -4.4863] | 2.7243 | 幅度 | `r1:metrics_deepseek:case-b:C3:T4` | `aggregate:case-b:c_b3` | `r1:metrics_deepseek:case-b` |
| p0-formal-s29 | `metrics_deepseek` | case-b | `c_b3` | 1 | nll/full | decrease | [-0.0500, -0.0100] | 0.0111 [0.0094, 0.0128] | 2.7243 | 方向/幅度 | `r1:metrics_deepseek:case-b:C3:T5` | `aggregate:case-b:c_b3` | `r1:metrics_deepseek:case-b` |
| p0-formal-s29 | `metrics_deepseek` | case-b | `c_b3` | 1 | free_end_response/end | decrease | [-1.5000, -0.3000] | -4.7172 [-4.8391, -4.5971] | 2.7243 | 幅度 | `r1:metrics_deepseek:case-b:C3:T7` | `aggregate:case-b:c_b3` | `r1:metrics_deepseek:case-b` |
| p0-formal-s29 | `metrics_gpt` | case-a | `c_a1` | 1 | clamped_rmse/full | decrease | [-1.8000, -0.2000] | -2.3406 [-2.5169, -2.1527] | 8.4853 | 幅度 | `r1:metrics_gpt:case-a:C1:T2` | `aggregate:case-a:c_a1` | `r1:metrics_gpt:case-a` |
| p0-formal-s29 | `metrics_gpt` | case-a | `c_a1` | 1 | free_rmse/full | decrease | [-5.5000, -1.2000] | -8.4853 [-8.7119, -8.2605] | 8.4853 | 幅度 | `r1:metrics_gpt:case-a:C1:T3` | `aggregate:case-a:c_a1` | `r1:metrics_gpt:case-a` |
| p0-formal-s29 | `metrics_gpt` | case-a | `c_a1` | 1 | free_end_response/end | decrease | [-7.0000, -1.0000] | -13.1030 [-13.4862, -12.7010] | 8.4853 | 幅度 | `r1:metrics_gpt:case-a:C1:T7` | `aggregate:case-a:c_a1` | `r1:metrics_gpt:case-a` |
| p0-formal-s29 | `metrics_gpt` | case-a | `c_a2` | 1 | free_rmse/full | decrease | [-4.8000, -0.7000] | -5.8784 [-6.0950, -5.6670] | 5.8784 | 幅度 | `r1:metrics_gpt:case-a:C2:T3` | `aggregate:case-a:c_a2` | — |
| p0-formal-s29 | `metrics_gpt` | case-a | `c_a3` | 1 | clamped_rmse/full | decrease | [-1.8000, -0.2000] | -2.3406 [-2.5169, -2.1527] | 1.8478 | 幅度 | `r1:metrics_gpt:case-a:C3:T2` | `aggregate:case-a:c_a3` | — |
| p0-formal-s29 | `metrics_gpt` | case-a | `c_a3` | 1 | nll/full | unchanged | [-0.0200, 0.0200] | 0.0257 [0.0244, 0.0272] | 1.8478 | 等效 | `r1:metrics_gpt:case-a:C3:T5` | `aggregate:case-a:c_a3` | — |
| p0-formal-s29 | `metrics_gpt` | case-b | `c_b2` | 1 | clamped_rmse/full | decrease | [-0.8000, -0.1200] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:metrics_gpt:case-b:C2:T2` | `aggregate:case-b:c_b2` | — |
| p0-formal-s29 | `metrics_gpt` | case-b | `c_b3` | 1 | clamped_rmse/full | decrease | [-0.8000, -0.1200] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:metrics_gpt:case-b:C3:T2` | `aggregate:case-b:c_b3` | `r1:metrics_gpt:case-b` |
| p0-formal-s29 | `metrics_gpt` | case-b | `c_b3` | 1 | free_rmse/full | decrease | [-0.8500, -0.2600] | -3.1125 [-3.2079, -2.9999] | 2.7243 | 幅度 | `r1:metrics_gpt:case-b:C3:T3` | `aggregate:case-b:c_b3` | `r1:metrics_gpt:case-b` |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a1` | 2 | background_bias/execution | decrease | [-0.1500, -0.0500] | -0.3946 [-0.4105, -0.3795] | 8.4853 | 幅度 | `r2:evidence_deepseek:case-a:C1:T1` | `aggregate:case-a:c_a1` | — |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a1` | 2 | clamped_rmse/full | increase | [0.1000, 0.6000] | -2.3406 [-2.5169, -2.1527] | 8.4853 | 方向/幅度 | `r2:evidence_deepseek:case-a:C1:T2` | `aggregate:case-a:c_a1` | — |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a1` | 2 | free_rmse/full | increase | [0.5000, 2.5000] | -8.4853 [-8.7119, -8.2605] | 8.4853 | 方向/幅度 | `r2:evidence_deepseek:case-a:C1:T3` | `aggregate:case-a:c_a1` | — |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a1` | 2 | end_response/end | decrease | [-1.5000, -0.4000] | -4.1842 [-4.3227, -4.0459] | 8.4853 | 幅度 | `r2:evidence_deepseek:case-a:C1:T4` | `aggregate:case-a:c_a1` | — |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a1` | 2 | nll/full | increase | [0.0100, 0.0600] | 0.0272 [0.0262, 0.0282] | 8.4853 | 方向 | `r2:evidence_deepseek:case-a:C1:T5` | `aggregate:case-a:c_a1` | — |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a1` | 2 | background_bias/recovery | decrease | [-0.1500, -0.0500] | -0.2398 [-0.2563, -0.2239] | 8.4853 | 幅度 | `r2:evidence_deepseek:case-a:C1:T6` | `aggregate:case-a:c_a1` | — |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a1` | 2 | free_end_response/end | decrease | [-1.5000, -0.4000] | -13.1030 [-13.4862, -12.7010] | 8.4853 | 幅度 | `r2:evidence_deepseek:case-a:C1:T7` | `aggregate:case-a:c_a1` | — |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a2` | 2 | background_bias/execution | decrease | [-0.1500, -0.0500] | -0.3946 [-0.4105, -0.3795] | 5.8784 | 幅度 | `r2:evidence_deepseek:case-a:C2:T1` | `aggregate:case-a:c_a2` | `r2:evidence_deepseek:case-a` |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a2` | 2 | clamped_rmse/full | decrease | [-0.4000, -0.1000] | 0.0000 [0.0000, 0.0000] | 5.8784 | 方向/幅度 | `r2:evidence_deepseek:case-a:C2:T2` | `aggregate:case-a:c_a2` | `r2:evidence_deepseek:case-a` |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a2` | 2 | free_rmse/full | decrease | [-2.0000, -0.3000] | -5.8784 [-6.0950, -5.6670] | 5.8784 | 幅度 | `r2:evidence_deepseek:case-a:C2:T3` | `aggregate:case-a:c_a2` | `r2:evidence_deepseek:case-a` |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a2` | 2 | end_response/end | decrease | [-1.2000, -0.3000] | 0.0000 [0.0000, 0.0000] | 5.8784 | 方向/幅度 | `r2:evidence_deepseek:case-a:C2:T4` | `aggregate:case-a:c_a2` | `r2:evidence_deepseek:case-a` |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a2` | 2 | nll/full | decrease | [-0.0500, -0.0100] | 0.0000 [0.0000, 0.0000] | 5.8784 | 方向/幅度 | `r2:evidence_deepseek:case-a:C2:T5` | `aggregate:case-a:c_a2` | `r2:evidence_deepseek:case-a` |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a2` | 2 | background_bias/recovery | decrease | [-0.1500, -0.0500] | -0.2398 [-0.2563, -0.2239] | 5.8784 | 幅度 | `r2:evidence_deepseek:case-a:C2:T6` | `aggregate:case-a:c_a2` | `r2:evidence_deepseek:case-a` |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a2` | 2 | free_end_response/end | decrease | [-1.2000, -0.3000] | -8.8117 [-9.2243, -8.4075] | 5.8784 | 幅度 | `r2:evidence_deepseek:case-a:C2:T7` | `aggregate:case-a:c_a2` | `r2:evidence_deepseek:case-a` |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a3` | 2 | background_bias/execution | increase | [0.0300, 0.1200] | -0.0265 [-0.0311, -0.0219] | 1.8478 | 方向/幅度 | `r2:evidence_deepseek:case-a:C3:T1` | `aggregate:case-a:c_a3` | — |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a3` | 2 | clamped_rmse/full | increase | [0.2000, 0.9000] | -2.3406 [-2.5169, -2.1527] | 1.8478 | 方向/幅度 | `r2:evidence_deepseek:case-a:C3:T2` | `aggregate:case-a:c_a3` | — |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a3` | 2 | free_rmse/full | increase | [0.8000, 3.0000] | -1.8478 [-1.9701, -1.7274] | 1.8478 | 方向/幅度 | `r2:evidence_deepseek:case-a:C3:T3` | `aggregate:case-a:c_a3` | — |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a3` | 2 | end_response/end | increase | [0.2000, 1.2000] | -4.1842 [-4.3227, -4.0459] | 1.8478 | 方向/幅度 | `r2:evidence_deepseek:case-a:C3:T4` | `aggregate:case-a:c_a3` | — |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a3` | 2 | nll/full | increase | [0.0100, 0.0600] | 0.0257 [0.0244, 0.0272] | 1.8478 | 方向 | `r2:evidence_deepseek:case-a:C3:T5` | `aggregate:case-a:c_a3` | — |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a3` | 2 | background_bias/recovery | increase | [0.0300, 0.1200] | -0.0132 [-0.0193, -0.0071] | 1.8478 | 方向/幅度 | `r2:evidence_deepseek:case-a:C3:T6` | `aggregate:case-a:c_a3` | — |
| p0-formal-s29 | `evidence_deepseek` | case-a | `c_a3` | 2 | free_end_response/end | increase | [0.2000, 1.2000] | -2.0679 [-2.2899, -1.8551] | 1.8478 | 方向/幅度 | `r2:evidence_deepseek:case-a:C3:T7` | `aggregate:case-a:c_a3` | — |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a1` | 1 | background_bias/execution | decrease | [-0.1800, -0.0600] | -0.0265 [-0.0311, -0.0219] | 1.8478 | 方向/幅度 | `r1:evidence_deepseek:case-a:C1:T1` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a1` | 1 | clamped_rmse/full | decrease | [-0.7000, -0.2500] | -2.3406 [-2.5169, -2.1527] | 1.8478 | 幅度 | `r1:evidence_deepseek:case-a:C1:T2` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a1` | 1 | free_rmse/full | decrease | [-1.8000, -0.6000] | -1.8478 [-1.9701, -1.7274] | 1.8478 | 方向 | `r1:evidence_deepseek:case-a:C1:T3` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a1` | 1 | end_response/end | increase | [0.2000, 0.8000] | -4.1842 [-4.3227, -4.0459] | 1.8478 | 方向/幅度 | `r1:evidence_deepseek:case-a:C1:T4` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a1` | 1 | nll/full | decrease | [-0.0250, -0.0050] | 0.0257 [0.0244, 0.0272] | 1.8478 | 方向/幅度 | `r1:evidence_deepseek:case-a:C1:T5` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a1` | 1 | background_bias/recovery | decrease | [-0.1500, -0.0300] | -0.0132 [-0.0193, -0.0071] | 1.8478 | 方向/幅度 | `r1:evidence_deepseek:case-a:C1:T6` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a1` | 1 | free_end_response/end | increase | [0.2000, 1.0000] | -2.0679 [-2.2899, -1.8551] | 1.8478 | 方向/幅度 | `r1:evidence_deepseek:case-a:C1:T7` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a2` | 1 | background_bias/execution | decrease | [-0.2000, -0.0800] | -0.3946 [-0.4105, -0.3795] | 8.4853 | 幅度 | `r1:evidence_deepseek:case-a:C2:T1` | `aggregate:case-a:c_a2` | `r1:evidence_deepseek:case-a` |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a2` | 1 | clamped_rmse/full | decrease | [-0.9000, -0.3500] | -2.3406 [-2.5169, -2.1527] | 8.4853 | 幅度 | `r1:evidence_deepseek:case-a:C2:T2` | `aggregate:case-a:c_a2` | `r1:evidence_deepseek:case-a` |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a2` | 1 | free_rmse/full | decrease | [-2.4000, -0.9000] | -8.4853 [-8.7119, -8.2605] | 8.4853 | 幅度 | `r1:evidence_deepseek:case-a:C2:T3` | `aggregate:case-a:c_a2` | `r1:evidence_deepseek:case-a` |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a2` | 1 | end_response/end | increase | [0.3000, 1.0000] | -4.1842 [-4.3227, -4.0459] | 8.4853 | 方向/幅度 | `r1:evidence_deepseek:case-a:C2:T4` | `aggregate:case-a:c_a2` | `r1:evidence_deepseek:case-a` |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a2` | 1 | nll/full | decrease | [-0.0350, -0.0080] | 0.0272 [0.0262, 0.0282] | 8.4853 | 方向/幅度 | `r1:evidence_deepseek:case-a:C2:T5` | `aggregate:case-a:c_a2` | `r1:evidence_deepseek:case-a` |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a2` | 1 | background_bias/recovery | decrease | [-0.1800, -0.0400] | -0.2398 [-0.2563, -0.2239] | 8.4853 | 幅度 | `r1:evidence_deepseek:case-a:C2:T6` | `aggregate:case-a:c_a2` | `r1:evidence_deepseek:case-a` |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a2` | 1 | free_end_response/end | increase | [0.3000, 1.2000] | -13.1030 [-13.4862, -12.7010] | 8.4853 | 方向/幅度 | `r1:evidence_deepseek:case-a:C2:T7` | `aggregate:case-a:c_a2` | `r1:evidence_deepseek:case-a` |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a3` | 1 | background_bias/execution | decrease | [-0.2200, -0.0900] | -0.3946 [-0.4105, -0.3795] | 5.8784 | 幅度 | `r1:evidence_deepseek:case-a:C3:T1` | `aggregate:case-a:c_a3` | — |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a3` | 1 | clamped_rmse/full | decrease | [-0.6000, -0.1500] | 0.0000 [0.0000, 0.0000] | 5.8784 | 方向/幅度 | `r1:evidence_deepseek:case-a:C3:T2` | `aggregate:case-a:c_a3` | — |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a3` | 1 | free_rmse/full | decrease | [-1.5000, -0.4000] | -5.8784 [-6.0950, -5.6670] | 5.8784 | 幅度 | `r1:evidence_deepseek:case-a:C3:T3` | `aggregate:case-a:c_a3` | — |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a3` | 1 | nll/full | decrease | [-0.0200, -0.0030] | 0.0000 [0.0000, 0.0000] | 5.8784 | 方向/幅度 | `r1:evidence_deepseek:case-a:C3:T5` | `aggregate:case-a:c_a3` | — |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a3` | 1 | background_bias/recovery | decrease | [-0.2000, -0.0500] | -0.2398 [-0.2563, -0.2239] | 5.8784 | 幅度 | `r1:evidence_deepseek:case-a:C3:T6` | `aggregate:case-a:c_a3` | — |
| p0-formal-s41 | `evidence_deepseek` | case-a | `c_a3` | 1 | free_end_response/end | unchanged | [-0.2500, 0.2500] | -8.8117 [-9.2243, -8.4075] | 5.8784 | 等效 | `r1:evidence_deepseek:case-a:C3:T7` | `aggregate:case-a:c_a3` | — |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b1` | 1 | clamped_rmse/full | increase | [0.0500, 0.9000] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 方向/幅度 | `r1:evidence_deepseek:case-b:C1:T2` | `aggregate:case-b:c_b1` | `r2:evidence_deepseek:case-b` |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b1` | 1 | free_rmse/full | increase | [0.0500, 0.9000] | -3.1125 [-3.2079, -2.9999] | 2.7243 | 方向/幅度 | `r1:evidence_deepseek:case-b:C1:T3` | `aggregate:case-b:c_b1` | `r2:evidence_deepseek:case-b` |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b1` | 1 | end_response/end | unchanged | [-0.2500, 0.2500] | -4.6327 [-4.7882, -4.4863] | 2.7243 | 等效 | `r1:evidence_deepseek:case-b:C1:T4` | `aggregate:case-b:c_b1` | `r2:evidence_deepseek:case-b` |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b1` | 1 | nll/full | increase | [0.0040, 0.0600] | 0.0111 [0.0094, 0.0128] | 2.7243 | 方向 | `r1:evidence_deepseek:case-b:C1:T5` | `aggregate:case-b:c_b1` | `r2:evidence_deepseek:case-b` |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b1` | 1 | free_end_response/end | unchanged | [-0.2500, 0.2500] | -4.7172 [-4.8391, -4.5971] | 2.7243 | 等效 | `r1:evidence_deepseek:case-b:C1:T7` | `aggregate:case-b:c_b1` | `r2:evidence_deepseek:case-b` |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b3` | 1 | background_bias/execution | increase | [0.0500, 0.5000] | 0.0000 [0.0000, 0.0000] | 2.7243 | 方向/幅度 | `r1:evidence_deepseek:case-b:C3:T1` | `aggregate:case-b:c_b3` | — |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b3` | 1 | clamped_rmse/full | increase | [0.1000, 1.3000] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 方向/幅度 | `r1:evidence_deepseek:case-b:C3:T2` | `aggregate:case-b:c_b3` | — |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b3` | 1 | free_rmse/full | increase | [0.1000, 1.3000] | -3.0954 [-3.1928, -2.9821] | 2.7243 | 方向/幅度 | `r1:evidence_deepseek:case-b:C3:T3` | `aggregate:case-b:c_b3` | — |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b3` | 1 | end_response/end | increase | [0.1000, 1.1000] | -4.6327 [-4.7882, -4.4863] | 2.7243 | 方向/幅度 | `r1:evidence_deepseek:case-b:C3:T4` | `aggregate:case-b:c_b3` | — |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b3` | 1 | nll/full | increase | [0.0050, 0.0800] | 0.0119 [0.0099, 0.0138] | 2.7243 | 方向 | `r1:evidence_deepseek:case-b:C3:T5` | `aggregate:case-b:c_b3` | — |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b3` | 1 | background_bias/recovery | increase | [0.0300, 0.4000] | 0.0000 [0.0000, 0.0000] | 2.7243 | 方向/幅度 | `r1:evidence_deepseek:case-b:C3:T6` | `aggregate:case-b:c_b3` | — |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b3` | 1 | free_end_response/end | increase | [0.1000, 1.1000] | -4.8479 [-4.9742, -4.7176] | 2.7243 | 方向/幅度 | `r1:evidence_deepseek:case-b:C3:T7` | `aggregate:case-b:c_b3` | — |
| p0-formal-s41 | `evidence_gpt` | case-a | `c_a1` | 1 | free_rmse/full | decrease | [-3.2000, -0.4000] | -1.8478 [-1.9701, -1.7274] | 1.8478 | 方向 | `r1:evidence_gpt:case-a:C1:T3` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `evidence_gpt` | case-a | `c_a1` | 1 | nll/full | unchanged | [-0.0200, 0.0200] | 0.0257 [0.0244, 0.0272] | 1.8478 | 等效 | `r1:evidence_gpt:case-a:C1:T5` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `evidence_gpt` | case-a | `c_a2` | 1 | free_rmse/full | decrease | [-7.2000, -2.1000] | -8.4853 [-8.7119, -8.2605] | 8.4853 | 幅度 | `r1:evidence_gpt:case-a:C2:T3` | `aggregate:case-a:c_a2` | `r1:evidence_gpt:case-a` |
| p0-formal-s41 | `evidence_gpt` | case-a | `c_a2` | 1 | free_end_response/end | decrease | [-10.5000, -0.8000] | -13.1030 [-13.4862, -12.7010] | 8.4853 | 幅度 | `r1:evidence_gpt:case-a:C2:T7` | `aggregate:case-a:c_a2` | `r1:evidence_gpt:case-a` |
| p0-formal-s41 | `evidence_gpt` | case-b | `c_b1` | 1 | clamped_rmse/full | decrease | [-1.5000, -0.2000] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:evidence_gpt:case-b:C1:T2` | `aggregate:case-b:c_b1` | `r1:evidence_gpt:case-b` |
| p0-formal-s41 | `evidence_gpt` | case-b | `c_b1` | 1 | free_rmse/full | decrease | [-1.5000, -0.2500] | -3.1125 [-3.2079, -2.9999] | 2.7243 | 幅度 | `r1:evidence_gpt:case-b:C1:T3` | `aggregate:case-b:c_b1` | `r1:evidence_gpt:case-b` |
| p0-formal-s41 | `evidence_gpt` | case-b | `c_b3` | 1 | clamped_rmse/full | decrease | [-1.5000, -0.2000] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:evidence_gpt:case-b:C3:T2` | `aggregate:case-b:c_b3` | — |
| p0-formal-s41 | `evidence_gpt` | case-b | `c_b3` | 1 | free_rmse/full | decrease | [-1.5000, -0.1000] | -3.0954 [-3.1928, -2.9821] | 2.7243 | 幅度 | `r1:evidence_gpt:case-b:C3:T3` | `aggregate:case-b:c_b3` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a1` | 1 | background_bias/execution | decrease | [-0.3000, -0.0500] | -0.0265 [-0.0311, -0.0219] | 1.8478 | 方向/幅度 | `r1:metrics_deepseek:case-a:C1:T1` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a1` | 1 | clamped_rmse/full | unchanged | [-0.1000, 0.1000] | -2.3406 [-2.5169, -2.1527] | 1.8478 | 等效 | `r1:metrics_deepseek:case-a:C1:T2` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a1` | 1 | free_rmse/full | unchanged | [-0.2500, 0.2500] | -1.8478 [-1.9701, -1.7274] | 1.8478 | 等效 | `r1:metrics_deepseek:case-a:C1:T3` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a1` | 1 | end_response/end | increase | [0.2000, 1.5000] | -4.1842 [-4.3227, -4.0459] | 1.8478 | 方向/幅度 | `r1:metrics_deepseek:case-a:C1:T4` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a1` | 1 | nll/full | unchanged | [-0.0200, 0.0200] | 0.0257 [0.0244, 0.0272] | 1.8478 | 等效 | `r1:metrics_deepseek:case-a:C1:T5` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a1` | 1 | background_bias/recovery | decrease | [-0.3000, -0.0500] | -0.0132 [-0.0193, -0.0071] | 1.8478 | 方向/幅度 | `r1:metrics_deepseek:case-a:C1:T6` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a1` | 1 | free_end_response/end | increase | [0.2000, 1.5000] | -2.0679 [-2.2899, -1.8551] | 1.8478 | 方向/幅度 | `r1:metrics_deepseek:case-a:C1:T7` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a2` | 1 | clamped_rmse/full | unchanged | [-0.1000, 0.1000] | -2.3406 [-2.5169, -2.1527] | 8.4853 | 等效 | `r1:metrics_deepseek:case-a:C2:T2` | `aggregate:case-a:c_a2` | `r2:metrics_deepseek:case-a` |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a2` | 1 | free_rmse/full | unchanged | [-0.2500, 0.2500] | -8.4853 [-8.7119, -8.2605] | 8.4853 | 等效 | `r1:metrics_deepseek:case-a:C2:T3` | `aggregate:case-a:c_a2` | `r2:metrics_deepseek:case-a` |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a2` | 1 | end_response/end | increase | [0.3000, 2.0000] | -4.1842 [-4.3227, -4.0459] | 8.4853 | 方向/幅度 | `r1:metrics_deepseek:case-a:C2:T4` | `aggregate:case-a:c_a2` | `r2:metrics_deepseek:case-a` |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a2` | 1 | nll/full | unchanged | [-0.0200, 0.0200] | 0.0272 [0.0262, 0.0282] | 8.4853 | 等效 | `r1:metrics_deepseek:case-a:C2:T5` | `aggregate:case-a:c_a2` | `r2:metrics_deepseek:case-a` |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a2` | 1 | free_end_response/end | increase | [0.3000, 2.0000] | -13.1030 [-13.4862, -12.7010] | 8.4853 | 方向/幅度 | `r1:metrics_deepseek:case-a:C2:T7` | `aggregate:case-a:c_a2` | `r2:metrics_deepseek:case-a` |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a3` | 1 | free_rmse/full | unchanged | [-0.2500, 0.2500] | -5.8784 [-6.0950, -5.6670] | 5.8784 | 等效 | `r1:metrics_deepseek:case-a:C3:T3` | `aggregate:case-a:c_a3` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a3` | 1 | free_end_response/end | unchanged | [-0.2500, 0.2500] | -8.8117 [-9.2243, -8.4075] | 5.8784 | 等效 | `r1:metrics_deepseek:case-a:C3:T7` | `aggregate:case-a:c_a3` | — |
| p0-formal-s41 | `metrics_deepseek` | case-b | `c_b1` | 1 | background_bias/execution | decrease | [-0.1200, -0.0200] | 0.0000 [0.0000, 0.0000] | 2.7243 | 方向/幅度 | `r1:metrics_deepseek:case-b:C1:T1` | `aggregate:case-b:c_b1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-b | `c_b1` | 1 | clamped_rmse/full | decrease | [-0.9000, -0.2000] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:metrics_deepseek:case-b:C1:T2` | `aggregate:case-b:c_b1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-b | `c_b1` | 1 | free_rmse/full | decrease | [-0.7000, -0.1500] | -3.1125 [-3.2079, -2.9999] | 2.7243 | 幅度 | `r1:metrics_deepseek:case-b:C1:T3` | `aggregate:case-b:c_b1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-b | `c_b1` | 1 | end_response/end | decrease | [-0.8000, -0.1500] | -4.6327 [-4.7882, -4.4863] | 2.7243 | 幅度 | `r1:metrics_deepseek:case-b:C1:T4` | `aggregate:case-b:c_b1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-b | `c_b1` | 1 | nll/full | decrease | [-0.0400, -0.0050] | 0.0111 [0.0094, 0.0128] | 2.7243 | 方向/幅度 | `r1:metrics_deepseek:case-b:C1:T5` | `aggregate:case-b:c_b1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-b | `c_b1` | 1 | background_bias/recovery | decrease | [-0.1200, -0.0200] | 0.0000 [0.0000, 0.0000] | 2.7243 | 方向/幅度 | `r1:metrics_deepseek:case-b:C1:T6` | `aggregate:case-b:c_b1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-b | `c_b1` | 1 | free_end_response/end | decrease | [-0.8000, -0.1500] | -4.7172 [-4.8391, -4.5971] | 2.7243 | 幅度 | `r1:metrics_deepseek:case-b:C1:T7` | `aggregate:case-b:c_b1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-b | `c_b3` | 1 | background_bias/execution | decrease | [-0.1800, -0.0400] | 0.0000 [0.0000, 0.0000] | 2.7243 | 方向/幅度 | `r1:metrics_deepseek:case-b:C3:T1` | `aggregate:case-b:c_b3` | `r1:metrics_deepseek:case-b` |
| p0-formal-s41 | `metrics_deepseek` | case-b | `c_b3` | 1 | clamped_rmse/full | decrease | [-1.2000, -0.3000] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:metrics_deepseek:case-b:C3:T2` | `aggregate:case-b:c_b3` | `r1:metrics_deepseek:case-b` |
| p0-formal-s41 | `metrics_deepseek` | case-b | `c_b3` | 1 | free_rmse/full | decrease | [-1.0000, -0.2000] | -3.0954 [-3.1928, -2.9821] | 2.7243 | 幅度 | `r1:metrics_deepseek:case-b:C3:T3` | `aggregate:case-b:c_b3` | `r1:metrics_deepseek:case-b` |
| p0-formal-s41 | `metrics_deepseek` | case-b | `c_b3` | 1 | end_response/end | decrease | [-0.9000, -0.1500] | -4.6327 [-4.7882, -4.4863] | 2.7243 | 幅度 | `r1:metrics_deepseek:case-b:C3:T4` | `aggregate:case-b:c_b3` | `r1:metrics_deepseek:case-b` |
| p0-formal-s41 | `metrics_deepseek` | case-b | `c_b3` | 1 | nll/full | decrease | [-0.0600, -0.0100] | 0.0119 [0.0099, 0.0138] | 2.7243 | 方向/幅度 | `r1:metrics_deepseek:case-b:C3:T5` | `aggregate:case-b:c_b3` | `r1:metrics_deepseek:case-b` |
| p0-formal-s41 | `metrics_deepseek` | case-b | `c_b3` | 1 | background_bias/recovery | decrease | [-0.1800, -0.0400] | 0.0000 [0.0000, 0.0000] | 2.7243 | 方向/幅度 | `r1:metrics_deepseek:case-b:C3:T6` | `aggregate:case-b:c_b3` | `r1:metrics_deepseek:case-b` |
| p0-formal-s41 | `metrics_deepseek` | case-b | `c_b3` | 1 | free_end_response/end | decrease | [-0.9000, -0.1500] | -4.8479 [-4.9742, -4.7176] | 2.7243 | 幅度 | `r1:metrics_deepseek:case-b:C3:T7` | `aggregate:case-b:c_b3` | `r1:metrics_deepseek:case-b` |
| p0-formal-s41 | `metrics_gpt` | case-a | `c_a1` | 1 | clamped_rmse/full | decrease | [-1.8000, -0.2000] | -2.3406 [-2.5169, -2.1527] | 1.8478 | 幅度 | `r1:metrics_gpt:case-a:C1:T2` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `metrics_gpt` | case-a | `c_a1` | 1 | free_rmse/full | decrease | [-2.5000, -0.3000] | -1.8478 [-1.9701, -1.7274] | 1.8478 | 方向 | `r1:metrics_gpt:case-a:C1:T3` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `metrics_gpt` | case-a | `c_a1` | 1 | nll/full | unchanged | [-0.0200, 0.0200] | 0.0257 [0.0244, 0.0272] | 1.8478 | 等效 | `r1:metrics_gpt:case-a:C1:T5` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `metrics_gpt` | case-a | `c_a2` | 1 | clamped_rmse/full | decrease | [-1.8000, -0.2000] | -2.3406 [-2.5169, -2.1527] | 8.4853 | 幅度 | `r1:metrics_gpt:case-a:C2:T2` | `aggregate:case-a:c_a2` | `r1:metrics_gpt:case-a` |
| p0-formal-s41 | `metrics_gpt` | case-a | `c_a2` | 1 | free_rmse/full | decrease | [-6.5000, -1.5000] | -8.4853 [-8.7119, -8.2605] | 8.4853 | 幅度 | `r1:metrics_gpt:case-a:C2:T3` | `aggregate:case-a:c_a2` | `r1:metrics_gpt:case-a` |
| p0-formal-s41 | `metrics_gpt` | case-a | `c_a2` | 1 | free_end_response/end | decrease | [-11.0000, -1.0000] | -13.1030 [-13.4862, -12.7010] | 8.4853 | 幅度 | `r1:metrics_gpt:case-a:C2:T7` | `aggregate:case-a:c_a2` | `r1:metrics_gpt:case-a` |
| p0-formal-s41 | `metrics_gpt` | case-a | `c_a3` | 1 | free_rmse/full | decrease | [-5.5000, -0.5000] | -5.8784 [-6.0950, -5.6670] | 5.8784 | 幅度 | `r1:metrics_gpt:case-a:C3:T3` | `aggregate:case-a:c_a3` | — |
| p0-formal-s41 | `metrics_gpt` | case-b | `c_b1` | 1 | clamped_rmse/full | decrease | [-1.2000, -0.1500] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:metrics_gpt:case-b:C1:T2` | `aggregate:case-b:c_b1` | `r1:metrics_gpt:case-b` |
| p0-formal-s41 | `metrics_gpt` | case-b | `c_b1` | 1 | free_rmse/full | decrease | [-1.4000, -0.3000] | -3.1125 [-3.2079, -2.9999] | 2.7243 | 幅度 | `r1:metrics_gpt:case-b:C1:T3` | `aggregate:case-b:c_b1` | `r1:metrics_gpt:case-b` |
| p0-formal-s41 | `metrics_gpt` | case-b | `c_b3` | 1 | clamped_rmse/full | decrease | [-1.2000, -0.1500] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r1:metrics_gpt:case-b:C3:T2` | `aggregate:case-b:c_b3` | — |
| p0-formal-s41 | `metrics_gpt` | case-b | `c_b3` | 1 | free_rmse/full | decrease | [-1.6000, -0.1000] | -3.0954 [-3.1928, -2.9821] | 2.7243 | 幅度 | `r1:metrics_gpt:case-b:C3:T3` | `aggregate:case-b:c_b3` | — |
| p0-formal-s41 | `metrics_gpt` | case-b | `c_b3` | 1 | nll/full | decrease | [-0.0900, -0.0050] | 0.0119 [0.0099, 0.0138] | 2.7243 | 方向/幅度 | `r1:metrics_gpt:case-b:C3:T5` | `aggregate:case-b:c_b3` | — |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b1` | 2 | background_bias/execution | decrease | [-0.1500, -0.0500] | 0.0000 [0.0000, 0.0000] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C1:T1` | `aggregate:case-b:c_b1` | `r2:evidence_deepseek:case-b` |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b1` | 2 | clamped_rmse/full | decrease | [-0.6000, -0.1000] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r2:evidence_deepseek:case-b:C1:T2` | `aggregate:case-b:c_b1` | `r2:evidence_deepseek:case-b` |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b1` | 2 | free_rmse/full | decrease | [-0.5000, -0.1000] | -3.1125 [-3.2079, -2.9999] | 2.7243 | 幅度 | `r2:evidence_deepseek:case-b:C1:T3` | `aggregate:case-b:c_b1` | `r2:evidence_deepseek:case-b` |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b1` | 2 | end_response/end | increase | [0.1000, 0.8000] | -4.6327 [-4.7882, -4.4863] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C1:T4` | `aggregate:case-b:c_b1` | `r2:evidence_deepseek:case-b` |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b1` | 2 | nll/full | decrease | [-0.0400, -0.0100] | 0.0111 [0.0094, 0.0128] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C1:T5` | `aggregate:case-b:c_b1` | `r2:evidence_deepseek:case-b` |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b1` | 2 | background_bias/recovery | decrease | [-0.1500, -0.0500] | 0.0000 [0.0000, 0.0000] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C1:T6` | `aggregate:case-b:c_b1` | `r2:evidence_deepseek:case-b` |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b1` | 2 | free_end_response/end | increase | [0.1000, 0.8000] | -4.7172 [-4.8391, -4.5971] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C1:T7` | `aggregate:case-b:c_b1` | `r2:evidence_deepseek:case-b` |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b3` | 2 | background_bias/execution | decrease | [-0.2500, -0.0500] | 0.0000 [0.0000, 0.0000] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C3:T1` | `aggregate:case-b:c_b3` | — |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b3` | 2 | clamped_rmse/full | decrease | [-0.7000, -0.1000] | -2.7243 [-2.8866, -2.5586] | 2.7243 | 幅度 | `r2:evidence_deepseek:case-b:C3:T2` | `aggregate:case-b:c_b3` | — |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b3` | 2 | free_rmse/full | decrease | [-0.6000, -0.1000] | -3.0954 [-3.1928, -2.9821] | 2.7243 | 幅度 | `r2:evidence_deepseek:case-b:C3:T3` | `aggregate:case-b:c_b3` | — |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b3` | 2 | end_response/end | increase | [0.1000, 1.0000] | -4.6327 [-4.7882, -4.4863] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C3:T4` | `aggregate:case-b:c_b3` | — |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b3` | 2 | nll/full | decrease | [-0.0500, -0.0100] | 0.0119 [0.0099, 0.0138] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C3:T5` | `aggregate:case-b:c_b3` | — |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b3` | 2 | background_bias/recovery | decrease | [-0.2500, -0.0500] | 0.0000 [0.0000, 0.0000] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C3:T6` | `aggregate:case-b:c_b3` | — |
| p0-formal-s41 | `evidence_deepseek` | case-b | `c_b3` | 2 | free_end_response/end | increase | [0.1000, 1.0000] | -4.8479 [-4.9742, -4.7176] | 2.7243 | 方向/幅度 | `r2:evidence_deepseek:case-b:C3:T7` | `aggregate:case-b:c_b3` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a1` | 2 | background_bias/execution | decrease | [-0.2000, -0.0500] | -0.0265 [-0.0311, -0.0219] | 1.8478 | 方向/幅度 | `r2:metrics_deepseek:case-a:C1:T1` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a1` | 2 | clamped_rmse/full | decrease | [-0.4000, -0.0500] | -2.3406 [-2.5169, -2.1527] | 1.8478 | 幅度 | `r2:metrics_deepseek:case-a:C1:T2` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a1` | 2 | free_rmse/full | decrease | [-1.5000, -0.3000] | -1.8478 [-1.9701, -1.7274] | 1.8478 | 方向/幅度 | `r2:metrics_deepseek:case-a:C1:T3` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a1` | 2 | end_response/end | unchanged | [-0.2500, 0.2500] | -4.1842 [-4.3227, -4.0459] | 1.8478 | 等效 | `r2:metrics_deepseek:case-a:C1:T4` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a1` | 2 | nll/full | decrease | [-0.0250, -0.0050] | 0.0257 [0.0244, 0.0272] | 1.8478 | 方向/幅度 | `r2:metrics_deepseek:case-a:C1:T5` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a1` | 2 | background_bias/recovery | decrease | [-0.1500, -0.0300] | -0.0132 [-0.0193, -0.0071] | 1.8478 | 方向/幅度 | `r2:metrics_deepseek:case-a:C1:T6` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a1` | 2 | free_end_response/end | unchanged | [-0.2500, 0.2500] | -2.0679 [-2.2899, -1.8551] | 1.8478 | 等效 | `r2:metrics_deepseek:case-a:C1:T7` | `aggregate:case-a:c_a1` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a2` | 2 | background_bias/execution | decrease | [-0.3000, -0.1000] | -0.3946 [-0.4105, -0.3795] | 8.4853 | 幅度 | `r2:metrics_deepseek:case-a:C2:T1` | `aggregate:case-a:c_a2` | `r2:metrics_deepseek:case-a` |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a2` | 2 | clamped_rmse/full | decrease | [-0.6000, -0.1000] | -2.3406 [-2.5169, -2.1527] | 8.4853 | 幅度 | `r2:metrics_deepseek:case-a:C2:T2` | `aggregate:case-a:c_a2` | `r2:metrics_deepseek:case-a` |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a2` | 2 | free_rmse/full | decrease | [-2.5000, -0.5000] | -8.4853 [-8.7119, -8.2605] | 8.4853 | 幅度 | `r2:metrics_deepseek:case-a:C2:T3` | `aggregate:case-a:c_a2` | `r2:metrics_deepseek:case-a` |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a2` | 2 | end_response/end | decrease | [-1.0000, -0.2000] | -4.1842 [-4.3227, -4.0459] | 8.4853 | 幅度 | `r2:metrics_deepseek:case-a:C2:T4` | `aggregate:case-a:c_a2` | `r2:metrics_deepseek:case-a` |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a2` | 2 | nll/full | decrease | [-0.0350, -0.0080] | 0.0272 [0.0262, 0.0282] | 8.4853 | 方向/幅度 | `r2:metrics_deepseek:case-a:C2:T5` | `aggregate:case-a:c_a2` | `r2:metrics_deepseek:case-a` |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a2` | 2 | background_bias/recovery | decrease | [-0.2000, -0.0400] | -0.2398 [-0.2563, -0.2239] | 8.4853 | 幅度 | `r2:metrics_deepseek:case-a:C2:T6` | `aggregate:case-a:c_a2` | `r2:metrics_deepseek:case-a` |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a2` | 2 | free_end_response/end | decrease | [-1.0000, -0.2000] | -13.1030 [-13.4862, -12.7010] | 8.4853 | 幅度 | `r2:metrics_deepseek:case-a:C2:T7` | `aggregate:case-a:c_a2` | `r2:metrics_deepseek:case-a` |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a3` | 2 | background_bias/execution | decrease | [-0.3000, -0.1000] | -0.3946 [-0.4105, -0.3795] | 5.8784 | 幅度 | `r2:metrics_deepseek:case-a:C3:T1` | `aggregate:case-a:c_a3` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a3` | 2 | free_rmse/full | decrease | [-1.5000, -0.2000] | -5.8784 [-6.0950, -5.6670] | 5.8784 | 幅度 | `r2:metrics_deepseek:case-a:C3:T3` | `aggregate:case-a:c_a3` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a3` | 2 | nll/full | decrease | [-0.0200, -0.0020] | 0.0000 [0.0000, 0.0000] | 5.8784 | 方向/幅度 | `r2:metrics_deepseek:case-a:C3:T5` | `aggregate:case-a:c_a3` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a3` | 2 | background_bias/recovery | decrease | [-0.2000, -0.0400] | -0.2398 [-0.2563, -0.2239] | 5.8784 | 幅度 | `r2:metrics_deepseek:case-a:C3:T6` | `aggregate:case-a:c_a3` | — |
| p0-formal-s41 | `metrics_deepseek` | case-a | `c_a3` | 2 | free_end_response/end | unchanged | [-0.2500, 0.2500] | -8.8117 [-9.2243, -8.4075] | 5.8784 | 等效 | `r2:metrics_deepseek:case-a:C3:T7` | `aggregate:case-a:c_a3` | — |
