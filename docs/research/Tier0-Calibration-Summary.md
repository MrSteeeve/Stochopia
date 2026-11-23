# Tier 0 开发集校准（Step 4.1）

**结论（2026-11-24，北京时间）**。增强信息漂移的配置在开发集上区分了两个诊断题。A 的背景流隔离使自由路径 RMSE 减少 6.593 tick，95% 区间为 [−6.958, −6.251]。B 在共用独立背景权重时加入自身标记价格分支，钳制路径 RMSE 减少 1.775 tick，95% 区间为 [−1.900, −1.643]。独立模拟日复跑确认了两项总体改善，B 有一个训练种子反向。建议沿用增强配置，正式阈值经人工确认后再登记。

## 数据、模型与配对口径

使用 Stochopia `d0f09c3`、OpenHyra `b3f252c` 的冻结实现。校准只拟合模拟 `train`，在模拟 `dev` 生成诊断与候选结果。三个配置共享 GRU 训练方案和五个独立初始化种子 `103, 137, 211, 293, 367`。每个配置 16 个训练日、32 个开发日、每种子 120 步。CPU 单线程、隐藏维度 32、1 层、batch 16、窗口 64、学习率 0.003，`own_start=inside`、标记无漏报或误报。背景权重对照额外拟合一个匿名对角 SSM，所有候选的价格主干均为 GRU。

| 开发配置 | train/dev 命名空间前缀 | B 题及共同训练世界 `info_drift` | `informed_frac` | 用途 |
|---|---|---:|---:|---|
| 标准 | `p0-step41-calibration-v1` | 0.85 | 0.40 | 复测脚本的默认世界 |
| 增强 | `p0-step41-calibration-v1` | 1.80 | 0.25 | 用同一日种子比较世界设置，下表主结果 |
| 增强复跑 | `p0-step41-replication-v1` | 1.80 | 0.25 | 用另一个 train/dev 命名空间重训和复测，保留五个初始化种子 |

A 的评估世界是 `plain`，`info_drift=0`、`informed_frac=0`。B 是 `informed`。两题共用各自配置登记的 informed 训练世界。其余共同参数为每模拟日 2,048 笔、预热 256 笔、`noise_std=0.50`、成交流 `n_slots=2, alpha=1.24, max_len=2048`、传播子 `beta=0.28, tau_max=96, n_scales=6`，其余字段沿用冻结代码默认。每个模拟日评估 6 个固定起点，上下文 48 笔。买卖两侧分别注入 12 笔，自身子单在合并时钟上间隔 3 个事件，`spacing=2`、`relax=18`。全路径 54 个合并事件。停手后的背景偏移从最后一个子单之后算起，共 20 个背景事件。

同一题内，候选与父模型共享交易日、训练种子、注入计划和逐日抽样输入。读数是候选减父模型。标量先取日均再等权汇总训练种子。RMSE 先对日路径按时间点平均，再对平均路径和真值做 RMSE，最后等权汇总各训练种子的 RMSE 差。95% 区间用 2,000 次成组日重抽，单次重抽的日期同时用于两个模型、真值和全部训练种子。区间以已训练模型为条件，训练噪声用五个种子的点估计范围另报。A/B 共用日种子，两个题的天数不相加。下一事件 NLL 使用归一化收益和无注入事件。冻结实现省略了高斯归一化常数，配对差值中该常数会抵消。

## F1 两类诊断题

`Runner.diagnose()` 的流残差为自由路径减钳制路径的 RMSE，价格残差为钳制路径减真值的 RMSE。分类要求主通道残差至少 0.20 tick，且超出另一通道至少 0.15 tick。诊断区间使用与 `diagnose()` 相同的 `random_inputs_ref="diagnosis"` 重播逐日路径，逐种子核对聚合结果后再重抽。A 的优势记为流减价，B 记为价减流。两题以区间下端超过 0.15 且五个种子分类一致作为诊断通过条件。

| 配置 | A 流优势，95% 日区间 | B 价优势，95% 日区间 | 分类一致性 |
|---|---:|---:|---|
| 标准 | 3.326 [3.048, 3.609] | 2.974 [2.867, 3.028] | A = flow 5/5，B = price 5/5 |
| 增强 | 3.881 [3.383, 4.352] | 3.468 [3.210, 3.699] | A = flow 5/5，B = price 5/5 |
| 增强复跑 | 3.234 [2.790, 3.713] | 4.055 [3.706, 4.290] | A = flow 5/5，B = price 5/5 |

增强配置的逐种子 `Runner.diagnose()` 读数（tick）。

| 训练种子 | A 流残差 | A 价格残差 | A 优势 | B 流残差 | B 价格残差 | B 优势 |
|---:|---:|---:|---:|---:|---:|---:|
| 103 | 5.775 | 1.754 | 4.021 | 0.233 | 3.382 | 3.149 |
| 137 | 5.778 | 2.074 | 3.705 | 0.091 | 3.367 | 3.276 |
| 211 | 7.925 | 3.887 | 4.038 | 0.161 | 4.813 | 4.652 |
| 293 | 6.774 | 3.198 | 3.575 | 0.299 | 3.890 | 3.591 |
| 367 | 6.475 | 2.411 | 4.064 | 0.265 | 2.936 | 2.672 |

增强配置的优势跨种子极差分别为 0.488、1.980 tick。优势均值除以跨种子样本标准差为 17.27、4.69，仅作信噪比描述。诊断重抽种子为 A `1823350129`、B `3306676263`。

## 候选菜单与开发结果

| 题 | 父模型 | 背景流改动 | 价格导向改动 | 联合改动 |
|---|---|---|---|---|
| A | `x00` 匿名 GRU | `x01` / `separate_flow`，保持父价格权重，用无注入流 | `a_self` / `self`，加自身标记和价格输出，保留联合生成 | `a_joint` / `separate_flow_own_price`，以 `x00` 提供独立流，加自身标记价格分支 |
| B | `x11` 独立流 + 匿名 GRU 价格 | `x13` / `separate_flow`，保持父价格权重，把背景模型换成 `x02` 匿名 SSM | `x12` / `separate_flow_own_price`，保持与 `x11` 相同的独立背景权重 | `b_joint` / `separate_flow_own_price`，加入自身标记价格分支并换到 `x02` 背景 |

A 的 `self` 会重训共享 GRU 状态和方向头，因此是对准价格的候选，无法宣称它在自由生成时只作用于价格。B 的 `x12` 与父模型持有相同背景权重，提供严格的价格通道对照。独立流结构给买卖注入共用无注入方向，因此背景偏移为零是实现核对。B 的 `x13` 共用父价格权重，钳制时又共用历史方向，钳制零差由结构保证。

下表每个候选单元格为 **配对点估计 [95% 日区间] / [五训练种子点估计最小、最大]**。父模型列为同口径的绝对值。背景偏移是无量纲的平均有符号成交方向，正值表示沿自身注入方向偏移。两题的模拟器末点真值均为 1.366 tick，末点读数的改善方向取决于父模型位于真值哪一侧。

### A 题

| 读数 | 父模型 | `x01` | `a_self` | `a_joint` |
|---|---:|---:|---:|---:|
| 下一事件 NLL | 0.084 | 0.000 [0.000, 0.000] / [0.000, 0.000] | 0.026 [0.023, 0.028] / [−0.012, 0.086] | 0.026 [0.024, 0.029] / [−0.012, 0.087] |
| 自由路径 RMSE | 8.968 | −6.593 [−6.958, −6.251] / [−7.817, −5.660] | −0.604 [−0.747, −0.467] / [−4.798, 1.286] | −7.705 [−8.021, −7.386] / [−11.069, −6.121] |
| 钳制路径 RMSE | 2.485 | 0.000 [0.000, 0.000] / [0.000, 0.000] | −1.059 [−1.156, −0.956] / [−3.090, −0.233] | −1.059 [−1.156, −0.956] / [−3.090, −0.233] |
| 执行期背景偏移 | 0.458 | −0.458 [−0.478, −0.439] / [−0.484, −0.442] | −0.062 [−0.069, −0.055] / [−0.129, −0.035] | −0.458 [−0.478, −0.439] / [−0.484, −0.442] |
| 停手后背景偏移 | 0.473 | −0.473 [−0.508, −0.439] / [−0.492, −0.456] | −0.048 [−0.060, −0.037] / [−0.074, −0.033] | −0.473 [−0.508, −0.439] / [−0.492, −0.456] |
| 钳制末点响应 | 4.443 | 0.000 [0.000, 0.000] / [0.000, 0.000] | −1.743 [−1.912, −1.587] / [−4.472, −0.711] | −1.743 [−1.912, −1.587] / [−4.472, −0.711] |
| 自由末点响应 | 16.070 | −11.854 [−12.559, −11.155] / [−14.516, −10.310] | −0.870 [−1.174, −0.575] / [−7.863, 2.220] | −13.695 [−14.337, −13.037] / [−19.381, −11.157] |

### B 题

| 读数 | 父模型 | `x13` | `x12` | `b_joint` |
|---|---:|---:|---:|---:|
| 下一事件 NLL | 0.438 | 0.021 [0.011, 0.030] / [0.008, 0.034] | 0.008 [0.006, 0.010] / [−0.017, 0.025] | 0.028 [0.018, 0.039] / [0.006, 0.060] |
| 自由路径 RMSE | 3.522 | 0.558 [0.475, 0.645] / [0.527, 0.609] | −1.706 [−1.845, −1.565] / [−3.725, −0.570] | −1.205 [−1.381, −1.033] / [−3.246, −0.073] |
| 钳制路径 RMSE | 3.700 | 0.000 [0.000, 0.000] / [0.000, 0.000] | −1.775 [−1.900, −1.643] / [−3.861, −0.864] | −1.775 [−1.900, −1.643] / [−3.861, −0.864] |
| 执行期背景偏移 | 0.000 | 0.000 [0.000, 0.000] / [0.000, 0.000] | 0.000 [0.000, 0.000] / [0.000, 0.000] | 0.000 [0.000, 0.000] / [0.000, 0.000] |
| 停手后背景偏移 | 0.000 | 0.000 [0.000, 0.000] / [0.000, 0.000] | 0.000 [0.000, 0.000] / [0.000, 0.000] | 0.000 [0.000, 0.000] / [0.000, 0.000] |
| 钳制末点响应 | 6.171 | 0.000 [0.000, 0.000] / [0.000, 0.000] | −2.746 [−2.980, −2.531] / [−5.567, −1.248] | −2.746 [−2.980, −2.531] / [−5.567, −1.248] |
| 自由末点响应 | 5.830 | 1.055 [0.899, 1.211] / [0.862, 1.258] | −2.707 [−2.917, −2.467] / [−5.528, −0.771] | −1.759 [−2.033, −1.476] / [−4.585, 0.080] |

### 正例、错配与复跑

| 配置 | A 隔离流 · 自由 RMSE | A 价格导向 · 自由 RMSE | B 改价格 · 钳制 RMSE | B 改流 · 钳制 RMSE |
|---|---:|---:|---:|---:|
| 标准 | −6.109 [−6.342, −5.881] | −0.825 [−0.934, −0.717] | −0.690 [−0.779, −0.605] | 0.000 [0.000, 0.000] |
| 增强 | −6.593 [−6.958, −6.251] | −0.604 [−0.747, −0.467] | −1.775 [−1.900, −1.643] | 0.000 [0.000, 0.000] |
| 增强复跑 | −6.516 [−6.781, −6.261] | −1.635 [−1.808, −1.456] | −2.427 [−2.613, −2.227] | 0.000 [0.000, 0.000] |

- A 的隔离流方案在两个增强样本的五个种子均降低自由 RMSE，区间上端均低于建议达标线 −2.00 tick。增强配置的 `self` 仍留下执行期偏移 0.397 [0.379, 0.415] 和停手后偏移 0.426 [0.392, 0.458]。它对钳制 RMSE 有改善，无法单独消除方向偏移。复跑仍有 0.372 的执行期偏移。
- B 的 `x12` 在增强样本五个种子都降低钳制 RMSE。复跑的总体区间也通过 −0.50 tick 线，种子 293 恶化 0.352 tick，其余四个改善。复跑的种子范围为 [−3.880, +0.352] tick，增加开发日只缩窄以这些拟合为条件的区间，种子差异仍需报告。
- B 的 `x13` 钳制 RMSE 在两次增强运行都精确落在建议等效带 [−0.10, +0.10] 内。B 的联合改动与只改价格的钳制读数相同，主运行中改用 SSM 背景却令联合候选的自由 RMSE 相对 `x12` 高 0.500 tick。A 的联合改动有更低的自由 RMSE，单通道对照依然提供直接的归因证据。

## 正式 P0 建议

建议 `train / dev / test = 16 / 32 / 64` 个模拟日、`K=5`、每训练 120 步、每模拟日 6 个窗口、2,000 次日重抽。根据下表的开发集精度，64 天可作为正式复验的精度余量，实际检验力尚未测量。现有 `AnalysisSpec` 在开发和冻结侧共用，建议登记 `min_pairs=32, min_train_repeats=5`，正式测试清单另登记 64 个计划日并报告完成数。开发与正式复验使用独立的模拟日命名空间。训练和模拟日使用分开的种子流。

120 步还保留了先前开发预校准的有效设置。旧增强预校准（`p0-stronger-2026-11-23`，20 个开发日、同样 5 个种子）在同一世界参数下，B 价格候选减流对照的钳制 RMSE 在 120 步为 −1.770 tick [−1.990, −1.525]，在 200 步变为 +0.813 tick [+0.608, +1.014]。训练步数因此应与数据和候选一起登记。

| 开发子集 | A 隔离流自由 RMSE 区间宽度 | B 价格改动钳制 RMSE 区间宽度 |
|---:|---:|---:|
| 8 | 1.361 | 0.552 |
| 16 | 0.972 | 0.377 |
| 24 | 0.832 | 0.331 |
| 32 | 0.707 | 0.257 |

增强配置实测的单种子 GRU 拟合中位数 0.74 秒，匿名 SSM 背景拟合中位数 0.16 秒，32 日含父模和候选的配对测量中位数 0.47 秒，范围 0.39–0.65 秒。父模已拟合时，涉及重训的单种子试验典型合计约 1.2 秒。首次使用替代背景时另计背景拟合。每组 5 种子的单候选串行试验约 6.1 秒，未计跨种子 2,000 次汇总重抽和写盘。

| 分析项 | 最小效应 ε | 近零等效带 | 建议的达标条件 |
|---|---:|---:|---|
| A 自由路径 RMSE（tick） | 2.00 | [−0.25, +0.25] | 主项 `max_delta = −2.00` |
| B 钳制路径 RMSE（tick） | 0.50 | [−0.10, +0.10] | 主项 `max_delta = −0.50` |
| A 钳制路径 RMSE（tick） | 0.50 | [−0.10, +0.10] | A 中只改流的预测核对使用等效带，允许其他合格候选改善该项 |
| B 自由路径 RMSE（tick） | 0.50 | [−0.25, +0.25] | 报告与预测核对 |
| 执行期、停手后背景偏移（方向均值） | 0.10 | [−0.025, +0.025] | A 执行期 `max_delta = −0.10`，其余用于报告或结构核对 |
| 自由、钳制末点响应（tick） | 0.75 | [−0.25, +0.25] | 报告水平及与真值的差 |
| 下一事件 NLL（nats/event） | 0.03 | [−0.02, +0.02] | 两题护栏 `max_delta = +0.05` |

达标以上侧区间端点判定，等效需要整个区间落在等效带内。训练重复的范围继续独立报告。A 的自由路径主阈值落在正例与价格导向对照的区间之间，B 的钳制阈值与两次增强运行的总体区间均有余量，复跑的反向种子仍计入报告。等效带优先用于有结构依据的预判，普通非零读数区间太宽时可以是不可判定。建议两题都允许联合候选在效果上达标，机制归因以保持另一通道的配对为依据。正式评分若另需一个唯一参考答案，需在 Step 4.2 启动前确定如何处理联合候选。

把建议的主项和护栏代入现有 `assess_candidate()`，增强配置及其复跑中，A 的 `x01`、`a_joint` 与 B 的 `x12`、`b_joint` 均达标，`a_self`、`x13` 均被相应主项反驳。该结果来自阈值校准用的开发数据。

## 复现与回归

本机在 macOS arm64 上用 Python 3.12.6、PyTorch 2.14.0 运行。下列命令在 Stochopia 根目录执行，OpenHyra 保持相邻目录。默认复现增强配置。标准配置把 `STEP41_DRIFT`、`STEP41_FRACTION` 改为 `0.85`、`0.40`。增强复跑把 `STEP41_NAMESPACE` 改为 `p0-step41-replication-v1`。每次使用独立的 `STEP41_OUT`。

<details>
<summary>复现命令</summary>

```bash
env PYTHONPATH=.:../OpenHyra STEP41_OUT=outputs/phase4-step41/reproduced \
  STEP41_NAMESPACE=p0-step41-calibration-v1 STEP41_DRIFT=1.8 STEP41_FRACTION=0.25 \
  .venv/bin/python - <<'PY'
from dataclasses import replace
from os import environ
from pathlib import Path

import torch
from tasks.stochopia_tier0 import calibrate
from stochopia.common import atomic_write_json
from stochopia.harness.contract import CandidateRef, EvaluationTask
from stochopia.harness.report import summarize_candidate
from stochopia.harness.runner import Runner
from stochopia.stats import derive_seed

torch.set_num_threads(1)
output = Path(environ['STEP41_OUT'])
output.mkdir(parents=True, exist_ok=False)
namespace = environ['STEP41_NAMESPACE']
manifest = calibrate.build_manifest(
    namespace, steps=120, n_train_days=16, n_dev_days=32, train_seeds=(103, 137, 211, 293, 367),
    n_resamples=2000, namespace=namespace, hidden=32, windows_per_day=6,
    info_drift=float(environ['STEP41_DRIFT']), informed_fraction=float(environ['STEP41_FRACTION']),
)
a, b = manifest.scenarios
scenes = (
    replace(a, candidate_ids=('x00', 'x01', 'a_self', 'a_joint')),
    replace(b, candidate_ids=('x11', 'x12', 'x13', 'b_joint')),
)
added = (
    CandidateRef(candidate_id='a_self', backbone='gru', recipe='self', parent_id='x00', channels=('price',)),
    CandidateRef(candidate_id='a_joint', backbone='gru', recipe='separate_flow_own_price',
                 parent_id='x00', channels=('flow', 'price')),
    CandidateRef(candidate_id='b_joint', backbone='gru', recipe='separate_flow_own_price',
                 parent_id='x11', channels=('flow', 'price'), implementation={'background_candidate_id': 'x02'}),
)
analyses = list(manifest.analysis)
extra_readouts = {'background_bias': ('background_bias', 'recovery'), 'end_response': ('free_end_response', 'end')}
for spec in manifest.analysis:
    if spec.readout in extra_readouts:
        name, window = extra_readouts[spec.readout]
        analyses.append(replace(spec, analysis_id=f'{spec.scenario_id}:{name}:{window}', readout=name, window=window,
                                acceptance={'role': 'prediction'}, seed=derive_seed(
                                    'stochopia-tier0-calibration-bootstrap', spec.scenario_id, name, window)))
manifest = replace(manifest, scenarios=scenes, candidates=(*manifest.candidates, *added), analysis=tuple(analyses),
                   prediction_candidate_set={scene.scenario_id: scene.candidate_ids[1:] for scene in scenes})
runner = Runner(manifest)
days = manifest.splits['dev'].unit_ids
repeats = tuple(manifest.train_seeds)
atomic_write_json(output / 'manifest.json', manifest.to_dict())
for scene in scenes:
    sid = scene.scenario_id
    rows = {candidate: [] for candidate in scene.candidate_ids[1:]}
    direct, diagnosis_rows = {}, []
    for repeat in repeats:
        parent = runner.fit_candidate(scene.parent_id, repeat, output / 'weights')
        direct[repeat] = runner.diagnose(sid, parent)
        for candidate_id in rows:
            fit = runner.fit_candidate(candidate_id, repeat, output / 'weights')
            task = EvaluationTask(task_id=f'{sid}-{candidate_id}-{repeat}', split='dev', scenario_id=sid,
                                  parent_fit_id=parent.fit_id, candidate_fit_id=fit.fit_id, unit_ids=days,
                                  action_ref=f'{sid}:action', random_inputs_ref=f'{sid}:dev:rollout')
            pair = runner.run_paired_eval(task, parent, fit, train_seconds=fit.config['train_seconds'])
            rows[candidate_id].append(pair)
            atomic_write_json(output / 'measurements' / f'{task.task_id}.json', pair.to_dict())
            if candidate_id == scene.candidate_ids[1]:
                diagnosis_task = replace(task, task_id=f'diagnosis-{task.task_id}', random_inputs_ref='diagnosis')
                diagnosis_rows.append(runner.run_paired_eval(diagnosis_task, parent, fit))
    diagnose = calibrate._diagnostic_grid(scene, diagnosis_rows, direct, repeats, days, 2000)
    for repeat in repeats:
        for readout in ('flow_rmse', 'price_rmse'):
            if abs(diagnose['diagnoses'][repeat][readout] - direct[repeat][readout]) > 2e-5:
                raise ValueError(f'{sid}/{repeat}/{readout} differs from Runner.diagnose()')
    effects = {}
    for candidate_id, pairs in rows.items():
        estimates = summarize_candidate(manifest.analysis, pairs, required_unit_ids=days)
        effects[candidate_id] = {key: calibrate._delta_summary(delta) for key, delta in estimates.items()}
    atomic_write_json(output / f'{sid}-summary.json', {'diagnosis': diagnose, 'direct': direct, 'effects': effects})
PY
```

</details>

模拟日日种子由 `calibrate._split()` 派生。世界标识为 `p0-tier0-common-rng`，命名空间在上表前缀后加 `-train` 或 `-dev`，索引从零开始。全局 rollout 和 delta 种子分别为 `45721`、`34913`。读数重抽种子由 `AnalysisSpec.seed` 登记，保存在复现命令输出的 `manifest.json`。样本量敏感性用同一次增强运行的前 8/16/24/32 个日 ID 重算，训练种子集合保持一致。权重、逐日读数和运行参数留在忽略目录 `outputs/phase4-step41/`。

本地 `.venv/bin/python -m pytest -q tests --ignore=tests/test_harness_p0_acceptance.py` 通过 224 项，有 3 条上游 JAX-LOB 弃用提示。回归测试包含原有的进程内客户端固定响应夹具，校准的真实与替身 LLM 调用均为零。该轮只发布校准报告，正式 `test` 数据、冻结复验和三组对照留在阈值确认之后。
