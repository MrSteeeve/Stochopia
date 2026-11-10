# Payoff DSL：指针与 Stochopia 适配说明

规范的唯一权威版本在 StructurerDSL 仓库：`~/GitHub/StructurerDSL/docs/SPEC.md`。
本文件只记录 Stochopia 作为消费者的适配约定，不重复语言定义。

## 依赖方向

Stochopia 依赖 `structurerdsl`（editable 安装在 `.venv`），StructurerDSL 不依赖 Stochopia。
语言、解释器、类型系统、验证器的改动一律在 StructurerDSL 仓库进行，并回写其 `docs/SPEC.md`。

## Stochopia 侧的责任

1. **等价性测试**：`tests/test_structurerdsl_equivalence.py` 把 DSL 解释器钉在本仓库的
   确定性定价器上。雪球与自动赎回在与 `pricing._mc_stats` 完全相同的随机路径上逐路径对账
   （精度 1e-9，不是蒙特卡洛标准误）；vanilla 对 `bs_call` / `bs_put`（3 倍标准误）；
   障碍期权对连续监测的 `barrier_option` 做 Broadie–Glasserman–Kou 障碍平移后比对。
2. **雪球到期规则差异**：本仓库 `_mc_stats` 的规则是"敲入且到期低于期初才亏损，否则付全额票息
   （即使曾敲入）"；StructurerDSL `catalog.snowball` 的规则是"未敲入才付票息"。等价性测试里
   用一段显式 DSL 程序复刻本仓库规则，两种规则都能表达，协议冻结期间本仓库口径不变。
3. **日期映射**：DSL 只认网格索引；本仓库的月末快照映射到月频网格（N = 月数，`t_years = i/12`），
   日频路径映射到日频网格。障碍监测频率由传给 `touched` 的日程表决定。
4. **迁移步骤**（对应 SPEC 第 13 节）：`parse_product_spec` 增加 `to_dsl()` 把 `ProductSpec`
   编译为 DSL 程序；环境 `step()` 的定价与结算切换到解释器时，`ProductSpec` 路径保留为兼容层。
   在 v3 协议冻结解除前不切换。

## 研究适配器的支付下界规则（`stochopia.adapters.structurer_dsl`）

供 OpenHyra 的 DSL 设计实验使用，正式 Benchmark v3 环境不消费这个适配器；规则按名义本金计量，
不意味着保护实际投入，需求评价仍须比较支付下界与 Q 现值购买价。

1. **无条件到期支付**（2026-10-10）：根节点为 `premium`、唯一直接子节点为到期 `pay` 时，
   采用检查器计算的 `report.client_lo` 作为下界。此前所有 `pay` 都被视为可能被抑制而给零下界，
   曾把表达式下界 0.85 的无条件合约误算成 100% 本金损失。
2. **组合支付**（2026-10-11）：无自动终止时，无条件到期支付的下界通过组合、缩放、方向与资金
   包装传播；进入条件分支不继承无条件假设。含自动终止的合约仍用保守规则（提前终止可能抑制到期腿）。
3. 条件支付、自动终止及其他形式沿用保守下界；不扩展语言，不声称通用验证器的完整性。
   回归用例：固定到期偿付 1.00 / 0.85 加条件票息 0.24 / 0.19，修正后下界分别为 1.00 / 0.85；
   含提前终止的对照仍为 0。测试在 `tests/test_structurerdsl_adapter.py`。

规则变更不重评旧实验结果；用新适配器离线重评旧候选必须另存并标注评价版本。

## 状态

- 2026-09-16：StructurerDSL v0.1 解释器、定价、类型系统、检查器就绪；本仓库等价性测试加入。
- 2026-10-09：`stochopia.environment.dsl.StochopiaDSLEnv`（level `DSLExperimental`）与
  `stochopia.adapters.structurer_dsl` 接通，DSL 合约可在实验环境中报价、成交、结算、重放；
  正式 Level-0 环境与 `ProductSpec` 路径不变，`to_dsl()` 未实现，v3 协议冻结未解除。
