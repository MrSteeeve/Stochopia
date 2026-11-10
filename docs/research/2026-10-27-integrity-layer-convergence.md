# 完整性机制收敛（决策记录）

日期：2026-10-27。作者：Claude Fable 5.1（主会话）。状态：已实施。

## 问题

哈希、manifest、审计报告、版本串的密度远高于它们承担的证明责任：约 3500 行相关代码；同一份
raw `manifest.sha256` 被 9 个模块各自重验；52 个 schema 版本串、30 个版本常量而无任何兼容分支；
`TrajectoryMetadata` 24 个字段中 20 个是哈希或版本；`ScenarioProvenance` 46 个字段约 20 个可推导；
8 份入库审计 JSON，其中记录的构建器源码哈希已与代码不一致。根因是每接一个数据源就复制上一轮的
"冻结仪式"，且下游一律不信任上游。防御对象偏向"自己的上一个函数篡改了文件"，而不是项目真正
面对的两个约束：原始数据不能再分发；评测结果要能复现。

## 决定

用户裁定：这是科研实验代码，凡是只为防篡改存在、又在运行时反复计算哈希的机制一律删除，
审计材料全部移除；不能因为安全增加计算复杂度。

### 删除

- 审计层：`raw_data_audit`、`historical_index_audit`、`formal_gate_audit`、`data/audit/*`、
  `data/reference/*_protocol.json`、四份数据审计文档；同日随后整体删除了无人消费的
  `multi_asset_data` 采集模块及其原始数据（备份在 `outputs/backups/`）。
- 原始数据层：append-only 预检与 `manifest.sha256` 写出。
- 派生输入：日历与 Q 侧 bundle 改为固定文件名，不写 manifest；删除对 raw manifest 的重验、
  构建器源码哈希、许可证文件哈希、P/Q 语料哈希；`ScenarioProvenance` 瘦身，`GeneratedMarketPath`
  不再对照 provenance 自校验；`MarketSnapshot.source` 改为 `synthetic;schema=…;episode=…;seed=…`。
- 评测层：每步 `state_hash`、轨迹逐条哈希链、`root_hash`、`implementation_hash`、工作树 diff 哈希、
  各类配置哈希；轨迹格式升为 `stochopia.trajectory-json.v2`，v1 轨迹被评测器明确拒收。
  `evaluate-trajectory` 默认按记录的 transition 计分，`--replay` 才重建环境逐项比对；
  `aggregate-v3` 不再要求 `replay_verified` 与 `evaluation_hash`；续跑不再校验与重放。

### 保留

缓存 sidecar 的 `sha256`（加载时校验一次，识别不完整下载）；`task_hash` / `suite_hash` /
`policy_config_hash` 作为分组 ID；`EnvResponseCache` 的 key；`run-manifest` 的 run/job fingerprint；
原始动作输出的 `raw_hash`；`git_sha`。这些要么是功能性 ID，要么只在一次性路径上计算一次。

> 2026-11-08 补：其中缓存 sidecar `sha256`、`run-manifest` fingerprint 与 `raw_output_sha256`
> 已按用户裁定删除，详见 `BENCHMARK_PROTOCOL.md`「2026-11-08：完整性机制收敛（续）」。

### 不变

定价、结算、约束、指标口径；`economic_replay` 主张（改为按需执行）；`conversation_replay` 主张。

## 验证

全量测试 513 个通过；用本地原始数据重跑日历与 Q 侧构建，2674 个开市日与 80 条 Q 记录与改动前
完全相同；`pricing.py`、`benchmark.py`、`products.py` 与改动前逐字节一致（独立审查核对）。
协议文档对应章节见 `BENCHMARK_PROTOCOL.md` 的「2026-10-27：完整性机制收敛」。
动手前的完整工作区备份：`outputs/backups/pre-integrity-removal-2026-10-27.tgz`。
