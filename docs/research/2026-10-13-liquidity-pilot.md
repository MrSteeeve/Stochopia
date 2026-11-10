# 共同流动性最小合成研究环境

此环境与 Benchmark v3 分离，仅用于研究不同可执行机制怎样表达共同流动性观点。不是历史订单簿回测，不给出新机制首创或投资绩效结论。

接口：`generate_panel(seed=0, n_paths=64, config={}, scenario='normal')` 返回 `MarketPanel`，有 `paths`、`data`、`steps`、`dt`、`scenario`、`seed`、`config`。默认10股票、20日×78步，固定 `dt=1/(252*78)`；`days` 与 `steps_per_day` 可缩短测试，但不改变步长。默认 `rate=drift=0, sigma=.2`。`paths.spot` 为 `s0..s9,basket`；`data` 同时提供价格别名与公开价差、深度、恢复、`vol/variance` 和辅助 `liquidity` 指标；每股另有 `spread_s0/depth_s0/recovery_s0` 等。后者只是公开观测，不是可交易资产或评分真值。

`normal`、各单因子 `spread/depth/recovery` 的 `bad/good`、`common_bad/good`、`price_up/down`、`vol_high/low` 共用随机数。所有情景期初价格、流动性与期权报价严格相同；纯流动性干预不改变股票价格与期权报价。三个流动性因子有彼此独立的共同创新与每股特异创新；价格因子、波动率因子另行生成。`bad` 分别使后续价差×2、深度×.5、恢复×.5；`good` 相反。

`instrument_prices(panel)` 返回字典，每个矩阵形状 `(P,T+1)`，含股票、篮子与到期 ATM 欧式 `call/put`（行权价1）。期权用公开当期篮子波动率的标准 BS 公式合成报价；随机波动率与算术篮子使该公式只是报价近似，不能称精确无套利 Q 定价。股票与篮子价格以期初1归一化。

`execute_strategy(artifact,panel,risk_config)` 消费 `{track:'synthetic',policy:{targets:{instrument:sexpr},rebalance_every:78},params:{},rationale:''}`。每个权重是该工具市值/初始买方资本，可为负；空 `targets={}` 是现金对照。复用现有 DSL `parse/eval_expr`；只允许可观察的当期/历史表达式，批量累计函数按前缀计算；拒绝 `terminal/ret/hit_time/var_clock` 等未来或不支持形式。共享参数适用于全部目标。调仓仅市价且部分成交；现金不足或超过 gross 上限的订单削减到可融资部分，同时永久记录失败，不补充资本。

`risk_config` 必须显式给 `buyer_capital,seller_capital,fee_bps,borrow_rate,gross_limit`。本轮用户批准资本双方各100万元、禁止额外注资；根评价器落实留出 ES95≤资本10%、压力损失≤20%及报价。测试费用为 `fee_bps=1,borrow_rate=.03,gross_limit=2`，仅是合成假设。执行默认可覆盖参数：`short_margin_rate=.5,option_margin_rate=1,impact_bps=2,option_spread_bps=20`。价差为全宽/中价，单次交易收费半价差，加手续费和临时冲击；衍生品全价差至少20bp。每股深度单位为该步可成交名义/初始资本；篮子每单位对应每股.1单位并共享容量，故同深度时篮子总容量约10倍单股深度。衍生品容量单独限制为 `.25*depth*capital` 的标的名义，不假定免费流动性后盾。

冲击状态随交易改变，以 `1-recovery` 每步衰减，再影响后续执行成本；恢复越低，半衰期越长。短股票锁定卖出所得并额外收50%市值保证金，按年化借券费逐步扣现金；短期权锁定权利金所得及100%标的名义。Gross 中期权按标的名义计算。未使用卖方资本作为买方交易账户可借资金。现金余额与抵押品检查每一步；已有失败不能通过后续盈利洗白。

返回 `pnl,costs,peak_funding,turnover` 为 `(P,)`，`funding_violation,unfinished` 为布尔 `(P,)`，另有 `final_inventory` `(P,K)`、`weights` `(P,T+1,K)`、排序后的 `instrument_names`、`ending_cash`。`trace=True` 添加逐步 `cash/funding`。PnL只计实际现金减初始资本，零仓位PnL为0（无现金利息），不把未平仓公允价算已完成盈利。股票期末通过相同价差、冲击与容量清算；容量不足保留库存并标记失败。到期期权按法律现金结算内在价值，不伪造市场成交。根评价器必须拒绝资金失败/未结清路径，尤其未平空头收款不代表可分配收益。

`contract_cashflows(artifact,panel)` 消费 `{track:'contract',contract_sexpr:...,params:{},notional:...}`，返回 `(P,T+1)` 买方视角现金流。直接复用原生 `Program/check/run`，核对成熟期、股票和外部序列名称以及因果性；外部序列的无界负支付必须通过 DSL `clip` 等有限损失构造通过检查。保留中途负现金流，不自动注资。本轮仅接受 `premium` 根节点，不接受 `funded`；原生运行输出若有非零期初经营支付则报错。初始模型报价由根评价器另行处理。

`reference_metrics(panel,capital)` 为固定参考服务探针，不消费候选策略。请求量 `[.01,.1,.5]` 倍资本，期限 `[1,6,24]` 步；每个期初之后的公开状态冻结报价、深度、恢复，在期限内均匀买入篮子再反向卖出。返回 `cost/unfinished` `(P,3,3)`，分别为双边已成交成本/资本与未服务请求比例；`recovery_half_life` `(P,)` 为平均恢复半衰期（步数）。另有三乘三路径均值、请求量与期限数组。采用执行器2bp冲击系数及状态衰减，不含手续费，不称真实ADV或真实执行回测。容量不足时已成交成本可能下降，因此深度应按未服务比例评价，不能把少成交误解为更廉价。

验证：`.venv/bin/python -m pytest tests/test_liquidity_research.py tests/test_path_research_market.py tests/test_structurerdsl_adapter.py tests/test_structurerdsl_equivalence.py -q`，52项通过。仅新增本研究模块、测试与此说明，未修改 StructurerDSL、正式协议或原有未提交文件；未执行 API 调用。
