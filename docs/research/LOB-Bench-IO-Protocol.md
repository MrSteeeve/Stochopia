# LOB‑Bench 输入与输出协议

LOB‑Bench 评测限价订单簿（LOB）生成模型。模型交付可采样的未来轨迹，每条轨迹包含**逐笔消息**和**每条消息处理后的盘口状态**。评估器读取生成轨迹与历史真实轨迹，输出各项统计量的分布距离。论文比较了 LOBS5、RWKV‑4、RWKV‑6、Coletta 的生成模型和 Cont 等人的参数化基线。它们使用各自的训练方法和模型输入，通过统一的 LOBSTER 格式交给评估器。本文的代码行为以官方仓库 `276baf3` 版本为准。 [LOB‑Bench 论文 §4–6](https://arxiv.org/html/2502.09172v2#S4) · [官方代码版本](https://github.com/peernagy/lob_bench/tree/276baf3039bf7630b555891b9d04ad49a92b3356)

## 1. 模型和评估器的输入输出

| 环节 | 输入 | 输出 |
|---|---|---|
| 条件生成模型 | 截至观测时刻的历史上下文，由各模型定义。LOBS5 使用逐笔消息和盘口状态，论文中的 RWKV 只使用逐笔消息。 | 未来市场消息。以 LOBS5 为例，撮合环境根据消息更新盘口，导出逐笔消息及事件后的盘口轨迹。 |
| LOB‑Bench 评估器 | 已导出的真实轨迹、生成轨迹，以及可选的历史条件轨迹。 | 各统计量的分布距离、用于绘图和检查的逐项得分表，以及按预测步长计算的误差。 |

在条件生成实验里，真实轨迹是观测边界之后的历史实际结果，生成轨迹是从该边界出发的模型样本。多个随机生成样本可以对应同一个真实窗口。训练窗口、模型结构、训练目标和生成方式由实验自行定义。 [LOB‑Bench §4、§6](https://arxiv.org/html/2502.09172v2#S6) · [LOBS5 原论文](https://arxiv.org/html/2309.00638)

## 2. 轨迹文件格式

每条轨迹由两个**无表头、无索引的 CSV**组成。两个文件行数相等，盘口文件的第 (i) 行对应处理消息文件第 (i) 条消息**之后**的状态。历史、生成和条件轨迹使用相同的列顺序。 [LOBSTER 官方格式](https://php.lobsterdata.com/info/DataStructure.php) · [LOB‑Bench 解析实现](https://github.com/peernagy/lob_bench/blob/276baf3039bf7630b555891b9d04ad49a92b3356/data_loading.py#L54-L113)

### `message` 文件

列顺序固定为 `time,event_type,order_id,size,price,direction`。

| 列 | 含义与编码 |
|---|---|
| `time` | 交易日当地午夜以来的秒数，写成带小数点的十进制字符串，例如 `34200.000` 表示当地 09:30:00。日期由加载器从文件名读取。同一时刻多笔消息按原始消息顺序写入。 |
| `event_type` | `1` 新限价单，`2` 部分撤单，`3` 整单删除，`4` 可见挂单成交，`5` 隐藏单成交，`6` 集合撮合等交叉成交，`7` 停牌状态。原始 LOBS5 连续竞价实验使用类型 `1–4`。 |
| `order_id` | 订单引用。撤单或可见成交引用受影响的原挂单。导出需保持同一订单的生命周期引用一致。 |
| `size` | 这次事件涉及的股数。 |
| `price` | 原始 LOBSTER 的价格用美元乘 10,000 存为整数。 |
| `direction` | 挂单方向，`1` 买、`-1` 卖。`event_type=4` 记录的是**被成交的挂单方向**。例如主动买入吃掉卖单时，此列为 `-1`。 |

加载器把 `order_id` 等五个数值列强制转成 `int32`。源数据使用较大的订单号时，导出前应在相关窗口内重新编号，确保初始化订单、条件段和后续消息之间的引用一致。 [LOBSTER 官方格式](https://php.lobsterdata.com/info/DataStructure.php) · [LOB‑Bench 消息解析器](https://github.com/peernagy/lob_bench/blob/276baf3039bf7630b555891b9d04ad49a92b3356/data_loading.py#L54-L100)

### `orderbook` 文件

每档四列，从最优价向外排列。十档盘口每行 40 列。档位指有挂单的价格层，相邻档位之间可以隔多于一个最小价格单位。原始 LOBSTER 对缺失卖档填价格 `9999999999`、数量 `0`，对缺失买档填价格 `-9999999999`、数量 `0`。 [LOBSTER 官方格式](https://php.lobsterdata.com/info/DataStructure.php)

```text
ask_price_1,ask_size_1,bid_price_1,bid_size_1,ask_price_2,ask_size_2,bid_price_2,bid_size_2,...
```

下面两对一档记录用于说明行语义。第一条新增 100 股卖单，第二条在该价位成交 40 股，卖一剩余 60 股。标准 CLI 的默认深度指标使用十档，正式导出应提供所用指标要求的档数。

```text
34200.001,1,101,100,100100,-1
34200.003,4,101,40,100100,-1
```

```text
100100,100,100000,80
100100,60,100000,80
```

### 目录和配对

`Simple_Loader` 接受 `real_data_path`、`gen_data_path` 和 `cond_data_path` 三个路径。使用 CLI 时，目录按 `<data_dir>/<model>/<stock>/<time_period>/[<model_version>/]` 组织，下面是一个窗口、一条生成样本的兼容命名示例。

```text
data_real/
  300750_2026-09-01_message_real_id_12.csv
  300750_2026-09-01_orderbook_real_id_12.csv
data_gen/
  300750_2026-09-01_message_real_id_12_gen_id_00.csv
  300750_2026-09-01_orderbook_real_id_12_gen_id_00.csv
data_cond/
  300750_2026-09-01_message_real_id_12.csv
  300750_2026-09-01_orderbook_real_id_12.csv
```

- 文件名的第二个下划线分段是日期。加载器以日期和 `real_id` 查找生成和条件文件，每条生成轨迹用 `gen_id` 标识。历史消息和历史盘口分别按文件名排序后配对，生成文件也分别排序。消息和盘口宜使用相同的命名前缀与后缀，仅交换 `message` 和 `orderbook`，并在评估前核对配对。
- 条件目录里的文件可省略。标准分布评估需要显式提供真实和生成两侧的盘口文件。该版本的加载器直接读盘口 CSV，自动从消息重建盘口的逻辑未实现。
- 数据结构允许一个 `real_id` 下有多个 `gen_id`。该版本的延迟读取函数在循环中共享文件路径变量，多样本读取会重复返回最后一个生成文件。原样使用时可先用每窗口一条生成轨迹核对接口。正式评估多次采样需先验证每个 `gen_id` 都读到了对应的消息和盘口。

[官方加载器实现](https://github.com/peernagy/lob_bench/blob/276baf3039bf7630b555891b9d04ad49a92b3356/data_loading.py#L279-L376) · [CLI 目录规则](https://github.com/peernagy/lob_bench/blob/276baf3039bf7630b555891b9d04ad49a92b3356/run_bench.py#L183-L202)

## 3. 评估输出

| 输出 | 计算方式 |
|---|---|
| 无条件分布距离 | 分别从真实与生成轨迹计算价差、最优档和多档深度、事件到达间隔、撤单等待时间、挂撤单档位、成交量及订单流失衡（OFI）等统计量。总变差距离用分箱后的频率计算，一阶 Wasserstein 距离用真实与生成样本合并标准化后的取值计算。数值越低，所比较的分布越接近。 |
| 条件分布距离 | 按条件量分组，比较每组的统计量分布再加权汇总，例如按价差比较卖一挂单量，或按日内时段、波动率比较价差。这里的分组条件来自参与评估的真实和生成轨迹。 |
| 随生成长度变化的误差 | 按生成消息位置划分区间，报告各区间统计量的分布距离，观察模型自由生成时误差如何累积。官方 CLI 默认用总变差距离，区间长度以消息条数计。 |
| 事件级价格响应 | 论文另比较事件后的带符号中间价响应曲线，按事件类别和是否改变中间价分别报告。该分析由单独的价格响应模块提供。 |

Python 评分接口返回 `(scores, score_dfs, plot_fns)`。CLI 将前两项写入 `scores/` 下的 `scores_uncond_*.pkl`、`scores_cond_*.pkl`、`scores_div_*.pkl`，文件内容为 gzip 压缩的 pickle 二元组。 [LOB‑Bench §4–6](https://arxiv.org/html/2502.09172v2#S4) · [距离指标实现](https://github.com/peernagy/lob_bench/blob/276baf3039bf7630b555891b9d04ad49a92b3356/metrics.py#L55-L116) · [评分接口](https://github.com/peernagy/lob_bench/blob/276baf3039bf7630b555891b9d04ad49a92b3356/scoring.py#L300-L353) · [CLI 输出](https://github.com/peernagy/lob_bench/blob/276baf3039bf7630b555891b9d04ad49a92b3356/run_bench.py#L204-L251)

## 4. Stochopia 接入时要确认的口径

- **行情和逐笔的对齐。** 深市的 `行情` 是周期性十档快照。可以用它核对逐笔回放，再导出每条消息处理后的盘口。连续事件即使有相同毫秒时间也要保留已经校验过的交易所消息顺序。导出 `time` 时把 `HHMMSSmmm` 转为午夜以来的秒数。 [A 股 Level 2 数据说明](https://huggingface.co/datasets/venvoo/china-a-share-l2-level2-limit-order-book-tick-data)
- **撮合时钟。** 回放发送给 JAX-LOB 的秒、纳秒以首事件为零点，保留真实时间间隔。CSV 保留输入时钟。同价同刻的挂单按原始消息顺序成交。JAX-LOB 0.19 在优先级选择中将 `999999999` 用作掩码值。适配器在撮合前拒绝相对秒数到达该值的轨迹，并在纳秒分量等于该值时，用当前挂单的实际价格、时间和入队顺序选择订单，再交给 JAX-LOB 撮合。 [JAX-LOB 撮合实现](https://github.com/KangOxford/AlphaTrade/blob/jaxV3/gymnax_exchange/jaxob/JaxOrderBookArrays.py)
- **成交和撤单语义。** 深市档案在逐笔成交流里编码撤单，导出时按剩余量区分部分撤单和整单删除。可见成交的 `order_id` 和 `direction` 采用受影响的挂单口径。应明确回放层与生成器中谁产生撮合成交，确保同一次成交只扣一次盘口量。价格单位在同一轮评估中一致，并让使用最小价格单位的指标采用对应市场的报价尺度。 [A 股 Level 2 数据说明](https://huggingface.co/datasets/venvoo/china-a-share-l2-level2-limit-order-book-tick-data) · [LOBSTER 官方格式](https://php.lobsterdata.com/info/DataStructure.php)
- **大单是独立的干预实验。** 复用同样的事件和盘口输出，在固定历史与初始盘口上分别生成注单和无注单轨迹，测执行期间和停手后的响应。LOB‑Bench 的事件级统计用于历史与生成轨迹的对照。主动注入大单的流程可参考后续的 LOB‑Impact。 [LOB‑Impact 注单协议](https://arxiv.org/html/2610.27100v1#S2.SS1)
