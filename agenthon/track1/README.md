# Agenthon 2026 Track 1：最小 agent 与本地评测 harness

Track 1 的提交是一个 Docker 镜像：`<image> solve --task-dir /input --out /app/output`，
运行时只能调用主办方的 House 模型（`nvidia/nemotron-3-super-120b-a12b`），每题 25 次请求、
每次最多 4000 输出 token。本目录包含：

| 路径 | 内容 |
|---|---|
| `t1agent/` | agent 本体（纯 Python，只依赖 httpx + 科学计算栈） |
| `harness/` | 本地评测：批量运行、用题目自带 checks 判分、失败分类、汇总报告 |
| `harness/splits.json` | 固定的 dev（59 题）/ holdout（27 题）切分 |
| `tests/` | 单元测试（不需要模型、不需要 root） |
| `Dockerfile`, `requirements.txt` | 提交镜像 |

## agent 流程（`t1agent/agent.py`）

```
准备（0 次调用）  重放题目镜像的 COPY 行，把 /app/data/x.csv 等路径软链接到 /input 下的数据；
                 /app/output 与 /output 指向同一目录；去掉 canary 行（模型永远看不到 GUID）；
                 从题面抽取交付文件、JSON 顶层键、表格列（对公开题的召回：352/353）
① plan    1 次   开 thinking，输出方法、约定、陷阱和交付契约（JSON）
② code    1 次   关 thinking，写完整脚本
③ run+check      子进程运行脚本；检查文件是否存在、可解析、缺键缺列、NaN/Inf、canary、
                 不得写 reward.json；每次运行后给输出拍快照
⑤ repair  n 次   带着报错和检查结果重写脚本，直到检查通过、请求用完或时间将尽
   review ≤1 次  检查通过但有 warning（如 NaN）时，让模型复核一次
结束             恢复"错误最少、其次 warning 最少"的那次快照
```

日志写在 `T1A_LOG_DIR`（默认 `/tmp/t1agent`，不写进输出目录）：`llm_calls.jsonl`、
`agent_stats.json`、`attempts/`（每次的脚本和输出快照）。

可调环境变量（均可选）：`T1A_MAX_REQUESTS`（默认 25）、`T1A_PLAN_THINKING`（1）、
`T1A_CODE_THINKING`（0）、`T1A_TEMPERATURE`（0）、`T1A_SEED`（0）、`T1A_RESERVE_S`（90）、
`T1A_MAX_SCRIPT_S`（900）、`T1A_REQUEST_TIMEOUT_S`（600）。

## 本地评测

准备一个 Python 3.13 环境，装好 `requirements.txt` 里的包（ta-lib 需要 C 库，本地可以跳过，
公开题都没用到）和 `pytest pytest-json-report pytest-timeout`。

```bash
cd agenthon/track1
export UNITS=/path/to/track1-coding-public/units
export MODEL_ENDPOINT=https://<origin>   # 不带 /v1，agent 会拼 /v1/chat/completions
export MODEL_NAME=<model id>
export MODEL_TOKEN=<key>

# ns 后端：Linux + root，不需要 Docker 守护进程（用 unshare 做挂载隔离）
sudo -E python -m harness.run_local --units-dir $UNITS --split dev --jobs 4 --name baseline \
     --python $(which python)

# docker 后端：先按官方 README 构建 finance-bench-sandbox:latest，再构建本镜像
docker build --platform linux/amd64 -t t1agent:dev .
python -m harness.run_local --units-dir $UNITS --split dev --backend docker --image t1agent:dev
```

结果在 `runs/<name>/`：每题一个目录（`input/` 不含 checks、`output/` 是 agent 原始输出、
`agent_logs/`、`check/` 是判分副本、`result.json`），以及 `results.jsonl` 和 `summary.md`
（pass@1、失败类别、按类别/难度分组、每题第一个失败的检查）。已完成的题默认跳过，
`--force` 重跑，`--units a,b` 只跑指定题。

失败类别：`agent_timeout`、`agent_crash`、`llm_error`、`no_code`、`no_output`、
`checker_error`、`missing_or_schema`、`wrong_values`。

**holdout 的用法**：只用来估计泛化能力。调 prompt 时不要看 holdout 的失败原因，
也不要把 holdout 题当 few-shot 示例；否则它就不再能代表私有测试集。

不接模型也能验证整条链路：

```bash
python -m harness.mock_llm --port 8765 &
MODEL_ENDPOINT=http://127.0.0.1:8765 MODEL_TOKEN=x T1A_MAX_REQUESTS=5 \
  sudo -E python -m harness.run_local --units-dir $UNITS \
  --units t1-EXAMPLE-bs-greeks-pde,t1-zero-coupon-bootstrapping --name smoke
# 预期：示例题 passed（mock 返回正确解法），另一题 missing_or_schema
```

## ns 后端与真实评测的差异

- 网络没有断开（agent 需要访问模型），生成的代码在本地理论上能联网，正式评测不能。
- 判分在 ns 里复现题目镜像的数据路径（很多 checks 直接读 `/app/data/...`），
  docker 后端则直接构建题目镜像来判分。

## 提交

```bash
docker build --platform linux/amd64 -t <registry>/<repo>:<tag> . && docker push <registry>/<repo>:<tag>
# 按 Agenthon2026-public 的 SUBMISSION-DESCRIPTOR 写 submission.json（category = "api"，填镜像 digest、
# models[] 的版本和 training_cutoff），然后：
qfbench2 submission pack --descriptor submission.json --team-number <N> --out submission.zip
```

## 已知限制 / 下一步

- 没有函数库、没有 few-shot 示例、没有按类别的不变量检查，这些是下一步的主要提分点。
- thinking token 是否计入 4000 输出上限、单次调用延迟、推理是否可复现，都需要用真实的
  House 模型实测后再调整默认值。
