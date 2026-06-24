# 逐轮时间序列抽取

## 目的

本阶段把 CoderForge 轨迹整理成“每个 assistant turn 一行”的时间序列。后续所有波形基学习、前缀预测、当前等待体验实验都依赖这张表。

## 数据来源

主要数据来自 ModelScope 上的 CoderForge Preview tokenized trajectories：

```text
togethercomputer/CoderForge-Preview
trajectories-tokenized_qwencoder
```

本仓库不直接提交原始 parquet 和全量中间表，因为体积较大。下载和批处理流程见：

```text
docs/experiments/00_data_pipeline.md
scripts/run_coderforge_pipeline.sh
scripts/build_full_timeseries.py
```

## 输出表

正式流程会生成：

```text
outputs/coderforge_full/turn_timeseries.parquet
```

核心字段：

| 字段 | 中文名 | 说明 |
|---|---|---|
| `source_file` | 来源文件 | 原始 parquet 分片 |
| `trajectory_id` | 轨迹 ID | 一个任务求解轨迹 |
| `turn_index` | 回合序号 | assistant turn 的位置 |
| `assistant_tokens` | 模型输出长度 | 来自 tokenized label span，较接近真实训练 token |
| `thinking_tokens_est` | 思考规划长度估计 | 从文本结构中估算 |
| `user_tokens_est` | 环境反馈长度估计 | 用户消息、工具返回、报错上下文的长度估计 |
| `edit_count` | 编辑次数 | 代码修改事件数 |
| `test_count` | 测试次数 | 测试/验证事件数 |
| `error_feedback_count` | 错误反馈次数 | 报错、失败、异常等反馈 |
| `finish_call_count` | 结束事件次数 | 结束/提交类事件 |
| `reward` | 轨迹结果 | 原数据中的任务完成结果 |

## 研究信号

后续研究把主信号收敛为三条：

1. 模型输出长度：`assistant_tokens`
2. 思考规划长度：`thinking_tokens_est`
3. 环境反馈长度：`user_tokens_est`

编辑、测试、错误、结束等不作为主波形通道，而是作为事件标记或机器学习增强特征。

## 主要限制

1. CoderForge 没有真实流式时间戳，因此“当前正在输出”的实验只能用 token 前缀模拟。
2. `thinking_tokens_est` 和 `user_tokens_est` 是估计值，不是模型服务端真实计数。
3. 工具调用计数在当前数据里信息量较低，前期检查发现很多轨迹近似每轮固定，因此没有作为主信号。
4. 数据来自编码任务轨迹，不等价于真实 Codex 桌面使用日志；应用到桌宠前需要重新校准阈值。

## 复现入口

服务器全量流程：

```bash
nohup ./scripts/run_coderforge_pipeline.sh > coderforge_pipeline.log 2>&1 &
```

单文件/局部调试可直接运行：

```bash
python scripts/extract_turn_timeseries.py \
  --input path/to/source.parquet \
  --out-dir outputs/local_timeseries
```

全量聚合分析：

```bash
python scripts/build_full_timeseries.py \
  --input-dir data/coderforge/trajectories-tokenized_qwencoder \
  --out-dir outputs/coderforge_full
```
