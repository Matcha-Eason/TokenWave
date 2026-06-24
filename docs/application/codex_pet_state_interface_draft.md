# Codex 事件到桌宠状态接口草案

## 目标

桌宠状态反映的是“用户可感知的等待体验”，不是模型内部真实算力消耗，也不是精确剩余轮数。

因此第一版接口关注三个问题：

1. 当前这轮是否已经让用户感觉“等得久”。
2. 当前状态是否应该显示为读题、稳定工作、深度输出、红温调试或收束。
3. 状态如何在流式 token、工具事件和错误反馈到来时平滑更新，避免频繁跳变。

## 实验依据

当前等待体验实验见：

```text
scripts/current_waiting_experience_experiment.py
outputs/current_waiting_experience_experiment/current_waiting_experience_report.md
```

正式实验使用 180,000 个模拟流式前缀样本，来自 3,000 条轨迹。核心结果：

| 输入信息 | 当前状态 accuracy | 当前状态宏平均 F1 |
|---|---:|---:|
| 规则状态机 | 0.689 | 0.547 |
| 只看历史统计 | 0.828 | 0.785 |
| 历史统计 + 当前流式前缀 | 0.937 | 0.904 |

按流式前缀看，“历史统计 + 当前流式前缀”的状态宏平均 F1：

| 当前轮已流出比例 | 宏平均 F1 |
|---:|---:|
| 0% | 0.818 |
| 25% | 0.921 |
| 50% | 0.930 |
| 75% | 0.924 |
| 100% | 0.924 |

结论：

- 只用历史统计已经能给出可用状态，尤其适合识别红温调试这类有上下文惯性的状态。
- 当前流式 token 是桌宠体验的关键输入；25% 前缀时已经能显著提升当前状态识别。
- 桌宠第一版应以规则状态机作为稳定 fallback，以离线模型或学习出的阈值作为校准层。

## 输入事件

Codex 插件侧建议把事件统一成 append-only event stream。每个事件都带 `session_id`、`turn_id`、`event_id`、`timestamp_ms`。

| 事件 | 含义 | 主要字段 |
|---|---|---|
| `user_message` | 用户发起或继续任务 | `text_chars`, `token_estimate` |
| `assistant_start` | assistant 开始本轮响应 | `mode`, `goal_mode`, `model` |
| `assistant_token_delta` | assistant 普通输出流式增量 | `delta_chars`, `delta_tokens_est` |
| `thinking_delta` | 思考/规划内容增量，如果可见或可计数 | `delta_chars`, `delta_tokens_est` |
| `tool_call_start` | 工具调用开始 | `tool_name`, `call_kind` |
| `tool_call_end` | 工具调用结束 | `tool_name`, `success`, `duration_ms`, `output_tokens_est` |
| `file_edit` | 文件编辑事件 | `path`, `edit_kind`, `changed_lines_est` |
| `test_run_start` | 测试/验证开始 | `command_kind` |
| `test_run_end` | 测试/验证结束 | `success`, `duration_ms`, `output_tokens_est`, `failure_count_est` |
| `error_feedback` | 明确错误反馈、异常、失败测试、lint 错误 | `error_kind`, `severity`, `token_estimate` |
| `assistant_end` | assistant 本轮结束 | `finish_reason`, `total_tokens_est` |
| `task_end_signal` | 明确收束信号 | `source`, `confidence` |

如果真实 Codex 事件无法提供 thinking token，可以把 `thinking_delta` 置空，并用 assistant 首 token 延迟、工具前停顿、计划性文本和 goal mode 作为替代特征。

## 在线特征

在线特征按三个层次维护。

### 1. 历史累计特征

这些特征只使用当前 turn 之前的信息：

| 中文名 | 字段建议 | 说明 |
|---|---|---|
| 累计模型输出 | `history.assistant_tokens_sum` | 历史输出规模 |
| 累计思考规划 | `history.thinking_tokens_sum` | 历史规划压力 |
| 累计环境反馈 | `history.feedback_tokens_sum` | 用户/工具/错误上下文规模 |
| 累计编辑次数 | `history.edit_count_sum` | 实际推进量 |
| 累计测试次数 | `history.test_count_sum` | 验证强度 |
| 累计错误反馈 | `history.error_count_sum` | 红温压力惯性 |
| 当前轮序号 | `history.turn_index` | 粗略任务阶段 |

### 2. 近期窗口特征

默认窗口为最近 5 轮，可配置为 3-8 轮：

| 中文名 | 字段建议 | 说明 |
|---|---|---|
| 近期错误密度 | `recent.error_count_sum` | 红温调试最强信号之一 |
| 近期测试密度 | `recent.test_count_sum` | 调试/验证压力 |
| 近期编辑密度 | `recent.edit_count_sum` | 工作推进强度 |
| 近期输出峰值 | `recent.assistant_tokens_max` | 是否已进入长输出惯性 |
| 近期环境反馈峰值 | `recent.feedback_tokens_max` | 是否被大段日志/报错拖住 |
| 近期负载斜率 | `recent.visible_load_slope` | 是否正在升温或收束 |

### 3. 当前流式特征

这些特征在本轮 assistant 输出过程中持续更新：

| 中文名 | 字段建议 | 说明 |
|---|---|---|
| 当前已流出模型 token | `current.assistant_tokens_streamed` | 用户等待体验的核心信号 |
| 当前已流出思考 token | `current.thinking_tokens_streamed` | 认真思考/规划信号 |
| 当前可见负载 | `current.visible_load` | `assistant + thinking` |
| 当前环境反馈长度 | `current.feedback_tokens` | 本轮开始时已知的上下文压力 |
| 当前工具等待时长 | `current.tool_wait_ms` | 工具执行造成的等待 |
| 当前无输出等待时长 | `current.silent_wait_ms` | 用户感知强烈，尤其 goal mode |
| 当前输出速度 | `current.tokens_per_second` | 慢速长输出和快速短输出体验不同 |
| 当前错误压力 | `current.error_pressure` | 当前已知错误、失败测试、异常 |
| 当前相对历史输出 | `current.output_vs_recent_mean` | 是否显著超过最近平均 |

## 状态定义

第一版保留 5 个状态。

| 状态 | 字段 | 用户感知 |
|---|---|---|
| 读题理解 | `reading_understanding` | 刚开始，输出少，事件少，像是在读题或理解上下文 |
| 稳定工作 | `steady_work` | 有持续推进，但等待压力不高 |
| 深度输出 | `deep_output` | 当前输出/思考明显变长，用户会感到它正在认真产出 |
| 红温调试 | `overheat_debugging` | 错误、失败测试、重复验证、长日志造成明显压力 |
| 收束 | `closing` | 输出下降、错误消失、出现总结/完成/等待确认信号 |

## 状态优先级

同一时刻可能多个信号同时触发。建议按以下优先级裁决：

```text
红温调试 > 深度输出 > 收束 > 读题理解 > 稳定工作
```

理由：

- 红温调试对用户等待体验最强，应该覆盖普通长输出。
- 深度输出代表本轮正在变长，即使任务接近尾声，也应优先表现为认真输出。
- 收束需要更保守，避免模型刚开始总结但后面又进入调试。
- 读题理解只适合早期低负载阶段。
- 稳定工作是默认状态。

## 状态评分

第一版可以先用规则分数，后续用实验脚本训练出的模型替换或校准。

建议输出 0-1 的内部信号：

```text
output_load       = 当前已流出模型 token / 长输出阈值
thinking_load     = 当前已流出思考 token / 长思考阈值
feedback_load     = 当前环境反馈 token / 高环境反馈阈值
error_pressure    = 近期错误 + 当前错误 + 失败测试 + 大日志
tool_wait_load    = 当前工具等待时长 / 工具等待阈值
silent_wait_load  = 当前无输出等待时长 / 无输出等待阈值
closing_signal    = 完成信号 + 负载下降 + 无新增错误
```

推荐第一版状态规则：

```text
if error_pressure >= 0.75:
    overheat_debugging
elif max(output_load, thinking_load, silent_wait_load) >= 0.75:
    deep_output
elif closing_signal >= 0.70 and error_pressure < 0.35:
    closing
elif turn_index <= 2 and output_load < 0.35 and error_pressure < 0.25:
    reading_understanding
else:
    steady_work
```

对于 goal mode，建议提高 `silent_wait_load` 权重。因为 goal mode 下用户可能看到更长时间的“无输出等待”，这本身就是用户可感知体验。

## 状态平滑规则

桌宠前端不应直接逐事件切状态，需要平滑。

建议规则：

1. 每 500-1000 ms 更新一次状态分数。
2. 状态切换需要新状态连续胜出至少 2 个 tick，或新状态分数比当前状态高 0.20 以上。
3. `deep_output` 最小驻留时间 2 秒，避免长输出刚触发就消失。
4. `overheat_debugging` 最小驻留时间 4 秒，且只有当错误压力低于 0.45 持续 2 个 tick 才能退出。
5. `closing` 需要完成信号或连续负载下降，不应只凭当前输出短就进入。
6. 如果 3 秒内无任何事件，但 assistant 未结束，增加 `silent_wait_load`，逐步从稳定工作进入深度输出或红温调试。

## 输出 JSON

建议插件内部每次状态更新输出如下 JSON：

```json
{
  "schema_version": "codex_pet_state.v0",
  "session_id": "session_123",
  "turn_id": "turn_17",
  "timestamp_ms": 1782268560123,
  "state": "deep_output",
  "state_zh": "深度输出",
  "intensity": 0.78,
  "confidence": 0.84,
  "reason": "当前模型输出已明显超过近期平均，且仍在持续流出",
  "signals": {
    "output_load": 0.91,
    "thinking_load": 0.42,
    "feedback_load": 0.28,
    "error_pressure": 0.12,
    "tool_wait_load": 0.0,
    "silent_wait_load": 0.08,
    "closing_signal": 0.18
  },
  "online_features": {
    "history_turn_index": 17,
    "recent_error_count_sum": 0,
    "recent_test_count_sum": 1,
    "recent_assistant_tokens_max": 420,
    "current_assistant_tokens_streamed": 318,
    "current_thinking_tokens_streamed": 41,
    "current_feedback_tokens": 390,
    "current_tokens_per_second": 22.4,
    "current_silent_wait_ms": 350
  },
  "state_scores": {
    "reading_understanding": 0.03,
    "steady_work": 0.32,
    "deep_output": 0.84,
    "overheat_debugging": 0.15,
    "closing": 0.11
  },
  "transition_risk": {
    "deep_output": 0.84,
    "overheat_debugging": 0.15,
    "closing": 0.11
  },
  "smoothing": {
    "previous_state": "steady_work",
    "raw_state": "deep_output",
    "ticks_in_state": 3,
    "changed": true
  }
}
```

## 第一版落地建议

1. 先实现规则状态机和在线特征聚合，不急着部署机器学习模型。
2. 把本实验学习到的阈值作为初始值：长输出约 300 token，长思考约 120 token，高环境反馈约 1200 token。
3. 在真实 Codex 日志上采样后重新校准阈值，因为 CoderForge 轨迹和桌面 Codex 的事件分布不一定一致。
4. 机器学习模型先作为离线评估器，目标是学习更好的权重和状态边界。
5. 后续 skill 可以暴露两个能力：`update(event) -> state_json` 和 `reset(session_id)`。
