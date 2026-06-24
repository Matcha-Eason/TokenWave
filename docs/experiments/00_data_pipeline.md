# CoderForge 全量分析服务器流程

这个流程会从 ModelScope 下载 `togethercomputer/CoderForge-Preview` 的 `trajectories-tokenized_qwencoder` 目录，抽取逐回合时间序列指标，并生成 parquet、汇总表和可视化。

## 一键运行

把本项目上传到服务器后，在项目根目录执行：

```bash
chmod +x scripts/run_coderforge_pipeline.sh
./scripts/run_coderforge_pipeline.sh
```

脚本会自动：

1. 创建 `.venv`
2. 安装 `requirements.txt` 中的 Python 依赖
3. 下载 ModelScope 数据
4. 扫描 parquet 分片并抽取指标
5. 输出可视化和汇总结果

如果你想手动安装环境，见 `ENVIRONMENT.md`。

## 输出

默认输出目录：

```text
outputs/coderforge_full/
```

主要文件：

- `turn_timeseries.parquet`：全量逐回合指标，每个 assistant 回合一行。
- `source_file_summary.csv`：每个 parquet 分片的处理摘要。
- `aggregate_envelope.csv`：按归一化进度聚合后的全量包络数据。
- `summary.json`：总处理摘要。
- `turn_timeseries_visualization.html`：最长样本轨迹的交互式图。
- `longest_trajectory_timeseries.png`：最长样本轨迹静态图。
- `aggregate_envelope.png`：全量聚合包络图。

默认不写 CSV 版全量明细，避免文件过大。需要时：

```bash
WRITE_CSV=1 ./scripts/run_coderforge_pipeline.sh
```

## 常用参数

只试跑前 3 个 parquet：

```bash
MAX_FILES=3 ./scripts/run_coderforge_pipeline.sh
```

已有数据，只跑分析：

```bash
SKIP_DOWNLOAD=1 ./scripts/run_coderforge_pipeline.sh
```

指定已有数据目录：

```bash
DATA_DIR=/data/CoderForge-Preview/trajectories-tokenized_qwencoder \
SKIP_DOWNLOAD=1 \
./scripts/run_coderforge_pipeline.sh
```

指定下载根目录，脚本会把 `trajectories-tokenized_qwencoder` 下载到这个目录下面：

```bash
DOWNLOAD_ROOT=/data/CoderForge-Preview ./scripts/run_coderforge_pipeline.sh
```

指定输出目录：

```bash
OUT_DIR=/data/coderforge_timeseries ./scripts/run_coderforge_pipeline.sh
```

提高或降低下载并发：

```bash
MODEL_SCOPE_MAX_WORKERS=16 ./scripts/run_coderforge_pipeline.sh
```

如果服务器访问 ModelScope 需要鉴权：

```bash
MODELSCOPE_TOKEN=你的token ./scripts/run_coderforge_pipeline.sh
```

## 指标说明

- `user_tokens_est`：前一条 user 消息的文本 token 估算。注意 OpenHands 轨迹中 user 消息经常是工具 observation。
- `assistant_tokens`：由源 parquet 的 `labels != -100` 连续 span 计算，优先作为 assistant 输出 token。
- `thinking_tokens_est`：`function=think` 中 `thought` 参数的文本 token 估算。
- `tool_call_count`：assistant 回合内全部 tool call 数。
- `external_tool_call_count`：排除 `think`、`finish` 等内部工具后的外部工具调用数。
- `edit_count`：`str_replace_editor` 的 `create`、`str_replace`、`insert`、`undo_edit` 次数。
- `test_count`：bash 命令中疑似测试运行的次数。
- `error_feedback_count`：下一条工具响应中错误、失败、traceback 等关键词命中次数。
- `duration_seconds`：源 parquet 没有时间戳，暂为空。
- `macro_load_index`：第一版宏观负载指标，用 assistant token、thinking token、工具、编辑、测试和错误反馈加权得到。

## 直接运行 Python

如果已经下载好了 parquet，也可以跳过 shell：

```bash
python scripts/build_full_timeseries.py \
  --input data/CoderForge-Preview/trajectories-tokenized_qwencoder \
  --out-dir outputs/coderforge_full
```
