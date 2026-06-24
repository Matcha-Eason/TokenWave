# Results Index

本仓库的 `results/` 目录只保存轻量结果快照，方便在 GitHub 上直接阅读。全量数据、中间 parquet、训练样本 parquet 和 smoke run 输出不提交。

## Waveform Basis Experiment

目录：

```text
results/waveform_basis_experiment/
```

内容：

| 文件 | 说明 |
|---|---|
| `experiment_report.md` | NMF 波形基实验自动报告 |
| `experiment_summary.json` | 机器可读摘要 |
| `nmf_reconstruction_metrics.csv` | NMF 重构误差 |
| `prefix_future_peak_metrics.csv` | 前缀未来峰预测指标 |
| `channel_scaling.csv` | 三通道缩放参数 |
| `nmf_basis.npy` | 学到的 NMF 波形基 |
| `nmf_waveform_bases.png` | 波形基可视化 |
| `prefix_50_reconstruction_examples.png` | 50% 前缀重构示例 |

## Online Dialogue State Prediction

目录：

```text
results/pet_state_experiment/
```

说明：该阶段名称沿用早期脚本里的 `pet_state`，但在研究语境中对应“在线对话状态预测”。

| 文件 | 说明 |
|---|---|
| `pet_state_report.md` | 在线状态预测报告 |
| `pet_state_summary.json` | 机器可读摘要 |
| `pet_state_metrics.csv` | 主要预测指标 |
| `pet_state_label_distribution.csv` | 状态标签分布 |
| `pet_state_classification_report.csv` | 分类详细指标 |
| `pet_state_confusion_matrix.csv` | 混淆矩阵 |
| `feature_importance_pet_state.csv` | 状态分类特征重要性 |
| `pet_state_experiment_overview.png` | 概览图 |

## Prefix Continuation And ML

目录：

```text
results/prefix_pet_target_experiment/
```

| 文件 | 说明 |
|---|---|
| `prefix_pet_target_report.md` | 前缀续写与 ML 增强报告 |
| `prefix_pet_target_summary.json` | 机器可读摘要 |
| `prefix_continuation_metrics.csv` | 纯 NMF 前缀续写指标 |
| `prefix_ml_metrics.csv` | 统计特征、波形特征、增强特征对比 |
| `pet_state_classification_report.csv` | 固定前缀状态分类报告 |
| `feature_importance_future_long_output.csv` | 长输出目标特征重要性 |
| `feature_importance_future_overheat.csv` | 红温目标特征重要性 |
| `feature_importance_future_peak_any.csv` | 未来峰目标特征重要性 |
| `prefix_pet_target_overview.png` | 概览图 |

## Current Waiting Experience

目录：

```text
results/current_waiting_experience_experiment/
```

这是目前最接近实际在线应用的实验。

| 文件 | 说明 |
|---|---|
| `current_waiting_experience_report.md` | 当前等待体验报告 |
| `current_waiting_experience_summary.json` | 机器可读摘要 |
| `current_waiting_experience_metrics.csv` | 主要指标 |
| `current_wait_state_distribution.csv` | 当前状态标签分布 |
| `history_only_classification_report.csv` | 只看历史统计的分类报告 |
| `history_plus_stream_classification_report.csv` | 历史 + 当前流式前缀的分类报告 |
| `feature_importance_history_only_state.csv` | 只看历史统计的特征重要性 |
| `feature_importance_history_plus_stream_state.csv` | 历史 + 流式前缀的特征重要性 |
| `current_waiting_experience_overview.png` | 概览图 |
