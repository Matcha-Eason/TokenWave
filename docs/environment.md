# 环境与依赖

建议使用 Python 3.10+，优先 Python 3.11 或 3.12。服务器上建议新建虚拟环境，避免和系统 Python 混在一起。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果只想跑已有的 parquet 抽取与可视化，最低依赖是：

```text
numpy
pandas
pyarrow
matplotlib
plotly
modelscope
```

如果要继续做“多通道波形基学习、NMF、前缀预测、可预测性验证”，还需要：

```text
scipy
scikit-learn
```

## 依赖用途

| 包 | 用途 |
|---|---|
| `numpy` | 数值计算、重采样、归一化、FFT |
| `pandas` | 表格处理、CSV/parquet 读写的 DataFrame 层 |
| `pyarrow` | 读写大规模 parquet，处理全量逐轮数据必需 |
| `matplotlib` | 生成静态 PNG 图 |
| `plotly` | 生成交互式 HTML 可视化 |
| `modelscope` | 从 ModelScope 下载 CoderForge 数据 |
| `scipy` | 后续波形处理、插值、距离/优化等实验 |
| `scikit-learn` | 后续 NMF、聚类、回归/分类基线模型 |

## 当前脚本对应关系

| 脚本 | 需要的主要依赖 |
|---|---|
| `scripts/run_coderforge_pipeline.sh` | `modelscope` 加下面两个 Python 脚本的依赖 |
| `scripts/build_full_timeseries.py` | `numpy`、`pandas`、`pyarrow`、`matplotlib`、`plotly` |
| `scripts/extract_turn_timeseries.py` | `numpy`、`pandas`、`matplotlib`、`plotly` |
| `scripts/analyze_timeseries_results.py` | `numpy`、`pandas`、`pyarrow` |

## 后续建模实验建议

下一步如果要完成“三通道清洗 + 64 点重采样 + NMF 波形基 + 前缀预测验证”，建议直接基于这个环境继续加脚本，不需要额外安装深度学习框架。第一版用 `scikit-learn` 足够。
