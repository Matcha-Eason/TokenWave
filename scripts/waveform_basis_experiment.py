#!/usr/bin/env python3
"""Learn non-negative waveform bases and test prefix predictability.

The experiment uses three main channels:
- assistant output energy: assistant token count
- planning energy: thinking token estimate
- environment feedback energy: previous user/tool-response token estimate

Event signals are kept for interpretation, but the first basis model learns from
the three token/length channels only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import NMF
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


MAIN_CHANNELS = [
    "assistant_tokens",
    "thinking_tokens_est",
    "user_tokens_est",
]
EVENT_CHANNELS = [
    "edit_count",
    "test_count",
    "error_feedback_count",
    "finish_call_count",
]
READ_COLUMNS = [
    "source_file",
    "trajectory_id",
    "reward",
    "turn_index",
    *MAIN_CHANNELS,
    *EVENT_CHANNELS,
]
CHANNEL_LABELS_ZH = {
    "assistant_tokens": "模型输出波",
    "thinking_tokens_est": "思考规划波",
    "user_tokens_est": "环境反馈波",
    "edit_count": "编辑事件",
    "test_count": "测试事件",
    "error_feedback_count": "错误反馈事件",
    "finish_call_count": "结束事件",
}
CHANNEL_LABELS_PLOT = {
    "assistant_tokens": "assistant output",
    "thinking_tokens_est": "planning",
    "user_tokens_est": "environment feedback",
}


@dataclass
class Scaling:
    channel_scales: np.ndarray

    def transform(self, tensor: np.ndarray) -> np.ndarray:
        out = np.log1p(np.maximum(tensor, 0.0))
        out = out / self.channel_scales[None, None, :]
        per_traj = np.percentile(out, 95, axis=1, keepdims=True)
        return out / np.maximum(per_traj, 1.0)


def resample_1d(values: np.ndarray, points: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.zeros(points, dtype=float)
    if len(values) == 1:
        return np.full(points, values[0], dtype=float)
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, points)
    return np.interp(x_new, x_old, values)


def build_tensor(
    input_path: Path,
    points: int,
    max_trajectories: int,
    min_turns: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    pf = pq.ParquetFile(input_path)
    rng = np.random.default_rng(seed)
    tensors: list[np.ndarray] = []
    event_tensors: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    seen = 0

    row_groups = list(range(pf.num_row_groups))
    rng.shuffle(row_groups)

    for row_group_index in row_groups:
        df = pf.read_row_group(row_group_index, columns=READ_COLUMNS).to_pandas()
        for (source_file, trajectory_id), group in df.groupby(
            ["source_file", "trajectory_id"], sort=False
        ):
            group = group.sort_values("turn_index")
            if len(group) < min_turns:
                continue
            channels = np.stack(
                [
                    resample_1d(group[channel].to_numpy(dtype=float), points)
                    for channel in MAIN_CHANNELS
                ],
                axis=1,
            )
            events = np.stack(
                [
                    resample_1d(group[channel].to_numpy(dtype=float), points)
                    for channel in EVENT_CHANNELS
                ],
                axis=1,
            )
            tensors.append(channels)
            event_tensors.append(events)
            rows.append(
                {
                    "source_file": source_file,
                    "trajectory_id": trajectory_id,
                    "trajectory_key": f"{source_file}::{trajectory_id}",
                    "reward": float(group["reward"].iloc[0]),
                    "turns": int(len(group)),
                    "success": bool(group["reward"].iloc[0] > 0),
                }
            )
            seen += 1
            if max_trajectories and seen >= max_trajectories:
                break
        if max_trajectories and seen >= max_trajectories:
            break

    if not tensors:
        raise SystemExit("No trajectories matched the filters.")
    return np.stack(tensors), np.stack(event_tensors), pd.DataFrame(rows)


def fit_scaling(train_tensor: np.ndarray) -> Scaling:
    logged = np.log1p(np.maximum(train_tensor, 0.0))
    scales = np.percentile(logged.reshape(-1, logged.shape[-1]), 75, axis=0)
    scales = np.maximum(scales, 1.0)
    return Scaling(channel_scales=scales)


def flatten(tensor: np.ndarray) -> np.ndarray:
    return tensor.reshape(tensor.shape[0], -1)


def fit_nmf(train_x: np.ndarray, components: int, seed: int) -> NMF:
    model = NMF(
        n_components=components,
        init="nndsvda",
        solver="mu",
        beta_loss="frobenius",
        max_iter=600,
        random_state=seed,
        alpha_W=0.0001,
        alpha_H=0.0001,
        l1_ratio=0.0,
    )
    model.fit(train_x)
    return model


def solve_weights_from_prefix(
    basis: np.ndarray,
    prefix_x: np.ndarray,
    points: int,
    channels: int,
    prefix_ratio: float,
    iterations: int = 250,
) -> np.ndarray:
    prefix_points = max(2, int(round(points * prefix_ratio)))
    mask = np.zeros(points * channels, dtype=bool)
    mask[: prefix_points * channels] = True
    basis_prefix = np.maximum(basis[:, mask], 1e-12)
    x_prefix = np.maximum(prefix_x[:, mask], 1e-12)

    rng = np.random.default_rng(17)
    weights = rng.random((prefix_x.shape[0], basis.shape[0])) + 0.1
    numerator_const = x_prefix @ basis_prefix.T
    denom_eps = 1e-10
    for _ in range(iterations):
        denom = (weights @ basis_prefix) @ basis_prefix.T + denom_eps
        weights *= numerator_const / denom
    return weights


def evaluate_reconstruction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    points: int,
    channels: int,
    prefix_ratio: float | None = None,
) -> dict[str, float]:
    if prefix_ratio is not None:
        start = max(2, int(round(points * prefix_ratio))) * channels
        true = y_true[:, start:]
        pred = y_pred[:, start:]
    else:
        true = y_true
        pred = y_pred
    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    var = float(np.var(true))
    return {
        "mse": float(mse),
        "mae": float(mae),
        "normalized_mse": float(mse / max(var, 1e-12)),
        "r2": float(r2_score(true, pred)),
    }


def prefix_mask(points: int, channels: int, prefix_ratio: float) -> np.ndarray:
    prefix_points = max(2, int(round(points * prefix_ratio)))
    mask = np.zeros(points * channels, dtype=bool)
    mask[: prefix_points * channels] = True
    return mask


def future_peak_labels(x: np.ndarray, points: int, channels: int, prefix_ratio: float) -> np.ndarray:
    tensor = x.reshape(x.shape[0], points, channels)
    start = max(2, int(round(points * prefix_ratio)))
    observed_peak = tensor[:, :start, :].max(axis=1)
    future_peak = tensor[:, start:, :].max(axis=1)
    return (future_peak >= 1.15 * np.maximum(observed_peak, 1e-6)).astype(int)


def future_peak_predictions(
    pred_x: np.ndarray, points: int, channels: int, prefix_ratio: float
) -> np.ndarray:
    tensor = pred_x.reshape(pred_x.shape[0], points, channels)
    start = max(2, int(round(points * prefix_ratio)))
    observed_peak = tensor[:, :start, :].max(axis=1)
    future_peak = tensor[:, start:, :].max(axis=1)
    return (future_peak >= 1.15 * np.maximum(observed_peak, 1e-6)).astype(int)


def peak_prevalence(y_true: np.ndarray) -> dict[str, float]:
    positive_rate = float((y_true == 1).mean())
    return {
        "positive_rate": positive_rate,
        "majority_accuracy": max(positive_rate, 1.0 - positive_rate),
    }


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # Report simple accuracy/F1 without depending on probability calibration.
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    tn = float(((y_true == 0) & (y_pred == 0)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1.0),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plot_bases(model: NMF, points: int, channels: int, out_dir: Path) -> None:
    basis = model.components_.reshape(model.n_components, points, channels)
    fig, axes = plt.subplots(model.n_components, 1, figsize=(12, 2.2 * model.n_components), sharex=True)
    if model.n_components == 1:
        axes = [axes]
    x = np.linspace(0, 1, points)
    for idx, ax in enumerate(axes):
        for channel_idx, channel in enumerate(MAIN_CHANNELS):
            ax.plot(x, basis[idx, :, channel_idx], label=CHANNEL_LABELS_PLOT[channel])
        ax.set_ylabel(f"basis {idx + 1}")
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("normalized progress")
    fig.suptitle("NMF non-negative waveform bases")
    fig.tight_layout()
    fig.savefig(out_dir / "nmf_waveform_bases.png", dpi=180)
    plt.close(fig)


def plot_reconstruction_examples(
    true_x: np.ndarray,
    pred_x: np.ndarray,
    points: int,
    channels: int,
    out_dir: Path,
    prefix_ratio: float,
    count: int = 6,
) -> None:
    true = true_x.reshape(true_x.shape[0], points, channels)
    pred = pred_x.reshape(pred_x.shape[0], points, channels)
    x = np.linspace(0, 1, points)
    chosen = np.linspace(0, true.shape[0] - 1, min(count, true.shape[0]), dtype=int)
    fig, axes = plt.subplots(len(chosen), channels, figsize=(14, 2.5 * len(chosen)), sharex=True)
    if len(chosen) == 1:
        axes = axes[None, :]
    split = prefix_ratio
    for row_idx, sample_idx in enumerate(chosen):
        for channel_idx, channel in enumerate(MAIN_CHANNELS):
            ax = axes[row_idx, channel_idx]
            ax.plot(x, true[sample_idx, :, channel_idx], label="真实", color="#1f77b4")
            ax.plot(x, pred[sample_idx, :, channel_idx], label="predicted", color="#d62728", alpha=0.8)
            ax.axvline(split, color="#666", linestyle="--", linewidth=1)
            ax.set_title(CHANNEL_LABELS_PLOT[channel] if row_idx == 0 else "")
            ax.grid(alpha=0.2)
            if row_idx == 0 and channel_idx == channels - 1:
                ax.legend(["true", "predicted"], loc="upper right")
    fig.suptitle(f"Prefix {int(prefix_ratio * 100)}% full-curve reconstruction examples")
    fig.tight_layout()
    fig.savefig(out_dir / f"prefix_{int(prefix_ratio * 100)}_reconstruction_examples.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/coderforge_full/turn_timeseries.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/waveform_basis_experiment"))
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--max-trajectories", type=int, default=30000)
    parser.add_argument("--min-turns", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_tensor, event_tensor, meta = build_tensor(
        args.input,
        points=args.points,
        max_trajectories=args.max_trajectories,
        min_turns=args.min_turns,
        seed=args.seed,
    )

    idx = np.arange(len(meta))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.25,
        random_state=args.seed,
        stratify=meta["success"] if meta["success"].nunique() == 2 else None,
    )
    scaling = fit_scaling(raw_tensor[train_idx])
    scaled_tensor = scaling.transform(raw_tensor)
    x = flatten(scaled_tensor)
    train_x = x[train_idx]
    test_x = x[test_idx]

    model = fit_nmf(train_x, args.components, args.seed)
    train_weights = model.transform(train_x)
    test_weights = model.transform(test_x)
    train_pred = model.inverse_transform(train_weights)
    test_pred = model.inverse_transform(test_weights)
    mean_template = train_x.mean(axis=0, keepdims=True)

    rows = []
    full_train = evaluate_reconstruction(train_x, train_pred, args.points, len(MAIN_CHANNELS))
    full_test = evaluate_reconstruction(test_x, test_pred, args.points, len(MAIN_CHANNELS))
    rows.append({"task": "full_reconstruction", "split": "train", "prefix_ratio": 1.0, **full_train})
    rows.append({"task": "full_reconstruction", "split": "test", "prefix_ratio": 1.0, **full_test})

    prefix_rows = []
    prefix_predictions: dict[float, np.ndarray] = {}
    for prefix_ratio in [0.25, 0.50, 0.75]:
        mask = prefix_mask(args.points, len(MAIN_CHANNELS), prefix_ratio)
        weights = solve_weights_from_prefix(
            model.components_,
            test_x,
            args.points,
            len(MAIN_CHANNELS),
            prefix_ratio,
        )
        pred = weights @ model.components_
        prefix_predictions[prefix_ratio] = pred
        metrics = evaluate_reconstruction(
            test_x,
            pred,
            args.points,
            len(MAIN_CHANNELS),
            prefix_ratio=prefix_ratio,
        )
        rows.append({"task": "future_curve_prediction_unsupervised_nmf", "split": "test", "prefix_ratio": prefix_ratio, **metrics})

        mean_pred = np.repeat(mean_template, len(test_x), axis=0)
        mean_metrics = evaluate_reconstruction(
            test_x,
            mean_pred,
            args.points,
            len(MAIN_CHANNELS),
            prefix_ratio=prefix_ratio,
        )
        rows.append({"task": "future_curve_prediction_mean_template", "split": "test", "prefix_ratio": prefix_ratio, **mean_metrics})

        ridge = Ridge(alpha=1.0)
        ridge.fit(train_x[:, mask], train_weights)
        ridge_weights = np.maximum(ridge.predict(test_x[:, mask]), 0.0)
        ridge_pred = ridge_weights @ model.components_
        ridge_metrics = evaluate_reconstruction(
            test_x,
            ridge_pred,
            args.points,
            len(MAIN_CHANNELS),
            prefix_ratio=prefix_ratio,
        )
        rows.append({"task": "future_curve_prediction_prefix_ridge_nmf", "split": "test", "prefix_ratio": prefix_ratio, **ridge_metrics})

        labels = future_peak_labels(test_x, args.points, len(MAIN_CHANNELS), prefix_ratio)
        preds = future_peak_predictions(pred, args.points, len(MAIN_CHANNELS), prefix_ratio)
        ridge_peak_preds = future_peak_predictions(ridge_pred, args.points, len(MAIN_CHANNELS), prefix_ratio)
        mean_peak_preds = future_peak_predictions(mean_pred, args.points, len(MAIN_CHANNELS), prefix_ratio)
        for channel_idx, channel in enumerate(MAIN_CHANNELS):
            label_channel = labels[:, channel_idx]
            prevalence = peak_prevalence(label_channel)
            score = classification_scores(label_channel, preds[:, channel_idx])
            prefix_rows.append(
                {
                    "model": "unsupervised_nmf_prefix_fit",
                    "prefix_ratio": prefix_ratio,
                    "channel": channel,
                    "channel_zh": CHANNEL_LABELS_ZH[channel],
                    **prevalence,
                    **score,
                }
            )
            ridge_score = classification_scores(label_channel, ridge_peak_preds[:, channel_idx])
            prefix_rows.append(
                {
                    "model": "prefix_ridge_to_nmf",
                    "prefix_ratio": prefix_ratio,
                    "channel": channel,
                    "channel_zh": CHANNEL_LABELS_ZH[channel],
                    **prevalence,
                    **ridge_score,
                }
            )
            mean_score = classification_scores(label_channel, mean_peak_preds[:, channel_idx])
            prefix_rows.append(
                {
                    "model": "mean_template",
                    "prefix_ratio": prefix_ratio,
                    "channel": channel,
                    "channel_zh": CHANNEL_LABELS_ZH[channel],
                    **prevalence,
                    **mean_score,
                }
            )

    metrics_df = pd.DataFrame(rows)
    peak_df = pd.DataFrame(prefix_rows)
    weights = model.transform(x)
    weight_cols = [f"basis_{i + 1}_weight" for i in range(args.components)]
    weight_df = pd.concat([meta.reset_index(drop=True), pd.DataFrame(weights, columns=weight_cols)], axis=1)
    basis = model.components_.reshape(args.components, args.points, len(MAIN_CHANNELS))

    metrics_df.to_csv(args.out_dir / "nmf_reconstruction_metrics.csv", index=False)
    peak_df.to_csv(args.out_dir / "prefix_future_peak_metrics.csv", index=False)
    weight_df.to_parquet(args.out_dir / "trajectory_basis_weights.parquet", index=False)
    np.save(args.out_dir / "nmf_basis.npy", basis)
    pd.DataFrame(
        {
            "channel": MAIN_CHANNELS,
            "channel_zh": [CHANNEL_LABELS_ZH[c] for c in MAIN_CHANNELS],
            "log1p_p75_scale": scaling.channel_scales,
        }
    ).to_csv(args.out_dir / "channel_scaling.csv", index=False)

    plot_bases(model, args.points, len(MAIN_CHANNELS), args.out_dir)
    plot_reconstruction_examples(
        test_x,
        prefix_predictions[0.50],
        args.points,
        len(MAIN_CHANNELS),
        args.out_dir,
        prefix_ratio=0.50,
    )

    summary = {
        "input": str(args.input),
        "trajectories_used": int(len(meta)),
        "train_trajectories": int(len(train_idx)),
        "test_trajectories": int(len(test_idx)),
        "points": args.points,
        "components": args.components,
        "main_channels": [
            {"field": channel, "zh": CHANNEL_LABELS_ZH[channel]}
            for channel in MAIN_CHANNELS
        ],
        "event_channels_not_in_basis": [
            {"field": channel, "zh": CHANNEL_LABELS_ZH[channel]}
            for channel in EVENT_CHANNELS
        ],
        "full_reconstruction_test_normalized_mse": float(full_test["normalized_mse"]),
        "full_reconstruction_test_r2": float(full_test["r2"]),
        "prefix_prediction": metrics_df[
            metrics_df["task"].str.startswith("future_curve_prediction")
        ].to_dict(orient="records"),
        "future_peak_prediction": peak_df.to_dict(orient="records"),
    }
    (args.out_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    def table_text(df: pd.DataFrame) -> str:
        return df.round(4).to_csv(index=False).strip()

    lines = [
        "# 多通道波形基实验结果",
        "",
        f"- 使用轨迹数：{len(meta):,}",
        f"- 训练/测试：{len(train_idx):,} / {len(test_idx):,}",
        f"- 重采样点数：{args.points}",
        f"- NMF 波形基数量：{args.components}",
        "",
        "## 完整曲线重构",
        table_text(metrics_df[metrics_df["task"] == "full_reconstruction"]),
        "",
        "## 前缀预测后续曲线",
        table_text(metrics_df[metrics_df["task"].str.startswith("future_curve_prediction")]),
        "",
        "## 未来高峰预测",
        table_text(peak_df),
        "",
        "## 初步解释",
        "- 如果完整曲线重构的测试集归一化均方误差显著小于 1，说明少量非负波形基能解释相当一部分轨迹形态。",
        "- 平均模板是朴素基线；无监督 NMF 前缀拟合检验纯数学模板能否续写；岭回归到 NMF 权重检验轻量机器学习增强是否带来预测提升。",
        "- 如果前缀预测后续曲线的归一化均方误差低于平均模板，说明当前前缀对未来波形有可预测性。",
        "- 如果未来高峰预测的 F1 高于朴素猜测，说明波形基对后续阶段变化有实际预警价值。",
    ]
    (args.out_dir / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
