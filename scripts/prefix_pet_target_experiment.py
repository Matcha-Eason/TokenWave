#!/usr/bin/env python3
"""Fixed-prefix waveform continuation and ML-enhanced pet target prediction.

This script completes steps 3 and 4 of the refined plan:
- Step 3: use 25/50/75% prefixes to fit NMF basis weights and continue the
  three-channel waveform.
- Step 4: feed basis weights, phase, slopes, and event densities into a
  gradient-boosting model for pet-facing targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None
    LGBMRegressor = None

from pet_state_experiment import LONG_OUTPUT_TOKEN_THRESHOLD, STATE_LABELS_ZH, state_label
from waveform_basis_experiment import (
    CHANNEL_LABELS_ZH,
    EVENT_CHANNELS,
    MAIN_CHANNELS,
    Scaling,
    fit_scaling,
    resample_1d,
    solve_weights_from_prefix,
)


READ_COLUMNS = [
    "source_file",
    "trajectory_id",
    "reward",
    "turn_index",
    *MAIN_CHANNELS,
    *EVENT_CHANNELS,
]

PREFIX_RATIOS = [0.25, 0.50, 0.75]
FUTURE_PEAK_MARGIN = 1.15


def load_scaling(path: Path | None, raw_tensor: np.ndarray) -> Scaling:
    if path and path.exists():
        df = pd.read_csv(path)
        scales = []
        for channel in MAIN_CHANNELS:
            matched = df[df["channel"] == channel]
            if matched.empty:
                raise SystemExit(f"Missing channel scale for {channel} in {path}")
            scales.append(float(matched["log1p_p75_scale"].iloc[0]))
        return Scaling(channel_scales=np.asarray(scales, dtype=float))
    return fit_scaling(raw_tensor)


def transform_with_prefix_scale(
    raw_tensor: np.ndarray,
    scaling: Scaling,
    prefix_points: int,
) -> np.ndarray:
    logged = np.log1p(np.maximum(raw_tensor, 0.0))
    scaled = logged / scaling.channel_scales[None, None, :]
    per_traj = np.percentile(scaled[:, :prefix_points, :], 95, axis=1, keepdims=True)
    return scaled / np.maximum(per_traj, 1.0)


def slope(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def safe_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        arr = np.zeros(1, dtype=float)
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_last": float(arr[-1]),
        f"{prefix}_sum": float(arr.sum()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_slope": slope(arr),
    }


def build_raw_dataset(
    input_path: Path,
    points: int,
    max_trajectories: int,
    min_turns: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[pd.DataFrame]]:
    pf = pq.ParquetFile(input_path)
    rng = np.random.default_rng(seed)
    row_groups = list(range(pf.num_row_groups))
    rng.shuffle(row_groups)

    raw_tensors: list[np.ndarray] = []
    event_tensors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    groups: list[pd.DataFrame] = []
    seen = 0
    for row_group_index in row_groups:
        df = pf.read_row_group(row_group_index, columns=READ_COLUMNS).to_pandas()
        for (source_file, trajectory_id), group in df.groupby(["source_file", "trajectory_id"], sort=False):
            group = group.sort_values("turn_index").reset_index(drop=True)
            if len(group) < min_turns:
                continue
            raw_tensors.append(
                np.stack(
                    [resample_1d(group[channel].to_numpy(dtype=float), points) for channel in MAIN_CHANNELS],
                    axis=1,
                )
            )
            event_tensors.append(
                np.stack(
                    [resample_1d(group[channel].to_numpy(dtype=float), points) for channel in EVENT_CHANNELS],
                    axis=1,
                )
            )
            key = f"{source_file}::{trajectory_id}"
            rows.append(
                {
                    "source_file": source_file,
                    "trajectory_id": trajectory_id,
                    "trajectory_key": key,
                    "reward": float(group["reward"].iloc[0]),
                    "turns": int(len(group)),
                    "success": bool(group["reward"].iloc[0] > 0),
                }
            )
            groups.append(group)
            seen += 1
            if max_trajectories and seen >= max_trajectories:
                break
        if max_trajectories and seen >= max_trajectories:
            break

    if not raw_tensors:
        raise SystemExit("No trajectories matched filters.")
    return np.stack(raw_tensors), np.stack(event_tensors), pd.DataFrame(rows), groups


def future_peak_label(scaled_tensor: np.ndarray, prefix_points: int) -> np.ndarray:
    observed_peak = scaled_tensor[:prefix_points, :].max(axis=0)
    future_peak = scaled_tensor[prefix_points:, :].max(axis=0)
    return future_peak >= FUTURE_PEAK_MARGIN * np.maximum(observed_peak, 1e-6)


def future_peak_prediction(pred_scaled_tensor: np.ndarray, prefix_points: int) -> np.ndarray:
    observed_peak = pred_scaled_tensor[:prefix_points, :].max(axis=0)
    future_peak = pred_scaled_tensor[prefix_points:, :].max(axis=0)
    return future_peak >= FUTURE_PEAK_MARGIN * np.maximum(observed_peak, 1e-6)


def future_overheat_label(future: pd.DataFrame) -> bool:
    return bool(
        future["error_feedback_count"].sum() >= 4
        or (future["test_count"].sum() >= 2 and future["error_feedback_count"].sum() >= 2)
    )


def future_normalized_mse(true_scaled: np.ndarray, pred_scaled: np.ndarray, prefix_points: int) -> float:
    true_future = true_scaled[prefix_points:, :]
    pred_future = pred_scaled[prefix_points:, :]
    mse = float(np.mean((true_future - pred_future) ** 2))
    variance = float(np.var(true_future))
    return mse / max(variance, 1e-12)


def feature_columns(df: pd.DataFrame, group_name: str) -> list[str]:
    exclude = {
        "trajectory_key",
        "trajectory_id",
        "source_file",
        "prefix_ratio",
        "prefix_turn",
        "total_turns",
        "remaining_turns",
        "remaining_turn_bucket",
        "future_end_within_5",
        "future_long_output",
        "future_overheat",
        "pet_state",
        "pet_state_zh",
        "future_peak_any",
        "future_peak_assistant",
        "future_peak_thinking",
        "future_peak_feedback",
    }
    if group_name == "stats_only":
        include_prefixes = ("phase_", "recent_", "cum_", "density_", "current_")
    elif group_name == "waveform_only":
        include_prefixes = ("nmf_", "reconstruction_", "predicted_future_", "phase_")
    elif group_name == "enhanced":
        include_prefixes = ("phase_", "recent_", "cum_", "density_", "current_", "nmf_", "reconstruction_", "predicted_future_")
    else:
        raise ValueError(f"Unknown feature group: {group_name}")

    cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.startswith(include_prefixes) and pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def make_classifier(seed: int):
    if LGBMClassifier is not None:
        return LGBMClassifier(
            n_estimators=220,
            learning_rate=0.045,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            verbose=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.055,
        l2_regularization=0.02,
        early_stopping=False,
        random_state=seed,
    )


def make_regressor(seed: int):
    if LGBMRegressor is not None:
        return LGBMRegressor(
            n_estimators=260,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            verbose=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=240,
        learning_rate=0.05,
        l2_regularization=0.02,
        early_stopping=False,
        random_state=seed,
    )


def binary_metrics(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if len(np.unique(y_true)) < 2:
        balanced = float(accuracy_score(y_true, pred))
        auc = float("nan")
    else:
        balanced = float(balanced_accuracy_score(y_true, pred))
        auc = float(roc_auc_score(y_true, proba))
    out = {
        "positive_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": balanced,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": auc,
    }
    return out


def save_importance(model, x_test: pd.DataFrame, y_test: np.ndarray, path: Path, scoring: str, seed: int) -> None:
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=2,
        random_state=seed,
        scoring=scoring,
        n_jobs=1,
    )
    pd.DataFrame(
        {
            "feature": x_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).to_csv(path, index=False)


def build_examples(
    meta: pd.DataFrame,
    groups: list[pd.DataFrame],
    raw_tensor: np.ndarray,
    event_tensor: np.ndarray,
    scaling: Scaling,
    basis: np.ndarray,
    prefix_ratios: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = raw_tensor.shape[1]
    channels = raw_tensor.shape[2]
    basis_flat = basis.reshape(basis.shape[0], -1)

    examples: list[dict[str, Any]] = []
    continuation_rows: list[dict[str, Any]] = []
    mean_template = transform_with_prefix_scale(raw_tensor, scaling, prefix_points=points).reshape(len(raw_tensor), -1).mean(axis=0)

    for idx, group in enumerate(groups):
        total_turns = len(group)
        meta_row = meta.iloc[idx]
        for prefix_ratio in prefix_ratios:
            prefix_points = max(2, int(round(points * prefix_ratio)))
            prefix_turn = min(total_turns - 2, max(0, int(np.ceil(total_turns * prefix_ratio)) - 1))
            prefix_df = group.iloc[: prefix_turn + 1]
            recent = group.iloc[max(0, prefix_turn - 5 + 1) : prefix_turn + 1]
            future = group.iloc[prefix_turn + 1 :]
            horizon3 = group.iloc[prefix_turn + 1 : min(total_turns, prefix_turn + 4)]
            horizon5 = group.iloc[prefix_turn + 1 : min(total_turns, prefix_turn + 6)]
            remaining_turns = total_turns - prefix_turn - 1

            scaled = transform_with_prefix_scale(raw_tensor[idx : idx + 1], scaling, prefix_points)[0]
            x = scaled.reshape(1, -1)
            weights = solve_weights_from_prefix(
                basis_flat,
                x,
                points,
                channels,
                prefix_ratio=prefix_ratio,
                iterations=160,
            )
            pred = (weights @ basis_flat).reshape(points, channels)
            mean_pred = mean_template.reshape(points, channels)

            nmf_future_mse = future_normalized_mse(scaled, pred, prefix_points)
            mean_future_mse = future_normalized_mse(scaled, mean_pred, prefix_points)
            true_peak = future_peak_label(scaled, prefix_points)
            pred_peak = future_peak_prediction(pred, prefix_points)
            mean_peak = future_peak_prediction(mean_pred, prefix_points)
            for channel_idx, channel in enumerate(MAIN_CHANNELS):
                continuation_rows.append(
                    {
                        "trajectory_key": meta_row["trajectory_key"],
                        "prefix_ratio": prefix_ratio,
                        "model": "unsupervised_nmf_prefix_fit",
                        "channel": channel,
                        "channel_zh": CHANNEL_LABELS_ZH[channel],
                        "future_curve_normalized_mse": nmf_future_mse,
                        "future_peak_true": bool(true_peak[channel_idx]),
                        "future_peak_pred": bool(pred_peak[channel_idx]),
                    }
                )
                continuation_rows.append(
                    {
                        "trajectory_key": meta_row["trajectory_key"],
                        "prefix_ratio": prefix_ratio,
                        "model": "mean_template",
                        "channel": channel,
                        "channel_zh": CHANNEL_LABELS_ZH[channel],
                        "future_curve_normalized_mse": mean_future_mse,
                        "future_peak_true": bool(true_peak[channel_idx]),
                        "future_peak_pred": bool(mean_peak[channel_idx]),
                    }
                )

            row: dict[str, Any] = {
                "trajectory_key": meta_row["trajectory_key"],
                "trajectory_id": meta_row["trajectory_id"],
                "source_file": meta_row["source_file"],
                "prefix_ratio": float(prefix_ratio),
                "prefix_turn": int(prefix_turn),
                "total_turns": int(total_turns),
                "remaining_turns": int(remaining_turns),
                "remaining_turn_bucket": "short" if remaining_turns <= 5 else "medium" if remaining_turns <= 20 else "long",
                "future_end_within_5": bool(remaining_turns <= 5),
                "future_long_output": bool((horizon3["assistant_tokens"] >= LONG_OUTPUT_TOKEN_THRESHOLD).any()),
                "future_overheat": future_overheat_label(horizon5),
                "future_peak_assistant": bool(true_peak[0]),
                "future_peak_thinking": bool(true_peak[1]),
                "future_peak_feedback": bool(true_peak[2]),
                "future_peak_any": bool(true_peak.any()),
                "pet_state": state_label(
                    current_turn=prefix_turn,
                    remaining_turns=remaining_turns,
                    current_assistant=float(group.iloc[prefix_turn]["assistant_tokens"]),
                    recent_assistant_max=float(recent["assistant_tokens"].max()),
                    recent_error_sum=float(recent["error_feedback_count"].sum()),
                    recent_test_sum=float(recent["test_count"].sum()),
                    recent_edit_sum=float(recent["edit_count"].sum()),
                ),
            }
            row["pet_state_zh"] = STATE_LABELS_ZH[row["pet_state"]]

            row.update(
                {
                    "phase_prefix_ratio": float(prefix_ratio),
                    "phase_observed_turns": float(prefix_turn + 1),
                    "current_assistant_tokens": float(group.iloc[prefix_turn]["assistant_tokens"]),
                }
            )
            for channel in [*MAIN_CHANNELS, "edit_count", "test_count", "error_feedback_count"]:
                row.update(safe_stats(recent[channel].to_numpy(dtype=float), f"recent_{channel}"))
                cumulative = prefix_df[channel].to_numpy(dtype=float)
                row[f"cum_{channel}_sum"] = float(cumulative.sum())
                row[f"density_{channel}_per_turn"] = float(cumulative.sum() / max(prefix_turn + 1, 1))

            prefix_true = scaled.reshape(1, -1)[:, : prefix_points * channels]
            prefix_pred = pred.reshape(1, -1)[:, : prefix_points * channels]
            row["reconstruction_prefix_mae"] = float(np.mean(np.abs(prefix_true - prefix_pred)))
            row["reconstruction_future_normalized_mse"] = float(nmf_future_mse)
            row["reconstruction_vs_mean_delta"] = float(mean_future_mse - nmf_future_mse)

            future_pred = pred[prefix_points:, :]
            for channel_idx, channel in enumerate(MAIN_CHANNELS):
                row[f"predicted_future_{channel}_max"] = float(future_pred[:, channel_idx].max()) if len(future_pred) else 0.0
                row[f"predicted_future_{channel}_sum"] = float(future_pred[:, channel_idx].sum())
                row[f"predicted_future_peak_{channel}"] = float(pred_peak[channel_idx])
            for basis_idx, value in enumerate(weights[0], start=1):
                row[f"nmf_basis_{basis_idx}_prefix_weight"] = float(value)

            examples.append(row)
    return pd.DataFrame(examples), pd.DataFrame(continuation_rows)


def continuation_summary(rows: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (prefix_ratio, model, channel, channel_zh), group in rows.groupby(["prefix_ratio", "model", "channel", "channel_zh"]):
        y_true = group["future_peak_true"].astype(bool).to_numpy()
        y_pred = group["future_peak_pred"].astype(bool).to_numpy()
        summaries.append(
            {
                "prefix_ratio": prefix_ratio,
                "model": model,
                "channel": channel,
                "channel_zh": channel_zh,
                "future_curve_normalized_mse_mean": float(group["future_curve_normalized_mse"].mean()),
                "future_peak_positive_rate": float(y_true.mean()),
                "future_peak_accuracy": float(accuracy_score(y_true, y_pred)),
                "future_peak_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "future_peak_f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return pd.DataFrame(summaries).sort_values(["prefix_ratio", "channel", "model"])


def run_ml(examples: pd.DataFrame, out_dir: Path, seed: int) -> pd.DataFrame:
    groups = examples["trajectory_key"]
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(examples, groups=groups))
    results: list[dict[str, Any]] = []

    binary_targets = [
        "future_end_within_5",
        "future_long_output",
        "future_overheat",
        "future_peak_any",
    ]
    feature_groups = ["stats_only", "waveform_only", "enhanced"]
    for feature_group in feature_groups:
        cols = feature_columns(examples, feature_group)
        x_train = examples.iloc[train_idx][cols].replace([np.inf, -np.inf], 0).fillna(0)
        x_test = examples.iloc[test_idx][cols].replace([np.inf, -np.inf], 0).fillna(0)
        for target in binary_targets:
            y_train = examples.iloc[train_idx][target].astype(int).to_numpy()
            y_test = examples.iloc[test_idx][target].astype(int).to_numpy()
            if len(np.unique(y_train)) < 2:
                constant = float(y_train[0])
                proba = np.full(len(y_test), constant, dtype=float)
            else:
                clf = make_classifier(seed)
                clf.fit(x_train, y_train)
                if hasattr(clf, "predict_proba"):
                    proba_matrix = clf.predict_proba(x_test)
                    proba = proba_matrix[:, 1] if proba_matrix.shape[1] > 1 else np.full(len(y_test), float(clf.classes_[0]))
                else:
                    proba = clf.predict(x_test)
            pred = proba >= 0.5
            results.append(
                {
                    "feature_group": feature_group,
                    "target": target,
                    "model_family": "lightgbm" if LGBMClassifier is not None else "sklearn_hist_gradient_boosting",
                    "features": len(cols),
                    **binary_metrics(y_test, proba, pred),
                }
            )
            if feature_group == "enhanced" and len(np.unique(y_train)) >= 2 and len(np.unique(y_test)) >= 2:
                save_importance(
                    clf,
                    x_test,
                    y_test,
                    out_dir / f"feature_importance_{target}.csv",
                    "balanced_accuracy",
                    seed,
                )

        y_train_state = examples.iloc[train_idx]["pet_state"].to_numpy()
        y_test_state = examples.iloc[test_idx]["pet_state"].to_numpy()
        state_clf = make_classifier(seed)
        state_clf.fit(x_train, y_train_state)
        state_pred = state_clf.predict(x_test)
        labels = sorted(set(y_test_state) | set(state_pred))
        results.append(
            {
                "feature_group": feature_group,
                "target": "pet_state_multiclass",
                "model_family": "lightgbm" if LGBMClassifier is not None else "sklearn_hist_gradient_boosting",
                "features": len(cols),
                "accuracy": float(accuracy_score(y_test_state, state_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test_state, state_pred)),
                "macro_f1": float(f1_score(y_test_state, state_pred, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_test_state, state_pred, average="weighted", zero_division=0)),
            }
        )
        if feature_group == "enhanced":
            pd.DataFrame(
                classification_report(
                    y_test_state,
                    state_pred,
                    labels=labels,
                    output_dict=True,
                    zero_division=0,
                )
            ).T.to_csv(out_dir / "pet_state_classification_report.csv")

        y_train_reg = np.log1p(examples.iloc[train_idx]["remaining_turns"].to_numpy(dtype=float))
        y_test_remaining = examples.iloc[test_idx]["remaining_turns"].to_numpy(dtype=float)
        reg = make_regressor(seed)
        reg.fit(x_train, y_train_reg)
        pred_remaining = np.expm1(reg.predict(x_test))
        results.append(
            {
                "feature_group": feature_group,
                "target": "remaining_turns_research",
                "model_family": "lightgbm" if LGBMRegressor is not None else "sklearn_hist_gradient_boosting",
                "features": len(cols),
                "mae_turns": float(mean_absolute_error(y_test_remaining, pred_remaining)),
                "median_abs_error_turns": float(median_absolute_error(y_test_remaining, pred_remaining)),
                "r2_log_remaining": float(r2_score(np.log1p(y_test_remaining), np.log1p(np.maximum(pred_remaining, 0)))),
            }
        )
        if feature_group == "enhanced":
            save_importance(
                reg,
                x_test,
                np.log1p(y_test_remaining),
                out_dir / "feature_importance_remaining_turns_research.csv",
                "neg_mean_absolute_error",
                seed,
            )

    return pd.DataFrame(results)


def plot_overview(continuation: pd.DataFrame, ml_metrics: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    cont = continuation[
        (continuation["model"] == "unsupervised_nmf_prefix_fit")
        & (continuation["channel"] == "assistant_tokens")
    ].sort_values("prefix_ratio")
    axes[0].plot(cont["prefix_ratio"], cont["future_curve_normalized_mse_mean"], marker="o", label="NMF prefix fit")
    mean = continuation[
        (continuation["model"] == "mean_template")
        & (continuation["channel"] == "assistant_tokens")
    ].sort_values("prefix_ratio")
    axes[0].plot(mean["prefix_ratio"], mean["future_curve_normalized_mse_mean"], marker="o", label="mean template")
    axes[0].set_title("Future assistant-wave continuation")
    axes[0].set_xlabel("prefix ratio")
    axes[0].set_ylabel("future normalized MSE")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    subset = ml_metrics[ml_metrics["target"].isin(["future_long_output", "future_overheat", "future_end_within_5"])]
    pivot = subset.pivot(index="target", columns="feature_group", values="auc")
    pivot[["stats_only", "waveform_only", "enhanced"]].plot.bar(ax=axes[1])
    axes[1].set_title("ML enhancement AUC")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "prefix_pet_target_overview.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/coderforge_full/turn_timeseries.parquet"))
    parser.add_argument("--basis", type=Path, default=Path("outputs/waveform_basis_experiment/nmf_basis.npy"))
    parser.add_argument("--channel-scaling", type=Path, default=Path("outputs/waveform_basis_experiment/channel_scaling.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/prefix_pet_target_experiment"))
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--max-trajectories", type=int, default=30000)
    parser.add_argument("--min-turns", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    basis = np.load(args.basis)
    raw_tensor, event_tensor, meta, groups = build_raw_dataset(
        args.input,
        points=args.points,
        max_trajectories=args.max_trajectories,
        min_turns=args.min_turns,
        seed=args.seed,
    )
    scaling = load_scaling(args.channel_scaling, raw_tensor)
    examples, continuation_rows = build_examples(
        meta=meta,
        groups=groups,
        raw_tensor=raw_tensor,
        event_tensor=event_tensor,
        scaling=scaling,
        basis=basis,
        prefix_ratios=PREFIX_RATIOS,
    )
    continuation_metrics = continuation_summary(continuation_rows)
    ml_metrics = run_ml(examples, args.out_dir, args.seed)

    examples.to_parquet(args.out_dir / "prefix_pet_examples.parquet", index=False)
    continuation_rows.to_parquet(args.out_dir / "prefix_continuation_rows.parquet", index=False)
    continuation_metrics.to_csv(args.out_dir / "prefix_continuation_metrics.csv", index=False)
    ml_metrics.to_csv(args.out_dir / "prefix_ml_metrics.csv", index=False)
    plot_overview(continuation_metrics, ml_metrics, args.out_dir)

    summary = {
        "examples": int(len(examples)),
        "trajectories": int(meta["trajectory_key"].nunique()),
        "prefix_ratios": PREFIX_RATIOS,
        "basis_components": int(basis.shape[0]),
        "model_family": "lightgbm" if LGBMClassifier is not None else "sklearn_hist_gradient_boosting",
        "continuation_metrics": continuation_metrics.to_dict(orient="records"),
        "ml_metrics": ml_metrics.to_dict(orient="records"),
    }
    (args.out_dir / "prefix_pet_target_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    def table(df: pd.DataFrame) -> str:
        return df.round(4).to_csv(index=False).strip()

    lines = [
        "# 前缀续写与机器学习增强实验",
        "",
        f"- 轨迹数：{meta['trajectory_key'].nunique():,}",
        f"- 前缀样本数：{len(examples):,}",
        f"- 前缀比例：{', '.join(str(x) for x in PREFIX_RATIOS)}",
        f"- 波形基数量：{basis.shape[0]}",
        f"- 监督模型：{'LightGBM' if LGBMClassifier is not None else 'sklearn HistGradientBoosting fallback'}",
        "",
        "## 第三步：前缀续写指标",
        table(continuation_metrics),
        "",
        "## 第四步：机器学习增强指标",
        table(ml_metrics),
        "",
        "## 解释",
        "- `stats_only`：只用相位、近期斜率、累计量和事件密度。",
        "- `waveform_only`：只用 NMF 权重、前缀重构误差和 NMF 续写出的未来波形摘要。",
        "- `enhanced`：合并统计特征和波形特征，用于检验波形基是否能增强桌宠目标预测。",
        "- `remaining_turns_research` 保留为科研指标，不建议直接作为桌宠展示目标。",
    ]
    (args.out_dir / "prefix_pet_target_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
