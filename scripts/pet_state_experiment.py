#!/usr/bin/env python3
"""Predict pet-facing progress states from online conversation prefixes.

Targets:
- remaining turns: how much longer the task may run
- future long output: whether the next few turns will produce high assistant tokens
- future overheat: whether the next few turns look like debugging pressure
- pet state: a first-pass 5-state label for desktop pet behavior
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from waveform_basis_experiment import (
    MAIN_CHANNELS,
    Scaling,
    fit_scaling,
    prefix_mask,
    solve_weights_from_prefix,
)


READ_COLUMNS = [
    "source_file",
    "trajectory_id",
    "reward",
    "turn_index",
    "assistant_tokens",
    "thinking_tokens_est",
    "user_tokens_est",
    "edit_count",
    "test_count",
    "error_feedback_count",
    "finish_call_count",
]

STATE_LABELS = [
    "reading_understanding",
    "steady_work",
    "deep_output",
    "overheat_debugging",
    "closing",
]
STATE_LABELS_ZH = {
    "reading_understanding": "读题理解",
    "steady_work": "稳定工作",
    "deep_output": "深度输出",
    "overheat_debugging": "红温调试",
    "closing": "收束",
}
LONG_OUTPUT_TOKEN_THRESHOLD = 250.0


def resample_prefix(values: np.ndarray, observed_turn: int, points: int) -> np.ndarray:
    prefix = np.asarray(values[: observed_turn + 1], dtype=float)
    if len(prefix) == 1:
        return np.full(points, prefix[0], dtype=float)
    old_x = np.linspace(0.0, 1.0, len(prefix))
    new_x = np.linspace(0.0, 1.0, points)
    return np.interp(new_x, old_x, prefix)


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
    }


def slope(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def state_label(
    current_turn: int,
    remaining_turns: int,
    current_assistant: float,
    recent_assistant_max: float,
    recent_error_sum: float,
    recent_test_sum: float,
    recent_edit_sum: float,
) -> str:
    if remaining_turns <= 3:
        return "closing"
    if recent_error_sum >= 3 or (recent_test_sum >= 2 and recent_error_sum >= 1):
        return "overheat_debugging"
    if current_turn <= 2 and recent_test_sum == 0 and recent_edit_sum == 0 and recent_error_sum == 0:
        return "reading_understanding"
    if current_assistant >= LONG_OUTPUT_TOKEN_THRESHOLD or recent_assistant_max >= LONG_OUTPUT_TOKEN_THRESHOLD:
        return "deep_output"
    return "steady_work"


def build_examples(
    input_path: Path,
    max_trajectories: int,
    max_examples: int,
    prefix_points: int,
    nmf_basis_path: Path | None,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray | None, Scaling | None]:
    pf = pq.ParquetFile(input_path)
    rng = np.random.default_rng(seed)
    row_groups = list(range(pf.num_row_groups))
    rng.shuffle(row_groups)
    trajectory_rows: list[pd.DataFrame] = []
    seen = 0

    for row_group_index in row_groups:
        df = pf.read_row_group(row_group_index, columns=READ_COLUMNS).to_pandas()
        for (source_file, trajectory_id), group in df.groupby(["source_file", "trajectory_id"], sort=False):
            group = group.sort_values("turn_index")
            if len(group) < 12:
                continue
            group = group.copy()
            group["trajectory_key"] = f"{source_file}::{trajectory_id}"
            trajectory_rows.append(group)
            seen += 1
            if max_trajectories and seen >= max_trajectories:
                break
        if max_trajectories and seen >= max_trajectories:
            break

    if not trajectory_rows:
        raise SystemExit("No trajectories found.")

    # Fit the same shape scaling on full sampled trajectories, then reuse it for prefixes.
    tensors = []
    for group in trajectory_rows:
        tensors.append(
            np.stack(
                [
                    np.interp(
                        np.linspace(0.0, 1.0, prefix_points),
                        np.linspace(0.0, 1.0, len(group)),
                        group[channel].to_numpy(dtype=float),
                    )
                    for channel in MAIN_CHANNELS
                ],
                axis=1,
            )
        )
    scaling = fit_scaling(np.stack(tensors))
    nmf_basis = np.load(nmf_basis_path) if nmf_basis_path and nmf_basis_path.exists() else None
    basis_flat = nmf_basis.reshape(nmf_basis.shape[0], -1) if nmf_basis is not None else None

    examples: list[dict[str, object]] = []
    for group in trajectory_rows:
        total_turns = len(group)
        if total_turns < 12:
            continue
        candidate_turns = np.unique(
            np.concatenate(
                [
                    np.array([0, 1, 2], dtype=int),
                    np.round(np.linspace(3, total_turns - 2, min(10, total_turns - 3))).astype(int),
                ]
            )
        )
        for current_turn in candidate_turns:
            recent = group.iloc[max(0, current_turn - 5 + 1) : current_turn + 1]
            horizon3 = group.iloc[current_turn + 1 : min(total_turns, current_turn + 4)]
            horizon5 = group.iloc[current_turn + 1 : min(total_turns, current_turn + 6)]
            progress = current_turn / max(total_turns - 1, 1)
            remaining_turns = total_turns - current_turn - 1
            future_long = bool((horizon3["assistant_tokens"] >= LONG_OUTPUT_TOKEN_THRESHOLD).any())
            future_token_pressure = float(horizon3["assistant_tokens"].sum())
            future_overheat = bool(
                horizon5["error_feedback_count"].sum() >= 4
                or (
                    horizon5["test_count"].sum() >= 2
                    and horizon5["error_feedback_count"].sum() >= 2
                )
            )
            label = state_label(
                current_turn=current_turn,
                remaining_turns=remaining_turns,
                current_assistant=float(group.iloc[current_turn]["assistant_tokens"]),
                recent_assistant_max=float(recent["assistant_tokens"].max()),
                recent_error_sum=float(recent["error_feedback_count"].sum()),
                recent_test_sum=float(recent["test_count"].sum()),
                recent_edit_sum=float(recent["edit_count"].sum()),
            )

            row: dict[str, object] = {
                "trajectory_key": group.iloc[0]["trajectory_key"],
                "trajectory_id": group.iloc[0]["trajectory_id"],
                "source_file": group.iloc[0]["source_file"],
                "reward": float(group.iloc[0]["reward"]),
                "current_turn": int(current_turn),
                "total_turns": int(total_turns),
                "progress_observed": float(progress),
                "remaining_turns": int(remaining_turns),
                "remaining_turn_bucket": (
                    "short" if remaining_turns <= 5 else "medium" if remaining_turns <= 20 else "long"
                ),
                "future_end_within_5": bool(remaining_turns <= 5),
                "future_long_output_3": future_long,
                "future_token_pressure_3": future_token_pressure,
                "future_overheat_5": future_overheat,
                "pet_state": label,
                "pet_state_zh": STATE_LABELS_ZH[label],
                "current_assistant_tokens": float(group.iloc[current_turn]["assistant_tokens"]),
                "long_output_threshold": LONG_OUTPUT_TOKEN_THRESHOLD,
            }

            for channel in ["assistant_tokens", "thinking_tokens_est", "user_tokens_est", "error_feedback_count", "test_count", "edit_count"]:
                values = recent[channel].to_numpy(dtype=float)
                row.update(safe_stats(values, f"recent_{channel}"))
                row[f"recent_{channel}_slope"] = slope(values)
                row[f"cum_{channel}_sum"] = float(group.iloc[: current_turn + 1][channel].sum())

            if basis_flat is not None:
                prefix_tensor = np.stack(
                    [
                        resample_prefix(group[channel].to_numpy(dtype=float), current_turn, prefix_points)
                        for channel in MAIN_CHANNELS
                    ],
                    axis=1,
                )[None, :, :]
                scaled = scaling.transform(prefix_tensor).reshape(1, -1)
                weights = solve_weights_from_prefix(
                    basis_flat,
                    scaled,
                    prefix_points,
                    len(MAIN_CHANNELS),
                    prefix_ratio=1.0,
                    iterations=120,
                )[0]
                for idx, value in enumerate(weights, start=1):
                    row[f"nmf_basis_{idx}_prefix_weight"] = float(value)

            examples.append(row)
            if max_examples and len(examples) >= max_examples:
                return pd.DataFrame(examples), nmf_basis, scaling

    return pd.DataFrame(examples), nmf_basis, scaling


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "trajectory_key",
        "trajectory_id",
        "source_file",
        "reward",
        "total_turns",
        "progress_observed",
        "long_output_threshold",
        "remaining_turns",
        "remaining_turn_bucket",
        "future_end_within_5",
        "future_long_output_3",
        "future_token_pressure_3",
        "future_overheat_5",
        "pet_state",
        "pet_state_zh",
    }
    cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def binary_metrics(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    out = {
        "positive_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, proba))
    else:
        out["auc"] = float("nan")
    return out


def save_feature_importance(model, x_test: pd.DataFrame, y_test, out_path: Path, scoring: str, seed: int) -> None:
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=3,
        random_state=seed,
        scoring=scoring,
        n_jobs=1,
    )
    imp = pd.DataFrame(
        {
            "feature": x_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    imp.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/coderforge_full/turn_timeseries.parquet"))
    parser.add_argument("--basis", type=Path, default=Path("outputs/waveform_basis_experiment/nmf_basis.npy"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/pet_state_experiment"))
    parser.add_argument("--max-trajectories", type=int, default=30000)
    parser.add_argument("--max-examples", type=int, default=120000)
    parser.add_argument("--prefix-points", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    examples, _, _ = build_examples(
        args.input,
        max_trajectories=args.max_trajectories,
        max_examples=args.max_examples,
        prefix_points=args.prefix_points,
        nmf_basis_path=args.basis,
        seed=args.seed,
    )
    examples.to_parquet(args.out_dir / "pet_state_examples.parquet", index=False)

    cols = feature_columns(examples)
    groups = examples["trajectory_key"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(examples, groups=groups))
    x_train = examples.iloc[train_idx][cols].replace([np.inf, -np.inf], 0).fillna(0)
    x_test = examples.iloc[test_idx][cols].replace([np.inf, -np.inf], 0).fillna(0)

    results: list[dict[str, object]] = []

    reg = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.06, l2_regularization=0.02, random_state=args.seed)
    y_train_reg = np.log1p(examples.iloc[train_idx]["remaining_turns"].to_numpy(dtype=float))
    y_test_remaining = examples.iloc[test_idx]["remaining_turns"].to_numpy(dtype=float)
    reg.fit(x_train, y_train_reg)
    pred_remaining = np.expm1(reg.predict(x_test))
    results.append(
        {
            "task": "remaining_turns_regression",
            "mae_turns": float(mean_absolute_error(y_test_remaining, pred_remaining)),
            "median_abs_error_turns": float(median_absolute_error(y_test_remaining, pred_remaining)),
            "r2_log_remaining": float(r2_score(np.log1p(y_test_remaining), np.log1p(np.maximum(pred_remaining, 0)))),
        }
    )
    save_feature_importance(reg, x_test, np.log1p(y_test_remaining), args.out_dir / "feature_importance_remaining_turns.csv", "neg_mean_absolute_error", args.seed)

    binary_tasks = [
        ("future_end_within_5", "future_end_within_5"),
        ("future_long_output_3", "future_long_output_3"),
        ("future_overheat_5", "future_overheat_5"),
    ]
    for task_name, target in binary_tasks:
        clf = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.06, l2_regularization=0.02, random_state=args.seed)
        y_train = examples.iloc[train_idx][target].astype(int).to_numpy()
        y_test = examples.iloc[test_idx][target].astype(int).to_numpy()
        clf.fit(x_train, y_train)
        proba = clf.predict_proba(x_test)[:, 1]
        pred = proba >= 0.5
        results.append({"task": task_name, **binary_metrics(y_test, proba, pred)})
        save_feature_importance(clf, x_test, y_test, args.out_dir / f"feature_importance_{task_name}.csv", "balanced_accuracy", args.seed)

    state_clf = RandomForestClassifier(
        n_estimators=220,
        max_depth=14,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )
    y_train_state = examples.iloc[train_idx]["pet_state"].to_numpy()
    y_test_state = examples.iloc[test_idx]["pet_state"].to_numpy()
    state_clf.fit(x_train, y_train_state)
    state_pred = state_clf.predict(x_test)
    labels = [label for label in STATE_LABELS if label in set(y_test_state) | set(state_pred)]
    state_report = classification_report(
        y_test_state,
        state_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(state_report).T.to_csv(args.out_dir / "pet_state_classification_report.csv")
    pd.DataFrame(
        confusion_matrix(y_test_state, state_pred, labels=labels),
        index=labels,
        columns=labels,
    ).to_csv(args.out_dir / "pet_state_confusion_matrix.csv")
    state_importance = pd.DataFrame(
        {
            "feature": cols,
            "importance": state_clf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    state_importance.to_csv(args.out_dir / "feature_importance_pet_state.csv", index=False)
    results.append(
        {
            "task": "pet_state_multiclass",
            "accuracy": float(accuracy_score(y_test_state, state_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test_state, state_pred)),
            "macro_f1": float(f1_score(y_test_state, state_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test_state, state_pred, average="weighted", zero_division=0)),
        }
    )

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.out_dir / "pet_state_metrics.csv", index=False)

    state_counts = examples["pet_state"].value_counts().rename_axis("state").reset_index(name="count")
    state_counts["state_zh"] = state_counts["state"].map(STATE_LABELS_ZH)
    state_counts["rate"] = state_counts["count"] / len(examples)
    state_counts.to_csv(args.out_dir / "pet_state_label_distribution.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test_remaining, pred_remaining, s=4, alpha=0.18)
    lim = max(float(np.percentile(y_test_remaining, 99)), float(np.percentile(pred_remaining, 99)), 1.0)
    axes[0].plot([0, lim], [0, lim], color="black", linewidth=1)
    axes[0].set_xlim(0, lim)
    axes[0].set_ylim(0, lim)
    axes[0].set_xlabel("true remaining turns")
    axes[0].set_ylabel("predicted remaining turns")
    axes[0].set_title("Remaining-turn prediction")
    state_counts.sort_values("rate", ascending=True).plot.barh(x="state", y="rate", ax=axes[1], legend=False)
    axes[1].set_title("Pet state label distribution")
    axes[1].set_xlabel("rate")
    fig.tight_layout()
    fig.savefig(args.out_dir / "pet_state_experiment_overview.png", dpi=180)
    plt.close(fig)

    summary = {
        "examples": int(len(examples)),
        "trajectories": int(examples["trajectory_key"].nunique()),
        "train_examples": int(len(train_idx)),
        "test_examples": int(len(test_idx)),
        "features": int(len(cols)),
        "state_labels": STATE_LABELS_ZH,
        "metrics": results,
    }
    (args.out_dir / "pet_state_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 桌宠状态预测实验",
        "",
        f"- 样本数：{len(examples):,}",
        f"- 轨迹数：{examples['trajectory_key'].nunique():,}",
        f"- 训练/测试样本：{len(train_idx):,} / {len(test_idx):,}",
        f"- 特征数：{len(cols)}",
        "",
        "## 预测指标",
        results_df.round(4).to_csv(index=False).strip(),
        "",
        "## 状态标签分布",
        state_counts.round(4).to_csv(index=False).strip(),
        "",
        "## 状态定义",
        "- 读题理解：早期阶段，尚未进入强输出或红温。",
        "- 稳定工作：中间常规推进。",
        "- 深度输出：当前或最近 5 轮存在长输出压力。",
        "- 红温调试：近期 5 轮错误/测试压力高。",
        "- 收束：预计 3 轮内结束。",
    ]
    (args.out_dir / "pet_state_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
