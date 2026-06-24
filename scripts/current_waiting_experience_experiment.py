#!/usr/bin/env python3
"""Evaluate current-turn waiting-experience states for a Codex desktop pet.

This experiment is intentionally closer than the earlier "future 3/5 turns"
targets. It simulates a streaming assistant turn by exposing 0%, 25%, 50%, 75%,
or 100% of the current turn's output tokens, then compares:

- history_only: all previous turns, no current output tokens
- history_plus_stream: previous turns plus current visible context and streamed tokens
- rule_baseline: deployable threshold rules on the same stream features
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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


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

PREFIX_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
NUMERIC_CHANNELS = [
    "assistant_tokens",
    "thinking_tokens_est",
    "user_tokens_est",
    "edit_count",
    "test_count",
    "error_feedback_count",
]


@dataclass(frozen=True)
class Thresholds:
    assistant_long: float
    thinking_long: float
    feedback_heavy: float
    visible_load_deep: float
    overheat_score_high: float


def safe_quantile(values: pd.Series, q: float, fallback: float) -> float:
    value = float(values.quantile(q))
    if not np.isfinite(value) or value <= 0:
        return fallback
    return value


def fit_thresholds(groups: list[pd.DataFrame]) -> Thresholds:
    sampled = pd.concat(groups, ignore_index=True)
    visible_load = sampled["assistant_tokens"] + sampled["thinking_tokens_est"]
    overheat_score = (
        sampled["error_feedback_count"] * 2.0
        + sampled["test_count"] * 1.4
        + np.log1p(sampled["user_tokens_est"]) * 0.35
    )
    return Thresholds(
        assistant_long=max(250.0, safe_quantile(sampled["assistant_tokens"], 0.80, 250.0)),
        thinking_long=max(120.0, safe_quantile(sampled["thinking_tokens_est"], 0.90, 120.0)),
        feedback_heavy=max(1200.0, safe_quantile(sampled["user_tokens_est"], 0.85, 1200.0)),
        visible_load_deep=max(300.0, safe_quantile(visible_load, 0.82, 300.0)),
        overheat_score_high=max(4.0, safe_quantile(overheat_score, 0.85, 4.0)),
    )


def slope(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def add_stats(row: dict[str, object], prefix: str, values: pd.Series | np.ndarray) -> None:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        arr = np.zeros(1, dtype=float)
    row[f"{prefix}_sum"] = float(arr.sum())
    row[f"{prefix}_mean"] = float(arr.mean())
    row[f"{prefix}_max"] = float(arr.max())
    row[f"{prefix}_last"] = float(arr[-1])
    row[f"{prefix}_std"] = float(arr.std())
    row[f"{prefix}_slope"] = slope(arr)


def current_wait_state(
    *,
    current_turn: int,
    total_turns: int,
    assistant_tokens: float,
    thinking_tokens: float,
    user_tokens: float,
    test_count: float,
    error_feedback_count: float,
    previous_recent_errors: float,
    previous_recent_tests: float,
    previous_recent_edits: float,
    thresholds: Thresholds,
) -> str:
    remaining = total_turns - current_turn - 1
    current_visible_load = assistant_tokens + thinking_tokens
    overheat_score = (
        error_feedback_count * 2.0
        + previous_recent_errors * 1.2
        + previous_recent_tests * 0.7
        + np.log1p(user_tokens) * 0.35
    )

    if overheat_score >= thresholds.overheat_score_high or (
        previous_recent_errors >= 2 and previous_recent_tests >= 1
    ):
        return "overheat_debugging"
    if (
        assistant_tokens >= thresholds.assistant_long
        or thinking_tokens >= thresholds.thinking_long
        or current_visible_load >= thresholds.visible_load_deep
    ):
        return "deep_output"
    if remaining <= 2 and current_visible_load < thresholds.visible_load_deep and previous_recent_errors <= 1:
        return "closing"
    if current_turn <= 2 and previous_recent_edits == 0 and previous_recent_tests == 0 and previous_recent_errors == 0:
        return "reading_understanding"
    return "steady_work"


def rule_state(row: pd.Series, thresholds: Thresholds) -> str:
    if row["stream_overheat_score"] >= thresholds.overheat_score_high or (
        row["hist_recent_error_feedback_count_sum"] >= 2 and row["hist_recent_test_count_sum"] >= 1
    ):
        return "overheat_debugging"
    if (
        row["stream_current_assistant_tokens"] >= thresholds.assistant_long
        or row["stream_current_thinking_tokens"] >= thresholds.thinking_long
        or row["stream_visible_load"] >= thresholds.visible_load_deep
    ):
        return "deep_output"
    if row["stream_prefix_fraction"] == 0 and row["current_turn"] <= 2 and row["hist_recent_edit_count_sum"] == 0:
        return "reading_understanding"
    if row["hist_recent_finish_call_count_sum"] > 0 or (
        row["hist_recent_assistant_tokens_slope"] < 0
        and row["hist_recent_error_feedback_count_sum"] == 0
        and row["stream_visible_load"] < thresholds.visible_load_deep
    ):
        return "closing"
    return "steady_work"


def load_trajectories(input_path: Path, max_trajectories: int, seed: int) -> list[pd.DataFrame]:
    pf = pq.ParquetFile(input_path)
    rng = np.random.default_rng(seed)
    row_groups = list(range(pf.num_row_groups))
    rng.shuffle(row_groups)

    groups: list[pd.DataFrame] = []
    seen_keys: set[str] = set()
    for row_group_index in row_groups:
        df = pf.read_row_group(row_group_index, columns=READ_COLUMNS).to_pandas()
        for (source_file, trajectory_id), group in df.groupby(["source_file", "trajectory_id"], sort=False):
            key = f"{source_file}::{trajectory_id}"
            if key in seen_keys:
                continue
            group = group.sort_values("turn_index").copy()
            if len(group) < 8:
                continue
            group["trajectory_key"] = key
            groups.append(group)
            seen_keys.add(key)
            if max_trajectories and len(groups) >= max_trajectories:
                return groups
    return groups


def candidate_turns(total_turns: int, max_turns_per_trajectory: int) -> np.ndarray:
    if total_turns <= max_turns_per_trajectory:
        return np.arange(total_turns, dtype=int)
    anchors = np.array([0, 1, 2, total_turns - 3, total_turns - 2, total_turns - 1], dtype=int)
    middle_count = max(max_turns_per_trajectory - len(np.unique(anchors)), 1)
    middle = np.round(np.linspace(3, total_turns - 4, middle_count)).astype(int)
    return np.unique(np.clip(np.concatenate([anchors, middle]), 0, total_turns - 1))


def build_examples(
    groups: list[pd.DataFrame],
    thresholds: Thresholds,
    max_examples: int,
    max_turns_per_trajectory: int,
) -> pd.DataFrame:
    examples: list[dict[str, object]] = []
    for group in groups:
        total_turns = len(group)
        for current_turn in candidate_turns(total_turns, max_turns_per_trajectory):
            current = group.iloc[current_turn]
            history = group.iloc[:current_turn]
            recent_history = group.iloc[max(0, current_turn - 5) : current_turn]
            previous_recent_errors = float(recent_history["error_feedback_count"].sum())
            previous_recent_tests = float(recent_history["test_count"].sum())
            previous_recent_edits = float(recent_history["edit_count"].sum())
            label = current_wait_state(
                current_turn=current_turn,
                total_turns=total_turns,
                assistant_tokens=float(current["assistant_tokens"]),
                thinking_tokens=float(current["thinking_tokens_est"]),
                user_tokens=float(current["user_tokens_est"]),
                test_count=float(current["test_count"]),
                error_feedback_count=float(current["error_feedback_count"]),
                previous_recent_errors=previous_recent_errors,
                previous_recent_tests=previous_recent_tests,
                previous_recent_edits=previous_recent_edits,
                thresholds=thresholds,
            )

            base: dict[str, object] = {
                "trajectory_key": current["trajectory_key"],
                "trajectory_id": current["trajectory_id"],
                "source_file": current["source_file"],
                "reward": float(current["reward"]),
                "current_turn": int(current_turn),
                "total_turns": int(total_turns),
                "current_wait_state": label,
                "current_wait_state_zh": STATE_LABELS_ZH[label],
                "true_current_assistant_tokens": float(current["assistant_tokens"]),
                "true_current_thinking_tokens": float(current["thinking_tokens_est"]),
                "true_current_user_tokens": float(current["user_tokens_est"]),
                "true_current_error_feedback_count": float(current["error_feedback_count"]),
                "true_current_test_count": float(current["test_count"]),
                "true_current_edit_count": float(current["edit_count"]),
                "is_deep_output": label == "deep_output",
                "is_overheat_debugging": label == "overheat_debugging",
            }
            for channel in NUMERIC_CHANNELS:
                add_stats(base, f"hist_cum_{channel}", history[channel] if len(history) else np.zeros(0))
                add_stats(base, f"hist_recent_{channel}", recent_history[channel] if len(recent_history) else np.zeros(0))
            add_stats(
                base,
                "hist_recent_finish_call_count",
                recent_history["finish_call_count"] if len(recent_history) else np.zeros(0),
            )

            for fraction in PREFIX_FRACTIONS:
                row = dict(base)
                observed_assistant = float(current["assistant_tokens"]) * fraction
                observed_thinking = float(current["thinking_tokens_est"]) * fraction
                visible_feedback = float(current["user_tokens_est"])
                row["stream_prefix_fraction"] = float(fraction)
                row["stream_current_assistant_tokens"] = observed_assistant
                row["stream_current_thinking_tokens"] = observed_thinking
                row["stream_current_user_tokens"] = visible_feedback
                row["stream_visible_load"] = observed_assistant + observed_thinking
                row["stream_context_load"] = visible_feedback
                row["stream_output_vs_history_mean"] = observed_assistant / max(
                    float(base["hist_recent_assistant_tokens_mean"]), 1.0
                )
                row["stream_thinking_vs_history_mean"] = observed_thinking / max(
                    float(base["hist_recent_thinking_tokens_est_mean"]), 1.0
                )
                row["stream_overheat_score"] = (
                    previous_recent_errors * 1.2
                    + previous_recent_tests * 0.7
                    + float(current["error_feedback_count"]) * 2.0
                    + np.log1p(visible_feedback) * 0.35
                )
                row["rule_state"] = rule_state(pd.Series(row), thresholds)
                examples.append(row)
                if max_examples and len(examples) >= max_examples:
                    return pd.DataFrame(examples)
    return pd.DataFrame(examples)


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    excluded = {
        "trajectory_key",
        "trajectory_id",
        "source_file",
        "reward",
        "total_turns",
        "current_wait_state",
        "current_wait_state_zh",
        "rule_state",
        "true_current_assistant_tokens",
        "true_current_thinking_tokens",
        "true_current_user_tokens",
        "true_current_error_feedback_count",
        "true_current_test_count",
        "true_current_edit_count",
        "is_deep_output",
        "is_overheat_debugging",
    }
    history = [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col]) and not col.startswith("stream_")
    ]
    stream = [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col]) and (col in history or col.startswith("stream_"))
    ]
    return {
        "history_only": history,
        "history_plus_stream": stream,
    }


def binary_metrics(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    out = {
        "positive_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    out["auc"] = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) == 2 else float("nan")
    return out


def train_and_evaluate(examples: pd.DataFrame, out_dir: Path, seed: int) -> pd.DataFrame:
    groups = examples["trajectory_key"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(splitter.split(examples, groups=groups))
    feature_sets = feature_groups(examples)
    metrics: list[dict[str, object]] = []

    y_test_state = examples.iloc[test_idx]["current_wait_state"].to_numpy()
    rule_pred = examples.iloc[test_idx]["rule_state"].to_numpy()
    metrics.append(
        {
            "model": "rule_baseline",
            "target": "current_wait_state",
            "prefix_fraction": "all",
            "features": 0,
            "accuracy": float(accuracy_score(y_test_state, rule_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test_state, rule_pred)),
            "macro_f1": float(f1_score(y_test_state, rule_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test_state, rule_pred, average="weighted", zero_division=0)),
        }
    )

    labels = [label for label in STATE_LABELS if label in set(y_test_state) | set(rule_pred)]
    pd.DataFrame(
        confusion_matrix(y_test_state, rule_pred, labels=labels),
        index=labels,
        columns=labels,
    ).to_csv(out_dir / "rule_baseline_confusion_matrix.csv")

    for feature_group, cols in feature_sets.items():
        x_train = examples.iloc[train_idx][cols].replace([np.inf, -np.inf], 0).fillna(0)
        x_test = examples.iloc[test_idx][cols].replace([np.inf, -np.inf], 0).fillna(0)

        state_clf = RandomForestClassifier(
            n_estimators=260,
            max_depth=16,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
        y_train_state = examples.iloc[train_idx]["current_wait_state"].to_numpy()
        state_clf.fit(x_train, y_train_state)
        state_pred = state_clf.predict(x_test)
        metrics.append(
            {
                "model": feature_group,
                "target": "current_wait_state",
                "prefix_fraction": "all",
                "features": len(cols),
                "accuracy": float(accuracy_score(y_test_state, state_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test_state, state_pred)),
                "macro_f1": float(f1_score(y_test_state, state_pred, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_test_state, state_pred, average="weighted", zero_division=0)),
            }
        )
        report = classification_report(
            y_test_state,
            state_pred,
            labels=[label for label in STATE_LABELS if label in set(y_test_state) | set(state_pred)],
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).T.to_csv(out_dir / f"{feature_group}_classification_report.csv")
        pd.DataFrame(
            confusion_matrix(
                y_test_state,
                state_pred,
                labels=[label for label in STATE_LABELS if label in set(y_test_state) | set(state_pred)],
            ),
            index=[label for label in STATE_LABELS if label in set(y_test_state) | set(state_pred)],
            columns=[label for label in STATE_LABELS if label in set(y_test_state) | set(state_pred)],
        ).to_csv(out_dir / f"{feature_group}_confusion_matrix.csv")

        importance = pd.DataFrame({"feature": cols, "importance": state_clf.feature_importances_}).sort_values(
            "importance", ascending=False
        )
        importance.to_csv(out_dir / f"feature_importance_{feature_group}_state.csv", index=False)

        for target in ["is_deep_output", "is_overheat_debugging"]:
            clf = HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.06,
                l2_regularization=0.02,
                random_state=seed,
            )
            y_train = examples.iloc[train_idx][target].astype(int).to_numpy()
            y_test = examples.iloc[test_idx][target].astype(int).to_numpy()
            clf.fit(x_train, y_train)
            proba = clf.predict_proba(x_test)[:, 1]
            pred = proba >= 0.5
            metrics.append(
                {
                    "model": feature_group,
                    "target": target,
                    "prefix_fraction": "all",
                    "features": len(cols),
                    **binary_metrics(y_test, proba, pred),
                }
            )

        for fraction in PREFIX_FRACTIONS:
            mask = examples.iloc[test_idx]["stream_prefix_fraction"].to_numpy() == fraction
            if not mask.any():
                continue
            subset_true = y_test_state[mask]
            subset_pred = state_pred[mask]
            metrics.append(
                {
                    "model": feature_group,
                    "target": "current_wait_state",
                    "prefix_fraction": fraction,
                    "features": len(cols),
                    "accuracy": float(accuracy_score(subset_true, subset_pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(subset_true, subset_pred)),
                    "macro_f1": float(f1_score(subset_true, subset_pred, average="macro", zero_division=0)),
                    "weighted_f1": float(f1_score(subset_true, subset_pred, average="weighted", zero_division=0)),
                }
            )

    return pd.DataFrame(metrics)


def plot_results(metrics: pd.DataFrame, examples: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    dist = examples["current_wait_state"].value_counts(normalize=True).reindex(STATE_LABELS).fillna(0)
    axes[0].barh(list(dist.index), dist.values)
    axes[0].set_title("Current waiting-state distribution")
    axes[0].set_xlabel("rate")

    state_rows = metrics[
        (metrics["target"] == "current_wait_state")
        & (metrics["model"].isin(["history_only", "history_plus_stream"]))
        & (metrics["prefix_fraction"] != "all")
    ].copy()
    for model, group in state_rows.groupby("model"):
        group = group.sort_values("prefix_fraction")
        axes[1].plot(group["prefix_fraction"], group["macro_f1"], marker="o", label=model)
    axes[1].set_title("State macro F1 by streamed prefix")
    axes[1].set_xlabel("streamed current-turn fraction")
    axes[1].set_ylabel("macro F1")
    axes[1].legend()

    binary = metrics[(metrics["target"].isin(["is_deep_output", "is_overheat_debugging"]))].copy()
    labels = [f"{row.model}\n{row.target}" for row in binary.itertuples()]
    axes[2].bar(range(len(binary)), binary["auc"].fillna(0))
    axes[2].set_xticks(range(len(binary)), labels, rotation=45, ha="right")
    axes[2].set_title("Current deep-output / overheat AUC")
    axes[2].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "current_waiting_experience_overview.png", dpi=180)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, examples: pd.DataFrame, thresholds: Thresholds, out_dir: Path) -> None:
    state_dist = examples["current_wait_state"].value_counts().rename_axis("state").reset_index(name="count")
    state_dist["state_zh"] = state_dist["state"].map(STATE_LABELS_ZH)
    state_dist["rate"] = state_dist["count"] / len(examples)
    state_dist.to_csv(out_dir / "current_wait_state_distribution.csv", index=False)

    all_state = metrics[(metrics["target"] == "current_wait_state") & (metrics["prefix_fraction"] == "all")]
    by_prefix = metrics[
        (metrics["target"] == "current_wait_state")
        & (metrics["model"].isin(["history_only", "history_plus_stream"]))
        & (metrics["prefix_fraction"] != "all")
    ]
    binary = metrics[metrics["target"].isin(["is_deep_output", "is_overheat_debugging"])]

    lines = [
        "# 当前等待体验实验",
        "",
        "本实验把目标从“未来几轮”拉近到“当前正在输出的这一轮”。由于 CoderForge 没有真实流式时间戳，脚本把当前轮的模型输出和思考长度切成 0%、25%、50%、75%、100% 五个前缀，模拟桌宠在输出过程中的可见信息。",
        "",
        "## 实验设置",
        "",
        f"- 样本数：{len(examples):,}",
        f"- 轨迹数：{examples['trajectory_key'].nunique():,}",
        "- 对照组：只看历史统计、历史统计 + 当前流式前缀、规则状态机。",
        "- 目标：当前等待体验状态、当前是否深度输出、当前是否红温调试。",
        "",
        "## 阈值",
        "",
        f"- 长模型输出阈值：{thresholds.assistant_long:.1f}",
        f"- 长思考规划阈值：{thresholds.thinking_long:.1f}",
        f"- 高环境反馈阈值：{thresholds.feedback_heavy:.1f}",
        f"- 深度可见负载阈值：{thresholds.visible_load_deep:.1f}",
        f"- 红温压力阈值：{thresholds.overheat_score_high:.2f}",
        "",
        "## 整体状态识别",
        "",
        all_state.round(4).to_csv(index=False).strip(),
        "",
        "## 流式前缀增益",
        "",
        by_prefix.round(4).to_csv(index=False).strip(),
        "",
        "## 当前二分类目标",
        "",
        binary.round(4).to_csv(index=False).strip(),
        "",
        "## 标签分布",
        "",
        state_dist.round(4).to_csv(index=False).strip(),
        "",
        "## 结论",
        "",
        "- 只看历史统计主要反映任务惯性，适合识别红温调试这类有上下文延续性的状态。",
        "- 加入当前流式 token 后，深度输出识别会更贴近桌宠的真实等待体验，因为用户真正感知到的是当前输出正在变长。",
        "- 规则状态机可以作为第一版插件 baseline；机器学习模型适合做离线校准和阈值学习。",
        "- 这版目标比“未来 3/5 轮”更适合桌宠，因为输出可以在本轮内随 token 和事件持续更新。",
    ]
    (out_dir / "current_waiting_experience_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/coderforge_full/turn_timeseries.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/current_waiting_experience_experiment"))
    parser.add_argument("--max-trajectories", type=int, default=30000)
    parser.add_argument("--max-examples", type=int, default=180000)
    parser.add_argument("--max-turns-per-trajectory", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    groups = load_trajectories(args.input, args.max_trajectories, args.seed)
    if not groups:
        raise SystemExit("No trajectories found.")

    thresholds = fit_thresholds(groups)
    examples = build_examples(groups, thresholds, args.max_examples, args.max_turns_per_trajectory)
    examples.to_parquet(args.out_dir / "current_waiting_experience_examples.parquet", index=False)

    metrics = train_and_evaluate(examples, args.out_dir, args.seed)
    metrics.to_csv(args.out_dir / "current_waiting_experience_metrics.csv", index=False)
    plot_results(metrics, examples, args.out_dir)
    write_report(metrics, examples, thresholds, args.out_dir)

    summary = {
        "examples": int(len(examples)),
        "trajectories": int(examples["trajectory_key"].nunique()),
        "prefix_fractions": PREFIX_FRACTIONS,
        "thresholds": thresholds.__dict__,
        "metrics": json.loads(metrics.replace({np.nan: None}).to_json(orient="records", force_ascii=False)),
    }
    (args.out_dir / "current_waiting_experience_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print((args.out_dir / "current_waiting_experience_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
