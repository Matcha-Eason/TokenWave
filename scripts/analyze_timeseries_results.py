#!/usr/bin/env python3
"""Analyze extracted turn-level CoderForge time-series outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


METRIC_COLUMNS = [
    "source_file",
    "trajectory_id",
    "reward",
    "turn_index",
    "assistant_tokens",
    "user_tokens_est",
    "thinking_tokens_est",
    "external_tool_call_count",
    "edit_count",
    "test_count",
    "error_feedback_count",
    "finish_call_count",
    "macro_load_index",
]


def spectral_features(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 4 or np.allclose(values, values[0]):
        return {
            "spectral_entropy": 0.0,
            "dominant_frequency": 0.0,
            "low_frequency_power_ratio": 0.0,
        }
    centered = values - values.mean()
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    if len(spectrum) <= 1 or spectrum[1:].sum() <= 0:
        return {
            "spectral_entropy": 0.0,
            "dominant_frequency": 0.0,
            "low_frequency_power_ratio": 0.0,
        }
    power = spectrum[1:]
    probs = power / power.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12)) / np.log2(len(probs))
    dom_idx = int(np.argmax(power) + 1)
    low_cut = max(1, int(np.ceil(len(power) * 0.2)))
    return {
        "spectral_entropy": float(entropy),
        "dominant_frequency": float(dom_idx / len(values)),
        "low_frequency_power_ratio": float(power[:low_cut].sum() / power.sum()),
    }


def peak_features(group: pd.DataFrame, metric: str) -> dict[str, float]:
    values = group[metric].to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        return {}
    peak_idx = int(np.argmax(values))
    q25 = int(np.floor((n - 1) * 0.25))
    q50 = int(np.floor((n - 1) * 0.50))
    q75 = int(np.floor((n - 1) * 0.75))
    first_mean = float(values[: q25 + 1].mean())
    mid_mean = float(values[q25 : q75 + 1].mean())
    last_mean = float(values[q75:].mean())
    return {
        f"{metric}_mean": float(values.mean()),
        f"{metric}_max": float(values.max()),
        f"{metric}_peak_turn": peak_idx,
        f"{metric}_peak_progress": float(peak_idx / max(n - 1, 1)),
        f"{metric}_first_quarter_mean": first_mean,
        f"{metric}_middle_half_mean": mid_mean,
        f"{metric}_last_quarter_mean": last_mean,
        f"{metric}_middle_over_edges": float(
            mid_mean / max((first_mean + last_mean) / 2, 1e-9)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/coderforge_full/turn_timeseries.parquet"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/coderforge_full"))
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=0,
        help="0 means all trajectories. Use a smaller value for quick checks.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(args.input)
    rows = []
    seen = 0
    for row_group_index in range(pf.num_row_groups):
        df = pf.read_row_group(row_group_index, columns=METRIC_COLUMNS).to_pandas()
        for (source_file, trajectory_id), group in df.groupby(
            ["source_file", "trajectory_id"], sort=False
        ):
            group = group.sort_values("turn_index")
            row = {
                "source_file": source_file,
                "trajectory_id": trajectory_id,
                "trajectory_key": f"{source_file}::{trajectory_id}",
                "reward": float(group["reward"].iloc[0]),
                "turns": int(len(group)),
                "finish_turns": int(group["finish_call_count"].sum()),
                "total_assistant_tokens": int(group["assistant_tokens"].sum()),
                "total_user_tokens_est": int(group["user_tokens_est"].sum()),
                "total_thinking_tokens_est": int(group["thinking_tokens_est"].sum()),
                "total_external_tool_calls": int(group["external_tool_call_count"].sum()),
                "total_edits": int(group["edit_count"].sum()),
                "total_tests": int(group["test_count"].sum()),
                "total_error_feedback": int(group["error_feedback_count"].sum()),
            }
            for metric in ["assistant_tokens", "thinking_tokens_est", "macro_load_index"]:
                row.update(peak_features(group, metric))
            row.update(spectral_features(group["macro_load_index"].to_numpy()))
            rows.append(row)
            seen += 1
            if args.max_trajectories and seen >= args.max_trajectories:
                break
        if args.max_trajectories and seen >= args.max_trajectories:
            break

    traj = pd.DataFrame(rows)
    traj.to_parquet(args.out_dir / "trajectory_wave_features.parquet", index=False)
    traj.to_csv(args.out_dir / "trajectory_wave_features.csv", index=False)

    summary_lines = []
    summary_lines.append(f"trajectories: {len(traj):,}")
    summary_lines.append(f"turns mean/median: {traj['turns'].mean():.2f} / {traj['turns'].median():.0f}")
    summary_lines.append(
        f"reward mean, success rate: {traj['reward'].mean():.4f}, {(traj['reward'] > 0).mean():.2%}"
    )
    for metric in ["assistant_tokens", "thinking_tokens_est", "macro_load_index"]:
        summary_lines.append(
            f"{metric} peak progress median/mean: "
            f"{traj[f'{metric}_peak_progress'].median():.3f} / "
            f"{traj[f'{metric}_peak_progress'].mean():.3f}"
        )
        summary_lines.append(
            f"{metric} middle_over_edges median/mean: "
            f"{traj[f'{metric}_middle_over_edges'].median():.3f} / "
            f"{traj[f'{metric}_middle_over_edges'].mean():.3f}"
        )
    summary_lines.append(
        f"macro spectral entropy median/mean: "
        f"{traj['spectral_entropy'].median():.3f} / {traj['spectral_entropy'].mean():.3f}"
    )
    summary_lines.append(
        f"macro low-frequency power ratio median/mean: "
        f"{traj['low_frequency_power_ratio'].median():.3f} / "
        f"{traj['low_frequency_power_ratio'].mean():.3f}"
    )

    by_reward = (
        traj.assign(success=traj["reward"] > 0)
        .groupby("success")
        .agg(
            trajectories=("trajectory_id", "size"),
            turns_mean=("turns", "mean"),
            assistant_peak_progress_median=("assistant_tokens_peak_progress", "median"),
            macro_peak_progress_median=("macro_load_index_peak_progress", "median"),
            macro_middle_over_edges_median=("macro_load_index_middle_over_edges", "median"),
            spectral_entropy_median=("spectral_entropy", "median"),
            low_frequency_power_ratio_median=("low_frequency_power_ratio", "median"),
            tests_mean=("total_tests", "mean"),
            edits_mean=("total_edits", "mean"),
            error_feedback_mean=("total_error_feedback", "mean"),
        )
        .reset_index()
    )
    by_reward.to_csv(args.out_dir / "wave_features_by_reward.csv", index=False)
    summary_lines.append("\nby reward:")
    summary_lines.append(by_reward.round(4).to_string(index=False))

    text = "\n".join(summary_lines)
    (args.out_dir / "wave_analysis_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
