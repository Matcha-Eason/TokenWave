#!/usr/bin/env python3
"""Batch extraction for full CoderForge parquet shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from extract_turn_timeseries import extract_rows, write_visualizations


SOURCE_COLUMNS = [
    "trajectory_id",
    "reward",
    "chat_template_applied",
    "labels",
]


def find_parquets(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob(pattern))


def append_table(writer: pq.ParquetWriter | None, output: Path, df: pd.DataFrame):
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(output, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def write_aggregate_plot(aggregate: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(aggregate["progress"], aggregate["assistant_tokens"])
    axes[0].set_ylabel("assistant tokens")
    axes[1].plot(aggregate["progress"], aggregate["external_tool_call_count"])
    axes[1].plot(aggregate["progress"], aggregate["edit_count"], label="edits")
    axes[1].plot(aggregate["progress"], aggregate["test_count"], label="tests")
    axes[1].set_ylabel("mean counts")
    axes[1].legend(loc="upper right")
    axes[2].plot(aggregate["progress"], aggregate["macro_load_index"])
    axes[2].set_ylabel("load index")
    axes[2].set_xlabel("normalized progress")
    fig.suptitle("Aggregate trajectory envelope")
    fig.tight_layout()
    fig.savefig(out_dir / "aggregate_envelope.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/CoderForge-Preview/trajectories-tokenized_qwencoder"),
        help="A parquet file or a directory containing parquet shards.",
    )
    parser.add_argument(
        "--pattern",
        default="**/*.parquet",
        help="Glob used when --input is a directory.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/coderforge_full"))
    parser.add_argument("--max-files", type=int, default=0, help="0 means all files.")
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write a CSV. This can be very large for the full dataset.",
    )
    parser.add_argument(
        "--plot-max-trajectories",
        type=int,
        default=30,
        help="Number of longest trajectories kept for interactive HTML plots.",
    )
    args = parser.parse_args()

    input_files = find_parquets(args.input, args.pattern)
    if args.max_files:
        input_files = input_files[: args.max_files]
    if not input_files:
        raise SystemExit(f"No parquet files found under {args.input}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_parquet = args.out_dir / "turn_timeseries.parquet"
    output_csv = args.out_dir / "turn_timeseries.csv"
    if output_parquet.exists():
        output_parquet.unlink()
    if output_csv.exists():
        output_csv.unlink()

    writer: pq.ParquetWriter | None = None
    csv_header = True
    rows = 0
    trajectories: set[str] = set()
    source_summaries = []
    sample_frames: list[pd.DataFrame] = []
    aggregate_frames: list[pd.DataFrame] = []

    for file_index, parquet_path in enumerate(input_files, start=1):
        print(f"[{file_index}/{len(input_files)}] reading {parquet_path}", flush=True)
        df = pd.read_parquet(parquet_path, columns=SOURCE_COLUMNS)
        metrics = extract_rows(df)
        if metrics.empty:
            continue

        metrics.insert(0, "source_file", str(parquet_path))
        writer = append_table(writer, output_parquet, metrics)

        if args.write_csv:
            metrics.to_csv(
                output_csv,
                mode="a",
                header=csv_header,
                index=False,
            )
            csv_header = False

        rows += len(metrics)
        trajectories.update(metrics["trajectory_id"].astype(str).unique())
        source_summaries.append(
            {
                "source_file": str(parquet_path),
                "source_trajectories": int(df["trajectory_id"].nunique()),
                "turn_rows": int(len(metrics)),
                "assistant_tokens_mean": float(metrics["assistant_tokens"].mean()),
                "macro_load_mean": float(metrics["macro_load_index"].mean()),
            }
        )

        top_ids = (
            metrics.groupby("trajectory_id")["turn_index"]
            .max()
            .sort_values(ascending=False)
            .head(max(args.plot_max_trajectories, 1))
            .index
        )
        sample_frames.append(metrics[metrics["trajectory_id"].isin(top_ids)].copy())

        tmp = metrics.copy()
        tmp["progress_bin"] = pd.cut(
            tmp["progress"], bins=np.linspace(0, 1, 31), include_lowest=True
        )
        aggregate_frames.append(
            tmp.groupby("progress_bin", observed=True)
            .agg(
                progress=("progress", "mean"),
                assistant_tokens=("assistant_tokens", "mean"),
                user_tokens_est=("user_tokens_est", "mean"),
                thinking_tokens_est=("thinking_tokens_est", "mean"),
                external_tool_call_count=("external_tool_call_count", "mean"),
                edit_count=("edit_count", "mean"),
                test_count=("test_count", "mean"),
                error_feedback_count=("error_feedback_count", "mean"),
                macro_load_index=("macro_load_index", "mean"),
                n=("trajectory_id", "size"),
            )
            .reset_index(drop=True)
        )

        del df, metrics

    if writer is not None:
        writer.close()

    if not output_parquet.exists():
        raise SystemExit("No metrics were written.")

    source_summary = pd.DataFrame(source_summaries)
    source_summary.to_csv(args.out_dir / "source_file_summary.csv", index=False)

    aggregate = pd.concat(aggregate_frames, ignore_index=True)
    aggregate["progress_bin"] = pd.cut(
        aggregate["progress"], bins=np.linspace(0, 1, 31), include_lowest=True
    )
    weighted_rows = []
    for _, group in aggregate.groupby("progress_bin", observed=True):
        weights = group["n"]
        weighted_rows.append(
            {
                "progress": float(np.average(group["progress"], weights=weights)),
                "assistant_tokens": float(
                    np.average(group["assistant_tokens"], weights=weights)
                ),
                "user_tokens_est": float(
                    np.average(group["user_tokens_est"], weights=weights)
                ),
                "thinking_tokens_est": float(
                    np.average(group["thinking_tokens_est"], weights=weights)
                ),
                "external_tool_call_count": float(
                    np.average(group["external_tool_call_count"], weights=weights)
                ),
                "edit_count": float(np.average(group["edit_count"], weights=weights)),
                "test_count": float(np.average(group["test_count"], weights=weights)),
                "error_feedback_count": float(
                    np.average(group["error_feedback_count"], weights=weights)
                ),
                "macro_load_index": float(
                    np.average(group["macro_load_index"], weights=weights)
                ),
                "n": int(weights.sum()),
            }
        )
    aggregate = pd.DataFrame(weighted_rows)
    aggregate.to_csv(args.out_dir / "aggregate_envelope.csv", index=False)
    write_aggregate_plot(aggregate, args.out_dir)

    if sample_frames:
        sample = pd.concat(sample_frames, ignore_index=True)
        top_ids = (
            sample.groupby("trajectory_id")["turn_index"]
            .max()
            .sort_values(ascending=False)
            .head(max(args.plot_max_trajectories, 1))
            .index
        )
        sample = sample[sample["trajectory_id"].isin(top_ids)].copy()
        sample.to_parquet(args.out_dir / "plot_sample_timeseries.parquet", index=False)
        write_visualizations(sample, args.out_dir)

    summary = {
        "input": str(args.input),
        "parquet_files": len(input_files),
        "trajectories": len(trajectories),
        "turn_rows": rows,
        "output_parquet": str(output_parquet),
        "duration_seconds_available": False,
        "notes": [
            "assistant_tokens uses labels != -100 spans.",
            "user_tokens_est and thinking_tokens_est are regex estimates from text.",
            "Full interactive HTML is intentionally limited to the longest sampled trajectories.",
        ],
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
