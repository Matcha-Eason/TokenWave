#!/usr/bin/env python3
"""Extract per-turn time-series metrics from OpenHands-style trajectory parquet."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MESSAGE_RE = re.compile(r"<\|im_start\|>(\w+)\n")
FUNCTION_RE = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
TOKEN_RE = re.compile(
    r"[\u4e00-\u9fff]|[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\s]",
    re.UNICODE,
)
ERROR_RE = re.compile(
    r"Traceback|AssertionError|Exception|CancelledError|"
    r"\bERROR\b|\bFAILED\b|\bFAIL\b|failed|error|"
    r"exit code [1-9]\d*|timed out|Command timed out",
    re.IGNORECASE,
)
TEST_RE = re.compile(
    r"\bpytest\b|\bunittest\b|\btox\b|\bnox\b|"
    r"\btest[_\w-]*\.py\b|python(?:3)?\s+[^&|;\n]*test",
    re.IGNORECASE,
)

EDIT_COMMANDS = {"create", "str_replace", "insert", "undo_edit"}
NON_EXTERNAL_TOOLS = {"think", "finish", "example_function_name", "..."}


def token_estimate(text: str) -> int:
    if not text:
        return 0
    return len(TOKEN_RE.findall(text))


def parse_messages(text: str) -> list[dict[str, str]]:
    markers = list(MESSAGE_RE.finditer(text))
    messages: list[dict[str, str]] = []
    for i, marker in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        content = text[marker.end() : end].replace("<|im_end|>", "").strip()
        messages.append({"role": marker.group(1), "content": content})
    return messages


def assistant_label_spans(labels: Any) -> list[int]:
    arr = np.asarray(labels)
    if arr.size == 0:
        return []
    mask = arr != -100
    if not mask.any():
        return []
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    ends = np.flatnonzero(mask & np.r_[~mask[1:], True]) + 1
    return [int(end - start) for start, end in zip(starts, ends)]


def parse_tool_calls(assistant_content: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for match in FUNCTION_RE.finditer(assistant_content):
        function = match.group(1).strip()
        body = match.group(2)
        params = {
            name.strip(): value.strip()
            for name, value in PARAM_RE.findall(body)
        }
        calls.append({"function": function, "params": params})
    return calls


def first_param(calls: list[dict[str, Any]], function: str, param: str) -> list[str]:
    values = []
    for call in calls:
        if call["function"] == function and param in call["params"]:
            values.append(str(call["params"][param]))
    return values


def command_texts(calls: list[dict[str, Any]]) -> list[str]:
    texts = []
    for call in calls:
        params = call["params"]
        if "command" in params:
            texts.append(str(params["command"]))
    return texts


def extract_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trajectory_ordinal, record in df.reset_index(drop=True).iterrows():
        trajectory_id = record["trajectory_id"]
        reward = float(record["reward"]) if "reward" in record else math.nan
        text = record["chat_template_applied"]
        messages = parse_messages(text)
        assistant_spans = assistant_label_spans(record["labels"])
        assistant_idx = 0

        for message_index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue

            assistant_content = message["content"]
            prev_user = (
                messages[message_index - 1]["content"]
                if message_index > 0 and messages[message_index - 1]["role"] == "user"
                else ""
            )
            next_user = (
                messages[message_index + 1]["content"]
                if message_index + 1 < len(messages)
                and messages[message_index + 1]["role"] == "user"
                else ""
            )
            next_is_tool_response = next_user.lstrip().startswith("<tool_response>")

            calls = parse_tool_calls(assistant_content)
            functions = [call["function"] for call in calls]
            function_counts = Counter(functions)
            commands = command_texts(calls)
            thinking_text = "\n".join(first_param(calls, "think", "thought"))

            edit_count = 0
            for call in calls:
                if call["function"] != "str_replace_editor":
                    continue
                command = str(call["params"].get("command", "")).strip()
                if command in EDIT_COMMANDS:
                    edit_count += 1

            test_count = sum(1 for command in commands if TEST_RE.search(command))
            error_feedback_count = (
                len(ERROR_RE.findall(next_user)) if next_is_tool_response else 0
            )
            external_tool_call_count = sum(
                1 for fn in functions if fn not in NON_EXTERNAL_TOOLS
            )
            assistant_tokens_exact = (
                assistant_spans[assistant_idx]
                if assistant_idx < len(assistant_spans)
                else math.nan
            )

            rows.append(
                {
                    "trajectory_ordinal": trajectory_ordinal,
                    "trajectory_id": trajectory_id,
                    "reward": reward,
                    "turn_index": assistant_idx,
                    "message_index": message_index,
                    "user_tokens_est": token_estimate(prev_user),
                    "assistant_tokens": assistant_tokens_exact,
                    "assistant_tokens_est": token_estimate(assistant_content),
                    "thinking_tokens_est": token_estimate(thinking_text),
                    "tool_call_count": len(calls),
                    "external_tool_call_count": external_tool_call_count,
                    "think_call_count": function_counts.get("think", 0),
                    "bash_call_count": function_counts.get("execute_bash", 0),
                    "editor_call_count": function_counts.get("str_replace_editor", 0),
                    "finish_call_count": function_counts.get("finish", 0),
                    "edit_count": edit_count,
                    "test_count": test_count,
                    "duration_seconds": math.nan,
                    "error_feedback_count": error_feedback_count,
                    "next_is_tool_response": next_is_tool_response,
                    "assistant_text_chars": len(assistant_content),
                    "user_text_chars": len(prev_user),
                    "thinking_text_chars": len(thinking_text),
                }
            )
            assistant_idx += 1

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["progress"] = out.groupby("trajectory_id")["turn_index"].transform(
        lambda s: s / max(float(s.max()), 1.0)
    )
    out["macro_load_index"] = (
        np.log1p(out["assistant_tokens"].fillna(out["assistant_tokens_est"]))
        + np.log1p(out["thinking_tokens_est"])
        + 0.8 * out["external_tool_call_count"]
        + 1.2 * out["edit_count"]
        + 1.0 * out["test_count"]
        + 0.6 * out["error_feedback_count"]
    )
    out["assistant_tokens_ma5"] = out.groupby("trajectory_id")[
        "assistant_tokens"
    ].transform(lambda s: s.rolling(5, min_periods=1, center=True).mean())
    out["macro_load_ma5"] = out.groupby("trajectory_id")["macro_load_index"].transform(
        lambda s: s.rolling(5, min_periods=1, center=True).mean()
    )
    return out


def write_visualizations(metrics: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    out_dir.mkdir(parents=True, exist_ok=True)

    longest_id = (
        metrics.groupby("trajectory_id")["turn_index"].max().sort_values().index[-1]
    )
    longest = metrics[metrics["trajectory_id"] == longest_id]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(longest["turn_index"], longest["assistant_tokens"], label="assistant")
    axes[0].plot(longest["turn_index"], longest["user_tokens_est"], label="user est")
    axes[0].plot(
        longest["turn_index"],
        longest["thinking_tokens_est"],
        label="thinking est",
    )
    axes[0].set_ylabel("tokens")
    axes[0].legend(loc="upper right")

    axes[1].plot(longest["turn_index"], longest["external_tool_call_count"])
    axes[1].set_ylabel("tool calls")

    axes[2].plot(longest["turn_index"], longest["edit_count"], label="edits")
    axes[2].plot(longest["turn_index"], longest["test_count"], label="tests")
    axes[2].plot(
        longest["turn_index"],
        longest["error_feedback_count"],
        label="error feedback",
    )
    axes[2].set_ylabel("counts")
    axes[2].legend(loc="upper right")

    axes[3].plot(longest["turn_index"], longest["macro_load_index"], alpha=0.35)
    axes[3].plot(longest["turn_index"], longest["macro_load_ma5"])
    axes[3].set_ylabel("load index")
    axes[3].set_xlabel("turn")
    fig.suptitle(f"Longest trajectory: {longest_id}")
    fig.tight_layout()
    fig.savefig(out_dir / "longest_trajectory_timeseries.png", dpi=180)
    plt.close(fig)

    aggregate = metrics.copy()
    aggregate["progress_bin"] = pd.cut(
        aggregate["progress"], bins=np.linspace(0, 1, 31), include_lowest=True
    )
    aggregate = (
        aggregate.groupby("progress_bin", observed=True)
        .agg(
            progress=("progress", "mean"),
            assistant_tokens=("assistant_tokens", "mean"),
            thinking_tokens_est=("thinking_tokens_est", "mean"),
            external_tool_call_count=("external_tool_call_count", "mean"),
            edit_count=("edit_count", "mean"),
            test_count=("test_count", "mean"),
            error_feedback_count=("error_feedback_count", "mean"),
            macro_load_index=("macro_load_index", "mean"),
        )
        .reset_index(drop=True)
    )

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

    metric_names = [
        "assistant_tokens",
        "user_tokens_est",
        "thinking_tokens_est",
        "external_tool_call_count",
        "edit_count",
        "test_count",
        "error_feedback_count",
        "macro_load_index",
    ]
    subplot_titles = [
        "assistant tokens",
        "user tokens est",
        "thinking tokens est",
        "external tool calls",
        "edits",
        "tests",
        "error feedback",
        "macro load index",
    ]
    fig = make_subplots(
        rows=len(metric_names),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        subplot_titles=subplot_titles,
    )

    trajectory_ids = list(metrics["trajectory_id"].drop_duplicates())
    visible = []
    for trajectory_position, trajectory_id in enumerate(trajectory_ids):
        subset = metrics[metrics["trajectory_id"] == trajectory_id]
        is_visible = trajectory_position == 0
        visible.extend([is_visible] * len(metric_names))
        for row, metric in enumerate(metric_names, start=1):
            fig.add_trace(
                go.Scatter(
                    x=subset["turn_index"],
                    y=subset[metric],
                    mode="lines+markers",
                    name=metric,
                    legendgroup=metric,
                    showlegend=trajectory_position == 0,
                    visible=is_visible,
                ),
                row=row,
                col=1,
            )

    buttons = []
    for trajectory_position, trajectory_id in enumerate(trajectory_ids):
        mask = [False] * (len(trajectory_ids) * len(metric_names))
        start = trajectory_position * len(metric_names)
        for i in range(len(metric_names)):
            mask[start + i] = True
        buttons.append(
            {
                "label": str(trajectory_id)[:80],
                "method": "update",
                "args": [
                    {"visible": mask},
                    {"title": f"Turn time series: {trajectory_id}"},
                ],
            }
        )

    fig.update_layout(
        title=f"Turn time series: {trajectory_ids[0] if trajectory_ids else ''}",
        height=1500,
        width=1250,
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.0,
                "y": 1.04,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
    )
    fig.update_xaxes(title_text="turn", row=len(metric_names), col=1)
    fig.write_html(out_dir / "turn_timeseries_visualization.html")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="codeforge-datatest/R2E_Gym-00000-of-00224.parquet",
        type=Path,
    )
    parser.add_argument("--out-dir", default="outputs", type=Path)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    metrics = extract_rows(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(args.out_dir / "turn_timeseries.parquet", index=False)
    metrics.to_csv(args.out_dir / "turn_timeseries.csv", index=False)

    summary = {
        "source": str(args.input),
        "trajectories": int(metrics["trajectory_id"].nunique()) if not metrics.empty else 0,
        "turns": int(len(metrics)),
        "duration_seconds_available": False,
        "assistant_tokens": "exact label span length from parquet labels",
        "user_tokens_est": "regex estimate from message text; no tokenizer offsets in source parquet",
        "thinking_tokens_est": "regex estimate from function=think thought text",
    }
    pd.Series(summary).to_json(args.out_dir / "summary.json", force_ascii=False, indent=2)
    write_visualizations(metrics, args.out_dir)
    print(summary)
    print(f"wrote {args.out_dir / 'turn_timeseries.parquet'}")
    print(f"wrote {args.out_dir / 'turn_timeseries_visualization.html'}")


if __name__ == "__main__":
    main()
