"""
plot_results.py

Reads training/evaluation logs from:
    results/<run_name>/training_log.csv
    results/<run_name>/final_evaluation.csv

Creates:
    plots/training_curves.png
    plots/evaluation_curves.png
    plots/final_eval_barplot.png
    plots/training_time_barplot.png
    plots/summary_results.csv

Usage: python plot_results.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")


METHOD_ORDER = [
    "REINFORCE (baseline=0)",
    "REINFORCE (baseline=20)",
    "REINFORCE (baseline=50)",
    "REINFORCE (baseline=100)",
    "Actor-Critic one-step",
    "Actor-Critic n-step (n=10)",
]


def infer_label(row_or_name) -> str:
    """
    Creates a clean label from run_name or dataframe row.
    """

    if isinstance(row_or_name, str):
        run_name = row_or_name

        if run_name.startswith("reinforce_baseline_0"):
            return "REINFORCE (baseline=0)"

        if run_name.startswith("reinforce_baseline_20"):
            return "REINFORCE (baseline=20)"

        if run_name.startswith("reinforce_baseline_50"):
            return "REINFORCE (baseline=50)"

        if run_name.startswith("reinforce_baseline_100"):
            return "REINFORCE (baseline=100)"

        if run_name.startswith("actor_critic_nstep_10"):
            return "Actor-Critic n-step (n=10)"

        if run_name.startswith("actor_critic"):
            return "Actor-Critic one-step"

        return run_name

    algo = row_or_name.get("algo", "")
    baseline = row_or_name.get("baseline", "")
    run_name = row_or_name.get("run_name", "")

    if isinstance(run_name, str):
        if run_name.startswith("actor_critic_nstep_10"):
            return "Actor-Critic n-step (n=10)"
        if run_name.startswith("actor_critic"):
            return "Actor-Critic one-step"
        if run_name.startswith("reinforce_baseline_0"):
            return "REINFORCE (baseline=0)"
        if run_name.startswith("reinforce_baseline_20"):
            return "REINFORCE (baseline=20)"
        if run_name.startswith("reinforce_baseline_50"):
            return "REINFORCE (baseline=50)"
        if run_name.startswith("reinforce_baseline_100"):
            return "REINFORCE (baseline=100)"

    if algo == "reinforce":
        return f"REINFORCE (baseline={baseline:g})"

    if algo == "actor_critic":
        return "Actor-Critic"

    return str(algo)


def plot_reinforce_baseline_ablation(train_df: pd.DataFrame, output_path: Path) -> None:
    df = train_df[
        train_df["method"].isin([
            "REINFORCE (baseline=0)",
            "REINFORCE (baseline=20)",
            "REINFORCE (baseline=50)",
            "REINFORCE (baseline=100)",
        ])
    ].copy()

    df = df.dropna(subset=["eval_mean_return"])

    if df.empty:
        print("No REINFORCE evaluation data found. Skipping baseline ablation plot.")
        return

    agg = aggregate_curve(df, "episode", "eval_mean_return")

    plt.figure(figsize=(11, 6))

    for method in get_sorted_methods(agg):
        method_df = agg[agg["method"] == method].sort_values("episode")

        x = method_df["episode"]
        y = method_df["mean"]
        std = method_df["std"].fillna(0.0)

        plt.plot(x, y, marker="o", markersize=3, label=method)
        plt.fill_between(x, y - std, y + std, alpha=0.15)

    plt.xlabel("Training episode")
    plt.ylabel("Mean evaluation return")
    plt.title("REINFORCE baseline ablation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def method_sort_key(method: str) -> int:
    if method in METHOD_ORDER:
        return METHOD_ORDER.index(method)
    return len(METHOD_ORDER)


def load_training_logs(results_dir: Path) -> pd.DataFrame:
    dfs = []

    for log_path in results_dir.glob("*/training_log.csv"):
        run_name = log_path.parent.name
        df = pd.read_csv(log_path)

        df["run_name"] = run_name
        df["method"] = infer_label(run_name)

        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No training_log.csv files found in {results_dir}/<run_name>/")

    return pd.concat(dfs, ignore_index=True)


def load_final_evaluations(results_dir: Path) -> pd.DataFrame:
    dfs = []

    for eval_path in results_dir.glob("*/final_evaluation.csv"):
        run_name = eval_path.parent.name
        df = pd.read_csv(eval_path)

        df["run_name"] = run_name
        df["method"] = df.apply(infer_label, axis=1)

        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No final_evaluation.csv files found in {results_dir}/<run_name>/")

    return pd.concat(dfs, ignore_index=True)


def add_moving_average(df: pd.DataFrame, value_col: str, window: int = 25) -> pd.DataFrame:
    df = df.copy()
    df[f"{value_col}_ma"] = (
        df.groupby("run_name")[value_col]
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )
    return df


def aggregate_curve(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(["method", x_col])[y_col]
        .agg(["mean", "std"])
        .reset_index()
    )
    return grouped


def get_sorted_methods(df: pd.DataFrame) -> list[str]:
    methods = sorted(df["method"].dropna().unique(), key=method_sort_key)
    return methods


def plot_training_curves(train_df: pd.DataFrame, output_path: Path) -> None:
    df = add_moving_average(train_df, "episode_return", window=25)
    agg = aggregate_curve(df, "episode", "episode_return_ma")

    plt.figure(figsize=(11, 6))

    for method in get_sorted_methods(agg):
        method_df = agg[agg["method"] == method].sort_values("episode")

        x = method_df["episode"]
        y = method_df["mean"]
        std = method_df["std"].fillna(0.0)

        plt.plot(x, y, label=method)
        plt.fill_between(x, y - std, y + std, alpha=0.15)

    plt.xlabel("Training episode")
    plt.ylabel("Episode return, moving average")
    plt.title("Training return curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_evaluation_curves(train_df: pd.DataFrame, output_path: Path) -> None:
    df = train_df.dropna(subset=["eval_mean_return"]).copy()

    if df.empty:
        print("No evaluation data found. Skipping evaluation curve.")
        return

    agg = aggregate_curve(df, "episode", "eval_mean_return")

    plt.figure(figsize=(11, 6))

    for method in get_sorted_methods(agg):
        method_df = agg[agg["method"] == method].sort_values("episode")

        x = method_df["episode"]
        y = method_df["mean"]
        std = method_df["std"].fillna(0.0)

        plt.plot(x, y, marker="o", markersize=3, label=method)
        plt.fill_between(x, y - std, y + std, alpha=0.15)

    plt.xlabel("Training episode")
    plt.ylabel("Mean evaluation return")
    plt.title("Evaluation return during training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_final_eval_barplot(final_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    summary = (
        final_df.groupby("method")
        .agg(
            mean_final_return=("final_eval_mean_return", "mean"),
            std_final_return=("final_eval_mean_return", "std"),
            mean_eval_std=("final_eval_std_return", "mean"),
            mean_training_time_sec=("total_training_time_sec", "mean"),
            std_training_time_sec=("total_training_time_sec", "std"),
            n_runs=("run_name", "count"),
        )
        .reset_index()
    )

    summary["std_final_return"] = summary["std_final_return"].fillna(0.0)
    summary["std_training_time_sec"] = summary["std_training_time_sec"].fillna(0.0)
    summary["sort_key"] = summary["method"].apply(method_sort_key)
    summary = summary.sort_values("sort_key").drop(columns=["sort_key"])

    plt.figure(figsize=(10, 6))
    plt.bar(
        summary["method"],
        summary["mean_final_return"],
        yerr=summary["std_final_return"],
        capsize=5,
    )

    plt.xlabel("Algorithm")
    plt.ylabel("Final evaluation return")
    plt.title("Final evaluation performance")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return summary


def plot_training_time_barplot(summary: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.bar(
        summary["method"],
        summary["mean_training_time_sec"],
        yerr=summary["std_training_time_sec"].fillna(0.0),
        capsize=5,
    )

    plt.xlabel("Algorithm")
    plt.ylabel("Training time [seconds]")
    plt.title("Training time comparison")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_training_logs(RESULTS_DIR)
    final_df = load_final_evaluations(RESULTS_DIR)

    numeric_cols = [
        "episode",
        "episode_return",
        "episode_steps",
        "episode_time_sec",
        "elapsed_time_sec",
        "eval_mean_return",
        "eval_std_return",
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "mean_return",
        "mean_advantage",
        "advantage",
    ]

    for col in numeric_cols:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

    final_numeric_cols = [
        "baseline",
        "seed",
        "episodes",
        "final_eval_episodes",
        "final_eval_mean_return",
        "final_eval_std_return",
        "total_training_time_sec",
    ]

    for col in final_numeric_cols:
        if col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors="coerce")

    plot_training_curves(train_df, PLOTS_DIR / "training_curves.png")
    plot_evaluation_curves(train_df, PLOTS_DIR / "evaluation_curves.png")

    summary = plot_final_eval_barplot(final_df, PLOTS_DIR / "final_eval_barplot.png")
    plot_training_time_barplot(summary, PLOTS_DIR / "training_time_barplot.png")

    plot_reinforce_baseline_ablation(train_df, PLOTS_DIR / "reinforce_baseline_ablation.png")

    summary_path = PLOTS_DIR / "summary_results.csv"
    summary.to_csv(summary_path, index=False)

    print("\nPlots created:")
    print(f"- {PLOTS_DIR / 'training_curves.png'}")
    print(f"- {PLOTS_DIR / 'evaluation_curves.png'}")
    print(f"- {PLOTS_DIR / 'final_eval_barplot.png'}")
    print(f"- {PLOTS_DIR / 'training_time_barplot.png'}")
    print(f"- {PLOTS_DIR / 'reinforce_baseline_ablation.png'}")
    print(f"- {summary_path}")

    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()