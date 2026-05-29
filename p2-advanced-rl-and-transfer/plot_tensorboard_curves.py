import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass
class RunSpec:
    label: str
    pattern: str
    color: str
    linewidth: float = 1.9


FIGURES: Dict[str, List[RunSpec]] = {
    "ppo_vs_sac": [
        RunSpec(
            label="PPO baseline",
            pattern="tensorboard_logs/ppo_push_none_source_dense_1000k_lr_0p0004_ent_0p0_seed_0_1",
            color="#6B7280",
            linewidth=1.9,
        ),
        RunSpec(
            label="SAC source",
            pattern="tensorboard_logs/sac_push_none_source_dense_1000k_lr_0p0004_ent_0p01_seed_0_1",
            color="#2563EB",
            linewidth=1.9,
        ),
        RunSpec(
            label="SAC target",
            pattern="tensorboard_logs/sac_push_none_target_dense_1000k_lr_0p0004_ent_0p01_seed_0_1",
            color="#B91C1C",
            linewidth=1.9,
        ),
    ],

    "sac_domain_randomization": [
        RunSpec(
            label="SAC source",
            pattern="tensorboard_logs/sac_push_none_source_dense_1000k_lr_0p0004_ent_auto_seed_0_1",
            color="#2563EB",
            linewidth=1.9,
        ),
        RunSpec(
            label="SAC + UDR [1, 6]",
            pattern="tensorboard_logs/sac_push_udr_source_dense_1000k_lr_0p0004_ent_auto_seed_0_1",
            color="#059669",
            linewidth=1.9,
        ),
        RunSpec(
            label="SAC + ADR [1, 6]",
            pattern="tensorboard_logs/sac_push_adr_source_dense_1000k_lr_0p0004_ent_auto_seed_0_1",
            color="#7C3AED",
            linewidth=1.9,
        ),
    ],

    "adr_range_stress_test": [
        RunSpec(
            label="ADR [1, 6]",
            pattern="tensorboard_logs/sac_push_adr_source_dense_1000k_lr_0p0004_ent_auto_seed_0_1",
            color="#2563EB",
            linewidth=1.9,
        ),
        RunSpec(
            label="ADR [1, 8]",
            pattern="tensorboard_logs/sac_push_adr_source_dense_initmass_1p0_1p5_limitmass_1p0_8p0_1000k_lr_0p0004_ent_auto_seed_0_1",
            color="#059669",
            linewidth=1.9,
        ),
        RunSpec(
            label="ADR [1, 10]",
            pattern="tensorboard_logs/sac_push_adr_source_dense_initmass_1p0_1p5_limitmass_1p0_10p0_1000k_lr_0p0004_ent_auto_seed_0_1",
            color="#7C3AED",
            linewidth=1.9,
        ),
    ],

    "entropy_ablation_standard": [
        RunSpec(
            label="SAC source, ent=auto",
            pattern="tensorboard_logs/sac_push_none_source_dense_1000k_lr_0p0004_ent_auto_seed_0_1",
            color="#2563EB",
            linewidth=1.9,
        ),
        RunSpec(
            label="SAC source, ent=0.01",
            pattern="tensorboard_logs/sac_push_none_source_dense_1000k_lr_0p0004_ent_0p01_seed_0_1",
            color="#F97316",
            linewidth=1.9,
        ),
        RunSpec(
            label="SAC target, ent=auto",
            pattern="tensorboard_logs/sac_push_none_target_dense_1000k_lr_0p0004_ent_auto_seed_0_1",
            color="#7C3AED",
            linewidth=1.9,
        ),
        RunSpec(
            label="SAC target, ent=0.01",
            pattern="tensorboard_logs/sac_push_none_target_dense_1000k_lr_0p0004_ent_0p01_seed_0_1",
            color="#B91C1C",
            linewidth=1.9,
        ),
    ],

    "entropy_ablation_randomization": [
        RunSpec(
            label="UDR [1, 6], ent=auto",
            pattern="tensorboard_logs/sac_push_udr_source_dense_1000k_lr_0p0004_ent_auto_seed_0_1",
            color="#059669",
            linewidth=1.9,
        ),
        RunSpec(
            label="UDR [1, 6], ent=0.01",
            pattern="tensorboard_logs/sac_push_udr_source_dense_mass_1p0_6p0_1000k_lr_0p0004_ent_0p01_seed_0_1",
            color="#DC2626",
            linewidth=1.9,
        ),
        RunSpec(
            label="ADR [1, 6], ent=auto",
            pattern="tensorboard_logs/sac_push_adr_source_dense_1000k_lr_0p0004_ent_auto_seed_0_1",
            color="#2563EB",
            linewidth=1.9,
        ),
        RunSpec(
            label="ADR [1, 6], ent=0.01",
            pattern="tensorboard_logs/sac_push_adr_source_dense_initmass_1p0_1p5_limitmass_1p0_6p0_1000k_lr_0p0004_ent_0p01_seed_0_1",
            color="#7C3AED",
            linewidth=1.9,
        ),
    ],
}


TAGS_TO_PLOT = [
    ("rollout/success_rate", "Success rate", "success_rate"),
    ("rollout/ep_rew_mean", "Mean episode reward", "mean_reward"),
    ("rollout/ep_len_mean", "Mean episode length", "episode_length"),
]


def find_event_file(pattern: str) -> str:
    candidate_paths = glob.glob(pattern)

    if not candidate_paths:
        raise FileNotFoundError(f"No folders matched pattern: {pattern}")

    event_files = []

    for path in candidate_paths:
        if os.path.isfile(path) and "events.out.tfevents" in os.path.basename(path):
            event_files.append(path)
        else:
            event_files.extend(
                glob.glob(
                    os.path.join(path, "**", "events.out.tfevents.*"),
                    recursive=True,
                )
            )

    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found for pattern: {pattern}")

    event_files = sorted(event_files, key=os.path.getmtime, reverse=True)
    return event_files[0]


def read_scalar(event_file: str, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    accumulator = EventAccumulator(event_file)
    accumulator.Reload()

    available_tags = accumulator.Tags().get("scalars", [])

    if tag not in available_tags:
        raise KeyError(
            f"Tag '{tag}' not found in {event_file}.\n"
            f"Available scalar tags:\n{available_tags}"
        )

    events = accumulator.Scalars(tag)

    steps = np.array([event.step for event in events], dtype=float)
    values = np.array([event.value for event in events], dtype=float)

    return steps, values

def smooth_curve(values: np.ndarray, smoothing: float) -> np.ndarray:
    if len(values) == 0:
        return values

    if smoothing <= 0:
        return values

    smoothed = np.zeros_like(values, dtype=float)
    smoothed[0] = values[0]

    for i in range(1, len(values)):
        smoothed[i] = smoothing * smoothed[i - 1] + (1.0 - smoothing) * values[i]

    return smoothed


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values

    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def interpolate_curve(
    steps: np.ndarray,
    values: np.ndarray,
    num_points: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(steps) < 2:
        return steps, values

    unique_steps, unique_indices = np.unique(steps, return_index=True)
    unique_values = values[unique_indices]

    dense_steps = np.linspace(unique_steps.min(), unique_steps.max(), num_points)
    dense_values = np.interp(dense_steps, unique_steps, unique_values)

    return dense_steps, dense_values


def cut_curve(
    steps: np.ndarray,
    values: np.ndarray,
    max_step: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    if max_step is None:
        return steps, values

    mask = steps <= max_step
    return steps[mask], values[mask]

def style_axis(ax, tag: str) -> None:
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.set_facecolor("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", labelsize=10)
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)

    ax.set_xlabel("Timesteps (×10⁶)")

    if tag == "rollout/success_rate":
        ax.set_ylabel("Success rate")
        ax.set_ylim(-0.02, 1.05)

    elif tag == "rollout/ep_rew_mean":
        ax.set_ylabel("Mean episode reward")

    elif tag == "rollout/ep_len_mean":
        ax.set_ylabel("Mean episode length")


def plot_run_on_axis(
    ax,
    run: RunSpec,
    tag: str,
    smoothing: float,
    max_step: Optional[float],
    interpolation_points: int,
    raw_alpha: float,
    raw_linewidth: float,
    raw_ma_window: int,
) -> None:
    event_file = find_event_file(run.pattern)

    steps, raw_values = read_scalar(event_file, tag)
    steps, raw_values = cut_curve(steps, raw_values, max_step)

    if len(steps) == 0:
        raise ValueError(f"No points left after max_step filtering for run {run.label}")

    # Raw/background curve
    raw_display_values = moving_average(raw_values, raw_ma_window)

    raw_steps_dense, raw_values_dense = interpolate_curve(
        steps=steps,
        values=raw_display_values,
        num_points=interpolation_points,
    )

    ax.plot(
        raw_steps_dense / 1_000_000,
        raw_values_dense,
        color=run.color,
        linewidth=raw_linewidth,
        alpha=raw_alpha,
        linestyle="-",
        solid_capstyle="round",
        solid_joinstyle="round",
        antialiased=True,
    )

    smooth_values = smooth_curve(raw_values, smoothing)

    smooth_steps_dense, smooth_values_dense = interpolate_curve(
        steps=steps,
        values=smooth_values,
        num_points=interpolation_points,
    )

    ax.plot(
        smooth_steps_dense / 1_000_000,
        smooth_values_dense,
        label=run.label,
        color=run.color,
        linewidth=run.linewidth,
        alpha=1.0,
        linestyle="-",
        solid_capstyle="round",
        solid_joinstyle="round",
        antialiased=True,
    )

def plot_single_metric_figure(
    figure_name: str,
    runs: List[RunSpec],
    tag: str,
    title: str,
    metric_slug: str,
    output_dir: str,
    smoothing: float,
    max_step: Optional[float],
    interpolation_points: int,
    dpi: int,
    raw_alpha: float,
    raw_linewidth: float,
    raw_ma_window: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    fig.patch.set_facecolor("white")

    for run in runs:
        try:
            plot_run_on_axis(
                ax=ax,
                run=run,
                tag=tag,
                smoothing=smoothing,
                max_step=max_step,
                interpolation_points=interpolation_points,
                raw_alpha=raw_alpha,
                raw_linewidth=raw_linewidth,
                raw_ma_window=raw_ma_window,
            )
        except Exception as error:
            print(f"[WARNING] Skipping '{run.label}' for tag '{tag}': {error}")

    #ax.set_title(title)
    style_axis(ax, tag)

    ax.legend(
        frameon=True,
        fontsize=9,
        loc="best",
    )

    fig.tight_layout()

    png_path = os.path.join(output_dir, f"{figure_name}_{metric_slug}.png")
    pdf_path = os.path.join(output_dir, f"{figure_name}_{metric_slug}.pdf")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot clean TensorBoard curves for the RL report."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="report_figures",
        help="Directory where PNG and PDF figures are saved.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.95,
        help="Exponential smoothing factor. Use 0 for raw curves.",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=None,
        help="Optional maximum timestep to display.",
    )
    parser.add_argument(
        "--interpolation-points",
        type=int,
        default=2500,
        help="Number of interpolation points for smoother curves.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for PNG output.",
    )
    parser.add_argument(
        "--include-episode-length",
        action="store_true",
        help="Also save episode length figures.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=list(FIGURES.keys()),
        help="Plot only one figure group.",
    )
    parser.add_argument(
        "--raw-alpha",
        type=float,
        default=0.18,
        help="Transparency of the raw background curve.",
    )
    parser.add_argument(
        "--raw-linewidth",
        type=float,
        default=0.8,
        help="Line width of the raw background curve.",
    )
    parser.add_argument(
        "--raw-ma-window",
        type=int,
        default=3,
        help="Small moving-average window for the raw background curve.",
    )

    args = parser.parse_args()

    selected_figures = FIGURES

    if args.only is not None:
        selected_figures = {args.only: FIGURES[args.only]}

    default_tags = TAGS_TO_PLOT[:2]

    for figure_name, runs in selected_figures.items():
        for tag, title, metric_slug in default_tags:
            plot_single_metric_figure(
                figure_name=figure_name,
                runs=runs,
                tag=tag,
                title=title,
                metric_slug=metric_slug,
                output_dir=args.output_dir,
                smoothing=args.smoothing,
                max_step=args.max_step,
                interpolation_points=args.interpolation_points,
                dpi=args.dpi,
                raw_alpha=args.raw_alpha,
                raw_linewidth=args.raw_linewidth,
                raw_ma_window=args.raw_ma_window,
            )

        if args.include_episode_length:
            tag, title, metric_slug = TAGS_TO_PLOT[2]

            plot_single_metric_figure(
                figure_name=figure_name,
                runs=runs,
                tag=tag,
                title=title,
                metric_slug=metric_slug,
                output_dir=args.output_dir,
                smoothing=args.smoothing,
                max_step=args.max_step,
                interpolation_points=args.interpolation_points,
                dpi=args.dpi,
                raw_alpha=args.raw_alpha,
                raw_linewidth=args.raw_linewidth,
                raw_ma_window=args.raw_ma_window,
            )


if __name__ == "__main__":
    main()