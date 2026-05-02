import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def load_evaluations(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"File not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    print("\nAvailable keys in file:")
    for key in data.files:
        print(f"- {key}")

    timesteps = data["timesteps"]
    results = data["results"]

    mean_returns = results.mean(axis=1)
    std_returns = results.std(axis=1)

    successes = None
    mean_success = None
    std_success = None

    if "successes" in data.files:
        successes = data["successes"]

        # successes may be object arrays depending on SB3/env
        try:
            success_array = np.array([np.mean(s) for s in successes], dtype=float)
            mean_success = success_array
            std_success = np.array([np.std(s) for s in successes], dtype=float)
        except Exception:
            print("Warning: could not parse successes correctly.")

    return timesteps, mean_returns, std_returns, mean_success, std_success


def plot_returns(timesteps, mean_returns, std_returns, title, output_path=None):
    plt.figure(figsize=(8, 5))

    plt.plot(timesteps, mean_returns, marker="o", label="Mean eval return")
    plt.fill_between(
        timesteps,
        mean_returns - std_returns,
        mean_returns + std_returns,
        alpha=0.2,
        label="±1 std",
    )

    best_idx = int(np.argmax(mean_returns))
    best_timestep = timesteps[best_idx]
    best_return = mean_returns[best_idx]

    plt.scatter([best_timestep], [best_return], marker="x", s=100, label="Best checkpoint")

    plt.xlabel("Training timesteps")
    plt.ylabel("Evaluation return")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved return plot to: {output_path}")

    plt.show()

    print("\nBest checkpoint according to mean return:")
    print(f"Timestep: {best_timestep}")
    print(f"Mean return: {best_return:.3f}")
    print(f"Std return: {std_returns[best_idx]:.3f}")


def plot_success_rate(timesteps, mean_success, std_success, title, output_path=None):
    if mean_success is None:
        print("\nNo success-rate data found in evaluations.npz.")
        return

    plt.figure(figsize=(8, 5))

    plt.plot(timesteps, mean_success, marker="o", label="Mean success rate")

    if std_success is not None:
        plt.fill_between(
            timesteps,
            mean_success - std_success,
            mean_success + std_success,
            alpha=0.2,
            label="±1 std",
        )

    best_idx = int(np.argmax(mean_success))
    best_timestep = timesteps[best_idx]
    best_success = mean_success[best_idx]

    plt.scatter([best_timestep], [best_success], marker="x", s=100, label="Best checkpoint")

    plt.xlabel("Training timesteps")
    plt.ylabel("Success rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved success-rate plot to: {output_path}")

    plt.show()

    print("\nBest checkpoint according to success rate:")
    print(f"Timestep: {best_timestep}")
    print(f"Success rate: {best_success:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training/evaluation curves from SB3 EvalCallback evaluations.npz"
    )

    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to evaluations.npz file",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Evaluation curve",
        help="Plot title",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Optional directory where plots are saved",
    )

    args = parser.parse_args()

    timesteps, mean_returns, std_returns, mean_success, std_success = load_evaluations(args.file)

    print("\nEvaluation summary:")
    print(f"Number of evaluations: {len(timesteps)}")
    print(f"First timestep: {timesteps[0]}")
    print(f"Last timestep: {timesteps[-1]}")
    print(f"Best mean return: {mean_returns.max():.3f}")
    print(f"Best mean return timestep: {timesteps[int(np.argmax(mean_returns))]}")

    return_plot_path = None
    success_plot_path = None

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        return_plot_path = os.path.join(args.save_dir, f"{base_name}_returns.png")
        success_plot_path = os.path.join(args.save_dir, f"{base_name}_success.png")

    plot_returns(
        timesteps=timesteps,
        mean_returns=mean_returns,
        std_returns=std_returns,
        title=args.title + " - Return",
        output_path=return_plot_path,
    )

    plot_success_rate(
        timesteps=timesteps,
        mean_success=mean_success,
        std_success=std_success,
        title=args.title + " - Success Rate",
        output_path=success_plot_path,
    )


if __name__ == "__main__":
    main()