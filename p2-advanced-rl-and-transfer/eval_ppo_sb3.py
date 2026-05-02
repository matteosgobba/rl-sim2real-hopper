import argparse
import csv
import os
import random

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401
import torch
from stable_baselines3 import PPO


def evaluate(
    model_path: str,
    n_episodes: int,
    deterministic: bool,
    render: bool,
    env_type: str,
    seed: int,
    results_path: str,
) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure you saved your trained model with model.save(...)."
        )

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    render_mode = "human" if render else "rgb_array"

    env = gym.make(
        "PandaPush-v3",
        render_mode=render_mode,
        type=env_type,
        reward_type="dense",
    )

    model = PPO.load(model_path, env=env)

    episode_returns = []
    successes = []

    for episode in range(1, n_episodes + 1):
        obs, info = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        episode_return = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)

        episode_returns.append(episode_return)

        if isinstance(info, dict) and "is_success" in info:
            successes.append(float(info["is_success"]))

        print(f"Episode {episode:03d} | return = {episode_return:.3f}")

    env.close()

    returns = np.array(episode_returns, dtype=np.float32)
    mean_return = float(returns.mean())
    std_return = float(returns.std())
    min_return = float(returns.min())
    max_return = float(returns.max())
    success_rate = float(np.mean(successes)) if successes else None

    print("\n=== Evaluation summary ===")
    print(f"Algorithm: PPO")
    print(f"Environment type: {env_type}")
    print(f"Episodes: {n_episodes}")
    print(f"Mean return: {mean_return:.3f}")
    print(f"Std return:  {std_return:.3f}")
    print(f"Min return:  {min_return:.3f}")
    print(f"Max return:  {max_return:.3f}")

    if success_rate is not None:
        print(f"Success rate: {success_rate:.2%}")

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(results_path)

    with open(results_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "algorithm",
                "model_path",
                "env_type",
                "n_episodes",
                "seed",
                "deterministic",
                "mean_return",
                "std_return",
                "min_return",
                "max_return",
                "success_rate",
            ])

        writer.writerow([
            "PPO",
            model_path,
            env_type,
            n_episodes,
            seed,
            deterministic,
            mean_return,
            std_return,
            min_return,
            max_return,
            success_rate if success_rate is not None else "",
        ])

    print(f"\nResults saved to: {results_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO on PandaPush-v3")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a PPO model zip file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render with a window",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="target",
        choices=["source", "target"],
        help="Type of environment to evaluate on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="results/sb3_results.csv",
        help="Path to the CSV results file",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluate(
        model_path=args.model_path,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
        env_type=args.env_type,
        seed=args.seed,
        results_path=args.results_path,
    )