import argparse
import os
import random

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401
import torch
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback

from rand_wrapper import RandomizationWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on PandaPush-v3")

    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Number of training timesteps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory where the trained model is saved",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Evaluate the model every N timesteps",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Number of episodes used during callback evaluation",
    )

    return parser.parse_args()


def make_env(env_type: str, sampling_strategy: str, seed: int):
    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=env_type,
        reward_type="dense",
    )

    env.reset(seed=seed)

    if sampling_strategy != "none":
        env = RandomizationWrapper(env, strategy=sampling_strategy)

    return env


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.model_dir, exist_ok=True)

    save_name = (
        f"sac_push_{args.sampling_strategy}_{args.env_type}_"
        f"{args.timesteps // 1000}k_seed_{args.seed}"
    )

    save_path = os.path.join(args.model_dir, save_name)

    best_model_dir = os.path.join(
        args.model_dir,
        "best",
        save_name,
    )

    log_dir = os.path.join(
        "logs",
        save_name,
    )

    env = make_env(
        env_type=args.env_type,
        sampling_strategy=args.sampling_strategy,
        seed=args.seed,
    )

    eval_env = make_env(
        env_type=args.env_type,
        sampling_strategy=args.sampling_strategy,
        seed=args.seed + 10_000,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        verbose=1,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        seed=args.seed,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
    )

    model.save(save_path)

    print(f"\nFinal model saved to: {save_path}.zip")
    print(f"Best model saved to: {os.path.join(best_model_dir, 'best_model.zip')}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()