import argparse
import os
import random

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from rand_wrapper import RandomizationWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on PandaPush-v3")

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
        default=50,
        help="Number of episodes used during callback evaluation",
    )

    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO clipping range",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.0,
        help="Entropy coefficient for PPO",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda parameter",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="Value function loss coefficient",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for PPO",
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

    env = Monitor(env)
    return env


def sanitize_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.model_dir, exist_ok=True)

    lr_str = sanitize_float(args.learning_rate)
    ent_str = sanitize_float(args.ent_coef)

    save_name = (
        f"ppo_push_{args.sampling_strategy}_{args.env_type}_"
        f"{args.timesteps // 1000}k_"
        f"lr_{lr_str}_ent_{ent_str}_seed_{args.seed}"
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

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=0.5,
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