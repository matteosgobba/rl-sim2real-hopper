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

    # Environment
    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type used for training",
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        default="dense",
        choices=["dense", "sparse"],
        help="Reward type",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Domain randomization strategy",
    )

    # Training
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
        help="Directory where trained models are saved",
    )

    # PPO hyperparameters to tune
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for PPO",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.0,
        help="Entropy coefficient for PPO",
    )

    # Domain randomization parameters
    parser.add_argument("--mass-min", type=float, default=1.0)
    parser.add_argument("--mass-max", type=float, default=5.0)

    parser.add_argument("--initial-mass-min", type=float, default=1.0)
    parser.add_argument("--initial-mass-max", type=float, default=1.5)
    parser.add_argument("--mass-limit-min", type=float, default=0.5)
    parser.add_argument("--mass-limit-max", type=float, default=6.0)
    parser.add_argument("--adr-step", type=float, default=0.25)
    parser.add_argument("--boundary-prob", type=float, default=0.5)
    parser.add_argument("--verbose-wrapper", action="store_true")

    # Evaluation callback
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Evaluate every N timesteps",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Number of episodes for callback evaluation",
    )

    # TensorBoard logging
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default="tensorboard_logs",
        help="Directory where TensorBoard logs are saved",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging",
    )

    return parser.parse_args()


def sanitize(value) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def make_env(args: argparse.Namespace, seed: int, use_randomization: bool = True):
    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=args.env_type,
        reward_type=args.reward_type,
    )

    if use_randomization and args.sampling_strategy != "none":
        if args.sampling_strategy == "udr":
            env = RandomizationWrapper(
                env,
                mode="udr",
                mass_range=(args.mass_min, args.mass_max),
                verbose=args.verbose_wrapper,
            )

        elif args.sampling_strategy == "adr":
            env = RandomizationWrapper(
                env,
                mode="adr",
                initial_mass_range=(args.initial_mass_min, args.initial_mass_max),
                mass_limits=(args.mass_limit_min, args.mass_limit_max),
                adr_step=args.adr_step,
                boundary_prob=args.boundary_prob,
                verbose=args.verbose_wrapper,
            )

    env = Monitor(env)
    env.reset(seed=seed)

    return env


def build_mass_info(args: argparse.Namespace) -> str:
    if args.sampling_strategy == "udr":
        mass_min_str = sanitize(args.mass_min)
        mass_max_str = sanitize(args.mass_max)
        return f"mass_{mass_min_str}_{mass_max_str}_"

    if args.sampling_strategy == "adr":
        initial_mass_min_str = sanitize(args.initial_mass_min)
        initial_mass_max_str = sanitize(args.initial_mass_max)
        mass_limit_min_str = sanitize(args.mass_limit_min)
        mass_limit_max_str = sanitize(args.mass_limit_max)
        adr_step_str = sanitize(args.adr_step)
        boundary_prob_str = sanitize(args.boundary_prob)

        return (
            f"initmass_{initial_mass_min_str}_{initial_mass_max_str}_"
            f"limitmass_{mass_limit_min_str}_{mass_limit_max_str}_"
            f"adrstep_{adr_step_str}_bprob_{boundary_prob_str}_"
        )

    return ""


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.model_dir, exist_ok=True)

    lr_str = sanitize(args.learning_rate)
    ent_str = sanitize(args.ent_coef)
    mass_info = build_mass_info(args)

    save_name = (
        f"ppo_push_{args.sampling_strategy}_{args.env_type}_"
        f"{args.reward_type}_"
        f"{mass_info}"
        f"{args.timesteps // 1000}k_"
        f"lr_{lr_str}_ent_{ent_str}_seed_{args.seed}"
    )

    save_path = os.path.join(args.model_dir, save_name)
    best_model_dir = os.path.join(args.model_dir, "best", save_name)
    log_dir = os.path.join("logs", save_name)

    env = make_env(args, seed=args.seed, use_randomization=True)
    eval_env = make_env(args, seed=args.seed + 10_000, use_randomization=False)

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
        tensorboard_log=None if args.no_tensorboard else args.tensorboard_log,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        seed=args.seed,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        tb_log_name=save_name,
    )

    model.save(save_path)

    print(f"\nFinal model saved to: {save_path}.zip")
    print(f"Best model saved to: {os.path.join(best_model_dir, 'best_model.zip')}")
    print(f"Eval logs saved to: {os.path.join(log_dir, 'evaluations.npz')}")

    if not args.no_tensorboard:
        print(f"TensorBoard logs saved to: {args.tensorboard_log}/{save_name}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()