import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import panda_gym  # type: ignore[import-not-found]
from stable_baselines3 import PPO
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=args.env_type,
        reward_type="dense",
    )

    #TODO: add randomization wrapper here
    # Randomization wrapper
    if args.sampling_strategy != "none":
        env = RandomizationWrapper(env, strategy=args.sampling_strategy)
        
    #TODO: create model and train it
    
    # PPO model
    model = PPO(
        policy="MultiInputPolicy",  # PandaPush usa dict observation
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )

    # Training
    model.learn(total_timesteps=args.timesteps)
    
    save_name = f"ppo_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    # TODO: model.save(save_name)
    model.save(save_name)


if __name__ == "__main__":
    main()