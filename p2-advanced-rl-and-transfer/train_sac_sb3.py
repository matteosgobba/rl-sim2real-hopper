import argparse

import gymnasium as gym
import numpy as np
import panda_gym  # NECESSARIO
from stable_baselines3 import SAC, HerReplayBuffer
from rand_wrapper import RandomizationWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on PandaPush-v3")
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
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
    
    # SAC model
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
    )
    
    # Training
    model.learn(total_timesteps=args.timesteps)

    save_name = f"sac_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    # TODO: model.save(save_name)
    model.save(save_name)


if __name__ == "__main__":
    main()