"""Test a random policy on the Gym Hopper environment

    Play around with this code to get familiar with the
    Hopper environment.

    For example, what happens if you don't reset the environment
    even after the episode is over?
    When exactly is the episode over?
    What is an action here?
"""

import argparse
import gymnasium as gym
import numpy as np
import panda_gym  # type: ignore[import-not-found]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-type", type=str, default="target", choices=["source", "target"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    env = gym.make(
        "PandaPush-v3",
        render_mode="human" if args.render else "rgb_array",
        type=args.env_type,
        reward_type="dense",
    )

    print("State space:", env.observation_space)
    print("Action space:", env.action_space)

    episode_returns = []
    successes = []

    for ep in range(args.episodes):
        done = False
        ep_return = 0.0

        state, info = env.reset(seed=args.seed + ep)

        while not done:
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            ep_return += float(reward)

            if args.render:
                env.render()

        episode_returns.append(ep_return)

        if isinstance(info, dict) and "is_success" in info:
            successes.append(float(info["is_success"]))

        print(f"Episode {ep + 1:03d} | return = {ep_return:.3f}")

    env.close()

    returns = np.array(episode_returns)

    print("\n=== Random policy results ===")
    print(f"Environment type: {args.env_type}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean return: {returns.mean():.3f}")
    print(f"Std return:  {returns.std():.3f}")
    print(f"Min return:  {returns.min():.3f}")
    print(f"Max return:  {returns.max():.3f}")

    if successes:
        print(f"Success rate: {np.mean(successes):.2%}")


if __name__ == "__main__":
    main()