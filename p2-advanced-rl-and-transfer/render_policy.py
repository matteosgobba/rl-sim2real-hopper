import argparse
import os
import time

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import panda_gym  # noqa: F401
from stable_baselines3 import SAC, PPO


def parse_args():
    parser = argparse.ArgumentParser(description="Render a trained PandaPush policy and save frames.")

    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["sac", "ppo"],
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="target",
        choices=["source", "target"]
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        default="dense",
        choices=["dense", "sparse"]
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="render_outputs"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1
    )
    parser.add_argument(
        "--gif",
        action="store_true"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0
    )

    return parser.parse_args()


def load_model(algo: str, model_path: str):
    if algo == "sac":
        return SAC.load(model_path)

    if algo == "ppo":
        return PPO.load(model_path)

    raise ValueError(f"Unsupported algo: {algo}")


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=args.env_type,
        reward_type=args.reward_type,
    )

    model = load_model(args.algo, args.model_path)

    all_gif_frames = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        step = 0
        episode_return = 0.0

        episode_dir = os.path.join(args.output_dir, f"episode_{ep + 1:03d}")
        os.makedirs(episode_dir, exist_ok=True)

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += float(reward)

            frame = env.render()

            if step % args.save_every == 0:
                frame_path = os.path.join(episode_dir, f"frame_{step:04d}.png")
                imageio.imwrite(frame_path, frame)
                all_gif_frames.append(frame)

            step += 1

            if args.sleep > 0:
                time.sleep(args.sleep)

        success = info.get("is_success", None)

        print(
            f"Episode {ep + 1:03d} | "
            f"return = {episode_return:.3f} | "
            f"steps = {step} | "
            f"success = {success}"
        )

    if args.gif and len(all_gif_frames) > 0:
        gif_path = os.path.join(args.output_dir, "policy_render.gif")
        imageio.mimsave(gif_path, all_gif_frames, fps=args.fps)
        print(f"GIF saved to: {gif_path}")

    env.close()

    print(f"Frames saved to: {args.output_dir}")


if __name__ == "__main__":
    main()