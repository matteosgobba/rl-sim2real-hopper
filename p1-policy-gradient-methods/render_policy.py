"""
Render a trained REINFORCE or Actor-Critic policy on Hopper-v4.

Examples:

Live rendering:
    python render_policy.py \
        --algo actor_critic \
        --model-path results/actor_critic_nstep_10_seed_0/model.pt \
        --render-mode human

Save frames:
    python render_policy.py \
        --algo actor_critic \
        --model-path results/actor_critic_nstep_10_seed_0/model.pt \
        --render-mode rgb_array \
        --save-frames \
        --frames-dir frames/nstep_ac_seed0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from agent import ActorCriticAgent, ReinforceAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a trained policy on Hopper-v4.")

    parser.add_argument("--algo", choices=["reinforce", "actor_critic"], required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="Hopper-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument(
        "--render-mode",
        choices=["human", "rgb_array"],
        default="human",
        help="human shows the live MuJoCo window; rgb_array allows saving screenshots.",
    )

    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--frames-dir", type=str, default="frames")
    parser.add_argument("--save-every", type=int, default=10)

    return parser.parse_args()


def make_env(env_id: str, seed: int, render_mode: Optional[str]) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def clip_action(env: gym.Env, action: np.ndarray) -> np.ndarray:
    if isinstance(env.action_space, gym.spaces.Box):
        return np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
    return action


def create_agent(args: argparse.Namespace, state_dim: int, action_dim: int):
    """
    The architecture must match the one used in training.
    If we train with hidden_dim=64, then we keep hidden_dim=64 here.
    """
    if args.algo == "reinforce":
        return ReinforceAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            device=args.device,
        )

    return ActorCriticAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        device=args.device,
    )


def main() -> None:
    args = parse_args()

    env = make_env(args.env_id, seed=args.seed, render_mode=args.render_mode)

    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError("This script expects a continuous Box observation space.")
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("This script expects a continuous Box action space.")

    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    agent = create_agent(args, state_dim, action_dim)
    agent.load(args.model_path)

    frames_dir = Path(args.frames_dir)
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    global_frame_idx = 0

    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed + episode)
        episode_return = 0.0

        for step in range(args.max_steps):
            action, _ = agent.get_action(state, evaluation=True)
            action = clip_action(env, action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_return += float(reward)
            state = next_state

            if args.render_mode == "rgb_array" and args.save_frames:
                frame = env.render()

                if frame is not None and step % args.save_every == 0:
                    image = Image.fromarray(frame)
                    image_path = frames_dir / f"episode_{episode:02d}_step_{step:04d}.png"
                    image.save(image_path)
                    global_frame_idx += 1

            if done:
                break

        print(f"Episode {episode + 1} | return = {episode_return:.2f} | steps = {step + 1}")

    env.close()

    if args.save_frames:
        print(f"Saved frames in: {frames_dir}")


if __name__ == "__main__":
    main()