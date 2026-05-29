"""
Training script for Part 1.

Examples:
    python train.py --algo reinforce --baseline 0 --episodes 1000 --seed 0 --eval-every 20
    python train.py --algo reinforce --baseline 20 --episodes 1000 --seed 0 --eval-every 20
    python train.py --algo actor_critic --ac-variant one_step --episodes 1000 --seed 0 --eval-every 20
    python train.py --algo actor_critic --ac-variant n_step --n-steps 10 --episodes 1000 --seed 0 --eval-every 20

Outputs:
    results/<run_name>/training_log.csv
    results/<run_name>/final_evaluation.csv
    results/<run_name>/model.pt

Optional TensorBoard commmand:
    python train.py --algo actor_critic --episodes 1000 --tensorboard
    tensorboard --logdir results
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from agent import ActorCriticAgent, ReinforceAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train REINFORCE or Actor-Critic on Hopper-v4.")

    parser.add_argument("--algo", choices=["reinforce", "actor_critic"], default="reinforce")
    parser.add_argument("--env-id", type=str, default="Hopper-v4")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Shared hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    # REINFORCE hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--baseline", type=float, default=0.0)
    parser.add_argument(
        "--normalize-advantages",
        action="store_true",
        help="Optional stability trick. Do not use for the main baseline=0 vs baseline=20 comparison.",
    )

    # Actor-Critic hyperparameters
    parser.add_argument("--ac-variant", choices=["one_step", "n_step"], default="one_step")
    parser.add_argument("--n-steps", type=int, default=5, help="Number of steps for n-step Actor-Critic.")
    parser.add_argument(
        "--normalize-ac-advantages",
        action="store_true",
        help="Optional advantage normalization for n-step Actor-Critic only.",
    )
    parser.add_argument("--lr-actor", type=float, default=5e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--value-coef", type=float, default=0.7)
    parser.add_argument("--entropy-coef", type=float, default=0.02)

    # Evaluation and logging
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--final-eval-episodes", type=int, default=50)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--render-eval", action="store_true")

    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_id: str, seed: int, render_mode: Optional[str] = None) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def clip_action(env: gym.Env, action: np.ndarray) -> np.ndarray:
    """Clip Gaussian actions to the Box limits required by MuJoCo/Gymnasium"""
    if isinstance(env.action_space, gym.spaces.Box):
        return np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
    return action


def evaluate_policy(
    agent,
    env_id: str,
    seed: int,
    n_episodes: int = 50,
    max_steps: int = 1000,
    render: bool = False,
) -> Tuple[float, float]:
    render_mode = "human" if render else None
    env = make_env(env_id, seed=seed, render_mode=render_mode)

    returns = []

    for episode_idx in range(n_episodes):
        state, _ = env.reset(seed=seed + episode_idx)
        episode_return = 0.0

        for _ in range(max_steps):
            action, _ = agent.get_action(state, evaluation=True)
            action = clip_action(env, action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_return += float(reward)
            state = next_state

            if done:
                break

        returns.append(episode_return)

    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name is not None:
        return args.run_name

    if args.algo == "reinforce":
        return f"reinforce_baseline_{args.baseline:g}_seed_{args.seed}"

    if args.ac_variant == "n_step":
        return f"actor_critic_nstep_{args.n_steps}_seed_{args.seed}"

    return f"actor_critic_onestep_seed_{args.seed}"


def write_csv_header(path: Path, header: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_csv_row(path: Path, row: list) -> None:
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def maybe_create_tensorboard_writer(args: argparse.Namespace, run_dir: Path):
    if not args.tensorboard:
        return None

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard is not installed. Continuing with CSV logging only.")
        return None

    return SummaryWriter(log_dir=str(run_dir / "tensorboard"))


def create_agent(args: argparse.Namespace, state_dim: int, action_dim: int):
    grad_clip = None if args.grad_clip <= 0 else args.grad_clip

    if args.algo == "reinforce":
        return ReinforceAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=args.lr,
            gamma=args.gamma,
            baseline=args.baseline,
            hidden_dim=args.hidden_dim,
            normalize_advantages=args.normalize_advantages,
            grad_clip=grad_clip,
            device=args.device,
        )

    return ActorCriticAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        hidden_dim=args.hidden_dim,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        grad_clip=grad_clip,
        device=args.device,
    )


def train_reinforce_episode(agent: ReinforceAgent, env: gym.Env, max_steps: int) -> Tuple[float, Dict[str, float], int]:
    state, _ = env.reset()
    episode_return = 0.0
    steps = 0

    for _ in range(max_steps):
        action, log_prob = agent.get_action(state, evaluation=False)
        clipped_action = clip_action(env, action)

        next_state, reward, terminated, truncated, _ = env.step(clipped_action)
        done = terminated or truncated

        agent.store_outcome(state, next_state, log_prob, float(reward), done)

        episode_return += float(reward)
        steps += 1
        state = next_state

        if done:
            break

    metrics = agent.update_policy()
    return episode_return, metrics, steps


def train_actor_critic_one_step_episode(
    agent: ActorCriticAgent,
    env: gym.Env,
    max_steps: int,
) -> Tuple[float, Dict[str, float], int]:
    state, _ = env.reset()
    episode_return = 0.0
    steps = 0
    step_metrics = []

    for _ in range(max_steps):
        action, log_prob = agent.get_action(state, evaluation=False)
        clipped_action = clip_action(env, action)

        next_state, reward, terminated, truncated, _ = env.step(clipped_action)
        done = terminated or truncated

        metrics = agent.update_step(state, log_prob, float(reward), next_state, done)
        step_metrics.append(metrics)

        episode_return += float(reward)
        steps += 1
        state = next_state

        if done:
            break

    avg_metrics: Dict[str, float] = {}
    if step_metrics:
        for key in step_metrics[0].keys():
            avg_metrics[key] = float(np.mean([m[key] for m in step_metrics]))

    return episode_return, avg_metrics, steps


def train_actor_critic_n_step_episode(
    agent: ActorCriticAgent,
    env: gym.Env,
    max_steps: int,
    n_steps: int,
    normalize_advantages: bool = False,
) -> Tuple[float, Dict[str, float], int]:
    state, _ = env.reset()
    episode_return = 0.0
    steps = 0

    states = []
    log_probs = []
    rewards = []
    next_states = []
    dones = []

    for _ in range(max_steps):
        action, log_prob = agent.get_action(state, evaluation=False)
        clipped_action = clip_action(env, action)

        next_state, reward, terminated, truncated, _ = env.step(clipped_action)
        done = terminated or truncated

        states.append(state)
        log_probs.append(log_prob)
        rewards.append(float(reward))
        next_states.append(next_state)
        dones.append(done)

        episode_return += float(reward)
        steps += 1
        state = next_state

        if done:
            break

    metrics = agent.update_n_step_episode(
        states=states,
        action_log_probs=log_probs,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        n_steps=n_steps,
        normalize_advantages=normalize_advantages,
    )

    return episode_return, metrics, steps


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    run_name = build_run_name(args)
    run_dir = Path(args.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.env_id, seed=args.seed)

    print("Environment:", args.env_id)
    print("State space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Algorithm:", args.algo)
    if args.algo == "actor_critic":
        print("Actor-Critic variant:", args.ac_variant)
        if args.ac_variant == "n_step":
            print("n-steps:", args.n_steps)
    print("Device:", args.device)
    print("Run directory:", run_dir)

    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError("This script expects a continuous Box observation space.")
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("This script expects a continuous Box action space.")

    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    agent = create_agent(args, state_dim, action_dim)
    writer = maybe_create_tensorboard_writer(args, run_dir)

    log_path = run_dir / "training_log.csv"
    final_eval_path = run_dir / "final_evaluation.csv"
    model_path = run_dir / "model.pt"

    write_csv_header(
        log_path,
        [
            "algo",
            "ac_variant",
            "n_steps",
            "baseline",
            "seed",
            "episode",
            "episode_return",
            "episode_steps",
            "episode_time_sec",
            "elapsed_time_sec",
            "eval_mean_return",
            "eval_std_return",
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "mean_return",
            "mean_advantage",
            "advantage",
        ],
    )

    training_start_time = time.time()
    best_eval_mean = -float("inf")

    for episode in range(1, args.episodes + 1):
        episode_start_time = time.time()

        if args.algo == "reinforce":
            episode_return, metrics, episode_steps = train_reinforce_episode(agent, env, args.max_steps)
        elif args.ac_variant == "one_step":
            episode_return, metrics, episode_steps = train_actor_critic_one_step_episode(agent, env, args.max_steps)
        else:
            episode_return, metrics, episode_steps = train_actor_critic_n_step_episode(
                agent,
                env,
                args.max_steps,
                n_steps=args.n_steps,
                normalize_advantages=args.normalize_ac_advantages,
            )

        episode_time = time.time() - episode_start_time
        elapsed_time = time.time() - training_start_time

        eval_mean = ""
        eval_std = ""

        if args.eval_every > 0 and episode % args.eval_every == 0:
            eval_mean_float, eval_std_float = evaluate_policy(
                agent=agent,
                env_id=args.env_id,
                seed=args.seed + 10_000 + episode,
                n_episodes=args.eval_episodes,
                max_steps=args.max_steps,
                render=False,
            )
            eval_mean = eval_mean_float
            eval_std = eval_std_float

            if eval_mean_float > best_eval_mean:
                best_eval_mean = eval_mean_float
                agent.save(str(run_dir / "best_model.pt"))

            print(
                f"Episode {episode:04d} | "
                f"return={episode_return:9.2f} | "
                f"eval={eval_mean_float:9.2f} +/- {eval_std_float:7.2f} | "
                f"time={elapsed_time:8.1f}s"
            )
        else:
            print(
                f"Episode {episode:04d} | "
                f"return={episode_return:9.2f} | "
                f"steps={episode_steps:4d} | "
                f"time={elapsed_time:8.1f}s"
            )

        row = [
            args.algo,
            args.ac_variant if args.algo == "actor_critic" else "",
            args.n_steps if args.algo == "actor_critic" and args.ac_variant == "n_step" else "",
            args.baseline if args.algo == "reinforce" else "",
            args.seed,
            episode,
            episode_return,
            episode_steps,
            episode_time,
            elapsed_time,
            eval_mean,
            eval_std,
            metrics.get("loss", ""),
            metrics.get("policy_loss", ""),
            metrics.get("value_loss", ""),
            metrics.get("entropy", ""),
            metrics.get("mean_return", ""),
            metrics.get("mean_advantage", ""),
            metrics.get("advantage", ""),
        ]
        append_csv_row(log_path, row)

        if writer is not None:
            writer.add_scalar("train/episode_return", episode_return, episode)
            writer.add_scalar("train/episode_steps", episode_steps, episode)
            writer.add_scalar("time/episode_time_sec", episode_time, episode)
            writer.add_scalar("time/elapsed_time_sec", elapsed_time, episode)

            for key, value in metrics.items():
                writer.add_scalar(f"losses/{key}", value, episode)

            if eval_mean != "":
                writer.add_scalar("eval/mean_return", eval_mean, episode)
                writer.add_scalar("eval/std_return", eval_std, episode)

    total_training_time = time.time() - training_start_time
    agent.save(str(model_path))

    final_eval_mean, final_eval_std = evaluate_policy(
        agent=agent,
        env_id=args.env_id,
        seed=args.seed + 50_000,
        n_episodes=args.final_eval_episodes,
        max_steps=args.max_steps,
        render=args.render_eval,
    )

    write_csv_header(
        final_eval_path,
        [
            "algo",
            "ac_variant",
            "n_steps",
            "baseline",
            "seed",
            "episodes",
            "final_eval_episodes",
            "final_eval_mean_return",
            "final_eval_std_return",
            "total_training_time_sec",
            "model_path",
        ],
    )
    append_csv_row(
        final_eval_path,
        [
            args.algo,
            args.ac_variant if args.algo == "actor_critic" else "",
            args.n_steps if args.algo == "actor_critic" and args.ac_variant == "n_step" else "",
            args.baseline if args.algo == "reinforce" else "",
            args.seed,
            args.episodes,
            args.final_eval_episodes,
            final_eval_mean,
            final_eval_std,
            total_training_time,
            str(model_path),
        ],
    )

    if writer is not None:
        writer.add_scalar("final_eval/mean_return", final_eval_mean, args.episodes)
        writer.add_scalar("final_eval/std_return", final_eval_std, args.episodes)
        writer.close()

    env.close()

    print("\nTraining completed.")
    print(f"Final evaluation over {args.final_eval_episodes} episodes: {final_eval_mean:.2f} +/- {final_eval_std:.2f}")
    print(f"Total training time: {total_training_time:.1f}s")
    print(f"Saved model: {model_path}")
    print(f"Saved training log: {log_path}")
    print(f"Saved final evaluation: {final_eval_path}")


if __name__ == "__main__":
    main()
