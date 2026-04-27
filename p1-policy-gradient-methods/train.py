"""Sample script for training a control policy on the Hopper environment

Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import argparse
import time
import numpy as np
import gymnasium as gym
import torch
import random

from agent import Policy, ReinforceAgent


def evaluate_agent(env, agent, n_episodes=5, seed=123):
    rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            state = next_state

        rewards.append(ep_reward)

    return float(np.mean(rewards))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='reinforce', choices=['reinforce'])
    parser.add_argument('--baseline', type=float, default=0.0)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_every', type=int, default=20)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    render_mode = 'human' if args.render else None
    env = gym.make('Hopper-v4', render_mode=render_mode)
    eval_env = gym.make('Hopper-v4')

    print('State space:', env.observation_space)
    print('Action space:', env.action_space)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = Policy(state_dim, action_dim)

    if args.algo == 'reinforce':
        agent = ReinforceAgent(
            policy=policy,
            baseline=args.baseline,
            lr=args.lr,
            gamma=args.gamma,
            device='cpu'
        )
    else:
        raise ValueError(f'Unsupported algorithm: {args.algo}')

    episode_rewards = []
    eval_rewards = []
    episode_times = []

    total_start = time.time()

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        done = False
        ep_reward = 0.0

        ep_start = time.time()

        while not done:
            action, action_log_prob = agent.get_action(state, evaluation=False)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_outcome(state, next_state, action_log_prob, reward, done)

            ep_reward += reward
            state = next_state

        loss, _ = agent.update_policy()
        ep_time = time.time() - ep_start

        episode_rewards.append(ep_reward)
        episode_times.append(ep_time)

        if episode % args.eval_every == 0:
            avg_eval_reward = evaluate_agent(eval_env, agent, n_episodes=5, seed=args.seed + 10000)
            eval_rewards.append((episode, avg_eval_reward))

            print(
                f"Episode {episode:4d} | "
                f"train reward = {ep_reward:8.2f} | "
                f"eval reward = {avg_eval_reward:8.2f} | "
                f"loss = {loss:10.4f} | "
                f"time = {ep_time:6.2f}s"
            )
        else:
            print(
                f"Episode {episode:4d} | "
                f"train reward = {ep_reward:8.2f} | "
                f"loss = {loss:10.4f} | "
                f"time = {ep_time:6.2f}s"
            )

    total_time = time.time() - total_start

    print("\nTraining completed.")
    print(f"Baseline: {args.baseline}")
    print(f"Episodes: {args.episodes}")
    print(f"Average episode reward: {np.mean(episode_rewards):.2f}")
    print(f"Average episode time: {np.mean(episode_times):.2f}s")
    print(f"Total training time: {total_time:.2f}s")

    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()