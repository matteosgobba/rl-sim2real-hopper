"""Sample script for training a control policy on the Hopper environment.

This script supports:
- REINFORCE
- Actor-Critic
"""

import argparse
import time
import random
import numpy as np
import gymnasium as gym
import torch

from agent import Policy, ReinforceAgent, ActorCriticAgent, PolicyNetwork, ValueNetwork


def evaluate_agent(env, agent, n_episodes=5, seed=123):
    rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            state = next_state

        rewards.append(ep_reward)

    return float(np.mean(rewards))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algo', type=str, default='reinforce', choices=['reinforce', 'actor_critic'])

    parser.add_argument('--baseline', type=float, default=0.0)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_actor', type=float, default=5e-4)
    parser.add_argument('--lr_critic', type=float, default=1e-3)

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

    if args.algo == 'reinforce':
        policy = Policy(state_dim, action_dim)

        agent = ReinforceAgent(
            policy=policy,
            baseline=args.baseline,
            lr=args.lr,
            gamma=args.gamma,
            device='cpu'
        )

    elif args.algo == 'actor_critic':
        actor_net = PolicyNetwork(state_dim, action_dim)
        critic_net = ValueNetwork(state_dim)

        agent = ActorCriticAgent(
            actor=actor_net,
            critic=critic_net,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            gamma=args.gamma,
            value_coef=0.7,
            entropy_coef=0.02,
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
        ep_loss = 0.0
        steps = 0

        ep_start = time.time()

        while not done:
            if args.algo == 'reinforce':
                action, action_log_prob = agent.get_action(state, evaluation=False)
                action = np.clip(action, env.action_space.low, env.action_space.high)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.store_outcome(
                    state=state,
                    next_state=next_state,
                    action_log_prob=action_log_prob,
                    reward=reward,
                    done=done
                )

                ep_reward += reward
                state = next_state
                steps += 1

            elif args.algo == 'actor_critic':
                action, action_info = agent.get_action(state, evaluation=False)
                action_log_prob, entropy, value = action_info

                action = np.clip(action, env.action_space.low, env.action_space.high)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                step_loss = agent.update_step(
                    next_state=next_state,
                    action_log_prob=action_log_prob,
                    reward=reward,
                    terminal=terminated,
                    value=value,
                    entropy=entropy
                )

                ep_loss += step_loss
                ep_reward += reward
                state = next_state
                steps += 1

        if args.algo == 'reinforce':
            loss, _ = agent.update_policy()
        else:
            loss = ep_loss / steps if steps > 0 else 0.0

        ep_time = time.time() - ep_start

        episode_rewards.append(ep_reward)
        episode_times.append(ep_time)

        if episode % args.eval_every == 0:
            avg_eval_reward = evaluate_agent(
                eval_env,
                agent,
                n_episodes=5,
                seed=args.seed + 10000
            )

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
    print(f"Algorithm: {args.algo}")

    if args.algo == 'reinforce':
        print(f"Baseline: {args.baseline}")

    print(f"Episodes: {args.episodes}")
    print(f"Average episode reward: {np.mean(episode_rewards):.2f}")
    print(f"Average episode time: {np.mean(episode_times):.2f}s")
    print(f"Total training time: {total_time:.2f}s")

    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()