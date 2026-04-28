import gymnasium as gym
import numpy as np
import csv
import os


def evaluate_policy(policy, env_name="Hopper-v4", n_episodes=50, seed=42,
                    policy_name="policy", results_path="results.csv"):
    env = gym.make(env_name)
    rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        total_reward = 0
        done = False

        while not done:
            action = policy(obs)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        print(f"Episode {episode + 1}/{n_episodes} - Reward: {total_reward:.2f}")

    env.close()

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    print(f"\nRESULTS ON {n_episodes} EPISODES")
    print(f"Mean reward:  {mean_reward:.2f}")
    print(f"Std reward:   {std_reward:.2f}")

    file_exists = os.path.isfile(results_path)

    with open(results_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["policy", "n_episodes", "mean_reward", "std_reward"])

        writer.writerow([policy_name, n_episodes, mean_reward, std_reward])

    return mean_reward, std_reward