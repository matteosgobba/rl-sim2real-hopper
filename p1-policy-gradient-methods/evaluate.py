import gymnasium as gym
import numpy as np

def evaluate_policy(policy, env_name="Hopper-v4", n_episodes=50, seed=42):
    env = gym.make(env_name)
    rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        total_reward = 0
        done = False

        while not done:
            action = policy(obs)
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
    print(f"Std reward:    {std_reward:.2f}")

    return mean_reward, std_reward