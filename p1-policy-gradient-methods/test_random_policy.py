"""Test a random policy on the Gym Hopper environment

    Play around with this code to get familiar with the
    Hopper environment.

    For example, what happens if you don't reset the environment
    even after the episode is over?
    When exactly is the episode over?
    What is an action here?
"""
import gymnasium as gym
import os
import csv
from datetime import datetime
from evaluate import evaluate_policy

def main():
    render = False

    if render:
        env = gym.make('Hopper-v4', render_mode='human')
    else:
        env = gym.make('Hopper-v4', render_mode='rgb_array')
    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space

    n_episodes = 50
    rewards = []

    for ep in range(n_episodes):  
        done = False
        state, info = env.reset()  # Reset environment to initial state

        while not done:  # Until the episode is over
            action = env.action_space.sample()  # Sample random action

            state, reward, terminated, truncated, _ = env.step(action)  # Step the simulator to the next timestep
            done = terminated or truncated
            rewards.append(reward)
            if render:
                env.render()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/run_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    results_path = os.path.join(save_dir, "results.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "n_episodes", "mean_reward", "std_reward"])
        writer.writerow(["random", n_episodes,
                         sum(rewards)/len(rewards),
                         (sum((r - sum(rewards)/len(rewards))**2
                         for r in rewards)/len(rewards))**0.5])

    print(f"Results saved in: {results_path}")

if __name__ == '__main__':
    main()