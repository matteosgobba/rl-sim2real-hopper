import gymnasium as gym

env = gym.make("Hopper-v4")

print("STATE SPACE")
print("Dimension:", env.observation_space.shape)
print("Lower bounds:", env.observation_space.low)
print("Upper bounds:", env.observation_space.high)

print("\nACTION SPACE")
print("Dimension:", env.action_space.shape)
print("Lower bounds:", env.action_space.low)
print("Upper bounds:", env.action_space.high)

print("\nMASSES")
for i, mass in enumerate(env.unwrapped.model.body_mass):
    print(f"  Body {i}: {mass:.4f} kg")

env.close()