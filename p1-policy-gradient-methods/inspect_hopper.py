import gymnasium as gym

env = gym.make("Hopper-v4")

print("=== STATE SPACE ===")
print("Dimensione:", env.observation_space.shape)
print("Limiti bassi:", env.observation_space.low)
print("Limiti alti:", env.observation_space.high)

print("\n=== ACTION SPACE ===")
print("Dimensione:", env.action_space.shape)
print("Limiti bassi:", env.action_space.low)
print("Limiti alti:", env.action_space.high)

print("\n=== MASSE ===")
for i, mass in enumerate(env.unwrapped.model.body_mass):
    print(f"  Body {i}: {mass:.4f} kg")

env.close()