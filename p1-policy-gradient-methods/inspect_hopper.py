import gymnasium as gym

env = gym.make("Hopper-v4")
model = env.unwrapped.model

print("STATE SPACE")
print("Dimension:", env.observation_space.shape)
print("Lower bounds:", env.observation_space.low)
print("Upper bounds:", env.observation_space.high)

print("\nACTION SPACE")
print("Dimension:", env.action_space.shape)
print("Lower bounds:", env.action_space.low)
print("Upper bounds:", env.action_space.high)

print("\nMASSES")
print(f"{'Body ID':<8} {'Body name':<20} {'Mass [kg]':>12}")
print("-" * 44)

for i, mass in enumerate(model.body_mass):
    body_name = model.body(i).name
    print(f"{i:<8} {body_name:<20} {mass:>12.4f}")

print("\nMODEL INFO")
print("Number of bodies:", model.nbody)
print("Number of DoFs:", model.nv)
print("Number of actuators:", model.nu)

env.close()