import gymnasium as gym
from lib.gym import EcoEnv

# Register your environment
gym.register(
    id='EcoEvo-v0',
    entry_point='lib.gym:EcoEnv',
)

# Create and use the environment
env = gym.make('EcoEvo-v0', render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()