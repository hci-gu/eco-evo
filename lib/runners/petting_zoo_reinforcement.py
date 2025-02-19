import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gymnasium import spaces
from lib.environments.petting_zoo import env as pettingzoo_env

# Wrapper to convert a multi-agent PettingZoo AEC environment into a single-agent Gym environment.
class SingleAgentWrapper(gym.Env):
    """
    Wraps a multi-agent PettingZoo AEC environment so that only one specified agent is exposed.
    This wrapper uses the Gymnasium API: reset() returns (obs, info) and step() returns
    (obs, reward, terminated, truncated, info).
    """
    def __init__(self, aec_env, agent):
        super().__init__()
        self.aec_env = aec_env
        self.agent = agent
        self.observation_space = aec_env.observation_space(agent)
        self.action_space = aec_env.action_space(agent)
        self.last_reward = 0
        self.done = False

    def reset(self, **kwargs):
        self.aec_env.reset(**kwargs)
        # Advance until it's our agent's turn.
        while self.aec_env.agent_selection != self.agent:
            self.aec_env.step(None)
        obs, reward, termination, truncation, info = self.aec_env.last()
        self.last_reward = reward
        self.done = termination or truncation
        return obs, info

    def step(self, action):
        self.aec_env.step(action)
        # Advance until it's our agent's turn or the episode ends.
        while True:
            if self.aec_env.agent_selection == self.agent:
                break
            self.aec_env.step(None)
            if self.done:
                break
        obs, reward, termination, truncation, info = self.aec_env.last()
        self.last_reward = reward
        self.done = termination or truncation
        return obs, reward, termination, truncation, info

    def render(self, mode="human"):
        return self.aec_env.render(mode)

# Wrapper to convert dict observations/actions into flat Box spaces for RL.
class RLAgentWrapper(gym.Env):
    """
    Converts an environment that uses Dict observation and Dict action spaces into a standard Gym env.
    The RL agent sees:
      - Observation: only the "observation" field (a Box space).
      - Action: only the "action" field (a Box space) with a symmetric range [-1,1].
    The wrapper stores the "positions" from the observation and attaches them to the action dict.
    """
    def __init__(self, env):
        super().__init__()
        self.env = env
        # Use the underlying observation "observation" Box.
        self.observation_space = env.observation_space.spaces["observation"]
        # Instead of using the underlying action space (which is [0,1]), create a symmetric Box.
        orig_action_space = env.action_space.spaces["action"]
        self.action_space = spaces.Box(
            low=-1, high=1, shape=orig_action_space.shape, dtype=np.float32
        )
        self.last_positions = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Convert positions to a NumPy array with proper dtype.
        self.last_positions = np.array(obs["positions"], dtype=np.int32)
        # Convert observation field to NumPy array.
        if isinstance(obs["observation"], torch.Tensor):
            obs_np = obs["observation"].cpu().detach().numpy()
        else:
            obs_np = np.array(obs["observation"])
        return obs_np, info

    def step(self, action):
        # Scale action from [-1, 1] to [0, 1].
        scaled_action = (action + 1) / 2.0
        # Reconstruct the dict action by attaching stored positions.
        dict_action = {"action": scaled_action, "positions": self.last_positions}
        self.env.step(dict_action)
        obs, reward, termination, truncation, info = self.env.last()
        # Update last_positions (convert to NumPy with correct dtype).
        self.last_positions = np.array(obs["positions"], dtype=np.int32)
        if isinstance(obs["observation"], torch.Tensor):
            obs_np = obs["observation"].cpu().detach().numpy()
        else:
            obs_np = np.array(obs["observation"])
        return obs_np, reward, termination, truncation, info

    def render(self, mode="human"):
        return self.env.render(mode)

def make_single_agent_env(aec_env, agent, render_mode="none"):
    """
    Given a multi-agent PettingZoo environment, create a single-agent Gym environment for the specified agent.
    """
    single_agent_env = SingleAgentWrapper(aec_env, agent)
    single_agent_env = RLAgentWrapper(single_agent_env)
    return single_agent_env

class MultiAgentRunner:
    def __init__(self, render_mode="none"):
        # Create the base PettingZoo environment.
        self.aec_env = pettingzoo_env(render_mode=render_mode)
        self.aec_env.reset()
        self.agents = self.aec_env.possible_agents

        # Create a PPO model for each agent using its own wrapped environment.
        self.models = {}
        self.envs = {}  # Store single-agent environments for each agent.
        for agent in self.agents:
            single_agent_env = make_single_agent_env(self.aec_env, agent, render_mode=render_mode)
            # Check the environment for compatibility.
            check_env(single_agent_env, warn=True)
            self.envs[agent] = single_agent_env
            self.models[agent] = PPO("MlpPolicy", single_agent_env, verbose=1)

    def train(self, timesteps=10000):
        # Train each agent independently.
        for agent, model in self.models.items():
            print(f"Training agent: {agent}")
            model.learn(total_timesteps=timesteps)
            model.save(f"ppo_{agent}_model")

    def evaluate(self, episodes=5):
        # Evaluate each trained agent.
        for agent, model in self.models.items():
            env = self.envs[agent]
            obs, _ = env.reset()
            print(f"Evaluating agent: {agent}")
            for episode in range(episodes):
                terminated, truncated = False, False
                total_reward = 0
                while not (terminated or truncated):
                    action, _ = model.predict(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    env.render()
                print(f"Agent {agent} Episode {episode} reward: {total_reward}")
                obs, _ = env.reset()

