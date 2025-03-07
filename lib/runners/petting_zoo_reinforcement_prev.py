import lib.constants as const
from lib.environments.petting_zoo import env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from torch.utils.tensorboard import SummaryWriter
import random

# Define a simple policy network that outputs the mean for a Gaussian distribution.
# We use a learnable log_std parameter to define the standard deviation.
class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()  # actions are in [0, 1]
        )
        # Initialize log_std as a learnable parameter.
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        mean = self.net(x)
        std = torch.exp(self.log_std)
        return distributions.Normal(mean, std)

class PettingZooRLRunner():
    def __init__(self):
        # Create the environment (set render_mode to "none" if visualization is not needed).
        self.env = env(render_mode="none")
        self.empty_action = self.env.action_space("plankton").sample()
        self.env.reset()
        
        # Set up TensorBoard writer.
        self.writer = SummaryWriter("runlogs/models")
        
        # RL parameters.
        self.gamma = 0.99  # Discount factor (if you expand the algorithm later).
        self.current_episode = 0

        # Use all species except plankton.
        self.species_list = [species for species in const.SPECIES_MAP.keys() if species != "plankton"]

        # Create a policy network and optimizer for each species.
        self.policies = {}
        self.optimizers = {}
        # Use the observation vector length from the environment.
        input_dim = self.env.obs_vector_length  
        # Assume the action space for any non-plankton agent defines the output dimension.
        output_dim = self.env.action_space(self.species_list[0]).shape[1]
        for species in self.species_list:
            self.policies[species] = Policy(input_dim, output_dim)
            self.optimizers[species] = optim.Adam(self.policies[species].parameters(), lr=1e-3)

    def run_episode(self):
        """
        Runs a single episode using RL. Returns cumulative rewards and losses per species.
        """
        self.env.reset()
        episode_rewards = {species: 0 for species in self.species_list}
        episode_losses = {species: 0 for species in self.species_list}

        # Run until all agents are terminated or truncated.
        while not all(self.env.terminations.values()) and not all(self.env.truncations.values()):
            current_agent = self.env.agent_selection
            # If the current agent is plankton, take the empty action.
            if current_agent == "plankton":
                self.env.step(self.empty_action)
            elif current_agent in self.species_list:
                # Get the observation batch and average over the batch for simplicity.
                obs, _, termination, truncation, info = self.env.last()
                print(obs)
                print(obs.shape)
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                # Forward pass to get an action distribution.
                dist = self.policies[current_agent](obs_tensor)
                action_tensor = dist.sample()
                log_prob = dist.log_prob(action_tensor).sum()
                # Expand the single action to match the expected batch size.
                batch_size = self.env.observation_positions[current_agent].shape[0]
                action = action_tensor.detach().numpy().repeat(batch_size, axis=0)
                # Step the environment with the chosen action.
                print(action)
                print(action.shape)
                self.env.step(action)
                reward = self.env.rewards[current_agent]
                episode_rewards[current_agent] += reward
                # Compute a simple policy gradient loss (REINFORCE).
                loss = -log_prob * reward
                episode_losses[current_agent] += loss.item()
                # Update the policy.
                self.optimizers[current_agent].zero_grad()
                loss.backward()
                self.optimizers[current_agent].step()
            else:
                # For any unknown agent, perform an empty action.
                self.env.step(self.empty_action)
        return episode_rewards, episode_losses

    def train(self, num_episodes=500):
        """
        Trains the RL policies for a given number of episodes.
        Logs rewards and losses to TensorBoard.
        """
        for episode in range(num_episodes):
            self.current_episode = episode
            rewards, losses = self.run_episode()
            total_reward = sum(rewards.values())
            # Log metrics for each species.
            for species in self.species_list:
                self.writer.add_scalar(f"Reward/{species}", rewards[species], episode)
                self.writer.add_scalar(f"Loss/{species}", losses[species], episode)
            self.writer.add_scalar("Reward/Total", total_reward, episode)
            print(f"Episode {episode+1}: Rewards {rewards}, Total Reward: {total_reward}")
        self.writer.close()
