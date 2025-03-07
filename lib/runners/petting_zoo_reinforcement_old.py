# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import time
# from pettingzoo.utils.conversions import aec_to_parallel
# from torch.utils.tensorboard import SummaryWriter


# from lib.environments.petting_zoo import env  # Replace with your actual module name

# # -----------------------------
# # Define the Policy Network
# # -----------------------------
# class PolicyNetwork(nn.Module):
#     def __init__(self, hidden_dim=128, action_dim=6):
#         super(PolicyNetwork, self).__init__()
#         # Our observation for each agent is a batch of 500 patches, each of length 99.
#         # For simplicity, we process each patch with a linear layer and then average.
#         self.fc1 = nn.Linear(99, hidden_dim)
#         self.actor_head = nn.Linear(hidden_dim, action_dim)
#         self.critic_head = nn.Linear(hidden_dim, 1)

#     def forward(self, x):
#         # x: Tensor of shape (500, 99)
#         x = torch.relu(self.fc1(x))  # shape: (500, hidden_dim)
#         x_mean = x.mean(dim=0)       # aggregate across patches => shape: (hidden_dim)
#         logits = self.actor_head(x_mean)  # (action_dim,)
#         value = self.critic_head(x_mean)  # (1,)
#         return logits, value

#     def act(self, observation):
#         """
#         Given a NumPy observation of shape (500, 99), compute the action.
#         Returns:
#           - action (int): chosen discrete action (0...5)
#           - log_prob (Tensor): log probability of that action
#           - value (Tensor): estimated state value
#         """
#         # Convert observation to tensor.
#         obs_tensor = torch.from_numpy(observation).float()
#         logits, value = self.forward(obs_tensor)
#         probs = torch.softmax(logits, dim=-1)
#         dist = torch.distributions.Categorical(probs)
#         action = dist.sample()
#         log_prob = dist.log_prob(action)
#         return action.item(), log_prob, value


# # -----------------------------
# # Define the RL Runner Class
# # -----------------------------
# class RLRunner:
#     def __init__(self, discount=0.99):
#         aec_env = env(render_mode="none")
#         aec_env.metadata["is_parallelizable"] = True
#         parallel_env = aec_to_parallel(aec_env)
#         self.env = parallel_env

#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.policy = PolicyNetwork(hidden_dim=128, action_dim=6).to(device)
#         self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
#         self.discount = discount
#         self.device = device
#         self.writer = SummaryWriter(log_dir="logs/models")

#     def run_episode(self):
#         """
#         Runs one episode and collects transitions for all agents.
#         Returns:
#           - transitions: dict mapping agent id to a list of tuples
#               (obs, action, log_prob, value, reward)
#           - total_rewards: dict mapping agent id to total episode reward
#         """
#         obs, _ = self.env.reset()  # obs: dict {agent: observation_dict}
#         dones = {agent: False for agent in obs.keys()}
#         transitions = {agent: [] for agent in obs.keys()}
#         total_rewards = {agent: 0.0 for agent in obs.keys()}

#         while not all(dones.values()):
#             actions = {}
#             # For each agent, select an action using the shared policy.
#             for agent, agent_obs in obs.items():
#                 # We use only the "observation" key (shape: (500, 99)).
#                 patch_obs = agent_obs["observation"]
#                 action, log_prob, value = self.policy.act(patch_obs)
#                 # Convert the discrete action to the format expected by your env:
#                 # Create a one-hot vector of length 6 and repeat it for the fixed batch size.
#                 one_hot = np.eye(6, dtype=np.float32)[action]  # shape (6,)
#                 action_array = np.repeat(one_hot[np.newaxis, :], repeats=patch_obs.shape[0], axis=0)
#                 # For positions, simply pass through the observation's positions.
#                 actions[agent] = {"action": action_array, "positions": agent_obs["positions"]}
#                 # Save transition (we will later append reward once we get it).
#                 transitions[agent].append([patch_obs, action, log_prob, value, None])
#             # Step the environment.
#             next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
#             dones = {agent: terminations[agent] or truncations[agent] for agent in terminations}

#             # Update rewards in transitions and total rewards.
#             for agent in obs.keys():
#                 total_rewards[agent] += (rewards[agent] / 4.0)  # Divide by 4 to match the original reward scale.
#                 # Append the reward to the last transition tuple.
#                 transitions[agent][-1][-1] = rewards[agent]
#             obs = next_obs
#         return transitions, total_rewards

#     def update_policy(self, transitions):
#         """
#         Updates the policy using an actor-critic loss computed over all transitions.
#         """
#         all_obs, all_actions, all_log_probs, all_returns, all_values = [], [], [], [], []
        
#         # Combine trajectories from all agents.
#         for agent, traj in transitions.items():
#             rewards = [t[4] for t in traj]
#             returns = []
#             R = 0.0
#             # Compute discounted returns (backwards).
#             for r in rewards[::-1]:
#                 R = r + self.discount * R
#                 returns.insert(0, R)
#             for (obs_np, action, log_prob, value, _), R in zip(traj, returns):
#                 # For each observation (shape: (500, 99)), we use the mean representation.
#                 obs_tensor = torch.from_numpy(obs_np).float().mean(dim=0)
#                 all_obs.append(obs_tensor)
#                 all_actions.append(action)
#                 all_log_probs.append(log_prob)
#                 all_values.append(value)
#                 all_returns.append(R)
        
#         # Stack lists into tensors.
#         obs_tensor = torch.stack(all_obs).to(self.device)              # shape: (N, 99)
#         actions_tensor = torch.tensor(all_actions, dtype=torch.long).to(self.device)
#         old_log_probs = torch.stack(all_log_probs).to(self.device)
#         returns_tensor = torch.tensor(all_returns, dtype=torch.float32).to(self.device)
#         values_tensor = torch.stack(all_values).squeeze().to(self.device)

#         # Compute advantages.
#         advantages = returns_tensor - values_tensor

#         # Re-run policy on the observations.
#         logits, values = self.policy.forward(obs_tensor)
#         # logits, values = self.policy.forward(obs_tensor.unsqueeze(0).expand(obs_tensor.size(0), -1))
#         probs = torch.softmax(logits, dim=-1)
#         dist = torch.distributions.Categorical(probs)
#         new_log_probs = dist.log_prob(actions_tensor)

#         # Compute the loss: policy loss + value loss.
#         policy_loss = -(new_log_probs * advantages.detach()).mean()
#         value_loss = advantages.pow(2).mean()
#         loss = policy_loss + 0.5 * value_loss

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         return loss.item()
    
#     def save_model(self, path):
#         """
#         Saves the model and optimizer states to the given path.
#         """
#         torch.save({
#             'model_state_dict': self.policy.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict()
#         }, path)
#         print(f"Model saved to {path}")

#     def load_model(self, path):
#         """
#         Loads the model and optimizer states from the given path.
#         """
#         checkpoint = torch.load(path, map_location=self.device)
#         self.policy.load_state_dict(checkpoint['model_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         print(f"Model loaded from {path}")

#     def train(self, num_episodes):
#         for ep in range(num_episodes):
#             transitions, total_rewards = self.run_episode()
#             loss = self.update_policy(transitions)
#             print(f"Episode {ep} - Loss: {loss:.4f}, Rewards: {total_rewards}")
            
#             # Log metrics to TensorBoard
#             self.writer.add_scalar("Loss", loss, ep)
#             for agent, reward in total_rewards.items():
#                 self.writer.add_scalar(f"Reward/{agent}", reward, ep)

#             if ep % 25 == 0:
#                 save_path = f"logs/models/model_ep_{ep}.pt"
#                 self.save_model(save_path)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.utils.conversions import aec_to_parallel
from torch.utils.tensorboard import SummaryWriter
import time

# Import your updated environment.
from lib.environments.petting_zoo import env  # Ensure this points to your updated env module

# -----------------------------
# Define the Policy Network
# -----------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, hidden_dim=128, action_dim=6):
        super(PolicyNetwork, self).__init__()
        # Each observation is a batch of patches of shape (n, 99)
        self.fc1 = nn.Linear(99, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Expects x of shape (batch, n, 99) where n is the number of patches.
        Returns:
          - logits: (batch, action_dim)
          - value: (batch, 1)
        """
        x = torch.relu(self.fc1(x))          # shape: (batch, n, hidden_dim)
        x_mean = x.mean(dim=1)                 # aggregate over patches dimension -> (batch, hidden_dim)
        logits = self.actor_head(x_mean)       # (batch, action_dim)
        value = self.critic_head(x_mean)       # (batch, 1)
        return logits, value

    def act(self, observation):
        """
        Given a NumPy observation of shape (n, 99), computes the action.
        Returns:
          - action (int): chosen discrete action (0...5)
          - log_prob (Tensor): log probability of that action
          - value (Tensor): estimated state value
        """
        # Add batch dimension: (1, n, 99)
        obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)
        logits, value = self.forward(obs_tensor)  # logits: (1, action_dim), value: (1, 1)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs.squeeze(0))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.squeeze(0)


# -----------------------------
# Define the RL Runner Class
# -----------------------------
class RLRunner:
    def __init__(self, discount=0.99):
        # Create and wrap the environment in parallel mode.
        aec_env = env(render_mode="none")
        aec_env.metadata["is_parallelizable"] = True
        parallel_env = aec_to_parallel(aec_env)
        self.env = parallel_env

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = PolicyNetwork(hidden_dim=128, action_dim=6).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.discount = discount
        self.writer = SummaryWriter(log_dir="logs/models")

    def run_episode(self):
        obs, _ = self.env.reset()  
        dones = {agent: False for agent in obs.keys()}
        transitions = {agent: [] for agent in obs.keys()}
        total_rewards = {agent: 0.0 for agent in obs.keys()}

        while not all(dones.values()):
            actions = {}
            # For each agent, select an action using the shared policy.
            for agent, patch_obs in obs.items():
                counter = 9
                while counter >= 0:
                    # Now patch_obs is directly a NumPy array of shape (n, 99)
                    action, log_prob, value = self.policy.act(patch_obs)
                    # Create one-hot vector for the chosen discrete action.
                    one_hot = np.eye(6, dtype=np.float32)[action]  # shape (6,)
                    # Repeat it for each patch in the observation.
                    action_array = np.repeat(one_hot[np.newaxis, :], repeats=patch_obs.shape[0], axis=0)
                    actions[agent] = action_array
                    # Save transition (reward will be appended after stepping the env).
                    transitions[agent].append([patch_obs, action, log_prob, value, None])
                    # Step the environment.
                    print(actions)
                    next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
                    dones = {agent: terminations[agent] or truncations[agent] for agent in terminations}
                    counter -= 1
                    

            # Update rewards in transitions and accumulate total rewards.
            for agent in obs.keys():
                # Adjust reward scaling if needed (here we still divide by 4).
                total_rewards[agent] += (rewards[agent] / 4.0)
                transitions[agent][-1][-1] = rewards[agent]
            obs = next_obs

        return transitions, total_rewards

    def update_policy(self, transitions):
        all_obs, all_actions, all_log_probs, all_returns, all_values = [], [], [], [], []
        
        # Combine trajectories from all agents.
        for agent, traj in transitions.items():
            rewards = [t[4] for t in traj]
            returns = []
            R = 0.0
            # Compute discounted returns (backwards).
            for r in rewards[::-1]:
                R = r + self.discount * R
                returns.insert(0, R)
            for (obs_np, action, log_prob, value, _), R in zip(traj, returns):
                # Do NOT average over patches here; let the network perform the patch-level aggregation.
                obs_tensor = torch.from_numpy(obs_np).float()  # shape: (n, 99)
                all_obs.append(obs_tensor)
                all_actions.append(action)
                all_log_probs.append(log_prob)
                all_values.append(value)
                all_returns.append(R)
        
        # Stack observations along a new batch dimension: (batch, n, 99)
        obs_tensor = torch.stack(all_obs).to(self.device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.stack(all_log_probs).to(self.device)
        returns_tensor = torch.tensor(all_returns, dtype=torch.float32).to(self.device)
        values_tensor = torch.stack(all_values).squeeze()  # shape: (batch,)
        
        # Compute advantages.
        advantages = returns_tensor - values_tensor

        # Re-run policy on the observations.
        logits, values = self.policy.forward(obs_tensor)  # logits: (batch, action_dim), values: (batch, 1)
        values = values.squeeze(1)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions_tensor)

        # Compute the loss: policy loss plus value loss.
        policy_loss = -(new_log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def save_model(self, path):
        """
        Saves the model and optimizer states to the given path.
        """
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Loads the model and optimizer states from the given path.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")

    def train(self, num_episodes):
        for ep in range(num_episodes):
            transitions, total_rewards = self.run_episode()
            loss = self.update_policy(transitions)
            print(f"Episode {ep} - Loss: {loss:.4f}, Rewards: {total_rewards}")
            
            # Log metrics to TensorBoard.
            self.writer.add_scalar("Loss", loss, ep)
            for agent, reward in total_rewards.items():
                self.writer.add_scalar(f"Reward/{agent}", reward, ep)

            if ep % 25 == 0:
                save_path = f"logs/models/model_ep_{ep}.pt"
                self.save_model(save_path)


