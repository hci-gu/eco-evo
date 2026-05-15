import torch
import torch.nn as nn
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=30):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (Batch, input_dim)
        logits = self.net(x)
        return self.softmax(logits)

    def get_action_probs(self, state_tensor):
        # state_tensor shape: (H, W, input_dim)
        H, W, D = state_tensor.shape
        flat_state = state_tensor.reshape(-1, D)
        with torch.no_grad():
            probs = self.forward(flat_state)
        return probs.view(H, W, -1).permute(2, 0, 1).numpy()

    def get_action_probs_torch(self, flat_state):
        """Batched forward returning torch tensor (N, output_dim) without numpy round-trip."""
        with torch.no_grad():
            return self.forward(flat_state)
