import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=30, uniform_bias_init=False,
                 hidden_layers=2, activation="sig"):
        super(PolicyNetwork, self).__init__()
        # ``hidden_layers`` = number of hidden Linear layers (each followed by
        # the activation specified by ``activation``: 'sig' -> Sigmoid (default,
        # legacy), 'relu' -> ReLU, 'tanh' -> Tanh). ``hidden_dim`` = nodes per
        # hidden layer. The output layer is always a final
        # Linear(hidden_dim, output_dim). Default 2x30 with sigmoid
        # reproduces the legacy two-hidden-layer architecture exactly.
        n_hidden = max(1, int(hidden_layers))
        act_key = str(activation).lower()
        if act_key in ("sig", "sigmoid"):
            act_cls = nn.Sigmoid
        elif act_key == "relu":
            act_cls = nn.ReLU
        elif act_key == "tanh":
            act_cls = nn.Tanh
        else:
            raise ValueError(
                f"PolicyNetwork: unknown activation {activation!r}; "
                f"expected one of 'sig', 'relu', 'tanh'.")
        self.activation = act_key
        layers = []
        prev = input_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(act_cls())
            prev = hidden_dim
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=-1)

        # Optional uniform-bias init on the output layer (off by default,
        # enabled via --uniform_bias_init in train.py):
        #   - bias = 0  -> no action preferred a priori
        #   - weights downscaled by 0.01 -> input*W ~= 0 on average regardless
        #     of input distribution or activation asymmetry
        # Result: logits ~= 0 at gen 1 -> softmax ~= uniform. Prevents
        # species with systematically asymmetric input (e.g. seals) from
        # locking into a saturated rest=100% attractor already at gen 1.
        # Default OFF: in line with konvergensproblem.txt's recommendation
        # to let the biological rules drive behaviour without
        # hacks that mask symptoms.
        if uniform_bias_init:
            with torch.no_grad():
                output_layer = self.net[-1]
                output_layer.bias.zero_()
                output_layer.weight.mul_(0.01)

    def forward(self, x):
        # x shape: (Batch, input_dim)
        logits = self.net(x)
        return self.softmax(logits)

    def get_action_logits_torch(self, flat_state):
        """Batched forward returning pre-softmax logits (N, output_dim).

        Used by the environment to apply masking *before* softmax, so that
        invalid actions (e.g. eat with no prey present, move when DM cannot
        move) contribute no probability mass and produce no gradient.
        """
        with torch.no_grad():
            return self.net(flat_state)
