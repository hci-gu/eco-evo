"""Packed policy inference with the existing per-species checkpoint layout."""

import torch
import torch.nn.functional as F

from lib.runners.policy import PolicyNetwork


class PolicyBank:
    def __init__(self, model, hidden_dim=30, hidden_layers=2, activation="sig",
                 uniform_bias_init=False, seed=0):
        if hidden_dim < 1 or hidden_layers < 1:
            raise ValueError("Policy dimensions must be positive")
        self.model = model
        self.activation = "sig" if activation == "sigmoid" else activation
        if self.activation not in ("sig", "relu", "tanh"):
            raise ValueError("Activation must be sig, relu, or tanh")
        # Initialize on CPU using the same Torch initialization as the reference,
        # then upload once. Forking avoids changing the caller's RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.policies = {
                fid: PolicyNetwork(model.in_dims[i], model.A, hidden_dim=hidden_dim,
                                   hidden_layers=hidden_layers, activation=self.activation,
                                   uniform_bias_init=uniform_bias_init).to(model.device)
                for i, fid in enumerate(model.dm_ids)
            }
        for policy in self.policies.values():
            policy.requires_grad_(False)
        self.shapes = []
        for fid in model.dm_ids:
            self.shapes.append(tuple(tuple(p.shape) for p in self.policies[fid].parameters()))

    def flat_weights(self):
        return [torch.cat([p.detach().reshape(-1) for p in self.policies[f].parameters()])
                for f in self.model.dm_ids]

    def install(self, flat_weights):
        for fid, flat in zip(self.model.dm_ids, flat_weights):
            offset = 0
            for parameter in self.policies[fid].parameters():
                count = parameter.numel()
                parameter.copy_(flat[offset:offset + count].view_as(parameter))
                offset += count

    def pack(self, candidates):
        """candidates[d]: [candidate, original parameter count]."""
        weights, biases = [], []
        offsets = [0] * self.model.D
        for layer in range(len(self.shapes[0]) // 2):
            ws, bs = [], []
            for d, flat in enumerate(candidates):
                out_dim, in_dim = self.shapes[d][2 * layer]
                count = out_dim * in_dim
                offset = offsets[d]
                w = flat[:, offset:offset + count].reshape(-1, out_dim, in_dim).transpose(1, 2)
                if layer == 0:
                    w = F.pad(w, (0, 0, 0, self.model.F - in_dim))
                ws.append(w)
                bs.append(flat[:, offset + count:offset + count + out_dim])
                offsets[d] += count + out_dim
            weights.append(torch.stack(ws, 1))
            biases.append(torch.stack(bs, 1))
        return tuple(weights), tuple(biases)

    def forward(self, observations, weights, biases):
        """[candidate, D, worlds*cells, F] with weights shared across worlds."""
        h = observations
        for k in range(len(weights)):
            h = torch.matmul(h, weights[k]) + biases[k][:, :, None]
            if k < len(weights) - 1:
                if self.activation == "sig":
                    h = torch.sigmoid(h)
                elif self.activation == "relu":
                    h = torch.relu(h)
                else:
                    h = torch.tanh(h)
        return h
