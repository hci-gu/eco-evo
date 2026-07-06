import numpy as np
import torch

from lib.environments.ecosystem_env import decisions


def rebuild_batched_weights(env):
    env._batched_ready = False
    env._Ws = None
    env._bs = None
    if not all(fid in env.policies for fid in env.dm_ids):
        return

    first = env.policies[env.dm_ids[0]]
    layers0 = [module for module in first.net if isinstance(module, torch.nn.Linear)]
    n_layers = len(layers0)
    if n_layers < 2:
        return

    ref_shapes = [(lin.in_features, lin.out_features) for lin in layers0]
    out_dim = ref_shapes[-1][1]
    max_in_dim = int(env.max_in_dim)

    for i, fid in enumerate(env.dm_ids):
        layers = [
            module for module in env.policies[fid].net
            if isinstance(module, torch.nn.Linear)
        ]
        if len(layers) != n_layers:
            return
        if layers[0].in_features != int(env.per_dm_in_dim[i]):
            return
        if layers[0].out_features != ref_shapes[0][1]:
            return
        for k in range(1, n_layers):
            if (
                layers[k].in_features != ref_shapes[k][0]
                or layers[k].out_features != ref_shapes[k][1]
            ):
                return

    weights = []
    biases = []
    for k, (in_k, out_k) in enumerate(ref_shapes):
        if k == 0:
            weight = torch.zeros(env.N_dm, max_in_dim, out_k)
        else:
            weight = torch.empty(env.N_dm, in_k, out_k)
        bias = torch.empty(env.N_dm, out_k)
        weights.append(weight)
        biases.append(bias)

    for i, fid in enumerate(env.dm_ids):
        layers = [
            module for module in env.policies[fid].net
            if isinstance(module, torch.nn.Linear)
        ]
        in_dim_i = int(env.per_dm_in_dim[i])
        weights[0][i, :in_dim_i, :] = layers[0].weight.detach().t()
        biases[0][i] = layers[0].bias.detach()
        for k in range(1, n_layers):
            weights[k][i] = layers[k].weight.detach().t()
            biases[k][i] = layers[k].bias.detach()

    env._Ws = weights
    env._bs = biases
    if n_layers == 3:
        env._W1, env._b1 = weights[0], biases[0]
        env._W2, env._b2 = weights[1], biases[1]
        env._W3, env._b3 = weights[2], biases[2]
    env._in_dim = max_in_dim
    env._out_dim = out_dim
    env._batched_ready = True


def batched_policy_forward(env, obs_batch, return_logits=False):
    weights = env._Ws
    biases = env._bs
    h = obs_batch
    for k in range(len(weights) - 1):
        h = torch.sigmoid(torch.bmm(h, weights[k]) + biases[k].unsqueeze(1))
    logits = torch.bmm(h, weights[-1]) + biases[-1].unsqueeze(1)
    if return_logits:
        return logits
    return torch.softmax(logits, dim=-1)


class PolicyController:
    def __init__(self, env):
        self.env = env

    def forward(self, observation):
        env = self.env
        obs_np = observation.features
        obs_t = torch.from_numpy(obs_np)
        num_actions = observation.action_mask.shape[1]

        if env._batched_ready and obs_np.shape[-1] == env._in_dim:
            with torch.no_grad():
                logits_t = batched_policy_forward(
                    env, obs_t, return_logits=True)
            logits = logits_t.detach().cpu().numpy()
        else:
            logits = np.empty(
                (env.N_dm, env.H * env.W, num_actions), dtype=env.dtype)
            for i, fid in enumerate(env.dm_ids):
                if fid in env.policies:
                    in_dim_i = int(env.per_dm_in_dim[i])
                    logits_i = env.policies[fid].get_action_logits_torch(
                        obs_t[i, :, :in_dim_i]).detach().cpu().numpy()
                else:
                    logits_i = np.zeros(
                        (env.H * env.W, num_actions), dtype=env.dtype)
                logits[i] = logits_i

        probs = decisions.masked_softmax(env, logits, observation.action_mask)
        return decisions.actions_from_probabilities(
            env, probs, observation.subthreshold_mask)
