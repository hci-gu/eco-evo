import numpy as np
import random
from pettingzoo.utils import wrappers
# from lib.model import SingleSpeciesModel
from lib.environments.petting_zoo import env as petting_zoo_env
import lib.constants as const
from torch.utils.tensorboard import SummaryWriter
from pettingzoo.utils.conversions import aec_to_parallel
import supersuit as ss

from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.distributions import (
    DiagGaussianDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            for k, v in info.items():
                if k.startswith("info--"):
                    self.logger.record_mean(k[6:], 0 if np.isnan(v) else v)
        return True


class PbmCnn(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super(PbmCnn, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, features_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(features_dim, features_dim, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(observations)


class PbmValueNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        n_dim = x.ndim
        if n_dim == 3:
            x = x[None]
        result = nn.functional.avg_pool3d(x, x.shape[-3:])
        return result.item() if n_dim == 3 else result


class PbmActionNet(nn.Module):
    def __init__(self, in_dim, n_actions):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=n_actions, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        if x.ndim == 3:
            x = th.permute(x, (1, 2, 0))
        else:
            x = th.permute(x, (0, 2, 3, 1))
        return x


class PbmFeatureExtractor(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        device="auto",
    ):
        super(PbmFeatureExtractor, self).__init__()
        device = get_device(device)

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Conv2d(feature_dim, last_layer_dim_pi, 1),
            nn.ReLU(),
            nn.Conv2d(last_layer_dim_pi, last_layer_dim_pi, 1),
            nn.ReLU(),
        ).to(device)
        # Value network
        self.value_net = nn.Sequential(
            nn.Conv2d(feature_dim, last_layer_dim_vf, 1),
            nn.ReLU(),
            nn.Conv2d(last_layer_dim_pi, last_layer_dim_pi, 1),
            nn.ReLU(),
        ).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class PbmActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(PbmActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PbmFeatureExtractor(self.features_dim, device=self.device)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        # We need the action distribution to be only for one cell, hence we reconstruct
        # with the last action_space_dimension.
        high = self.action_space.high[0, 0, 0]
        low = self.action_space.low[0, 0, 0]
        space = gym.spaces.Box(low=low, high=high, shape=self.action_space.shape[-1:])
        self.action_dist = make_proba_distribution(
            space, use_sde=self.use_sde, dist_kwargs=self.dist_kwargs
        )

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi,
                latent_sde_dim=latent_dim_pi,
                log_std_init=self.log_std_init,
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        # self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # We need a somewhat different value net since the input is not one dimensioanl.
        self.value_net = PbmValueNet(self.mlp_extractor.latent_dim_vf)
        # And action net
        n_actions = self.action_space.shape[-1]
        self.action_net = PbmActionNet(self.mlp_extractor.latent_dim_pi, n_actions)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]

                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = self._log_prob(distribution, actions)
        # We need to reduce the dimension of `log_prob` by summing over the grid
        # dimensions.
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = self._log_prob(
            distribution, actions.reshape((-1, *self.action_space.shape))
        )
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def _log_prob(self, distribution, actions):
        """This is replacement for log_prob in DiagGaussianDistribution. We don't want
        to collapse all dimensions."""
        tensor = distribution.log_prob(actions)
        dims = tuple(range(1, tensor.ndim))
        tensor = tensor.sum(dim=dims)
        return tensor


def get_lr(lr_low=1e-5, lr_high=1e-5):
    def _lr(x):
        # assert 0 <= x <= 1 # This seems to break at the final step, got negative x values.
        warmup_fraction = 0.1
        warmup_lr = 1e-6
        if (1 - x) < warmup_fraction:
            warmup_progress = (1 - x) / warmup_fraction
            return warmup_lr + warmup_progress * (lr_low - warmup_lr)
        return lr_low + x * (lr_high - lr_low)

    return _lr

class ChannelFlip(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        grid_shape = env.observation_space.shape[:2]
        n_channels = np.prod(env.observation_space.shape[2:])
        new_shape = (n_channels, *grid_shape)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=new_shape)

    def observation(self, observation):
        return observation.reshape(*observation.shape[:2], -1).transpose(2, 0, 1)


class FlattenAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = env.action_space.shape
        # We flatten the last two dimensions that are the actual actions.
        new_low = env.action_space.low.reshape(*old_shape[:2], -1).copy()
        new_shape = new_low.shape
        new_high = env.action_space.high.reshape(new_shape).copy()
        self.action_space = gym.spaces.Box(new_low, new_high, shape=new_shape)

    def action(self, action):
        return action.reshape(self.env.action_space.shape)


class DropPrimaryProducerAction(gym.ActionWrapper):
    """Drop the action from the primary producer since it does not have any active
    actions anyway."""

    def __init__(self, env, pp_index=0):
        super().__init__(env)
        new_low = np.delete(env.action_space.low, pp_index, axis=2)
        new_high = np.delete(env.action_space.high, pp_index, axis=2)
        self.action_space = gym.spaces.Box(new_low, new_high, shape=new_low.shape)
        self.pp_index = pp_index

    def action(self, action):
        return np.insert(action, self.pp_index, 0, axis=2)


def create_env():
    env = gym.make('gymnasium_env/Ecotwin-v0', render_mode="none")
    # aec_env = petting_zoo_env(render_mode="none")
    # aec_env.metadata["is_parallelizable"] = True
    # parallel_env = aec_to_parallel(aec_env)
    # vec_env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
    # wrapped_env = SingleAgentWrapper(vec_env, agent_key="cod")
    
    env = ChannelFlip(env)
    # env = FlattenAction(DropPrimaryProducerAction(env, pp_index=0))
    env = FlattenAction(env)
    # env = Monitor(env)
    return env

class RLRunner:
    def __init__(self, num_episodes=100, learning_rate=1e-3, gamma=0.99, log_dir="logs/models"):
        env = SubprocVecEnv([lambda: create_env() for i in range(6)])
        self.env = env

        self.num_episodes = num_episodes
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def evaluate(self, model_file, num_episodes=10):
        model = PPO.load(model_file)
        model.set_env(self.env)

        # for episode in range(num_episodes):

        # if normalize:
        #     if norm_file is None:
        #         print("A parameter file for the normalized values must be set.")
        #         exit()
        #     env = VecNormalize.load(norm_file, env.venv)

    def train(self):
        policy_kwargs = dict(
            features_extractor_class=PbmCnn,
            features_extractor_kwargs=dict(features_dim=64),
            share_features_extractor=True,
            net_arch=[],
        )
        model = PPO(
            PbmActorCriticPolicy,
            self.env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=256,
            tensorboard_log="logs/models",
            device="cpu",
            learning_rate=1e-4,
        )

        callback = [
            EvalCallback(self.env, eval_freq=5 * model.n_steps),
            TensorboardCallback(),
        ]

        
        total_timesteps = 100000
        interval = 10000  # number of timesteps per learning chunk
        
        timesteps_done = 0
        while timesteps_done < total_timesteps:
            model.learn(
                total_timesteps=interval,
                reset_num_timesteps=False,
                callback=callback,
            )
            timesteps_done += interval
            now = datetime.now().strftime("%Y%m%d-%H%M")
            model.save(f"./models/model_{model.num_timesteps}_steps_{now}.pth")
        # for episode in range(self.num_episodes):
        #     self.env.reset()
        #     done = False

        #     now = datetime.now().strftime("%Y%m%d-%H%M")
        #     model.save(f"./models/model_{model.num_timesteps}_steps_{now}.pth")

        #     # Run the environment with the agent_iter loop.
        #     while not done:
        #         for agent in self.env.agent_iter():
        #             observation, reward, terminated, truncated, info = self.env.last()
                    
        #             model = self.models[agent]
        #             action = model.forward(observation)
                    
        #             self.env.step(action)
        #             print(agent)
                    
        #             if terminated or truncated:
        #                 done = True
        #                 break
            
        #     print(f"Episode {episode+1} complete")
        # self.writer.close()
        # return self.models

