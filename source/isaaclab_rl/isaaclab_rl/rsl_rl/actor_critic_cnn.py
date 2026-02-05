# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any

from rsl_rl.networks import MLP, EmpiricalNormalization


class ActorCriticCNN(nn.Module):
    """Actor-Critic network with CNN support for image observations."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        actor_cnn_cfg: Any | None = None,
        critic_cnn_cfg: Any | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticCNN.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        # Get the observation dimensions
        self.obs_groups = obs_groups

        # Process actor observations
        actor_1d_obs_size = 0
        actor_image_obs = None
        for obs_group in obs_groups["policy"]:
            if len(obs[obs_group].shape) == 4:  # Image observation (B, C, H, W)
                actor_image_obs = obs[obs_group]
            else:  # 1D observation
                actor_1d_obs_size += obs[obs_group].shape[-1]

        # Process critic observations
        critic_1d_obs_size = 0
        critic_image_obs = None
        for obs_group in obs_groups["critic"]:
            if len(obs[obs_group].shape) == 4:  # Image observation (B, C, H, W)
                critic_image_obs = obs[obs_group]
            else:  # 1D observation
                critic_1d_obs_size += obs[obs_group].shape[-1]

        # Actor CNN encoder
        self.actor_cnn = None
        if actor_image_obs is not None and actor_cnn_cfg is not None:
            self.actor_cnn = self._build_cnn_encoder(actor_image_obs, actor_cnn_cfg).to(actor_image_obs.device)
            actor_cnn_output_size = self._get_cnn_output_size(actor_image_obs, self.actor_cnn)
            actor_1d_obs_size += actor_cnn_output_size

        # Critic CNN encoder
        self.critic_cnn = None
        if critic_image_obs is not None and critic_cnn_cfg is not None:
            self.critic_cnn = self._build_cnn_encoder(critic_image_obs, critic_cnn_cfg).to(critic_image_obs.device)
            critic_cnn_output_size = self._get_cnn_output_size(critic_image_obs, self.critic_cnn)
            critic_1d_obs_size += critic_cnn_output_size

        # Normalization
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(shape=[actor_1d_obs_size], until=1.0e8)
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[critic_1d_obs_size], until=1.0e8)

        # Policy
        self.actor = MLP(
            input_dim=actor_1d_obs_size,
            output_dim=num_actions,
            hidden_dims=actor_hidden_dims,
            activation=activation,
        )

        # Value function
        self.critic = MLP(
            input_dim=critic_1d_obs_size,
            output_dim=1,
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )

        # Action noise
        self.state_dependent_std = state_dependent_std
        self.noise_std_type = noise_std_type
        if state_dependent_std:
            self.std = MLP(
                input_dim=actor_1d_obs_size,
                output_dim=num_actions,
                hidden_dims=actor_hidden_dims,
                activation=activation,
            )
        else:
            if noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif noise_std_type == "fixed_diagonal":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions), requires_grad=False)
        
        # Distribution placeholder (will be updated in act())
        self.distribution = None
        # Store action mean, std, and entropy for PPO algorithm
        self.action_mean = None
        self.action_std = None
        self.entropy = None

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _build_cnn_encoder(self, obs: torch.Tensor, cnn_cfg: Any) -> nn.Module:
        """Build a CNN encoder from configuration."""
        layers = []
        in_channels = obs.shape[1]  # C dimension from (B, C, H, W)

        # Handle dict or object config
        def get_cfg(cfg, key, default=None):
            return cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)
        
        def set_cfg(cfg, key, value):
            if isinstance(cfg, dict):
                cfg[key] = value
            else:
                setattr(cfg, key, value)

        # Handle single config or list of configs
        output_channels = get_cfg(cnn_cfg, 'output_channels')
        if not isinstance(output_channels, (list, tuple)):
            set_cfg(cnn_cfg, 'output_channels', [output_channels])
            set_cfg(cnn_cfg, 'kernel_size', [get_cfg(cnn_cfg, 'kernel_size')])
            stride = get_cfg(cnn_cfg, 'stride')
            set_cfg(cnn_cfg, 'stride', [stride if stride is not None else 1])
            max_pool = get_cfg(cnn_cfg, 'max_pool')
            set_cfg(cnn_cfg, 'max_pool', [max_pool if max_pool is not None else False])

        # Build convolutional layers
        output_channels = get_cfg(cnn_cfg, 'output_channels')
        kernel_size = get_cfg(cnn_cfg, 'kernel_size')
        stride = get_cfg(cnn_cfg, 'stride')
        padding = get_cfg(cnn_cfg, 'padding', 'zeros')
        norm = get_cfg(cnn_cfg, 'norm')
        
        for i, out_channels in enumerate(output_channels):
            # Conv layer
            layers.append(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size[i] if isinstance(kernel_size, list) else kernel_size,
                stride=stride[i] if isinstance(stride, list) else stride,
                padding=1 if padding == "zeros" else 0,
            ))

            # Normalization
            if norm == "batch" or (isinstance(norm, list) and norm[i] == "batch"):
                layers.append(nn.BatchNorm2d(out_channels))
            elif norm == "layer" or (isinstance(norm, list) and norm[i] == "layer"):
                # Layer norm for 2D needs to specify normalized shape
                layers.append(nn.GroupNorm(1, out_channels))

            # Activation
            activation = get_cfg(cnn_cfg, 'activation', 'relu')
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "elu":
                layers.append(nn.ELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())

            # Max pooling
            max_pool = get_cfg(cnn_cfg, 'max_pool')
            if max_pool[i] if isinstance(max_pool, list) else max_pool:
                layers.append(nn.MaxPool2d(2, 2))

            in_channels = out_channels

        # Global pooling
        global_pool = get_cfg(cnn_cfg, 'global_pool')
        if global_pool == "max":
            layers.append(nn.AdaptiveMaxPool2d(1))
        elif global_pool == "avg":
            layers.append(nn.AdaptiveAvgPool2d(1))

        # Flatten
        flatten = get_cfg(cnn_cfg, 'flatten', True)
        if flatten:
            layers.append(nn.Flatten())

        return nn.Sequential(*layers)

    def _get_cnn_output_size(self, obs: torch.Tensor, cnn: nn.Module) -> int:
        """Compute the output size of the CNN encoder."""
        with torch.no_grad():
            sample = torch.zeros((1,) + obs.shape[1:], device=obs.device)
            output = cnn(sample)
            return output.shape[-1]

    def _process_observations(self, obs: TensorDict, obs_groups: list[str], cnn: nn.Module | None) -> torch.Tensor:
        """Process observations, concatenating 1D and CNN-encoded image observations."""
        obs_list = []

        for obs_group in obs_groups:
            if len(obs[obs_group].shape) == 4:  # Image observation
                if cnn is not None:
                    encoded = cnn(obs[obs_group])
                    obs_list.append(encoded)
            else:  # 1D observation
                obs_list.append(obs[obs_group])

        return torch.cat(obs_list, dim=-1)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        """Compute actions from observations."""
        actor_obs = self._process_observations(obs, self.obs_groups["policy"], self.actor_cnn)

        if self.actor_obs_normalization:
            actor_obs = self.actor_obs_normalizer(actor_obs)

        self._update_distribution(actor_obs)
        return self.distribution.sample()
    
    def _update_distribution(self, actor_obs: torch.Tensor):
        """Update the action distribution based on actor observations."""
        self.action_mean = self.actor(actor_obs)

        if self.state_dependent_std:
            self.action_std = torch.exp(self.std(actor_obs))
        else:
            self.action_std = torch.exp(self.std)

        self.distribution = Normal(self.action_mean, self.action_std)
        self.entropy = self.distribution.entropy().sum(dim=-1)

    @torch.no_grad()
    def act_inference(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        """Compute deterministic actions from observations (inference mode)."""
        actor_obs = self._process_observations(obs, self.obs_groups["policy"], self.actor_cnn)

        if self.actor_obs_normalization:
            actor_obs = self.actor_obs_normalizer(actor_obs)

        actions_mean = self.actor(actor_obs)
        return actions_mean

    def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        """Evaluate observations to get critic value."""
        critic_obs = self._process_observations(obs, self.obs_groups["critic"], self.critic_cnn)

        if self.critic_obs_normalization:
            critic_obs = self.critic_obs_normalizer(critic_obs)

        return self.critic(critic_obs)
    
    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probability of actions using the current distribution."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_std(self):
        """Get the standard deviation of the action distribution."""
        if self.state_dependent_std:
            return None
        else:
            return torch.exp(self.std)

    def set_std(self, std: torch.Tensor):
        """Set the standard deviation of the action distribution."""
        if not self.state_dependent_std:
            self.std.data = torch.log(std)
    
    def update_normalization(self, obs: TensorDict):
        """Update the normalization statistics for actor and critic observations."""
        if self.actor_obs_normalization:
            actor_obs = self._process_observations(obs, self.obs_groups["policy"], self.actor_cnn)
            self.actor_obs_normalizer.update(actor_obs)
        
        if self.critic_obs_normalization:
            critic_obs = self._process_observations(obs, self.obs_groups["critic"], self.critic_cnn)
            self.critic_obs_normalizer.update(critic_obs)