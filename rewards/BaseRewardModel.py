from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseReward(nn.Module, ABC):
    def __init__(
        self,
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.reward_scale = float(reward_scale)
        self.reward_shift = float(reward_shift)

    @abstractmethod
    def forward(
        self, agent_latents: torch.Tensor, expert_latents: torch.Tensor
    ) -> torch.Tensor:
        """
        Expected shapes (batch-first):
          - agent_latents: (B, D) or (B, T, D)
          - expert_latents: (B, D) or (B, T, D)
        Returns:
          - reward: (B,) or (B, T) (or (B, T-1) if the reward uses transitions)
        """
        raise NotImplementedError

    def reward(
        self, agent_latents: torch.Tensor, expert_latents: torch.Tensor
    ) -> torch.Tensor:
        reward = self.forward(agent_latents, expert_latents)
        return self._apply_reward_transform(reward)

    def _apply_reward_transform(self, reward: torch.Tensor) -> torch.Tensor:
        return reward * self.reward_scale + self.reward_shift
