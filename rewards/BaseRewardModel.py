from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseReward(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

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
        return self.forward(agent_latents, expert_latents)
