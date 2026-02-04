from __future__ import annotations

import torch
from torch import nn

from .BaseRewardModel import BaseReward


class Discriminator(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 256, layers: int = 2, act: str = "SiLU"
    ) -> None:
        super().__init__()
        act_cls = getattr(nn, act)
        net = []
        dim = input_dim
        for _ in range(layers):
            net.append(nn.Linear(dim, hidden_dim))
            net.append(act_cls())
            dim = hidden_dim
        net.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x).squeeze(-1)
        probs = torch.sigmoid(logits)
        return logits, probs


class GailReward(BaseReward):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        layers: int = 2,
        lr: float = 1e-4,
        act: str = "SiLU",
        use_transitions: bool = True,
    ) -> None:
        super().__init__()
        self.use_transitions = use_transitions
        disc_input_dim = input_dim * 2 if use_transitions else input_dim
        self.discriminator = Discriminator(
            input_dim=disc_input_dim, hidden_dim=hidden_dim, layers=layers, act=act
        )
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        agent_latents: torch.Tensor,
        expert_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if expert_latents is not None:
            self.train_step(agent_latents, expert_latents)
        return self._reward(agent_latents)

    def train_step(
        self,
        agent_latents: torch.Tensor,
        expert_latents: torch.Tensor,
    ) -> torch.Tensor:
        agent_inputs = self._build_inputs(agent_latents.detach())
        expert_inputs = self._build_inputs(expert_latents.detach())
        agent_flat, _ = self._flatten(agent_inputs)
        expert_flat, _ = self._flatten(expert_inputs)

        agent_logits, _ = self.discriminator(agent_flat)
        expert_logits, _ = self.discriminator(expert_flat)

        agent_targets = torch.zeros_like(agent_logits)
        expert_targets = torch.ones_like(expert_logits)

        loss = self.loss_fn(agent_logits, agent_targets) + self.loss_fn(
            expert_logits, expert_targets
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def _reward(self, agent_latents: torch.Tensor) -> torch.Tensor:
        agent_inputs = self._build_inputs(agent_latents)
        agent_flat, orig_shape = self._flatten(agent_inputs)
        with torch.no_grad():
            _, probs = self.discriminator(agent_flat)
        reward = probs.reshape(orig_shape)
        return reward

    def _build_inputs(self, latents: torch.Tensor) -> torch.Tensor:
        if not self.use_transitions:
            return latents
        if latents.shape[1] < 2:
            raise ValueError("Need at least 2 time steps to build transitions.")
        prev = latents[:, :-1, :]
        nxt = latents[:, 1:, :]
        return torch.cat([prev, nxt], dim=-1)

    @staticmethod
    def _flatten(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
        orig_shape = latents.shape[:-1]
        flat = latents.reshape(-1, latents.shape[-1])
        return flat, orig_shape
