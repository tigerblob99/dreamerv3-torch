from __future__ import annotations

import torch

from .BaseRewardModel import BaseReward

class DittoReward(BaseReward):
    def __init__(
        self,
        metric: str = "max_cos",
    ) -> None:
        super().__init__()
        self.metric = metric.lower()

    def forward(self, agent_latents: torch.Tensor, expert_latents: torch.Tensor) -> torch.Tensor:
        # Inputs: (Batch, Dim) or (Batch, Time, Dim)
        
        if self.metric == "mse":
            dist = (agent_latents - expert_latents).pow(2).sum(dim=-1)
            return -dist

        elif self.metric == "max_cos":
            n_i = torch.norm(agent_latents, dim=-1, keepdim=True)
            n_t = torch.norm(expert_latents, dim=-1, keepdim=True)

            # Find max norm between the pair
            max_norm = torch.max(n_i, n_t)
            
            dot_prod = (agent_latents * expert_latents).sum(dim=-1, keepdim=True)
            
            # max_cos = Dot / Max_Norm^2
            score = dot_prod / (max_norm.pow(2) + 1e-8)
            return score.squeeze(-1)

        else:
            raise NotImplementedError(f"Metric {self.metric} not implemented")
