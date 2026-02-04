import copy

import torch
from torch import nn

import networks
import tools


class Critic(nn.Module):
    def __init__(self, config, feat_dim: int):
        super().__init__()
        self._config = config
        self._use_amp = True if config.precision == 16 else False
        self.value = networks.MLP(
            feat_dim,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )

    def update(
        self,
        features,
        rewards,
        discounts=None,
        return_targets=False,
    ):
        """
        features: Tensor shaped (B, T, D).
        rewards: Output from BaseRewardModel, shaped (B, T) or (B, T, 1).
                 Transition rewards (length T-1) are supported.
        discounts: Optional per-step discount/continuation, shaped like rewards.
        return_targets: If True, also return (targets, weights, metrics).
        """
        self._update_slow_target()

        if rewards.ndim == 2:
            rewards = rewards.unsqueeze(-1)
        if discounts is None:
            discounts = torch.full_like(rewards, self._config.discount)
        elif discounts.ndim == 2:
            discounts = discounts.unsqueeze(-1)

        feats_t = features.permute(1, 0, 2)
        rewards_t = rewards.permute(1, 0, 2)
        discounts_t = discounts.permute(1, 0, 2)
        reward_len = rewards_t.shape[0]
        if discounts_t.shape[0] != reward_len:
            discounts_t = discounts_t[:reward_len]

        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda", enabled=self._use_amp, dtype=torch.float16
            ):
                values = self.value(feats_t).mode()
            target, weights, value_seq = self._lambda_return(
                rewards_t, values, discounts_t
            )
            target = torch.stack(target, dim=1)

        value_input = feats_t[:reward_len]
        with tools.RequiresGrad(self.value):
            with torch.amp.autocast(
                device_type="cuda", enabled=self._use_amp, dtype=torch.float16
            ):
                value_dist = self.value(value_input.detach())
                value_loss = -value_dist.log_prob(target.detach())
                if self._config.critic["slow_target"]:
                    slow_target = self._slow_value(value_input.detach())
                    value_loss -= value_dist.log_prob(slow_target.mode().detach())
                value_loss = torch.mean(weights * value_loss[:, :, None])

        metrics = {}
        metrics.update(tools.tensorstats(value_seq, "value"))
        metrics.update(tools.tensorstats(target, "target"))
        with tools.RequiresGrad(self):
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        if return_targets:
            return target, weights, value_seq, metrics
        return metrics

    def _lambda_return(self, reward, values, discount):
        reward_len = reward.shape[0]
        if values.shape[0] < reward_len:
            raise ValueError("Not enough value steps for reward sequence.")
        value_seq = values[:reward_len]
        bootstrap = values[reward_len] if values.shape[0] > reward_len else values[-1]
        target = tools.lambda_return(
            reward,
            value_seq,
            discount,
            bootstrap=bootstrap,
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value_seq

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
