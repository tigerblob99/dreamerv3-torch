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
        stop_gradient=True,
        is_first=None,
    ):
        """Compatibility wrapper for callers expecting one-step critic update."""
        self.update_slow_target()
        target, value_seq = self.compute_targets(
            features, rewards, discounts, stop_gradient=stop_gradient, is_first=is_first
        )
        metrics = self.update_from_targets(features, target)
        if return_targets:
            return target, value_seq, metrics
        return metrics

    def compute_targets(self, features, rewards, discounts=None, stop_gradient=True, is_first=None):
        """Compute lambda-return targets without applying optimizer updates.

        features: (B, T, D)
        rewards: (B, Tr) or (B, Tr, 1)
        discounts: optional, same leading shape as rewards. Kept for backward
            compatibility and ignored by target computation.
        is_first: optional (B, T) tensor of boundary flags. When provided,
            continuation masks are derived as ``1 - is_first`` so that
            lambda returns do not bootstrap across episode splice points.
        returns:
            target: (Tr-1, B, 1)
            value_seq: (Tr-1, B, 1)

        Non-terminal lambda targets use Dreamer-style recursion with fixed
        gamma=config.discount and lambda=config.discount_lambda. Terminal anchor
        is Director-style R=V(s_T) from the last predicted value.
        """
        if rewards.ndim == 2:
            rewards = rewards.unsqueeze(-1)
        feats_t = features.permute(1, 0, 2)
        rewards_t = rewards.permute(1, 0, 2)

        # Build continuation mask from is_first: (B, T) -> (T, B, 1)
        cont = None
        if is_first is not None:
            cont = (1.0 - is_first).unsqueeze(-1).permute(1, 0, 2)

        if stop_gradient:
            with torch.no_grad():
                with torch.amp.autocast(
                    device_type="cuda", enabled=self._use_amp, dtype=torch.float16
                ):
                    values = self.value(feats_t).mode()
                target, value_seq = self._lambda_return(rewards_t, values, cont=cont)
            return target, value_seq

        self.value.requires_grad_(False)
        try:
            with torch.amp.autocast(
                device_type="cuda", enabled=self._use_amp, dtype=torch.float16
            ):
                values = self.value(feats_t).mode()
            target, value_seq = self._lambda_return(rewards_t, values, cont=cont)
        finally:
            self.value.requires_grad_(True)
        return target, value_seq

    def update_from_targets(self, features, target):
        """Optimize value network from precomputed lambda-return targets."""
        feats_t = features.permute(1, 0, 2)
        target_t = target
        if target_t.ndim == 2:
            target_t = target_t.unsqueeze(-1)
        reward_len = target_t.shape[0]

        value_input = feats_t[:reward_len]
        with tools.RequiresGrad(self.value):
            with torch.amp.autocast(
                device_type="cuda", enabled=self._use_amp, dtype=torch.float16
            ):
                value_dist = self.value(value_input.detach())
                value_loss = -value_dist.log_prob(target_t.detach())
                if self._config.critic["slow_target"]:
                    slow_target = self._slow_value(value_input.detach())
                    value_loss -= value_dist.log_prob(slow_target.mode().detach())
                value_loss = torch.mean(value_loss)

        metrics = {}
        metrics.update(tools.tensorstats(value_dist.mode(), "value"))
        metrics.update(tools.tensorstats(target_t, "target"))
        with tools.RequiresGrad(self):
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return metrics

    def _lambda_return(self, reward, values, cont=None):
        """Compute lambda-return targets.

        Parameters
        ----------
        reward : (T, B, 1)
        values : (T, B, 1)
        cont : optional (T, B, 1) continuation mask.  ``cont[t] = 0`` at
            episode splice boundaries prevents bootstrapping across them.
            When *None*, all continuations default to 1 (backward compatible).
        """
        if values.shape[0] < reward.shape[0]:
            raise ValueError("Not enough value steps for reward sequence.")

        reward_len = reward.shape[0]
        if reward_len < 2:
            raise ValueError(
                "Lambda targets require reward horizon >= 2."
            )

        # Keep non-terminal alignment: target length is reward_len - 1.
        reward_l = reward[:-1]
        value_t = values[: reward_len - 1]
        value_tp1 = values[1:reward_len]

        # cont_l[t] gates the transition from step t to t+1.
        if cont is not None:
            cont_l = cont[1:reward_len]
        else:
            cont_l = torch.ones_like(reward_l)

        gamma = float(self._config.discount)
        lambda_ = float(self._config.discount_lambda)
        # R_T = V(s_T), where s_T is the last
        # available state in the sequence.
        ret = values[reward_len - 1]
        returns = []
        for t in range(reward_l.shape[0] - 1, -1, -1):
            ret = reward_l[t] + gamma * cont_l[t] * ((1.0 - lambda_) * value_tp1[t] + lambda_ * ret)
            returns.append(ret)
        target = torch.stack(returns[::-1], dim=0)
        return target, value_t

    def update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

    def _update_slow_target(self):
        # Backward-compatible alias.
        self.update_slow_target()
