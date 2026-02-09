import torch
from torch import nn

import tools
from joint_train import ActionMLP


to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """Running quantile range for reward/return normalization."""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class Actor(nn.Module):
    def __init__(self, config, feat_dim: int, act_dim: int):
        super().__init__()
        self._config = config
        self._use_amp = True if config.precision == 16 else False
        self._use_reward_ema = bool(getattr(config, "reward_EMA", False))
        layers = int(getattr(config, "mlp_layers", config.actor["layers"]))
        units = int(getattr(config, "policy_units", 1024))
        self.actionMLP = ActionMLP(
            feat_dim,
            (act_dim,),
            layers=layers,
            units=units,
            act=config.act,
            norm=config.norm,
            dist=config.actor["dist"],
            std=config.actor["std"],
            min_std=config.actor["min_std"],
            max_std=config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            device=config.device,
        )
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actionMLP.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        if self._use_reward_ema:
            self.register_buffer("ema_vals", torch.zeros((2,), device=config.device))
            self.reward_ema = RewardEMA(device=config.device)

    def generate_actions(self, features, sample=True, return_dist=False):
        with torch.amp.autocast(
            device_type="cuda", enabled=self._use_amp, dtype=torch.float16
        ):
            if return_dist:
                return self.actionMLP(features, return_dist=True)
            return self.actionMLP(features, sample=sample)

    def update(self, features, actions, target, weights, value_seq):
        feats_t = features.permute(1, 0, 2)
        actions_t = actions.permute(1, 0, 2)
        target_t = target
        weights_t = weights
        baseline_t = value_seq
        reward_len = target_t.shape[0]
        baseline_t = baseline_t[:reward_len]
        if weights_t.shape[0] != reward_len:
            weights_t = weights_t[:reward_len]
        if target_t.ndim == 2:
            target_t = target_t.unsqueeze(-1)
        if weights_t.ndim == 2:
            weights_t = weights_t.unsqueeze(-1)
        if baseline_t.ndim == 2:
            baseline_t = baseline_t.unsqueeze(-1)

        with tools.RequiresGrad(self.actionMLP):
            with torch.amp.autocast(
                device_type="cuda", enabled=self._use_amp, dtype=torch.float16
            ):
                policy = self.actionMLP(feats_t[:reward_len].detach(), return_dist=True)
                log_prob = policy.log_prob(actions_t[:reward_len])[:, :, None]
                entropy = policy.entropy()[:, :, None]
                if self._use_reward_ema:
                    offset, scale = self.reward_ema(target_t, self.ema_vals)
                    normed_target = (target_t - offset) / scale
                    normed_base = (baseline_t - offset) / scale
                    advantage = (normed_target - normed_base).detach()
                else:
                    normed_target = None
                    advantage = (target_t - baseline_t).detach()
                actor_loss = -weights_t[:reward_len] * log_prob * advantage
                actor_loss -= self._config.actor["entropy"] * entropy
                actor_loss = torch.mean(actor_loss)

        metrics = {"actor_entropy": to_np(torch.mean(entropy))}
        if normed_target is not None:
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actionMLP.parameters()))
        return metrics
