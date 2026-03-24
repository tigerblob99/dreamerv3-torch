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

    @staticmethod
    def _normal_dist(dist, name):
        if isinstance(dist, tools.ContDist):
            normal = dist.base_dist
        elif isinstance(dist, tools.FaithfulContDist):
            normal = dist.base_dist.base_dist
        else:
            raise TypeError(
                f"{name} must be ContDist or FaithfulContDist, got {type(dist).__name__}."
            )
        if type(normal) is not torch.distributions.Normal:
            raise TypeError(
                f"{name} must wrap the exact Normal used by actor dist='normal' or "
                f"'faithful_normal', got {type(normal).__name__}."
            )
        if normal.loc.ndim != 3 or normal.loc.shape != normal.scale.shape:
            raise ValueError(
                f"{name} must have loc/scale shape [time, batch, action], got "
                f"loc={tuple(normal.loc.shape)}, scale={tuple(normal.scale.shape)}."
            )
        return normal

    @staticmethod
    def _distribution_kl(policy, reference_policy, batch_size):
        """Return a per-sample KL with shape [time, batch, 1]."""
        policy_dist = Actor._normal_dist(policy, "policy")
        reference_dist = Actor._normal_dist(reference_policy, "reference_policy")
        if policy_dist.loc.shape != reference_dist.loc.shape:
            raise ValueError(
                "policy and reference_policy must have matching [time, batch, action] "
                f"shapes, got {tuple(policy_dist.loc.shape)} and "
                f"{tuple(reference_dist.loc.shape)}."
            )
        if policy_dist.loc.shape[1] != batch_size:
            raise ValueError(
                f"policy batch dimension must be {batch_size}, got "
                f"{policy_dist.loc.shape[1]}."
            )
        kl = torch.distributions.kl_divergence(policy_dist, reference_dist)
        return kl.sum(dim=-1, keepdim=True)

    def _bc_kl(self, bc_actor, features):
        feats_t = features.permute(1, 0, 2)
        policy = self.actionMLP(feats_t.detach(), return_dist=True)
        with torch.no_grad():
            bc_policy = bc_actor.actionMLP(feats_t.detach(), return_dist=True)
        return self._distribution_kl(policy, bc_policy, feats_t.shape[1])

    def update(
        self, features, actions, target, value_seq, bc_actor=None, bc_kl_features=None
    ):
        feats_t = features.permute(1, 0, 2)
        actions_t = actions.permute(1, 0, 2)
        target_t = target
        baseline_t = value_seq
        mode = str(getattr(self._config, "imag_gradient", "reinforce")).lower()
        if target_t.ndim == 2:
            target_t = target_t.unsqueeze(-1)
        if baseline_t.ndim == 2:
            baseline_t = baseline_t.unsqueeze(-1)
        batch_size = feats_t.shape[1]
        if target_t.ndim != 3 or target_t.shape[1:] != (batch_size, 1):
            raise ValueError(
                f"target must have shape [time, {batch_size}, 1], got "
                f"{tuple(target_t.shape)}."
            )
        if baseline_t.ndim != 3 or baseline_t.shape[1:] != (batch_size, 1):
            raise ValueError(
                f"value_seq must have shape [time, {batch_size}, 1], got "
                f"{tuple(baseline_t.shape)}."
            )
        reward_len = min(target_t.shape[0], feats_t.shape[0], actions_t.shape[0])
        target_t = target_t[:reward_len]
        baseline_t = baseline_t[:reward_len]

        with torch.amp.autocast(
            device_type="cuda", enabled=self._use_amp, dtype=torch.float16
        ):
            policy = self.actionMLP(feats_t[:reward_len].detach(), return_dist=True)
            log_prob = policy.log_prob(actions_t[:reward_len])[:, :, None]
            entropy = policy.entropy()[:, :, None]
            bc_kl = None
            bc_kl_scale = float(self._config.actor.get("bc_kl_scale", 0.0))
            if bc_actor is not None and bc_kl_scale > 0.0:
                kl_features = features if bc_kl_features is None else bc_kl_features
                bc_kl = self._bc_kl(bc_actor, kl_features[:, :reward_len])
            if self._use_reward_ema:
                offset, scale = self.reward_ema(target_t, self.ema_vals)
                normed_target = (target_t - offset) / scale
                normed_base = (baseline_t - offset) / scale
            else:
                normed_target = None
                normed_base = None

            if mode == "dynamics":
                dynamics_target = (
                    normed_target if normed_target is not None else target_t
                )
                actor_loss = -dynamics_target
            elif mode == "reinforce":
                if normed_target is not None and normed_base is not None:
                    advantage = (normed_target - normed_base).detach()
                else:
                    advantage = (target_t - baseline_t).detach()
                actor_loss = -log_prob * advantage
            else:
                raise NotImplementedError(
                    "imag_gradient must be 'dynamics' or 'reinforce'."
                )
            actor_loss -= self._config.actor["entropy"] * entropy
            if bc_kl is not None:
                actor_loss += bc_kl_scale * bc_kl
            actor_loss = torch.mean(actor_loss)

        metrics = {
            "actor_entropy": to_np(torch.mean(entropy)),
            "imag_gradient_mode": 1.0 if mode == "dynamics" else 0.0,
        }
        if bc_kl is not None:
            metrics["actor_bc_kl"] = to_np(torch.mean(bc_kl))
            metrics["actor_bc_kl_scale"] = bc_kl_scale
        if normed_target is not None:
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])
        metrics.update(self._actor_opt(actor_loss, self.actionMLP.parameters()))
        return metrics
