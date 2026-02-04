import torch
from torch import nn

import tools
from joint_train import ActionMLP


to_np = lambda x: x.detach().cpu().numpy()


class Actor(nn.Module):
    def __init__(self, config, feat_dim: int, act_dim: int):
        super().__init__()
        self._config = config
        self._use_amp = True if config.precision == 16 else False
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
        reward_len = target_t.shape[0]
        value_seq = value_seq[:reward_len]
        if weights_t.shape[0] != reward_len:
            weights_t = weights_t[:reward_len]
        if target_t.ndim == 2:
            target_t = target_t.unsqueeze(-1)

        with tools.RequiresGrad(self.actionMLP):
            with torch.amp.autocast(
                device_type="cuda", enabled=self._use_amp, dtype=torch.float16
            ):
                policy = self.actionMLP(feats_t[:reward_len].detach(), return_dist=True)
                log_prob = policy.log_prob(actions_t[:reward_len])
                entropy = policy.entropy()
                advantage = (target_t - value_seq).detach()
                actor_loss = (
                    -weights_t[:reward_len] * log_prob[:, :, None] * advantage
                )
                actor_loss -= (
                    self._config.actor["entropy"] * entropy[:, :, None]
                )
                actor_loss = torch.mean(actor_loss)

        metrics = {"actor_entropy": to_np(torch.mean(entropy))}
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actionMLP.parameters()))
        return metrics
