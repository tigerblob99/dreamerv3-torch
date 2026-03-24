import unittest
from types import SimpleNamespace

import torch

from AC_RL.actor import Actor


def _actor_config():
    return SimpleNamespace(
        precision=32,
        reward_EMA=False,
        mlp_layers=2,
        policy_units=64,
        act="SiLU",
        norm=True,
        device="cpu",
        weight_decay=0.0,
        opt="adam",
        imag_gradient="dynamics",
        actor={
            "layers": 2,
            "dist": "normal",
            "std": "learned",
            "min_std": 0.1,
            "max_std": 1.0,
            "temp": 0.1,
            "unimix_ratio": 0.01,
            "outscale": 1.0,
            "lr": 1e-3,
            "eps": 1e-5,
            "grad_clip": 100.0,
            "entropy": 0.0,
        },
    )


def _grad_sum(module: torch.nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.abs().sum().item())
    return total


class ActorGradFlowTest(unittest.TestCase):
    def _build_actor(self, feat_dim=6, act_dim=6):
        return Actor(_actor_config(), feat_dim=feat_dim, act_dim=act_dim)

    @staticmethod
    def _zero_grads(module: torch.nn.Module):
        for param in module.parameters():
            param.grad = None

    @staticmethod
    def _imagined_rollout_loss(actor: Actor, feat_dim: int, horizon: int = 5) -> torch.Tensor:
        batch = 8
        feat = torch.randn(batch, feat_dim)
        losses = []
        for _ in range(horizon):
            action = actor.generate_actions(feat, sample=True)
            # Keep a differentiable dependency from the next latent to action.
            feat = 0.9 * feat + 0.1 * action
            losses.append(-(feat**2).mean())
        return torch.stack(losses).mean()

    def test_gradients_flow_when_action_mlp_is_trainable(self):
        torch.manual_seed(0)
        feat_dim = 6
        actor = self._build_actor(feat_dim=feat_dim, act_dim=feat_dim)
        actor.actionMLP.requires_grad_(True)
        self._zero_grads(actor.actionMLP)

        loss = self._imagined_rollout_loss(actor, feat_dim=feat_dim)
        loss.backward()

        self.assertGreater(_grad_sum(actor.actionMLP), 0.0)

    def test_gradients_are_blocked_when_action_mlp_is_frozen(self):
        torch.manual_seed(0)
        feat_dim = 6
        actor = self._build_actor(feat_dim=feat_dim, act_dim=feat_dim)
        actor.actionMLP.requires_grad_(False)
        self._zero_grads(actor.actionMLP)

        loss = self._imagined_rollout_loss(actor, feat_dim=feat_dim)
        self.assertFalse(loss.requires_grad)
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            loss.backward()
        self.assertEqual(_grad_sum(actor.actionMLP), 0.0)


if __name__ == "__main__":
    unittest.main()
