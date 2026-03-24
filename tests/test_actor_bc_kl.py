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
        imag_gradient="reinforce",
        actor={
            "layers": 2,
            "dist": "faithful_normal",
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
            "bc_kl_scale": 0.01,
        },
    )


class ActorBcKlTest(unittest.TestCase):
    def _run_bc_kl_update(self, dist):
        torch.manual_seed(0)
        config = _actor_config()
        config.actor["dist"] = dist
        actor = Actor(config, feat_dim=6, act_dim=4)
        bc_actor = Actor(config, feat_dim=6, act_dim=4)

        batch = 3
        horizon = 5
        features = torch.randn(batch, horizon, 6)
        actions = torch.randn(batch, horizon, 4)
        target = torch.randn(horizon, batch, 1)
        value_seq = torch.randn(horizon, batch, 1)

        metrics = actor.update(
            features, actions, target, value_seq, bc_actor=bc_actor
        )

        self.assertIn("actor_bc_kl", metrics)
        self.assertGreaterEqual(float(metrics["actor_bc_kl"]), 0.0)

    def test_normal_bc_kl_update_runs(self):
        self._run_bc_kl_update("normal")

    def test_faithful_normal_bc_kl_update_runs(self):
        self._run_bc_kl_update("faithful_normal")

    def test_update_requires_time_major_targets(self):
        torch.manual_seed(0)
        config = _actor_config()
        actor = Actor(config, feat_dim=6, act_dim=4)

        batch = 3
        horizon = 5
        features = torch.randn(batch, horizon, 6)
        actions = torch.randn(batch, horizon, 4)
        target = torch.randn(batch, horizon, 1)
        value_seq = torch.randn(horizon, batch, 1)

        with self.assertRaisesRegex(ValueError, r"target must have shape \[time, 3, 1\]"):
            actor.update(features, actions, target, value_seq)


if __name__ == "__main__":
    unittest.main()
