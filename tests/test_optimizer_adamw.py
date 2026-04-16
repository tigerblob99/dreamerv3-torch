import unittest

import torch

import tools


class OptimizerAdamWTest(unittest.TestCase):
    def test_explicit_adamw_uses_decoupled_weight_decay(self):
        module = torch.nn.Linear(4, 3)
        optimizer = tools.Optimizer(
            "test",
            module.parameters(),
            lr=1e-3,
            eps=1e-8,
            wd=0.05,
            opt="adamw",
        )

        self.assertIsInstance(optimizer._opt, torch.optim.AdamW)
        self.assertAlmostEqual(optimizer._opt.defaults["weight_decay"], 0.05)

    def test_adam_alias_remains_backward_compatible(self):
        module = torch.nn.Linear(4, 3)
        optimizer = tools.Optimizer(
            "test",
            module.parameters(),
            lr=1e-3,
            eps=1e-8,
            wd=0.05,
            opt="adam",
        )

        self.assertIsInstance(optimizer._opt, torch.optim.AdamW)


if __name__ == "__main__":
    unittest.main()
