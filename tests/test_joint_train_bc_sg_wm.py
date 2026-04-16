import unittest

import torch
import torch.nn.functional as F


def _mods():
    torch.manual_seed(0)
    return {
        "wm": torch.nn.Linear(4, 4, bias=False),
        "pi": torch.nn.Linear(4, 2, bias=False),
        "head": torch.nn.Linear(4, 1, bias=False),
    }


def _grads(stop_bc: bool, wm_only: bool = False):
    mods = _mods()
    wm = mods["wm"]
    pi = mods["pi"]
    head = mods["head"]

    x = torch.randn(5, 4)
    y_bc = torch.randn(5, 2)
    y_wm = torch.randn(5, 1)

    feat = wm(x)
    feat_bc = feat
    if stop_bc:
        feat_bc = feat_bc.detach()

    wm_loss = F.mse_loss(head(feat), y_wm)
    if wm_only:
        loss = wm_loss
    else:
        bc_loss = F.mse_loss(pi(feat_bc), y_bc)
        loss = wm_loss + bc_loss

    loss.backward()
    out = {}
    for name, mod in mods.items():
        out[name] = None if mod.weight.grad is None else mod.weight.grad.detach().clone()
    return out


class JointTrainBCStopGradTest(unittest.TestCase):
    def test_bc_reaches_wm_when_disabled(self):
        g_all = _grads(stop_bc=False)
        g_wm = _grads(stop_bc=False, wm_only=True)
        self.assertIsNotNone(g_all["wm"])
        self.assertIsNotNone(g_wm["wm"])
        self.assertGreater(float((g_all["wm"] - g_wm["wm"]).abs().sum().item()), 0.0)
        self.assertGreater(float(g_all["pi"].abs().sum().item()), 0.0)

    def test_bc_does_not_reach_wm_when_enabled(self):
        g_all = _grads(stop_bc=True)
        g_wm = _grads(stop_bc=True, wm_only=True)
        self.assertIsNotNone(g_all["wm"])
        self.assertIsNotNone(g_wm["wm"])
        torch.testing.assert_close(g_all["wm"], g_wm["wm"])
        self.assertGreater(float(g_all["pi"].abs().sum().item()), 0.0)

    def test_wm_head_grads_are_unchanged_when_enabled(self):
        g_all = _grads(stop_bc=True)
        g_wm = _grads(stop_bc=True, wm_only=True)
        self.assertIsNotNone(g_all["head"])
        self.assertIsNotNone(g_wm["head"])
        torch.testing.assert_close(g_all["head"], g_wm["head"])


if __name__ == "__main__":
    unittest.main()
