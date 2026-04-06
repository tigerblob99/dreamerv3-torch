import unittest

import torch

import tools


class FaithfulNormalLossTest(unittest.TestCase):
    def test_faithful_loss_matches_paper_objective(self):
        torch.manual_seed(0)
        mean = torch.randn(4, 3, 2, requires_grad=True)
        std = torch.full_like(mean, 0.7, requires_grad=True)
        target = torch.randn(4, 3, 2)

        dist = tools.FaithfulContDist(mean, std)
        loss = dist.faithful_loss(target)

        std_dist = torch.distributions.Independent(
            torch.distributions.Normal(mean.detach(), std), 1
        )
        expected = torch.square(target - mean).sum(dim=-1) - std_dist.log_prob(target)
        torch.testing.assert_close(loss, expected)

    def test_regression_loss_uses_exact_mean_gradient_for_faithful_dist(self):
        torch.manual_seed(1)
        mean = torch.randn(2, 3, requires_grad=True)
        std = torch.full_like(mean, 0.9, requires_grad=True)
        target = torch.randn(2, 3)

        dist = tools.FaithfulContDist(mean, std)
        loss = tools.regression_loss(dist, target).sum()
        loss.backward()

        expected_mean_grad = 2.0 * (mean.detach() - target)
        torch.testing.assert_close(mean.grad, expected_mean_grad)

    def test_regression_loss_defaults_to_negative_log_prob(self):
        torch.manual_seed(2)
        mean = torch.randn(5, 2)
        std = torch.full_like(mean, 0.5)
        target = torch.randn(5, 2)
        dist = tools.ContDist(
            torch.distributions.Independent(
                torch.distributions.Normal(mean, std), 1
            )
        )

        loss = tools.regression_loss(dist, target)
        expected = -dist.log_prob(target)
        torch.testing.assert_close(loss, expected)


if __name__ == "__main__":
    unittest.main()
