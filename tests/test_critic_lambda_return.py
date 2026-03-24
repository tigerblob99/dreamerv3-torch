import unittest
from types import SimpleNamespace

import torch

from AC_RL.critic import Critic


def _reference_lambda_return(rewards, values, lambda_, gamma):
    """Reference recursion from the user-provided function."""
    ret = values[-1]
    returns = [ret]
    rewards_less_last = rewards[:-1]
    values_less_first = values[1:]
    for reward_t, value_tplus1 in zip(
        rewards_less_last.flip(0), values_less_first.flip(0)
    ):
        ret = reward_t + gamma * ((1.0 - lambda_) * value_tplus1 + lambda_ * ret)
        returns.insert(0, ret)
    return torch.stack(returns, dim=0)


class CriticLambdaReturnTest(unittest.TestCase):
    def _make_critic(self, gamma=0.99, lambda_=0.95):
        critic = Critic.__new__(Critic)
        critic._config = SimpleNamespace(discount=gamma, discount_lambda=lambda_)
        return critic

    def test_matches_reference_recursion_truncated(self):
        torch.manual_seed(0)
        critic = self._make_critic(gamma=0.99, lambda_=0.95)

        time_len, batch = 7, 4
        rewards = torch.randn(time_len, batch, 1)
        values = torch.randn(time_len, batch, 1)

        target, _ = critic._lambda_return(rewards, values)
        reference = _reference_lambda_return(
            rewards,
            values,
            gamma=critic._config.discount,
            lambda_=critic._config.discount_lambda,
        )
        self.assertTrue(torch.allclose(target, reference[:-1], atol=1e-6, rtol=1e-6))

    def test_returns_expected_value_sequence(self):
        torch.manual_seed(1)
        critic = self._make_critic(gamma=0.997, lambda_=0.9)

        time_len, batch = 6, 3
        rewards = torch.randn(time_len, batch, 1)
        values = torch.randn(time_len, batch, 1)

        target_a, value_seq_a = critic._lambda_return(rewards, values)
        target_b, value_seq_b = critic._lambda_return(rewards, values)

        self.assertTrue(torch.allclose(target_a, target_b, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(value_seq_a, values[:-1]))
        self.assertTrue(torch.allclose(value_seq_b, values[:-1]))

    def test_shape_contract(self):
        critic = self._make_critic(gamma=0.98, lambda_=0.95)

        time_len, batch = 5, 2
        rewards = torch.randn(time_len, batch, 1)
        values = torch.randn(time_len, batch, 1)

        target, value_seq = critic._lambda_return(rewards, values)
        self.assertEqual(target.shape, (time_len - 1, batch, 1))
        self.assertEqual(value_seq.shape, (time_len - 1, batch, 1))

    def test_raises_when_reward_horizon_too_short(self):
        critic = self._make_critic()

        rewards = torch.randn(1, 2, 1)
        values = torch.randn(1, 2, 1)

        with self.assertRaises(ValueError):
            critic._lambda_return(rewards, values)

    def test_raises_when_values_shorter_than_rewards(self):
        critic = self._make_critic()

        rewards = torch.randn(4, 2, 1)
        values = torch.randn(3, 2, 1)

        with self.assertRaises(ValueError):
            critic._lambda_return(rewards, values)


if __name__ == "__main__":
    unittest.main()
