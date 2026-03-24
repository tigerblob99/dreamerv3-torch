import unittest

from AC_RL.RlDataset import _episode_start_weights


class RlDatasetTerminalWeightsTest(unittest.TestCase):
    def test_progressive_tail_weights_inverse_coverage(self):
        weights = _episode_start_weights(ep_len=10, batch_length=4)

        self.assertEqual(len(weights), 7)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=7)

        # Tail starts for N=7, H=4 are indices [3, 4, 5, 6].
        self.assertLess(float(weights[3]), float(weights[4]))
        self.assertLess(float(weights[4]), float(weights[5]))
        self.assertLess(float(weights[5]), float(weights[6]))

        # Normalization cancels out in ratios.
        self.assertAlmostEqual(float(weights[6] / weights[5]), 2.0, places=7)
        self.assertAlmostEqual(float(weights[6] / weights[4]), 3.0, places=7)
        self.assertAlmostEqual(float(weights[6] / weights[3]), 4.0, places=7)

    def test_short_episode_edge_case(self):
        weights = _episode_start_weights(ep_len=5, batch_length=4)

        self.assertEqual(len(weights), 2)
        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
