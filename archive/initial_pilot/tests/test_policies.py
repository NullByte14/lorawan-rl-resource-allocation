"""Focused correctness checks for shared pilot learning policies."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policies import MixMAB, QPolicy


class PolicyTests(unittest.TestCase):
    def test_q_update_matches_worked_equation(self):
        policy = QPolicy(n_actions=3)
        policy.q[(0,)] = np.array([0.4, 0, 0], dtype=np.float32)
        policy.q[(1,)] = np.array([0.8, 0, 0], dtype=np.float32)
        policy.update((0,), 0, 1, (1,))
        self.assertAlmostEqual(float(policy.q[(0,)][0]), 0.532, places=6)

    def test_frozen_q_is_immutable_even_for_unseen_states(self):
        policy = QPolicy(n_actions=3)
        policy.update((0,), 0, 1, (1,))
        policy.update_enabled = False
        original = copy.deepcopy(policy.q)
        policy.choose((3,), np.random.default_rng(7))
        policy.update((0,), 0, -10, (3,))
        self.assertEqual(policy.q.keys(), original.keys())
        for key in original:
            np.testing.assert_array_equal(policy.q[key], original[key])

    def test_q_zero_rows_randomize_ties(self):
        policy = QPolicy(n_actions=3, epsilon=0)
        rng = np.random.default_rng(8)
        self.assertEqual({policy.choose((0,), rng) for _ in range(50)}, {0, 1, 2})

    def test_mix_preprocess_exactly_five_rounds(self):
        policy = MixMAB(n_actions=3)
        rng = np.random.default_rng(7)
        sequence = [policy.choose((), rng) for _ in range(15)]
        self.assertEqual(sequence, [0, 1, 2] * 5)
        np.testing.assert_array_equal(policy.counts, [5, 5, 5])
        self.assertFalse(policy.in_round_robin)
        self.assertEqual(policy.min_selection_count, 5)
        for action in sequence:
            policy.update((), action, -1000, (), success=1, energy_norm=999)
        self.assertEqual(policy.update_count, 15)
        self.assertAlmostEqual(policy.probabilities().sum(), 1)

    def test_mix_failure_does_not_change_weights(self):
        policy = MixMAB(n_actions=3)
        action = policy.choose((), np.random.default_rng(2))
        before = policy.log_weights.copy()
        policy.update((), action, 999, (), success=0)
        np.testing.assert_array_equal(policy.log_weights, before)

    def test_mix_fractional_block_reward_preserved(self):
        policy = MixMAB(n_actions=3)
        action = policy.choose((), np.random.default_rng(2))
        probability = policy.pending[action][0][0]
        increment = policy.gamma * 0.25 / (policy.n_actions * probability)
        policy.update((), action, 999, (), success=0.25)
        expected = np.full(3, -increment)
        expected[action] = 0.0
        np.testing.assert_allclose(policy.log_weights, expected)
        self.assertEqual(policy.update_count, 1)

    def test_mix_rejects_out_of_range_block_reward(self):
        for invalid in [-0.1, 1.1, np.nan, None]:
            policy = MixMAB(n_actions=3)
            action = policy.choose((), np.random.default_rng(2))
            with self.assertRaises(ValueError):
                policy.update((), action, 0, (), success=invalid)

    def test_mix_unknown_feedback_never_becomes_failure(self):
        policy = MixMAB(n_actions=3)
        action = policy.choose((), np.random.default_rng(2))
        policy.discard_pending(action)
        self.assertEqual(policy.update_count, 0)
        self.assertFalse(policy.pending[action])

    def test_mix_elimination_and_restart_reopen_actions(self):
        policy = MixMAB(n_actions=2, l_exp=1, l_ee=3)
        rng = np.random.default_rng(7)
        for _ in range(2):
            action = policy.choose((), rng)
            policy.update((), action, 0, (), success=1)
        policy.log_weights[:] = [0, -15]
        policy.counts[1] = 2
        policy.pending[1].append((0.1, policy.epoch))
        policy.update((), 1, 0, (), success=0)
        self.assertFalse(policy.active[1])
        policy.counts[0] = 3
        policy.choose((), rng)
        self.assertEqual(policy.restart_count, 1)
        self.assertTrue(policy.active.all())
        self.assertEqual(policy.alpha, 2)
        np.testing.assert_array_equal(policy.counts, [0, 0])

    def test_mix_stable_long_delayed_run(self):
        policy = MixMAB(n_actions=30, horizon=3000)
        rng = np.random.default_rng(19)
        for _ in range(60):
            batch = [policy.choose((), rng) for _ in range(10)]
            for action in batch:
                policy.update((), action, 0, (), success=int(rng.random() < 0.8))
            probabilities = policy.probabilities()
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertTrue((probabilities >= 0).all())
            self.assertAlmostEqual(probabilities.sum(), 1)


if __name__ == "__main__":
    unittest.main()
