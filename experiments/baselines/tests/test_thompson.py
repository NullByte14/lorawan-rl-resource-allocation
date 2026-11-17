"""Posterior, information, and local-copy checks for the adapted reference."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import run_online as baseline


class ThompsonTests(unittest.TestCase):
    def test_prior_is_beta_one_one_and_only_resolved_labels_update(self):
        policy=baseline.ThompsonPolicy(n_actions=3)
        np.testing.assert_array_equal(policy.successes,np.zeros(3))
        np.testing.assert_array_equal(policy.failures,np.zeros(3))
        policy.update((55,),1,-100.,(),success=1,energy_norm=999.)
        policy.update((),1,100.,(),success=0,energy_norm=0.)
        np.testing.assert_array_equal(policy.successes,[0,1,0])
        np.testing.assert_array_equal(policy.failures,[0,1,0])
        self.assertEqual(policy.update_count,2)

    def test_unknown_labels_are_rejected_not_counted_as_failures(self):
        policy=baseline.ThompsonPolicy()
        for success in (None,-1,2):
            with self.assertRaises(ValueError):
                policy.update((),0,0.,(),success=success)
        self.assertEqual(policy.update_count,0)
        self.assertEqual(policy.failures.sum(),0.)

    def test_posterior_sampling_stays_in_common_action_set(self):
        policy=baseline.ThompsonPolicy()
        rng=np.random.default_rng(23)
        actions=[policy.choose((55,4),rng) for _ in range(100)]
        self.assertTrue(all(0<=action<30 for action in actions))
        self.assertGreater(len(set(actions)),1)

    def test_each_deployed_copy_updates_independently(self):
        base=baseline.ThompsonPolicy()
        devices=[copy.deepcopy(base),copy.deepcopy(base)]
        before=baseline.policy_hash(devices[1])
        devices[0].update((),4,0.,(),success=1)
        self.assertNotEqual(baseline.policy_hash(devices[0]),before)
        self.assertEqual(baseline.policy_hash(devices[1]),before)
        self.assertEqual(baseline.policy_hash(base),before)

    def test_freezing_prevents_posterior_changes(self):
        policy=baseline.ThompsonPolicy(update_enabled=False)
        before=baseline.policy_hash(policy)
        policy.update((),0,0.,(),success=1)
        self.assertEqual(before,baseline.policy_hash(policy))
        self.assertEqual(policy.update_count,0)

    def test_no_feedback_provides_no_online_learning(self):
        policy=baseline.ThompsonPolicy()
        before=baseline.policy_hash(policy)
        result=baseline.run_case(1001,8,3200,policy,"reward","no_feedback","D")
        self.assertGreater(sum(p["attempted"] for p in result["phases"]),0)
        self.assertEqual(sum(p["reports"] for p in result["phases"]),0)
        self.assertEqual(sum(p["resolved"] for p in result["phases"]),0)
        self.assertEqual(policy.update_count,0)
        self.assertEqual(before,baseline.policy_hash(policy))


if __name__=="__main__":
    unittest.main()
