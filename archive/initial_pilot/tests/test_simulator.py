"""Targeted simulator invariants for the mechanism pilot."""

from contextlib import contextmanager
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pilot
from policies import QPolicy


class RecordingPolicy:
    def __init__(self, action=4):
        self.action = action
        self.update_count = 0
        self.policy_size_bytes = 0
        self.observations = []
        self.updates = []

    def choose(self, state, rng):
        self.observations.append(state)
        return self.action

    def update(self, state, action, reward, next_state, **kwargs):
        self.updates.append((state, action, reward, next_state, kwargs))
        self.update_count += 1


@contextmanager
def custom_trace(losses, events):
    previous = pilot.make_trace
    pilot.make_trace = lambda seed, n, seconds: (
        np.zeros((len(losses), 2)), np.asarray(losses, dtype=float), events
    )
    try:
        yield
    finally:
        pilot.make_trace = previous


class SimulatorTests(unittest.TestCase):
    def test_semtech_airtime_reference(self):
        self.assertAlmostEqual(pilot.airtime(7, 20), .056576, places=9)
        self.assertAlmostEqual(pilot.airtime(12, 20), 1.318912, places=9)

    def test_printed_reward_cases(self):
        self.assertAlmostEqual(pilot.original_reward(9, -11.5), 1/9 + 1/81)
        self.assertEqual(pilot.original_reward(9, -12.5), 0)
        self.assertEqual(pilot.original_reward(9, -13.5), -9)
        self.assertEqual(pilot.original_reward(9, None), 0)

    def test_same_sf_capture(self):
        events = [(0., 0, 0, 0., 1., 0.), (0., 1, 0, 0., 1., 0.)]
        with custom_trace([114, 121], events):
            result = pilot.run(11, 2, 3200, RecordingPolicy(), "original_online")
        self.assertEqual(result["phases"][0]["received"], 1)
        self.assertEqual(result["phases"][0]["collisions"], 1)
        with custom_trace([114, 117], events):
            result = pilot.run(11, 2, 3200, RecordingPolicy(), "original_online")
        self.assertEqual(result["phases"][0]["received"], 0)
        self.assertEqual(result["phases"][0]["collisions"], 2)

    def test_orthogonal_channels(self):
        events = [(0., 0, 0, 0., 1., 0.), (0., 1, 1, 0., 1., 0.)]
        with custom_trace([114, 114], events):
            result = pilot.run(11, 2, 3200, RecordingPolicy(), "original_online")
        self.assertEqual(result["phases"][0]["received"], 2)

    def test_gateway_downlink_blocks_uplink(self):
        events = [(float(t), 0, 0, 0., 1., 0.) for t in range(0, 1600, 200)]
        events.append((1403., 1, 1, 0., 1., 0.))
        events.sort()
        with custom_trace([114, 114], events):
            result = pilot.run(11, 2, 3200, RecordingPolicy(), "original_online")
        phase = result["phases"][0]
        self.assertEqual(phase["attempted"], 9)
        self.assertEqual(phase["received"], 8)
        self.assertEqual(phase["half_duplex"], 1)

    def test_device_duty_cycle_rejection(self):
        events = [(0., 0, 0, 0., 1., 0.), (1., 0, 0, 0., 1., 0.)]
        with custom_trace([114], events):
            result = pilot.run(11, 1, 3200, RecordingPolicy(), "original_online")
        phase = result["phases"][0]
        self.assertEqual(phase["generated"], 2)
        self.assertEqual(phase["attempted"], 1)
        self.assertEqual(phase["received"], 1)
        self.assertEqual(phase["duty_dropped"], 1)

    def test_unknown_feedback_cannot_update(self):
        # All uplinks succeed; the eighth requests feedback, which is erased.
        events = [(float(t), 0, 0, 0., 0., 0.) for t in range(0, 1600, 200)]
        policy = RecordingPolicy()
        with custom_trace([114], events):
            result = pilot.run(11, 1, 3200, policy, "both_online")
        self.assertEqual(result["phases"][0]["received"], 8)
        self.assertEqual(result["phases"][0]["reports"], 0)
        self.assertEqual(policy.update_count, 0)
        self.assertGreater(result["phases"][0]["dl_airtime_s"], 0)
        self.assertTrue(all(state[0] == 55 for state in policy.observations))

    def test_successful_report_updates_only_after_feedback(self):
        events = [(float(t), 0, 0, 0., 1., 0.) for t in range(0, 3200, 200)]
        policy = RecordingPolicy()
        with custom_trace([114], events):
            result = pilot.run(11, 1, 3200, policy, "both_online")
        self.assertGreater(result["phases"][0]["reports"], 0)
        self.assertGreater(policy.update_count, 0)
        self.assertTrue(all(state[0] == 55 for state in policy.observations[:8]))
        self.assertTrue(any(state[0] != 55 for state in policy.observations[8:]))

    def test_frozen_table_and_trace_pairing(self):
        records = []
        for variant in ("original_frozen", "reward_frozen", "state_frozen", "both_frozen"):
            learner = QPolicy(epsilon=0., update_enabled=False)
            records.append(pilot.run(11, 3, 3200, learner, variant))
            self.assertEqual(learner.update_count, 0)
            self.assertEqual(learner.q, {})
        self.assertEqual(len({record["trace_sha256"] for record in records}), 1)
        for record in records:
            for phase in record["phases"]:
                self.assertLessEqual(phase["received"], phase["attempted"])
                self.assertLessEqual(phase["attempted"], phase["generated"])
                self.assertGreaterEqual(phase["total_j"], phase["tx_j"])


if __name__ == "__main__":
    unittest.main()
