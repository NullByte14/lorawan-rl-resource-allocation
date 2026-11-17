"""Deterministic physical and information-flow checks for the feedback experiment.

These tests use fixed traffic and a recording policy, rather than asserting that
one algorithm wins a random simulation. No production data files are modified.
"""

import copy
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine
from policies import QPolicy


class RecordingPolicy:
    """Choose SF9/14 dBm by default and retain exactly what learning observes."""

    def __init__(self, action=14, update_enabled=True):
        self.action = action
        self.update_enabled = update_enabled
        self.update_count = 0
        self.policy_size_bytes = 0
        self.q_entries = 0
        self.observations = []
        self.updates = []

    def choose(self, state, rng):
        self.observations.append(tuple(state))
        return self.action

    def update(self, state, action, reward, next_state, **kwargs):
        if not self.update_enabled:
            return
        self.updates.append((tuple(state), action, reward, tuple(next_state), kwargs))
        self.update_count += 1


@contextmanager
def custom_trace(losses, events):
    original = engine.make_trace
    engine.make_trace = lambda seed, n, seconds: (
        np.zeros((len(losses), 2)), np.asarray(losses, dtype=float), sorted(events)
    )
    try:
        yield
    finally:
        engine.make_trace = original


def periodic_events(count, spacing=200, device=0, dl_uniform=.5):
    return [(float(i * spacing), device, 0, 0., dl_uniform, 0.)
            for i in range(count)]


def total(result, field):
    return sum(phase[field] for phase in result["phases"])


class AirtimeTests(unittest.TestCase):
    def test_ul_dl_crc_reference_values(self):
        self.assertAlmostEqual(engine.airtime(7, 20, crc_on=True), .056576, places=9)
        self.assertAlmostEqual(engine.airtime(7, 20, crc_on=False), .051456, places=9)
        self.assertAlmostEqual(engine.airtime(12, 35, crc_on=False), 1.810432, places=9)
        self.assertAlmostEqual(engine.airtime(12, 16, crc_on=False), 1.155072, places=9)
        self.assertAlmostEqual(engine.airtime(12, 33, crc_on=True), 1.810432, places=9)
        self.assertAlmostEqual(engine.airtime(12, 37, crc_on=True), 1.974272, places=9)

    def test_short_downlink_payloads_share_sf12_symbol_plateau(self):
        durations = [engine.airtime(12, size, crc_on=False) for size in range(14, 18)]
        self.assertEqual(len(set(durations)), 1)

    def test_reference_energy_remains_fixed(self):
        self.assertAlmostEqual(engine.E_REF, .36383221248, places=12)

    def test_original_reward_reference_cases(self):
        self.assertAlmostEqual(engine.original_reward(9, -11.5), 1/9 + 1/81)
        self.assertEqual(engine.original_reward(9, -12.5), 0)
        self.assertEqual(engine.original_reward(9, -13.5), -9)
        self.assertEqual(engine.original_reward(9, None), 0)


class PhysicalTests(unittest.TestCase):
    def test_same_sf_capture_and_separate_channel(self):
        events = [(0., 0, 0, 0., .5, 0.), (0., 1, 0, 0., .5, 0.)]
        with custom_trace([114, 121], events):
            captured = engine.run(11, 2, 3200, RecordingPolicy(4), "reward")
        self.assertEqual(total(captured, "received"), 1)
        with custom_trace([114, 117], events):
            collided = engine.run(11, 2, 3200, RecordingPolicy(4), "reward")
        self.assertEqual(total(collided, "received"), 0)
        separate = [(0., 0, 0, 0., .5, 0.), (0., 1, 1, 0., .5, 0.)]
        with custom_trace([114, 114], separate):
            orthogonal = engine.run(11, 2, 3200, RecordingPolicy(4), "reward")
        self.assertEqual(total(orthogonal, "received"), 2)

    def test_gateway_transmission_blocks_an_uplink(self):
        events = periodic_events(8)
        events.append((1403., 1, 1, 0., .5, 0.))
        with custom_trace([114, 114], events):
            result = engine.run(11, 2, 3200, RecordingPolicy(4), "reward",
                                feedback=engine.CONFIGS["A"])
        self.assertEqual(total(result, "attempted"), 9)
        self.assertEqual(total(result, "received"), 8)
        self.assertEqual(total(result, "half_duplex"), 1)

    def test_device_duty_drop_does_not_consume_sequence_number(self):
        events = periodic_events(8)
        events.append((1., 0, 0, 0., .5, 0.))
        with custom_trace([139], events):
            result = engine.run(11, 1, 3200, RecordingPolicy(), "both",
                                feedback=engine.CONFIGS["D"], debug=True)
        self.assertEqual(total(result, "generated"), 9)
        self.assertEqual(total(result, "attempted"), 8)
        self.assertEqual(total(result, "duty_dropped"), 1)
        reports = [e for e in result["debug_events"] if e["kind"] == "report_sent"]
        self.assertEqual([e["anchor"] for e in reports], [8])

    def test_request_bytes_change_tx_energy_by_reference_increment(self):
        records = {}
        for arm in ("A", "C"):
            with custom_trace([139], periodic_events(8)):
                records[arm] = engine.run(11, 1, 3200, RecordingPolicy(29), "reward",
                                         feedback=engine.CONFIGS[arm])
        expected = 3.3 * .035 * (1.974272 - 1.810432)
        self.assertAlmostEqual(total(records["A"], "tx_j") - total(records["C"], "tx_j"),
                               expected, places=10)
        self.assertAlmostEqual(total(records["A"], "rx_j"), total(records["C"], "rx_j"),
                               places=10)

    def test_shorter_report_releases_gateway_duty_budget_earlier(self):
        events = periodic_events(8, spacing=100)
        events += [(float(14 + i*100), 1, 1, 0., .5, 0.) for i in range(8)]
        records = {}
        for arm in ("A", "B"):
            with custom_trace([114, 114], events):
                records[arm] = engine.run(11, 2, 3200, RecordingPolicy(4), "reward",
                                         feedback=engine.CONFIGS[arm], debug=True)
        self.assertEqual(total(records["A"], "dl_sent"), 1)
        self.assertEqual(total(records["B"], "dl_sent"), 2)
        self.assertAlmostEqual(total(records["A"], "dl_airtime_s"), 1.810432, places=9)
        self.assertAlmostEqual(total(records["B"], "dl_airtime_s"), 2 * 1.155072, places=9)

    def test_lost_report_still_costs_receive_energy(self):
        records = {}
        for loss in (0., 1.):
            configuration = replace(engine.CONFIGS["B"], downlink_loss=loss)
            with custom_trace([139], periodic_events(8)):
                records[loss] = engine.run(11, 1, 3200, RecordingPolicy(), "reward",
                                          feedback=configuration)
        self.assertEqual(total(records[0.], "reports"), 1)
        self.assertEqual(total(records[1.], "reports"), 0)
        self.assertEqual(total(records[1.], "dl_sent"), 1)
        self.assertGreater(total(records[1.], "feedback_j"), 0)
        self.assertAlmostEqual(total(records[0.], "rx_j"), total(records[1.], "rx_j"), places=10)

    def test_report_compression_receive_energy_matches_shorter_duration(self):
        records = {}
        for arm in ("A", "B"):
            with custom_trace([139], periodic_events(8)):
                records[arm] = engine.run(11, 1, 3200, RecordingPolicy(), "reward",
                                         feedback=engine.CONFIGS[arm])
        expected = 3.3 * .0108 * (1.810432 - 1.155072)
        self.assertAlmostEqual(total(records["A"], "tx_j"), total(records["B"], "tx_j"), places=10)
        self.assertAlmostEqual(total(records["A"], "rx_j") - total(records["B"], "rx_j"),
                               expected, places=10)


class FeedbackTests(unittest.TestCase):
    def test_rich_and_compact_serialization_agree_across_snr_boundaries(self):
        for snr in [-25., *[value/2 for value in range(-40, 13)], 10.]:
            with self.subTest(snr=snr):
                history = {seq: snr for seq in range(1, 17) if seq != 5}
                rich = engine.parse_report("rich", engine.build_report("rich", 16, history), 16)
                compact = engine.parse_report("compact", engine.build_report("compact", 16, history), 16)
                self.assertEqual(rich.receipts, compact.receipts)
                self.assertEqual(rich.snr_bin, compact.snr_bin)
                self.assertEqual(rich.snrs[4], None)
                self.assertEqual(rich.snrs[-1], snr)
                self.assertTrue(all(value is None for value in compact.snrs))

    def test_startup_padding_has_no_nonexistent_successes(self):
        history = {seq: -8. for seq in range(1, 9)}
        for report_format in ("rich", "compact"):
            report = engine.parse_report(report_format,
                engine.build_report(report_format, 8, history), 8)
            self.assertEqual(report.receipts[:8], (False,)*8)
            self.assertEqual(report.receipts[8:], (True,)*8)

    def test_fresh_report_contract_rejects_invalid_trigger_and_stale_rich_anchor(self):
        with self.assertRaises(ValueError):
            engine.build_report("compact", 16, {15: -8.})
        payload = engine.build_report("rich", 16, {16: -8.})
        with self.assertRaises(ValueError):
            engine.parse_report("rich", payload, 24)
        # Valid generic bitmap bytes cannot represent a fresh response if its
        # final bit says the triggering uplink itself was not received.
        with self.assertRaises(ValueError):
            engine.parse_report("compact", bytes([0x80, 0, 0]), 16)

    def test_no_valid_report_means_unknown_state_and_no_updates(self):
        policy = RecordingPolicy()
        config = replace(engine.CONFIGS["D"], downlink_loss=1.)
        with custom_trace([139], periodic_events(16)):
            result = engine.run(11, 1, 3200, policy, "both", feedback=config)
        self.assertEqual(total(result, "received"), 16)
        self.assertEqual(total(result, "reports"), 0)
        self.assertEqual(policy.update_count, 0)
        self.assertTrue(all(state[0] == 55 for state in policy.observations))
        self.assertEqual(result["pending_labels"], 16)

    def test_overlapping_reports_resolve_each_transition_once(self):
        policy = RecordingPolicy()
        with custom_trace([139], periodic_events(17)):
            result = engine.run(11, 1, 3200, policy, "both",
                                feedback=engine.CONFIGS["D"], debug=True)
        updates = [e for e in result["debug_events"] if e["kind"] == "update"]
        self.assertEqual([e["seq"] for e in updates], list(range(1, 17)))
        self.assertEqual(policy.update_count, 16)
        self.assertTrue(all(state[0] == 55 for state in policy.observations[:8]))
        self.assertNotEqual(policy.observations[8][0], 55)
        self.assertEqual(result["pending_labels"], 1)

    def test_next_window_recovers_labels_after_one_lost_report(self):
        events = periodic_events(16)
        events[7] = (*events[7][:4], 0., events[7][5])
        policy = RecordingPolicy()
        with custom_trace([139], events):
            result = engine.run(11, 1, 3200, policy, "both",
                                feedback=engine.CONFIGS["D"], debug=True)
        self.assertEqual(total(result, "reports"), 1)
        self.assertEqual(policy.update_count, 16)
        self.assertEqual(result["pending_labels"], 0)
        self.assertEqual(result["expired_labels"], 0)

    def test_history_outside_recovered_window_expires_unknown(self):
        events = periodic_events(24)
        for index in (7, 15):
            events[index] = (*events[index][:4], 0., events[index][5])
        policy = RecordingPolicy()
        with custom_trace([139], events):
            result = engine.run(11, 1, 3200, policy, "both",
                                feedback=engine.CONFIGS["D"], debug=True)
        updates = [e for e in result["debug_events"] if e["kind"] == "update"]
        self.assertEqual([e["seq"] for e in updates], list(range(9, 25)))
        self.assertEqual(result["expired_labels"], 8)
        self.assertTrue(all(e["success"] == 1 for e in updates))

    def test_compact_and_rich_reports_preserve_receipts_and_snr_bin(self):
        events = periodic_events(17)
        # Frame 5 fails at the gateway; all other frames have a non-extreme SNR.
        events[4] = (*events[4][:3], -20., events[4][4], events[4][5])
        records, policies = {}, {}
        for arm in ("A", "B"):
            policies[arm] = RecordingPolicy()
            with custom_trace([139], events):
                records[arm] = engine.run(11, 1, 3200, policies[arm], "both",
                                         feedback=engine.CONFIGS[arm], debug=True)
        self.assertEqual(policies["A"].observations, policies["B"].observations)
        for arm in records:
            updates = [e for e in records[arm]["debug_events"] if e["kind"] == "update"]
            self.assertEqual([e["success"] for e in updates], [1]*4 + [0] + [1]*11)
            first = next(e for e in records[arm]["debug_events"] if e["kind"] == "report_sent")
            self.assertNotEqual(first["snr_bin"], 55)
            self.assertTrue(first["receipts"][-1])
        rich = next(e for e in records["A"]["debug_events"] if e["kind"] == "report_sent")
        compact = next(e for e in records["B"]["debug_events"] if e["kind"] == "report_sent")
        self.assertEqual(rich["snr_bin"], compact["snr_bin"])
        self.assertEqual(rich["receipts"], compact["receipts"])
        self.assertEqual(rich["phy_bytes"], 35)
        self.assertIn(compact["phy_bytes"], (14, 15, 16))

    def test_stored_next_state_is_not_rewritten_by_late_feedback(self):
        policy = RecordingPolicy()
        with custom_trace([139], periodic_events(9)):
            result = engine.run(11, 1, 3200, policy, "both",
                                feedback=engine.CONFIGS["D"], debug=True)
        updates = [e for e in result["debug_events"] if e["kind"] == "update"]
        self.assertEqual(updates[0]["state"][0], 55)
        self.assertEqual(updates[0]["next_state"][0], 55)
        self.assertNotEqual(policy.observations[-1][0], 55)

    def test_delivery_energy_reward_uses_report_label_and_local_cost(self):
        events = periodic_events(8)
        events[4] = (*events[4][:3], -20., events[4][4], events[4][5])
        with custom_trace([139], events):
            result = engine.run(11, 1, 3200, RecordingPolicy(), "both",
                                feedback=engine.CONFIGS["D"], debug=True)
        updates = [e for e in result["debug_events"] if e["kind"] == "update"]
        self.assertEqual(len(updates), 8)
        for update in updates:
            self.assertAlmostEqual(update["reward"],
                update["success"] - .2 * update["energy_j"] / engine.E_REF, places=12)
        self.assertLess(updates[4]["reward"], 0)
        self.assertGreater(updates[-1]["energy_j"], updates[0]["energy_j"])

    def test_original_reward_training_rejects_compact_feedback(self):
        with self.assertRaises(ValueError):
            engine.run(11, 1, 3200, QPolicy(), "original",
                       feedback=engine.CONFIGS["B"], training=True)

    def test_original_frozen_policy_can_consume_compact_state(self):
        policy = QPolicy(update_enabled=False, epsilon=0.)
        result = engine.run(11, 2, 3200, policy, "original",
                            feedback=engine.CONFIGS["D"])
        self.assertEqual(result["updates"], 0)


class PairingAndPolicyTests(unittest.TestCase):
    def test_q_update_equation(self):
        policy = QPolicy(n_actions=3)
        policy.q[(0,)] = np.array([.4, 0., 0.], dtype=np.float32)
        policy.q[(1,)] = np.array([.8, 0., 0.], dtype=np.float32)
        policy.update((0,), 0, 1., (1,))
        self.assertAlmostEqual(float(policy.q[(0,)][0]), .532, places=6)

    def test_frozen_q_rows_and_unknown_states_remain_unchanged(self):
        policy = QPolicy(epsilon=0., update_enabled=False)
        policy.q[(55, 4, 3)] = np.arange(30, dtype=np.float32)
        before = copy.deepcopy(policy.q)
        engine.run(11, 3, 3200, policy, "both", feedback=engine.CONFIGS["D"])
        self.assertEqual(policy.update_count, 0)
        self.assertEqual(policy.q.keys(), before.keys())
        for state in before:
            np.testing.assert_array_equal(policy.q[state], before[state])

    def test_all_four_configs_have_matching_generated_environment(self):
        records = [engine.run(11, 4, 3200, QPolicy(epsilon=0., update_enabled=False),
                              "both", feedback=engine.CONFIGS[arm])
                   for arm in "ABCD"]
        self.assertEqual(len({r["trace_sha256"] for r in records}), 1)
        self.assertEqual(len({r["topology_sha256"] for r in records}), 1)
        for record in records:
            self.assertEqual(total(record, "generated"), sum(b["generated"] for b in record["bins"]))
            for phase in record["phases"]:
                self.assertLessEqual(phase["received"], phase["attempted"])
                self.assertLessEqual(phase["attempted"], phase["generated"])
                self.assertAlmostEqual(phase["total_j"], phase["tx_j"] + phase["rx_j"] + phase["sleep_j"])
            self.assertEqual(total(record, "resolved") + record["expired_labels"] + record["pending_labels"],
                             total(record, "attempted"))


if __name__ == "__main__":
    unittest.main()
