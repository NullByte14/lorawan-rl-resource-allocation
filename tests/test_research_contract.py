"""Independent checks of the archived evidence and bounded physical cases.

These checks deliberately do not require the proposed policy to win. They
check complete comparisons, information availability, accounting and source
provenance. The historical dataset is a versioned research artifact.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEEDBACK = ROOT / "experiments" / "feedback"
sys.path.insert(0, str(FEEDBACK))
import engine
from policies import QPolicy
from run_experiment import q_sha256


class ConstantPolicy:
    """A fixed radio configuration: no hidden observations or learning."""
    update_enabled = False
    update_count = 0
    policy_size_bytes = 0

    def choose(self, state, rng):
        return 29  # SF12 and 14dBm.

    def update(self, *args, **kwargs):
        raise AssertionError("A fixed policy must not receive learning updates")


@contextmanager
def isolated_trace(loss_db, events):
    saved = engine.make_trace
    engine.make_trace = lambda seed, n, seconds: (
        np.zeros((1, 2), dtype=float), np.asarray([loss_db], dtype=float), sorted(events))
    try:
        yield
    finally:
        engine.make_trace = saved


def isolated_events(count=24, spacing=400.):
    return [(float(i * spacing), 0, 0, 0., .5, 0.) for i in range(count)]


def total(result, metric):
    return sum(phase[metric] for phase in result["phases"])


class ArchivedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads((FEEDBACK / "results" / "raw.json").read_text(encoding="utf-8"))

    def test_repository_uses_research_artifact_names(self):
        ignored = {".git", ".venv", "__pycache__", ".tmp"}
        offending = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
                     if p.is_file() and not ignored.intersection(p.relative_to(ROOT).parts)
                     and "mentor_summary" in p.name.casefold()]
        self.assertEqual(offending, [])

    def test_complete_versioned_evaluation_grid(self):
        config = self.raw["config"]
        self.assertEqual(config["seeds"], list(range(1001, 1031)))
        self.assertEqual(config["node_counts"], [30, 60, 120])
        self.assertEqual(config["train_episodes"], 3)
        self.assertEqual(config["phase_seconds"], 8000)
        expected = {(mode, n, seed, learner, arm)
                    for mode in ("frozen", "online")
                    for n in ([30, 60, 120] if mode == "frozen" else [60])
                    for seed in range(1001, 1031)
                    for learner in (("original", "reward", "both") if mode == "frozen" else ("reward", "both"))
                    for arm in "ABCD"}
        actual = [(r["mode"], r["nodes"], r["seed"], r["learner"], r["condition"])
                  for r in self.raw["records"]]
        self.assertEqual(len(actual), 1320)
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(set(actual), expected)

    def test_training_has_no_missing_episodes_or_evaluation_seed_overlap(self):
        training = self.raw["training"]
        self.assertEqual(len(training), 1530)
        expected = {(mode, n, seed, learner, condition, ep)
                    for mode in ("frozen", "online")
                    for n in ([30, 60, 120] if mode == "frozen" else [60])
                    for seed in range(1001, 1031)
                    for learner in (("original", "reward", "both") if mode == "frozen" else ("reward", "both"))
                    for condition in (("common",) if mode == "frozen" else tuple("ABCD"))
                    for ep in range(3)}
        actual = [(r["mode"], r["nodes"], r["seed"], r["learner"], r["condition"], r["episode"])
                  for r in training]
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), len(set(actual)))
        self.assertFalse({r["environment_seed"] for r in training}.intersection(range(1001, 1031)))
        for r in training:
            self.assertEqual(r["environment_seed"], r["seed"] + 50000 + 1000 * r["episode"])

    def test_archive_keeps_its_exact_simulation_sources(self):
        for name, expected in self.raw["source_sha256"].items():
            with self.subTest(source=name):
                self.assertEqual(hashlib.sha256((FEEDBACK / name).read_bytes()).hexdigest(), expected)

    def test_all_arms_share_exogenous_conditions_and_frozen_tables(self):
        traces, tables = {}, {}
        for r in self.raw["records"]:
            environmental_key = (r["nodes"], r["seed"])
            value = (r["trace_sha256"], r["topology_sha256"])
            self.assertEqual(traces.setdefault(environmental_key, value), value)
            if r["mode"] == "frozen":
                key = (r["nodes"], r["seed"], r["learner"])
                self.assertEqual(tables.setdefault(key, r["initial_q_sha256"]), r["initial_q_sha256"])
                self.assertEqual(r["initial_q_sha256"], r["final_q_sha256"])
                self.assertEqual(r["evaluation_updates"], 0)
        self.assertEqual(len(traces), 90)
        self.assertEqual(len(tables), 270)

    def test_all_attempts_are_resolved_expired_or_pending(self):
        for r in self.raw["records"]:
            with self.subTest(mode=r["mode"], n=r["nodes"], seed=r["seed"], learner=r["learner"], arm=r["condition"]):
                for p in r["phases"]:
                    self.assertLessEqual(0, p["received"])
                    self.assertLessEqual(p["received"], p["attempted"])
                    self.assertLessEqual(p["attempted"], p["generated"])
                    self.assertEqual(p["resolved"] + p["unresolved"] + p["tail_unknown"], p["attempted"])
                    self.assertAlmostEqual(p["total_j"], p["tx_j"] + p["rx_j"] + p["sleep_j"], places=10)
                    self.assertTrue(all(math.isfinite(p[k]) and p[k] >= 0
                                        for k in ("tx_j", "rx_j", "sleep_j", "total_j", "dl_airtime_s")))
                    self.assertEqual(p["reports"] + p["dl_lost"], p["dl_sent"])
                    if r["condition"] in ("A", "C"):
                        self.assertEqual(p["report_phy_bytes"], 35 * p["dl_sent"])
                    else:
                        self.assertLessEqual(14 * p["dl_sent"], p["report_phy_bytes"])
                        self.assertLessEqual(p["report_phy_bytes"], 16 * p["dl_sent"])
                self.assertEqual(total(r, "generated"), sum(b["generated"] for b in r["bins"]))
                self.assertEqual(total(r, "received"), sum(b["received"] for b in r["bins"]))
                self.assertEqual(total(r, "attempted"), sum(r["action_counts"]))

    def test_saved_initial_tables_match_their_canonical_hashes(self):
        cache = {}
        for r in self.raw["records"]:
            name = r["training_table_file"]
            if name not in cache:
                with np.load(FEEDBACK / "results" / name, allow_pickle=False) as saved:
                    q = QPolicy()
                    q.q = {tuple(map(int, state)): values.copy()
                           for state, values in zip(saved["states"], saved["values"])}
                    self.assertEqual(saved["values"].shape[1], 30)
                    self.assertTrue(np.all(np.isfinite(saved["values"])))
                    self.assertAlmostEqual(float(saved["alpha"]), .1)
                    self.assertAlmostEqual(float(saved["gamma"]), .9)
                    cache[name] = (q_sha256(q), int(saved["training_updates"]))
            digest, training_updates = cache[name]
            self.assertEqual(digest, r["training_table_sha256"])
            if r["mode"] == "frozen":
                self.assertEqual(digest, r["initial_q_sha256"])
                self.assertEqual(r["updates"], training_updates)
            else:
                aggregate = hashlib.sha256(json.dumps([digest] * r["nodes"], separators=(",", ":")).encode()).hexdigest()
                self.assertEqual(aggregate, r["initial_q_sha256"])
                self.assertEqual(r["updates"] - training_updates * r["nodes"], r["evaluation_updates"])
        self.assertEqual(len(cache), 510)


class BoundedPhysicalTests(unittest.TestCase):
    def test_isolated_four_arm_energy_order_without_policy_feedback_effects(self):
        records = {}
        for arm in "ABCD":
            with isolated_trace(139., isolated_events()):
                records[arm] = engine.run(17, 1, 4000, ConstantPolicy(), "reward", feedback=engine.CONFIGS[arm])
        self.assertEqual({total(r, "received") for r in records.values()}, {24})
        self.assertEqual({total(r, "dl_sent") for r in records.values()}, {3})
        self.assertAlmostEqual(total(records["A"], "tx_j"), total(records["B"], "tx_j"), places=10)
        self.assertLess(total(records["B"], "rx_j"), total(records["A"], "rx_j"))
        self.assertLess(total(records["C"], "tx_j"), total(records["A"], "tx_j"))
        self.assertAlmostEqual(total(records["C"], "rx_j"), total(records["A"], "rx_j"), places=10)
        self.assertLess(total(records["D"], "total_j"), total(records["A"], "total_j"))

    def test_erased_reports_still_consume_airtime_and_energy(self):
        config = replace(engine.CONFIGS["D"], downlink_loss=1.)
        with isolated_trace(139., isolated_events()):
            result = engine.run(17, 1, 4000, ConstantPolicy(), "reward", feedback=config)
        self.assertEqual(total(result, "received"), 24)
        self.assertEqual(total(result, "reports"), 0)
        self.assertEqual(total(result, "dl_lost"), 3)
        self.assertEqual(total(result, "resolved"), 0)
        self.assertGreater(total(result, "dl_airtime_s"), 0)
        self.assertGreater(total(result, "feedback_j"), 0)

    def test_isolated_path_loss_cannot_improve_reception(self):
        received = []
        for loss in (125., 145., 160.):
            with isolated_trace(loss, isolated_events(16)):
                result = engine.run(17, 1, 4000, ConstantPolicy(), "reward")
            received.append(total(result, "received"))
            self.assertTrue(math.isfinite(total(result, "total_j")))
        self.assertEqual(received, [16, 16, 0])

    def test_horizon_drain_counts_and_charges_only_admitted_attempts(self):
        # The first packet starts before the horizon and finishes after it.
        # Generation at exactly the horizon is excluded.
        events = [(9599.9, 0, 0, 0., .5, 0.), (9600., 0, 0, 0., .5, 0.)]
        with isolated_trace(139., events):
            result = engine.run(17, 1, 3200, ConstantPolicy(), "reward")
        self.assertEqual(total(result, "generated"), 1)
        self.assertEqual(total(result, "attempted"), 1)
        self.assertEqual(total(result, "received"), 1)
        self.assertEqual(result["post_horizon_finishes"], 1)
        self.assertEqual(result["pending_labels"], 1)
        expected_tx = 3.3 * .035 * engine.airtime(12, 33, crc_on=True)
        self.assertAlmostEqual(total(result, "tx_j"), expected_tx, places=12)


if __name__ == "__main__":
    unittest.main()
