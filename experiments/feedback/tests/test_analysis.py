"""Statistical-unit and audit checks, independent of the radio implementation."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze


def example_raw():
    records = []
    for seed in (1001, 1002, 1003):
        for arm in "ABCD":
            cost = {"A": 10., "B": 8., "C": 9., "D": 7.}[arm]
            phases = [dict(generated=10, attempted=8, received=5, total_j=cost,
                           tx_j=cost*.6, rx_j=cost*.3, sleep_j=cost*.1,
                           dl_airtime_s=cost/2, feedback_age_count=2,
                           feedback_age_sum_s=20., reports=2, requested=3,
                           dl_sent=2, report_phy_bytes=32, resolved=8,
                           unresolved=0) for _ in range(3)]
            records.append(dict(mode="frozen", nodes=60, seed=seed,
                                learner="reward", condition=arm, phases=phases,
                                trace_sha256=str(seed), topology_sha256=str(seed),
                                initial_q_sha256="same", final_q_sha256="same",
                                evaluation_updates=0))
    return dict(config={"seeds": [1001, 1002, 1003]},
                training=[{"environment_seed": 51001}], records=records)


class AnalysisTests(unittest.TestCase):
    def test_constant_bootstrap_and_invalid_input(self):
        self.assertEqual(analyze.bootstrap_mean([3., 3., 3.], "constant", 200), (3., 3., 3.))
        self.assertEqual(analyze.bootstrap_mean([1., 2., 4.], "repeat", 200),
                         analyze.bootstrap_mean([1., 2., 4.], "repeat", 200))
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                analyze.bootstrap_mean(values, resamples=200)

    def test_totals_use_ratio_of_sums_not_phase_ratio_average(self):
        raw = example_raw()
        phases = raw["records"][0]["phases"]
        phases[0].update(generated=100, attempted=100, received=10)
        totals = analyze.phase_rows([raw["records"][0]])[-1]
        self.assertEqual(totals["phase"], "total")
        self.assertAlmostEqual(totals["app_delivery"], 20 / 120)
        self.assertAlmostEqual(totals["pdr"], 20 / 116)
        self.assertAlmostEqual(totals["j_per_delivery"], 30 / 20)

    def test_pairing_and_frozen_audits(self):
        raw = example_raw()
        self.assertEqual(analyze.verify_records(raw)["evaluations"], 12)
        for key, value in (("trace_sha256", "other"), ("topology_sha256", "other"),
                           ("final_q_sha256", "changed"), ("evaluation_updates", 1),
                           ("initial_q_sha256", "other")):
            broken = copy.deepcopy(raw)
            broken["records"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                analyze.verify_records(broken)
        missing = copy.deepcopy(raw)
        missing["records"] = [r for r in missing["records"] if r["condition"] != "D"]
        with self.assertRaises(ValueError):
            analyze.verify_records(missing)

    def test_whole_run_label_delay_is_weighted_by_resolved_packets(self):
        record = example_raw()["records"][0]
        record["phases"][0].update(feedback_age_count=1, feedback_age_sum_s=100., feedback_age_mean_s=100.)
        record["phases"][1].update(feedback_age_count=4, feedback_age_sum_s=40., feedback_age_mean_s=10.)
        record["phases"][2].update(feedback_age_count=0, feedback_age_sum_s=0., feedback_age_mean_s=None)
        total = analyze.phase_rows([record])[-1]
        self.assertEqual(total["mean_label_delay_s"], 28.)
        self.assertEqual(total["feedback_age_mean_s"], 28.)

    def test_paired_difference_and_interaction(self):
        rows = analyze.phase_rows(example_raw()["records"])
        paired = analyze.paired_rows(rows, 200)
        d_minus_a = next(r for r in paired if r["phase"] == "total" and
                         r["metric"] == "j_per_delivery" and r["contrast"] == "D-A")
        self.assertAlmostEqual(d_minus_a["delta_mean"], -.6)
        self.assertEqual(d_minus_a["n_total"], 3)
        self.assertEqual(d_minus_a["n_valid"], 3)
        interaction = next(r for r in paired if r["phase"] == "total" and
                           r["metric"] == "total_j" and r["contrast"] == "interaction_D-C-B+A")
        self.assertEqual(interaction["delta_mean"], 0.)

    def test_undefined_primary_denominator_not_silently_dropped(self):
        rows = analyze.phase_rows(example_raw()["records"])
        rows[0]["j_per_delivery"] = None
        with self.assertRaises(ValueError):
            analyze.summaries(rows, 200)
        with self.assertRaises(ValueError):
            analyze.paired_rows(rows, 200)

    def test_source_integrity(self):
        writable_tmp = Path(__file__).resolve().parents[1] / ".tmp"
        writable_tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=writable_tmp) as temporary:
            directory = Path(temporary)
            path = directory / "source.py"
            path.write_bytes(b"x=1\n")
            raw = {"source_sha256": {"source.py": hashlib.sha256(path.read_bytes()).hexdigest()}}
            self.assertEqual(analyze.verify_source_hashes(raw, directory), 1)
            path.write_bytes(b"x=2\n")
            with self.assertRaises(ValueError):
                analyze.verify_source_hashes(raw, directory)


if __name__ == "__main__":
    unittest.main()
