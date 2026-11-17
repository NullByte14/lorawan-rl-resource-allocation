"""Stress-profile and control-policy contracts, separate from radio unit tests."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_suite as suite


class ProfileTests(unittest.TestCase):
    def test_six_registered_cases_and_profiles(self):
        self.assertEqual(set(suite.CASES),{"stable_low","stable_high","load_step","channel_step","lossy_feedback","no_feedback"})
        self.assertEqual(suite.CASES["stable_low"].intervals,(400.,400.,400.))
        self.assertEqual(suite.CASES["stable_high"].intervals,(160.,160.,160.))
        self.assertEqual(suite.CASES["lossy_feedback"].downlink_loss,.30)
        self.assertEqual(suite.CASES["no_feedback"].downlink_loss,1.)

    def test_load_step_matches_preserved_generator(self):
        positions,loss,events=suite.make_case_trace(1001,4,3200,"load_step")
        expected_positions,expected_loss,expected_events=suite.BASE_MAKE_TRACE(1001,4,3200)
        np.testing.assert_array_equal(positions,expected_positions)
        np.testing.assert_array_equal(loss,expected_loss)
        self.assertEqual(events,expected_events)

    def test_channel_step_is_paired_bidirectional_attenuation(self):
        _,_,base=suite.make_case_trace(1001,3,3200,"load_step")
        _,_,changed=suite.make_case_trace(1001,3,3200,"channel_step")
        self.assertEqual(len(base),len(changed))
        for a,b in zip(base,changed):
            self.assertEqual(a[:3],b[:3])
            self.assertEqual(a[4],b[4])
            shift=6 if a[0]>=3200 else 0
            self.assertAlmostEqual(a[3]-b[3],shift)
            self.assertAlmostEqual(a[5]-b[5],shift)

    def test_stable_case_offered_counts(self):
        for case,expected in (("stable_low",4*8),("stable_high",4*20)):
            _,_,events=suite.make_case_trace(1001,4,3200,case)
            self.assertEqual([sum(int(t//3200)==phase for t,*_ in events) for phase in range(3)],[expected]*3)

    def test_positions_and_shadowing_paired_between_cases(self):
        position,loss,_=suite.make_case_trace(1001,4,3200,"load_step")
        for case in suite.CASES:
            actual_position,actual_loss,_=suite.make_case_trace(1001,4,3200,case)
            np.testing.assert_array_equal(position,actual_position)
            np.testing.assert_array_equal(loss,actual_loss)

    def test_trace_context_restored_after_failure(self):
        previous=suite.engine.make_trace
        with self.assertRaises(RuntimeError):
            with suite.case_trace("stable_low"):
                self.assertIsNot(suite.engine.make_trace,previous)
                raise RuntimeError("test")
        self.assertIs(suite.engine.make_trace,previous)
        with self.assertRaises(ValueError):
            with suite.case_trace("unknown"):
                pass
        self.assertIs(suite.engine.make_trace,previous)


class SimulatorContracts(unittest.TestCase):
    def test_no_feedback_never_produces_labels_or_snr(self):
        policy=suite.QPolicy(epsilon=.25)
        result=suite.run_case(1001,4,3200,policy,"both","no_feedback","D",debug=True)
        self.assertEqual(sum(p["reports"] for p in result["phases"]),0)
        self.assertEqual(sum(p["resolved"] for p in result["phases"]),0)
        self.assertEqual(policy.update_count,0)
        decisions=[e for e in result["debug_events"] if e["kind"]=="decision"]
        self.assertTrue(decisions)
        self.assertTrue(all(e["state"][0]==55 for e in decisions))
        self.assertTrue(all(e["state"][-1]==3 for e in decisions))

    def test_fixed_control_only_uses_declared_action(self):
        result=suite.run_case(1001,4,3200,suite.FixedConservative(),"original","load_step")
        self.assertGreater(result["action_counts"][29],0)
        self.assertEqual(sum(result["action_counts"][:29]),0)
        self.assertEqual(result["updates"],0)

    def test_fixed_no_feedback_has_no_request_or_report_cost(self):
        result=suite.run_case(1001,4,3200,suite.FixedConservative(),"original","load_step","N")
        self.assertGreater(sum(p["attempted"] for p in result["phases"]),0)
        self.assertEqual(sum(p["request_frames"] for p in result["phases"]),0)
        self.assertEqual(sum(p["request_airtime_s"] for p in result["phases"]),0.)
        self.assertEqual(sum(p["dl_sent"] for p in result["phases"]),0)
        self.assertEqual(sum(p["reports"] for p in result["phases"]),0)

    def test_feedback_conditions_share_each_case_environment(self):
        for case in suite.CASES:
            signatures=set()
            for condition in ("A","D"):
                result=suite.run_case(1001,3,3200,suite.FixedConservative(),"original",case,condition)
                signatures.add((result["trace_sha256"],result["topology_sha256"]))
            self.assertEqual(len(signatures),1)

    def test_total_ratios_and_label_delay_are_pooled(self):
        stats=[dict(generated=100,attempted=100,received=10,total_j=10,dl_airtime_s=1,feedback_age_count=1,feedback_age_sum_s=100,feedback_age_mean_s=100),
               dict(generated=10,attempted=8,received=5,total_j=5,dl_airtime_s=2,feedback_age_count=4,feedback_age_sum_s=40,feedback_age_mean_s=10),
               dict(generated=10,attempted=8,received=5,total_j=5,dl_airtime_s=2,feedback_age_count=0,feedback_age_sum_s=0,feedback_age_mean_s=None)]
        record=dict(scenario="load_step",nodes=3,seed=1,learner="both",condition="A",mode="frozen",phases=stats)
        total=suite.make_rows([record])[-1]
        self.assertAlmostEqual(total["app_delivery"],20/120)
        self.assertAlmostEqual(total["j_per_delivery"],1.)
        self.assertAlmostEqual(total["mean_label_delay_s"],28.)


if __name__=="__main__":
    unittest.main()
