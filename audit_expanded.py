"""Independent full-matrix, saved-prior and accounting audit of expanded studies.

Unlike a simulation or bootstrap analysis, this reads every completed checkpoint
and saved initial policy. It never requires the proposed policy to outperform a
control. Run after the full robustness and online-baseline matrices finish.
"""
from pathlib import Path
import copy
import hashlib
import gzip
import json
import math
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/"experiments/feedback"))
sys.path.insert(0, str(ROOT/"experiments/robustness"))
sys.path.insert(0, str(ROOT/"experiments/baselines"))
from policies import QPolicy
from run_experiment import q_sha256
from run_online import ThompsonPolicy, policy_hash
from run_suite import run_case

CASES = ("stable_low", "stable_high", "load_step", "channel_step", "lossy_feedback", "no_feedback")
Q_MODELS = ("original", "reward", "state", "both")
FIXED = ("fixed_conservative", "fixed_no_feedback")


def load(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def saved_policy(path, learner):
    with np.load(path, allow_pickle=False) as archive:
        if learner == "thompson":
            policy = ThompsonPolicy()
            policy.successes = archive["successes"].copy()
            policy.failures = archive["failures"].copy()
        else:
            policy = QPolicy()
            policy.q = {tuple(map(int, state)): values.copy()
                        for state, values in zip(archive["states"], archive["values"])}
        policy.update_count = int(archive["training_updates"])
    return policy


def check_accounting(record):
    for p in record["phases"]:
        assert 0 <= p["received"] <= p["attempted"] <= p["generated"]
        assert p["resolved"]+p["unresolved"]+p["tail_unknown"] == p["attempted"]
        assert p["reports"]+p["dl_lost"] == p["dl_sent"]
        assert math.isclose(p["total_j"], p["tx_j"]+p["rx_j"]+p["sleep_j"], rel_tol=1e-12)
        assert all(math.isfinite(p[k]) and p[k] >= 0 for k in ("tx_j", "rx_j", "sleep_j", "total_j", "dl_airtime_s"))
        expected_duration = 1.810432 if record["condition"] in ("A", "C") else 1.155072
        assert math.isclose(p["dl_airtime_s"], p["dl_sent"]*expected_duration, abs_tol=1e-7)
        if record["scenario"] == "no_feedback":
            assert p["reports"] == p["resolved"] == 0
        if record["learner"] == "fixed_no_feedback":
            assert p["dl_sent"] == p["request_frames"] == p["reports"] == 0
    for key in ("generated", "attempted", "received"):
        assert sum(p[key] for p in record["phases"]) == sum(b[key] for b in record["bins"])


def audit_study(study):
    folder = ROOT/"experiments"/study/"results"
    index = load(folder/"index.json")
    config = index["config"]
    assert config["seeds"] == list(range(1001, 1031))
    assert config["cases"] == list(CASES)
    assert config["train_episodes"] == 3 and config["phase_seconds"] == 8000
    for name, expected in index["source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    records, training = [], []
    for relative in index["chunks"]:
        path = folder/relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == index["chunk_sha256"][relative]
        chunk = load(path)
        records.extend(chunk["records"])
        training.extend(chunk["training"])
    if study == "robustness":
        assert config["node_counts"] == [30, 60, 120]
        expected = {(case, n, s, l, a) for case in CASES for n in (30, 60, 120)
                    for s in range(1001, 1031) for l in Q_MODELS+FIXED
                    for a in (("N",) if l == "fixed_no_feedback" else ("A",) if l == "fixed_conservative" else tuple("ABCD"))}
        train_expected = {(n, s, l, ep) for n in (30, 60, 120) for s in range(1001, 1031)
                          for l in Q_MODELS for ep in range(3)}
        train_actual = [(t["nodes"], t["seed"], t["learner"], t["episode"]) for t in training]
    else:
        assert config["nodes"] == 60
        expected = {(case, 60, s, l, a) for case in CASES for s in range(1001, 1031)
                    for l in ("thompson", "reward", "both") for a in ("A", "D")}
        train_expected = {(s, l, a, ep) for s in range(1001, 1031)
                          for l in ("thompson", "reward", "both") for a in ("A", "D") for ep in range(3)}
        train_actual = [(t["seed"], t["learner"], t["condition"], t["episode"]) for t in training]
    assert set(train_actual) == train_expected and len(train_actual) == len(set(train_actual))
    assert not set(config["seeds"]) & {t["environment_seed"] for t in training}
    for t in training:
        assert t["environment_seed"] == t["seed"]+50000+1000*t["episode"]
    actual = [(r["scenario"], r["nodes"], r["seed"], r["learner"], r["condition"]) for r in records]
    assert set(actual) == expected and len(actual) == len(set(actual))
    environments, initial = {}, {}
    priors = {}
    for r in records:
        check_accounting(r)
        key = r["scenario"], r["nodes"], r["seed"]
        signature = r["trace_sha256"], r["topology_sha256"]
        assert environments.setdefault(key, signature) == signature
        if study == "robustness":
            assert r["initial_q_sha256"] == r["final_q_sha256"] and r["evaluation_updates"] == 0
            if r["learner"] in Q_MODELS:
                key = r["nodes"], r["seed"], r["learner"]
                assert initial.setdefault(key, r["initial_q_sha256"]) == r["initial_q_sha256"]
                if key not in priors:
                    policy = saved_policy(folder/"tables"/f"n{key[0]}_s{key[1]}_{key[2]}.npz", key[2])
                    priors[key] = q_sha256(policy)
                assert priors[key] == r["initial_q_sha256"]
        else:
            assert r["evaluation_updates"] == sum(p["resolved"] for p in r["phases"])
            key = r["seed"], r["learner"], r["condition"]
            assert initial.setdefault(key, r["initial_policy_sha256"]) == r["initial_policy_sha256"]
            if key not in priors:
                policy = saved_policy(folder/r["training_table_file"], r["learner"])
                digest = policy_hash(policy)
                repeated = hashlib.sha256(json.dumps([digest]*r["nodes"], separators=(",", ":")).encode()).hexdigest()
                priors[key] = digest, repeated
            assert priors[key][0] == r["training_policy_sha256"]
            assert priors[key][1] == r["initial_policy_sha256"]
    return dict(status="passed", study=study, evaluations=len(records), training_episodes=len(training),
                complete_case_count=len(CASES), paired_environment_groups=len(environments),
                saved_initial_policy_fingerprints_checked=len(priors), source_hashes_verified=len(index["source_sha256"]),
                all_unfavorable_and_negative_cases_retained=True)


def main():
    audits = [audit_study(name) for name in ("robustness", "baselines")]
    replayed = []
    specifications = ([("robustness", 120, 1001, "both", "D", "channel_step"),
                       ("baselines", 60, 1001, "thompson", "D", "load_step")]
                      if platform.system() == "Windows" else [])
    for study, nodes, seed, learner, condition, scenario in specifications:
        folder = ROOT/"experiments"/study/"results"
        index = load(folder/"index.json")
        name = f"chunks/n{nodes}_s{seed}.json.gz" if study == "robustness" else f"chunks/s{seed}.json.gz"
        chunk = load(folder/name)
        row = next(r for r in chunk["records"] if r["learner"] == learner
                   and r["condition"] == condition and r["scenario"] == scenario)
        relative = f"tables/n{nodes}_s{seed}_{learner}.npz" if study == "robustness" else row["training_table_file"]
        initial = saved_policy(folder/relative, learner)
        initial.update_enabled = study == "baselines"
        if isinstance(initial, QPolicy):
            initial.epsilon = 0. if study == "robustness" else .05
        policies = initial if study == "robustness" else [copy.deepcopy(initial) for _ in range(nodes)]
        replay = run_case(seed, nodes, index["config"]["phase_seconds"], policies,
                          "reward" if learner == "thompson" else learner, scenario, condition)
        checked = 0
        for key, value in replay.items():
            if key not in ("positions", "elapsed_seconds"):
                assert value == row[key], f"Replay mismatch: {study}/{scenario}/{key}"
                checked += 1
        final_key = "final_q_sha256" if study == "robustness" else "final_policy_sha256"
        assert policy_hash(policies) == row[final_key]
        replayed.append(dict(study=study, nodes=nodes, seed=seed, learner=learner,
                             scenario=scenario, condition=condition,
                             deterministic_fields_checked=checked, status="passed"))
    result = dict(status="passed", studies=audits,
                  representative_saved_prior_replays=replayed,
                  replay_scope="Historical bitwise replays run on the recorded Windows platform; other platforms validate fresh paired execution through the quick studies",
                  interpretation="Complete declared matrices and internal consistency; not external physical validation")
    (ROOT/"validation/expanded_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
