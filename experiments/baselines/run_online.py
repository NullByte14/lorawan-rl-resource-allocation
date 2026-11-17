"""Matched online Q-learning and delayed binary Thompson-sampling reference.

Full matrix: six cases, A/D feedback, 30 paired seeds and three learners at
60 devices. Run --quick for a small complete-matrix mechanics check.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FEEDBACK = HERE.parent / "feedback"
ROBUSTNESS = HERE.parent / "robustness"
sys.path.insert(0, str(ROBUSTNESS))
sys.path.insert(0, str(FEEDBACK))
from policies import QPolicy
from run_experiment import q_sha256, save_q
from run_suite import CASES, run_case, dump_gzip, load_gzip
import engine

MODELS = ("thompson", "reward", "both")
CONDITIONS = ("A", "D")


class ThompsonPolicy:
    """Beta(1,1) Bernoulli Thompson sampler over the common thirty actions.

    Only a resolved binary reception label updates a posterior. Scalar energy
    reward, SNR/state, and unreported outcomes are deliberately not used.
    This is a delayed-feedback 30-arm adaptation, not a Navas replication.
    """
    def __init__(self, n_actions=30, update_enabled=True):
        if n_actions < 1:
            raise ValueError("n_actions must be positive")
        self.n_actions = n_actions
        self.successes = np.zeros(n_actions, dtype="<f8")
        self.failures = np.zeros(n_actions, dtype="<f8")
        self.update_enabled = update_enabled
        self.update_count = 0

    def choose(self, state, rng):
        return int(np.argmax(rng.beta(self.successes + 1., self.failures + 1.)))

    def update(self, state, action, reward, next_state, success=None, energy_norm=None):
        if not self.update_enabled:
            return
        if not 0 <= action < self.n_actions or success not in (0, 1):
            raise ValueError("Thompson update needs an action and a resolved binary label")
        self.successes[action] += int(success)
        self.failures[action] += 1 - int(success)
        self.update_count += 1

    @property
    def policy_size_bytes(self):
        return self.successes.nbytes + self.failures.nbytes


def policy_hash(policy):
    if isinstance(policy, list):
        return hashlib.sha256(json.dumps([policy_hash(p) for p in policy], separators=(",", ":")).encode()).hexdigest()
    if isinstance(policy, ThompsonPolicy):
        return hashlib.sha256(b"beta-bernoulli-ts-v1" + policy.successes.astype("<f8").tobytes()
                              + policy.failures.astype("<f8").tobytes()).hexdigest()
    return q_sha256(policy)


def save_policy(policy, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(policy, ThompsonPolicy):
        np.savez_compressed(path, successes=policy.successes, failures=policy.failures,
                            training_updates=policy.update_count, prior_alpha=1., prior_beta=1.)
    else:
        save_q(policy, path)


def sources():
    paths = ["experiments/feedback/engine.py", "experiments/feedback/policies.py",
             "experiments/feedback/feedback_codec.py", "experiments/feedback/run_experiment.py",
             "experiments/feedback/analyze.py", "experiments/robustness/run_suite.py",
             "experiments/baselines/run_online.py", "experiments/baselines/analyze.py"]
    return {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in paths}


def train(seed, learner, condition, args, output):
    policy = ThompsonPolicy() if learner == "thompson" else QPolicy(alpha=.1, gamma=.9, epsilon=.25)
    records = []
    for episode in range(args.train_episodes):
        environment_seed = seed + 50000 + 1000 * episode
        before = policy.update_count
        result = run_case(environment_seed, args.nodes, args.phase_seconds, policy,
                          "reward" if learner == "thompson" else learner,
                          "load_step", condition, training=True)
        records.append(dict(seed=seed, nodes=args.nodes, learner=learner, condition=condition,
            episode=episode, environment_seed=environment_seed, scenario="load_step",
            total_j=sum(p["total_j"] for p in result["phases"]),
            attempted=sum(p["attempted"] for p in result["phases"]),
            received=sum(p["received"] for p in result["phases"]),
            reports=sum(p["reports"] for p in result["phases"]),
            updates=policy.update_count-before, policy_sha256=policy_hash(policy),
            trace_sha256=result["trace_sha256"], topology_sha256=result["topology_sha256"],
            elapsed_seconds=result["elapsed_seconds"]))
    relative = f"tables/s{seed}_{learner}_{condition}.npz"
    save_policy(policy, output / relative)
    return policy, records, relative


def run_group(seed, args, output):
    records, training = [], []
    for learner in MODELS:
        for condition in CONDITIONS:
            pretrained, train_records, table_file = train(seed, learner, condition, args, output)
            training.extend(train_records)
            for scenario in args.cases:
                initial = copy.deepcopy(pretrained)
                initial.update_enabled = True
                if isinstance(initial, QPolicy):
                    initial.epsilon = .05
                policies = [copy.deepcopy(initial) for _ in range(args.nodes)]
                before = sum(p.update_count for p in policies)
                initial_hash = policy_hash(policies)
                result = run_case(seed, args.nodes, args.phase_seconds, policies,
                                  "reward" if learner == "thompson" else learner,
                                  scenario, condition)
                result.pop("positions")  # Regenerable and fingerprinted, avoid repeated geometry.
                result.update(mode="online", scenario=scenario, seed=seed, nodes=args.nodes,
                    learner=learner, condition=condition, initial_policy_sha256=initial_hash,
                    final_policy_sha256=policy_hash(policies), training_table_file=table_file,
                    training_policy_sha256=policy_hash(pretrained),
                    evaluation_updates=result["updates"]-before)
                if result["evaluation_updates"] != sum(p["resolved"] for p in result["phases"]):
                    raise AssertionError("A resolved packet did not produce exactly one update")
                if scenario == "no_feedback" and result["evaluation_updates"]:
                    raise AssertionError("Unknown outcomes became learning labels")
                records.append(result)
    for scenario in args.cases:
        signatures = {(r["trace_sha256"], r["topology_sha256"])
                      for r in records if r["scenario"] == scenario}
        if len(signatures) != 1:
            raise AssertionError("Baseline learners did not share the exogenous environment")
    return dict(seed=seed, training=training, records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--phase-seconds", type=int)
    parser.add_argument("--train-episodes", type=int)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    args.nodes = args.nodes or (8 if args.quick else 60)
    args.seeds = args.seeds or ([1001, 1002] if args.quick else list(range(1001, 1031)))
    args.phase_seconds = args.phase_seconds or (3200 if args.quick else 8000)
    args.train_episodes = args.train_episodes or (1 if args.quick else 3)
    output = args.output or HERE / ("quick_results" if args.quick else "results")
    figures = args.figures or HERE / ("quick_figures" if args.quick else "figures")
    if args.nodes < 1 or args.phase_seconds < 3200 or args.phase_seconds % 800 or args.train_episodes < 1:
        parser.error("Positive counts and phase length>=3200 divisible by800 required")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.cases)) != len(args.cases):
        parser.error("Do not repeat seeds or cases")
    if {s+50000+1000*ep for s in args.seeds for ep in range(args.train_episodes)}.intersection(args.seeds):
        parser.error("Training and evaluation environment seeds overlap")
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.json"
    if not args.analyze_only:
        config = dict(nodes=args.nodes, seeds=args.seeds, cases=args.cases,
            case_specs={c: asdict(CASES[c]) for c in args.cases}, phase_seconds=args.phase_seconds,
            train_episodes=args.train_episodes, learners=list(MODELS), conditions=list(CONDITIONS),
            prior="Thompson Beta(1,1) independently per action", common_training_case="load_step",
            q_epsilon_train=.25, q_epsilon_evaluation=.05, alpha=.1, gamma=.9,
            beta=engine.BETA, energy_reference_j=engine.E_REF, python_version=platform.python_version(),
            numpy_version=np.__version__,
            comparison="All online, same30SF/TP options and costed A/D protocols; TS objective is binary success, Q objectives include energy",
            reference_status="Adapted delayed binary Thompson sampler, not an exact Navas energy-aware bandit replication")
        source = sources()
        if args.resume and index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if index["config"] != config or index["source_sha256"] != source:
                raise ValueError("Cannot resume changed sources or configuration")
        else:
            index = dict(config=config, source_sha256=source, chunks=[], chunk_sha256={})
        for seed in args.seeds:
            relative = f"chunks/s{seed}.json.gz"
            path = output / relative
            if args.resume and relative in index["chunks"]:
                if hashlib.sha256(path.read_bytes()).hexdigest() != index["chunk_sha256"][relative]:
                    raise ValueError("Changed checkpoint")
            else:
                dump_gzip(path, run_group(seed, args, output))
                if relative not in index["chunks"]:
                    index["chunks"].append(relative)
                index["chunk_sha256"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
                index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
            print(f"Online baseline seed={seed}: all learners/protocols/scenarios saved", flush=True)
    subprocess.run([sys.executable, str(HERE / "analyze.py"), "--input", str(index_path),
                    "--figures", str(figures)], check=True)


if __name__ == "__main__":
    main()
