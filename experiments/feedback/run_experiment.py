"""Reproducible paired feedback experiment; see README.md for assumptions.

Default: 30 held-out seeds, N=30/60/120 frozen evaluations, then N=60
protocol-matched local online evaluations. Each phase lasts 8,000 seconds.
The first pilot is never modified or imported.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

import engine
from policies import QPolicy

HERE = Path(__file__).resolve().parent
CORE_FILES = ("engine.py", "policies.py", "feedback_codec.py", "run_experiment.py",
              "analyze.py", "reproduce.py")
FROZEN_LEARNERS = ("original", "reward", "both")
ONLINE_LEARNERS = ("reward", "both")


def q_arrays(policy):
    keys = sorted(policy.q)
    width = len(keys[0]) if keys else 0
    states = np.asarray(keys, dtype="<i2").reshape(len(keys), width)
    values = (np.stack([policy.q[key] for key in keys]).astype("<f4")
              if keys else np.empty((0, policy.n_actions), dtype="<f4"))
    return states, values


def q_sha256(policy):
    if isinstance(policy, list):
        return hashlib.sha256(json.dumps([q_sha256(p) for p in policy],
                                        separators=(",", ":")).encode()).hexdigest()
    states, values = q_arrays(policy)
    header = json.dumps([states.shape, values.shape], separators=(",", ":")).encode()
    return hashlib.sha256(header + states.tobytes() + values.tobytes()).hexdigest()


def save_q(policy, path):
    states, values = q_arrays(policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, states=states, values=values,
                        alpha=policy.alpha, gamma=policy.gamma,
                        training_updates=policy.update_count)


def source_hashes():
    return {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
            for name in CORE_FILES}


def write_raw(path, payload):
    # Replace only this experiment's generated output, never the first pilot.
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def assert_consistency(records):
    groups = {}
    frozen_tables = {}
    for row in records:
        key = (row["mode"], row["nodes"], row["seed"])
        hashes = (row["trace_sha256"], row["topology_sha256"])
        if key in groups and groups[key] != hashes:
            raise AssertionError(f"Unpaired exogenous traces: {key}")
        groups[key] = hashes
        for phase in row["phases"]:
            if not 0 <= phase["received"] <= phase["attempted"] <= phase["generated"]:
                raise AssertionError("Invalid packet accounting")
            if phase["resolved"]+phase["unresolved"]+phase["tail_unknown"] != phase["attempted"]:
                raise AssertionError("Outcome labels are not a partition of attempts")
            expected = phase["tx_j"]+phase["rx_j"]+phase["sleep_j"]
            if not np.isclose(expected, phase["total_j"], atol=1e-12):
                raise AssertionError("Invalid energy accounting")
        if row["mode"] == "frozen":
            if row["evaluation_updates"] != 0 or row["initial_q_sha256"] != row["final_q_sha256"]:
                raise AssertionError("Frozen table changed")
            key = (row["nodes"], row["seed"], row["learner"])
            previous = frozen_tables.setdefault(key, row["initial_q_sha256"])
            if previous != row["initial_q_sha256"]:
                raise AssertionError("Feedback arms did not receive identical tables")


def train(seed, nodes, learner_name, condition, mode, args, payload, output):
    policy = QPolicy(alpha=.1, gamma=.9, epsilon=.25, update_enabled=True)
    feedback = engine.CONFIGS["A" if condition == "common" else condition]
    for episode in range(args.train_episodes):
        environment_seed = seed+50000+episode*1000
        before = policy.update_count
        result = engine.run(environment_seed, nodes, args.phase_seconds, policy,
                            learner_name, feedback=feedback, training=True)
        payload["training"].append(dict(
            mode=mode, seed=seed, nodes=nodes, learner=learner_name,
            condition=condition, episode=episode, environment_seed=environment_seed,
            total_j=sum(p["total_j"] for p in result["phases"]),
            attempted=sum(p["attempted"] for p in result["phases"]),
            received=sum(p["received"] for p in result["phases"]),
            dl_airtime_s=sum(p["dl_airtime_s"] for p in result["phases"]),
            updates=policy.update_count-before, table_sha256=q_sha256(policy),
            trace_sha256=result["trace_sha256"],
            elapsed_seconds=result["elapsed_seconds"]))
    snapshot = Path("tables")/f"{mode}_n{nodes}_s{seed}_{learner_name}_{condition}.npz"
    save_q(policy, output/snapshot)
    return policy, snapshot.as_posix()


def evaluate(seed, nodes, learner_name, condition, mode, pretrained, snapshot, args):
    policy = copy.deepcopy(pretrained)
    policy.epsilon = 0. if mode == "frozen" else .05
    policy.update_enabled = mode == "online"
    if mode == "online":
        policy = [copy.deepcopy(policy) for _ in range(nodes)]
    before = sum(p.update_count for p in policy) if isinstance(policy, list) else policy.update_count
    initial_hash = q_sha256(policy)
    result = engine.run(seed, nodes, args.phase_seconds, policy, learner_name,
                        feedback=engine.CONFIGS[condition])
    result.update(mode=mode, seed=seed, nodes=nodes, learner=learner_name,
                  condition=condition, initial_q_sha256=initial_hash,
                  final_q_sha256=q_sha256(policy), training_table_file=snapshot,
                  training_table_sha256=q_sha256(pretrained),
                  evaluation_updates=result["updates"]-before)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "frozen", "online"), default="all")
    parser.add_argument("--append", action="store_true", help="append online evaluations to a frozen run")
    parser.add_argument("--nodes", type=int, nargs="+", default=[30, 60, 120])
    parser.add_argument("--online-nodes", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1001, 1031)))
    parser.add_argument("--phase-seconds", type=int, default=8000)
    parser.add_argument("--train-episodes", type=int, default=3)
    parser.add_argument("--output", type=Path, default=HERE/"results")
    args = parser.parse_args()
    if (args.phase_seconds < 3200 or args.phase_seconds % 800 or args.train_episodes < 1
            or min(args.nodes+[args.online_nodes]) < 1 or len(set(args.seeds)) != len(args.seeds)
            or len(set(args.nodes)) != len(args.nodes)):
        parser.error("Use positive unique node counts/seeds, >=1 training episode, phases>=3200 divisible by800")
    if args.append and args.stage != "online":
        parser.error("--append is only supported for --stage online")
    evaluation_seeds = set(args.seeds)
    training_seeds = {s+50000+1000*ep for s in args.seeds for ep in range(args.train_episodes)}
    if evaluation_seeds & training_seeds:
        parser.error("Evaluation and training seeds overlap")
    return args


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output/"raw.json"
    current_sources = source_hashes()
    if args.append:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        if payload["source_sha256"] != current_sources:
            raise ValueError("Source changed since frozen run; rerun instead of appending")
        for name in ("seeds", "node_counts", "phase_seconds", "train_episodes", "online_nodes"):
            expected = args.nodes if name == "node_counts" else getattr(args, name)
            if payload["config"][name] != expected:
                raise ValueError(f"Config mismatch while appending: {name}")
        if any(r["mode"] == "online" for r in payload["records"]):
            raise ValueError("Online data already exists; refusing duplicate records")
        assert_consistency(payload["records"])
    else:
        config = dict(
            schema_version=1, seeds=args.seeds, node_counts=args.nodes,
            online_nodes=args.online_nodes, phase_seconds=args.phase_seconds,
            train_episodes=args.train_episodes, alpha=.1, gamma=.9,
            epsilon_train=.25, epsilon_frozen=0., epsilon_online=.05,
            beta=engine.BETA, energy_reference_j=engine.E_REF,
            feedback={key: asdict(value) for key, value in engine.CONFIGS.items()},
            traffic_intervals_s=[400, 160, 400], gateway_count=1, channels=3,
            python_version=platform.python_version(), numpy_version=np.__version__,
            matplotlib_version=importlib.metadata.version("matplotlib"),
            baseline_status="Paper-based DSF2R reconstruction; not the published authors' executable training",
            frozen_training="Once per learner/seed/N with common condition A; copied to all four arms",
            online_training="Separate per protocol/learner/seed, then independent per-device evaluation copies",
            training_seed_rule="evaluation_seed + 50000 + 1000 * episode_index",
            uncertainty="10000 paired percentile bootstrap resamples over complete evaluation seeds",
            units="seconds, joules, logical frames; NbTrans=1, no retry/queue",
            stage_order=[])
        payload = dict(config=config, source_sha256=current_sources, training=[], records=[])
    started = time.perf_counter()
    if args.stage in ("all", "frozen"):
        for nodes in args.nodes:
            for seed in args.seeds:
                for learner_name in FROZEN_LEARNERS:
                    pretrained, snapshot = train(seed, nodes, learner_name, "common", "frozen", args, payload, args.output)
                    for condition in engine.CONFIGS:
                        payload["records"].append(evaluate(seed, nodes, learner_name, condition,
                            "frozen", pretrained, snapshot, args))
                assert_consistency(payload["records"])
                write_raw(raw_path, payload)
                print(f"Frozen N={nodes} seed={seed}: all learners/arms saved", flush=True)
        payload["config"]["stage_order"].append("frozen: pairing, label accounting and unchanged-table checks passed")
        write_raw(raw_path, payload)
    if args.stage in ("all", "online"):
        if payload["records"]:
            assert_consistency(payload["records"])
        for seed in args.seeds:
            for learner_name in ONLINE_LEARNERS:
                for condition in engine.CONFIGS:
                    pretrained, snapshot = train(seed, args.online_nodes, learner_name,
                                                 condition, "online", args, payload, args.output)
                    payload["records"].append(evaluate(seed, args.online_nodes, learner_name,
                        condition, "online", pretrained, snapshot, args))
            assert_consistency(payload["records"])
            write_raw(raw_path, payload)
            print(f"Online N={args.online_nodes} seed={seed}: both learners/all arms saved", flush=True)
        payload["config"]["stage_order"].append("online: protocol-matched training and local evaluation copies completed")
    payload["config"]["runner_elapsed_seconds_last_stage"] = time.perf_counter()-started
    write_raw(raw_path, payload)
    assert_consistency(payload["records"])
    print(f"Saved {len(payload['records'])} evaluations and {len(payload['training'])} training episodes to {raw_path}", flush=True)


if __name__ == "__main__":
    main()
