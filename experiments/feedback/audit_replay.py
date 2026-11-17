"""Replay one saved frozen run and one online run from their saved Q-tables.

Default cases: combined learner, D, 60 devices, seed1001. All deterministic
engine outputs and final Q hashes must match. Wall-clock timing is excluded.
"""
from pathlib import Path
import copy
import json

import numpy as np

import engine
from policies import QPolicy
from run_experiment import q_sha256

HERE = Path(__file__).resolve().parent


def main():
    output = HERE/"validation"
    output.mkdir(exist_ok=True)
    raw = json.loads((HERE/"results/raw.json").read_text(encoding="utf-8"))
    checks = []
    for mode in ("frozen", "online"):
        record = next(r for r in raw["records"] if r["mode"] == mode
                      and r["nodes"] == 60 and r["seed"] == 1001
                      and r["learner"] == "both" and r["condition"] == "D")
        policy = QPolicy(epsilon=0. if mode == "frozen" else .05,
                         update_enabled=mode == "online")
        with np.load(HERE/"results"/record["training_table_file"], allow_pickle=False) as saved:
            policy.q = {tuple(map(int, state)): values.copy()
                        for state, values in zip(saved["states"], saved["values"])}
            policy.update_count = int(saved["training_updates"])
        if q_sha256(policy) != record["training_table_sha256"]:
            raise AssertionError("Saved table does not match its recorded canonical hash")
        if mode == "online":
            policy = [copy.deepcopy(policy) for _ in range(60)]
        if q_sha256(policy) != record["initial_q_sha256"]:
            raise AssertionError("Replay initial Q hash mismatch")
        result = engine.run(1001, 60, raw["config"]["phase_seconds"], policy,
                            "both", feedback=engine.CONFIGS["D"])
        keys = [key for key in result if key != "elapsed_seconds"]
        for key in keys:
            if result[key] != record[key]:
                raise AssertionError(f"{mode}: deterministic replay differs in {key}")
        if q_sha256(policy) != record["final_q_sha256"]:
            raise AssertionError("Replay final Q hash mismatch")
        checks.append(dict(mode=mode, nodes=60, seed=1001, learner="both", condition="D",
                           deterministic_fields_checked=len(keys), status="passed"))
    document = dict(status="passed", cases=checks,
                    excluded_field="elapsed_seconds: desktop timing is nondeterministic")
    (output/"deterministic_replay.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
