"""Verify delivered file hashes and the complete default experimental matrix.

Uses Python's standard library only. Run: python verify.py
"""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent


def main():
    manifest = json.loads((HERE/"MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    for name, expected in manifest["files"].items():
        path = (HERE/name).resolve()
        if not path.is_relative_to(HERE) or not path.is_file():
            failures.append(f"Missing or invalid path: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"Changed: {name}")
    if failures:
        raise SystemExit("\n".join(failures))
    raw = json.loads((HERE/"results/raw.json").read_text(encoding="utf-8"))
    config = raw["config"]
    expected = {(mode, nodes, seed, learner, condition)
                for mode in ("frozen", "online")
                for nodes in (config["node_counts"] if mode == "frozen" else [config["online_nodes"]])
                for seed in config["seeds"]
                for learner in (("original", "reward", "both") if mode == "frozen" else ("reward", "both"))
                for condition in ("A", "B", "C", "D")}
    actual = {(r["mode"], r["nodes"], r["seed"], r["learner"], r["condition"])
              for r in raw["records"]}
    if expected != actual or len(actual) != len(raw["records"]):
        raise SystemExit("Incomplete or duplicate evaluation matrix")
    expected_training = (len(config["node_counts"])*len(config["seeds"])*3 +
                         len(config["seeds"])*2*4)*config["train_episodes"]
    if len(raw["training"]) != expected_training:
        raise SystemExit("Incomplete training matrix")
    for name, expected_hash in raw["source_sha256"].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest() != expected_hash:
            raise SystemExit(f"Simulation source changed: {name}")
    print(json.dumps(dict(files_verified=len(manifest["files"]),
                          evaluations=len(actual), training_episodes=expected_training,
                          paired_seeds=len(config["seeds"]), status="passed"), indent=2))


if __name__ == "__main__":
    main()
