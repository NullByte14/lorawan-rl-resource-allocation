"""Resume robustness checkpoints with isolated deterministic worker processes.

The scientific runner and its recorded configuration/source hashes are unchanged.
Workers own disjoint (device count, seed) groups. Only the parent updates the
checkpoint index. Run the serial runner once to initialize the configuration.
"""
from __future__ import annotations

from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
HERE = ROOT/"experiments/robustness"
sys.path.insert(0, str(HERE))


def worker(nodes, seed, config, output):
    from run_suite import evaluate_group, dump_gzip
    args = Namespace(**config)
    result = evaluate_group(seed, nodes, args, Path(output))
    relative = f"chunks/n{nodes}_s{seed}.json.gz"
    path = Path(output)/relative
    dump_gzip(path, result)
    return relative, hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("workers must be1–12")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    output = HERE/"results"
    path = output/"index.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    from run_suite import sources, analyze
    if sources() != index["source_sha256"]:
        raise ValueError("Scientific sources changed; cannot resume existing study")
    for relative in index["chunks"]:
        if hashlib.sha256((output/relative).read_bytes()).hexdigest() != index["chunk_sha256"][relative]:
            raise ValueError("Checkpoint fingerprint mismatch")
    config = index["config"]
    pending = [(n, s) for n in config["node_counts"] for s in config["seeds"]
               if f"chunks/n{n}_s{s}.json.gz" not in index["chunks"]]
    print(f"Verified {len(index['chunks'])} checkpoints; pending groups={len(pending)}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, n, s, config, str(output)): (n, s) for n, s in pending}
        for future in as_completed(futures):
            relative, digest = future.result()
            index["chunks"].append(relative)
            index["chunks"].sort()
            index["chunk_sha256"][relative] = digest
            temporary = path.with_suffix(".tmp.json")
            temporary.write_text(json.dumps(index, indent=2), encoding="utf-8")
            temporary.replace(path)
            print(f"Completed {relative}; saved groups={len(index['chunks'])}", flush=True)
    validation = ROOT/"validation"
    validation.mkdir(exist_ok=True)
    (validation/"parallel_execution.json").write_text(json.dumps(dict(
        status="passed", workers=args.workers, resumed_groups=len(pending),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        policy="Isolated independent seed groups; parent-only atomic checkpoint index updates",
        source_hashes_verified=len(index["source_sha256"])), indent=2), encoding="utf-8")
    analyze(path, HERE/"figures")


if __name__ == "__main__":
    main()
