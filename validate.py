"""Run mechanism/data-contract checks and optional small all-case simulations."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(arguments, cwd=ROOT):
    print("Running: python " + " ".join(map(str, arguments)), flush=True)
    subprocess.run([sys.executable, *map(str, arguments)], cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="also execute and plot small all-case studies")
    args = parser.parse_args()
    temporary = ROOT/".tmp"
    temporary.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(temporary)
    run(["artifact_manifest.py"])
    run(["audit_expanded.py"])
    run(["-m", "unittest", "discover", "-s", "tests", "-v"])
    run(["-m", "unittest", "discover", "-s", "tests", "-v"], ROOT/"archive/initial_pilot")
    run(["-m", "unittest", "discover", "-s", "tests", "-v"], ROOT/"experiments/robustness")
    run(["-m", "unittest", "discover", "-s", "tests", "-v"], ROOT/"experiments/baselines")
    if args.quick:
        run(["experiments/feedback/reproduce.py", "--quick"])
        run(["experiments/robustness/run_suite.py", "--quick"])
        run(["experiments/baselines/run_online.py", "--quick"])
    else:
        run(["-m", "unittest", "discover", "-s", "tests", "-v"], ROOT/"experiments/feedback")
    print("All requested validation commands passed.", flush=True)


if __name__ == "__main__":
    main()
