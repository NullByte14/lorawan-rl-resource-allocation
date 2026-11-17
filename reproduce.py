"""Reproduce the complete studies into separate output directories.

All defaults are the full thirty-seed matrices. For a short mechanics run use
`python validate.py --quick`. Historical delivered results are not overwritten.
"""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("all", "feedback", "robustness", "baselines"), default="all")
    args = parser.parse_args()
    commands = []
    if args.study in ("all", "feedback"):
        commands.append(["experiments/feedback/reproduce.py"])
    if args.study in ("all", "robustness"):
        base = ROOT/"experiments/robustness"
        commands.append([base/"run_suite.py", "--output", base/"reproduced_results",
                         "--figures", base/"reproduced_figures"])
    if args.study in ("all", "baselines"):
        base = ROOT/"experiments/baselines"
        commands.append([base/"run_online.py", "--output", base/"reproduced_results",
                         "--figures", base/"reproduced_figures"])
    for command in commands:
        subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
