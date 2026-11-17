"""Regenerate all current figures and statistical tables from saved raw data."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    commands = [
        ["experiments/feedback/analyze.py"],
        ["experiments/robustness/run_suite.py", "--analyze-only"],
        ["experiments/baselines/run_online.py", "--analyze-only"],
    ]
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
