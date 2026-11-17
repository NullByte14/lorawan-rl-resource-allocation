"""One command to test, simulate, and regenerate all four figures.

    python reproduce.py

Writes reproduced_results/ and reproduced_figures/, preserving delivered data.
Use --quick for a tiny mechanics demonstration, never for performance claims.
"""
from pathlib import Path
import argparse
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    output = HERE/("quick_results" if args.quick else "reproduced_results")
    figures = HERE/("quick_figures" if args.quick else "reproduced_figures")
    commands = [
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        [sys.executable, "run_experiment.py", "--output", str(output)],
        [sys.executable, "analyze.py", "--input", str(output/"raw.json"),
         "--output", str(output), "--figures", str(figures)],
    ]
    if args.quick:
        commands[1].extend(["--nodes", "8", "--online-nodes", "8", "--seeds", "1001", "1002",
                            "--phase-seconds", "3200", "--train-episodes", "1"])
    for command in commands:
        subprocess.run(command, cwd=HERE, check=True)
    print(f"Completed. Read {output/'RESULTS.md'}; figures: {figures}")


if __name__ == "__main__":
    main()
