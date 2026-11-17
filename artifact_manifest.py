"""Verify delivered source/data bytes, or explicitly refresh their manifest.

    python artifact_manifest.py           # verify only
    python artifact_manifest.py --write   # after a reviewed artifact update

Simulation-source and raw-checkpoint fingerprints are also verified by the
experiment analyses; this manifest records the repository deliverable itself.
"""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT/"ARTIFACT_MANIFEST.json"
IGNORED = {".git", ".venv", ".tmp", ".mplcache", ".mplconfig", "__pycache__",
           "quick_results", "quick_figures", "reproduced_results", "reproduced_figures",
           "smoke_results", "smoke_figures"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    files = sorted(p for p in ROOT.rglob("*") if p.is_file()
                   and not IGNORED.intersection(p.relative_to(ROOT).parts)
                   and p != MANIFEST and p.suffix != ".zip")
    actual = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files}
    if args.write:
        MANIFEST.write_text(json.dumps(dict(algorithm="SHA256", files=actual), indent=2), encoding="utf-8")
        print(f"Wrote {len(actual)} artifact fingerprints")
        return
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    missing = sorted(set(expected)-set(actual))
    changed = sorted(k for k in set(expected)&set(actual) if expected[k] != actual[k])
    unexpected = sorted(set(actual)-set(expected))
    if missing or changed or unexpected:
        raise SystemExit(json.dumps(dict(missing=missing, changed=changed, unexpected=unexpected), indent=2))
    print(f"Verified all {len(expected)} delivered artifacts")


if __name__ == "__main__":
    main()
