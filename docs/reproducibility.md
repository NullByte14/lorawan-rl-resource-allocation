# Reproduction and data organization

## Environment

The supplied controlled study was executed with Python 3.14.7, NumPy 2.3.5 and Matplotlib 3.11.2 on Windows. The lock file records all installed scientific dependencies. Ubuntu GitHub Actions independently executes the unit/data-contract tests and fresh all-case smoke studies. Bitwise historical replay was checked on the recorded platform; cross-platform floating-point implementations can produce different last bits and topology fingerprints.

Create a Python 3.14 environment from the repository root:

```shell
python -m venv .venv
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`; on Linux/macOS use `source .venv/bin/activate`. Then:

```shell
python -m pip install -r requirements-lock.txt
python -m pip check
python validate.py --quick
```

If a sandbox denies writes to its default temporary directory, create `.tmp` inside the repository and set `TMPDIR` to that absolute path before creating the environment. Runtime caches and generated quick/reproduced outputs are excluded from Git.

## Three reproduction levels

1. `python validate.py` verifies the repository artifact manifest, audits every completed expanded-study checkpoint and saved initial policy, and runs mechanism, legacy and data-contract tests without a full new experiment.
2. `python validate.py --quick` also trains, simulates and plots small studies containing every declared scenario. Small runs demonstrate operation; the archived full matrices provide the performance evidence.
3. `python reproduce.py` executes the complete studies using their thirty-seed defaults. Use `--study feedback`, `--study robustness` or `--study baselines` to select one. It writes separate `reproduced_results/` and `reproduced_figures/` directories.

The full studies are substantially larger than smoke tests. Each script records its configuration, actual environment, source hashes and individual results. Checkpoint support is described in each experiment README. A resumed job must retain the same source and configuration; changed experiments require a new output directory.

## Regenerate figures without simulation

```shell
python regenerate_figures.py
```

This rebuilds tables, intervals, reports and figures from the committed raw results. PNG files are convenient for viewing; SVG files retain editable vector elements. Render timestamps can differ without changing scientific estimates. Each comparison resamples the same complete paired seed sets; undefined denominators are reported or rejected rather than silently dropping unfavorable cases.

## Data and provenance

- `experiments/feedback/results/raw.json` preserves the complete1,320-evaluation controlled study and 1,530 training episodes. Its510 pretrained tables are in `results/tables/`.
- `experiments/robustness/results/index.json` records scenario settings, source fingerprints and compressed per-seed checkpoints. CSV tables expose every case, learner, density, condition and phase.
- `experiments/baselines/results/` records the separate online Q-learning/Thompson comparison and protocol-matched training.
- `archive/initial_pilot/` preserves the earlier allocation code, data and documented MIX-MAB adaptation. It is historical evidence, not an additional independent confirmatory sample.
- `experiments/feedback/validation/` records independent table, accounting, figure and deterministic-replay checks. The root test suite additionally verifies the declared data matrix.

Run counts across studies are execution counts. Some seed/case combinations intentionally overlap the controlled study; they are not additional independent replications and are not pooled to narrow intervals.

Source/data bytes are preserved through `.gitattributes`; automatic line-ending conversion would invalidate archived cryptographic fingerprints. Python accepts the preserved line endings on both tested platforms. The repository manifest verifies committed artifacts separately from scientifically deterministic replay.

`python artifact_manifest.py` verifies the delivered source/data/figure fingerprints without modifying them. `--write` is an explicit maintenance action after reviewed changes, not a way to repair inconsistent simulation-source hashes. `python audit_expanded.py` independently checks the complete six-case matrices, saved priors, unknown-label partitions and energy accounting.

`resume_parallel.py` can resume the robustness study's supplied configuration using isolated seed-group workers. It verifies scientific-source and completed-checkpoint fingerprints before starting. Only the parent process updates the index, atomically. This changes execution concurrency, not the declared experiment or per-seed random streams. The regular serial runner remains the default reproduction path.

## Reporting rules

Distinguish generated frames, attempted uplinks, received frames, report opportunities, reports actually transmitted and reports successfully decoded. Keep missing labels unknown. Report training costs separately from deployment costs and do not treat desktop runtime as embedded-device energy.

Every declared case remains in tables and figures. The six-case matrix bounds the study; it is not exhaustive of every physical deployment. No confidence interval is a guarantee of physical accuracy, formal delivery equivalence or publication novelty.
