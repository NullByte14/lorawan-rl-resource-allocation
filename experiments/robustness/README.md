# Frozen-policy robustness study

This study evaluates all declared cases, including lost feedback and channel deterioration. It preserves the feedback experiment's radio engine, reward normalization and action set. It is a synthetic model evaluation, not ns-3 or hardware validation.

## Declared matrix

Six cases are crossed with 30/60/120 devices and evaluation seeds 1001–1030:

| Case | Traffic intervals in the three phases (s) | Additional feedback erasure | Channel change |
|---|---|---:|---|
| stable_low | 400 / 400 / 400 | 5% | None |
| stable_high | 160 / 160 / 160 | 5% | None |
| load_step | 400 / 160 / 400 | 5% | None |
| channel_step | 400 / 160 / 400 | 5% | 6 dB attenuation in both directions from phase 1 |
| lossy_feedback | 400 / 160 / 400 | 30% | None |
| no_feedback | 400 / 160 / 400 | 100% | None |

Each phase lasts 8,000 seconds. For channel changes, the downlink fade belongs to the triggering uplink; the same per-frame approximation is used in every feedback condition. Stable profiles regenerate the declared periodic arrivals using the same topology distribution; pairing is within each case and density. Topology/shadowing is paired across cases too.

Four Q-learning variants are evaluated: reconstructed original, energy-reward only, delivery-state only, and their combination. Every variant is crossed with A (rich+request), B (compact+request), C (rich without request), and D (compact without request). Two fixed SF12/14 dBm controls are included: one pays matched protocol A to isolate allocation effects, while the other sends no learning feedback or requests (N). The latter avoids penalizing a fixed configuration with communication it does not need. Both still pay modeled standard Class A receive windows. Neither control claims to represent ADR. A faithful ADR baseline requires a separately specified server/controller and command-delivery path; a renamed heuristic would obscure that difference.

There are **9,720 evaluation runs** and **1,080 training episodes** in the full matrix. Each learner is trained for three episodes on the common load-step/A configuration with separate seeds `evaluation_seed + 50000 + 1000 * episode`. The resulting table is frozen and copied to every case and feedback condition. This isolates feedback and environment effects under a common prior; it is not evidence of online relearning after a channel change. The independent online study is reported separately.

## Reproduce

From the repository root, using an environment with the root requirements installed:

```text
python experiments/robustness/run_suite.py --quick
python experiments/robustness/run_suite.py
```

`--quick` uses eight devices, two seeds, one training episode and 3,200-second phases. It checks the pipeline and is not performance evidence. The full run stores `results/index.json`, compressed raw checkpoints, pretrained tables, all per-seed and aggregate CSVs, and four figures in PNG/SVG. `--resume` verifies source/configuration and checkpoint hashes before skipping completed groups. `--analyze-only` regenerates tables and figures from complete checkpoints.

Raw checkpoints are partitioned by density and seed to avoid repeatedly rewriting a large growing JSON file. Each includes all 108 planned evaluations in that group, the twelve training-episode records, and shared topology coordinates. The index supplies SHA256 hashes for checkpoints and source code. Evaluation records retain phase counts, energy components, delay sums, time bins, action counts, trace hashes and unchanged-table hashes. No case is silently filtered by significance.

## Figures and interpretation

1. `01_case_density_effects`: D−A differences for all four Q learners, cases and device counts. Green favors D; marked cells have pointwise paired intervals excluding zero.
2. `02_model_comparison`: the Q learners and matched fixed control under A, plus the fixed no-feedback control N, at the main density, with mean application delivery and energy per delivered packet.
3. `03_delivery_energy_tradeoffs`: A and D trade-off points for each learner in every case.
4. `04_factor_effects`: B−A, C−A and D−A for the combined learner, keeping compression, request removal and their combination distinct.

The 95% intervals use 10,000 paired seed-bootstrap resamples. Seeds, not packets, are the independent units. Intervals are pointwise and not corrected for the many displayed comparisons. Undefined secondary delay metrics in the all-feedback-erased case retain an explicit `n_valid=0`; missing feedback never becomes a failure label or an artificial zero delay.

This matrix tests the declared conditions. It does not cover every possible topology, PHY receiver, traffic process, device mobility pattern or parameter setting. Positive cells, negative cells and intervals spanning zero must be interpreted together. Neither bitmap compression nor sparse feedback alone establishes novelty.
