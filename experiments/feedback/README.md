# LoRaWAN feedback experiment: start here

This package extends the earlier SF–power allocation pilot with compact feedback. It contains executable experiments, saved individual results, four figures, and a formal research interpretation. The original allocation experiment is preserved in `../../archive/initial_pilot/`.

**Simulator:** an independently implemented Python/NumPy discrete-event model. This is not ns-3, LoRaSim, a hardware test, a complete LoRaWAN stack, or an exact reproduction of published DSF2R. All propagation, receiver and radio-current values below are declared model assumptions.

Read `../../docs/evidence_scope.md` for the evidence scope and `results/RESULTS.md` for measured outcomes and confidence intervals. Open the four PNG figures in `figures/`; editable SVG copies are included.

## The question and controlled comparisons

Does reducing learning-feedback overhead improve the delivery–energy trade-off of an SF–power allocator?

| Condition | Report | Extra request bytes on every eighth attempted uplink |
|---|---|---:|
| A | Rich, 35 complete PHYPayload bytes | 4 |
| B | Compact, 14–16 PHYPayload bytes | 4 |
| C | Rich | 0 |
| D | Compact | 0 |

All conditions report at the same cadence, every eight attempted logical uplinks, with a 16-frame receipt window. No adaptive cadence is added. A and C carry a 22-byte application report: four-byte anchor, two-byte receipt bitmap, and sixteen half-dB SNR slots. B and D carry an actually encoded/decoded one-, two- or three-byte application report containing mode bits, latest SNR-state bin and individual receipt information. All include 13 modeled framing bytes. The full LoRaWAN frame, encryption and radio bits are not serialized.

The four request bytes are a modeled uplink-size increment; all conditions share a configured reporting schedule. In request-free conditions the server uses the received frame counter. A lost triggering uplink yields no report. Reports use the current uplink's RX2 opportunity and are discarded if unavailable; stale queued snapshots are never reassociated with a later uplink. Repetitions, retries and multiple sessions are not modeled (`NbTrans=1`).

The compact format is not a standardized MAC command. Individual receipt bits preserve which action succeeded. Silence means unknown, not failed or successful delivery. Later overlapping windows can recover labels. Each outcome is applied once; unrecoverable and end-of-run outcomes remain unknown and are counted.

## Learners and training

All learners choose among the same 30 SF–power pairs: SF7–12 and 6/8/10/12/14 dBm.

- `original`: paper-based DSF2R reconstruction, with SNR-bin and current-power state and the printed SF/SNR-margin reward. Failed-frame SNR uses the value available when its action was chosen, or zero reward if unknown. This declared reconstruction is not verified as the authors' unpublished training behavior.
- `reward`: same state, with `reward = received - 0.2 * attempt_energy / E_REF`.
- `both`: that reward plus a delivery-history state derived from the last 20 resolved packet outcomes, with bins below 0.5, 0.5–0.8, at least 0.8, and initially unknown.

There are 55 known SNR-state bins plus unknown. The latest bin arrives directly from decoded reports; compact reports do not invent a dB value. The reward reference remains **E_REF = 0.36383221248 J**, alpha=0.1 and gamma=0.9 throughout. Attempt energy includes modeled TX and receive costs; background sleep enters evaluation totals rather than reward.

**Frozen comparison:** three training episodes per learner, seed and device count using condition A, epsilon=0.25. Four identical copies of each resulting table then enter A–D, with updates disabled and epsilon=0. A frozen table still selects different settings when observed state changes. Initial/final Q-table hashes verify that evaluation does not change it.

**Online confirmation:** at 60 devices, train `reward` and `both` separately for each protocol, for three episodes. Each deployed device receives its own table copy, updates locally, and uses epsilon=0.05. No device obtains another's updates for free. This comparison includes protocol effects during both training and deployment; it does not isolate evaluation updates alone. Differences from frozen evaluation also include exploration.

Original-reward training always retains historical per-frame SNR in rich reports. Its compact evaluations are frozen and do not evaluate that reward. Original online learning with compact reports is explicitly rejected. No unsupported historical SNR is substituted.

Training energy is reported separately and means simulated radio/sleep expenditure in the training environment, not desktop CPU training energy. Plotted evaluation energy is not a lifetime or training-amortized battery estimate. Numeric tables are saved in `results/tables/` as non-pickle NumPy archives with states, float32 values and learning parameters.

## Network and experiment settings

The full run uses 30 paired seeds, **1001–1030**, at 30, 60 and 120 devices for all three frozen learners and all four conditions. Online confirmation uses both energy-reward learners and all conditions at 60 devices. This gives **1,320 evaluation runs and 1,530 training episodes**.

Training environment seeds are `evaluation_seed + 50000 + 1000 * episode_index`, separate from evaluation. Within each seed and device count, generated positions, traffic times, channel choices, shadowing and fading are paired. Actual attempts may differ legitimately because airtime affects duty admission. Raw trace and topology hashes verify the pairing.

| Setting | Value |
|---|---|
| Topology | One fixed half-duplex gateway; stationary devices, area-uniform radii 200–2600 m |
| Traffic | Three 8,000-second phases; per-device intervals 400 → 160 → 400 seconds, randomized initial offsets |
| Channels | Three random uplink channels; a separate fixed downlink channel |
| Payload | 20 application bytes plus 13 modeled uplink framing bytes and optional request |
| LoRa modulation | BW125 kHz, CR4/5, explicit header, eight-symbol preamble, low-rate optimization at SF11–12 |
| CRC | Payload CRC on uplinks, off on downlinks; modeled MIC/framing remains |
| Downlink | RX2 after two seconds, SF12, gateway power 14 dBm; 5% extra independent erasure |
| Duty factors | 1% device uplink, 10% gateway downlink, minimum off-time; configured assumptions, not universal regional limits |
| Propagation | Path loss32.4 +32log10(distance in metres), fixed 3 dB shadowing, independent1.5 dB fading |
| Receiver | SF thresholds and imperfect-orthogonality SIR matrix inherited from pilot; approximate8-demodulator cap |
| Device energy | 3.3 V; TX20/22/25/29/35 mA; RX10.8 mA; sleep1.5 microamps |

There is no queue or retry: generation during duty off-time or listening is dropped. Every admitted attempt pays for RX1/RX2 preamble-detection windows; additional listening when a report is transmitted is charged even if the report is erased. The full downlink duration determines gateway duty release and half-duplex uplink blocking. Background sleep is charged over the horizon in addition to active currents, slightly overcounting leakage during active periods. Processing, gateway electrical and backhaul energy are excluded.

Packets admitted before the horizon are fully charged and allowed to finish just after it; no report may start or finish at/after the horizon. Counts are attributed to the packet-generation phase. `post_horizon_finishes` makes the drain explicit. The experiment is 6 h 40 min of simulated time per run, not desktop runtime.

## Reading the metrics and plots

- **Application delivery** = received/generated. This counts the impact of duty drops and is the main reliability view.
- **PDR per attempt** = received/attempted. A high PDR alone can hide generated data that was never transmitted.
- **Energy per delivered packet** = total modeled end-device TX+RX+sleep joules / received application frames.
- **Downlink airtime** includes every sent report, including erased reports. The first figure normalizes it per 1,000 generated packets so density comparisons retain a common offered-traffic denominator. Actual total seconds and report counts remain in the tables.
- **Label delay** runs from a packet's transmission start to the valid report resolving its outcome. It is not a report-gap or an observation-age distribution. Unresolved outcomes are reported separately, not assigned artificial delays.

Means are across independent evaluation seeds. Confidence intervals are percentile bootstrap intervals from 10,000 seed resamples. Paired differences resample within-seed contrasts. Individual packets are not treated as independent replications. Intervals are pointwise/descriptive, without multiple-comparison adjustment or a predeclared noninferiority margin.

`paired_differences.csv` includes B−A (compression with requests), C−A (request removal with rich feedback), D−A (combined), D−B, D−C, and the interaction D−C−B+A. Frozen learner comparisons use the same feedback condition. Whole-run ratios use pooled packet/energy counts within each run; means of phase ratios are not substituted. The two primary learner strata and all unfavorable runs remain visible.

At SF12 the compact payloads all occupy the same symbol plateau:14–16 PHY bytes take1.155072 s versus1.810432 s for35 bytes. This is 36.2% less per report. **Total network savings need not equal36.2%** because shorter reports can enable more reports and alter feedback, decisions and duty eligibility. Do not equate airtime reduction with battery-life improvement.

## Run and reproduce

Recorded environment: Python 3.14.7, NumPy 2.3.5, Matplotlib 3.11.2. Dependencies are isolated in `.venv/` locally and excluded from the portable ZIP. Use Python 3.14 with the exact lock file for the recorded dependency versions.

From this package directory in PowerShell:

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null
$env:TMPDIR = (Resolve-Path .tmp).Path
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe reproduce.py
```

The final command runs tests, the entire frozen experiment, checked local online confirmation and all four plots. It writes `reproduced_results/` and `reproduced_figures/`, preserving the supplied `results/` and `figures/`. Byte hashes of newly generated raw files differ because runtime measurements differ; scientific metrics are deterministic for the recorded NumPy version, seeds and code.

For a small runnable demonstration:

```powershell
.\.venv\Scripts\python.exe reproduce.py --quick
```

This uses 8 devices, two seeds, one training episode and shorter phases. It is for code demonstrations and must not replace the thirty-seed results in the presentation.

To regenerate the supplied figures from saved raw data without simulating:

```powershell
.\.venv\Scripts\python.exe analyze.py
```

The analysis refuses mismatched source hashes, incomplete seed pairing, changed frozen tables or undefined primary denominators. To run stages independently, use `run_experiment.py --stage frozen`, inspect the checks, then `run_experiment.py --stage online --append`. Appending requires exactly matching settings and sources and rejects duplicate online evaluations.

## Package contents and limits

`engine.py` models radio events and reports; `policies.py` contains Q-learning; `feedback_codec.py` encodes reports; `run_experiment.py` trains/runs/saves; `analyze.py` audits/statistically summarizes/plots; `reproduce.py` is the one-command entry point. The tests cover reference airtime/energy, truth separation, unknown feedback, deduplication, SNR preservation, frozen policies, pairing and statistical aggregation. `results/raw.json` retains every evaluation and training record; CSVs provide spreadsheet-friendly views. `MANIFEST.json` records delivered file hashes; `verify.py` checks them.

The public DSF2R Raspberry Pi application receives measurements over MQTT/IP, unlike our modeled LoRa reports. Its released code/weights and unavailable training details have documented discrepancies; `reference/PROTOTYPE_AUDIT.md` preserves that audit. Comparisons here concern a matched reconstruction and proposed feedback adaptation, not a demonstrated improvement over the published implementation.

The next validation step is to cross-check receiver/accounting behavior and port the selected comparison to a suitable ns-3 LoRaWAN configuration or measured testbed. More seeds improve sampling precision, not the accuracy of uncalibrated radio assumptions. This package establishes a reproducible synthetic experiment; it makes no claim of a new adaptive reporting algorithm or guaranteed reliability.

Reference materials: [LoRaWAN L2 1.0.4](https://lora-alliance.org/wp-content/uploads/2021/11/LoRaWAN-Link-Layer-Specification-v1.0.4.pdf), [RP002-1.0.3 p.86](https://lora-alliance.org/wp-content/uploads/2021/05/RP002-1.0.3-FINAL-1.pdf), [Semtech airtime implementation](https://github.com/Lora-net/LoRaMac-node/blob/master/src/radio/sx1276/sx1276.c), [DSF2R2025](https://doi.org/10.1109/IWCMC65282.2025.11059434), [delayed-feedback bandit code/data](https://zenodo.org/records/7876244).
