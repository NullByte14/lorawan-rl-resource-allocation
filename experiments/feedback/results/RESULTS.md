# Measured feedback experiment results

These are executed synthetic Python-simulator results. They are not ns-3, hardware measurements, or an exact reproduction of published DSF2R. The original policy is a documented paper-based reconstruction.

Evaluated **1320 held-out runs**, **1530 training episodes**, and **30 paired seeds** per comparison. Frozen-table and trace/topology pairing checks passed. Training and evaluation environment seeds are disjoint.

## Complete evaluation at the main density

The following averages use 60 devices across all three traffic phases. Device energy includes modeled TX, RX and sleep; processing and gateway electrical energy are excluded. Training is separate in `training.csv`.

| Evaluation | Learner | Arm | Application delivery | PDR per attempt | mJ/delivery | DL seconds | Valid reports |
|---|---|---|---:|---:|---:|---:|---:|
| frozen | original | A | 64.92% | 66.51% | 72.36 | 285.14 | 149.1 |
| frozen | original | B | 64.37% | 65.94% | 72.26 | 232.32 | 190.9 |
| frozen | original | C | 64.92% | 66.53% | 72.02 | 285.57 | 149.0 |
| frozen | original | D | 64.40% | 65.97% | 71.58 | 231.94 | 190.6 |
| frozen | reward | A | 71.00% | 74.59% | 90.69 | 343.02 | 180.1 |
| frozen | reward | B | 70.97% | 74.20% | 86.71 | 279.18 | 229.1 |
| frozen | reward | C | 71.14% | 74.67% | 89.54 | 343.56 | 180.5 |
| frozen | reward | D | 71.06% | 74.22% | 85.77 | 279.80 | 229.9 |
| frozen | both | A | 71.04% | 75.11% | 97.52 | 345.49 | 181.2 |
| frozen | both | B | 70.75% | 74.28% | 92.24 | 273.44 | 224.3 |
| frozen | both | C | 71.11% | 75.16% | 96.38 | 344.71 | 180.6 |
| frozen | both | D | 70.91% | 74.40% | 91.37 | 274.37 | 225.5 |
| online | reward | A | 70.77% | 74.82% | 92.14 | 421.35 | 221.5 |
| online | reward | B | 72.38% | 76.01% | 94.33 | 321.61 | 264.8 |
| online | reward | C | 71.15% | 74.85% | 91.41 | 416.70 | 219.9 |
| online | reward | D | 72.04% | 75.68% | 90.62 | 319.76 | 263.3 |
| online | both | A | 70.37% | 74.77% | 97.76 | 414.11 | 217.1 |
| online | both | B | 71.04% | 74.79% | 89.25 | 319.30 | 262.1 |
| online | both | C | 70.68% | 74.58% | 92.89 | 413.26 | 217.3 |
| online | both | D | 71.04% | 74.57% | 91.06 | 319.03 | 262.6 |

A = rich report + 4 request bytes; B = compact + request; C = rich without request; D = compact without request. Cadence is every 8 attempts and the receipt window is 16. All frozen arms of one learner start with the same pretrained table. Online training is matched to each protocol, so online comparisons include both training-protocol and deployment effects.

## Paired changes: D minus A

Each confidence interval resamples matched evaluation seeds. Negative energy/airtime changes favor D; positive delivery changes favor D. A confidence interval spanning zero does not establish an improvement or equivalence. Intervals are descriptive, pointwise and not adjusted for multiple comparisons.

| Evaluation | Learner | Period | Application delivery change (percentage points) | Energy change (mJ/delivery) | DL airtime change (s) |
|---|---|---|---:|---:|---:|
| frozen | reward | all phases | +0.06 [-0.52, +0.65] | -4.92 [-8.06, -2.08] | -63.22 [-73.29, -53.13] |
| frozen | reward | high load | +0.62 [-0.25, +1.52] | -6.29 [-10.14, -2.72] | -23.01 [-29.23, -16.83] |
| frozen | both | all phases | -0.13 [-1.03, +0.78] | -6.15 [-9.73, -2.92] | -71.12 [-81.51, -60.48] |
| frozen | both | high load | +0.53 [-0.77, +1.84] | -8.05 [-12.71, -3.87] | -29.23 [-36.03, -22.56] |
| online | reward | all phases | +1.27 [-0.89, +3.44] | -1.52 [-10.32, +7.04] | -101.59 [-117.99, -84.01] |
| online | reward | high load | +1.96 [-0.37, +4.45] | -3.68 [-10.31, +2.73] | -47.97 [-60.25, -35.55] |
| online | both | all phases | +0.67 [-1.62, +2.95] | -6.70 [-18.55, +4.29] | -95.08 [-108.48, -82.47] |
| online | both | high load | +1.20 [-1.76, +4.27] | -6.94 [-16.94, +2.40] | -39.53 [-50.58, -29.07] |

## What the results support

- **frozen, reward:** D used 18.4% less mean gateway downlink airtime than A (ratio of arm means). Mean energy per delivery was lower, and its interval excludes zero; application delivery was higher, and its interval spans zero. D had lower energy per delivery in 20/30 paired runs.
- **frozen, both:** D used 20.6% less mean gateway downlink airtime than A (ratio of arm means). Mean energy per delivery was lower, and its interval excludes zero; application delivery was lower, and its interval spans zero. D had lower energy per delivery in 21/30 paired runs.
- **online, reward:** D used 24.1% less mean gateway downlink airtime than A (ratio of arm means). Mean energy per delivery was lower, and its interval spans zero; application delivery was higher, and its interval spans zero. D had lower energy per delivery in 14/30 paired runs.
- **online, both:** D used 23.0% less mean gateway downlink airtime than A (ratio of arm means). Mean energy per delivery was lower, and its interval spans zero; application delivery was higher, and its interval spans zero. D had lower energy per delivery in 16/30 paired runs.

The 36.2% analytical per-report saving is not automatically the total network saving: shorter reports change gateway availability, valid-report counts, device choices and duty eligibility. Results above charge those consequences. Byte compression within the 14–16-byte SF12 plateau has the same packet duration; request removal affects uplink airtime separately.

## Limits and interpretation

- Receiver/capture behavior, propagation and radio currents remain simplified and uncalibrated. No performance claim about the published DSF2R implementation follows from this reconstruction.
- More independent seeds reduce sampling uncertainty but do not validate physical assumptions. The seeds are the independent units, not individual packets.
- No new adaptive reporting policy was added; reporting cadence is held fixed. The plots do not prove a recovery-time advantage.
- Lower airtime alone is not proof of better application performance. Read application delivery and energy together, and keep unfavorable runs visible.
- Original-reward training retains historical packet SNR in rich feedback. Compact original evaluation is frozen and needs only the current SNR state.

## Files

- `per_seed.csv`: per-phase and whole-run metrics, including PDR, offered delivery, energy, report counts and unresolved labels. `mean_label_delay_s` is delay from transmission start to receipt of a valid outcome report; it is not time since the last report or observation age at every decision.
- `summary.csv`: means and percentile bootstrap intervals, with explicit total/valid sample counts.
- `paired_differences.csv`: compression, request-removal, combined and interaction contrasts, plus matched-arm learner comparisons.
- `training.csv`: separate training energy, updates and table hashes.
- `raw.json`: configuration, source hashes and individual evaluation/training records.
- `../figures/`: four PNG/SVG figures. Every figure includes the four planned feedback arms.
