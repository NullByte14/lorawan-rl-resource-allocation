# Frozen-policy robustness results

Executed 9,720 evaluations and 1,080 training episodes across 30 paired seeds per comparison. All planned cases and feedback conditions are included.

This is the declared Python receiver/energy model, not ns-3 or hardware evidence. Policies are pretrained once on the load-step case and frozen across stress cases; settings still respond to observed states. Both fixed controls use SF12/14 dBm: one pays matched protocol A, the other sends no learning feedback (N). No faithful ADR implementation is claimed.

## Combined learner, D minus A over all phases

Negative energy/downlink differences favor D. Positive delivery differences favor D. Intervals spanning zero do not establish improvement or equivalence.

| Case | Devices | Delivery change (pp) | Energy change (mJ/delivery) | DL change (s/1,000 generated) |
|---|---:|---:|---:|---:|
| stable_low | 30 | -0.18 [-0.78, +0.42] | -4.12 [-5.64, -2.58] | -20.22 [-21.41, -18.95] |
| stable_low | 60 | -0.65 [-1.34, +0.04] | -4.59 [-7.27, -2.15] | -9.42 [-10.57, -8.27] |
| stable_low | 120 | -0.36 [-0.87, +0.13] | -0.81 [-2.94, +1.13] | -3.97 [-4.45, -3.51] |
| stable_high | 30 | -0.70 [-1.57, +0.09] | -0.95 [-2.62, +0.71] | -23.03 [-24.96, -21.00] |
| stable_high | 60 | -0.45 [-1.33, +0.42] | -5.30 [-9.25, -1.72] | -10.36 [-11.85, -8.88] |
| stable_high | 120 | -0.55 [-1.38, +0.33] | -4.22 [-7.71, -1.34] | -4.84 [-5.73, -3.98] |
| load_step | 30 | +0.41 [-0.23, +1.06] | -3.71 [-5.11, -2.35] | -25.44 [-28.15, -22.98] |
| load_step | 60 | -0.13 [-1.02, +0.76] | -6.15 [-9.71, -2.91] | -13.17 [-15.10, -11.27] |
| load_step | 120 | -0.66 [-1.29, -0.02] | -3.02 [-5.50, -0.84] | -4.99 [-6.17, -3.87] |
| channel_step | 30 | -0.19 [-0.77, +0.46] | -3.44 [-5.38, -1.51] | -17.38 [-19.13, -15.58] |
| channel_step | 60 | -0.71 [-1.31, -0.08] | -4.47 [-7.96, -1.15] | -10.98 [-12.42, -9.50] |
| channel_step | 120 | -0.91 [-1.58, -0.28] | -2.04 [-4.90, +0.60] | -5.12 [-6.12, -4.07] |
| lossy_feedback | 30 | +0.23 [-0.32, +0.76] | -4.75 [-6.55, -3.11] | -24.25 [-26.32, -22.26] |
| lossy_feedback | 60 | -0.22 [-0.96, +0.53] | -4.74 [-7.53, -2.10] | -11.65 [-13.33, -10.01] |
| lossy_feedback | 120 | -0.37 [-0.99, +0.19] | -3.13 [-5.14, -1.26] | -3.67 [-4.62, -2.76] |
| no_feedback | 30 | +0.08 [-0.03, +0.19] | -1.75 [-2.10, -1.44] | -14.45 [-15.96, -12.97] |
| no_feedback | 60 | +0.04 [-0.06, +0.14] | -1.58 [-1.90, -1.26] | -6.20 [-6.94, -5.45] |
| no_feedback | 120 | +0.11 [+0.03, +0.18] | -1.27 [-1.69, -0.90] | -2.97 [-3.27, -2.68] |

## Interpretation boundaries

- The all-feedback-erased case is a negative control: no receipts or SNR reach the device, though sent reports still consume airtime and receive energy.
- Channel attenuation changes both directions from the second phase; the per-frame downlink fade is attached to the originating uplink, including any boundary crossing.
- Frozen evaluations assess transfer of pretrained state-responsive policies, not online relearning after each change. Online baseline comparisons are a separate experiment.
- Confidence intervals are pointwise paired seed bootstrap intervals, without multiplicity correction. Report all cases rather than select significant cells.
- The table exposes successes, failures and inconclusive cases. It does not prove a universal research gap, optimality, or superiority to published implementations.
- Exact intervals and every model/condition/phase are in summary.csv and paired_differences.csv; individual runs remain in per_seed.csv and compressed raw checkpoints.
