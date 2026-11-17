# Matched online baseline results

Executed synthetic results: all three models learn locally during evaluation and use the same30 SF–power options and costed feedback protocols. Thompson uses Beta(1,1) priors and binary reception observations; the Q learners use delivery/energy reward. The Thompson reference is an adaptation, not an exact Navas energy-aware policy or paper replication.

The complete matrix contains **1080 evaluations**, **540 training episodes**, and **30 paired evaluation seeds** at **60 devices**. All six declared cases are retained when using the default configuration.

## Compact request-free deployment, condition D

| Case | Online model | Application delivery | PDR/attempt | Energy (mJ/delivery) | DL airtime (s) |
|---|---|---:|---:|---:|---:|
| stable_low | Delayed binary Thompson | 94.02% | 94.07% | 91.97 | 180.92 |
| stable_low | Energy-reward Q | 78.42% | 78.48% | 95.17 | 170.68 |
| stable_low | Delivery-state + energy Q | 77.32% | 77.39% | 95.00 | 169.76 |
| stable_high | Delayed binary Thompson | 83.24% | 88.59% | 96.70 | 572.65 |
| stable_high | Energy-reward Q | 65.30% | 72.16% | 89.49 | 533.64 |
| stable_high | Delivery-state + energy Q | 64.39% | 71.42% | 90.95 | 531.60 |
| load_step | Delayed binary Thompson | 88.36% | 91.39% | 94.25 | 327.23 |
| load_step | Energy-reward Q | 72.04% | 75.68% | 90.62 | 319.76 |
| load_step | Delivery-state + energy Q | 71.04% | 74.57% | 91.06 | 319.03 |
| channel_step | Delayed binary Thompson | 72.87% | 75.40% | 113.88 | 285.73 |
| channel_step | Energy-reward Q | 49.26% | 52.15% | 135.79 | 231.67 |
| channel_step | Delivery-state + energy Q | 48.30% | 50.92% | 136.67 | 228.01 |
| lossy_feedback | Delayed binary Thompson | 88.38% | 91.41% | 94.19 | 327.23 |
| lossy_feedback | Energy-reward Q | 72.52% | 76.56% | 92.68 | 320.57 |
| lossy_feedback | Delivery-state + energy Q | 71.34% | 75.24% | 93.01 | 318.38 |
| no_feedback | Delayed binary Thompson | 88.32% | 91.35% | 94.27 | 327.66 |
| no_feedback | Energy-reward Q | 73.08% | 78.80% | 112.60 | 279.95 |
| no_feedback | Delivery-state + energy Q | 73.14% | 78.56% | 110.20 | 281.76 |

## Paired feedback effects, D minus A

| Case | Online model | Delivery change (percentage points) | Energy change (mJ/delivery) |
|---|---|---:|---:|
| stable_low | Delayed binary Thompson | -0.06 [-0.59, +0.53] | -5.17 [-14.65, +4.38] |
| stable_low | Energy-reward Q | +0.88 [-2.08, +3.64] | +0.65 [-10.64, +11.40] |
| stable_low | Delivery-state + energy Q | -0.03 [-2.57, +2.44] | -6.74 [-20.83, +6.46] |
| stable_high | Delayed binary Thompson | +0.58 [-2.59, +3.61] | -6.28 [-16.26, +3.77] |
| stable_high | Energy-reward Q | +0.99 [-1.08, +3.05] | -1.70 [-7.55, +3.89] |
| stable_high | Delivery-state + energy Q | -0.04 [-2.52, +2.47] | -2.82 [-11.05, +4.93] |
| load_step | Delayed binary Thompson | +0.34 [-1.56, +2.24] | -5.99 [-15.72, +3.99] |
| load_step | Energy-reward Q | +1.27 [-0.85, +3.47] | -1.52 [-10.23, +7.22] |
| load_step | Delivery-state + energy Q | +0.67 [-1.72, +2.99] | -6.70 [-18.44, +4.32] |
| channel_step | Delayed binary Thompson | +0.05 [-1.71, +1.79] | -6.79 [-17.43, +4.35] |
| channel_step | Energy-reward Q | +0.24 [-1.82, +2.42] | -0.39 [-15.38, +13.88] |
| channel_step | Delivery-state + energy Q | -0.65 [-2.61, +1.29] | -5.28 [-22.26, +11.21] |
| lossy_feedback | Delayed binary Thompson | +0.37 [-1.54, +2.26] | -6.05 [-15.83, +3.94] |
| lossy_feedback | Energy-reward Q | +1.04 [-1.30, +3.50] | -0.85 [-11.92, +9.67] |
| lossy_feedback | Delivery-state + energy Q | +0.13 [-2.44, +2.77] | -6.02 [-19.05, +6.26] |
| no_feedback | Delayed binary Thompson | +0.42 [-1.45, +2.24] | -6.15 [-16.10, +3.85] |
| no_feedback | Energy-reward Q | +0.90 [-3.29, +5.15] | +6.49 [-19.53, +32.24] |
| no_feedback | Delivery-state + energy Q | +2.12 [-2.08, +6.42] | -11.53 [-39.51, +13.92] |

## Interpretation

- All intervals are pointwise95% paired seed-bootstrap intervals, without multiple-comparison adjustment. An interval crossing zero is inconclusive, not evidence of equivalence.
- Initial tables/posteriors are reused across scenarios; each device receives an independent local copy. There is no free cross-device update sharing during deployment.
- Training uses the normal load-step case separately for A/D. Therefore D−A includes training-protocol and deployment effects. Training radio energy is reported separately.
- In the all-feedback-erased negative control, no learning outcome reaches any model. Sent reports still consume gateway airtime and receive energy.
- Thompson and Q have different optimization objectives. Read delivery and energy together, including cases where the Q extension loses to the bandit. Do not select only favorable cases or infer universal superiority.
- The reconstructed DSF2R and state-only ablation are evaluated in the separate frozen robustness study. No frozen-versus-online comparison here is labelled a causal algorithm improvement.
- Receiver calibration, independent ns-3 validation and hardware measurements remain outside this synthetic study.

Every model/protocol/phase and paired Q−Thompson contrast is in the CSV files; compressed raw checkpoints and trained priors are retained.
