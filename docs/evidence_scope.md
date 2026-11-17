# Evidence, claims and limits

The research question is whether explicit delivery/energy information and a less expensive feedback protocol improve the trade-off achieved by an SF–power allocator. This document identifies which parts are implemented, which comparisons support a claim, and where the evidence remains insufficient.

The simulator is independently implemented in Python/NumPy. Its propagation, receiver and device-current assumptions are declared rather than measured calibration data. A test passing validates the checked behavior, not the physical model as a whole.

## Preserved controlled experiment

The [feedback experiment](../experiments/feedback/) contains 1,320 evaluation runs and 1,530 training episodes, with 30 paired seeds per comparison. Frozen evaluations use 30, 60 and 120 devices; local online confirmation uses 60 devices. Every condition uses an eight-attempt reporting interval and a 16-frame receipt window.

| Condition | Report format | Extra request bytes |
|---|---|---:|
| A | Rich | 4 |
| B | Compact | 4 |
| C | Rich | 0 |
| D | Compact | 0 |

Within a frozen learner, all conditions receive identical pretrained Q-tables. Generated topology, traffic, channel choices and fading are paired. Actual transmission attempts can differ because packet duration changes duty eligibility. Online training is matched to each protocol; those comparisons consequently include both training and deployment effects.

Full means, individual runs and paired 95% bootstrap intervals are in [the measured results](../experiments/feedback/results/RESULTS.md), [per-seed data](../experiments/feedback/results/per_seed.csv) and [paired contrasts](../experiments/feedback/results/paired_differences.csv). Independent seeds, not individual packets, are the statistical units. Intervals are pointwise/descriptive and are not adjusted for multiple comparisons.

## Claim-to-evidence matrix

| Proposed claim | Evidence and appropriate interpretation | Boundary |
|---|---|---|
| The extended allocator explicitly uses delivery and energy information. | Code implements delivery-history state and `received - 0.2 E/E_REF`; tests verify that reported labels and local cost enter the reward. | This is an implementation fact. It does not by itself establish novelty or better performance. |
| The combined allocation extension can improve delivery in the declared model. | At 60 devices under the same rich/request condition A, combined versus original reconstruction increased mean application delivery by 6.12 percentage points; paired interval [3.46, 8.81]. | Energy per delivery also increased by 25.16 mJ [14.15, 37.48]. This contrast changes state and reward together; it does not isolate the delivery-state contribution. |
| Compact, request-free feedback reduces the combined learner's controlled radio cost. | At 60 devices with fixed tables, D versus A reduced mean energy from 97.52 to 91.37 mJ/delivery, a 6.31% ratio-of-means reduction. Paired energy difference: −6.15 mJ [−9.73, −2.92]. Total mean downlink airtime fell from 345.49 to 274.37 s, or 20.59%. | The clean claim is within the same learner and scenario. It is not a gain over the published DSF2R implementation. Different conditions and seed-level outcomes remain in the data. |
| Delivery is unchanged after feedback optimization. | Mean application delivery changed by −0.13 percentage points, with interval [−1.03, 0.78]. Attempt PDR changed by −0.71 points [−1.46, 0.07]. | **Not established.** An interval spanning zero does not prove equivalence or rule out a delivery penalty. There is no predeclared noninferiority margin. |
| The same energy improvement is established during local online learning. | For the 60-device combined learner, mean D−A energy was −6.70 mJ/delivery, with interval [−18.55, 4.29]. | **Not established.** The estimate is uncertain. Report the full interval and the different protocol-specific training tables. |
| A smaller report produces a fixed 36.2% total network saving. | Under the declared SF12 radio settings, a 35-byte report takes 1.810432 s and a 14–16-byte report takes 1.155072 s: 36.2% less per report. | **Not established for total network airtime.** Shorter reports permit more reports and change information, actions and gateway availability. Use measured total airtime and report counts. |
| Missing feedback is handled without fabricating failures. | Device records obtain outcomes only from decoded reports; tests cover complete report loss, later recovery, overlap deduplication and expiry as unknown. | This is a checked protocol property under `NbTrans=1` and the declared fresh-window contract. It is not a novel result about LoRaWAN generally. |
| The new method is uniformly better than existing models. | The matched reconstruction, reward-only and combined learners expose different delivery–energy trade-offs. | **Not established.** The original reconstruction uses less mean energy in some comparisons; online and per-seed outcomes are mixed. Other literature methods need faithful or explicitly adapted reproductions. |
| The method recovers faster after a change. | Time curves show behavior during light/heavy/light traffic. | **Not established by the preserved feedback study.** A recovery definition, sufficient observation horizon and common quality target are required. |

## The precise gap addressed by the implementation

The project makes a specified local allocation pipeline consume packet-reception evidence and charge for obtaining it. This addresses two operational omissions in the chosen SNR-state reconstruction: delivery history is not an explicit state coordinate, and its printed reward does not explicitly contain a received-packet indicator or energy in joules. The implemented report protocol also makes delay, loss and airtime costs observable in the evaluation.

This is a narrower statement than claiming that existing research ignores reliability, energy, congestion or feedback. [Related work](related_work.md) documents methods that already address those topics. The present evidence supports a controlled feedback-cost result and a measured allocation trade-off; it does not establish a universal literature gap or a theoretical guarantee.

## Additional cases and validation status

The completed [online baseline study](../experiments/baselines/results/RESULTS.md) adds 1,080 evaluations and 540 protocol-matched training episodes across all six scenarios at 60 devices. It compares two Q variants against a delayed binary Thompson-sampling adaptation over the same 30 actions and the same A/D feedback protocols. It does not reproduce the Navas energy-aware reward or full ns-3 experiment.

This broader baseline is a substantive limitation of the current allocator. Under protocol D in the load-step case, Thompson delivers 88.36% of generated packets and the combined Q learner delivers 71.04%. The paired combined-Q−Thompson difference is −17.32 percentage points [−19.49,−15.12], while its energy difference is −3.18 mJ/delivery [−12.64,+5.88]. Under channel attenuation, combined Q delivers 24.57 points fewer [−26.52,−22.67] and uses 22.79 mJ/delivery more [10.06,35.05]. These findings rule out presenting the Q extension as the strongest tested online allocator. The controlled feedback-cost claim remains a separate within-learner result.

The completed [frozen robustness study](../experiments/robustness/results/RESULTS.md) contains 9,720 evaluations and 1,080 training episodes, including four state/reward variants and two fixed-radio controls. For the combined learner's D−A comparison, all 18 case/density cells have lower mean downlink airtime with pointwise95% intervals below zero;15 of 18 have energy intervals below zero. This supports a broad feedback-cost result within the model, not a simultaneous guarantee across deployments.

Delivery is not uniformly preserved. The load-step/120-device and channel-step/60- and 120-device cells have negative delivery intervals. For example, channel-step/60-device delivery changes by −0.71 percentage points [−1.31,−0.08], while energy changes by −4.47 mJ/delivery [−7.96,−1.15]. These are cost–reliability trade-offs, and every unfavorable case remains visible. Complete-feedback-erasure is a negative control: no new outcomes are learned even when pretrained settings continue to deliver packets.

Each study's outputs use its own recorded configuration and bootstrap resamples; the preserved feedback study's exact confidence bounds are not reused for the expanded cases. The additional state-only ablation separates its information change from the energy-reward change. Fixed N receives no gratuitous learning-feedback cost; a faithful conventional ADR implementation remains outside scope.

An experiment matrix is a bounded specification. “All cases” means every declared case and test, including unfavorable outcomes; it cannot mean every physical deployment, possible failure sequence or unpublished implementation. Explicit coverage and missing coverage are more informative than a blanket completeness claim.

Before making publication-level external performance claims, the remaining work includes receiver/accounting cross-checks against a suitable ns-3 configuration or measurements, propagation/current calibration, baseline fidelity review, and sensitivity to training exposure and reward weight. Processing and gateway electrical energy are outside the present device-radio metric. Frozen evaluation and local online learning are separate protocols, not interchangeable evidence of adaptation.

## Reading the figures

- Application delivery is received/generated; PDR is received/attempted. Duty-related admission differences can change the two denominators differently.
- Energy per delivery includes modeled device TX, RX and sleep during evaluation. Simulated training expenditure is separate; desktop CPU training energy is not measured.
- Normalized downlink airtime is seconds per 1,000 generated packets. A lower per-packet value at higher density can result from a shared gateway bottleneck, not better service per device.
- Bands on individual means and intervals for paired differences answer different questions. Overlapping mean bands do not invalidate a more precise paired contrast.
- Report counts, unresolved outcomes and label delay accompany the main metrics. Label delay is not the age of the last observation at every decision.
