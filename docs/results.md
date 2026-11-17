# Research results and reasoning

## What was executed

The current package contains three studies:1,320 controlled-feedback evaluations,9,720 frozen robustness evaluations and 1,080 matched online-baseline evaluations. Their training records contain1,530,1,080 and 540 episodes respectively. Each comparison uses thirty paired evaluation seeds. Shared cases and seed IDs remain paired and are not pooled as extra independent evidence.

The 83 automated checks cover the simulator/codec, statistical analysis, archived-data contracts, scenario transformations, fixed controls and Thompson posterior updates. Full-data audits verify the declared matrices, raw checkpoint/source hashes, saved initial-policy fingerprints and accounting. Representative saved-prior replays match the recorded Windows results. Eleven PNG/SVG figures and complete per-seed CSVs accompany the compressed raw checkpoints.

## Allocation: more delivery can cost more energy

At60 devices under rich reports with request bytes, the combined delivery-state/energy learner delivers 71.04% of generated packets at 97.52 mJ per delivery. The reconstructed original delivers 64.92% at 72.36 mJ. The paired difference is +6.12 percentage points in delivery [3.46,8.81], accompanied by +25.16 mJ/delivery [14.15,37.48].

This supports an altered delivery–energy trade-off within the reconstruction. It does not show that adding delivery history alone is responsible, nor that the extension dominates the original. The expanded study includes the state-only and reward-only variants so that the two changes remain distinguishable. [Controlled results](../experiments/feedback/results/RESULTS.md) · [Robustness model comparison](../experiments/robustness/figures/02_model_comparison.png).

## Feedback: a supported cost reduction within a learner

For the same combined learner at 60 devices, compact request-free feedback reduces mean energy from 97.52 to 91.37 mJ/delivery and downlink airtime from 345.49 to 274.37 s. The paired differences are−6.15 mJ [−9.73,−2.92] and −71.12 s [−81.51,−60.48]. Application delivery changes by −0.13 percentage points [−1.03,+0.78].

The reason is concrete: the device already knows its actions and local cost, and the energy reward does not need every historical successful-packet SNR. A compact receipt window retains individual action outcomes and the latest state bin. Shorter reports reduce receive time and gateway blocking; eliminating request bytes reduces uplink duration. Actual network effects include changed duty eligibility and report counts, so the total downlink reduction is 20.59%, rather than the analytical 36.2% per-report reduction.

In the six-case frozen robustness matrix, every combined-learner D−A downlink-airtime interval is below zero across 18 case/density cells. Energy intervals are below zero in 15 cells. These are pointwise intervals without multiple-comparison adjustment; they are not a simultaneous claim that every deployment benefits.

Reliability penalties remain. In the channel-step case at 60 devices, application delivery falls by 0.71 percentage points [−1.31,−0.08], alongside an energy reduction of 4.47 mJ/delivery [−7.96,−1.15]. The load-step/120-device and channel-step/120-device comparisons also have negative delivery intervals. The plots retain these cases rather than treating lower cost as sufficient evidence of success. [All feedback contrasts](../experiments/robustness/results/paired_differences.csv).

## Online comparison: the present Q allocator is not the strongest reference

All online models use independent local copies, the same30 SF–power actions and costed A/D feedback. The Thompson comparator learns binary reception probability; Q includes an energy penalty and uses larger state spaces. It is an explicit adaptation, not a faithful Navas reproduction.

Under load-step/protocol D at 60 devices, Thompson delivers 88.36% of generated packets at 94.25 mJ/delivery; combined Q delivers 71.04% at 91.06 mJ. The paired Q−Thompson delivery difference is −17.32 percentage points [−19.49,−15.12], while the energy difference is −3.18 mJ [−12.64,+5.88]. A small mean energy advantage therefore does not establish a better overall allocation result.

Under the channel step, combined Q has 24.57 percentage points lower delivery [−26.52,−22.67] and 22.79 mJ higher energy per delivery [10.06,35.05]. This is a counterexample to uniform superiority. Thompson has higher delivery means than both Q variants in all six tested D cases. Fixed and reconstructed references remain in the separate frozen study; they are not conflated with the online learners. [All online results](../experiments/baselines/results/RESULTS.md) · [Q−Thompson contrasts](../experiments/baselines/figures/03_q_vs_thompson.png).

The large state/action table and limited training exposure are plausible factors to investigate, not established explanations from this experiment. Weight tuning, longer training and state/credit sensitivity require separate validation seeds rather than selection on these results.

## The gap addressed and what remains

The implementation makes delivery information explicit in a chosen SNR-based allocation reconstruction and measures the delay, loss and radio cost of obtaining that information. It demonstrates a compact-feedback cost reduction and exposes allocation/reliability trade-offs. This is the defensible operational contribution of the current package.

It does not establish a first in the literature, a universally improved allocator, equal delivery, a recovery-time advantage or hardware battery savings. Receiver calibration, independent ns-3/testbed checks, faithful additional baselines and convergence/sensitivity studies remain open. [Evidence scope](evidence_scope.md) · [Related work](related_work.md) · [Limitations](limitations.md).
