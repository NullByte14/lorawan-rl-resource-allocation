# Figure guide

Every graph has a PNG for viewing and an SVG for vector editing. The exact estimates, sample counts and intervals are in the matching study's CSV files. All declared cases are retained; the graphs are not a selection of favorable seeds.

## Controlled feedback study

These figures isolate report compression and removal of explicit request bytes. A/B/C/D represent large+request, compact+request, large request-free and compact request-free. Reporting cadence and receipt-window length are unchanged.

| Figure | What it shows | How to interpret it |
|---|---|---|
| [Feedback airtime](../experiments/feedback/figures/01_feedback_airtime.png) | Frozen curves at 30/60/120 devices, plus online 60-device results; downlink seconds per 1,000 generated packets. | This includes lost reports. Total airtime and report counts are saved separately. Lower normalized airtime at higher density can reflect a gateway bottleneck, not improved per-device service. |
| [Energy per delivered packet](../experiments/feedback/figures/02_energy_per_delivery.png) | Modeled device energy divided by delivered application frames, in mJ. | Read alongside delivery. The gray original/A reconstruction is a context reference; within-learner A–D differences are the clean feedback contrasts. Online training is matched to each protocol, unlike common-prior frozen comparisons. |
| [Application delivery](../experiments/feedback/figures/03_application_delivery.png) | Received/generated frames, including the effect of duty-admission drops. | PDR per attempt is in the adjacent result tables. Small mean differences and overlapping intervals do not establish equivalence. |
| [Delivery over time](../experiments/feedback/figures/04_delivery_over_time.png) | Delivery in 800-second bins; the shaded middle phase has heavier traffic. | These are descriptive curves, not proof of faster recovery. Each point summarizes a time interval; plotted bands are pointwise. |

All four protocol arms appear for both the reward-only and combined learners. Frozen tables do not learn during evaluation, but their selected settings still change with observed state. Individual-mean bands and paired-difference intervals are different estimates: overlapping bands can coexist with a paired difference whose interval excludes zero.

## Frozen robustness study

The rows/cases include stable low load, stable high load, a load step, a channel attenuation step,30% feedback erasure and complete feedback erasure. The complete matrix uses thirty paired seeds at each of 30/60/120 devices. Policy priors are trained on the same normal load-step scenario and reused across cases; this tests transfer and feedback behavior, not online relearning.

| Figure | What it shows | How to interpret it |
|---|---|---|
| [Case/density effects](../experiments/robustness/figures/01_case_density_effects.png) | D−A delivery, energy and downlink-airtime differences for each Q variant, case and density. | Green indicates a favorable sign, not a guarantee. Cells print values; an asterisk means a pointwise paired interval excludes zero. Color limits differ by panel, so compare printed values rather than color intensity across metrics. |
| [Allocation model comparison](../experiments/robustness/figures/02_model_comparison.png) | Original reconstruction, reward-only, state-only, combined and two fixed-radio controls at 60 devices. | Q models and the matched fixed control use protocol A. Fixed N uses SF12/14 dBm without learning feedback, avoiding a gratuitous reporting penalty. Neither fixed control is ADR. |
| [Delivery–energy trade-offs](../experiments/robustness/figures/03_delivery_energy_tradeoffs.png) | All allocation variants and fixed controls in delivery-versus-energy space, with A/D/N markers. | Higher delivery and lower energy are preferable; no single ordering follows when one improves and the other worsens. This overview displays means; intervals remain in the tables. |
| [Separate factor effects](../experiments/robustness/figures/04_factor_effects.png) | Combined learner B−A, C−A and D−A, separated by case and density. | These distinguish compression, request removal and their combination. The interaction is also saved in the CSV rather than attributed to one factor. |

In the complete-feedback-erasure case no new receipt or SNR information reaches the device. Existing pretrained information remains available. Reports that are sent still consume airtime/receive energy. Successful delivery during that case must not be described as learning from absent outcomes.

## Matched online baseline study

All three models update locally during evaluation with independent device copies and protocol-matched priors. The Thompson reference is a thirty-action binary-success adaptation; the Q models have an energy penalty and different state spaces. Objective differences are explicit, and frozen/online methods are not mixed into a causal ranking.

| Figure | What it shows | How to interpret it |
|---|---|---|
| [Online delivery–energy comparison](../experiments/baselines/figures/01_online_delivery_energy.png) | All six cases at 60 devices with mean and95% intervals for delivery and energy, using A/D markers. | Thompson achieves higher delivery means than the current Q variants in the tested D cases. Under channel attenuation it also has lower energy per delivered packet. These counterexamples remain part of the study. |
| [Online feedback effects](../experiments/baselines/figures/02_online_feedback_effects.png) | Within-model D−A differences for delivery, energy and downlink airtime. | A confidence interval crossing zero is inconclusive. These differences include protocol effects during training and deployment, not only report-size changes at evaluation. |
| [Q versus Thompson](../experiments/baselines/figures/03_q_vs_thompson.png) | Paired Q−Thompson differences under matched protocols for all cases. | Positive delivery differences favor Q; negative energy differences favor Q. This graph makes losses to the reference visible rather than hiding them behind a favorable single metric. |

The uncertainty unit throughout is the independent seed, with 10,000 paired percentile-bootstrap resamples. Intervals are descriptive/pointwise, without multiple-comparison correction. A simulator-internal difference does not establish hardware accuracy or novelty relative to all prior work.
