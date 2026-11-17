# Limitations and unresolved questions

## Physical validity

Propagation, fading, currents, sensitivity, capture and demodulator limits are uncalibrated assumptions in an independently written simulator. Passing reference and metamorphic tests checks implementation consistency; it does not validate the receiver against measurements. An ns-3 cross-check or hardware experiment remains necessary for deployment claims. Increased seed count cannot remove model bias.

## Baseline fidelity

The original DSF2R policy is reconstructed. Released application/table discrepancies and absent training details are documented in [the source audit](../experiments/feedback/reference/PROTOTYPE_AUDIT.md). Its Raspberry Pi application obtains observations through MQTT/IP, whereas these experiments model LoRa feedback. The Thompson reference adapts the bandit family to thirty actions and common reports; it is not a reproduction of the six-SF, fixed-power Navas implementation. The archived MIX-MAB comparator also changes action/feedback assumptions. A fixed SF12/14 dBm control is a conservative reference, not optimized ADR.

## Information quality

Recent resolved outcomes can be stale and selected by feedback availability. The model cannot infer an uplink outcome from absent feedback. Severe outages and changes before feedback delivery can leave all policies poorly informed. Compact latest-SNR feedback is sufficient for the energy reward and frozen state construction, but not for original SNR-dependent online training.

## Statistical and computational scope

The finite scenario matrix is explicit and includes negative cases. It does not cover every device population, traffic distribution, fading process, regional channel plan or failure mode. Reported intervals summarize variation across the chosen seeds and are not simultaneous guarantees. Hyperparameters and model assumptions require independent validation; the prior three training episodes do not establish convergence.

Evaluation energy excludes CPU processing, gateway electrical energy and backhaul. Training radio costs are recorded separately, without a battery-lifetime amortization claim. The slight active/sleep leakage overlap and full listening charge for erased reports are conservative approximations.

## Interpretation

The current controlled evidence supports feedback-airtime and fixed-table energy reductions within a learner. It also shows an allocation delivery–energy trade-off. Online energy intervals include zero, delivery equivalence has not been established, and the time curves do not prove faster recovery. Joint SF/power adaptation, receipt compression and sparse feedback already exist in prior work. Their combination here is an evaluated extension, not an established first.

The additional matched-action online Thompson comparator has higher application-delivery means than either Q variant in all six tested protocol D scenarios. In the channel-step case, it also has lower energy per delivered packet, with pointwise paired intervals excluding zero for the combined-Q comparison. This is a measured weakness of the present Q configuration; it must remain visible in any paper or presentation. Training exposure, state coverage and the reward/credit structure need investigation before proposing the combined Q learner as a performance improvement over that reference.
