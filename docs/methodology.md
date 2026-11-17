# Methodology

## Separation of questions

The allocation question compares the reconstructed original state/reward, delivery-history-only, energy-reward-only and combined variants under the same feedback protocol. The feedback question holds a learner fixed and changes report representation and request bytes. Online comparisons retain independent per-device policy copies and charge protocol-specific training separately.

The preserved study uses original/reward/combined policies. The scenario robustness study adds the missing state-only ablation. The online baseline study adds a stateless Thompson-sampling reference with the same 30 actions, explicitly adapted from the delayed-feedback bandit family rather than presented as an exact paper reproduction.

## Radio and event model

The engine is a custom discrete-event model with one fixed gateway, stationary devices, SF7–12, powers 6/8/10/12/14 dBm, three uplink channels, BW125 kHz and CR4/5. Packet duration follows the Semtech LoRa expression, with explicit header, eight-symbol preamble and low-data-rate optimization at SF11–12. Payload CRC is on for uplinks and off for downlinks. Reports use SF12 RX2 after the triggering uplink.

The assumed log-distance model is `32.4 + 32 log10(distance_m)`, with fixed 3 dB shadowing and independent 1.5 dB packet fading. Capture uses a declared inter-SF SIR matrix and conservative overlap logic. The eight-demodulator approximation and half-duplex gateway are explicit simplifications. The configured uplink/downlink duty factors are 1%/10%; they are experimental assumptions, not universal regulatory values.

Generated packets during off-time or receive-window occupancy are dropped. There is no queue or retransmission (`NbTrans=1`). An admitted packet can finish just after the horizon; complete attempt cost and outcome are assigned to its generation phase. No report starts or finishes after the horizon.

## Information and reward

Receiver success and SNR remain at the server until serialized feedback is delivered. Device records retain their actual decision state, action, next observed decision state and local energy. Delayed information never rewrites a past decision state. Unknown labels are recovered through later overlapping windows or expire as unknown; labels are applied once.

The delivery-history feature uses the last twenty **resolved** outcomes. It is not the true PDR of every recent attempt. Outcome selection and staleness remain relevant when reports are lost; the outage cases expose this limitation.

The original reconstructed reward depends on SF and SNR margin. Successful frames need their historical quantized SNR. Failed frames use the SNR available when their action was chosen, with zero reward if unknown. This proxy is a declared assumption because the authors' executable training path was unavailable. Compact reports cannot preserve that original online reward; original/state training therefore uses rich feedback.

Energy-reward variants use `received − 0.2 × attempt_energy/E_REF`, with fixed E_REF=0.36383221248 J. Device voltage is 3.3 V; modeled TX currents are 20/22/25/29/35 mA, RX is 10.8 mA and sleep is 1.5 microamps. Every uplink pays for receive-window detection; a sent report pays its longer listening duration even when erased. Background leakage slightly overlaps active-time accounting. Processing, gateway and backhaul electrical costs are excluded.

## Feedback factorial

All four conditions use an eight-attempt cadence and sixteen-frame receipt window:

| Condition | Report | Request bytes |
|---|---|---:|
| A | Rich | 4 |
| B | Compact | 4 |
| C | Rich | 0 |
| D | Compact | 0 |

The rich application payload has a four-byte counter, two-byte receipt bitmap and sixteen half-dB SNR slots; with framing it is 35 bytes. The compact payload is one to three bytes plus 13 framing bytes. It binds its implicit anchor to the uplink that opened the current receive windows; unavailable reports are dropped rather than queued. These are proposed application payloads, not standardized LoRaWAN MAC commands. The full LoRaWAN encryption and framing bits are not implemented.

## Experimental units and interpretation

Each condition receives the same generated topology, traffic, channel selection and fading for a given seed and node count. Actual admitted transmissions can differ through airtime and duty constraints. Training and evaluation environment seeds are disjoint. The thirty seeds are independent experimental units; multiple policies and cases on one seed are paired rather than counted as additional independent replications.

The studies deliberately reuse seed IDs, common priors and some load-step cases. Counts of evaluation runs or training episodes describe executed invocations, not that many statistically independent samples. Overlapping cases are not pooled as extra evidence, and intervals in each study retain thirty paired seeds rather than treating repeated policies or archived reruns as independent observations.

Means, paired differences and percentile bootstrap intervals are descriptive and pointwise. Multiple comparisons are visible, with no familywise-significance claim. The studies are exploratory; no formal reliability-equivalence margin or universal energy guarantee was preregistered. Strong claims require validated radio behavior and independent external reproduction.
