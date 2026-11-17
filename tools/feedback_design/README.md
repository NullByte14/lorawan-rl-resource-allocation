# Reducing feedback airtime for the LoRaWAN allocation project

Prepared 16 November 2026. This is a proposed protocol, a working payload codec, and an analytical airtime calculation. It has not been integrated into the network pilot or tested in ns-3. The analytical design predates the integrated experiment; measured results are in `../../experiments/feedback/`.

The practical starting point is to remove explicit feedback requests, return a short receipt bitmap plus the latest SNR state, and then test longer reporting intervals. For our success-and-energy reward, a 16-frame report needs at most **three application bytes**. Longer intervals can save more airtime than further byte compression, but can delay learning.

## What is already known locally?

The device knows its own frame sequence, chosen SF and transmit power, and local energy accounting. Do not send these values back. It can compute PDR from reported receipt outcomes. The server supplies only reception outcomes and the SNR information measured at the gateway.

The existing pilot budgets 35 PHYPayload bytes per report: 13 framing + 4 counter + 2 bitmap + 16 historical SNR bytes. It also adds four request bytes to every eighth attempted uplink. These are design choices in this pilot, not mandatory LoRaWAN reporting fields.

LoRaWAN application framing without FOpts costs 13 bytes: MHDR 1, minimum FHDR 7, FPort 1, MIC 4. A proprietary learning report belongs in an application payload on an agreed FPort; arbitrary bytes must not be presented as standardized MAC commands. See [LoRaWAN L2 1.0.4, pp. 15, 17, 23](https://lora-alliance.org/wp-content/uploads/2021/11/LoRaWAN-Link-Layer-Specification-v1.0.4.pdf).

## The proposed wire format

For a fixed 16-frame window, put the latest SNR-bin index in six bits and a format mode in the remaining two bits of a control byte. The pilot has 55 SNR bins plus an unknown state, so codes 0 through 55 fit. Codes 56 through 63 are reserved. Keep this as a bin index when integrating; do not reconstruct arbitrary dB values and digitize them again.

| Reception pattern | Application contents | Application bytes | PHYPayload bytes |
|---|---|---:|---:|
| All 16 arrived | Control/SNR byte | 1 | 14 |
| Exactly one missing | Control/SNR + missing position | 2 | 15 |
| Any other pattern | Control/SNR + 16 receipt bits | 3 | 16 |

Example: a receipt bit of 1 means the server received that logical frame; 0 means it did not. The device pairs each outcome with its locally stored action. Reporting only “14 out of 16 arrived” would lose that pairing if SF/power changed between frames.

`feedback_codec.py` implements these modes and checks all 65,536 bitmap patterns. Mode 00 is all received; 01 adds a byte containing a loss position 0–15; 10 adds a little-endian two-byte bitmap, with the oldest frame at bit 0; 11 is reserved. It validates field ranges, length, and position padding. It does not implement radio timing, session security, or a network server.

For arbitrary outcomes and 56 SNR states, a fixed 16-frame representation needs more than 16 bits, so three bytes is a reasonable worst-case byte budget. This is not a universal minimum for every learner or network architecture.

### When the counter can disappear

An implicit sequence anchor is safe only under this protocol contract:

1. Generate the report from the uplink that opened the current Class A RX1/RX2 windows. The bitmap ends at that uplink's frame counter and covers the preceding 15 logical frames.
2. Send that fresh report only in those receive windows. Drop it if the opportunity is missed. Never later transmit a stale queued snapshot with no anchor.
3. Retain the session context, LoRaWAN integrity/replay checks, and bounded history. Ignore positions before the start of the current session during startup.
4. Use unconfirmed uplinks with `NbTrans=1`, as in this pilot. Retransmissions can share FCnt; per-physical-attempt learning would require extra identification, or an objective defined per logical frame with all its attempt energy.

Class A windows follow the immediately preceding uplink; see [L2 1.0.4, pp. 12–13](https://lora-alliance.org/wp-content/uploads/2021/11/LoRaWAN-Link-Layer-Specification-v1.0.4.pdf). The implicit report association is our application rule, not something LoRaWAN guarantees for arbitrary queued application data. If that rule cannot be enforced, retain an explicit anchor. A truncated anchor also needs bounded wraparound and session rules.

No four-byte request is required if the server has a configured reporting rule, for example on received frames with `FCnt mod 8 = 0`. Every ordinary Class A uplink already provides receive windows. A lost trigger produces no report; it does not tell the node the outcome. Do not create dummy uplinks merely to poll for feedback.

## What can be saved mathematically?

Assumptions: SF12, BW 125 kHz, CR 4/5, explicit header, eight-symbol preamble, low-data-rate optimization enabled, and downlink payload CRC disabled. The MIC remains. RP002 specifies CRC on uplinks and off on downlinks ([RP002-1.0.3, p. 86, Table 109](https://lora-alliance.org/wp-content/uploads/2021/05/RP002-1.0.3-FINAL-1.pdf)). The old pilot used CRC-on for both directions; its 35-byte SF12 number happens to be unchanged by the correction because of symbol rounding.

| Report profile | PHY bytes | Report every K uplinks | Receipt window W | Seconds/report | DL seconds per 1,000 uplinks | Reduction |
|---|---:|---:|---:|---:|---:|---:|
| Existing | 35 | 8 | 16 | 1.810432 | 226.304 | — |
| Compact | 16 | 8 | 16 | 1.155072 | 144.384 | 36.2% |
| Larger batch | 18 | 16 | 32 | 1.318912 | 82.432 | 63.6% |
| Larger batch | 22 | 32 | 64 | 1.318912 | 41.216 | 81.8% |

The latter two rows budget a fixed bitmap of 4 or 8 bytes plus one SNR byte and 13 framing bytes. Their formats are byte-budget calculations; the provided codec implements only W=16. For different window profiles, agree their meaning in advance, for example using separate application FPorts. A fully adaptive scheduler should use a sufficiently large fixed window or explicitly identify its window profile; changing W silently is ambiguous.

W=2K allows a subsequent report to recover one missed reporting interval under the ideal periodic schedule. Older outcomes can still be lost after consecutive missed reports. Do not update the same transition twice when windows overlap.

These are normalized long-run airtime rates: 1,000/K reports, each sent once. They exclude missed opportunities, losses, retransmissions, queues, link feasibility, and energy effects. They are **not network performance measurements**. Doubling K increases feedback delay: at the pilot's 160-second high-load generation interval, K=32 is roughly 85 minutes between opportunities before duty-cycle drops. That could be too slow to react to a change.

All 14-, 15-, 16-, and 17-byte reports take **1.155072 seconds at SF12** under these settings. The adaptive codec therefore saves bytes but does not beat the fixed three-byte format's SF12 airtime. At SF7, a 14-byte report takes 41.216 ms and a 16-byte report 46.336 ms, so the same compression can save airtime there.

Removing a four-byte request from a 37-byte SF12 uplink reduces its length to 33 bytes and airtime by 163.84 ms. At one request per eight uplinks, that is another 20.48 seconds per 1,000 attempted uplinks in a homogeneous-SF12 calculation. This saving is not included in the downlink table. Neither percentage describes total battery saving.

The calculator uses the symbol expression implemented in the [Semtech SX1276 driver](https://raw.githubusercontent.com/Lora-net/LoRaMac-node/master/src/radio/sx1276/sx1276.c). LoRa airtime grows in symbol steps, so minimizing bytes alone is insufficient.

## Reporting and receive-window strategy

First compare the current profile with compact feedback at the same K=8. This isolates the format change. Then test K=16 and K=32 with adequate overlap, rather than assuming the largest interval is acceptable.

A proposed server-side adaptive rule is to lengthen spacing only after several stable receipt windows, and shorten it when observed receipt rate deteriorates, observed SF/TP changes substantially, or the time since the last report attempt exceeds a limit. Thresholds and the age limit are parameters to evaluate, not established optimal values. The server knows report attempts, not necessarily report delivery; an unconfirmed report supplies no delivery confirmation. It also cannot observe a device's private Q-value uncertainty without extra telemetry.

The device separately tracks the age of its last valid report and uses a bounded local fallback if feedback stays absent. That fallback must be specified and evaluated with the learner. An age limit requests an attempt at a suitable receive opportunity; it cannot guarantee delivery when uplinks or downlinks are unavailable. Avoid rapid feedback-request retries that would consume more airtime during an outage.

Use existing application downlinks for piggybacking when they occur within the age limit. Rebuild a fresh snapshot for the actual current uplink. Piggybacking can avoid an additional frame header, but provides no saving if there is no other downlink to share. A missing report always leaves outcomes **unknown**, never automatically successful or failed. Use later overlapping windows to recover labels, then expire unrecoverable history as unknown.

Evaluate RX1 when it permits a faster, reliable reply. A 16-byte downlink takes 46.336 ms at SF7 versus 1.155072 seconds at SF12. RX1 data rate depends on the uplink rate and configured offset; it cannot arbitrarily use SF7 after an SF12 uplink. RX2 and RX1 also have different regional channel/duty budgets. Faster downlink operation needs measured reception performance, not just a favorable uplink SNR. See [RP002-1.0.3, EU863–870 RX1/RX2 settings, p. 29](https://lora-alliance.org/wp-content/uploads/2021/05/RP002-1.0.3-FINAL-1.pdf) and Zorbas, [Downlink Spreading Factor selection in LoRaWAN, 2024](https://doi.org/10.1016/j.comcom.2023.12.013) (publisher abstract reviewed; full text unavailable).

Do not eliminate standard Class A receive windows merely because no learning report is expected. Other network/application commands may need them. Receiving a valid RX1 downlink avoids RX2 under the standard behavior; otherwise include both receive windows in energy accounting.

## Does this preserve our learning model?

For the proposed reward `r = success - 0.2 E/Eref`, per-frame success comes from the bitmap and energy comes from local accounting. Only the latest SNR bin is needed for the next decision's SNR state. Preserve each stored decision state, action, and next-decision state; do not rewrite past states using a report that arrived later. Process new outcomes chronologically and exactly once. Keep the same PDR-history convention.

This information is sufficient for the pilot's success/energy online variant and frozen evaluation variants. It is **not sufficient for online or offline training with the original SNR-dependent reward**, which uses each successful frame's historical SNR. Keep those SNR samples for that baseline, or explicitly define and test a different approximation. Report training feedback costs separately. For a format-only causal comparison, keep the information supplied to each policy and the reporting schedule matched; for an end-to-end deployment comparison, allow necessary policy-specific messages and charge all their airtime.

On integration, consume SNR-bin codes directly in state construction. Recompute airtime, device RX/TX energy, gateway duty scheduling, half-duplex blocking, unknown-outcome expiry, and training on the actual new feedback protocol. Keep beta and Eref fixed across format ablations; changing Eref also changes the effective reward penalty. None of these integrations is implemented by this payload-only package.

## What the literature already supplies

- Navas et al., [Energy-Aware Spreading Factor Selection in LoRaWAN Using Delayed-Feedback Bandits, 2023](https://doi.org/10.23919/IFIPNetworking57963.2023.10186444), explicitly study batched feedback. Their response carries per-SF receipt counts because their Thompson sampler does not need outcome order. This motivates delayed feedback but does not justify replacing per-action Q-learning outcomes with one overall PDR count. The [author's code/data release](https://zenodo.org/records/7876244) provides an ns-3 starting point.
- [RFC 8724, section 8.3.2.1](https://www.rfc-editor.org/rfc/rfc8724.html#section-8.3.2.1) describes compressed bitmap feedback. [RFC 9441, section 3](https://www.rfc-editor.org/rfc/rfc9441.html#section-3) combines multiple windows into one response to reduce transmissions. These are SCHC fragmentation mechanisms; the codec here borrows their principles and is not a standardized SCHC ACK.
- Standard `LinkADRReq` can carry data-rate and power changes in five MAC-command bytes including CID, with a two-byte `LinkADRAns` including CID. A server-side learner could retain SNR/outcomes and send changes only when needed, removing repeated learning-state reports. It still needs command delivery/recovery and knowledge of which configuration actually applied. This is a centralized architecture, with its own constraints, rather than unchanged local DSF2R. See [L2 1.0.4, pp. 26, 30–31](https://lora-alliance.org/wp-content/uploads/2021/11/LoRaWAN-Link-Layer-Specification-v1.0.4.pdf).

Batching, bitmap compression, and server-based control already exist. A defensible contribution would be an evaluated rule for when feedback is worth sending, coupled to the SF/TP learner, that reduces total radio cost while meeting an explicitly stated reliability and recovery target. Do not claim novelty from compression alone.

## Reproduce the included checks

```powershell
python -X utf8 outputs/feedback_optimization/feedback_codec.py --self-check
python -X utf8 outputs/feedback_optimization/airtime.py
```

`airtime.csv` contains SF7–SF12 lengths and uplink request increments; `comparison.csv` contains the four reporting profiles. The next network experiment must compare PDR, delivery per generated application frame, energy per delivered frame, feedback airtime, report age, unresolved outcomes, and recovery under a controlled load/channel change. The smallest wire format is useful only if the resulting feedback delay still supports the application.
