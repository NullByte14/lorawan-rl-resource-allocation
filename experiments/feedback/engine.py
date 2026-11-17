"""Independent, deliberately simplified LoRaWAN feedback experiment engine.

Derived from the preserved first pilot. Radio/path-loss/interference assumptions
are unchanged. This engine corrects downlink CRC, separates gateway observations
from device records, and transports feedback through actual payload codecs.
It is not ns-3, a LoRaWAN stack, or an exact published DSF2R reproduction.

Generated uplinks before the horizon are charged for their complete attempt and
allowed to finish (a short drain). No new report starts or finishes at/after the
horizon. Outcomes are attributed to the generating phase; background sleep is
charged over the stated horizon as in the first pilot. No retransmission/queue.
"""

from collections import deque
from dataclasses import dataclass
import hashlib
import heapq
import math
import struct
import time

import numpy as np

from feedback_codec import encode_feedback, decode_feedback


SF = np.arange(7, 13)
TP = np.array([6, 8, 10, 12, 14])
ACTIONS = [(int(s), int(p)) for s in SF for p in TP]
REQUIRED = np.array([-7.5, -10, -12.5, -15, -17.5, -20])
SENSITIVITY = np.array([-126.5, -127.25, -131.25, -132.75, -133.25, -134.5])
SIR = np.array([[6,-8,-9,-9,-9,-9],[-11,6,-11,-12,-13,-13],
                [-15,-13,6,-13,-14,-15],[-19,-18,-17,6,-17,-18],
                [-22,-22,-21,-20,6,-20],[-25,-25,-25,-24,-23,6]])
TX_MA = {6:20, 8:22, 10:25, 12:29, 14:35}
NOISE_DBM = -117.0
E_REF = 0.36383221248  # Fixed first-pilot numeric normalization, never re-tuned.
BETA = 0.2
SNR_THRESHOLDS = np.linspace(-20, 6, 55)


@dataclass(frozen=True)
class FeedbackConfig:
    report_format: str = "rich"
    request_bytes: int = 4
    interval: int = 8
    window: int = 16
    downlink_loss: float = 0.05

    def __post_init__(self):
        if self.report_format not in ("rich", "compact"):
            raise ValueError("report_format must be rich or compact")
        if self.request_bytes not in (0, 4):
            raise ValueError("request_bytes must be 0 or 4")
        if self.interval < 1 or self.window != 16:
            raise ValueError("interval must be positive and window must be 16")
        if not 0 <= self.downlink_loss <= 1:
            raise ValueError("downlink_loss must lie in [0,1]")


CONFIGS = {
    "A": FeedbackConfig("rich", 4),
    "B": FeedbackConfig("compact", 4),
    "C": FeedbackConfig("rich", 0),
    "D": FeedbackConfig("compact", 0),
}


def airtime(sf, phy_bytes, bw=125000, crc_on=True):
    """Semtech LoRa expression: explicit header, CR4/5, eight-symbol preamble."""
    if sf not in range(7, 13) or not 0 <= phy_bytes <= 255 or bw <= 0:
        raise ValueError("Unsupported SF, PHYPayload length or bandwidth")
    low_rate = int(sf >= 11)
    symbols = 8 + max(math.ceil((8*phy_bytes - 4*sf + 28 + 16*int(crc_on)) /
                               (4*(sf - 2*low_rate))) * 5, 0)
    return (8 + 4.25 + symbols) * 2**sf / bw


def snr_to_bin(snr):
    return 55 if snr is None else int(np.clip(
        np.digitize(snr, SNR_THRESHOLDS)-1, 0, 54))


def original_reward(sf, snr):
    if snr is None:
        return 0.0
    margin = snr - REQUIRED[sf-7]
    return float((1/sf + margin)/(margin*sf) if margin > 0 else sf*margin)


def state_key(node, extended):
    # A compact report supplies this exact coordinate, not synthetic dB.
    snr_bin = node.get("snr_bin", snr_to_bin(node.get("snr")))
    base = (int(snr_bin), ACTIONS[node["last_action"]][1]//2-3)
    if not extended:
        return base
    history = node["known"]
    pdr_bin = 3 if not history else int(np.digitize(np.mean(history), [.5, .8]))
    return base + (pdr_bin,)


@dataclass(frozen=True)
class Report:
    anchor: int
    receipts: tuple
    snr_bin: int
    snrs: tuple  # Raw quantized values only in rich reports.


def build_report(report_format, anchor, server_history):
    """Serialize a fresh gateway snapshot; missing entries are actual UL gaps.

    server_history maps successfully decoded logical sequence numbers to SNR.
    The rich application payload is exactly 22 bytes: counter4 + bitmap2 +
    sixteen one-byte half-dB SNR slots. Receipt bits determine slot validity.
    Half-dB SNR must fit -64..63.5 dB; otherwise fail rather than silently clip.
    Compact payloads have no anchor and are tied to current RX-window context.
    """
    if anchor < 1 or anchor not in server_history:
        raise ValueError("Fresh report requires the triggering uplink received")
    seqs = range(anchor-15, anchor+1)
    receipts = tuple(seq >= 1 and seq in server_history for seq in seqs)
    latest = server_history[anchor]
    if report_format == "compact":
        return encode_feedback(receipts, snr_to_bin(latest))
    if report_format != "rich":
        raise ValueError("Unknown report format")
    bitmap = sum(int(value) << i for i, value in enumerate(receipts))
    snr_codes = []
    for seq, received in zip(range(anchor-15, anchor+1), receipts):
        quantized = int(round(server_history[seq]*2)) if received else -128
        if received and not -128 <= quantized <= 127:
            raise ValueError("Rich report half-dB SNR exceeds its byte range")
        snr_codes.append(quantized+128)
    return struct.pack("<IH", anchor, bitmap) + bytes(snr_codes)


def parse_report(report_format, payload, anchor_context):
    """Decode only transmitted information using the current Class A context."""
    if report_format == "compact":
        decoded = decode_feedback(payload)
        result = Report(anchor_context, decoded.receipts, decoded.snr_bin,
                        (None,)*16)
    elif report_format == "rich":
        if len(payload) != 22:
            raise ValueError("Rich application payload must be 22 bytes")
        anchor, bitmap = struct.unpack("<IH", payload[:6])
        if anchor != anchor_context:
            raise ValueError("Stale rich report does not match RX context")
        receipts = tuple(bool(bitmap & (1 << i)) for i in range(16))
        snrs = tuple((payload[6+i]-128)/2 if received else None
                     for i, received in enumerate(receipts))
        result = Report(anchor, receipts, snr_to_bin(snrs[-1]), snrs)
    else:
        raise ValueError("Unknown report format")
    if result.anchor < 1 or not result.receipts[-1]:
        raise ValueError("Fresh current uplink must have a received final bit")
    return result


def make_trace(seed, n, phase_seconds):
    rng = np.random.default_rng(seed)
    radii = np.sqrt(rng.uniform(200**2, 2600**2, n))
    angles = rng.uniform(0, 2*np.pi, n)
    positions = np.column_stack((radii*np.cos(angles), radii*np.sin(angles)))
    loss = 32.4 + 32*np.log10(radii) + rng.normal(0, 3, n)
    events = []
    for phase, interval in enumerate([400., 160., 400.]):
        for device in range(n):
            offset = rng.uniform(0, interval)
            times = np.arange(phase*phase_seconds+offset,
                              (phase+1)*phase_seconds, interval)
            for t in times:
                events.append((float(t), device, int(rng.integers(3)),
                               float(rng.normal(0, 1.5)), float(rng.random()),
                               float(rng.normal(0, 1.5))))
    events.sort()
    return positions, loss, events


def run(seed, n, phase_seconds, policy, variant, feedback=CONFIGS["A"],
        training=False, debug=False):
    started = time.perf_counter()
    variant = variant.split("_")[0]
    if variant not in ("original", "reward", "both", "state"):
        raise ValueError("Unknown learner variant")
    if n < 1 or phase_seconds <= 0:
        raise ValueError("Require positive n and phase_seconds")
    learners = policy if isinstance(policy, list) else [policy]
    if isinstance(policy, list) and len(policy) != n:
        raise ValueError("Per-device policy list must have n entries")
    original = variant in ("original", "state")
    if original and feedback.report_format == "compact" and (
            training or any(getattr(p, "update_enabled", True) for p in learners)):
        raise ValueError("Original reward training/online updates need rich SNR")
    extended = variant in ("both", "state")
    positions, loss, trace = make_trace(seed, n, phase_seconds)
    end = 3*phase_seconds
    rng = np.random.default_rng(seed+100000)
    nodes = [dict(last_action=29, snr=None, snr_bin=55, known=deque(maxlen=20),
                  pending=deque(maxlen=64), seq=0, next_tx=0., rx_until=0.)
             for _ in range(n)]
    server_history = [{} for _ in range(n)]
    stats = [dict(generated=0, attempted=0, received=0, tx_j=0., rx_j=0.,
                  duty_dropped=0, requested=0, request_frames=0, reports=0,
                  feedback_j=0., unresolved=0, updates=0, dl_airtime_s=0.,
                  dl_sent=0, dl_blocked=0, dl_lost=0, report_phy_bytes=0,
                  ul_airtime_s=0., request_airtime_s=0.,
                  resolved=0, tail_unknown=0, feedback_age_sum_s=0.,
                  feedback_age_count=0, collisions=0, sensitivity=0,
                  half_duplex=0, demodulator=0) for _ in range(3)]
    bins = [dict(start_s=i*800, generated=0, attempted=0, received=0,
                 energy_j=0.) for i in range(math.ceil(end/800))]
    active, queue, debug_events = [], [], []
    action_counts = [0]*30
    counter, expired, post_horizon_finishes = 0, 0, 0
    gw_busy, gw_next_dl = -1., 0.

    def emit(kind, t, **kwargs):
        if debug:
            debug_events.append(dict(kind=kind, time=float(t), **kwargs))

    def push(t, priority, kind, data):
        nonlocal counter
        heapq.heappush(queue, (t, priority, counter, kind, data))
        counter += 1

    def bin_at(t):
        return bins[min(int(t//800), len(bins)-1)]

    def add_energy(frame, rx=0., tx=0., feedback_energy=0.):
        local = frame["local"]
        p = stats[local["phase"]]
        p["tx_j"] += tx
        p["rx_j"] += rx
        p["feedback_j"] += feedback_energy
        local["energy"] += tx+rx
        bin_at(local["start"])["energy_j"] += tx+rx

    def check_interference():
        for frame in active:
            total = sum(10**((other["rssi"]-frame["rssi"] +
                              SIR[frame["sf"]-7, other["sf"]-7])/10)
                        for other in active if other is not frame and
                        other["channel"] == frame["channel"])
            if total > 1:
                frame["failure"].add("collisions")

    for event in trace:
        push(event[0], 2, "generate", event)
    while queue:
        t, _, _, kind, data = heapq.heappop(queue)
        if t >= end and kind != "finish":
            continue
        if kind == "generate":
            _, device, channel, fade, feedback_uniform, dl_fade = data
            node = nodes[device]
            phase = min(int(t//phase_seconds), 2)
            stats[phase]["generated"] += 1
            bin_at(t)["generated"] += 1
            if t < node["next_tx"] or t < node["rx_until"]:
                stats[phase]["duty_dropped"] += 1
                continue
            learner = policy[device] if isinstance(policy, list) else policy
            state = state_key(node, extended)
            if node["pending"] and node["pending"][-1]["next_state"] is None:
                node["pending"][-1]["next_state"] = state
            action = learner.choose(state, rng)
            sf, tp = ACTIONS[action]
            node["last_action"] = action
            node["seq"] += 1
            opportunity = node["seq"] % feedback.interval == 0
            request_size = feedback.request_bytes if opportunity else 0
            duration = airtime(sf, 33+request_size, crc_on=True)
            rx1 = .01 + 8*2**sf/125000
            rx2 = .01 + 8*2**12/125000
            local = dict(seq=node["seq"], action=int(action), sf=sf,
                         start=t, phase=phase, energy=0., state=state,
                         next_state=None, prior_snr=node["snr"])
            frame = dict(device=device, sf=sf, channel=channel, stop=t+duration,
                         local=local, rssi=float(tp-loss[device]+fade),
                         failure=set(), opportunity=opportunity,
                         dl_uniform=feedback_uniform, dl_fade=dl_fade, rx2=rx2)
            if len(node["pending"]) == 64:
                expired_local = node["pending"].popleft()
                stats[expired_local["phase"]]["unresolved"] += 1
                expired += 1
            node["pending"].append(local)
            stats[phase]["attempted"] += 1
            stats[phase]["request_frames"] += int(request_size > 0)
            stats[phase]["ul_airtime_s"] += duration
            if request_size:
                stats[phase]["request_airtime_s"] += duration-airtime(sf, 33, crc_on=True)
            action_counts[action] += 1
            bin_at(t)["attempted"] += 1
            add_energy(frame, tx=3.3*TX_MA[tp]/1000*duration,
                       rx=3.3*.0108*(rx1+rx2))
            node["next_tx"] = t + duration/.01
            node["rx_until"] = t+duration+2+rx2
            if frame["rssi"] < SENSITIVITY[sf-7] or frame["rssi"]-NOISE_DBM < REQUIRED[sf-7]:
                frame["failure"].add("sensitivity")
            if t < gw_busy:
                frame["failure"].add("half_duplex")
            if sum(not f["failure"] for f in active) >= 8:
                frame["failure"].add("demodulator")
            active.append(frame)
            check_interference()
            emit("decision", t, device=device, seq=local["seq"],
                 state=state, action=int(action), energy_j=local["energy"])
            push(frame["stop"], 0, "finish", frame)
        elif kind == "finish":
            frame = data
            active.remove(frame)
            local, device = frame["local"], frame["device"]
            phase = local["phase"]
            if t >= end:
                post_horizon_finishes += 1
            success = not frame["failure"]
            for reason in frame["failure"]:
                stats[phase][reason] += 1
            if success:
                stats[phase]["received"] += 1
                bin_at(local["start"])["received"] += 1
                server_history[device][local["seq"]] = round(
                    (frame["rssi"]-NOISE_DBM)*2)/2
            # The server stores actual received packets, not device-side truth.
            cutoff = local["seq"]-64
            for seq in list(server_history[device]):
                if seq < cutoff:
                    del server_history[device][seq]
            if frame["opportunity"]:
                stats[phase]["requested"] += 1
                if success and t+2 < end:
                    push(t+2, 1, "downlink", frame)
        elif kind == "downlink":
            frame = data
            local, device = frame["local"], frame["device"]
            phase, anchor = local["phase"], local["seq"]
            if t < gw_busy or t < gw_next_dl:
                stats[phase]["dl_blocked"] += 1
                continue
            payload = build_report(feedback.report_format, anchor,
                                   server_history[device])
            phy_bytes = 13+len(payload)
            duration = airtime(12, phy_bytes, crc_on=False)
            if t+duration >= end:
                continue
            gw_busy = t+duration
            gw_next_dl = t+duration/.10
            stats[phase]["dl_airtime_s"] += duration
            stats[phase]["dl_sent"] += 1
            stats[phase]["report_phy_bytes"] += phy_bytes
            for other in active:
                other["failure"].add("half_duplex")
            down_rssi = 14-loss[device]+frame["dl_fade"]
            valid = (frame["dl_uniform"] >= feedback.downlink_loss and
                     down_rssi >= SENSITIVITY[5] and
                     down_rssi-NOISE_DBM >= REQUIRED[5])
            extra_rx = 3.3*.0108*max(0, duration-frame["rx2"])
            add_energy(frame, rx=extra_rx, feedback_energy=extra_rx)
            nodes[device]["rx_until"] = t+duration
            snapshot = parse_report(feedback.report_format, payload, anchor)
            emit("report_sent", t, device=device, anchor=anchor,
                 phy_bytes=phy_bytes, duration=duration,
                 receipts=snapshot.receipts, snr_bin=snapshot.snr_bin,
                 payload_hex=payload.hex(), valid=bool(valid))
            if valid:
                push(t+duration, 0, "report", (device, local, payload))
            else:
                stats[phase]["dl_lost"] += 1
        elif kind == "report":
            device, triggering_local, payload = data
            node = nodes[device]
            report = parse_report(feedback.report_format, payload,
                                  triggering_local["seq"])
            earliest = max(1, report.anchor-15)
            too_old = [f for f in node["pending"] if f["seq"] < earliest]
            for f in too_old:
                stats[f["phase"]]["unresolved"] += 1
            expired += len(too_old)
            covered = [f for f in node["pending"]
                       if earliest <= f["seq"] <= report.anchor]
            for f in covered:
                receipt_index = f["seq"]-(report.anchor-15)
                node["known"].append(int(report.receipts[receipt_index]))
            node["snr_bin"] = report.snr_bin
            node["snr"] = report.snrs[-1]  # None for compact: never fake dB.
            learner = policy[device] if isinstance(policy, list) else policy
            next_state = state_key(node, extended)
            old_count = learner.update_count
            for f in covered:
                receipt_index = f["seq"]-(report.anchor-15)
                success = int(report.receipts[receipt_index])
                energy_norm = f["energy"]/E_REF
                stats[f["phase"]]["resolved"] += 1
                stats[f["phase"]]["feedback_age_sum_s"] += t-f["start"]
                stats[f["phase"]]["feedback_age_count"] += 1
                if getattr(learner, "update_enabled", True):
                    observed = report.snrs[receipt_index] if success else f["prior_snr"]
                    reward = (original_reward(f["sf"], observed) if original
                              else success-BETA*energy_norm)
                    actual_next = f["next_state"] if f["next_state"] is not None else next_state
                    learner.update(f["state"], f["action"], reward, actual_next,
                                   success=success, energy_norm=energy_norm)
                    emit("update", t, device=device, seq=f["seq"],
                         state=f["state"], next_state=actual_next,
                         success=success, reward=float(reward), energy_j=f["energy"])
            stats[triggering_local["phase"]]["updates"] += learner.update_count-old_count
            stats[triggering_local["phase"]]["reports"] += 1
            emit("report_received", t, device=device, anchor=report.anchor,
                 snr_bin=report.snr_bin, resolved_seq=[f["seq"] for f in covered])
            node["pending"] = deque([f for f in node["pending"]
                                     if f["seq"] > report.anchor], maxlen=64)

    for node in nodes:
        for f in node["pending"]:
            stats[f["phase"]]["tail_unknown"] += 1
    for p in stats:
        p["sleep_j"] = n*phase_seconds*3.3*1.5e-6
        p["total_j"] = p["tx_j"]+p["rx_j"]+p["sleep_j"]
        p["pdr"] = p["received"]/p["attempted"] if p["attempted"] else None
        p["app_delivery"] = p["received"]/p["generated"] if p["generated"] else None
        p["j_per_delivery"] = p["total_j"]/p["received"] if p["received"] else None
        p["feedback_age_mean_s"] = (p["feedback_age_sum_s"]/p["feedback_age_count"]
                                      if p["feedback_age_count"] else None)
    result = dict(phases=stats, bins=bins,
                  trace_sha256=hashlib.sha256(repr(trace).encode()).hexdigest(),
                  topology_sha256=hashlib.sha256(positions.tobytes()+loss.tobytes()).hexdigest(),
                  positions=positions.tolist(),
                  updates=sum(p.update_count for p in learners),
                  table_bytes=sum(p.policy_size_bytes for p in learners),
                  expired_labels=expired,
                  pending_labels=sum(len(node["pending"]) for node in nodes),
                  post_horizon_finishes=post_horizon_finishes,
                  action_counts=action_counts,
                  elapsed_seconds=time.perf_counter()-started)
    if debug:
        result["debug_events"] = debug_events
    return result
