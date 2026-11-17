"""Small, explicitly simplified LoRaWAN allocation experiment (NumPy only).

This is an independent mechanism pilot, not LoRaSim or an exact DSF2R
reproduction. Decisions occur for each uplink, with updates delayed to a report. The same
delayed/lossy/costed reporting protocol is used for every method.
"""
import argparse
import copy
import csv
import hashlib
import heapq
import json
import math
import platform
from collections import deque
from pathlib import Path
import time

import numpy as np
from policies import QPolicy, MixMAB

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
METHODS = ['original_frozen', 'reward_frozen', 'state_frozen', 'both_frozen',
           'original_online', 'both_online', 'mix_delayed_adapted']


def airtime(sf, phy_bytes, bw=125000):
    """Explicit header, CRC on, eight preamble symbols, coding rate 4/5."""
    low_rate = int(sf >= 11)
    symbols = 8 + max(math.ceil((8*phy_bytes - 4*sf + 28 + 16) /
                               (4*(sf - 2*low_rate))) * 5, 0)
    return (8 + 4.25 + symbols) * 2**sf / bw


E_REF = 3.3 * .035 * airtime(12, 37) + 3.3 * .0108 * (2 + airtime(12, 35))


def original_reward(sf, snr):
    """Printed 2025 equation, using reported (not hidden failed-frame) SNR.

    Observed SNR is quantized to .5 dB, as are thresholds. Positive margins
    therefore cannot approach zero continuously. Exact zero uses else branch.
    """
    if snr is None:
        return 0.0
    margin = snr - REQUIRED[sf-7]
    if margin > 0:
        return (1/sf + margin) / (margin*sf)
    return float(sf*margin)


def state_key(node, extended):
    snr_bin = 55 if node['snr'] is None else int(np.clip(
        np.digitize(node['snr'], np.linspace(-20, 6, 55))-1, 0, 54))
    base = (snr_bin, ACTIONS[node['last_action']][1]//2 - 3)
    if not extended:
        return base
    history = node['known']
    pdr_bin = 3 if not history else int(np.digitize(np.mean(history), [.5,.8]))
    return base + (pdr_bin,)


def make_trace(seed, n, phase_seconds):
    """Exogenous positions, channels, fading and traffic are identical across policies."""
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


def run(seed, n, phase_seconds, policy, variant, training=False):
    started = time.perf_counter()
    positions, loss, trace = make_trace(seed, n, phase_seconds)
    end = 3*phase_seconds
    rng = np.random.default_rng(seed+100000)
    is_mix = variant == 'mix_delayed_adapted'
    extended = variant.startswith('state') or variant.startswith('both')
    energy_reward = variant.startswith('reward') or variant.startswith('both')
    nodes = [dict(last_action=29, action=None, state=None, snr=None,
                  known=deque(maxlen=20), pending=deque(maxlen=64), seq=0,
                  next_tx=0., rx_until=0.) for _ in range(n)]
    stats = [dict(generated=0, attempted=0, received=0, tx_j=0., rx_j=0.,
                  duty_dropped=0, requested=0, reports=0, feedback_j=0.,
                  unresolved=0, updates=0, dl_airtime_s=0., collisions=0,
                  sensitivity=0, half_duplex=0, demodulator=0) for _ in range(3)]
    bins = [dict(start_s=i*800, attempted=0, received=0, energy_j=0.)
            for i in range(math.ceil(end/800))]
    active = []
    queue = []
    counter = 0
    gw_busy = -1.
    gw_next_dl = 0.
    sequence = 0
    expired = 0
    def push(t, priority, kind, data):
        nonlocal counter
        heapq.heappush(queue, (t, priority, counter, kind, data))
        counter += 1
    for event in trace:
        push(event[0], 2, 'generate', event)
    def phase_of(t):
        return min(int(t//phase_seconds), 2)
    def add_energy(frame, rx=0., tx=0., feedback=0.):
        p = stats[frame['phase']]
        p['tx_j'] += tx
        p['rx_j'] += rx
        p['feedback_j'] += feedback
        frame['energy'] += tx+rx
        bins[min(int(frame['start']//800),len(bins)-1)]['energy_j'] += tx+rx
    def check_interference():
        for frame in active:
            interfering = [other for other in active
                           if other is not frame and other['channel']==frame['channel']]
            total = sum(10**((other['rssi']-frame['rssi'] +
                             SIR[frame['sf']-7,other['sf']-7])/10)
                        for other in interfering)
            if total > 1:
                frame['failure'].add('collisions')
    while queue:
        t, _, _, kind, data = heapq.heappop(queue)
        if t >= end and kind != 'finish':
            continue
        if kind == 'generate':
            _, device, channel, fade, feedback_uniform, dl_fade = data
            node = nodes[device]
            phase = phase_of(t)
            stats[phase]['generated'] += 1
            if t < node['next_tx'] or t < node['rx_until']:
                stats[phase]['duty_dropped'] += 1
                continue
            learner = policy[device] if isinstance(policy,list) else policy
            node['state'] = state_key(node, extended)
            # Store the next state actually observable at the next decision.
            # This prevents substituting future report knowledge into a past state.
            if node['pending'] and node['pending'][-1]['next_state'] is None:
                node['pending'][-1]['next_state'] = node['state']
            node['action'] = learner.choose(node['state'], rng)
            action = node['action']
            sf, tp = ACTIONS[action]
            node['last_action'] = action
            node['seq'] += 1
            request = node['seq'] % 8 == 0
            duration = airtime(sf, 33+4*request)
            rx1 = .01 + 8*2**sf/125000
            rx2 = .01 + 8*2**12/125000
            frame = dict(device=device, action=action, sf=sf, tp=tp, channel=channel,
                         start=t, stop=t+duration, phase=phase, request=request,
                         rssi=float(tp-loss[device]+fade), failure=set(), success=False,
                         energy=0., dl_uniform=feedback_uniform, dl_fade=dl_fade,
                         seq=node['seq'], rx2=rx2, state=node['state'], next_state=None,
                         prior_snr=node['snr'])
            if len(node['pending']) == 64:
                old_frame = node['pending'].popleft()
                if is_mix:
                    learner.discard_pending(old_frame['action'])
                stats[phase]['unresolved'] += 1
                expired += 1
            node['pending'].append(frame)
            stats[phase]['attempted'] += 1
            bins[min(int(t//800),len(bins)-1)]['attempted'] += 1
            add_energy(frame, tx=3.3*TX_MA[tp]/1000*duration,
                       rx=3.3*.0108*(rx1+rx2))
            node['next_tx'] = t + duration/.01
            node['rx_until'] = t+duration+2+rx2
            if frame['rssi'] < SENSITIVITY[sf-7] or frame['rssi']-NOISE_DBM < REQUIRED[sf-7]:
                frame['failure'].add('sensitivity')
            if t < gw_busy:
                frame['failure'].add('half_duplex')
            if sum(not f['failure'] for f in active) >= 8:
                frame['failure'].add('demodulator')
            active.append(frame)
            check_interference()
            push(frame['stop'], 0, 'finish', frame)
        elif kind == 'finish':
            frame = data
            active.remove(frame)
            frame['success'] = not frame['failure']
            for reason in frame['failure']:
                stats[frame['phase']][reason] += 1
            if frame['success']:
                stats[frame['phase']]['received'] += 1
                bins[min(int(frame['start']//800),len(bins)-1)]['received'] += 1
            if frame['request']:
                stats[frame['phase']]['requested'] += 1
                if frame['success'] and t+2 < end:
                    push(t+2, 1, 'downlink', frame)
        elif kind == 'downlink':
            frame = data
            if t < gw_busy or t < gw_next_dl:
                continue
            # 13B framing +4B counters +2B receipts +16 SNR bytes =35B.
            duration = airtime(12, 35)
            if t+duration >= end:
                continue
            gw_busy = t+duration
            gw_next_dl = t+duration/.10
            stats[frame['phase']]['dl_airtime_s'] += duration
            # A half-duplex gateway cannot receive UL while transmitting DL.
            for other in active:
                other['failure'].add('half_duplex')
            device = frame['device']
            down_rssi = 14-loss[device]+frame['dl_fade']
            valid = (frame['dl_uniform'] >= .05 and down_rssi >= SENSITIVITY[5]
                     and down_rssi-NOISE_DBM >= REQUIRED[5])
            extra_rx = 3.3*.0108*max(0,duration-frame['rx2'])
            add_energy(frame, rx=extra_rx, feedback=extra_rx)
            nodes[device]['rx_until'] = t+duration
            if valid:
                push(t+duration, 0, 'report', frame)
        elif kind == 'report':
            frame = data
            node = nodes[frame['device']]
            # The server returns a closed sequence-window receipt bitmap.
            # Losses are inferred only inside a successfully received report.
            earliest = max(1,frame['seq']-15)
            too_old = [f for f in node['pending'] if f['seq']<earliest]
            for f in too_old:
                if is_mix:
                    policy[frame['device']].discard_pending(f['action'])
            stats[frame['phase']]['unresolved'] += len(too_old)
            expired += len(too_old)
            covered = [f for f in node['pending'] if earliest<=f['seq']<=frame['seq']]
            if not covered:
                continue
            for f in covered:
                node['known'].append(int(f['success']))
            successful = [f for f in covered if f['success']]
            if successful:
                measured = successful[-1]['rssi']-NOISE_DBM
                node['snr'] = round(measured*2)/2
            learner = policy[frame['device']] if isinstance(policy,list) else policy
            next_state = state_key(node, extended)
            old_count = learner.update_count
            for f in covered:
                success = int(f['success'])
                energy_norm = f['energy']/E_REF
                # Successful frames return their own quantized SNR. Failed
                # frames retain the SNR known when choosing that frame's action.
                # Unpublished original failure handling remains unverified.
                observed_snr = round((f['rssi']-NOISE_DBM)*2)/2 if success else f['prior_snr']
                reward = success-.2*energy_norm if energy_reward else original_reward(f['sf'],observed_snr)
                learner.update(f['state'], f['action'], reward,
                               f['next_state'] if f['next_state'] is not None else next_state,
                               success=success, energy_norm=energy_norm)
            stats[frame['phase']]['updates'] += learner.update_count-old_count
            stats[frame['phase']]['reports'] += 1
            node['pending'] = deque([f for f in node['pending'] if f['seq']>frame['seq']], maxlen=64)
    # Sleep contribution is added exactly once per node and phase. Active time
    # is small; use elapsed-time background leakage, explicitly documented.
    for p in stats:
        p['sleep_j'] = n*phase_seconds*3.3*1.5e-6
        p['total_j'] = p['tx_j']+p['rx_j']+p['sleep_j']
        p['pdr'] = p['received']/p['attempted'] if p['attempted'] else None
        p['app_delivery'] = p['received']/p['generated'] if p['generated'] else None
        p['j_per_delivery'] = p['total_j']/p['received'] if p['received'] else None
    learners = policy if isinstance(policy,list) else [policy]
    # Unreported tail outcomes remain unknown. Remove cached propensities
    # before reusing a trained learner in an independent simulation.
    if is_mix:
        for node, learner in zip(nodes,policy):
            for frame in node['pending']:
                learner.discard_pending(frame['action'])
    return dict(phases=stats, bins=bins,
                trace_sha256=hashlib.sha256(repr(trace).encode()).hexdigest(),
                positions=positions.tolist(),
                updates=sum(l.update_count for l in learners),
                table_bytes=sum(l.policy_size_bytes for l in learners),
                mix_round_robin_nodes=sum(l.in_round_robin for l in learners) if is_mix else 0,
                expired_labels=expired,
                pending_labels=sum(len(node['pending']) for node in nodes),
                elapsed_seconds=time.perf_counter()-started)


def mean_ci(values):
    a = np.asarray([v for v in values if v is not None],float)
    # Student-t 97.5% quantiles; no scipy dependency. Small-pilot intervals wide.
    critical = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,9:2.262,19:2.093}
    if len(a)<2:
        return float(a.mean()) if len(a) else None, None
    df=len(a)-1
    t=critical[df] if df in critical else next(v for k,v in sorted(critical.items(),reverse=True) if k<df)
    return float(a.mean()), float(t*a.std(ddof=1)/math.sqrt(len(a)))


def recovery(bins, phase_seconds):
    """Within 5 percentage points of own late high-load PDR, two bins in a row.

    A descriptive settling-time measure, not a common quality target. A policy
    can settle quickly at a poor PDR, so this MUST be read beside PDR.
    """
    high=[b for b in bins if phase_seconds<=b['start_s']<2*phase_seconds and b['attempted']]
    if len(high)<4:
        return None
    reference=sum(b['received'] for b in high[-3:])/sum(b['attempted'] for b in high[-3:])
    for i in range(len(high)-1):
        if all(abs(b['received']/b['attempted']-reference)<=.05 for b in high[i:i+2]):
            return high[i]['start_s']-phase_seconds+1600
    return None


def common_recovery(bins, phase_seconds, target=.8):
    high=[b for b in bins if phase_seconds<=b['start_s']<2*phase_seconds and b['attempted']]
    for i in range(len(high)-1):
        if all(b['received']/b['attempted']>=target for b in high[i:i+2]):
            return high[i]['start_s']-phase_seconds+1600
    return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,nargs='+',default=[101,202,303])
    parser.add_argument('--nodes',type=int,default=60)
    parser.add_argument('--phase-seconds',type=int,default=8000)
    parser.add_argument('--train-episodes',type=int,default=3)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    args=parser.parse_args()
    if args.nodes<1 or args.phase_seconds<3200 or args.phase_seconds%800 or args.train_episodes<1:
        parser.error('Require nodes>=1, phase-seconds>=3200 divisible by800, train-episodes>=1')
    args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    training_records=[]
    for seed in args.seeds:
        trained={}
        for variant in ['original','reward','state','both']:
            learner=QPolicy(epsilon=.25)
            for ep in range(args.train_episodes):
                train=run(seed+50000+ep*1000,args.nodes,args.phase_seconds,learner,variant,True)
                training_records.append(dict(seed=seed,variant=variant,episode=ep,
                    environment_seed=seed+50000+ep*1000,
                    total_j=sum(p['total_j'] for p in train['phases']),
                    attempted=sum(p['attempted'] for p in train['phases']),
                    reported_updates=sum(p['updates'] for p in train['phases']),
                    desktop_runtime_s=train['elapsed_seconds']))
            trained[variant]=learner
        mix_trained=[MixMAB(horizon=5000) for _ in range(args.nodes)]
        for ep in range(args.train_episodes):
            train=run(seed+50000+ep*1000,args.nodes,args.phase_seconds,mix_trained,'mix_delayed_adapted',True)
            training_records.append(dict(seed=seed,variant='mix_delayed_adapted',episode=ep,
                environment_seed=seed+50000+ep*1000,
                total_j=sum(p['total_j'] for p in train['phases']),
                attempted=sum(p['attempted'] for p in train['phases']),
                reported_updates=sum(p['updates'] for p in train['phases']),
                desktop_runtime_s=train['elapsed_seconds']))
        for method in METHODS:
            if method=='mix_delayed_adapted':
                learner=copy.deepcopy(mix_trained)
            else:
                variant,mode=method.split('_')
                learner=copy.deepcopy(trained[variant])
                learner.update_enabled=mode=='online'
                learner.epsilon=.05 if mode=='online' else 0.
                if mode=='online':
                    # Each deployed device updates its own copy. Sharing one
                    # live table would leak updates across devices for free.
                    learner=[copy.deepcopy(learner) for _ in range(args.nodes)]
            before=sum(x.update_count for x in learner) if isinstance(learner,list) else learner.update_count
            result=run(seed,args.nodes,args.phase_seconds,learner,method)
            result['evaluation_updates']=result['updates']-before
            result.update(seed=seed,method=method,
                          own_reference_settling_s=recovery(result['bins'],args.phase_seconds),
                          time_to_80pct_s=common_recovery(result['bins'],args.phase_seconds))
            records.append(result)
            print(seed,method,'high-load PDR',round(result['phases'][1]['pdr'],4),flush=True)
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [Path(__file__),Path(__file__).with_name('policies.py')]}
    config=vars(args).copy();config['output']=str(config['output'])
    config['python_version']=platform.python_version()
    config['numpy_version']=np.__version__
    config['baseline_status']='Paper-aligned Q-learning reconstruction; not released DSF2R training or weights'
    config['feedback']='Every8 attempted uplinks request last16 receipts and received-frame SNR; RX2 SF12; DL10% duty;5% extra erasure'
    (args.output/'raw.json').write_text(json.dumps(dict(config=config,source_sha256=source_hashes,
                                                      training=training_records,records=records),indent=2),encoding='utf8')
    with (args.output/'training.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(training_records[0]));writer.writeheader();writer.writerows(training_records)
    with (args.output/'per_seed.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','method','phase']+list(records[0]['phases'][0]))
        writer.writeheader()
        for r in records:
            for phase,s in enumerate(r['phases']):
                writer.writerow(dict(seed=r['seed'],method=r['method'],phase=phase,**s))
    rows=[]
    for method in METHODS:
        selected=[r for r in records if r['method']==method]
        for phase in range(3):
            row=dict(method=method,phase=phase,seeds=len(selected))
            for metric in ['pdr','app_delivery','j_per_delivery','dl_airtime_s','reports']:
                avg,half=mean_ci([r['phases'][phase][metric] for r in selected])
                row[metric+'_mean']=avg;row[metric+'_ci95_half']=half
            rows.append(row)
    with (args.output/'summary.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    paired=[]
    for method in METHODS[1:]:
        for phase in range(3):
            for metric in ['pdr','j_per_delivery']:
                diffs=[]
                for seed in args.seeds:
                    baseline=next(r for r in records if r['seed']==seed and r['method']=='original_frozen')
                    other=next(r for r in records if r['seed']==seed and r['method']==method)
                    a,b=other['phases'][phase][metric],baseline['phases'][phase][metric]
                    if a is not None and b is not None: diffs.append(a-b)
                avg,half=mean_ci(diffs)
                paired.append(dict(method=method,phase=phase,metric=metric,
                                   delta_mean=avg,delta_ci95_half=half))
    with (args.output/'paired_differences.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    print('Saved pilot results',args.output,flush=True)


if __name__=='__main__':
    main()
