"""Build the readable pilot report from recorded outputs, without rerunning."""
import csv
import hashlib
import json
from pathlib import Path

from pilot import METHODS, mean_ci


def main():
    folder=Path(__file__).parent
    results=folder/'results'
    raw=json.loads((results/'raw.json').read_text(encoding='utf8'))
    records=raw['records']
    summary=list(csv.DictReader((results/'summary.csv').open(encoding='utf8')))
    for name,expected in raw['source_sha256'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Recorded source hash differs: '+name)
    for seed in raw['config']['seeds']:
        group=[r for r in records if r['seed']==seed]
        assert len({r['trace_sha256'] for r in group})==1
        assert len(group)==len(METHODS)
    text='''# Pilot results — delivery-aware DSF2R extension

These are actual synthetic pilot results from the supplied code. The baseline is a paper-based reconstruction with declared training, feedback and failed-frame assumptions. It is not an exact execution of the authors' pretrained DSF2R. All variants received the same reporting opportunities and paid for the same reporting protocol. Feedback deliveries differ as a consequence of their radio choices.

Three paired seeds, 60 fixed devices and one gateway were evaluated. Generation intervals changed from 400 to 160 and back to 400 seconds across three 8000-second phases. Every learner received three independent simulated training episodes. No measured propagation/current calibration dataset was used.

## High-load phase

Values below are means across seeds. Energy includes device TX, RX windows, feedback listening and background sleep during evaluation. Uncalibrated processing energy and gateway electrical energy are excluded. Training expenditure is recorded separately.

| Method | PDR per attempted packet | Delivery per generated packet | Energy per delivered packet | Valid reports |
|---|---:|---:|---:|---:|
'''
    for method in METHODS:
        row=next(s for s in summary if s['method']==method and s['phase']=='1')
        text+=f"| {method} | {100*float(row['pdr_mean']):.2f}% | {100*float(row['app_delivery_mean']):.2f}% | {1000*float(row['j_per_delivery_mean']):.2f} mJ | {float(row['reports_mean']):.1f} |\n"
    text+='''
## Paired evidence

Differences compare the same held-out seeds. Intervals are Student-t 95% intervals over three seed differences. They are very wide and should not be interpreted as proof of improvement or equivalence.

'''
    for method,baseline in [('both_frozen','original_frozen'),('both_online','original_online')]:
        for metric,multiplier,units in [('pdr',100,'percentage points'),('j_per_delivery',1000,'mJ/delivered packet')]:
            delta=[]
            for seed in raw['config']['seeds']:
                a=next(r for r in records if r['seed']==seed and r['method']==method)
                b=next(r for r in records if r['seed']==seed and r['method']==baseline)
                delta.append((a['phases'][1][metric]-b['phases'][1][metric])*multiplier)
            avg,half=mean_ci(delta)
            text+=f'- {method} minus {baseline}, {metric}: mean {avg:+.2f} {units}; 95% interval[{avg-half:+.2f}, {avg+half:+.2f}].\n'
    text+='''
## Per-seed check

| Seed | Original frozen PDR | Both changes frozen PDR | Original energy | Both changes energy |
|---|---:|---:|---:|---:|
'''
    for seed in raw['config']['seeds']:
        a=next(r for r in records if r['seed']==seed and r['method']=='original_frozen')['phases'][1]
        b=next(r for r in records if r['seed']==seed and r['method']=='both_frozen')['phases'][1]
        text+=f"| {seed} | {a['pdr']*100:.2f}% | {b['pdr']*100:.2f}% | {a['j_per_delivery']*1000:.2f} mJ | {b['j_per_delivery']*1000:.2f} mJ |\n"
    text+='''
## Adaptation diagnostic

This pilot counts time to two consecutive high-load windows with PDR at least 0.8. Failure to reach that criterion is right-censored by the end of the phase. It does not mean an infinite recovery time. No consistent recovery advantage is established.

| Method | Runs reaching criterion / 3 | Observed times since traffic increase |
|---|---:|---|
'''
    for method in METHODS:
        times=[r['time_to_80pct_s'] for r in records if r['method']==method and r['time_to_80pct_s'] is not None]
        text+=f"| {method} | {len(times)}/3 | {', '.join(str(x)+' s' for x in times) if times else 'Not reached in these runs'} |\n"
    mix=[r['mix_round_robin_nodes'] for r in records if r['method']=='mix_delayed_adapted']
    text+=f'\nMIX-MAB adaptation round-robin node counts at evaluation end were {mix}. This checks one exploration-stage concern, but does not establish a bit-exact reproduction of MIX-MAB.\n'
    text+='''
## Interpretation and limits

The full frozen extension has a higher average high-load PDR and higher energy per delivery in this pilot. Its behavior differs across seeds, including a PDR decrease on seed 303. Changing the reward alone has a lower average energy per delivery here, but these few runs cannot establish a reliable advantage. Adding state dimensions also changes sample requirements and unknown-state coverage. More training may alter the ranking.

This is sufficient to show a working allocation/feedback/measurement pipeline and an honest trade-off. It does not show that published DSF2R has been beaten. The reconstructed failed-packet reward, simplified collision receiver, uncalibrated energy currents, small sample and adapted MIX-MAB all limit external interpretation. See README.md and PROTOTYPE_AUDIT.md for details.

## What can be presented to the mentor

We implemented a delivery-aware extension of a paper-based DSF2R reconstruction and separated the effects of state and reward. A three-seed traffic-change pilot produced mixed results: the combined extension improved average delivery while using more energy per delivered packet, and uncertainty remains large. We will validate the reconstruction and receiver assumptions before scaling the experiment or claiming an improvement.

## Reproduce and inspect

- README.md explains the algorithm, information availability, modeled costs and run commands.
- results/raw.json preserves all run settings, trace hashes, source hashes, phase counts, 800-second windows, update counts, numeric model storage and desktop simulation times.
- results/training.csv separates training energy and updates from evaluation.
- results/per_seed.csv and results/summary.csv contain the numerical outcomes.
- results/paired_differences.csv compares each method to the frozen reconstruction.
- python summarize.py regenerates this report and verifies the code hashes and trace pairing.
'''
    (folder/'RESULTS.md').write_text(text,encoding='utf8')
    print('Wrote RESULTS.md; source hashes and paired traces verified.')


if __name__=='__main__':
    main()
