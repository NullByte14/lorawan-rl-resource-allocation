# Pilot results — delivery-aware DSF2R extension

These are actual synthetic pilot results from the supplied code. The baseline is a paper-based reconstruction with declared training, feedback and failed-frame assumptions. It is not an exact execution of the authors' pretrained DSF2R. All variants received the same reporting opportunities and paid for the same reporting protocol. Feedback deliveries differ as a consequence of their radio choices.

Three paired seeds, 60 fixed devices and one gateway were evaluated. Generation intervals changed from 400 to 160 and back to 400 seconds across three 8000-second phases. Every learner received three independent simulated training episodes. No measured propagation/current calibration dataset was used.

## High-load phase

Values below are means across seeds. Energy includes device TX, RX windows, feedback listening and background sleep during evaluation. Uncalibrated processing energy and gateway electrical energy are excluded. Training expenditure is recorded separately.

| Method | PDR per attempted packet | Delivery per generated packet | Energy per delivered packet | Valid reports |
|---|---:|---:|---:|---:|
| original_frozen | 70.91% | 69.39% | 67.08 mJ | 62.0 |
| reward_frozen | 72.47% | 71.10% | 61.67 mJ | 76.0 |
| state_frozen | 71.63% | 66.12% | 87.07 mJ | 84.7 |
| both_frozen | 75.62% | 73.14% | 70.50 mJ | 78.7 |
| original_online | 70.46% | 68.43% | 67.90 mJ | 87.0 |
| both_online | 75.89% | 72.76% | 77.74 mJ | 95.7 |
| mix_delayed_adapted | 69.93% | 59.68% | 105.17 mJ | 121.3 |

## Paired evidence

Differences compare the same held-out seeds. Intervals are Student-t 95% intervals over three seed differences. They are very wide and should not be interpreted as proof of improvement or equivalence.

- both_frozen minus original_frozen, pdr: mean +4.71 percentage points; 95% interval[-20.44, +29.85].
- both_frozen minus original_frozen, j_per_delivery: mean +3.42 mJ/delivered packet; 95% interval[-6.49, +13.32].
- both_online minus original_online, pdr: mean +5.43 percentage points; 95% interval[-9.31, +20.17].
- both_online minus original_online, j_per_delivery: mean +9.84 mJ/delivered packet; 95% interval[-10.69, +30.37].

## Per-seed check

| Seed | Original frozen PDR | Both changes frozen PDR | Original energy | Both changes energy |
|---|---:|---:|---:|---:|
| 101 | 68.57% | 77.51% | 51.33 mJ | 50.48 mJ |
| 202 | 67.64% | 79.66% | 77.32 mJ | 81.39 mJ |
| 303 | 76.54% | 69.70% | 72.59 mJ | 79.63 mJ |

## Adaptation diagnostic

This pilot counts time to two consecutive high-load windows with PDR at least 0.8. Failure to reach that criterion is right-censored by the end of the phase. It does not mean an infinite recovery time. No consistent recovery advantage is established.

| Method | Runs reaching criterion / 3 | Observed times since traffic increase |
|---|---:|---|
| original_frozen | 1/3 | 1600 s |
| reward_frozen | 0/3 | Not reached in these runs |
| state_frozen | 1/3 | 1600 s |
| both_frozen | 1/3 | 1600 s |
| original_online | 1/3 | 1600 s |
| both_online | 1/3 | 1600 s |
| mix_delayed_adapted | 0/3 | Not reached in these runs |

MIX-MAB adaptation round-robin node counts at evaluation end were [0, 0, 0]. This checks one exploration-stage concern, but does not establish a bit-exact reproduction of MIX-MAB.

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
