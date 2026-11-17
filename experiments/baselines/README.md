# Matched online algorithm references

This experiment compares two energy-reward Q learners with an independently implemented delayed Bernoulli Thompson sampler. All three use the same thirty SF–power choices, paired network environments and explicitly costed feedback protocols. Every device learns locally during evaluation.

## Reference definition

The Thompson sampler maintains separate Beta(1,1) priors for each SF–power pair. It samples one reception probability per action and selects the largest sample. A successfully returned receipt label increments the selected action's success or failure count. Missing feedback produces no update. SNR, the Q reward and energy expenditure do not enter the posterior.

This is a **thirty-action delayed binary Thompson reference**, not an exact reproduction of Navas et al.'s energy-aware six-SF/fixed-power method. Delayed bandit feedback is established prior work; see [Navas et al., 2023](https://doi.org/10.23919/IFIPNetworking57963.2023.10186444). The adaptation provides a distinct existing learning family under a common action set rather than attributing our extension to their published implementation.

The `reward` Q learner uses SNR and current power; `both` additionally uses the last twenty resolved reception outcomes. Both receive the unchanged reward `received - 0.2 * attempt_energy / E_REF`, with alpha 0.1, gamma 0.9 and fixed reference energy 0.36383221248 J. Their evaluation exploration is0.05. Thompson samples its posteriors without an added epsilon rule. Its objective is reception probability, while Q balances delivery and energy. These are disclosed trade-offs, not identical optimization objectives.

## Declared comparison matrix

- Sixty devices, one gateway and thirty paired evaluation seeds 1001–1030.
- Six cases: stable light load, stable heavy load, load step, channel attenuation step,30% feedback erasure, and all feedback erased.
- Two protocols: A, rich report with explicit request bytes; D, compact report without request bytes. Both report every eight attempted uplinks over a sixteen-frame window.
- Three independently trained models per protocol and seed; three training episodes on the normal load-step case, with seeds disjoint from evaluation. The trained prior is reused across cases, with fresh independent per-device copies for every evaluation.

The default matrix contains **1,080 online evaluations and 540 training episodes**. This separate online study complements the frozen reconstructed-DSF2R, state-only, reward-only and combined-policy comparisons in `../robustness/`. Frozen and online policies are not ranked together as if learning mode were controlled.

Training is protocol-matched, so A versus D includes changes during training and deployment. Actual attempted frames may differ through duty admission. All models pay for their reports, including erased reports, and receive outcomes only through decoded feedback. During the all-feedback-erased case no online label reaches any learner. The simulator still charges transmitted reports and modeled receive energy.

## Run

From the repository root, after installing the locked dependencies:

```text
python -m unittest discover -s experiments/baselines/tests -v
python experiments/baselines/run_online.py --quick
python experiments/baselines/run_online.py
```

The quick mode uses eight devices, two seeds, one training episode and shorter phases. It exercises every case/model/protocol but is not quantitative publication evidence. Full runs checkpoint one compressed file per evaluation seed. Use `--resume` only with identical configuration and source hashes. To regenerate tables and figures from saved checkpoints, use `run_online.py --analyze-only`.

## Results and interpretation

`results/index.json` records the complete configuration, source fingerprints and chunk hashes. Compressed checkpoints retain individual evaluations and training costs; `tables/` retains pretrained Q-tables or Thompson counts. Final per-device aggregate hashes are recorded, while final arrays are not all persisted. CSV files expose every model/protocol/phase and matched-seed contrast.

Three PNG/SVG figures show all-case delivery–energy trade-offs, within-model D−A effects, and Q−Thompson differences under the same protocol. Confidence intervals use10,000 pointwise paired seed-bootstrap samples. Training radio energy is separate from evaluation energy; CPU and gateway electrical energy remain excluded. Unknown labels are counted separately rather than labelled failures or assigned artificial delays.

The report must retain cases in which the proposed Q learner loses or the interval is inconclusive. Binary Thompson sampling is not an energy optimizer; a higher delivery rate at higher energy is not proof that either model dominates. This synthetic comparison does not validate the receiver model or prove superiority over a published paper's implementation.
