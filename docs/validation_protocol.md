# Validation protocol

The repository separates implementation checks, archived-data audits, and comparative experiments. None is treated as a substitute for receiver calibration or an independent simulator/testbed comparison. A successful test run does not require the proposed learner to outperform a reference policy.

## Implementation checks

The feedback experiment's existing tests check the Q update, UL/DL CRC and airtime references, radio-energy calculations, sensitivity/capture cases, channel separation, gateway half-duplex loss, duty admission, and receive costs for erased downlinks. They also check delayed transitions, unknown outcomes, duplicate protection, window expiry, startup padding, direct SNR-bin transport, and all 65,536 receipt bitmaps.

The root research-contract tests add bounded whole-system checks. In a deliberately isolated fixed-policy trace, all four protocol arms deliver the same packets and send the same number of reports: compression must reduce receive energy, request removal must reduce transmit energy, and their combination must reduce total energy. This monotonicity claim applies to that controlled trace, not to a learning network whose feedback can change later decisions. Other checks cover all-erased reports, a link below receiver sensitivity, and admitted packets completing after the simulation horizon.

Commands from the repository root:

```text
python -m unittest discover -s tests -v
python -m unittest discover -s experiments/feedback/tests -v
python -m unittest discover -s experiments/robustness/tests -v
python -m unittest discover -s experiments/baselines/tests -v
```

## Archived evidence audit

The versioned feedback study contains 1,320 evaluations and 1,530 training episodes. Contract tests require the complete declared grid: thirty paired evaluation seeds, three frozen device counts, all four feedback settings, three frozen learner variants, and both protocol-trained online learners at sixty devices. The tests check all arms rather than selecting favorable cases.

Every attempted packet must be resolved by a valid report, expire as unknown, or remain pending. Received/attempted/generated counts, energy partitions, report-byte bounds, report-loss counts, and time-bin totals must be consistent. Original-reward training retains rich historical SNR; compact original evaluation is frozen. Source-byte fingerprints are verified before interpreting the saved data. Canonical hashes of all 510 saved pretrained Q-tables are independently recomputed; frozen tables must be identical across feedback arms and unchanged in evaluation. Online initialization hashes must represent sixty independent copies of the correct protocol-trained table.

The final online per-device tables are not all persisted as arrays. Their aggregate hashes are recorded; selected saved-table replays provide an additional deterministic check. This distinction is retained rather than describing every online final state as independently audited from disk.

## Statistical interpretation

The independent unit is an evaluation seed, not an individual packet. Within a seed, exogenous traffic/topology/fading are paired across methods. Initial Q-tables are shared across frozen protocol comparisons; online tables are trained separately for each protocol. Whole-run delivery and energy ratios use sums within each run before averaging across seeds. Label-delay means are weighted by the number of resolved packets.

Confidence intervals use 10,000 seed-level percentile bootstrap samples. Paired contrasts include compression, request removal, their combination and their interaction. The intervals are pointwise and descriptive, without multiplicity adjustment or a prespecified delivery-noninferiority margin. An interval spanning zero does not establish equivalence. A lower mean in one policy is not a universal superiority claim.

## Continuous integration

The GitHub Actions workflow installs the locked dependencies under Python 3.14 on a fresh Ubuntu runner and invokes `python validate.py --quick`. This audits the archived dataset, executes the mechanism and baseline-posterior tests, and regenerates all three small studies: controlled feedback, six-case frozen robustness, and matched online Thompson/Q comparisons. Regenerated smoke evidence is retained as a workflow artifact. The quick runs are mechanics checks; the full thirty-seed studies remain the quantitative evidence.

Official action versions are taken from the [checkout](https://github.com/actions/checkout), [setup-python](https://github.com/actions/setup-python), and [upload-artifact](https://github.com/actions/upload-artifact) repositories and pinned to their verified commit SHAs. A configured workflow is not described as passing until its remote run succeeds.

Exact historical replays were validated in the recorded Windows/NumPy environment. Linux CI validates fresh paired execution and invariant accounting. Floating-point and library differences may prevent bit-identical historical trace hashes on another platform; that must not be disguised as radio validation.

## External validity

The radio receiver, capture model, demodulator cap, propagation and currents are declared simplifying assumptions. Retransmissions, queues, mobility, multiple gateways, gateway electrical energy, backhaul and calibrated processing energy are outside the current model. Standard ADR or another published method must not be labelled a faithful replication unless its decision information, feedback and implementation have been reproduced. Independent ns-3 and measured-radio checks remain necessary before strong deployment claims.
