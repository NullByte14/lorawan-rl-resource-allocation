# Initial allocation experiment

This directory preserves the earlier independently implemented allocation pilot and its data. It includes the reconstructed DSF2R state/reward, delivery-aware ablations, frozen and online Q-learning, and an explicitly adapted MIX-MAB policy. The current controlled feedback and scenario studies are documented at the repository root.

The raw run has three paired seeds,60 devices, one gateway and light→heavy→light traffic. It is preliminary evidence with wide uncertainty, not a reproduction of the published DSF2R experiment or an independent confirmation of the later thirty-seed study. Path loss, radio currents and receiver behavior are declared synthetic assumptions.

This snapshot used one CRC-on airtime function for both directions. The subsequent feedback engine corrects downlink CRC and implements actual feedback payload serialization. The rich35-byte SF12 downlink happens to retain the same airtime after the CRC correction because of symbol quantization. Preserve this historical distinction when comparing versions.

The original simulator uses NumPy. To run its checks from this directory:

```shell
python -m unittest discover -s tests -v
```

`pilot.py` generates the recorded model family; `policies.py` contains Q-learning and a delayed-feedback MIX-MAB adaptation; `summarize.py` checks source and trace hashes before reporting results. The documented MIX-MAB adaptation changes the published action/observation assumptions and should not be presented as exact replication. `PROTOTYPE_AUDIT.md` records discrepancies in the public DSF2R application and the unavailable executable training path.

`RESULTS.md` and `results/` preserve all earlier outcomes, including the unfavorable third seed and the energy cost of improved delivery. The old dataset's code fingerprints refer to the preserved simulation sources, not this updated explanatory README.
