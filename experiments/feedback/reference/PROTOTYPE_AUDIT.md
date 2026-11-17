# DSF2R prototype verification

Inspected on 16 November 2026. Public repository HEAD was `98f74f47dfef4fd4b49d0d22b26fca27cc726cb6`. The recursive GitHub response was not truncated and listed ten paths, including directories. Exact acquired files and the commit/tree API responses are in `work/upstream/`.

## Verified application behavior

- `app.py:206–239` implements inference only. It loads `./config/Q_model-LORA-rob.pkl` using pickle on each call and applies `np.argmax`. There are no Q updates, reward calculations, PDR statistics, table writes, or training loops in this application.
- Its documented input is raw SNR, and the actual state lookup uses raw SNR plus a power-index expression. This supports the paper's `(SNR, TP)` definition rather than `(SNR margin, TP)`.
- `app.py:340–350` reads `best_gw.lsnr` from the MQTT queue and selects settings using `q_model(lsnr, selected_tp)`. The callback receives ChirpStack messages for the selected device and inserts them into that queue (`175–196`). After a message arrives, queued older messages are discarded (`341–344`).
- This feedback reaches the Raspberry Pi application through its IP/MQTT connection. The source does not send these observations as LoRa downlink payloads. A simulated periodic LoRa delivery report would therefore be a deliberately specified, costed feedback adaptation of this prototype.
- `app.py:332` calls `module.send_uplink(json_string, False)`. `RN2483.py:1017` defines `type_confirmed`, and `1079–1081` selects `uncnf` unless that flag is true. Uplinks here are unconfirmed.
- While the MQTT queue is empty, `app.py:303–334` keeps sending with the selected configuration. The pretrained table is frozen, but the resulting choice can change after a new SNR message. Describing it as completely nonadaptive would be inaccurate.
- The complete current tree has no separate training program or calibration dataset. A README section describes training but does not contain executable training code. Failure handling during the unpublished training process remains unverified. Some serial-command failure handling exists in the application, which is a different issue from learning from failed packet reception.

## Pretrained table and paper/code discrepancies

The weights ARE publicly available. The source pickle contains one little-endian float64 numeric array with shape `(55, 5, 35)`, not 30 action columns. `extract_qtable.py` uses `pickletools` to parse its byte payload, checks the expected dtype strings and payload length, and saves `q_table.npy`. No pickle instructions were executed. `q_table_metadata.json` records the summary.

- `app.py:219` defines power levels `[6,8,10,12,14]` dBm.
- `app.py:221` includes `SF13` in `[7,8,9,10,11,12,13]`. `itertools.product(sfs,power_levels)` gives action index `a=5*(SF-7)+power_position`, with positions zero through four. Action indices 30–34 correspond to SF13, outside the paper's SF7–12 set and ordinary RN2483 LoRaWAN data-rate mapping used here.
- Three of the 275 SNR/power state rows have their highest Q-value in action 30 or 31. Hence the extra action columns are not merely unused padding. With the literal application's always-last power column, one SNR bin selects action31 (SF13,8dBm), and `new_dr=12-sf` would yield `-1`.
- `app.py:223` defines 55 descending SNR thresholds `np.linspace(6.5,-20.5,55)`, spaced by 0.5 dB. `232` takes `min(np.digitize(snr,snr_space),54)`; preserve this exact mapping if evaluating the table's original inference.
- The application's `selected_tp` is the RN2483 power index, initialized to1 (`270`) and updated with the mapping `{6:5,8:4,10:3,12:2,14:1}` (`226–239,364`). However, `233` digitizes it against dBm thresholds `[6,8,10,12,14]`. Every legal index1–5 then becomes `-1`, selecting the table's final power column. This is a verified units/index mismatch.

A sensible adapted reference could convert the current power index to dBm before indexing and restrict candidate actions to the first30 columns. Label this as a **corrected/adapted pretrained prototype reference**. It is not a literal execution of the released application. The original learned weights also reflect an unknown training environment.

For an ablation of state and reward, the cleaner primary comparison is a jointly reconstructed Q-learning family trained under the same simulation and feedback conditions. Call the original-state/original-reward member a **paper-based DSF2R reconstruction**. Its missing training details and any handling of zero SNR margin, failed receptions, feedback delay, and initialization need to be stated explicitly. Do not attribute those chosen details to the authors.

## Consistency with paper extracts

- 2025 paper PDF p.4 specifies state `(SNR,TP)` and action `(SF,TP)`. Table III lists five powers and SF7–12. It trains in simulation using measured radio values, but the complete code path is absent from this repository.
- 2026 paper PDF p.5 says updates occur upon receiving a gateway downlink. Its Algorithm1 says55 SNR values, while p.6 prose says50 levels and six power choices. These statements are inconsistent with one another and with the current 55×5 table. A reconstruction should choose and declare one specification.

## Exact sources and hashes

Immutable application link: https://github.com/sofianebatta/LoRaWAN-RL-Parametrizing/blob/98f74f47dfef4fd4b49d0d22b26fca27cc726cb6/app/app.py

Immutable radio driver link: https://github.com/sofianebatta/LoRaWAN-RL-Parametrizing/blob/98f74f47dfef4fd4b49d0d22b26fca27cc726cb6/app/RN2483.py

SHA256:

- app.py `67c8eb79bae877d4cf073410e1539885b78d19838c1703b804a82fe6fae6ea63`
- RN2483.py `1f3ca75e1277b8a5b5865f0aba0395305361c97020504dfae8d5b608f2c556fc`
- README.md `7a444a42f302899ec0de5b09bec5ab281b4f9470a4dcbc1eca556d9d39bcd387`
- Q_model-LORA-rob.pkl `78298f216a8328e12292a14a9225eacc314690e42cee9f33dce9c3e9659bb177`

Fetch route used GitHub `/commits/main`, `/git/trees/main?recursive=1`, pinned raw file URLs, and the table's Git blob API URL from the recursive response. The first raw table transfer failed; GitHub's blob API content was decoded from base64. This did not require executing repository code.

All four acquired source files were additionally checked against the recursive response's Git blob SHA1 using the exact Git blob header and content bytes. Each matched its advertised blob hash, including app.py initially fetched from the main branch.
