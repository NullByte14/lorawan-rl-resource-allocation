# Source and data provenance

## Project code

The Python simulator, controlled comparisons, codecs, analysis, tests and data-audit tools are project implementations. The historical feedback source files are preserved byte-for-byte because their outputs record their SHA256 fingerprints. New studies import that engine rather than silently changing it. Root orchestration and documentation are separate from those frozen scientific sources.

The first allocation pilot and its results remain in `archive/initial_pilot/`. The compact-codec design and airtime calculation are retained in `tools/feedback_design/`. Experimental configuration/source and checkpoint fingerprints are recorded beside each dataset. The repository-wide artifact manifest additionally covers code, data, documentation and figures.

## DSF2R source inspection

The source audit examined the authors' public application at commit `98f74f47dfef4fd4b49d0d22b26fca27cc726cb6`. Public immutable sources include the [application](https://github.com/sofianebatta/LoRaWAN-RL-Parametrizing/blob/98f74f47dfef4fd4b49d0d22b26fca27cc726cb6/app/app.py) and [radio driver](https://github.com/sofianebatta/LoRaWAN-RL-Parametrizing/blob/98f74f47dfef4fd4b49d0d22b26fca27cc726cb6/app/RN2483.py).

The [inspection report](../experiments/feedback/reference/PROTOTYPE_AUDIT.md) retains the original audit text. Its `work/upstream/` references describe paths in the original inspection workspace, not files promised at those paths in this repository. This repository includes the acquired commit/tree metadata, source hashes and table metadata in [reference/upstream](../reference/upstream/). The external application, radio driver and pretrained binary are not redistributed here or used as a silently corrected official baseline.

The public application uses MQTT/IP SNR observations and a pretrained lookup table. Its executable training path was unavailable in the inspected tree, and action/power indexing discrepancies were documented. Consequently, the experimental original-state/original-reward member is explicitly a **paper-based reconstruction**, not the authors' executable training or an exact measured comparison with their published results.

`tools/feedback_design/extract_qtable.py` is the safe inspection utility from that audit. It uses `pickletools` to parse known numeric payload structure without executing arbitrary pickle opcodes. It requires the external file identified in the audit and is not needed to run this repository's experiments.

## Research references and claims

The [related-work comparison](related_work.md) distinguishes verified paper facts, implementation availability and reproduction status. Third-party numerical results are not imported into a graph as though they shared this simulator's conditions. The adapted Thompson and archived MIX-MAB references are labelled explicitly; unimplemented methods remain literature context.

Dependency versions are recorded in the lock file. Dependencies retain their original licenses. The project has not selected an open-source license; private GitHub publication does not grant reuse rights to third-party source or original PDFs.
