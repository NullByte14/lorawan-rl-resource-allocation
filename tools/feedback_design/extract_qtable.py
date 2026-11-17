"""Extract the public numeric array without executing pickle instructions."""

from pathlib import Path
import hashlib
import json
import pickletools

import numpy as np


def main():
    folder = Path(__file__).resolve().parent
    source = folder / "Q_model-LORA-rob.pkl"
    raw = source.read_bytes()
    operations = list(pickletools.genops(raw))
    strings = [arg for op, arg, _ in operations if op.name == "SHORT_BINUNICODE"]
    if strings != ["numpy.core.multiarray", "_reconstruct", "numpy", "ndarray", "dtype", "f8", "<"]:
        raise ValueError("Unexpected pickle structure; review before extracting")
    payloads = [arg for op, arg, _ in operations if op.name == "BINBYTES"]
    if len(payloads) != 1 or len(payloads[0]) != 55 * 5 * 35 * 8:
        raise ValueError("Unexpected raw array payload")
    table = np.frombuffer(payloads[0], dtype="<f8").reshape(55, 5, 35).copy()
    if not np.isfinite(table).all():
        raise ValueError("Nonfinite Q-value")
    np.save(folder / "q_table.npy", table, allow_pickle=False)
    policy = np.argmax(table, axis=2)
    actions, counts = np.unique(policy, return_counts=True)
    metadata = {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(table.shape),
        "dtype": str(table.dtype),
        "value_range": [float(table.min()), float(table.max())],
        "nonzero_values": int(np.count_nonzero(table)),
        "argmax_action_counts": {str(int(a)): int(c) for a, c in zip(actions, counts)},
        "argmax_sf13_states": int(np.count_nonzero(policy >= 30)),
        "literal_power_index_minus_one_policy": policy[:, -1].tolist(),
    }
    (folder / "q_table_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
