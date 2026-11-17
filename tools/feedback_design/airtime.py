"""Analytical LoRaWAN feedback airtime accounting; no network simulation.

Run with: python -X utf8 outputs/feedback_optimization/airtime.py
Writes airtime.csv and comparison.csv beside this script using only stdlib.

All packet sizes are PHYPayload bytes, including LoRaWAN headers and MIC,
but excluding the LoRa physical header, preamble, and PHY payload CRC.
Radio assumptions: 125 kHz bandwidth, CR 4/5, eight preamble symbols,
explicit header, low-data-rate optimization for SF11 and SF12. Payload CRC
is disabled for downlinks and enabled for uplinks, per RP002-1.0.3 p.86.
The symbol expression is the Semtech SX1276 airtime formula (SF7--SF12).

comparison.csv normalizes the long-run report rate to 1000/K attempted
uplinks; this can be fractional and is not a finite-trace packet count.
Every scheduled report is assumed transmitted exactly once. There are no
retries, missed scheduling opportunities, duty-cycle queues, collisions,
delivery losses, receive-window costs, or energy estimates. Each row uses
one homogeneous downlink SF. It does not establish downlink feasibility.
Savings from removing the four-byte uplink request are calculated separately
in airtime.csv and are excluded from comparison.csv.

Sources:
https://lora-alliance.org/wp-content/uploads/2021/05/RP002-1.0.3-FINAL-1.pdf
https://raw.githubusercontent.com/Lora-net/LoRaMac-node/master/src/radio/sx1276/sx1276.c
"""

import csv
import math
from pathlib import Path


BANDWIDTH_HZ = 125_000
CODING_RATE_INDEX = 1  # CR=4/(4+1)=4/5.
PREAMBLE_SYMBOLS = 8
SPREADING_FACTORS = tuple(range(7, 13))
DOWNLINK_PHY_BYTES = (14, 15, 16, 17, 18, 20, 22, 35)
PROTOCOLS = (
    ("existing", 35, 8, 16),
    ("compact_k8_w16", 16, 8, 16),
    ("compact_k16_w32", 18, 16, 32),
    ("compact_k32_w64", 22, 32, 64),
)


def airtime_seconds(sf: int, phy_bytes: int, *, crc_on: bool) -> float:
    """Return physical airtime; phy_bytes does not include a PHY CRC."""
    if sf not in SPREADING_FACTORS:
        raise ValueError("This calculator supports SF7 through SF12 only")
    if not 0 <= phy_bytes <= 255:
        raise ValueError("PHYPayload length must be between 0 and 255 bytes")
    low_data_rate_optimization = int(sf >= 11)
    # Explicit header means IH=0, so the usual -20*IH term is zero.
    numerator = 8 * phy_bytes - 4 * sf + 28 + 16 * int(crc_on)
    denominator = 4 * (sf - 2 * low_data_rate_optimization)
    payload_symbols = 8 + max(
        math.ceil(numerator / denominator) * (CODING_RATE_INDEX + 4), 0
    )
    symbol_seconds = (2**sf) / BANDWIDTH_HZ
    return (PREAMBLE_SYMBOLS + 4.25 + payload_symbols) * symbol_seconds


def verify_calculations() -> None:
    """Check the requested anchors and monotonicity, allowing symbol plateaus."""
    assert math.isclose(
        airtime_seconds(12, 35, crc_on=False), 1.810432, abs_tol=1e-12
    )
    assert math.isclose(
        airtime_seconds(12, 16, crc_on=False), 1.155072, abs_tol=1e-12
    )
    for crc_on in (False, True):
        for sf in SPREADING_FACTORS:
            values = [airtime_seconds(sf, size, crc_on=crc_on) for size in range(256)]
            assert all(a <= b for a, b in zip(values, values[1:]))
        for size in DOWNLINK_PHY_BYTES + (33, 37):
            values = [airtime_seconds(sf, size, crc_on=crc_on) for sf in SPREADING_FACTORS]
            assert all(a <= b for a, b in zip(values, values[1:]))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_airtime_rows() -> list[dict]:
    rows = []
    for sf in SPREADING_FACTORS:
        for size in DOWNLINK_PHY_BYTES:
            rows.append(
                {
                    "direction": "downlink",
                    "sf": sf,
                    "phy_bytes": size,
                    "payload_crc_on": 0,
                    "airtime_s": f"{airtime_seconds(sf, size, crc_on=False):.9f}",
                    "comparison_baseline_phy_bytes": 35,
                    "increment_vs_baseline_s": f"{airtime_seconds(sf, size, crc_on=False) - airtime_seconds(sf, 35, crc_on=False):.9f}",
                }
            )
        for size in (33, 37):
            rows.append(
                {
                    "direction": "uplink",
                    "sf": sf,
                    "phy_bytes": size,
                    "payload_crc_on": 1,
                    "airtime_s": f"{airtime_seconds(sf, size, crc_on=True):.9f}",
                    "comparison_baseline_phy_bytes": 33,
                    "increment_vs_baseline_s": f"{airtime_seconds(sf, size, crc_on=True) - airtime_seconds(sf, 33, crc_on=True):.9f}",
                }
            )
    return rows


def make_comparison_rows() -> list[dict]:
    rows = []
    for sf in SPREADING_FACTORS:
        baseline = airtime_seconds(sf, 35, crc_on=False) * 1000 / 8
        for name, size, interval, window in PROTOCOLS:
            report_airtime = airtime_seconds(sf, size, crc_on=False)
            total_airtime = report_airtime * 1000 / interval
            rows.append(
                {
                    "protocol": name,
                    "sf": sf,
                    "report_phy_bytes": size,
                    "report_interval_k_ul": interval,
                    "feedback_window_w_ul": window,
                    "scheduled_reports_per1000_ul": f"{1000 / interval:.6f}",
                    "report_airtime_s": f"{report_airtime:.9f}",
                    "downlink_airtime_per1000_ul_s": f"{total_airtime:.9f}",
                    "reduction_vs35k8_fraction": f"{1 - total_airtime / baseline:.9f}",
                    "reduction_vs35k8_percent": f"{100 * (1 - total_airtime / baseline):.6f}",
                }
            )
    return rows


def main() -> None:
    verify_calculations()
    output_dir = Path(__file__).resolve().parent
    airtime_rows = make_airtime_rows()
    comparison_rows = make_comparison_rows()
    write_csv(output_dir / "airtime.csv", airtime_rows)
    write_csv(output_dir / "comparison.csv", comparison_rows)
    print("Analytical checks passed: SF12 35 B = 1.810432 s; 16 B = 1.155072 s.")
    print("Airtime is nondecreasing with packet length and SF under these assumptions.")
    print(f"Wrote {len(airtime_rows)} airtime rows and {len(comparison_rows)} protocol rows.")
    print("\nSF12 report comparison (normalized long-run rate, not a network run):")
    for row in comparison_rows:
        if row["sf"] == 12:
            print(
                f"  {row['protocol']}: {row['report_airtime_s']} s/report, "
                f"{row['downlink_airtime_per1000_ul_s']} s/1000 UL, "
                f"{row['reduction_vs35k8_percent']}% reduction"
            )
    print("\nFour-byte uplink request cost, PHY33 -> PHY37 with CRC on:")
    for sf in SPREADING_FACTORS:
        base = airtime_seconds(sf, 33, crc_on=True)
        request = airtime_seconds(sf, 37, crc_on=True)
        print(f"  SF{sf}: {base:.9f} -> {request:.9f} s; increment {request - base:.9f} s")
    print("\nExclude UL request savings, losses, retries, queues, RX energy, and feasibility.")


if __name__ == "__main__":
    main()
