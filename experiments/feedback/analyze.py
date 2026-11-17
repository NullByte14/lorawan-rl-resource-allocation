"""Summarize paired simulator runs and render four presentation figures.

All estimates use one independent evaluation seed as the sampling unit.
Percentile bootstrap intervals use 10,000 paired seed resamples by default.
These intervals describe this synthetic model, not hardware uncertainty.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CONDITIONS = ("A", "B", "C", "D")
LEARNERS = ("original", "reward", "both")
COLORS = {"A": "#8c564b", "B": "#377eb8", "C": "#e59c20", "D": "#15936d"}
LINESTYLES = {"A": "--", "B": "--", "C": "-", "D": "-"}
DRAW_ORDER = ("C", "D", "A", "B")  # Dashed curves stay visible over exact solid overlaps.
LABELS = {"A": "A: large + request", "B": "B: compact + request",
          "C": "C: large, no request", "D": "D: compact, no request"}
TITLES = {"reward": "Delivery/energy reward", "both": "Delivery history + energy reward"}
METRICS = ("pdr", "app_delivery", "j_per_delivery", "dl_airtime_s",
           "dl_airtime_per1000_generated_s", "total_j", "tx_j", "rx_j", "sleep_j",
           "generated", "attempted", "received", "requested", "reports", "dl_sent",
           "report_phy_bytes", "resolved", "unresolved", "tail_unknown",
           "ul_airtime_s", "request_airtime_s", "mean_label_delay_s",
           "mean_report_phy_bytes", "mean_report_airtime_s")
PRIMARY = ("pdr", "app_delivery", "j_per_delivery")
_BOOTSTRAP_CACHE = (None, None)


def stable_seed(label: str, base: int = 20261117) -> int:
    return (base + int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")) % (2**64)


def bootstrap_mean(values, label: str = "", resamples: int = 10_000):
    """Return mean and percentile bounds, with no silent nonfinite filtering."""
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or len(a) == 0 or not np.all(np.isfinite(a)):
        raise ValueError("Bootstrap needs a nonempty finite vector")
    if resamples < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    global _BOOTSTRAP_CACHE
    # Related metrics/time points share the same resamples of independent seeds.
    # Cache only the most recent group so memory stays bounded.
    group_label = label.rsplit(":", 1)[0] if ":" in label else label
    cache_key = (group_label, len(a), resamples)
    if _BOOTSTRAP_CACHE[0] != cache_key:
        rng = np.random.default_rng(stable_seed(group_label))
        weights = rng.multinomial(len(a), np.full(len(a), 1 / len(a)), size=resamples)
        _BOOTSTRAP_CACHE = (cache_key, weights)
    else:
        weights = _BOOTSTRAP_CACHE[1]
    # Multinomial counts are equivalent to sampling seed indices with replacement.
    estimates = weights @ a / len(a)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(a.mean()), float(low), float(high)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def derived(stats: dict) -> dict:
    s = stats.copy()
    for key in ("generated", "attempted", "received", "total_j", "dl_airtime_s"):
        if key not in s:
            raise ValueError(f"Required simulator metric missing: {key}")
    s["pdr"] = s["received"] / s["attempted"] if s["attempted"] else None
    s["app_delivery"] = s["received"] / s["generated"] if s["generated"] else None
    s["j_per_delivery"] = s["total_j"] / s["received"] if s["received"] else None
    s["dl_airtime_per1000_generated_s"] = (
        1000 * s["dl_airtime_s"] / s["generated"] if s["generated"] else None)
    count = s.get("feedback_age_count", 0)
    s["mean_label_delay_s"] = s.get("feedback_age_sum_s", 0) / count if count else None
    s["feedback_age_mean_s"] = s["mean_label_delay_s"]
    reports_sent = s.get("dl_sent", 0)
    s["mean_report_phy_bytes"] = s.get("report_phy_bytes", 0) / reports_sent if reports_sent else None
    s["mean_report_airtime_s"] = s["dl_airtime_s"] / reports_sent if reports_sent else None
    return s


def phase_rows(records: list[dict]) -> list[dict]:
    rows = []
    no_sum = {"pdr", "app_delivery", "j_per_delivery", "mean_label_delay_s", "feedback_age_mean_s",
              "mean_report_phy_bytes", "mean_report_airtime_s"}
    for record in records:
        meta = {k: record[k] for k in ("mode", "nodes", "seed", "learner", "condition")}
        for phase, stats in enumerate(record["phases"]):
            rows.append({**meta, "phase": phase, **derived(stats)})
        total = {}
        for stats in record["phases"]:
            for key, value in stats.items():
                if key not in no_sum and isinstance(value, (int, float)) and not isinstance(value, bool):
                    total[key] = total.get(key, 0) + value
        rows.append({**meta, "phase": "total", **derived(total)})
    return rows


def verify_records(raw: dict) -> dict:
    """Reject broken pairing and table mutation before making any figures."""
    records = raw["records"]
    if not records:
        raise ValueError("No evaluation records")
    traces = defaultdict(set)
    initial = defaultdict(set)
    groups = defaultdict(set)
    seen = set()
    expected_seeds = set(raw["config"]["seeds"])
    for r in records:
        key = (r["mode"], r["nodes"], r["seed"], r["learner"], r["condition"])
        if key in seen:
            raise ValueError(f"Duplicate evaluation: {key}")
        seen.add(key)
        if r["condition"] not in CONDITIONS:
            raise ValueError(f"Unknown arm: {r['condition']}")
        if len(r["phases"]) != 3:
            raise ValueError(f"Expected three phases: {key}")
        traces[(r["nodes"], r["seed"])].add((r["trace_sha256"], r["topology_sha256"]))
        groups[(r["mode"], r["nodes"], r["learner"], r["condition"])].add(r["seed"])
        if r["mode"] == "frozen":
            initial[(r["nodes"], r["seed"], r["learner"])].add(r["initial_q_sha256"])
            if r["initial_q_sha256"] != r["final_q_sha256"] or r["evaluation_updates"] != 0:
                raise ValueError(f"Frozen Q-table changed: {key}")
    if any(len(values) != 1 for values in traces.values()):
        raise ValueError("Compared conditions have different generated traces/topologies")
    if any(len(values) != 1 for values in initial.values()):
        raise ValueError("Frozen conditions did not start with identical Q-tables")
    for key, seeds in groups.items():
        if seeds != expected_seeds:
            raise ValueError(f"Incomplete seed group {key}: {sorted(seeds)}")
    families = {(mode, nodes, learner) for mode, nodes, learner, condition in groups}
    for family in families:
        if any((*family, condition) not in groups for condition in CONDITIONS):
            raise ValueError(f"Incomplete four-arm comparison: {family}")
    evaluation_seeds = expected_seeds
    training_seeds = {r["environment_seed"] for r in raw["training"]}
    if evaluation_seeds & training_seeds:
        raise ValueError("Training and evaluation environment seeds overlap")
    return {"evaluations": len(records), "training_episodes": len(raw["training"]),
            "paired_trace_groups": len(traces), "frozen_table_groups": len(initial),
            "seeds_per_group": len(expected_seeds), "pairing_and_frozen_checks": "passed"}


def verify_source_hashes(raw: dict, source_dir: Path = HERE) -> int:
    sources = raw.get("source_sha256", {})
    if not sources:
        raise ValueError("Raw results have no source hashes")
    for name, expected in sources.items():
        path = source_dir / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed since simulation: {name}")
    return len(sources)


def summaries(rows: list[dict], resamples: int) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ("mode", "nodes", "learner", "condition", "phase"))].append(row)
    out = []
    for key, selected in sorted(groups.items(), key=lambda item: str(item[0])):
        selected.sort(key=lambda r: r["seed"])
        for metric in METRICS:
            available = [r[metric] for r in selected if r.get(metric) is not None]
            if metric in PRIMARY and len(available) != len(selected):
                raise ValueError(f"Undefined primary denominator in {key}: {metric}")
            estimate = bootstrap_mean(available, f"summary:{key}:{metric}", resamples) if available else (None,) * 3
            out.append({**dict(zip(("mode", "nodes", "learner", "condition", "phase"), key)),
                        "metric": metric, "n_total": len(selected), "n_valid": len(available),
                        "mean": estimate[0], "ci95_low": estimate[1], "ci95_high": estimate[2]})
    return out


def paired_rows(rows: list[dict], resamples: int) -> list[dict]:
    groups = defaultdict(dict)
    for r in rows:
        key = (r["mode"], r["nodes"], r["learner"], r["phase"])
        groups[key][(r["seed"], r["condition"])] = r
    out = []
    contrasts = {"B-A": {"B": 1, "A": -1}, "C-A": {"C": 1, "A": -1},
                 "D-A": {"D": 1, "A": -1}, "D-B": {"D": 1, "B": -1},
                 "D-C": {"D": 1, "C": -1}, "interaction_D-C-B+A": {"D": 1, "C": -1, "B": -1, "A": 1}}
    for key, selected in sorted(groups.items(), key=lambda item: str(item[0])):
        seeds = sorted({s for s, c in selected})
        for contrast, weights in contrasts.items():
            for metric in METRICS:
                values = []
                for seed in seeds:
                    v = [selected[(seed, condition)].get(metric) for condition in weights]
                    if any(value is None for value in v):
                        if metric in PRIMARY:
                            raise ValueError(f"Undefined paired primary metric {key}: {metric}")
                        continue
                    values.append(sum(weight * selected[(seed, condition)][metric]
                                      for condition, weight in weights.items()))
                estimate = bootstrap_mean(values, f"paired:{key}:{contrast}:{metric}", resamples) if values else (None,) * 3
                out.append({**dict(zip(("mode", "nodes", "learner", "phase"), key)),
                            "contrast": contrast, "metric": metric, "n_total": len(seeds),
                            "n_valid": len(values), "delta_mean": estimate[0],
                            "ci95_low": estimate[1], "ci95_high": estimate[2]})
    # Learner comparisons stay within one feedback arm and share evaluation seeds.
    frozen = {(r["nodes"], r["seed"], r["learner"], r["condition"], r["phase"]): r
              for r in rows if r["mode"] == "frozen"}
    strata = sorted({(r["nodes"], r["condition"], str(r["phase"])) for r in rows if r["mode"] == "frozen"})
    for nodes, condition, phase_string in strata:
        phase = "total" if phase_string == "total" else int(phase_string)
        for learner in ("reward", "both"):
            seeds = sorted({seed for n, seed, l, c, p in frozen if n == nodes and l == learner and c == condition and p == phase})
            if not seeds or any((nodes, seed, "original", condition, phase) not in frozen for seed in seeds):
                continue
            for metric in PRIMARY + ("dl_airtime_s",):
                values = [frozen[nodes, seed, learner, condition, phase][metric] -
                          frozen[nodes, seed, "original", condition, phase][metric] for seed in seeds]
                estimate = bootstrap_mean(values, f"learner:{nodes}:{condition}:{phase}:{learner}:{metric}", resamples)
                out.append({"mode": "frozen", "nodes": nodes, "learner": learner, "phase": phase,
                            "contrast": f"{learner}-original_same_arm_{condition}", "metric": metric,
                            "n_total": len(seeds), "n_valid": len(seeds), "delta_mean": estimate[0],
                            "ci95_low": estimate[1], "ci95_high": estimate[2]})
    return out


def make_figures(raw: dict, summary: list[dict], figures: Path, resamples: int):
    os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    lookup = {(r["mode"], r["nodes"], r["learner"], r["condition"], r["phase"], r["metric"]): r for r in summary}
    densities = sorted({r["nodes"] for r in raw["records"] if r["mode"] == "frozen"})
    online_nodes = sorted({r["nodes"] for r in raw["records"] if r["mode"] == "online"})
    online_n = 60 if 60 in online_nodes else (online_nodes[0] if online_nodes else None)
    legend = [Line2D([0], [0], color=COLORS[c], lw=2.3, ls=LINESTYLES[c], marker="o", label=LABELS[c]) for c in CONDITIONS]
    definitions = [("dl_airtime_per1000_generated_s", 1, "Downlink seconds / 1,000 generated packets", "Feedback airtime", "01_feedback_airtime"),
                   ("j_per_delivery", 1000, "Device energy / delivered packet (mJ)", "Energy per delivered packet", "02_energy_per_delivery"),
                   ("app_delivery", 100, "Received / generated application packets (%)", "Application delivery", "03_application_delivery")]
    for metric, factor, ylabel, title, filename in definitions:
        fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.7), constrained_layout=False)
        for column, learner in enumerate(("reward", "both")):
            ax = axes[0, column]
            for condition in DRAW_ORDER:
                rows = [lookup["frozen", n, learner, condition, "total", metric] for n in densities]
                means = np.array([r["mean"] for r in rows]) * factor
                low = np.array([r["ci95_low"] for r in rows]) * factor
                high = np.array([r["ci95_high"] for r in rows]) * factor
                ax.plot(densities, means, color=COLORS[condition], marker="o", ls=LINESTYLES[condition], lw=2)
                ax.fill_between(densities, low, high, color=COLORS[condition], alpha=0.08)
            reference = [lookup.get(("frozen", n, "original", "A", "total", metric)) for n in densities]
            if all(reference):
                ax.plot(densities, [r["mean"] * factor for r in reference], color="#656565", ls="--", lw=1.6)
            ax.set(title=f"Fixed Q-table: {TITLES[learner]}", xlabel="Devices", ylabel=ylabel)
            ax.set_xticks(densities)
            ax = axes[1, column]
            if online_n is None:
                ax.text(0.5, 0.5, "Online evaluation not included in this run", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            else:
                rows = [lookup["online", online_n, learner, c, "total", metric] for c in CONDITIONS]
                means = np.array([r["mean"] for r in rows]) * factor
                low = np.array([r["ci95_low"] for r in rows]) * factor
                high = np.array([r["ci95_high"] for r in rows]) * factor
                ax.bar(np.arange(4), means, color=[COLORS[c] for c in CONDITIONS], alpha=0.85, width=0.65)
                ax.errorbar(np.arange(4), means, yerr=[np.maximum(0, means-low), np.maximum(0, high-means)], color="#252525", capsize=4, fmt="none")
                ax.set(title=f"Local online learning: {online_n} devices", xlabel="Feedback condition", ylabel=ylabel)
                ax.set_xticks(np.arange(4), CONDITIONS)
            if metric == "app_delivery":
                for row in range(2):
                    axes[row, column].set_ylim(0, 100)
        handles = legend + [Line2D([0], [0], color="#656565", ls="--", label="Original reconstruction, A (fixed only)")]
        fig.suptitle(title + " — complete light/heavy/light evaluation", fontsize=15, y=0.985)
        fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.015), frameon=False)
        fig.text(0.5, 0.105, "Mean and 95% seed-bootstrap intervals. Synthetic Python model; training energy reported separately.", ha="center", fontsize=9)
        fig.subplots_adjust(top=0.90, bottom=0.19, left=0.08, right=0.985, hspace=0.43, wspace=0.28)
        for ext in ("png", "svg"):
            fig.savefig(figures / f"{filename}.{ext}", dpi=190, bbox_inches="tight")
        plt.close(fig)

    phase_seconds = raw["config"]["phase_seconds"]
    main_n = 60 if 60 in densities else densities[0]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.7))
    for row, mode in enumerate(("frozen", "online")):
        for column, learner in enumerate(("reward", "both")):
            ax = axes[row, column]
            selected_n = main_n if mode == "frozen" else online_n
            if selected_n is None:
                ax.set_axis_off()
                continue
            selected = [r for r in raw["records"] if r["mode"] == mode and r["nodes"] == selected_n and r["learner"] == learner]
            for condition in DRAW_ORDER:
                runs = sorted([r for r in selected if r["condition"] == condition], key=lambda r: r["seed"])
                if not runs:
                    continue
                starts = [b["start_s"] for b in runs[0]["bins"]]
                if any([b["start_s"] for b in r["bins"]] != starts for r in runs):
                    raise ValueError("Time bins differ between paired runs")
                estimates = []
                for i, start in enumerate(starts):
                    if any(r["bins"][i]["generated"] == 0 for r in runs):
                        raise ValueError("Undefined application-delivery time bin; do not silently omit it")
                    values = [r["bins"][i]["received"] / r["bins"][i]["generated"] for r in runs]
                    estimates.append(bootstrap_mean(values, f"time:{mode}:{learner}:{condition}:{start}", resamples))
                values = np.asarray(estimates) * 100
                # Each point represents a whole 800-second interval, shown at its midpoint.
                x = (np.asarray(starts) + 400) / 3600
                ax.plot(x, values[:, 0], color=COLORS[condition], ls=LINESTYLES[condition], lw=1.7)
                ax.fill_between(x, values[:, 1], values[:, 2], color=COLORS[condition], alpha=0.07)
            ax.axvspan(phase_seconds/3600, 2*phase_seconds/3600, color="#777777", alpha=0.08)
            for boundary in (phase_seconds, 2*phase_seconds):
                ax.axvline(boundary/3600, color="#555555", ls=":", lw=1)
            ax.set(title=f"{'Fixed Q-table' if mode == 'frozen' else 'Local online'}: {TITLES[learner]}",
                   xlabel="Simulated time (hours)", ylabel="Application delivery per 800 s (%)", ylim=(0, 100))
    fig.suptitle(f"Delivery around traffic changes — {main_n} devices (shading: heavier traffic)", fontsize=15, y=0.985)
    fig.legend(handles=legend, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.015), frameon=False)
    fig.text(0.5, 0.105, "Mean and 95% seed-bootstrap intervals. Time curves do not establish a recovery-time advantage.", ha="center", fontsize=9)
    fig.subplots_adjust(top=0.90, bottom=0.19, left=0.08, right=0.985, hspace=0.43, wspace=0.28)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"04_delivery_over_time.{ext}", dpi=190, bbox_inches="tight")
    plt.close(fig)


def render_report(raw: dict, summary: list[dict], paired: list[dict], audit: dict, output: Path):
    lookup = {(r["mode"], r["nodes"], r["learner"], r["condition"], r["phase"], r["metric"]): r for r in summary}
    pair_lookup = {(r["mode"], r["nodes"], r["learner"], r["phase"], r["contrast"], r["metric"]): r for r in paired}
    densities = sorted({r["nodes"] for r in raw["records"] if r["mode"] == "frozen"})
    main_n = 60 if 60 in densities else densities[0]
    modes = [m for m in ("frozen", "online") if any(r["mode"] == m and r["nodes"] == main_n for r in raw["records"])]
    lines = ["# Measured feedback experiment results", "",
             "These are executed synthetic Python-simulator results. They are not ns-3, hardware measurements, or an exact reproduction of published DSF2R. The original policy is a documented paper-based reconstruction.", "",
             f"Evaluated **{audit['evaluations']} held-out runs**, **{audit['training_episodes']} training episodes**, and **{audit['seeds_per_group']} paired seeds** per comparison. Frozen-table and trace/topology pairing checks passed. Training and evaluation environment seeds are disjoint.", "",
             "## Complete evaluation at the main density", "",
             f"The following averages use {main_n} devices across all three traffic phases. Device energy includes modeled TX, RX and sleep; processing and gateway electrical energy are excluded. Training is separate in `training.csv`.", "",
             "| Evaluation | Learner | Arm | Application delivery | PDR per attempt | mJ/delivery | DL seconds | Valid reports |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for mode in modes:
        learners = ("original", "reward", "both") if mode == "frozen" else ("reward", "both")
        for learner in learners:
            for c in CONDITIONS:
                def value(metric):
                    return lookup[mode, main_n, learner, c, "total", metric]["mean"]
                lines.append(f"| {mode} | {learner} | {c} | {100*value('app_delivery'):.2f}% | {100*value('pdr'):.2f}% | {1000*value('j_per_delivery'):.2f} | {value('dl_airtime_s'):.2f} | {value('reports'):.1f} |")
    lines += ["", "A = rich report + 4 request bytes; B = compact + request; C = rich without request; D = compact without request. Cadence is every 8 attempts and the receipt window is 16. All frozen arms of one learner start with the same pretrained table. Online training is matched to each protocol, so online comparisons include both training-protocol and deployment effects.", "",
              "## Paired changes: D minus A", "",
              "Each confidence interval resamples matched evaluation seeds. Negative energy/airtime changes favor D; positive delivery changes favor D. A confidence interval spanning zero does not establish an improvement or equivalence. Intervals are descriptive, pointwise and not adjusted for multiple comparisons.", "",
              "| Evaluation | Learner | Period | Application delivery change (percentage points) | Energy change (mJ/delivery) | DL airtime change (s) |",
              "|---|---|---|---:|---:|---:|"]
    for mode in modes:
        for learner in ("reward", "both"):
            for phase in ("total", 1):
                cells = []
                for metric, factor in (("app_delivery", 100), ("j_per_delivery", 1000), ("dl_airtime_s", 1)):
                    r = pair_lookup[mode, main_n, learner, phase, "D-A", metric]
                    cells.append(f"{r['delta_mean']*factor:+.2f} [{r['ci95_low']*factor:+.2f}, {r['ci95_high']*factor:+.2f}]")
                lines.append(f"| {mode} | {learner} | {'all phases' if phase == 'total' else 'high load'} | " + " | ".join(cells) + " |")
    lines += ["", "## What the results support", ""]
    rows = phase_rows(raw["records"])
    for mode in modes:
        for learner in ("reward", "both"):
            e = pair_lookup[mode, main_n, learner, "total", "D-A", "j_per_delivery"]
            d = pair_lookup[mode, main_n, learner, "total", "D-A", "app_delivery"]
            a = lookup[mode, main_n, learner, "A", "total", "dl_airtime_s"]["mean"]
            z = lookup[mode, main_n, learner, "D", "total", "dl_airtime_s"]["mean"]
            e_direction = "lower" if e["delta_mean"] < 0 else "higher"
            d_direction = "higher" if d["delta_mean"] > 0 else "lower"
            e_qualifier = "its interval spans zero" if e["ci95_low"] <= 0 <= e["ci95_high"] else "its interval excludes zero"
            d_qualifier = "its interval spans zero" if d["ci95_low"] <= 0 <= d["ci95_high"] else "its interval excludes zero"
            selected = {(r["seed"], r["condition"]): r for r in rows if r["mode"] == mode and r["nodes"] == main_n and r["learner"] == learner and r["phase"] == "total"}
            seeds = sorted({s for s, c in selected})
            lower_count = sum(selected[s, "D"]["j_per_delivery"] < selected[s, "A"]["j_per_delivery"] for s in seeds)
            lines.append(f"- **{mode}, {learner}:** D used {100*(1-z/a):.1f}% less mean gateway downlink airtime than A (ratio of arm means). Mean energy per delivery was {e_direction}, and {e_qualifier}; application delivery was {d_direction}, and {d_qualifier}. D had lower energy per delivery in {lower_count}/{len(seeds)} paired runs.")
    lines += ["", "The 36.2% analytical per-report saving is not automatically the total network saving: shorter reports change gateway availability, valid-report counts, device choices and duty eligibility. Results above charge those consequences. Byte compression within the 14–16-byte SF12 plateau has the same packet duration; request removal affects uplink airtime separately.", "",
              "## Limits and interpretation", "",
              "- Receiver/capture behavior, propagation and radio currents remain simplified and uncalibrated. No performance claim about the published DSF2R implementation follows from this reconstruction.",
              "- More independent seeds reduce sampling uncertainty but do not validate physical assumptions. The seeds are the independent units, not individual packets.",
              "- No new adaptive reporting policy was added; reporting cadence is held fixed. The plots do not prove a recovery-time advantage.",
              "- Lower airtime alone is not proof of better application performance. Read application delivery and energy together, and keep unfavorable runs visible.",
              "- Original-reward training retains historical packet SNR in rich feedback. Compact original evaluation is frozen and needs only the current SNR state.", "",
              "## Files", "",
              "- `per_seed.csv`: per-phase and whole-run metrics, including PDR, offered delivery, energy, report counts and unresolved labels. `mean_label_delay_s` is delay from transmission start to receipt of a valid outcome report; it is not time since the last report or observation age at every decision.",
              "- `summary.csv`: means and percentile bootstrap intervals, with explicit total/valid sample counts.",
              "- `paired_differences.csv`: compression, request-removal, combined and interaction contrasts, plus matched-arm learner comparisons.",
              "- `training.csv`: separate training energy, updates and table hashes.",
              "- `raw.json`: configuration, source hashes and individual evaluation/training records.",
              "- `../figures/`: four PNG/SVG figures. Every figure includes the four planned feedback arms.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "results" / "raw.json")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--figures", type=Path, default=None,
                        help="figure folder; defaults to the output folder's sibling figures")
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    audit = verify_records(raw)
    audit["source_hashes_verified"] = verify_source_hashes(raw)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = phase_rows(raw["records"])
    summary = summaries(rows, args.resamples)
    paired = paired_rows(rows, args.resamples)
    write_csv(args.output / "per_seed.csv", rows)
    write_csv(args.output / "training.csv", raw["training"])
    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "paired_differences.csv", paired)
    (args.output / "analysis_checks.json").write_text(json.dumps({**audit, "bootstrap_resamples": args.resamples,
        "bootstrap_method": "seed-level paired percentile; deterministic SHA256-derived PRNG seeds",
        "statistical_unit": "independent evaluation seed", "undefined_secondary_metrics": "n_valid reported explicitly",
        "primary_metrics": list(PRIMARY)}, indent=2), encoding="utf-8")
    render_report(raw, summary, paired, audit, args.output)
    make_figures(raw, summary, args.figures or args.output.parent / "figures", args.resamples)
    print(json.dumps({**audit, "summary_rows": len(summary), "paired_rows": len(paired),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
